"""
Mechanical Motion Primitives


Behavioral classification-based implementation of mechanical-to-mathematical mappings
env: Python 3.10.10
"""

from __future__ import annotations
import math
from math import gcd

import random
from fractions import Fraction

from dataclasses import dataclass, field
from typing import Protocol, Tuple, Optional, Callable, Any
from typing import runtime_checkable
from abc import ABC, abstractmethod

import hashlib



# ============================================================================
# CORE PROTOCOLS - Define behavioral contracts
# ============================================================================

@runtime_checkable
class Invertible(Protocol):
    def forward(self, x: Any, **kwargs) -> Any: ...
    def inverse(self, y: Any, **kwargs) -> Any: ...

@runtime_checkable
class OneWay(Protocol):
    def forward(self, x: Any, **kwargs) -> Any: ...
    # No inverse

class Configurable(Protocol):
    """Mechanisms whose behavior changes based on configuration"""
    def set_mode(self, mode: str) -> None: ...
    def get_mode(self) -> str: ...

class Periodic(Protocol):
    """Mechanisms with cyclic behavior"""
    def period(self) -> float: ...
    def normalize_angle(self, theta: float) -> float: ...

# ============================================================================
# CLASS I: LINEAR SCALING (Affine Maps)
# Continuous, invertible, constant ratio
# ============================================================================

@dataclass
class SpurGear:
    ratio: float | Fraction
    modulus: int | None = None
    
    def __post_init__(self):
        if self.modulus:
            # Convert float to Fraction for exact arithmetic
            if isinstance(self.ratio, float):
                # This is heuristic - better to require Fraction
                self.ratio = Fraction(self.ratio).limit_denominator(1000)
            
            # Now ratio is Fraction, get numerator/denominator
            num = self.ratio.numerator
            den = self.ratio.denominator
            
            # Need modular inverse of numerator (mod modulus)
            if gcd(num, self.modulus) != 1:
                raise ValueError(f"Ratio numerator {num} must be coprime to modulus {self.modulus}")
            
            # Store for inverse operation
            self._num = num
            self._den = den
            self._num_inv = pow(num, -1, self.modulus)
    
    def forward(self, x: float) -> float:
        if self.modulus:
            return (x * self._num * pow(self._den, -1, self.modulus)) % self.modulus
        return x * self.ratio
    
    def inverse(self, y: float) -> float:
        if self.modulus:
            return (y * self._num_inv * self._den) % self.modulus
        return y / self.ratio

@dataclass
class CompoundGearTrain:
    """
    Multiple gears cascaded - product of ratios
    Behavioral class: I - Linear Scaling (cascaded)
    """
    ratios: list[float]
    modulus: Optional[int] = None
    
    @property
    def total_ratio(self) -> float:
        prod = 1.0
        for r in self.ratios:
            prod *= r
        return prod
    
    def forward(self, x: float) -> float:
        return SpurGear(self.total_ratio, self.modulus).forward(x)
    
    def inverse(self, y: float) -> float:
        return SpurGear(1.0/self.total_ratio, self.modulus).forward(y)

@dataclass
class RackAndPinion:
    """
    Rotation <-> Linear translation
    Behavioral class: I - Linear Scaling (type-changing)
    """
    pitch_radius: float  # r in y = r*theta
    
    def rotation_to_linear(self, theta: float) -> float:
        """Rotation (radians) -> linear displacement"""
        return self.pitch_radius * theta
    
    def linear_to_rotation(self, y: float) -> float:
        """Linear displacement -> rotation (radians)"""
        return y / self.pitch_radius
    
    # Aliases for consistent interface
    forward = rotation_to_linear
    inverse = linear_to_rotation

@dataclass
class Wedge:
    angle_rad: float
    
    @property
    def mechanical_advantage(self) -> float:
        """For horizontal force -> vertical lift: MA = cot(angle)"""
        return 1.0 / math.tan(self.angle_rad)
    
    @property
    def displacement_ratio(self) -> float:
        """Horizontal displacement -> vertical displacement = tan(angle)"""
        return math.tan(self.angle_rad)
    
    def forward(self, x: float) -> float:
        """Horizontal displacement -> vertical displacement"""
        return x * self.displacement_ratio
    
    def inverse(self, y: float) -> float:
        """Vertical displacement -> horizontal displacement"""
        return y / self.displacement_ratio

@dataclass
class Pantograph:
    """
    2D scaling mechanism
    Behavioral class: I - Linear Scaling (2D)
    """
    scale: float  # k
    
    def forward(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        return (x * self.scale, y * self.scale)
    
    def inverse(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        return (x / self.scale, y / self.scale)

@dataclass
class OldhamCoupling:
    """Transmits rotation between parallel offset shafts"""
    offset_x: float
    offset_y: float
    
    def forward_rotation(self, angle: float) -> float:
        """Input shaft angle -> output shaft angle (same angle)"""
        # Oldham couples transmit angle 1:1
        return angle
    
    def forward_position(self, center_xy: tuple[float, float]) -> tuple[float, float]:
        """Input shaft center -> output shaft center"""
        x, y = center_xy
        return (x + self.offset_x, y + self.offset_y)

# ============================================================================
# CLASS II: PERIODIC NON-LINEAR (Trigonometric)
# Oscillatory, bounded, non-injective without domain restriction
# ============================================================================

@dataclass
class ScotchYoke:
    """
    Crank and yoke - pure sinusoidal motion
    Behavioral class: II - Periodic Non-linear
    Note: Only invertible on [-π/2, π/2]
    """
    amplitude: float  # A
    phase: float = 0.0  # phi
    
    def forward(self, theta: float) -> float:
        """Rotation angle -> linear displacement"""
        return self.amplitude * math.sin(theta + self.phase)
    
    def inverse(self, y: float, branch: str = 'principal') -> float:
        """
        Linear displacement -> rotation angle
        Requires branch selection due to sin periodicity
        """
        if abs(y) > self.amplitude:
            raise ValueError(f"y={y} outside amplitude range [-{self.amplitude}, {self.amplitude}]")
        
        raw_theta = math.asin(y / self.amplitude) - self.phase
        
        if branch == 'principal':
            return raw_theta  # [-π/2, π/2]
        elif branch == 'supplementary':
            return math.pi - raw_theta  # [π/2, 3π/2]
        else:
            # General solution: θ = arcsin(y/A) + 2πn or π - arcsin(y/A) + 2πn
            raise NotImplementedError("Use branch='principal' or 'supplementary'")
    
    def period(self) -> float:
        return 2 * math.pi

@dataclass
class EccentricCam:
    """
    Circular cam with follower
    Behavioral class: II - Periodic Non-linear (with sqrt)
    """
    eccentricity: float  # e
    follower_radius: float  # r
    
    def forward(self, theta: float) -> float:
        """Cam rotation -> follower displacement"""
        e, r = self.eccentricity, self.follower_radius
        return e * math.cos(theta) + math.sqrt(r**2 - (e * math.sin(theta))**2)
    
    def inverse_numerical(self, y: float, theta_guess: float = 0.0) -> float:
        """
        Numerical inverse - no closed form
        Uses Newton-Raphson
        """
        def f(theta):
            return self.forward(theta) - y
        
        def f_prime(theta):
            e, r = self.eccentricity, self.follower_radius
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            # Derivative of forward function
            term1 = -e * sin_t
            term2 = -(e**2 * sin_t * cos_t) / math.sqrt(r**2 - (e * sin_t)**2)
            return term1 + term2
        
        theta = theta_guess
        for _ in range(100):  # Max iterations
            delta = f(theta) / f_prime(theta)
            theta -= delta
            if abs(delta) < 1e-10:
                return theta
        raise ValueError("Failed to converge")

@dataclass
class CrankSlider:
    """
    Crank-slider mechanism
    Behavioral class: II - Periodic Non-linear (connecting rod)
    """
    crank_length: float  # r
    rod_length: float  # L
    
    def forward(self, theta: float) -> float:
        """Crank angle -> slider position"""
        r, L = self.crank_length, self.rod_length
        return r * math.cos(theta) + math.sqrt(L**2 - (r * math.sin(theta))**2)
    
    def approximate(self, theta: float) -> float:
        """Small-angle approximation (L >> r)"""
        r, L = self.crank_length, self.rod_length
        return L + r * math.cos(theta) - (r**2 / (2*L)) * math.sin(theta)**2

@dataclass
class HookesJoint:
    """
    Universal joint (Hooke's coupling)
    Behavioral class: II - Periodic Non-linear (velocity fluctuation)
    """
    shaft_angle: float  # alpha in radians
    
    def forward(self, theta_in: float) -> float:
        """Input shaft angle -> output shaft angle"""
        alpha = self.shaft_angle
        # tan(θ_out) = cos(α) * tan(θ_in)
        tan_out = math.cos(alpha) * math.tan(theta_in)
        return math.atan2(math.sin(theta_in) * math.cos(alpha), math.cos(theta_in))
    
    def angular_velocity_ratio(self, theta_in: float) -> float:
        """ω_out / ω_in at given input angle"""
        alpha = self.shaft_angle
        cos_in = math.cos(theta_in)
        sin_in = math.sin(theta_in)
        denom = (cos_in**2 + (sin_in * math.cos(alpha))**2)
        return math.cos(alpha) / denom

# ============================================================================
# CLASS III: CONFIGURATION-DEPENDENT (Mode-Switchable)
# Same hardware, different equations based on constraint
# ============================================================================

@dataclass
class PlanetaryGear:
    """
    Epicyclic gearing - behavior changes based on which element is fixed
    Behavioral class: III - Configuration-dependent
    """
    sun_teeth: int  # N_s
    planet_teeth: int  # N_p
    ring_teeth: int  # N_r
    carrier_teeth: Optional[int] = None
    
    def __post_init__(self):
        self.modes = ['sun_fixed', 'ring_fixed', 'carrier_fixed']
        self.current_mode = 'ring_fixed'  # default
    
    def set_mode(self, mode: str) -> None:
        if mode not in self.modes:
            raise ValueError(f"Mode must be one of {self.modes}")
        self.current_mode = mode
    
    def get_mode(self) -> str:
        return self.current_mode
    
    def angular_velocity(self, sun_w: float, ring_w: float, carrier_w: float) -> dict:
        """
        Calculate all angular velocities based on fixed element
        Returns dictionary of all velocities
        """
        N_s, N_r = self.sun_teeth, self.ring_teeth
        
        if self.current_mode == 'ring_fixed':
            # Ring fixed (ω_r = 0)
            # ω_c = (N_s / (N_s + N_r)) * ω_s
            carrier_w = (N_s / (N_s + N_r)) * sun_w
            return {'sun': sun_w, 'ring': 0, 'carrier': carrier_w}
        
        elif self.current_mode == 'sun_fixed':
            # Sun fixed (ω_s = 0)
            # ω_c = (N_r / (N_s + N_r)) * ω_r
            carrier_w = (N_r / (N_s + N_r)) * ring_w
            return {'sun': 0, 'ring': ring_w, 'carrier': carrier_w}
        
        elif self.current_mode == 'carrier_fixed':
            # Carrier fixed (ω_c = 0)
            # ω_r = -(N_s / N_r) * ω_s
            ring_w = -(N_s / N_r) * sun_w
            return {'sun': sun_w, 'ring': ring_w, 'carrier': 0}
    
    def forward(self, input_speed: float, input_element: str = 'sun') -> float:
        """
        Convenience method - get output speed at carrier
        """
        if input_element == 'sun':
            vels = self.angular_velocity(input_speed, 0, 0)
        elif input_element == 'ring':
            vels = self.angular_velocity(0, input_speed, 0)
        else:
            raise ValueError("input_element must be 'sun' or 'ring'")
        
        return vels['carrier']

@dataclass
class DifferentialGear:
    """
    Automotive differential - lossy sum
    Behavioral class: III - Configuration-dependent (lossy)
    """
    def forward(self, left: float, right: float) -> float:
        """Two inputs sum to carrier output (lossy)"""
        return (left + right) / 2
    
    def inverse(self, output: float, known_input: float) -> float:
        """Recover unknown input given output and one input"""
        return 2 * output - known_input
    
    def split(self, input_speed: float, diff_speed: float = 0) -> Tuple[float, float]:
        """
        Carrier input splits to two outputs
        diff_speed represents turning difference
        """
        left = input_speed + diff_speed/2
        right = input_speed - diff_speed/2
        return left, right

@dataclass
class FourBarLinkage:
    """Four-bar mechanism - Freudenstein's equation form"""
    a: float  # input crank
    b: float  # coupler
    c: float  # output crank
    d: float  # ground
    
    def __post_init__(self):
        # Normalized Freudenstein coefficients
        # R1 = d/a, R2 = d/c, R3 = (a² - b² + c² + d²)/(2ac)
        self.R1 = self.d / self.a
        self.R2 = self.d / self.c
        self.R3 = (self.a**2 - self.b**2 + self.c**2 + self.d**2) / (2 * self.a * self.c)
    
    def freudenstein(self, theta_in: float, theta_out: float) -> float:
        """
        Freudenstein equation in standard form:
        R1 * cos(theta_out) - R2 * cos(theta_in) + R3 - cos(theta_in - theta_out) = 0
        """
        return (self.R1 * math.cos(theta_out) - 
                self.R2 * math.cos(theta_in) + 
                self.R3 - 
                math.cos(theta_in - theta_out))
    
    def forward_numerical(self, theta_in: float, guess: float = 0.0) -> float:
        """Find theta_out given theta_in"""
        def f(theta_out):
            return self.freudenstein(theta_in, theta_out)
        
        def f_prime(theta_out):
            return (-self.R1 * math.sin(theta_out) +
                    math.sin(theta_in - theta_out))
        
        theta = guess
        for _ in range(100):
            residual = f(theta)
            if abs(residual) < 1e-10:
                return theta
            
            deriv = f_prime(theta)
            if abs(deriv) < 1e-10:
                raise ValueError("Derivative near zero")
            
            theta -= residual / deriv
        
        raise ValueError("Failed to converge")

# ============================================================================
# CLASS IV: QUANTIZING / DISCRETIZING
# Continuous input → stepped output
# ============================================================================

@dataclass
class Ratchet:
    tooth_pitch: float = 1.0
    # State fields with init=False
    position: float = field(init=False, default=0.0)
    max_position: float = field(init=False, default=0.0)
    
    def __post_init__(self):
        self.position = 0.0
        self.max_position = 0.0
    
    def forward(self, x: float) -> float:
        """
        Advance ratchet - can only increase
        Returns current position after engagement
        """
        # Quantize to nearest tooth
        tooth = round(x / self.tooth_pitch) * self.tooth_pitch
        
        # Can only move forward
        if tooth > self.max_position:
            self.max_position = tooth
        
        self.position = self.max_position
        return self.position
    
    def reset(self) -> None:
        """Manual reset (not part of normal operation)"""
        self.position = 0.0
        self.max_position = 0.0
    
    def __call__(self, x: float) -> float:
        return self.forward(x)


@dataclass
class Detent:
    step: float
    hysteresis: float = 0.1
    last_position: float = field(init=False, default=0.0)
    
    def forward(self, x: float) -> float:
        # Nearest detent position
        stepped = round(x / self.step) * self.step
        
        # Hysteresis - only move if outside deadband FROM CURRENT POSITION
        if abs(stepped - self.last_position) < self.hysteresis:
            return self.last_position
        
        self.last_position = stepped
        return stepped

class Escapement:
    """
    Clock escapement - continuous force → discrete ticks
    Behavioral class: IV - Quantizing (clocked)
    """
    def __init__(self, advance_per_tick: float = 1.0, period: float = 1.0):
        self.advance = advance_per_tick
        self.period = period
        self.state = 0.0
        self.last_release = 0.0
    
    def forward(self, time: float, force: float = 1.0) -> Optional[float]:
        """
        Time input -> tick output (if time for release)
        Force must be sufficient to overcome escapement
        """
        if force <= 0:
            return None
        
        # Time since last release
        delta = time - self.last_release
        
        if delta >= self.period:
            # Release! Advance by one tick
            self.state += self.advance
            self.last_release = time
            return self.state
        
        return None  # No tick this call
    
    def tick_count(self) -> float:
        return self.state

@dataclass
class GenevaDrive:
    num_slots: int = 4
    current_slot: int = field(init=False, default=-1)
    output_angle: float = field(init=False, default=0.0)
    slot_positions: list = field(init=False, default_factory=list)
    engagement_window: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.engagement_window = math.pi / self.num_slots
        self.slot_positions = [i * 2 * math.pi / self.num_slots for i in range(self.num_slots)]
    
    def _engagement_profile(self, theta: float, entry_angle: float) -> float:
        """
        Proper Geneva kinematics:
        tan(ψ) = sin(θ) / (λ - cos(θ)) where λ = center_distance/crank_radius
        """
        # θ is input angle relative to engagement start
        lambda_ratio = self.center_distance / self.crank_radius
        
        # During engagement, output angle ψ follows:
        tan_psi = math.sin(theta) / (lambda_ratio - math.cos(theta))
        psi = math.atan2(math.sin(theta), lambda_ratio - math.cos(theta))
        
        # Normalize to slot position
        return psi % self.slot_angle
        
    def forward(self, input_angle: float) -> float:
        """Input rotation -> output rotation (handles engagement and update)"""
        theta = input_angle % (2 * math.pi)

        for i, center in enumerate(self.slot_positions):
            if abs(theta - center) <= self.engagement_window:
                if i != self.current_slot:
                    self.current_slot = i
                    self.output_angle = self._engagement_profile(theta, center)
                return self.output_angle

        return self.output_angle


# ============================================================================
# CLASS V: ONE-WAY / IRREVERSIBLE
# Physically non-backdrivable
# ============================================================================

@dataclass
class WormGear:
    """Pure mechanical worm gear - no hashing logic"""
    num_starts: int = 1
    friction_coefficient: float = 0.1

    def __post_init__(self):
        self.position: float = 0.0  # mutable state, not a dataclass field

    @property
    def is_backdrivable(self) -> bool:
        lead_angle = math.atan(self.num_starts / (2 * math.pi))
        return lead_angle > math.atan(self.friction_coefficient)

    def forward(self, worm_rotations: float) -> float:
        advance = worm_rotations / self.num_starts
        if self.is_backdrivable or advance > 0:
            self.position += advance
        return self.position


class IrreversibleMixer:
    """Separate construction for hash-like behavior"""
    def __init__(self, worm: WormGear, quantizer: Optional[Detent] = None):
        self.worm = worm
        self.quantizer = quantizer or Detent(step=0.001)
    
    def mix(self, data: bytes, state: int = 0) -> int:
        for byte in data:
            # Mechanical forward (irreversible if worm isn't backdrivable)
            self.worm.forward(byte / 255.0)
            # Quantization loses information (like friction)
            quantized = self.quantizer.forward(self.worm.position)
            # Mix into state
            state = (state << 1) ^ int(quantized * 1000)
        return state

# ============================================================================
# CLASS VI: HIGH-RATIO / OBFUSCATING
# Extreme scaling that obscures relationship
# ============================================================================

@dataclass
class HarmonicDrive:
    """
    Harmonic drive / strain wave gearing
    Behavioral class: VI - High-ratio / Obfuscating
    """
    flexspline_teeth: int
    circular_spline_teeth: int
    
    @property
    def ratio(self) -> float:
        """Reduction ratio"""
        return self.flexspline_teeth / (self.circular_spline_teeth - self.flexspline_teeth)
    
    def forward(self, input_rotations: float) -> float:
        """Input (wave generator) -> output (flexspline)"""
        return input_rotations / self.ratio
    
    def strain_component(self, input_rotations: float) -> float:
        """Add non-linear strain wave component (obfuscation)"""
        base = self.forward(input_rotations)
        # Strain wave introduces small non-linear perturbation
        perturbation = 0.01 * math.sin(2 * math.pi * input_rotations * self.ratio)
        return base + perturbation

@dataclass
class DifferentialScrew:
    pitch1: float  # p₁ - thread pitch (linear distance per rotation)
    pitch2: float  # p₂ - thread pitch (linear distance per rotation)
    
    @property
    def effective_pitch(self) -> float:
        """Net linear advance per rotation"""
        return self.pitch1 - self.pitch2
    
    def forward(self, rotations: float) -> float:
        """Rotations -> linear displacement (rotations, not radians)"""
        return rotations * self.effective_pitch
    
    def forward_radians(self, radians: float) -> float:
        """If input is in radians instead of rotations"""
        return (radians / (2 * math.pi)) * self.effective_pitch
 
class CVT:
    """
    Continuously Variable Transmission
    Behavioral class: VI - High-ratio / Obfuscating (key-parameterized)
    """
    def __init__(self, min_ratio: float = 0.5, max_ratio: float = 2.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.control_position = 0.5  # 0 to 1
    
    def set_control(self, position: float) -> None:
        """Set CVT control position (acts as key)"""
        self.control_position = max(0.0, min(1.0, position))
    
    @property
    def current_ratio(self) -> float:
        """Current ratio based on control position"""
        return self.min_ratio + self.control_position * (self.max_ratio - self.min_ratio)
    
    def forward(self, input_speed: float) -> float:
        """Input speed -> output speed with current ratio"""
        return input_speed * self.current_ratio
    
    def inverse(self, output_speed: float) -> float:
        """If we know ratio, can invert"""
        return output_speed / self.current_ratio

@dataclass
class ToroidalCVT:
    """
    Toroidal CVT - ratio varies as tan(α)
    Behavioral class: VI - High-ratio / Obfuscating (trigonometric)
    """
    def __post_init__(self):
        self.tilt_angle = 0.0  # α in radians
    
    def set_tilt(self, angle_rad: float) -> None:
        """Set roller tilt angle (key)"""
        # Avoid pi/2 singularity
        if abs(abs(angle_rad) - math.pi/2) < 0.01:
            raise ValueError("Angle too close to pi/2")
        self.tilt_angle = angle_rad
    
    @property
    def ratio(self) -> float:
        """Ratio = tan(α)"""
        return math.tan(self.tilt_angle)
    
    def forward(self, input_speed: float) -> float:
        return input_speed * self.ratio
    
    def inverse(self, output_speed: float) -> float:
        return output_speed / self.ratio

# ============================================================================
# CLASS VII: COMPLEX PATH / MULTI-DIMENSIONAL
# Produces curves in 2D or higher spaces
# ============================================================================

@dataclass
class TrochoidalGear:
    """
    Trochoidal gear (Wankel-style) - 1D input → 2D output
    Behavioral class: VII - Complex path / Multi-dimensional
    """
    R: float  # fixed circle radius
    r: float  # rolling circle radius
    d: float  # tracing point offset
    
    def position(self, t: float) -> Tuple[float, float]:
        """
        t = rotation parameter (radians)
        Returns (x, y) position
        """
        R, r, d = self.R, self.r, self.d
        x = (R + r) * math.cos(t) - d * math.cos((R + r) * t / r)
        y = (R + r) * math.sin(t) - d * math.sin((R + r) * t / r)
        return (x, y)
    
    def is_closed(self, tolerance: float = 1e-10) -> bool:
        """Returns True if path is closed (R/r rational)"""
        ratio = self.R / self.r
        # Get best rational approximation with reasonable denominator
        frac = Fraction(ratio).limit_denominator(10000)
        # Check if approximation is exact (within floating error)
        return abs(ratio - float(frac)) < tolerance
    
    def period(self) -> Optional[float]:
        """If closed, return fundamental period"""
        if not self.is_closed():
            return None
        
        ratio = Fraction(self.R / self.r).limit_denominator(10000)
        # Period relates to denominator of simplified ratio
        # For epitrochoid: period = 2π * denominator
        return 2 * math.pi * ratio.denominator

@dataclass
class Spirograph:
    """
    Spirograph / Hypotrochoid
    Behavioral class: VII - Complex path / Multi-dimensional
    """
    R: float  # fixed gear radius
    r: float  # moving gear radius
    d: float  # pen offset from moving gear center
    
    def position(self, t: float) -> Tuple[float, float]:
        """
        t = rotation parameter
        Returns (x, y) position of pen
        """
        R, r, d = self.R, self.r, self.d
        x = (R - r) * math.cos(t) + d * math.cos((R - r) * t / r)
        y = (R - r) * math.sin(t) - d * math.sin((R - r) * t / r)
        return (x, y)
    
    def num_loops(self) -> int:
        """Number of loops before closing (if rational)"""
        from fractions import Fraction
        ratio = Fraction(self.r / self.R).limit_denominator()
        return ratio.denominator

# ============================================================================
# CLASS VIII: COINCIDENCE / RESONANT
# Output only at specific alignments
# ============================================================================

@dataclass
class VernierScale:
    """
    Vernier scale - precise measurement via coincidence
    Behavioral class: VIII - Coincidence / Resonant
    """
    main_scale_spacing: float
    vernier_spacing: float
    
    def __post_init__(self):
        self.n = self.main_scale_spacing / (self.main_scale_spacing - self.vernier_spacing)
    
    def find_coincidence(self, offset: float) -> tuple[int, int, float]:
        best_n, best_m = 0, 0
        min_error = float('inf')
        
        for n in range(int(self.n * 10)):
            m = round((n * self.main_scale_spacing - offset) / self.vernier_spacing)
            position = n * self.main_scale_spacing
            error = abs(position - (m * self.vernier_spacing + offset))
            
            # Use <= instead of < to prefer later indices on ties
            if error <= min_error:
                min_error = error
                best_n, best_m = n, m
                
                if error < 1e-6:
                    break
        
        return (best_n, best_m, best_n * self.main_scale_spacing)
    
    def measure(self, offset: float) -> float:
        """Measure unknown offset using vernier coincidence"""
        _, _, position = self.find_coincidence(offset)
        return position

"""
 Mechanical Chain Compositor
"""

class MechanicalChain:
    """Compose multiple mechanisms in sequence"""
    def __init__(self, mechanisms: list):
        self.mechanisms = mechanisms
        self._check_compatibility()
    
    def _check_compatibility(self):
        """Verify we can chain these mechanisms"""
        for i, mech in enumerate(self.mechanisms):
            if i < len(self.mechanisms) - 1:
                # Check that output type matches next input type
                # This is heuristic - actual checking depends on your type system
                pass
    
    def forward(self, x: Any) -> Any:
        """Pass input through all mechanisms in order"""
        for mech in self.mechanisms:
            x = mech.forward(x)
        return x
    
    def inverse(self, y: Any) -> Any:
        """Reverse pass - only works if all mechanisms are invertible"""
        if not self.is_fully_invertible():
            raise ValueError("Chain contains non-invertible mechanisms")
        
        for mech in reversed(self.mechanisms):
            y = mech.inverse(y)
        return y
    
    def is_fully_invertible(self) -> bool:
        return all(isinstance(m, Invertible) for m in self.mechanisms)
    
    def __or__(self, other):
        """Operator overload for chaining: mech1 | mech2"""
        if isinstance(other, MechanicalChain):
            return MechanicalChain(self.mechanisms + other.mechanisms)
        return MechanicalChain(self.mechanisms + [other])