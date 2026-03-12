"""
Mechanical Motion Primitives


Behavioral classification-based implementation of mechanical-to-mathematical mappings
env: Python 3.10.10
"""

from __future__ import annotations

import math
import random
import hashlib

from math           import gcd
from fractions      import Fraction

from abc            import ABC, abstractmethod
from dataclasses    import dataclass, field
from typing         import Protocol, Tuple, Optional, Callable, Any
from typing         import runtime_checkable
from enum           import Enum, auto


# ============================================================================
# CORE: MECHANICAL LIMITS - Physical constants
# ============================================================================

class Dimension(Enum):
    ANGLE        = auto()   # radians
    LENGTH       = auto()   # meters
    RATIO        = auto()   # dimensionless
    VELOCITY     = auto()   # rad/s or m/s — context dependent
    GENERIC      = auto()   # unknown/any — use sparingly

# ============================================================================
# CORE: DOMAIN
# Geometric truth — what a primitive can physically produce or accept
# None implies unbounded (±inf)
# ============================================================================

@dataclass(frozen=True)
class Domain:
    """
    Geometric validity envelope — declared by primitive, enforced by compositor.

    None implies unbounded in that direction.
    MechanicalLimits are compositor-level warnings, not primitive-level walls.

    input_unit / output_unit carry physical dimension for compatibility checks.
    """
    min:         Optional[float] = None
    max:         Optional[float] = None
    input_unit:  Dimension       = Dimension.GENERIC
    output_unit: Dimension       = Dimension.GENERIC

    def contains(self, x: float, tolerance: float = 1e-9) -> bool:
        if self.min is not None and x < self.min - tolerance:
            return False
        if self.max is not None and x > self.max + tolerance:
            return False
        return True

    def is_finite(self) -> bool:
        return self.min is not None and self.max is not None

# ============================================================================
# CORE: MECHANICAL LIMITS
# Compositor-level sanity thresholds — warning triggers, not hard walls
# Nothing physical exceeds these — if it does, it is a bug in the caller
# ============================================================================

class MechanicalLimits:
    MAX_ANGLE        = 4 * math.pi   # two full rotations
    MAX_RATIO        = 1000.0        # no real gearbox exceeds 1000:1
    MAX_DISPLACEMENT = 1e6           # 1km — mechanically absurd
    MAX_VELOCITY     = 1e4           # 10,000 rad/s — jet turbine territory
    TOLERANCE        = 1e-9          # float drift forgiveness

# ============================================================================
# CORE: PROTOCOLS
# ============================================================================
# Primitive          → float → float          (base, everything)
# Invertible         → float → float          (bidirectional)
# OneWay             → float → float          (forward only)
# Configurable       → mode switching         (Class III)
# Periodic           → float → float cyclic   (Class II)
# VectorPrimitive    → DEFERRED               (Class VII)

@runtime_checkable
class Primitive(Protocol):
    """
    Base contract for all MMP primitives.
    domain carries geometric truth — read by compositor, never enforced here.
    """
    domain: Domain
    is_invertible: bool

    def forward(self, x: float) -> float: ...

@runtime_checkable
class Invertible(Primitive, Protocol):
    """
    Bidirectional mapping — inverse mathematically guaranteed.
    is_invertible must be True.
    """
    domain: Domain
    is_invertible: bool

    def forward(self, x: float) -> float: ...
    def inverse(self, y: float) -> float: ...

@runtime_checkable
class OneWay(Primitive, Protocol):
    """
    Explicitly non-invertible — forward only, by mechanical law.
    is_invertible must be False — enforced by implementor.
    """
    domain: Domain
    is_invertible: bool

    def forward(self, x: float) -> float: ...

@runtime_checkable
class Configurable(Protocol):
    """
    Mode-switchable mechanisms — same hardware, different equations.
    available_modes() must return all valid mode strings.
    set_mode() must reject unknown modes loudly.
    """
    def set_mode(self, mode: str) -> None: ...
    def get_mode(self) -> str: ...
    def available_modes(self) -> list[str]: ...

@runtime_checkable
class Periodic(Primitive, Protocol):
    """
    Cyclic mechanisms — output repeats with known period.
    normalize() wraps any input to the canonical [0, period) window.
    """
    domain: Domain
    is_invertible: bool

    def forward(self, x: float) -> float: ...
    def period(self) -> float: ...
    def normalize(self, x: float) -> float: ...

@runtime_checkable
class PeriodicBijective(Periodic, Invertible, Protocol):
    """
    Periodic but fully invertible within one period.
    No branch selection needed — forward and inverse are
    the same structural form.
    Example: HookesJoint
    """
    domain: Domain
    is_invertible: bool
    is_analytically_invertible: bool

    def forward(self, x: float) -> float: ...
    def inverse(self, y: float) -> float: ...
    def period(self) -> float: ...
    def normalize(self, x: float) -> float: ...


@runtime_checkable
class PeriodicBranchDependent(Periodic, Invertible, Protocol):
    """
    Periodic and non-injective over full domain.
    Inverse requires branch selection — stored as instance state,
    not passed as parameter.
    theta_guess seeds numerical solvers — also instance state.
    is_analytically_invertible distinguishes closed-form vs numerical.
    Example: ScotchYoke, CrankSlider, EccentricCam
    """
    domain: Domain
    is_invertible: bool
    is_analytically_invertible: bool
    branch: str
    available_branches: list[str]
    theta_guess: float

    def forward(self, x: float) -> float: ...
    def inverse(self, y: float) -> float: ...
    def period(self) -> float: ...
    def normalize(self, x: float) -> float: ...

# ============================================================================
# CORE: EXCEPTIONS
# ============================================================================

class MMPError(Exception):
    """Base exception for all MMP errors"""

class CompositionError(MMPError):
    """Raised when two primitives cannot be legally connected"""

class DimensionMismatchError(CompositionError):
    """Output unit of stage N does not match input unit of stage N+1"""

class DomainViolationError(CompositionError):
    """Output range of stage N exceeds input domain of stage N+1"""

class InverseUndefinedError(MMPError):
    """Inverse requested on a non-invertible or lossy chain"""

# ============================================================================
# CORE: GOVERNOR
# Stack-level output clamp — wraps any Primitive
# Mechanically: centrifugal governor, pressure relief valve, torque limiter
# ============================================================================

@dataclass
class Governor:
    """
    Transparent stack wrapper — clamps output to [min_val, max_val].

    is_invertible is permanently False.
    Clamping is a lossy operation — information destroyed at limits
    cannot be recovered. The entire chain becomes OneWay the moment
    a Governor is inserted.

    Compositor reads is_invertible=False and marks chain as lossy.
    """
    primitive:  Primitive
    min_val:    float
    max_val:    float
    hysteresis: float = 0.0
    domain: Domain = field(init=False)
    
    def __post_init__(self):
        self.domain = Domain(...)
        if self.min_val >= self.max_val:
            raise ValueError(
                f"min_val ({self.min_val}) must be less than "
                f"max_val ({self.max_val})"
            )
        if self.hysteresis < 0:
            raise ValueError("Hysteresis must be non-negative")
        if self.hysteresis >= (self.max_val - self.min_val) / 2:
            raise ValueError(
                "Hysteresis too large — would invert the governed range"
            )
        self.domain = Domain(
            min=self.min_val,
            max=self.max_val,
            input_unit=self.primitive.domain.input_unit,
            output_unit=self.primitive.domain.output_unit
        )

    @property
    def is_invertible(self) -> bool:
        return False  # clamping destroys invertibility — always, permanently

    def forward(self, x: float) -> float:
        y = self.primitive.forward(x)
        return max(self.min_val, min(self.max_val, y))

# ============================================================================
# CLASS I: LINEAR SCALING (Affine Maps)
# Continuous, invertible, constant ratio
# Satisfies: Primitive, Invertible
# is_invertible: True (all Class I are bijective)
# ============================================================================

@dataclass
class SpurGear:
    """
    Simple gear ratio — pure linear scaling
    y = ratio * x
    forward:  x -> x * ratio
    inverse:  y -> y / ratio

    input_unit:  ANGLE  (input shaft rotation)
    output_unit: ANGLE  (output shaft rotation)
    """
    ratio: float

    def __post_init__(self):
        if self.ratio == 0:
            raise ValueError("Gear ratio cannot be zero — undefined inverse")
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.ANGLE
        )

    @property
    def is_invertible(self) -> bool:
        return True

    def forward(self, x: float) -> float:
        return x * self.ratio

    def inverse(self, y: float) -> float:
        return y / self.ratio


@dataclass
class CompoundGearTrain:
    """
    Multiple gears cascaded — true sequential application
    y = r1 * r2 * ... * rN * x
    forward:  applies each ratio left to right
    inverse:  applies each ratio right to left (reversed)

    input_unit:  ANGLE  (input shaft rotation)
    output_unit: ANGLE  (output shaft rotation)
    """
    ratios: list[float]

    def __post_init__(self):
        if not self.ratios:
            raise ValueError(
                "CompoundGearTrain requires at least one ratio"
            )
        if any(r == 0 for r in self.ratios):
            raise ValueError("All gear ratios must be non-zero")
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.ANGLE
        )

    @property
    def is_invertible(self) -> bool:
        return True

    def forward(self, x: float) -> float:
        result = x
        for r in self.ratios:
            result = SpurGear(r).forward(result)
        return result

    def inverse(self, y: float) -> float:
        result = y
        for r in reversed(self.ratios):
            result = SpurGear(r).inverse(result)
        return result


@dataclass
class RackAndPinion:
    """
    Rotation <-> Linear translation
    y = pitch_radius * theta
    forward:  rotation (radians) -> linear displacement
    inverse:  linear displacement -> rotation (radians)

    input_unit:  ANGLE   (input shaft rotation)
    output_unit: LENGTH  (linear displacement)

    Natural adapter: ANGLE -> LENGTH
    Compositor will require this primitive when bridging
    rotational and linear primitives in a chain.
    """
    pitch_radius: float

    def __post_init__(self):
        if self.pitch_radius <= 0:
            raise ValueError("Pitch radius must be positive")
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH
        )

    @property
    def is_invertible(self) -> bool:
        return True

    def forward(self, x: float) -> float:
        """Rotation (radians) -> linear displacement"""
        return self.pitch_radius * x

    def inverse(self, y: float) -> float:
        """Linear displacement -> rotation (radians)"""
        return y / self.pitch_radius


@dataclass
class Wedge:
    """
    Inclined plane — horizontal displacement to vertical lift
    y = x * tan(angle)
    forward:  horizontal displacement -> vertical displacement
    inverse:  vertical displacement -> horizontal displacement

    mechanical_advantage = cot(angle) = 1 / tan(angle)

    input_unit:  LENGTH  (horizontal displacement)
    output_unit: LENGTH  (vertical displacement)

    Domain: angle strictly in (0, π/2)
    At 0: output collapses — no lift
    At π/2: tan blows up — infinite slope
    """
    angle_rad: float

    def __post_init__(self):
        if not (0 < self.angle_rad < math.pi / 2):
            raise ValueError(
                "Wedge angle must be in (0, π/2) exclusive — "
                "at 0 no lift occurs, at π/2 tan is undefined"
            )
        self.domain = Domain(
            input_unit=Dimension.LENGTH,
            output_unit=Dimension.LENGTH
        )

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def mechanical_advantage(self) -> float:
        """Force amplification: MA = cot(angle) = 1/tan(angle)"""
        return 1.0 / math.tan(self.angle_rad)

    def forward(self, x: float) -> float:
        """Horizontal displacement -> vertical displacement"""
        return x * math.tan(self.angle_rad)

    def inverse(self, y: float) -> float:
        """Vertical displacement -> horizontal displacement"""
        return y / math.tan(self.angle_rad)


@dataclass
class OldhamCoupling:
    """
    Transmits rotation 1:1 between parallel offset shafts
    y = x  (identity on angle — ratio is always 1.0)
    forward:  input shaft angle -> output shaft angle
    inverse:  output shaft angle -> input shaft angle

    The physical shaft offset is a geometric property only.
    It does not enter the scalar mapping — angle is preserved exactly.
    shaft_offset exposes the physical translation for reference.

    input_unit:  ANGLE
    output_unit: ANGLE
    """
    offset_x: float
    offset_y: float

    def __post_init__(self):
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.ANGLE
        )

    @property
    def is_invertible(self) -> bool:
        return True

    def forward(self, x: float) -> float:
        """Input shaft angle -> output shaft angle (1:1)"""
        return x

    def inverse(self, y: float) -> float:
        """Output shaft angle -> input shaft angle (1:1)"""
        return y

    @property
    def shaft_offset(self) -> tuple[float, float]:
        """Physical offset between shaft centers — not part of scalar map"""
        return (self.offset_x, self.offset_y)

# ============================================================================
# CLASS II: PERIODIC NON-LINEAR (Trigonometric)
# Oscillatory, bounded, non-injective without domain restriction
# Satisfies: Primitive, Periodic
# Subdivided: PeriodicBijective | PeriodicBranchDependent
# ============================================================================

@dataclass
class ScotchYoke:
    """
    Crank and yoke — pure sinusoidal motion
    y = A * sin(x + phi)
    forward:  rotation angle -> linear displacement
    inverse:  linear displacement -> rotation angle

    Non-injective over R — branch stored as instance state.
    branch='principal'     -> inverse in [-π/2,  π/2]
    branch='supplementary' -> inverse in [ π/2, 3π/2]

    Analytical inverse exists — is_analytically_invertible = True

    input_unit:  ANGLE
    output_unit: LENGTH
    """
    amplitude: float
    phase:     float = 0.0
    branch:    str   = 'principal'

    def __post_init__(self):
        if self.amplitude == 0:
            raise ValueError(
                "Amplitude cannot be zero — output collapses to point"
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches}"
            )
        self.domain = Domain(
            min=-abs(self.amplitude),
            max= abs(self.amplitude),
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH
        )

    @property
    def available_branches(self) -> list[str]:
        return ['principal', 'supplementary']

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def is_analytically_invertible(self) -> bool:
        return True  # closed form via arcsin

    @property
    def theta_guess(self) -> float:
        return 0.0  # not used — analytical inverse

    def forward(self, x: float) -> float:
        """Rotation angle (radians) -> linear displacement"""
        return self.amplitude * math.sin(x + self.phase)

    def inverse(self, y: float) -> float:
        """
        Linear displacement -> rotation angle
        Branch determines which solution is returned — instance state
        """
        if abs(y) > abs(self.amplitude):
            raise ValueError(
                f"y={y} outside amplitude range "
                f"[{-abs(self.amplitude)}, {abs(self.amplitude)}]"
            )
        raw = math.asin(y / self.amplitude) - self.phase
        if self.branch == 'principal':
            return raw
        else:  # supplementary
            return math.pi - raw

    def period(self) -> float:
        return 2 * math.pi

    def normalize(self, x: float) -> float:
        return x % (2 * math.pi)


@dataclass
class EccentricCam:
    """
    Circular cam with follower — offset rotation to lift
    y = e*cos(θ) + sqrt(r² - (e*sin(θ))²)
    forward:  cam rotation angle -> follower displacement
    inverse:  follower displacement -> cam angle (Newton-Raphson)

    No closed form inverse — is_analytically_invertible = False
    theta_guess seeds the solver — instance state, selects branch

    input_unit:  ANGLE
    output_unit: LENGTH
    """
    eccentricity:    float
    follower_radius: float
    branch:          str   = 'principal'
    theta_guess:     float = 0.0

    def __post_init__(self):
        if self.eccentricity <= 0:
            raise ValueError("Eccentricity must be positive")
        if self.follower_radius <= self.eccentricity:
            raise ValueError(
                "Follower radius must exceed eccentricity — "
                "otherwise cam exceeds follower range"
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches}"
            )
        self.domain = Domain(
            min=self.follower_radius - self.eccentricity,
            max=self.follower_radius + self.eccentricity,
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH
        )

    @property
    def available_branches(self) -> list[str]:
        return ['principal', 'supplementary']

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def is_analytically_invertible(self) -> bool:
        return False  # Newton-Raphson required

    def _derivative(self, x: float) -> float:
        """Analytical derivative of forward — used by Newton-Raphson"""
        e, r = self.eccentricity, self.follower_radius
        sin_x = math.sin(x)
        cos_x = math.cos(x)
        return (
            -e * sin_x
            - (e**2 * sin_x * cos_x)
            / math.sqrt(r**2 - (e * sin_x)**2)
        )

    def forward(self, x: float) -> float:
        """Cam rotation angle (radians) -> follower displacement"""
        e, r = self.eccentricity, self.follower_radius
        return e * math.cos(x) + math.sqrt(r**2 - (e * math.sin(x))**2)

    """
    # VERSION 1 (bad)
    def inverse(self, y: float) -> float:
        # Follower displacement -> cam rotation angle
        # Newton-Raphson from theta_guess — instance state selects branch
        
        theta = self.theta_guess
        for _ in range(100):
            delta = (self.forward(theta) - y) / self._derivative(theta)
            theta -= delta
            theta = theta % (2 * math.pi)  # stay within one period
            if abs(delta) < 1e-10:
                return theta
        raise ValueError(
            f"Newton-Raphson failed to converge for y={y}, "
            f"theta_guess={self.theta_guess}"
        )
    
    # VERSION 2 (also bad...)
    def inverse(self, y: float) -> float:
        theta = self.theta_guess
        for _ in range(100):
            deriv = self._derivative(theta)
            if abs(deriv) < 1e-12:
                raise ValueError(
                    f"Newton-Raphson derivative near zero at theta={theta} "
                    f"— try a different theta_guess"
                )
            delta = (self.forward(theta) - y) / deriv
            theta -= delta
            theta = theta % (2 * math.pi)  # stay within one period
            if abs(delta) < 1e-10:
                return theta
        raise ValueError(
            f"Newton-Raphson failed to converge for y={y}, "
            f"theta_guess={self.theta_guess}"
        )
    """
    
    """ VERSION 3 (direction mistake...)
    def inverse(self, y: float) -> float:
        theta = self.theta_guess
        for _ in range(100):
            delta = (self.forward(theta) - y) / self._derivative(theta)
            theta -= delta
            if abs(delta) < 1e-10:
                # Normalize and then adjust to correct branch
                theta = self.normalize(theta)
                if self.branch == 'principal' and theta > math.pi:
                    theta -= 2*math.pi
                elif self.branch == 'supplementary' and theta < math.pi:
                    theta += math.pi
                return theta
        raise ValueError(...)
    """
    
    """ VERSION 4 (another dud...)
    def inverse(self, y: float) -> float:
        # First check if y is in domain
        if not self.domain.contains(y):
            raise ValueError(f"y={y} outside domain [{self.domain.min}, {self.domain.max}]")
        
        theta = self.theta_guess
        for _ in range(100):
            delta = (self.forward(theta) - y) / self._derivative(theta)
            theta -= delta
            if abs(delta) < 1e-10:
                # Normalize to [0, 2π)
                theta = self.normalize(theta)
                
                # Adjust to correct branch
                if self.branch == 'principal':
                    # Principal branch should be in [-π/2, π/2]
                    if theta > math.pi/2 and theta < 3*math.pi/2:
                        # We're in supplementary region - move to principal
                        theta = theta - math.pi if theta > math.pi else theta + math.pi
                    # Ensure in [-π/2, π/2] range
                    if theta > math.pi/2:
                        theta -= 2*math.pi
                        
                elif self.branch == 'supplementary':
                    # Supplementary branch should be in [π/2, 3π/2]
                    if theta < math.pi/2 or theta > 3*math.pi/2:
                        # We're in principal region - move to supplementary
                        theta = theta + math.pi if theta < math.pi/2 else theta - math.pi
                        
                return theta
                
        raise ValueError(
            f"Newton-Raphson failed to converge for y={y}, "
            f"theta_guess={self.theta_guess}"
        )
    """
    
    """ VERSION 5 (Another dud...)
    def inverse(self, y: float) -> float:
        # Domain check first — before any math
        if not self.domain.contains(y):
            raise ValueError(
                f"y={y} outside domain "
                f"[{self.domain.min}, {self.domain.max}]"
            )

        # Seed Newton-Raphson to the correct branch region
        # rather than trying to correct after convergence
        if self.theta_guess != 0.0:
            theta = self.theta_guess
        elif self.branch == 'principal':
            theta = 0.0          # seeds into [0, π] region
        else:  # supplementary
            theta = math.pi      # seeds into [π, 2π] region

        for _ in range(100):
            deriv = self._derivative(theta)
            if abs(deriv) < 1e-12:
                raise ValueError(
                    f"Newton-Raphson derivative near zero at theta={theta} "
                    f"— try a different theta_guess"
                )
            delta = (self.forward(theta) - y) / deriv
            theta -= delta
            if abs(delta) < 1e-10:
                # Normalize to [0, 2π)
                theta = self.normalize(theta)
                return theta

        raise ValueError(
            f"Newton-Raphson failed to converge for y={y}, "
            f"theta_guess={self.theta_guess}"
        )
    """
    
    def inverse(self, y: float) -> float:
        """
        Follower displacement -> cam rotation angle via Newton-Raphson.
        
        CRITICAL: Must constrain iteration to branch region to prevent
        convergence to the wrong symmetric solution.
        
        EccentricCam forward is symmetric: forward(θ) == forward(2π - θ)
        Branch selection determines which of the two solutions is returned.
        """
        # 1. Domain check FIRST — before any math
        if not self.domain.contains(y):
            raise ValueError(
                f"y={y} outside domain [{self.domain.min}, {self.domain.max}]"
            )
        
        # 2. Check for extrema — derivative is zero at min/max displacement
        #    At these points, only one solution exists (θ=0 or θ=π)
        if abs(y - self.domain.max) < MechanicalLimits.TOLERANCE: # was 'self.domain.tolerance'
            return 0.0  # Maximum displacement at θ=0
        if abs(y - self.domain.min) < MechanicalLimits.TOLERANCE: # was 'self.domain.tolerance'
            return math.pi  # Minimum displacement at θ=π
        
        # 3. Seed to correct branch region
        if self.theta_guess != 0.0:
            theta = self.theta_guess
        elif self.branch == 'principal':
            # Principal branch: [0, π] — peak at θ=0
            theta = 0.5  # Seed away from extrema
        else:  # supplementary
            # Supplementary branch: [π, 2π] — peak at θ=2π (equiv to 0)
            theta = math.pi + 0.5  # Seed in middle of region
        
        # 4. Newton-Raphson with branch constraints
        for iteration in range(100):
            deriv = self._derivative(theta)
            
            # Handle near-zero derivative — project to nearest branch boundary
            if abs(deriv) < 1e-12:
                if self.branch == 'principal':
                    theta = 0.0 if theta < math.pi else math.pi
                else:
                    theta = math.pi if theta < 1.5 * math.pi else 2 * math.pi
                break
            
            delta = (self.forward(theta) - y) / deriv
            
            # 5. CRITICAL: Constrain step to prevent branch jumping
            #    Limit step size to stay within branch region
            max_step = math.pi / 4  # Don't jump more than 45° per iteration
            delta = max(-max_step, min(max_step, delta))
            
            theta -= delta
            
            # 6. Project theta back to branch region if it escapes
            if self.branch == 'principal':
                # Keep in [0, π]
                theta = max(0.0, min(math.pi, theta))
            else:
                # Keep in [π, 2π]
                theta = max(math.pi, min(2 * math.pi, theta))
            
            # 7. Convergence check
            if abs(delta) < 1e-10:
                return self.normalize(theta)
        
        # 8. Final validation — ensure forward(inverse(y)) ≈ y
        result = self.normalize(theta)
        if abs(self.forward(result) - y) > 1e-6:
            raise ValueError(
                f"Newton-Raphson converged but forward(result)={self.forward(result)} "
                f"does not match y={y} — try different theta_guess"
            )
        
        return result
    
    def period(self) -> float:
        return 2 * math.pi

    def normalize(self, x: float) -> float:
        return x % (2 * math.pi)


@dataclass
class CrankSlider:
    """
    Crank-slider mechanism — crank rotation to slider translation
    y = r*cos(θ) + sqrt(L² - (r*sin(θ))²)
    forward:  crank angle -> slider position
    inverse:  slider position -> crank angle (Newton-Raphson)

    No closed form inverse — is_analytically_invertible = False
    theta_guess seeds the solver — instance state, selects branch
    approximate() available for L >> r regime — not part of protocol

    input_unit:  ANGLE
    output_unit: LENGTH
    """
    crank_length: float
    rod_length:   float
    branch:       str   = 'principal'
    theta_guess:  float = 0.0

    def __post_init__(self):
        if self.crank_length <= 0:
            raise ValueError("Crank length must be positive")
        if self.rod_length <= self.crank_length:
            raise ValueError(
                "Rod length must exceed crank length — "
                "otherwise slider cannot complete full rotation"
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches}"
            )
        self.domain = Domain(
            min=self.rod_length - self.crank_length,
            max=self.rod_length + self.crank_length,
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH
        )

    @property
    def available_branches(self) -> list[str]:
        return ['principal', 'supplementary']

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def is_analytically_invertible(self) -> bool:
        return False  # Newton-Raphson required

    def _derivative(self, x: float) -> float:
        """Analytical derivative of forward — used by Newton-Raphson"""
        r, L = self.crank_length, self.rod_length
        sin_x = math.sin(x)
        cos_x = math.cos(x)
        return (
            -r * sin_x
            - (r**2 * sin_x * cos_x)
            / math.sqrt(L**2 - (r * sin_x)**2)
        )

    def forward(self, x: float) -> float:
        """Crank angle (radians) -> slider position"""
        r, L = self.crank_length, self.rod_length
        return r * math.cos(x) + math.sqrt(L**2 - (r * math.sin(x))**2)

    def inverse(self, y: float) -> float:
        """
        Slider position -> crank angle via Newton-Raphson.

        CrankSlider forward is symmetric: forward(θ) == forward(2π - θ)
        Branch region constrains iteration — prevents convergence to wrong solution.

        Correctness criterion: forward(inverse(y)) == y within branch region
        Not: inverse(forward(x)) == x  (not guaranteed due to symmetry)
        """
        
        # 1. Domain check first — before any math
        if not self.domain.contains(y):
            raise ValueError(
                f"y={y} outside domain "
                f"[{self.domain.min}, {self.domain.max}]"
            )

        # 2. Extrema bypass — derivative is zero at max/min displacement
        #    Only one solution exists at these points
        if abs(y - self.domain.max) < MechanicalLimits.TOLERANCE:
            return 0.0       # Maximum displacement at θ=0
        if abs(y - self.domain.min) < MechanicalLimits.TOLERANCE:
            return math.pi   # Minimum displacement at θ=π

        # 3. Seed to correct branch region
        if self.theta_guess != 0.0:
            theta = self.theta_guess
        elif self.branch == 'principal':
            theta = 0.5              # Seeds into [0, π]
        else:
            theta = math.pi + 0.5   # Seeds into [π, 2π]

        # 4. Newton-Raphson with branch constraints
        for _ in range(100):
            deriv = self._derivative(theta)

            if abs(deriv) < 1e-12:
                raise ValueError(
                    f"Newton-Raphson derivative near zero at theta={theta} "
                    f"— try a different theta_guess"
                )

            delta = (self.forward(theta) - y) / deriv

            # 5. Limit step size — prevents single-step branch jumping
            max_step = math.pi / 4
            delta = max(-max_step, min(max_step, delta))

            theta -= delta

            # 6. Clamp to branch region — iteration cannot escape
            if self.branch == 'principal':
                theta = max(0.0, min(math.pi, theta))
            else:
                theta = max(math.pi, min(2 * math.pi, theta))

            # 7. Convergence check
            if abs(delta) < 1e-10:
                return self.normalize(theta)

        # 8. Final validation
        result = self.normalize(theta)
        if abs(self.forward(result) - y) > 1e-6:
            raise ValueError(
                f"Newton-Raphson converged but forward(result)={self.forward(result)} "
                f"does not match y={y} — try different theta_guess"
            )
        return result

    def approximate(self, x: float) -> float:
        """
        Small-angle approximation for L >> r regime
        y ≈ L + r*cos(θ) - (r²/2L)*sin²(θ)
        Valid when r/L << 1 — not part of encode/decode protocol
        """
        r, L = self.crank_length, self.rod_length
        return L + r * math.cos(x) - (r**2 / (2 * L)) * math.sin(x)**2

    def period(self) -> float:
        return 2 * math.pi

    def normalize(self, x: float) -> float:
        return x % (2 * math.pi)


@dataclass
class HookesJoint:
    """
    Universal joint — angled shaft transmission
    tan(θ_out) = cos(α) * tan(θ_in)
    forward:  input shaft angle -> output shaft angle
    inverse:  output shaft angle -> input shaft angle

    Fully invertible within period — PeriodicBijective
    No branch selection needed — inverse is same form, reciprocal cos(α)
    Analytical inverse — is_analytically_invertible = True

    Singularity at θ_in = π/2 + nπ — tan blows up
    Shaft angle must be in [0, π/2) — at π/2 joint locks

    input_unit:  ANGLE
    output_unit: ANGLE
    """
    shaft_angle: float

    def __post_init__(self):
        if not (0 <= self.shaft_angle < math.pi / 2):
            raise ValueError(
                "Shaft angle must be in [0, π/2) — "
                "at π/2 the joint locks (cos=0)"
            )
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.ANGLE
        )

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def is_analytically_invertible(self) -> bool:
        return True  # closed form, same structural form both directions

    @property
    def branch(self) -> str:
        return 'none'  # PeriodicBijective — no branch needed

    @property
    def available_branches(self) -> list[str]:
        return ['none']

    @property
    def theta_guess(self) -> float:
        return 0.0  # not used — analytical inverse

    def forward(self, x: float) -> float:
        """Input shaft angle (radians) -> output shaft angle (radians)"""
        return math.atan2(
            math.sin(x) * math.cos(self.shaft_angle),
            math.cos(x)
        )

    def inverse(self, y: float) -> float:
        """
        Output shaft angle (radians) -> input shaft angle (radians)
        Inverse form: tan(θ_in) = tan(θ_out) / cos(α)
        """
        return math.atan2(
            math.sin(y) / math.cos(self.shaft_angle),
            math.cos(y)
        )

    def angular_velocity_ratio(self, x: float) -> float:
        """
        ω_out / ω_in at given input angle
        Exposes velocity fluctuation — characteristic of Hooke's joints
        Not part of encode/decode protocol
        """
        cos_x = math.cos(x)
        sin_x = math.sin(x)
        denom = cos_x**2 + (sin_x * math.cos(self.shaft_angle))**2
        return math.cos(self.shaft_angle) / denom

    def period(self) -> float:
        return 2 * math.pi

    def normalize(self, x: float) -> float:
        return x % (2 * math.pi)