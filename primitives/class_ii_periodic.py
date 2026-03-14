from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.base import Domain, Dimension
from core.exceptions import MMPError
from core.constants import MechanicalLimits

# Utils (not implementation yet)
#from utils.math_helpers import newton_raphson, unwrap_angle
#from utils.validation import validate_branch, validate_range


"""
    ============================================================================
    CLASS II: PERIODIC NON-LINEAR (Trigonometric)
    Oscillatory, bounded, non-injective without domain restriction
    Satisfies: Primitive, Periodic
    Subdivided: PeriodicBijective | PeriodicBranchDependent
    ============================================================================
"""

@dataclass(frozen=True)
class ScotchYoke:
    amplitude: float
    phase: float = 0.0
    branch: str = 'principal'
    theta_guess: float = 0.0
    is_analytically_invertible: bool = True
    is_monotonic: bool = False
    domain: Domain = field(init=False)

    def __post_init__(self):
        if self.amplitude == 0:
            raise ValueError("Amplitude cannot be zero ")
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches} "
            )
        # Set domain explicitly
        object.__setattr__(self, 'domain', Domain(
            min=None,
            max=None,
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH,
            output_min=-abs(self.amplitude),
            output_max=abs(self.amplitude),
        ))

    @property
    def is_invertible(self) -> bool:
        return True

    def forward(self, x: float) -> float:
        return self.amplitude * math.sin(x + self.phase)

    def derivative(self, x: float) -> float:
        return self.amplitude * math.cos(x + self.phase)

    def inverse(self, y: float, branch: Optional[str] = None, 
                guess: Optional[float] = None) -> float:
        """Branch passed explicitly, defaults to self.branch"""
        active_branch = branch if branch is not None else self.branch
        
        if active_branch not in ['principal', 'supplementary']:
            raise ValueError(f"Invalid branch: {active_branch} ")
        if abs(y) > abs(self.amplitude):
            raise ValueError(f"y={y} outside amplitude range ")
        
        raw = math.asin(y / self.amplitude) - self.phase
        if active_branch == 'principal':
            return raw
        return math.pi - raw

    @property
    def available_branches(self) -> list[str]:
        return ['principal', 'supplementary']

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
    """
    eccentricity:    float
    follower_radius: float
    branch:          str   = 'principal'
    theta_guess:     float = 0.0
    is_analytically_invertible: bool = False
    is_monotonic: bool = False
    domain: Domain = field(init=False)

    def __post_init__(self):
        if self.eccentricity <= 0:
            raise ValueError("Eccentricity must be positive ")
        if self.follower_radius <= self.eccentricity:
            raise ValueError(
                "Follower radius must exceed eccentricity —  "
                "otherwise cam exceeds follower range "
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches} "
            )
        # Set domain explicitly
        self.domain = Domain(
            min=None,
            max=None,
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH,
            output_min=self.follower_radius - self.eccentricity,
            output_max=self.follower_radius + self.eccentricity,
        )

    @property
    def available_branches(self) -> list[str]:
        return ['principal', 'supplementary']

    @property
    def is_invertible(self) -> bool:
        return True

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

    def derivative(self, x: float) -> float:
        """Public derivative required by Primitive protocol"""
        return self._derivative(x)

    def forward(self, x: float) -> float:
        """Cam rotation angle (radians) -> follower displacement"""
        e, r = self.eccentricity, self.follower_radius
        return e * math.cos(x) + math.sqrt(r**2 - (e * math.sin(x))**2)

    def inverse(self, y: float, branch: Optional[str] = None, 
                guess: Optional[float] = None) -> float:
        """
        Follower displacement -> cam rotation angle via Newton-Raphson.
        Uses active_branch to constrain iteration.
        """
        active_branch = branch if branch is not None else self.branch
        active_guess = guess if guess is not None else self.theta_guess

        # 1. Domain check FIRST — before any math.
        #    Use output_min/output_max (the bounded output range), not domain.min/max
        #    (which are now None — the input angle is unbounded).
        out_lo = self.domain.output_min
        out_hi = self.domain.output_max
        if not self.domain.output_contains(y):
            raise ValueError(
                f"y={y} outside output range [{out_lo}, {out_hi}] "
            )
        
        # 2. Check for extrema — derivative is zero at min/max displacement
        #    At these points, only one solution exists (θ=0 or θ=π)
        if abs(y - out_hi)  < MechanicalLimits.TOLERANCE:
            return 0.0  # Maximum displacement at θ=0
        if abs(y - out_lo)  < MechanicalLimits.TOLERANCE:
            return math.pi  # Minimum displacement at θ=π
        
        # 3. Seed to correct branch region
        if active_guess != 0.0:
            theta = active_guess
        elif active_branch == 'principal':
            # Principal branch: [0, π] — peak at θ=0
            theta = 0.5  # Seed away from extrema
        else:  # supplementary
            # Supplementary branch: [π, 2π] — peak at θ=2π (equiv to 0)
            theta = math.pi + 0.5  # Seed in middle of region
        
        # 4. Newton-Raphson with branch constraints
        for iteration in range(100):
            deriv = self._derivative(theta)
            
            # Handle near-zero derivative — project to nearest branch boundary
            if abs(deriv)  < 1e-12:
                if active_branch == 'principal':
                    theta = 0.0 if theta  < math.pi else math.pi
                else:
                    theta = math.pi if theta  < 1.5 * math.pi else 2 * math.pi
                break
            
            delta = (self.forward(theta) - y) / deriv
            
            # 5. CRITICAL: Constrain step to prevent branch jumping
            #    Limit step size to stay within branch region
            max_step = math.pi / 4  # Don't jump more than 45° per iteration
            delta = max(-max_step, min(max_step, delta))
            
            theta -= delta
            
            # 6. Project theta back to branch region if it escapes
            if active_branch == 'principal':
                # Keep in [0, π]
                theta = max(0.0, min(math.pi, theta))
            else:
                # Keep in [π, 2π]
                theta = max(math.pi, min(2 * math.pi, theta))
            
            # 7. Convergence check
            if abs(delta)  < 1e-10:
                return self.normalize(theta)
        
        # 8. Final validation — ensure forward(inverse(y)) ≈ y
        result = self.normalize(theta)
        if abs(self.forward(result)  - y)  > 1e-6:
            raise ValueError(
                f"Newton-Raphson converged but forward(result)={self.forward(result)}  "
                f"does not match y={y} — try different theta_guess "
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
    """
    crank_length: float
    rod_length:   float
    branch:       str   = 'principal'
    theta_guess:  float = 0.0
    is_analytically_invertible: bool = False
    is_monotonic: bool = False
    domain: Domain = field(init=False)

    def __post_init__(self):
        if self.crank_length <= 0:
            raise ValueError("Crank length must be positive ")
        if self.rod_length <= self.crank_length:
            raise ValueError(
                "Rod length must exceed crank length —  "
                "otherwise slider cannot complete full rotation "
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches} "
            )
        # Set domain explicitly
        self.domain = Domain(
            min=None,
            max=None,
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH,
            output_min=self.rod_length - self.crank_length,
            output_max=self.rod_length + self.crank_length,
        )

    @property
    def available_branches(self) -> list[str]:
        return ['principal', 'supplementary']

    @property
    def is_invertible(self) -> bool:
        return True

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

    def derivative(self, x: float) -> float:
        """Public derivative required by Primitive protocol"""
        return self._derivative(x)

    def forward(self, x: float) -> float:
        """Crank angle (radians) -> slider position"""
        r, L = self.crank_length, self.rod_length
        return r * math.cos(x) + math.sqrt(L**2 - (r * math.sin(x))**2)

    def inverse(self, y: float, branch: Optional[str] = None, 
            guess: Optional[float] = None) -> float:
        """
        Slider position -> crank angle via Newton-Raphson.
        Uses active_branch to constrain iteration.

        """
        
        active_branch = branch if branch is not None else self.branch
        active_guess = guess if guess is not None else self.theta_guess

        # 1. Domain check FIRST — before any math.
        out_lo = self.domain.output_min
        out_hi = self.domain.output_max
        if not self.domain.output_contains(y):
            raise ValueError(
                f"y={y} outside output range [{out_lo}, {out_hi}]"
            )

        # 2. Extrema bypass — derivative is zero at max/min displacement
        #    Only one solution exists at these points
        if abs(y - out_hi)  < MechanicalLimits.TOLERANCE:
            return 0.0       # Maximum displacement at θ=0
        if abs(y - out_lo)  < MechanicalLimits.TOLERANCE:
            return math.pi   # Minimum displacement at θ=π

        # 3. Seed to correct branch region
        if active_guess != 0.0:
            theta = active_guess
        elif active_branch == 'principal':
            theta = 0.5              # Seeds into [0, π]
        else:
            theta = math.pi + 0.5   # Seeds into [π, 2π]

        # 4. Newton-Raphson with branch constraints
        for _ in range(100):
            deriv = self._derivative(theta)

            if abs(deriv)  < 1e-12:
                raise ValueError(
                    f"Newton-Raphson derivative near zero at theta={theta}  "
                    f"— try a different theta_guess "
                )

            delta = (self.forward(theta) - y) / deriv

            # 5. Limit step size — prevents single-step branch jumping
            max_step = math.pi / 4
            delta = max(-max_step, min(max_step, delta))

            theta -= delta

            # 6. Clamp to branch region — iteration cannot escape
            if active_branch == 'principal':
                theta = max(0.0, min(math.pi, theta))
            else:
                theta = max(math.pi, min(2 * math.pi, theta))

            # 7. Convergence check
            if abs(delta)  < 1e-10:
                return self.normalize(theta)

        # 8. Final validation
        result = self.normalize(theta)
        if abs(self.forward(result) - y)  > 1e-6:
            raise ValueError(
                f"Newton-Raphson converged but forward(result)={self.forward(result)}  "
                f"does not match y={y} — try different theta_guess "
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
    Universal joint, angled shaft transmission
    tan(θ_out) = cos(α) * tan(θ_in)
    FIX: Uses atan2 for stability at pi/2, with unwrapping for continuity.
    """
    shaft_angle: float
    is_analytically_invertible: bool = True
    branch: str = 'none'
    theta_guess: float = 0.0
    is_monotonic: bool = False  # Technically monotonic over 2pi, but velocity fluctuates

    def __post_init__(self):
        if not (0  <= self.shaft_angle  < math.pi / 2):
            raise ValueError(
                "Shaft angle must be in [0, π/2) —  "
                "at π/2 the joint locks (cos=0) "
            )
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.ANGLE
        )

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def available_branches(self) -> list[str]:
        return ['none']

    def forward(self, x: float) -> float:
        """
        Input shaft angle (radians) -> output shaft angle (radians)
        Stable implementation using atan2 to avoid tan(pi/2) singularity.
        Unwraps result to match input continuity.
        """
        cos_alpha = math.cos(self.shaft_angle)
        # Stable calculation: atan2(y, x) handles cos(x)=0 gracefully
        raw = math.atan2(cos_alpha * math.sin(x), math.cos(x))
        
        # Unwrap to match input x continuity (preserve rotation count)
        # We expect output to be roughly close to input (scaled ~1.0)
        two_pi = 2 * math.pi
        k = round((x - raw) / two_pi)
        return raw + k * two_pi

    def inverse(self, y: float, branch: Optional[str] = None,
                guess: Optional[float] = None) -> float:
        """
        Output shaft angle (radians) -> input shaft angle (radians)
        Inverse form: tan(θ_in) = tan(θ_out) / cos(α)
        """
        cos_alpha = math.cos(self.shaft_angle)
        if abs(cos_alpha) < MechanicalLimits.TOLERANCE:
            raise ValueError("Joint near lock-up (alpha ~ 90 deg)")
            
        # Stable calculation
        raw = math.atan2(math.sin(y) / cos_alpha, math.cos(y))
        
        # Unwrap to match output y continuity
        two_pi = 2 * math.pi
        k = round((y - raw) / two_pi)
        return raw + k * two_pi

    def derivative(self, x: float) -> float:
        """
        Derivative of output angle with respect to input angle.
        Angular velocity ratio.
        """
        cos_x = math.cos(x)
        sin_x = math.sin(x)
        cos_alpha = math.cos(self.shaft_angle)
        denom = cos_x**2 + (sin_x * cos_alpha)**2
        # Avoid division by zero
        if abs(denom) < MechanicalLimits.TOLERANCE:
            return 0.0
        return cos_alpha / denom

    def angular_velocity_ratio(self, x: float) -> float:
        """ω_out / ω_in at given input angle"""
        return self.derivative(x)

    def period(self) -> float:
        return 2 * math.pi

    def normalize(self, x: float) -> float:
        return x % (2 * math.pi)