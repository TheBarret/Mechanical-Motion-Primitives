from __future__ import annotations
"""
Mechanical Motion Primitives
Behavioral classification-based implementation of mechanical-to-mathematical mappings
env: Python 3.10.10
"""

import math
from abc            import ABC, abstractmethod
from dataclasses    import dataclass, field
from typing         import Protocol, Tuple, Optional, Callable, Any
from typing         import runtime_checkable
from enum           import Enum, auto

"""
============================================================================
CORE: MECHANICAL LIMITS - Physical constants
============================================================================
"""

class Dimension(Enum):
    ANGLE        = auto()   # radians
    LENGTH       = auto()   # meters
    RATIO        = auto()   # dimensionless
    VELOCITY     = auto()   # rad/s or m/s — context dependent
    GENERIC      = auto()   # unknown/any — use sparingly

"""
============================================================================
CORE: DOMAIN
Geometric truth — what a primitive can physically produce or accept
None implies unbounded (±inf)
============================================================================
"""

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

"""
============================================================================
CORE: MECHANICAL LIMITS
Compositor-level sanity thresholds — warning triggers, not hard walls
Nothing physical exceeds these — if it does, it is a bug in the caller
============================================================================
"""

class MechanicalLimits:
    MAX_ANGLE        = 4 * math.pi   # two full rotations
    MAX_RATIO        = 1000.0        # no real gearbox exceeds 1000:1
    MAX_DISPLACEMENT = 1e6           # 1km — mechanically absurd
    MAX_VELOCITY     = 1e4           # 10,000 rad/s — jet turbine territory
    TOLERANCE        = 1e-9          # float drift forgiveness

"""
============================================================================
CORE: PROTOCOLS
============================================================================

 Primitive          → float → float          (base, everything)
 Invertible         → float → float          (bidirectional)
 OneWay             → float → float          (forward only)
 Periodic           → float → float cyclic   (Class II)
 Configurable       → mode switching         (Class III)
 VectorPrimitive    → DEFERRED               (Class VII)
"""

@runtime_checkable
class Primitive(Protocol):
    """
    Base contract for all MMP primitives.
    domain carries geometric truth — read by compositor, never enforced here.
    """
    domain: Domain
    is_invertible: bool
    is_monotonic: bool
    
    def forward(self, x: float) -> float: ...
    def derivative(self, x: float) -> float: ...

@runtime_checkable
class Invertible(Primitive, Protocol):
    """
    Bidirectional mapping — inverse mathematically guaranteed.
    is_invertible must be True.
    """
    domain: Domain
    is_invertible: bool
    
    def forward(self, x: float) -> float: ...
    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float: ...
    
    @property
    def available_branches(self) -> list[str]: ...

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
    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float: ...
    def period(self) -> float: ...
    def normalize(self, x: float) -> float: ...

@runtime_checkable
class PeriodicBranchDependent(Periodic, Invertible, Protocol):
    """
    Periodic and non-injective over full domain.
    Inverse requires branch selection - passed as parameter, not stored.
    """
    domain: Domain
    is_invertible: bool
    is_analytically_invertible: bool
    branch: str
    theta_guess: float
    
    def forward(self, x: float) -> float: ...
    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float: ...
    def period(self) -> float: ...
    def normalize(self, x: float) -> float: ...
    
    @property
    def available_branches(self) -> list[str]: ...
    
""" 
    ============================================================================
    CORE: EXCEPTIONS
    ============================================================================
""" 

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
    
"""
============================================================================
CORE: Composite Primitive
============================================================================
"""

@dataclass(frozen=True)
class CompositePrimitive:
    """
    Immutable chain of primitives.
    All validation happens at construction.
    """
    primitives: tuple[Primitive, ...]

    def __post_init__(self):
        # Validate unit compatibility
        for i in range(len(self.primitives) - 1):
            out_unit = self.primitives[i].domain.output_unit
            in_unit =  self.primitives[i+1].domain.input_unit
            if not self._units_compatible(out_unit, in_unit):
                raise DimensionMismatchError(
                    f"Stage {i} output {out_unit} != Stage {i+1} input {in_unit} "
                )
        # Compute and store derived properties
        object.__setattr__(self, '_input_domain', self._compute_input_domain())
        object.__setattr__(self, '_output_domain', self._compute_output_domain())
        object.__setattr__(self, '_period', self._compute_period())

    @property
    def domain(self) -> Domain:
        """Required by Primitive protocol"""
        return Domain(
            min=self._input_domain.min,      # Back-propagated input bounds
            max=self._input_domain.max,
            input_unit=self.primitives[0].domain.input_unit,
            output_unit=self.primitives[-1].domain.output_unit
        )

    @property
    def is_invertible(self) -> bool:
        return all(p.is_invertible for p in self.primitives)

    @property
    def input_domain(self) -> Domain:
        """Valid input range (back-propagated constraints)"""
        return self._input_domain

    @property
    def output_domain(self) -> Domain:
        """Achievable output range"""
        return self._output_domain

    def forward(self, x: float) -> float:
        """Apply chain forward with input validation warning"""
        if not self._input_domain.contains(x):
            pass # Warning suppressed per original design
        result = x
        for p in self.primitives:
            result = p.forward(result)
        return result

    def derivative(self, x: float) -> float:
        """Chain rule: dy/dx = fₙ'(...f₂'(f₁'(x))...)"""
        deriv = 1.0
        current = x
        intermediates = []
        # Forward pass to get intermediate values
        for p in self.primitives:
            intermediates.append(current)
            current = p.forward(current)
        # Backward pass applying derivatives
        for i, p in enumerate(reversed(self.primitives)):
            deriv *= p.derivative(intermediates[-(i+1)])
        return deriv

    def inverse(self, y: float, branches: Optional[list[str]] = None, guesses: Optional[list[float]] = None) -> float:
        """
        Inverse with per-stage branch context.
        """
        if not self.is_invertible:
            raise InverseUndefinedError("Chain is not invertible")
        result = y
        # Iterate backwards through primitives
        for i, p in enumerate(reversed(self.primitives)):
            idx = len(self.primitives) - 1 - i
            branch = branches[idx] if branches and idx < len(branches) else None
            guess = guesses[idx] if guesses and idx < len(guesses) else None
            if hasattr(p, 'inverse'):
                result = p.inverse(result, branch=branch, guess=guess)
            else:
                raise InverseUndefinedError(
                    f"Primitive {p} claims invertible but lacks inverse method "
                )
        return result

    def period(self) -> Optional[float]:
        """LCM of all periodic primitives, or None if aperiodic"""
        return self._period

    # ----------------------------------------------------------------------
    # FIX 1: Added missing helper method
    # ----------------------------------------------------------------------
    @staticmethod
    def _generate_sample_points(min_val: Optional[float], max_val: Optional[float], num_samples: int = 100) -> list[float]:
        """Generate linearly spaced sample points within bounds."""
        if min_val is None or max_val is None:
            # Fallback for unbounded domains (use mechanical limits)
            min_val = -MechanicalLimits.MAX_ANGLE
            max_val = MechanicalLimits.MAX_ANGLE
        
        if min_val > max_val:
            return []
            
        if num_samples <= 1:
            return [(min_val + max_val) / 2] if min_val != max_val else [min_val]

        step = (max_val - min_val) / (num_samples - 1)
        return [min_val + i * step for i in range(num_samples)]

    def _compute_input_domain(self) -> Domain:
        """
        Back-propagate constraints to find valid input range.
        FIX 2: Checks is_monotonic before analytic inversion.
        """
        if not self.primitives:
            return Domain(input_unit=Dimension.GENERIC, output_unit=Dimension.GENERIC)

        current_min = self.primitives[0].domain.min
        current_max = self.primitives[0].domain.max

        for i, p in enumerate(self.primitives[1:], start=1):
            if p.domain.min is None and p.domain.max is None:
                continue
            
            prefix = self.primitives[:i]
            
            # FIX: Check monotonicity. Invertible != Monotonic (e.g. ScotchYoke)
            all_invertible = all(getattr(pp, 'is_invertible', False) for pp in prefix)
            all_monotonic = all(getattr(pp, 'is_monotonic', False) for pp in prefix)

            constraints = []
            # Only use analytic inversion if chain is BOTH invertible AND monotonic
            if all_invertible and all_monotonic:
                if p.domain.min is not None:
                    try:
                        y = p.domain.min
                        for pp in reversed(prefix):
                            if hasattr(pp, 'inverse'):
                                y = pp.inverse(y)
                            else:
                                raise ValueError(f"Primitive {pp} lacks inverse")
                        constraints.append(y)
                    except Exception:
                        pass
                if p.domain.max is not None:
                    try:
                        y = p.domain.max
                        for pp in reversed(prefix):
                            if hasattr(pp, 'inverse'):
                                y = pp.inverse(y)
                            else:
                                raise ValueError(f"Primitive {pp} lacks inverse")
                        constraints.append(y)
                    except Exception:
                        pass
            
            # If analytic failed or not monotonic, force sampling
            if not constraints:
                sample_points = self._generate_sample_points(current_min, current_max)
                feasible_inputs = []
                for x in sample_points:
                    try:
                        val = x
                        for j in range(i):
                            val = self.primitives[j].forward(val)
                        if p.domain.contains(val):
                            feasible_inputs.append(x)
                    except Exception:
                        continue
                
                if feasible_inputs:
                    constraints = [min(feasible_inputs), max(feasible_inputs)]
                else:
                    # No feasible inputs
                    return Domain(min=float('inf'), max=-float('inf'),
                                  input_unit=self.primitives[0].domain.input_unit,
                                  output_unit=self.primitives[0].domain.output_unit)

            if constraints:
                new_min = min(constraints)
                new_max = max(constraints)
                if current_min is not None:
                    current_min = max(current_min, new_min)
                else:
                    current_min = new_min
                if current_max is not None:
                    current_max = min(current_max, new_max)
                else:
                    current_max = new_max

            if current_min is not None and current_max is not None:
                if current_min > current_max + MechanicalLimits.TOLERANCE:
                    return Domain(min=float('inf'), max=-float('inf'),
                                  input_unit=self.primitives[0].domain.input_unit,
                                  output_unit=self.primitives[0].domain.output_unit)

        return Domain(
            min=None if current_min in (None, float('inf'), -float('inf')) else current_min,
            max=None if current_max in (None, float('inf'), -float('inf')) else current_max,
            input_unit=self.primitives[0].domain.input_unit,
            output_unit=self.primitives[0].domain.output_unit
        )

    def _compute_output_domain(self) -> Domain:
        """
        Forward-propagate to find achievable output range.
        """
        if not self.primitives:
            return Domain(input_unit=Dimension.GENERIC, output_unit=Dimension.GENERIC)

        current_min = self.primitives[0].domain.min
        current_max = self.primitives[0].domain.max

        if current_min is None or current_max is None:
            current_min = -MechanicalLimits.MAX_ANGLE
            current_max = MechanicalLimits.MAX_ANGLE

        for i, p in enumerate(self.primitives):
            try:
                if getattr(p, 'is_monotonic', False):
                    y_min = p.forward(current_min)
                    y_max = p.forward(current_max)
                    current_min = min(y_min, y_max)
                    current_max = max(y_min, y_max)
                else:
                    samples = self._generate_sample_points(current_min, current_max, num_samples=50)
                    outputs = [p.forward(x) for x in samples if self._input_feasible(x, i)]
                    if outputs:
                        current_min = min(outputs)
                        current_max = max(outputs)
                    else:
                        return Domain(min=float('inf'), max=-float('inf'),
                                      input_unit=self.primitives[0].domain.input_unit,
                                      output_unit=self.primitives[-1].domain.output_unit)
            except Exception:
                if p.domain.min is not None:
                    current_min = max(current_min, p.domain.min) if current_min is not None else p.domain.min
                if p.domain.max is not None:
                    current_max = min(current_max, p.domain.max) if current_max is not None else p.domain.max

        last = self.primitives[-1]
        if last.domain.min is not None:
            current_min = max(current_min, last.domain.min) if current_min is not None else last.domain.min
        if last.domain.max is not None:
            current_max = min(current_max, last.domain.max) if current_max is not None else last.domain.max

        if current_min is not None and current_max is not None:
            if current_min > current_max + MechanicalLimits.TOLERANCE:
                return Domain(min=float('inf'), max=-float('inf'),
                              input_unit=self.primitives[0].domain.input_unit,
                              output_unit=self.primitives[-1].domain.output_unit)

        return Domain(
            min=None if current_min in (None, float('inf'), -float('inf')) else current_min,
            max=None if current_max in (None, float('inf'), -float('inf')) else current_max,
            input_unit=self.primitives[0].domain.input_unit,
            output_unit=self.primitives[-1].domain.output_unit
        )

    def _input_feasible(self, x: float, up_to_stage: int) -> bool:
        """Check if input x produces valid intermediate values for all stages up to up_to_stage."""
        try:
            val = x
            for i in range(up_to_stage + 1):
                p = self.primitives[i]
                if not p.domain.contains(val):
                    return False
                val = p.forward(val)
            return True
        except Exception:
            return False

    def _compute_period(self) -> Optional[float]:
        """Return period if all primitives share the exact same period."""
        periods = set()
        for p in self.primitives:
            if hasattr(p, 'period'):
                periods.add(p.period())
            else:
                return None
        if len(periods) == 1:
            return periods.pop()
        elif len(periods)  > 1:
            print("compositePrimitive._compute_period(): mixed periods detected ")
            return None
        return None

    @staticmethod
    def _units_compatible(out_unit: Dimension, in_unit: Dimension) -> bool:
        """Unit compatibility with GENERIC wildcard"""
        if out_unit == Dimension.GENERIC or in_unit == Dimension.GENERIC:
            return True
        return out_unit == in_unit
        
"""
============================================================================
CORE: GOVERNOR
Stack-level output clamp — wraps any Primitive
Mechanically: centrifugal governor, pressure relief valve, torque limiter
============================================================================
"""

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
        # validate bounds
        if self.min_val  >= self.max_val:
            raise ValueError(
                f"min_val ({self.min_val}) must be less than  "
                f"max_val ({self.max_val}) "
            )
        
        # validate hysteresis
        if self.hysteresis  < 0:
            raise ValueError("Hysteresis must be non-negative ")
        
        if self.hysteresis  >= (self.max_val - self.min_val) / 2:
            raise ValueError(
                "Hysteresis too large — would invert the governed range "
            )
        
        # set domain once
        self.domain = Domain(
            min=self.min_val,
            max=self.max_val,
            input_unit=self.primitive.domain.input_unit,
            output_unit=self.primitive.domain.output_unit
        )

    @property
    def is_invertible(self) -> bool:
        # clamping destroys invertibility
        return False

    @property
    def is_monotonic(self) -> bool:
        return False  # clamping makes output non-monotonic at limits

    def forward(self, x: float) -> float:
        y = self.primitive.forward(x)
        return max(self.min_val, min(self.max_val, y))
        
    def derivative(self, x: float) -> float:
        """Derivative of clamped output"""
        y = self.primitive.forward(x)
        if y  <= self.min_val or y  >= self.max_val:
            # derivative is zero (clipping)
            return 0.0
        return self.primitive.derivative(x)
"""
============================================================================
CLASS I: LINEAR SCALING (Affine Maps)
Continuous, invertible, constant ratio
Satisfies: Primitive, Invertible
is_invertible: True (all Class I are bijective)
============================================================================
"""

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

    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float:
        return y / self.ratio
        
    # Constant, independent of x
    def derivative(self, x: float) -> float:
        return self.ratio
        
    @property
    def is_monotonic(self) -> bool:
        # all spur gears are monotonic
        return True

    @property
    def available_branches(self) -> list[str]:
        return ['none']

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
                "CompoundGearTrain requires at least one ratio "
            )
        if any(r == 0 for r in self.ratios):
            raise ValueError("All gear ratios must be non-zero ")
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

    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float:
        result = y
        for r in reversed(self.ratios):
            result = SpurGear(r).inverse(result)
        return result

    @property
    def is_monotonic(self) -> bool:
        return True  # product of non-zero ratios is always monotonic

    @property
    def available_branches(self) -> list[str]:
        return ['none']

    def derivative(self, x: float) -> float:
        """Product of all gear ratios (constant)"""
        # Could also compute via chain rule, but product is simpler
        product = 1.0
        for r in self.ratios:
            product *= r
        return product

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
        if self.pitch_radius  <= 0:
            raise ValueError("Pitch radius must be positive ")
        self.domain = Domain(
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH
        )

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def is_monotonic(self) -> bool:
        return True  # linear scaling is always monotonic

    def forward(self, x: float) -> float:
        """Rotation (radians) -> linear displacement"""
        return self.pitch_radius * x

    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float:
        """Linear displacement -> rotation (radians)"""
        return y / self.pitch_radius

    @property
    def available_branches(self) -> list[str]:
        return ['none']

    def derivative(self, x: float) -> float:
        """dy/dx = pitch_radius (constant, independent of x)"""
        return self.pitch_radius

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
        if not (0  < self.angle_rad  < math.pi / 2):
            raise ValueError(
                "Wedge angle must be in (0, π/2) exclusive —  "
                "at 0 no lift occurs, at π/2 tan is undefined "
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

    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float:
        """Vertical displacement -> horizontal displacement"""
        return y / math.tan(self.angle_rad)

    @property
    def is_monotonic(self) -> bool:
        return True  # tan(angle) is a positive constant in (0, π/2)

    @property
    def available_branches(self) -> list[str]:
        return ['none']

    def derivative(self, x: float) -> float:
        """dy/dx = tan(angle) (constant, independent of x)"""
        return math.tan(self.angle_rad)

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

    def inverse(self, y: float, branch: Optional[str] = None, guess: Optional[float] = None) -> float:
        """Output shaft angle -> input shaft angle (1:1)"""
        return y

    @property
    def shaft_offset(self) -> tuple[float, float]:
        """Physical offset between shaft centers — not part of scalar map"""
        return (self.offset_x, self.offset_y)

    @property
    def is_monotonic(self) -> bool:
        return True  # identity is trivially monotonic

    @property
    def available_branches(self) -> list[str]:
        return ['none']

    def derivative(self, x: float) -> float:
        """dy/dx = 1 (identity mapping)"""
        return 1.0
        
"""
    ============================================================================
    CLASS II: PERIODIC NON-LINEAR (Trigonometric)
    Oscillatory, bounded, non-injective without domain restriction
    Satisfies: Primitive, Periodic
    Subdivided: PeriodicBijective | PeriodicBranchDependent
    ============================================================================
"""

@dataclass(frozen=True)  # IMMUTABLE
class ScotchYoke:
    amplitude: float
    phase: float = 0.0
    branch: str = 'principal'
    theta_guess: float = 0.0
    is_analytically_invertible: bool = True
    is_monotonic: bool = False

    def __post_init__(self):
        if self.amplitude == 0:
            raise ValueError("Amplitude cannot be zero ")
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches} "
            )
        object.__setattr__(self, 'domain', Domain(
            min=-abs(self.amplitude),
            max=abs(self.amplitude),
            input_unit=Dimension.ANGLE,
            output_unit=Dimension.LENGTH
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

    def __post_init__(self):
        if self.eccentricity  <= 0:
            raise ValueError("Eccentricity must be positive ")
        if self.follower_radius  <= self.eccentricity:
            raise ValueError(
                "Follower radius must exceed eccentricity —  "
                "otherwise cam exceeds follower range "
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches} "
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

        # 1. Domain check FIRST — before any math
        if not self.domain.contains(y):
            raise ValueError(
                f"y={y} outside domain [{self.domain.min}, {self.domain.max}] "
            )
        
        # 2. Check for extrema — derivative is zero at min/max displacement
        #    At these points, only one solution exists (θ=0 or θ=π)
        if abs(y - self.domain.max)  < MechanicalLimits.TOLERANCE:
            return 0.0  # Maximum displacement at θ=0
        if abs(y - self.domain.min)  < MechanicalLimits.TOLERANCE:
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

    def __post_init__(self):
        if self.crank_length  <= 0:
            raise ValueError("Crank length must be positive ")
        if self.rod_length  <= self.crank_length:
            raise ValueError(
                "Rod length must exceed crank length —  "
                "otherwise slider cannot complete full rotation "
            )
        if self.branch not in self.available_branches:
            raise ValueError(
                f"branch must be one of {self.available_branches} "
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

        # 1. Domain check first — before any math
        if not self.domain.contains(y):
            raise ValueError(
                f"y={y} outside domain  "
                f"[{self.domain.min}, {self.domain.max}] "
            )

        # 2. Extrema bypass — derivative is zero at max/min displacement
        #    Only one solution exists at these points
        if abs(y - self.domain.max)  < MechanicalLimits.TOLERANCE:
            return 0.0       # Maximum displacement at θ=0
        if abs(y - self.domain.min)  < MechanicalLimits.TOLERANCE:
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

"""
============================================================================
Immutable Chain Builder
============================================================================

example:

wrist_actuator = (ChainBuilder()
        .add(SpurGear, ratio=2.5)                     # Motor gearbox: 2.5x speed reduction
        .add(HookesJoint, shaft_angle=0.3)            # Angled transmission (≈17°)
        .add(RackAndPinion, pitch_radius=0.1)         # Convert rotation to linear motion
        .add_governor(min_val=-2.0, max_val=2.0)      # Safety limits: ±2cm travel
        .build()) 

motor_angle = 1.5  # radians
actuator_position = wrist_actuator.forward(motor_angle)

print(f"-> wrist_actuator.forward({motor_angle})")
print(f"    Motor at {motor_angle}rad → Actuator at {actuator_position:.3f}m")

"""
class ChainBuilder:
    """
    Each operation returns a NEW builder.
    """
    def __init__(self, primitives: tuple[Primitive, ...] = ()):
        self._primitives = primitives

    def add(self, primitive_class, **kwargs) -> 'ChainBuilder':
        """Add a primitive, returning NEW builder"""
        new_primitive = primitive_class(**kwargs)
        
        # Validate unit compatibility with last primitive
        if self._primitives:
            last = self._primitives[-1]
            if not self._units_compatible(last.domain.output_unit, 
                                          new_primitive.domain.input_unit):
                raise DimensionMismatchError(
                    f"Cannot add {primitive_class.__name__}:  "
                    f"expects {new_primitive.domain.input_unit} but  "
                    f"previous stage outputs {last.domain.output_unit} "
                )
        
        return ChainBuilder(self._primitives + (new_primitive,))

    def add_governor(self, min_val: float, max_val: float, 
                     hysteresis: float = 0.0)  -> 'ChainBuilder':
        """Wrap last primitive in Governor"""
        if not self._primitives:
            raise ValueError("Cannot add governor to empty chain ")
        
        last = self._primitives[-1]
        governor = Governor(last, min_val, max_val, hysteresis)
        
        # Replace last primitive with governor
        return ChainBuilder(self._primitives[:-1] + (governor,))

    def build(self) -> CompositePrimitive:
        """Create immutable composite"""
        if not self._primitives:
            raise ValueError("Cannot build empty chain ")
        return CompositePrimitive(self._primitives)

    @staticmethod
    def _units_compatible(out_unit: Dimension, in_unit: Dimension) -> bool:
        if out_unit == Dimension.GENERIC or in_unit == Dimension.GENERIC:
            return True
        return out_unit == in_unit
