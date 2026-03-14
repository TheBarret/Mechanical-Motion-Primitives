from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Optional, Tuple, runtime_checkable
from enum import Enum, auto

# Optional

from core.constants import MechanicalLimits
from core.exceptions import DomainViolationError

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

    min / max         — valid INPUT range for this primitive (None = unbounded).
    output_min / max  — achievable OUTPUT range (None = unbounded / same as input).
    Periodic primitives (ScotchYoke, CrankSlider, EccentricCam) accept any input
    angle but produce bounded output — they set min/max=None and declare output bounds
    via output_min/output_max.  _compute_output_domain and _input_feasible use only
    min/max for input gating; output clamping uses output_min/output_max.
    """
    min:         Optional[float] = None
    max:         Optional[float] = None
    input_unit:  Dimension       = Dimension.GENERIC
    output_unit: Dimension       = Dimension.GENERIC
    output_min:  Optional[float] = None   # Achievable output lower bound
    output_max:  Optional[float] = None   # Achievable output upper bound

    def contains(self, x: float, tolerance: float = 1e-9) -> bool:
        """Check whether x lies within the INPUT domain."""
        if self.min is not None and x < self.min - tolerance:
            return False
        if self.max is not None and x > self.max + tolerance:
            return False
        return True
    
    def output_contains(self, y: float, tolerance: float = 1e-9) -> bool:
        """
        Check whether y lies within the achievable OUTPUT range.
        Falls back to input bounds (min/max) when output_min/output_max are not set,
        preserving backwards-compatible behaviour for linear primitives where input
        and output share the same unbounded nature.
        """
        # Handle None input - this should never happen in normal use,
        # but protects against tests and edge cases
        if y is None:
            return False
            
        lo = self.output_min if self.output_min is not None else self.min
        hi = self.output_max if self.output_max is not None else self.max
        
        # Safe comparisons with None
        if lo is not None and y < lo - tolerance:
            return False
        if hi is not None and y > hi + tolerance:
            return False
        return True

    def is_finite(self) -> bool:
        """Input domain is fully bounded."""
        return self.min is not None and self.max is not None

    def output_is_finite(self) -> bool:
        """Output range is fully bounded."""
        lo = self.output_min if self.output_min is not None else self.min
        hi = self.output_max if self.output_max is not None else self.max
        return lo is not None and hi is not None
        
"""
============================================================================
CORE: PROTOCOLS
============================================================================
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