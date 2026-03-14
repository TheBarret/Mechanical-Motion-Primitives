from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, ClassVar
import math

from core.base import Domain, Dimension
from core.exceptions import MMPError


"""
============================================================================
CLASS III: Admissible / Inadmissible adapters [Experimental]

These are "fictional" mathematical transformations that don't correspond to
a single physical mechanism, but allow us to compose arbitrary chains
for analysis, simulation, or mathematical exploration.

They carry the same unit semantics as real primitives, enabling the
compositor to validate dimensional flow even in fictional constructions.
============================================================================
"""

@dataclass
class UnitAdapter:
    """
    Abstract base for all dimensional adapters.
    Not intended for direct instantiation.
    """
    domain: Domain = field(init=False)
    
    def __post_init__(self):
        if not hasattr(self, '_input_unit') or not hasattr(self, '_output_unit'):
            raise TypeError("UnitAdapter subclasses must set _input_unit and _output_unit")
        
        self.domain = Domain(
            input_unit=self._input_unit,
            output_unit=self._output_unit
        )
    
    @property
    def is_invertible(self) -> bool:
        return True
    
    @property
    def is_monotonic(self) -> bool:
        return True
    
    @property
    def available_branches(self) -> list[str]:
        return ['none']


@dataclass
class AngleToLength(UnitAdapter):
    """
    Fictional: Pure mathematical conversion from angle to length.
    Useful for constructing test chains or mathematical explorations.
    y = scale * x
    
    Real-world equivalent would be RackAndPinion, but this adapter
    makes no physical claims - it's a pure mathematical transformation.
    """
    scale: float = 1.0
    _input_unit: ClassVar = Dimension.ANGLE
    _output_unit: ClassVar = Dimension.LENGTH
    
    def __post_init__(self):
        super().__post_init__()
        if self.scale == 0:
            raise ValueError("Scale cannot be zero")
    
    def forward(self, x: float) -> float:
        return x * self.scale
    
    def inverse(self, y: float, branch: Optional[str] = None, 
                guess: Optional[float] = None) -> float:
        return y / self.scale
    
    def derivative(self, x: float) -> float:
        return self.scale


@dataclass
class LengthToAngle(UnitAdapter):
    """
    Fictional: Pure mathematical conversion from length to angle.
    The inverse of AngleToLength - useful for closing kinematic loops
    or constructing mathematical thought experiments.
    y = scale * x
    """
    scale: float = 1.0
    _input_unit: ClassVar = Dimension.LENGTH
    _output_unit: ClassVar = Dimension.ANGLE
    
    def __post_init__(self):
        super().__post_init__()
        if self.scale == 0:
            raise ValueError("Scale cannot be zero")
    
    def forward(self, x: float) -> float:
        return x * self.scale
    
    def inverse(self, y: float, branch: Optional[str] = None,
                guess: Optional[float] = None) -> float:
        return y / self.scale
    
    def derivative(self, x: float) -> float:
        return self.scale


@dataclass
class UnitlessScaling(UnitAdapter):
    """
    Fictional: Pure scaling that preserves units.
    Useful for mathematical transformations that don't change dimension.
    y = scale * x
    
    Input and output units must match (both specified at construction).
    """
    scale: float = 1.0
    unit: Dimension = Dimension.GENERIC
    _input_unit: Dimension = field(init=False)
    _output_unit: Dimension = field(init=False)
    
    def __post_init__(self):
        self._input_unit = self.unit
        self._output_unit = self.unit
        super().__post_init__()
        if self.scale == 0:
            raise ValueError("Scale cannot be zero")
    
    def forward(self, x: float) -> float:
        return x * self.scale
    
    def inverse(self, y: float, branch: Optional[str] = None,
                guess: Optional[float] = None) -> float:
        return y / self.scale
    
    def derivative(self, x: float) -> float:
        return self.scale


@dataclass
class Bias(UnitAdapter):
    """
    Fictional: Add a constant offset while preserving units.
    y = x + bias
    
    Useful for shifting coordinate systems or reference frames.
    Input and output units must match.
    """
    bias: float = 0.0
    unit: Dimension = Dimension.GENERIC
    _input_unit: Dimension = field(init=False)
    _output_unit: Dimension = field(init=False)
    
    def __post_init__(self):
        self._input_unit = self.unit
        self._output_unit = self.unit
        super().__post_init__()
    
    def forward(self, x: float) -> float:
        return x + self.bias
    
    def inverse(self, y: float, branch: Optional[str] = None,
                guess: Optional[float] = None) -> float:
        return y - self.bias
    
    @property
    def is_monotonic(self) -> bool:
        return True
    
    def derivative(self, x: float) -> float:
        return 1.0


@dataclass
class FunctionAdapter(UnitAdapter):
    """
    Fictional: Arbitrary mathematical function with known inverse.
    Allows injection of any invertible function into a chain.
    
    Example:
        square = FunctionAdapter(
            fwd=lambda x: x**2,
            inv=lambda y: math.sqrt(y),
            input_unit=Dimension.LENGTH,
            output_unit=Dimension.ANGLE,
            name="square"
        )
    """
    fwd: Callable[[float], float]
    inv: Callable[[float], float]
    input_unit: Dimension
    output_unit: Dimension
    name: str = "custom"
    deriv: Optional[Callable[[float], float]] = None
    
    _input_unit: Dimension = field(init=False)
    _output_unit: Dimension = field(init=False)
    
    def __post_init__(self):
        self._input_unit = self.input_unit
        self._output_unit = self.output_unit
        super().__post_init__()
    
    def forward(self, x: float) -> float:
        return self.fwd(x)
    
    def inverse(self, y: float, branch: Optional[str] = None,
                guess: Optional[float] = None) -> float:
        return self.inv(y)
    
    def derivative(self, x: float) -> float:
        if self.deriv is not None:
            return self.deriv(x)
        # Fallback to numerical approximation
        h = 1e-8
        return (self.fwd(x + h) - self.fwd(x)) / h
    
    @property
    def is_analytically_invertible(self) -> bool:
        return True
    
    @property
    def is_monotonic(self) -> bool:
        # no guarantee, assume False
        return False