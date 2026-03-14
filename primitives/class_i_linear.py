from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.base import Domain, Dimension
from core.exceptions import MMPError
from core.constants import MechanicalLimits

# Optional (not implemented yet)
#from utils.validation import validate_positive, validate_non_zero

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