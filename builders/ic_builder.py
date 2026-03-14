from __future__ import annotations
from typing import Optional, Type, Any

from core.base import Dimension, Primitive
from core.exceptions import DimensionMismatchError
from composite.chain import CompositePrimitive
from composite.governor import Governor
from primitives.class_i_linear import SpurGear, CompoundGearTrain, RackAndPinion, Wedge, OldhamCoupling
from primitives.class_ii_periodic import ScotchYoke, EccentricCam, CrankSlider, HookesJoint
from primitives.class_iii_adapters import AngleToLength, LengthToAngle, UnitlessScaling, Bias, FunctionAdapter


"""
============================================================================
Immutable Chain Builder
============================================================================

example:

wrist_actuator = (ICBuilder()
        .add(SpurGear, ratio=2.5)                     # Motor gearbox: 2.5x speed reduction
        .add(HookesJoint, shaft_angle=0.3)            # Angled transmission (≈17°)
        .add(RackAndPinion, pitch_radius=0.1)         # Convert rotation to linear motion
        .add_governor(min_val=-2.0, max_val=2.0)      # Safety limits: ±2cm travel
        .build()) 
"""

class ICBuilder:
    """
    Each operation returns a NEW builder.
    """
    def __init__(self, primitives: tuple[Primitive, ...] = ()):
        self._primitives = primitives

    def add(self, primitive_class, **kwargs) -> 'ICBuilder':
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
        
        return ICBuilder(self._primitives + (new_primitive,))

    def add_governor(self, min_val: float, max_val: float, 
                     hysteresis: float = 0.0)  -> 'ICBuilder':
        """Wrap last primitive in Governor"""
        if not self._primitives:
            raise ValueError("Cannot add governor to empty chain ")
        
        last = self._primitives[-1]
        governor = Governor(last, min_val, max_val, hysteresis)
        
        # Replace last primitive with governor
        return ICBuilder(self._primitives[:-1] + (governor,))

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