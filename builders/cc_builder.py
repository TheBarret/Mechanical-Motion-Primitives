from __future__ import annotations
from typing import Optional

from core.base import Dimension
from core.exceptions import DimensionMismatchError
from composite.chain import CompositePrimitive
from composite.governor import Governor
from primitives.class_i_linear import SpurGear, CompoundGearTrain, RackAndPinion, Wedge, OldhamCoupling
from primitives.class_ii_periodic import ScotchYoke, EccentricCam, CrankSlider, HookesJoint
from primitives.class_iii_adapters import AngleToLength, LengthToAngle, UnitlessScaling, Bias, FunctionAdapter

"""        
============================================================================
 Creative Chain Builder
 Adapter support, mathematically admissible but physically inadmissible.
============================================================================

example:

experiment = (CCBuilder()
        .add(SpurGear, ratio=2.5)                           # Current unit: ANGLE
        .add(RackAndPinion, pitch_radius=0.1)               # Current unit: LENGTH
        .add_adapter(to_unit=Dimension.ANGLE, scale=10.0)   # LENGTH → ANGLE
        .add(HookesJoint, shaft_angle=0.3)                  # Current unit: ANGLE again
        .build())
        
"""
class CCBuilder:
    """
    Creative builder with adapter convenience methods.
    """
    def __init__(self, primitives: tuple[Primitive, ...] = ()):
        self._primitives = primitives
    
    def add(self, primitive_class, **kwargs) -> 'CCBuilder':
        """Add a primitive, returning NEW builder"""
        new_primitive = primitive_class(**kwargs)
        
        # Validate unit compatibility with last primitive
        if self._primitives:
            last = self._primitives[-1]
            if not self._units_compatible(last.domain.output_unit, 
                                          new_primitive.domain.input_unit):
                raise DimensionMismatchError(
                    f"Cannot add {primitive_class.__name__}: "
                    f"expects {new_primitive.domain.input_unit} but "
                    f"previous stage outputs {last.domain.output_unit}"
                )
        
        return CCBuilder(self._primitives + (new_primitive,))
    
    def add_adapter(self, to_unit: Dimension, scale: float = 1.0) -> 'CCBuilder':
        """
        Add appropriate adapter between units.
        Infers from_unit from the tail of the chain.
        """
        if not self._primitives:
            raise ValueError("Cannot add adapter to empty chain — no from_unit to infer")
        
        from_unit = self._primitives[-1].domain.output_unit

        if from_unit == to_unit:
            if scale != 1.0:
                return self.add(UnitlessScaling, scale=scale, unit=from_unit)
            return CCBuilder(self._primitives)  # immutable copy, not self
        
        if from_unit == Dimension.ANGLE and to_unit == Dimension.LENGTH:
            return self.add(AngleToLength, scale=scale)
        
        if from_unit == Dimension.LENGTH and to_unit == Dimension.ANGLE:
            return self.add(LengthToAngle, scale=scale)
        
        raise DimensionMismatchError(
            f"No direct adapter from {from_unit} to {to_unit}"
        )
    
    def add_governor(self, min_val: float, max_val: float, 
                     hysteresis: float = 0.0) -> 'CCBuilder':
        """Wrap last primitive in Governor"""
        if not self._primitives:
            raise ValueError("Cannot add governor to empty chain")
        
        last = self._primitives[-1]
        governor = Governor(last, min_val, max_val, hysteresis)
        
        # Replace last primitive with governor
        return CCBuilder(self._primitives[:-1] + (governor,))
    
    def build(self) -> CompositePrimitive:
        """Create immutable composite"""
        if not self._primitives:
            raise ValueError("Cannot build empty chain")
        return CompositePrimitive(self._primitives)
    
    @staticmethod
    def _units_compatible(out_unit: Dimension, in_unit: Dimension) -> bool:
        if out_unit == Dimension.GENERIC or in_unit == Dimension.GENERIC:
            return True
        return out_unit == in_unit