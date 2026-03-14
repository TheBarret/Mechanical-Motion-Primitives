"""
Primitives layer: All mechanical motion primitives organized by class.
"""

from primitives.class_i_linear import (
    SpurGear, CompoundGearTrain, RackAndPinion, Wedge, OldhamCoupling
)
from primitives.class_ii_periodic import (
    ScotchYoke, EccentricCam, CrankSlider, HookesJoint
)
from primitives.class_iii_adapters import (
    UnitAdapter, AngleToLength, LengthToAngle, UnitlessScaling,
    Bias, FunctionAdapter
)

__all__ = [
    # Class I: Linear
    "SpurGear",
    "CompoundGearTrain",
    "RackAndPinion",
    "Wedge",
    "OldhamCoupling",
    
    # Class II: Periodic
    "ScotchYoke",
    "EccentricCam",
    "CrankSlider",
    "HookesJoint",
    
    # Class III: Adapters
    "UnitAdapter",
    "AngleToLength",
    "LengthToAngle",
    "UnitlessScaling",
    "Bias",
    "FunctionAdapter",
]