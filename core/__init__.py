"""
Core layer: Fundamental types, protocols, and exceptions.
No dependencies on other MMP layers.
"""

from core.base import (
    Dimension, Domain, Primitive, Invertible, OneWay,
    Configurable, Periodic, PeriodicBijective, PeriodicBranchDependent
)
from core.exceptions import (
    MMPError, CompositionError, DimensionMismatchError,
    DomainViolationError, InverseUndefinedError
)
from core.constants import MechanicalLimits

__all__ = [
    # Base
    "Dimension",
    "Domain",
    "Primitive",
    "Invertible",
    "OneWay",
    "Configurable",
    "Periodic",
    "PeriodicBijective",
    "PeriodicBranchDependent",
    
    # Exceptions
    "MMPError",
    "CompositionError",
    "DimensionMismatchError",
    "DomainViolationError",
    "InverseUndefinedError",
    
    # Constants
    "MechanicalLimits",
]