import math
from typing import Optional, List, Any

from core.exceptions import MMPError

def validate_positive(value: float, name: str = "value") -> None:
    if value <= 0:
        raise MMPError(f"{name} must be positive, got {value}")

def validate_non_zero(value: float, name: str = "value") -> None:
    if value == 0:
        raise MMPError(f"{name} cannot be zero")

def validate_range(
    value: float,
    min_val: Optional[float],
    max_val: Optional[float],
    name: str = "value"
) -> None:
    if min_val is not None and value < min_val:
        raise MMPError(f"{name} {value} < minimum {min_val}")
    if max_val is not None and value > max_val:
        raise MMPError(f"{name} {value} > maximum {max_val}")

def validate_branch(branch: str, available: List[str]) -> None:
    if branch not in available:
        raise MMPError(f"Branch '{branch}' not in {available}")