from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from core.base import Domain, Dimension, Primitive
from core.exceptions import MMPError

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
        
        # set domain once (bounds)
        # unbounded fallback path and substituted the mechanical limit
        self.domain = Domain(
            min=None,              # ← Input is NOT constrained
            max=None,              # ← Input is NOT constrained
            input_unit=self.primitive.domain.input_unit,
            output_unit=self.primitive.domain.output_unit,
            output_min=self.min_val,  # ← Output IS constrained
            output_max=self.max_val,  # ← Output IS constrained
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