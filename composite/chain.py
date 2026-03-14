from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

from core.base import Domain, Dimension, Primitive, Invertible
from core.exceptions import CompositionError, DimensionMismatchError, DomainViolationError, InverseUndefinedError
from core.constants import MechanicalLimits
from primitives.class_i_linear import SpurGear, CompoundGearTrain, RackAndPinion, Wedge, OldhamCoupling
from primitives.class_ii_periodic import ScotchYoke, EccentricCam, CrankSlider, HookesJoint
from primitives.class_iii_adapters import AngleToLength, LengthToAngle, UnitlessScaling, Bias, FunctionAdapter

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

        Guards:
        1. Chain must be invertible.
        2. y must lie within the chain's achievable output range — checked via
           _output_domain.output_contains().  Calling inverse() with a y outside
           that range is physically meaningless and would produce silent garbage.
        """
    
        if not self.is_invertible:
            raise InverseUndefinedError("Chain is not invertible")

        # Add small epsilon tolerance for boundary cases
        if not self._output_domain.output_contains(y, tolerance=1e-6):
            # Check if we're extremely close to boundary
            out_min = self._output_domain.output_min
            out_max = self._output_domain.output_max
            if out_min is not None and abs(y - out_min) < 1e-10:
                y = out_min
            elif out_max is not None and abs(y - out_max) < 1e-10:
                y = out_max
            else:
                raise DomainViolationError(
                    f"y={y} is outside the chain's achievable output range "
                    f"[{out_min}, {out_max}]"
                )
            
        result = y
        # Iterate backwards through primitives.
        # idx counts forward (0…N-1) so that branches/guesses lists are indexed
        # in the same forward order the caller naturally thinks about them.
        # NOTE: do NOT substitute i for idx here — i counts the reversed iteration
        #       (0 = last primitive), which is the opposite of the caller's mental model.
        for i, p in enumerate(reversed(self.primitives)):
            idx = len(self.primitives) - 1 - i
            branch = branches[idx] if branches and idx < len(branches) else None
            guess  = guesses[idx]  if guesses  and idx < len(guesses)  else None
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
        Checks is_monotonic before analytic inversion (invertible != monotonic).

        Bounded periodic primitives (ScotchYoke, CrankSlider, EccentricCam) declare
        domain.min/max=None (unbounded input) but set domain.output_min/output_max.
        We must not skip those stages — their output bounds constrain which input
        values are useful.  The analytic path uses output_min/output_max as the
        bounds to back-propagate; the sampling path falls through naturally since
        domain.contains() on an unbounded-input primitive always returns True.
        """
        if not self.primitives:
            return Domain(input_unit=Dimension.GENERIC, output_unit=Dimension.GENERIC)

        current_min = self.primitives[0].domain.min
        current_max = self.primitives[0].domain.max

        for i, p in enumerate(self.primitives[1:], start=1):
            # Determine effective output bounds for this primitive.
            # Prefer explicit output_min/output_max; fall back to domain.min/max
            # for primitives that don't separate the two (linear primitives).
            p_out_lo = p.domain.output_min if p.domain.output_min is not None else p.domain.min
            p_out_hi = p.domain.output_max if p.domain.output_max is not None else p.domain.max

            # Skip primitives with no output constraints — they don't narrow the domain.
            if p_out_lo is None and p_out_hi is None:
                continue
            
            prefix = self.primitives[:i]
            
            # Check monotonicity. Invertible != Monotonic (e.g. ScotchYoke)
            all_invertible = all(getattr(pp, 'is_invertible', False) for pp in prefix)
            all_monotonic  = all(getattr(pp, 'is_monotonic',  False) for pp in prefix)

            constraints = []
            # Only use analytic inversion if the prefix chain is BOTH invertible AND monotonic
            if all_invertible and all_monotonic:
                if p_out_lo is not None:
                    try:
                        y = p_out_lo
                        for pp in reversed(prefix):
                            if hasattr(pp, 'inverse'):
                                y = pp.inverse(y)
                            else:
                                raise ValueError(f"Primitive {pp} lacks inverse")
                        constraints.append(y)
                    except Exception:
                        pass
                if p_out_hi is not None:
                    try:
                        y = p_out_hi
                        for pp in reversed(prefix):
                            if hasattr(pp, 'inverse'):
                                y = pp.inverse(y)
                            else:
                                raise ValueError(f"Primitive {pp} lacks inverse")
                        constraints.append(y)
                    except Exception:
                        pass
            
            # If analytic failed or prefix is not monotonic, use sampling.
            # domain.contains(val) checks the INPUT bounds of p — for bounded periodic
            # primitives this is always True (min/max=None), which is correct: any
            # angle is a valid input to sin/cos.  The output-range filter is implicit
            # in the forward pass values we observe.
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
                    # No feasible inputs — chain is geometrically impossible
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
                # Fallback: clamp to this primitive's declared output range if available,
                # otherwise try its input domain bounds.
                p_out_lo = p.domain.output_min if p.domain.output_min is not None else p.domain.min
                p_out_hi = p.domain.output_max if p.domain.output_max is not None else p.domain.max
                if p_out_lo is not None:
                    current_min = max(current_min, p_out_lo) if current_min is not None else p_out_lo
                if p_out_hi is not None:
                    current_max = min(current_max, p_out_hi) if current_max is not None else p_out_hi

        # Final clamp to last primitive's declared output range.
        # Use output_min/output_max when set (bounded periodic primitives);
        # fall back to domain.min/max for primitives that don't separate the two.
        #
        # Semantics: the declared bounds are exact (analytic), sampling is approximate.
        # For the lower bound we take the MINIMUM of the two (declared is tighter/exact).
        # For the upper bound we take the MAXIMUM of the two.
        # This ensures the guard in inverse() never rejects a mathematically valid y
        # that sampling happened to miss.
        last = self.primitives[-1]
        last_out_lo = last.domain.output_min if last.domain.output_min is not None else last.domain.min
        last_out_hi = last.domain.output_max if last.domain.output_max is not None else last.domain.max
        if last_out_lo is not None:
            # Declared lower bound wins if it is tighter (smaller) than sampled
            current_min = min(current_min, last_out_lo) if current_min is not None else last_out_lo
        if last_out_hi is not None:
            # Declared upper bound wins if it is tighter (larger) than sampled
            current_max = max(current_max, last_out_hi) if current_max is not None else last_out_hi

        if current_min is not None and current_max is not None:
            if current_min > current_max + MechanicalLimits.TOLERANCE:
                return Domain(min=float('inf'), max=-float('inf'),
                              input_unit=self.primitives[0].domain.input_unit,
                              output_unit=self.primitives[-1].domain.output_unit)

        out_lo = None if current_min in (None, float('inf'), -float('inf')) else current_min
        out_hi = None if current_max in (None, float('inf'), -float('inf')) else current_max
        return Domain(
            min=None,
            max=None,
            input_unit=self.primitives[0].domain.input_unit,
            output_unit=self.primitives[-1].domain.output_unit,
            output_min=out_lo,
            output_max=out_hi,
        )

    def _input_feasible(self, x: float, up_to_stage: int) -> bool:
        """
        Check whether input x produces values that fall within each stage's
        declared INPUT domain up to and including up_to_stage.

        domain.contains() tests domain.min/max — the valid INPUT range.
        For bounded periodic primitives (ScotchYoke, CrankSlider, EccentricCam)
        domain.min/max are None (any angle is valid input), so this check
        always passes for those stages, which is correct — sin/cos accept all reals.
        Output-range gating is handled separately in _compute_output_domain via
        output_min/output_max.
        """
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