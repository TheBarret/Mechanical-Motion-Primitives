"""
MMP GA Stress Tester
"""

import random
import inspect
import math
import traceback
from typing import Callable
from typing import Type, Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from mmp import *

# ============================================================================
# Genome Foundation
# ============================================================================

Genome = List[Tuple[Type[Primitive], Dict[str, Any]]]

# ============================================================================
# FAULT DETECTION
# ============================================================================

@dataclass
class FaultReport:
    """What went wrong, where, and why"""
    exception_type: str
    exception_msg: str
    stage: int  # -1 for composition, N for primitive N
    primitive_name: str
    trace: str
    genome: List[Tuple[Type, Dict]]
    
class FaultDetector:
    """
    Special fitness that doesn't want success - it wants FAILURES
    """
    
    def __init__(self):
        self.faults_found: List[FaultReport] = []
        self.seen_faults: Set[str] = set()  # Deduplicate by exception msg
        
    def fitness(self, genome: List[Tuple[Type, Dict]], 
                target_input: float, target_output: float) -> float:
        """
        Score: higher for interesting failures, lower for success!
        """
        try:
            # Try to build - this might raise CompositionError
            builder = ICBuilder()
            for cls, kwargs in genome:
                builder = builder.add(cls, **kwargs)
            chain = builder.build()
            
            # Try forward pass - this might raise various errors
            result = chain.forward(target_input)
            
            # SUCCESS - boring! Low score
            error = abs(result - target_output)
            return 0.1 / (1.0 + error)  # Max 0.1 for success
            
        except Exception as e:
            # FAILURE - interesting! High score
            return self._score_failure(e, genome)
    
    def _is_expected_validation(self, e: Exception) -> bool:
        """
        Distinguish 'framework working correctly' from 'actual bug'.
        """
        # These exception types indicate proper validation
        if isinstance(e, (DimensionMismatchError, DomainViolationError)):
            return True
        
        # Common validation messages that are expected
        msg = str(e).lower()
        expected_phrases = [
            "must be positive",
            "must exceed",
            "must be in",
            "cannot be zero",
            "outside domain",
            "expects dimension",
            "previous stage outputs",
        ]
        return any(phrase in msg for phrase in expected_phrases)

    def _score_failure(self, e: Exception, genome: List[Tuple[Type, Dict]]) -> float:
        msg = str(e)[:100]
        if msg in self.seen_faults:
            return 0.5  # Already seen
        
        self.seen_faults.add(msg)
        
        # Record fault with improved stage detection
        stage = self._find_failing_stage(e, genome)
        fault = FaultReport(
            exception_type=type(e).__name__,
            exception_msg=str(e),
            stage=stage,
            primitive_name=genome[stage][0].__name__ if 0 <= stage < len(genome) else "unknown",
            trace=traceback.format_exc(),
            genome=genome
        )
        self.faults_found.append(fault)
        
        # Score: expected validations get low score, surprises get high score
        if self._is_expected_validation(e):
            return 0.3  # Working as intended
        elif isinstance(e, ZeroDivisionError):
            return 10.0  # Math singularity — very interesting
        elif isinstance(e, ValueError) and ("math domain" in msg or "sqrt" in msg):
            return 9.0  # Domain hole in primitive math
        elif isinstance(e, DomainViolationError):
            return 8.0  # Domain propagation issue
        elif isinstance(e, InverseUndefinedError):
            return 7.0  # Invertibility tracking bug
        else:
            return 4.0  # Unexpected but not critical
            
    def _find_failing_stage(self, e: Exception, genome: List[Tuple[Type, Dict]]) -> int:
        """
        Identify which stage failed.
        Strategy: 
        1. Check if primitive name appears in exception message (most reliable)
        2. Fallback: scan traceback for __init__ frames
        """
        msg = str(e).lower()
        
        # Strategy 1: Primitive names often appear in error messages
        # e.g., "Cannot add CrankSlider: expects..." or "Shaft angle must be in..."
        for i, (cls, _) in enumerate(genome):
            if cls.__name__.lower() in msg:
                return i
        
        # Strategy 2: Traceback parsing (fallback)
        tb = traceback.extract_tb(e.__traceback__)
        for frame in tb:
            # Check if frame is in a primitive's __init__ or forward method
            for i, (cls, _) in enumerate(genome):
                if cls.__name__ in frame.filename:
                    if '__init__' in frame.name or 'forward' in frame.name:
                        return i
                    # Check source line if available
                    if frame.line and cls.__name__ in frame.line:
                        return i
        
        return -1  # Unknown
        
    def _find_failing_primitive(self, e: Exception, 
                                genome: List[Tuple[Type, Dict]]) -> str:
        stage = self._find_failing_stage(e, genome)
        if stage >= 0:
            return genome[stage][0].__name__
        return "unknown"

# ============================================================================
# COVERAGE TRACKING
# ============================================================================

class CoverageTracker:
    """Track what we've tested to find gaps"""
    
    def __init__(self):
        self.tested_pairs: Set[Tuple[str, str]] = set()
        self.tested_triples: Set[Tuple[str, str, str]] = set()
        self.tested_primitives: Set[str] = set()
        self.dimension_paths: Set[Tuple[str, str]] = set()  # (in_dim, out_dim)
        
    def update(self, genome: List[Tuple[Type, Dict]]):
        """Record coverage from a genome"""
        names = [cls.__name__ for cls, _ in genome]
        
        # Individual primitives
        for name in names:
            self.tested_primitives.add(name)
            
        # Pairs
        for i in range(len(names) - 1):
            self.tested_pairs.add((names[i], names[i+1]))
            
        # Triples
        for i in range(len(names) - 2):
            self.tested_triples.add((names[i], names[i+1], names[i+2]))
            
        # Dimension paths (build a dummy instance to get domains)
        try:
            builder = ICBuilder()
            for cls, kwargs in genome:
                builder = builder.add(cls, **kwargs)
            chain = builder.build()
            in_dim = chain.domain.input_unit.name
            out_dim = chain.domain.output_unit.name
            self.dimension_paths.add((in_dim, out_dim))
        except:
            pass
            
    def coverage_report(self) -> Dict:
        """What's missing?"""
        all_primitives = [SpurGear, RackAndPinion, Wedge, HookesJoint, 
                          ScotchYoke, EccentricCam, CrankSlider, OldhamCoupling]
        all_names = {cls.__name__ for cls in all_primitives}
        
        missing_prims = all_names - self.tested_primitives
        
        return {
            'primitives_tested': len(self.tested_primitives),
            'primitives_total': len(all_names),
            'missing_primitives': missing_prims,
            'pairs_tested': len(self.tested_pairs),
            'triples_tested': len(self.tested_triples),
            'dimension_paths': len(self.dimension_paths)
        }

# ============================================================================
# MUTATION
# ============================================================================

class CoverageAwareMutator:
    """Mutation that tries to fill coverage gaps"""
    
    def __init__(self, primitive_pool: List[Type], coverage: CoverageTracker):
        self.pool = primitive_pool
        self.coverage = coverage
        
    def mutate(self, genome: Genome) -> Genome:
        """Mutate with bias toward untested combinations"""
        import copy
        new_genome = copy.deepcopy(genome)
        
        if not new_genome or random.random() < 0.3:
            # Add a new primitive, biased toward untested ones
            all_names = {cls.__name__ for cls in self.pool}
            tested = self.coverage.tested_primitives
            untested = list(all_names - tested)
            
            if untested and random.random() < 0.7:
                # Strong bias toward untested primitives
                cls_name = random.choice(untested)
                cls = next(c for c in self.pool if c.__name__ == cls_name)
            else:
                cls = random.choice(self.pool)
                
            pos = random.randint(0, len(new_genome))
            new_genome.insert(pos, (cls, _random_params(cls)))
            return new_genome
            
        if len(new_genome) > 1 and random.random() < 0.3:
            # Remove a gene, maybe creating a new untested pair
            idx = random.randint(0, len(new_genome) - 1)
            new_genome.pop(idx)
            return new_genome
            
        # Parameter tweak
        if new_genome:
            idx = random.randint(0, len(new_genome) - 1)
            cls, params = new_genome[idx]
            
            # Try to push parameters to extremes to find edge cases
            numeric_keys = [k for k, v in params.items() 
                          if isinstance(v, (int, float))]
            if numeric_keys:
                key = random.choice(numeric_keys)
                if random.random() < 0.3:
                    # Extreme value (find singularities)
                    params[key] *= random.choice([0.01, 100.0])
                else:
                    # Normal tweak
                    params[key] *= random.uniform(0.5, 2.0)
                    
        return new_genome

# ============================================================================
# MAIN EVOLUTION LOOP
# ============================================================================

def hunt_for_faults(
    primitive_pool: List[Type[Primitive]],
    generations: int = 500,
    pop_size: int = 100,
    target_input: float = 1.0,  # Doesn't really matter
    target_output: float = 0.5   # Doesn't really matter
) -> List[FaultReport]:
    """
    Evolve chains to find framework bugs, not to solve problems.
    """
    detector = FaultDetector()
    coverage = CoverageTracker()
    mutator = CoverageAwareMutator(primitive_pool, coverage)
    
    # Initial population: random chains
    population = []
    for _ in range(pop_size):
        length = random.randint(1, 5)
        genome = []
        for _ in range(length):
            cls = random.choice(primitive_pool)
            genome.append((cls, _random_params(cls)))
        population.append(genome)
    
    for gen in range(generations):
        # Evaluate - we want FAILURES
        scored = [(detector.fitness(g, target_input, target_output), g) 
                  for g in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update coverage from top genomes
        for _, g in scored[:10]:
            coverage.update(g)
        
        # Report progress
        if gen % 100 == 0:
            cov = coverage.coverage_report()
            print(f"\nGen {gen}:")
            print(f"  Faults found: {len(detector.faults_found)}")
            print(f"  Coverage: {cov['primitives_tested']}/{cov['primitives_total']} primitives")
            print(f"  Missing: {cov['missing_primitives']}")
            
            # Show most interesting recent fault
            if detector.faults_found:
                latest = detector.faults_found[-1]
                print(f"  Latest fault: {latest.exception_type} at stage {latest.stage}")
        
        # Elitism - keep top 20% (the most broken ones!)
        keep = max(1, pop_size // 5)
        next_gen = [g for s, g in scored[:keep]]
        
        # Fill rest with mutations
        while len(next_gen) < pop_size:
            parent = random.choice(scored[:pop_size//2])[1]
            child = mutator.mutate(parent)
            next_gen.append(child)
            
        population = next_gen
    
    # Final report
    print("\n" + "="*50)
    print("FAULT HUNT COMPLETE")
    print("="*50)
    
    cov = coverage.coverage_report()
    print(f"\nCoverage:")
    print(f"  Primitives: {cov['primitives_tested']}/{cov['primitives_total']}")
    if cov['missing_primitives']:
        print(f"  Untested: {cov['missing_primitives']}")
    print(f"  Unique pairs: {cov['pairs_tested']}")
    print(f"  Dimension paths: {cov['dimension_paths']}")
    
    print(f"\nFaults Found: {len(detector.faults_found)}")
    
    # Group by type
    from collections import Counter
    fault_types = Counter([f.exception_type for f in detector.faults_found])
    print("\nFault Types:")
    for ftype, count in fault_types.most_common():
        print(f"  {ftype}: {count}")
    
    # Show most interesting faults
    print("\nFaults:")
    for i, fault in enumerate(detector.faults_found[-10:]):
        print(f"\n{i+1}. {fault.exception_type}")
        print(f"   {fault.exception_msg}")
        print(f"   Stage: {fault.stage} ({fault.primitive_name})")
        chain_str = " → ".join([f"{cls.__name__}" for cls, _ in fault.genome])
        print(f"   Chain: {chain_str}")
    
    return detector.faults_found

# ============================================================================
# VALIDATION - Test specific hypotheses
# ============================================================================

def test_hypothesis(h: Dict) -> Dict[str, Any]:
    """
    Run a single hypothesis test with expectation checking.
    Returns detailed results for analysis.
    """
    results = {
        'name': h['name'],
        'passed': True,
        'notes': [],
        'errors': []
    }
    
    genome = h['genome']
    test_inputs = h.get('test_inputs', [1.0])
    expect = h.get('expect', 'No errors')
    
    try:
        # Build chain
        builder = ICBuilder()
        for cls, kwargs in genome:
            builder = builder.add(cls, **kwargs)
        
        # Apply post-build hook if present (e.g., add_governor)
        if 'post_build' in h:
            builder = h['post_build'](builder)
            
        chain = builder.build()
        results['chain_built'] = True
        
        # Run forward tests
        outputs = []
        for x in test_inputs:
            try:
                y = chain.forward(x)
                outputs.append((x, y))
            except Exception as e:
                results['errors'].append(f"forward({x}): {type(e).__name__}: {e}")
                results['passed'] = False
        
        # Round-trip test if invertible and requested
        if chain.is_invertible and outputs:
            for x, y in outputs[:3]:  # Test first few
                try:
                    x_rec = chain.inverse(y)
                    if abs(x_rec - x) > MechanicalLimits.TOLERANCE:
                        results['notes'].append(
                            f"Round-trip error at x={x}: |{x_rec} - {x}| = {abs(x_rec-x)}"
                        )
                except Exception as e:
                    results['errors'].append(f"inverse({y}): {e}")
        
        # Derivative consistency check if requested
        if 'derivative' in expect.lower():
            for x in test_inputs[:3]:
                try:
                    analytical = chain.derivative(x)
                    numerical = numerical_derivative(chain.forward, x)
                    if abs(analytical - numerical) > 1e-5:
                        results['notes'].append(
                            f"Derivative mismatch at x={x}: "
                            f"analytical={analytical:.6f}, numerical={numerical:.6f}"
                        )
                except Exception as e:
                    results['errors'].append(f"derivative({x}): {e}")
        
        # Invertibility check
        if 'non-invertible' in expect.lower() or 'governor' in h['name'].lower():
            if chain.is_invertible:
                results['notes'].append("WARNING: Chain reports invertible but expected non-invertible")
            else:
                try:
                    chain.inverse(0.0)
                    results['notes'].append("WARNING: inverse() didn't raise on non-invertible chain")
                except InverseUndefinedError:
                    pass  # Expected
        
        results['expectation'] = expect
        
    except Exception as e:
        results['passed'] = False
        results['errors'].append(f"build: {type(e).__name__}: {e}")
    
    return results

# ============================================================================
# TEST ENVIROMENT - Specific hypotheses
# ============================================================================

HYPOTHESES = [
    # ========================================================================
    # 1. SINGULARITY NEAR-MISSES
    # ========================================================================
    {
        'name': "HookesJoint near lock-up (α → π/2)",
        'genome': [(HookesJoint, {'shaft_angle': math.pi/2 - 1e-6})],
        'test_inputs': [0, 0.1, math.pi/4, math.pi/2 - 1e-4, math.pi/2 + 1e-4],
        'expect': "derivative() spikes but doesn't crash; forward() stays bounded"
    },
    {
        'name': "HookesJoint continuity across π boundary",
        'genome': [(HookesJoint, {'shaft_angle': 0.5})],
        'test_inputs': [math.pi - 0.1, math.pi, math.pi + 0.1, 2*math.pi - 0.1, 2*math.pi],
        'expect': "forward() output continuous; no π-jump artifacts"
    },
    {
        'name': "Wedge near vertical (angle → π/2)",
        'genome': [(Wedge, {'angle_rad': math.pi/2 - 1e-6})],
        'test_inputs': [0.01, 0.1, 1.0],
        'expect': "forward() returns large values but no OverflowError"
    },
    {
        'name': "Wedge near flat (angle → 0)",
        'genome': [(Wedge, {'angle_rad': 1e-6})],
        'test_inputs': [0.1, 1.0, 10.0],
        'expect': "forward() ≈ 0; derivative ≈ tan(angle) ≈ angle"
    },
    {
        'name': "CrankSlider near kinematic limit (L ≈ r)",
        'genome': [(CrankSlider, {'crank_length': 1.0, 'rod_length': 1.001})],
        'test_inputs': [0, math.pi/2, math.pi, 3*math.pi/2],
        'expect': "forward() stays in domain; inverse() doesn't diverge"
    },
    {
        'name': "CrankSlider L >> r regime",
        'genome': [(CrankSlider, {'crank_length': 0.1, 'rod_length': 5.0})],
        'test_inputs': [0, 1, 2, 3, 4, 5, 6],
        'expect': "forward() ≈ L + r*cos(θ); derivative smooth"
    },
    {
        'name': "EccentricCam near follower limit",
        'genome': [(EccentricCam, {'eccentricity': 0.1, 'follower_radius': 0.101})],
        'test_inputs': [0, math.pi/2, math.pi, 3*math.pi/2],
        'expect': "forward() bounded; inverse() converges with guess"
    },
    {
        'name': "ScotchYoke amplitude boundary",
        'genome': [(ScotchYoke, {'amplitude': 1.0})],
        'test_inputs': [-1.0, -0.999, 0, 0.999, 1.0],
        'expect': "forward() clamped to ±amplitude; inverse() handles boundaries"
    },

    # ========================================================================
    # 2. DOMAIN PROPAGATION EDGE CASES
    # ========================================================================
    {
        'name': "Non-monotonic primitive in middle of chain",
        'genome': [
            (SpurGear, {'ratio': 2.0}),
            (ScotchYoke, {'amplitude': 1.0}),
            (SpurGear, {'ratio': 0.5})
        ],
        'test_inputs': [0, 1, 2, 3, 4, 5, 6],
        'expect': "_compute_input_domain() uses sampling fallback, not analytic inversion"
    },
    {
        'name': "Unbounded input propagates conservatively",
        'genome': [(SpurGear, {'ratio': 10.0}), (RackAndPinion, {'pitch_radius': 0.1})],
        'test_inputs': [-1e6, -1e3, 0, 1e3, 1e6],
        'expect': "output_domain uses MechanicalLimits as fallback, not ±inf"
    },
    {
        'name': "Domain intersection empties chain",
        'genome': [
            (SpurGear, {'ratio': 100.0}),
            (ScotchYoke, {'amplitude': 0.01})  # tiny output domain
        ],
        'test_inputs': [0, 0.1, 1.0],
        'expect': "input_domain shrinks or becomes impossible"
    },
    {
        'name': "Mixed bounded/unbounded chain",
        'genome': [
            (SpurGear, {'ratio': 2.0}),  # unbounded
            (ScotchYoke, {'amplitude': 1.0})  # bounded output
        ],
        'test_inputs': [-10, -1, 0, 1, 10],
        'expect': "output_domain = ±amplitude; input_domain back-propagated"
    },

    # ========================================================================
    # 3. ROUND-TRIP CONSISTENCY (Invertible Chains)
    # ========================================================================
    {
        'name': "Forward → Inverse round-trip",
        'genome': [
            (SpurGear, {'ratio': 3.7}),
            (RackAndPinion, {'pitch_radius': 0.15})
        ],
        'test_inputs': [0, 0.5, 1.0, 2.5, 5.0],
        'expect': "chain.inverse(chain.forward(x)) ≈ x within TOLERANCE"
    },
    {
        'name': "ScotchYoke branch-aware inverse",
        'genome': [(ScotchYoke, {'amplitude': 2.0, 'phase': 0.3})],
        'test_inputs': [0.5, 1.0, 1.5],
        'expect': "inverse(y, branch='principal') and branch='supplementary' give distinct x"
    },
    {
        'name': "HookesJoint round-trip with continuity",
        'genome': [(HookesJoint, {'shaft_angle': 0.4})],
        'test_inputs': [0, 1, 2, 3, 4, 5, 6],
        'expect': "inverse(forward(x)) ≈ x; no π-wrap discontinuities"
    },
    {
        'name': "CompoundGearTrain round-trip",
        'genome': [(CompoundGearTrain, {'ratios': [2.0, 0.5, 3.0]})],
        'test_inputs': [-5, -1, 0, 1, 5],
        'expect': "inverse(forward(x)) ≈ x; net ratio = 3.0"
    },
    {
        'name': "OldhamCoupling identity round-trip",
        'genome': [(OldhamCoupling, {'offset_x': 0.1, 'offset_y': 0.2})],
        'test_inputs': [-10, -1, 0, 1, 10],
        'expect': "forward(x) = x; inverse(y) = y exactly"
    },

    # ========================================================================
    # 4. DERIVATIVE CONSISTENCY
    # ========================================================================
    {
        'name': "Analytical vs numerical derivative: HookesJoint",
        'genome': [(HookesJoint, {'shaft_angle': 0.4})],
        'test_inputs': [0.1, 0.5, 1.0, 2.0],
        'expect': "chain.derivative(x) ≈ numerical_diff(forward, x) within 1e-6"
    },
    {
        'name': "Derivative chain rule: multi-stage",
        'genome': [
            (SpurGear, {'ratio': 2.0}),
            (RackAndPinion, {'pitch_radius': 0.1}),
            (Wedge, {'angle_rad': 0.5})
        ],
        'test_inputs': [0.1, 0.5, 1.0],
        'expect': "chain.derivative(x) ≈ product of stage derivatives"
    },
    {
        'name': "Governor derivative at clamp boundary",
        'genome': [(SpurGear, {'ratio': 5.0})],
        'post_build': lambda b: b.add_governor(min_val=-2.0, max_val=2.0),
        'test_inputs': [-1.0, -0.5, 0, 0.5, 1.0],
        'expect': "derivative() returns 0.0 when output is clamped"
    },
    {
        'name': "ScotchYoke derivative zero-crossing",
        'genome': [(ScotchYoke, {'amplitude': 1.0})],
        'test_inputs': [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi],
        'expect': "derivative = amplitude*cos(x); zero at π/2, 3π/2"
    },

    # ========================================================================
    # 5. UNIT PROPAGATION & GENERIC HANDLING
    # ========================================================================
    {
        'name': "ANGLE → LENGTH → ANGLE roundtrip",
        'genome': [
            (RackAndPinion, {'pitch_radius': 0.1}),
            (RackAndPinion, {'pitch_radius': 0.1})
        ],
        'test_inputs': [0, 0.5, 1.0, 2.0],
        'expect': "output ≈ input * (0.1 * 10) = input (within numeric tolerance)"
    },
    {
        'name': "LENGTH → LENGTH chain: Wedge → Wedge",
        'genome': [
            (Wedge, {'angle_rad': 0.3}),
            (Wedge, {'angle_rad': 0.4})
        ],
        'test_inputs': [0.1, 0.5, 1.0],
        'expect': "output = input * tan(0.3) * tan(0.4)"
    },
    {
        'name': "GENERIC unit doesn't leak incorrectly",
        'genome': [(SpurGear, {'ratio': 2.0})],
        'test_inputs': [1.0],
        'expect': "chain.domain.input_unit == ANGLE, not GENERIC"
    },
    {
        'name': "Unit mismatch caught at build time",
        'genome': [
            (SpurGear, {'ratio': 2.0}),  # ANGLE → ANGLE
            (Wedge, {'angle_rad': 0.3})   # expects LENGTH
        ],
        'test_inputs': [1.0],
        'expect': "ICBuilder.add() raises DimensionMismatchError"
    },

    # ========================================================================
    # 6. GOVERNOR INTEGRATION
    # ========================================================================
    {
        'name': "Governor makes chain non-invertible",
        'genome': [(SpurGear, {'ratio': 2.0})],
        'post_build': lambda b: b.add_governor(min_val=-1.0, max_val=1.0),
        'test_inputs': [-2.0, -0.5, 0, 0.5, 2.0],
        'expect': "chain.is_invertible == False; inverse() raises InverseUndefinedError"
    },
    {
        'name': "Governor hysteresis doesn't affect forward()",
        'genome': [(SpurGear, {'ratio': 1.0})],
        'post_build': lambda b: b.add_governor(min_val=-1.0, max_val=1.0, hysteresis=0.2),
        'test_inputs': [-2.0, -1.5, -1.0, 0, 1.0, 1.5, 2.0],
        'expect': "forward() clamps identically regardless of hysteresis value"
    },
    {
        'name': "Governor at chain start",
        'genome': [(SpurGear, {'ratio': 1.0})],
        'post_build': lambda b: b.add_governor(min_val=0.0, max_val=5.0),
        'test_inputs': [-10, -1, 0, 1, 10],
        'expect': "output clamped to [0, 5]; derivative zero outside bounds"
    },
    {
        'name': "Governor with tight bounds on high-ratio chain",
        'genome': [(SpurGear, {'ratio': 100.0})],
        'post_build': lambda b: b.add_governor(min_val=-0.1, max_val=0.1),
        'test_inputs': [-0.01, 0, 0.01],
        'expect': "tiny input range produces clamped output; derivative sensitive"
    },

    # ========================================================================
    # 7. PERIOD HANDLING
    # ========================================================================
    {
        'name': "Mixed-period chain reports aperiodic",
        'genome': [
            (SpurGear, {'ratio': 2.0}),  # Not periodic (linear, unbounded)
            (HookesJoint, {'shaft_angle': 0.3})
        ],
        'test_inputs': [0, 1, 2],
        'expect': "chain.period() == None"
    },
    {
        'name': "Same-period chain preserves period",
        'genome': [
            (HookesJoint, {'shaft_angle': 0.3}),  # period=2π
            (ScotchYoke, {'amplitude': 1.0}),     # period=2π
        ],
        'test_inputs': [0, 1, 2, 3, 4, 5, 6],
        'expect': "chain.period() == 2π"
    },
    {
        'name': "Normalize on periodic primitive",
        'genome': [(ScotchYoke, {'amplitude': 1.0})],
        'test_inputs': [-10, -2*math.pi, 0, 2*math.pi, 10],
        'expect': "normalize(x) wraps to [0, 2π); forward(normalize(x)) consistent"
    },

    # ========================================================================
    # 8. NUMERICAL STABILITY & EDGE VALUES
    # ========================================================================
    {
        'name': "Extreme ratio cascade overflow check",
        'genome': [
            (SpurGear, {'ratio': 1e4}),
            (SpurGear, {'ratio': 1e4}),
            (SpurGear, {'ratio': 1e-8})
        ],
        'test_inputs': [1e-6, 1e-3, 1.0, 1e3, 1e6],
        'expect': "no OverflowError; net ratio = 1.0; forward(x) ≈ x"
    },
    {
        'name': "Tiny amplitude ScotchYoke precision",
        'genome': [(ScotchYoke, {'amplitude': 1e-6})],
        'test_inputs': [0, 0.1, 1.0, 10.0],
        'expect': "forward() returns values ~1e-6; no underflow to zero"
    },
    {
        'name': "Large input to bounded primitive",
        'genome': [(ScotchYoke, {'amplitude': 1.0})],
        'test_inputs': [-1e6, -1e3, 0, 1e3, 1e6],
        'expect': "forward() bounded to ±1.0; no NaN/inf"
    },
    {
        'name': "Zero input to all primitives",
        'genome': [
            (SpurGear, {'ratio': 2.0}),
            (RackAndPinion, {'pitch_radius': 0.1}),
            (Wedge, {'angle_rad': 0.5})
        ],
        'test_inputs': [0.0],
        'expect': "forward(0) = 0 for linear primitives; sin(0)=0 for ScotchYoke"
    },

    # ========================================================================
    # 9. INVERTIBILITY EDGE CASES
    # ========================================================================
    {
        'name': "Non-invertible primitive in chain",
        'genome': [
            (SpurGear, {'ratio': 2.0}),
            (Governor, {'primitive': SpurGear(1.0), 'min_val': -1.0, 'max_val': 1.0})
        ],
        'test_inputs': [-2, -1, 0, 1, 2],
        'expect': "chain.is_invertible == False; inverse() raises"
    },
    {
        'name': "Branch selection propagation in inverse",
        'genome': [
            (SpurGear, {'ratio': 1.0}),
            (ScotchYoke, {'amplitude': 1.0})
        ],
        'test_inputs': [0.5],
        'expect': "inverse(y, branches=[None, 'supplementary']) uses correct branch"
    },
    {
        'name': "Guess propagation in Newton-Raphson inverse",
        'genome': [(EccentricCam, {'eccentricity': 0.1, 'follower_radius': 0.2})],
        'test_inputs': [0.15],
        'expect': "inverse(y, guess=1.0) converges faster than default"
    },

    # ========================================================================
    # 10. COMPOSITE PRIMITIVE BEHAVIOR
    # ========================================================================
    {
        'name': "Empty chain handling",
        'genome': [],
        'test_inputs': [1.0],
        'expect': "ICBuilder.build() raises ValueError on empty chain"
    },
    {
        'name': "Single primitive chain",
        'genome': [(SpurGear, {'ratio': 3.0})],
        'test_inputs': [-2, -1, 0, 1, 2],
        'expect': "forward(x) = 3*x; derivative = 3; inverse(y) = y/3"
    },
    {
        'name': "Long chain performance",
        'genome': [(SpurGear, {'ratio': 1.1}) for _ in range(20)],
        'test_inputs': [1.0],
        'expect': "forward() completes without stack overflow; net ratio ≈ 1.1^20"
    },
    {
        'name': "Domain propagation through long chain",
        'genome': [
            (SpurGear, {'ratio': 0.9}),
            (SpurGear, {'ratio': 0.9}),
            (SpurGear, {'ratio': 0.9}),
            (ScotchYoke, {'amplitude': 1.0})
        ],
        'test_inputs': [-10, -1, 0, 1, 10],
        'expect': "input_domain back-propagated through scaling stages"
    },
]

# ========================================================================
# HELPERS
# ========================================================================

def _random_params(cls: Type[Primitive]) -> Dict[str, Any]:
    """Generate sane random parameters for a primitive class"""
    import inspect
    params = {}
    sig = inspect.signature(cls.__init__)
    
    # Common parameter patterns with safe ranges
    param_ranges = {
        'ratio': (0.1, 10.0),
        'shaft_angle': (0.01, 1.5),  # radians, avoid π/2
        'pitch_radius': (0.01, 0.5),
        'angle_rad': (0.1, 1.4),
        'amplitude': (0.1, 5.0),
        'phase': (0, 2*math.pi),
        'eccentricity': (0.01, 0.3),
        'follower_radius': (0.1, 0.5),
        'crank_length': (0.1, 2.0),
        'rod_length': (0.5, 3.0),
        'offset_x': (-0.5, 0.5),
        'offset_y': (-0.5, 0.5),
    }
    
    for name, param in sig.parameters.items():
        if name in ('self', 'branch', 'theta_guess', 'is_analytically_invertible'):
            continue
            
        # Try to match by name pattern first
        matched = False
        for key, (min_val, max_val) in param_ranges.items():
            if key in name:
                params[name] = random.uniform(min_val, max_val)
                matched = True
                break
        
        if not matched:
            # Fallback: use default if available, else skip
            if param.default is not inspect.Parameter.empty:
                params[name] = param.default
            # Otherwise skip — let __post_init__ use its own defaults
    
    return params
def run_first_phase():
    print("="*60)
    print("Stress Test I")
    
    # First, hunt for random faults
    PRIMITIVE_POOL = [
        SpurGear, RackAndPinion, Wedge, HookesJoint,
        ScotchYoke, EccentricCam, CrankSlider, OldhamCoupling
    ]
    faults = hunt_for_faults(
        primitive_pool=PRIMITIVE_POOL,
        generations=500,
        pop_size=100
    )
    
def run_second_phase():
    print("=" * 60)
    print("Stress Test II")
    
    passed = 0
    for h in HYPOTHESES:
        result = test_hypothesis(h)
        status = "+" if result['passed'] else "!"
        print(f"\n{status} {result['name']}")
        
        if result.get('notes'):
            for note in result['notes']:
                print(f"   {note}")
        if result.get('errors'):
            for err in result['errors'][:2]:  # Show first 2 errors
                print(f"   ! {err}")
        
        if result['passed']:
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(HYPOTHESES)} hypotheses passed")

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Central difference approximation"""
    return (f(x + h) - f(x - h)) / (2 * h)


if __name__ == "__main__":
    #run_first_phase()
    run_second_phase()
