"""
MMP Framework Test Application
"""

import math
import sys
from pprint import pprint

from core import (
    Dimension, Domain, Primitive, Invertible, Periodic,
    MechanicalLimits, MMPError, CompositionError, DimensionMismatchError
)
from primitives import (
    # Class I
    SpurGear, CompoundGearTrain, RackAndPinion, Wedge, OldhamCoupling,
    # Class II
    ScotchYoke, EccentricCam, CrankSlider, HookesJoint,
    # Class III Adapters
    AngleToLength, LengthToAngle, UnitlessScaling, Bias, FunctionAdapter
)
from composite import CompositePrimitive, Governor
from builders import ICBuilder, CCBuilder

def print_header(title: str) -> None:
    """Pretty print section headers"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_core_layer() -> None:
    """Test core domain and dimension functionality"""
    print_header("TESTING CORE LAYER")
    
    # Test Domain creation
    angle_domain = Domain(
        min=-math.pi,
        max=math.pi,
        input_unit=Dimension.ANGLE,
        output_unit=Dimension.ANGLE
    )
    #print(f"Angle Domain: {angle_domain}")
    #print(f"Contains 0? {angle_domain.contains(0)}")
    #print(f"Contains 4? {angle_domain.contains(4)}")
    
    # Test bounded periodic domain
    scotch_domain = Domain(
        min=None,  # unbounded input
        max=None,
        input_unit=Dimension.ANGLE,
        output_unit=Dimension.LENGTH,
        output_min=-2.0,
        output_max=2.0
    )
    #print(f"\nScotchYoke Domain: {scotch_domain}")
    print(f"Output contains 1.5? {scotch_domain.output_contains(1.5)}")
    print(f"Output contains 3.0? {scotch_domain.output_contains(3.0)}")
    
    # Test MechanicalLimits constants
    print(f"\nMechanical Limits:")
    print(f"  MAX_ANGLE = {MechanicalLimits.MAX_ANGLE:.2f} rad")
    print(f"  MAX_RATIO = {MechanicalLimits.MAX_RATIO}")


def test_class_i_primitives() -> None:
    """Test linear/scaling primitives"""
    print_header("TESTING CLASS I: LINEAR PRIMITIVES")
    
    # Spur Gear
    gear = SpurGear(ratio=2.5)
    print(f"SpurGear (ratio=2.5):")
    print(f"  forward(1.0) = {gear.forward(1.0)}")
    print(f"  inverse(2.5) = {gear.inverse(2.5)}")
    print(f"  derivative(any) = {gear.derivative(0)}")
    print(f"  is_monotonic? {gear.is_monotonic}")
    
    # Compound Gear Train
    gear_train = CompoundGearTrain(ratios=[2.0, 1.5, 0.5])
    print(f"\nCompoundGearTrain (ratios=[2.0, 1.5, 0.5]):")
    print(f"  forward(1.0) = {gear_train.forward(1.0)} (product = 1.5)")
    print(f"  inverse(1.5) = {gear_train.inverse(1.5)}")
    
    # Rack and Pinion
    rack = RackAndPinion(pitch_radius=0.1)
    print(f"\nRackAndPinion (radius=0.1m):")
    print(f"  forward(2π rad) = {rack.forward(2*math.pi):.4f} m")
    print(f"  inverse(0.628m) = {rack.inverse(0.628):.4f} rad")
    
    # Wedge
    wedge = Wedge(angle_rad=math.radians(30))
    print(f"\nWedge (30°):")
    print(f"  mechanical_advantage = {wedge.mechanical_advantage:.4f}")
    print(f"  forward(10cm) = {wedge.forward(10):.4f} cm")
    
    # Oldham Coupling
    oldham = OldhamCoupling(offset_x=0.05, offset_y=0.03)
    print(f"\nOldhamCoupling (offset=(0.05,0.03)):")
    print(f"  forward(1.57) = {oldham.forward(1.57)} (identity)")
    print(f"  shaft_offset = {oldham.shaft_offset}")


def test_class_ii_primitives() -> None:
    """Test periodic/trigonometric primitives"""
    print_header("TESTING CLASS II: PERIODIC PRIMITIVES")
    
    # Scotch Yoke
    yoke = ScotchYoke(amplitude=2.0, phase=0.0)
    print(f"ScotchYoke (amplitude=2.0):")
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
        print(f"  forward({angle:6.2f}) = {yoke.forward(angle):6.2f}")
    print(f"  period = {yoke.period():.4f}")
    print(f"  inverse(0, 'principal') = {yoke.inverse(0, branch='principal'):.4f}")
    print(f"  inverse(0, 'supplementary') = {yoke.inverse(0, branch='supplementary'):.4f}")
    
    # Crank Slider
    slider = CrankSlider(crank_length=0.5, rod_length=1.0)
    print(f"\nCrankSlider (crank=0.5, rod=1.0):")
    print(f"  range = [{slider.domain.output_min:.3f}, {slider.domain.output_max:.3f}]")
    for angle in [0, math.pi/2, math.pi]:
        print(f"  forward({angle:6.2f}) = {slider.forward(angle):6.4f}")
    
    # Hooke's Joint
    hooke = HookesJoint(shaft_angle=math.radians(30))
    print(f"\nHookesJoint (30°):")
    for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4]:
        print(f"  forward({angle:6.2f}) = {hooke.forward(angle):6.2f}")
        print(f"  velocity_ratio = {hooke.angular_velocity_ratio(angle):6.4f}")


def test_class_iii_adapters() -> None:
    """Test fictional adapter primitives"""
    print_header("TESTING CLASS III: ADAPTERS")
    
    # Angle to Length
    a2l = AngleToLength(scale=0.1)
    print(f"AngleToLength (scale=0.1):")
    print(f"  forward(2π) = {a2l.forward(2*math.pi):.4f} m")
    print(f"  inverse(0.628) = {a2l.inverse(0.628):.4f} rad")
    
    # Length to Angle
    l2a = LengthToAngle(scale=10.0)
    print(f"\nLengthToAngle (scale=10.0):")
    print(f"  forward(0.628) = {l2a.forward(0.628):.4f} rad")
    
    # Unitless Scaling
    scaling = UnitlessScaling(scale=2.0, unit=Dimension.ANGLE)
    print(f"\nUnitlessScaling (scale=2.0):")
    print(f"  forward(1.57) = {scaling.forward(1.57):.4f} rad")
    
    # Bias
    bias = Bias(bias=1.0, unit=Dimension.LENGTH)
    print(f"\nBias (+1.0):")
    print(f"  forward(2.0) = {bias.forward(2.0)} m")
    print(f"  inverse(3.0) = {bias.inverse(3.0)} m")
    
    # Function Adapter
    square = FunctionAdapter(
        fwd=lambda x: x**2,
        inv=lambda y: math.sqrt(y),
        input_unit=Dimension.LENGTH,
        output_unit=Dimension.ANGLE,
        name="square",
        deriv=lambda x: 2*x
    )
    print(f"\nFunctionAdapter (square):")
    print(f"  forward(3.0) = {square.forward(3.0)}")
    print(f"  inverse(9.0) = {square.inverse(9.0)}")
    print(f"  derivative(3.0) = {square.derivative(3.0)}")


def test_governor() -> None:
    """Test Governor wrapper"""
    print_header("TESTING GOVERNOR")
    
    # Create a primitive and wrap it
    gear = SpurGear(ratio=2.0)
    governor = Governor(gear, min_val=-5.0, max_val=5.0, hysteresis=0.1)
    
    print(f"Governor (min=-5, max=5, hyst=0.1):")
    for x in [-3.0, -2.5, 0.0, 2.5, 3.0]:
        y_gear = gear.forward(x)
        y_gov = governor.forward(x)
        print(f"  x={x:4.1f} -> gear={y_gear:5.1f}, governed={y_gov:5.1f}")
    
    print(f"  is_invertible? {governor.is_invertible} (always False)")
    print(f"  derivative at limit (x=2.6): {governor.derivative(2.6)}")


def test_composite_layer() -> None:
    """Test CompositePrimitive with various chains"""
    print_header("TESTING COMPOSITE LAYER")
    
    # Simple gear train
    primitives = (
        SpurGear(ratio=2.0),
        SpurGear(ratio=1.5),
        SpurGear(ratio=0.5)
    )
    composite = CompositePrimitive(primitives)
    print(f"Simple composite (gears):")
    print(f"  forward(1.0) = {composite.forward(1.0)} (should be 1.5)")
    print(f"  inverse(1.5) = {composite.inverse(1.5)}")
    print(f"  derivative = {composite.derivative(1.0)}")
    print(f"  is_invertible? {composite.is_invertible}")
    
    # Mixed units chain (rotation -> linear -> rotation)
    mixed = CompositePrimitive((
        SpurGear(ratio=2.0),                    # ANGLE -> ANGLE
        RackAndPinion(pitch_radius=0.1),        # ANGLE -> LENGTH
        LengthToAngle(scale=10.0)                # LENGTH -> ANGLE
    ))
    print(f"\nMixed units composite (ANGLE->LENGTH->ANGLE):")
    print(f"  forward(1.57) = {mixed.forward(1.57):.4f} rad")
    print(f"  inverse(3.14) = {mixed.inverse(3.14):.4f} rad")
    print(f"  input unit: {mixed.domain.input_unit}")
    print(f"  output unit: {mixed.domain.output_unit}")
    
    # Test domain computation
    print(f"\nComposite domain:")
    print(f"  input range: [{mixed.input_domain.min}, {mixed.input_domain.max}]")
    print(f"  output range: [{mixed.output_domain.output_min}, {mixed.output_domain.output_max}]")


def test_ic_builder() -> None:
    """Test Industrial/Conservative builder"""
    print_header("TESTING IC BUILDER")
    
    # Build a wrist actuator chain
    try:
        wrist = (ICBuilder()
            .add(SpurGear, ratio=2.5)                     # Motor gearbox
            .add(HookesJoint, shaft_angle=0.3)            # Angled transmission
            .add(RackAndPinion, pitch_radius=0.1)         # Convert to linear
            .add_governor(min_val=-2.0, max_val=2.0)      # Safety limits
            .build())
        
        print(f"Wrist actuator chain built successfully!")
        print(f"  Number of primitives: {len(wrist.primitives)}")
        print(f"  Is invertible? {wrist.is_invertible} (False due to governor)")
        
        # Test forward pass
        for angle in [0, 0.5, 1.0, 1.5, 2.0]:
            pos = wrist.forward(angle)
            print(f"  input {angle:4.2f} rad -> output {pos:6.4f} m")
        
    except CompositionError as e:
        print(f"Composition error: {e}")


def test_cc_builder() -> None:
    """Test Creative/Experimental builder with adapters"""
    print_header("TESTING CC BUILDER")
    
    # Build an experimental chain with adapters
    try:
        experiment = (CCBuilder()
            .add(SpurGear, ratio=2.5)                      # ANGLE
            .add(RackAndPinion, pitch_radius=0.1)          # ANGLE -> LENGTH
            .add_adapter(to_unit=Dimension.ANGLE, scale=10.0)  # LENGTH -> ANGLE
            .add(HookesJoint, shaft_angle=0.3)             # ANGLE
            .add(ScotchYoke, amplitude=0.05)               # ANGLE -> LENGTH
            .build())
        
        print(f"Experimental chain built successfully!")
        print(f"  Primitive count: {len(experiment.primitives)}")
        print(f"  Input unit: {experiment.domain.input_unit}")
        print(f"  Output unit: {experiment.domain.output_unit}")
        
        # Test a few points
        for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
            result = experiment.forward(angle)
            print(f"  input {angle:6.2f} rad -> output {result:6.4f} m")
        
    except CompositionError as e:
        print(f"Composition error: {e}")


def test_error_handling() -> None:
    """Test error cases and validation"""
    print_header("TESTING ERROR HANDLING")
    
    # Dimension mismatch
    print("Testing dimension mismatch:")
    try:
        composite = CompositePrimitive((
            SpurGear(ratio=2.0),           # ANGLE output
            ScotchYoke(amplitude=1.0)       # expects ANGLE input (this is actually fine)
        ))
        print("  ✓ ANGLE->ANGLE works")
    except CompositionError as e:
        print(f"  ✗ Unexpected error: {e}")
    
    try:
        composite = CompositePrimitive((
            RackAndPinion(pitch_radius=0.1),   # LENGTH output
            SpurGear(ratio=2.0)                  # expects ANGLE input
        ))
        print("  This should not print (error expected)")
    except DimensionMismatchError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    # Domain violation
    print("\nTesting domain violation:")
    try:
        yoke = ScotchYoke(amplitude=2.0)
        # Try to inverse with value outside output range
        result = yoke.inverse(3.0)  # Should raise
        print("  This should not print")
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")


def test_utils_functionality() -> None:
    """Test utils (if implemented)"""
    print_header("TESTING UTILITIES")
    
    # This assumes you've implemented the utils module
    try:
        from mmp.utils.math_helpers import generate_sample_points, unwrap_angle
        
        samples = generate_sample_points(-1.0, 1.0, num_samples=5)
        print(f"Sample points [-1,1] (5 samples): {samples}")
        
        # Test angle unwrapping
        unwrapped = unwrap_angle(6.5, reference=0.0)
        print(f"Unwrap 6.5 rad -> {unwrapped:.4f} rad")
        
    except ImportError:
        print("Utils module not fully implemented yet - skipping")


def main():
    """Main test routine"""
    print("\n" + "=" * 80)
    print(" MMP FRAMEWORK TEST SUITE - MODULAR LAYOUT")
    print("=" * 80)
    
    # Test all layers
    test_core_layer()
    test_class_i_primitives()
    test_class_ii_primitives()
    test_class_iii_adapters()
    test_governor()
    test_composite_layer()
    test_ic_builder()
    test_cc_builder()
    test_error_handling()
    test_utils_functionality()
    
    print_header("ALL TESTS COMPLETED")
    print("If you see this without uncaught exceptions, the modular layout works!")


if __name__ == "__main__":
    main()