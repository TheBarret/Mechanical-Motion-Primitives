import math
import pytest
from fractions import Fraction

from mmp import (
    SpurGear, RackAndPinion, Wedge, Pantograph,
    ScotchYoke, CrankSlider, HookesJoint,
    PlanetaryGear, DifferentialGear,
    Ratchet, Detent, Escapement, GenevaDrive,
    WormGear, HarmonicDrive, DifferentialScrew, CVT, ToroidalCVT,
    TrochoidalGear, Spirograph, VernierScale,
    MechanicalChain
)

# ============================================================================
# CLASS I: LINEAR SCALING
# ============================================================================

def test_spur_gear_basic():
    """Gear with 2:1 ratio: 10 turns in = 20 turns out"""
    gear = SpurGear(ratio=2.0)
    assert gear.forward(10) == 20
    assert gear.inverse(20) == 10

def test_spur_gear_modular():
    """In modular arithmetic, 3 * 5 ≡ 1 (mod 7)"""
    gear = SpurGear(ratio=3, modulus=7)
    assert gear.forward(5) == 1  # (5*3) % 7 = 15 % 7 = 1
    assert gear.inverse(1) == 5  # (1 * 3^-1) % 7 = 1*5 % 7 = 5

def test_rack_and_pinion():
    """Pinion radius 2cm: 1 rotation = 12.57cm linear"""
    rack = RackAndPinion(pitch_radius=2.0)
    assert abs(rack.rotation_to_linear(math.pi) - (2 * math.pi)) < 1e-10
    assert abs(rack.linear_to_rotation(2*math.pi) - math.pi) < 1e-10

def test_wedge():
    """30° wedge: 10cm horizontal = 5.77cm vertical"""
    wedge = Wedge(angle_rad=math.radians(30))
    vertical = wedge.forward(10)
    assert abs(vertical - 10 * math.tan(math.radians(30))) < 1e-10
    assert abs(wedge.inverse(vertical) - 10) < 1e-10
    # Mechanical advantage at 30° = cot(30°) ≈ 1.732
    assert abs(wedge.mechanical_advantage - (1/math.tan(math.radians(30)))) < 1e-10

def test_pantograph():
    """2x scaling: (2,3) → (4,6)"""
    panto = Pantograph(scale=2.0)
    assert panto.forward((2, 3)) == (4, 6)
    assert panto.inverse((4, 6)) == (2, 3)

# ============================================================================
# CLASS II: PERIODIC NON-LINEAR
# ============================================================================

def test_scotch_yoke():
    """Amplitude 5: sin(π/2) = 5"""
    yoke = ScotchYoke(amplitude=5.0)
    assert abs(yoke.forward(math.pi/2) - 5.0) < 1e-10
    # Inverse on principal branch
    assert abs(yoke.inverse(5.0) - math.pi/2) < 1e-10
    # Check that sin(θ) = sin(π-θ)
    val = yoke.forward(math.pi/4)
    assert abs(yoke.inverse(val, 'principal') - math.pi/4) < 1e-10
    assert abs(yoke.inverse(val, 'supplementary') - (math.pi - math.pi/4)) < 1e-10

def test_crank_slider():
    """Crank 1, rod 10: at 0° position = 11"""
    slider = CrankSlider(crank_length=1.0, rod_length=10.0)
    assert abs(slider.forward(0) - 11.0) < 1e-10
    # At 90°: position = sqrt(10² - 1²) = sqrt(99) ≈ 9.95
    assert abs(slider.forward(math.pi/2) - 9.9498743710662) < 1e-5

def test_hookes_joint():
    """30° shaft angle: at 45° input, output ≠ 45°"""
    joint = HookesJoint(shaft_angle=math.radians(30))
    out = joint.forward(math.radians(45))
    # Known: tan(θ_out) = cos(30°) * tan(45°) = 0.866
    assert abs(math.tan(out) - 0.8660254) < 1e-5

# ============================================================================
# CLASS III: CONFIGURATION-DEPENDENT
# ============================================================================

def test_planetary_gear():
    """Standard planetary: sun 30, ring 70, planet 20"""
    planetary = PlanetaryGear(sun_teeth=30, planet_teeth=20, ring_teeth=70)
    
    # Ring fixed: carrier = (30/(30+70)) * sun = 0.3 * 100 = 30
    planetary.set_mode('ring_fixed')
    v = planetary.angular_velocity(sun_w=100, ring_w=0, carrier_w=0)
    assert abs(v['carrier'] - 30) < 1e-10
    
    # Sun fixed: carrier = (70/(30+70)) * ring = 0.7 * 100 = 70
    planetary.set_mode('sun_fixed')
    v = planetary.angular_velocity(sun_w=0, ring_w=100, carrier_w=0)
    assert abs(v['carrier'] - 70) < 1e-10
    
    # Carrier fixed: ring = -(30/70) * sun = -42.857
    planetary.set_mode('carrier_fixed')
    v = planetary.angular_velocity(sun_w=100, ring_w=0, carrier_w=0)
    assert abs(v['ring'] - (-30/70 * 100)) < 1e-10

def test_differential():
    """Classic diff: (10+20)/2 = 15"""
    diff = DifferentialGear()
    assert diff.forward(10, 20) == 15
    # Split: 15 input, 5 diff speed = 17.5, 12.5
    l, r = diff.split(15, diff_speed=5)
    assert abs(l - 17.5) < 1e-10
    assert abs(r - 12.5) < 1e-10
    # Inverse: given output 15 and left 10, right must be 20
    assert diff.inverse(15, 10) == 20

# ============================================================================
# CLASS IV: QUANTIZING / DISCRETIZING
# ============================================================================

def test_ratchet():
    """Ratchet with 2mm teeth: can only increase"""
    ratchet = Ratchet(tooth_pitch=2.0)
    assert ratchet.forward(1.5) == 2.0  # Rounds to nearest tooth
    assert ratchet.forward(0.5) == 2.0  # Can't go back
    assert ratchet.forward(3.2) == 4.0  # Advances to next

def test_detent():
    """Detent with 1.0 steps, 0.2 hysteresis"""
    detent = Detent(step=1.0, hysteresis=0.2)
    detent.forward(0.0)  # Initialize at 0
    assert detent.forward(0.15) == 0.0   # Within hysteresis of current position
    assert detent.forward(1.2) == 1.0    # Crossed threshold
    assert detent.forward(0.9) == 1.0    # Within hysteresis of new position

def test_escapement():
    """Clock escapement: ticks every 1.0 seconds"""
    escapement = Escapement(advance_per_tick=1.0, period=1.0)
    assert escapement.forward(0.5) is None   # No tick
    assert escapement.forward(1.0) == 1.0    # Tick at 1.0
    assert escapement.forward(1.5) is None   # No tick
    assert escapement.forward(2.0) == 2.0    # Tick at 2.0

def test_geneva_basic():
    geneva = GenevaDrive(num_slots=4)
    # First engagement might behave differently
    out1 = geneva.forward(0.1)
    out2 = geneva.forward(math.pi/4)
    # Add explicit assertion about initial state
    assert geneva.current_slot >= 0  # Should have engaged a slot
    assert abs(out2 - out1) < 1e-10

# ============================================================================
# CLASS V: ONE-WAY
# ============================================================================

def test_worm_gear_irreversible():
    """High friction worm: can't backdrive"""
    worm = WormGear(num_starts=1, friction_coefficient=0.3)
    # Lead angle small, friction high → not backdrivable
    assert not worm.is_backdrivable
    
    pos = worm.position
    worm.forward(10)      # Advance
    assert worm.position > pos
    
    pos = worm.position
    worm.forward(-5)      # Try to reverse
    assert worm.position == pos  # Should not move

def test_worm_gear_reversible():
    """Low friction worm: can backdrive"""
    worm = WormGear(num_starts=4, friction_coefficient=0.05)
    # Multi-start, low friction → backdrivable
    assert worm.is_backdrivable
    
    pos = worm.position
    worm.forward(10)
    assert worm.position > pos
    
    worm.forward(-5)      # Should reverse
    assert worm.position < pos + 10  # Moved backward

# ============================================================================
# CLASS VI: HIGH-RATIO
# ============================================================================

def test_harmonic_drive():
    """200/2 = 100:1 reduction"""
    harmonic = HarmonicDrive(flexspline_teeth=200, circular_spline_teeth=202)
    assert abs(harmonic.ratio - 100) < 1e-10
    # 100 input rotations = 1 output rotation
    assert abs(harmonic.forward(100) - 1.0) < 1e-10

def test_differential_screw():
    """Threads 2.0mm and 1.9mm: 0.1mm per rotation"""
    screw = DifferentialScrew(pitch1=2.0, pitch2=1.9)
    assert abs(screw.effective_pitch - 0.1) < 1e-10
    assert abs(screw.forward(10) - 1.0) < 1e-10  # 10 rotations = 1mm

def test_cvt():
    """CVT with control 0.75 between 0.5-2.0: ratio = 1.625"""
    cvt = CVT(min_ratio=0.5, max_ratio=2.0)
    cvt.set_control(0.75)
    assert abs(cvt.current_ratio - 1.625) < 1e-10
    assert cvt.forward(100) == 162.5

# ============================================================================
# CLASS VII: COMPLEX PATH
# ============================================================================

def test_trochoidal_position():
    """Known point on trochoid"""
    trochoid = TrochoidalGear(R=5, r=2, d=3)
    x, y = trochoid.position(0)
    assert abs(x - (5+2 - 3)) < 1e-10  # At t=0: x = R+r-d
    assert abs(y - 0) < 1e-10

def test_spirograph_closure():
    """R=60, r=20 gives 3 loops before closing"""
    spiro = Spirograph(R=60, r=20, d=30)
    assert spiro.num_loops() == 3
    # Position at t=0 and t=2π should be same
    x0, y0 = spiro.position(0)
    x1, y1 = spiro.position(2 * math.pi)
    assert abs(x0 - x1) < 1e-10
    assert abs(y0 - y1) < 1e-10

# ============================================================================
# CLASS VIII: COINCIDENCE
# ============================================================================

# !! currently broke !!
def test_vernier():
    """Main 1.0mm, vernier 0.9mm: 10:1 resolution"""
    vernier = VernierScale(main_scale_spacing=1.0, vernier_spacing=0.9)
    
    # Offset 0.0: perfect alignment at n=0
    n, m, pos = vernier.find_coincidence(0.0)
    assert n == 0
    
    # Offset 0.1: vernier mark 1 aligns with main mark 1
    n, m, pos = vernier.find_coincidence(0.1)
    assert n == 1
    
    # Offset 0.5: vernier mark 5 aligns with main mark 5
    n, m, pos = vernier.find_coincidence(0.5)
    assert n == 5
    
    # Measure method should return position
    assert abs(vernier.measure(0.3) - 3.0) < 1e-10

# ============================================================================
# COMPOSITION
# ============================================================================

# !! currently broke !!
def test_mechanical_chain():
    """Chain of invertible mechanisms should be invertible"""
    chain = MechanicalChain([
        SpurGear(2.0),
        Pantograph(1.5),
        Wedge(math.radians(15))
    ])
    
    assert chain.is_fully_invertible()
    
    # Test round-trip
    x = 10
    y = chain.forward(x)
    x2 = chain.inverse(y)
    assert abs(x2 - x) < 1e-10

def test_chain_with_oneway():
    """Chain with non-invertible should raise on inverse"""
    chain = MechanicalChain([
        SpurGear(2.0),
        Ratchet(tooth_pitch=1.0)  # Not invertible
    ])
    
    assert not chain.is_fully_invertible()
    
    # Forward works
    y = chain.forward(5)
    assert y > 0
    
    # Inverse should raise
    with pytest.raises(ValueError, match="non-invertible"):
        chain.inverse(y)

# ============================================================================
# EDGE CASES
# ============================================================================

def test_scotch_yoke_out_of_range():
    """Requesting inverse outside [-A,A] should raise"""
    yoke = ScotchYoke(amplitude=5.0)
    with pytest.raises(ValueError, match="outside amplitude range"):
        yoke.inverse(10.0)

def test_planetary_invalid_mode():
    """Setting invalid mode should raise"""
    planetary = PlanetaryGear(30, 20, 70)
    with pytest.raises(ValueError, match="Mode must be one of"):
        planetary.set_mode('invalid')

def test_toroidal_cvt_near_singularity():
    """Angle near π/2 should raise"""
    cvt = ToroidalCVT()
    with pytest.raises(ValueError, match="too close to pi/2"):
        cvt.set_tilt(math.pi/2 - 0.001)


def test_compound_gear_train():
    """Multiple gears: 2:1 * 3:1 = 6:1 total"""
    train = CompoundGearTrain(ratios=[2.0, 3.0])
    assert train.total_ratio == 6.0
    assert train.forward(10) == 60
    assert train.inverse(60) == 10

def test_four_bar_linkage():
    """Freudenstein equation should equal 0 for valid configuration"""
    linkage = FourBarLinkage(a=4, b=5, c=3, d=6)
    # At a valid configuration, Freudenstein equation = 0
    theta_out = linkage.forward_numerical(math.pi/4, guess=math.pi/3)
    residual = linkage.freudenstein(math.pi/4, theta_out)
    assert abs(residual) < 1e-8

def test_eccentric_cam_numerical():
    """Cam inverse should converge numerically"""
    cam = EccentricCam(eccentricity=0.5, follower_radius=2.0)
    y = cam.forward(math.pi/4)
    theta = cam.inverse_numerical(y, theta_guess=math.pi/4)
    assert abs(theta - math.pi/4) < 1e-6

def test_toroidal_cvt_basic():
    """Toroidal CVT ratio = tan(α)"""
    cvt = ToroidalCVT()
    cvt.set_tilt(math.pi/4)  # 45°
    assert abs(cvt.ratio - 1.0) < 1e-10  # tan(45°) = 1
    assert cvt.forward(100) == 100

def test_escapement_force_threshold():
    """Escapement requires positive force"""
    escapement = Escapement(advance_per_tick=1.0, period=1.0)
    assert escapement.forward(1.0, force=0) is None
    assert escapement.forward(1.0, force=-1) is None
    assert escapement.forward(1.0, force=1) == 1.0

def test_mechanical_chain_1d():
    """Chain of 1D invertible mechanisms should be invertible"""
    chain = MechanicalChain([
        SpurGear(2.0),
        RackAndPinion(pitch_radius=1.0),
        Wedge(math.radians(15))
    ])
    
    assert chain.is_fully_invertible()
    
    x = 10.0
    y = chain.forward(x)
    x2 = chain.inverse(y)
    assert abs(x2 - x) < 1e-10

def test_mechanical_chain_2d():
    """Chain of 2D invertible mechanisms should be invertible"""
    chain = MechanicalChain([
        Pantograph(scale=2.0),
        OldhamCoupling(delta_x=1.0, delta_y=1.0),
    ])
    
    assert chain.is_fully_invertible()
    
    point = (10.0, 20.0)
    result = chain.forward(point)
    original = chain.inverse(result)
    assert abs(original[0] - point[0]) < 1e-10
    assert abs(original[1] - point[1]) < 1e-10

if __name__ == "__main__":
    import sys
    # Run tests with pytest or manually
    print("Running basic validation tests...")
    test_spur_gear_basic()
    print("✓ Spur gear")
    test_rack_and_pinion()
    print("✓ Rack and pinion")
    test_scotch_yoke()
    print("✓ Scotch yoke")
    test_planetary_gear()
    print("✓ Planetary gear")
    test_ratchet()
    print("✓ Ratchet")
    test_worm_gear_irreversible()
    print("✓ Worm gear")
    test_harmonic_drive()
    print("✓ Harmonic drive")
    test_vernier()
    print("✓ Vernier scale")
    #test_mechanical_chain()
    #print("✓ Mechanical chain")
    print("\nAll basic tests passed!")
    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)