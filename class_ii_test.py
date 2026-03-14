"""
Test suite for Mechanical Motion Primitives Class II

"""

import pytest
import math

from core import (
    Dimension, Domain, Primitive, Invertible, Periodic, OneWay, 
    PeriodicBranchDependent, PeriodicBijective,
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

# ============================================================================
# Test ScotchYoke (PeriodicBranchDependent, analytical inverse)
# ============================================================================

class TestScotchYoke:
    def test_creation(self):
        """Valid yoke creation"""
        y = ScotchYoke(amplitude=2.0, phase=0.5, branch='principal')
        assert y.amplitude == 2.0
        assert y.phase == 0.5
        assert y.branch == 'principal'
        assert y.is_invertible is True
        assert y.is_analytically_invertible is True
        assert y.domain.input_unit == Dimension.ANGLE
        assert y.domain.output_unit == Dimension.LENGTH
        # Input domain is unbounded
        assert y.domain.min is None
        assert y.domain.max is None
        # Output bounds are in output_min/output_max
        assert y.domain.output_min == -2.0
        assert y.domain.output_max == 2.0

    def test_zero_amplitude_raises(self):
        """Amplitude cannot be zero"""
        with pytest.raises(ValueError, match="cannot be zero"):
            ScotchYoke(amplitude=0.0)

    def test_invalid_branch_raises(self):
        """Branch must be valid"""
        with pytest.raises(ValueError, match="branch must be one of"):
            ScotchYoke(amplitude=2.0, branch='invalid')

    def test_available_branches(self):
        """Should list both branches"""
        y = ScotchYoke(amplitude=2.0)
        assert set(y.available_branches) == {'principal', 'supplementary'}

    def test_forward(self):
        """Forward: rotation -> displacement"""
        y = ScotchYoke(amplitude=2.0, phase=0.0)
        
        # sin(0) = 0
        assert y.forward(0.0) == 0.0
        
        # sin(π/2) = 1
        assert y.forward(math.pi/2) == pytest.approx(2.0)
        
        # sin(π) = 0
        assert y.forward(math.pi) == pytest.approx(0.0)
        
        # sin(3π/2) = -1
        assert y.forward(3*math.pi/2) == pytest.approx(-2.0)

    def test_forward_with_phase(self):
        """Phase shifts the sine wave"""
        y = ScotchYoke(amplitude=2.0, phase=math.pi/2)
        
        # sin(0 + π/2) = 1
        assert y.forward(0.0) == pytest.approx(2.0)
        
        # sin(π/2 + π/2) = sin(π) = 0
        assert y.forward(math.pi/2) == pytest.approx(0.0)

    def test_inverse_principal_branch(self):
        """Principal branch: inverse in [-π/2, π/2]"""
        y = ScotchYoke(amplitude=2.0, phase=0.0, branch='principal')
        
        # y = 0 -> θ = 0
        assert y.inverse(0.0) == pytest.approx(0.0)
        
        # y = 2.0 -> θ = π/2
        assert y.inverse(2.0) == pytest.approx(math.pi/2)
        
        # y = -2.0 -> θ = -π/2
        assert y.inverse(-2.0) == pytest.approx(-math.pi/2)
        
        # y = 1.0 -> θ = arcsin(0.5) = π/6
        assert y.inverse(1.0) == pytest.approx(math.pi/6)

    def test_inverse_supplementary_branch(self):
        """Supplementary branch: inverse in [π/2, 3π/2]"""
        y = ScotchYoke(amplitude=2.0, phase=0.0, branch='supplementary')
        
        # y = 0 -> θ = π (sin(π) = 0)
        assert y.inverse(0.0) == pytest.approx(math.pi)
        
        # y = 2.0 -> θ = π/2 (sin(π/2) = 1)
        assert y.inverse(2.0) == pytest.approx(math.pi/2)
        
        # y = -2.0 -> θ = 3π/2 (sin(3π/2) = -1)
        assert y.inverse(-2.0) == pytest.approx(3*math.pi/2)
        
        # y = 1.0 -> θ = π - π/6 = 5π/6
        assert y.inverse(1.0) == pytest.approx(5*math.pi/6)

    def test_inverse_out_of_range_raises(self):
        """Inverse fails for |y| > amplitude"""
        y = ScotchYoke(amplitude=2.0)
        with pytest.raises(ValueError, match="outside amplitude range|outside output range"):
            y.inverse(3.0)

    def test_period(self):
        """Period should be 2π"""
        y = ScotchYoke(amplitude=2.0)
        assert y.period() == 2 * math.pi

    def test_normalize(self):
        """Normalize wraps angle to [0, 2π)"""
        y = ScotchYoke(amplitude=2.0)
        assert y.normalize(3 * math.pi) == pytest.approx(math.pi)  # 3π → π
        assert y.normalize(-math.pi/2) == pytest.approx(3*math.pi/2)

    def test_roundtrip_principal(self):
        """forward then inverse returns original (principal branch)"""
        y = ScotchYoke(amplitude=2.0, phase=0.0, branch='principal')
        theta = 0.7  # Within principal branch range
        assert y.inverse(y.forward(theta)) == pytest.approx(theta)

    def test_roundtrip_supplementary(self):
        """forward then inverse returns original (supplementary branch)"""
        y = ScotchYoke(amplitude=2.0, phase=0.0, branch='supplementary')
        theta = 2.5  # Within supplementary branch range
        assert y.inverse(y.forward(theta)) == pytest.approx(theta)


# ============================================================================
# Test EccentricCam (PeriodicBranchDependent, numerical inverse)
# ============================================================================

class TestEccentricCam:
    def test_creation(self):
        """Valid cam creation"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0, 
                         branch='principal', theta_guess=0.5)
        assert c.eccentricity == 1.0
        assert c.follower_radius == 5.0
        assert c.branch == 'principal'
        assert c.theta_guess == 0.5
        assert c.is_invertible is True
        assert c.is_analytically_invertible is False
        assert c.domain.input_unit == Dimension.ANGLE
        assert c.domain.output_unit == Dimension.LENGTH
        # Input domain is unbounded
        assert c.domain.min is None
        assert c.domain.max is None
        # Output bounds in output_min/output_max
        assert c.domain.output_min == 4.0  # r - e
        assert c.domain.output_max == 6.0  # r + e

    def test_eccentricity_zero_raises(self):
        """Eccentricity must be positive"""
        with pytest.raises(ValueError, match="positive"):
            EccentricCam(eccentricity=0.0, follower_radius=5.0)

    def test_follower_radius_too_small_raises(self):
        """Follower radius must exceed eccentricity"""
        with pytest.raises(ValueError, match="must exceed"):
            EccentricCam(eccentricity=3.0, follower_radius=2.0)

    def test_invalid_branch_raises(self):
        """Branch must be valid"""
        with pytest.raises(ValueError):
            EccentricCam(eccentricity=1.0, follower_radius=5.0, branch='invalid')

    def test_forward_at_key_points(self):
        """Forward: cam angle -> follower displacement"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)
        
        # θ = 0: e*1 + sqrt(25 - 0) = 1 + 5 = 6
        assert c.forward(0.0) == pytest.approx(6.0)
        
        # θ = π/2: e*0 + sqrt(25 - 1) = 0 + sqrt(24) ≈ 4.898979
        assert c.forward(math.pi/2) == pytest.approx(4.898979, rel=1e-5)
        
        # θ = π: e*(-1) + sqrt(25 - 0) = -1 + 5 = 4
        assert c.forward(math.pi) == pytest.approx(4.0)
        
        # θ = 3π/2: e*0 + sqrt(25 - 1) = 0 + sqrt(24) ≈ 4.898979
        assert c.forward(3*math.pi/2) == pytest.approx(4.898979, rel=1e-5)

    def test_derivative(self):
        """Analytical derivative should match numerical approximation"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)
        theta = 1.0
        
        # Analytical
        d_analytical = c._derivative(theta)
        
        # Numerical approximation
        h = 1e-8
        d_numerical = (c.forward(theta + h) - c.forward(theta)) / h
        
        assert d_analytical == pytest.approx(d_numerical, rel=1e-5)

    def test_inverse_converges(self):
        """Inverse should find correct angle"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0, 
                         theta_guess=0.5)
        
        # Pick a test angle
        theta = 1.2
        y = c.forward(theta)
        
        # Inverse should recover theta
        assert c.inverse(y) == pytest.approx(theta, rel=1e-6)
    
    def test_inverse_with_different_guess(self):
        """
        EccentricCam forward is symmetric: forward(θ) == forward(2π - θ)
        Branch determines which of the two solutions is returned.
        We verify the result is in the correct region and reproduces y.
        """
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)

        # Principal branch — solutions in [0, π]
        c.branch = 'principal'
        theta = 0.8
        y = c.forward(theta)
        result = c.inverse(y)
        assert 0 <= result <= math.pi, f"Result {result} not in principal branch [0, π]"
        assert c.forward(result) == pytest.approx(y, rel=1e-6)

        # Supplementary branch — solutions in [π, 2π]
        c.branch = 'supplementary'
        theta = 4.0  # in [π, 2π]
        y = c.forward(theta)
        result = c.inverse(y)
        assert math.pi <= result <= 2 * math.pi, f"Result {result} not in supplementary branch [π, 2π]"
        assert c.forward(result) == pytest.approx(y, rel=1e-6)


    def test_inverse_out_of_domain_raises(self):
        """Inverse fails for y outside [r-e, r+e]"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)
        
        # Should raise domain error, not convergence error
        with pytest.raises(ValueError, match="outside output range"):
            c.inverse(10.0)  # > max (6.0)
        
        with pytest.raises(ValueError, match="outside output range"):
            c.inverse(2.0)  # < min (4.0)


    def test_inverse_at_extrema(self):
        """Inverse at min/max displacement should return exact angles"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)
        
        # Maximum displacement at θ=0
        y_max = c.domain.output_max  # 6.0, not domain.max
        result = c.inverse(y_max)
        assert result == pytest.approx(0.0, abs=1e-9)
        
        # Minimum displacement at θ=π
        y_min = c.domain.output_min  # 4.0, not domain.min
        result = c.inverse(y_min)
        assert result == pytest.approx(math.pi, abs=1e-9)

    def test_period(self):
        """Period should be 2π"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)
        assert c.period() == 2 * math.pi

    def test_normalize(self):
        """Normalize wraps angle"""
        c = EccentricCam(eccentricity=1.0, follower_radius=5.0)
        assert c.normalize(3 * math.pi) == pytest.approx(math.pi)


# ============================================================================
# Test CrankSlider (PeriodicBranchDependent, numerical inverse)
# ============================================================================

class TestCrankSlider:
    def test_creation(self):
        """Valid crank-slider creation"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0,
                         branch='principal', theta_guess=0.5)
        assert cs.crank_length == 1.0
        assert cs.rod_length == 4.0
        assert cs.branch == 'principal'
        assert cs.theta_guess == 0.5
        assert cs.is_invertible is True
        assert cs.is_analytically_invertible is False
        assert cs.domain.input_unit == Dimension.ANGLE
        assert cs.domain.output_unit == Dimension.LENGTH
        # Input domain is unbounded
        assert cs.domain.min is None
        assert cs.domain.max is None
        # Output bounds in output_min/output_max
        assert cs.domain.output_min == 3.0
        assert cs.domain.output_max == 5.0

    def test_crank_length_zero_raises(self):
        """Crank length must be positive"""
        with pytest.raises(ValueError, match="positive"):
            CrankSlider(crank_length=0.0, rod_length=4.0)

    def test_rod_length_too_short_raises(self):
        """Rod length must exceed crank length"""
        with pytest.raises(ValueError, match="must exceed"):
            CrankSlider(crank_length=3.0, rod_length=2.0)

    def test_forward_at_key_points(self):
        """Forward: crank angle -> slider position"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)

        # θ=0: r*1 + sqrt(16-0) = 1+4 = 5
        assert cs.forward(0.0) == pytest.approx(5.0)

        # θ=π/2: r*0 + sqrt(16-1) = sqrt(15) ≈ 3.87298
        assert cs.forward(math.pi/2) == pytest.approx(3.87298, rel=1e-5)

        # θ=π: r*(-1) + sqrt(16-0) = -1+4 = 3
        assert cs.forward(math.pi) == pytest.approx(3.0)

        # θ=3π/2: r*0 + sqrt(16-1) = sqrt(15) ≈ 3.87298
        assert cs.forward(3*math.pi/2) == pytest.approx(3.87298, rel=1e-5)

    def test_derivative(self):
        """Analytical derivative should match numerical approximation"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)
        theta = 1.0

        d_analytical = cs._derivative(theta)
        h = 1e-8
        d_numerical = (cs.forward(theta + h) - cs.forward(theta)) / h

        assert d_analytical == pytest.approx(d_numerical, rel=1e-5)

    def test_inverse_principal_branch(self):
        """Inverse returns solution in [0, π] for principal branch"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0, branch='principal')
        theta = 0.8
        y = cs.forward(theta)
        result = cs.inverse(y)

        assert 0 <= result <= math.pi
        assert cs.forward(result) == pytest.approx(y, rel=1e-6)

    def test_inverse_supplementary_branch(self):
        """Inverse returns solution in [π, 2π] for supplementary branch"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0, branch='supplementary')
        theta = 4.0  # In [π, 2π]
        y = cs.forward(theta)
        result = cs.inverse(y)

        assert math.pi <= result <= 2 * math.pi
        assert cs.forward(result) == pytest.approx(y, rel=1e-6)

    def test_inverse_at_extrema(self):
        """Inverse at min/max displacement returns exact angles"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)

        # Maximum displacement at θ=0
        result = cs.inverse(cs.domain.output_max)  # Use output_max, not domain.max
        assert result == pytest.approx(0.0, abs=1e-9)

        # Minimum displacement at θ=π
        result = cs.inverse(cs.domain.output_min)  # Use output_min, not domain.min
        assert result == pytest.approx(math.pi, abs=1e-9)

    def test_inverse_out_of_domain_raises(self):
        """Inverse fails for y outside [L-r, L+r]"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)

        with pytest.raises(ValueError, match="outside output range"):
            cs.inverse(10.0)  # above max

        with pytest.raises(ValueError, match="outside output range"):
            cs.inverse(1.0)   # below min

    def test_round_trip_all_branches(self):
        """
        Encode/decode round-trip for both branches.
        Correctness criterion: forward(inverse(y)) == y
        Not: inverse(forward(x)) == x  (not guaranteed due to symmetry)
        """
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)

        test_angles = {
            'principal':     [0.1, 0.5, 1.0, 2.0, math.pi - 0.1],
            'supplementary': [math.pi + 0.1, 4.0, 5.0, 2*math.pi - 0.1]
        }

        for branch, angles in test_angles.items():
            cs.branch = branch
            for theta in angles:
                y = cs.forward(theta)
                result = cs.inverse(y)

                # Round-trip must hold
                assert cs.forward(result) == pytest.approx(y, rel=1e-6)

                # Result must be in correct branch region
                if branch == 'principal':
                    assert 0 <= result <= math.pi
                else:
                    assert math.pi <= result <= 2 * math.pi

    def test_approximation_small_angle(self):
        """Small-angle approximation should be close for L >> r"""
        cs = CrankSlider(crank_length=1.0, rod_length=10.0)
        theta = 0.3

        exact = cs.forward(theta)
        approx = cs.approximate(theta)

        assert abs(exact - approx) < 0.01

    def test_period(self):
        """Period should be 2π"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)
        assert cs.period() == 2 * math.pi

    def test_normalize(self):
        """Normalize wraps angle to [0, 2π)"""
        cs = CrankSlider(crank_length=1.0, rod_length=4.0)
        assert cs.normalize(3 * math.pi) == pytest.approx(math.pi)
        assert cs.normalize(-math.pi/2) == pytest.approx(3*math.pi/2)


# ============================================================================
# Test HookesJoint (PeriodicBijective, analytical inverse)
# ============================================================================

class TestHookesJoint:
    def test_creation(self):
        """Valid joint creation"""
        h = HookesJoint(shaft_angle=math.pi/6)  # 30°
        assert h.shaft_angle == math.pi/6
        assert h.is_invertible is True
        assert h.is_analytically_invertible is True
        assert h.branch == 'none'
        assert h.available_branches == ['none']
        assert h.domain.input_unit == Dimension.ANGLE
        assert h.domain.output_unit == Dimension.ANGLE

    def test_shaft_angle_limits(self):
        """Shaft angle must be in [0, π/2)"""
        HookesJoint(shaft_angle=0.0)  # Should work
        HookesJoint(shaft_angle=math.pi/4)  # Should work
        
        with pytest.raises(ValueError, match="must be in"):
            HookesJoint(shaft_angle=math.pi/2)  # At limit - locks
        
        with pytest.raises(ValueError, match="must be in"):
            HookesJoint(shaft_angle=2.0)  # > π/2

    def test_forward_at_key_points(self):
        """Forward: input angle -> output angle"""
        h = HookesJoint(shaft_angle=math.pi/6)  # cos(30°) = 0.8660
        
        # θ_in = 0 -> tan(θ_out) = cos(α)*0 = 0 -> θ_out = 0
        assert h.forward(0.0) == pytest.approx(0.0)
        
        # θ_in = π/4 (45°) -> tan(θ_out) = 0.8660 * 1 = 0.8660 -> θ_out ≈ 40.9°
        expected = math.atan(0.8660254)
        assert h.forward(math.pi/4) == pytest.approx(expected, rel=1e-5)
        
        # θ_in = π/2 (90°) is singularity - skip

    def test_inverse(self):
        """Inverse: output angle -> input angle"""
        h = HookesJoint(shaft_angle=math.pi/6)
        
        # Test roundtrip
        theta_in = 0.7
        theta_out = h.forward(theta_in)
        assert h.inverse(theta_out) == pytest.approx(theta_in, rel=1e-5)

    def test_angular_velocity_ratio(self):
        """Velocity ratio should be >1 at some angles"""
        h = HookesJoint(shaft_angle=math.pi/6)
        
        # At θ=0, ratio = cos(α) / 1² = 0.866
        assert h.angular_velocity_ratio(0.0) == pytest.approx(math.cos(math.pi/6))
        
        # At some angles, ratio > 1 (output speeds up)
        ratio_near_singularity = h.angular_velocity_ratio(math.pi/3)  # 60°
        assert ratio_near_singularity > 1.0

    def test_period(self):
        """Period should be 2π"""
        h = HookesJoint(shaft_angle=math.pi/6)
        assert h.period() == 2 * math.pi

    def test_normalize(self):
        """Normalize wraps angle"""
        h = HookesJoint(shaft_angle=math.pi/6)
        assert h.normalize(3 * math.pi) == pytest.approx(math.pi)


# ============================================================================
# Test Protocol Compliance for Class II
# ============================================================================

def test_periodic_protocol_compliance():
    """All Class II primitives satisfy Periodic protocol"""
    primitives = [
        ScotchYoke(amplitude=2.0),
        EccentricCam(eccentricity=1.0, follower_radius=5.0),
        CrankSlider(crank_length=1.0, rod_length=4.0),
        HookesJoint(shaft_angle=math.pi/6)
    ]
    
    for p in primitives:
        assert hasattr(p, 'forward')
        assert hasattr(p, 'period')
        assert hasattr(p, 'normalize')
        assert hasattr(p, 'is_invertible')
        assert p.is_invertible is True
        assert isinstance(p, Periodic)

def test_periodic_branch_dependent_protocol():
    """ScotchYoke, EccentricCam, CrankSlider satisfy PeriodicBranchDependent"""
    primitives = [
        ScotchYoke(amplitude=2.0),
        EccentricCam(eccentricity=1.0, follower_radius=5.0),
        CrankSlider(crank_length=1.0, rod_length=4.0)
    ]
    
    for p in primitives:
        assert hasattr(p, 'branch')
        assert hasattr(p, 'available_branches')
        assert hasattr(p, 'theta_guess')
        assert hasattr(p, 'is_analytically_invertible')
        assert isinstance(p, PeriodicBranchDependent)

def test_periodic_bijective_protocol():
    """HookesJoint satisfies PeriodicBijective"""
    h = HookesJoint(shaft_angle=math.pi/6)
    
    assert hasattr(h, 'is_analytically_invertible')
    assert h.is_analytically_invertible is True
    assert isinstance(h, PeriodicBijective)
    
    # Should NOT have branch attributes
    assert not hasattr(h, 'available_branches') or h.available_branches == ['none']


# ============================================================================
# Test Domain Validation for Class II
# ============================================================================

def test_class2_domains():
    """All Class II primitives have correct domain bounds"""
    
    # ScotchYoke input domain unbounded, output bounded
    y = ScotchYoke(amplitude=2.5)
    assert y.domain.min is None
    assert y.domain.max is None
    assert y.domain.output_min == -2.5
    assert y.domain.output_max == 2.5
    
    # EccentricCam input domain unbounded, output bounded
    c = EccentricCam(eccentricity=2.0, follower_radius=5.0)
    assert c.domain.min is None
    assert c.domain.max is None
    assert c.domain.output_min == 3.0
    assert c.domain.output_max == 7.0
    
    # CrankSlider input domain unbounded, output bounded
    cs = CrankSlider(crank_length=1.5, rod_length=5.0)
    assert cs.domain.min is None
    assert cs.domain.max is None
    assert cs.domain.output_min == 3.5
    assert cs.domain.output_max == 6.5
    
    # HookesJoint domain is unbounded (angles)
    h = HookesJoint(shaft_angle=math.pi/6)
    assert h.domain.min is None
    assert h.domain.max is None


# ============================================================================
# Run tests if script executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])