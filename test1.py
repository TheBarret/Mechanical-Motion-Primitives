"""
Test suite for Mechanical Motion Primitives Class I

"""

import pytest
import math
from mmp import *

# ============================================================================
# Test Domain and Dimension
# ============================================================================

def test_domain_contains():
    """Domain bounds checking with tolerance"""
    d = Domain(min=-10.0, max=10.0)
    
    assert d.contains(0.0) is True
    assert d.contains(10.0) is True
    assert d.contains(-10.0) is True
    assert d.contains(10.0 + 1e-10) is True  # Within tolerance
    assert d.contains(10.1) is False
    assert d.contains(-10.1) is False
    
    # Unbounded domains
    d_unbounded = Domain()
    assert d_unbounded.contains(1e6) is True
    assert d_unbounded.contains(-1e6) is True

def test_domain_is_finite():
    """Finite domain detection"""
    assert Domain(min=-5, max=5).is_finite() is True
    assert Domain(min=-5, max=None).is_finite() is False
    assert Domain(min=None, max=5).is_finite() is False
    assert Domain().is_finite() is False

def test_dimension_enum():
    """Dimension enum has expected members"""
    assert Dimension.ANGLE.value == 1
    assert Dimension.LENGTH.value == 2
    assert Dimension.RATIO.value == 3


# ============================================================================
# Test SpurGear
# ============================================================================

class TestSpurGear:
    def test_creation(self):
        """Valid gear creation"""
        g = SpurGear(ratio=2.5)
        assert g.ratio == 2.5
        assert g.is_invertible is True
        assert g.domain.input_unit == Dimension.ANGLE
        assert g.domain.output_unit == Dimension.ANGLE
    
    def test_zero_ratio_raises(self):
        """Gear ratio cannot be zero"""
        with pytest.raises(ValueError, match="cannot be zero"):
            SpurGear(ratio=0)
    
    def test_forward(self):
        """Forward transformation"""
        g = SpurGear(ratio=3.0)
        assert g.forward(2.0) == 6.0
        assert g.forward(-2.0) == -6.0
    
    def test_inverse(self):
        """Inverse transformation"""
        g = SpurGear(ratio=3.0)
        assert g.inverse(6.0) == 2.0
        assert g.inverse(-6.0) == -2.0
    
    def test_roundtrip(self):
        """forward then inverse returns original"""
        g = SpurGear(ratio=2.5)
        x = 4.0
        assert g.inverse(g.forward(x)) == pytest.approx(x)


# ============================================================================
# Test CompoundGearTrain
# ============================================================================

class TestCompoundGearTrain:
    def test_creation(self):
        """Valid train creation"""
        c = CompoundGearTrain([2.0, 3.0, 0.5])
        assert c.ratios == [2.0, 3.0, 0.5]
        assert c.is_invertible is True
    
    def test_empty_ratios_raises(self):
        """Must have at least one ratio"""
        with pytest.raises(ValueError, match="at least one ratio"):
            CompoundGearTrain([])
    
    def test_zero_ratio_raises(self):
        """No zero ratios allowed"""
        with pytest.raises(ValueError, match="must be non-zero"):
            CompoundGearTrain([2.0, 0, 3.0])
    
    def test_forward(self):
        """Forward cascades correctly"""
        c = CompoundGearTrain([2.0, 3.0, 0.5])  # net = 3.0
        assert c.forward(5.0) == 15.0
    
    def test_inverse(self):
        """Inverse reverses correctly"""
        c = CompoundGearTrain([2.0, 3.0, 0.5])  # net = 3.0
        assert c.inverse(15.0) == 5.0
    
    def test_roundtrip(self):
        """forward then inverse returns original"""
        c = CompoundGearTrain([2.0, 3.0, 0.5])
        x = 7.0
        assert c.inverse(c.forward(x)) == pytest.approx(x)


# ============================================================================
# Test RackAndPinion
# ============================================================================

class TestRackAndPinion:
    def test_creation(self):
        """Valid creation"""
        r = RackAndPinion(pitch_radius=0.1)
        assert r.pitch_radius == 0.1
        assert r.domain.input_unit == Dimension.ANGLE
        assert r.domain.output_unit == Dimension.LENGTH
    
    def test_non_positive_radius_raises(self):
        """Radius must be positive"""
        with pytest.raises(ValueError, match="positive"):
            RackAndPinion(pitch_radius=0)
        with pytest.raises(ValueError, match="positive"):
            RackAndPinion(pitch_radius=-0.1)
    
    def test_rotation_to_linear(self):
        """Forward converts rotation to displacement"""
        r = RackAndPinion(pitch_radius=0.1)
        assert r.forward(math.pi) == pytest.approx(0.1 * math.pi)
    
    def test_linear_to_rotation(self):
        """Inverse converts displacement to rotation"""
        r = RackAndPinion(pitch_radius=0.1)
        assert r.inverse(0.5) == pytest.approx(5.0)  # 0.5 / 0.1 = 5.0 rad
    
    def test_roundtrip(self):
        """forward then inverse returns original"""
        r = RackAndPinion(pitch_radius=0.1)
        theta = 2.5
        assert r.inverse(r.forward(theta)) == pytest.approx(theta)


# ============================================================================
# Test Wedge
# ============================================================================

class TestWedge:
    def test_creation(self):
        """Valid wedge angles"""
        w = Wedge(angle_rad=math.pi/4)  # 45°
        assert w.angle_rad == math.pi/4
        assert w.domain.input_unit == Dimension.LENGTH
        assert w.domain.output_unit == Dimension.LENGTH
    
    def test_invalid_angles_raises(self):
        """Angles must be strictly between 0 and π/2"""
        with pytest.raises(ValueError):
            Wedge(angle_rad=0)
        with pytest.raises(ValueError):
            Wedge(angle_rad=math.pi/2)
        with pytest.raises(ValueError):
            Wedge(angle_rad=-0.1)
        with pytest.raises(ValueError):
            Wedge(angle_rad=2.0)  # > π/2
    
    def test_mechanical_advantage(self):
        """MA = cot(angle)"""
        w = Wedge(angle_rad=math.pi/4)  # 45°, MA = 1
        assert w.mechanical_advantage == pytest.approx(1.0)
        
        w2 = Wedge(angle_rad=math.pi/6)  # 30°, MA = cot(30°) = √3 ≈ 1.732
        assert w2.mechanical_advantage == pytest.approx(math.sqrt(3))
    
    def test_forward(self):
        """Horizontal to vertical displacement"""
        w = Wedge(angle_rad=math.pi/6)  # 30°, tan=1/√3 ≈ 0.577
        assert w.forward(10.0) == pytest.approx(10.0 * math.tan(math.pi/6))
    
    def test_inverse(self):
        """Vertical to horizontal displacement"""
        w = Wedge(angle_rad=math.pi/6)
        assert w.inverse(5.0) == pytest.approx(5.0 / math.tan(math.pi/6))
    
    def test_roundtrip(self):
        """forward then inverse returns original"""
        w = Wedge(angle_rad=math.pi/6)
        x = 7.0
        assert w.inverse(w.forward(x)) == pytest.approx(x)


# ============================================================================
# Test OldhamCoupling
# ============================================================================

class TestOldhamCoupling:
    def test_creation(self):
        """Valid coupling creation"""
        o = OldhamCoupling(offset_x=0.1, offset_y=0.05)
        assert o.offset_x == 0.1
        assert o.offset_y == 0.05
        assert o.domain.input_unit == Dimension.ANGLE
        assert o.domain.output_unit == Dimension.ANGLE
    
    def test_forward_identity(self):
        """Forward preserves angle"""
        o = OldhamCoupling(0.1, 0.05)
        assert o.forward(2.5) == 2.5
        assert o.forward(-1.3) == -1.3
    
    def test_inverse_identity(self):
        """Inverse preserves angle"""
        o = OldhamCoupling(0.1, 0.05)
        assert o.inverse(2.5) == 2.5
    
    def test_shaft_offset(self):
        """Shaft offset stored as metadata"""
        o = OldhamCoupling(0.1, 0.05)
        assert o.shaft_offset == (0.1, 0.05)
    
    def test_roundtrip(self):
        """forward then inverse returns original"""
        o = OldhamCoupling(0.1, 0.05)
        theta = 3.7
        assert o.inverse(o.forward(theta)) == theta


# ============================================================================
# Test Governor (UPDATED: Now properly OneWay)
# ============================================================================

class TestGovernor:
    def test_creation(self):
        """Valid governor creation"""
        g = SpurGear(2.0)
        gov = Governor(g, min_val=-5.0, max_val=5.0)
        assert gov.min_val == -5.0
        assert gov.max_val == 5.0
        assert gov.is_invertible is False
    
    def test_invalid_bounds_raises(self):
        """min must be less than max"""
        g = SpurGear(2.0)
        with pytest.raises(ValueError, match="less than"):
            Governor(g, min_val=5.0, max_val=5.0)
        with pytest.raises(ValueError, match="less than"):
            Governor(g, min_val=5.0, max_val=4.0)
    
    def test_negative_hysteresis_raises(self):
        """Hysteresis must be non-negative"""
        g = SpurGear(2.0)
        with pytest.raises(ValueError, match="non-negative"):
            Governor(g, -5.0, 5.0, hysteresis=-0.1)
    
    def test_hysteresis_too_large_raises(self):
        """Hysteresis cannot exceed half the range"""
        g = SpurGear(2.0)
        with pytest.raises(ValueError, match="too large"):
            Governor(g, 0.0, 10.0, hysteresis=6.0)  # > 5.0
    
    def test_forward_clamps_output(self):
        """Governor clamps to [min, max]"""
        g = SpurGear(3.0)  # forward multiplies by 3
        gov = Governor(g, min_val=-5.0, max_val=5.0)
        
        # Within bounds
        assert gov.forward(1.0) == 3.0  # 3*1 = 3 (within ±5)
        
        # Above max
        assert gov.forward(2.0) == 5.0  # 3*2 = 6 → clamped to 5
        
        # Below min
        assert gov.forward(-2.0) == -5.0  # 3*(-2) = -6 → clamped to -5
    
    def test_domain_reflects_clamp(self):
        """Governor domain matches clamp range"""
        g = SpurGear(3.0)
        gov = Governor(g, min_val=-5.0, max_val=5.0)
        
        assert gov.domain.min == -5.0
        assert gov.domain.max == 5.0
        assert gov.domain.input_unit == Dimension.ANGLE
        assert gov.domain.output_unit == Dimension.ANGLE
    
    def test_no_inverse_method(self):
        """Governor should NOT have an inverse method - it's OneWay"""
        g = SpurGear(2.0)
        gov = Governor(g, -5.0, 5.0)
        
        # Should not have inverse attribute
        assert not hasattr(gov, 'inverse')
        
        # Should satisfy OneWay protocol
        assert isinstance(gov, OneWay)
        
        # Should NOT satisfy Invertible protocol
        assert not isinstance(gov, Invertible)


# ============================================================================
# Test Protocol Compliance
# ============================================================================

def test_invertible_protocol_compliance():
    """All Class I primitives satisfy Invertible protocol"""
    from typing import runtime_checkable, cast
    
    primitives = [
        SpurGear(2.0),
        CompoundGearTrain([2.0, 3.0]),
        RackAndPinion(0.1),
        Wedge(math.pi/4),
        OldhamCoupling(0.1, 0.05)
    ]
    
    for p in primitives:
        # Duck typing check
        assert hasattr(p, 'forward')
        assert hasattr(p, 'inverse')
        assert hasattr(p, 'is_invertible')
        assert p.is_invertible is True
        
        # Runtime protocol check
        assert isinstance(p, Invertible)


def test_oneway_protocol_compliance():
    """Governor satisfies OneWay protocol"""
    g = SpurGear(2.0)
    gov = Governor(g, -5.0, 5.0)
    
    assert hasattr(gov, 'forward')
    assert hasattr(gov, 'is_invertible')
    assert gov.is_invertible is False
    
    # Should satisfy OneWay
    assert isinstance(gov, OneWay)
    
    # Should NOT be Invertible (missing inverse)
    assert not isinstance(gov, Invertible)


def test_periodic_protocols_placeholder():
    """Placeholder for future periodic mechanism tests"""
    # Once you implement periodic mechanisms (cam, etc.)
    pass


# ============================================================================
# Test Composition Patterns (Conceptual - would need Compositor class)
# ============================================================================

def test_unit_compatibility_concept():
    """
    Conceptual test for unit compatibility checking.
    This would be implemented in a Compositor class.
    """
    rack = RackAndPinion(0.1)  # ANGLE -> LENGTH
    wedge = Wedge(math.pi/4)    # LENGTH -> LENGTH
    
    # These should be compatible: rack's output (LENGTH) matches wedge's input (LENGTH)
    assert rack.domain.output_unit == wedge.domain.input_unit
    
    # This would fail: gear expects ANGLE, rack outputs LENGTH
    gear = SpurGear(2.0)  # ANGLE -> ANGLE
    assert gear.domain.input_unit == Dimension.ANGLE
    assert rack.domain.output_unit == Dimension.LENGTH
    # gear.forward(rack.forward(1.0)) would be dimensionally wrong


def test_invertible_chain_concept():
    """Chain of invertible primitives remains invertible"""
    # This would be implemented in a Compositor
    chain = [SpurGear(2.0), RackAndPinion(0.1), Wedge(math.pi/4)]
    
    # All are invertible
    assert all(p.is_invertible for p in chain)
    
    # Composite would also be invertible


def test_oneway_chain_concept():
    """Once a Governor is inserted, chain becomes OneWay"""
    chain = [
        SpurGear(2.0),
        Governor(SpurGear(1.0), -5.0, 5.0),  # Lossy stage
        RackAndPinion(0.1)
    ]
    
    # First stage invertible, second is OneWay
    assert chain[0].is_invertible is True
    assert chain[1].is_invertible is False
    assert not hasattr(chain[1], 'inverse')
    
    # Composite would be OneWay, not Invertible


# ============================================================================
# Test Encoding vs Hashing Conceptual Distinction
# ============================================================================

def test_encoding_vs_hashing_distinction():
    """
    Demonstrates the architectural separation:
    - Invertible chain → encoding/decoding
    - OneWay chain → hashing/signatures
    """
    # Encoding pipeline (fully reversible)
    encoder = [
        SpurGear(2.0),      # Can decode
        RackAndPinion(0.1), # Can decode
        Wedge(math.pi/4)    # Can decode
    ]
    assert all(isinstance(p, Invertible) for p in encoder)
    
    # Hashing pipeline (lossy)
    hasher = [
        SpurGear(2.0),                      # Still invertible here
        Governor(SpurGear(1.0), 0.0, 100.0), # Information lost!
        RackAndPinion(0.1)                   # Can't recover pre-governor values
    ]
    assert isinstance(hasher[0], Invertible)
    assert isinstance(hasher[1], OneWay)  # Governor is OneWay
    assert not hasattr(hasher[1], 'inverse')


# ============================================================================
# Run tests if script executed directly
# ============================================================================

if __name__ == "__main__":

    pytest.main([__file__, "-v"])
