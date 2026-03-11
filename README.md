# Mechanical-Motion-Primitives
Mechanical implementation to mathematical mappings


Almost complete, needs some work.
```
mm-test.py::test_spur_gear_basic PASSED                                                                                                [  2%]
mm-test.py::test_spur_gear_modular PASSED                                                                                              [  5%]
mm-test.py::test_rack_and_pinion PASSED                                                                                                [  8%]
mm-test.py::test_wedge PASSED                                                                                                          [ 11%]
mm-test.py::test_pantograph PASSED                                                                                                     [ 14%]
mm-test.py::test_scotch_yoke PASSED                                                                                                    [ 17%]
mm-test.py::test_crank_slider PASSED                                                                                                   [ 20%]
mm-test.py::test_hookes_joint PASSED                                                                                                   [ 23%]
mm-test.py::test_planetary_gear PASSED                                                                                                 [ 26%]
mm-test.py::test_differential PASSED                                                                                                   [ 29%]
mm-test.py::test_ratchet PASSED                                                                                                        [ 32%]
mm-test.py::test_detent PASSED                                                                                                         [ 35%]
mm-test.py::test_escapement PASSED                                                                                                     [ 38%]
mm-test.py::test_geneva_basic FAILED                                                                                                   [ 41%]
mm-test.py::test_worm_gear_irreversible PASSED                                                                                         [ 44%]
mm-test.py::test_worm_gear_reversible PASSED                                                                                           [ 47%]
mm-test.py::test_harmonic_drive PASSED                                                                                                 [ 50%]
mm-test.py::test_differential_screw PASSED                                                                                             [ 52%]
mm-test.py::test_cvt PASSED                                                                                                            [ 55%]
mm-test.py::test_trochoidal_position PASSED                                                                                            [ 58%]
mm-test.py::test_spirograph_closure PASSED                                                                                             [ 61%]
mm-test.py::test_vernier PASSED                                                                                                        [ 64%]
mm-test.py::test_mechanical_chain FAILED                                                                                               [ 67%]
mm-test.py::test_chain_with_oneway PASSED                                                                                              [ 70%]
mm-test.py::test_scotch_yoke_out_of_range PASSED                                                                                       [ 73%]
mm-test.py::test_planetary_invalid_mode PASSED                                                                                         [ 76%]
mm-test.py::test_toroidal_cvt_near_singularity PASSED                                                                                  [ 79%]
mm-test.py::test_compound_gear_train FAILED                                                                                            [ 82%]
mm-test.py::test_four_bar_linkage FAILED                                                                                               [ 85%]
mm-test.py::test_eccentric_cam_numerical FAILED                                                                                        [ 88%]
mm-test.py::test_toroidal_cvt_basic FAILED                                                                                             [ 91%]
mm-test.py::test_escapement_force_threshold PASSED                                                                                     [ 94%]
mm-test.py::test_mechanical_chain_1d PASSED                                                                                            [ 97%]
mm-test.py::test_mechanical_chain_2d FAILED                                                                                            [100%]
```

```py
    def test_geneva_basic():
        geneva = GenevaDrive(num_slots=4)
        # First engagement might behave differently
>       out1 = geneva.forward(0.1)

mm-test.py:145:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
mmp.py:568: in forward
    self.output_angle = self._engagement_profile(theta, center)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = GenevaDrive(num_slots=4, current_slot=0, output_angle=0.0, slot_positions=[0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469], engagement_window=0.7853981633974483)
theta = 0.1, entry_angle = 0.0

    def _engagement_profile(self, theta: float, entry_angle: float) -> float:
        """
        Proper Geneva kinematics:
        tan(ψ) = sin(θ) / (λ - cos(θ)) where λ = center_distance/crank_radius
        """
        # θ is input angle relative to engagement start
>       lambda_ratio = self.center_distance / self.crank_radius
E       AttributeError: 'GenevaDrive' object has no attribute 'center_distance'

mmp.py:551: AttributeError
___________________________________________________________ test_mechanical_chain ___________________________________________________________

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
>       y = chain.forward(x)

mm-test.py:268:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
mmp.py:849: in forward
    x = mech.forward(x)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = Pantograph(scale=1.5), point = 20.0

    def forward(self, point: Tuple[float, float]) -> Tuple[float, float]:
>       x, y = point
E       TypeError: cannot unpack non-iterable float object

mmp.py:162: TypeError
_________________________________________________________ test_compound_gear_train __________________________________________________________

    def test_compound_gear_train():
        """Multiple gears: 2:1 * 3:1 = 6:1 total"""
>       train = CompoundGearTrain(ratios=[2.0, 3.0])
E       NameError: name 'CompoundGearTrain' is not defined

mm-test.py:314: NameError
___________________________________________________________ test_four_bar_linkage ___________________________________________________________

    def test_four_bar_linkage():
        """Freudenstein equation should equal 0 for valid configuration"""
>       linkage = FourBarLinkage(a=4, b=5, c=3, d=6)
E       NameError: name 'FourBarLinkage' is not defined

mm-test.py:321: NameError
_______________________________________________________ test_eccentric_cam_numerical ________________________________________________________

    def test_eccentric_cam_numerical():
        """Cam inverse should converge numerically"""
>       cam = EccentricCam(eccentricity=0.5, follower_radius=2.0)
E       NameError: name 'EccentricCam' is not defined

mm-test.py:329: NameError
__________________________________________________________ test_toroidal_cvt_basic __________________________________________________________

    def test_toroidal_cvt_basic():
        """Toroidal CVT ratio = tan(α)"""
        cvt = ToroidalCVT()
        cvt.set_tilt(math.pi/4)  # 45°
        assert abs(cvt.ratio - 1.0) < 1e-10  # tan(45°) = 1
>       assert cvt.forward(100) == 100
E       assert 99.99999999999999 == 100
E        +  where 99.99999999999999 = forward(100)
E        +    where forward = ToroidalCVT().forward

mm-test.py:339: AssertionError
_________________________________________________________ test_mechanical_chain_2d __________________________________________________________

    def test_mechanical_chain_2d():
        """Chain of 2D invertible mechanisms should be invertible"""
        chain = MechanicalChain([
            Pantograph(scale=2.0),
>           OldhamCoupling(delta_x=1.0, delta_y=1.0),
        ])
E       NameError: name 'OldhamCoupling' is not defined

mm-test.py:367: NameError
========================================================== short test summary info ==========================================================
FAILED mm-test.py::test_geneva_basic - AttributeError: 'GenevaDrive' object has no attribute 'center_distance'
FAILED mm-test.py::test_mechanical_chain - TypeError: cannot unpack non-iterable float object
FAILED mm-test.py::test_compound_gear_train - NameError: name 'CompoundGearTrain' is not defined
FAILED mm-test.py::test_four_bar_linkage - NameError: name 'FourBarLinkage' is not defined
FAILED mm-test.py::test_eccentric_cam_numerical - NameError: name 'EccentricCam' is not defined
FAILED mm-test.py::test_toroidal_cvt_basic - assert 99.99999999999999 == 100
FAILED mm-test.py::test_mechanical_chain_2d - NameError: name 'OldhamCoupling' is not defined
======================================================= 7 failed, 27 passed in 0.79s ========================================================
```
