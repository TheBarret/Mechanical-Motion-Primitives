# Mechanical Motion Primitives
Mechanical implementation to mathematical mappings

A little project for modeling mechanical transmission chains as composable mathematical mappings.  
Each mechanism is expressed as a typed, domain-aware function that can be chained, inverted,  
and differentiated treating mechanical motion the same way a compiler treats types.  

# Primitive Classes

The (MMP) models each stage as a primitive: a forward() mapping with a declared physical domain, 
a unit type, and (where mechanically valid) an inverse() and derivative(). Primitives compose into chains. 
The chain validates itself at construction you cannot connect an angular output to a linear input without an explicit adapter, 
and a Governor (physical clamp) permanently marks the chain as non-invertible. 
 
- Class I - Linear Scaling (Affine Maps)  
These are continuous, invertible, constant ratio, the workhorses of rotational and linear transmissions.  
All Class I primitives are monotonic and fully invertible.  

- Class II - Periodic Non-Linear (Trigonometric)  
These are oscillatory, bounded output, non-injective over the full domain, subdivided by inversion behavior. 
The `Periodic Bijective` are invertible within one period, no branch selection needed, 
as where the `Periodic Branch Dependent` is non-injective, inverse requires branch selection required. 

<img width="869" height="456" alt="image" src="https://github.com/user-attachments/assets/282fe725-5eef-4064-bd44-afd45296a37a" />  

### Governor 
A Governor wraps any primitive and clamps its output inserting one permanently sets `is_invertible = False` on the entire chain.

## Domain and Units
Every primitive declares a `Domain` the physical envelope it can accept and produce. 

```py
@dataclass(frozen=True)
class Domain:
    min: Optional[float]      # None = unbounded
    max: Optional[float]
    input_unit:  Dimension    # ANGLE | LENGTH | RATIO | VELOCITY | GENERIC
    output_unit: Dimension
```

The `CompositePrimitive` back-propagates output constraints to compute `input_domain`, 
the range of motor inputs that keeps every intermediate stage within its physical bounds. 
For monotonic invertible prefix chains this is exact; for non-monotonic stages it falls back conservatively. 

## Exceptions

- `DimensionMismatchError` 
   Connecting incompatible unit types at build time 
- `DomainViolationError` 
   Output range of stage N exceeds input domain of stage N+1 
- `InverseUndefinedError` 
   Calling inverse() on a non-invertible or Governor-containing chain 

## Composition & Example

Chains are built with ChainBuilder and are immutable once constructed. Validation happens at build time. 

```py
import math
import numpy as np
from mmp import *

if __name__ == "__main__":
    wrist = (ChainBuilder()
        .add(SpurGear, ratio=2.5)
        .add(HookesJoint, shaft_angle=0.05)
        .add(RackAndPinion, pitch_radius=0.1)
        .build())
    
    # Forward kinematics
    position = wrist_actuator.forward(1.5)      # motor angle → actuator position
    
    # Inverse kinematics (raises InverseUndefinedError if Governor was present)
    motor_angle = wrist_actuator.inverse(1.2)
    
    # Instantaneous transmission ratio via chain rule
    ratio = wrist_actuator.derivative(1.5)
    
    # Check operating range
    wrist_actuator.input_domain.contains(theta) # back-propagated from Governor bounds
```

### Behavior
The chain enforces unit compatibility at every junction, connecting a SpurGear (ANGLE output) 
directly to a Wedge (LENGTH input) raises `DimensionMismatchError` at build time, not at runtime.

## (Stress) Testing
<img width="1024" alt="wrist_complex" src="https://github.com/user-attachments/assets/75938112-c96e-4fde-8d9e-789148d14a69" />

```
Stress Tester
============================================================
Testing...

Gen 0:
  Faults found: 10
  Coverage: 8/8 primitives
  Missing: set()
  Latest fault: DimensionMismatchError at stage 1

Gen 100:
  Faults found: 12
  Coverage: 8/8 primitives
  Missing: set()
  Latest fault: ValueError at stage 0

Gen 200:
  Faults found: 12
  Coverage: 8/8 primitives
  Missing: set()
  Latest fault: ValueError at stage 0

Gen 300:
  Faults found: 12
  Coverage: 8/8 primitives
  Missing: set()
  Latest fault: ValueError at stage 0

Gen 400:
  Faults found: 12
  Coverage: 8/8 primitives
  Missing: set()
  Latest fault: ValueError at stage 0

==================================================
FAULT HUNT COMPLETE
==================================================

Coverage:
  Primitives: 8/8
  Unique pairs: 19
  Dimension paths: 0

Faults Found: 12

Fault Types:
  DimensionMismatchError: 8
  ValueError: 4

Faults:

1. DimensionMismatchError
   Cannot add HookesJoint:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 1 (HookesJoint)
   Chain: Wedge → HookesJoint → SpurGear

2. DimensionMismatchError
   Cannot add RackAndPinion:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 1 (RackAndPinion)
   Chain: CrankSlider → RackAndPinion

3. DimensionMismatchError
   Cannot add CrankSlider:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 2 (CrankSlider)
   Chain: SpurGear → ScotchYoke → CrankSlider → HookesJoint

4. ValueError
   Follower radius must exceed eccentricity —  otherwise cam exceeds follower range
   Stage: -1 (unknown)
   Chain: EccentricCam → HookesJoint → OldhamCoupling → Wedge

5. DimensionMismatchError
   Cannot add EccentricCam:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 1 (EccentricCam)
   Chain: CrankSlider → EccentricCam → HookesJoint → RackAndPinion

6. DimensionMismatchError
   Cannot add SpurGear:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 1 (SpurGear)
   Chain: ScotchYoke → SpurGear → SpurGear → CrankSlider

7. DimensionMismatchError
   Cannot add OldhamCoupling:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 1 (OldhamCoupling)
   Chain: Wedge → OldhamCoupling → SpurGear → SpurGear

8. DimensionMismatchError
   Cannot add ScotchYoke:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH
   Stage: 1 (ScotchYoke)
   Chain: RackAndPinion → ScotchYoke → SpurGear → HookesJoint → Wedge

9. ValueError
   Shaft angle must be in [0, π/2) —  at π/2 the joint locks (cos=0)
   Stage: -1 (unknown)
   Chain: ScotchYoke → HookesJoint → Wedge → Wedge

10. ValueError
   Wedge angle must be in (0, π/2) exclusive —  at 0 no lift occurs, at π/2 tan is undefined
   Stage: 0 (Wedge)
   Chain: Wedge → HookesJoint

============================================================
Running hypothesis tests...
Stress Test
============================================================

+ HookesJoint near lock-up (α → π/2)

+ HookesJoint continuity across π boundary

+ Wedge near vertical (angle → π/2)

+ Wedge near flat (angle → 0)

+ CrankSlider near kinematic limit (L ≈ r)

+ CrankSlider L >> r regime

+ EccentricCam near follower limit

+ ScotchYoke amplitude boundary

! Non-monotonic primitive in middle of chain
   ! build: DimensionMismatchError: Cannot add SpurGear:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH

+ Unbounded input propagates conservatively

+ Domain intersection empties chain
   Round-trip error at x=0.1: |-0.005752220392306202 - 0.1| = 0.10575222039230621
   Round-trip error at x=1.0: |-0.005309649148733836 - 1.0| = 1.0053096491487339

+ Mixed bounded/unbounded chain
   Round-trip error at x=-10: |-0.5752220392306203 - -10| = 9.42477796076938
   Round-trip error at x=-1: |-0.5707963267948967 - -1| = 0.42920367320510333

+ Forward → Inverse round-trip

+ ScotchYoke branch-aware inverse
   Round-trip error at x=1.5: |1.041592653589793 - 1.5| = 0.458407346410207

+ HookesJoint round-trip with continuity

+ CompoundGearTrain round-trip

+ OldhamCoupling identity round-trip

+ Analytical vs numerical derivative: HookesJoint

+ Derivative chain rule: multi-stage

+ Governor derivative at clamp boundary

+ ScotchYoke derivative zero-crossing
   Round-trip error at x=3.141592653589793: |1.2246467991473532e-16 - 3.141592653589793| = 3.141592653589793

! ANGLE → LENGTH → ANGLE roundtrip
   ! build: DimensionMismatchError: Cannot add RackAndPinion:  expects Dimension.ANGLE but  previous stage outputs Dimension.LENGTH

+ LENGTH → LENGTH chain: Wedge → Wedge

+ GENERIC unit doesn't leak incorrectly

! Unit mismatch caught at build time
   ! build: DimensionMismatchError: Cannot add Wedge:  expects Dimension.LENGTH but  previous stage outputs Dimension.ANGLE

+ Governor makes chain non-invertible

+ Governor hysteresis doesn't affect forward()

+ Governor at chain start
   Derivative mismatch at x=0: analytical=0.000000, numerical=0.500000

+ Governor with tight bounds on high-ratio chain

+ Mixed-period chain reports aperiodic

+ Same-period chain preserves period
   Round-trip error at x=2: |1.141592653589793 - 2| = 0.8584073464102071

+ Normalize on periodic primitive
   Round-trip error at x=-10: |0.5752220392306202 - -10| = 10.57522203923062
   Round-trip error at x=-6.283185307179586: |2.4492935982947064e-16 - -6.283185307179586| = 6.283185307179586

+ Extreme ratio cascade overflow check

+ Tiny amplitude ScotchYoke precision

+ Large input to bounded primitive
   Round-trip error at x=-1000000.0: |0.357564167085735 - -1000000.0| = 1000000.3575641671
   Round-trip error at x=-1000.0: |-0.97353615844575 - -1000.0| = 999.0264638415542

+ Zero input to all primitives

+ Non-invertible primitive in chain

+ Branch selection propagation in inverse

+ Guess propagation in Newton-Raphson inverse

! Empty chain handling
   ! build: ValueError: Cannot build empty chain

+ Single primitive chain

+ Long chain performance

+ Domain propagation through long chain
   Round-trip error at x=-10: |-1.3810901136082492 - -10| = 8.618909886391751

============================================================
Results: 39/43 hypotheses passed
```


