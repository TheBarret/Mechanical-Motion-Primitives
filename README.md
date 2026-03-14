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

## Chain Composition

There are two types of chain builders, both reflect real vs unreal.
`ICBuilder` and `CCBuilder` both immutable and only once constructed, validation happens at build time. 

```py
import math
import numpy as np
from mmp import *

if __name__ == "__main__":
    wrist = (ICBuilder()
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

### CCBuilder vs ICBuilder Behavior

The `ICBuilder` chain enforces unit compatibility at every junction. 

- Connecting a SpurGear `(ANGLE output)` 
- Directly to a Wedge `(LENGTH input)` 
 
Raises `DimensionMismatchError` at build time, not at runtime. 

The `CCBuilder` can be used the same way as `ICBuilder` but supports `adapters` chains.

Chain adapters in `Class III` are 'Admissible' and 'Inadmissible' types, 
these are "fictional" mathematical transformations that don't correspond to a single physical mechanism, 
but allow you to compose arbitrary chains for (mathematical) exploration, carry the same unit semantics as real primitives, 
enabling the compositor to validate dimensional flow even in fictional constructions. 

Types:
- `UnitAdapter()`: Abstract base for all dimensional adapters
- `AngleToLength(UnitAdapter)`: Conversion from angle to length.
- `LengthToAngle(UnitAdapter)`: Conversion from length to angle.
- `UnitlessScaling(UnitAdapter)`: Scaling that preserves units.
- `Bias(UnitAdapter)`: Add a constant offset while preserving units.
- `FunctionAdapter(UnitAdapter)`: Arbitrary mathematical function with known inverse.

Example:

```py
experiment = (CCBuilder()
    .add(SpurGear, ratio=2.5)                          # Current unit: ANGLE
    .add(RackAndPinion, pitch_radius=0.1)              # Current unit: LENGTH
    .add_adapter(to_unit=Dimension.ANGLE, scale=10.0)  # LENGTH → ANGLE  (Inadmissible)
    .add(HookesJoint, shaft_angle=0.3)                 # Current unit: ANGLE again
    .build())
```

Output:
```
* Testing illegal actuator *
-> chain.forward(1.5)
    -> Motor at 1.5rad → Actuator at 3.729m
-> chain.inverse(1.2)
    Need 1.2m → Motor at 0.486rad
At 10.0 rad/s motor speed → Actuator moves at 24.585 m/s
-> chain.forward(0.5)
  θ=0.5rad → valid, position=1.236m
-> chain.forward(1.0)
  θ=1.0rad → valid, position=2.522m
-> chain.forward(1.5)
  θ=1.5rad → valid, position=3.729m
-> chain.forward(2.0)
  θ=2.0rad → valid, position=5.013m
-> chain.forward(2.5)
  θ=2.5rad → valid, position=6.251m
```


## EOF
