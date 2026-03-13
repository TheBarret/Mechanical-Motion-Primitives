# Mechanical Motion Primitives
Mechanical implementation to mathematical mappings

<img width="869" height="456" alt="image" src="https://github.com/user-attachments/assets/282fe725-5eef-4064-bd44-afd45296a37a" />

## Testing

<img width="1024" alt="wrist_complex" src="https://github.com/user-attachments/assets/532b17e2-571a-4e52-ad2c-36aaf13f266f" />

```py
import math
import numpy as np
from mmpv3 import *


if __name__ == "__main__":
    wrist_actuator = (ChainBuilder()
        .add(SpurGear, ratio=2.5)                     # Motor gearbox: 2.5x speed reduction
        .add(HookesJoint, shaft_angle=0.3)            # Angled transmission (≈17°)
        .add(RackAndPinion, pitch_radius=0.1)         # Convert rotation to linear motion
        .add_governor(min_val=-2.0, max_val=2.0)      # Safety limits: ±2cm travel
        .build()) 
        
        
    # Forward kinematics: Given motor angle, find actuator position
    motor_angle = 1.5  # radians
    actuator_position = wrist_actuator.forward(motor_angle)
    print(f"-> wrist_actuator.forward({motor_angle})")
    print(f"    Motor at {motor_angle}rad → Actuator at {actuator_position:.3f}m")

    # Inverse kinematics: Desired position → required motor angle
    desired_position = 1.2  # meters
    print(f"-> wrist_actuator.inverse({desired_position})")
    try:
        motor_angle_needed = wrist_actuator.inverse(desired_position)
        print(f"    Need {desired_position}m → Motor at {motor_angle_needed:.3f}rad")
    except InverseUndefinedError:
        print("     [InverseUndefinedError]: governor makes chain lossy")

    # How fast does actuator move given motor speed?
    motor_speed = 10.0  # rad/s
    actuator_speed = wrist_actuator.derivative(motor_angle) * motor_speed
    print(f"At {motor_speed} rad/s motor speed → Actuator moves at {actuator_speed:.3f} m/s")

    # check operating limits
    test_inputs = [0.5, 1.0, 1.5, 2.0, 2.5]
    for theta in test_inputs:
        if wrist_actuator.input_domain.contains(theta):
            print(f"-> wrist_actuator.forward({theta})")
            pos = wrist_actuator.forward(theta)
            print(f"  θ={theta:.1f}rad → valid, position={pos:.3f}m")
        else:
            print(f"  θ={theta:.1f}rad → WARNING: Outside recommended input range")
```
