# Mechanical Motion Primitives
Mechanical implementation to mathematical mappings

<img width="869" height="456" alt="image" src="https://github.com/user-attachments/assets/282fe725-5eef-4064-bd44-afd45296a37a" />

## Example

```py
import math
import numpy as np
from mmpv3 import *

if __name__ == "__main__":
    wrist = (ChainBuilder()
        .add(SpurGear, ratio=2.5)
        .add(HookesJoint, shaft_angle=0.05)
        .add(RackAndPinion, pitch_radius=0.1)
        .build())
```

## Testing
<img width="1024" alt="wrist_complex" src="https://github.com/user-attachments/assets/75938112-c96e-4fde-8d9e-789148d14a69" />

