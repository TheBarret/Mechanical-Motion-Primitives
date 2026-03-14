import math

"""
============================================================================
CORE: MECHANICAL LIMITS
Compositor-level sanity thresholds — warning triggers, not hard walls
Nothing physical exceeds these — if it does, it is a bug in the caller
============================================================================
"""

class MechanicalLimits:
    MAX_ANGLE        = 4 * math.pi   # two full rotations
    MAX_RATIO        = 1000.0        # no real gearbox exceeds 1000:1
    MAX_DISPLACEMENT = 1e6           # 1km — mechanically absurd
    MAX_VELOCITY     = 1e4           # 10,000 rad/s — jet turbine territory
    TOLERANCE        = 1e-9          # float drift forgiveness
 