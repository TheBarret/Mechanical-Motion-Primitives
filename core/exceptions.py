""" 
    ============================================================================
    CORE: EXCEPTIONS
    ============================================================================
""" 

class MMPError(Exception):
    """Base exception for all MMP errors"""

class CompositionError(MMPError):
    """Raised when two primitives cannot be legally connected"""

class DimensionMismatchError(CompositionError):
    """Output unit of stage N does not match input unit of stage N+1"""

class DomainViolationError(CompositionError):
    """Output range of stage N exceeds input domain of stage N+1"""

class InverseUndefinedError(MMPError):
    """Inverse requested on a non-invertible or lossy chain"""