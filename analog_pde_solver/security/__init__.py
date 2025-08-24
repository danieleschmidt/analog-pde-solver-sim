"""Security monitoring and protection for analog PDE solver."""

# Import available modules with graceful fallback
try:
    from .threat_detector import ThreatDetector, SecurityAudit
    _HAS_THREAT_DETECTOR = True
except ImportError:
    _HAS_THREAT_DETECTOR = False
    ThreatDetector = None
    SecurityAudit = None

try:
    from .input_sanitizer import InputSanitizer, SecurityValidation
    _HAS_INPUT_SANITIZER = True
except ImportError:
    _HAS_INPUT_SANITIZER = False
    InputSanitizer = None
    SecurityValidation = None

try:
    from .input_validation import InputValidator, SecurityValidator, validate_all_inputs
    _HAS_INPUT_VALIDATION = True
except ImportError:
    _HAS_INPUT_VALIDATION = False
    InputValidator = None
    SecurityValidator = None
    validate_all_inputs = None

__all__ = []

# Add available components to __all__
if _HAS_THREAT_DETECTOR:
    __all__.extend(["ThreatDetector", "SecurityAudit"])

if _HAS_INPUT_SANITIZER:
    __all__.extend(["InputSanitizer", "SecurityValidation"])

if _HAS_INPUT_VALIDATION:
    __all__.extend(["InputValidator", "SecurityValidator", "validate_all_inputs"])