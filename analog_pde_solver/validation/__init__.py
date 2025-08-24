"""Validation and verification module for analog PDE solver."""

__version__ = "0.1.0"

# Import available modules with graceful fallback
try:
    from .pde_validator import PDEValidator, ValidationResult
    _HAS_PDE_VALIDATOR = True
except ImportError:
    _HAS_PDE_VALIDATOR = False
    PDEValidator = None
    ValidationResult = None

try:
    from .hardware_validator import HardwareValidator, HardwareValidationResult
    _HAS_HARDWARE_VALIDATOR = True
except ImportError:
    _HAS_HARDWARE_VALIDATOR = False
    HardwareValidator = None
    HardwareValidationResult = None

try:
    from .performance_validator import PerformanceValidator, PerformanceBounds
    _HAS_PERFORMANCE_VALIDATOR = True
except ImportError:
    _HAS_PERFORMANCE_VALIDATOR = False
    PerformanceValidator = None
    PerformanceBounds = None

try:
    from .quality_gates import QualityGateRunner, run_comprehensive_quality_check
    _HAS_QUALITY_GATES = True
except ImportError:
    _HAS_QUALITY_GATES = False
    QualityGateRunner = None
    run_comprehensive_quality_check = None

try:
    from .comprehensive_quality_gates import ComprehensiveQualityGates
    _HAS_COMPREHENSIVE_QUALITY_GATES = True
except ImportError:
    _HAS_COMPREHENSIVE_QUALITY_GATES = False
    ComprehensiveQualityGates = None

__all__ = []

# Add available components to __all__
if _HAS_PDE_VALIDATOR:
    __all__.extend(["PDEValidator", "ValidationResult"])

if _HAS_HARDWARE_VALIDATOR:
    __all__.extend(["HardwareValidator", "HardwareValidationResult"])

if _HAS_PERFORMANCE_VALIDATOR:
    __all__.extend(["PerformanceValidator", "PerformanceBounds"])

if _HAS_QUALITY_GATES:
    __all__.extend(["QualityGateRunner", "run_comprehensive_quality_check"])

if _HAS_COMPREHENSIVE_QUALITY_GATES:
    __all__.extend(["ComprehensiveQualityGates"])