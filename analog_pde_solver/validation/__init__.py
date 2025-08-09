"""Validation and verification module for analog PDE solver."""

__version__ = "0.1.0"

from .pde_validator import PDEValidator, ValidationResult
from .hardware_validator import HardwareValidator, HardwareValidationResult
from .performance_validator import PerformanceValidator, PerformanceBounds

__all__ = [
    "PDEValidator",
    "ValidationResult", 
    "HardwareValidator",
    "HardwareValidationResult",
    "PerformanceValidator", 
    "PerformanceBounds"
]