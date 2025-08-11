"""Advanced Validation Layer for Analog PDE Solver Algorithms.

This module provides comprehensive validation mechanisms to ensure
algorithmic correctness, numerical stability, and result integrity
across all advanced analog computing algorithms.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from ..utils.logger import get_logger


class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = 1     # Basic input/output validation
    STANDARD = 2    # Input validation + numerical stability checks
    STRICT = 3      # Full validation including mathematical properties
    PARANOID = 4    # Exhaustive validation for critical applications


class ValidationResult(Enum):
    """Validation result types."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    level: ValidationResult
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    overall_result: ValidationResult
    issues: List[ValidationIssue]
    metrics: Dict[str, float]
    timestamp: float
    algorithm_name: str
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return any(issue.level == ValidationResult.CRITICAL for issue in self.issues)
    
    @property
    def has_failures(self) -> bool:
        """Check if report contains failures."""
        return any(issue.level in [ValidationResult.FAILED, ValidationResult.CRITICAL] 
                  for issue in self.issues)


class NumericalValidator:
    """Validates numerical properties and stability."""
    
    def __init__(self, tolerance: float = 1e-12):
        self.logger = get_logger('numerical_validator')
        self.tolerance = tolerance
    
    def validate_matrix_properties(
        self,
        matrix: np.ndarray,
        expected_properties: List[str]
    ) -> List[ValidationIssue]:
        """Validate mathematical properties of matrices.
        
        Args:
            matrix: Matrix to validate
            expected_properties: List of expected properties
                ('symmetric', 'positive_definite', 'sparse', 'bounded', etc.)
        """
        issues = []
        
        if not isinstance(matrix, np.ndarray):
            issues.append(ValidationIssue(
                ValidationResult.CRITICAL,
                "type_error",
                "Matrix is not a numpy array",
                {"type": str(type(matrix))}
            ))
            return issues
        
        # Check for NaN or infinite values
        if not np.isfinite(matrix).all():
            nan_count = np.isnan(matrix).sum()
            inf_count = np.isinf(matrix).sum()
            issues.append(ValidationIssue(
                ValidationResult.CRITICAL,
                "numerical_instability",
                f"Matrix contains {nan_count} NaN and {inf_count} infinite values",
                {"nan_count": int(nan_count), "inf_count": int(inf_count)},
                "Check input data and algorithm numerical stability"
            ))
        
        # Validate specific properties
        for prop in expected_properties:
            if prop == "symmetric":
                if matrix.shape[0] == matrix.shape[1]:
                    symmetry_error = np.max(np.abs(matrix - matrix.T))
                    if symmetry_error > self.tolerance:
                        issues.append(ValidationIssue(
                            ValidationResult.WARNING,
                            "symmetry_violation",
                            f"Matrix not symmetric (max error: {symmetry_error:.2e})",
                            {"symmetry_error": float(symmetry_error)},
                            "Ensure algorithm preserves symmetry constraints"
                        ))
            
            elif prop == "positive_definite":
                if matrix.shape[0] == matrix.shape[1]:
                    try:
                        eigenvals = np.linalg.eigvals(matrix)
                        min_eigenval = np.min(eigenvals)
                        if min_eigenval <= 0:
                            issues.append(ValidationIssue(
                                ValidationResult.FAILED,
                                "positive_definite_violation",
                                f"Matrix not positive definite (min eigenvalue: {min_eigenval:.2e})",
                                {"min_eigenvalue": float(min_eigenval)},
                                "Check problem formulation and boundary conditions"
                            ))
                    except np.linalg.LinAlgError as e:
                        issues.append(ValidationIssue(
                            ValidationResult.WARNING,
                            "eigenvalue_computation_failed",
                            f"Cannot compute eigenvalues: {e}",
                            {"error": str(e)}
                        ))
            
            elif prop == "bounded":
                matrix_max = np.max(np.abs(matrix))
                if matrix_max > 1e10:  # Arbitrary large number threshold
                    issues.append(ValidationIssue(
                        ValidationResult.WARNING,
                        "large_values",
                        f"Matrix contains very large values (max: {matrix_max:.2e})",
                        {"max_absolute_value": float(matrix_max)},
                        "Consider scaling or preconditioning"
                    ))
            
            elif prop == "sparse":
                sparsity = np.mean(np.abs(matrix) < self.tolerance)
                if sparsity < 0.5:  # Less than 50% sparse
                    issues.append(ValidationIssue(
                        ValidationResult.WARNING,
                        "low_sparsity",
                        f"Matrix not very sparse (sparsity: {sparsity:.1%})",
                        {"sparsity_level": float(sparsity)},
                        "Consider alternative algorithms for dense problems"
                    ))
        
        return issues
    
    def validate_convergence(
        self,
        residual_history: List[float],
        tolerance: float
    ) -> List[ValidationIssue]:
        """Validate convergence properties."""
        issues = []
        
        if not residual_history:
            issues.append(ValidationIssue(
                ValidationResult.FAILED,
                "empty_history",
                "No convergence history provided",
                fix_suggestion="Ensure solver records iteration history"
            ))
            return issues
        
        residuals = np.array(residual_history)
        
        # Check for convergence
        final_residual = residuals[-1]
        if final_residual > tolerance:
            issues.append(ValidationIssue(
                ValidationResult.WARNING,
                "poor_convergence",
                f"Final residual {final_residual:.2e} exceeds tolerance {tolerance:.2e}",
                {
                    "final_residual": float(final_residual),
                    "target_tolerance": float(tolerance),
                    "convergence_ratio": float(final_residual / tolerance)
                },
                "Increase iteration count or improve preconditioning"
            ))
        
        # Check convergence rate
        if len(residuals) > 5:
            recent_residuals = residuals[-5:]
            if np.all(recent_residuals[1:] >= recent_residuals[:-1]):  # Not decreasing
                issues.append(ValidationIssue(
                    ValidationResult.WARNING,
                    "stagnant_convergence",
                    "Residual not decreasing in recent iterations",
                    {"recent_trend": "non_decreasing"},
                    "Check for numerical issues or adjust solver parameters"
                ))
        
        # Check for oscillatory behavior
        if len(residuals) > 10:
            # Simple oscillation detection: count sign changes in residual differences
            diffs = np.diff(residuals[-10:])
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes > 5:  # More than half are sign changes
                issues.append(ValidationIssue(
                    ValidationResult.WARNING,
                    "oscillatory_convergence",
                    f"Detected oscillatory behavior ({sign_changes} sign changes in last 10 iterations)",
                    {"sign_changes": int(sign_changes)},
                    "Reduce relaxation parameter or improve damping"
                ))
        
        return issues


class PhysicsValidator:
    """Validates physics-related properties and constraints."""
    
    def __init__(self):
        self.logger = get_logger('physics_validator')
    
    def validate_conservation_laws(
        self,
        solution: np.ndarray,
        domain_info: Dict[str, Any],
        conservation_type: str = 'mass'
    ) -> List[ValidationIssue]:
        """Validate conservation law compliance."""
        issues = []
        
        if conservation_type == 'mass':
            # Simple mass conservation check (integral should remain constant)
            total_mass = np.sum(solution)
            
            if 'expected_mass' in domain_info:
                expected_mass = domain_info['expected_mass']
                mass_error = abs(total_mass - expected_mass) / abs(expected_mass)
                
                if mass_error > 1e-6:
                    issues.append(ValidationIssue(
                        ValidationResult.WARNING,
                        "mass_conservation_violation",
                        f"Mass conservation error: {mass_error:.2e}",
                        {
                            "computed_mass": float(total_mass),
                            "expected_mass": float(expected_mass),
                            "relative_error": float(mass_error)
                        },
                        "Check discretization scheme and boundary conditions"
                    ))
        
        elif conservation_type == 'energy':
            # Energy conservation checks would go here
            pass
        
        return issues
    
    def validate_boundary_conditions(
        self,
        solution: np.ndarray,
        boundary_spec: Dict[str, Any],
        domain_shape: Tuple[int, ...]
    ) -> List[ValidationIssue]:
        """Validate boundary condition compliance."""
        issues = []
        
        if 'dirichlet' in boundary_spec:
            dirichlet_value = boundary_spec.get('dirichlet_value', 0.0)
            tolerance = boundary_spec.get('tolerance', 1e-6)
            
            # Check 2D boundary conditions
            if len(domain_shape) == 2:
                solution_2d = solution.reshape(domain_shape)
                
                # Check all four boundaries
                boundaries = [
                    solution_2d[0, :],    # Top
                    solution_2d[-1, :],   # Bottom
                    solution_2d[:, 0],    # Left
                    solution_2d[:, -1]    # Right
                ]
                
                for i, boundary in enumerate(boundaries):
                    max_violation = np.max(np.abs(boundary - dirichlet_value))
                    if max_violation > tolerance:
                        boundary_names = ['top', 'bottom', 'left', 'right']
                        issues.append(ValidationIssue(
                            ValidationResult.WARNING,
                            "boundary_condition_violation",
                            f"Dirichlet BC violation on {boundary_names[i]} boundary (max error: {max_violation:.2e})",
                            {
                                "boundary": boundary_names[i],
                                "max_violation": float(max_violation),
                                "expected_value": float(dirichlet_value)
                            },
                            "Check boundary condition implementation"
                        ))
        
        return issues


class AlgorithmValidator:
    """High-level algorithm validator."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.logger = get_logger('algorithm_validator')
        self.validation_level = validation_level
        self.numerical_validator = NumericalValidator()
        self.physics_validator = PhysicsValidator()
    
    def validate_algorithm_execution(
        self,
        algorithm_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        algorithm_metadata: Dict[str, Any]
    ) -> ValidationReport:
        """Comprehensive algorithm validation.
        
        Args:
            algorithm_name: Name of the algorithm
            inputs: Algorithm inputs
            outputs: Algorithm outputs  
            algorithm_metadata: Additional algorithm information
        """
        import time
        
        issues = []
        metrics = {}
        
        # Basic input/output validation (all levels)
        if self.validation_level.value >= ValidationLevel.MINIMAL.value:
            issues.extend(self._validate_basic_io(inputs, outputs))
        
        # Numerical stability validation (standard and above)
        if self.validation_level.value >= ValidationLevel.STANDARD.value:
            if 'solution' in outputs:
                solution = outputs['solution']
                issues.extend(self.numerical_validator.validate_matrix_properties(
                    solution, ['bounded']
                ))
            
            if 'convergence_history' in outputs:
                tolerance = algorithm_metadata.get('tolerance', 1e-6)
                issues.extend(self.numerical_validator.validate_convergence(
                    outputs['convergence_history'], tolerance
                ))
        
        # Strict mathematical validation (strict and above)
        if self.validation_level.value >= ValidationLevel.STRICT.value:
            if 'solution' in outputs and 'domain_info' in algorithm_metadata:
                # Physics validation
                conservation_type = algorithm_metadata.get('conservation_type', 'mass')
                issues.extend(self.physics_validator.validate_conservation_laws(
                    outputs['solution'], 
                    algorithm_metadata['domain_info'],
                    conservation_type
                ))
                
                # Boundary condition validation
                if 'boundary_spec' in algorithm_metadata:
                    domain_shape = algorithm_metadata.get('domain_shape', outputs['solution'].shape)
                    issues.extend(self.physics_validator.validate_boundary_conditions(
                        outputs['solution'],
                        algorithm_metadata['boundary_spec'],
                        domain_shape
                    ))
        
        # Paranoid validation (paranoid level)
        if self.validation_level.value >= ValidationLevel.PARANOID.value:
            # Additional extensive checks would go here
            pass
        
        # Compute overall result
        if any(issue.level == ValidationResult.CRITICAL for issue in issues):
            overall_result = ValidationResult.CRITICAL
        elif any(issue.level == ValidationResult.FAILED for issue in issues):
            overall_result = ValidationResult.FAILED
        elif any(issue.level == ValidationResult.WARNING for issue in issues):
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.PASSED
        
        # Compute metrics
        metrics['total_issues'] = len(issues)
        metrics['critical_issues'] = sum(1 for i in issues if i.level == ValidationResult.CRITICAL)
        metrics['failed_issues'] = sum(1 for i in issues if i.level == ValidationResult.FAILED)
        metrics['warning_issues'] = sum(1 for i in issues if i.level == ValidationResult.WARNING)
        
        return ValidationReport(
            overall_result=overall_result,
            issues=issues,
            metrics=metrics,
            timestamp=time.time(),
            algorithm_name=algorithm_name
        )
    
    def _validate_basic_io(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Basic input/output validation."""
        issues = []
        
        # Check for required output fields
        required_outputs = ['solution']
        for field in required_outputs:
            if field not in outputs:
                issues.append(ValidationIssue(
                    ValidationResult.CRITICAL,
                    "missing_output",
                    f"Required output '{field}' missing",
                    {"missing_field": field}
                ))
        
        # Validate solution format
        if 'solution' in outputs:
            solution = outputs['solution']
            if not isinstance(solution, np.ndarray):
                issues.append(ValidationIssue(
                    ValidationResult.CRITICAL,
                    "invalid_solution_type",
                    f"Solution must be numpy array, got {type(solution)}",
                    {"solution_type": str(type(solution))}
                ))
            elif solution.size == 0:
                issues.append(ValidationIssue(
                    ValidationResult.CRITICAL,
                    "empty_solution",
                    "Solution array is empty"
                ))
        
        return issues


# Global validator instance for easy access
default_validator = AlgorithmValidator(ValidationLevel.STANDARD)


def validate_algorithm_result(
    algorithm_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    metadata: Dict[str, Any] = None,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationReport:
    """Convenience function for algorithm validation.
    
    Args:
        algorithm_name: Name of algorithm
        inputs: Algorithm inputs
        outputs: Algorithm outputs
        metadata: Additional metadata for validation
        validation_level: Level of validation strictness
    
    Returns:
        Comprehensive validation report
    """
    validator = AlgorithmValidator(validation_level)
    return validator.validate_algorithm_execution(
        algorithm_name, inputs, outputs, metadata or {}
    )