"""PDE solution validation and verification tools."""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    RESEARCH = "research"


@dataclass
class ValidationResult:
    """Result of PDE validation."""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    error_metrics: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    validation_level: ValidationLevel
    test_cases_passed: int
    test_cases_total: int
    
    def summary(self) -> str:
        """Generate validation summary."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"""
PDE Validation Summary
{status} (Confidence: {self.confidence_score:.2%})
Tests: {self.test_cases_passed}/{self.test_cases_total}
Level: {self.validation_level.value.upper()}
Errors: {len(self.errors)} | Warnings: {len(self.warnings)}
"""


class PDEValidator:
    """Comprehensive PDE solution validator."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize PDE validator.
        
        Args:
            validation_level: Strictness of validation
        """
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds based on level
        self.thresholds = self._get_validation_thresholds(validation_level)
        
    def _get_validation_thresholds(self, level: ValidationLevel) -> Dict[str, float]:
        """Get validation thresholds for given level."""
        thresholds = {
            ValidationLevel.BASIC: {
                "max_l2_error": 1e-1,
                "max_l_inf_error": 1e0,
                "min_convergence_rate": -1.0,
                "max_residual": 1e-1,
                "max_boundary_violation": 1e-2
            },
            ValidationLevel.STANDARD: {
                "max_l2_error": 1e-3,
                "max_l_inf_error": 1e-2,
                "min_convergence_rate": 0.1,
                "max_residual": 1e-3,
                "max_boundary_violation": 1e-4
            },
            ValidationLevel.STRICT: {
                "max_l2_error": 1e-6,
                "max_l_inf_error": 1e-5,
                "min_convergence_rate": 0.5,
                "max_residual": 1e-6,
                "max_boundary_violation": 1e-8
            },
            ValidationLevel.RESEARCH: {
                "max_l2_error": 1e-12,
                "max_l_inf_error": 1e-10,
                "min_convergence_rate": 0.8,
                "max_residual": 1e-12,
                "max_boundary_violation": 1e-15
            }
        }
        
        return thresholds[level]
    
    def validate_solution(
        self,
        computed_solution: np.ndarray,
        reference_solution: Optional[np.ndarray] = None,
        pde_equation = None,
        boundary_conditions: Optional[Dict[str, Any]] = None,
        convergence_history: Optional[List[float]] = None
    ) -> ValidationResult:
        """Validate PDE solution comprehensively.
        
        Args:
            computed_solution: Numerical solution to validate
            reference_solution: Known analytical/reference solution (optional)
            pde_equation: PDE equation object
            boundary_conditions: Boundary condition specifications
            convergence_history: Error history during solving
            
        Returns:
            Validation result with detailed analysis
        """
        self.logger.info(f"Starting PDE validation at {self.validation_level.value} level")
        
        errors = []
        warnings = []
        error_metrics = {}
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Basic solution properties
        tests_total += 1
        if self._validate_solution_properties(computed_solution):
            tests_passed += 1
        else:
            errors.append("Solution has invalid properties (NaN/Inf/wrong shape)")
        
        # Test 2: Boundary conditions (if provided)
        if boundary_conditions:
            tests_total += 1
            bc_error = self._validate_boundary_conditions(computed_solution, boundary_conditions)
            error_metrics["boundary_violation"] = bc_error
            
            if bc_error <= self.thresholds["max_boundary_violation"]:
                tests_passed += 1
            else:
                errors.append(f"Boundary condition violation: {bc_error:.2e}")
        
        # Test 3: PDE residual check (if equation provided)
        if pde_equation:
            tests_total += 1
            residual_error = self._validate_pde_residual(computed_solution, pde_equation)
            error_metrics["residual_error"] = residual_error
            
            if residual_error <= self.thresholds["max_residual"]:
                tests_passed += 1
            else:
                errors.append(f"High PDE residual error: {residual_error:.2e}")
        
        # Test 4: Comparison with reference solution
        if reference_solution is not None:
            tests_total += 2
            
            l2_error = self._compute_l2_error(computed_solution, reference_solution)
            linf_error = self._compute_linf_error(computed_solution, reference_solution)
            
            error_metrics["l2_error"] = l2_error
            error_metrics["l_inf_error"] = linf_error
            
            if l2_error <= self.thresholds["max_l2_error"]:
                tests_passed += 1
            else:
                errors.append(f"L2 error too high: {l2_error:.2e}")
            
            if linf_error <= self.thresholds["max_l_inf_error"]:
                tests_passed += 1
            else:
                errors.append(f"L∞ error too high: {linf_error:.2e}")
        
        # Test 5: Convergence analysis
        if convergence_history:
            tests_total += 1
            conv_rate = self._analyze_convergence(convergence_history)
            error_metrics["convergence_rate"] = conv_rate
            
            if conv_rate >= self.thresholds["min_convergence_rate"]:
                tests_passed += 1
            else:
                warnings.append(f"Slow convergence rate: {conv_rate:.3f}")
        
        # Test 6: Physical plausibility
        tests_total += 1
        if self._validate_physical_plausibility(computed_solution):
            tests_passed += 1
        else:
            warnings.append("Solution may not be physically plausible")
        
        # Test 7: Numerical stability
        tests_total += 1
        stability_score = self._assess_numerical_stability(computed_solution)
        error_metrics["stability_score"] = stability_score
        
        if stability_score >= 0.8:
            tests_passed += 1
        elif stability_score >= 0.6:
            warnings.append(f"Moderate stability concerns: {stability_score:.2f}")
        else:
            errors.append(f"Poor numerical stability: {stability_score:.2f}")
        
        # Calculate overall validation result
        is_valid = len(errors) == 0 and tests_passed >= int(0.8 * tests_total)
        confidence_score = tests_passed / tests_total if tests_total > 0 else 0.0
        
        # Adjust confidence based on validation level
        if self.validation_level == ValidationLevel.RESEARCH and confidence_score < 0.95:
            is_valid = False
        elif self.validation_level == ValidationLevel.STRICT and confidence_score < 0.90:
            is_valid = False
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            error_metrics=error_metrics,
            warnings=warnings,
            errors=errors,
            validation_level=self.validation_level,
            test_cases_passed=tests_passed,
            test_cases_total=tests_total
        )
        
        self.logger.info(f"Validation completed: {'PASSED' if is_valid else 'FAILED'} "
                        f"({confidence_score:.1%} confidence)")
        
        return result
    
    def _validate_solution_properties(self, solution: np.ndarray) -> bool:
        """Validate basic solution properties."""
        try:
            if not isinstance(solution, np.ndarray):
                return False
            
            # Check for NaN or Inf
            if not np.isfinite(solution).all():
                return False
            
            # Check reasonable magnitude
            if np.abs(solution).max() > 1e10:
                return False
                
            # Check for zero solution (usually invalid unless trivial problem)
            if np.allclose(solution, 0, atol=1e-15) and solution.size > 4:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_boundary_conditions(
        self, 
        solution: np.ndarray, 
        boundary_conditions: Dict[str, Any]
    ) -> float:
        """Validate boundary condition satisfaction."""
        try:
            # For 1D case
            if solution.ndim == 1:
                error = 0.0
                if "dirichlet" in str(boundary_conditions.get("type", "")).lower():
                    # Check left boundary
                    if "left" in boundary_conditions:
                        target = boundary_conditions["left"]
                        actual = solution[0]
                        error += abs(actual - target)
                    
                    # Check right boundary  
                    if "right" in boundary_conditions:
                        target = boundary_conditions["right"]
                        actual = solution[-1]
                        error += abs(actual - target)
                
                return error
            
            # For 2D case (simplified)
            elif solution.ndim == 2:
                error = 0.0
                if "dirichlet" in str(boundary_conditions.get("type", "")).lower():
                    # Check all boundaries
                    boundary_values = np.concatenate([
                        solution[0, :],   # Top
                        solution[-1, :],  # Bottom
                        solution[:, 0],   # Left
                        solution[:, -1]   # Right
                    ])
                    
                    target_value = boundary_conditions.get("value", 0.0)
                    error = np.sum(np.abs(boundary_values - target_value))
                
                return error
            
            return 0.0  # Unknown case
            
        except Exception as e:
            self.logger.warning(f"Boundary condition validation failed: {e}")
            return float('inf')
    
    def _validate_pde_residual(self, solution: np.ndarray, pde_equation) -> float:
        """Compute PDE residual error."""
        try:
            # This is a simplified residual check
            # In practice, would apply the PDE operator to the solution
            
            if hasattr(pde_equation, 'compute_residual'):
                return pde_equation.compute_residual(solution)
            
            # Fallback: compute Laplacian residual for elliptic PDEs
            if solution.ndim == 1:
                # Second derivative via finite differences
                residual = np.zeros_like(solution)
                h = 1.0 / (len(solution) - 1)
                
                for i in range(1, len(solution) - 1):
                    laplacian = (solution[i+1] - 2*solution[i] + solution[i-1]) / (h**2)
                    residual[i] = abs(laplacian)  # For Laplace equation: ∇²u = 0
                
                return np.mean(residual[1:-1])  # Exclude boundaries
            
            elif solution.ndim == 2:
                # 2D Laplacian residual
                laplacian = np.zeros_like(solution)
                h = 1.0 / (solution.shape[0] - 1)
                
                for i in range(1, solution.shape[0] - 1):
                    for j in range(1, solution.shape[1] - 1):
                        d2u_dx2 = (solution[i+1, j] - 2*solution[i, j] + solution[i-1, j]) / (h**2)
                        d2u_dy2 = (solution[i, j+1] - 2*solution[i, j] + solution[i, j-1]) / (h**2)
                        laplacian[i, j] = d2u_dx2 + d2u_dy2
                
                return np.mean(np.abs(laplacian[1:-1, 1:-1]))
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"PDE residual validation failed: {e}")
            return float('inf')
    
    def _compute_l2_error(self, computed: np.ndarray, reference: np.ndarray) -> float:
        """Compute L2 error between computed and reference solutions."""
        try:
            if computed.shape != reference.shape:
                self.logger.warning("Shape mismatch in L2 error computation")
                return float('inf')
            
            return np.sqrt(np.mean((computed - reference)**2))
        except Exception:
            return float('inf')
    
    def _compute_linf_error(self, computed: np.ndarray, reference: np.ndarray) -> float:
        """Compute L∞ error between computed and reference solutions."""
        try:
            if computed.shape != reference.shape:
                return float('inf')
            
            return np.max(np.abs(computed - reference))
        except Exception:
            return float('inf')
    
    def _analyze_convergence(self, convergence_history: List[float]) -> float:
        """Analyze convergence rate from error history."""
        try:
            if len(convergence_history) < 3:
                return -1.0
            
            # Compute convergence rate over last iterations
            errors = np.array(convergence_history[-10:])  # Last 10 iterations
            
            if len(errors) < 2:
                return -1.0
            
            # Linear fit in log space to estimate convergence rate
            iterations = np.arange(len(errors))
            
            # Avoid log(0)
            errors = np.maximum(errors, 1e-15)
            log_errors = np.log(errors)
            
            # Linear regression
            if len(iterations) >= 2:
                coeff = np.polyfit(iterations, log_errors, 1)
                convergence_rate = -coeff[0]  # Negative slope indicates convergence
                return max(0.0, convergence_rate)
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Convergence analysis failed: {e}")
            return -1.0
    
    def _validate_physical_plausibility(self, solution: np.ndarray) -> bool:
        """Check if solution is physically plausible."""
        try:
            # Basic checks for physical plausibility
            
            # 1. Solution should be smooth (no wild oscillations)
            if solution.ndim == 1 and len(solution) > 2:
                gradients = np.diff(solution)
                second_derivatives = np.diff(gradients)
                
                # Check for excessive oscillations
                if np.std(second_derivatives) > 10 * np.std(gradients):
                    return False
            
            # 2. Solution magnitude should be reasonable
            if np.abs(solution).max() > 1e6:
                return False
            
            # 3. Solution should not have extreme local variations
            if solution.ndim == 1:
                local_variation = np.max(np.abs(np.diff(solution)))
                global_range = np.abs(solution.max() - solution.min())
                
                if global_range > 0 and local_variation / global_range > 0.5:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _assess_numerical_stability(self, solution: np.ndarray) -> float:
        """Assess numerical stability of solution."""
        try:
            stability_score = 1.0
            
            # 1. Check condition number (for matrix-like solutions)
            if solution.ndim == 2 and solution.shape[0] == solution.shape[1]:
                try:
                    cond_num = np.linalg.cond(solution)
                    if cond_num > 1e12:
                        stability_score *= 0.5
                    elif cond_num > 1e8:
                        stability_score *= 0.7
                except:
                    pass
            
            # 2. Check for near-singular behavior
            solution_range = solution.max() - solution.min()
            if solution_range < 1e-12:
                stability_score *= 0.6
            
            # 3. Check smoothness
            if solution.ndim == 1 and len(solution) > 3:
                smoothness = 1.0 / (1.0 + np.var(np.diff(solution, n=2)))
                stability_score *= smoothness
            
            # 4. Check for monotonic behavior where expected
            # This is problem-dependent, so we skip it for generality
            
            return min(1.0, stability_score)
            
        except Exception:
            return 0.0
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate detailed validation report."""
        report_lines = [
            "=" * 60,
            "ANALOG PDE SOLVER - VALIDATION REPORT",
            "=" * 60,
            f"Validation Level: {result.validation_level.value.upper()}",
            f"Overall Result: {'✅ PASSED' if result.is_valid else '❌ FAILED'}",
            f"Confidence Score: {result.confidence_score:.1%}",
            f"Tests Passed: {result.test_cases_passed}/{result.test_cases_total}",
            ""
        ]
        
        # Error metrics
        if result.error_metrics:
            report_lines.append("Error Metrics:")
            for metric, value in result.error_metrics.items():
                report_lines.append(f"  {metric}: {value:.2e}")
            report_lines.append("")
        
        # Errors
        if result.errors:
            report_lines.append("❌ Errors:")
            for error in result.errors:
                report_lines.append(f"  • {error}")
            report_lines.append("")
        
        # Warnings
        if result.warnings:
            report_lines.append("⚠️  Warnings:")
            for warning in result.warnings:
                report_lines.append(f"  • {warning}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("📋 Recommendations:")
        if result.confidence_score < 0.8:
            report_lines.append("  • Consider increasing solver precision or iterations")
        if "l2_error" in result.error_metrics and result.error_metrics["l2_error"] > 1e-6:
            report_lines.append("  • High L2 error indicates poor solution accuracy")
        if "convergence_rate" in result.error_metrics and result.error_metrics["convergence_rate"] < 0.5:
            report_lines.append("  • Slow convergence - check problem conditioning")
        if len(result.warnings) > 0:
            report_lines.append("  • Address warnings to improve solution quality")
        
        report_lines.extend([
            "",
            "=" * 60,
            "Report generated by Terragon Labs Validation Suite",
            "=" * 60
        ])
        
        return "\n".join(report_lines)