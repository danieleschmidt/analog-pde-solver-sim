"""Comprehensive input validation and sanitization for analog PDE solver."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any
    warnings: List[str]
    errors: List[str]
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_crossbar_size(self, size: Any) -> ValidationResult:
        """Validate and sanitize crossbar size parameter."""
        warnings = []
        errors = []
        
        # Type validation
        if not isinstance(size, (int, np.integer)):
            try:
                size = int(size)
                warnings.append(f"Crossbar size converted to integer: {size}")
            except (ValueError, TypeError):
                errors.append(f"Crossbar size must be integer, got {type(size)}")
                return ValidationResult(False, None, warnings, errors)
        
        # Range validation
        if size <= 0:
            errors.append(f"Crossbar size must be positive, got {size}")
            return ValidationResult(False, None, warnings, errors)
        
        if size > 10000:
            errors.append(f"Crossbar size too large (>{10000}), got {size}")
            return ValidationResult(False, None, warnings, errors)
        
        # Performance warnings
        if size > 1000:
            warnings.append(f"Large crossbar size ({size}) may impact performance")
        elif size < 4:
            warnings.append(f"Small crossbar size ({size}) may limit accuracy")
            
        return ValidationResult(True, int(size), warnings, errors)
    
    def validate_conductance_range(self, range_val: Any) -> ValidationResult:
        """Validate and sanitize conductance range."""
        warnings = []
        errors = []
        
        # Type validation
        if not isinstance(range_val, (tuple, list)):
            errors.append(f"Conductance range must be tuple/list, got {type(range_val)}")
            return ValidationResult(False, None, warnings, errors)
        
        if len(range_val) != 2:
            errors.append(f"Conductance range must have 2 elements, got {len(range_val)}")
            return ValidationResult(False, None, warnings, errors)
        
        try:
            g_min, g_max = float(range_val[0]), float(range_val[1])
        except (ValueError, TypeError):
            errors.append("Conductance range values must be numeric")
            return ValidationResult(False, None, warnings, errors)
        
        # Physical constraints
        if g_min <= 0 or g_max <= 0:
            errors.append(f"Conductance values must be positive, got ({g_min}, {g_max})")
            return ValidationResult(False, None, warnings, errors)
        
        if g_min >= g_max:
            errors.append(f"g_min must be < g_max, got ({g_min}, {g_max})")
            return ValidationResult(False, None, warnings, errors)
        
        # Realistic ranges for memristive devices
        if g_min < 1e-12:
            warnings.append(f"Very small g_min ({g_min:.2e}S) may be unrealistic")
        if g_max > 1e-3:
            warnings.append(f"Very large g_max ({g_max:.2e}S) may be unrealistic")
        
        # Dynamic range check
        dynamic_range = g_max / g_min
        if dynamic_range < 10:
            warnings.append(f"Small dynamic range ({dynamic_range:.1f}) may limit precision")
        elif dynamic_range > 1e6:
            warnings.append(f"Large dynamic range ({dynamic_range:.2e}) may cause numerical issues")
            
        return ValidationResult(True, (g_min, g_max), warnings, errors)
    
    def validate_noise_model(self, noise_model: Any) -> ValidationResult:
        """Validate noise model specification."""
        warnings = []
        errors = []
        
        if not isinstance(noise_model, str):
            errors.append(f"Noise model must be string, got {type(noise_model)}")
            return ValidationResult(False, None, warnings, errors)
        
        valid_models = ["none", "gaussian", "realistic", "device", "thermal"]
        noise_model_clean = noise_model.lower().strip()
        
        if noise_model_clean not in valid_models:
            errors.append(f"Invalid noise model '{noise_model}'. Valid: {valid_models}")
            return ValidationResult(False, None, warnings, errors)
        
        # Warn about computational impact
        if noise_model_clean in ["realistic", "device"]:
            warnings.append("Realistic noise models may significantly slow computation")
            
        return ValidationResult(True, noise_model_clean, warnings, errors)
    
    def validate_domain_size(self, domain_size: Any) -> ValidationResult:
        """Validate PDE domain size."""
        warnings = []
        errors = []
        
        # Handle different input formats
        if isinstance(domain_size, (int, np.integer)):
            domain_size = (int(domain_size),)
        elif isinstance(domain_size, (tuple, list)):
            try:
                domain_size = tuple(int(s) for s in domain_size)
            except (ValueError, TypeError):
                errors.append("Domain size elements must be integers")
                return ValidationResult(False, None, warnings, errors)
        else:
            errors.append(f"Domain size must be int or tuple, got {type(domain_size)}")
            return ValidationResult(False, None, warnings, errors)
        
        # Validate dimensions
        if len(domain_size) > 3:
            errors.append(f"Maximum 3D domains supported, got {len(domain_size)}D")
            return ValidationResult(False, None, warnings, errors)
        
        # Validate sizes
        for i, size in enumerate(domain_size):
            if size <= 2:
                errors.append(f"Domain size[{i}] must be > 2, got {size}")
                return ValidationResult(False, None, warnings, errors)
            if size > 10000:
                errors.append(f"Domain size[{i}] too large (>{10000}), got {size}")
                return ValidationResult(False, None, warnings, errors)
        
        # Performance warnings
        total_points = np.prod(domain_size)
        if total_points > 1000000:
            warnings.append(f"Large domain ({total_points} points) may require significant memory")
        elif total_points < 16:
            warnings.append(f"Small domain ({total_points} points) may have poor accuracy")
            
        return ValidationResult(True, domain_size, warnings, errors)
    
    def validate_source_function(self, source_func: Any) -> ValidationResult:
        """Validate source function."""
        warnings = []
        errors = []
        
        if source_func is None:
            return ValidationResult(True, None, warnings, errors)
        
        if not callable(source_func):
            errors.append(f"Source function must be callable, got {type(source_func)}")
            return ValidationResult(False, None, warnings, errors)
        
        # Test function with sample inputs
        try:
            # Test 1D case
            result_1d = source_func(0.5, 0.0)
            if not isinstance(result_1d, (int, float, np.number)):
                warnings.append("Source function should return numeric values")
            
            # Test for potential numerical issues
            test_points = [0.0, 0.5, 1.0]
            results = []
            for x in test_points:
                try:
                    val = source_func(x, 0.0)
                    results.append(val)
                except Exception as e:
                    warnings.append(f"Source function failed at x={x}: {e}")
            
            if results:
                max_val = max(abs(r) for r in results if np.isfinite(r))
                if max_val > 1e6:
                    warnings.append(f"Source function values very large (max: {max_val:.2e})")
                elif max_val == 0:
                    warnings.append("Source function appears to return only zeros")
                    
        except Exception as e:
            warnings.append(f"Could not validate source function: {e}")
            
        return ValidationResult(True, source_func, warnings, errors)
    
    def validate_boundary_conditions(self, bc: Any) -> ValidationResult:
        """Validate boundary conditions."""
        warnings = []
        errors = []
        
        if not isinstance(bc, str):
            errors.append(f"Boundary conditions must be string, got {type(bc)}")
            return ValidationResult(False, None, warnings, errors)
        
        valid_bcs = ["dirichlet", "neumann", "periodic", "mixed"]
        bc_clean = bc.lower().strip()
        
        if bc_clean not in valid_bcs:
            errors.append(f"Invalid boundary condition '{bc}'. Valid: {valid_bcs}")
            return ValidationResult(False, None, warnings, errors)
        
        if bc_clean == "mixed":
            warnings.append("Mixed boundary conditions may require special handling")
            
        return ValidationResult(True, bc_clean, warnings, errors)
    
    def validate_solve_parameters(self, 
                                iterations: Any, 
                                threshold: Any) -> ValidationResult:
        """Validate solver parameters."""
        warnings = []
        errors = []
        
        # Validate iterations
        if not isinstance(iterations, (int, np.integer)):
            try:
                iterations = int(iterations)
                warnings.append(f"Iterations converted to integer: {iterations}")
            except (ValueError, TypeError):
                errors.append(f"Iterations must be integer, got {type(iterations)}")
                return ValidationResult(False, None, warnings, errors)
        
        if iterations <= 0:
            errors.append(f"Iterations must be positive, got {iterations}")
            return ValidationResult(False, None, warnings, errors)
        
        if iterations > 100000:
            errors.append(f"Too many iterations (>{100000}), got {iterations}")
            return ValidationResult(False, None, warnings, errors)
        
        if iterations < 10:
            warnings.append(f"Few iterations ({iterations}) may not allow convergence")
        elif iterations > 10000:
            warnings.append(f"Many iterations ({iterations}) may indicate convergence issues")
        
        # Validate threshold
        if not isinstance(threshold, (int, float, np.number)):
            errors.append(f"Threshold must be numeric, got {type(threshold)}")
            return ValidationResult(False, None, warnings, errors)
        
        threshold = float(threshold)
        
        if threshold <= 0:
            errors.append(f"Threshold must be positive, got {threshold}")
            return ValidationResult(False, None, warnings, errors)
        
        if threshold > 1.0:
            warnings.append(f"Large threshold ({threshold}) may give inaccurate results")
        elif threshold < 1e-12:
            warnings.append(f"Very small threshold ({threshold:.2e}) may never be reached")
            
        return ValidationResult(True, (int(iterations), float(threshold)), warnings, errors)
    
    def sanitize_matrix(self, matrix: np.ndarray) -> ValidationResult:
        """Sanitize and validate matrix inputs."""
        warnings = []
        errors = []
        
        if not isinstance(matrix, np.ndarray):
            errors.append(f"Matrix must be numpy array, got {type(matrix)}")
            return ValidationResult(False, None, warnings, errors)
        
        # Check for non-finite values
        if not np.all(np.isfinite(matrix)):
            nan_count = np.sum(~np.isfinite(matrix))
            errors.append(f"Matrix contains {nan_count} non-finite values")
            return ValidationResult(False, None, warnings, errors)
        
        # Check matrix properties
        if matrix.ndim != 2:
            errors.append(f"Matrix must be 2D, got {matrix.ndim}D")
            return ValidationResult(False, None, warnings, errors)
        
        if matrix.shape[0] != matrix.shape[1]:
            errors.append(f"Matrix must be square, got {matrix.shape}")
            return ValidationResult(False, None, warnings, errors)
        
        # Check for numerical issues
        matrix_norm = np.linalg.norm(matrix)
        if matrix_norm > 1e6:
            warnings.append(f"Large matrix norm ({matrix_norm:.2e}) may cause numerical issues")
        elif matrix_norm < 1e-12:
            warnings.append(f"Small matrix norm ({matrix_norm:.2e}) may be singular")
        
        # Check condition number
        try:
            condition_num = np.linalg.cond(matrix)
            if condition_num > 1e12:
                warnings.append(f"Ill-conditioned matrix (condition: {condition_num:.2e})")
        except np.linalg.LinAlgError:
            warnings.append("Could not compute matrix condition number")
        
        return ValidationResult(True, matrix.copy(), warnings, errors)
    
    def sanitize_vector(self, vector: np.ndarray) -> ValidationResult:
        """Sanitize and validate vector inputs."""
        warnings = []
        errors = []
        
        if not isinstance(vector, np.ndarray):
            try:
                vector = np.array(vector)
                warnings.append("Input converted to numpy array")
            except:
                errors.append(f"Could not convert input to numpy array")
                return ValidationResult(False, None, warnings, errors)
        
        if vector.ndim != 1:
            if vector.ndim == 2 and min(vector.shape) == 1:
                vector = vector.flatten()
                warnings.append("Reshaped 2D vector to 1D")
            else:
                errors.append(f"Vector must be 1D, got {vector.ndim}D")
                return ValidationResult(False, None, warnings, errors)
        
        # Check for non-finite values
        if not np.all(np.isfinite(vector)):
            nan_count = np.sum(~np.isfinite(vector))
            errors.append(f"Vector contains {nan_count} non-finite values")
            return ValidationResult(False, None, warnings, errors)
        
        # Check magnitude
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 1e6:
            warnings.append(f"Large vector norm ({vector_norm:.2e})")
        
        return ValidationResult(True, vector.copy(), warnings, errors)


class SecurityValidator:
    """Security-focused validation to prevent malicious inputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_function_source(self, func: Callable) -> ValidationResult:
        """Validate source function for potential security issues."""
        warnings = []
        errors = []
        
        if not callable(func):
            errors.append("Source function must be callable")
            return ValidationResult(False, None, warnings, errors)
        
        # Check function source code for dangerous patterns
        try:
            import inspect
            source = inspect.getsource(func)
            
            # Dangerous patterns to look for
            dangerous_patterns = [
                r'import\s+os',
                r'import\s+subprocess', 
                r'import\s+sys',
                r'exec\s*\(',
                r'eval\s*\(',
                r'__import__',
                r'open\s*\(',
                r'file\s*\(',
                r'input\s*\(',
                r'raw_input\s*\('
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    warnings.append(f"Potentially dangerous pattern found: {pattern}")
                    
        except (OSError, TypeError):
            # Can't get source (e.g., built-in function, lambda)
            warnings.append("Could not inspect function source code")
        
        return ValidationResult(True, func, warnings, errors)
    
    def check_resource_limits(self, crossbar_size: int, domain_size: tuple) -> ValidationResult:
        """Check if requested resources are within safe limits."""
        warnings = []
        errors = []
        
        # Memory estimation
        total_points = np.prod(domain_size) if domain_size else crossbar_size
        estimated_memory_mb = (crossbar_size**2 * 8 + total_points * 8) / (1024**2)
        
        if estimated_memory_mb > 1000:  # 1GB limit
            errors.append(f"Estimated memory usage ({estimated_memory_mb:.1f}MB) exceeds limit")
            return ValidationResult(False, None, warnings, errors)
        elif estimated_memory_mb > 100:  # 100MB warning
            warnings.append(f"High memory usage estimated: {estimated_memory_mb:.1f}MB")
        
        # Computation complexity estimation
        complexity_score = crossbar_size**2 * total_points
        if complexity_score > 1e9:
            errors.append(f"Computational complexity too high: {complexity_score:.2e}")
            return ValidationResult(False, None, warnings, errors)
        elif complexity_score > 1e7:
            warnings.append(f"High computational complexity: {complexity_score:.2e}")
        
        return ValidationResult(True, None, warnings, errors)


def validate_all_inputs(**kwargs) -> Dict[str, ValidationResult]:
    """Comprehensive validation of all solver inputs."""
    validator = InputValidator()
    security_validator = SecurityValidator()
    
    results = {}
    
    # Validate each parameter if provided
    if 'crossbar_size' in kwargs:
        results['crossbar_size'] = validator.validate_crossbar_size(kwargs['crossbar_size'])
    
    if 'conductance_range' in kwargs:
        results['conductance_range'] = validator.validate_conductance_range(kwargs['conductance_range'])
    
    if 'noise_model' in kwargs:
        results['noise_model'] = validator.validate_noise_model(kwargs['noise_model'])
    
    if 'domain_size' in kwargs:
        results['domain_size'] = validator.validate_domain_size(kwargs['domain_size'])
    
    if 'source_function' in kwargs:
        results['source_function'] = validator.validate_source_function(kwargs['source_function'])
        if kwargs['source_function'] is not None:
            results['source_security'] = security_validator.validate_function_source(kwargs['source_function'])
    
    if 'boundary_conditions' in kwargs:
        results['boundary_conditions'] = validator.validate_boundary_conditions(kwargs['boundary_conditions'])
    
    if 'iterations' in kwargs and 'convergence_threshold' in kwargs:
        results['solve_params'] = validator.validate_solve_parameters(
            kwargs['iterations'], kwargs['convergence_threshold']
        )
    
    # Security check for resource limits
    if 'crossbar_size' in kwargs and 'domain_size' in kwargs:
        crossbar_size = kwargs['crossbar_size']
        domain_size = kwargs['domain_size']
        if isinstance(domain_size, (int, np.integer)):
            domain_size = (domain_size,)
        results['resource_limits'] = security_validator.check_resource_limits(crossbar_size, domain_size)
    
    return results