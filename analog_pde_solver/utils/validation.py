"""Input validation and error handling utilities."""

import numpy as np
from typing import Union, Tuple, Optional, Any
import logging


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


def validate_domain_size(domain_size: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Validate and normalize domain size parameter.
    
    Args:
        domain_size: Domain size specification
        
    Returns:
        Normalized domain size tuple
        
    Raises:
        ValidationError: If domain size is invalid
    """
    if isinstance(domain_size, int):
        if domain_size <= 0:
            raise ValidationError(f"Domain size must be positive, got {domain_size}")
        if domain_size < 2:
            raise ValidationError(f"Domain size too small for finite differences: {domain_size}")
        if domain_size > 10000:
            logger.warning(f"Large domain size {domain_size} may cause memory issues")
        return (domain_size,)
    
    elif isinstance(domain_size, (tuple, list)):
        domain_tuple = tuple(domain_size)
        
        if len(domain_tuple) == 0:
            raise ValidationError("Domain size cannot be empty")
        
        if len(domain_tuple) > 3:
            raise ValidationError(f"Unsupported domain dimension: {len(domain_tuple)}")
            
        for dim in domain_tuple:
            if not isinstance(dim, int):
                raise ValidationError(f"Domain dimensions must be integers, got {type(dim)}")
            if dim <= 0:
                raise ValidationError(f"Domain dimensions must be positive, got {dim}")
            if dim < 2:
                raise ValidationError(f"Domain dimension too small: {dim}")
                
        total_size = np.prod(domain_tuple)
        if total_size > 1e8:
            logger.warning(f"Very large problem size {total_size:.0e} may cause issues")
            
        return domain_tuple
    
    else:
        raise ValidationError(f"Invalid domain size type: {type(domain_size)}")


def validate_conductance_range(conductance_range: Tuple[float, float]) -> Tuple[float, float]:
    """Validate conductance range parameters.
    
    Args:
        conductance_range: (min_conductance, max_conductance) in Siemens
        
    Returns:
        Validated conductance range
        
    Raises:
        ValidationError: If conductance range is invalid
    """
    if not isinstance(conductance_range, (tuple, list)):
        raise ValidationError(f"Conductance range must be tuple/list, got {type(conductance_range)}")
    
    if len(conductance_range) != 2:
        raise ValidationError(f"Conductance range must have 2 elements, got {len(conductance_range)}")
    
    g_min, g_max = conductance_range
    
    if not isinstance(g_min, (int, float)) or not isinstance(g_max, (int, float)):
        raise ValidationError("Conductance values must be numeric")
    
    if g_min <= 0 or g_max <= 0:
        raise ValidationError("Conductance values must be positive")
    
    if g_min >= g_max:
        raise ValidationError(f"Min conductance {g_min} must be less than max {g_max}")
    
    # Check realistic range for memristive devices
    if g_min < 1e-12:
        logger.warning(f"Very low min conductance {g_min:.2e} S may be unrealistic")
    
    if g_max > 1e-3:
        logger.warning(f"Very high max conductance {g_max:.2e} S may be unrealistic")
    
    if g_max / g_min > 1e6:
        logger.warning(f"Large conductance ratio {g_max/g_min:.2e} may affect precision")
    
    return (float(g_min), float(g_max))


def validate_crossbar_size(size: int) -> int:
    """Validate crossbar array size.
    
    Args:
        size: Crossbar array size (square)
        
    Returns:
        Validated size
        
    Raises:
        ValidationError: If size is invalid
    """
    if not isinstance(size, int):
        raise ValidationError(f"Crossbar size must be integer, got {type(size)}")
    
    if size <= 0:
        raise ValidationError(f"Crossbar size must be positive, got {size}")
    
    if size < 2:
        raise ValidationError(f"Crossbar size too small: {size}")
    
    if size > 100000:
        raise ValidationError(f"Crossbar size too large: {size}")
    
    # Check for power of 2 for efficiency
    if size & (size - 1) != 0:
        logger.info(f"Crossbar size {size} is not power of 2, may be less efficient")
    
    return size


def validate_noise_model(noise_model: str) -> str:
    """Validate noise model parameter.
    
    Args:
        noise_model: Noise modeling approach
        
    Returns:
        Validated noise model string
        
    Raises:
        ValidationError: If noise model is invalid
    """
    if not isinstance(noise_model, str):
        raise ValidationError(f"Noise model must be string, got {type(noise_model)}")
    
    valid_models = {"none", "gaussian", "realistic", "shot", "thermal", "flicker"}
    
    if noise_model not in valid_models:
        raise ValidationError(f"Invalid noise model '{noise_model}', "
                            f"must be one of {valid_models}")
    
    return noise_model


def validate_convergence_params(
    iterations: int, 
    threshold: float
) -> Tuple[int, float]:
    """Validate convergence parameters.
    
    Args:
        iterations: Maximum number of iterations
        threshold: Convergence threshold
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(iterations, int):
        raise ValidationError(f"Iterations must be integer, got {type(iterations)}")
    
    if iterations <= 0:
        raise ValidationError(f"Iterations must be positive, got {iterations}")
    
    if iterations > 100000:
        logger.warning(f"High iteration count {iterations} may take long time")
    
    if not isinstance(threshold, (int, float)):
        raise ValidationError(f"Threshold must be numeric, got {type(threshold)}")
    
    if threshold <= 0:
        raise ValidationError(f"Threshold must be positive, got {threshold}")
    
    if threshold < 1e-15:
        logger.warning(f"Very strict threshold {threshold:.2e} may not converge")
    
    if threshold > 1e-2:
        logger.warning(f"Loose threshold {threshold:.2e} may give inaccurate results")
    
    return iterations, float(threshold)


def safe_array_operation(
    func: callable,
    *arrays: np.ndarray,
    error_msg: str = "Array operation failed"
) -> Any:
    """Safely perform numpy array operations with error handling.
    
    Args:
        func: Function to call on arrays
        arrays: Input arrays
        error_msg: Error message on failure
        
    Returns:
        Result of function call
        
    Raises:
        ValidationError: If operation fails
    """
    try:
        # Check for NaN or infinity
        for i, arr in enumerate(arrays):
            if not np.isfinite(arr).all():
                raise ValidationError(f"Array {i} contains NaN or infinity")
        
        result = func(*arrays)
        
        # Check result
        if hasattr(result, 'dtype') and not np.isfinite(result).all():
            raise ValidationError(f"Operation result contains NaN or infinity")
        
        return result
        
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        raise ValidationError(f"{error_msg}: {str(e)}")


def validate_matrix_properties(
    matrix: np.ndarray,
    name: str = "matrix",
    check_symmetric: bool = False,
    check_positive_definite: bool = False
) -> np.ndarray:
    """Validate matrix properties for numerical stability.
    
    Args:
        matrix: Input matrix
        name: Matrix name for error messages
        check_symmetric: Whether to check symmetry
        check_positive_definite: Whether to check positive definiteness
        
    Returns:
        Validated matrix
        
    Raises:
        ValidationError: If matrix properties are invalid
    """
    if not isinstance(matrix, np.ndarray):
        raise ValidationError(f"{name} must be numpy array, got {type(matrix)}")
    
    if matrix.ndim != 2:
        raise ValidationError(f"{name} must be 2D array, got {matrix.ndim}D")
    
    if not np.isfinite(matrix).all():
        raise ValidationError(f"{name} contains NaN or infinity values")
    
    # Check condition number
    try:
        cond = np.linalg.cond(matrix)
        if cond > 1e12:
            logger.warning(f"{name} is ill-conditioned (cond={cond:.2e})")
    except np.linalg.LinAlgError:
        logger.warning(f"Could not compute condition number for {name}")
    
    # Check symmetry if requested
    if check_symmetric:
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            logger.warning(f"{name} is not symmetric")
    
    # Check positive definiteness if requested
    if check_positive_definite:
        try:
            eigenvals = np.linalg.eigvals(matrix)
            if not np.all(eigenvals > 0):
                logger.warning(f"{name} is not positive definite")
        except np.linalg.LinAlgError:
            logger.warning(f"Could not check positive definiteness of {name}")
    
    return matrix