"""Input sanitization and validation for security."""

import re
import numpy as np
from typing import Any, Dict, List, Union, Optional
import logging


class InputSanitizer:
    """Sanitize and validate inputs for security."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Allowed patterns
        self.safe_patterns = {
            'parameter_name': r'^[a-zA-Z_][a-zA-Z0-9_]{0,50}$',
            'file_extension': r'^\.(py|json|txt|md|yml|yaml)$',
            'numeric_string': r'^-?\d+\.?\d*([eE][+-]?\d+)?$'
        }
        
        # Maximum sizes
        self.size_limits = {
            'string_length': 1000,
            'array_size': 1000000,
            'dict_keys': 100,
            'nesting_depth': 10
        }
    
    def sanitize_solver_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize solver parameters."""
        sanitized = {}
        
        # Validate crossbar_size
        if 'crossbar_size' in params:
            size = params['crossbar_size']
            if not isinstance(size, int) or size <= 0 or size > 10000:
                raise ValueError(f"Invalid crossbar_size: {size}")
            sanitized['crossbar_size'] = size
        
        # Validate conductance_range
        if 'conductance_range' in params:
            cond_range = params['conductance_range']
            if not isinstance(cond_range, (list, tuple)) or len(cond_range) != 2:
                raise ValueError(f"Invalid conductance_range: {cond_range}")
            
            g_min, g_max = float(cond_range[0]), float(cond_range[1])
            if g_min <= 0 or g_max <= 0 or g_min >= g_max:
                raise ValueError(f"Invalid conductance values: {g_min}, {g_max}")
            
            sanitized['conductance_range'] = (g_min, g_max)
        
        # Validate noise_model
        if 'noise_model' in params:
            noise_model = params['noise_model']
            allowed_models = {'none', 'gaussian', 'realistic', 'shot', 'thermal', 'flicker'}
            if noise_model not in allowed_models:
                raise ValueError(f"Invalid noise_model: {noise_model}")
            sanitized['noise_model'] = str(noise_model)
        
        # Validate convergence parameters
        if 'iterations' in params:
            iterations = params['iterations']
            if not isinstance(iterations, int) or iterations <= 0 or iterations > 100000:
                raise ValueError(f"Invalid iterations: {iterations}")
            sanitized['iterations'] = iterations
        
        if 'convergence_threshold' in params:
            threshold = params['convergence_threshold']
            if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 1:
                raise ValueError(f"Invalid convergence_threshold: {threshold}")
            sanitized['convergence_threshold'] = float(threshold)
        
        return sanitized
    
    def sanitize_array_input(self, array: np.ndarray, name: str = "array") -> np.ndarray:
        """Sanitize numpy array inputs."""
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be numpy array")
        
        # Check size limits
        if array.size > self.size_limits['array_size']:
            raise ValueError(f"{name} too large: {array.size} elements")
        
        # Check for invalid values
        if not np.isfinite(array).all():
            self.logger.warning(f"{name} contains NaN or infinity, cleaning...")
            array = np.nan_to_num(array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for reasonable value ranges
        if np.abs(array).max() > 1e12:
            self.logger.warning(f"{name} contains very large values, clipping...")
            array = np.clip(array, -1e12, 1e12)
        
        return array.copy()
    
    def sanitize_file_path(self, filepath: str) -> str:
        """Sanitize file path for security."""
        if not isinstance(filepath, str):
            raise TypeError("File path must be string")
        
        # Remove null bytes and control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', filepath)
        
        # Check for directory traversal
        if '..' in cleaned or '~' in cleaned:
            raise ValueError("Directory traversal not allowed")
        
        # Check for absolute paths outside allowed directories
        if os.path.isabs(cleaned):
            allowed_roots = ['/tmp', '/var/tmp', os.getcwd()]
            if not any(cleaned.startswith(root) for root in allowed_roots):
                raise ValueError("Absolute path not in allowed directories")
        
        # Validate file extension
        _, ext = os.path.splitext(cleaned)
        if ext and not re.match(self.safe_patterns['file_extension'], ext):
            raise ValueError(f"File extension not allowed: {ext}")
        
        return cleaned
    
    def validate_json_input(self, data: Dict[str, Any], max_depth: int = 5) -> Dict[str, Any]:
        """Validate and sanitize JSON-like input data."""
        return self._validate_dict_recursive(data, current_depth=0, max_depth=max_depth)
    
    def _validate_dict_recursive(self, data: Dict[str, Any], current_depth: int, max_depth: int) -> Dict[str, Any]:
        """Recursively validate dictionary data."""
        if current_depth > max_depth:
            raise ValueError(f"Dictionary nesting too deep: {current_depth} > {max_depth}")
        
        if len(data) > self.size_limits['dict_keys']:
            raise ValueError(f"Too many dictionary keys: {len(data)}")
        
        sanitized = {}
        
        for key, value in data.items():
            # Validate key
            if not isinstance(key, str):
                raise TypeError(f"Dictionary keys must be strings, got {type(key)}")
            
            if len(key) > 100:
                raise ValueError(f"Key too long: {len(key)} characters")
            
            # Sanitize key
            clean_key = re.sub(r'[^\w\-.]', '', key)
            if not clean_key:
                raise ValueError(f"Invalid key after sanitization: {key}")
            
            # Validate and sanitize value
            if isinstance(value, dict):
                sanitized[clean_key] = self._validate_dict_recursive(value, current_depth + 1, max_depth)
            elif isinstance(value, str):
                if len(value) > self.size_limits['string_length']:
                    raise ValueError(f"String value too long: {len(value)}")
                sanitized[clean_key] = value
            elif isinstance(value, (int, float, bool)):
                sanitized[clean_key] = value
            elif isinstance(value, list):
                if len(value) > 1000:
                    raise ValueError(f"List too long: {len(value)}")
                sanitized[clean_key] = value
            else:
                raise TypeError(f"Unsupported value type: {type(value)}")
        
        return sanitized


class SecurityValidation:
    """High-level security validation orchestrator."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.threat_detector = ThreatDetector()
        self.logger = logging.getLogger(__name__)
    
    def validate_all_inputs(self, **kwargs) -> Dict[str, Any]:
        """Comprehensive input validation pipeline."""
        try:
            # Step 1: Threat detection
            for key, value in kwargs.items():
                if not self.threat_detector.scan_input(value, source=key):
                    raise SecurityError(f"Threat detected in input: {key}")
            
            # Step 2: Input sanitization  
            if 'solver_params' in kwargs:
                kwargs['solver_params'] = self.sanitizer.sanitize_solver_params(
                    kwargs['solver_params']
                )
            
            if 'array_data' in kwargs:
                kwargs['array_data'] = self.sanitizer.sanitize_array_input(
                    kwargs['array_data'], name='input_array'
                )
            
            if 'config_data' in kwargs:
                kwargs['config_data'] = self.sanitizer.validate_json_input(
                    kwargs['config_data']
                )
            
            # Step 3: Rate limiting check
            for operation in ['solve', 'program_crossbar', 'file_access']:
                if operation in kwargs:
                    if not self.threat_detector.check_rate_limits(operation):
                        raise SecurityError(f"Rate limit exceeded for {operation}")
            
            self.logger.debug("All inputs validated successfully")
            return kwargs
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            raise


class SecurityError(Exception):
    """Security validation error."""
    pass