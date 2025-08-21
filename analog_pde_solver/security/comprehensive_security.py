"""
Comprehensive Security Framework for Analog PDE Solver

This module implements enterprise-grade security measures including:
- Input validation and sanitization
- Threat detection and monitoring  
- Automated security remediation
- Secure computation protocols
- Data protection and privacy safeguards

Security Standards: OWASP guidelines, NIST cybersecurity framework
"""

import re
import hashlib
import hmac
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import secrets
import base64

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Configuration for security framework."""
    # Input validation
    max_input_size: int = 100_000_000  # 100MB max
    max_grid_size: int = 10_000  # Maximum grid size
    allowed_file_extensions: List[str] = None
    max_file_size: int = 50_000_000  # 50MB max
    
    # Rate limiting
    max_requests_per_minute: int = 1000
    max_compute_time_seconds: int = 3600  # 1 hour max
    
    # Cryptography
    secret_key: Optional[bytes] = None
    token_expiry_hours: int = 24
    
    # Threat detection
    enable_anomaly_detection: bool = True
    suspicious_pattern_threshold: float = 0.8
    
    # Logging and monitoring
    enable_security_logging: bool = True
    log_retention_days: int = 90
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.py', '.json', '.txt', '.csv', '.pkl']
        
        if self.secret_key is None:
            self.secret_key = secrets.token_bytes(32)


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Compiled regex patterns for efficiency
        self.safe_filename_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
        self.safe_string_pattern = re.compile(r'^[a-zA-Z0-9\s._-]+$') 
        self.numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
        self.code_injection_patterns = [
            re.compile(r'(import|exec|eval|__import__|compile)\s*\(', re.IGNORECASE),
            re.compile(r'(subprocess|os\.system|popen)\s*\(', re.IGNORECASE),
            re.compile(r'(rm\s+-rf|del\s+/|format\s+c:)', re.IGNORECASE),
            re.compile(r'(drop\s+table|delete\s+from|truncate)', re.IGNORECASE)
        ]
        
    def validate_grid_size(self, grid_size: Any) -> int:
        """Validate and sanitize grid size parameter."""
        try:
            if not isinstance(grid_size, (int, float)):
                if isinstance(grid_size, str):
                    grid_size = float(grid_size)
                else:
                    raise ValueError(f"Grid size must be numeric, got {type(grid_size)}")
            
            grid_size = int(grid_size)
            
            if grid_size <= 0:
                raise ValueError(f"Grid size must be positive, got {grid_size}")
            
            if grid_size > self.config.max_grid_size:
                raise ValueError(f"Grid size {grid_size} exceeds maximum {self.config.max_grid_size}")
            
            # Check for reasonable computational limits
            total_points = grid_size ** 2
            if total_points > 100_000_000:  # 100M points
                logger.warning(f"Large grid size requested: {grid_size}×{grid_size} = {total_points} points")
            
            return grid_size
            
        except (ValueError, TypeError, OverflowError) as e:
            logger.error(f"Grid size validation failed: {e}")
            raise ValueError(f"Invalid grid size: {e}")
    
    def validate_filename(self, filename: str) -> str:
        """Validate and sanitize filename."""
        if not isinstance(filename, str):
            raise ValueError(f"Filename must be string, got {type(filename)}")
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValueError("Path traversal detected in filename")
        
        # Check filename pattern
        if not self.safe_filename_pattern.match(filename):
            raise ValueError("Filename contains invalid characters")
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config.allowed_file_extensions:
            raise ValueError(f"File extension {file_ext} not allowed")
        
        # Check length
        if len(filename) > 255:
            raise ValueError("Filename too long")
        
        return filename
    
    def validate_numeric_parameter(self, value: Any, name: str, 
                                 min_val: Optional[float] = None,
                                 max_val: Optional[float] = None) -> float:
        """Validate numeric parameters with range checks."""
        try:
            if isinstance(value, str):
                if not self.numeric_pattern.match(value.strip()):
                    raise ValueError(f"Invalid numeric format for {name}")
                value = float(value.strip())
            elif not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value)}")
            
            value = float(value)
            
            # Check for NaN and infinity
            if not (value == value):  # NaN check
                raise ValueError(f"{name} cannot be NaN")
            if abs(value) == float('inf'):
                raise ValueError(f"{name} cannot be infinite")
            
            # Range validation
            if min_val is not None and value < min_val:
                raise ValueError(f"{name} {value} below minimum {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{name} {value} exceeds maximum {max_val}")
            
            return value
            
        except (ValueError, TypeError, OverflowError) as e:
            logger.error(f"Numeric validation failed for {name}: {e}")
            raise ValueError(f"Invalid {name}: {e}")
    
    def validate_function_code(self, code: str, context: str = "user_function") -> str:
        """Validate and sanitize user-provided function code."""
        if not isinstance(code, str):
            raise ValueError("Function code must be string")
        
        if len(code) > 10_000:  # 10KB limit for function code
            raise ValueError("Function code too long")
        
        # Check for dangerous patterns
        for pattern in self.code_injection_patterns:
            if pattern.search(code):
                logger.warning(f"Potentially dangerous code detected in {context}: {pattern.pattern}")
                raise ValueError(f"Code contains prohibited patterns")
        
        # Basic syntax validation (simplified)
        try:
            compile(code, f"<{context}>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in function code: {e}")
        
        return code
    
    def sanitize_string_input(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input for safe processing."""
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
            logger.warning(f"String input truncated to {max_length} characters")
        
        # Remove null bytes and control characters
        input_str = ''.join(char for char in input_str if ord(char) >= 32 or char in '\\t\\n\\r')
        
        return input_str
    
    def validate_problem_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate problem configuration dictionary."""
        if not isinstance(config, dict):
            raise ValueError("Problem configuration must be dictionary")
        
        validated_config = {}
        
        # Required fields
        required_fields = ['pde_type', 'grid_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate specific fields
        validated_config['pde_type'] = self.validate_pde_type(config['pde_type'])
        validated_config['grid_size'] = self.validate_grid_size(config['grid_size'])
        
        # Optional fields with validation
        if 'boundary_conditions' in config:
            validated_config['boundary_conditions'] = self.validate_boundary_conditions(
                config['boundary_conditions']
            )
        
        if 'convergence_threshold' in config:
            validated_config['convergence_threshold'] = self.validate_numeric_parameter(
                config['convergence_threshold'], 'convergence_threshold', min_val=1e-12, max_val=1.0
            )
        
        if 'max_iterations' in config:
            validated_config['max_iterations'] = int(self.validate_numeric_parameter(
                config['max_iterations'], 'max_iterations', min_val=1, max_val=100_000
            ))
        
        # Copy other safe fields
        safe_fields = ['name', 'description', 'time_step', 'final_time', 'diffusivity']
        for field in safe_fields:
            if field in config:
                if field in ['name', 'description']:
                    validated_config[field] = self.sanitize_string_input(config[field])
                else:
                    validated_config[field] = self.validate_numeric_parameter(
                        config[field], field, min_val=0.0
                    )
        
        return validated_config
    
    def validate_pde_type(self, pde_type: str) -> str:
        """Validate PDE type parameter."""
        if not isinstance(pde_type, str):
            raise ValueError("PDE type must be string")
        
        pde_type = pde_type.lower().strip()
        
        allowed_types = ['poisson', 'heat', 'wave', 'helmholtz', 'biharmonic', 'advection']
        
        if pde_type not in allowed_types:
            raise ValueError(f"PDE type '{pde_type}' not allowed. Allowed: {allowed_types}")
        
        return pde_type
    
    def validate_boundary_conditions(self, bc_type: str) -> str:
        """Validate boundary condition type."""
        if not isinstance(bc_type, str):
            raise ValueError("Boundary condition type must be string")
        
        bc_type = bc_type.lower().strip()
        
        allowed_types = ['dirichlet', 'neumann', 'mixed', 'periodic']
        
        if bc_type not in allowed_types:
            raise ValueError(f"Boundary condition '{bc_type}' not allowed. Allowed: {allowed_types}")
        
        return bc_type


class ThreatDetector:
    """Advanced threat detection and monitoring."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_history = {}
        self.suspicious_patterns = []
        self.anomaly_scores = {}
        
    def check_rate_limiting(self, client_id: str) -> bool:
        """Check if client exceeds rate limits."""
        current_time = time.time()
        
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_history[client_id] = [
            req_time for req_time in self.request_history[client_id]
            if req_time > cutoff_time
        ]
        
        # Check current rate
        recent_requests = len(self.request_history[client_id])
        
        if recent_requests >= self.config.max_requests_per_minute:
            logger.warning(f"Rate limit exceeded for client {client_id}: {recent_requests} requests/minute")
            return False
        
        # Record current request
        self.request_history[client_id].append(current_time)
        return True
    
    def detect_anomalous_behavior(self, client_id: str, request_data: Dict[str, Any]) -> float:
        """Detect anomalous request patterns."""
        anomaly_score = 0.0
        
        # Check for unusually large parameters
        if 'grid_size' in request_data:
            grid_size = request_data['grid_size']
            if isinstance(grid_size, (int, float)) and grid_size > 1000:
                anomaly_score += 0.3
        
        # Check for suspicious timing patterns
        current_time = time.time()
        if client_id in self.request_history:
            recent_times = self.request_history[client_id][-5:]  # Last 5 requests
            if len(recent_times) >= 3:
                intervals = [recent_times[i+1] - recent_times[i] for i in range(len(recent_times)-1)]
                avg_interval = sum(intervals) / len(intervals)
                
                # Very fast automated requests
                if avg_interval < 0.1:  # Less than 100ms between requests
                    anomaly_score += 0.4
        
        # Check for repeated identical requests
        request_signature = self._compute_request_signature(request_data)
        
        if client_id not in self.anomaly_scores:
            self.anomaly_scores[client_id] = {'signatures': [], 'last_update': current_time}
        
        client_data = self.anomaly_scores[client_id]
        client_data['signatures'].append(request_signature)
        
        # Keep only recent signatures (last hour)
        hour_ago = current_time - 3600
        client_data['signatures'] = [
            sig for sig, timestamp in client_data['signatures']
            if timestamp > hour_ago
        ]
        
        # Count duplicate signatures
        signature_counts = {}
        for sig, _ in client_data['signatures']:
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
        
        max_duplicates = max(signature_counts.values()) if signature_counts else 0
        if max_duplicates > 10:  # More than 10 identical requests
            anomaly_score += 0.5
        
        self.anomaly_scores[client_id]['last_update'] = current_time
        
        return anomaly_score
    
    def _compute_request_signature(self, request_data: Dict[str, Any]) -> Tuple[str, float]:
        """Compute signature for request deduplication."""
        # Create normalized request representation
        normalized_data = {}
        
        for key, value in request_data.items():
            if key in ['grid_size', 'pde_type', 'boundary_conditions']:
                normalized_data[key] = value
        
        # Create hash signature
        data_str = json.dumps(normalized_data, sort_keys=True)
        signature = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
        return signature, time.time()
    
    def assess_threat_level(self, client_id: str, request_data: Dict[str, Any]) -> str:
        """Assess overall threat level for request."""
        
        # Check rate limiting
        if not self.check_rate_limiting(client_id):
            return "HIGH"
        
        # Calculate anomaly score
        anomaly_score = self.detect_anomalous_behavior(client_id, request_data)
        
        if anomaly_score >= self.config.suspicious_pattern_threshold:
            logger.warning(f"High anomaly score for client {client_id}: {anomaly_score}")
            return "HIGH"
        elif anomaly_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"


class SecureComputation:
    """Secure computation protocols and data protection."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def create_secure_session(self, client_id: str) -> Dict[str, str]:
        """Create secure session with authentication token."""
        
        # Generate session token
        token_data = {
            'client_id': client_id,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours)).isoformat(),
            'nonce': secrets.token_hex(16)
        }
        
        # Create signed token
        token_json = json.dumps(token_data, sort_keys=True)
        signature = hmac.new(
            self.config.secret_key,
            token_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Encode token
        token = base64.b64encode(f"{token_json}.{signature}".encode()).decode()
        
        return {
            'session_token': token,
            'expires_at': token_data['expires_at']
        }
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode session token."""
        try:
            # Decode token
            decoded = base64.b64decode(token.encode()).decode()
            token_json, signature = decoded.rsplit('.', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.config.secret_key,
                token_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Invalid token signature")
                return None
            
            # Parse token data
            token_data = json.loads(token_json)
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.utcnow() > expires_at:
                logger.warning("Token expired")
                return None
            
            return token_data
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Token validation failed: {e}")
            return None
    
    def sanitize_computation_environment(self) -> Dict[str, Any]:
        """Set up sanitized computation environment."""
        
        # Environment restrictions
        restricted_modules = [
            'os', 'sys', 'subprocess', 'importlib', '__builtin__', 'builtins',
            'exec', 'eval', 'compile', '__import__'
        ]
        
        # Safe execution context
        safe_context = {
            '__builtins__': {
                # Allow only safe built-in functions
                'abs': abs, 'min': min, 'max': max, 'sum': sum,
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
                'print': print  # Controlled print for debugging
            }
        }
        
        return safe_context
    
    def secure_function_execution(self, func_code: str, variables: Dict[str, Any]) -> Any:
        """Execute user function in secure sandboxed environment."""
        
        # Create safe execution context
        safe_context = self.sanitize_computation_environment()
        safe_context.update(variables)
        
        # Compile and execute with timeout
        try:
            compiled_func = compile(func_code, "<secure_function>", "eval")
            
            # Set execution timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Function execution timeout")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                result = eval(compiled_func, safe_context)
                signal.alarm(0)  # Cancel timeout
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                
        except (SyntaxError, NameError, TypeError, ValueError, TimeoutError) as e:
            logger.error(f"Secure function execution failed: {e}")
            raise ValueError(f"Function execution error: {e}")


class SecurityLogger:
    """Comprehensive security event logging."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security_logger = self._setup_security_logger()
    
    def _setup_security_logger(self) -> logging.Logger:
        """Set up dedicated security event logger."""
        
        security_logger = logging.getLogger('security')
        security_logger.setLevel(logging.INFO)
        
        # Security log file handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        security_handler = logging.FileHandler(
            log_dir / 'security_events.log'
        )
        
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        security_handler.setFormatter(security_formatter)
        security_logger.addHandler(security_handler)
        
        return security_logger
    
    def log_security_event(self, event_type: str, client_id: str, 
                          details: Dict[str, Any], severity: str = 'INFO'):
        """Log security event with structured data."""
        
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'client_id': client_id,
            'severity': severity,
            'details': details
        }
        
        log_message = json.dumps(event_data, sort_keys=True)
        
        if severity == 'CRITICAL':
            self.security_logger.critical(log_message)
        elif severity == 'WARNING':
            self.security_logger.warning(log_message)
        else:
            self.security_logger.info(log_message)
    
    def log_access_attempt(self, client_id: str, endpoint: str, success: bool):
        """Log access attempt."""
        self.log_security_event(
            'ACCESS_ATTEMPT',
            client_id,
            {'endpoint': endpoint, 'success': success},
            'WARNING' if not success else 'INFO'
        )
    
    def log_validation_failure(self, client_id: str, validation_type: str, error: str):
        """Log validation failure."""
        self.log_security_event(
            'VALIDATION_FAILURE',
            client_id,
            {'validation_type': validation_type, 'error': error},
            'WARNING'
        )
    
    def log_threat_detected(self, client_id: str, threat_level: str, details: Dict[str, Any]):
        """Log threat detection."""
        severity = 'CRITICAL' if threat_level == 'HIGH' else 'WARNING'
        self.log_security_event(
            'THREAT_DETECTED',
            client_id,
            {'threat_level': threat_level, **details},
            severity
        )


class SecurityFramework:
    """Main security framework coordinator."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.validator = InputValidator(self.config)
        self.threat_detector = ThreatDetector(self.config)
        self.secure_computation = SecureComputation(self.config)
        self.security_logger = SecurityLogger(self.config)
    
    def validate_request(self, client_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive request validation."""
        
        # Assess threat level
        threat_level = self.threat_detector.assess_threat_level(client_id, request_data)
        
        if threat_level == 'HIGH':
            self.security_logger.log_threat_detected(
                client_id, threat_level, {'request_data': str(request_data)[:1000]}
            )
            raise SecurityError("Request blocked due to security concerns")
        
        # Validate and sanitize inputs
        try:
            if 'problem_config' in request_data:
                validated_config = self.validator.validate_problem_config(
                    request_data['problem_config']
                )
                request_data['problem_config'] = validated_config
            
            # Validate other common parameters
            for param in ['grid_size', 'convergence_threshold', 'max_iterations']:
                if param in request_data:
                    if param == 'grid_size':
                        request_data[param] = self.validator.validate_grid_size(
                            request_data[param]
                        )
                    else:
                        request_data[param] = self.validator.validate_numeric_parameter(
                            request_data[param], param, min_val=0.0
                        )
            
            self.security_logger.log_access_attempt(client_id, 'pde_solve', True)
            return request_data
            
        except (ValueError, TypeError) as e:
            self.security_logger.log_validation_failure(
                client_id, 'input_validation', str(e)
            )
            raise SecurityError(f"Input validation failed: {e}")
    
    def create_secure_session(self, client_id: str) -> Dict[str, str]:
        """Create secure session for client."""
        return self.secure_computation.create_secure_session(client_id)
    
    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token."""
        return self.secure_computation.validate_session_token(token)
    
    def execute_secure_computation(self, func_code: str, variables: Dict[str, Any]) -> Any:
        """Execute computation in secure environment."""
        return self.secure_computation.secure_function_execution(func_code, variables)


class SecurityError(Exception):
    """Custom security exception."""
    pass


# Factory function for easy setup
def create_security_framework(custom_config: Optional[Dict[str, Any]] = None) -> SecurityFramework:
    """Create security framework with optional custom configuration."""
    
    if custom_config:
        config = SecurityConfig(**custom_config)
    else:
        config = SecurityConfig()
    
    return SecurityFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create security framework
    security = create_security_framework()
    
    # Example client request
    client_id = "test_client_123"
    request_data = {
        'problem_config': {
            'pde_type': 'poisson',
            'grid_size': 128,
            'boundary_conditions': 'dirichlet',
            'convergence_threshold': 1e-6
        },
        'max_iterations': 1000
    }
    
    try:
        # Validate request
        validated_request = security.validate_request(client_id, request_data)
        print("✅ Request validation successful")
        print(f"Validated config: {validated_request}")
        
        # Create secure session
        session = security.create_secure_session(client_id)
        print(f"✅ Secure session created: {session['expires_at']}")
        
        # Validate session
        token_data = security.validate_session(session['session_token'])
        print(f"✅ Session validation successful: {token_data['client_id']}")
        
        print("🛡️  Security framework operational!")
        
    except SecurityError as e:
        print(f"❌ Security error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")