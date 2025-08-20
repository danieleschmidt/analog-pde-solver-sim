"""Robust analog PDE solver with comprehensive error handling."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import psutil
import threading
from .crossbar import AnalogCrossbarArray
from ..utils.validation import (
    validate_crossbar_size, validate_conductance_range,
    validate_noise_model, validate_convergence_params,
    safe_array_operation, validate_matrix_properties
)
from ..utils.logging_config import PerformanceMonitor, get_logger


class RobustAnalogPDESolver:
    """Enhanced analog PDE solver with robust error handling and monitoring."""
    
    def __init__(
        self,
        crossbar_size: int = 128,
        conductance_range: tuple = (1e-9, 1e-6),
        noise_model: str = "realistic"
    ):
        """Initialize robust analog PDE solver.
        
        Args:
            crossbar_size: Size of crossbar array
            conductance_range: Min/max conductance values in Siemens
            noise_model: Noise modeling approach
        """
        # Validate inputs
        self.crossbar_size = validate_crossbar_size(crossbar_size)
        self.conductance_range = validate_conductance_range(conductance_range)
        self.noise_model = validate_noise_model(noise_model)
        
        # Initialize logger
        self.logger = get_logger('robust_solver')
        
        # Create crossbar array with error handling
        try:
            self.crossbar = AnalogCrossbarArray(
                crossbar_size, crossbar_size
            )
            self.logger.info(
                f"Initialized {crossbar_size}x{crossbar_size} crossbar array"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize crossbar: {e}")
            raise
            
        # Solver state
        self.is_programmed = False
        self.last_solution = None
        self.convergence_history = []
        self.performance_stats = {}
        
        # Advanced robustness features
        self.circuit_breaker = CircuitBreaker()
        self.memory_monitor = MemoryMonitor()
        self.security_monitor = SecurityMonitor()
        self.auto_recovery = AutoRecoverySystem()
        
        # Health monitoring
        self.health_status = {
            'solver_healthy': True,
            'memory_healthy': True,
            'crossbar_healthy': True,
            'last_health_check': time.time()
        }
        
    def map_pde_to_crossbar(self, pde) -> Dict[str, Any]:
        """Map PDE discretization matrix to crossbar conductances."""
        with PerformanceMonitor("PDE mapping", self.logger):
            try:
                # Generate finite difference Laplacian matrix
                size = self.crossbar_size
                laplacian = self._create_laplacian_matrix(size)
                
                # Validate matrix properties
                laplacian = validate_matrix_properties(
                    laplacian, "Laplacian matrix"
                )
                
                # Program crossbar with error handling
                self.crossbar.program_conductances(laplacian)
                self.is_programmed = True
                
                self.logger.info(f"Successfully mapped {size}x{size} PDE to crossbar")
                
                return {
                    "matrix_size": size,
                    "conductance_range": self.conductance_range,
                    "programming_success": True,
                    "matrix_condition": float(np.linalg.cond(laplacian))
                }
                
            except Exception as e:
                self.logger.error(f"Failed to map PDE to crossbar: {e}")
                self.is_programmed = False
                return {
                    "matrix_size": size,
                    "conductance_range": self.conductance_range,
                    "programming_success": False,
                    "error": str(e)
                }
    
    def solve(
        self, 
        pde,
        iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> np.ndarray:
        """Solve PDE using analog crossbar computation with robust error handling."""
        # Validate inputs
        iterations, convergence_threshold = validate_convergence_params(
            iterations, convergence_threshold
        )
        
        with PerformanceMonitor("PDE solving", self.logger):
            try:
                # Map PDE to crossbar if not already done
                if not self.is_programmed:
                    config = self.map_pde_to_crossbar(pde)
                    if not config["programming_success"]:
                        raise RuntimeError(f"Failed to program crossbar: {config.get('error')}")
                else:
                    config = {"matrix_size": self.crossbar_size}
                
                # Initialize solution vector
                size = config["matrix_size"]
                phi = self._initialize_solution(size)
                
                # Create source term
                source = self._create_source_term(pde, size)
                
                # Reset convergence tracking
                self.convergence_history = []
                
                # Iterative analog solver with robust error handling
                for i in range(iterations):
                    try:
                        # Analog matrix-vector multiplication with error handling
                        residual = safe_array_operation(
                            self.crossbar.compute_vmm, phi,
                            error_msg="Crossbar computation failed"
                        ) + source
                        
                        # Adaptive Jacobi-style update with damping
                        damping = self._adaptive_damping(i, len(self.convergence_history))
                        phi_new = phi - damping * residual
                        
                        # Apply boundary conditions
                        phi_new = self._apply_boundary_conditions(phi_new, pde)
                        
                        # Check for numerical issues
                        if not np.isfinite(phi_new).all():
                            self.logger.error(f"Numerical instability at iteration {i}")
                            break
                        
                        # Check convergence
                        error = np.linalg.norm(phi_new - phi)
                        self.convergence_history.append(error)
                        phi = phi_new.copy()
                        
                        # Log progress periodically
                        if i % max(1, iterations // 10) == 0:
                            self.logger.debug(f"Iteration {i}: error = {error:.2e}")
                        
                        if error < convergence_threshold:
                            self.logger.info(f"Converged after {i+1} iterations")
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Error in iteration {i}: {e}")
                        if i == 0:
                            raise  # Re-raise if first iteration fails
                        break  # Use best solution so far
                
                else:
                    self.logger.warning(
                        f"Did not converge after {iterations} iterations "
                        f"(final error: {self.convergence_history[-1]:.2e})"
                    )
                
                # Store and return solution
                self.last_solution = phi
                self._update_performance_stats()
                return phi
                
            except Exception as e:
                self.logger.error(f"PDE solving failed: {e}")
                raise
    
    def _create_laplacian_matrix(self, size: int) -> np.ndarray:
        """Create finite difference Laplacian matrix with error handling."""
        try:
            laplacian = np.zeros((size, size), dtype=np.float64)
            
            # Main diagonal (central difference)
            np.fill_diagonal(laplacian, -2.0)
            
            # Off-diagonals (neighbors)
            for i in range(size - 1):
                laplacian[i, i + 1] = 1.0
                laplacian[i + 1, i] = 1.0
            
            # Scale by grid spacing squared
            dx = 1.0 / (size + 1)
            laplacian /= dx**2
            
            return laplacian
            
        except Exception as e:
            self.logger.error(f"Failed to create Laplacian matrix: {e}")
            raise
    
    def _initialize_solution(self, size: int) -> np.ndarray:
        """Initialize solution vector with small random values."""
        np.random.seed(42)  # Reproducible initialization
        phi = np.random.uniform(-0.01, 0.01, size)
        return phi.astype(np.float64)
    
    def _create_source_term(self, pde, size: int) -> np.ndarray:
        """Create source term for PDE."""
        try:
            if hasattr(pde, 'source_function') and pde.source_function:
                x = np.linspace(0, 1, size)
                source = np.array([pde.source_function(xi, 0) for xi in x])
            else:
                source = np.ones(size) * 0.01  # Small default source
            
            return source.astype(np.float64)
        except Exception as e:
            self.logger.warning(f"Failed to create source term: {e}, using default")
            return np.ones(size) * 0.01
    
    def _apply_boundary_conditions(self, phi: np.ndarray, pde) -> np.ndarray:
        """Apply boundary conditions to solution vector."""
        phi_bc = phi.copy()
        
        try:
            # Default Dirichlet boundary conditions
            if hasattr(pde, 'boundary_conditions'):
                if pde.boundary_conditions == "dirichlet":
                    phi_bc[0] = 0.0
                    phi_bc[-1] = 0.0
                elif pde.boundary_conditions == "neumann":
                    # Neumann: zero derivative at boundaries
                    phi_bc[0] = phi_bc[1]
                    phi_bc[-1] = phi_bc[-2]
                elif pde.boundary_conditions == "periodic":
                    # Periodic: phi[0] = phi[-1]
                    avg = (phi_bc[0] + phi_bc[-1]) / 2
                    phi_bc[0] = avg
                    phi_bc[-1] = avg
            else:
                # Default to Dirichlet
                phi_bc[0] = 0.0
                phi_bc[-1] = 0.0
                
        except Exception as e:
            self.logger.warning(f"Error applying boundary conditions: {e}")
            # Fall back to Dirichlet
            phi_bc[0] = 0.0
            phi_bc[-1] = 0.0
        
        return phi_bc
    
    def _adaptive_damping(self, iteration: int, history_length: int) -> float:
        """Compute adaptive damping factor for stability."""
        base_damping = 0.1
        
        # Reduce damping if convergence is slow
        if history_length > 5:
            recent_errors = self.convergence_history[-5:]
            if len(recent_errors) >= 2:
                # If error is not decreasing, reduce damping
                if recent_errors[-1] >= recent_errors[-2] * 0.95:
                    base_damping *= 0.8
        
        # Increase damping early in iteration for stability
        if iteration < 10:
            base_damping *= 0.5
        
        return max(0.01, min(0.3, base_damping))
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        if self.convergence_history:
            self.performance_stats = {
                "iterations": len(self.convergence_history),
                "final_error": self.convergence_history[-1],
                "initial_error": self.convergence_history[0],
                "convergence_rate": self._compute_convergence_rate(),
                "total_operations": len(self.convergence_history) * self.crossbar_size**2
            }
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get information about solver convergence."""
        if not self.convergence_history:
            return {"status": "not_solved"}
        
        return {
            "status": "solved" if len(self.convergence_history) > 0 else "failed",
            "iterations": len(self.convergence_history),
            "final_error": self.convergence_history[-1] if self.convergence_history else None,
            "initial_error": self.convergence_history[0] if self.convergence_history else None,
            "convergence_rate": self._compute_convergence_rate(),
            "performance_stats": self.performance_stats
        }
    
    def _compute_convergence_rate(self) -> Optional[float]:
        """Compute convergence rate from error history."""
        if len(self.convergence_history) < 10:
            return None
        
        # Linear regression on log of errors
        errors = np.array(self.convergence_history[-10:])
        errors = errors[errors > 0]  # Remove any zeros
        
        if len(errors) < 5:
            return None
        
        try:
            log_errors = np.log(errors)
            iterations = np.arange(len(log_errors))
            
            # Linear fit: log(error) = a * iteration + b
            coeffs = np.polyfit(iterations, log_errors, 1)
            rate = -coeffs[0]  # Negative because errors should decrease
            
            return float(rate)
            
        except Exception:
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            "timestamp": np.datetime64('now').item().isoformat(),
            "crossbar_programmed": self.is_programmed,
            "crossbar_size": self.crossbar_size,
            "last_solve_successful": self.last_solution is not None,
            "memory_usage": "unknown"
        }
        
        # Check for memory issues
        try:
            from ..utils.logging_config import MemoryMonitor
            MemoryMonitor.log_memory_usage(self.logger)
            health["memory_check"] = "passed"
        except Exception as e:
            health["memory_check"] = f"failed: {e}"
        
        # Check crossbar integrity
        try:
            test_vector = np.ones(self.crossbar_size)
            result = self.crossbar.compute_vmm(test_vector)
            health["crossbar_check"] = "passed" if np.isfinite(result).all() else "failed"
        except Exception as e:
            health["crossbar_check"] = f"failed: {e}"
        
        return health


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


class MemoryMonitor:
    """Monitor memory usage and prevent OOM conditions."""
    
    def __init__(self, memory_threshold: float = 0.85):
        self.memory_threshold = memory_threshold
        
    def check_memory(self) -> Dict[str, Any]:
        """Check current memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "warning": memory.percent > self.memory_threshold * 100
            }
        except Exception:
            return {"warning": True, "error": "Failed to get memory info"}
    
    def ensure_memory_available(self, required_bytes: int):
        """Ensure sufficient memory is available."""
        memory_info = self.check_memory()
        if memory_info.get("warning", False):
            raise RuntimeError("Insufficient memory available")


class SecurityMonitor:
    """Monitor for security-related issues."""
    
    def __init__(self):
        self.suspicious_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"open\s*\(",
            r"file\s*\(",
        ]
        
    def validate_input(self, data: Any) -> bool:
        """Validate input data for security issues."""
        if isinstance(data, str):
            import re
            for pattern in self.suspicious_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    raise SecurityError(f"Suspicious pattern detected: {pattern}")
        return True


class SecurityError(Exception):
    """Security-related error."""
    pass


class AutoRecoverySystem:
    """Automatic recovery from failures."""
    
    def __init__(self):
        self.recovery_strategies = {
            "memory_error": self._recover_from_memory_error,
            "convergence_failure": self._recover_from_convergence_failure,
            "numerical_instability": self._recover_from_numerical_error
        }
        
    def attempt_recovery(self, error_type: str, context: Dict[str, Any]):
        """Attempt to recover from specified error type."""
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](context)
        return False
    
    def _recover_from_memory_error(self, context: Dict[str, Any]) -> bool:
        """Recover from memory-related errors."""
        # Reduce problem size or precision
        if "crossbar_size" in context:
            context["crossbar_size"] = min(64, context["crossbar_size"] // 2)
            return True
        return False
    
    def _recover_from_convergence_failure(self, context: Dict[str, Any]) -> bool:
        """Recover from convergence failures."""
        # Adjust convergence parameters
        if "threshold" in context:
            context["threshold"] *= 10  # Relax threshold
        if "damping" in context:
            context["damping"] *= 0.5  # Reduce damping
        return True
    
    def _recover_from_numerical_error(self, context: Dict[str, Any]) -> bool:
        """Recover from numerical instability."""
        # Switch to more stable algorithm
        if "solver_type" in context:
            context["solver_type"] = "stable"
        return True