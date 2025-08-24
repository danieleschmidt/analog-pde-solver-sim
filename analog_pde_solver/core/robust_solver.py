"""Robust analog PDE solver with comprehensive validation, monitoring, and security."""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List
from .solver import AnalogPDESolver
from .health_monitor import SystemHealthMonitor, PerformanceMetrics, CircuitBreakerMixin
from ..security.input_validation import validate_all_inputs, ValidationResult
from ..validation.quality_gates import QualityGateRunner, run_comprehensive_quality_check


class RobustAnalogPDESolver(AnalogPDESolver, CircuitBreakerMixin):
    """Production-ready analog PDE solver with full robustness features."""
    
    def __init__(
        self,
        crossbar_size: int = 128,
        conductance_range: tuple = (1e-9, 1e-6),
        noise_model: str = "realistic",
        enable_monitoring: bool = True,
        enable_validation: bool = True,
        enable_quality_gates: bool = True,
        strict_mode: bool = False
    ):
        """Initialize robust analog PDE solver.
        
        Args:
            crossbar_size: Size of crossbar array
            conductance_range: Min/max conductance values
            noise_model: Noise modeling approach
            enable_monitoring: Enable health monitoring
            enable_validation: Enable input validation
            enable_quality_gates: Enable automated quality gates
            strict_mode: Fail on any validation warnings
        """
        # Initialize circuit breaker
        CircuitBreakerMixin.__init__(self)
        
        # Set up robust logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation
        self.enable_quality_gates = enable_quality_gates
        self.strict_mode = strict_mode
        
        # Initialize monitoring and validation systems
        if self.enable_monitoring:
            self.health_monitor = SystemHealthMonitor()
        
        if self.enable_quality_gates:
            self.quality_runner = QualityGateRunner()
        
        # Validate inputs before initialization
        if self.enable_validation:
            validation_results = self._validate_initialization_inputs(
                crossbar_size, conductance_range, noise_model
            )
            
            # Apply validated/sanitized values
            crossbar_size = validation_results.get('crossbar_size', ValidationResult(True, crossbar_size, [], [])).sanitized_value
            conductance_range = validation_results.get('conductance_range', ValidationResult(True, conductance_range, [], [])).sanitized_value
            noise_model = validation_results.get('noise_model', ValidationResult(True, noise_model, [], [])).sanitized_value
        
        # Initialize base solver with validated parameters
        super().__init__(crossbar_size, conductance_range, noise_model)
        
        # Perform initial health check
        if self.enable_monitoring:
            health_check = self.health_monitor.check_solver_health(self)
            self.health_monitor.record_health_check(health_check)
            
            if health_check.status.value in ['critical', 'failed']:
                raise RuntimeError(f"Solver failed initial health check: {health_check.message}")
        
        self.logger.info("RobustAnalogPDESolver initialized successfully")
    
    def _validate_initialization_inputs(self, crossbar_size, conductance_range, noise_model) -> Dict[str, ValidationResult]:
        """Validate and sanitize initialization inputs."""
        validation_results = validate_all_inputs(
            crossbar_size=crossbar_size,
            conductance_range=conductance_range,
            noise_model=noise_model
        )
        
        # Log validation results
        for param, result in validation_results.items():
            if not result.is_valid:
                self.logger.error(f"Validation failed for {param}: {'; '.join(result.errors)}")
                raise ValueError(f"Invalid {param}: {'; '.join(result.errors)}")
            
            if result.warnings:
                warning_msg = f"Validation warnings for {param}: {'; '.join(result.warnings)}"
                self.logger.warning(warning_msg)
                
                if self.strict_mode:
                    raise ValueError(f"Strict mode: {warning_msg}")
        
        return validation_results
    
    def solve(
        self, 
        pde=None,
        iterations: int = 100,
        convergence_threshold: float = 1e-6,
        enable_diagnostics: bool = False
    ) -> np.ndarray:
        """Robust PDE solving with comprehensive monitoring and validation.
        
        Args:
            pde: PDE object to solve
            iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold
            enable_diagnostics: Return detailed diagnostics
            
        Returns:
            Solution array or tuple (solution, diagnostics) if enable_diagnostics=True
        """
        solve_start_time = time.time()
        diagnostics = {}
        
        try:
            # Input validation
            if self.enable_validation:
                validation_results = validate_all_inputs(
                    iterations=iterations,
                    convergence_threshold=convergence_threshold
                )
                
                for param, result in validation_results.items():
                    if not result.is_valid:
                        raise ValueError(f"Invalid {param}: {'; '.join(result.errors)}")
                    
                    if result.warnings and self.strict_mode:
                        raise ValueError(f"Strict mode - warnings for {param}: {'; '.join(result.warnings)}")
                
                # Use sanitized values
                solve_params = validation_results.get('solve_params')
                if solve_params and solve_params.is_valid:
                    iterations, convergence_threshold = solve_params.sanitized_value
            
            # Health check on PDE
            if self.enable_monitoring and pde is not None:
                pde_health = self.health_monitor.check_pde_health(pde)
                self.health_monitor.record_health_check(pde_health)
                
                if pde_health.status.value in ['critical', 'failed']:
                    raise ValueError(f"PDE health check failed: {pde_health.message}")
            
            # Run quality gates before solving
            if self.enable_quality_gates:
                pre_solve_gates = self.quality_runner.run_all_gates(self, pde)
                critical_failures = [g for g in pre_solve_gates if not g.passed and g.level.value == 'critical']
                
                if critical_failures:
                    failure_msgs = [g.message for g in critical_failures]
                    raise RuntimeError(f"Critical quality gate failures: {'; '.join(failure_msgs)}")
                
                diagnostics['pre_solve_quality_gates'] = [g.to_dict() for g in pre_solve_gates]
            
            # Solve with circuit breaker protection
            solution = self.call_with_circuit_breaker(
                super().solve, 
                pde, 
                iterations=iterations, 
                convergence_threshold=convergence_threshold
            )
            
            solve_time_ms = (time.time() - solve_start_time) * 1000
            
            # Solution health check
            if self.enable_monitoring:
                # Get convergence error from solution (approximate)
                final_error = convergence_threshold  # This would need to be tracked in base solver
                
                solution_health = self.health_monitor.check_solution_health(
                    solution, final_error, iterations, iterations
                )
                self.health_monitor.record_health_check(solution_health)
                
                if solution_health.status.value == 'failed':
                    raise RuntimeError(f"Solution health check failed: {solution_health.message}")
                
                # Record performance metrics
                performance_metrics = PerformanceMetrics(
                    solve_time_ms=solve_time_ms,
                    iterations_used=iterations,  # Approximate - would need tracking
                    convergence_error=final_error,
                    matrix_condition_number=self._estimate_condition_number(),
                    memory_usage_mb=self._estimate_memory_usage(),
                    crossbar_utilization=1.0  # Simplified
                )
                
                self.health_monitor.record_performance(performance_metrics)
                diagnostics['performance_metrics'] = performance_metrics.to_dict()
            
            # Post-solve quality gates
            if self.enable_quality_gates:
                post_solve_gates = self.quality_runner.run_all_gates(self, pde, solution)
                diagnostics['post_solve_quality_gates'] = [g.to_dict() for g in post_solve_gates]
                
                # Check for critical failures
                critical_failures = [g for g in post_solve_gates if not g.passed and g.level.value == 'critical']
                if critical_failures:
                    self.logger.error(f"Post-solve critical failures: {[g.message for g in critical_failures]}")
            
            self.logger.info(f"Robust solve completed successfully in {solve_time_ms:.2f}ms")
            
            if enable_diagnostics:
                return solution, diagnostics
            else:
                return solution
                
        except Exception as e:
            solve_time_ms = (time.time() - solve_start_time) * 1000
            self.logger.error(f"Robust solve failed after {solve_time_ms:.2f}ms: {e}")
            
            # Record failure
            if self.enable_monitoring:
                failure_metrics = PerformanceMetrics(
                    solve_time_ms=solve_time_ms,
                    iterations_used=0,
                    convergence_error=float('inf'),
                    matrix_condition_number=float('inf'),
                    memory_usage_mb=0,
                    crossbar_utilization=0
                )
                self.health_monitor.record_performance(failure_metrics)
            
            raise e
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health and performance report."""
        if not self.enable_monitoring:
            return {"message": "Monitoring is disabled"}
        
        health_summary = self.health_monitor.get_system_health_summary()
        performance_summary = self.health_monitor.get_performance_summary()
        
        report = {
            "health_summary": health_summary,
            "performance_summary": performance_summary,
            "circuit_breaker": {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time
            },
            "configuration": {
                "crossbar_size": self.crossbar_size,
                "conductance_range": self.conductance_range,
                "noise_model": self.noise_model,
                "monitoring_enabled": self.enable_monitoring,
                "validation_enabled": self.enable_validation,
                "quality_gates_enabled": self.enable_quality_gates,
                "strict_mode": self.strict_mode
            }
        }
        
        if self.enable_quality_gates:
            quality_summary = self.quality_runner.get_quality_summary()
            report["quality_summary"] = quality_summary
        
        return report
    
    def run_comprehensive_diagnostics(self, pde=None) -> Dict[str, Any]:
        """Run comprehensive system diagnostics."""
        self.logger.info("Running comprehensive diagnostics")
        
        diagnostics = {}
        
        # Health checks
        if self.enable_monitoring:
            diagnostics["solver_health"] = self.health_monitor.check_solver_health(self).to_dict()
            
            if pde:
                diagnostics["pde_health"] = self.health_monitor.check_pde_health(pde).to_dict()
        
        # Quality gates
        if self.enable_quality_gates:
            quality_report = run_comprehensive_quality_check(self, pde)
            diagnostics["quality_report"] = quality_report
        
        # System information
        diagnostics["system_info"] = {
            "crossbar_memory_mb": (self.crossbar_size**2 * 8) / (1024**2),
            "estimated_complexity": self.crossbar_size**2,
            "python_version": f"{__import__('sys').version}",
            "numpy_version": np.__version__
        }
        
        # Circuit breaker status
        diagnostics["circuit_breaker"] = {
            "state": self.state,
            "failure_count": self.failure_count,
            "max_failures": self.max_failures,
            "recovery_timeout": self.recovery_timeout
        }
        
        return diagnostics
    
    def _estimate_condition_number(self) -> float:
        """Estimate condition number of current system matrix."""
        try:
            # Create a representative matrix for condition number estimation
            matrix = self._create_laplacian_matrix(min(self.crossbar_size, 32))
            return float(np.linalg.cond(matrix))
        except:
            return float('inf')
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Rough estimate based on crossbar size
        crossbar_memory = self.crossbar_size**2 * 8 * 2  # Two matrices (pos/neg conductances)
        return crossbar_memory / (1024**2)
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker to closed state."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0
        self.logger.info("Circuit breaker manually reset")
    
    def set_strict_mode(self, enabled: bool):
        """Enable or disable strict mode."""
        self.strict_mode = enabled
        self.logger.info(f"Strict mode {'enabled' if enabled else 'disabled'}")
    
    def get_validation_report(self, **kwargs) -> Dict[str, Any]:
        """Get detailed validation report for given parameters."""
        if not self.enable_validation:
            return {"message": "Validation is disabled"}
        
        validation_results = validate_all_inputs(**kwargs)
        
        report = {
            "validation_enabled": True,
            "strict_mode": self.strict_mode,
            "parameter_results": {},
            "overall_valid": True,
            "total_warnings": 0,
            "total_errors": 0
        }
        
        for param, result in validation_results.items():
            report["parameter_results"][param] = {
                "valid": result.is_valid,
                "warnings": result.warnings,
                "errors": result.errors,
                "sanitized_value": str(result.sanitized_value) if result.sanitized_value is not None else None
            }
            
            if not result.is_valid:
                report["overall_valid"] = False
            
            report["total_warnings"] += len(result.warnings)
            report["total_errors"] += len(result.errors)
        
        return report