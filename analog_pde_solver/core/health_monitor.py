"""Health monitoring and system resilience for analog PDE solver."""

import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    solve_time_ms: float
    iterations_used: int
    convergence_error: float
    matrix_condition_number: float
    memory_usage_mb: float
    crossbar_utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_history: List[HealthCheck] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        
    def check_solver_health(self, solver) -> HealthCheck:
        """Check analog solver health."""
        try:
            # Basic initialization check
            if not hasattr(solver, 'crossbar') or solver.crossbar is None:
                return HealthCheck(
                    component="solver",
                    status=HealthStatus.FAILED,
                    message="Crossbar not initialized",
                    timestamp=time.time()
                )
            
            # Check crossbar size
            if solver.crossbar_size <= 0 or solver.crossbar_size > 10000:
                return HealthCheck(
                    component="solver", 
                    status=HealthStatus.CRITICAL,
                    message=f"Invalid crossbar size: {solver.crossbar_size}",
                    timestamp=time.time()
                )
            
            # Check conductance range
            g_min, g_max = solver.conductance_range
            if g_min <= 0 or g_max <= g_min or g_max > 1e-3:
                return HealthCheck(
                    component="solver",
                    status=HealthStatus.WARNING,
                    message=f"Unusual conductance range: {solver.conductance_range}",
                    timestamp=time.time()
                )
            
            return HealthCheck(
                component="solver",
                status=HealthStatus.HEALTHY,
                message="Solver initialized correctly",
                timestamp=time.time(),
                metrics={
                    "crossbar_size": solver.crossbar_size,
                    "conductance_range": solver.conductance_range,
                    "noise_model": solver.noise_model
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error checking solver health: {e}")
            return HealthCheck(
                component="solver",
                status=HealthStatus.FAILED,
                message=f"Health check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_pde_health(self, pde) -> HealthCheck:
        """Check PDE object health."""
        try:
            # Check required attributes
            if not hasattr(pde, 'domain_size'):
                return HealthCheck(
                    component="pde",
                    status=HealthStatus.FAILED,
                    message="Missing domain_size attribute",
                    timestamp=time.time()
                )
            
            # Check domain size validity
            if isinstance(pde.domain_size, tuple):
                if any(s <= 0 for s in pde.domain_size):
                    return HealthCheck(
                        component="pde",
                        status=HealthStatus.CRITICAL,
                        message=f"Invalid domain size: {pde.domain_size}",
                        timestamp=time.time()
                    )
            else:
                if pde.domain_size <= 0:
                    return HealthCheck(
                        component="pde",
                        status=HealthStatus.CRITICAL,
                        message=f"Invalid domain size: {pde.domain_size}",
                        timestamp=time.time()
                    )
            
            # Check boundary conditions
            if hasattr(pde, 'boundary_conditions'):
                valid_bcs = ["dirichlet", "neumann", "periodic"]
                if pde.boundary_conditions.lower() not in valid_bcs:
                    return HealthCheck(
                        component="pde",
                        status=HealthStatus.WARNING,
                        message=f"Unknown boundary condition: {pde.boundary_conditions}",
                        timestamp=time.time()
                    )
            
            return HealthCheck(
                component="pde",
                status=HealthStatus.HEALTHY,
                message="PDE object is valid",
                timestamp=time.time(),
                metrics={
                    "pde_type": type(pde).__name__,
                    "domain_size": pde.domain_size,
                    "boundary_conditions": getattr(pde, 'boundary_conditions', 'unknown')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error checking PDE health: {e}")
            return HealthCheck(
                component="pde",
                status=HealthStatus.FAILED,
                message=f"PDE health check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_solution_health(self, solution: np.ndarray, 
                            convergence_error: float,
                            iterations: int,
                            max_iterations: int) -> HealthCheck:
        """Check solution health and convergence quality."""
        try:
            # Check solution validity
            if not np.all(np.isfinite(solution)):
                return HealthCheck(
                    component="solution",
                    status=HealthStatus.FAILED,
                    message="Solution contains NaN or Inf values",
                    timestamp=time.time()
                )
            
            # Check solution magnitude
            solution_norm = np.linalg.norm(solution)
            if solution_norm > 1e6:
                return HealthCheck(
                    component="solution",
                    status=HealthStatus.CRITICAL,
                    message=f"Solution magnitude too large: {solution_norm:.2e}",
                    timestamp=time.time()
                )
            
            # Check convergence
            status = HealthStatus.HEALTHY
            message = "Solution converged successfully"
            
            if convergence_error > 1e-3:
                status = HealthStatus.WARNING
                message = f"High convergence error: {convergence_error:.2e}"
            elif iterations >= max_iterations * 0.9:
                status = HealthStatus.WARNING  
                message = f"Used {iterations}/{max_iterations} iterations"
            
            return HealthCheck(
                component="solution",
                status=status,
                message=message,
                timestamp=time.time(),
                metrics={
                    "solution_norm": solution_norm,
                    "convergence_error": convergence_error,
                    "iterations_used": iterations,
                    "max_iterations": max_iterations
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error checking solution health: {e}")
            return HealthCheck(
                component="solution",
                status=HealthStatus.FAILED,
                message=f"Solution health check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.performance_history.append(metrics)
        
        # Limit history size
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size//2:]
        
        self.logger.info(f"Performance recorded: {metrics.solve_time_ms:.2f}ms, "
                        f"{metrics.iterations_used} iterations, "
                        f"error: {metrics.convergence_error:.2e}")
    
    def record_health_check(self, health_check: HealthCheck):
        """Record health check result."""
        self.health_history.append(health_check)
        
        # Limit history size
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size//2:]
        
        # Log critical issues
        if health_check.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            self.logger.error(f"Health check {health_check.component}: {health_check.message}")
        elif health_check.status == HealthStatus.WARNING:
            self.logger.warning(f"Health check {health_check.component}: {health_check.message}")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.health_history:
            return {
                "overall_status": HealthStatus.HEALTHY.value,
                "message": "No health checks performed yet",
                "last_check": None,
                "component_status": {}
            }
        
        # Get latest health check for each component
        latest_checks = {}
        for check in reversed(self.health_history):
            if check.component not in latest_checks:
                latest_checks[check.component] = check
        
        # Determine overall status
        statuses = [check.status for check in latest_checks.values()]
        if any(s == HealthStatus.FAILED for s in statuses):
            overall_status = HealthStatus.FAILED
        elif any(s == HealthStatus.CRITICAL for s in statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(s == HealthStatus.WARNING for s in statuses):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "message": f"System health: {overall_status.value}",
            "last_check": max(check.timestamp for check in latest_checks.values()),
            "component_status": {
                comp: {
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp
                }
                for comp, check in latest_checks.items()
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 runs
        
        solve_times = [m.solve_time_ms for m in recent_metrics]
        iterations = [m.iterations_used for m in recent_metrics]
        errors = [m.convergence_error for m in recent_metrics]
        
        return {
            "recent_runs": len(recent_metrics),
            "avg_solve_time_ms": np.mean(solve_times),
            "min_solve_time_ms": np.min(solve_times),
            "max_solve_time_ms": np.max(solve_times),
            "avg_iterations": np.mean(iterations),
            "avg_convergence_error": np.mean(errors),
            "best_convergence_error": np.min(errors),
            "worst_convergence_error": np.max(errors)
        }
    
    def diagnose_convergence_issues(self, 
                                  convergence_history: List[float]) -> List[str]:
        """Diagnose potential convergence issues."""
        issues = []
        
        if len(convergence_history) < 5:
            return ["Insufficient convergence data for analysis"]
        
        # Check for stagnation
        recent_errors = convergence_history[-10:]
        if len(set(f"{e:.6f}" for e in recent_errors)) <= 2:
            issues.append("Solution appears to be stagnating")
        
        # Check for oscillation  
        diffs = np.diff(convergence_history[-20:])
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        if sign_changes > len(diffs) * 0.8:
            issues.append("Solution is oscillating, reduce damping factor")
        
        # Check for divergence
        if len(convergence_history) > 5:
            recent_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
            if recent_trend > 0:
                issues.append("Solution appears to be diverging")
        
        # Check for slow convergence
        if convergence_history[-1] > 1e-2 and len(convergence_history) > 50:
            issues.append("Very slow convergence, consider different solver parameters")
        
        return issues if issues else ["No obvious convergence issues detected"]


class CircuitBreakerMixin:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self):
        self.failure_count = 0
        self.max_failures = 5
        self.recovery_timeout = 60  # seconds
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def call_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        current_time = time.time()
        
        # Check if we should attempt recovery
        if (self.state == "open" and 
            current_time - self.last_failure_time > self.recovery_timeout):
            self.state = "half-open"
            self.logger.info("Circuit breaker transitioning to half-open state")
        
        if self.state == "open":
            raise RuntimeError("Circuit breaker is open - too many recent failures")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("Circuit breaker closed - service recovered")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.max_failures:
                self.state = "open"
                self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e