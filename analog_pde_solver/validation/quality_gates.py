"""Automated quality gates for analog PDE solver."""

import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import sys
import os


class QualityLevel(Enum):
    """Quality gate levels."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    level: QualityLevel
    message: str
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['level'] = self.level.value
        return result


class QualityGateRunner:
    """Automated quality gate execution and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_history: List[List[QualityGateResult]] = []
        
    def run_all_gates(self, solver, pde=None, solution=None) -> List[QualityGateResult]:
        """Run all quality gates and return results."""
        results = []
        
        # Core functionality gates
        results.extend(self._run_core_gates(solver))
        
        # PDE-specific gates
        if pde is not None:
            results.extend(self._run_pde_gates(solver, pde))
        
        # Solution quality gates  
        if solution is not None:
            results.extend(self._run_solution_gates(solution))
        
        # Performance gates
        results.extend(self._run_performance_gates(solver))
        
        # Security gates
        results.extend(self._run_security_gates(solver, pde))
        
        # Code quality gates (if in development environment)
        if self._is_development_environment():
            results.extend(self._run_code_quality_gates())
        
        # Store results
        self.results_history.append(results)
        
        # Log summary
        self._log_results_summary(results)
        
        return results
    
    def _run_core_gates(self, solver) -> List[QualityGateResult]:
        """Run core functionality quality gates."""
        results = []
        
        # Gate: Solver initialization
        start_time = time.time()
        try:
            assert hasattr(solver, 'crossbar'), "Solver missing crossbar"
            assert hasattr(solver, 'crossbar_size'), "Solver missing crossbar_size"
            assert solver.crossbar_size > 0, "Invalid crossbar size"
            
            results.append(QualityGateResult(
                gate_name="solver_initialization",
                passed=True,
                level=QualityLevel.CRITICAL,
                message="Solver properly initialized",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="solver_initialization", 
                passed=False,
                level=QualityLevel.CRITICAL,
                message=f"Solver initialization failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        # Gate: Matrix creation
        start_time = time.time()
        try:
            laplacian = solver._create_laplacian_matrix(4)
            assert laplacian.shape == (4, 4), "Wrong matrix shape"
            assert np.all(np.isfinite(laplacian)), "Matrix contains non-finite values"
            condition_num = np.linalg.cond(laplacian)
            
            results.append(QualityGateResult(
                gate_name="matrix_creation",
                passed=True,
                level=QualityLevel.HIGH,
                message="Matrix creation working correctly",
                score=1.0 / max(condition_num, 1.0),  # Lower condition number = better score
                details={"condition_number": condition_num},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="matrix_creation",
                passed=False,
                level=QualityLevel.HIGH,
                message=f"Matrix creation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return results
    
    def _run_pde_gates(self, solver, pde) -> List[QualityGateResult]:
        """Run PDE-specific quality gates."""
        results = []
        
        # Gate: PDE validation
        start_time = time.time()
        try:
            assert hasattr(pde, 'domain_size'), "PDE missing domain_size"
            
            domain_size = pde.domain_size[0] if isinstance(pde.domain_size, tuple) else pde.domain_size
            assert domain_size > 0, "Invalid domain size"
            
            results.append(QualityGateResult(
                gate_name="pde_validation",
                passed=True,
                level=QualityLevel.HIGH,
                message="PDE object is valid",
                details={"pde_type": type(pde).__name__, "domain_size": pde.domain_size},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="pde_validation",
                passed=False,
                level=QualityLevel.HIGH,
                message=f"PDE validation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        # Gate: PDE-to-crossbar mapping
        start_time = time.time()
        try:
            config = solver.map_pde_to_crossbar(pde)
            assert config["programming_success"], "PDE mapping failed"
            assert "matrix_condition_number" in config, "Missing condition number"
            
            condition_score = 1.0 / max(config["matrix_condition_number"], 1.0)
            
            results.append(QualityGateResult(
                gate_name="pde_crossbar_mapping",
                passed=True,
                level=QualityLevel.HIGH,
                message="PDE successfully mapped to crossbar",
                score=condition_score,
                details=config,
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="pde_crossbar_mapping",
                passed=False,
                level=QualityLevel.HIGH,
                message=f"PDE mapping failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return results
    
    def _run_solution_gates(self, solution: np.ndarray) -> List[QualityGateResult]:
        """Run solution quality gates."""
        results = []
        
        # Gate: Solution validity
        start_time = time.time()
        try:
            assert isinstance(solution, np.ndarray), "Solution must be numpy array"
            assert solution.ndim == 1, "Solution must be 1D array"
            assert np.all(np.isfinite(solution)), "Solution contains non-finite values"
            
            solution_norm = np.linalg.norm(solution)
            assert solution_norm < 1e6, f"Solution norm too large: {solution_norm}"
            
            results.append(QualityGateResult(
                gate_name="solution_validity",
                passed=True,
                level=QualityLevel.CRITICAL,
                message="Solution is numerically valid",
                details={"solution_norm": solution_norm, "solution_shape": solution.shape},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="solution_validity",
                passed=False,
                level=QualityLevel.CRITICAL,
                message=f"Solution validation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        # Gate: Boundary conditions
        start_time = time.time()
        try:
            # Check if boundary conditions are satisfied (assuming Dirichlet BC = 0)
            boundary_error = abs(solution[0]) + abs(solution[-1])
            
            if boundary_error < 1e-10:
                bc_score = 1.0
                message = "Boundary conditions perfectly satisfied"
            elif boundary_error < 1e-6:
                bc_score = 0.8
                message = "Boundary conditions well satisfied"
            else:
                bc_score = 0.5
                message = f"Boundary conditions loosely satisfied (error: {boundary_error:.2e})"
            
            results.append(QualityGateResult(
                gate_name="boundary_conditions",
                passed=boundary_error < 1e-3,
                level=QualityLevel.HIGH,
                message=message,
                score=bc_score,
                details={"boundary_error": boundary_error},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="boundary_conditions",
                passed=False,
                level=QualityLevel.HIGH,
                message=f"Boundary condition check failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return results
    
    def _run_performance_gates(self, solver) -> List[QualityGateResult]:
        """Run performance quality gates."""
        results = []
        
        # Gate: Memory efficiency
        start_time = time.time()
        try:
            # Estimate memory usage
            crossbar_memory = solver.crossbar_size**2 * 8  # 8 bytes per float64
            estimated_mb = crossbar_memory / (1024**2)
            
            if estimated_mb < 1:
                memory_score = 1.0
                message = f"Excellent memory efficiency: {estimated_mb:.2f}MB"
            elif estimated_mb < 10:
                memory_score = 0.8
                message = f"Good memory efficiency: {estimated_mb:.1f}MB" 
            elif estimated_mb < 100:
                memory_score = 0.6
                message = f"Moderate memory usage: {estimated_mb:.1f}MB"
            else:
                memory_score = 0.3
                message = f"High memory usage: {estimated_mb:.1f}MB"
            
            results.append(QualityGateResult(
                gate_name="memory_efficiency",
                passed=estimated_mb < 500,  # 500MB limit
                level=QualityLevel.MEDIUM,
                message=message,
                score=memory_score,
                details={"estimated_memory_mb": estimated_mb},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="memory_efficiency",
                passed=False,
                level=QualityLevel.MEDIUM,
                message=f"Memory check failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        # Gate: Computational complexity
        start_time = time.time()
        try:
            # Rough complexity estimate
            complexity = solver.crossbar_size**2
            
            if complexity < 1000:
                complexity_score = 1.0
                message = f"Low computational complexity: {complexity}"
            elif complexity < 10000:
                complexity_score = 0.8
                message = f"Moderate computational complexity: {complexity}"
            elif complexity < 100000:
                complexity_score = 0.6
                message = f"High computational complexity: {complexity}"
            else:
                complexity_score = 0.3
                message = f"Very high computational complexity: {complexity}"
            
            results.append(QualityGateResult(
                gate_name="computational_complexity",
                passed=complexity < 1000000,
                level=QualityLevel.LOW,
                message=message,
                score=complexity_score,
                details={"complexity_estimate": complexity},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="computational_complexity",
                passed=False,
                level=QualityLevel.LOW,
                message=f"Complexity check failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return results
    
    def _run_security_gates(self, solver, pde) -> List[QualityGateResult]:
        """Run security quality gates."""
        results = []
        
        # Gate: Input validation
        start_time = time.time()
        try:
            # Check solver parameters are within safe ranges
            assert 0 < solver.crossbar_size <= 10000, "Crossbar size out of safe range"
            
            g_min, g_max = solver.conductance_range
            assert 0 < g_min < g_max < 1e-2, "Conductance range potentially unsafe"
            
            results.append(QualityGateResult(
                gate_name="input_validation",
                passed=True,
                level=QualityLevel.HIGH,
                message="All inputs within safe ranges",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="input_validation",
                passed=False,
                level=QualityLevel.HIGH,
                message=f"Input validation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        # Gate: Resource limits
        start_time = time.time()
        try:
            # Check memory and computation limits
            estimated_ops = solver.crossbar_size**3  # Rough estimate
            
            if estimated_ops > 1e9:
                results.append(QualityGateResult(
                    gate_name="resource_limits",
                    passed=False,
                    level=QualityLevel.MEDIUM,
                    message=f"Resource usage too high: {estimated_ops:.2e} operations",
                    details={"estimated_operations": estimated_ops},
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
            else:
                results.append(QualityGateResult(
                    gate_name="resource_limits",
                    passed=True,
                    level=QualityLevel.MEDIUM,
                    message="Resource usage within limits",
                    details={"estimated_operations": estimated_ops},
                    execution_time_ms=(time.time() - start_time) * 1000
                ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="resource_limits",
                passed=False,
                level=QualityLevel.MEDIUM,
                message=f"Resource check failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return results
    
    def _run_code_quality_gates(self) -> List[QualityGateResult]:
        """Run code quality gates (in development environment)."""
        results = []
        
        # Gate: Import check
        start_time = time.time()
        try:
            import analog_pde_solver
            results.append(QualityGateResult(
                gate_name="import_check",
                passed=True,
                level=QualityLevel.LOW,
                message="Package imports successfully",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        except Exception as e:
            results.append(QualityGateResult(
                gate_name="import_check",
                passed=False,
                level=QualityLevel.LOW,
                message=f"Import failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return results
    
    def _is_development_environment(self) -> bool:
        """Check if running in development environment."""
        # Simple check - look for common dev indicators
        dev_indicators = [
            os.path.exists('.git'),
            os.path.exists('setup.py'),
            os.path.exists('pyproject.toml'),
            'dev' in os.getcwd().lower(),
            'test' in os.getcwd().lower()
        ]
        return any(dev_indicators)
    
    def _log_results_summary(self, results: List[QualityGateResult]):
        """Log summary of quality gate results."""
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.passed)
        
        # Count by level
        level_counts = {}
        failed_critical = []
        
        for result in results:
            level = result.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
            
            if not result.passed and result.level == QualityLevel.CRITICAL:
                failed_critical.append(result.gate_name)
        
        self.logger.info(f"Quality Gates: {passed_gates}/{total_gates} passed")
        self.logger.info(f"Gate distribution: {level_counts}")
        
        if failed_critical:
            self.logger.error(f"CRITICAL gates failed: {failed_critical}")
        
        # Calculate overall score
        total_score = 0
        scored_gates = 0
        for result in results:
            if result.score is not None:
                total_score += result.score
                scored_gates += 1
        
        if scored_gates > 0:
            avg_score = total_score / scored_gates
            self.logger.info(f"Average quality score: {avg_score:.3f}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall quality summary."""
        if not self.results_history:
            return {"message": "No quality gate results available"}
        
        latest_results = self.results_history[-1]
        
        total_gates = len(latest_results)
        passed_gates = sum(1 for r in latest_results if r.passed)
        
        # Calculate scores by level
        level_scores = {}
        for level in QualityLevel:
            level_results = [r for r in latest_results if r.level == level and r.passed]
            level_total = len([r for r in latest_results if r.level == level])
            if level_total > 0:
                level_scores[level.value] = len(level_results) / level_total
        
        # Overall quality assessment
        if passed_gates == total_gates:
            quality_status = "excellent"
        elif passed_gates / total_gates >= 0.8:
            quality_status = "good"
        elif passed_gates / total_gates >= 0.6:
            quality_status = "fair"
        else:
            quality_status = "poor"
        
        return {
            "overall_status": quality_status,
            "gates_passed": passed_gates,
            "total_gates": total_gates,
            "pass_rate": passed_gates / total_gates,
            "level_scores": level_scores,
            "last_run": time.time(),
            "critical_failures": [
                r.gate_name for r in latest_results 
                if not r.passed and r.level == QualityLevel.CRITICAL
            ]
        }


def run_comprehensive_quality_check(solver, pde=None, solution=None) -> Dict[str, Any]:
    """Run comprehensive quality check and return detailed report."""
    runner = QualityGateRunner()
    
    start_time = time.time()
    results = runner.run_all_gates(solver, pde, solution)
    total_time = (time.time() - start_time) * 1000
    
    summary = runner.get_quality_summary()
    
    return {
        "summary": summary,
        "detailed_results": [r.to_dict() for r in results],
        "execution_time_ms": total_time,
        "timestamp": time.time()
    }