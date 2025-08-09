"""Advanced optimization algorithms for analog PDE solving."""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Advanced algorithm types for optimization."""
    MULTIGRID = "multigrid"
    ADAPTIVE_MESH_REFINEMENT = "amr"
    PRECONDITIONING = "preconditioning"
    KRYLOV_SUBSPACE = "krylov"
    DOMAIN_DECOMPOSITION = "domain_decomp"
    NEURAL_ACCELERATION = "neural_accel"


@dataclass 
class OptimizationResult:
    """Result from advanced optimization algorithm."""
    solution: np.ndarray
    convergence_rate: float
    iterations: int
    computation_time: float
    memory_usage: float
    algorithm_used: AlgorithmType
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]


class MultiGridSolver:
    """Multigrid solver for efficient hierarchical PDE solving."""
    
    def __init__(
        self,
        levels: int = 4,
        smoother: str = "jacobi",
        coarsening_factor: int = 2,
        max_iterations_per_level: int = 10
    ):
        """Initialize multigrid solver.
        
        Args:
            levels: Number of grid levels
            smoother: Smoothing method (jacobi, gauss_seidel, sor)
            coarsening_factor: Grid coarsening ratio
            max_iterations_per_level: Max smoothing iterations per level
        """
        self.levels = levels
        self.smoother = smoother
        self.coarsening_factor = coarsening_factor
        self.max_iterations = max_iterations_per_level
        self.logger = logging.getLogger(__name__)
        
    def solve(
        self,
        crossbar_hierarchy: List[Any],
        initial_solution: np.ndarray,
        rhs: np.ndarray,
        tolerance: float = 1e-6
    ) -> OptimizationResult:
        """Solve using V-cycle multigrid algorithm.
        
        Args:
            crossbar_hierarchy: List of crossbar arrays for each level
            initial_solution: Initial guess
            rhs: Right-hand side vector
            tolerance: Convergence tolerance
            
        Returns:
            Optimization result with solution and metrics
        """
        start_time = time.perf_counter()
        
        # Initialize grid hierarchy
        solutions = [initial_solution]
        rhs_vectors = [rhs]
        
        # Create coarser levels
        for level in range(1, self.levels):
            coarse_size = len(solutions[-1]) // self.coarsening_factor
            coarse_solution = self._restrict(solutions[-1], coarse_size)
            coarse_rhs = self._restrict(rhs_vectors[-1], coarse_size)
            
            solutions.append(coarse_solution)
            rhs_vectors.append(coarse_rhs)
        
        residuals = []
        v_cycles = 0
        
        # V-cycle iterations
        while v_cycles < 50:  # Max V-cycles
            old_solution = solutions[0].copy()
            
            # V-cycle down (restriction)
            for level in range(self.levels - 1):
                solutions[level] = self._smooth(
                    crossbar_hierarchy[level],
                    solutions[level],
                    rhs_vectors[level],
                    self.max_iterations
                )
                
                # Compute residual and restrict to coarser level
                residual = self._compute_residual(
                    crossbar_hierarchy[level],
                    solutions[level],
                    rhs_vectors[level]
                )
                
                if level + 1 < len(rhs_vectors):
                    rhs_vectors[level + 1] = self._restrict(residual, len(solutions[level + 1]))
            
            # Coarsest level solve
            solutions[-1] = self._direct_solve(
                crossbar_hierarchy[-1] if len(crossbar_hierarchy) >= self.levels else None,
                solutions[-1],
                rhs_vectors[-1]
            )
            
            # V-cycle up (prolongation and correction)
            for level in range(self.levels - 2, -1, -1):
                # Prolongate correction from coarser level
                correction = self._prolongate(solutions[level + 1], len(solutions[level]))
                solutions[level] += correction
                
                # Post-smoothing
                solutions[level] = self._smooth(
                    crossbar_hierarchy[level],
                    solutions[level],
                    rhs_vectors[level],
                    self.max_iterations // 2
                )
            
            # Check convergence
            residual_norm = np.linalg.norm(solutions[0] - old_solution)
            residuals.append(residual_norm)
            v_cycles += 1
            
            self.logger.debug(f"V-cycle {v_cycles}: residual = {residual_norm:.2e}")
            
            if residual_norm < tolerance:
                break
        
        computation_time = time.perf_counter() - start_time
        
        # Calculate convergence rate
        if len(residuals) > 1:
            convergence_rate = np.log(residuals[-1] / residuals[0]) / len(residuals)
        else:
            convergence_rate = 0.0
        
        return OptimizationResult(
            solution=solutions[0],
            convergence_rate=abs(convergence_rate),
            iterations=v_cycles,
            computation_time=computation_time,
            memory_usage=sum(sol.nbytes for sol in solutions) / 1024**2,  # MB
            algorithm_used=AlgorithmType.MULTIGRID,
            accuracy_metrics={"final_residual": residuals[-1] if residuals else 0.0},
            performance_metrics={
                "v_cycles": v_cycles,
                "avg_time_per_cycle": computation_time / max(1, v_cycles)
            }
        )
    
    def _restrict(self, fine_vector: np.ndarray, coarse_size: int) -> np.ndarray:
        """Restrict fine grid vector to coarse grid."""
        if coarse_size >= len(fine_vector):
            return fine_vector.copy()
        
        # Simple injection for now (could use full weighting)
        stride = len(fine_vector) // coarse_size
        return fine_vector[::stride][:coarse_size]
    
    def _prolongate(self, coarse_vector: np.ndarray, fine_size: int) -> np.ndarray:
        """Prolongate coarse grid vector to fine grid."""
        if fine_size <= len(coarse_vector):
            return coarse_vector[:fine_size]
        
        # Linear interpolation
        fine_vector = np.zeros(fine_size)
        ratio = len(coarse_vector) / fine_size
        
        for i in range(fine_size):
            coarse_idx = i * ratio
            idx_low = int(np.floor(coarse_idx))
            idx_high = min(idx_low + 1, len(coarse_vector) - 1)
            
            if idx_low == idx_high:
                fine_vector[i] = coarse_vector[idx_low]
            else:
                weight = coarse_idx - idx_low
                fine_vector[i] = (1 - weight) * coarse_vector[idx_low] + weight * coarse_vector[idx_high]
        
        return fine_vector
    
    def _smooth(
        self,
        crossbar: Any,
        solution: np.ndarray,
        rhs: np.ndarray,
        iterations: int
    ) -> np.ndarray:
        """Apply smoothing iterations."""
        current_solution = solution.copy()
        
        for _ in range(iterations):
            if self.smoother == "jacobi":
                current_solution = self._jacobi_step(crossbar, current_solution, rhs)
            elif self.smoother == "gauss_seidel":
                current_solution = self._gauss_seidel_step(crossbar, current_solution, rhs)
            else:
                # Default to simple damped iteration
                residual = self._compute_residual(crossbar, current_solution, rhs)
                current_solution -= 0.1 * residual
        
        return current_solution
    
    def _jacobi_step(self, crossbar: Any, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Single Jacobi smoothing step."""
        if crossbar is None:
            return solution
        
        # Compute residual using crossbar
        residual = crossbar.compute_vmm(solution) - rhs
        
        # Jacobi update with damping
        return solution - 0.2 * residual  # 0.2 is damping factor
    
    def _gauss_seidel_step(self, crossbar: Any, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Single Gauss-Seidel smoothing step."""
        if crossbar is None:
            return solution
        
        # Simplified Gauss-Seidel (would need diagonal extraction for true GS)
        residual = crossbar.compute_vmm(solution) - rhs
        return solution - 0.15 * residual
    
    def _compute_residual(self, crossbar: Any, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Compute residual r = Ax - b."""
        if crossbar is None:
            return np.zeros_like(solution)
        
        return crossbar.compute_vmm(solution) - rhs
    
    def _direct_solve(self, crossbar: Any, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Direct solve on coarsest level."""
        if crossbar is None or len(solution) > 10:
            # Too large for direct solve, use iterative
            return self._smooth(crossbar, solution, rhs, 20)
        
        # For very small systems, could use direct methods
        return self._smooth(crossbar, solution, rhs, 50)


class AdaptiveMeshRefinement:
    """Adaptive mesh refinement for analog PDE solving."""
    
    def __init__(
        self,
        refinement_threshold: float = 0.1,
        coarsening_threshold: float = 0.01,
        max_refinement_levels: int = 3
    ):
        """Initialize adaptive mesh refinement.
        
        Args:
            refinement_threshold: Error threshold for refinement
            coarsening_threshold: Error threshold for coarsening  
            max_refinement_levels: Maximum refinement levels
        """
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.max_levels = max_refinement_levels
        self.logger = logging.getLogger(__name__)
    
    def solve_with_amr(
        self,
        base_solver: Any,
        pde: Any,
        initial_mesh_size: int = 64
    ) -> OptimizationResult:
        """Solve PDE with adaptive mesh refinement.
        
        Args:
            base_solver: Base PDE solver
            pde: PDE problem definition
            initial_mesh_size: Initial mesh size
            
        Returns:
            Optimization result with adapted solution
        """
        start_time = time.perf_counter()
        
        # Start with coarse mesh
        current_mesh_size = initial_mesh_size
        refinement_level = 0
        
        solutions = []
        error_estimates = []
        
        while refinement_level <= self.max_levels:
            self.logger.info(f"AMR level {refinement_level}: mesh size {current_mesh_size}")
            
            # Solve on current mesh
            if hasattr(pde, 'set_domain_size'):
                pde.set_domain_size((current_mesh_size,))
            
            solution = base_solver.solve(pde)
            solutions.append(solution)
            
            # Estimate error
            error_estimate = self._estimate_error(solution, pde)
            error_estimates.append(error_estimate)
            
            self.logger.debug(f"Error estimate at level {refinement_level}: {error_estimate:.2e}")
            
            # Check refinement criteria
            if error_estimate > self.refinement_threshold and refinement_level < self.max_levels:
                # Refine mesh
                current_mesh_size *= 2
                refinement_level += 1
                
                # Update solver for new mesh size
                if hasattr(base_solver, 'crossbar_size'):
                    base_solver.crossbar_size = current_mesh_size
                    base_solver.crossbar = type(base_solver.crossbar)(current_mesh_size, current_mesh_size)
                
            else:
                # Converged or max levels reached
                break
        
        computation_time = time.perf_counter() - start_time
        
        # Calculate convergence rate from error reduction
        if len(error_estimates) > 1:
            convergence_rate = np.log(error_estimates[-1] / error_estimates[0]) / len(error_estimates)
        else:
            convergence_rate = 0.0
        
        return OptimizationResult(
            solution=solutions[-1],
            convergence_rate=abs(convergence_rate),
            iterations=refinement_level + 1,
            computation_time=computation_time,
            memory_usage=sum(sol.nbytes for sol in solutions) / 1024**2,
            algorithm_used=AlgorithmType.ADAPTIVE_MESH_REFINEMENT,
            accuracy_metrics={"final_error_estimate": error_estimates[-1]},
            performance_metrics={
                "refinement_levels": refinement_level,
                "final_mesh_size": current_mesh_size,
                "error_reduction": error_estimates[0] / error_estimates[-1] if len(error_estimates) > 1 else 1.0
            }
        )
    
    def _estimate_error(self, solution: np.ndarray, pde: Any) -> float:
        """Estimate solution error using various indicators."""
        # Gradient-based error estimate
        if solution.ndim == 1:
            gradients = np.gradient(solution)
            second_derivatives = np.gradient(gradients)
            
            # High second derivative indicates need for refinement
            error_indicator = np.std(second_derivatives)
            
        elif solution.ndim == 2:
            # 2D gradient-based estimate
            grad_x, grad_y = np.gradient(solution)
            error_indicator = np.std(grad_x) + np.std(grad_y)
            
        else:
            # Fallback: use solution variation
            error_indicator = np.std(solution)
        
        return error_indicator


class PreconditionedSolver:
    """Preconditioned iterative solver for improved convergence."""
    
    def __init__(
        self,
        preconditioner_type: str = "jacobi",
        tolerance: float = 1e-6,
        max_iterations: int = 1000
    ):
        """Initialize preconditioned solver.
        
        Args:
            preconditioner_type: Type of preconditioner (jacobi, ilu, multigrid)
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        """
        self.preconditioner_type = preconditioner_type
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)
    
    def solve_preconditioned(
        self,
        crossbar: Any,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Solve using preconditioned conjugate gradient method.
        
        Args:
            crossbar: Crossbar array representing system matrix
            rhs: Right-hand side vector
            initial_guess: Initial solution guess
            
        Returns:
            Optimization result with preconditioned solution
        """
        start_time = time.perf_counter()
        
        # Initialize
        n = len(rhs)
        x = initial_guess if initial_guess is not None else np.zeros(n)
        
        # Build preconditioner
        preconditioner = self._build_preconditioner(crossbar)
        
        # Initial residual
        r = rhs - crossbar.compute_vmm(x)
        z = self._apply_preconditioner(preconditioner, r)
        p = z.copy()
        
        residuals = []
        rsold = np.dot(r, z)
        
        for iteration in range(self.max_iterations):
            # CG step
            Ap = crossbar.compute_vmm(p)
            alpha = rsold / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            
            residual_norm = np.linalg.norm(r)
            residuals.append(residual_norm)
            
            if residual_norm < self.tolerance:
                self.logger.info(f"Preconditioned CG converged in {iteration + 1} iterations")
                break
            
            # Update search direction
            z = self._apply_preconditioner(preconditioner, r)
            rsnew = np.dot(r, z)
            beta = rsnew / rsold
            p = z + beta * p
            rsold = rsnew
        
        computation_time = time.perf_counter() - start_time
        
        # Calculate convergence rate
        if len(residuals) > 1:
            convergence_rate = np.log(residuals[-1] / residuals[0]) / len(residuals)
        else:
            convergence_rate = 0.0
        
        return OptimizationResult(
            solution=x,
            convergence_rate=abs(convergence_rate),
            iterations=len(residuals),
            computation_time=computation_time,
            memory_usage=(x.nbytes + len(residuals) * 8) / 1024**2,
            algorithm_used=AlgorithmType.PRECONDITIONING,
            accuracy_metrics={"final_residual": residuals[-1] if residuals else 0.0},
            performance_metrics={
                "preconditioner_type": self.preconditioner_type,
                "condition_improvement": self._estimate_condition_improvement(crossbar, preconditioner)
            }
        )
    
    def _build_preconditioner(self, crossbar: Any) -> Dict[str, Any]:
        """Build preconditioner matrix/operator."""
        if self.preconditioner_type == "jacobi":
            # Diagonal preconditioner
            diagonal = self._extract_diagonal(crossbar)
            return {"type": "jacobi", "diagonal": diagonal}
            
        elif self.preconditioner_type == "block_jacobi":
            # Block diagonal preconditioner
            return {"type": "block_jacobi", "crossbar": crossbar}
            
        else:
            # Identity preconditioner (no preconditioning)
            return {"type": "identity"}
    
    def _apply_preconditioner(self, preconditioner: Dict[str, Any], vector: np.ndarray) -> np.ndarray:
        """Apply preconditioner to vector."""
        if preconditioner["type"] == "jacobi":
            diagonal = preconditioner["diagonal"]
            return vector / (diagonal + 1e-12)  # Avoid division by zero
            
        elif preconditioner["type"] == "block_jacobi":
            # Simplified block application
            return vector * 0.5  # Simple scaling
            
        else:
            # Identity preconditioner
            return vector
    
    def _extract_diagonal(self, crossbar: Any) -> np.ndarray:
        """Extract diagonal elements from crossbar."""
        n = crossbar.rows
        diagonal = np.zeros(n)
        
        # Estimate diagonal by applying unit vectors
        for i in range(n):
            unit_vec = np.zeros(n)
            unit_vec[i] = 1.0
            result = crossbar.compute_vmm(unit_vec)
            diagonal[i] = result[i] if i < len(result) else 1.0
        
        return diagonal
    
    def _estimate_condition_improvement(self, crossbar: Any, preconditioner: Dict[str, Any]) -> float:
        """Estimate condition number improvement from preconditioning."""
        # Simplified estimate - in practice would need eigenvalue analysis
        if preconditioner["type"] == "jacobi":
            return 2.0  # Typical improvement for Jacobi preconditioning
        elif preconditioner["type"] == "block_jacobi":
            return 5.0  # Better improvement for block methods
        else:
            return 1.0  # No improvement


class AdvancedAlgorithmSuite:
    """Suite of advanced optimization algorithms for analog PDE solving."""
    
    def __init__(self):
        """Initialize advanced algorithm suite."""
        self.multigrid = MultiGridSolver()
        self.amr = AdaptiveMeshRefinement()
        self.preconditioned = PreconditionedSolver()
        self.logger = logging.getLogger(__name__)
        
        # Algorithm selection history for learning
        self.algorithm_performance: Dict[str, List[float]] = {}
        
    def auto_select_algorithm(
        self,
        pde_characteristics: Dict[str, Any],
        performance_targets: Dict[str, float]
    ) -> AlgorithmType:
        """Automatically select best algorithm based on problem characteristics.
        
        Args:
            pde_characteristics: PDE problem characteristics
            performance_targets: Desired performance metrics
            
        Returns:
            Recommended algorithm type
        """
        problem_size = pde_characteristics.get("size", 100)
        conditioning = pde_characteristics.get("condition_number", 100)
        sparsity = pde_characteristics.get("sparsity", 0.1)
        geometry_complexity = pde_characteristics.get("geometry_complexity", 0.5)
        
        target_time = performance_targets.get("max_time_seconds", 60.0)
        target_accuracy = performance_targets.get("accuracy", 1e-6)
        
        # Decision logic based on problem characteristics
        if problem_size > 1000 and geometry_complexity < 0.3:
            # Large, simple geometry -> Multigrid
            return AlgorithmType.MULTIGRID
            
        elif geometry_complexity > 0.7 or pde_characteristics.get("has_singularities", False):
            # Complex geometry or singularities -> AMR
            return AlgorithmType.ADAPTIVE_MESH_REFINEMENT
            
        elif conditioning > 1000 or sparsity > 0.8:
            # Ill-conditioned or very sparse -> Preconditioning
            return AlgorithmType.PRECONDITIONING
            
        else:
            # Default to multigrid for most problems
            return AlgorithmType.MULTIGRID
    
    def solve_with_algorithm(
        self,
        algorithm_type: AlgorithmType,
        solver: Any,
        pde: Any,
        **kwargs
    ) -> OptimizationResult:
        """Solve using specified advanced algorithm.
        
        Args:
            algorithm_type: Algorithm to use
            solver: Base solver instance
            pde: PDE problem
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Optimization result
        """
        start_time = time.perf_counter()
        
        try:
            if algorithm_type == AlgorithmType.MULTIGRID:
                # Create crossbar hierarchy
                crossbar_hierarchy = self._create_crossbar_hierarchy(solver, **kwargs)
                initial_solution = np.random.random(solver.crossbar_size) * 0.1
                rhs = np.ones(solver.crossbar_size) * 0.1
                
                result = self.multigrid.solve(crossbar_hierarchy, initial_solution, rhs)
                
            elif algorithm_type == AlgorithmType.ADAPTIVE_MESH_REFINEMENT:
                result = self.amr.solve_with_amr(solver, pde)
                
            elif algorithm_type == AlgorithmType.PRECONDITIONING:
                rhs = np.ones(solver.crossbar_size) * 0.1
                result = self.preconditioned.solve_preconditioned(solver.crossbar, rhs)
                
            else:
                raise ValueError(f"Algorithm type {algorithm_type} not implemented")
            
            # Record performance for future algorithm selection
            self._record_algorithm_performance(algorithm_type, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Algorithm {algorithm_type} failed: {e}")
            
            # Fallback result
            return OptimizationResult(
                solution=np.zeros(solver.crossbar_size),
                convergence_rate=0.0,
                iterations=0,
                computation_time=time.perf_counter() - start_time,
                memory_usage=0.0,
                algorithm_used=algorithm_type,
                accuracy_metrics={"error": "algorithm_failed"},
                performance_metrics={"status": "failed"}
            )
    
    def _create_crossbar_hierarchy(self, solver: Any, **kwargs) -> List[Any]:
        """Create hierarchy of crossbar arrays for multigrid."""
        hierarchy = [solver.crossbar]
        
        levels = kwargs.get("levels", 4)
        current_size = solver.crossbar_size
        
        for level in range(1, levels):
            current_size = max(4, current_size // 2)
            
            # Create smaller crossbar for coarser level
            coarse_crossbar = type(solver.crossbar)(current_size, current_size)
            
            # Initialize with appropriate conductances
            coarse_crossbar.g_positive = np.random.uniform(1e-8, 1e-6, (current_size, current_size))
            coarse_crossbar.g_negative = np.random.uniform(1e-9, 1e-7, (current_size, current_size))
            
            hierarchy.append(coarse_crossbar)
        
        return hierarchy
    
    def _record_algorithm_performance(self, algorithm_type: AlgorithmType, result: OptimizationResult):
        """Record algorithm performance for learning."""
        if algorithm_type.value not in self.algorithm_performance:
            self.algorithm_performance[algorithm_type.value] = []
        
        # Compute performance score (higher is better)
        score = result.convergence_rate / max(0.001, result.computation_time)
        
        self.algorithm_performance[algorithm_type.value].append(score)
        
        # Keep only recent performance data
        if len(self.algorithm_performance[algorithm_type.value]) > 100:
            self.algorithm_performance[algorithm_type.value] = \
                self.algorithm_performance[algorithm_type.value][-50:]
    
    def get_algorithm_recommendations(self, pde_characteristics: Dict[str, Any]) -> List[Tuple[AlgorithmType, float]]:
        """Get ranked algorithm recommendations based on historical performance.
        
        Args:
            pde_characteristics: Problem characteristics
            
        Returns:
            List of (algorithm, confidence_score) tuples, ranked by recommendation
        """
        recommendations = []
        
        for algorithm_name, performance_history in self.algorithm_performance.items():
            if performance_history:
                avg_performance = np.mean(performance_history[-20:])  # Recent performance
                confidence = min(1.0, len(performance_history) / 20.0)  # More data = higher confidence
                
                algorithm_type = AlgorithmType(algorithm_name)
                recommendations.append((algorithm_type, avg_performance * confidence))
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def benchmark_algorithms(
        self,
        solver: Any,
        pde: Any,
        algorithms: Optional[List[AlgorithmType]] = None
    ) -> Dict[AlgorithmType, OptimizationResult]:
        """Benchmark multiple algorithms on the same problem.
        
        Args:
            solver: Base solver instance
            pde: PDE problem
            algorithms: List of algorithms to benchmark (default: all available)
            
        Returns:
            Dictionary mapping algorithms to their results
        """
        if algorithms is None:
            algorithms = [
                AlgorithmType.MULTIGRID,
                AlgorithmType.ADAPTIVE_MESH_REFINEMENT,
                AlgorithmType.PRECONDITIONING
            ]
        
        results = {}
        
        for algorithm in algorithms:
            self.logger.info(f"Benchmarking {algorithm.value}")
            
            try:
                # Create fresh solver instance to avoid state contamination
                fresh_solver = type(solver)(
                    crossbar_size=solver.crossbar_size,
                    conductance_range=solver.conductance_range,
                    noise_model=solver.noise_model
                )
                
                result = self.solve_with_algorithm(algorithm, fresh_solver, pde)
                results[algorithm] = result
                
                self.logger.info(f"{algorithm.value}: {result.iterations} iterations, "
                               f"{result.computation_time:.3f}s, "
                               f"convergence rate: {result.convergence_rate:.3f}")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {algorithm.value}: {e}")
                
                # Create failure result
                results[algorithm] = OptimizationResult(
                    solution=np.zeros(solver.crossbar_size),
                    convergence_rate=0.0,
                    iterations=0,
                    computation_time=float('inf'),
                    memory_usage=0.0,
                    algorithm_used=algorithm,
                    accuracy_metrics={"status": "failed"},
                    performance_metrics={"error": str(e)}
                )
        
        return results
    
    def generate_algorithm_report(self, results: Dict[AlgorithmType, OptimizationResult]) -> str:
        """Generate comprehensive algorithm benchmark report.
        
        Args:
            results: Algorithm benchmark results
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "ADVANCED ALGORITHM BENCHMARK REPORT",
            "=" * 60,
            ""
        ]
        
        # Sort results by performance (fastest convergence with lowest time)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].convergence_rate / max(0.001, x[1].computation_time),
            reverse=True
        )
        
        report_lines.append("🏆 Algorithm Performance Ranking:")
        for i, (algorithm, result) in enumerate(sorted_results, 1):
            status = "✅" if result.computation_time < float('inf') else "❌"
            
            report_lines.append(f"{i}. {status} {algorithm.value.upper()}")
            report_lines.append(f"   Time: {result.computation_time:.3f}s")
            report_lines.append(f"   Iterations: {result.iterations}")
            report_lines.append(f"   Convergence Rate: {result.convergence_rate:.3f}")
            report_lines.append(f"   Memory: {result.memory_usage:.1f} MB")
            report_lines.append("")
        
        # Performance metrics comparison
        report_lines.extend([
            "📊 Detailed Metrics Comparison:",
            ""
        ])
        
        metrics_table = []
        headers = ["Algorithm", "Time (s)", "Iterations", "Conv. Rate", "Memory (MB)"]
        metrics_table.append(" | ".join(f"{h:>12}" for h in headers))
        metrics_table.append("-" * 65)
        
        for algorithm, result in sorted_results:
            row = [
                algorithm.value[:12],
                f"{result.computation_time:.3f}",
                f"{result.iterations}",
                f"{result.convergence_rate:.3f}",
                f"{result.memory_usage:.1f}"
            ]
            metrics_table.append(" | ".join(f"{v:>12}" for v in row))
        
        report_lines.extend(metrics_table)
        report_lines.extend([
            "",
            "=" * 60,
            "Report generated by Terragon Labs Advanced Algorithm Suite",
            "=" * 60
        ])
        
        return "\n".join(report_lines)