"""Advanced performance optimization for analog PDE solvers."""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from dataclasses import dataclass
from ..utils.logging_config import get_logger, PerformanceMonitor


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_parallel_crossbars: bool = True
    max_worker_threads: int = 4
    enable_adaptive_precision: bool = True
    enable_caching: bool = True
    enable_prefetch: bool = True
    enable_load_balancing: bool = True
    memory_pool_size_mb: int = 512
    compression_ratio: float = 0.7


class PerformanceOptimizer:
    """Advanced performance optimization for analog PDE solving."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize performance optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = get_logger('optimizer')
        
        # Performance caching
        self.matrix_cache: Dict[str, np.ndarray] = {}
        self.solution_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Resource pooling
        self.thread_pool = None
        self.process_pool = None
        
        # Adaptive precision tracking
        self.precision_history: List[Dict[str, Any]] = []
        
        # Load balancing
        self.worker_loads: List[float] = []
        self.load_lock = threading.Lock()
        
        self.logger.info(f"Performance optimizer initialized with config: {self.config}")
    
    def optimize_crossbar_operations(
        self,
        crossbars: List[Any],
        input_vectors: List[np.ndarray],
        operation_type: str = "vmm"
    ) -> List[np.ndarray]:
        """Optimize parallel crossbar operations.
        
        Args:
            crossbars: List of crossbar arrays
            input_vectors: Input vectors for each crossbar
            operation_type: Type of operation to perform
            
        Returns:
            List of results from each crossbar
        """
        if not self.config.enable_parallel_crossbars or len(crossbars) == 1:
            # Sequential processing
            return [self._execute_crossbar_operation(cb, vec, operation_type) 
                   for cb, vec in zip(crossbars, input_vectors)]
        
        # Parallel processing with load balancing
        with PerformanceMonitor("Parallel crossbar operations", self.logger):
            if self.thread_pool is None:
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.config.max_worker_threads
                )
            
            # Load balancing: distribute work based on crossbar sizes
            work_items = self._balance_crossbar_workload(crossbars, input_vectors)
            
            # Submit tasks
            futures = []
            for crossbar, input_vec, worker_id in work_items:
                future = self.thread_pool.submit(
                    self._execute_crossbar_operation_with_monitoring,
                    crossbar, input_vec, operation_type, worker_id
                )
                futures.append(future)
            
            # Collect results
            results = [future.result() for future in futures]
            
            return results
    
    def _execute_crossbar_operation(
        self, 
        crossbar: Any, 
        input_vector: np.ndarray,
        operation_type: str
    ) -> np.ndarray:
        """Execute single crossbar operation with caching."""
        # Check cache first
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(crossbar, input_vector, operation_type)
            if cache_key in self.solution_cache:
                self.cache_hits += 1
                self.logger.debug(f"Cache hit for operation {operation_type}")
                return self.solution_cache[cache_key]
            else:
                self.cache_misses += 1
        
        # Execute operation
        if operation_type == "vmm":
            result = crossbar.compute_vmm(input_vector)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        # Cache result
        if self.config.enable_caching:
            self._update_cache(cache_key, result)
        
        return result
    
    def _execute_crossbar_operation_with_monitoring(
        self,
        crossbar: Any,
        input_vector: np.ndarray,
        operation_type: str,
        worker_id: int
    ) -> np.ndarray:
        """Execute crossbar operation with worker monitoring."""
        start_time = time.perf_counter()
        
        try:
            result = self._execute_crossbar_operation(crossbar, input_vector, operation_type)
            
            # Update worker load statistics
            elapsed = time.perf_counter() - start_time
            with self.load_lock:
                if worker_id >= len(self.worker_loads):
                    self.worker_loads.extend([0.0] * (worker_id - len(self.worker_loads) + 1))
                self.worker_loads[worker_id] = elapsed
            
            return result
            
        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed: {e}")
            raise
    
    def _balance_crossbar_workload(
        self,
        crossbars: List[Any],
        input_vectors: List[np.ndarray]
    ) -> List[Tuple[Any, np.ndarray, int]]:
        """Balance workload across workers based on crossbar complexity."""
        if not self.config.enable_load_balancing:
            # Simple round-robin assignment
            return [(cb, vec, i % self.config.max_worker_threads) 
                   for i, (cb, vec) in enumerate(zip(crossbars, input_vectors))]
        
        # Calculate complexity scores for each crossbar
        complexities = []
        for crossbar in crossbars:
            # Complexity based on array size and programming status
            size_complexity = crossbar.rows * crossbar.cols
            prog_complexity = 1.2 if getattr(crossbar, 'is_programmed', True) else 0.5
            complexities.append(size_complexity * prog_complexity)
        
        # Assign work to balance total complexity per worker
        worker_loads = [0.0] * self.config.max_worker_threads
        work_assignments = []
        
        for i, (crossbar, input_vec) in enumerate(zip(crossbars, input_vectors)):
            # Assign to least loaded worker
            worker_id = np.argmin(worker_loads)
            worker_loads[worker_id] += complexities[i]
            work_assignments.append((crossbar, input_vec, worker_id))
        
        self.logger.debug(f"Load balancing assigned work: {worker_loads}")
        return work_assignments
    
    def optimize_memory_usage(
        self,
        large_matrices: List[np.ndarray],
        compression_enabled: bool = True
    ) -> List[np.ndarray]:
        """Optimize memory usage for large matrix operations."""
        if not compression_enabled:
            return large_matrices
        
        optimized_matrices = []
        
        for i, matrix in enumerate(large_matrices):
            with PerformanceMonitor(f"Matrix {i} optimization", self.logger):
                # Compress sparse matrices
                if self._is_sparse(matrix):
                    compressed = self._compress_sparse_matrix(matrix)
                    optimized_matrices.append(compressed)
                    self.logger.debug(f"Matrix {i} compressed (sparse)")
                    
                # Use memory mapping for very large dense matrices
                elif matrix.nbytes > 100 * 1024 * 1024:  # 100MB threshold
                    mapped = self._create_memory_mapped_matrix(matrix, f"matrix_{i}")
                    optimized_matrices.append(mapped)
                    self.logger.debug(f"Matrix {i} memory mapped")
                    
                else:
                    # Use in-place operations where possible
                    optimized_matrices.append(matrix)
        
        return optimized_matrices
    
    def adaptive_precision_solver(
        self,
        solver: Any,
        pde: Any,
        target_accuracy: float = 1e-6,
        max_precision_bits: int = 32
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve PDE with adaptive precision for optimal performance."""
        if not self.config.enable_adaptive_precision:
            # Use standard precision
            return solver.solve(pde), {"precision_used": "standard"}
        
        # Start with low precision and increase as needed
        precision_levels = [4, 8, 16, 32]
        precision_levels = [p for p in precision_levels if p <= max_precision_bits]
        
        best_solution = None
        best_stats = {}
        
        for precision in precision_levels:
            with PerformanceMonitor(f"{precision}-bit precision solve", self.logger):
                try:
                    # Configure solver for this precision
                    original_precision = getattr(solver, 'precision_bits', None)
                    if hasattr(solver, 'precision_bits'):
                        solver.precision_bits = precision
                    
                    # Solve with current precision
                    solution = solver.solve(pde)
                    
                    # Check accuracy
                    if hasattr(pde, 'solve_digital'):
                        reference = pde.solve_digital()
                        error = np.mean(np.abs(solution - reference))
                    else:
                        # Use convergence error as proxy
                        convergence_info = solver.get_convergence_info()
                        error = convergence_info.get('final_error', 1.0)
                    
                    stats = {
                        'precision_bits': precision,
                        'accuracy_error': float(error),
                        'converged': error <= target_accuracy
                    }
                    
                    self.precision_history.append(stats)
                    
                    # If accuracy target met, use this solution
                    if error <= target_accuracy:
                        best_solution = solution
                        best_stats = stats
                        self.logger.info(
                            f"Adaptive precision: {precision} bits achieved target accuracy"
                        )
                        break
                    
                    # Otherwise, keep as best so far
                    if best_solution is None or error < best_stats['accuracy_error']:
                        best_solution = solution
                        best_stats = stats
                    
                    # Restore original precision
                    if original_precision is not None and hasattr(solver, 'precision_bits'):
                        solver.precision_bits = original_precision
                        
                except Exception as e:
                    self.logger.error(f"Adaptive precision {precision}-bit failed: {e}")
                    continue
        
        if best_solution is None:
            raise RuntimeError("Adaptive precision solver failed at all precision levels")
        
        return best_solution, best_stats
    
    def prefetch_and_pipeline(
        self,
        solver: Any,
        pde_sequence: List[Any],
        pipeline_depth: int = 2
    ) -> List[np.ndarray]:
        """Execute sequence of PDE solves with prefetching and pipelining."""
        if not self.config.enable_prefetch or len(pde_sequence) <= 1:
            # Sequential execution
            return [solver.solve(pde) for pde in pde_sequence]
        
        solutions = []
        
        with PerformanceMonitor("Pipelined PDE solving", self.logger):
            if self.process_pool is None:
                self.process_pool = ProcessPoolExecutor(
                    max_workers=min(pipeline_depth, len(pde_sequence))
                )
            
            # Submit initial batch
            futures = {}
            for i in range(min(pipeline_depth, len(pde_sequence))):
                future = self.process_pool.submit(self._solve_pde_isolated, solver, pde_sequence[i])
                futures[future] = i
            
            # Process results as they complete and submit new work
            completed_count = 0
            next_submit_index = pipeline_depth
            
            while futures:
                # Wait for next completion
                from concurrent.futures import as_completed
                
                for future in as_completed(futures, timeout=60):
                    result_index = futures[future]
                    
                    try:
                        solution = future.result()
                        # Store solution in correct order
                        while len(solutions) <= result_index:
                            solutions.append(None)
                        solutions[result_index] = solution
                        
                        completed_count += 1
                        self.logger.debug(f"Completed PDE {result_index}")
                        
                    except Exception as e:
                        self.logger.error(f"PDE {result_index} failed: {e}")
                        solutions.append(None)  # Placeholder for failed solve
                    
                    # Remove completed future
                    del futures[future]
                    
                    # Submit next work if available
                    if next_submit_index < len(pde_sequence):
                        new_future = self.process_pool.submit(
                            self._solve_pde_isolated, 
                            solver, 
                            pde_sequence[next_submit_index]
                        )
                        futures[new_future] = next_submit_index
                        next_submit_index += 1
                    
                    break  # Process one completion at a time
        
        return solutions
    
    def _solve_pde_isolated(self, solver: Any, pde: Any) -> np.ndarray:
        """Solve PDE in isolated process (for multiprocessing)."""
        return solver.solve(pde)
    
    def _generate_cache_key(
        self, 
        crossbar: Any, 
        input_vector: np.ndarray,
        operation_type: str
    ) -> str:
        """Generate cache key for crossbar operation."""
        # Use hash of crossbar state and input
        crossbar_hash = hash((
            crossbar.rows, crossbar.cols,
            getattr(crossbar, 'operation_count', 0) // 100  # Coarse granularity
        ))
        
        input_hash = hash(input_vector.tobytes())
        
        return f"{operation_type}_{crossbar_hash}_{input_hash}"
    
    def _update_cache(self, key: str, result: np.ndarray):
        """Update solution cache with size management."""
        max_cache_size = 1000  # Maximum cached entries
        
        if len(self.solution_cache) >= max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.solution_cache.keys())[:100]
            for k in keys_to_remove:
                del self.solution_cache[k]
        
        self.solution_cache[key] = result.copy()
    
    def _is_sparse(self, matrix: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if matrix is sparse (has many zeros)."""
        return np.count_nonzero(matrix) / matrix.size < threshold
    
    def _compress_sparse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Compress sparse matrix using scipy if available."""
        try:
            from scipy import sparse
            return sparse.csr_matrix(matrix)
        except ImportError:
            self.logger.warning("Scipy not available for sparse matrix compression")
            return matrix
    
    def _create_memory_mapped_matrix(self, matrix: np.ndarray, filename: str) -> np.ndarray:
        """Create memory-mapped version of large matrix."""
        try:
            import tempfile
            import os
            
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            filepath = os.path.join(temp_dir, f"{filename}.npy")
            
            # Save matrix to file
            np.save(filepath, matrix)
            
            # Create memory-mapped array
            mapped_array = np.load(filepath, mmap_mode='r+')
            
            self.logger.debug(f"Created memory-mapped matrix: {filepath}")
            return mapped_array
            
        except Exception as e:
            self.logger.warning(f"Failed to create memory-mapped matrix: {e}")
            return matrix
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0.0
        
        stats = {
            "cache_statistics": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate,
                "cache_size": len(self.solution_cache)
            },
            "worker_statistics": {
                "thread_pool_active": self.thread_pool is not None,
                "process_pool_active": self.process_pool is not None,
                "max_workers": self.config.max_worker_threads,
                "current_loads": self.worker_loads.copy()
            },
            "optimization_config": {
                "parallel_crossbars": self.config.enable_parallel_crossbars,
                "adaptive_precision": self.config.enable_adaptive_precision,
                "caching": self.config.enable_caching,
                "prefetch": self.config.enable_prefetch,
                "load_balancing": self.config.enable_load_balancing
            },
            "precision_history": self.precision_history[-10:]  # Last 10 entries
        }
        
        return stats
    
    def cleanup(self):
        """Cleanup resources and worker pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
            
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        # Clear caches to free memory
        self.matrix_cache.clear()
        self.solution_cache.clear()
        
        self.logger.info("Performance optimizer cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup