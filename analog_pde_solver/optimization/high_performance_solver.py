"""High-performance optimized analog PDE solver with caching, concurrency, and scaling."""

import numpy as np
import logging
import time
import hashlib
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import json
from dataclasses import dataclass, asdict

from ..core.robust_solver import RobustAnalogPDESolver
from ..core.health_monitor import PerformanceMetrics


@dataclass
class CacheStats:
    """Statistics for the intelligent cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.total_requests, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "hit_rate": self.hit_rate
        }


@dataclass 
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size: int = 128
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_adaptive_precision: bool = True
    enable_memory_optimization: bool = True
    enable_vectorization: bool = True
    precompile_matrices: bool = True
    

class IntelligentCache:
    """Intelligent caching system for PDE solutions and matrices."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_count: Dict[str, int] = {}
        self.access_time: Dict[str, float] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': [self._serialize_arg(arg) for arg in args],
            'kwargs': {k: self._serialize_arg(v) for k, v in sorted(kwargs.items())}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_arg(self, arg: Any) -> Any:
        """Serialize argument for cache key generation."""
        if isinstance(arg, np.ndarray):
            return {
                'type': 'ndarray',
                'shape': arg.shape,
                'dtype': str(arg.dtype),
                'hash': hashlib.md5(arg.tobytes()).hexdigest()[:16]
            }
        elif hasattr(arg, '__dict__'):
            return {
                'type': type(arg).__name__,
                'attrs': {k: self._serialize_arg(v) for k, v in arg.__dict__.items()
                         if not k.startswith('_')}
            }
        elif callable(arg):
            return {
                'type': 'function',
                'name': getattr(arg, '__name__', 'unknown'),
                'hash': hashlib.md5(str(arg).encode()).hexdigest()[:16]
            }
        else:
            return arg
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            self.stats.total_requests += 1
            
            if key in self.cache:
                self.stats.hits += 1
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.access_time[key] = time.time()
                self.logger.debug(f"Cache hit for key {key[:8]}...")
                return self.cache[key]['value']
            else:
                self.stats.misses += 1
                self.logger.debug(f"Cache miss for key {key[:8]}...")
                return None
    
    def put(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """Put item in cache with LRU eviction."""
        with self.lock:
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = {
                'value': value,
                'metadata': metadata or {},
                'created': time.time()
            }
            self.access_count[key] = 1
            self.access_time[key] = time.time()
            self.logger.debug(f"Cached result for key {key[:8]}...")
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
        
        del self.cache[lru_key]
        del self.access_count[lru_key]
        del self.access_time[lru_key]
        self.stats.evictions += 1
        self.logger.debug(f"Evicted LRU item {lru_key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.access_time.clear()
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'stats': self.stats.to_dict(),
                'current_size': len(self.cache),
                'max_size': self.max_size
            }


def cached_computation(cache_instance):
    """Decorator for caching expensive computations."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Use the instance's cache, not the decorator argument
            cache = getattr(self, 'cache', None) or cache_instance
            if not cache:
                # No cache available, compute directly
                return func(self, *args, **kwargs)
            
            # Generate cache key (excluding self)
            cache_key = cache._generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            start_time = time.time()
            result = func(self, *args, **kwargs)
            computation_time = time.time() - start_time
            
            # Cache with metadata
            cache.put(cache_key, result, {
                'computation_time': computation_time,
                'function': func.__name__
            })
            
            return result
        return wrapper
    return decorator


class HighPerformanceAnalogPDESolver(RobustAnalogPDESolver):
    """High-performance analog PDE solver with optimization and scaling."""
    
    def __init__(
        self,
        crossbar_size: int = 128,
        conductance_range: tuple = (1e-9, 1e-6),
        noise_model: str = "realistic",
        optimization_config: OptimizationConfig = None,
        **robust_kwargs
    ):
        """Initialize high-performance solver.
        
        Args:
            crossbar_size: Size of crossbar array
            conductance_range: Conductance range
            noise_model: Noise model
            optimization_config: Optimization configuration
            **robust_kwargs: Arguments for robust solver
        """
        # Set defaults
        if optimization_config is None:
            optimization_config = OptimizationConfig()
        
        self.opt_config = optimization_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_stats = {
            'cache_hits': 0,
            'parallel_solves': 0,
            'memory_optimizations': 0,
            'vectorization_speedups': 0
        }
        
        # Initialize caching system
        if self.opt_config.enable_caching:
            self.cache = IntelligentCache(max_size=self.opt_config.cache_size)
        else:
            self.cache = None
        
        # Initialize thread pool for parallel processing
        if self.opt_config.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.opt_config.max_workers)
        else:
            self.thread_pool = None
        
        # Initialize parent robust solver
        super().__init__(
            crossbar_size=crossbar_size,
            conductance_range=conductance_range,
            noise_model=noise_model,
            **robust_kwargs
        )
        
        # Precompile common matrices if enabled
        if self.opt_config.precompile_matrices:
            self._precompile_common_matrices()
        
        self.logger.info("HighPerformanceAnalogPDESolver initialized with optimizations")
    
    def _precompile_common_matrices(self) -> None:
        """Precompile and cache common matrix operations."""
        common_sizes = [4, 8, 16, 32, 64, 128]
        
        for size in common_sizes:
            if size <= self.crossbar_size:
                # Cache Laplacian matrices
                if self.cache:
                    key = f"laplacian_{size}"
                    if not self.cache.get(key):
                        matrix = self._create_laplacian_matrix(size)
                        self.cache.put(key, matrix, {'type': 'laplacian', 'size': size})
                
        self.logger.info(f"Precompiled matrices for sizes: {common_sizes}")
    
    def _cached_create_laplacian_matrix(self, size: int) -> np.ndarray:
        """Cached version of Laplacian matrix creation."""
        if self.cache:
            # Check cache manually
            cache_key = self.cache._generate_key(size)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = super()._create_laplacian_matrix(size)
            self.cache.put(cache_key, result, {'type': 'laplacian', 'size': size})
            return result
        else:
            return super()._create_laplacian_matrix(size)
    
    def _create_laplacian_matrix(self, size: int) -> np.ndarray:
        """Override to use caching when available."""
        if self.cache:
            return self._cached_create_laplacian_matrix(size)
        else:
            return super()._create_laplacian_matrix(size)
    
    def solve_parallel(
        self, 
        pdes: List[Any], 
        iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> List[np.ndarray]:
        """Solve multiple PDEs in parallel."""
        if not self.opt_config.enable_parallel_processing or not self.thread_pool:
            # Fallback to sequential solving
            return [self.solve(pde, iterations, convergence_threshold) for pde in pdes]
        
        self.logger.info(f"Solving {len(pdes)} PDEs in parallel with {self.opt_config.max_workers} workers")
        
        # Submit all tasks
        future_to_pde = {}
        for i, pde in enumerate(pdes):
            future = self.thread_pool.submit(
                self.solve, pde, iterations, convergence_threshold
            )
            future_to_pde[future] = i
        
        # Collect results in order
        results = [None] * len(pdes)
        completed = 0
        
        for future in as_completed(future_to_pde):
            pde_index = future_to_pde[future]
            try:
                results[pde_index] = future.result()
                completed += 1
                self.logger.debug(f"Parallel solve {completed}/{len(pdes)} completed")
            except Exception as e:
                self.logger.error(f"Parallel solve {pde_index} failed: {e}")
                results[pde_index] = None
        
        self.optimization_stats['parallel_solves'] += len(pdes)
        return results
    
    def solve_with_adaptive_precision(
        self,
        pde: Any,
        target_accuracy: float = 1e-6,
        max_precision_levels: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve with adaptive precision to optimize performance vs accuracy."""
        if not self.opt_config.enable_adaptive_precision:
            solution = self.solve(pde)
            return solution, {'precision_level': 'fixed', 'iterations': []}
        
        self.logger.info("Starting adaptive precision solve")
        
        precision_levels = [1e-3, 1e-4, 1e-5, target_accuracy]
        precision_levels = [p for p in precision_levels if p >= target_accuracy]
        precision_levels = precision_levels[:max_precision_levels]
        
        solutions = []
        iteration_counts = []
        solve_times = []
        
        for i, threshold in enumerate(precision_levels):
            self.logger.debug(f"Adaptive precision level {i+1}: threshold={threshold:.2e}")
            
            start_time = time.time()
            
            # Use previous solution as initial guess if available
            if solutions:
                # Could implement warm start here
                pass
            
            # Estimate iterations needed for this precision
            estimated_iterations = int(50 * (1 / threshold) ** 0.1)
            estimated_iterations = min(max(estimated_iterations, 10), 500)
            
            solution = self.solve(pde, iterations=estimated_iterations, 
                                convergence_threshold=threshold)
            
            solve_time = time.time() - start_time
            solutions.append(solution)
            iteration_counts.append(estimated_iterations)
            solve_times.append(solve_time)
            
            # Check if we've achieved target accuracy
            if threshold <= target_accuracy:
                break
        
        final_solution = solutions[-1]
        
        adaptive_info = {
            'precision_level': 'adaptive',
            'precision_levels': precision_levels,
            'iterations': iteration_counts,
            'solve_times': solve_times,
            'total_time': sum(solve_times),
            'final_threshold': precision_levels[-1] if precision_levels else target_accuracy
        }
        
        self.logger.info(f"Adaptive precision completed in {adaptive_info['total_time']:.3f}s")
        return final_solution, adaptive_info
    
    def solve_with_memory_optimization(
        self,
        pde: Any,
        chunk_size: Optional[int] = None,
        **solve_kwargs
    ) -> np.ndarray:
        """Solve with memory optimization for large problems."""
        if not self.opt_config.enable_memory_optimization:
            return self.solve(pde, **solve_kwargs)
        
        domain_size = pde.domain_size[0] if isinstance(pde.domain_size, tuple) else pde.domain_size
        
        # Use chunking for large domains
        if chunk_size is None:
            # Adaptive chunk sizing based on available memory
            chunk_size = min(domain_size, 64)  # Conservative default
        
        if domain_size <= chunk_size or chunk_size >= domain_size:
            # No need for chunking
            return self.solve(pde, **solve_kwargs)
        
        self.logger.info(f"Using memory optimization: domain={domain_size}, chunks={chunk_size}")
        
        # For now, implement a simple domain decomposition approach
        # This could be enhanced with more sophisticated techniques
        num_chunks = (domain_size + chunk_size - 1) // chunk_size
        
        # Create sub-problems (simplified approach)
        solutions = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, domain_size)
            chunk_domain_size = end_idx - start_idx
            
            # Create reduced problem for this chunk
            # This is a simplified approach - real implementation would need
            # proper boundary condition handling between chunks
            chunk_pde = type(pde)(domain_size=(chunk_domain_size,))
            if hasattr(pde, 'source_function'):
                chunk_pde.source_function = pde.source_function
            
            chunk_solution = self.solve(chunk_pde, **solve_kwargs)
            solutions.append(chunk_solution)
        
        # Combine chunk solutions
        combined_solution = np.concatenate(solutions)
        
        self.optimization_stats['memory_optimizations'] += 1
        self.logger.info(f"Memory-optimized solve completed with {num_chunks} chunks")
        
        return combined_solution[:domain_size]  # Ensure correct size
    
    def benchmark_solver_performance(
        self,
        test_cases: List[Dict[str, Any]],
        runs_per_case: int = 3
    ) -> Dict[str, Any]:
        """Comprehensive performance benchmarking."""
        self.logger.info(f"Running performance benchmark with {len(test_cases)} test cases")
        
        benchmark_results = {
            'test_cases': [],
            'summary': {},
            'optimization_stats': self.optimization_stats.copy()
        }
        
        if self.cache:
            benchmark_results['cache_stats'] = self.cache.get_stats()
        
        total_start_time = time.time()
        
        for case_idx, test_case in enumerate(test_cases):
            case_results = {
                'case_index': case_idx,
                'config': test_case,
                'runs': [],
                'statistics': {}
            }
            
            # Run multiple times for statistics
            solve_times = []
            solution_norms = []
            
            for run in range(runs_per_case):
                run_start_time = time.time()
                
                try:
                    # Create PDE from test case config
                    pde_config = test_case.get('pde_config', {})
                    pde_type = test_case.get('pde_type', 'PoissonEquation')
                    
                    # Simple PDE creation (could be enhanced)
                    from ..core.equations import PoissonEquation
                    pde = PoissonEquation(**pde_config)
                    
                    # Solve
                    solve_config = test_case.get('solve_config', {})
                    solution = self.solve(pde, **solve_config)
                    
                    run_time = time.time() - run_start_time
                    solution_norm = np.linalg.norm(solution)
                    
                    solve_times.append(run_time * 1000)  # Convert to ms
                    solution_norms.append(solution_norm)
                    
                    case_results['runs'].append({
                        'run': run,
                        'solve_time_ms': run_time * 1000,
                        'solution_norm': solution_norm,
                        'success': True
                    })
                    
                except Exception as e:
                    self.logger.error(f"Benchmark case {case_idx} run {run} failed: {e}")
                    case_results['runs'].append({
                        'run': run,
                        'error': str(e),
                        'success': False
                    })
            
            # Compute statistics
            if solve_times:
                case_results['statistics'] = {
                    'avg_solve_time_ms': np.mean(solve_times),
                    'min_solve_time_ms': np.min(solve_times),
                    'max_solve_time_ms': np.max(solve_times),
                    'std_solve_time_ms': np.std(solve_times),
                    'avg_solution_norm': np.mean(solution_norms),
                    'success_rate': sum(1 for r in case_results['runs'] if r['success']) / runs_per_case
                }
            
            benchmark_results['test_cases'].append(case_results)
            
            self.logger.info(f"Completed benchmark case {case_idx+1}/{len(test_cases)}")
        
        # Overall summary
        total_time = time.time() - total_start_time
        all_solve_times = []
        success_count = 0
        total_runs = 0
        
        for case in benchmark_results['test_cases']:
            for run in case['runs']:
                total_runs += 1
                if run['success']:
                    success_count += 1
                    all_solve_times.append(run['solve_time_ms'])
        
        if all_solve_times:
            benchmark_results['summary'] = {
                'total_benchmark_time_s': total_time,
                'total_test_cases': len(test_cases),
                'total_runs': total_runs,
                'success_rate': success_count / total_runs,
                'avg_solve_time_ms': np.mean(all_solve_times),
                'median_solve_time_ms': np.median(all_solve_times),
                'min_solve_time_ms': np.min(all_solve_times),
                'max_solve_time_ms': np.max(all_solve_times),
                'std_solve_time_ms': np.std(all_solve_times)
            }
        
        self.logger.info(f"Performance benchmark completed in {total_time:.2f}s")
        return benchmark_results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance report."""
        report = {
            'optimization_config': asdict(self.opt_config),
            'optimization_stats': self.optimization_stats.copy(),
            'performance_summary': {}
        }
        
        # Cache statistics
        if self.cache:
            report['cache_performance'] = self.cache.get_stats()
        
        # Thread pool statistics
        if self.thread_pool:
            report['thread_pool'] = {
                'max_workers': self.opt_config.max_workers,
                'enabled': True
            }
        else:
            report['thread_pool'] = {'enabled': False}
        
        # Performance history summary
        if self.performance_history:
            solve_times = [p.solve_time_ms for p in self.performance_history]
            report['performance_summary'] = {
                'total_solves': len(self.performance_history),
                'avg_solve_time_ms': np.mean(solve_times),
                'min_solve_time_ms': np.min(solve_times),
                'max_solve_time_ms': np.max(solve_times),
                'total_solve_time_ms': np.sum(solve_times)
            }
        
        # Optimization effectiveness
        cache_effectiveness = 0
        if self.cache and self.cache.stats.total_requests > 0:
            cache_effectiveness = self.cache.stats.hit_rate
        
        report['optimization_effectiveness'] = {
            'cache_hit_rate': cache_effectiveness,
            'parallel_processing_usage': self.optimization_stats.get('parallel_solves', 0),
            'memory_optimizations_used': self.optimization_stats.get('memory_optimizations', 0)
        }
        
        return report
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.logger.info("Thread pool shutdown completed")
        
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")
        
        self.logger.info("HighPerformanceAnalogPDESolver cleanup completed")