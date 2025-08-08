"""GPU-accelerated analog PDE solver implementation."""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from ..core.solver import AnalogPDESolver
from ..utils.logger import get_logger, PerformanceLogger

# Attempt to import GPU libraries with fallback
HAS_CUPY = False
HAS_NUMBA_CUDA = False

try:
    import cupy as cp
    HAS_CUPY = True
    CupyArray = cp.ndarray
except ImportError:
    cp = None
    HAS_CUPY = False
    # Create dummy type for annotations
    CupyArray = type(None)

try:
    from numba import cuda
    import numba
    HAS_NUMBA_CUDA = True
except ImportError:
    cuda = None
    numba = None
    HAS_NUMBA_CUDA = False


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration."""
    device_id: int = 0
    memory_pool_size_gb: float = 4.0
    use_streams: bool = True
    num_streams: int = 4
    block_size: int = 256
    preferred_backend: str = 'cupy'  # 'cupy' or 'numba'


class GPUAcceleratedSolver:
    """GPU-accelerated analog PDE solver."""
    
    def __init__(
        self,
        base_solver: AnalogPDESolver,
        gpu_config: Optional[GPUConfig] = None,
        fallback_to_cpu: bool = True
    ):
        """Initialize GPU-accelerated solver.
        
        Args:
            base_solver: Base analog PDE solver
            gpu_config: GPU configuration
            fallback_to_cpu: Whether to fallback to CPU if GPU unavailable
        """
        self.logger = get_logger('gpu_accelerated_solver')
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.base_solver = base_solver
        self.config = gpu_config or GPUConfig()
        self.fallback_to_cpu = fallback_to_cpu
        
        # GPU availability check
        self.gpu_available = self._check_gpu_availability()
        self.backend = self._select_backend()
        
        if self.gpu_available:
            self._initialize_gpu_resources()
        else:
            self.logger.warning("GPU acceleration not available, using CPU fallback")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        if not (HAS_CUPY or HAS_NUMBA_CUDA):
            return False
        
        try:
            if self.config.preferred_backend == 'cupy' and HAS_CUPY:
                # Test CuPy availability
                cp.cuda.Device(self.config.device_id).use()
                test_array = cp.array([1, 2, 3])
                _ = cp.sum(test_array)
                return True
            elif self.config.preferred_backend == 'numba' and HAS_NUMBA_CUDA:
                # Test Numba CUDA availability
                cuda.select_device(self.config.device_id)
                return True
            else:
                # Try any available backend
                if HAS_CUPY:
                    cp.cuda.Device(self.config.device_id).use()
                    return True
                elif HAS_NUMBA_CUDA:
                    cuda.select_device(self.config.device_id)
                    return True
        except Exception as e:
            self.logger.debug(f"GPU availability check failed: {e}")
        
        return False
    
    def _select_backend(self) -> str:
        """Select the best available GPU backend."""
        if not self.gpu_available:
            return 'cpu'
        
        if self.config.preferred_backend == 'cupy' and HAS_CUPY:
            return 'cupy'
        elif self.config.preferred_backend == 'numba' and HAS_NUMBA_CUDA:
            return 'numba'
        elif HAS_CUPY:
            return 'cupy'
        elif HAS_NUMBA_CUDA:
            return 'numba'
        else:
            return 'cpu'
    
    def _initialize_gpu_resources(self) -> None:
        """Initialize GPU resources."""
        if self.backend == 'cupy':
            self._initialize_cupy_resources()
        elif self.backend == 'numba':
            self._initialize_numba_resources()
        
        self.logger.info(f"GPU acceleration initialized with {self.backend} backend")
    
    def _initialize_cupy_resources(self) -> None:
        """Initialize CuPy resources."""
        cp.cuda.Device(self.config.device_id).use()
        
        # Initialize memory pool
        memory_pool_bytes = int(self.config.memory_pool_size_gb * 1024**3)
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=memory_pool_bytes)
        
        # Initialize streams if requested
        if self.config.use_streams:
            self.streams = [cp.cuda.Stream() for _ in range(self.config.num_streams)]
        else:
            self.streams = None
    
    def _initialize_numba_resources(self) -> None:
        """Initialize Numba CUDA resources."""
        cuda.select_device(self.config.device_id)
        
        # Initialize streams if requested
        if self.config.use_streams:
            self.streams = [cuda.stream() for _ in range(self.config.num_streams)]
        else:
            self.streams = None
    
    def solve_gpu(
        self,
        pde,
        iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve PDE using GPU acceleration.
        
        Args:
            pde: PDE object to solve
            iterations: Maximum iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            Tuple of (solution, solve_info)
        """
        if not self.gpu_available and not self.fallback_to_cpu:
            raise RuntimeError("GPU not available and CPU fallback disabled")
        
        if not self.gpu_available:
            self.logger.info("Using CPU fallback")
            return self.base_solver.solve(pde, iterations, convergence_threshold), {
                'method': 'cpu_fallback',
                'gpu_available': False
            }
        
        solve_info = {
            'method': f'gpu_{self.backend}',
            'gpu_available': True,
            'device_id': self.config.device_id
        }
        
        self.perf_logger.start_timer('gpu_solve_total')
        
        try:
            if self.backend == 'cupy':
                solution = self._solve_cupy(pde, iterations, convergence_threshold)
            elif self.backend == 'numba':
                solution = self._solve_numba(pde, iterations, convergence_threshold)
            else:
                raise RuntimeError(f"Unknown GPU backend: {self.backend}")
            
            solve_time = self.perf_logger.end_timer('gpu_solve_total')
            solve_info['solve_time'] = solve_time
            
            return solution, solve_info
            
        except Exception as e:
            self.logger.error(f"GPU solving failed: {e}")
            if self.fallback_to_cpu:
                self.logger.info("Falling back to CPU solver")
                solution = self.base_solver.solve(pde, iterations, convergence_threshold)
                solve_info.update({
                    'method': 'cpu_fallback_after_gpu_error',
                    'gpu_error': str(e)
                })
                return solution, solve_info
            else:
                raise
    
    def _solve_cupy(
        self,
        pde,
        iterations: int,
        convergence_threshold: float
    ) -> np.ndarray:
        """Solve using CuPy backend."""
        # Transfer data to GPU
        size = self.base_solver.crossbar_size
        
        # Create Laplacian matrix on GPU
        laplacian_gpu = self._create_laplacian_cupy(size)
        
        # Initialize solution on GPU
        phi_gpu = cp.random.random(size).astype(cp.float32) * 0.1
        
        # Create source term on GPU
        if hasattr(pde, 'source_function') and pde.source_function:
            x = cp.linspace(0, 1, size)
            source_gpu = cp.array([pde.source_function(float(xi), 0) for xi in x])
        else:
            source_gpu = cp.ones(size, dtype=cp.float32) * 0.1
        
        # Iterative solver on GPU
        for i in range(iterations):
            # Matrix-vector multiplication
            residual = cp.dot(laplacian_gpu, phi_gpu) + source_gpu
            
            # Jacobi update
            phi_new = phi_gpu - 0.1 * residual
            
            # Apply boundary conditions
            phi_new[0] = 0.0
            phi_new[-1] = 0.0
            
            # Check convergence
            error = cp.linalg.norm(phi_new - phi_gpu)
            phi_gpu = phi_new
            
            if error < convergence_threshold:
                self.logger.debug(f"GPU solver converged after {i+1} iterations")
                break
        
        # Transfer result back to CPU
        return cp.asnumpy(phi_gpu)
    
    def _solve_numba(
        self,
        pde,
        iterations: int,
        convergence_threshold: float
    ) -> np.ndarray:
        """Solve using Numba CUDA backend."""
        size = self.base_solver.crossbar_size
        
        # Create data on CPU first
        phi = np.random.random(size).astype(np.float32) * 0.1
        laplacian = self._create_laplacian_matrix_numpy(size).astype(np.float32)
        
        if hasattr(pde, 'source_function') and pde.source_function:
            x = np.linspace(0, 1, size)
            source = np.array([pde.source_function(xi, 0) for xi in x], dtype=np.float32)
        else:
            source = np.ones(size, dtype=np.float32) * 0.1
        
        # Transfer to GPU
        phi_gpu = cuda.to_device(phi)
        laplacian_gpu = cuda.to_device(laplacian)
        source_gpu = cuda.to_device(source)
        
        # Create work arrays on GPU
        residual_gpu = cuda.device_array(size, dtype=np.float32)
        phi_new_gpu = cuda.device_array(size, dtype=np.float32)
        
        # Configure CUDA kernel
        threads_per_block = min(self.config.block_size, size)
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
        
        # Iterative solver
        for i in range(iterations):
            # Matrix-vector multiplication kernel
            self._matvec_kernel[blocks_per_grid, threads_per_block](
                laplacian_gpu, phi_gpu, residual_gpu, size
            )
            
            # Update kernel
            self._jacobi_update_kernel[blocks_per_grid, threads_per_block](
                phi_gpu, residual_gpu, source_gpu, phi_new_gpu, size
            )
            
            # Apply boundary conditions
            self._apply_bc_kernel[1, 1](phi_new_gpu, size)
            
            # Check convergence (simplified)
            if i % 10 == 0:
                phi_cpu = phi_new_gpu.copy_to_host()
                phi_old_cpu = phi_gpu.copy_to_host()
                error = np.linalg.norm(phi_cpu - phi_old_cpu)
                
                if error < convergence_threshold:
                    self.logger.debug(f"GPU solver converged after {i+1} iterations")
                    break
            
            # Swap arrays
            phi_gpu, phi_new_gpu = phi_new_gpu, phi_gpu
        
        # Transfer result back to CPU
        return phi_gpu.copy_to_host()
    
    def _create_laplacian_cupy(self, size: int) -> CupyArray:
        """Create Laplacian matrix using CuPy."""
        laplacian = cp.zeros((size, size), dtype=cp.float32)
        
        # Main diagonal
        cp.fill_diagonal(laplacian, -2.0)
        
        # Off-diagonals
        for i in range(size - 1):
            laplacian[i, i + 1] = 1.0
            laplacian[i + 1, i] = 1.0
        
        return laplacian
    
    def _create_laplacian_matrix_numpy(self, size: int) -> np.ndarray:
        """Create Laplacian matrix using NumPy."""
        laplacian = np.zeros((size, size), dtype=np.float32)
        
        # Main diagonal
        np.fill_diagonal(laplacian, -2.0)
        
        # Off-diagonals
        for i in range(size - 1):
            laplacian[i, i + 1] = 1.0
            laplacian[i + 1, i] = 1.0
        
        return laplacian
    
    @property
    def _matvec_kernel(self):
        """Matrix-vector multiplication kernel."""
        if not hasattr(self, '_matvec_kernel_cached'):
            @cuda.jit
            def matvec_kernel(A, x, y, n):
                i = cuda.grid(1)
                if i < n:
                    result = 0.0
                    for j in range(n):
                        result += A[i, j] * x[j]
                    y[i] = result
            
            self._matvec_kernel_cached = matvec_kernel
        
        return self._matvec_kernel_cached
    
    @property
    def _jacobi_update_kernel(self):
        """Jacobi update kernel."""
        if not hasattr(self, '_jacobi_update_kernel_cached'):
            @cuda.jit
            def jacobi_update_kernel(phi, residual, source, phi_new, n):
                i = cuda.grid(1)
                if i < n:
                    phi_new[i] = phi[i] - 0.1 * (residual[i] + source[i])
            
            self._jacobi_update_kernel_cached = jacobi_update_kernel
        
        return self._jacobi_update_kernel_cached
    
    @property
    def _apply_bc_kernel(self):
        """Apply boundary conditions kernel."""
        if not hasattr(self, '_apply_bc_kernel_cached'):
            @cuda.jit
            def apply_bc_kernel(phi, n):
                phi[0] = 0.0
                phi[n-1] = 0.0
            
            self._apply_bc_kernel_cached = apply_bc_kernel
        
        return self._apply_bc_kernel_cached
    
    def benchmark_gpu_vs_cpu(
        self,
        pde,
        iterations: int = 100,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance.
        
        Args:
            pde: PDE to solve
            iterations: Number of iterations per solve
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting GPU vs CPU benchmark with {num_runs} runs")
        
        results = {
            'num_runs': num_runs,
            'iterations': iterations,
            'gpu_available': self.gpu_available,
            'backend': self.backend,
            'problem_size': self.base_solver.crossbar_size
        }
        
        # CPU benchmark
        cpu_times = []
        for i in range(num_runs):
            self.perf_logger.start_timer(f'cpu_run_{i}')
            _ = self.base_solver.solve(pde, iterations=iterations)
            cpu_time = self.perf_logger.end_timer(f'cpu_run_{i}')
            cpu_times.append(cpu_time)
        
        results['cpu_times'] = cpu_times
        results['avg_cpu_time'] = np.mean(cpu_times)
        results['std_cpu_time'] = np.std(cpu_times)
        
        # GPU benchmark
        if self.gpu_available:
            gpu_times = []
            for i in range(num_runs):
                self.perf_logger.start_timer(f'gpu_run_{i}')
                _ = self.solve_gpu(pde, iterations=iterations)
                gpu_time = self.perf_logger.end_timer(f'gpu_run_{i}')
                gpu_times.append(gpu_time)
            
            results['gpu_times'] = gpu_times
            results['avg_gpu_time'] = np.mean(gpu_times)
            results['std_gpu_time'] = np.std(gpu_times)
            results['speedup'] = results['avg_cpu_time'] / results['avg_gpu_time']
        else:
            results['gpu_times'] = None
            results['speedup'] = None
        
        self.logger.info(f"Benchmark completed. Speedup: {results.get('speedup', 'N/A'):.2f}x")
        return results
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        info = {'gpu_available': True, 'backend': self.backend}
        
        try:
            if self.backend == 'cupy':
                mempool = cp.get_default_memory_pool()
                info.update({
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes(),
                    'limit_bytes': mempool.get_limit(),
                    'n_free_blocks': mempool.n_free_blocks()
                })
            elif self.backend == 'numba':
                # Numba doesn't provide detailed memory info
                info['memory_details'] = 'Not available with Numba backend'
        except Exception as e:
            info['error'] = str(e)
        
        return info


class GPUMemoryManager:
    """GPU memory management utilities."""
    
    def __init__(self, backend: str = 'cupy'):
        """Initialize GPU memory manager.
        
        Args:
            backend: GPU backend ('cupy' or 'numba')
        """
        self.backend = backend
        self.logger = get_logger('gpu_memory_manager')
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.backend == 'cupy' and HAS_CUPY:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            self.logger.info("CuPy memory cache cleared")
        elif self.backend == 'numba' and HAS_NUMBA_CUDA:
            # Numba doesn't have explicit cache clearing
            self.logger.info("Numba CUDA cache management not available")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        stats = {'backend': self.backend}
        
        if self.backend == 'cupy' and HAS_CUPY:
            try:
                mempool = cp.get_default_memory_pool()
                stats.update({
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes(),
                    'n_free_blocks': mempool.n_free_blocks(),
                    'limit_bytes': mempool.get_limit()
                })
            except Exception as e:
                stats['error'] = str(e)
        
        return stats