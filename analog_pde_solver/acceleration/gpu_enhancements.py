"""Advanced GPU acceleration enhancements for analog PDE solving."""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import GPU libraries with graceful fallback
HAS_CUPY = False
HAS_TORCH = False
HAS_JAX = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None

try:
    import torch
    HAS_TORCH = torch.cuda.is_available() if torch is not None else False
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = len(jax.devices('gpu')) > 0 if jax is not None else False
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


class AccelerationType(Enum):
    """Types of GPU acceleration available."""
    CUPY = "cupy"
    PYTORCH = "pytorch"
    JAX = "jax"
    MULTI_GPU = "multi_gpu"
    MIXED_PRECISION = "mixed_precision"


@dataclass
class GPUEnhancementConfig:
    """Configuration for advanced GPU acceleration."""
    use_mixed_precision: bool = True
    use_tensor_cores: bool = True
    use_multi_gpu: bool = False
    num_gpus: int = 1
    prefetch_factor: int = 2
    pin_memory: bool = True
    use_compilation: bool = True  # JIT compilation
    batch_size: int = 32
    memory_fraction: float = 0.8
    gradient_checkpointing: bool = False


class MultiGPUManager:
    """Manager for multi-GPU PDE solving."""
    
    def __init__(self, num_gpus: int = None):
        """Initialize multi-GPU manager.
        
        Args:
            num_gpus: Number of GPUs to use (auto-detect if None)
        """
        self.logger = logging.getLogger(__name__)
        self.available_gpus = self._detect_gpus()
        self.num_gpus = min(num_gpus or len(self.available_gpus), len(self.available_gpus))
        
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for multi-GPU acceleration")
        
        self.logger.info(f"Initialized multi-GPU manager with {self.num_gpus} GPUs")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        gpus = []
        
        if HAS_CUPY:
            try:
                for i in range(cp.cuda.runtime.getDeviceCount()):
                    cp.cuda.Device(i).use()
                    gpus.append(i)
            except Exception as e:
                self.logger.debug(f"CuPy GPU detection failed: {e}")
        
        if HAS_TORCH:
            try:
                for i in range(torch.cuda.device_count()):
                    gpus.append(i)
            except Exception as e:
                self.logger.debug(f"PyTorch GPU detection failed: {e}")
        
        return list(set(gpus))  # Remove duplicates
    
    def distribute_problem(
        self,
        problem_data: Dict[str, np.ndarray],
        method: str = "domain_decomposition"
    ) -> List[Dict[str, Any]]:
        """Distribute PDE problem across multiple GPUs.
        
        Args:
            problem_data: Problem matrices and vectors
            method: Distribution method (domain_decomposition, matrix_splitting)
            
        Returns:
            List of sub-problems for each GPU
        """
        if method == "domain_decomposition":
            return self._domain_decomposition(problem_data)
        elif method == "matrix_splitting":
            return self._matrix_splitting(problem_data)
        else:
            raise ValueError(f"Unknown distribution method: {method}")
    
    def _domain_decomposition(self, problem_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Decompose domain spatially across GPUs."""
        sub_problems = []
        
        # Get problem size
        if 'matrix' in problem_data:
            problem_size = problem_data['matrix'].shape[0]
        elif 'solution' in problem_data:
            problem_size = len(problem_data['solution'])
        else:
            raise ValueError("Cannot determine problem size")
        
        # Calculate subdomain sizes
        subdomain_size = problem_size // self.num_gpus
        overlap = max(1, subdomain_size // 10)  # 10% overlap
        
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * subdomain_size
            end_idx = min((gpu_id + 1) * subdomain_size + overlap, problem_size)
            
            if gpu_id == self.num_gpus - 1:
                end_idx = problem_size  # Last GPU gets remainder
            
            sub_problem = {
                'gpu_id': self.available_gpus[gpu_id],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': end_idx - start_idx,
                'overlap': overlap if gpu_id < self.num_gpus - 1 else 0
            }
            
            # Extract subdomain data
            for key, data in problem_data.items():
                if isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        sub_problem[key] = data[start_idx:end_idx]
                    elif data.ndim == 2:
                        sub_problem[key] = data[start_idx:end_idx, start_idx:end_idx]
                    else:
                        sub_problem[key] = data  # Copy full data for higher dims
            
            sub_problems.append(sub_problem)
        
        return sub_problems
    
    def _matrix_splitting(self, problem_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Split matrix operations across GPUs."""
        sub_problems = []
        
        if 'matrix' not in problem_data:
            raise ValueError("Matrix splitting requires 'matrix' in problem_data")
        
        matrix = problem_data['matrix']
        rows_per_gpu = matrix.shape[0] // self.num_gpus
        
        for gpu_id in range(self.num_gpus):
            start_row = gpu_id * rows_per_gpu
            end_row = (gpu_id + 1) * rows_per_gpu
            
            if gpu_id == self.num_gpus - 1:
                end_row = matrix.shape[0]  # Last GPU gets remainder
            
            sub_problem = {
                'gpu_id': self.available_gpus[gpu_id],
                'matrix': matrix[start_row:end_row, :],
                'row_start': start_row,
                'row_end': end_row
            }
            
            # Copy other data
            for key, data in problem_data.items():
                if key != 'matrix' and isinstance(data, np.ndarray):
                    sub_problem[key] = data.copy()
            
            sub_problems.append(sub_problem)
        
        return sub_problems
    
    def solve_distributed(
        self,
        sub_problems: List[Dict[str, Any]],
        solver_func: Callable,
        communication_rounds: int = 10
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve distributed problem with inter-GPU communication.
        
        Args:
            sub_problems: List of sub-problems for each GPU
            solver_func: Solver function to apply to each sub-problem
            communication_rounds: Number of communication/synchronization rounds
            
        Returns:
            Tuple of (global_solution, solve_info)
        """
        start_time = time.perf_counter()
        
        # Initialize solutions on each GPU
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit initial solves
            futures = []
            for sub_problem in sub_problems:
                future = executor.submit(self._solve_subproblem, solver_func, sub_problem)
                futures.append(future)
            
            # Collect initial solutions
            sub_solutions = [future.result() for future in futures]
        
        # Communication and refinement rounds
        for round_num in range(communication_rounds):
            # Update boundary conditions based on neighboring solutions
            self._update_boundary_conditions(sub_problems, sub_solutions)
            
            # Refine solutions
            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                futures = []
                for i, sub_problem in enumerate(sub_problems):
                    sub_problem['initial_guess'] = sub_solutions[i]
                    future = executor.submit(self._solve_subproblem, solver_func, sub_problem)
                    futures.append(future)
                
                sub_solutions = [future.result() for future in futures]
            
            # Check global convergence
            if self._check_global_convergence(sub_solutions):
                self.logger.info(f"Distributed solver converged after {round_num + 1} rounds")
                break
        
        # Assemble global solution
        global_solution = self._assemble_global_solution(sub_problems, sub_solutions)
        
        solve_time = time.perf_counter() - start_time
        
        solve_info = {
            'method': 'multi_gpu_distributed',
            'num_gpus': self.num_gpus,
            'communication_rounds': round_num + 1,
            'solve_time': solve_time
        }
        
        return global_solution, solve_info
    
    def _solve_subproblem(self, solver_func: Callable, sub_problem: Dict[str, Any]) -> np.ndarray:
        """Solve individual sub-problem on assigned GPU."""
        gpu_id = sub_problem['gpu_id']
        
        # Set GPU context
        if HAS_CUPY:
            with cp.cuda.Device(gpu_id):
                return solver_func(sub_problem)
        elif HAS_TORCH:
            with torch.cuda.device(gpu_id):
                return solver_func(sub_problem)
        else:
            return solver_func(sub_problem)
    
    def _update_boundary_conditions(
        self,
        sub_problems: List[Dict[str, Any]],
        sub_solutions: List[np.ndarray]
    ):
        """Update boundary conditions for domain decomposition."""
        for i in range(len(sub_problems)):
            # Update left boundary (from previous subdomain)
            if i > 0:
                overlap_size = sub_problems[i].get('overlap', 0)
                if overlap_size > 0:
                    # Use solution from previous subdomain
                    left_boundary = sub_solutions[i-1][-overlap_size:]
                    sub_problems[i]['left_boundary'] = left_boundary
            
            # Update right boundary (from next subdomain)
            if i < len(sub_problems) - 1:
                overlap_size = sub_problems[i].get('overlap', 0)
                if overlap_size > 0:
                    # Use solution from next subdomain
                    right_boundary = sub_solutions[i+1][:overlap_size]
                    sub_problems[i]['right_boundary'] = right_boundary
    
    def _check_global_convergence(self, sub_solutions: List[np.ndarray]) -> bool:
        """Check if global solution has converged."""
        # Simple convergence check based on boundary matching
        if len(sub_solutions) < 2:
            return True
        
        for i in range(len(sub_solutions) - 1):
            # Check overlap region convergence
            left_solution = sub_solutions[i]
            right_solution = sub_solutions[i + 1]
            
            overlap_size = min(10, len(left_solution) // 10, len(right_solution) // 10)
            if overlap_size > 0:
                left_boundary = left_solution[-overlap_size:]
                right_boundary = right_solution[:overlap_size]
                
                boundary_error = np.linalg.norm(left_boundary - right_boundary)
                if boundary_error > 1e-4:  # Convergence threshold
                    return False
        
        return True
    
    def _assemble_global_solution(
        self,
        sub_problems: List[Dict[str, Any]],
        sub_solutions: List[np.ndarray]
    ) -> np.ndarray:
        """Assemble global solution from sub-solutions."""
        # Calculate total size
        total_size = sum(sub_problem['size'] for sub_problem in sub_problems)
        
        # Account for overlaps
        for sub_problem in sub_problems[:-1]:  # All except last
            total_size -= sub_problem.get('overlap', 0)
        
        global_solution = np.zeros(total_size)
        current_pos = 0
        
        for i, (sub_problem, sub_solution) in enumerate(zip(sub_problems, sub_solutions)):
            overlap = sub_problem.get('overlap', 0)
            
            if i == len(sub_problems) - 1:
                # Last subdomain - use full solution
                global_solution[current_pos:] = sub_solution
            else:
                # Use solution without overlap region
                solution_size = len(sub_solution) - overlap
                global_solution[current_pos:current_pos + solution_size] = sub_solution[:solution_size]
                current_pos += solution_size
        
        return global_solution


class MixedPrecisionAccelerator:
    """Mixed precision acceleration for analog PDE solving."""
    
    def __init__(self, config: GPUEnhancementConfig):
        """Initialize mixed precision accelerator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check tensor core availability
        self.has_tensor_cores = self._check_tensor_cores()
        
    def _check_tensor_cores(self) -> bool:
        """Check if tensor cores are available."""
        if HAS_TORCH:
            try:
                # Check for tensor core capable GPUs (compute capability >= 7.0)
                for i in range(torch.cuda.device_count()):
                    capability = torch.cuda.get_device_capability(i)
                    if capability[0] >= 7:  # Volta or newer
                        return True
            except Exception:
                pass
        
        return False
    
    def solve_mixed_precision(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        iterations: int = 1000
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using mixed precision (FP16/FP32) acceleration.
        
        Args:
            matrix: System matrix
            rhs: Right-hand side vector
            initial_guess: Initial solution guess
            iterations: Maximum iterations
            
        Returns:
            Tuple of (solution, solve_info)
        """
        start_time = time.perf_counter()
        
        if HAS_TORCH and torch.cuda.is_available():
            return self._solve_pytorch_mixed(matrix, rhs, initial_guess, iterations)
        elif HAS_CUPY:
            return self._solve_cupy_mixed(matrix, rhs, initial_guess, iterations)
        else:
            raise RuntimeError("No suitable GPU framework available for mixed precision")
    
    def _solve_pytorch_mixed(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray],
        iterations: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using PyTorch mixed precision."""
        device = torch.device('cuda')
        
        # Convert to PyTorch tensors
        matrix_tensor = torch.from_numpy(matrix).to(device, dtype=torch.float32)
        rhs_tensor = torch.from_numpy(rhs).to(device, dtype=torch.float32)
        
        if initial_guess is not None:
            x = torch.from_numpy(initial_guess).to(device, dtype=torch.float32)
        else:
            x = torch.zeros(len(rhs), device=device, dtype=torch.float32)
        
        # Use automatic mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.use_mixed_precision else None
        
        residuals = []
        
        for i in range(iterations):
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Cast to FP16 for computation
                    matrix_fp16 = matrix_tensor.half()
                    x_fp16 = x.half()
                    rhs_fp16 = rhs_tensor.half()
                    
                    # Matrix-vector multiplication in FP16
                    residual_fp16 = torch.matmul(matrix_fp16, x_fp16) - rhs_fp16
                    
                    # Convert back to FP32 for update
                    residual = residual_fp16.float()
            else:
                residual = torch.matmul(matrix_tensor, x) - rhs_tensor
            
            # Jacobi update
            x = x - 0.1 * residual
            
            # Check convergence
            residual_norm = torch.norm(residual).item()
            residuals.append(residual_norm)
            
            if residual_norm < 1e-6:
                break
        
        # Convert back to NumPy
        solution = x.cpu().numpy()
        
        solve_time = time.perf_counter() - start_time
        
        return solution, {
            'method': 'pytorch_mixed_precision',
            'iterations': len(residuals),
            'final_residual': residuals[-1] if residuals else 0.0,
            'solve_time': solve_time,
            'used_tensor_cores': self.has_tensor_cores and self.config.use_tensor_cores
        }
    
    def _solve_cupy_mixed(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray],
        iterations: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using CuPy mixed precision."""
        # Convert to CuPy arrays
        matrix_gpu = cp.asarray(matrix, dtype=cp.float32)
        rhs_gpu = cp.asarray(rhs, dtype=cp.float32)
        
        if initial_guess is not None:
            x_gpu = cp.asarray(initial_guess, dtype=cp.float32)
        else:
            x_gpu = cp.zeros(len(rhs), dtype=cp.float32)
        
        residuals = []
        
        for i in range(iterations):
            if self.config.use_mixed_precision:
                # Use FP16 for computation
                matrix_fp16 = matrix_gpu.astype(cp.float16)
                x_fp16 = x_gpu.astype(cp.float16)
                rhs_fp16 = rhs_gpu.astype(cp.float16)
                
                # Matrix-vector multiplication in FP16
                residual_fp16 = cp.dot(matrix_fp16, x_fp16) - rhs_fp16
                
                # Convert back to FP32
                residual = residual_fp16.astype(cp.float32)
            else:
                residual = cp.dot(matrix_gpu, x_gpu) - rhs_gpu
            
            # Update solution
            x_gpu = x_gpu - 0.1 * residual
            
            # Check convergence
            residual_norm = float(cp.linalg.norm(residual))
            residuals.append(residual_norm)
            
            if residual_norm < 1e-6:
                break
        
        # Convert back to NumPy
        solution = cp.asnumpy(x_gpu)
        
        solve_time = time.perf_counter() - start_time
        
        return solution, {
            'method': 'cupy_mixed_precision',
            'iterations': len(residuals),
            'final_residual': residuals[-1] if residuals else 0.0,
            'solve_time': solve_time
        }


class GPUEnhancementSuite:
    """Comprehensive GPU enhancement suite for analog PDE solving."""
    
    def __init__(self, config: GPUEnhancementConfig = None):
        """Initialize GPU enhancement suite."""
        self.config = config or GPUEnhancementConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.multi_gpu = None
        self.mixed_precision = None
        
        if self.config.use_multi_gpu and self.config.num_gpus > 1:
            try:
                self.multi_gpu = MultiGPUManager(self.config.num_gpus)
            except Exception as e:
                self.logger.warning(f"Multi-GPU initialization failed: {e}")
        
        if self.config.use_mixed_precision:
            self.mixed_precision = MixedPrecisionAccelerator(self.config)
        
    def auto_select_acceleration(
        self,
        problem_size: int,
        problem_characteristics: Dict[str, Any]
    ) -> AccelerationType:
        """Automatically select best acceleration method.
        
        Args:
            problem_size: Size of the PDE problem
            problem_characteristics: Problem characteristics
            
        Returns:
            Best acceleration type for this problem
        """
        complexity_score = problem_characteristics.get('complexity', 1.0)
        sparsity = problem_characteristics.get('sparsity', 0.1)
        
        # Decision logic
        if self.multi_gpu and problem_size > 10000 and complexity_score > 0.5:
            return AccelerationType.MULTI_GPU
        elif self.mixed_precision and problem_size > 1000:
            return AccelerationType.MIXED_PRECISION
        elif HAS_JAX:
            return AccelerationType.JAX
        elif HAS_TORCH:
            return AccelerationType.PYTORCH
        elif HAS_CUPY:
            return AccelerationType.CUPY
        else:
            raise RuntimeError("No GPU acceleration available")
    
    def solve_accelerated(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        acceleration_type: Optional[AccelerationType] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using selected GPU acceleration.
        
        Args:
            matrix: System matrix
            rhs: Right-hand side vector
            acceleration_type: Acceleration method (auto-select if None)
            **kwargs: Additional solver parameters
            
        Returns:
            Tuple of (solution, solve_info)
        """
        if acceleration_type is None:
            problem_chars = kwargs.get('problem_characteristics', {})
            acceleration_type = self.auto_select_acceleration(len(rhs), problem_chars)
        
        self.logger.info(f"Using {acceleration_type.value} acceleration")
        
        if acceleration_type == AccelerationType.MULTI_GPU and self.multi_gpu:
            return self._solve_multi_gpu(matrix, rhs, **kwargs)
        elif acceleration_type == AccelerationType.MIXED_PRECISION and self.mixed_precision:
            return self.mixed_precision.solve_mixed_precision(matrix, rhs, **kwargs)
        elif acceleration_type == AccelerationType.JAX and HAS_JAX:
            return self._solve_jax(matrix, rhs, **kwargs)
        elif acceleration_type == AccelerationType.PYTORCH and HAS_TORCH:
            return self._solve_pytorch(matrix, rhs, **kwargs)
        elif acceleration_type == AccelerationType.CUPY and HAS_CUPY:
            return self._solve_cupy(matrix, rhs, **kwargs)
        else:
            raise RuntimeError(f"Acceleration type {acceleration_type} not available")
    
    def _solve_multi_gpu(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using multi-GPU acceleration."""
        problem_data = {'matrix': matrix, 'rhs': rhs}
        sub_problems = self.multi_gpu.distribute_problem(problem_data)
        
        def solver_func(sub_problem):
            # Simple iterative solver for sub-problem
            sub_matrix = sub_problem['matrix']
            sub_rhs = sub_problem['rhs']
            x = np.zeros(len(sub_rhs))
            
            for _ in range(100):  # Fixed iterations for simplicity
                residual = np.dot(sub_matrix, x) - sub_rhs
                x = x - 0.1 * residual
            
            return x
        
        return self.multi_gpu.solve_distributed(sub_problems, solver_func)
    
    def _solve_jax(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        iterations: int = 1000,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using JAX GPU acceleration."""
        start_time = time.perf_counter()
        
        # Convert to JAX arrays on GPU
        matrix_jax = jnp.asarray(matrix)
        rhs_jax = jnp.asarray(rhs)
        x = jnp.zeros_like(rhs_jax)
        
        # JIT compile the solver step
        @jax.jit
        def solver_step(x, matrix, rhs):
            residual = jnp.dot(matrix, x) - rhs
            return x - 0.1 * residual
        
        # Iterative solver
        for i in range(iterations):
            x = solver_step(x, matrix_jax, rhs_jax)
            
            if i % 100 == 0:  # Check convergence periodically
                residual = jnp.dot(matrix_jax, x) - rhs_jax
                residual_norm = jnp.linalg.norm(residual)
                if residual_norm < 1e-6:
                    break
        
        solution = np.asarray(x)
        solve_time = time.perf_counter() - start_time
        
        return solution, {
            'method': 'jax_gpu',
            'iterations': i + 1,
            'solve_time': solve_time
        }
    
    def _solve_pytorch(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        iterations: int = 1000,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using PyTorch GPU acceleration."""
        start_time = time.perf_counter()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        matrix_tensor = torch.from_numpy(matrix).to(device)
        rhs_tensor = torch.from_numpy(rhs).to(device)
        x = torch.zeros_like(rhs_tensor)
        
        for i in range(iterations):
            residual = torch.matmul(matrix_tensor, x) - rhs_tensor
            x = x - 0.1 * residual
            
            if i % 100 == 0:
                residual_norm = torch.norm(residual).item()
                if residual_norm < 1e-6:
                    break
        
        solution = x.cpu().numpy()
        solve_time = time.perf_counter() - start_time
        
        return solution, {
            'method': 'pytorch_gpu',
            'iterations': i + 1,
            'solve_time': solve_time
        }
    
    def _solve_cupy(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
        iterations: int = 1000,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using CuPy GPU acceleration."""
        start_time = time.perf_counter()
        
        matrix_gpu = cp.asarray(matrix)
        rhs_gpu = cp.asarray(rhs)
        x_gpu = cp.zeros_like(rhs_gpu)
        
        for i in range(iterations):
            residual = cp.dot(matrix_gpu, x_gpu) - rhs_gpu
            x_gpu = x_gpu - 0.1 * residual
            
            if i % 100 == 0:
                residual_norm = float(cp.linalg.norm(residual))
                if residual_norm < 1e-6:
                    break
        
        solution = cp.asnumpy(x_gpu)
        solve_time = time.perf_counter() - start_time
        
        return solution, {
            'method': 'cupy_gpu',
            'iterations': i + 1,
            'solve_time': solve_time
        }
    
    def benchmark_accelerations(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray
    ) -> Dict[AccelerationType, Dict[str, Any]]:
        """Benchmark all available acceleration methods.
        
        Args:
            matrix: Test system matrix
            rhs: Test right-hand side vector
            
        Returns:
            Dictionary of benchmark results for each acceleration type
        """
        available_types = []
        
        if HAS_CUPY:
            available_types.append(AccelerationType.CUPY)
        if HAS_TORCH:
            available_types.append(AccelerationType.PYTORCH)
        if HAS_JAX:
            available_types.append(AccelerationType.JAX)
        if self.mixed_precision:
            available_types.append(AccelerationType.MIXED_PRECISION)
        if self.multi_gpu:
            available_types.append(AccelerationType.MULTI_GPU)
        
        results = {}
        
        for acc_type in available_types:
            try:
                self.logger.info(f"Benchmarking {acc_type.value}")
                solution, solve_info = self.solve_accelerated(matrix, rhs, acc_type)
                
                # Verify solution quality
                residual = np.dot(matrix, solution) - rhs
                solve_info['solution_error'] = np.linalg.norm(residual)
                solve_info['success'] = True
                
                results[acc_type] = solve_info
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {acc_type.value}: {e}")
                results[acc_type] = {
                    'success': False,
                    'error': str(e),
                    'solve_time': float('inf')
                }
        
        return results
    
    def generate_benchmark_report(self, results: Dict[AccelerationType, Dict[str, Any]]) -> str:
        """Generate comprehensive benchmark report.
        
        Args:
            results: Benchmark results from benchmark_accelerations
            
        Returns:
            Formatted benchmark report
        """
        report_lines = [
            "=" * 60,
            "GPU ACCELERATION BENCHMARK REPORT",
            "=" * 60,
            ""
        ]
        
        # Sort by performance (lowest solve time)
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1].get('solve_time', float('inf'))
        )
        
        if sorted_results:
            report_lines.append("🏆 Performance Ranking:")
            for i, (acc_type, result) in enumerate(sorted_results, 1):
                report_lines.append(f"{i}. {acc_type.value.upper()}")
                report_lines.append(f"   Solve Time: {result['solve_time']:.3f}s")
                report_lines.append(f"   Iterations: {result.get('iterations', 'N/A')}")
                report_lines.append(f"   Solution Error: {result.get('solution_error', 'N/A'):.2e}")
                report_lines.append("")
        
        # Failed methods
        failed_results = {k: v for k, v in results.items() if not v.get('success', True)}
        if failed_results:
            report_lines.append("❌ Failed Methods:")
            for acc_type, result in failed_results.items():
                report_lines.append(f"   {acc_type.value}: {result.get('error', 'Unknown error')}")
            report_lines.append("")
        
        # Performance comparison table
        if len(successful_results) > 1:
            baseline_time = min(r['solve_time'] for r in successful_results.values())
            
            report_lines.extend([
                "📊 Speedup Analysis:",
                ""
            ])
            
            for acc_type, result in sorted_results:
                speedup = baseline_time / result['solve_time']
                report_lines.append(f"   {acc_type.value}: {speedup:.2f}× speedup")
        
        report_lines.extend([
            "",
            "=" * 60,
            "Report generated by Terragon Labs GPU Enhancement Suite",
            "=" * 60
        ])
        
        return "\n".join(report_lines)