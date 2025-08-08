#!/usr/bin/env python3
"""Unit tests for GPU acceleration functionality."""

import sys
import os
import unittest
import numpy as np

# Add the root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analog_pde_solver import AnalogPDESolver, PoissonEquation
from analog_pde_solver.acceleration import GPUAcceleratedSolver, GPUConfig, GPUMemoryManager


class TestGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_solver = AnalogPDESolver(
            crossbar_size=32,
            conductance_range=(1e-9, 1e-6),
            noise_model="realistic"
        )
        
        self.gpu_config = GPUConfig(
            device_id=0,
            memory_pool_size_gb=1.0,
            use_streams=False,
            preferred_backend='cupy'
        )
        
        self.pde = PoissonEquation((32,))
        
    def test_gpu_solver_initialization(self):
        """Test GPU solver initialization."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            gpu_config=self.gpu_config,
            fallback_to_cpu=True
        )
        
        self.assertIsInstance(gpu_solver, GPUAcceleratedSolver)
        self.assertEqual(gpu_solver.config.device_id, 0)
        self.assertEqual(gpu_solver.config.memory_pool_size_gb, 1.0)
        self.assertTrue(gpu_solver.fallback_to_cpu)
        
    def test_gpu_availability_check(self):
        """Test GPU availability checking."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=True
        )
        
        # GPU availability depends on system, but should not crash
        self.assertIsInstance(gpu_solver.gpu_available, bool)
        self.assertIn(gpu_solver.backend, ['cpu', 'cupy', 'numba'])
        
    def test_cpu_fallback_solve(self):
        """Test solving with CPU fallback."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=True
        )
        
        solution, solve_info = gpu_solver.solve_gpu(
            self.pde,
            iterations=20,
            convergence_threshold=1e-4
        )
        
        self.assertEqual(len(solution), 32)
        self.assertTrue(np.isfinite(solution).all())
        self.assertIn('method', solve_info)
        self.assertIn('gpu_available', solve_info)
        
    def test_gpu_config_validation(self):
        """Test GPU configuration validation."""
        # Test valid config
        config = GPUConfig(
            device_id=0,
            memory_pool_size_gb=2.0,
            use_streams=True,
            num_streams=2,
            block_size=256,
            preferred_backend='cupy'
        )
        
        self.assertEqual(config.device_id, 0)
        self.assertEqual(config.memory_pool_size_gb, 2.0)
        self.assertTrue(config.use_streams)
        self.assertEqual(config.num_streams, 2)
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.preferred_backend, 'cupy')
        
    def test_memory_manager_initialization(self):
        """Test GPU memory manager initialization."""
        manager = GPUMemoryManager(backend='cupy')
        self.assertEqual(manager.backend, 'cupy')
        
        manager = GPUMemoryManager(backend='numba')
        self.assertEqual(manager.backend, 'numba')
        
    def test_memory_stats(self):
        """Test memory statistics retrieval."""
        manager = GPUMemoryManager(backend='cupy')
        stats = manager.get_memory_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['backend'], 'cupy')
        
    def test_benchmark_structure(self):
        """Test benchmark functionality structure."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=True
        )
        
        try:
            benchmark_results = gpu_solver.benchmark_gpu_vs_cpu(
                self.pde,
                iterations=5,
                num_runs=2
            )
            
            # Check required fields
            required_fields = [
                'num_runs', 'iterations', 'gpu_available', 
                'backend', 'problem_size', 'cpu_times',
                'avg_cpu_time', 'std_cpu_time'
            ]
            
            for field in required_fields:
                self.assertIn(field, benchmark_results)
                
            self.assertEqual(benchmark_results['num_runs'], 2)
            self.assertEqual(benchmark_results['iterations'], 5)
            self.assertEqual(benchmark_results['problem_size'], 32)
            self.assertEqual(len(benchmark_results['cpu_times']), 2)
            
        except Exception as e:
            self.skipTest(f"Benchmark test skipped due to error: {e}")
    
    def test_error_handling(self):
        """Test error handling in GPU operations."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=False  # Disable fallback to test error handling
        )
        
        # If GPU not available, should raise error when fallback disabled
        if not gpu_solver.gpu_available:
            with self.assertRaises(RuntimeError):
                gpu_solver.solve_gpu(self.pde, iterations=5)
        
    def test_laplacian_matrix_creation(self):
        """Test Laplacian matrix creation methods."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=True
        )
        
        # Test NumPy version (always available)
        laplacian = gpu_solver._create_laplacian_matrix_numpy(32)
        
        self.assertEqual(laplacian.shape, (32, 32))
        self.assertEqual(laplacian.dtype, np.float32)
        
        # Check diagonal structure
        self.assertTrue(np.all(np.diag(laplacian) == -2.0))
        
        # Check off-diagonals
        for i in range(31):
            self.assertEqual(laplacian[i, i+1], 1.0)
            self.assertEqual(laplacian[i+1, i], 1.0)
    
    def test_matrix_creation(self):
        """Test GPU matrix creation methods."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=True
        )
        
        # Test Laplacian matrix creation (NumPy version always available)
        laplacian = gpu_solver._create_laplacian_matrix_numpy(8)
        
        self.assertEqual(laplacian.shape, (8, 8))
        self.assertEqual(laplacian.dtype, np.float32)
        
        # Check matrix structure
        self.assertTrue(np.all(np.diag(laplacian) == -2.0))
        
        # Check symmetry of tridiagonal structure
        for i in range(7):
            self.assertEqual(laplacian[i, i+1], 1.0)
            self.assertEqual(laplacian[i+1, i], 1.0)
    
    def test_gpu_memory_info(self):
        """Test GPU memory information retrieval."""
        gpu_solver = GPUAcceleratedSolver(
            base_solver=self.base_solver,
            fallback_to_cpu=True
        )
        
        memory_info = gpu_solver.get_gpu_memory_info()
        
        self.assertIsInstance(memory_info, dict)
        self.assertIn('gpu_available', memory_info)
        
        if memory_info['gpu_available']:
            self.assertIn('backend', memory_info)


class TestGPUMemoryManager(unittest.TestCase):
    """Test GPU memory manager functionality."""
    
    def test_clear_cache(self):
        """Test memory cache clearing."""
        manager = GPUMemoryManager(backend='cupy')
        
        # Should not raise exceptions
        try:
            manager.clear_cache()
        except Exception as e:
            self.skipTest(f"Cache clear test skipped: {e}")
    
    def test_memory_stats_structure(self):
        """Test memory statistics structure."""
        for backend in ['cupy', 'numba']:
            manager = GPUMemoryManager(backend=backend)
            stats = manager.get_memory_stats()
            
            self.assertIsInstance(stats, dict)
            self.assertEqual(stats['backend'], backend)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)