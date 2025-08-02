"""
Performance benchmarking tests for analog PDE solver.
"""
import pytest
import time
import psutil
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any

# Import the performance fixtures
from tests.fixtures.pde_fixtures import performance_benchmarks


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        
    @contextmanager
    def monitor(self):
        """Context manager for performance monitoring."""
        # Record start metrics
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        self.peak_memory = self.start_memory
        
        try:
            yield self
        finally:
            # Record end metrics
            self.end_time = time.perf_counter()
            self.end_memory = psutil.Process().memory_info().rss
            self.peak_memory = max(self.peak_memory, self.end_memory)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def memory_usage(self) -> int:
        """Get peak memory usage in bytes."""
        return self.peak_memory - self.start_memory if self.peak_memory and self.start_memory else 0


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    def test_small_problem_performance(self, performance_benchmarks):
        """Test performance on small problems."""
        benchmark = performance_benchmarks["small"]
        monitor = PerformanceMonitor()
        
        with monitor.monitor():
            # Simulate analog PDE solving
            grid_size = benchmark["grid_size"]
            matrix = np.random.random(grid_size)
            
            # Simulate iterative solving
            for _ in range(100):
                matrix = 0.25 * (
                    np.roll(matrix, 1, axis=0) +
                    np.roll(matrix, -1, axis=0) +
                    np.roll(matrix, 1, axis=1) +
                    np.roll(matrix, -1, axis=1)
                )
        
        # Verify performance constraints
        assert monitor.elapsed_time < benchmark["max_time"], \
            f"Small problem took {monitor.elapsed_time:.2f}s, expected < {benchmark['max_time']}s"
        
        assert monitor.memory_usage < benchmark["max_memory"], \
            f"Small problem used {monitor.memory_usage} bytes, expected < {benchmark['max_memory']} bytes"
    
    @pytest.mark.slow
    def test_medium_problem_performance(self, performance_benchmarks):
        """Test performance on medium problems."""
        benchmark = performance_benchmarks["medium"]
        monitor = PerformanceMonitor()
        
        with monitor.monitor():
            grid_size = benchmark["grid_size"]
            matrix = np.random.random(grid_size)
            
            # Simulate more intensive computation
            for _ in range(50):
                matrix = 0.25 * (
                    np.roll(matrix, 1, axis=0) +
                    np.roll(matrix, -1, axis=0) +
                    np.roll(matrix, 1, axis=1) +
                    np.roll(matrix, -1, axis=1)
                )
        
        assert monitor.elapsed_time < benchmark["max_time"]
        assert monitor.memory_usage < benchmark["max_memory"]
    
    @pytest.mark.slow
    @pytest.mark.hardware
    def test_large_problem_performance(self, performance_benchmarks):
        """Test performance on large problems (requires hardware acceleration)."""
        benchmark = performance_benchmarks["large"]
        monitor = PerformanceMonitor()
        
        with monitor.monitor():
            grid_size = benchmark["grid_size"]
            # For large problems, we'd typically use hardware acceleration
            matrix = np.random.random(grid_size)
            
            # Simulate hardware-accelerated computation
            for _ in range(10):  # Fewer iterations due to hardware speedup
                matrix = 0.25 * (
                    np.roll(matrix, 1, axis=0) +
                    np.roll(matrix, -1, axis=0) +
                    np.roll(matrix, 1, axis=1) +
                    np.roll(matrix, -1, axis=1)
                )
        
        assert monitor.elapsed_time < benchmark["max_time"]
        assert monitor.memory_usage < benchmark["max_memory"]
    
    def test_convergence_performance(self):
        """Test convergence rate performance."""
        grid_size = (128, 128)
        tolerance = 1e-6
        max_iterations = 1000
        
        # Create test problem
        solution = np.random.random(grid_size)
        target = np.zeros_like(solution)
        monitor = PerformanceMonitor()
        
        with monitor.monitor():
            for iteration in range(max_iterations):
                old_solution = solution.copy()
                
                # Gauss-Seidel iteration
                solution[1:-1, 1:-1] = 0.25 * (
                    solution[0:-2, 1:-1] +  # up
                    solution[2:, 1:-1] +    # down
                    solution[1:-1, 0:-2] +  # left
                    solution[1:-1, 2:]      # right
                )
                
                # Check convergence
                residual = np.max(np.abs(solution - old_solution))
                if residual < tolerance:
                    break
        
        # Verify reasonable convergence
        assert iteration < max_iterations * 0.8, \
            f"Convergence took {iteration} iterations, expected < {max_iterations * 0.8}"
        
        # Verify performance
        time_per_iteration = monitor.elapsed_time / (iteration + 1)
        assert time_per_iteration < 0.1, \
            f"Average time per iteration: {time_per_iteration:.3f}s, expected < 0.1s"


@pytest.mark.performance
def test_memory_scaling():
    """Test memory usage scaling with problem size."""
    sizes = [(32, 32), (64, 64), (128, 128)]
    memory_usage = []
    
    for size in sizes:
        monitor = PerformanceMonitor()
        
        with monitor.monitor():
            # Allocate arrays similar to analog solver
            matrix = np.random.random(size)
            conductance_pos = np.random.random(size)
            conductance_neg = np.random.random(size)
            
            # Simulate some computation
            result = matrix @ conductance_pos.T - matrix @ conductance_neg.T
        
        memory_usage.append(monitor.memory_usage)
    
    # Verify roughly linear scaling (within 2x tolerance)
    for i in range(1, len(memory_usage)):
        size_ratio = (sizes[i][0] * sizes[i][1]) / (sizes[i-1][0] * sizes[i-1][1])
        memory_ratio = memory_usage[i] / memory_usage[i-1]
        
        assert memory_ratio < size_ratio * 2, \
            f"Memory scaling worse than expected: {memory_ratio:.2f} vs {size_ratio:.2f}"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])