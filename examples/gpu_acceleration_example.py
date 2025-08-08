#!/usr/bin/env python3
"""Example demonstrating GPU acceleration for analog PDE solving."""

import sys
import os
import numpy as np

# Add the root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analog_pde_solver import AnalogPDESolver, PoissonEquation
from analog_pde_solver.acceleration import GPUAcceleratedSolver, GPUConfig


def main():
    """Demonstrate GPU acceleration capabilities."""
    print("=== Analog PDE Solver - GPU Acceleration Demo ===\n")
    
    # Create base solver
    base_solver = AnalogPDESolver(
        crossbar_size=128,
        conductance_range=(1e-9, 1e-6),
        noise_model="realistic"
    )
    
    # Create GPU configuration
    gpu_config = GPUConfig(
        device_id=0,
        memory_pool_size_gb=2.0,
        use_streams=True,
        num_streams=2,
        preferred_backend='cupy'
    )
    
    # Create GPU-accelerated solver
    print("Initializing GPU-accelerated solver...")
    gpu_solver = GPUAcceleratedSolver(
        base_solver=base_solver,
        gpu_config=gpu_config,
        fallback_to_cpu=True
    )
    
    print(f"GPU available: {gpu_solver.gpu_available}")
    print(f"Selected backend: {gpu_solver.backend}")
    
    # Create test PDE
    pde = PoissonEquation(
        domain_size=(128,),
        boundary_conditions="dirichlet"
    )
    
    def source_function(x, t):
        """Source function for Poisson equation."""
        return np.sin(2 * np.pi * x) * np.exp(-x)
    
    pde.source_function = source_function
    
    print(f"\nSolving Poisson equation on {base_solver.crossbar_size}x{base_solver.crossbar_size} grid")
    print("Source function: sin(2πx) * exp(-x)")
    
    # Solve with GPU acceleration
    print("\n--- GPU Accelerated Solve ---")
    try:
        solution_gpu, solve_info_gpu = gpu_solver.solve_gpu(
            pde,
            iterations=100,
            convergence_threshold=1e-6
        )
        
        print(f"Method used: {solve_info_gpu['method']}")
        print(f"Solve time: {solve_info_gpu.get('solve_time', 'N/A'):.4f} seconds")
        print(f"Solution norm: {np.linalg.norm(solution_gpu):.6f}")
        print(f"Solution range: [{np.min(solution_gpu):.6f}, {np.max(solution_gpu):.6f}]")
        
    except Exception as e:
        print(f"GPU solve failed: {e}")
        solution_gpu = None
    
    # Compare with CPU solve for validation
    print("\n--- CPU Reference Solve ---")
    try:
        solution_cpu = base_solver.solve(
            pde,
            iterations=100,
            convergence_threshold=1e-6
        )
        
        print(f"CPU solution norm: {np.linalg.norm(solution_cpu):.6f}")
        print(f"CPU solution range: [{np.min(solution_cpu):.6f}, {np.max(solution_cpu):.6f}]")
        
        # Compare solutions if both available
        if solution_gpu is not None:
            difference = np.linalg.norm(solution_gpu - solution_cpu)
            relative_error = difference / np.linalg.norm(solution_cpu)
            print(f"\nGPU vs CPU difference: {difference:.8f}")
            print(f"Relative error: {relative_error:.8f}")
            
            if relative_error < 0.01:
                print("✅ GPU and CPU solutions match (< 1% error)")
            else:
                print("⚠️  GPU and CPU solutions differ significantly")
        
    except Exception as e:
        print(f"CPU solve failed: {e}")
    
    # Benchmark performance if GPU available
    if gpu_solver.gpu_available:
        print("\n--- Performance Benchmark ---")
        try:
            benchmark_results = gpu_solver.benchmark_gpu_vs_cpu(
                pde,
                iterations=50,
                num_runs=3
            )
            
            print(f"Problem size: {benchmark_results['problem_size']}x{benchmark_results['problem_size']}")
            print(f"Iterations per solve: {benchmark_results['iterations']}")
            print(f"Number of benchmark runs: {benchmark_results['num_runs']}")
            print()
            print(f"Average CPU time: {benchmark_results['avg_cpu_time']:.4f} ± {benchmark_results['std_cpu_time']:.4f} s")
            
            if benchmark_results['speedup']:
                print(f"Average GPU time: {benchmark_results['avg_gpu_time']:.4f} ± {benchmark_results['std_gpu_time']:.4f} s")
                print(f"Speedup: {benchmark_results['speedup']:.2f}x")
                
                if benchmark_results['speedup'] > 1.0:
                    print("🚀 GPU acceleration successful!")
                else:
                    print("📊 GPU overhead higher than CPU for this problem size")
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    # Display GPU memory information
    print("\n--- GPU Memory Information ---")
    memory_info = gpu_solver.get_gpu_memory_info()
    
    if memory_info['gpu_available']:
        print(f"Backend: {memory_info['backend']}")
        
        if 'used_bytes' in memory_info:
            used_mb = memory_info['used_bytes'] / (1024**2)
            total_mb = memory_info.get('total_bytes', 0) / (1024**2)
            limit_mb = memory_info.get('limit_bytes', 0) / (1024**2)
            
            print(f"Used memory: {used_mb:.2f} MB")
            print(f"Total allocated: {total_mb:.2f} MB") 
            print(f"Memory limit: {limit_mb:.2f} MB")
            print(f"Free blocks: {memory_info.get('n_free_blocks', 0)}")
        else:
            print("Detailed memory info not available")
    else:
        print("GPU not available")
    
    print("\n=== GPU Acceleration Demo Complete ===")


if __name__ == "__main__":
    main()