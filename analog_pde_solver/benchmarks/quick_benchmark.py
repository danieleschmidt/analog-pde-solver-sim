"""Quick benchmark implementation for analog PDE solver."""

import time
import numpy as np
from ..core.solver import AnalogPDESolver
from ..core.equations import PoissonEquation


def run_poisson_benchmark(size: int = 64) -> dict:
    """Run basic Poisson equation benchmark."""
    # Create test problem
    pde = PoissonEquation(
        domain_size=size,
        boundary_conditions="dirichlet",
        source_function=lambda x, y: np.sin(np.pi * x)
    )
    
    # Digital solution timing
    start_time = time.time()
    digital_solution = pde.solve_digital()
    digital_time = time.time() - start_time
    
    # Analog solution timing
    solver = AnalogPDESolver(
        crossbar_size=size,
        conductance_range=(1e-9, 1e-6),
        noise_model="realistic"
    )
    
    start_time = time.time()
    analog_solution = solver.solve(pde, iterations=50)
    analog_time = time.time() - start_time
    
    # Calculate error
    error = np.mean(np.abs(analog_solution - digital_solution))
    
    return {
        "problem_size": size,
        "digital_time": digital_time,
        "analog_time": analog_time,
        "speedup": digital_time / analog_time if analog_time > 0 else float('inf'),
        "mean_absolute_error": error,
        "digital_solution_norm": np.linalg.norm(digital_solution),
        "analog_solution_norm": np.linalg.norm(analog_solution)
    }


def run_quick_benchmarks() -> dict:
    """Run suite of quick benchmarks."""
    results = []
    sizes = [32, 64, 128]
    
    print("Running analog PDE solver benchmarks...")
    
    for size in sizes:
        print(f"  Testing size {size}x{size}...")
        result = run_poisson_benchmark(size)
        results.append(result)
        
        print(f"    Digital: {result['digital_time']:.4f}s")
        print(f"    Analog:  {result['analog_time']:.4f}s")
        print(f"    Speedup: {result['speedup']:.2f}x")
        print(f"    Error:   {result['mean_absolute_error']:.2e}")
    
    return {"benchmark_results": results}


if __name__ == "__main__":
    results = run_quick_benchmarks()
    print("\nBenchmark Summary:")
    for result in results["benchmark_results"]:
        print(f"Size {result['problem_size']}: {result['speedup']:.2f}x speedup")