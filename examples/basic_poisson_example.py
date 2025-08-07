#!/usr/bin/env python3
"""Basic Poisson equation solver example using analog crossbar."""

import numpy as np
import matplotlib.pyplot as plt
from analog_pde_solver import AnalogPDESolver, PoissonEquation


def main():
    """Run basic Poisson equation example."""
    print("Analog PDE Solver - Basic Poisson Example")
    print("=" * 45)
    
    # Define Poisson equation: ∇²φ = -ρ/ε₀
    pde = PoissonEquation(
        domain_size=64,
        boundary_conditions="dirichlet",
        source_function=lambda x, y: np.exp(-(x**2 + y**2))
    )
    
    print(f"Problem size: {pde.domain_size}")
    print(f"Boundary conditions: {pde.boundary_conditions}")
    
    # Create analog solver
    solver = AnalogPDESolver(
        crossbar_size=64,
        conductance_range=(1e-9, 1e-6),
        noise_model="realistic"
    )
    
    print(f"Crossbar size: {solver.crossbar_size}")
    print(f"Conductance range: {solver.conductance_range}")
    
    # Solve using analog method
    print("\nSolving with analog crossbar...")
    analog_solution = solver.solve(
        pde,
        iterations=100,
        convergence_threshold=1e-6
    )
    
    # Compare with digital solution
    print("Computing digital reference...")
    digital_solution = pde.solve_digital()
    
    # Calculate error metrics
    error = np.mean(np.abs(analog_solution - digital_solution))
    rel_error = error / np.mean(np.abs(digital_solution))
    
    print(f"\nResults:")
    print(f"  Mean absolute error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2%}")
    print(f"  Analog solution norm: {np.linalg.norm(analog_solution):.4f}")
    print(f"  Digital solution norm: {np.linalg.norm(digital_solution):.4f}")
    
    # Plot comparison
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        x = np.linspace(0, 1, len(digital_solution))
        
        ax1.plot(x, digital_solution, 'b-', label='Digital', linewidth=2)
        ax1.set_title('Digital Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('φ(x)')
        ax1.grid(True)
        
        ax2.plot(x, analog_solution, 'r-', label='Analog', linewidth=2)
        ax2.set_title('Analog Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('φ(x)')
        ax2.grid(True)
        
        error_plot = np.abs(analog_solution - digital_solution)
        ax3.plot(x, error_plot, 'g-', linewidth=2)
        ax3.set_title('Absolute Error')
        ax3.set_xlabel('x')
        ax3.set_ylabel('|φ_analog - φ_digital|')
        ax3.grid(True)
        ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('poisson_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as 'poisson_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()