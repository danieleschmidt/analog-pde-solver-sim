#!/usr/bin/env python3
"""Heat equation solver example using analog crossbar."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from analog_pde_solver import AnalogPDESolver, HeatEquation


def main():
    """Run heat equation example."""
    print("Analog PDE Solver - Heat Equation Example")
    print("=" * 42)
    
    # Define heat equation: ∂T/∂t = α∇²T
    heat_eq = HeatEquation(
        domain_size=(64,),
        thermal_diffusivity=0.1,
        time_step=0.001,
        boundary_conditions="dirichlet"
    )
    
    print(f"Problem size: {heat_eq.domain_size}")
    print(f"Thermal diffusivity: {heat_eq.thermal_diffusivity}")
    print(f"Time step: {heat_eq.time_step}")
    
    # Initialize with Gaussian temperature profile
    def initial_temp(x):
        return np.exp(-((x - 0.5) / 0.1)**2)
    
    heat_eq.initialize_field(initial_condition=initial_temp)
    
    print(f"Initial temperature range: [{heat_eq.temperature_field.min():.4f}, {heat_eq.temperature_field.max():.4f}]")
    
    # Create analog solver  
    solver = AnalogPDESolver(
        crossbar_size=64,
        conductance_range=(1e-9, 1e-6),
        noise_model="realistic"
    )
    
    print(f"Crossbar size: {solver.crossbar_size}")
    
    # Solve using analog method for multiple time steps
    print("\nSolving with analog crossbar...")
    
    temperatures = []
    for step in range(20):
        # Update heat equation one time step
        temp = heat_eq.step()
        
        # Also solve with analog solver
        analog_solution = solver.solve(heat_eq, iterations=10)
        
        temperatures.append({
            'step': step,
            'digital': temp.copy(),
            'analog': analog_solution.copy(),
            'time': step * heat_eq.time_step
        })
        
        if step % 5 == 0:
            print(f"  Step {step:2d}: Digital max={temp.max():.6f}, Analog max={analog_solution.max():.6f}")
    
    # Compare final states
    final_digital = temperatures[-1]['digital']
    final_analog = temperatures[-1]['analog']
    
    error = np.mean(np.abs(final_analog - final_digital))
    rel_error = error / (np.mean(np.abs(final_digital)) + 1e-12)
    
    print(f"\nFinal Results:")
    print(f"  Mean absolute error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2%}")
    print(f"  Digital temperature norm: {np.linalg.norm(final_digital):.4f}")
    print(f"  Analog temperature norm: {np.linalg.norm(final_analog):.4f}")
    
    # Analyze heat diffusion
    initial_max = temperatures[0]['digital'].max()
    final_max = temperatures[-1]['digital'].max()
    diffusion_ratio = final_max / initial_max
    
    print(f"\nHeat Diffusion Analysis:")
    print(f"  Initial peak temperature: {initial_max:.4f}")
    print(f"  Final peak temperature: {final_max:.4f}")
    print(f"  Diffusion ratio: {diffusion_ratio:.4f}")
    
    print("\nHeat equation example completed successfully!")


if __name__ == "__main__":
    main()