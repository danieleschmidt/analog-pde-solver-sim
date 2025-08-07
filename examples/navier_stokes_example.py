#!/usr/bin/env python3
"""Navier-Stokes flow simulation using analog crossbar arrays."""

import numpy as np
from analog_pde_solver import NavierStokesAnalog


def main():
    """Run Navier-Stokes simulation example."""
    print("Analog PDE Solver - Navier-Stokes Example")
    print("=" * 42)
    
    # Create Navier-Stokes solver
    ns_solver = NavierStokesAnalog(
        resolution=(64, 64),
        reynolds_number=1000,
        time_step=0.001
    )
    
    print(f"Resolution: {ns_solver.resolution}")
    print(f"Reynolds number: {ns_solver.reynolds_number}")
    print(f"Time step: {ns_solver.time_step}")
    
    # Configure analog hardware
    ns_solver.configure_hardware(
        num_crossbars=4,
        precision_bits=8,
        update_scheme="semi-implicit"
    )
    
    print(f"Configured {len(ns_solver.crossbars)} crossbar arrays")
    print(f"Precision: {ns_solver.precision_bits} bits")
    
    # Initialize flow field
    print("\nInitializing flow field...")
    
    # Add initial perturbation
    center_x, center_y = 32, 32
    sigma = 10
    x, y = np.meshgrid(np.arange(64), np.arange(64))
    
    # Gaussian velocity perturbation
    initial_u = 0.1 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    ns_solver.velocity_u = initial_u
    
    print("Initial conditions set")
    
    # Run fluid simulation
    print("\nRunning fluid simulation...")
    max_timesteps = 20
    
    for timestep in range(max_timesteps):
        # Update velocity field
        u, v = ns_solver.update_velocity()
        
        # Update pressure (Poisson solve)
        pressure = ns_solver.solve_pressure_poisson()
        
        # Apply pressure correction
        u, v = ns_solver.apply_pressure_correction(u, v, pressure)
        
        # Visualize every 5 steps
        if timestep % 5 == 0:
            print(f"\nTimestep {timestep}:")
            ns_solver.visualize_flow(u, v, pressure)
    
    # Analyze power consumption
    print("\nPower Analysis:")
    power_analysis = ns_solver.analyze_power()
    print(f"  Average power: {power_analysis.avg_power_mw:.2f} mW")
    print(f"  Energy per iteration: {power_analysis.energy_per_iter_nj:.2f} nJ")
    
    # Calculate energy efficiency
    digital_power_estimate = 50.0  # mW (typical GPU power for similar problem)
    efficiency_gain = digital_power_estimate / power_analysis.avg_power_mw
    
    print(f"  Estimated digital power: {digital_power_estimate:.1f} mW")
    print(f"  Energy efficiency gain: {efficiency_gain:.1f}x")
    
    print("\nNavier-Stokes simulation completed!")
    print("Analog crossbar arrays successfully simulated fluid dynamics")


if __name__ == "__main__":
    main()