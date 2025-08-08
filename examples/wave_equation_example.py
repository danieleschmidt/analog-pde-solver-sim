#!/usr/bin/env python3
"""Wave equation solver example using analog crossbar."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from analog_pde_solver import AnalogPDESolver, WaveEquation


def main():
    """Run wave equation example."""
    print("Analog PDE Solver - Wave Equation Example")
    print("=" * 42)
    
    # Define wave equation: ∂²u/∂t² = c²∇²u
    wave_eq = WaveEquation(
        domain_size=(64,),
        wave_speed=1.0,
        time_step=0.01,
        boundary_conditions="dirichlet"
    )
    
    print(f"Problem size: {wave_eq.domain_size}")
    print(f"Wave speed: {wave_eq.wave_speed}")
    print(f"Time step: {wave_eq.time_step}")
    
    # Initialize with Gaussian pulse
    def initial_displacement(x):
        return np.exp(-((x - 0.3) / 0.05)**2)
    
    def initial_velocity(x):
        return 0.0  # Start from rest
    
    u_current, u_previous = wave_eq.initialize_field(
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity
    )
    
    print(f"Initial displacement range: [{u_current.min():.4f}, {u_current.max():.4f}]")
    
    # Create analog solver  
    solver = AnalogPDESolver(
        crossbar_size=64,
        conductance_range=(1e-9, 1e-6),
        noise_model="realistic"
    )
    
    print(f"Crossbar size: {solver.crossbar_size}")
    
    # Calculate CFL number for stability
    dx = 1.0 / 64
    cfl = wave_eq.wave_speed * wave_eq.time_step / dx
    print(f"CFL number: {cfl:.3f} (should be ≤ 1.0 for stability)")
    
    # Solve using both methods for multiple time steps
    print("\nSolving with wave propagation...")
    
    wave_states = []
    for step in range(50):
        # Update wave equation one time step
        u_digital = wave_eq.step()
        
        # Also solve with analog solver (approximate)
        analog_solution = solver.solve(wave_eq, iterations=5)
        
        wave_states.append({
            'step': step,
            'digital': u_digital.copy(),
            'analog': analog_solution.copy(),
            'time': step * wave_eq.time_step
        })
        
        if step % 10 == 0:
            max_displacement = np.max(np.abs(u_digital))
            analog_max = np.max(np.abs(analog_solution))
            print(f"  Step {step:2d}: Max |u| = {max_displacement:.6f} (digital), {analog_max:.6f} (analog)")
    
    # Compare final states
    final_digital = wave_states[-1]['digital']
    final_analog = wave_states[-1]['analog']
    
    error = np.mean(np.abs(final_analog - final_digital))
    rel_error = error / (np.mean(np.abs(final_digital)) + 1e-12)
    
    print(f"\nFinal Results:")
    print(f"  Mean absolute error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2%}")
    print(f"  Digital wave energy: {np.sum(final_digital**2):.4f}")
    print(f"  Analog wave energy: {np.sum(final_analog**2):.4f}")
    
    # Analyze wave propagation
    initial_energy = np.sum(wave_states[0]['digital']**2)
    final_energy = np.sum(wave_states[-1]['digital']**2)
    energy_conservation = final_energy / initial_energy
    
    print(f"\nWave Propagation Analysis:")
    print(f"  Initial wave energy: {initial_energy:.4f}")
    print(f"  Final wave energy: {final_energy:.4f}")
    print(f"  Energy conservation: {energy_conservation:.4f} (should be ~1.0)")
    
    # Find pulse location over time
    initial_center = np.argmax(np.abs(wave_states[0]['digital'])) / 64
    final_center = np.argmax(np.abs(wave_states[-1]['digital'])) / 64
    distance_traveled = abs(final_center - initial_center)
    
    print(f"  Initial pulse center: x = {initial_center:.3f}")
    print(f"  Final pulse center: x = {final_center:.3f}")
    print(f"  Distance traveled: {distance_traveled:.3f}")
    
    print("\nWave equation example completed successfully!")


if __name__ == "__main__":
    main()