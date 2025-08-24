#!/usr/bin/env python3
"""
Analytical solution verification for analog PDE solver.
Tests against known analytical solutions to validate correctness.
"""

import numpy as np
from analog_pde_solver.core.solver import AnalogPDESolver
from analog_pde_solver.core.equations import PoissonEquation, HeatEquation, WaveEquation


def test_poisson_analytical():
    """Test Poisson equation against analytical solution."""
    print("=== Testing Poisson Equation Against Analytical Solution ===")
    
    # Problem: -d²φ/dx² = π²sin(πx) with φ(0) = φ(1) = 0
    # Analytical solution: φ(x) = sin(πx)
    def analytical_solution(x):
        return np.sin(np.pi * x)
    
    def source_function(x, y):
        return np.pi**2 * np.sin(np.pi * x)
    
    # Test with different grid sizes
    sizes = [8, 16, 32]
    
    for n in sizes:
        print(f"\n--- Testing with {n} grid points ---")
        
        # Create PDE
        pde = PoissonEquation(
            domain_size=(n,),
            source_function=source_function,
            boundary_conditions="dirichlet"
        )
        
        # Analytical solution on grid
        x_grid = np.linspace(0, 1, n)
        phi_analytical = analytical_solution(x_grid)
        
        # Digital solution
        phi_digital = pde.solve_digital()
        
        # Analog solution
        solver = AnalogPDESolver(crossbar_size=n, noise_model="none")
        phi_analog = solver.solve(pde, iterations=100, convergence_threshold=1e-8)
        
        # Compare solutions
        error_digital = np.linalg.norm(phi_digital - phi_analytical)
        error_analog = np.linalg.norm(phi_analog - phi_analytical)
        
        print(f"Analytical solution norm: {np.linalg.norm(phi_analytical):.6f}")
        print(f"Digital solution norm:    {np.linalg.norm(phi_digital):.6f}")
        print(f"Analog solution norm:     {np.linalg.norm(phi_analog):.6f}")
        print(f"Digital vs analytical L2 error: {error_digital:.6f}")
        print(f"Analog vs analytical L2 error:  {error_analog:.6f}")
        print(f"Digital vs analytical rel error: {error_digital/np.linalg.norm(phi_analytical):.6f}")
        print(f"Analog vs analytical rel error:  {error_analog/np.linalg.norm(phi_analytical):.6f}")
        
        # Check if solutions are reasonable
        if error_digital / np.linalg.norm(phi_analytical) < 0.1:
            print("✓ Digital solution is accurate")
        else:
            print("✗ Digital solution has high error")
            
        if error_analog / np.linalg.norm(phi_analytical) < 0.5:
            print("✓ Analog solution is reasonable")
        else:
            print("✗ Analog solution has high error")


def test_heat_analytical():
    """Test heat equation against analytical solution."""
    print("\n=== Testing Heat Equation Against Analytical Solution ===")
    
    # Problem: ∂T/∂t = α∇²T with initial condition T(x,0) = sin(πx)
    # Analytical solution: T(x,t) = exp(-α*π²*t) * sin(πx)
    
    alpha = 0.1
    t_final = 0.1
    
    def analytical_solution(x, t):
        return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
    
    def initial_condition(x):
        return np.sin(np.pi * x)
    
    n = 16
    dt = 0.001
    num_steps = int(t_final / dt)
    
    print(f"Testing with {n} grid points, dt={dt}, {num_steps} time steps")
    
    # Create heat equation
    pde = HeatEquation(
        domain_size=(n,),
        thermal_diffusivity=alpha,
        time_step=dt
    )
    
    # Grid and analytical solution
    x_grid = np.linspace(0, 1, n)
    T_analytical = analytical_solution(x_grid, t_final)
    
    # Digital solution
    pde.initialize_field(initial_condition)
    for _ in range(num_steps):
        pde.step()
    T_digital = pde.temperature_field
    
    # Simple analog solution test (just check it runs)
    solver = AnalogPDESolver(crossbar_size=n, noise_model="none")
    T_analog = solver.solve(pde, iterations=50)
    
    # Compare
    error_digital = np.linalg.norm(T_digital - T_analytical)
    error_analog = np.linalg.norm(T_analog - T_analytical)
    
    print(f"Analytical solution norm: {np.linalg.norm(T_analytical):.6f}")
    print(f"Digital solution norm:    {np.linalg.norm(T_digital):.6f}")
    print(f"Analog solution norm:     {np.linalg.norm(T_analog):.6f}")
    print(f"Digital vs analytical error: {error_digital:.6f}")
    print(f"Analog vs analytical error:  {error_analog:.6f}")
    
    if T_analytical.max() > 0:
        print(f"Digital relative error: {error_digital/np.linalg.norm(T_analytical):.6f}")
        print(f"Analog relative error:  {error_analog/np.linalg.norm(T_analytical):.6f}")


def test_simple_validation():
    """Simple validation tests to verify basic correctness."""
    print("\n=== Simple Validation Tests ===")
    
    # Test 1: Zero source should give zero solution
    print("\n--- Test 1: Zero Source ---")
    pde_zero = PoissonEquation(
        domain_size=(8,),
        source_function=lambda x, y: 0.0
    )
    
    solver = AnalogPDESolver(crossbar_size=8, noise_model="none")
    solution_zero = solver.solve(pde_zero, iterations=50)
    
    print(f"Zero source solution norm: {np.linalg.norm(solution_zero):.8f}")
    if np.linalg.norm(solution_zero) < 1e-6:
        print("✓ Zero source gives approximately zero solution")
    else:
        print("✗ Zero source should give zero solution")
    
    # Test 2: Constant source should give parabolic solution
    print("\n--- Test 2: Constant Source ---")
    pde_const = PoissonEquation(
        domain_size=(16,),
        source_function=lambda x, y: 1.0
    )
    
    solution_const = solver.solve(pde_const, iterations=100)
    digital_const = pde_const.solve_digital()
    
    # Solution should be symmetric and have maximum in middle
    mid_idx = len(solution_const) // 2
    if solution_const[mid_idx] > solution_const[0] and solution_const[mid_idx] > solution_const[-1]:
        print("✓ Constant source gives symmetric solution with maximum in middle")
    else:
        print("✗ Solution should have maximum in middle")
    
    print(f"Analog solution max: {np.max(solution_const):.6f}")
    print(f"Digital solution max: {np.max(digital_const):.6f}")
    
    # Test 3: Boundary conditions
    print("\n--- Test 3: Boundary Conditions ---")
    if abs(solution_const[0]) < 1e-10 and abs(solution_const[-1]) < 1e-10:
        print("✓ Dirichlet boundary conditions are satisfied")
    else:
        print(f"✗ Boundary conditions violated: φ(0)={solution_const[0]}, φ(1)={solution_const[-1]}")


def test_convergence_analysis():
    """Test convergence behavior."""
    print("\n=== Convergence Analysis ===")
    
    pde = PoissonEquation(
        domain_size=(8,),
        source_function=lambda x, y: np.sin(np.pi * x)
    )
    
    solver = AnalogPDESolver(crossbar_size=8, noise_model="none")
    
    # Test convergence with different iteration counts
    iteration_counts = [10, 50, 100, 200]
    solutions = []
    
    for iters in iteration_counts:
        solution = solver.solve(pde, iterations=iters, convergence_threshold=1e-10)
        solutions.append(solution)
        print(f"Iterations: {iters:3d}, Solution norm: {np.linalg.norm(solution):.6f}")
    
    # Check if solutions are converging
    for i in range(1, len(solutions)):
        diff = np.linalg.norm(solutions[i] - solutions[i-1])
        print(f"Change from {iteration_counts[i-1]} to {iteration_counts[i]} iterations: {diff:.6f}")
        
        if diff < 0.1:
            print(f"✓ Solution stabilizing at {iteration_counts[i]} iterations")
            break
    else:
        print("⚠ Solution may not be fully converged")


if __name__ == "__main__":
    print("Analytical Solution Verification for Analog PDE Solver\n")
    
    try:
        test_simple_validation()
        test_poisson_analytical()
        test_heat_analytical()
        test_convergence_analysis()
        
        print("\n" + "="*60)
        print("ANALYTICAL VALIDATION COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()