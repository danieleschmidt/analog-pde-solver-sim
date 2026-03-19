#!/usr/bin/env python3
"""
Demo: Solve 2-D Poisson equation on a 20×20 grid.

Problem: ∇²u = f(x,y) where f = -2π²·sin(πx)·sin(πy)
Exact solution: u(x,y) = sin(πx)·sin(πy)

Boundary conditions: u = 0 on all edges.

Run with:
    ~/anaconda3/bin/python3 examples/poisson_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from analog_pde_solver import AnalogPDESolver, DigitalSolver, EnergyModel

GRID = 20

def zero_boundary(i, j, rows, cols):
    if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
        return 0.0
    return None

def poisson_source(i, j, rows, cols):
    """Source term f = -2π²·sin(πx)·sin(πy)."""
    x = i / (rows - 1)
    y = j / (cols - 1)
    return -2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_solution(rows, cols):
    """Exact: u(x,y) = sin(πx)·sin(πy)."""
    u = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            x = i / (rows - 1)
            y = j / (cols - 1)
            u[i, j] = np.sin(np.pi * x) * np.sin(np.pi * y)
    return u

print("=" * 60)
print("  Analog PDE Solver — Poisson equation demo")
print("=" * 60)

analog  = AnalogPDESolver(rows=GRID, cols=GRID)
digital = DigitalSolver(rows=GRID, cols=GRID)
exact   = exact_solution(GRID, GRID)

u_a = analog.solve(zero_boundary, source_fn=poisson_source, tol=1e-7)
u_d = digital.solve(zero_boundary, source_fn=poisson_source, tol=1e-7)

err_a = np.max(np.abs(u_a - exact))
err_d = np.max(np.abs(u_d - exact))

print(f"\nAnalog  max error vs exact : {err_a:.4e}  ({analog.iterations_run} iters)")
print(f"Digital max error vs exact : {err_d:.4e}  ({digital.iterations_run} iters)")

em = EnergyModel()
e_a = analog.energy_joules(em)
e_d = digital.energy_joules(em)
print(f"\nAnalog  energy : {e_a*1e12:.3f} pJ")
print(f"Digital energy : {e_d*1e12:.3f} pJ")
print(f"Efficiency gain: {em.efficiency_ratio(e_a, e_d):.1f}×")
print("\nDone.")
