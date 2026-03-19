#!/usr/bin/env python3
"""
Demo: Solve 2-D Laplace equation (heat distribution) on a 20×20 grid.

Problem setup
-------------
A square plate 1 × 1.  Boundary conditions:

    Top edge:    u = 1  (hot wall, 100 °C normalised to 1)
    All others:  u = 0  (cold / insulated)

The steady-state temperature satisfies ∇²u = 0 (Laplace equation).

We compare:
  * AnalogPDESolver  — resistive mesh simulation
  * DigitalSolver    — conventional FD Gauss–Seidel
  * EnergyModel      — estimated energy consumption

Run with:
    ~/anaconda3/bin/python3 examples/laplace_heat_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from analog_pde_solver import AnalogPDESolver, DigitalSolver, EnergyModel

GRID = 20

# ── Boundary condition ────────────────────────────────────────────────────────
def top_hot(i, j, rows, cols):
    """Top row = 1 (hot), all others = 0."""
    if i == 0:
        return 1.0
    if i == rows - 1 or j == 0 or j == cols - 1:
        return 0.0
    return None  # interior → free to compute


# ── Solve with both solvers ───────────────────────────────────────────────────
print("=" * 60)
print("  Analog PDE Solver — Laplace / heat distribution demo")
print("=" * 60)

# Ideal analog (no noise)
analog = AnalogPDESolver(rows=GRID, cols=GRID, noise_level=0.0, rng_seed=42)
u_analog = analog.solve(top_hot, tol=1e-6, max_iter=20_000)

print(f"\nAnalog solver  (resistive mesh, no noise)")
print(f"  Iterations to converge : {analog.iterations_run}")
print(f"  Final residual          : {analog.residual_history[-1]:.2e}")

# Analog with 1 % resistor mismatch
analog_noisy = AnalogPDESolver(rows=GRID, cols=GRID, noise_level=0.01, rng_seed=0)
u_analog_noisy = analog_noisy.solve(top_hot, tol=1e-6, max_iter=20_000)

print(f"\nAnalog solver  (1 % resistor mismatch / noise)")
print(f"  Iterations to converge : {analog_noisy.iterations_run}")

# Digital reference
digital = DigitalSolver(rows=GRID, cols=GRID)
u_digital = digital.solve(top_hot, tol=1e-6, max_iter=20_000)

print(f"\nDigital solver (finite-difference Gauss–Seidel)")
print(f"  Iterations to converge : {digital.iterations_run}")
print(f"  Wall time               : {digital.wall_time_s*1000:.1f} ms")

# ── Accuracy comparison ───────────────────────────────────────────────────────
max_err   = np.max(np.abs(u_analog - u_digital))
rms_err   = np.sqrt(np.mean((u_analog - u_digital) ** 2))
max_err_n = np.max(np.abs(u_analog_noisy - u_digital))

print(f"\nAccuracy (analog ideal   vs digital): max err = {max_err:.2e}, RMS = {rms_err:.2e}")
print(f"Accuracy (analog noisy  vs digital): max err = {max_err_n:.2e}")

# ── Energy comparison ─────────────────────────────────────────────────────────
em = EnergyModel(resistor_ohms=10_000, supply_voltage=1.0)

e_analog  = analog.energy_joules(em)
e_digital = digital.energy_joules(em)
ratio     = em.efficiency_ratio(e_analog, e_digital)

print(f"\nEnergy model ({GRID}×{GRID} grid)")
print(f"  Analog  energy : {e_analog*1e12:.3f} pJ")
print(f"  Digital energy : {e_digital*1e12:.3f} pJ")
print(f"  Efficiency gain: {ratio:.1f}× (analog vs digital)")

# ── ASCII visualisation ───────────────────────────────────────────────────────
def ascii_heatmap(grid, title, width=20):
    shades = " .:-=+*#%@"
    print(f"\n{title}")
    rows, cols = grid.shape
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            idx = int(grid[i, j] * (len(shades) - 1))
            row_str += shades[max(0, min(idx, len(shades) - 1))] * 2
        print(row_str)

ascii_heatmap(u_digital,    "Digital solution  (ground truth)")
ascii_heatmap(u_analog,     "Analog  solution  (resistive mesh, no noise)")
ascii_heatmap(u_analog_noisy, "Analog  solution  (1% resistor mismatch)")

# ── Centre-column temperature profile ─────────────────────────────────────────
mid = GRID // 2
print(f"\nCentre-column temperature profile (col={mid})")
print(f"  {'Row':>4}  {'Digital':>9}  {'Analog':>9}  {'Noisy':>9}")
for i in range(GRID):
    print(f"  {i:4d}  {u_digital[i, mid]:9.4f}  {u_analog[i, mid]:9.4f}  {u_analog_noisy[i, mid]:9.4f}")

print("\nDone.")
