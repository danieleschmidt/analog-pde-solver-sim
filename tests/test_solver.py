"""Tests for AnalogPDESolver, DigitalSolver, and EnergyModel."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from analog_pde_solver import AnalogPDESolver, DigitalSolver, EnergyModel


# ── Shared boundary/source helpers ───────────────────────────────────────────

def top_hot(i, j, rows, cols):
    """Laplace: top = 1, rest = 0."""
    if i == 0:
        return 1.0
    if i == rows - 1 or j == 0 or j == cols - 1:
        return 0.0
    return None


def zero_boundary(i, j, rows, cols):
    if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
        return 0.0
    return None


def sin_source(i, j, rows, cols):
    """Poisson source for u=sin(πx)·sin(πy) exact solution."""
    x = i / (rows - 1)
    y = j / (cols - 1)
    return -2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)


# ── AnalogPDESolver tests ─────────────────────────────────────────────────────

class TestAnalogPDESolver:
    def test_laplace_converges(self):
        solver = AnalogPDESolver(rows=10, cols=10)
        u = solver.solve(top_hot, tol=1e-5)
        assert solver.iterations_run < 10_000, "Should converge before hitting cap"

    def test_laplace_boundary_respected(self):
        solver = AnalogPDESolver(rows=10, cols=10)
        u = solver.solve(top_hot, tol=1e-6)
        assert np.isclose(u[0, :], 1.0).all(), "Top row should be 1"
        assert np.isclose(u[-1, :], 0.0).all(), "Bottom row should be 0"
        # Left/right cols: non-top-row rows should be 0 (top row corners = 1)
        assert np.isclose(u[1:, 0], 0.0).all(), "Left col (non-top) should be 0"
        assert np.isclose(u[1:, -1], 0.0).all(), "Right col (non-top) should be 0"

    def test_laplace_interior_bounded(self):
        """Interior values must lie within boundary range [0, 1]."""
        solver = AnalogPDESolver(rows=10, cols=10)
        u = solver.solve(top_hot, tol=1e-6)
        assert u.min() >= -1e-8, "Solution should not undershoot 0"
        assert u.max() <= 1.0 + 1e-8, "Solution should not exceed 1"

    def test_interior_decreases_top_to_bottom(self):
        """Heat should decrease monotonically from top to bottom along centre."""
        solver = AnalogPDESolver(rows=10, cols=10)
        u = solver.solve(top_hot, tol=1e-6)
        mid = 5
        col = u[:, mid]
        assert all(col[k] >= col[k + 1] - 1e-8 for k in range(len(col) - 1)), \
            "Temperature should decrease (or stay equal) from top to bottom"

    def test_symmetry_laplace(self):
        """Top-hot problem is left-right symmetric."""
        solver = AnalogPDESolver(rows=10, cols=10)
        u = solver.solve(top_hot, tol=1e-6)
        # Interior only
        assert np.allclose(u[:, 1:9], u[:, 8:0:-1], atol=1e-5), \
            "Solution should be left-right symmetric"

    def test_poisson_accuracy(self):
        """Poisson solve should be close to exact sin·sin solution."""
        rows, cols = 15, 15
        solver = AnalogPDESolver(rows=rows, cols=cols)
        u = solver.solve(zero_boundary, source_fn=sin_source, tol=1e-7)
        # Exact solution
        exact = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                x, y = i / (rows - 1), j / (cols - 1)
                exact[i, j] = np.sin(np.pi * x) * np.sin(np.pi * y)
        max_err = np.max(np.abs(u[1:-1, 1:-1] - exact[1:-1, 1:-1]))
        assert max_err < 0.05, f"Max Poisson error {max_err:.4f} exceeds 5 %"

    def test_noisy_analog_still_close(self):
        """1 % resistor mismatch should give < 5 % error vs ideal."""
        rows, cols = 10, 10
        ideal = AnalogPDESolver(rows=rows, cols=cols, noise_level=0.0)
        noisy = AnalogPDESolver(rows=rows, cols=cols, noise_level=0.01, rng_seed=7)
        u_i = ideal.solve(top_hot, tol=1e-6)
        u_n = noisy.solve(top_hot, tol=1e-6)
        max_err = np.max(np.abs(u_i - u_n))
        assert max_err < 0.05, f"Noisy max err {max_err:.4f} too large for 1% noise"

    def test_residual_history_decreasing(self):
        """Residual should trend downward (monotone decrease not guaranteed but overall)."""
        solver = AnalogPDESolver(rows=10, cols=10)
        solver.solve(top_hot, tol=1e-6)
        hist = solver.residual_history
        assert len(hist) > 0
        # First residual should be larger than last
        assert hist[0] > hist[-1], "Residual should decrease over iterations"

    def test_energy_positive(self):
        solver = AnalogPDESolver(rows=10, cols=10)
        solver.solve(top_hot, tol=1e-5)
        e = solver.energy_joules()
        assert e > 0, "Energy should be positive"


# ── DigitalSolver tests ───────────────────────────────────────────────────────

class TestDigitalSolver:
    def test_laplace_converges(self):
        solver = DigitalSolver(rows=10, cols=10)
        solver.solve(top_hot, tol=1e-5)
        assert solver.iterations_run < 10_000

    def test_boundary_conditions(self):
        solver = DigitalSolver(rows=10, cols=10)
        u = solver.solve(top_hot, tol=1e-6)
        assert np.isclose(u[0, :], 1.0).all()
        assert np.isclose(u[-1, :], 0.0).all()

    def test_poisson_accuracy(self):
        rows, cols = 15, 15
        solver = DigitalSolver(rows=rows, cols=cols)
        u = solver.solve(zero_boundary, source_fn=sin_source, tol=1e-7)
        exact = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                x, y = i / (rows - 1), j / (cols - 1)
                exact[i, j] = np.sin(np.pi * x) * np.sin(np.pi * y)
        max_err = np.max(np.abs(u[1:-1, 1:-1] - exact[1:-1, 1:-1]))
        assert max_err < 0.05

    def test_wall_time_recorded(self):
        solver = DigitalSolver(rows=10, cols=10)
        solver.solve(top_hot, tol=1e-5)
        assert solver.wall_time_s > 0

    def test_energy_positive(self):
        solver = DigitalSolver(rows=10, cols=10)
        solver.solve(top_hot, tol=1e-5)
        e = solver.energy_joules()
        assert e > 0


# ── Cross-solver comparison ───────────────────────────────────────────────────

class TestCrossComparison:
    def test_analog_vs_digital_close(self):
        """Analog and digital should agree to within 1e-4 on same problem."""
        analog  = AnalogPDESolver(rows=15, cols=15, noise_level=0.0)
        digital = DigitalSolver(rows=15, cols=15)
        u_a = analog.solve(top_hot, tol=1e-7)
        u_d = digital.solve(top_hot, tol=1e-7)
        max_err = np.max(np.abs(u_a - u_d))
        assert max_err < 1e-4, f"Analog vs digital max err {max_err:.2e} too large"

    def test_analog_more_energy_efficient(self):
        """Analog energy should be lower than digital for a 20×20 grid."""
        analog  = AnalogPDESolver(rows=20, cols=20)
        digital = DigitalSolver(rows=20, cols=20)
        analog.solve(top_hot, tol=1e-6)
        digital.solve(top_hot, tol=1e-6)
        em = EnergyModel()
        e_a = analog.energy_joules(em)
        e_d = digital.energy_joules(em)
        assert e_a < e_d, "Analog should use less energy than digital"

    def test_same_convergence_iterations(self):
        """Both solvers implement the same algorithm — should take same iterations."""
        analog  = AnalogPDESolver(rows=10, cols=10, noise_level=0.0)
        digital = DigitalSolver(rows=10, cols=10)
        analog.solve(top_hot, tol=1e-5)
        digital.solve(top_hot, tol=1e-5)
        # Allow ±1 iteration difference (floating-point order may differ slightly)
        assert abs(analog.iterations_run - digital.iterations_run) <= 2, \
            f"Iteration count mismatch: analog={analog.iterations_run} digital={digital.iterations_run}"


# ── EnergyModel tests ─────────────────────────────────────────────────────────

class TestEnergyModel:
    def test_efficiency_ratio_analog_wins(self):
        """Digital should be more expensive than analog for sufficient iterations."""
        em = EnergyModel(resistor_ohms=10_000, flop_energy_add_pj=1.0)
        # 20×20 grid, 500 iterations
        e_d = em.digital_energy_joules(20, 20, 500)
        # Make a simple uniform-voltage solution for analog
        u = np.ones((20, 20)) * 0.5
        e_a = em.analog_energy_joules(u)
        ratio = em.efficiency_ratio(e_a, e_d)
        assert ratio > 1.0, "Analog should be more efficient than digital"

    def test_zero_voltage_solution_nonzero_energy_digital(self):
        """Digital energy depends on iteration count, not solution values."""
        em = EnergyModel()
        e = em.digital_energy_joules(10, 10, 100)
        assert e > 0

    def test_efficiency_ratio_infinite_on_zero_analog(self):
        em = EnergyModel()
        ratio = em.efficiency_ratio(0.0, 1e-9)
        assert ratio == float("inf")

    def test_energy_scales_with_iterations(self):
        em = EnergyModel()
        e1 = em.digital_energy_joules(10, 10, 100)
        e2 = em.digital_energy_joules(10, 10, 200)
        assert np.isclose(e2, 2 * e1), "Energy should scale linearly with iterations"
