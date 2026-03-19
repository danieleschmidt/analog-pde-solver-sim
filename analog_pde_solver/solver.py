"""
Analog PDE Solver Simulation
=============================

Simulates how a resistive mesh (analog computer) solves the Laplace / Poisson
equation, and compares it to a conventional finite-difference digital solver.

Analog computing principle
---------------------------
A 2-D resistive mesh is a physical analog computer for the Laplace equation.
Place equal resistors between every neighbouring node.  Kirchhoff's current
law (KCL) at each interior node states:

    (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1] - 4·V[i,j]) / R = 0

which is exactly the finite-difference discretisation of ∇²V = 0.
The network relaxes to the correct solution in continuous time — no clock,
no multiplications, just resistors obeying Ohm's law.

For the Poisson equation ∇²u = f(x,y), current-source injections at nodes
model the source term f.

Energy model
-------------
* Analog: power is dissipated in resistors while the mesh equilibrates.
  We estimate P = sum(V² / R) for each resistor.  Once settled (~5 RC time
  constants), the computation is done.
* Digital: each floating-point addition / subtraction costs ~1 pJ (modern CPU),
  and each multiply costs ~5 pJ.  Gauss–Seidel requires ~5 FLOP per node per
  iteration (for a 5-point stencil).

Reference energy advantage: 100-1000× for large grids at moderate accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

@dataclass
class EnergyModel:
    """Estimates energy consumption for analog vs. digital computation.

    Parameters
    ----------
    resistor_ohms:
        Resistance value used in the analog mesh (Ω).  Lower resistance →
        faster settling but higher static power.
    supply_voltage:
        Nominal voltage applied at boundary nodes (V).
    flop_energy_add_pj:
        Energy per floating-point add/subtract on a digital processor (pJ).
    flop_energy_mul_pj:
        Energy per floating-point multiply on a digital processor (pJ).
    """

    resistor_ohms: float = 10_000.0       # 10 kΩ — typical thin-film resistor
    supply_voltage: float = 1.0           # 1 V boundary
    flop_energy_add_pj: float = 1.0       # 1 pJ / FLOP (modern CPU, ~1 GHz)
    flop_energy_mul_pj: float = 5.0       # 5 pJ / FLOP

    def analog_energy_joules(
        self,
        solution: np.ndarray,
        settling_time_s: Optional[float] = None,
    ) -> float:
        """Estimate total energy for analog mesh to reach solution.

        The mesh dissipates power P = ΣV²/R across each resistor.  We
        approximate settling time as 5·RC where C is a representative node
        capacitance (~1 pF for thin-film).
        """
        C_node_F = 1e-12  # 1 pF parasitic node capacitance
        rows, cols = solution.shape

        # Horizontal resistors
        dV_h = np.diff(solution, axis=1)
        power_h = np.sum(dV_h ** 2) / self.resistor_ohms
        # Vertical resistors
        dV_v = np.diff(solution, axis=0)
        power_v = np.sum(dV_v ** 2) / self.resistor_ohms

        total_power_W = power_h + power_v

        if settling_time_s is None:
            # 5 RC time constants to ~99 % accuracy
            settling_time_s = 5.0 * self.resistor_ohms * C_node_F

        return total_power_W * settling_time_s

    def digital_energy_joules(
        self,
        grid_rows: int,
        grid_cols: int,
        iterations: int,
        flops_per_node_per_iter: int = 5,
    ) -> float:
        """Estimate energy for digital Gauss–Seidel relaxation.

        Gauss–Seidel 5-point stencil: 4 additions + 1 division ≈ 5 FLOP.
        """
        # 4 adds + 1 divide treated as 1 mul
        adds = 4 * grid_rows * grid_cols * iterations
        muls = 1 * grid_rows * grid_cols * iterations
        energy_pj = (
            adds * self.flop_energy_add_pj
            + muls * self.flop_energy_mul_pj
        )
        return energy_pj * 1e-12  # convert to Joules

    def efficiency_ratio(
        self,
        analog_J: float,
        digital_J: float,
    ) -> float:
        """How many times more efficient is analog vs digital (>1 = analog wins)."""
        if analog_J == 0:
            return float("inf")
        return digital_J / analog_J


# ---------------------------------------------------------------------------
# Boundary condition helpers
# ---------------------------------------------------------------------------

def apply_dirichlet(
    grid: np.ndarray,
    boundary_fn: Callable[[int, int, int, int], Optional[float]],
) -> np.ndarray:
    """Apply Dirichlet (fixed-value) boundary conditions in-place.

    Parameters
    ----------
    grid:
        2-D solution array; interior values will be computed, edge values
        set by *boundary_fn*.
    boundary_fn:
        Called as ``boundary_fn(i, j, rows, cols)``; should return the
        prescribed value at ``(i, j)`` for boundary nodes and ``None`` for
        interior nodes.
    """
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            val = boundary_fn(i, j, rows, cols)
            if val is not None:
                grid[i, j] = val
    return grid


# ---------------------------------------------------------------------------
# Analog solver — resistive mesh / Gauss–Seidel relaxation
# ---------------------------------------------------------------------------

class AnalogPDESolver:
    """Simulate a resistive-mesh analog computer solving the Laplace / Poisson eq.

    The resistive mesh is a physical realisation of the 5-point finite-difference
    stencil for ∇²u = f.  Each interior node obeys Kirchhoff's current law:

        (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4·u[i,j]) / R = -f[i,j] · R_src

    which is equivalent to:

        u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h²·f[i,j]) / 4

    The solver runs Gauss–Seidel relaxation — mimicking the physical settling
    of the RC network toward its DC equilibrium.

    Parameters
    ----------
    rows, cols:
        Grid dimensions (including boundary nodes).
    resistor_ohms:
        Resistor value for energy estimation.
    noise_level:
        Fractional noise amplitude (0 = ideal analog, 0.01 = 1 % component
        mismatch — realistic for thin-film resistors).
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        resistor_ohms: float = 10_000.0,
        noise_level: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.resistor_ohms = resistor_ohms
        self.noise_level = noise_level
        self._rng = np.random.default_rng(rng_seed)

        self.solution: np.ndarray = np.zeros((rows, cols))
        self.iterations_run: int = 0
        self.residual_history: list[float] = []

        # Resistor mismatch matrix (multiplicative noise on each link)
        if noise_level > 0:
            self._noise_h = 1.0 + noise_level * self._rng.standard_normal((rows, cols - 1))
            self._noise_v = 1.0 + noise_level * self._rng.standard_normal((rows - 1, cols))
        else:
            self._noise_h = np.ones((rows, cols - 1))
            self._noise_v = np.ones((rows - 1, cols))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        boundary_fn: Callable[[int, int, int, int], Optional[float]],
        source_fn: Optional[Callable[[int, int, int, int], float]] = None,
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """Solve ∇²u = source (or ∇²u = 0 if source_fn is None).

        Parameters
        ----------
        boundary_fn:
            Returns the fixed boundary value at ``(i, j)`` or ``None`` for
            interior nodes.
        source_fn:
            Returns the source-term value f(i,j) for Poisson problems.
            Defaults to zero (Laplace equation).
        tol:
            Convergence tolerance on the maximum change per iteration.
        max_iter:
            Safety cap on iteration count (the physical network always converges,
            but numerical relaxation may be slow for large grids).

        Returns
        -------
        np.ndarray
            Converged solution array of shape (rows, cols).
        """
        rows, cols = self.rows, self.cols

        # Initialise with boundary conditions
        grid = np.zeros((rows, cols))
        apply_dirichlet(grid, boundary_fn)

        # Build source term array h² · f[i,j]  (h = 1 / (n-1) so h² = 1/(n-1)²)
        h = 1.0 / (max(rows, cols) - 1)
        source = np.zeros((rows, cols))
        if source_fn is not None:
            for i in range(rows):
                for j in range(cols):
                    source[i, j] = h ** 2 * source_fn(i, j, rows, cols)

        # Gauss–Seidel relaxation (mimics RC settling)
        self.residual_history = []
        for iteration in range(max_iter):
            max_change = 0.0

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Neighbour voltages with optional resistor noise
                    north = grid[i - 1, j] / self._noise_v[i - 1, j]
                    south = grid[i + 1, j] / self._noise_v[i, j]
                    west  = grid[i, j - 1] / self._noise_h[i, j - 1]
                    east  = grid[i, j + 1] / self._noise_h[i, j]

                    # KCL at node: ∇²u = f  →  u[i,j] = (neighbors - h²·f) / 4
                    new_val = (north + south + west + east - source[i, j]) / 4.0
                    change = abs(new_val - grid[i, j])
                    if change > max_change:
                        max_change = change
                    grid[i, j] = new_val

            self.residual_history.append(max_change)

            if max_change < tol:
                self.iterations_run = iteration + 1
                break
        else:
            self.iterations_run = max_iter

        self.solution = grid
        return grid

    def energy_joules(self, energy_model: Optional[EnergyModel] = None) -> float:
        """Estimate energy consumed during the analog computation."""
        em = energy_model or EnergyModel(resistor_ohms=self.resistor_ohms)
        return em.analog_energy_joules(self.solution)


# ---------------------------------------------------------------------------
# Digital solver — explicit finite-difference for comparison
# ---------------------------------------------------------------------------

class DigitalSolver:
    """Finite-difference Gauss–Seidel solver for ∇²u = f.

    This is the straightforward digital equivalent of the analog mesh.
    It uses the same 5-point stencil but executes as floating-point
    arithmetic on a CPU — no physical settling, just number-crunching.
    """

    def __init__(self, rows: int = 20, cols: int = 20) -> None:
        self.rows = rows
        self.cols = cols
        self.solution: np.ndarray = np.zeros((rows, cols))
        self.iterations_run: int = 0
        self.residual_history: list[float] = []
        self.wall_time_s: float = 0.0

    def solve(
        self,
        boundary_fn: Callable[[int, int, int, int], Optional[float]],
        source_fn: Optional[Callable[[int, int, int, int], float]] = None,
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """Solve ∇²u = f using Gauss–Seidel relaxation.

        Parameters are identical to :meth:`AnalogPDESolver.solve`.
        """
        rows, cols = self.rows, self.cols
        h = 1.0 / (max(rows, cols) - 1)

        grid = np.zeros((rows, cols))
        apply_dirichlet(grid, boundary_fn)

        source = np.zeros((rows, cols))
        if source_fn is not None:
            for i in range(rows):
                for j in range(cols):
                    source[i, j] = h ** 2 * source_fn(i, j, rows, cols)

        t0 = time.perf_counter()
        self.residual_history = []

        for iteration in range(max_iter):
            max_change = 0.0

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # ∇²u = f  →  u[i,j] = (sum_neighbors - h²·f) / 4
                    new_val = (
                        grid[i - 1, j]
                        + grid[i + 1, j]
                        + grid[i, j - 1]
                        + grid[i, j + 1]
                        - source[i, j]
                    ) / 4.0
                    change = abs(new_val - grid[i, j])
                    if change > max_change:
                        max_change = change
                    grid[i, j] = new_val

            self.residual_history.append(max_change)

            if max_change < tol:
                self.iterations_run = iteration + 1
                break
        else:
            self.iterations_run = max_iter

        self.wall_time_s = time.perf_counter() - t0
        self.solution = grid
        return grid

    def energy_joules(
        self,
        energy_model: Optional[EnergyModel] = None,
        flops_per_node_per_iter: int = 5,
    ) -> float:
        """Estimate energy consumed during digital computation."""
        em = energy_model or EnergyModel()
        return em.digital_energy_joules(
            self.rows,
            self.cols,
            self.iterations_run,
            flops_per_node_per_iter,
        )
