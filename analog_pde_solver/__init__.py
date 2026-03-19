"""
Analog PDE Solver Simulation

Simulates how analog circuits (resistive meshes / RC networks) solve
Partial Differential Equations — naturally and at ultra-low energy.
"""

from .solver import AnalogPDESolver, DigitalSolver, EnergyModel

__all__ = ["AnalogPDESolver", "DigitalSolver", "EnergyModel"]
__version__ = "1.0.0"
