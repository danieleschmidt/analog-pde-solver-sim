"""
Analog PDE Solver Simulation Framework

A Python + Verilog playground for prototyping in-memory analog accelerators
that solve partial differential equations with 100-1000× energy efficiency.
"""

__version__ = "0.1.0"

from .core.solver import AnalogPDESolver
from .core.equations import PoissonEquation, NavierStokesEquation

__all__ = [
    "AnalogPDESolver",
    "PoissonEquation", 
    "NavierStokesEquation",
]