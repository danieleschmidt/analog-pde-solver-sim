"""Core analog PDE solver components."""

from .solver import AnalogPDESolver
from .equations import PoissonEquation, NavierStokesEquation
from .crossbar import AnalogCrossbarArray

__all__ = [
    "AnalogPDESolver",
    "PoissonEquation",
    "NavierStokesEquation", 
    "AnalogCrossbarArray",
]