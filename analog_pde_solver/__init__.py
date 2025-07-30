"""
Analog PDE Solver Simulation Package

A Python + Verilog playground for prototyping in-memory analog accelerators
for solving partial differential equations.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports for public API
try:
    from .core.solver import AnalogPDESolver
    from .core.equations import PoissonEquation, HeatEquation
    from .hardware.crossbar import AnalogCrossbarArray
    from .spice.simulator import SPICESimulator
    
    __all__ = [
        "AnalogPDESolver",
        "PoissonEquation", 
        "HeatEquation",
        "AnalogCrossbarArray",
        "SPICESimulator",
    ]
except ImportError:
    # Handle missing dependencies gracefully during development
    __all__ = []