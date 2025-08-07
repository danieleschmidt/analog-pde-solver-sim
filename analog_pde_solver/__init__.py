"""
Analog PDE Solver Simulation Framework

A Python + Verilog playground for prototyping in-memory analog accelerators
that solve partial differential equations with 100-1000× energy efficiency.
"""

__version__ = "0.1.0"

from .core.solver import AnalogPDESolver
from .core.equations import PoissonEquation, NavierStokesEquation
from .navier_stokes import NavierStokesAnalog
from .core.solver_robust import RobustAnalogPDESolver
from .optimization.performance_optimizer import PerformanceOptimizer, OptimizationConfig
from .optimization.auto_scaler import AutoScaler, ScalingPolicy
from .monitoring.health_monitor import SystemHealthMonitor
from .rtl.verilog_generator import VerilogGenerator, RTLConfig

__all__ = [
    "AnalogPDESolver",
    "PoissonEquation", 
    "NavierStokesEquation",
    "NavierStokesAnalog",
    "RobustAnalogPDESolver",
    "PerformanceOptimizer",
    "OptimizationConfig", 
    "AutoScaler",
    "ScalingPolicy",
    "SystemHealthMonitor",
    "VerilogGenerator",
    "RTLConfig",
]