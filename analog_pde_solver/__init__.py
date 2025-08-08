"""
Analog PDE Solver Simulation Framework

A Python + Verilog playground for prototyping in-memory analog accelerators
that solve partial differential equations with 100-1000× energy efficiency.
"""

__version__ = "0.1.0"

# Set up default logging
try:
    from .utils.logger import setup_logging
    _logger = setup_logging()
except ImportError:
    # Fallback if logger utils not available
    import logging
    logging.basicConfig(level=logging.INFO)

from .core.solver import AnalogPDESolver
from .core.equations import PoissonEquation, NavierStokesEquation, HeatEquation, WaveEquation
from .navier_stokes import NavierStokesAnalog
from .core.solver_robust import RobustAnalogPDESolver
from .optimization.performance_optimizer import PerformanceOptimizer, OptimizationConfig
from .optimization.auto_scaler import AutoScaler, ScalingPolicy
from .monitoring.health_monitor import SystemHealthMonitor
from .rtl.verilog_generator import VerilogGenerator, RTLConfig

# GPU acceleration (optional)
try:
    from .acceleration.gpu_solver import GPUAcceleratedSolver, GPUConfig, GPUMemoryManager
    _HAS_GPU_ACCELERATION = True
except ImportError:
    _HAS_GPU_ACCELERATION = False
    GPUAcceleratedSolver = None
    GPUConfig = None
    GPUMemoryManager = None

__all__ = [
    "AnalogPDESolver",
    "PoissonEquation", 
    "NavierStokesEquation",
    "HeatEquation",
    "WaveEquation",
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

# Add GPU acceleration to exports if available
if _HAS_GPU_ACCELERATION:
    __all__.extend([
        "GPUAcceleratedSolver",
        "GPUConfig", 
        "GPUMemoryManager"
    ])