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

# Optional imports with graceful fallback
try:
    from .navier_stokes import NavierStokesAnalog
    _HAS_NAVIER_STOKES = True
except ImportError:
    _HAS_NAVIER_STOKES = False
    NavierStokesAnalog = None

try:
    from .core.solver_robust import RobustAnalogPDESolver
    _HAS_ROBUST_SOLVER = True
except ImportError:
    _HAS_ROBUST_SOLVER = False
    RobustAnalogPDESolver = None

try:
    from .optimization.performance_optimizer import PerformanceOptimizer, OptimizationConfig
    from .optimization.auto_scaler import AutoScaler, ScalingPolicy
    _HAS_OPTIMIZATION = True
except ImportError:
    _HAS_OPTIMIZATION = False
    PerformanceOptimizer = OptimizationConfig = AutoScaler = ScalingPolicy = None

try:
    from .monitoring.health_monitor import SystemHealthMonitor
    _HAS_MONITORING = True
except ImportError:
    _HAS_MONITORING = False
    SystemHealthMonitor = None

try:
    from .rtl.verilog_generator import VerilogGenerator, RTLConfig
    _HAS_RTL = True
except ImportError:
    _HAS_RTL = False
    VerilogGenerator = RTLConfig = None

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
]

# Add optional components to exports if available
if _HAS_NAVIER_STOKES and NavierStokesAnalog:
    __all__.append("NavierStokesAnalog")

if _HAS_ROBUST_SOLVER and RobustAnalogPDESolver:
    __all__.append("RobustAnalogPDESolver")

if _HAS_OPTIMIZATION and PerformanceOptimizer:
    __all__.extend(["PerformanceOptimizer", "OptimizationConfig", "AutoScaler", "ScalingPolicy"])

if _HAS_MONITORING and SystemHealthMonitor:
    __all__.append("SystemHealthMonitor")

if _HAS_RTL and VerilogGenerator:
    __all__.extend(["VerilogGenerator", "RTLConfig"])

if _HAS_GPU_ACCELERATION:
    __all__.extend([
        "GPUAcceleratedSolver",
        "GPUConfig", 
        "GPUMemoryManager"
    ])