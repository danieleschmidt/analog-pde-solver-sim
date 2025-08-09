"""Visualization module for analog PDE solver results."""

__version__ = "0.1.0"

try:
    from .pde_visualizer import PDEVisualizer, PlotConfig
    from .hardware_monitor import HardwareMonitorDashboard
    from .benchmark_plots import BenchmarkVisualizer
    _HAS_VISUALIZATION = True
except ImportError as e:
    _HAS_VISUALIZATION = False
    PDEVisualizer = None
    PlotConfig = None
    HardwareMonitorDashboard = None
    BenchmarkVisualizer = None

__all__ = []

if _HAS_VISUALIZATION:
    __all__.extend([
        "PDEVisualizer",
        "PlotConfig", 
        "HardwareMonitorDashboard",
        "BenchmarkVisualizer"
    ])