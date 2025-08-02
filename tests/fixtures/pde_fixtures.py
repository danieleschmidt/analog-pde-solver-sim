"""
Advanced test fixtures for PDE solver testing.
"""
import pytest
import numpy as np
from typing import Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class PDETestCase:
    """Standard test case for PDE problems."""
    name: str
    equation_type: str
    grid_size: tuple
    boundary_conditions: Dict[str, Any]
    analytical_solution: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    expected_convergence_rate: float = None
    tolerance: float = 1e-6


@pytest.fixture
def poisson_2d_simple():
    """Simple 2D Poisson equation test case."""
    def analytical(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    return PDETestCase(
        name="poisson_2d_simple",
        equation_type="poisson",
        grid_size=(64, 64),
        boundary_conditions={"type": "dirichlet", "value": 0.0},
        analytical_solution=analytical,
        expected_convergence_rate=2.0,
        tolerance=1e-5
    )


@pytest.fixture
def heat_equation_gaussian():
    """Heat equation with Gaussian initial condition."""
    def analytical(x, y, t=0.1):
        sigma = 0.1
        return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (4 * sigma * t + 0.01))
    
    return PDETestCase(
        name="heat_gaussian",
        equation_type="heat",
        grid_size=(128, 128),
        boundary_conditions={"type": "neumann", "value": 0.0},
        analytical_solution=analytical,
        expected_convergence_rate=1.0,
        tolerance=1e-4
    )


@pytest.fixture
def navier_stokes_cavity():
    """Lid-driven cavity flow test case."""
    return PDETestCase(
        name="cavity_flow",
        equation_type="navier_stokes",
        grid_size=(256, 256),
        boundary_conditions={
            "top": {"type": "dirichlet", "u": 1.0, "v": 0.0},
            "walls": {"type": "no_slip"}
        },
        expected_convergence_rate=None,  # Nonlinear problem
        tolerance=1e-3
    )


@pytest.fixture(params=[
    (32, 32), (64, 64), (128, 128), (256, 256)
])
def grid_sizes(request):
    """Parametrized grid sizes for convergence studies."""
    return request.param


@pytest.fixture
def conductance_ranges():
    """Standard conductance ranges for analog testing."""
    return [
        (1e-9, 1e-6),  # Standard range
        (1e-8, 1e-5),  # Higher precision
        (1e-10, 1e-7), # Lower power
    ]


@pytest.fixture
def noise_models():
    """Different noise models for testing."""
    return {
        "ideal": {"enabled": False},
        "gaussian": {"enabled": True, "type": "gaussian", "sigma": 1e-3},
        "realistic": {
            "enabled": True,
            "thermal": True,
            "shot": True,
            "flicker": True,
            "rtv": 0.1  # Random telegraph noise
        }
    }


@pytest.fixture
def hardware_configurations():
    """Standard hardware configurations for testing."""
    return [
        {
            "name": "small_crossbar",
            "size": (64, 64),
            "precision": 6,
            "dac_bits": 8,
            "adc_bits": 10
        },
        {
            "name": "large_crossbar", 
            "size": (512, 512),
            "precision": 8,
            "dac_bits": 10,
            "adc_bits": 12
        },
        {
            "name": "high_precision",
            "size": (128, 128),
            "precision": 10,
            "dac_bits": 12,
            "adc_bits": 14
        }
    ]


@pytest.fixture
def performance_benchmarks():
    """Standard performance benchmark problems."""
    return {
        "small": {
            "grid_size": (64, 64),
            "max_time": 1.0,  # seconds
            "max_memory": 100e6  # bytes
        },
        "medium": {
            "grid_size": (256, 256),
            "max_time": 10.0,
            "max_memory": 1e9
        },
        "large": {
            "grid_size": (1024, 1024),
            "max_time": 60.0,
            "max_memory": 8e9
        }
    }