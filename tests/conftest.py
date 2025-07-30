"""
Pytest configuration and fixtures for analog-pde-solver-sim tests.
"""

import pytest
import numpy as np
from typing import Tuple, Dict, Any


@pytest.fixture
def sample_2d_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple 2D grid for testing."""
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)
    return X, Y


@pytest.fixture
def poisson_test_case() -> Dict[str, Any]:
    """Standard Poisson equation test case."""
    return {
        "domain_size": (32, 32),
        "boundary_conditions": "dirichlet",
        "source_function": lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
        "analytical_solution": lambda x, y: -np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)
    }


@pytest.fixture
def crossbar_config() -> Dict[str, Any]:
    """Default crossbar configuration for testing."""
    return {
        "size": 32,
        "conductance_range": (1e-9, 1e-6),
        "noise_model": "realistic",
        "device_type": "memristor"
    }


@pytest.fixture
def spice_sim_config() -> Dict[str, Any]:
    """SPICE simulation configuration."""
    return {
        "simulator": "ngspice",
        "temperature": 300,  # Kelvin
        "time_step": 1e-6,
        "stop_time": 1e-3
    }