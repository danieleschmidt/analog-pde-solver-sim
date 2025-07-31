import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_grid():
    """Create a sample 2D grid for testing."""
    return np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))


@pytest.fixture
def mock_spice_simulator():
    """Mock SPICE simulator for testing without external dependencies."""
    class MockSpiceSimulator:
        def __init__(self):
            self.components = []
            self.simulated = False
            
        def add_component(self, name, type_, **kwargs):
            self.components.append({"name": name, "type": type_, **kwargs})
            
        def simulate(self, **kwargs):
            self.simulated = True
            return {"success": True, "results": np.random.random((10, 10))}
    
    return MockSpiceSimulator()


@pytest.fixture
def sample_conductance_matrix():
    """Sample conductance matrix for crossbar testing."""
    return np.random.uniform(1e-9, 1e-6, (16, 16))


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def enable_slow_tests(request):
    """Enable slow tests when explicitly requested."""
    if request.config.getoption("--runslow"):
        return True
    pytest.skip("need --runslow option to run")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runhardware", action="store_true", default=False, help="run hardware tests"
    )