# Getting Started with Analog PDE Solver

This tutorial will guide you through setting up and running your first analog PDE simulation.

## Prerequisites

Before starting, ensure you have:

- Python 3.9 or higher
- Git
- A UNIX-like environment (Linux, macOS, or WSL on Windows)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/analog-pde-solver-sim.git
cd analog-pde-solver-sim
```

### 2. Quick Setup

Use the provided setup script:

```bash
./scripts/setup-dev.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Check for required system tools
- Run basic validation tests

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install ngspice iverilog verilator
```

## Verify Installation

Run the test suite to verify everything is working:

```bash
# Run fast tests
make test-fast

# Or using pytest directly
pytest -v -m "not slow and not hardware"
```

## Your First Simulation

Once the implementation is complete, you'll be able to run:

```python
from analog_pde_solver import AnalogPDESolver, PoissonEquation
import numpy as np

# Define a simple Poisson equation
pde = PoissonEquation(
    domain_size=(64, 64),
    boundary_conditions="dirichlet"
)

# Create analog solver
solver = AnalogPDESolver(crossbar_size=64)

# This will work once the core implementation exists
# solution = solver.solve(pde)
```

## Next Steps

- Read the [Architecture Overview](../ARCHITECTURE.md)
- Explore the [Development Guide](DEVELOPMENT.md)
- Check out [example notebooks](../../examples/)
- Contribute to the project following [CONTRIBUTING.md](../../CONTRIBUTING.md)

## Getting Help

If you encounter issues:

1. Check the [troubleshooting guide](troubleshooting.md)
2. Search existing [GitHub issues](https://github.com/yourusername/analog-pde-solver-sim/issues)
3. Create a new issue with details about your problem

## Development Workflow

For contributors:

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make typecheck

# Run all checks
make test-all
```