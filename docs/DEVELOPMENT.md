# Development Setup

## Prerequisites

- Python 3.9+
- SPICE simulator (NgSpice recommended)
- Verilog tools (Icarus Verilog, Verilator)
- CUDA-capable GPU (optional, for PyTorch acceleration)

## Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/yourusername/analog-pde-solver-sim.git
cd analog-pde-solver-sim
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ngspice iverilog verilator
```

**macOS:**
```bash
brew install ngspice icarus-verilog verilator
```

### 3. Verify Installation

```bash
python -c "import analog_pde_solver; print('Installation successful')"
ngspice --version
iverilog -V
```

## Development Workflow

### Code Quality
```bash
# Format code
black .

# Check style
flake8

# Type checking
mypy analog_pde_solver/

# Run tests
pytest --cov=analog_pde_solver
```

### Hardware Simulation
- SPICE netlists generated in `temp/spice/`
- Verilog testbenches in `tests/hardware/`
- Waveform outputs in `.vcd` format

### Documentation
```bash
cd docs
make html  # Generate Sphinx documentation
```

## Project Structure

```
analog-pde-solver-sim/
├── analog_pde_solver/       # Main package
│   ├── core/               # Core PDE classes
│   ├── analog/             # Analog hardware simulation
│   ├── compiler/           # RTL generation
│   └── visualization/      # Plotting and analysis
├── tests/                  # Test suite
├── examples/               # Usage examples
├── docs/                   # Documentation source
└── scripts/                # Development utilities
```