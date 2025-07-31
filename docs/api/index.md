# API Reference

Complete API documentation for the analog PDE solver framework.

## Core Modules

### PDE Abstractions
- [`analog_pde_solver.core`](core.md) - Base PDE classes and interfaces
- [`analog_pde_solver.equations`](equations.md) - Specific PDE implementations

### Analog Hardware
- [`analog_pde_solver.analog`](analog.md) - Crossbar arrays and analog simulation
- [`analog_pde_solver.spice`](spice.md) - SPICE integration and circuit modeling
- [`analog_pde_solver.noise`](noise.md) - Noise models and non-idealities

### Compilation and Generation
- [`analog_pde_solver.compiler`](compiler.md) - PyTorch to RTL compilation
- [`analog_pde_solver.rtl`](rtl.md) - Verilog generation and synthesis

### Utilities
- [`analog_pde_solver.visualization`](visualization.md) - Plotting and analysis tools
- [`analog_pde_solver.benchmarks`](benchmarks.md) - Performance benchmarking
- [`analog_pde_solver.utils`](utils.md) - Common utilities

## Quick Reference

### Basic Usage Pattern

```python
from analog_pde_solver import AnalogPDESolver, PoissonEquation

# 1. Define PDE
pde = PoissonEquation(domain_size=(128, 128))

# 2. Create solver
solver = AnalogPDESolver(crossbar_size=128)

# 3. Solve
solution = solver.solve(pde)
```

### Hardware Generation

```python
from analog_pde_solver.compiler import TorchToAnalog

# Compile PyTorch model to analog hardware
compiler = TorchToAnalog()
analog_model = compiler.compile(pytorch_model)
analog_model.export_rtl("solver.v")
```

### SPICE Simulation

```python
from analog_pde_solver.spice import SPICESimulator

# Circuit-level simulation
spice_sim = SPICESimulator()
results = spice_sim.simulate_crossbar(conductance_matrix)
```

## Type Annotations

The library uses comprehensive type hints. Import common types:

```python
from analog_pde_solver.types import (
    ConductanceMatrix,
    PDEConfig,
    SolutionArray,
    HardwareSpec
)
```

## Error Handling

Common exceptions:

- `PDESolverError`: Base exception for PDE solving errors
- `AnalogHardwareError`: Hardware simulation/generation errors
- `SPICESimulationError`: SPICE-specific errors
- `CompilationError`: RTL generation errors

## Configuration

Global configuration through environment variables:

```bash
# SPICE simulator path
export SPICE_SIMULATOR_PATH=/usr/local/bin/ngspice

# Verilog tools
export VERILOG_SIMULATOR=iverilog
export SYNTHESIS_TOOL=vivado

# Temporary directory for simulations
export ANALOG_PDE_TEMP_DIR=/tmp/analog_pde
```

Or programmatically:

```python
from analog_pde_solver import config

config.set_spice_path("/usr/local/bin/ngspice")
config.set_temp_dir("/tmp/my_simulations")
```

## Performance Tips

1. **Use appropriate precision**: Lower precision = faster simulation
2. **Cache conductance matrices**: Avoid repeated programming
3. **Parallel crossbars**: Use multiple arrays for large problems
4. **Warm-up simulations**: First SPICE run is slower due to model loading

## Development

For extending the library:

- Inherit from `BasePDE` for new equation types
- Implement `NoiseModel` interface for custom noise
- Use `RTLGenerator` base class for new hardware targets
- Follow type annotations and docstring conventions

## See Also

- [Architecture Overview](../ARCHITECTURE.md)
- [Development Guide](../DEVELOPMENT.md)
- [Tutorials](../tutorials/)
- [Examples](../../examples/)