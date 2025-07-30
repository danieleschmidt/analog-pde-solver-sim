# Architecture Overview

## System Design

The analog-pde-solver-sim project implements a multi-layer architecture for simulating analog computing solutions to partial differential equations.

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  Python API │ Jupyter Notebooks │ CLI Tools │ Visualization │
├─────────────────────────────────────────────────────────────┤
│                   Algorithm Layer                           │
│ PDE Solvers │ Multigrid │ Adaptive Precision │ Convergence  │
├─────────────────────────────────────────────────────────────┤
│                   Hardware Layer                            │
│ Crossbar Arrays │ Memristor Models │ DAC/ADC │ Noise Models │
├─────────────────────────────────────────────────────────────┤
│                  Simulation Layer                           │
│    SPICE Engine    │    Verilog HDL    │    Mixed Signal   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PDE Equation Framework

**Purpose**: Unified interface for defining and discretizing PDEs

```python
class PDEEquation(ABC):
    def discretize(self) -> np.ndarray: ...
    def boundary_conditions(self) -> Dict: ...
    def source_terms(self) -> np.ndarray: ...
```

**Supported Equations**:
- Poisson: ∇²φ = -ρ/ε₀
- Heat: ∂u/∂t = α∇²u
- Wave: ∂²u/∂t² = c²∇²u
- Navier-Stokes: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u

### 2. Analog Crossbar Engine

**Purpose**: Maps PDE matrix operations to conductance arrays

```python
class AnalogCrossbarArray:
    def program_conductances(self, matrix: np.ndarray): ...
    def compute_vmm(self, input_vector: np.ndarray): ...
    def add_noise(self, operation_type: str): ...
```

**Key Features**:
- Differential pair programming for signed values
- Realistic noise modeling (1/f, thermal, shot)
- Non-linear device characteristics
- Parasitic resistance effects

### 3. SPICE Integration Engine

**Purpose**: Circuit-level simulation of analog hardware

```python
class SPICESimulator:
    def generate_netlist(self, crossbar: CrossbarArray): ...
    def run_transient(self, stop_time: float): ...
    def extract_results(self) -> SimulationResults: ...
```

**Capabilities**:
- Automatic netlist generation
- Memristor SPICE models
- Mixed-signal interface simulation
- Parasitic extraction

### 4. Hardware Description Generator

**Purpose**: Synthesizable RTL for FPGA/ASIC implementation

```python
class VerilogGenerator:
    def generate_crossbar_module(self): ...
    def generate_control_logic(self): ...
    def generate_mixed_signal_interface(self): ...
```

## Data Flow Architecture

### Forward Simulation Path
1. **PDE Definition** → Discretized matrix equations
2. **Matrix Mapping** → Conductance programming values
3. **Hardware Simulation** → Analog computation results
4. **Digital Interface** → ADC quantized outputs
5. **Solution Reconstruction** → Final PDE solution

### Optimization Loop
1. **Error Analysis** → Compare analog vs. digital solutions
2. **Parameter Tuning** → Adjust conductance ranges, precision
3. **Hardware Reconfiguration** → Update crossbar programming
4. **Performance Analysis** → Energy, speed, accuracy metrics

## Memory Architecture

### Crossbar Array Organization
```
Row Drivers (DACs)     Crossbar Array        Column Sense (ADCs)
┌─────────────┐       ┌─────────────────┐    ┌─────────────────┐
│ V₁ ─────────┼───────┤ G₁₁  G₁₂  G₁₃   ├────┼──── I₁ → ADC₁  │
│ V₂ ─────────┼───────┤ G₂₁  G₂₂  G₂₃   ├────┼──── I₂ → ADC₂  │
│ V₃ ─────────┼───────┤ G₃₁  G₃₂  G₃₃   ├────┼──── I₃ → ADC₃  │
└─────────────┘       └─────────────────┘    └─────────────────┘
```

### Memory Hierarchy
- **L1**: On-chip crossbar arrays (128×128 typical)
- **L2**: Multi-crossbar systems for large problems
- **L3**: Host memory for intermediate results
- **Storage**: Persistent model checkpoints

## Scalability Design

### Horizontal Scaling
- **Multi-Crossbar Arrays**: Parallel processing of large matrices
- **Domain Decomposition**: Spatial partitioning for distributed solving
- **Pipeline Architecture**: Overlapped computation and communication

### Vertical Scaling
- **Hierarchical Multigrid**: Coarse-to-fine solution refinement
- **Adaptive Precision**: Dynamic bit-width optimization
- **Selective Computing**: Focus resources on high-gradient regions

## Performance Characteristics

### Computational Complexity
- **Matrix-Vector Multiply**: O(1) in analog domain
- **Iterative Solver**: O(k×n) where k = iterations, n = unknowns
- **Overall Complexity**: Dominated by convergence rate

### Energy Model
```
E_total = E_programming + E_computation + E_readout
E_computation = k × (E_DAC + E_crossbar + E_ADC)
```

### Accuracy Analysis
- **Quantization Error**: DAC/ADC resolution limits
- **Device Variation**: Memristor programming accuracy
- **Noise Effects**: Thermal, shot, and 1/f noise
- **Parasitic Effects**: Wire resistance, capacitance

## Design Patterns

### 1. Strategy Pattern
Different PDE types implement common solver interface

### 2. Factory Pattern
Hardware configurations created based on problem requirements

### 3. Observer Pattern
Monitoring and logging during long simulations

### 4. Template Method Pattern
Common optimization steps with problem-specific implementations

## Extension Points

### New PDE Types
1. Inherit from `PDEEquation` base class
2. Implement discretization method
3. Define boundary condition handling
4. Add to equation registry

### Custom Hardware Models
1. Extend `MemristorDevice` interface
2. Provide SPICE subcircuit model
3. Implement noise characteristics
4. Validate against experimental data

### Alternative Algorithms
1. Implement `IterativeSolver` interface
2. Define convergence criteria
3. Add performance benchmarking
4. Document accuracy trade-offs

This architecture balances flexibility for research exploration with performance for practical analog computing applications.