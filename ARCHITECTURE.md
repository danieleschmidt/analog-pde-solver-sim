# Architecture Overview

## System Design

The analog PDE solver simulation framework consists of three main layers:

### 1. PDE Abstraction Layer
- **Purpose**: Provide high-level PDE specification interface
- **Components**: 
  - `PoissonEquation`, `NavierStokesAnalog`, `HeatEquation` classes
  - Boundary condition handling
  - Multi-physics coupling interfaces

### 2. Analog Hardware Simulation Layer
- **Purpose**: Model analog crossbar arrays and peripheral circuits
- **Components**:
  - `AnalogCrossbarArray`: Conductance-based matrix operations
  - SPICE integration for circuit-level accuracy
  - Noise modeling (thermal, shot, flicker)
  - Non-ideality simulation (device variations, parasitic effects)

### 3. Hardware Generation Layer
- **Purpose**: Generate synthesizable RTL from PDE specifications
- **Components**:
  - PyTorch-to-Verilog compiler
  - Mixed-signal interface generation
  - Constraint generation for FPGA/ASIC flows

## Data Flow

```
PDE Specification → Matrix Decomposition → Conductance Mapping → 
SPICE Simulation → Analog Solution → RTL Generation
```

## Key Design Decisions

1. **Differential Conductance Encoding**: Positive and negative weights mapped to separate crossbar arrays
2. **Multi-Grid Hierarchy**: Coarse-to-fine solving for large problems
3. **Adaptive Precision**: Dynamic bit-width adjustment based on convergence
4. **Mixed-Signal Interface**: Standardized DAC/ADC interfaces for hardware portability

## Performance Model

Energy efficiency gains come from:
- Parallel analog computation (O(1) matrix-vector multiplication)
- Reduced data movement (in-memory computing)
- Lower precision requirements for iterative methods

## Extensibility Points

- New PDE types: Inherit from `BasePDE` class
- Custom noise models: Implement `NoiseModel` interface  
- Hardware targets: Add platform-specific RTL generators
- Optimization algorithms: Plugin architecture for solver methods