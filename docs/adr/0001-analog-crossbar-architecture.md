# ADR-0001: Analog Crossbar Architecture for PDE Solving

Date: 2025-08-02

## Status

Accepted

## Context

Traditional digital PDE solvers require significant computational resources and energy for iterative matrix operations. Analog crossbar arrays offer the potential for massive parallelization and energy efficiency through in-memory computing, but require careful architectural decisions for:

- Matrix decomposition strategies (handling negative values)
- Precision vs. energy tradeoffs
- Noise tolerance and error accumulation
- Hardware-software interface design

## Decision

We adopt a **differential conductance encoding** architecture with the following key components:

1. **Dual Crossbar Arrays**: Separate positive and negative weight matrices
2. **Differential Current Sensing**: Output = I_positive - I_negative
3. **Multi-Level Conductance**: 8-bit equivalent precision through conductance quantization
4. **Iterative Refinement**: Gauss-Seidel style updates leveraging analog noise as regularization

### Technical Specifications

- Conductance range: 1nS to 1μS (1000:1 dynamic range)
- Target precision: ~6-8 effective bits for PDE convergence
- Array sizes: 128x128 to 1024x1024 depending on problem scale
- Mixed-signal interface: 8-bit DACs, 10-bit ADCs

## Consequences

### Positive

- **Energy Efficiency**: 100-1000× reduction in energy per operation
- **Parallelization**: O(1) matrix-vector multiplication
- **Scalability**: Linear scaling with problem size
- **Noise Tolerance**: Iterative methods naturally handle analog imprecision

### Negative

- **Limited Precision**: Cannot achieve arbitrary numerical precision
- **Device Variations**: Requires calibration and compensation schemes
- **Complex Programming**: Conductance programming adds latency
- **Temperature Sensitivity**: Performance varies with environmental conditions

### Neutral

- **SPICE Integration**: Enables accurate modeling but increases simulation complexity
- **Hardware Dependency**: Solutions tied to specific analog architectures

## Implementation

- [x] Core crossbar array simulation (`analog_pde_solver/core/crossbar.py`)
- [x] SPICE integration framework (`analog_pde_solver/spice/simulator.py`)
- [x] Differential encoding algorithms
- [ ] Temperature compensation methods
- [ ] Adaptive precision control
- [ ] Hardware validation on FPGA platforms

## References

- Nature Photonics: "Sub-milliwatt in-pixel compute"
- EN100 IMC chip architecture documentation
- "Analog In-Memory Computing: Principles and Applications" - IEEE Proceedings
- Related: ADR-0002 (SPICE Integration Strategy)