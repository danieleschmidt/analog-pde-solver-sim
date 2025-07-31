# Analog Computing Fundamentals

This tutorial introduces the core concepts of analog computing for PDE solving.

## Why Analog Computing for PDEs?

Traditional digital computers solve PDEs through:
- Iterative matrix operations
- High precision floating-point arithmetic
- Sequential processing with limited parallelism

Analog computers leverage:
- **Parallel computation**: All matrix elements computed simultaneously
- **In-memory processing**: Computation happens where data is stored
- **Energy efficiency**: ~1000× lower power than digital equivalents
- **Natural noise**: Can accelerate certain iterative algorithms

## Crossbar Array Architecture

### Basic Principles

A crossbar array is a grid of programmable resistors (memristors):

```
    Col 0  Col 1  Col 2
      |      |      |
Row 0-R00----R01----R02-
      |      |      |
Row 1-R10----R11----R12-
      |      |      |
Row 2-R20----R21----R22-
      |      |      |
```

### Vector-Matrix Multiplication

Applying voltage `V` to rows produces current `I` at columns:

```
I = G × V
```

Where `G` is the conductance matrix (inverse of resistance).

### Mapping PDEs to Conductances

1. **Discretize PDE**: Convert continuous equation to matrix form
2. **Decompose matrix**: Split into positive and negative components
3. **Map to conductances**: Scale values to available conductance range
4. **Program crossbar**: Set each memristor to target conductance

## Finite Difference Example

For the 2D Poisson equation ∇²φ = -ρ/ε₀:

### Digital Approach
```python
# 5-point stencil finite difference
for i in range(1, N-1):
    for j in range(1, N-1):
        phi[i,j] = 0.25 * (
            phi[i+1,j] + phi[i-1,j] + 
            phi[i,j+1] + phi[i,j-1] + 
            rho[i,j] * dx**2
        )
```

### Analog Mapping
```python
# Create Laplacian operator matrix
L = create_laplacian_matrix(N)

# Map to conductances
G_pos = np.maximum(L, 0)  # Positive weights
G_neg = np.maximum(-L, 0) # Negative weights

# Program crossbars
crossbar_pos.program_conductances(G_pos)
crossbar_neg.program_conductances(G_neg)
```

## Noise and Non-Idealities

### Sources of Analog Noise

1. **Thermal noise**: Random fluctuations due to temperature
2. **Shot noise**: Discrete nature of charge carriers
3. **Flicker noise**: Low-frequency variations
4. **Device variations**: Manufacturing tolerances

### Noise as a Feature

For some algorithms, noise can be beneficial:
- **Simulated annealing**: Helps escape local minima
- **Stochastic optimization**: Natural exploration mechanism
- **Regularization**: Prevents overfitting in iterative methods

## Energy Efficiency Analysis

### Digital Energy Breakdown
- **Memory access**: ~100 pJ per operation
- **Floating-point math**: ~10-50 pJ per operation
- **Data movement**: ~10-100 pJ per bit

### Analog Energy Breakdown
- **Conductance read**: ~0.1 pJ per operation
- **DAC/ADC conversion**: ~1-10 pJ per conversion
- **Static power**: ~1 nW per crossbar

### Example Calculation

For 1024² Poisson equation:
- **Digital**: ~125 ms, ~12.5 J
- **Analog (projected)**: ~0.12 ms, ~12 mJ
- **Efficiency gain**: 1,042× faster, 1,000× lower energy

## Limitations and Challenges

### Precision Constraints
- Limited conductance precision (6-10 bits typical)
- ADC/DAC quantization effects
- Temperature and voltage variations

### Scalability Issues
- Wire resistance increases with array size
- Sneak path currents in large arrays
- Programming time scales with array size

### Algorithm Compatibility
- Not all algorithms map well to analog
- Iterative methods work best
- Global operations (FFT) are challenging

## When to Use Analog Computing

### Good Candidates
- ✅ Iterative PDE solvers (Jacobi, Gauss-Seidel)
- ✅ Large, sparse matrix operations
- ✅ Real-time applications
- ✅ Energy-constrained environments
- ✅ Moderate precision requirements (8-12 bits)

### Poor Candidates
- ❌ High precision requirements (>16 bits)
- ❌ Complex control flow
- ❌ Frequent matrix updates
- ❌ Algorithms requiring exact arithmetic

## Next Steps

- Learn about [PDE mapping techniques](03_pde_mapping.md)
- Explore [SPICE simulation](04_spice_simulation.md)
- Study [hardware design flow](05_hardware_flow.md)
- Practice with [example problems](../../examples/)

## References

1. Burr, G. W. et al. "Experimental demonstration and tolerancing of a large-scale neural network (165,000 synapses) using phase-change memory as the synaptic weight element." IEEE Trans. Electron Devices (2015).

2. Chi, P. et al. "PRIME: A novel processing-in-memory architecture for neural network computation in ReRAM-based main memory." ACM SIGARCH Computer Architecture News (2016).

3. Yao, P. et al. "Fully hardware-implemented memristor convolutional neural network." Nature (2020).