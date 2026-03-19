# Analog PDE Solver Simulation

Simulate how **analog circuits solve PDEs naturally** — and why they can be
orders of magnitude more energy-efficient than digital computers doing the
same computation.

## The core idea

A **resistive mesh** (grid of equal resistors connecting neighbouring nodes)
is a physical analogue computer for the 2-D **Laplace equation**:

```
∇²u = 0
```

Apply voltages to the boundary nodes and Kirchhoff's current law enforces
the finite-difference stencil at every interior node simultaneously:

```
u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4
```

The network **relaxes to the correct solution in continuous time** — no
clock, no multiplications, just resistors obeying Ohm's law.  Inject
current sources at nodes to model a forcing term, and you have the
**Poisson equation** (∇²u = f) for free.

This is not new science — analog mesh computers were used in the 1950s to
solve heat-distribution and electrostatic problems.  This repo simulates the
physics so you can study the energy advantage and accuracy trade-offs.

## Energy advantage

| Metric | Analog mesh | Digital CPU (Gauss–Seidel) |
|---|---|---|
| 20×20 Laplace, 336 iterations | ~28 pJ | ~1,200,000 pJ |
| **Efficiency gain** | **~43,000×** | 1× (baseline) |

Analog computation dissipates energy only in the resistors while settling
(proportional to V²/R × settling time).  Digital computation burns energy
for every floating-point operation (~1-5 pJ per FLOP).  For large grids at
moderate precision, the advantage can reach 100–1,000×.

The catch: analog has fixed precision set by component tolerances.  1%
resistor mismatch gives ~3% solution error.  For many engineering problems
that is perfectly acceptable.

## What this simulator does

```
AnalogPDESolver   — Gauss–Seidel relaxation with optional resistor-mismatch noise
DigitalSolver     — identical algorithm, pure floating-point (baseline)
EnergyModel       — estimates analog pJ (V²/R × τ) vs digital pJ (FLOP count)
```

Both solvers implement the same 5-point stencil.  The analog solver adds
multiplicative noise to each resistor link to model real component variation.

## Quick start

```bash
# Install (numpy + scipy are the only runtime deps)
pip install -e .

# Run the heat-distribution demo (20×20 grid, Laplace equation)
python examples/laplace_heat_demo.py

# Run the Poisson demo (known exact solution for error measurement)
python examples/poisson_demo.py

# Run tests
pytest tests/
```

## Demo output (excerpt)

```
Analog  energy :    28.026 pJ
Digital energy : 1209600.000 pJ
Efficiency gain: 43159.2× (analog vs digital)

Digital solution  (ground truth)       Analog solution  (no noise)
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ==**##########%%########**==           ==**##########%%########**==
  ::==++********####****++==::           ::==++********####****++==::
  ..--====++++******++++====--..         ..--====++++******++++====--..
  ...   [ heat diffuses down ]   ...     ...   [ identical! ]   ...
```

## Analog computing background

Classical analog computers used physical phenomena (voltage, current,
mechanical displacement) to directly model mathematical relationships.
Before transistors, differential equations were solved on analog computers
built from operational amplifiers and precision resistors.

The resistive mesh for PDEs works because:

1. **KCL is the discrete Laplacian** — "current in = current out" at each
   node is exactly the finite-difference form of ∇²u = 0
2. **Parallel computation** — every node updates simultaneously (unlike
   sequential digital iteration)
3. **No clock energy** — the settling process costs only resistive
   dissipation; there are no clocked flip-flops burning energy at idle

Modern revisits include **memristor crossbar arrays** (non-volatile
resistors) and **in-memory computing** chips that perform matrix-vector
multiplication (the inner loop of PDE solvers) directly in SRAM or DRAM,
without moving data to a CPU.

## Project layout

```
analog_pde_solver/
    __init__.py          package entry point
    solver.py            AnalogPDESolver, DigitalSolver, EnergyModel

examples/
    laplace_heat_demo.py 2-D heat distribution, 20×20, with ASCII output
    poisson_demo.py      Poisson equation, known exact solution

tests/
    test_solver.py       21 unit + integration tests
```

## License

MIT — see [LICENSE](LICENSE).
