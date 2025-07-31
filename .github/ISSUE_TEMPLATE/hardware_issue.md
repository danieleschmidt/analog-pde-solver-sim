---
name: Hardware Issue
about: Report issues with SPICE simulation or RTL generation
title: '[HARDWARE] '
labels: hardware, bug
assignees: ''

---

## Hardware Issue Description

Describe the hardware-related issue you're experiencing.

## Component

- [ ] SPICE simulation
- [ ] RTL generation
- [ ] Verilog synthesis
- [ ] Analog crossbar modeling
- [ ] Mixed-signal interface
- [ ] Other: ___________

## Configuration

**PDE Configuration:**
- PDE type: [e.g., Poisson, Heat, Navier-Stokes]
- Grid size: [e.g., 128x128]
- Boundary conditions: [e.g., Dirichlet, Neumann]

**Hardware Configuration:**
- Crossbar size: [e.g., 64x64]
- Precision: [e.g., 8-bit]
- Conductance range: [e.g., 1nS - 1μS]
- Noise model: [e.g., realistic, ideal]

**Tools:**
- SPICE simulator: [e.g., NgSpice 38]
- Verilog tools: [e.g., Icarus 12.0, Verilator 5.0]
- Synthesis tool: [e.g., Vivado 2023.1, Quartus 22.1]

## Error Output

```
Paste SPICE/Verilog error messages here
```

## Expected Hardware Behavior

What should happen in the hardware simulation/generation?

## Actual Hardware Behavior

What actually happened?

## Simulation/Synthesis Files

If possible, attach relevant .cir, .v, or log files.

## Performance Impact

- Simulation time: [e.g., expected vs actual]
- Resource usage: [e.g., LUTs, DSPs, BRAM]
- Power consumption: [e.g., estimated vs simulated]

## Workaround

If you found a temporary workaround, please describe it.