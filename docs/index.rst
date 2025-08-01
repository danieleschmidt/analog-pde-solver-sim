Analog PDE Solver Documentation
================================

Welcome to the Analog PDE Solver documentation! This framework enables prototyping 
of in-memory analog accelerators for solving partial differential equations with 
100-1000× energy efficiency over digital methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   tutorials/index
   hardware_generation
   benchmarking
   examples
   troubleshooting
   contributing

Quick Start 
-----------

Install the package:

.. code-block:: bash

   pip install -e ".[dev,hardware]"

Solve a Poisson equation:

.. code-block:: python

   from analog_pde_solver import AnalogPDESolver, PoissonEquation
   import numpy as np

   # Define Poisson equation: ∇²φ = -ρ/ε₀
   pde = PoissonEquation(
       domain_size=(128, 128),
       boundary_conditions="dirichlet",
       source_function=lambda x, y: np.exp(-(x**2 + y**2))
   )

   # Create analog solver
   solver = AnalogPDESolver(
       crossbar_size=128,
       conductance_range=(1e-9, 1e-6),
       noise_model="realistic"
   )

   # Solve
   solution = solver.solve(pde, iterations=100)

Key Features
------------

* **Conductance-Aware Quantization**: Maps PDE coefficients to analog crossbar arrays
* **SPICE-Level Noise Modeling**: Realistic analog behavior simulation  
* **PyTorch-to-RTL Transpiler**: Automatic hardware generation from PDE specifications
* **Multi-Physics Support**: Poisson, Navier-Stokes, heat, and wave equations

Performance Projections
-----------------------

.. list-table:: Performance Comparison
   :header-rows: 1
   :class: benchmark-table

   * - PDE Type
     - Digital (GPU)
     - Analog (Projected)
     - Speedup
   * - Poisson 2D (1024²)
     - 125 ms
     - 0.12 ms
     - 1,042×
   * - Heat 3D (256³)
     - 2.1 s
     - 1.8 ms
     - 1,167×
   * - Navier-Stokes 2D
     - 485 ms
     - 0.8 ms
     - 606×

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`