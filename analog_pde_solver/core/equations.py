"""PDE equation definitions and discretization."""

import numpy as np
from typing import Callable, Optional


class PoissonEquation:
    """Poisson equation: ∇²φ = -ρ/ε₀"""
    
    def __init__(
        self,
        domain_size: tuple,
        boundary_conditions: str = "dirichlet",
        source_function: Optional[Callable] = None
    ):
        self.domain_size = domain_size
        self.boundary_conditions = boundary_conditions
        self.source_function = source_function
        
    def solve_digital(self) -> np.ndarray:
        """Reference digital solution for comparison."""
        # Placeholder for digital finite difference solver
        return np.zeros(self.domain_size)


class NavierStokesEquation:
    """Incompressible Navier-Stokes equations."""
    
    def __init__(
        self,
        resolution: tuple,
        reynolds_number: float = 1000,
        time_step: float = 0.001
    ):
        self.resolution = resolution
        self.reynolds_number = reynolds_number
        self.time_step = time_step