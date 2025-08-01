"""Standard benchmark problems for analog PDE solvers."""

import numpy as np
from typing import Dict, Any, Callable, Optional

from ..core.equations import PoissonEquation, NavierStokesEquation


class StandardProblems:
    """Collection of standard benchmark problems for PDE solvers."""
    
    def __init__(self):
        self._problems = self._initialize_problems()
    
    def _initialize_problems(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the standard problem suite."""
        
        problems = {
            "poisson_2d_sine": {
                "description": "2D Poisson equation with sinusoidal source",
                "equation_factory": self._poisson_2d_sine,
                "reference_solution": self._poisson_2d_sine_solution,
                "crossbar_size": 64,
                "max_iterations": 100,
                "tolerance": 1e-6,
                "expected_iterations": 45,
                "category": "elliptic"
            },
            
            "poisson_2d_gaussian": {
                "description": "2D Poisson equation with Gaussian source",
                "equation_factory": self._poisson_2d_gaussian,
                "reference_solution": None,  # Numerical reference
                "crossbar_size": 128,
                "max_iterations": 150,
                "tolerance": 1e-6,
                "expected_iterations": 65,
                "category": "elliptic"
            },
            
            "heat_1d_transient": {
                "description": "1D transient heat equation",
                "equation_factory": self._heat_1d_transient,
                "reference_solution": self._heat_1d_analytical,
                "crossbar_size": 32,
                "max_iterations": 200,
                "tolerance": 1e-5,
                "expected_iterations": 120,
                "category": "parabolic"
            },
            
            "wave_1d_simple": {
                "description": "1D wave equation with simple boundary conditions",
                "equation_factory": self._wave_1d_simple,
                "reference_solution": None,
                "crossbar_size": 64,
                "max_iterations": 300,
                "tolerance": 1e-5,
                "expected_iterations": 180,
                "category": "hyperbolic"
            },
            
            "navier_stokes_cavity": {
                "description": "2D lid-driven cavity flow (Navier-Stokes)",
                "equation_factory": self._navier_stokes_cavity,
                "reference_solution": None,
                "crossbar_size": 256,
                "max_iterations": 500,
                "tolerance": 1e-4,
                "expected_iterations": 350,
                "category": "nonlinear"
            },
            
            "laplace_2d_dirichlet": {
                "description": "2D Laplace equation with Dirichlet boundary conditions",
                "equation_factory": self._laplace_2d_dirichlet,
                "reference_solution": self._laplace_2d_analytical,
                "crossbar_size": 32,
                "max_iterations": 80,
                "tolerance": 1e-7,
                "expected_iterations": 40,
                "category": "elliptic"
            },
            
            "poisson_3d_cube": {
                "description": "3D Poisson equation on unit cube",
                "equation_factory": self._poisson_3d_cube,
                "reference_solution": None,
                "crossbar_size": 512,
                "max_iterations": 300,
                "tolerance": 1e-5,
                "expected_iterations": 200,
                "category": "elliptic_3d"
            },
            
            "diffusion_reaction": {
                "description": "Reaction-diffusion equation",
                "equation_factory": self._diffusion_reaction,
                "reference_solution": None,
                "crossbar_size": 128,
                "max_iterations": 400,
                "tolerance": 1e-5,
                "expected_iterations": 250,
                "category": "reaction_diffusion"
            }
        }
        
        return problems
    
    def get_problem(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific problem configuration."""
        return self._problems.get(name)
    
    def get_all_problem_names(self) -> list[str]:
        """Get list of all available problem names."""
        return list(self._problems.keys())
    
    def get_problems_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all problems in a specific category."""
        return {
            name: config for name, config in self._problems.items()
            if config.get("category") == category
        }
    
    # Problem factory methods
    def _poisson_2d_sine(self) -> PoissonEquation:
        """2D Poisson equation with sinusoidal source term."""
        def source_function(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)
        
        return PoissonEquation(
            domain_size=(64, 64),
            boundary_conditions="dirichlet",
            source_function=source_function
        )
    
    def _poisson_2d_sine_solution(self) -> np.ndarray:
        """Analytical solution for 2D Poisson with sine source."""
        x = np.linspace(0, 1, 64)
        y = np.linspace(0, 1, 64)
        X, Y = np.meshgrid(x, y)
        
        # Analytical solution: φ = sin(πx)sin(πy) / (2π²)
        return np.sin(np.pi * X) * np.sin(np.pi * Y) / (2 * np.pi**2)
    
    def _poisson_2d_gaussian(self) -> PoissonEquation:
        """2D Poisson equation with Gaussian source term."""
        def source_function(x, y):
            return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
        
        return PoissonEquation(
            domain_size=(128, 128),
            boundary_conditions="dirichlet",
            source_function=source_function
        )
    
    def _heat_1d_transient(self):
        """1D transient heat equation."""
        # Placeholder - would return proper Heat equation class
        return PoissonEquation(domain_size=(32,), boundary_conditions="dirichlet")
    
    def _heat_1d_analytical(self) -> np.ndarray:
        """Analytical solution for 1D heat equation."""
        x = np.linspace(0, 1, 32)
        t = 0.1  # Fixed time
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    def _wave_1d_simple(self):
        """1D wave equation with simple boundary conditions."""
        # Placeholder - would return proper Wave equation class
        return PoissonEquation(domain_size=(64,), boundary_conditions="dirichlet")
    
    def _navier_stokes_cavity(self) -> NavierStokesEquation:
        """2D lid-driven cavity flow problem."""
        return NavierStokesEquation(
            resolution=(64, 64),
            reynolds_number=100,
            time_step=0.01
        )
    
    def _laplace_2d_dirichlet(self) -> PoissonEquation:
        """2D Laplace equation (Poisson with zero source)."""
        def zero_source(x, y):
            return np.zeros_like(x)
        
        return PoissonEquation(
            domain_size=(32, 32),
            boundary_conditions="dirichlet",
            source_function=zero_source
        )
    
    def _laplace_2d_analytical(self) -> np.ndarray:
        """Analytical solution for 2D Laplace equation."""
        # Simple case: linear solution
        x = np.linspace(0, 1, 32)
        y = np.linspace(0, 1, 32)
        X, Y = np.meshgrid(x, y)
        
        return X + Y  # Linear solution satisfies Laplace equation
    
    def _poisson_3d_cube(self) -> PoissonEquation:
        """3D Poisson equation on unit cube."""
        def source_3d(x, y, z):
            return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        
        # Note: This would need a 3D-capable PoissonEquation class
        return PoissonEquation(
            domain_size=(8, 8, 8),  # Smaller for memory constraints
            boundary_conditions="dirichlet",
            source_function=source_3d
        )
    
    def _diffusion_reaction(self):
        """Reaction-diffusion equation."""
        # Placeholder - would return proper ReactionDiffusion equation class
        return PoissonEquation(
            domain_size=(128, 128),
            boundary_conditions="neumann"
        )
    
    def get_problem_statistics(self) -> Dict[str, Any]:
        """Get statistics about the problem suite."""
        categories = {}
        crossbar_sizes = []
        
        for name, config in self._problems.items():
            category = config.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
            crossbar_sizes.append(config.get("crossbar_size", 64))
        
        return {
            "total_problems": len(self._problems),
            "categories": categories,
            "crossbar_size_range": {
                "min": min(crossbar_sizes),
                "max": max(crossbar_sizes),
                "mean": np.mean(crossbar_sizes)
            },
            "problem_types": list(categories.keys())
        }