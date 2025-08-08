"""Main analog PDE solver implementation."""

import numpy as np
import logging
from typing import Dict, Any, Optional, Union
from .crossbar import AnalogCrossbarArray


class AnalogPDESolver:
    """Analog crossbar-based PDE solver with noise modeling."""
    
    def __init__(
        self,
        crossbar_size: int = 128,
        conductance_range: tuple = (1e-9, 1e-6),
        noise_model: str = "realistic"
    ):
        """Initialize analog PDE solver.
        
        Args:
            crossbar_size: Size of crossbar array (must be > 0)
            conductance_range: Min/max conductance values in Siemens (min < max, both > 0)
            noise_model: Noise modeling approach ('none', 'gaussian', 'realistic')
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate inputs
        self._validate_initialization_parameters(crossbar_size, conductance_range, noise_model)
        
        self.crossbar_size = crossbar_size
        self.conductance_range = conductance_range
        self.noise_model = noise_model
        
        try:
            self.crossbar = AnalogCrossbarArray(crossbar_size, crossbar_size)
            self.logger.info(f"Initialized AnalogPDESolver with {crossbar_size}×{crossbar_size} crossbar")
        except Exception as e:
            self.logger.error(f"Failed to initialize crossbar: {e}")
            raise RuntimeError(f"Crossbar initialization failed: {e}") from e
        
    def map_pde_to_crossbar(self, pde) -> Dict[str, Any]:
        """Map PDE discretization matrix to crossbar conductances.
        
        Args:
            pde: PDE object to map
            
        Returns:
            Configuration dictionary with mapping results
            
        Raises:
            ValueError: If PDE is invalid
            RuntimeError: If mapping fails
        """
        self.logger.debug("Starting PDE to crossbar mapping")
        
        # Validate PDE object
        self._validate_pde_object(pde)
        
        try:
            # Generate finite difference Laplacian matrix
            size = self.crossbar_size
            laplacian = self._create_laplacian_matrix(size)
            
            # Program crossbar with Laplacian operator
            self.crossbar.program_conductances(laplacian)
            
            self.logger.info(f"Successfully mapped {size}×{size} PDE to crossbar")
            
            return {
                "matrix_size": size,
                "conductance_range": self.conductance_range,
                "programming_success": True,
                "pde_type": type(pde).__name__
            }
            
        except Exception as e:
            self.logger.error(f"PDE to crossbar mapping failed: {e}")
            raise RuntimeError(f"Failed to map PDE to crossbar: {e}") from e
        
    def solve(
        self, 
        pde,
        iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> np.ndarray:
        """Solve PDE using analog crossbar computation.
        
        Args:
            pde: PDE object to solve
            iterations: Maximum number of iterations (must be > 0)
            convergence_threshold: Convergence threshold (must be > 0)
            
        Returns:
            Solution array
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If solving fails
        """
        self.logger.debug(f"Starting PDE solve with {iterations} max iterations")
        
        # Validate inputs
        self._validate_solve_parameters(iterations, convergence_threshold)
        
        try:
            # Map PDE to crossbar
            config = self.map_pde_to_crossbar(pde)
            
            # Initialize solution vector
            size = config["matrix_size"]
            phi = np.random.random(size) * 0.1
            
            # Create source term
            if hasattr(pde, 'source_function') and pde.source_function:
                x = np.linspace(0, 1, size)
                source = np.array([pde.source_function(xi, 0) for xi in x])
            else:
                source = np.ones(size) * 0.1
            
            # Iterative analog solver with convergence tracking
            convergence_history = []
            
            for i in range(iterations):
                try:
                    # Analog matrix-vector multiplication
                    residual = self.crossbar.compute_vmm(phi) + source
                    
                    # Jacobi-style update with stability check
                    phi_new = phi - 0.1 * residual
                    
                    # Check for numerical instability
                    if not np.isfinite(phi_new).all():
                        self.logger.warning(f"Numerical instability detected at iteration {i}")
                        phi_new = np.clip(phi_new, -1e6, 1e6)  # Clamp values
                        phi_new[~np.isfinite(phi_new)] = 0.0   # Replace NaN/inf with zero
                    
                    # Apply boundary conditions
                    phi_new[0] = 0.0  # Dirichlet BC
                    phi_new[-1] = 0.0
                    
                    # Check convergence
                    error = np.linalg.norm(phi_new - phi)
                    convergence_history.append(error)
                    
                    phi = phi_new
                    
                    if error < convergence_threshold:
                        self.logger.info(f"Converged after {i+1} iterations (error: {error:.2e})")
                        break
                        
                    if i > 10 and error > convergence_history[-10]:
                        self.logger.warning("Convergence may be stalling or diverging")
                        
                except Exception as e:
                    self.logger.error(f"Error in iteration {i}: {e}")
                    raise RuntimeError(f"Solver failed at iteration {i}: {e}") from e
            else:
                self.logger.warning(f"Did not converge after {iterations} iterations (error: {convergence_history[-1]:.2e})")
            
            self.logger.debug(f"Solution computed, norm: {np.linalg.norm(phi):.6f}")
            return phi
            
        except Exception as e:
            self.logger.error(f"PDE solving failed: {e}")
            if isinstance(e, (ValueError, RuntimeError)):
                raise  # Re-raise validation and solver errors as-is
            else:
                raise RuntimeError(f"Unexpected error during solving: {e}") from e
    
    def _create_laplacian_matrix(self, size: int) -> np.ndarray:
        """Create finite difference Laplacian matrix."""
        laplacian = np.zeros((size, size))
        
        # Main diagonal
        np.fill_diagonal(laplacian, -2.0)
        
        # Off-diagonals
        for i in range(size - 1):
            laplacian[i, i + 1] = 1.0
            laplacian[i + 1, i] = 1.0
            
        return laplacian
    
    def _validate_initialization_parameters(
        self, 
        crossbar_size: int, 
        conductance_range: tuple, 
        noise_model: str
    ) -> None:
        """Validate initialization parameters.
        
        Args:
            crossbar_size: Crossbar array size
            conductance_range: Conductance range tuple
            noise_model: Noise model string
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        # Validate crossbar_size
        if not isinstance(crossbar_size, int):
            raise TypeError(f"crossbar_size must be int, got {type(crossbar_size)}")
        if crossbar_size <= 0:
            raise ValueError(f"crossbar_size must be positive, got {crossbar_size}")
        if crossbar_size > 10000:
            raise ValueError(f"crossbar_size too large (>{10000}), got {crossbar_size}")
            
        # Validate conductance_range
        if not isinstance(conductance_range, (tuple, list)):
            raise TypeError(f"conductance_range must be tuple/list, got {type(conductance_range)}")
        if len(conductance_range) != 2:
            raise ValueError(f"conductance_range must have 2 elements, got {len(conductance_range)}")
        
        g_min, g_max = conductance_range
        if not isinstance(g_min, (int, float)) or not isinstance(g_max, (int, float)):
            raise TypeError("conductance_range values must be numeric")
        if g_min <= 0 or g_max <= 0:
            raise ValueError(f"conductance values must be positive, got ({g_min}, {g_max})")
        if g_min >= g_max:
            raise ValueError(f"g_min must be < g_max, got ({g_min}, {g_max})")
            
        # Validate noise_model
        if not isinstance(noise_model, str):
            raise TypeError(f"noise_model must be str, got {type(noise_model)}")
        valid_noise_models = ["none", "gaussian", "realistic"]
        if noise_model not in valid_noise_models:
            raise ValueError(f"noise_model must be one of {valid_noise_models}, got '{noise_model}'")
    
    def _validate_pde_object(self, pde) -> None:
        """Validate PDE object has required attributes.
        
        Args:
            pde: PDE object to validate
            
        Raises:
            ValueError: If PDE object is invalid
            AttributeError: If required attributes are missing
        """
        if pde is None:
            raise ValueError("PDE object cannot be None")
            
        # Check for required attributes
        if not hasattr(pde, 'domain_size'):
            raise AttributeError("PDE object must have 'domain_size' attribute")
            
        # Validate domain size compatibility
        if isinstance(pde.domain_size, (tuple, list)):
            domain_size = pde.domain_size[0] if len(pde.domain_size) > 0 else 0
        else:
            domain_size = pde.domain_size
            
        if domain_size != self.crossbar_size:
            self.logger.warning(
                f"PDE domain size ({domain_size}) != crossbar size ({self.crossbar_size}). "
                "This may cause issues."
            )
    
    def _validate_solve_parameters(
        self, 
        iterations: int, 
        convergence_threshold: float
    ) -> None:
        """Validate solve method parameters.
        
        Args:
            iterations: Number of iterations
            convergence_threshold: Convergence threshold
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        if not isinstance(iterations, int):
            raise TypeError(f"iterations must be int, got {type(iterations)}")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if iterations > 100000:
            raise ValueError(f"iterations too large (>100000), got {iterations}")
            
        if not isinstance(convergence_threshold, (int, float)):
            raise TypeError(f"convergence_threshold must be numeric, got {type(convergence_threshold)}")
        if convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold must be positive, got {convergence_threshold}")
        if convergence_threshold > 1.0:
            raise ValueError(f"convergence_threshold too large (>1.0), got {convergence_threshold}")