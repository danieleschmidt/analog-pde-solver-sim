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
            # Get PDE domain size
            domain_size = pde.domain_size[0] if isinstance(pde.domain_size, tuple) else pde.domain_size
            
            # Use domain size if smaller than crossbar, otherwise use crossbar size
            size = min(domain_size, self.crossbar_size)
            
            # Create PDE-specific discretization matrix
            pde_type = type(pde).__name__
            if pde_type == "PoissonEquation":
                matrix = self._create_poisson_matrix(size)
                self.logger.debug(f"Created Poisson discretization matrix ({size}×{size})")
            elif pde_type == "HeatEquation":
                matrix = self._create_heat_equation_matrix(size, pde.thermal_diffusivity, pde.time_step)
                self.logger.debug(f"Created Heat equation matrix ({size}×{size})")
            elif pde_type == "WaveEquation":
                matrix = self._create_wave_equation_matrix(size, pde.wave_speed, pde.time_step)
                self.logger.debug(f"Created Wave equation matrix ({size}×{size})")
            else:
                # Fallback to generic Laplacian for unknown PDE types
                matrix = self._create_laplacian_matrix(size)
                self.logger.warning(f"Unknown PDE type {pde_type}, using generic Laplacian")
            
            # Program crossbar with PDE-specific operator
            self.crossbar.program_conductances(matrix)
            
            self.logger.info(f"Successfully mapped {pde_type} ({size}×{size}) to crossbar")
            
            return {
                "matrix_size": size,
                "domain_size": domain_size,
                "conductance_range": self.conductance_range,
                "programming_success": True,
                "pde_type": pde_type,
                "matrix_condition_number": np.linalg.cond(matrix)
            }
            
        except Exception as e:
            self.logger.error(f"PDE to crossbar mapping failed: {e}")
            raise RuntimeError(f"Failed to map PDE to crossbar: {e}") from e
        
    def solve(
        self, 
        pde=None,
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
            
            # Create source term based on PDE type
            source = self._create_source_term(pde, size)
            self.logger.debug(f"Created source term with norm: {np.linalg.norm(source):.6f}")
            
            # PDE-specific iterative solver
            pde_type = type(pde).__name__
            convergence_history = []
            
            # Initialize previous solution for time-dependent PDEs
            phi_prev = None
            if pde_type in ["HeatEquation", "WaveEquation"]:
                phi_prev = phi.copy()
            
            for i in range(iterations):
                try:
                    if pde_type == "PoissonEquation":
                        phi_new = self._solve_poisson_iteration(phi, source)
                    elif pde_type == "HeatEquation":
                        phi_new = self._solve_heat_iteration(phi, source, pde)
                    elif pde_type == "WaveEquation":
                        phi_new = self._solve_wave_iteration(phi, phi_prev, source, pde)
                        phi_prev = phi.copy()  # Update previous time step
                    else:
                        # Generic Laplacian-based solver
                        phi_new = self._solve_generic_iteration(phi, source)
                    
                    # Check for numerical instability
                    if not np.isfinite(phi_new).all():
                        self.logger.warning(f"Numerical instability detected at iteration {i}")
                        phi_new = np.clip(phi_new, -1e6, 1e6)  # Clamp values
                        phi_new[~np.isfinite(phi_new)] = 0.0   # Replace NaN/inf with zero
                    
                    # Apply boundary conditions
                    phi_new = self._apply_boundary_conditions(phi_new, pde)
                    
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
    
    def to_rtl(self, target: str = "xilinx_ultrascale", optimization: str = "area"):
        """Generate RTL hardware description.
        
        Args:
            target: Target FPGA/ASIC platform
            optimization: Optimization strategy ('area', 'speed', 'power')
            
        Returns:
            RTL generator object
        """
        self.logger.info(f"Generating RTL for {target} with {optimization} optimization")
        from ..rtl.verilog_generator import VerilogGenerator, RTLConfig
        
        config = RTLConfig(
            target_platform=target,
            optimization_strategy=optimization,
            crossbar_size=self.crossbar_size
        )
        
        return VerilogGenerator(config)
    
    def _create_laplacian_matrix(self, size: int) -> np.ndarray:
        """Create finite difference Laplacian matrix."""
        # Use normalized discretization for better numerical stability
        laplacian = np.zeros((size, size))
        
        # Standard three-point stencil: [1, -2, 1]
        # Main diagonal: -2
        np.fill_diagonal(laplacian, -2.0)
        
        # Off-diagonals: 1
        for i in range(size - 1):
            laplacian[i, i + 1] = 1.0
            laplacian[i + 1, i] = 1.0
        
        # Apply boundary conditions in the matrix
        # First and last rows are identity for Dirichlet BC
        laplacian[0, :] = 0.0
        laplacian[0, 0] = 1.0
        laplacian[-1, :] = 0.0  
        laplacian[-1, -1] = 1.0
            
        return laplacian
    
    def _create_poisson_matrix(self, size: int) -> np.ndarray:
        """Create discretization matrix for Poisson equation: ∇²φ = -ρ."""
        # For Poisson equation, we want to solve ∇²φ = -ρ
        # This becomes: -Laplacian * φ = ρ (negative Laplacian)
        return -self._create_laplacian_matrix(size)
    
    def _create_heat_equation_matrix(self, size: int, alpha: float, dt: float) -> np.ndarray:
        """Create discretization matrix for heat equation: ∂T/∂t = α∇²T."""
        dx = 1.0 / (size - 1)
        r = alpha * dt / (dx**2)  # Diffusion number
        
        # Implicit Euler: (I - r*L) * T^(n+1) = T^n
        # We solve: (I - r*L) * T^(n+1) = T^n + source
        laplacian = self._create_laplacian_matrix(size)
        identity = np.eye(size)
        
        # Ensure stability for explicit scheme or use implicit
        if r <= 0.5:  # Stable explicit scheme
            matrix = identity + r * laplacian
        else:  # Use implicit scheme
            matrix = identity - r * laplacian
        
        return matrix
    
    def _create_wave_equation_matrix(self, size: int, c: float, dt: float) -> np.ndarray:
        """Create discretization matrix for wave equation: ∂²u/∂t² = c²∇²u."""
        dx = 1.0 / (size - 1)
        r = (c * dt / dx) ** 2  # CFL parameter
        
        # Second-order finite difference: u^(n+1) = 2u^n - u^(n-1) + r*L*u^n
        # Rearranged: u^(n+1) = (2I + r*L)*u^n - u^(n-1)
        laplacian = self._create_laplacian_matrix(size)
        identity = np.eye(size)
        
        # Ensure CFL stability
        if r > 1.0:
            self.logger.warning(f"CFL condition violated (r={r:.3f} > 1), reducing time step")
            r = 0.9  # Clamp to stable value
        
        matrix = 2 * identity + r * laplacian
        return matrix
    
    def _create_source_term(self, pde, size: int) -> np.ndarray:
        """Create source term vector based on PDE type and properties."""
        x = np.linspace(0, 1, size)
        
        # Initialize source term
        source = np.zeros(size)
        
        # Check if PDE has source function
        if hasattr(pde, 'source_function') and pde.source_function is not None:
            try:
                # Try to evaluate source function
                if hasattr(pde, 'domain_size') and isinstance(pde.domain_size, tuple) and len(pde.domain_size) >= 2:
                    # 2D case - use midpoint for y
                    source = np.array([pde.source_function(xi, 0.5) for xi in x])
                else:
                    # 1D case
                    source = np.array([pde.source_function(xi, 0) for xi in x])
            except (TypeError, ValueError):
                # Handle functions that expect different arguments
                try:
                    source = np.array([pde.source_function(xi) for xi in x])
                except:
                    self.logger.warning("Could not evaluate source function, using default")
                    source = np.ones(size) * 0.1
        else:
            # Default source terms based on PDE type
            pde_type = type(pde).__name__
            if pde_type == "PoissonEquation":
                # Default: Gaussian source
                source = np.exp(-((x - 0.5) / 0.2)**2)
            elif pde_type == "HeatEquation":
                # Default: Initial temperature profile
                source = np.sin(np.pi * x)
            elif pde_type == "WaveEquation":
                # Default: Initial pulse
                source = np.exp(-((x - 0.5) / 0.1)**2)
            else:
                # Generic source
                source = np.ones(size) * 0.1
        
        # Scale source term to match matrix discretization
        dx = 1.0 / (size - 1)
        source = source * (dx**2)
        
        # Set boundary source terms to zero (enforced by BC)
        source[0] = 0.0
        source[-1] = 0.0
        
        return source
    
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
    
    def _solve_poisson_iteration(self, phi: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Single iteration for Poisson equation solving."""
        # For Poisson equation: -∇²φ = ρ
        # Matrix form: A*φ = b where A = -Laplacian, b = source
        # We use Jacobi iteration: φ^(k+1) = φ^k - ω * (A*φ^k - b) / diag(A)
        
        # Compute A*phi using crossbar
        A_phi = self.crossbar.compute_vmm(phi)
        
        # Residual: A*phi - b
        residual = A_phi - source
        
        # Jacobi update with damping
        # For our matrix with diagonal = 18, we need proper scaling
        omega = 0.05  # Small damping factor for stability
        diagonal_value = 18.0  # From our -Laplacian matrix
        
        phi_new = phi - omega * residual / diagonal_value
        
        return phi_new
    
    def _solve_heat_iteration(self, phi: np.ndarray, source: np.ndarray, pde) -> np.ndarray:
        """Single iteration for heat equation solving."""
        # For heat equation: ∂T/∂t = α∇²T + source
        # Using implicit or explicit scheme based on stability
        matrix_phi = self.crossbar.compute_vmm(phi)
        
        # Add source term (which could include previous time step)
        phi_new = matrix_phi + source
        
        return phi_new
    
    def _solve_wave_iteration(self, phi: np.ndarray, phi_prev: np.ndarray, source: np.ndarray, pde) -> np.ndarray:
        """Single iteration for wave equation solving."""
        # For wave equation: ∂²u/∂t² = c²∇²u
        # Using explicit finite difference scheme
        if phi_prev is None:
            phi_prev = phi.copy()
        
        # Matrix operation gives: (2I + r*L)*u^n
        matrix_phi = self.crossbar.compute_vmm(phi)
        
        # Wave equation update: u^(n+1) = matrix_result - u^(n-1) + source
        phi_new = matrix_phi - phi_prev + source
        
        return phi_new
    
    def _solve_generic_iteration(self, phi: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Generic iterative solver for unknown PDE types."""
        # Fallback to simple Jacobi iteration
        residual = self.crossbar.compute_vmm(phi) + source
        phi_new = phi - 0.1 * residual
        return phi_new
    
    def _apply_boundary_conditions(self, phi: np.ndarray, pde) -> np.ndarray:
        """Apply boundary conditions based on PDE type."""
        # Default Dirichlet boundary conditions
        phi_bc = phi.copy()
        
        if hasattr(pde, 'boundary_conditions'):
            if pde.boundary_conditions.lower() == "dirichlet":
                phi_bc[0] = 0.0
                phi_bc[-1] = 0.0
            elif pde.boundary_conditions.lower() == "neumann":
                # Neumann BC: zero gradient (copy neighboring values)
                phi_bc[0] = phi_bc[1]
                phi_bc[-1] = phi_bc[-2]
            elif pde.boundary_conditions.lower() == "periodic":
                # Periodic BC
                phi_bc[0] = phi_bc[-2]
                phi_bc[-1] = phi_bc[1]
        else:
            # Default: homogeneous Dirichlet
            phi_bc[0] = 0.0
            phi_bc[-1] = 0.0
        
        return phi_bc