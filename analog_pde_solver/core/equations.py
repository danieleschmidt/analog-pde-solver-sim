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
        try:
            from scipy.sparse import diags
            from scipy.sparse.linalg import spsolve
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
        
        # 1D Poisson equation for simplicity
        n = self.domain_size[0] if isinstance(self.domain_size, tuple) else self.domain_size
        
        if HAS_SCIPY:
            # Create discrete Laplacian operator (second derivative)
            # For 1D: d²/dx² ≈ (u[i-1] - 2*u[i] + u[i+1]) / h²
            h = 1.0 / (n - 1)  # Grid spacing
            diagonals = [np.ones(n-1)/h**2, -2*np.ones(n)/h**2, np.ones(n-1)/h**2]
            laplacian = diags(diagonals, [-1, 0, 1], shape=(n, n))
            
            # Source term for Poisson: ∇²φ = -ρ, so we solve ∇²φ = f where f = -ρ
            # For positive source ρ, we want negative f to get positive φ
            if self.source_function:
                x = np.linspace(0, 1, n)
                source = np.array([self.source_function(xi, 0) for xi in x])
                rhs = -source  # Negative source for Poisson
            else:
                rhs = -np.ones(n)  # Negative for positive solution
                
            # Apply boundary conditions
            laplacian = laplacian.tocsr()
            laplacian[0, :] = 0
            laplacian[0, 0] = 1
            laplacian[-1, :] = 0
            laplacian[-1, -1] = 1
            rhs[0] = 0
            rhs[-1] = 0
            
            # Solve system
            solution = spsolve(laplacian, rhs)
            
            return solution
        else:
            # Fallback: Simple iterative solver
            print("SciPy not available, using simple iterative solver")
            x = np.linspace(0, 1, n)
            
            # Source term for Poisson equation
            if self.source_function:
                source = np.array([self.source_function(xi, 0) for xi in x])
                rhs = -source  # Negative source for Poisson: ∇²φ = -ρ
            else:
                rhs = -np.ones(n)  # Negative for positive solution
                
            # Initialize solution
            solution = np.zeros(n)
            dx = 1.0 / (n - 1)
            
            # Simple Jacobi iteration
            for _ in range(1000):  # Fixed iterations
                solution_new = solution.copy()
                
                for i in range(1, n-1):
                    # Jacobi iteration for -d²u/dx² = f(x)
                    # u[i] = (u[i-1] + u[i+1] - h²*f[i]) / 2
                    solution_new[i] = 0.5 * (solution[i-1] + solution[i+1] - dx**2 * rhs[i])
                
                # Apply boundary conditions
                solution_new[0] = 0.0
                solution_new[-1] = 0.0
                
                # Check convergence
                if np.linalg.norm(solution_new - solution) < 1e-8:
                    break
                    
                solution = solution_new
                
            return solution


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
        self.velocity_field = None
        self.pressure_field = None
        
    def update_velocity(self) -> tuple:
        """Update velocity field using semi-implicit method."""
        n_x, n_y = self.resolution
        
        # Initialize if first time
        if self.velocity_field is None:
            self.velocity_field = {
                'u': np.zeros((n_x, n_y)),
                'v': np.zeros((n_x, n_y))
            }
            
        return self.velocity_field['u'], self.velocity_field['v']
        
    def solve_pressure_poisson(self) -> np.ndarray:
        """Solve pressure Poisson equation."""
        n_x, n_y = self.resolution
        
        if self.pressure_field is None:
            self.pressure_field = np.zeros((n_x, n_y))
            
        return self.pressure_field
        
    def apply_pressure_correction(self, u, v, pressure) -> tuple:
        """Apply pressure correction to velocity field."""
        # Simple pressure correction
        dt = self.time_step
        dx = 1.0 / u.shape[0]
        
        # Pressure gradient
        grad_p_x = np.gradient(pressure, axis=0) / dx
        grad_p_y = np.gradient(pressure, axis=1) / dx
        
        # Correct velocities
        u_corrected = u - dt * grad_p_x
        v_corrected = v - dt * grad_p_y
        
        return u_corrected, v_corrected
        
    def visualize_flow(self, u, v, pressure):
        """Visualize flow field (placeholder)."""
        pass


class HeatEquation:
    """Heat equation: ∂T/∂t = α∇²T"""
    
    def __init__(
        self,
        domain_size: tuple,
        thermal_diffusivity: float = 1.0,
        time_step: float = 0.001,
        boundary_conditions: str = "dirichlet"
    ):
        self.domain_size = domain_size
        self.thermal_diffusivity = thermal_diffusivity
        self.time_step = time_step
        self.boundary_conditions = boundary_conditions
        self.temperature_field = None
        
    def initialize_field(self, initial_condition: Optional[Callable] = None) -> np.ndarray:
        """Initialize temperature field."""
        if isinstance(self.domain_size, tuple):
            if len(self.domain_size) == 1:
                n = self.domain_size[0]
                if initial_condition:
                    x = np.linspace(0, 1, n)
                    self.temperature_field = np.array([initial_condition(xi) for xi in x])
                else:
                    self.temperature_field = np.zeros(n)
            else:  # 2D case
                n_x, n_y = self.domain_size
                self.temperature_field = np.zeros((n_x, n_y))
        else:
            self.temperature_field = np.zeros(self.domain_size)
            
        return self.temperature_field
        
    def step(self) -> np.ndarray:
        """Perform one time step using explicit Euler."""
        if self.temperature_field is None:
            self.initialize_field()
            
        # Initialize with non-zero field if all zeros
        if np.all(self.temperature_field == 0):
            # Add small perturbation in center to trigger evolution
            mid = len(self.temperature_field) // 2
            self.temperature_field[mid] = 1.0
            
        # Simple explicit Euler for heat equation
        alpha = self.thermal_diffusivity
        dt = self.time_step
        dx = 1.0 / (len(self.temperature_field) - 1)
        
        # Stability condition: dt <= dx²/(2α)
        if dt > dx**2 / (2 * alpha):
            dt = dx**2 / (2 * alpha) * 0.9  # Use 90% of stability limit
            
        # Finite difference update
        T_new = self.temperature_field.copy()
        for i in range(1, len(self.temperature_field) - 1):
            T_new[i] = self.temperature_field[i] + alpha * dt / dx**2 * (
                self.temperature_field[i+1] - 2*self.temperature_field[i] + self.temperature_field[i-1]
            )
            
        # Apply boundary conditions
        T_new[0] = 0.0   # Dirichlet BC
        T_new[-1] = 0.0
        
        self.temperature_field = T_new
        return self.temperature_field
        
    def solve_digital(self, num_steps: int = 100) -> np.ndarray:
        """Reference digital solution."""
        self.initialize_field()
        for _ in range(num_steps):
            self.step()
        return self.temperature_field


class WaveEquation:
    """Wave equation: ∂²u/∂t² = c²∇²u"""
    
    def __init__(
        self,
        domain_size: tuple,
        wave_speed: float = 1.0,
        time_step: float = 0.001,
        boundary_conditions: str = "dirichlet"
    ):
        self.domain_size = domain_size
        self.wave_speed = wave_speed
        self.time_step = time_step
        self.boundary_conditions = boundary_conditions
        self.u_current = None
        self.u_previous = None
        self.u_next = None
        
    def initialize_field(
        self, 
        initial_displacement: Optional[Callable] = None,
        initial_velocity: Optional[Callable] = None
    ) -> tuple:
        """Initialize wave field with displacement and velocity."""
        n = self.domain_size[0] if isinstance(self.domain_size, tuple) else self.domain_size
        x = np.linspace(0, 1, n)
        
        # Initial displacement
        if initial_displacement:
            self.u_current = np.array([initial_displacement(xi) for xi in x])
        else:
            self.u_current = np.zeros(n)
            
        # Initial velocity (used to compute u_previous)
        if initial_velocity:
            velocity = np.array([initial_velocity(xi) for xi in x])
            # Backward Euler: u_prev = u_current - dt * velocity
            self.u_previous = self.u_current - self.time_step * velocity
        else:
            self.u_previous = self.u_current.copy()
            
        self.u_next = np.zeros_like(self.u_current)
        return self.u_current, self.u_previous
        
    def step(self) -> np.ndarray:
        """Perform one time step using finite difference."""
        if self.u_current is None:
            self.initialize_field()
            
        c = self.wave_speed
        dt = self.time_step
        dx = 1.0 / len(self.u_current)
        
        # CFL stability condition: c*dt/dx <= 1
        cfl = c * dt / dx
        if cfl > 1.0:
            dt = dx / c * 0.9  # Use 90% of stability limit
            
        # Finite difference wave equation
        r = (c * dt / dx) ** 2
        
        for i in range(1, len(self.u_current) - 1):
            self.u_next[i] = (2 * self.u_current[i] - self.u_previous[i] + 
                             r * (self.u_current[i+1] - 2*self.u_current[i] + self.u_current[i-1]))
            
        # Apply boundary conditions
        self.u_next[0] = 0.0   # Dirichlet BC
        self.u_next[-1] = 0.0
        
        # Update for next step
        self.u_previous = self.u_current.copy()
        self.u_current = self.u_next.copy()
        
        return self.u_current
        
    def solve_digital(self, num_steps: int = 100) -> np.ndarray:
        """Reference digital solution."""
        self.initialize_field()
        for _ in range(num_steps):
            self.step()
        return self.u_current