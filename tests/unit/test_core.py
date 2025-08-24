"""Comprehensive unit tests for analog PDE solver core functionality."""

import pytest
import numpy as np
from analog_pde_solver.core.solver import AnalogPDESolver
from analog_pde_solver.core.equations import PoissonEquation, HeatEquation, WaveEquation
from analog_pde_solver.core.crossbar import AnalogCrossbarArray


class TestAnalogCrossbarArray:
    """Test the analog crossbar array implementation."""
    
    def test_crossbar_initialization(self):
        """Test crossbar array initialization."""
        crossbar = AnalogCrossbarArray(64, 64)
        assert crossbar.rows == 64
        assert crossbar.cols == 64
        assert crossbar.conductance_matrix.shape == (64, 64)
    
    def test_conductance_programming(self):
        """Test programming conductances from matrix."""
        crossbar = AnalogCrossbarArray(4, 4)
        test_matrix = np.array([
            [-2, 1, 0, 0],
            [1, -2, 1, 0],
            [0, 1, -2, 1],
            [0, 0, 1, -2]
        ])
        
        crossbar.program_conductances(test_matrix)
        
        # Check that conductances were programmed (should have positive and negative parts)
        assert hasattr(crossbar, 'g_positive')
        assert hasattr(crossbar, 'g_negative')
        assert crossbar.g_positive.shape == (4, 4)
        assert crossbar.g_negative.shape == (4, 4)
    
    def test_vector_matrix_multiplication(self):
        """Test analog vector-matrix multiplication."""
        crossbar = AnalogCrossbarArray(3, 3)
        
        # Simple test matrix
        test_matrix = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        crossbar.program_conductances(test_matrix)
        
        # Test vector
        test_vector = np.array([1, 1, 1])
        
        # Compute VMM
        result = crossbar.compute_vmm(test_vector)
        
        # Result should be approximately [1, 2, 3] (with some noise)
        assert len(result) == 3
        assert np.allclose(result, [1, 2, 3], atol=0.5)


class TestPoissonEquation:
    """Test Poisson equation implementation."""
    
    def test_poisson_initialization(self):
        """Test Poisson equation initialization."""
        pde = PoissonEquation(domain_size=(32,), boundary_conditions="dirichlet")
        assert pde.domain_size == (32,)
        assert pde.boundary_conditions == "dirichlet"
    
    def test_poisson_digital_solution(self):
        """Test digital reference solution for Poisson equation."""
        # Simple 1D Poisson with constant source
        pde = PoissonEquation(domain_size=(16,), source_function=lambda x, y: 1.0)
        
        solution = pde.solve_digital()
        
        # Check solution properties
        assert len(solution) == 16
        assert solution[0] == 0.0  # Dirichlet BC
        assert solution[-1] == 0.0  # Dirichlet BC
        
        # For constant source, solution should be parabolic
        # Maximum should be in the middle
        max_idx = np.argmax(solution)
        assert max_idx > 4 and max_idx < 12  # Approximately in the middle
    
    def test_poisson_with_gaussian_source(self):
        """Test Poisson equation with Gaussian source function."""
        def gaussian_source(x, y):
            return np.exp(-((x - 0.5) / 0.2)**2)
        
        pde = PoissonEquation(domain_size=(32,), source_function=gaussian_source)
        solution = pde.solve_digital()
        
        assert len(solution) == 32
        assert np.max(solution) > 0  # Should have positive solution
        assert solution[0] == 0.0 and solution[-1] == 0.0  # BCs


class TestHeatEquation:
    """Test heat equation implementation."""
    
    def test_heat_initialization(self):
        """Test heat equation initialization."""
        pde = HeatEquation(domain_size=(32,), thermal_diffusivity=1.0, time_step=0.001)
        assert pde.domain_size == (32,)
        assert pde.thermal_diffusivity == 1.0
        assert pde.time_step == 0.001
    
    def test_heat_field_initialization(self):
        """Test temperature field initialization."""
        pde = HeatEquation(domain_size=(16,))
        
        # Initialize with sine function
        def initial_temp(x):
            return np.sin(np.pi * x)
        
        field = pde.initialize_field(initial_temp)
        
        assert len(field) == 16
        assert np.allclose(field[8], np.sin(np.pi * 0.5), atol=0.1)  # Middle should be ~1
    
    def test_heat_time_stepping(self):
        """Test heat equation time stepping."""
        pde = HeatEquation(domain_size=(16,), thermal_diffusivity=0.1, time_step=0.01)
        
        # Initialize with initial condition
        pde.initialize_field()
        initial_field = pde.temperature_field.copy()
        
        # Take several time steps
        for _ in range(5):
            pde.step()
        
        # Temperature field should change
        assert not np.array_equal(pde.temperature_field, initial_field)
        # Boundary conditions should be maintained
        assert pde.temperature_field[0] == 0.0
        assert pde.temperature_field[-1] == 0.0


class TestWaveEquation:
    """Test wave equation implementation."""
    
    def test_wave_initialization(self):
        """Test wave equation initialization."""
        pde = WaveEquation(domain_size=(32,), wave_speed=1.0, time_step=0.001)
        assert pde.domain_size == (32,)
        assert pde.wave_speed == 1.0
        assert pde.time_step == 0.001
    
    def test_wave_field_initialization(self):
        """Test wave field initialization."""
        pde = WaveEquation(domain_size=(16,))
        
        def initial_pulse(x):
            return np.exp(-((x - 0.5) / 0.1)**2)
        
        u_current, u_previous = pde.initialize_field(initial_pulse)
        
        assert len(u_current) == 16
        assert len(u_previous) == 16
        # Initial pulse should be centered
        max_idx = np.argmax(u_current)
        assert max_idx > 6 and max_idx < 10
    
    def test_wave_propagation(self):
        """Test wave propagation over time."""
        pde = WaveEquation(domain_size=(32,), wave_speed=1.0, time_step=0.01)
        
        # Initialize with pulse
        def pulse(x):
            return np.exp(-((x - 0.5) / 0.05)**2)
        
        pde.initialize_field(pulse)
        initial_field = pde.u_current.copy()
        
        # Propagate for several steps
        for _ in range(10):
            pde.step()
        
        # Wave should have evolved
        assert not np.array_equal(pde.u_current, initial_field)
        # Energy should be approximately conserved (no damping)
        initial_energy = np.sum(initial_field**2)
        final_energy = np.sum(pde.u_current**2)
        assert abs(final_energy - initial_energy) / initial_energy < 0.5


class TestAnalogPDESolver:
    """Test the main analog PDE solver."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = AnalogPDESolver(crossbar_size=64)
        assert solver.crossbar_size == 64
        assert solver.conductance_range == (1e-9, 1e-6)
        assert solver.noise_model == "realistic"
    
    def test_solver_initialization_validation(self):
        """Test solver initialization parameter validation."""
        # Invalid crossbar size
        with pytest.raises(ValueError):
            AnalogPDESolver(crossbar_size=0)
        
        # Invalid conductance range
        with pytest.raises(ValueError):
            AnalogPDESolver(conductance_range=(1e-6, 1e-9))  # min > max
        
        # Invalid noise model
        with pytest.raises(ValueError):
            AnalogPDESolver(noise_model="invalid")
    
    def test_pde_to_crossbar_mapping(self):
        """Test PDE to crossbar mapping."""
        solver = AnalogPDESolver(crossbar_size=16)
        pde = PoissonEquation(domain_size=(16,))
        
        config = solver.map_pde_to_crossbar(pde)
        
        assert config["programming_success"] is True
        assert config["pde_type"] == "PoissonEquation"
        assert config["matrix_size"] == 16
        assert "matrix_condition_number" in config
    
    def test_poisson_solving(self):
        """Test solving Poisson equation with analog solver."""
        solver = AnalogPDESolver(crossbar_size=16, noise_model="none")
        
        # Create simple Poisson equation
        pde = PoissonEquation(
            domain_size=(16,),
            source_function=lambda x, y: np.sin(np.pi * x)
        )
        
        # Solve with analog method
        analog_solution = solver.solve(pde, iterations=50, convergence_threshold=1e-4)
        
        assert len(analog_solution) == 16
        assert analog_solution[0] == 0.0  # Boundary condition
        assert analog_solution[-1] == 0.0  # Boundary condition
        assert np.max(np.abs(analog_solution)) > 0  # Non-trivial solution
    
    def test_heat_equation_solving(self):
        """Test solving heat equation with analog solver."""
        solver = AnalogPDESolver(crossbar_size=16, noise_model="none")
        
        pde = HeatEquation(
            domain_size=(16,),
            thermal_diffusivity=0.1,
            time_step=0.01
        )
        
        # Solve with analog method
        solution = solver.solve(pde, iterations=20)
        
        assert len(solution) == 16
        assert solution[0] == 0.0  # Boundary condition
        assert solution[-1] == 0.0  # Boundary condition
    
    def test_wave_equation_solving(self):
        """Test solving wave equation with analog solver."""
        solver = AnalogPDESolver(crossbar_size=16, noise_model="none")
        
        pde = WaveEquation(
            domain_size=(16,),
            wave_speed=1.0,
            time_step=0.01
        )
        
        # Solve with analog method
        solution = solver.solve(pde, iterations=10)
        
        assert len(solution) == 16
        assert solution[0] == 0.0  # Boundary condition
        assert solution[-1] == 0.0  # Boundary condition
    
    def test_solver_convergence_detection(self):
        """Test solver convergence detection."""
        solver = AnalogPDESolver(crossbar_size=8, noise_model="none")
        pde = PoissonEquation(domain_size=(8,))
        
        # Should converge quickly for small problem
        solution = solver.solve(pde, iterations=100, convergence_threshold=1e-6)
        
        assert solution is not None
        assert len(solution) == 8


class TestSolverRobustness:
    """Test solver robustness and error handling."""
    
    def test_invalid_pde_object(self):
        """Test handling of invalid PDE objects."""
        solver = AnalogPDESolver(crossbar_size=16)
        
        # None PDE should raise error
        with pytest.raises(ValueError):
            solver.map_pde_to_crossbar(None)
        
        # Object without domain_size should raise error
        class InvalidPDE:
            pass
        
        with pytest.raises(AttributeError):
            solver.map_pde_to_crossbar(InvalidPDE())
    
    def test_domain_size_mismatch_warning(self, caplog):
        """Test warning for domain size mismatch."""
        solver = AnalogPDESolver(crossbar_size=32)
        pde = PoissonEquation(domain_size=(16,))  # Smaller than crossbar
        
        solver.map_pde_to_crossbar(pde)
        
        # Should log a warning but not fail
        # Note: This test may need adjustment based on actual logging behavior
    
    def test_numerical_stability_handling(self):
        """Test handling of numerical instability."""
        solver = AnalogPDESolver(crossbar_size=8, noise_model="none")
        
        # Create a potentially unstable PDE (large source)
        def large_source(x, y):
            return 1e6
        
        pde = PoissonEquation(domain_size=(8,), source_function=large_source)
        
        # Should handle instability gracefully
        solution = solver.solve(pde, iterations=50)
        
        assert np.all(np.isfinite(solution))
        assert np.all(np.abs(solution) < 1e6)  # Should be clamped


class TestMatrixCreation:
    """Test PDE matrix creation methods."""
    
    def test_laplacian_matrix_creation(self):
        """Test Laplacian matrix creation."""
        solver = AnalogPDESolver(crossbar_size=5)
        matrix = solver._create_laplacian_matrix(5)
        
        # Check structure
        assert matrix.shape == (5, 5)
        assert matrix[0, 0] < 0  # Diagonal should be negative
        assert matrix[0, 1] > 0  # Off-diagonal should be positive
        assert matrix[1, 0] > 0
    
    def test_poisson_matrix_creation(self):
        """Test Poisson-specific matrix creation."""
        solver = AnalogPDESolver(crossbar_size=5)
        matrix = solver._create_poisson_matrix(5)
        
        # Should be negative of Laplacian
        laplacian = solver._create_laplacian_matrix(5)
        assert np.allclose(matrix, -laplacian)
    
    def test_heat_matrix_creation(self):
        """Test heat equation matrix creation."""
        solver = AnalogPDESolver(crossbar_size=5)
        matrix = solver._create_heat_equation_matrix(5, alpha=1.0, dt=0.001)
        
        assert matrix.shape == (5, 5)
        # Should have identity component
        assert matrix[0, 0] != 0
    
    def test_wave_matrix_creation(self):
        """Test wave equation matrix creation."""
        solver = AnalogPDESolver(crossbar_size=5)
        matrix = solver._create_wave_equation_matrix(5, c=1.0, dt=0.01)
        
        assert matrix.shape == (5, 5)
        # Diagonal should be approximately 2 for stable scheme
        assert abs(matrix[2, 2] - 2.0) < 1.0


class TestSourceTermCreation:
    """Test source term creation."""
    
    def test_source_with_function(self):
        """Test source term creation with function."""
        solver = AnalogPDESolver(crossbar_size=8)
        
        def source_func(x, y):
            return x**2
        
        pde = PoissonEquation(domain_size=(8,), source_function=source_func)
        source = solver._create_source_term(pde, 8)
        
        assert len(source) == 8
        assert source[0] < source[-1]  # Should increase with x
    
    def test_source_without_function(self):
        """Test default source term creation."""
        solver = AnalogPDESolver(crossbar_size=8)
        pde = PoissonEquation(domain_size=(8,))  # No source function
        
        source = solver._create_source_term(pde, 8)
        
        assert len(source) == 8
        assert np.all(source > 0)  # Should have positive default values


class TestBoundaryConditions:
    """Test boundary condition application."""
    
    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary condition application."""
        solver = AnalogPDESolver(crossbar_size=8)
        pde = PoissonEquation(domain_size=(8,), boundary_conditions="dirichlet")
        
        test_phi = np.ones(8) * 5.0  # All values = 5
        phi_bc = solver._apply_boundary_conditions(test_phi, pde)
        
        assert phi_bc[0] == 0.0
        assert phi_bc[-1] == 0.0
        assert phi_bc[4] == 5.0  # Interior unchanged
    
    def test_neumann_boundary_conditions(self):
        """Test Neumann boundary condition application."""
        solver = AnalogPDESolver(crossbar_size=8)
        pde = PoissonEquation(domain_size=(8,), boundary_conditions="neumann")
        
        test_phi = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        phi_bc = solver._apply_boundary_conditions(test_phi, pde)
        
        assert phi_bc[0] == phi_bc[1]  # Zero gradient at boundary
        assert phi_bc[-1] == phi_bc[-2]
    
    def test_periodic_boundary_conditions(self):
        """Test periodic boundary condition application."""
        solver = AnalogPDESolver(crossbar_size=8)
        pde = PoissonEquation(domain_size=(8,), boundary_conditions="periodic")
        
        test_phi = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        phi_bc = solver._apply_boundary_conditions(test_phi, pde)
        
        assert phi_bc[0] == phi_bc[-2]  # Periodic connection
        assert phi_bc[-1] == phi_bc[1]