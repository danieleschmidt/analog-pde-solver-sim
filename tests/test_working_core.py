#!/usr/bin/env python3
"""Unit tests for analog PDE solver core functionality."""

import sys
import os
import unittest
import numpy as np

# Add the root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analog_pde_solver import (
    AnalogPDESolver, 
    PoissonEquation, 
    HeatEquation, 
    WaveEquation,
    VerilogGenerator,
    RTLConfig
)
from analog_pde_solver.core.crossbar import AnalogCrossbarArray


class TestAnalogCrossbarArray(unittest.TestCase):
    """Test analog crossbar array functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.crossbar = AnalogCrossbarArray(32, 32)
        
    def test_initialization(self):
        """Test crossbar initialization."""
        self.assertEqual(self.crossbar.rows, 32)
        self.assertEqual(self.crossbar.cols, 32)
        self.assertEqual(self.crossbar.cell_type, "1T1R")
        
    def test_conductance_programming(self):
        """Test conductance programming."""
        test_matrix = np.random.random((32, 32)) - 0.5  # Mixed positive/negative
        self.crossbar.program_conductances(test_matrix)
        
        # Check that conductances are in valid range
        self.assertTrue(np.all(self.crossbar.g_positive >= 0))
        self.assertTrue(np.all(self.crossbar.g_negative >= 0))
        self.assertTrue(np.all(self.crossbar.g_positive <= 1e-6))
        self.assertTrue(np.all(self.crossbar.g_negative <= 1e-6))
        
    def test_vector_matrix_multiplication(self):
        """Test analog vector-matrix multiplication."""
        # Simple identity matrix test
        identity = np.eye(32)
        self.crossbar.program_conductances(identity)
        
        input_vector = np.ones(32)
        output = self.crossbar.compute_vmm(input_vector)
        
        # Should be approximately the input (with noise)
        self.assertEqual(len(output), 32)
        self.assertTrue(np.all(np.abs(output) < 1.0))  # Reasonable bounds
        

class TestPoissonEquation(unittest.TestCase):
    """Test Poisson equation implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pde = PoissonEquation(
            domain_size=(32,),
            boundary_conditions="dirichlet"
        )
        
    def test_initialization(self):
        """Test Poisson equation initialization."""
        self.assertEqual(self.pde.domain_size, (32,))
        self.assertEqual(self.pde.boundary_conditions, "dirichlet")
        
    def test_digital_solver(self):
        """Test digital reference solver."""
        solution = self.pde.solve_digital()
        self.assertEqual(len(solution), 32)
        
        # Boundary conditions should be satisfied
        self.assertAlmostEqual(solution[0], 0.0, places=6)
        self.assertAlmostEqual(solution[-1], 0.0, places=6)
        

class TestHeatEquation(unittest.TestCase):
    """Test heat equation implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heat_eq = HeatEquation(
            domain_size=(32,),
            thermal_diffusivity=0.1,
            time_step=0.001
        )
        
    def test_initialization(self):
        """Test heat equation initialization."""
        self.assertEqual(self.heat_eq.domain_size, (32,))
        self.assertEqual(self.heat_eq.thermal_diffusivity, 0.1)
        self.assertEqual(self.heat_eq.time_step, 0.001)
        
    def test_field_initialization(self):
        """Test temperature field initialization."""
        def initial_temp(x):
            return np.sin(np.pi * x)
            
        field = self.heat_eq.initialize_field(initial_condition=initial_temp)
        self.assertEqual(len(field), 32)
        
        # Check boundary conditions
        self.assertAlmostEqual(field[0], 0.0, places=6)
        self.assertAlmostEqual(field[-1], 0.0, places=6)
        
    def test_time_stepping(self):
        """Test time stepping functionality."""
        self.heat_eq.initialize_field()
        initial_field = self.heat_eq.temperature_field.copy()
        
        # Take a time step
        new_field = self.heat_eq.step()
        
        # Field should evolve (unless it's zero everywhere)
        self.assertEqual(len(new_field), 32)
        

class TestWaveEquation(unittest.TestCase):
    """Test wave equation implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.wave_eq = WaveEquation(
            domain_size=(32,),
            wave_speed=1.0,
            time_step=0.01
        )
        
    def test_initialization(self):
        """Test wave equation initialization."""
        self.assertEqual(self.wave_eq.domain_size, (32,))
        self.assertEqual(self.wave_eq.wave_speed, 1.0)
        self.assertEqual(self.wave_eq.time_step, 0.01)
        
    def test_field_initialization(self):
        """Test wave field initialization."""
        def initial_displacement(x):
            return np.exp(-((x - 0.5) / 0.1)**2)
            
        u_current, u_previous = self.wave_eq.initialize_field(
            initial_displacement=initial_displacement
        )
        
        self.assertEqual(len(u_current), 32)
        self.assertEqual(len(u_previous), 32)
        
        # Check that initial displacement is set
        self.assertTrue(np.max(u_current) > 0.1)


class TestAnalogPDESolver(unittest.TestCase):
    """Test main analog PDE solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = AnalogPDESolver(
            crossbar_size=32,
            conductance_range=(1e-9, 1e-6),
            noise_model="realistic"
        )
        
    def test_initialization(self):
        """Test solver initialization."""
        self.assertEqual(self.solver.crossbar_size, 32)
        self.assertEqual(self.solver.conductance_range, (1e-9, 1e-6))
        self.assertEqual(self.solver.noise_model, "realistic")
        
    def test_laplacian_matrix_generation(self):
        """Test Laplacian matrix generation."""
        laplacian = self.solver._create_laplacian_matrix(32)
        
        self.assertEqual(laplacian.shape, (32, 32))
        
        # Check diagonal elements are -2
        self.assertTrue(np.all(np.diag(laplacian) == -2.0))
        
        # Check off-diagonals are 1
        for i in range(31):
            self.assertEqual(laplacian[i, i+1], 1.0)
            self.assertEqual(laplacian[i+1, i], 1.0)
            
    def test_pde_mapping(self):
        """Test PDE to crossbar mapping."""
        pde = PoissonEquation((32,))
        config = self.solver.map_pde_to_crossbar(pde)
        
        self.assertEqual(config["matrix_size"], 32)
        self.assertEqual(config["conductance_range"], (1e-9, 1e-6))
        self.assertTrue(config["programming_success"])
        
    def test_solving(self):
        """Test PDE solving functionality."""
        pde = PoissonEquation((32,))
        solution = self.solver.solve(pde, iterations=10)
        
        self.assertEqual(len(solution), 32)
        self.assertTrue(np.isfinite(solution).all())


class TestVerilogGenerator(unittest.TestCase):
    """Test Verilog RTL generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RTLConfig(
            dac_bits=8,
            adc_bits=10,
            clock_frequency_mhz=100.0
        )
        self.generator = VerilogGenerator(self.config)
        
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.config.dac_bits, 8)
        self.assertEqual(self.generator.config.adc_bits, 10)
        self.assertEqual(self.generator.config.clock_frequency_mhz, 100.0)
        
    def test_top_module_generation(self):
        """Test top-level module generation."""
        verilog_code = self.generator.generate_top_module(
            crossbar_size=16,
            num_crossbars=2,
            pde_type="poisson"
        )
        
        self.assertIsInstance(verilog_code, str)
        self.assertTrue(len(verilog_code) > 1000)
        self.assertIn("module analog_pde_solver_poisson", verilog_code)
        self.assertIn("parameter GRID_SIZE = 16", verilog_code)
        self.assertIn("NUM_CROSSBARS = 2", verilog_code)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)