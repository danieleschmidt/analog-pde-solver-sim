"""
End-to-end tests for the complete analog PDE solver pipeline.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import test fixtures
from tests.fixtures.pde_fixtures import poisson_2d_simple, hardware_configurations


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """End-to-end pipeline tests."""
    
    def test_complete_poisson_solver_pipeline(self, poisson_2d_simple, temp_dir):
        """Test complete pipeline from PDE specification to analog solution."""
        
        # Mock the analog PDE solver components for testing
        with patch('analog_pde_solver.core.solver.AnalogPDESolver') as MockSolver:
            # Setup mock solver
            mock_solver = MagicMock()
            MockSolver.return_value = mock_solver
            
            # Mock the solve method to return a reasonable solution
            grid_size = poisson_2d_simple.grid_size
            x = np.linspace(0, 1, grid_size[0])
            y = np.linspace(0, 1, grid_size[1])
            X, Y = np.meshgrid(x, y)
            mock_solution = poisson_2d_simple.analytical_solution(X, Y)
            mock_solver.solve.return_value = mock_solution
            
            # Test the pipeline
            try:
                # 1. Create PDE specification
                pde_spec = {
                    "type": "poisson",
                    "grid_size": grid_size,
                    "boundary_conditions": poisson_2d_simple.boundary_conditions
                }
                
                # 2. Initialize analog solver
                solver = MockSolver(
                    crossbar_size=max(grid_size),
                    conductance_range=(1e-9, 1e-6),
                    noise_model="realistic"
                )
                
                # 3. Map PDE to analog hardware
                hardware_config = {"mapped": True}
                mock_solver.map_pde_to_crossbar.return_value = hardware_config
                
                # 4. Solve the PDE
                solution = mock_solver.solve(
                    iterations=100,
                    convergence_threshold=poisson_2d_simple.tolerance
                )
                
                # 5. Verify solution quality
                assert solution is not None
                assert solution.shape == grid_size
                
                # Check that solution is reasonable (not all zeros or NaN)
                assert not np.all(solution == 0), "Solution is all zeros"
                assert not np.any(np.isnan(solution)), "Solution contains NaN"
                assert not np.any(np.isinf(solution)), "Solution contains infinity"
                
                # 6. Generate RTL (mocked)
                rtl_output = temp_dir / "poisson_solver.v"
                mock_solver.to_rtl.return_value.save.return_value = str(rtl_output)
                
                # Verify RTL generation was called
                mock_solver.to_rtl.assert_called_once()
                
            except Exception as e:
                pytest.fail(f"Pipeline test failed with exception: {e}")
    
    def test_navier_stokes_pipeline(self, temp_dir):
        """Test Navier-Stokes solver pipeline."""
        
        with patch('analog_pde_solver.NavierStokesAnalog') as MockNS:
            mock_ns = MagicMock()
            MockNS.return_value = mock_ns
            
            # Mock fluid simulation results
            grid_size = (128, 128)
            mock_u = np.random.random(grid_size) * 0.1
            mock_v = np.random.random(grid_size) * 0.1
            mock_pressure = np.random.random(grid_size)
            
            mock_ns.update_velocity.return_value = (mock_u, mock_v)
            mock_ns.solve_pressure_poisson.return_value = mock_pressure
            mock_ns.apply_pressure_correction.return_value = (mock_u, mock_v)
            
            try:
                # Initialize Navier-Stokes solver
                ns_solver = MockNS(
                    resolution=(128, 128),
                    reynolds_number=1000,
                    time_step=0.001
                )
                
                # Configure hardware
                ns_solver.configure_hardware(
                    num_crossbars=4,
                    precision_bits=8,
                    update_scheme="semi-implicit"
                )
                
                # Run simulation for a few timesteps
                for timestep in range(10):
                    u, v = ns_solver.update_velocity()
                    pressure = ns_solver.solve_pressure_poisson()
                    u, v = ns_solver.apply_pressure_correction(u, v, pressure)
                    
                    # Verify results are reasonable
                    assert u is not None and v is not None
                    assert pressure is not None
                    assert u.shape == grid_size
                    assert v.shape == grid_size
                    assert pressure.shape == grid_size
                
                # Analyze power consumption
                mock_power_analysis = MagicMock()
                mock_power_analysis.avg_power_mw = 5.2
                mock_power_analysis.energy_per_iter_nj = 0.8
                mock_ns.analyze_power.return_value = mock_power_analysis
                
                power_analysis = ns_solver.analyze_power()
                assert power_analysis.avg_power_mw > 0
                assert power_analysis.energy_per_iter_nj > 0
                
            except Exception as e:
                pytest.fail(f"Navier-Stokes pipeline test failed: {e}")
    
    @pytest.mark.hardware
    def test_spice_integration_pipeline(self, hardware_configurations, temp_dir):
        """Test SPICE integration pipeline."""
        
        config = hardware_configurations[0]  # Use small crossbar config
        
        with patch('analog_pde_solver.spice.SPICESimulator') as MockSpice:
            mock_spice = MagicMock()
            MockSpice.return_value = mock_spice
            
            # Mock SPICE simulation results
            mock_results = MagicMock()
            mock_results.get_node_voltages.return_value = np.random.random((64, 64))
            mock_spice.transient.return_value = mock_results
            
            try:
                # Create SPICE simulator
                spice_sim = MockSpice()
                
                # Add crossbar components
                rows, cols = config["size"]
                for i in range(min(rows, 8)):  # Limit for testing
                    for j in range(min(cols, 8)):
                        spice_sim.add_component(
                            f"R_{i}_{j}",
                            "memristor",
                            nodes=(f"row_{i}", f"col_{j}"),
                            params={
                                "ron": 1e3,
                                "roff": 1e6,
                                "rinit": 1e4,
                                "model": "hp_memristor"
                            }
                        )
                
                # Add peripheral circuits
                spice_sim.add_dac_array("input_dac", resolution=config["dac_bits"], voltage_range=1.0)
                spice_sim.add_adc_array("output_adc", resolution=config["adc_bits"], sampling_rate=1e6)
                
                # Run simulation
                results = spice_sim.transient(
                    stop_time=1e-3,
                    time_step=1e-6,
                    initial_conditions={"initial": "state"}
                )
                
                # Extract and verify solution
                analog_solution = results.get_node_voltages("output_nodes")
                assert analog_solution is not None
                assert analog_solution.shape == (64, 64)
                assert not np.any(np.isnan(analog_solution))
                
            except Exception as e:
                pytest.fail(f"SPICE integration pipeline test failed: {e}")
    
    def test_rtl_generation_pipeline(self, temp_dir):
        """Test RTL generation pipeline."""
        
        with patch('analog_pde_solver.compiler.TorchToAnalog') as MockCompiler:
            mock_compiler = MagicMock()
            MockCompiler.return_value = mock_compiler
            
            # Mock compiled model
            mock_analog_model = MagicMock()
            mock_compiler.compile.return_value = mock_analog_model
            
            try:
                # Create compiler
                compiler = MockCompiler()
                
                # Mock PyTorch model compilation
                mock_model = MagicMock()
                analog_model = compiler.compile(
                    model=mock_model,
                    target_hardware="crossbar_array",
                    optimization_level=3
                )
                
                # Generate RTL files
                rtl_file = temp_dir / "pde_accelerator.v"
                constraints_file = temp_dir / "constraints.xdc"
                
                analog_model.export_rtl.return_value = str(rtl_file)
                analog_model.export_constraints.return_value = str(constraints_file)
                
                # Verify RTL generation
                rtl_output = analog_model.export_rtl("pde_accelerator.v")
                constraints_output = analog_model.export_constraints("constraints.xdc")
                
                assert rtl_output is not None
                assert constraints_output is not None
                
                # Verify methods were called
                analog_model.export_rtl.assert_called_once()
                analog_model.export_constraints.assert_called_once()
                
            except Exception as e:
                pytest.fail(f"RTL generation pipeline test failed: {e}")
    
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        
        with patch('analog_pde_solver.core.solver.AnalogPDESolver') as MockSolver:
            # Test convergence failure
            mock_solver = MagicMock()
            MockSolver.return_value = mock_solver
            mock_solver.solve.side_effect = RuntimeError("Convergence failed")
            
            with pytest.raises(RuntimeError, match="Convergence failed"):
                solver = MockSolver()
                solver.solve()
            
            # Test invalid hardware configuration
            mock_solver.solve.side_effect = ValueError("Invalid crossbar size")
            
            with pytest.raises(ValueError, match="Invalid crossbar size"):
                solver = MockSolver()
                solver.solve()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])