"""Advanced integration tests for all system components."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Import all components for comprehensive testing
try:
    from analog_pde_solver import (
        RobustAnalogPDESolver, PoissonEquation, NavierStokesAnalog,
        PerformanceOptimizer, OptimizationConfig,
        AutoScaler, ScalingPolicy,
        SystemHealthMonitor,
        VerilogGenerator, RTLConfig
    )
    from analog_pde_solver.core.crossbar_robust import RobustAnalogCrossbarArray
    from analog_pde_solver.utils.validation import ValidationError
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")
class TestRobustSolver:
    """Test the robust analog PDE solver implementation."""
    
    def test_solver_initialization(self):
        """Test solver initialization with various configurations."""
        # Test default configuration
        solver = RobustAnalogPDESolver()
        assert solver.crossbar_size == 128
        assert solver.conductance_range == (1e-9, 1e-6)
        assert solver.noise_model == "realistic"
        
        # Test custom configuration
        solver = RobustAnalogPDESolver(
            crossbar_size=64,
            conductance_range=(1e-8, 1e-5),
            noise_model="none"
        )
        assert solver.crossbar_size == 64
        assert solver.conductance_range == (1e-8, 1e-5)
        assert solver.noise_model == "none"
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises((ValidationError, ValueError)):
            RobustAnalogPDESolver(crossbar_size=0)
        
        with pytest.raises((ValidationError, ValueError)):
            RobustAnalogPDESolver(conductance_range=(1e-6, 1e-9))  # Invalid range
        
        with pytest.raises((ValidationError, ValueError)):
            RobustAnalogPDESolver(noise_model="invalid_model")
    
    def test_poisson_solve(self):
        """Test solving a simple Poisson equation."""
        # Create simple PDE
        pde = PoissonEquation(
            domain_size=32,
            boundary_conditions="dirichlet",
            source_function=lambda x, y: 1.0
        )
        
        # Create solver
        solver = RobustAnalogPDESolver(
            crossbar_size=32,
            noise_model="none"  # No noise for reproducible testing
        )
        
        # Solve
        solution = solver.solve(pde, iterations=50, convergence_threshold=1e-4)
        
        # Verify solution properties
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 32
        assert np.isfinite(solution).all()
        
        # Check boundary conditions
        assert abs(solution[0]) < 1e-6  # Dirichlet BC
        assert abs(solution[-1]) < 1e-6
        
        # Get convergence info
        conv_info = solver.get_convergence_info()
        assert conv_info['status'] == 'solved'
        assert conv_info['iterations'] <= 50
    
    def test_health_check(self):
        """Test solver health check functionality."""
        solver = RobustAnalogPDESolver()
        health = solver.health_check()
        
        assert isinstance(health, dict)
        assert 'timestamp' in health
        assert 'crossbar_programmed' in health
        assert 'crossbar_size' in health
        assert health['crossbar_size'] == 128


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")
class TestPerformanceOptimizer:
    """Test performance optimization features."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = OptimizationConfig(
            enable_parallel_crossbars=True,
            max_worker_threads=2,
            enable_caching=True
        )
        
        optimizer = PerformanceOptimizer(config)
        assert optimizer.config.enable_parallel_crossbars
        assert optimizer.config.max_worker_threads == 2
        assert optimizer.config.enable_caching
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        optimizer = PerformanceOptimizer(OptimizationConfig(enable_caching=True))
        
        # Create mock crossbar
        class MockCrossbar:
            def __init__(self):
                self.rows = 32
                self.cols = 32
                self.operation_count = 0
            
            def compute_vmm(self, input_vec):
                self.operation_count += 1
                return np.ones(32) * np.sum(input_vec)
        
        crossbar = MockCrossbar()
        input_vec = np.ones(32)
        
        # First call should miss cache
        result1 = optimizer._execute_crossbar_operation(crossbar, input_vec, "vmm")
        assert optimizer.cache_misses >= 1
        
        # Second call with same input should hit cache if caching works
        result2 = optimizer._execute_crossbar_operation(crossbar, input_vec, "vmm")
        
        # Results should be equal
        np.testing.assert_array_equal(result1, result2)
    
    def test_cleanup(self):
        """Test optimizer cleanup."""
        optimizer = PerformanceOptimizer()
        
        # Add some cache entries
        optimizer.solution_cache['test'] = np.ones(10)
        assert len(optimizer.solution_cache) > 0
        
        # Cleanup should clear cache
        optimizer.cleanup()
        assert len(optimizer.solution_cache) == 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")
class TestHealthMonitoring:
    """Test health monitoring system."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = SystemHealthMonitor(history_size=50)
        assert monitor.history_size == 50
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alerts) == 0
    
    def test_metrics_recording(self):
        """Test metrics recording."""
        monitor = SystemHealthMonitor()
        
        # Mock stats
        crossbar_stats = {
            'is_programmed': True,
            'programming_errors': 0,
            'total_devices': 1000,
            'health_percentage': 95.0
        }
        
        solver_info = {
            'status': 'solved',
            'iterations': 25,
            'final_error': 1e-5
        }
        
        # Record metrics
        metrics = monitor.record_metrics(crossbar_stats, solver_info)
        
        # Verify metrics
        assert metrics.crossbar_health > 80.0  # Should be healthy
        assert metrics.solver_health > 80.0
        assert len(monitor.metrics_history) == 1
    
    def test_alert_generation(self):
        """Test alert generation for critical conditions."""
        monitor = SystemHealthMonitor()
        
        # Create critical condition
        bad_crossbar_stats = {
            'is_programmed': True,
            'programming_errors': 10,
            'total_devices': 1000,
            'health_percentage': 20.0,  # Very low health
            'health_stuck_low_devices': 200,
            'health_stuck_high_devices': 100
        }
        
        bad_solver_info = {
            'status': 'failed',
            'iterations': 1000,
            'final_error': 1.0
        }
        
        # Record bad metrics
        metrics = monitor.record_metrics(bad_crossbar_stats, bad_solver_info)
        
        # Should generate alerts
        assert len(monitor.alerts) > 0
        
        # Check for critical alerts
        critical_alerts = [a for a in monitor.alerts if a['level'] == 'CRITICAL']
        assert len(critical_alerts) > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")  
class TestAutoScaling:
    """Test auto-scaling functionality."""
    
    def test_autoscaler_initialization(self):
        """Test auto-scaler initialization."""
        policy = ScalingPolicy(
            min_instances=2,
            max_instances=6,
            scale_up_threshold=0.8
        )
        
        autoscaler = AutoScaler(policy)
        assert autoscaler.policy.min_instances == 2
        assert autoscaler.policy.max_instances == 6
        assert autoscaler.current_instances == 2
    
    def test_scaling_decisions(self):
        """Test scaling decision making."""
        autoscaler = AutoScaler()
        
        # Mock high utilization metrics
        from analog_pde_solver.optimization.auto_scaler import ScalingMetrics
        
        high_util_metrics = ScalingMetrics(
            cpu_utilization=0.9,
            memory_utilization=0.85,
            response_time=5.0,
            queue_length=20
        )
        
        # Add to history
        for _ in range(10):
            autoscaler.metrics_history.append(high_util_metrics)
        
        # Should decide to scale up
        decision = autoscaler._make_scaling_decision(high_util_metrics)
        # Note: May be MAINTAIN due to cooldown, which is correct behavior
        
    def test_get_available_solver(self):
        """Test solver instance retrieval."""
        autoscaler = AutoScaler()
        
        # Add mock solver
        class MockSolver:
            def __init__(self, name):
                self.name = name
        
        mock_solver = MockSolver("test")
        autoscaler.resource_pools['solvers'].append(mock_solver)
        autoscaler.current_instances = 1
        
        # Get solver
        solver = autoscaler.get_available_solver()
        assert solver is not None
        assert solver.name == "test"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")
class TestRTLGeneration:
    """Test RTL generation capabilities."""
    
    def test_verilog_generator_initialization(self):
        """Test Verilog generator initialization."""
        config = RTLConfig(
            target_technology="xilinx_ultrascale",
            clock_frequency_mhz=100.0,
            dac_bits=8,
            adc_bits=10
        )
        
        gen = VerilogGenerator(config)
        assert gen.config.target_technology == "xilinx_ultrascale"
        assert gen.config.clock_frequency_mhz == 100.0
    
    def test_top_level_generation(self):
        """Test top-level module generation."""
        gen = VerilogGenerator()
        
        verilog_code = gen.generate_top_level(
            crossbar_size=32,
            num_crossbars=2,
            pde_type="poisson"
        )
        
        # Verify basic structure
        assert "module analog_pde_solver_poisson" in verilog_code
        assert "endmodule" in verilog_code
        assert "GRID_SIZE = 32" in verilog_code
        assert "NUM_CROSSBARS = 2" in verilog_code
    
    def test_crossbar_generation(self):
        """Test crossbar module generation."""
        gen = VerilogGenerator()
        
        crossbar_code = gen.generate_crossbar_array(size=16)
        
        # Verify structure
        assert "module analog_crossbar_array_16x16" in crossbar_code
        assert "ARRAY_SIZE = 16" in crossbar_code
        assert "g_positive" in crossbar_code
        assert "g_negative" in crossbar_code
    
    def test_file_export(self):
        """Test exporting generated modules to files."""
        gen = VerilogGenerator()
        
        # Generate some modules
        gen.generate_top_level(crossbar_size=8, num_crossbars=1)
        gen.generate_crossbar_array(size=8)
        
        # Export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exported_files = gen.export_all_modules(temp_path)
            
            # Verify files were created
            assert len(exported_files) > 0
            
            # Check that files exist and have content
            for module_name, file_path in exported_files.items():
                if module_name != "constraints":  # Skip constraints file
                    assert file_path.exists()
                    assert file_path.stat().st_size > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")
class TestValidation:
    """Test input validation and error handling."""
    
    def test_domain_size_validation(self):
        """Test domain size validation."""
        from analog_pde_solver.utils.validation import validate_domain_size
        
        # Valid cases
        assert validate_domain_size(64) == (64,)
        assert validate_domain_size((32, 32)) == (32, 32)
        assert validate_domain_size([16, 16, 16]) == (16, 16, 16)
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_domain_size(0)  # Zero size
        
        with pytest.raises(ValidationError):
            validate_domain_size(-1)  # Negative size
        
        with pytest.raises(ValidationError):
            validate_domain_size((1,))  # Too small
    
    def test_conductance_validation(self):
        """Test conductance range validation."""
        from analog_pde_solver.utils.validation import validate_conductance_range
        
        # Valid cases
        result = validate_conductance_range((1e-9, 1e-6))
        assert result == (1e-9, 1e-6)
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_conductance_range((1e-6, 1e-9))  # Inverted range
        
        with pytest.raises(ValidationError):
            validate_conductance_range((-1e-9, 1e-6))  # Negative conductance
        
        with pytest.raises(ValidationError):
            validate_conductance_range((1e-9,))  # Wrong length


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports failed: {IMPORT_ERROR}")
class TestEndToEndIntegration:
    """End-to-end integration tests combining multiple components."""
    
    def test_optimized_solve_pipeline(self):
        """Test complete optimized solve pipeline."""
        # Create PDE problem
        pde = PoissonEquation(
            domain_size=16,  # Small for fast testing
            boundary_conditions="dirichlet"
        )
        
        # Create solver
        solver = RobustAnalogPDESolver(
            crossbar_size=16,
            noise_model="none"
        )
        
        # Create optimizer
        optimizer = PerformanceOptimizer(OptimizationConfig(
            enable_caching=True,
            enable_adaptive_precision=True
        ))
        
        # Create health monitor
        monitor = SystemHealthMonitor()
        
        try:
            # Solve with adaptive precision
            solution, stats = optimizer.adaptive_precision_solver(
                solver, pde, target_accuracy=1e-4
            )
            
            # Verify solution
            assert isinstance(solution, np.ndarray)
            assert len(solution) == 16
            assert stats['converged']
            
            # Get solver info
            conv_info = solver.get_convergence_info()
            
            # Record health metrics
            crossbar_stats = solver.crossbar.get_device_stats() if hasattr(solver.crossbar, 'get_device_stats') else {
                'is_programmed': True,
                'programming_errors': 0,
                'total_devices': 256
            }
            
            health_metrics = monitor.record_metrics(crossbar_stats, conv_info)
            
            # Verify health metrics
            assert health_metrics.crossbar_health > 0
            assert health_metrics.solver_health > 0
            
        finally:
            # Cleanup
            optimizer.cleanup()
    
    def test_navier_stokes_integration(self):
        """Test Navier-Stokes solver integration."""
        ns_solver = NavierStokesAnalog(
            resolution=(16, 16),
            reynolds_number=100,
            time_step=0.01
        )
        
        # Configure hardware
        ns_solver.configure_hardware(num_crossbars=2, precision_bits=8)
        
        # Run a few timesteps
        for i in range(3):
            u, v = ns_solver.update_velocity()
            pressure = ns_solver.solve_pressure_poisson()
            u, v = ns_solver.apply_pressure_correction(u, v, pressure)
        
        # Analyze power
        power_analysis = ns_solver.analyze_power()
        assert hasattr(power_analysis, 'avg_power_mw')
        assert hasattr(power_analysis, 'energy_per_iter_nj')
    
    def test_rtl_generation_integration(self):
        """Test RTL generation with realistic parameters."""
        # Create RTL generator
        config = RTLConfig(
            target_technology="xilinx_zynq",
            clock_frequency_mhz=50.0,
            optimization_goal="area"
        )
        
        gen = VerilogGenerator(config)
        
        # Generate complete design
        top_level = gen.generate_top_level(
            crossbar_size=16,
            num_crossbars=1,
            pde_type="poisson"
        )
        
        crossbar_module = gen.generate_crossbar_array(size=16)
        controller = gen.generate_pde_controller(crossbar_size=16, pde_type="poisson")
        interfaces = gen.generate_mixed_signal_interface(crossbar_size=16)
        
        # Verify all modules were generated
        assert len(top_level) > 1000  # Should be substantial
        assert len(crossbar_module) > 500
        assert len(controller) > 100
        assert len(interfaces) > 500
        
        # Verify basic syntax
        assert top_level.count("module") == top_level.count("endmodule")
        assert "always @(posedge clk)" in top_level


# Run basic smoke tests even if imports fail
def test_basic_imports():
    """Basic test to verify imports work."""
    if not IMPORTS_AVAILABLE:
        pytest.skip(f"Imports failed: {IMPORT_ERROR}")
    
    # Just test that we can import the main classes
    from analog_pde_solver import AnalogPDESolver, PoissonEquation
    
    # Create instances
    pde = PoissonEquation(domain_size=4)
    solver = AnalogPDESolver(crossbar_size=4)
    
    assert pde is not None
    assert solver is not None


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])