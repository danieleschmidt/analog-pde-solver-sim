"""Comprehensive tests for advanced analog algorithms.

Tests all novel algorithms implemented in the research module:
- Analog Physics-Informed Crossbar Networks (APICNs)
- Temporal Crossbar Cascading (TCC)
- Heterogeneous Precision Analog Computing (HPAC)
- Analog Multi-Physics Coupling (AMPC)
- Neuromorphic PDE Acceleration (NPA)
- Integrated Solver Framework
- Validation and Benchmarking
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analog_pde_solver.core.solver import AnalogPDESolver
from analog_pde_solver.core.crossbar import AnalogCrossbarArray
from analog_pde_solver.core.equations import PoissonEquation, HeatEquation

from analog_pde_solver.research.advanced_analog_algorithms import (
    AnalogPhysicsInformedCrossbar,
    TemporalCrossbarCascade,
    HeterogeneousPrecisionAnalogComputing,
    LocalErrorEstimator,
    PrecisionLevel,
    CrossbarRegion,
    PhysicsConstraint
)

from analog_pde_solver.research.multi_physics_coupling import (
    AnalogMultiPhysicsCoupler,
    PhysicsDomain,
    PhysicsDomainConfig,
    CouplingInterface
)

from analog_pde_solver.research.neuromorphic_acceleration import (
    NeuromorphicPDESolver,
    NeuromorphicSpikeEncoder,
    NeuromorphicSpikeDecoder,
    SparseEventBuffer,
    SpikeEncoding,
    SpikeEvent,
    NeuronState
)

from analog_pde_solver.research.ml_acceleration import (
    NeuralNetworkSurrogate,
    PhysicsInformedSurrogate,
    MLAcceleratedPDESolver,
    TrainingData
)

from analog_pde_solver.research.integrated_solver_framework import (
    AdvancedSolverFramework,
    AlgorithmType,
    ProblemCharacteristics,
    AlgorithmSelector,
    PerformanceTracker
)

from analog_pde_solver.research.validation_benchmark_suite import (
    ValidationBenchmarkSuite,
    BenchmarkType,
    BenchmarkProblem,
    BenchmarkResult
)


class TestAnalogPhysicsInformedCrossbar:
    """Test Analog Physics-Informed Crossbar Networks."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_crossbar = AnalogCrossbarArray(32, 32)
        
        # Create physics constraints
        self.physics_constraints = [
            PhysicsConstraint(
                constraint_type='conservation',
                constraint_function=lambda x: np.sum(x),
                weight=1.0,
                conductance_mapping=None,
                active_regions=[(0, 16, 0, 16)],
                conservation_required=True,
                bidirectional=False
            ),
            PhysicsConstraint(
                constraint_type='symmetry',
                constraint_function=None,
                weight=0.5,
                conductance_mapping=None,
                active_regions=[(16, 32, 16, 32)],
                conservation_required=False,
                bidirectional=True
            )
        ]
        
        self.apicn = AnalogPhysicsInformedCrossbar(
            self.base_crossbar,
            self.physics_constraints,
            residual_threshold=1e-6,
            adaptation_rate=0.01
        )
    
    def test_initialization(self):
        """Test APICN initialization."""
        assert self.apicn.base_crossbar is self.base_crossbar
        assert len(self.apicn.physics_constraints) == 2
        assert len(self.apicn.physics_conductances) == 2
        assert 'constraint_0' in self.apicn.physics_conductances
        assert 'constraint_1' in self.apicn.physics_conductances
    
    def test_physics_aware_programming(self):
        """Test physics-aware conductance programming."""
        target_matrix = np.random.random((32, 32))
        
        metrics = self.apicn.program_physics_aware_conductances(target_matrix)
        
        # Check that metrics are returned
        assert 'total_violation' in metrics
        assert 'programming_time' in metrics
        assert 'constraints_satisfied' in metrics
        
        # Check that conductances were modified
        assert not np.array_equal(
            self.apicn.base_crossbar.conductance_matrix,
            target_matrix
        )
    
    def test_solve_with_physics_constraints(self):
        """Test solving with physics constraint enforcement."""
        input_vector = np.random.random(32)
        
        solution, metrics = self.apicn.solve_with_physics_constraints(
            input_vector,
            max_physics_iterations=10
        )
        
        # Check solution properties
        assert solution.shape == (32,)
        assert np.isfinite(solution).all()
        
        # Check metrics
        assert 'iterations' in metrics
        assert 'final_violation' in metrics
        assert 'solve_time' in metrics
        assert metrics['iterations'] <= 10
    
    def test_constraint_residual_computation(self):
        """Test physics constraint residual computation."""
        conductances = np.random.random((32, 32))
        
        for constraint in self.physics_constraints:
            residual = self.apicn._compute_constraint_residual(
                constraint, conductances
            )
            
            assert isinstance(residual, (int, float))
            assert np.isfinite(residual)
    
    def test_conductance_adjustment(self):
        """Test conductance adjustment computation."""
        conductances = np.random.random((32, 32))
        
        for constraint in self.physics_constraints:
            residual = 0.1  # Test residual
            
            adjustment = self.apicn._compute_conductance_adjustment(
                constraint, residual, conductances
            )
            
            assert adjustment.shape == conductances.shape
            assert np.isfinite(adjustment).all()


class TestTemporalCrossbarCascade:
    """Test Temporal Crossbar Cascading."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cascade_crossbars = [
            AnalogCrossbarArray(16, 16) for _ in range(4)
        ]
        
        self.tcc = TemporalCrossbarCascade(
            self.cascade_crossbars,
            time_step=0.01,
            temporal_scheme='forward_euler',
            cascade_depth=4
        )
    
    def test_initialization(self):
        """Test TCC initialization."""
        assert len(self.tcc.base_crossbars) == 4
        assert self.tcc.time_step == 0.01
        assert self.tcc.temporal_scheme == 'forward_euler'
        assert self.tcc.cascade_depth == 4
    
    def test_temporal_pipeline_setup(self):
        """Test temporal pipeline setup."""
        spatial_operator = np.random.random((16, 16))
        boundary_conditions = {'dirichlet': True, 'dirichlet_value': 0.0}
        
        # Should not raise an exception
        self.tcc.setup_temporal_pipeline(spatial_operator, boundary_conditions)
        
        # Check that crossbars were programmed
        for crossbar in self.tcc.base_crossbars:
            assert not np.array_equal(
                crossbar.conductance_matrix,
                np.zeros((16, 16))
            )
    
    def test_sequential_pipeline_evolution(self):
        """Test sequential pipeline evolution."""
        spatial_operator = np.eye(16)  # Simple identity operator
        boundary_conditions = {'dirichlet': True}
        
        self.tcc.setup_temporal_pipeline(spatial_operator, boundary_conditions)
        
        initial_state = np.random.random(16)
        
        final_state, metrics = self.tcc.evolve_temporal_pipeline(
            initial_state,
            num_time_steps=10,
            parallel_execution=False
        )
        
        # Check results
        assert final_state.shape == initial_state.shape
        assert np.isfinite(final_state).all()
        
        # Check metrics
        assert 'time_steps' in metrics
        assert 'speedup_vs_sequential' in metrics
        assert 'evolution_time' in metrics
        assert metrics['time_steps'] == 10
    
    def test_parallel_pipeline_evolution(self):
        """Test parallel pipeline evolution."""
        spatial_operator = np.eye(16)
        boundary_conditions = {'dirichlet': True}
        
        self.tcc.setup_temporal_pipeline(spatial_operator, boundary_conditions)
        
        initial_state = np.random.random(16)
        
        final_state, metrics = self.tcc.evolve_temporal_pipeline(
            initial_state,
            num_time_steps=5,  # Fewer steps for parallel test
            parallel_execution=True
        )
        
        assert final_state.shape == initial_state.shape
        assert np.isfinite(final_state).all()
        assert 'pipeline_efficiency' in metrics


class TestHeterogeneousPrecisionAnalogComputing:
    """Test Heterogeneous Precision Analog Computing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_crossbar = AnalogCrossbarArray(64, 64)
        self.hpac = HeterogeneousPrecisionAnalogComputing(
            self.base_crossbar,
            precision_levels=list(PrecisionLevel),
            adaptation_threshold=1e-4,
            energy_weight=0.5
        )
    
    def test_initialization(self):
        """Test HPAC initialization."""
        assert self.hpac.base_crossbar is self.base_crossbar
        assert len(self.hpac.precision_levels) == 4
        assert len(self.hpac.crossbar_regions) > 0
    
    def test_region_initialization(self):
        """Test crossbar region initialization."""
        regions = self.hpac._initialize_regions()
        
        assert len(regions) == 16  # 4x4 grid
        
        for region in regions:
            assert isinstance(region, CrossbarRegion)
            assert region.precision in PrecisionLevel
            assert 0 <= region.start_row < region.end_row <= 64
            assert 0 <= region.start_col < region.end_col <= 64
    
    def test_precision_adaptation(self):
        """Test precision adaptation."""
        current_solution = np.random.random(64)
        
        metrics = self.hpac.adapt_precision_allocation(
            current_solution,
            target_accuracy=1e-6
        )
        
        # Check metrics
        assert 'regions_adapted' in metrics
        assert 'energy_reduction' in metrics
        assert 'adaptation_time' in metrics
        assert 'precision_distribution' in metrics
    
    def test_heterogeneous_vmm(self):
        """Test heterogeneous precision VMM computation."""
        input_vector = np.random.random(64)
        
        output, metrics = self.hpac.compute_heterogeneous_vmm(input_vector)
        
        # Check output
        assert output.shape == (64,)
        assert np.isfinite(output).all()
        
        # Check metrics
        assert 'total_energy' in metrics
        assert 'estimated_accuracy' in metrics
        assert 'computation_time' in metrics
        assert 'precision_utilization' in metrics
    
    def test_region_energy_computation(self):
        """Test region energy computation."""
        region = self.hpac.crossbar_regions[0]
        
        for precision in PrecisionLevel:
            energy = self.hpac._compute_region_energy(region, precision)
            assert energy >= 0
            assert np.isfinite(energy)
    
    def test_optimal_precision_selection(self):
        """Test optimal precision selection."""
        region_error = 1e-5
        target_accuracy = 1e-6
        current_precision = PrecisionLevel.MEDIUM
        
        new_precision = self.hpac._select_optimal_precision(
            region_error,
            target_accuracy,
            current_precision
        )
        
        assert new_precision in PrecisionLevel


class TestLocalErrorEstimator:
    """Test Local Error Estimator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.estimator = LocalErrorEstimator(32, 32)
    
    def test_initialization(self):
        """Test error estimator initialization."""
        assert self.estimator.rows == 32
        assert self.estimator.cols == 32
        assert self.estimator.previous_solution is None
    
    def test_error_estimation(self):
        """Test local error estimation."""
        solution = np.random.random(32 * 32)
        
        errors = self.estimator.estimate_local_errors(solution)
        
        assert errors.shape == (32, 32)
        assert np.isfinite(errors).all()
        assert (errors >= 0).all()
    
    def test_adaptation_indicators(self):
        """Test adaptation indicators."""
        errors = np.random.random((32, 32))
        
        indicators = self.estimator.get_adaptation_indicators(errors)
        
        assert 'high_error_regions' in indicators
        assert 'low_error_regions' in indicators
        assert 'stable_regions' in indicators
        assert 'error_magnitude' in indicators
        
        for key, indicator in indicators.items():
            if key != 'error_magnitude':
                assert indicator.dtype == bool
            assert indicator.shape == errors.shape


class TestNeuromorphicAcceleration:
    """Test Neuromorphic PDE Acceleration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_solver = AnalogPDESolver(crossbar_size=32)
        
        self.encoder = NeuromorphicSpikeEncoder(
            encoding_scheme=SpikeEncoding.RATE,
            time_window=1.0,
            max_spike_rate=1000.0
        )
        
        self.decoder = NeuromorphicSpikeDecoder(
            decoding_scheme=SpikeEncoding.RATE,
            time_window=1.0,
            output_size=32
        )
        
        self.npa_solver = NeuromorphicPDESolver(
            self.base_solver,
            spike_encoder=self.encoder,
            spike_decoder=self.decoder,
            sparsity_threshold=0.9
        )
    
    def test_spike_encoder_initialization(self):
        """Test spike encoder initialization."""
        assert self.encoder.encoding_scheme == SpikeEncoding.RATE
        assert self.encoder.time_window == 1.0
        assert self.encoder.max_spike_rate == 1000.0
    
    def test_spike_encoding(self):
        """Test spike encoding."""
        data = np.random.random(32)
        current_time = 0.0
        
        events = self.encoder.encode_data(data, current_time)
        
        assert isinstance(events, list)
        
        for event in events:
            assert isinstance(event, SpikeEvent)
            assert event.timestamp >= current_time
            assert event.timestamp <= current_time + self.encoder.time_window
            assert 0 <= event.neuron_id < len(data)
    
    def test_spike_decoding(self):
        """Test spike decoding."""
        # Create sample spike events
        events = [
            SpikeEvent(timestamp=0.1, neuron_id=0, spike_value=0.5),
            SpikeEvent(timestamp=0.3, neuron_id=1, spike_value=0.7),
            SpikeEvent(timestamp=0.8, neuron_id=0, spike_value=0.3)
        ]
        
        current_time = 1.0
        decoded_data = self.decoder.decode_events(events, current_time)
        
        assert decoded_data.shape == (32,)
        assert np.isfinite(decoded_data).all()
    
    def test_sparse_event_buffer(self):
        """Test sparse event buffer."""
        buffer = SparseEventBuffer(capacity=100)
        
        # Add events
        for i in range(10):
            event = SpikeEvent(
                timestamp=i * 0.1,
                neuron_id=i % 5,
                spike_value=0.5
            )
            buffer.add_event(event)
        
        assert len(buffer.events) == 10
        assert len(buffer.active_neurons) == 5
        
        # Test windowed retrieval
        events_in_window = buffer.get_events_in_window(0.0, 0.5, None)
        assert len(events_in_window) == 6  # Events at 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
        
        # Test sparsity statistics
        stats = buffer.get_sparsity_statistics()
        assert 'sparsity' in stats
        assert 'active_fraction' in stats
        assert 'event_rate' in stats
    
    def test_neuromorphic_solver_initialization(self):
        """Test neuromorphic solver initialization."""
        assert self.npa_solver.base_solver is self.base_solver
        assert self.npa_solver.sparsity_threshold == 0.9
        assert len(self.npa_solver.neuron_states) > 0
    
    def test_sparsity_computation(self):
        """Test sparsity level computation."""
        # Dense solution
        dense_solution = np.ones(32)
        sparsity = self.npa_solver._compute_sparsity(dense_solution)
        assert sparsity < 0.5
        
        # Sparse solution
        sparse_solution = np.zeros(32)
        sparse_solution[0] = 1.0  # Only one non-zero element
        sparsity = self.npa_solver._compute_sparsity(sparse_solution)
        assert sparsity > 0.9


class TestMultiPhysicsCoupling:
    """Test Analog Multi-Physics Coupling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.primary_crossbar = AnalogCrossbarArray(64, 64)
        
        # Create physics domains
        self.thermal_domain = PhysicsDomainConfig(
            domain_type=PhysicsDomain.THERMAL,
            governing_equations=['heat_equation'],
            crossbar_allocation=(0, 32, 0, 32),
            boundary_conditions={'dirichlet': True},
            material_properties={'conductivity': 1.0},
            source_terms=None,
            time_scale=1.0,
            length_scale=1.0
        )
        
        self.fluid_domain = PhysicsDomainConfig(
            domain_type=PhysicsDomain.FLUID,
            governing_equations=['navier_stokes'],
            crossbar_allocation=(32, 64, 0, 32),
            boundary_conditions={'dirichlet': True},
            material_properties={'viscosity': 1e-3},
            source_terms=None,
            time_scale=0.1,
            length_scale=1.0
        )
        
        # Create coupling interface
        self.coupling_interface = CouplingInterface(
            source_domain=PhysicsDomain.THERMAL,
            target_domain=PhysicsDomain.FLUID,
            coupling_type='source_term',
            coupling_strength=0.1,
            coupling_function=lambda x: 0.1 * x,
            interface_regions=[(16, 48, 16, 48)],
            conservation_required=True,
            bidirectional=False
        )
        
        self.ampc = AnalogMultiPhysicsCoupler(
            self.primary_crossbar,
            [self.thermal_domain, self.fluid_domain],
            [self.coupling_interface]
        )
    
    def test_initialization(self):
        """Test AMPC initialization."""
        assert len(self.ampc.physics_domains) == 2
        assert PhysicsDomain.THERMAL in self.ampc.physics_domains
        assert PhysicsDomain.FLUID in self.ampc.physics_domains
        assert len(self.ampc.coupling_interfaces) == 1
    
    def test_domain_crossbar_initialization(self):
        """Test domain crossbar initialization."""
        assert len(self.ampc.domain_crossbars) == 2
        
        thermal_info = self.ampc.domain_crossbars[PhysicsDomain.THERMAL]
        assert thermal_info['allocation'] == (0, 32, 0, 32)
        assert thermal_info['crossbar'].rows == 32
        assert thermal_info['crossbar'].cols == 32
    
    def test_coupling_matrix_initialization(self):
        """Test coupling matrix initialization."""
        interface_key = "thermal_to_fluid"
        assert interface_key in self.ampc.coupling_matrices
        
        coupling_matrix = self.ampc.coupling_matrices[interface_key]
        assert coupling_matrix.shape[0] > 0
        assert coupling_matrix.shape[1] > 0
    
    def test_solve_coupled_system(self):
        """Test coupled system solving."""
        initial_conditions = {
            PhysicsDomain.THERMAL: np.random.random(32),
            PhysicsDomain.FLUID: np.random.random(32)
        }
        
        final_states, metrics = self.ampc.solve_coupled_system(
            initial_conditions,
            time_span=(0.0, 0.1),
            num_time_steps=5,
            coupling_iterations=3
        )
        
        # Check results
        assert len(final_states) == 2
        assert PhysicsDomain.THERMAL in final_states
        assert PhysicsDomain.FLUID in final_states
        
        # Check metrics
        assert 'time_steps' in metrics
        assert 'coupling_iterations_per_step' in metrics
        assert 'conservation_errors' in metrics
        assert 'coupling_residuals' in metrics
    
    def test_coupling_efficiency_analysis(self):
        """Test coupling efficiency analysis."""
        analysis = self.ampc.analyze_coupling_efficiency()
        
        assert 'interface_utilization' in analysis
        assert 'domain_efficiency' in analysis
        assert 'conservation_quality' in analysis
        assert 'coupling_overhead_estimate' in analysis


class TestMLAcceleration:
    """Test Machine Learning acceleration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_solver = AnalogPDESolver(crossbar_size=32)
        
        self.nn_surrogate = NeuralNetworkSurrogate(
            input_dim=32,
            hidden_layers=[16, 8],
            activation='relu'
        )
        
        self.ml_solver = MLAcceleratedPDESolver(
            self.base_solver,
            surrogate_type='neural_network',
            training_threshold=5
        )
    
    def test_neural_network_initialization(self):
        """Test neural network surrogate initialization."""
        assert self.nn_surrogate.input_dim == 32
        assert self.nn_surrogate.hidden_layers == [16, 8]
        assert len(self.nn_surrogate.layers) == 3  # Input->16->8->Output
    
    def test_neural_network_forward_pass(self):
        """Test neural network forward pass."""
        input_data = np.random.random(32)
        
        output = self.nn_surrogate.forward(input_data)
        
        assert output.shape == (32,)
        assert np.isfinite(output).all()
    
    def test_neural_network_training(self):
        """Test neural network training."""
        # Create training data
        inputs = np.random.random((10, 32))
        outputs = np.random.random((10, 32))
        
        training_data = TrainingData(
            inputs=inputs,
            outputs=outputs,
            metadata={'test': True}
        )
        
        history = self.nn_surrogate.train(training_data, epochs=10)
        
        assert 'loss' in history
        assert len(history['loss']) == 10
    
    def test_ml_accelerated_solver_initialization(self):
        """Test ML accelerated solver initialization."""
        assert self.ml_solver.base_solver is self.base_solver
        assert self.ml_solver.training_threshold == 5
        assert self.ml_solver.solve_count == 0


class TestIntegratedSolverFramework:
    """Test Integrated Solver Framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = AdvancedSolverFramework(
            base_crossbar_size=32,
            performance_mode='balanced'
        )
        
        self.algorithm_selector = AlgorithmSelector('balanced')
        self.performance_tracker = PerformanceTracker()
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework.base_crossbar_size == 32
        assert self.framework.performance_mode == 'balanced'
        assert len(self.framework.algorithms) > 0
        assert AlgorithmType.BASE_ANALOG in self.framework.algorithms
    
    def test_problem_analysis(self):
        """Test problem characteristics analysis."""
        pde = PoissonEquation(
            domain_size=(32, 32),
            boundary_conditions='dirichlet'
        )
        
        characteristics = self.framework._analyze_problem(
            pde,
            convergence_threshold=1e-6,
            time_dependent=False
        )
        
        assert isinstance(characteristics, ProblemCharacteristics)
        assert characteristics.problem_size == (32, 32)
        assert characteristics.accuracy_requirement == 1e-6
        assert not characteristics.time_dependent
    
    def test_algorithm_selector(self):
        """Test algorithm selector."""
        characteristics = ProblemCharacteristics(
            problem_size=(64, 64),
            sparsity_level=0.95,  # High sparsity
            time_dependent=False,
            multi_physics=False,
            conservation_required=False,
            accuracy_requirement=1e-6,
            energy_budget=0.01,
            real_time_requirement=False,
            physics_constraints=[],
            boundary_complexity='simple'
        )
        
        available_algorithms = {
            AlgorithmType.BASE_ANALOG: Mock(),
            AlgorithmType.NEUROMORPHIC: Mock()
        }
        
        recommendation = self.algorithm_selector.recommend_algorithm(
            characteristics,
            available_algorithms,
            {}
        )
        
        assert recommendation.algorithm_type in available_algorithms
        assert 0 <= recommendation.confidence <= 1
    
    def test_performance_tracker(self):
        """Test performance tracker."""
        # Start tracking
        self.performance_tracker.start_tracking('test_operation')
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        # End tracking
        duration = self.performance_tracker.end_tracking(
            'test_operation',
            additional_metrics={'operations': 100}
        )
        
        assert duration > 0
        
        # Get summary
        summary = self.performance_tracker.get_performance_summary()
        assert 'test_operation' in summary
        assert summary['test_operation']['count'] == 1


class TestValidationBenchmarkSuite:
    """Test Validation and Benchmark Suite."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = AdvancedSolverFramework(base_crossbar_size=32)
        self.benchmark_suite = ValidationBenchmarkSuite(
            self.framework,
            output_directory="test_benchmark_results",
            num_statistical_runs=2  # Reduced for testing
        )
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        assert self.benchmark_suite.framework is self.framework
        assert len(self.benchmark_suite.benchmark_problems) > 0
        assert self.benchmark_suite.num_statistical_runs == 2
    
    def test_benchmark_problem_creation(self):
        """Test benchmark problem creation."""
        problems = self.benchmark_suite._create_benchmark_problems()
        
        assert len(problems) > 0
        
        for problem in problems:
            assert isinstance(problem, BenchmarkProblem)
            assert problem.name
            assert problem.pde_constructor
            assert problem.difficulty_level in ['easy', 'medium', 'hard', 'extreme']
    
    def test_performance_benchmark(self):
        """Test performance benchmark."""
        algorithms_to_test = [AlgorithmType.BASE_ANALOG]
        
        results = self.benchmark_suite.run_performance_benchmark(algorithms_to_test)
        
        assert 'solve_times' in results
        assert 'throughput' in results
        assert 'memory_usage' in results
    
    def test_statistical_analysis(self):
        """Test statistical analysis methods."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        effect_size = self.benchmark_suite._compute_effect_size(group1, group2)
        
        assert isinstance(effect_size, float)
        assert effect_size >= 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.framework = AdvancedSolverFramework(
            base_crossbar_size=32,
            performance_mode='balanced'
        )
    
    def test_end_to_end_solve(self):
        """Test complete end-to-end solving pipeline."""
        # Create a simple PDE
        pde = PoissonEquation(
            domain_size=(32, 32),
            boundary_conditions='dirichlet',
            source_function=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        )
        
        # Define problem characteristics
        characteristics = ProblemCharacteristics(
            problem_size=(32, 32),
            sparsity_level=0.2,
            time_dependent=False,
            multi_physics=False,
            conservation_required=False,
            accuracy_requirement=1e-6,
            energy_budget=None,
            real_time_requirement=False,
            physics_constraints=[],
            boundary_complexity='simple'
        )
        
        # Solve
        solution, solve_info = self.framework.solve_pde(
            pde,
            problem_characteristics=characteristics
        )
        
        # Verify results
        assert solution.shape == (32*32,) or solution.shape == (32, 32)
        assert np.isfinite(solution).all()
        assert 'selected_algorithm' in solve_info
        assert 'total_framework_time' in solve_info
    
    def test_algorithm_comparison(self):
        """Test comparison between different algorithms."""
        pde = PoissonEquation(
            domain_size=(16, 16),
            boundary_conditions='dirichlet'
        )
        
        characteristics = ProblemCharacteristics(
            problem_size=(16, 16),
            sparsity_level=0.5,
            time_dependent=False,
            multi_physics=False,
            conservation_required=False,
            accuracy_requirement=1e-6,
            energy_budget=None,
            real_time_requirement=False,
            physics_constraints=[],
            boundary_complexity='simple'
        )
        
        # Test multiple algorithms
        algorithms_to_test = [AlgorithmType.BASE_ANALOG]
        
        results = {}
        for algorithm in algorithms_to_test:
            try:
                solution, solve_info = self.framework.solve_pde(
                    pde,
                    problem_characteristics=characteristics,
                    algorithm_preference=algorithm
                )
                
                results[algorithm] = {
                    'solution': solution,
                    'solve_info': solve_info,
                    'success': True
                }
                
            except Exception as e:
                results[algorithm] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Verify at least one algorithm worked
        successful_algorithms = [alg for alg, result in results.items() if result['success']]
        assert len(successful_algorithms) > 0


# Pytest fixtures and configuration
@pytest.fixture
def sample_crossbar():
    """Sample crossbar for testing."""
    return AnalogCrossbarArray(16, 16)


@pytest.fixture  
def sample_pde():
    """Sample PDE for testing."""
    return PoissonEquation(
        domain_size=(16, 16),
        boundary_conditions='dirichlet',
        source_function=lambda x, y: np.ones_like(x)
    )


@pytest.fixture
def sample_solver():
    """Sample solver for testing."""
    return AnalogPDESolver(crossbar_size=16)


# Test configuration
def pytest_configure(config):
    """Configure pytest."""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])