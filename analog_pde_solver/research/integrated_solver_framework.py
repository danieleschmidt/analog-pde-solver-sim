"""Integrated Advanced Solver Framework.

This module provides a unified interface to all advanced algorithms,
enabling seamless integration and automatic algorithm selection based on
problem characteristics.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.solver import AnalogPDESolver
from ..core.crossbar import AnalogCrossbarArray
from ..utils.logger import get_logger, PerformanceLogger
from .validation_layer import validate_algorithm_result, ValidationLevel, ValidationResult

from .ml_acceleration import MLAcceleratedPDESolver
from .advanced_analog_algorithms import (
    AnalogPhysicsInformedCrossbar,
    TemporalCrossbarCascade,
    HeterogeneousPrecisionAnalogComputing,
    PrecisionLevel,
    PhysicsConstraint
)
from .multi_physics_coupling import (
    AnalogMultiPhysicsCoupler,
    PhysicsDomain,
    PhysicsDomainConfig,
    CouplingInterface
)
from .neuromorphic_acceleration import (
    NeuromorphicPDESolver,
    NeuromorphicSpikeEncoder,
    NeuromorphicSpikeDecoder,
    SpikeEncoding
)


class AlgorithmType(Enum):
    """Types of advanced algorithms available."""
    BASE_ANALOG = "base_analog"
    ML_ACCELERATED = "ml_accelerated"
    PHYSICS_INFORMED = "physics_informed"
    TEMPORAL_CASCADE = "temporal_cascade"
    HETEROGENEOUS_PRECISION = "heterogeneous_precision"
    MULTI_PHYSICS = "multi_physics"
    NEUROMORPHIC = "neuromorphic"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class ProblemCharacteristics:
    """Characteristics of PDE problem for algorithm selection."""
    problem_size: Tuple[int, ...]
    sparsity_level: float
    time_dependent: bool
    multi_physics: bool
    conservation_required: bool
    accuracy_requirement: float
    energy_budget: Optional[float]
    real_time_requirement: bool
    physics_constraints: List[str]
    boundary_complexity: str  # 'simple', 'complex', 'time_varying'


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with performance estimates."""
    algorithm_type: AlgorithmType
    confidence: float  # 0-1
    estimated_speedup: float
    estimated_energy_savings: float
    estimated_accuracy: float
    reasoning: str
    configuration: Dict[str, Any]


class AdvancedSolverFramework:
    """Unified framework for advanced analog PDE solving algorithms.
    
    Automatically selects optimal algorithms based on problem characteristics
    and provides seamless integration of all advanced techniques.
    """
    
    def __init__(
        self,
        base_crossbar_size: int = 128,
        enable_ml_acceleration: bool = True,
        enable_neuromorphic: bool = True,
        enable_multi_physics: bool = True,
        performance_mode: str = 'balanced'  # 'speed', 'accuracy', 'energy', 'balanced'
    ):
        """Initialize advanced solver framework.
        
        Args:
            base_crossbar_size: Size of base crossbar array
            enable_ml_acceleration: Enable ML acceleration
            enable_neuromorphic: Enable neuromorphic acceleration
            enable_multi_physics: Enable multi-physics coupling
            performance_mode: Performance optimization mode
        """
        self.logger = get_logger('advanced_framework')
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.base_crossbar_size = base_crossbar_size
        self.performance_mode = performance_mode
        
        # Initialize base components
        self.base_crossbar = AnalogCrossbarArray(base_crossbar_size, base_crossbar_size)
        self.base_solver = AnalogPDESolver(crossbar_size=base_crossbar_size)
        
        # Initialize advanced algorithm components
        self.algorithms = {}
        self._initialize_algorithms(enable_ml_acceleration, enable_neuromorphic, enable_multi_physics)
        
        # Algorithm selection and performance tracking
        self.algorithm_selector = AlgorithmSelector(self.performance_mode)
        self.performance_tracker = PerformanceTracker()
        
        # Problem history for adaptive learning
        self.problem_history = []
        self.performance_history = {}
        
        self.logger.info(f"Initialized Advanced Solver Framework with {len(self.algorithms)} algorithms")
    
    def _initialize_algorithms(
        self,
        enable_ml: bool,
        enable_neuromorphic: bool,
        enable_multi_physics: bool
    ) -> None:
        """Initialize all advanced algorithms."""
        
        # Base analog solver (always available)
        self.algorithms[AlgorithmType.BASE_ANALOG] = self.base_solver
        
        # ML-accelerated solver
        if enable_ml:
            try:
                self.algorithms[AlgorithmType.ML_ACCELERATED] = MLAcceleratedPDESolver(
                    self.base_solver,
                    surrogate_type='neural_network'
                )
                
                # Physics-informed variant
                physics_ml_solver = MLAcceleratedPDESolver(
                    self.base_solver,
                    surrogate_type='physics_informed'
                )
                self.algorithms[AlgorithmType.PHYSICS_INFORMED] = physics_ml_solver
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize ML algorithms: {e}")
        
        # Advanced analog algorithms
        try:
            # Physics-informed crossbar
            physics_constraints = [
                PhysicsConstraint(
                    constraint_type='conservation',
                    constraint_function=lambda x: np.sum(x),
                    weight=1.0,
                    conductance_mapping=None,
                    active_regions=[(0, 32, 0, 32)],  # Example region
                    conservation_required=True,
                    bidirectional=False
                )
            ]
            
            physics_crossbar = AnalogPhysicsInformedCrossbar(
                self.base_crossbar,
                physics_constraints
            )
            self.algorithms[AlgorithmType.PHYSICS_INFORMED] = physics_crossbar
            
            # Temporal cascade
            cascade_crossbars = [
                AnalogCrossbarArray(self.base_crossbar_size, self.base_crossbar_size)
                for _ in range(4)
            ]
            temporal_cascade = TemporalCrossbarCascade(
                cascade_crossbars,
                time_step=0.001
            )
            self.algorithms[AlgorithmType.TEMPORAL_CASCADE] = temporal_cascade
            
            # Heterogeneous precision
            hetero_precision = HeterogeneousPrecisionAnalogComputing(
                self.base_crossbar
            )
            self.algorithms[AlgorithmType.HETEROGENEOUS_PRECISION] = hetero_precision
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize advanced analog algorithms: {e}")
        
        # Neuromorphic acceleration
        if enable_neuromorphic:
            try:
                neuromorphic_solver = NeuromorphicPDESolver(
                    self.base_solver,
                    sparsity_threshold=0.9
                )
                self.algorithms[AlgorithmType.NEUROMORPHIC] = neuromorphic_solver
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize neuromorphic algorithms: {e}")
        
        # Multi-physics coupling
        if enable_multi_physics:
            try:
                # Example multi-physics configuration
                thermal_domain = PhysicsDomainConfig(
                    domain_type=PhysicsDomain.THERMAL,
                    governing_equations=['heat_equation'],
                    crossbar_allocation=(0, 64, 0, 64),
                    boundary_conditions={'dirichlet': True},
                    material_properties={'conductivity': 1.0},
                    source_terms=None,
                    time_scale=1.0,
                    length_scale=1.0
                )
                
                fluid_domain = PhysicsDomainConfig(
                    domain_type=PhysicsDomain.FLUID,
                    governing_equations=['navier_stokes'],
                    crossbar_allocation=(64, 128, 0, 64),
                    boundary_conditions={'dirichlet': True},
                    material_properties={'viscosity': 1e-3},
                    source_terms=None,
                    time_scale=0.1,
                    length_scale=1.0
                )
                
                coupling_interface = CouplingInterface(
                    source_domain=PhysicsDomain.THERMAL,
                    target_domain=PhysicsDomain.FLUID,
                    coupling_type='source_term',
                    coupling_strength=0.1,
                    coupling_function=lambda x: 0.1 * x,  # Simple linear coupling
                    interface_regions=[(32, 96, 32, 96)],
                    conservation_required=True,
                    bidirectional=True
                )
                
                multi_physics_coupler = AnalogMultiPhysicsCoupler(
                    self.base_crossbar,
                    [thermal_domain, fluid_domain],
                    [coupling_interface]
                )
                
                self.algorithms[AlgorithmType.MULTI_PHYSICS] = multi_physics_coupler
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize multi-physics algorithms: {e}")
    
    def solve_pde(
        self,
        pde,
        problem_characteristics: Optional[ProblemCharacteristics] = None,
        algorithm_preference: Optional[AlgorithmType] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve PDE using optimal algorithm selection.
        
        Args:
            pde: PDE problem to solve
            problem_characteristics: Problem characteristics for algorithm selection
            algorithm_preference: Preferred algorithm (overrides automatic selection)
            **kwargs: Additional solver parameters
            
        Returns:
            Tuple of (solution, solve_info)
        """
        self.perf_logger.start_timer('framework_solve')
        
        # Analyze problem if characteristics not provided
        if problem_characteristics is None:
            problem_characteristics = self._analyze_problem(pde, **kwargs)
        
        # Select optimal algorithm
        if algorithm_preference is None:
            recommendation = self.algorithm_selector.recommend_algorithm(
                problem_characteristics,
                self.algorithms,
                self.performance_history
            )
            selected_algorithm = recommendation.algorithm_type
        else:
            selected_algorithm = algorithm_preference
            recommendation = AlgorithmRecommendation(
                algorithm_type=selected_algorithm,
                confidence=1.0,
                estimated_speedup=1.0,
                estimated_energy_savings=0.0,
                estimated_accuracy=1e-6,
                reasoning="User specified",
                configuration={}
            )
        
        self.logger.info(f"Selected algorithm: {selected_algorithm.value} (confidence: {recommendation.confidence:.2f})")
        
        # Execute solve with selected algorithm
        solution, solve_info = self._execute_solve(
            selected_algorithm,
            pde,
            problem_characteristics,
            recommendation.configuration,
            **kwargs
        )
        
        total_time = self.perf_logger.end_timer('framework_solve')
        
        # Update performance tracking
        self._update_performance_history(
            selected_algorithm,
            problem_characteristics,
            solve_info,
            total_time
        )
        
        # Validate algorithm results
        validation_metadata = {
            'tolerance': problem_characteristics.accuracy_requirement,
            'domain_shape': problem_characteristics.problem_size,
            'conservation_type': 'mass' if problem_characteristics.conservation_required else None,
            'boundary_spec': {
                'dirichlet': hasattr(pde, 'boundary_conditions') and 'dirichlet' in str(pde.boundary_conditions).lower(),
                'dirichlet_value': 0.0,  # Default assumption
                'tolerance': problem_characteristics.accuracy_requirement
            } if hasattr(pde, 'boundary_conditions') else {}
        }
        
        # Add convergence history if available
        if 'convergence_history' in solve_info or 'residual_history' in solve_info:
            validation_metadata['convergence_history'] = (
                solve_info.get('convergence_history') or 
                solve_info.get('residual_history') or
                []
            )
        
        # Run validation
        try:
            validation_report = validate_algorithm_result(
                algorithm_name=selected_algorithm.value,
                inputs={'pde': pde, 'characteristics': problem_characteristics},
                outputs={'solution': solution},
                metadata=validation_metadata,
                validation_level=ValidationLevel.STANDARD
            )
            
            # Log validation results
            if validation_report.has_critical_issues:
                self.logger.error(f"Critical validation issues found: {len(validation_report.issues)}")
                for issue in validation_report.issues:
                    if issue.level == ValidationResult.CRITICAL:
                        self.logger.error(f"  - {issue.category}: {issue.message}")
            elif validation_report.has_failures:
                self.logger.warning(f"Validation failures found: {validation_report.metrics['failed_issues']}")
            elif validation_report.issues:
                self.logger.info(f"Validation completed with {len(validation_report.issues)} warnings")
            else:
                self.logger.debug("Validation passed successfully")
            
            # Add validation report to solve info
            solve_info['validation_report'] = {
                'overall_result': validation_report.overall_result.value,
                'issues_count': len(validation_report.issues),
                'critical_issues': validation_report.metrics.get('critical_issues', 0),
                'failed_issues': validation_report.metrics.get('failed_issues', 0),
                'warning_issues': validation_report.metrics.get('warning_issues', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            solve_info['validation_report'] = {
                'overall_result': 'validation_error',
                'error': str(e)
            }
        
        # Enhanced solve info with framework details
        solve_info.update({
            'framework_version': '2.0.0',
            'selected_algorithm': selected_algorithm.value,
            'algorithm_recommendation': {
                'confidence': recommendation.confidence,
                'reasoning': recommendation.reasoning,
                'estimated_speedup': recommendation.estimated_speedup,
                'estimated_energy_savings': recommendation.estimated_energy_savings
            },
            'problem_characteristics': problem_characteristics,
            'total_framework_time': total_time
        })
        
        return solution, solve_info
    
    def _analyze_problem(self, pde, **kwargs) -> ProblemCharacteristics:
        """Analyze PDE problem to extract characteristics."""
        
        # Extract problem size
        if hasattr(pde, 'domain_size'):
            if isinstance(pde.domain_size, tuple):
                problem_size = pde.domain_size
            else:
                problem_size = (pde.domain_size,)
        else:
            problem_size = (self.base_crossbar_size,)
        
        # Analyze sparsity (if solution or matrix available)
        sparsity_level = 0.0
        if hasattr(pde, 'get_matrix'):
            try:
                matrix = pde.get_matrix()
                sparsity_level = 1.0 - (np.count_nonzero(matrix) / matrix.size)
            except:
                pass
        
        # Check for time dependence
        time_dependent = (
            hasattr(pde, 'time_dependent') and pde.time_dependent or
            'time' in kwargs or
            'dt' in kwargs or
            'num_time_steps' in kwargs
        )
        
        # Check for multi-physics
        multi_physics = (
            len(getattr(pde, 'coupled_equations', [])) > 1 or
            hasattr(pde, 'physics_domains')
        )
        
        # Check conservation requirements
        conservation_required = (
            hasattr(pde, 'conservation_laws') or
            'conservation' in str(type(pde)).lower() or
            'navier' in str(type(pde)).lower()  # Navier-Stokes typically requires conservation
        )
        
        # Extract accuracy requirement
        accuracy_requirement = kwargs.get('convergence_threshold', 1e-6)
        
        # Check for real-time requirements
        real_time_requirement = kwargs.get('real_time', False)
        
        # Extract physics constraints
        physics_constraints = []
        if hasattr(pde, 'constraints'):
            physics_constraints = list(pde.constraints.keys())
        
        # Determine boundary complexity
        boundary_complexity = 'simple'  # Default
        if hasattr(pde, 'boundary_conditions'):
            bc = pde.boundary_conditions
            if isinstance(bc, dict) and len(bc) > 2:
                boundary_complexity = 'complex'
            elif callable(bc) or any(callable(v) for v in bc.values() if isinstance(bc, dict)):
                boundary_complexity = 'time_varying'
        
        return ProblemCharacteristics(
            problem_size=problem_size,
            sparsity_level=sparsity_level,
            time_dependent=time_dependent,
            multi_physics=multi_physics,
            conservation_required=conservation_required,
            accuracy_requirement=accuracy_requirement,
            energy_budget=kwargs.get('energy_budget'),
            real_time_requirement=real_time_requirement,
            physics_constraints=physics_constraints,
            boundary_complexity=boundary_complexity
        )
    
    def _execute_solve(
        self,
        algorithm_type: AlgorithmType,
        pde,
        characteristics: ProblemCharacteristics,
        configuration: Dict[str, Any],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute solve with specified algorithm."""
        
        if algorithm_type not in self.algorithms:
            self.logger.warning(f"Algorithm {algorithm_type.value} not available, falling back to base solver")
            algorithm_type = AlgorithmType.BASE_ANALOG
        
        algorithm = self.algorithms[algorithm_type]
        solve_info = {'algorithm_used': algorithm_type.value}
        
        try:
            if algorithm_type == AlgorithmType.BASE_ANALOG:
                # Base analog solver
                solution = algorithm.solve(
                    pde,
                    iterations=kwargs.get('iterations', 100),
                    convergence_threshold=kwargs.get('convergence_threshold', 1e-6)
                )
                
            elif algorithm_type == AlgorithmType.ML_ACCELERATED:
                # ML-accelerated solver
                solution, ml_info = algorithm.solve(
                    pde,
                    iterations=kwargs.get('iterations', 100),
                    convergence_threshold=kwargs.get('convergence_threshold', 1e-6)
                )
                solve_info.update(ml_info)
                
            elif algorithm_type == AlgorithmType.NEUROMORPHIC:
                # Neuromorphic solver
                if characteristics.time_dependent:
                    time_span = kwargs.get('time_span', (0.0, 1.0))
                    num_time_steps = kwargs.get('num_time_steps', 100)
                    initial_solution = kwargs.get('initial_solution', np.random.random(characteristics.problem_size[0]))
                    
                    solution, neuro_info = algorithm.solve_sparse_pde(
                        pde,
                        initial_solution,
                        time_span,
                        num_time_steps
                    )
                    solve_info.update(neuro_info)
                else:
                    # Fall back to base solver for non-time-dependent
                    solution = self.base_solver.solve(pde, iterations=kwargs.get('iterations', 100))
                    
            elif algorithm_type == AlgorithmType.MULTI_PHYSICS:
                # Multi-physics coupling
                if characteristics.multi_physics:
                    initial_conditions = kwargs.get('initial_conditions', {
                        PhysicsDomain.THERMAL: np.random.random(64),
                        PhysicsDomain.FLUID: np.random.random(64)
                    })
                    time_span = kwargs.get('time_span', (0.0, 1.0))
                    num_time_steps = kwargs.get('num_time_steps', 100)
                    
                    solutions, coupling_info = algorithm.solve_coupled_system(
                        initial_conditions,
                        time_span,
                        num_time_steps
                    )
                    
                    # Combine solutions (simplified)
                    solution = np.concatenate([s for s in solutions.values()])
                    solve_info.update(coupling_info)
                else:
                    # Fall back to base solver
                    solution = self.base_solver.solve(pde, iterations=kwargs.get('iterations', 100))
                    
            elif algorithm_type == AlgorithmType.TEMPORAL_CASCADE:
                # Temporal cascade
                if characteristics.time_dependent:
                    # Setup temporal pipeline
                    spatial_operator = np.random.random((characteristics.problem_size[0], characteristics.problem_size[0]))  # Placeholder
                    boundary_conditions = {'dirichlet': True}
                    
                    algorithm.setup_temporal_pipeline(spatial_operator, boundary_conditions)
                    
                    # Evolve solution
                    initial_state = kwargs.get('initial_solution', np.random.random(characteristics.problem_size[0]))
                    num_time_steps = kwargs.get('num_time_steps', 100)
                    
                    solution, temporal_info = algorithm.evolve_temporal_pipeline(
                        initial_state,
                        num_time_steps
                    )
                    solve_info.update(temporal_info)
                else:
                    # Fall back to base solver
                    solution = self.base_solver.solve(pde, iterations=kwargs.get('iterations', 100))
                    
            elif algorithm_type == AlgorithmType.HETEROGENEOUS_PRECISION:
                # Heterogeneous precision
                # Adapt precision based on current solution estimate
                initial_solution = kwargs.get('initial_solution', np.random.random(characteristics.problem_size[0]))
                
                adaptation_metrics = algorithm.adapt_precision_allocation(
                    initial_solution,
                    characteristics.accuracy_requirement
                )
                
                # Compute solution with adapted precision
                solution, computation_metrics = algorithm.compute_heterogeneous_vmm(
                    initial_solution
                )
                
                solve_info.update({
                    'adaptation_metrics': adaptation_metrics,
                    'computation_metrics': computation_metrics
                })
                
            else:
                # Default to base solver
                solution = self.base_solver.solve(
                    pde,
                    iterations=kwargs.get('iterations', 100),
                    convergence_threshold=kwargs.get('convergence_threshold', 1e-6)
                )
            
        except Exception as e:
            self.logger.error(f"Algorithm {algorithm_type.value} failed: {e}")
            # Fall back to base solver
            solution = self.base_solver.solve(
                pde,
                iterations=kwargs.get('iterations', 100),
                convergence_threshold=kwargs.get('convergence_threshold', 1e-6)
            )
            solve_info['fallback_used'] = True
            solve_info['error'] = str(e)
        
        return solution, solve_info
    
    def _update_performance_history(
        self,
        algorithm_type: AlgorithmType,
        characteristics: ProblemCharacteristics,
        solve_info: Dict[str, Any],
        total_time: float
    ) -> None:
        """Update performance history for future algorithm selection."""
        
        if algorithm_type not in self.performance_history:
            self.performance_history[algorithm_type] = []
        
        performance_record = {
            'problem_size': characteristics.problem_size,
            'sparsity_level': characteristics.sparsity_level,
            'time_dependent': characteristics.time_dependent,
            'multi_physics': characteristics.multi_physics,
            'solve_time': total_time,
            'solve_info': solve_info,
            'timestamp': time.time()
        }
        
        self.performance_history[algorithm_type].append(performance_record)
        
        # Keep only recent records (last 1000)
        if len(self.performance_history[algorithm_type]) > 1000:
            self.performance_history[algorithm_type] = self.performance_history[algorithm_type][-1000:]
    
    def get_algorithm_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all algorithms."""
        summary = {}
        
        for algorithm_type, records in self.performance_history.items():
            if records:
                solve_times = [r['solve_time'] for r in records]
                summary[algorithm_type.value] = {
                    'num_uses': len(records),
                    'avg_solve_time': np.mean(solve_times),
                    'min_solve_time': np.min(solve_times),
                    'max_solve_time': np.max(solve_times),
                    'std_solve_time': np.std(solve_times),
                    'recent_uses': len([r for r in records if time.time() - r['timestamp'] < 3600])  # Last hour
                }
        
        return summary
    
    def benchmark_algorithms(
        self,
        test_problems: List[Tuple[Any, Dict[str, Any]]],
        algorithms_to_test: Optional[List[AlgorithmType]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark multiple algorithms on test problems.
        
        Args:
            test_problems: List of (pde, kwargs) tuples
            algorithms_to_test: Specific algorithms to test (default: all available)
            
        Returns:
            Benchmark results
        """
        if algorithms_to_test is None:
            algorithms_to_test = list(self.algorithms.keys())
        
        benchmark_results = {}
        
        self.logger.info(f"Starting benchmark with {len(test_problems)} problems and {len(algorithms_to_test)} algorithms")
        
        for algorithm_type in algorithms_to_test:
            if algorithm_type not in self.algorithms:
                continue
                
            algorithm_results = {
                'solve_times': [],
                'errors': [],
                'successes': 0,
                'failures': 0
            }
            
            for i, (pde, kwargs) in enumerate(test_problems):
                try:
                    self.perf_logger.start_timer(f'benchmark_{algorithm_type.value}_{i}')
                    
                    solution, solve_info = self._execute_solve(
                        algorithm_type,
                        pde,
                        self._analyze_problem(pde, **kwargs),
                        {},
                        **kwargs
                    )
                    
                    solve_time = self.perf_logger.end_timer(f'benchmark_{algorithm_type.value}_{i}')
                    
                    algorithm_results['solve_times'].append(solve_time)
                    algorithm_results['successes'] += 1
                    
                    # Compute error if reference solution available
                    if 'reference_solution' in kwargs:
                        error = np.linalg.norm(solution - kwargs['reference_solution'])
                        algorithm_results['errors'].append(error)
                    
                except Exception as e:
                    self.logger.warning(f"Benchmark failed for {algorithm_type.value} on problem {i}: {e}")
                    algorithm_results['failures'] += 1
            
            # Compute summary statistics
            if algorithm_results['solve_times']:
                algorithm_results.update({
                    'avg_solve_time': np.mean(algorithm_results['solve_times']),
                    'min_solve_time': np.min(algorithm_results['solve_times']),
                    'max_solve_time': np.max(algorithm_results['solve_times']),
                    'success_rate': algorithm_results['successes'] / len(test_problems)
                })
                
                if algorithm_results['errors']:
                    algorithm_results.update({
                        'avg_error': np.mean(algorithm_results['errors']),
                        'max_error': np.max(algorithm_results['errors'])
                    })
            
            benchmark_results[algorithm_type.value] = algorithm_results
        
        self.logger.info("Benchmark completed")
        return benchmark_results


class AlgorithmSelector:
    """Intelligent algorithm selection based on problem characteristics."""
    
    def __init__(self, performance_mode: str = 'balanced'):
        """Initialize algorithm selector.
        
        Args:
            performance_mode: Optimization mode ('speed', 'accuracy', 'energy', 'balanced')
        """
        self.logger = get_logger('algorithm_selector')
        self.performance_mode = performance_mode
        
        # Performance mode weights (speed, accuracy, energy)
        self.mode_weights = {
            'speed': (0.8, 0.1, 0.1),
            'accuracy': (0.1, 0.8, 0.1),
            'energy': (0.1, 0.1, 0.8),
            'balanced': (0.33, 0.33, 0.34)
        }
    
    def recommend_algorithm(
        self,
        characteristics: ProblemCharacteristics,
        available_algorithms: Dict[AlgorithmType, Any],
        performance_history: Dict[AlgorithmType, List[Dict[str, Any]]]
    ) -> AlgorithmRecommendation:
        """Recommend optimal algorithm based on problem characteristics.
        
        Args:
            characteristics: Problem characteristics
            available_algorithms: Available algorithm instances
            performance_history: Historical performance data
            
        Returns:
            Algorithm recommendation
        """
        
        # Score each available algorithm
        algorithm_scores = {}
        
        for algorithm_type in available_algorithms.keys():
            score = self._score_algorithm(algorithm_type, characteristics, performance_history)
            algorithm_scores[algorithm_type] = score
        
        # Select highest scoring algorithm
        if algorithm_scores:
            best_algorithm = max(algorithm_scores.keys(), key=lambda alg: algorithm_scores[alg]['total_score'])
            best_score = algorithm_scores[best_algorithm]
            
            return AlgorithmRecommendation(
                algorithm_type=best_algorithm,
                confidence=best_score['confidence'],
                estimated_speedup=best_score['estimated_speedup'],
                estimated_energy_savings=best_score['estimated_energy_savings'],
                estimated_accuracy=best_score['estimated_accuracy'],
                reasoning=best_score['reasoning'],
                configuration=best_score['configuration']
            )
        else:
            # Fallback to base algorithm
            return AlgorithmRecommendation(
                algorithm_type=AlgorithmType.BASE_ANALOG,
                confidence=0.5,
                estimated_speedup=1.0,
                estimated_energy_savings=0.0,
                estimated_accuracy=characteristics.accuracy_requirement,
                reasoning="No algorithms available, using base solver",
                configuration={}
            )
    
    def _score_algorithm(
        self,
        algorithm_type: AlgorithmType,
        characteristics: ProblemCharacteristics,
        performance_history: Dict[AlgorithmType, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Score algorithm suitability for given problem characteristics."""
        
        # Base scores
        speed_score = 0.5
        accuracy_score = 0.5
        energy_score = 0.5
        confidence = 0.5
        estimated_speedup = 1.0
        estimated_energy_savings = 0.0
        estimated_accuracy = characteristics.accuracy_requirement
        reasoning_parts = []
        
        # Algorithm-specific scoring
        if algorithm_type == AlgorithmType.NEUROMORPHIC:
            # Neuromorphic excels for sparse problems
            if characteristics.sparsity_level > 0.9:
                speed_score = 0.95
                energy_score = 0.98
                confidence = 0.9
                estimated_speedup = 1000.0 * characteristics.sparsity_level
                estimated_energy_savings = 0.999 * characteristics.sparsity_level
                reasoning_parts.append("High sparsity favors neuromorphic")
            elif characteristics.sparsity_level > 0.5:
                speed_score = 0.7
                energy_score = 0.8
                confidence = 0.7
                estimated_speedup = 10.0 * characteristics.sparsity_level
                estimated_energy_savings = 0.5 * characteristics.sparsity_level
                reasoning_parts.append("Moderate sparsity benefits from neuromorphic")
            else:
                speed_score = 0.2
                energy_score = 0.3
                confidence = 0.3
                reasoning_parts.append("Low sparsity not suitable for neuromorphic")
                
        elif algorithm_type == AlgorithmType.TEMPORAL_CASCADE:
            # Temporal cascade excels for time-dependent problems
            if characteristics.time_dependent:
                speed_score = 0.9
                confidence = 0.85
                estimated_speedup = 100.0
                reasoning_parts.append("Time-dependent problem ideal for temporal cascade")
                
                # Better for larger problems
                problem_size = np.prod(characteristics.problem_size)
                if problem_size > 1000:
                    speed_score = 0.95
                    confidence = 0.9
                    reasoning_parts.append("Large time-dependent problem")
            else:
                speed_score = 0.3
                confidence = 0.2
                reasoning_parts.append("Not time-dependent, temporal cascade not beneficial")
                
        elif algorithm_type == AlgorithmType.MULTI_PHYSICS:
            # Multi-physics coupling for coupled problems
            if characteristics.multi_physics:
                speed_score = 0.8
                accuracy_score = 0.9
                confidence = 0.8
                estimated_speedup = 10.0
                reasoning_parts.append("Multi-physics problem benefits from direct coupling")
                
                if characteristics.conservation_required:
                    accuracy_score = 0.95
                    confidence = 0.85
                    reasoning_parts.append("Conservation requirements well-handled")
            else:
                speed_score = 0.1
                confidence = 0.1
                reasoning_parts.append("Single-physics problem doesn't need multi-physics coupling")
                
        elif algorithm_type == AlgorithmType.ML_ACCELERATED:
            # ML acceleration good for repeated similar problems
            if algorithm_type in performance_history and len(performance_history[algorithm_type]) > 10:
                speed_score = 0.8
                confidence = 0.7
                estimated_speedup = 50.0
                reasoning_parts.append("ML surrogate trained on similar problems")
            else:
                speed_score = 0.4
                confidence = 0.3
                reasoning_parts.append("ML surrogate not yet trained")
                
        elif algorithm_type == AlgorithmType.HETEROGENEOUS_PRECISION:
            # Heterogeneous precision good for multi-scale problems
            problem_size = np.prod(characteristics.problem_size)
            if problem_size > 10000:  # Large problems benefit more
                energy_score = 0.8
                speed_score = 0.7
                confidence = 0.7
                estimated_energy_savings = 0.5
                reasoning_parts.append("Large problem benefits from precision adaptation")
            else:
                energy_score = 0.6
                speed_score = 0.5
                confidence = 0.5
                reasoning_parts.append("Moderate benefits for smaller problems")
                
        elif algorithm_type == AlgorithmType.PHYSICS_INFORMED:
            # Physics-informed good when physics constraints are important
            if len(characteristics.physics_constraints) > 0:
                accuracy_score = 0.9
                confidence = 0.8
                reasoning_parts.append("Physics constraints important")
            
            if characteristics.conservation_required:
                accuracy_score = 0.95
                confidence = 0.85
                reasoning_parts.append("Conservation laws enforced in hardware")
                
        # Apply performance mode weights
        speed_weight, accuracy_weight, energy_weight = self.mode_weights[self.performance_mode]
        total_score = (speed_weight * speed_score + 
                      accuracy_weight * accuracy_score + 
                      energy_weight * energy_score)
        
        # Adjust based on historical performance
        if algorithm_type in performance_history:
            recent_records = [r for r in performance_history[algorithm_type] 
                            if time.time() - r['timestamp'] < 3600]  # Last hour
            
            if recent_records:
                avg_time = np.mean([r['solve_time'] for r in recent_records])
                if avg_time < 1.0:  # Fast recent performance
                    total_score += 0.1
                    confidence += 0.1
                    reasoning_parts.append("Good recent performance")
        
        return {
            'total_score': total_score,
            'speed_score': speed_score,
            'accuracy_score': accuracy_score,
            'energy_score': energy_score,
            'confidence': min(confidence, 1.0),
            'estimated_speedup': estimated_speedup,
            'estimated_energy_savings': estimated_energy_savings,
            'estimated_accuracy': estimated_accuracy,
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else "Default scoring",
            'configuration': {}  # Algorithm-specific configuration
        }


class PerformanceTracker:
    """Track performance metrics across all algorithms."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.logger = get_logger('performance_tracker')
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.thread_lock = threading.Lock()
    
    def start_tracking(self, identifier: str) -> None:
        """Start tracking performance for identifier."""
        with self.thread_lock:
            self.start_times[identifier] = time.time()
    
    def end_tracking(self, identifier: str, additional_metrics: Dict[str, Any] = None) -> float:
        """End tracking and record metrics."""
        end_time = time.time()
        
        with self.thread_lock:
            if identifier in self.start_times:
                duration = end_time - self.start_times[identifier]
                
                metric_record = {
                    'timestamp': end_time,
                    'duration': duration
                }
                
                if additional_metrics:
                    metric_record.update(additional_metrics)
                
                self.metrics[identifier].append(metric_record)
                
                # Clean up
                del self.start_times[identifier]
                
                return duration
            else:
                self.logger.warning(f"No start time found for identifier: {identifier}")
                return 0.0
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all tracked identifiers."""
        summary = {}
        
        with self.thread_lock:
            for identifier, records in self.metrics.items():
                if records:
                    durations = [r['duration'] for r in records]
                    summary[identifier] = {
                        'count': len(records),
                        'avg_duration': np.mean(durations),
                        'min_duration': np.min(durations),
                        'max_duration': np.max(durations),
                        'std_duration': np.std(durations),
                        'total_duration': np.sum(durations)
                    }
        
        return summary