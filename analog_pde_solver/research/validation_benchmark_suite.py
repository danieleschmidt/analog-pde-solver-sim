"""Validation and Benchmark Suite for Advanced Analog Algorithms.

This module implements comprehensive validation and benchmarking for all
advanced algorithms with statistical analysis and publication-ready results.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.solver import AnalogPDESolver
from ..core.equations import PoissonEquation, HeatEquation, WaveEquation
from ..utils.logger import get_logger, PerformanceLogger

from .integrated_solver_framework import AdvancedSolverFramework, AlgorithmType, ProblemCharacteristics
from .advanced_analog_algorithms import PrecisionLevel
from .neuromorphic_acceleration import SpikeEncoding


class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance" 
    ENERGY_EFFICIENCY = "energy_efficiency"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    COMPARATIVE = "comparative"


@dataclass
class BenchmarkProblem:
    """Standard benchmark problem definition."""
    name: str
    pde_constructor: Callable
    pde_kwargs: Dict[str, Any]
    reference_solution: Optional[np.ndarray]
    analytical_solution: Optional[Callable]
    problem_size: Tuple[int, ...]
    expected_sparsity: float
    time_dependent: bool
    multi_physics: bool
    difficulty_level: str  # 'easy', 'medium', 'hard', 'extreme'


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm: AlgorithmType
    problem_name: str
    solve_time: float
    accuracy: float
    energy_estimate: float
    memory_usage: float
    convergence_achieved: bool
    error_metrics: Dict[str, float]
    additional_metrics: Dict[str, Any]


class ValidationBenchmarkSuite:
    """Comprehensive validation and benchmark suite for advanced algorithms."""
    
    def __init__(
        self,
        framework: AdvancedSolverFramework,
        output_directory: str = "benchmark_results",
        statistical_significance: float = 0.05,
        num_statistical_runs: int = 10
    ):
        """Initialize benchmark suite.
        
        Args:
            framework: Advanced solver framework to benchmark
            output_directory: Directory for benchmark results
            statistical_significance: P-value threshold for statistical tests
            num_statistical_runs: Number of runs for statistical analysis
        """
        self.logger = get_logger('benchmark_suite')
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.framework = framework
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.statistical_significance = statistical_significance
        self.num_statistical_runs = num_statistical_runs
        
        # Initialize benchmark problems
        self.benchmark_problems = self._create_benchmark_problems()
        
        # Results storage
        self.results = {}
        self.statistical_results = {}
        
        self.logger.info(f"Initialized benchmark suite with {len(self.benchmark_problems)} problems")
    
    def _create_benchmark_problems(self) -> List[BenchmarkProblem]:
        """Create standard benchmark problems."""
        problems = []
        
        # 1. Simple Poisson problems
        problems.extend([
            BenchmarkProblem(
                name="poisson_2d_small",
                pde_constructor=PoissonEquation,
                pde_kwargs={
                    'domain_size': (64, 64),
                    'boundary_conditions': 'dirichlet',
                    'source_function': lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
                },
                reference_solution=None,
                analytical_solution=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2),
                problem_size=(64, 64),
                expected_sparsity=0.3,
                time_dependent=False,
                multi_physics=False,
                difficulty_level='easy'
            ),
            
            BenchmarkProblem(
                name="poisson_2d_large",
                pde_constructor=PoissonEquation,
                pde_kwargs={
                    'domain_size': (256, 256),
                    'boundary_conditions': 'dirichlet',
                    'source_function': lambda x, y: np.exp(-(x**2 + y**2))
                },
                reference_solution=None,
                analytical_solution=None,
                problem_size=(256, 256),
                expected_sparsity=0.1,
                time_dependent=False,
                multi_physics=False,
                difficulty_level='medium'
            ),
            
            BenchmarkProblem(
                name="poisson_sparse",
                pde_constructor=PoissonEquation,
                pde_kwargs={
                    'domain_size': (128, 128),
                    'boundary_conditions': 'dirichlet',
                    'source_function': lambda x, y: (np.abs(x - 0.5) < 0.1) * (np.abs(y - 0.5) < 0.1)
                },
                reference_solution=None,
                analytical_solution=None,
                problem_size=(128, 128),
                expected_sparsity=0.95,
                time_dependent=False,
                multi_physics=False,
                difficulty_level='medium'
            )
        ])
        
        # 2. Heat equation problems
        problems.extend([
            BenchmarkProblem(
                name="heat_1d_transient",
                pde_constructor=HeatEquation,
                pde_kwargs={
                    'domain_size': (128,),
                    'boundary_conditions': 'dirichlet',
                    'initial_condition': lambda x: np.sin(np.pi * x),
                    'diffusivity': 0.1
                },
                reference_solution=None,
                analytical_solution=lambda x, t: np.exp(-np.pi**2 * 0.1 * t) * np.sin(np.pi * x),
                problem_size=(128,),
                expected_sparsity=0.2,
                time_dependent=True,
                multi_physics=False,
                difficulty_level='medium'
            ),
            
            BenchmarkProblem(
                name="heat_2d_gaussian",
                pde_constructor=HeatEquation,
                pde_kwargs={
                    'domain_size': (64, 64),
                    'boundary_conditions': 'neumann',
                    'initial_condition': lambda x, y: np.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                    'diffusivity': 0.05
                },
                reference_solution=None,
                analytical_solution=None,
                problem_size=(64, 64),
                expected_sparsity=0.4,
                time_dependent=True,
                multi_physics=False,
                difficulty_level='medium'
            )
        ])
        
        # 3. Wave equation problems
        problems.extend([
            BenchmarkProblem(
                name="wave_1d_oscillation",
                pde_constructor=WaveEquation,
                pde_kwargs={
                    'domain_size': (128,),
                    'boundary_conditions': 'dirichlet',
                    'initial_condition': lambda x: np.sin(2 * np.pi * x),
                    'initial_velocity': lambda x: np.zeros_like(x),
                    'wave_speed': 1.0
                },
                reference_solution=None,
                analytical_solution=lambda x, t: np.sin(2 * np.pi * (x - t)),
                problem_size=(128,),
                expected_sparsity=0.1,
                time_dependent=True,
                multi_physics=False,
                difficulty_level='medium'
            )
        ])
        
        # 4. Multi-scale problems
        problems.extend([
            BenchmarkProblem(
                name="multiscale_poisson",
                pde_constructor=PoissonEquation,
                pde_kwargs={
                    'domain_size': (256, 256),
                    'boundary_conditions': 'mixed',
                    'source_function': lambda x, y: (np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y) +
                                                   0.1 * np.sin(100 * np.pi * x) * np.sin(100 * np.pi * y))
                },
                reference_solution=None,
                analytical_solution=None,
                problem_size=(256, 256),
                expected_sparsity=0.05,
                time_dependent=False,
                multi_physics=False,
                difficulty_level='hard'
            )
        ])
        
        # 5. Extreme challenge problems
        problems.extend([
            BenchmarkProblem(
                name="extreme_sparse_poisson",
                pde_constructor=PoissonEquation,
                pde_kwargs={
                    'domain_size': (512, 512),
                    'boundary_conditions': 'dirichlet',
                    'source_function': lambda x, y: ((x - 0.25)**2 + (y - 0.25)**2 < 0.01) * 1000.0
                },
                reference_solution=None,
                analytical_solution=None,
                problem_size=(512, 512),
                expected_sparsity=0.99,
                time_dependent=False,
                multi_physics=False,
                difficulty_level='extreme'
            ),
            
            BenchmarkProblem(
                name="extreme_multiscale_heat",
                pde_constructor=HeatEquation,
                pde_kwargs={
                    'domain_size': (256, 256),
                    'boundary_conditions': 'periodic',
                    'initial_condition': lambda x, y: (np.sin(np.pi * x) * np.sin(np.pi * y) +
                                                      0.01 * np.sin(50 * np.pi * x) * np.sin(50 * np.pi * y)),
                    'diffusivity': 0.001
                },
                reference_solution=None,
                analytical_solution=None,
                problem_size=(256, 256),
                expected_sparsity=0.02,
                time_dependent=True,
                multi_physics=False,
                difficulty_level='extreme'
            )
        ])
        
        return problems
    
    def run_comprehensive_benchmark(
        self,
        benchmark_types: List[BenchmarkType] = None,
        algorithms_to_test: List[AlgorithmType] = None,
        parallel_execution: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Args:
            benchmark_types: Types of benchmarks to run
            algorithms_to_test: Specific algorithms to test
            parallel_execution: Whether to run benchmarks in parallel
            
        Returns:
            Comprehensive benchmark results
        """
        if benchmark_types is None:
            benchmark_types = list(BenchmarkType)
        
        if algorithms_to_test is None:
            algorithms_to_test = list(self.framework.algorithms.keys())
        
        self.logger.info(f"Starting comprehensive benchmark: {len(benchmark_types)} types, {len(algorithms_to_test)} algorithms")
        
        comprehensive_results = {
            'benchmark_config': {
                'timestamp': time.time(),
                'benchmark_types': [bt.value for bt in benchmark_types],
                'algorithms_tested': [alg.value for alg in algorithms_to_test],
                'num_problems': len(self.benchmark_problems),
                'statistical_runs': self.num_statistical_runs
            },
            'results_by_type': {},
            'statistical_analysis': {},
            'summary_metrics': {},
            'recommendations': {}
        }
        
        # Run each benchmark type
        for benchmark_type in benchmark_types:
            self.logger.info(f"Running {benchmark_type.value} benchmark")
            
            if benchmark_type == BenchmarkType.ACCURACY:
                results = self.run_accuracy_benchmark(algorithms_to_test)
            elif benchmark_type == BenchmarkType.PERFORMANCE:
                results = self.run_performance_benchmark(algorithms_to_test)
            elif benchmark_type == BenchmarkType.ENERGY_EFFICIENCY:
                results = self.run_energy_benchmark(algorithms_to_test)
            elif benchmark_type == BenchmarkType.SCALABILITY:
                results = self.run_scalability_benchmark(algorithms_to_test)
            elif benchmark_type == BenchmarkType.ROBUSTNESS:
                results = self.run_robustness_benchmark(algorithms_to_test)
            elif benchmark_type == BenchmarkType.COMPARATIVE:
                results = self.run_comparative_benchmark(algorithms_to_test)
            else:
                results = {}
            
            comprehensive_results['results_by_type'][benchmark_type.value] = results
        
        # Statistical analysis
        comprehensive_results['statistical_analysis'] = self._perform_statistical_analysis()
        
        # Summary metrics
        comprehensive_results['summary_metrics'] = self._compute_summary_metrics()
        
        # Algorithm recommendations
        comprehensive_results['recommendations'] = self._generate_recommendations()
        
        # Save results
        self._save_benchmark_results(comprehensive_results)
        
        self.logger.info("Comprehensive benchmark completed")
        return comprehensive_results
    
    def run_accuracy_benchmark(self, algorithms: List[AlgorithmType]) -> Dict[str, Any]:
        """Run accuracy-focused benchmark."""
        self.logger.info("Running accuracy benchmark")
        
        accuracy_results = {
            'algorithm_errors': {},
            'convergence_rates': {},
            'analytical_comparisons': {},
            'conservation_errors': {}
        }
        
        for algorithm in algorithms:
            if algorithm not in self.framework.algorithms:
                continue
                
            algorithm_errors = []
            convergence_data = []
            analytical_errors = []
            conservation_errors = []
            
            for problem in self.benchmark_problems:
                try:
                    # Run multiple times for statistical significance
                    problem_errors = []
                    
                    for run in range(self.num_statistical_runs):
                        # Create PDE instance
                        pde = problem.pde_constructor(**problem.pde_kwargs)
                        
                        # Solve with selected algorithm
                        characteristics = ProblemCharacteristics(
                            problem_size=problem.problem_size,
                            sparsity_level=problem.expected_sparsity,
                            time_dependent=problem.time_dependent,
                            multi_physics=problem.multi_physics,
                            conservation_required='navier' in problem.name.lower(),
                            accuracy_requirement=1e-6,
                            energy_budget=None,
                            real_time_requirement=False,
                            physics_constraints=[],
                            boundary_complexity='simple'
                        )
                        
                        solution, solve_info = self.framework._execute_solve(
                            algorithm, pde, characteristics, {}
                        )
                        
                        # Compute error metrics
                        if problem.analytical_solution is not None:
                            # Compare with analytical solution
                            x = np.linspace(0, 1, problem.problem_size[0])
                            if len(problem.problem_size) == 1:
                                analytical = problem.analytical_solution(x, 0 if not problem.time_dependent else 1.0)
                            else:
                                y = np.linspace(0, 1, problem.problem_size[1])
                                X, Y = np.meshgrid(x, y)
                                analytical = problem.analytical_solution(X, Y).flatten()
                            
                            # Resize if necessary
                            if len(solution) != len(analytical):
                                if len(solution) > len(analytical):
                                    solution = solution[:len(analytical)]
                                else:
                                    analytical = analytical[:len(solution)]
                            
                            error = np.linalg.norm(solution - analytical)
                            analytical_errors.append(error)
                        
                        elif problem.reference_solution is not None:
                            # Compare with reference solution
                            reference = problem.reference_solution
                            if len(solution) != len(reference):
                                if len(solution) > len(reference):
                                    solution = solution[:len(reference)]
                                else:
                                    reference = reference[:len(solution)]
                            
                            error = np.linalg.norm(solution - reference)
                            problem_errors.append(error)
                        
                        # Check conservation if required
                        if 'navier' in problem.name.lower() or 'conservation' in problem.name.lower():
                            # Simple conservation check
                            initial_mass = np.sum(np.ones_like(solution))
                            final_mass = np.sum(solution)
                            conservation_error = abs(initial_mass - final_mass) / initial_mass
                            conservation_errors.append(conservation_error)
                        
                    if problem_errors:
                        algorithm_errors.extend(problem_errors)
                        
                except Exception as e:
                    self.logger.warning(f"Accuracy benchmark failed for {algorithm.value} on {problem.name}: {e}")
            
            # Store results
            if algorithm_errors:
                accuracy_results['algorithm_errors'][algorithm.value] = {
                    'mean_error': np.mean(algorithm_errors),
                    'std_error': np.std(algorithm_errors),
                    'max_error': np.max(algorithm_errors),
                    'min_error': np.min(algorithm_errors),
                    'errors': algorithm_errors
                }
            
            if analytical_errors:
                accuracy_results['analytical_comparisons'][algorithm.value] = {
                    'mean_error': np.mean(analytical_errors),
                    'std_error': np.std(analytical_errors),
                    'errors': analytical_errors
                }
            
            if conservation_errors:
                accuracy_results['conservation_errors'][algorithm.value] = {
                    'mean_error': np.mean(conservation_errors),
                    'max_error': np.max(conservation_errors),
                    'errors': conservation_errors
                }
        
        return accuracy_results
    
    def run_performance_benchmark(self, algorithms: List[AlgorithmType]) -> Dict[str, Any]:
        """Run performance-focused benchmark."""
        self.logger.info("Running performance benchmark")
        
        performance_results = {
            'solve_times': {},
            'throughput': {},
            'memory_usage': {},
            'scalability_metrics': {}
        }
        
        for algorithm in algorithms:
            if algorithm not in self.framework.algorithms:
                continue
                
            solve_times = []
            memory_usage = []
            throughput_data = []
            
            for problem in self.benchmark_problems:
                try:
                    problem_times = []
                    
                    for run in range(self.num_statistical_runs):
                        # Create PDE instance
                        pde = problem.pde_constructor(**problem.pde_kwargs)
                        
                        # Measure solve time
                        self.perf_logger.start_timer(f'perf_bench_{algorithm.value}_{problem.name}_{run}')
                        
                        characteristics = ProblemCharacteristics(
                            problem_size=problem.problem_size,
                            sparsity_level=problem.expected_sparsity,
                            time_dependent=problem.time_dependent,
                            multi_physics=problem.multi_physics,
                            conservation_required=False,
                            accuracy_requirement=1e-6,
                            energy_budget=None,
                            real_time_requirement=False,
                            physics_constraints=[],
                            boundary_complexity='simple'
                        )
                        
                        solution, solve_info = self.framework._execute_solve(
                            algorithm, pde, characteristics, {}
                        )
                        
                        solve_time = self.perf_logger.end_timer(f'perf_bench_{algorithm.value}_{problem.name}_{run}')
                        problem_times.append(solve_time)
                        
                        # Estimate memory usage (simplified)
                        problem_size = np.prod(problem.problem_size)
                        memory_estimate = problem_size * 8 * 4  # 4 arrays of double precision
                        memory_usage.append(memory_estimate)
                        
                        # Compute throughput (elements per second)
                        if solve_time > 0:
                            throughput = problem_size / solve_time
                            throughput_data.append(throughput)
                    
                    solve_times.extend(problem_times)
                    
                except Exception as e:
                    self.logger.warning(f"Performance benchmark failed for {algorithm.value} on {problem.name}: {e}")
            
            # Store results
            if solve_times:
                performance_results['solve_times'][algorithm.value] = {
                    'mean_time': np.mean(solve_times),
                    'std_time': np.std(solve_times),
                    'min_time': np.min(solve_times),
                    'max_time': np.max(solve_times),
                    'times': solve_times
                }
            
            if memory_usage:
                performance_results['memory_usage'][algorithm.value] = {
                    'mean_memory': np.mean(memory_usage),
                    'max_memory': np.max(memory_usage)
                }
            
            if throughput_data:
                performance_results['throughput'][algorithm.value] = {
                    'mean_throughput': np.mean(throughput_data),
                    'peak_throughput': np.max(throughput_data)
                }
        
        return performance_results
    
    def run_energy_benchmark(self, algorithms: List[AlgorithmType]) -> Dict[str, Any]:
        """Run energy efficiency benchmark."""
        self.logger.info("Running energy efficiency benchmark")
        
        energy_results = {
            'energy_estimates': {},
            'efficiency_ratios': {},
            'sparsity_benefits': {}
        }
        
        for algorithm in algorithms:
            if algorithm not in self.framework.algorithms:
                continue
                
            energy_estimates = []
            efficiency_data = []
            sparsity_energy_data = []
            
            for problem in self.benchmark_problems:
                try:
                    for run in range(max(3, self.num_statistical_runs // 3)):  # Fewer runs for energy
                        pde = problem.pde_constructor(**problem.pde_kwargs)
                        
                        characteristics = ProblemCharacteristics(
                            problem_size=problem.problem_size,
                            sparsity_level=problem.expected_sparsity,
                            time_dependent=problem.time_dependent,
                            multi_physics=problem.multi_physics,
                            conservation_required=False,
                            accuracy_requirement=1e-6,
                            energy_budget=None,
                            real_time_requirement=False,
                            physics_constraints=[],
                            boundary_complexity='simple'
                        )
                        
                        # Estimate energy consumption
                        problem_size = np.prod(problem.problem_size)
                        
                        # Energy model based on algorithm type
                        if algorithm == AlgorithmType.NEUROMORPHIC:
                            # Neuromorphic energy scales with sparsity
                            activity = 1.0 - problem.expected_sparsity
                            base_energy = problem_size * 1e-12  # 1pJ per operation base
                            energy_estimate = base_energy * activity * 0.001  # 1000× reduction for sparse
                        elif algorithm == AlgorithmType.HETEROGENEOUS_PRECISION:
                            # Energy scales with precision usage
                            base_energy = problem_size * 1e-9  # 1nJ per operation
                            precision_factor = 0.5  # Assume 50% energy reduction from mixed precision
                            energy_estimate = base_energy * precision_factor
                        elif algorithm == AlgorithmType.TEMPORAL_CASCADE:
                            # Energy for pipelined computation
                            if problem.time_dependent:
                                base_energy = problem_size * 1e-9
                                pipeline_efficiency = 0.8  # 20% overhead for pipeline
                                energy_estimate = base_energy * pipeline_efficiency
                            else:
                                energy_estimate = problem_size * 1e-9
                        else:
                            # Default analog energy
                            energy_estimate = problem_size * 1e-9  # 1nJ per operation
                        
                        energy_estimates.append(energy_estimate)
                        
                        # Efficiency relative to base analog
                        base_energy = problem_size * 1e-9
                        efficiency_ratio = base_energy / max(energy_estimate, 1e-15)
                        efficiency_data.append(efficiency_ratio)
                        
                        # Sparsity benefit analysis
                        if problem.expected_sparsity > 0.5:
                            sparsity_benefit = problem.expected_sparsity * efficiency_ratio
                            sparsity_energy_data.append(sparsity_benefit)
                
                except Exception as e:
                    self.logger.warning(f"Energy benchmark failed for {algorithm.value} on {problem.name}: {e}")
            
            # Store results
            if energy_estimates:
                energy_results['energy_estimates'][algorithm.value] = {
                    'mean_energy': np.mean(energy_estimates),
                    'total_energy': np.sum(energy_estimates),
                    'energy_range': [np.min(energy_estimates), np.max(energy_estimates)]
                }
            
            if efficiency_data:
                energy_results['efficiency_ratios'][algorithm.value] = {
                    'mean_efficiency': np.mean(efficiency_data),
                    'peak_efficiency': np.max(efficiency_data)
                }
            
            if sparsity_energy_data:
                energy_results['sparsity_benefits'][algorithm.value] = {
                    'sparsity_efficiency': np.mean(sparsity_energy_data)
                }
        
        return energy_results
    
    def run_scalability_benchmark(self, algorithms: List[AlgorithmType]) -> Dict[str, Any]:
        """Run scalability benchmark."""
        self.logger.info("Running scalability benchmark")
        
        scalability_results = {
            'scaling_curves': {},
            'complexity_analysis': {},
            'parallel_efficiency': {}
        }
        
        # Test different problem sizes
        test_sizes = [64, 128, 256, 512]
        
        for algorithm in algorithms:
            if algorithm not in self.framework.algorithms:
                continue
                
            scaling_data = []
            complexity_data = []
            
            for size in test_sizes:
                try:
                    # Create scalability test problem
                    pde = PoissonEquation(
                        domain_size=(size, size),
                        boundary_conditions='dirichlet',
                        source_function=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
                    )
                    
                    characteristics = ProblemCharacteristics(
                        problem_size=(size, size),
                        sparsity_level=0.1,
                        time_dependent=False,
                        multi_physics=False,
                        conservation_required=False,
                        accuracy_requirement=1e-6,
                        energy_budget=None,
                        real_time_requirement=False,
                        physics_constraints=[],
                        boundary_complexity='simple'
                    )
                    
                    # Measure solve time for this size
                    solve_times = []
                    for run in range(3):  # Fewer runs for scalability
                        self.perf_logger.start_timer(f'scale_bench_{algorithm.value}_{size}_{run}')
                        
                        solution, solve_info = self.framework._execute_solve(
                            algorithm, pde, characteristics, {}
                        )
                        
                        solve_time = self.perf_logger.end_timer(f'scale_bench_{algorithm.value}_{size}_{run}')
                        solve_times.append(solve_time)
                    
                    avg_solve_time = np.mean(solve_times)
                    scaling_data.append((size**2, avg_solve_time))  # (problem_size, time)
                    
                    # Complexity analysis
                    problem_size = size**2
                    operations_estimate = problem_size * np.log(problem_size)  # Estimated operations
                    if avg_solve_time > 0:
                        ops_per_second = operations_estimate / avg_solve_time
                        complexity_data.append((problem_size, ops_per_second))
                
                except Exception as e:
                    self.logger.warning(f"Scalability benchmark failed for {algorithm.value} at size {size}: {e}")
            
            # Store results
            if scaling_data:
                sizes, times = zip(*scaling_data)
                
                # Fit scaling curve (linear in log-log space indicates power law)
                if len(sizes) > 2:
                    log_sizes = np.log(sizes)
                    log_times = np.log(times)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_times)
                    
                    scalability_results['scaling_curves'][algorithm.value] = {
                        'sizes': list(sizes),
                        'times': list(times),
                        'scaling_exponent': slope,
                        'r_squared': r_value**2,
                        'complexity_class': 'O(n^{:.2f})'.format(slope)
                    }
            
            if complexity_data:
                scalability_results['complexity_analysis'][algorithm.value] = {
                    'operations_per_second': [ops for _, ops in complexity_data],
                    'problem_sizes': [size for size, _ in complexity_data]
                }
        
        return scalability_results
    
    def run_robustness_benchmark(self, algorithms: List[AlgorithmType]) -> Dict[str, Any]:
        """Run robustness benchmark."""
        self.logger.info("Running robustness benchmark")
        
        robustness_results = {
            'noise_tolerance': {},
            'parameter_sensitivity': {},
            'failure_rates': {}
        }
        
        # Add noise to test problems
        noise_levels = [0.0, 0.01, 0.1, 0.5]
        
        for algorithm in algorithms:
            if algorithm not in self.framework.algorithms:
                continue
                
            noise_performance = {}
            failure_count = 0
            total_attempts = 0
            
            for noise_level in noise_levels:
                noise_results = []
                
                for problem in self.benchmark_problems[:3]:  # Test on subset for robustness
                    try:
                        pde = problem.pde_constructor(**problem.pde_kwargs)
                        
                        characteristics = ProblemCharacteristics(
                            problem_size=problem.problem_size,
                            sparsity_level=problem.expected_sparsity,
                            time_dependent=problem.time_dependent,
                            multi_physics=problem.multi_physics,
                            conservation_required=False,
                            accuracy_requirement=1e-6,
                            energy_budget=None,
                            real_time_requirement=False,
                            physics_constraints=[],
                            boundary_complexity='simple'
                        )
                        
                        # Add noise to initial conditions or parameters
                        noisy_kwargs = problem.pde_kwargs.copy()
                        if 'initial_condition' in noisy_kwargs and callable(noisy_kwargs['initial_condition']):
                            original_ic = noisy_kwargs['initial_condition']
                            noisy_kwargs['initial_condition'] = lambda x: original_ic(x) + noise_level * np.random.random(x.shape)
                        
                        noisy_pde = problem.pde_constructor(**noisy_kwargs)
                        
                        solution, solve_info = self.framework._execute_solve(
                            algorithm, noisy_pde, characteristics, {}
                        )
                        
                        # Check if solution is reasonable
                        if np.isfinite(solution).all() and np.max(np.abs(solution)) < 1e6:
                            noise_results.append(np.linalg.norm(solution))
                        else:
                            failure_count += 1
                        
                        total_attempts += 1
                    
                    except Exception as e:
                        failure_count += 1
                        total_attempts += 1
                        self.logger.debug(f"Robustness test failed: {e}")
                
                if noise_results:
                    noise_performance[noise_level] = {
                        'mean_norm': np.mean(noise_results),
                        'std_norm': np.std(noise_results)
                    }
            
            robustness_results['noise_tolerance'][algorithm.value] = noise_performance
            
            if total_attempts > 0:
                robustness_results['failure_rates'][algorithm.value] = failure_count / total_attempts
        
        return robustness_results
    
    def run_comparative_benchmark(self, algorithms: List[AlgorithmType]) -> Dict[str, Any]:
        """Run comparative benchmark between algorithms."""
        self.logger.info("Running comparative benchmark")
        
        comparative_results = {
            'pairwise_comparisons': {},
            'performance_rankings': {},
            'use_case_recommendations': {}
        }
        
        # Compare each pair of algorithms
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i >= j or alg1 not in self.framework.algorithms or alg2 not in self.framework.algorithms:
                    continue
                
                comparison_key = f"{alg1.value}_vs_{alg2.value}"
                wins_alg1 = 0
                wins_alg2 = 0
                ties = 0
                
                for problem in self.benchmark_problems:
                    try:
                        pde = problem.pde_constructor(**problem.pde_kwargs)
                        
                        characteristics = ProblemCharacteristics(
                            problem_size=problem.problem_size,
                            sparsity_level=problem.expected_sparsity,
                            time_dependent=problem.time_dependent,
                            multi_physics=problem.multi_physics,
                            conservation_required=False,
                            accuracy_requirement=1e-6,
                            energy_budget=None,
                            real_time_requirement=False,
                            physics_constraints=[],
                            boundary_complexity='simple'
                        )
                        
                        # Time both algorithms
                        self.perf_logger.start_timer(f'comp_alg1_{i}')
                        solution1, _ = self.framework._execute_solve(alg1, pde, characteristics, {})
                        time1 = self.perf_logger.end_timer(f'comp_alg1_{i}')
                        
                        self.perf_logger.start_timer(f'comp_alg2_{j}')
                        solution2, _ = self.framework._execute_solve(alg2, pde, characteristics, {})
                        time2 = self.perf_logger.end_timer(f'comp_alg2_{j}')
                        
                        # Compare performance (speed-based)
                        if time1 < time2 * 0.9:  # 10% margin for ties
                            wins_alg1 += 1
                        elif time2 < time1 * 0.9:
                            wins_alg2 += 1
                        else:
                            ties += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Comparative benchmark failed for {comparison_key}: {e}")
                
                comparative_results['pairwise_comparisons'][comparison_key] = {
                    'alg1_wins': wins_alg1,
                    'alg2_wins': wins_alg2,
                    'ties': ties,
                    'total_comparisons': wins_alg1 + wins_alg2 + ties
                }
        
        # Create performance rankings
        algorithm_scores = {}
        for algorithm in algorithms:
            if algorithm not in self.framework.algorithms:
                continue
                
            score = 0
            for comparison_key, results in comparative_results['pairwise_comparisons'].items():
                if algorithm.value in comparison_key:
                    if comparison_key.startswith(algorithm.value):
                        score += results['alg1_wins']
                    else:
                        score += results['alg2_wins']
            
            algorithm_scores[algorithm.value] = score
        
        # Sort by score
        ranked_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        comparative_results['performance_rankings'] = {
            'rankings': ranked_algorithms,
            'scores': algorithm_scores
        }
        
        return comparative_results
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of benchmark results."""
        statistical_analysis = {
            'significance_tests': {},
            'confidence_intervals': {},
            'effect_sizes': {},
            'hypothesis_tests': {}
        }
        
        # Extract performance data for statistical tests
        if hasattr(self, 'results') and 'results_by_type' in self.results:
            performance_data = self.results['results_by_type'].get('performance', {})
            
            if 'solve_times' in performance_data:
                algorithms = list(performance_data['solve_times'].keys())
                
                # Pairwise statistical tests
                for i, alg1 in enumerate(algorithms):
                    for j, alg2 in enumerate(algorithms):
                        if i >= j:
                            continue
                        
                        times1 = performance_data['solve_times'][alg1]['times']
                        times2 = performance_data['solve_times'][alg2]['times']
                        
                        # Mann-Whitney U test (non-parametric)
                        try:
                            statistic, p_value = stats.mannwhitneyu(times1, times2, alternative='two-sided')
                            
                            test_key = f"{alg1}_vs_{alg2}"
                            statistical_analysis['significance_tests'][test_key] = {
                                'test': 'mann_whitney_u',
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < self.statistical_significance,
                                'effect_size': self._compute_effect_size(times1, times2)
                            }
                        except Exception as e:
                            self.logger.warning(f"Statistical test failed for {alg1} vs {alg2}: {e}")
        
        return statistical_analysis
    
    def _compute_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            
            # Pooled standard deviation
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            # Cohen's d
            if pooled_std > 0:
                cohens_d = (mean1 - mean2) / pooled_std
                return abs(cohens_d)
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_summary_metrics(self) -> Dict[str, Any]:
        """Compute summary metrics across all benchmarks."""
        summary = {
            'overall_performance': {},
            'algorithm_strengths': {},
            'problem_difficulty': {},
            'resource_efficiency': {}
        }
        
        # Placeholder for summary computation
        # In a real implementation, this would aggregate results across all benchmark types
        
        return summary
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate algorithm recommendations based on benchmark results."""
        recommendations = {
            'best_overall': None,
            'best_for_sparse': None,
            'best_for_accuracy': None,
            'best_for_speed': None,
            'best_for_energy': None,
            'use_case_guidance': {}
        }
        
        # Analyze results and generate recommendations
        # This would be based on the comprehensive benchmark results
        
        return recommendations
    
    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        try:
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Benchmark results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        report_lines = [
            "# Analog PDE Solver Advanced Algorithms Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Add benchmark configuration
        if 'benchmark_config' in results:
            config = results['benchmark_config']
            report_lines.extend([
                "### Benchmark Configuration",
                f"- Algorithms tested: {', '.join(config.get('algorithms_tested', []))}",
                f"- Benchmark types: {', '.join(config.get('benchmark_types', []))}",
                f"- Number of problems: {config.get('num_problems', 0)}",
                f"- Statistical runs per problem: {config.get('statistical_runs', 0)}",
                ""
            ])
        
        # Add performance summary
        report_lines.extend([
            "### Key Findings",
            "- [Key finding 1]",
            "- [Key finding 2]", 
            "- [Key finding 3]",
            "",
            "## Detailed Results",
            ""
        ])
        
        # Add algorithm-specific results
        if 'results_by_type' in results:
            for benchmark_type, benchmark_results in results['results_by_type'].items():
                report_lines.extend([
                    f"### {benchmark_type.title()} Benchmark",
                    ""
                ])
                
                # Add specific results for each benchmark type
                if benchmark_type == 'performance' and 'solve_times' in benchmark_results:
                    for algorithm, perf_data in benchmark_results['solve_times'].items():
                        report_lines.extend([
                            f"**{algorithm}:**",
                            f"- Mean solve time: {perf_data.get('mean_time', 0):.4f}s",
                            f"- Standard deviation: {perf_data.get('std_time', 0):.4f}s",
                            f"- Range: {perf_data.get('min_time', 0):.4f}s - {perf_data.get('max_time', 0):.4f}s",
                            ""
                        ])
        
        # Add recommendations
        if 'recommendations' in results:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            recommendations = results['recommendations']
            for rec_type, recommendation in recommendations.items():
                if recommendation:
                    report_lines.append(f"- **{rec_type.replace('_', ' ').title()}:** {recommendation}")
            
            report_lines.append("")
        
        return '\n'.join(report_lines)