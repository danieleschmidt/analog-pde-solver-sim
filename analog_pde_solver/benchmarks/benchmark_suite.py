"""Main benchmarking suite for analog PDE solvers."""

import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from ..core.solver import AnalogPDESolver
from .standard_problems import StandardProblems
from .performance_metrics import PerformanceMetrics


@dataclass
class BenchmarkResult:
    """Container for benchmark execution results."""
    problem_name: str
    solver_type: str
    execution_time: float
    memory_usage: float
    accuracy_error: float
    energy_estimate: float
    convergence_iterations: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class BenchmarkSuite:
    """Comprehensive benchmarking suite for analog PDE solvers."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.problems = StandardProblems()
        self.metrics = PerformanceMetrics()
        self.results: List[BenchmarkResult] = []
        
    def run_single_benchmark(
        self, 
        problem_name: str,
        solver_factory: Callable[[], AnalogPDESolver],
        iterations: int = 3
    ) -> BenchmarkResult:
        """Run a single benchmark problem multiple times and average results."""
        
        # Get the problem configuration
        problem_config = self.problems.get_problem(problem_name)
        if not problem_config:
            raise ValueError(f"Unknown problem: {problem_name}")
            
        execution_times = []
        memory_usages = []
        accuracy_errors = []
        convergence_iters = []
        
        for _ in range(iterations):
            # Create fresh solver instance
            solver = solver_factory()
            
            # Prepare problem
            equation = problem_config["equation_factory"]()
            expected_solution = problem_config.get("reference_solution")
            
            # Measure execution time and memory
            start_time = time.perf_counter()
            start_memory = self.metrics.get_memory_usage()
            
            # Execute solver
            try:
                result = solver.solve(
                    equation,
                    iterations=problem_config.get("max_iterations", 100),
                    convergence_threshold=problem_config.get("tolerance", 1e-6)
                )
                
                end_time = time.perf_counter()
                end_memory = self.metrics.get_memory_usage()
                
                execution_times.append(end_time - start_time)
                memory_usages.append(end_memory - start_memory)
                
                # Calculate accuracy if reference solution available
                if expected_solution is not None:
                    if callable(expected_solution):
                        expected = expected_solution()
                    else:
                        expected = expected_solution
                    
                    accuracy_error = np.mean(np.abs(result - expected))
                    accuracy_errors.append(accuracy_error)
                else:
                    accuracy_errors.append(0.0)
                    
                # Mock convergence iterations (would be real in actual solver)
                convergence_iters.append(problem_config.get("expected_iterations", 50))
                
            except Exception as e:
                print(f"Benchmark failed for {problem_name}: {e}")
                execution_times.append(float('inf'))
                memory_usages.append(0.0)
                accuracy_errors.append(float('inf'))
                convergence_iters.append(0)
        
        # Calculate averages
        avg_time = np.mean(execution_times) if execution_times else float('inf')
        avg_memory = np.mean(memory_usages) if memory_usages else 0.0
        avg_error = np.mean(accuracy_errors) if accuracy_errors else float('inf')
        avg_iterations = int(np.mean(convergence_iters)) if convergence_iters else 0
        
        # Estimate energy consumption (mock calculation)
        energy_estimate = self._estimate_energy_consumption(
            avg_time, 
            problem_config.get("crossbar_size", 128)
        )
        
        result = BenchmarkResult(
            problem_name=problem_name,
            solver_type=solver_factory().__class__.__name__,
            execution_time=avg_time,
            memory_usage=avg_memory,
            accuracy_error=avg_error,
            energy_estimate=energy_estimate,
            convergence_iterations=avg_iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        return result
    
    def run_full_suite(
        self, 
        solver_factory: Callable[[], AnalogPDESolver],
        problems: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """Run the complete benchmark suite."""
        
        if problems is None:
            problems = self.problems.get_all_problem_names()
            
        print(f"🚀 Running benchmark suite with {len(problems)} problems...")
        
        suite_results = []
        for problem_name in problems:
            print(f"  Running {problem_name}...")
            try:
                result = self.run_single_benchmark(problem_name, solver_factory)
                suite_results.append(result)
                print(f"    ✅ Completed in {result.execution_time:.3f}s")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                
        return suite_results
    
    def run_comparison_benchmark(
        self, 
        solvers: Dict[str, Callable[[], AnalogPDESolver]],
        problems: Optional[List[str]] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run comparative benchmarks across multiple solvers."""
        
        if problems is None:
            problems = self.problems.get_all_problem_names()
            
        comparison_results = {}
        
        for solver_name, solver_factory in solvers.items():
            print(f"\n🔧 Testing {solver_name}...")
            comparison_results[solver_name] = self.run_full_suite(
                solver_factory, problems
            )
            
        return comparison_results
    
    def generate_performance_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        
        if not results:
            return {"error": "No results to analyze"}
            
        # Calculate aggregate metrics
        total_problems = len(results)
        successful_runs = len([r for r in results if r.execution_time != float('inf')])
        
        execution_times = [r.execution_time for r in results if r.execution_time != float('inf')]
        memory_usages = [r.memory_usage for r in results]
        accuracy_errors = [r.accuracy_error for r in results if r.accuracy_error != float('inf')]
        energy_estimates = [r.energy_estimate for r in results]
        
        report = {
            "summary": {
                "total_problems": total_problems,
                "successful_runs": successful_runs,
                "success_rate": successful_runs / total_problems if total_problems > 0 else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance_metrics": {
                "execution_time": {
                    "mean": float(np.mean(execution_times)) if execution_times else 0,
                    "median": float(np.median(execution_times)) if execution_times else 0,
                    "std": float(np.std(execution_times)) if execution_times else 0,
                    "min": float(np.min(execution_times)) if execution_times else 0,
                    "max": float(np.max(execution_times)) if execution_times else 0
                },
                "memory_usage": {
                    "mean": float(np.mean(memory_usages)) if memory_usages else 0,
                    "peak": float(np.max(memory_usages)) if memory_usages else 0
                },
                "accuracy": {
                    "mean_error": float(np.mean(accuracy_errors)) if accuracy_errors else 0,
                    "max_error": float(np.max(accuracy_errors)) if accuracy_errors else 0
                },
                "energy_efficiency": {
                    "total_energy": float(np.sum(energy_estimates)) if energy_estimates else 0,
                    "average_energy": float(np.mean(energy_estimates)) if energy_estimates else 0
                }
            },
            "problem_breakdown": [result.to_dict() for result in results]
        }
        
        return report
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save benchmark results to JSON file."""
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        report = self.generate_performance_report(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📊 Results saved to {output_path}")
        return output_path
    
    def _estimate_energy_consumption(self, execution_time: float, crossbar_size: int) -> float:
        """Estimate energy consumption based on execution parameters."""
        
        # Mock energy model for analog crossbar
        # Real implementation would use detailed power models
        
        # Base power consumption (mW)
        crossbar_power = crossbar_size * crossbar_size * 0.001  # 1μW per cell
        peripheral_power = crossbar_size * 0.1  # 100μW per row/column
        digital_control_power = 10.0  # 10mW for control logic
        
        total_power_mw = crossbar_power + peripheral_power + digital_control_power
        
        # Energy = Power × Time (convert to nJ)
        energy_nj = total_power_mw * execution_time * 1e6
        
        return energy_nj
    
    def compare_with_baseline(
        self, 
        baseline_file: str, 
        current_results: List[BenchmarkResult],
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Compare current results with baseline performance."""
        
        try:
            with open(baseline_file) as f:
                baseline_data = json.load(f)
                
            baseline_results = baseline_data.get("problem_breakdown", [])
            
            comparisons = []
            
            for current in current_results:
                # Find matching baseline result
                baseline = next(
                    (b for b in baseline_results if b["problem_name"] == current.problem_name),
                    None
                )
                
                if baseline:
                    time_ratio = current.execution_time / baseline["execution_time"]
                    memory_ratio = current.memory_usage / max(baseline["memory_usage"], 1)
                    
                    comparison = {
                        "problem": current.problem_name,
                        "time_ratio": time_ratio,
                        "memory_ratio": memory_ratio,
                        "performance_change": "improved" if time_ratio < (1 - tolerance) else
                                            "degraded" if time_ratio > (1 + tolerance) else
                                            "stable",
                        "current_time": current.execution_time,
                        "baseline_time": baseline["execution_time"]
                    }
                    
                    comparisons.append(comparison)
                    
            return {
                "baseline_file": baseline_file,
                "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "comparisons": comparisons,
                "summary": {
                    "improved": len([c for c in comparisons if c["performance_change"] == "improved"]),
                    "degraded": len([c for c in comparisons if c["performance_change"] == "degraded"]),
                    "stable": len([c for c in comparisons if c["performance_change"] == "stable"])
                }
            }
            
        except FileNotFoundError:
            return {"error": f"Baseline file {baseline_file} not found"}
        except Exception as e:
            return {"error": f"Comparison failed: {e}"}