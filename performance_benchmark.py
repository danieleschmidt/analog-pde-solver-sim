#!/usr/bin/env python3
"""Performance benchmark suite for analog PDE solver."""

import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
import traceback

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analog_pde_solver import (
        AnalogPDESolver, PoissonEquation, 
        RobustAnalogPDESolver,
        NavierStokesAnalog
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = []
        self.system_info = self._get_system_info()
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("Starting Performance Benchmark Suite")
        print("=" * 50)
        
        if not IMPORTS_AVAILABLE:
            return {
                'status': 'failed',
                'error': f'Imports not available: {IMPORT_ERROR}',
                'system_info': self.system_info
            }
        
        benchmarks = [
            ('Basic Solver Performance', self._benchmark_basic_solver),
            ('Robust Solver Performance', self._benchmark_robust_solver),
            ('Scaling Performance', self._benchmark_scaling),
            ('Navier-Stokes Performance', self._benchmark_navier_stokes),
            ('Memory Usage', self._benchmark_memory_usage),
            ('Convergence Analysis', self._benchmark_convergence)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\\nRunning {name}...")
            try:
                start_time = time.time()
                result = benchmark_func()
                duration = time.time() - start_time
                
                result.update({
                    'benchmark_name': name,
                    'total_duration': duration,
                    'status': 'completed'
                })
                
                self.results.append(result)
                print(f"  Completed in {duration:.2f}s")
                
            except Exception as e:
                error_result = {
                    'benchmark_name': name,
                    'status': 'failed',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.results.append(error_result)
                print(f"  Failed: {e}")
        
        return self._generate_final_report()
    
    def _benchmark_basic_solver(self) -> Dict[str, Any]:
        """Benchmark basic analog PDE solver."""
        results = {
            'solver_type': 'basic',
            'test_cases': []
        }
        
        sizes = [16, 32, 64]  # Smaller sizes for testing
        
        for size in sizes:
            # Create test problem
            pde = PoissonEquation(
                domain_size=size,
                boundary_conditions="dirichlet",
                source_function=lambda x, y: np.sin(np.pi * x)
            )
            
            # Create solver
            solver = AnalogPDESolver(
                crossbar_size=size,
                conductance_range=(1e-9, 1e-6),
                noise_model="realistic"
            )
            
            # Benchmark solve
            start_time = time.time()
            try:
                solution = solver.solve(pde, iterations=100)
                solve_time = time.time() - start_time
                
                test_case = {
                    'size': size,
                    'solve_time': solve_time,
                    'solution_norm': float(np.linalg.norm(solution)),
                    'success': True
                }
                
            except Exception as e:
                test_case = {
                    'size': size,
                    'solve_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
            
            results['test_cases'].append(test_case)
        
        return results
    
    def _benchmark_robust_solver(self) -> Dict[str, Any]:
        """Benchmark robust analog PDE solver."""
        results = {
            'solver_type': 'robust',
            'test_cases': []
        }
        
        sizes = [16, 32, 64]
        
        for size in sizes:
            pde = PoissonEquation(
                domain_size=size,
                boundary_conditions="dirichlet"
            )
            
            solver = RobustAnalogPDESolver(
                crossbar_size=size,
                noise_model="realistic"
            )
            
            start_time = time.time()
            try:
                solution = solver.solve(pde, iterations=100)
                solve_time = time.time() - start_time
                
                # Get additional metrics
                conv_info = solver.get_convergence_info()
                health_info = solver.health_check()
                
                test_case = {
                    'size': size,
                    'solve_time': solve_time,
                    'solution_norm': float(np.linalg.norm(solution)),
                    'iterations': conv_info.get('iterations', 0),
                    'convergence_rate': conv_info.get('convergence_rate', 0.0),
                    'crossbar_health': health_info.get('crossbar_check', 'unknown'),
                    'success': True
                }
                
            except Exception as e:
                test_case = {
                    'size': size,
                    'solve_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
            
            results['test_cases'].append(test_case)
        
        return results
    
    def _benchmark_scaling(self) -> Dict[str, Any]:
        """Benchmark scaling performance."""
        results = {
            'scaling_type': 'problem_size',
            'measurements': []
        }
        
        sizes = [8, 16, 32, 64, 128]
        
        for size in sizes:
            pde = PoissonEquation(domain_size=size)
            solver = AnalogPDESolver(crossbar_size=size, noise_model="none")
            
            # Run multiple iterations for statistical significance
            times = []
            for _ in range(3):  # 3 runs per size
                start_time = time.time()
                try:
                    solution = solver.solve(pde, iterations=50)
                    times.append(time.time() - start_time)
                except Exception:
                    times.append(float('inf'))
            
            measurement = {
                'size': size,
                'complexity': size * size,  # O(n²) for matrix operations
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times))
            }
            
            results['measurements'].append(measurement)
        
        return results
    
    def _benchmark_navier_stokes(self) -> Dict[str, Any]:
        """Benchmark Navier-Stokes solver."""
        results = {
            'solver_type': 'navier_stokes',
            'test_cases': []
        }
        
        resolutions = [(16, 16), (32, 32)]
        
        for resolution in resolutions:
            ns_solver = NavierStokesAnalog(
                resolution=resolution,
                reynolds_number=100,
                time_step=0.01
            )
            
            ns_solver.configure_hardware(num_crossbars=2)
            
            # Benchmark time stepping
            start_time = time.time()
            try:
                for timestep in range(5):  # 5 timesteps
                    u, v = ns_solver.update_velocity()
                    pressure = ns_solver.solve_pressure_poisson()
                    u, v = ns_solver.apply_pressure_correction(u, v, pressure)
                
                total_time = time.time() - start_time
                
                # Analyze power
                power_analysis = ns_solver.analyze_power()
                
                test_case = {
                    'resolution': resolution,
                    'timesteps': 5,
                    'total_time': total_time,
                    'time_per_step': total_time / 5,
                    'avg_power_mw': power_analysis.avg_power_mw,
                    'energy_per_iter_nj': power_analysis.energy_per_iter_nj,
                    'success': True
                }
                
            except Exception as e:
                test_case = {
                    'resolution': resolution,
                    'total_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
            
            results['test_cases'].append(test_case)
        
        return results
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        results = {
            'measurement_type': 'memory_usage',
            'measurements': []
        }
        
        try:
            import psutil
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            sizes = [32, 64, 128]
            
            for size in sizes:
                # Measure memory before solver creation
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # Create solver and solve
                solver = RobustAnalogPDESolver(crossbar_size=size)
                pde = PoissonEquation(domain_size=size)
                
                memory_after_creation = process.memory_info().rss / 1024 / 1024
                
                try:
                    solution = solver.solve(pde, iterations=20)
                    memory_after_solve = process.memory_info().rss / 1024 / 1024
                    solve_success = True
                except Exception:
                    memory_after_solve = process.memory_info().rss / 1024 / 1024
                    solve_success = False
                
                measurement = {
                    'size': size,
                    'baseline_memory_mb': baseline_memory,
                    'memory_before_mb': memory_before,
                    'memory_after_creation_mb': memory_after_creation,
                    'memory_after_solve_mb': memory_after_solve,
                    'memory_increase_mb': memory_after_solve - memory_before,
                    'solve_success': solve_success
                }
                
                results['measurements'].append(measurement)
                
                # Cleanup
                del solver, pde
                
        except ImportError:
            results['error'] = 'psutil not available for memory measurements'
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _benchmark_convergence(self) -> Dict[str, Any]:
        """Benchmark convergence behavior."""
        results = {
            'analysis_type': 'convergence',
            'test_cases': []
        }
        
        # Test different convergence thresholds
        thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
        
        for threshold in thresholds:
            pde = PoissonEquation(domain_size=32)
            solver = RobustAnalogPDESolver(crossbar_size=32, noise_model="none")
            
            start_time = time.time()
            try:
                solution = solver.solve(pde, iterations=200, convergence_threshold=threshold)
                solve_time = time.time() - start_time
                
                conv_info = solver.get_convergence_info()
                
                test_case = {
                    'threshold': threshold,
                    'solve_time': solve_time,
                    'iterations_used': conv_info.get('iterations', 0),
                    'final_error': conv_info.get('final_error', float('inf')),
                    'converged': conv_info.get('status') == 'solved',
                    'convergence_rate': conv_info.get('convergence_rate', 0.0),
                    'success': True
                }
                
            except Exception as e:
                test_case = {
                    'threshold': threshold,
                    'solve_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
            
            results['test_cases'].append(test_case)
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        try:
            import psutil
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
            })
        except ImportError:
            pass
        
        try:
            import numpy as np
            info['numpy_version'] = np.__version__
        except ImportError:
            pass
        
        return info
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final benchmark report."""
        successful_benchmarks = [r for r in self.results if r.get('status') == 'completed']
        failed_benchmarks = [r for r in self.results if r.get('status') == 'failed']
        
        # Calculate summary statistics
        total_time = sum(r.get('total_duration', 0) for r in successful_benchmarks)
        
        # Analyze performance trends
        performance_summary = self._analyze_performance_trends()
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.system_info,
            'summary': {
                'total_benchmarks': len(self.results),
                'successful_benchmarks': len(successful_benchmarks),
                'failed_benchmarks': len(failed_benchmarks),
                'total_benchmark_time': total_time,
                'success_rate': len(successful_benchmarks) / len(self.results) if self.results else 0
            },
            'performance_summary': performance_summary,
            'detailed_results': self.results
        }
        
        return report
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across benchmarks."""
        analysis = {
            'scaling_efficiency': 'unknown',
            'memory_efficiency': 'unknown',
            'convergence_quality': 'unknown'
        }
        
        # Analyze scaling results
        scaling_results = next((r for r in self.results if r.get('scaling_type')), None)
        if scaling_results and 'measurements' in scaling_results:
            measurements = scaling_results['measurements']
            if len(measurements) >= 3:
                times = [m['mean_time'] for m in measurements if m['mean_time'] != float('inf')]
                sizes = [m['size'] for m in measurements if m['mean_time'] != float('inf')]
                
                if len(times) >= 2:
                    # Simple efficiency analysis
                    time_ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
                    size_ratio = sizes[-1] / sizes[0] if sizes[0] > 0 else float('inf')
                    
                    if time_ratio < size_ratio * 1.5:  # Better than linear scaling
                        analysis['scaling_efficiency'] = 'excellent'
                    elif time_ratio < size_ratio * 3:  # Reasonable scaling
                        analysis['scaling_efficiency'] = 'good'
                    else:
                        analysis['scaling_efficiency'] = 'poor'
        
        # Analyze memory efficiency
        memory_results = next((r for r in self.results if r.get('measurement_type') == 'memory_usage'), None)
        if memory_results and 'measurements' in memory_results:
            memory_increases = [m.get('memory_increase_mb', 0) for m in memory_results['measurements']]
            if memory_increases:
                avg_increase = np.mean(memory_increases)
                if avg_increase < 50:  # < 50MB average increase
                    analysis['memory_efficiency'] = 'excellent'
                elif avg_increase < 200:  # < 200MB average
                    analysis['memory_efficiency'] = 'good'
                else:
                    analysis['memory_efficiency'] = 'poor'
        
        # Analyze convergence quality
        convergence_results = next((r for r in self.results if r.get('analysis_type') == 'convergence'), None)
        if convergence_results and 'test_cases' in convergence_results:
            convergence_rates = []
            for test_case in convergence_results['test_cases']:
                if test_case.get('success') and test_case.get('converged'):
                    rate = test_case.get('convergence_rate', 0)
                    if rate > 0:
                        convergence_rates.append(rate)
            
            if convergence_rates:
                avg_rate = np.mean(convergence_rates)
                if avg_rate > 0.1:  # Fast convergence
                    analysis['convergence_quality'] = 'excellent'
                elif avg_rate > 0.05:  # Moderate convergence
                    analysis['convergence_quality'] = 'good'
                else:
                    analysis['convergence_quality'] = 'poor'
        
        return analysis


def print_benchmark_summary(report: Dict[str, Any]):
    """Print benchmark summary to console."""
    print("\\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\\nTimestamp: {report['timestamp']}")
    
    # System info
    sys_info = report['system_info']
    print(f"\\nSystem Information:")
    print(f"  Python: {sys_info['python_version'].split()[0]}")
    print(f"  Platform: {sys_info['platform']}")
    if 'cpu_count' in sys_info:
        print(f"  CPU Cores: {sys_info['cpu_count']}")
    if 'memory_gb' in sys_info:
        print(f"  Memory: {sys_info['memory_gb']:.1f} GB")
    
    # Summary
    summary = report['summary']
    print(f"\\nBenchmark Summary:")
    print(f"  Total benchmarks: {summary['total_benchmarks']}")
    print(f"  Successful: {summary['successful_benchmarks']}")
    print(f"  Failed: {summary['failed_benchmarks']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total time: {summary['total_benchmark_time']:.2f}s")
    
    # Performance analysis
    perf = report['performance_summary']
    print(f"\\nPerformance Analysis:")
    print(f"  Scaling efficiency: {perf['scaling_efficiency']}")
    print(f"  Memory efficiency: {perf['memory_efficiency']}")
    print(f"  Convergence quality: {perf['convergence_quality']}")
    
    # Detailed results summary
    print(f"\\nDetailed Results:")
    for result in report['detailed_results']:
        name = result.get('benchmark_name', 'Unknown')
        status = result.get('status', 'unknown')
        duration = result.get('total_duration', 0)
        
        if status == 'completed':
            print(f"  ✓ {name}: {duration:.2f}s")
        else:
            print(f"  ✗ {name}: FAILED")
    
    print("\\n" + "="*60)


def main():
    """Main entry point for benchmark suite."""
    benchmark = PerformanceBenchmark()
    
    try:
        report = benchmark.run_all_benchmarks()
        
        # Print summary
        print_benchmark_summary(report)
        
        # Save detailed report
        report_file = Path("performance_benchmark_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\nDetailed report saved to: {report_file}")
        
        # Exit code based on results
        success_rate = report['summary']['success_rate']
        if success_rate >= 0.8:  # 80% success rate
            print("\\nBenchmark PASSED")
            return 0
        else:
            print("\\nBenchmark FAILED (low success rate)")
            return 1
            
    except Exception as e:
        print(f"\\nBenchmark suite failed: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())