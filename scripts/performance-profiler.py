#!/usr/bin/env python3
"""
Advanced Performance Profiling Tool for Analog PDE Solver

This tool provides comprehensive performance analysis including:
- Memory usage profiling
- CPU utilization tracking  
- Hardware simulation performance
- Energy estimation
- Bottleneck identification
"""

import argparse
import cProfile
import json
import pstats
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Optional
import tracemalloc
import psutil
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from analog_pde_solver import AnalogPDESolver, PoissonEquation
    from analog_pde_solver.benchmarks import PDEBenchmarkSuite
except ImportError as e:
    print(f"Warning: Could not import analog_pde_solver: {e}")
    print("Run with dependencies installed: pip install -e '.[dev,hardware]'")


class PerformanceProfiler:
    """Advanced performance profiling for analog PDE solving."""
    
    def __init__(self, output_dir: Path = Path("performance_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function."""
        tracemalloc.start()
        
        # Baseline memory
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Memory after execution
        memory_after = process.memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_usage': {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024,
                'rss_before_mb': memory_before / 1024 / 1024,
                'rss_after_mb': memory_after / 1024 / 1024,
                'rss_delta_mb': (memory_after - memory_before) / 1024 / 1024
            }
        }
    
    def profile_cpu_usage(self, func, *args, **kwargs):
        """Profile CPU usage during function execution."""
        # CPU monitoring setup
        process = psutil.Process()
        cpu_samples = []
        
        def monitor_cpu():
            while hasattr(monitor_cpu, 'running'):
                cpu_samples.append(process.cpu_percent())
                time.sleep(0.1)
        
        # Start monitoring in background
        import threading
        monitor_cpu.running = True
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Execute function with profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs) 
        end_time = time.time()
        
        profiler.disable()
        
        # Stop monitoring
        delattr(monitor_cpu, 'running')
        monitor_thread.join()
        
        # Analyze profiling results
        stats_stream = StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative').print_stats(20)
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'cpu_usage': {
                'avg_percent': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
                'max_percent': max(cpu_samples) if cpu_samples else 0,
                'samples': len(cpu_samples)
            },
            'profiling_stats': stats_stream.getvalue()
        }
    
    def benchmark_solver_performance(self, problem_sizes: List[int] = None):
        """Comprehensive solver performance benchmarking."""
        if problem_sizes is None:
            problem_sizes = [64, 128, 256, 512]
            
        results = {}
        
        for size in problem_sizes:
            print(f"🔍 Benchmarking problem size: {size}x{size}")
            
            # Create test problem
            def create_and_solve():
                pde = PoissonEquation(
                    domain_size=(size, size),
                    boundary_conditions="dirichlet"
                )
                
                solver = AnalogPDESolver(
                    crossbar_size=size,
                    conductance_range=(1e-9, 1e-6)
                )
                
                return solver.solve(pde, iterations=50)
            
            # Profile memory and CPU
            memory_profile = self.profile_memory_usage(create_and_solve)
            cpu_profile = self.profile_cpu_usage(create_and_solve)
            
            results[f"{size}x{size}"] = {
                'problem_size': size * size,
                'execution_time': memory_profile['execution_time'],
                'memory_usage': memory_profile['memory_usage'],
                'cpu_usage': cpu_profile['cpu_usage'],
                'performance_metrics': {
                    'iterations_per_second': 50 / memory_profile['execution_time'],
                    'memory_efficiency': size * size / memory_profile['memory_usage']['peak_mb'],
                    'cpu_efficiency': 50 / (cpu_profile['cpu_usage']['avg_percent'] * memory_profile['execution_time'])
                }
            }
            
        return results
    
    def estimate_energy_consumption(self, benchmark_results: Dict):
        """Estimate energy consumption based on performance metrics."""
        energy_results = {}
        
        # Energy estimation constants (typical values)
        CPU_POWER_WATTS = 65  # Typical CPU TDP
        MEMORY_POWER_WATTS_PER_GB = 3  # DDR4 power consumption
        
        for problem_name, metrics in benchmark_results.items():
            exec_time = metrics['execution_time']
            cpu_usage = metrics['cpu_usage']['avg_percent'] / 100
            memory_gb = metrics['memory_usage']['peak_mb'] / 1024
            
            # Calculate energy consumption
            cpu_energy = CPU_POWER_WATTS * cpu_usage * exec_time
            memory_energy = MEMORY_POWER_WATTS_PER_GB * memory_gb * exec_time
            total_energy = cpu_energy + memory_energy
            
            # Analog hardware projections (based on research estimates)
            analog_speedup = 500  # Conservative estimate
            analog_energy_efficiency = 100  # Energy efficiency improvement
            
            projected_analog_time = exec_time / analog_speedup
            projected_analog_energy = total_energy / analog_energy_efficiency
            
            energy_results[problem_name] = {
                'digital_energy_joules': total_energy,
                'digital_power_watts': total_energy / exec_time,
                'analog_projected_energy_joules': projected_analog_energy,
                'analog_projected_time_seconds': projected_analog_time,
                'energy_efficiency_ratio': total_energy / projected_analog_energy,
                'speedup_ratio': exec_time / projected_analog_time
            }
            
        return energy_results
    
    def identify_bottlenecks(self, profiling_stats: str) -> Dict[str, Any]:
        """Identify performance bottlenecks from profiling data."""
        lines = profiling_stats.split('\n')
        bottlenecks = []
        
        # Parse profiling output to identify slow functions
        for line in lines:
            if 'function calls' in line or 'primitive calls' in line:
                continue
            if line.strip() and not line.startswith('ncalls'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        cumtime = float(parts[3])
                        if cumtime > 0.1:  # Functions taking > 100ms
                            function_name = ' '.join(parts[5:])
                            bottlenecks.append({
                                'function': function_name,
                                'cumulative_time': cumtime,
                                'calls': parts[0]
                            })
                    except (ValueError, IndexError):
                        continue
        
        return {
            'bottlenecks': sorted(bottlenecks, key=lambda x: x['cumulative_time'], reverse=True)[:10],
            'total_slow_functions': len(bottlenecks)
        }
    
    def generate_performance_report(self, benchmark_results: Dict, energy_results: Dict):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            },
            'benchmark_results': benchmark_results,
            'energy_analysis': energy_results,
            'summary': {}
        }
        
        # Calculate summary statistics
        execution_times = [r['execution_time'] for r in benchmark_results.values()]
        memory_usage = [r['memory_usage']['peak_mb'] for r in benchmark_results.values()]
        
        report['summary'] = {
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'avg_memory_usage_mb': sum(memory_usage) / len(memory_usage),
            'max_memory_usage_mb': max(memory_usage),
            'total_energy_digital_joules': sum(e['digital_energy_joules'] for e in energy_results.values()),
            'total_energy_analog_projected_joules': sum(e['analog_projected_energy_joules'] for e in energy_results.values()),
            'average_speedup_projection': sum(e['speedup_ratio'] for e in energy_results.values()) / len(energy_results),
            'average_energy_efficiency': sum(e['energy_efficiency_ratio'] for e in energy_results.values()) / len(energy_results)
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = None):
        """Save performance report to file."""
        if filename is None:
            filename = f"performance_report_{int(time.time())}.json"
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📊 Performance report saved to: {filepath}")
        return filepath
    
    def print_summary(self, report: Dict):
        """Print performance summary to console."""
        summary = report['summary']
        
        print("\n🚀 PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Average Execution Time: {summary['avg_execution_time']:.3f}s")
        print(f"Peak Memory Usage: {summary['max_memory_usage_mb']:.1f}MB")
        print(f"Total Digital Energy: {summary['total_energy_digital_joules']:.3f}J")
        print(f"Projected Analog Energy: {summary['total_energy_analog_projected_joules']:.6f}J")
        print(f"Average Speedup Projection: {summary['average_speedup_projection']:.1f}×")
        print(f"Average Energy Efficiency: {summary['average_energy_efficiency']:.1f}×")
        
        print("\n📈 DETAILED RESULTS")
        print("-" * 30)
        for problem, results in report['benchmark_results'].items():
            print(f"{problem}: {results['execution_time']:.3f}s, {results['memory_usage']['peak_mb']:.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="Advanced Performance Profiler for Analog PDE Solver")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256], 
                       help="Problem sizes to benchmark")
    parser.add_argument("--output", type=str, help="Output filename for report")
    parser.add_argument("--output-dir", type=Path, default=Path("performance_results"),
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🔬 Starting Advanced Performance Profiling...")
    print(f"Problem sizes: {args.sizes}")
    
    profiler = PerformanceProfiler(args.output_dir)
    
    try:
        # Run benchmarks
        benchmark_results = profiler.benchmark_solver_performance(args.sizes)
        
        # Analyze energy consumption
        energy_results = profiler.estimate_energy_consumption(benchmark_results)
        
        # Generate report
        report = profiler.generate_performance_report(benchmark_results, energy_results)
        
        # Save and display results
        profiler.save_report(report, args.output)
        profiler.print_summary(report)
        
        print(f"\n✅ Profiling complete! Results saved to {args.output_dir}")
        
    except ImportError:
        print("❌ Could not import analog_pde_solver package.")
        print("Please install dependencies: pip install -e '.[dev,hardware]'")
        return 1
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())