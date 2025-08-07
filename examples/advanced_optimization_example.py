#!/usr/bin/env python3
"""Advanced optimization example demonstrating Generation 3 capabilities."""

import numpy as np
import time
from pathlib import Path

from analog_pde_solver import (
    RobustAnalogPDESolver, PoissonEquation, 
    PerformanceOptimizer, OptimizationConfig,
    AutoScaler, ScalingPolicy,
    SystemHealthMonitor
)


def main():
    """Demonstrate advanced optimization features."""
    print("Analog PDE Solver - Advanced Optimization Example")
    print("=" * 52)
    
    # Configure optimization
    opt_config = OptimizationConfig(
        enable_parallel_crossbars=True,
        max_worker_threads=4,
        enable_adaptive_precision=True,
        enable_caching=True,
        enable_prefetch=True,
        enable_load_balancing=True
    )
    
    optimizer = PerformanceOptimizer(opt_config)
    print(f"Performance optimizer configured:")
    print(f"  Parallel crossbars: {opt_config.enable_parallel_crossbars}")
    print(f"  Worker threads: {opt_config.max_worker_threads}")
    print(f"  Adaptive precision: {opt_config.enable_adaptive_precision}")
    
    # Configure auto-scaling
    scaling_policy = ScalingPolicy(
        min_instances=2,
        max_instances=8,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        enable_predictive=True
    )
    
    autoscaler = AutoScaler(scaling_policy)
    print(f"\nAuto-scaler configured:")
    print(f"  Instance range: {scaling_policy.min_instances}-{scaling_policy.max_instances}")
    print(f"  Thresholds: {scaling_policy.scale_up_threshold:.1%} / {scaling_policy.scale_down_threshold:.1%}")
    
    # Initialize health monitoring
    health_monitor = SystemHealthMonitor(
        history_size=100,
        log_file=Path("health_logs.jsonl")
    )
    print(f"\nHealth monitor initialized with logging to health_logs.jsonl")
    
    # Create multiple PDE problems for testing
    pde_problems = []
    problem_sizes = [32, 64, 128, 256]
    
    for size in problem_sizes:
        pde = PoissonEquation(
            domain_size=size,
            boundary_conditions="dirichlet",
            source_function=lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
        )
        pde_problems.append(pde)
    
    print(f"\nCreated {len(pde_problems)} PDE problems with sizes: {problem_sizes}")
    
    # Create multiple solver instances for auto-scaling
    solvers = []
    for i in range(scaling_policy.min_instances):
        solver = RobustAnalogPDESolver(
            crossbar_size=128,
            conductance_range=(1e-9, 1e-6),
            noise_model="realistic"
        )
        solvers.append(solver)
        autoscaler.resource_pools['solvers'].append(solver)
    
    autoscaler.current_instances = len(solvers)
    print(f"Initialized {len(solvers)} solver instances")
    
    # Start monitoring
    autoscaler.start_monitoring(interval=5.0)  # Check every 5 seconds
    print("Auto-scaler monitoring started")
    
    # Performance benchmark with optimization
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKING WITH OPTIMIZATION")
    print("="*50)
    
    results = []
    
    for i, pde in enumerate(pde_problems):
        print(f"\nSolving problem {i+1}/{len(pde_problems)} (size: {problem_sizes[i]})")
        
        # Get solver from auto-scaler
        solver = autoscaler.get_available_solver()
        if solver is None:
            print("  No solver available, creating new instance")
            solver = RobustAnalogPDESolver(
                crossbar_size=problem_sizes[i],
                conductance_range=(1e-9, 1e-6),
                noise_model="realistic"
            )
        
        # Solve with adaptive precision
        start_time = time.time()
        
        try:
            solution, precision_stats = optimizer.adaptive_precision_solver(
                solver, pde, target_accuracy=1e-5
            )
            
            solve_time = time.time() - start_time
            
            # Get solver performance info
            convergence_info = solver.get_convergence_info()
            
            # Get crossbar stats if available
            crossbar_stats = {}
            if hasattr(solver.crossbar, 'get_device_stats'):
                crossbar_stats = solver.crossbar.get_device_stats()
            else:
                crossbar_stats = {
                    'is_programmed': True,
                    'programming_errors': 0,
                    'total_devices': solver.crossbar_size**2
                }
            
            # Record health metrics
            performance_data = {
                'solve_time': solve_time,
                'accuracy': precision_stats.get('accuracy_error', 0.0),
                'memory_efficiency': 0.8  # Placeholder
            }
            
            health_metrics = health_monitor.record_metrics(
                crossbar_stats, convergence_info, performance_data
            )
            
            result = {
                'problem_size': problem_sizes[i],
                'solve_time': solve_time,
                'precision_used': precision_stats.get('precision_bits', 'unknown'),
                'accuracy_error': precision_stats.get('accuracy_error', 0.0),
                'iterations': convergence_info.get('iterations', 0),
                'converged': convergence_info.get('status') == 'solved',
                'crossbar_health': health_metrics.crossbar_health,
                'solver_health': health_metrics.solver_health
            }
            
            results.append(result)
            
            print(f"  Completed in {solve_time:.3f}s")
            print(f"  Precision: {result['precision_used']} bits")
            print(f"  Accuracy: {result['accuracy_error']:.2e}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Health: Crossbar {result['crossbar_health']:.1f}%, Solver {result['solver_health']:.1f}%")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'problem_size': problem_sizes[i],
                'solve_time': float('inf'),
                'error': str(e)
            })
    
    # Performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        total_time = sum(r['solve_time'] for r in successful_results)
        avg_time = total_time / len(successful_results)
        
        print(f"Successful solves: {len(successful_results)}/{len(results)}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time: {avg_time:.3f}s")
        
        print("\\nPer-problem results:")
        for result in successful_results:
            print(f"  Size {result['problem_size']:3d}: {result['solve_time']:.3f}s, "
                  f"{result['precision_used']:2}bits, {result['iterations']:3d} iter")
    
    # Get optimization statistics
    print("\\n" + "="*50)
    print("OPTIMIZATION STATISTICS")
    print("="*50)
    
    opt_stats = optimizer.get_performance_stats()
    
    print("Cache Performance:")
    cache_stats = opt_stats['cache_statistics']
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Cache size: {cache_stats['cache_size']}")
    
    print("\\nWorker Statistics:")
    worker_stats = opt_stats['worker_statistics']
    print(f"  Thread pool active: {worker_stats['thread_pool_active']}")
    print(f"  Max workers: {worker_stats['max_workers']}")
    
    # Get auto-scaling statistics
    scaling_stats = autoscaler.get_scaling_stats()
    
    print("\\nAuto-Scaling Statistics:")
    print(f"  Current instances: {scaling_stats['current_instances']}")
    print(f"  Scale ups (24h): {scaling_stats['recent_events']['scale_ups_24h']}")
    print(f"  Scale downs (24h): {scaling_stats['recent_events']['scale_downs_24h']}")
    
    if 'efficiency' in scaling_stats:
        efficiency = scaling_stats['efficiency']
        print(f"  Average utilization: {efficiency['avg_utilization']:.1%}")
        print(f"  Target utilization: {efficiency['target_utilization']:.1%}")
    
    # Get health summary
    print("\\n" + "="*50)
    print("SYSTEM HEALTH SUMMARY")
    print("="*50)
    
    health_summary = health_monitor.get_system_summary()
    
    if health_summary.get('status') != 'no_data':
        print(f"Overall status: {health_summary['overall_status']}")
        print(f"Total operations: {health_summary['total_operations']}")
        print(f"Total errors: {health_summary['total_errors']}")
        print(f"System uptime: {health_summary['system_uptime']:.1f}s")
        
        current_metrics = health_summary.get('current_metrics', {})
        if current_metrics:
            print(f"\\nCurrent Metrics:")
            print(f"  Crossbar health: {current_metrics.get('crossbar_health', 0):.1f}%")
            print(f"  Solver health: {current_metrics.get('solver_health', 0):.1f}%")
            print(f"  Memory usage: {current_metrics.get('memory_usage_mb', 0):.1f} MB")
            print(f"  Error rate: {current_metrics.get('error_rate', 0):.2%}")
        
        # Show trends if available
        trends = health_summary.get('trends', {})
        if trends and trends.get('status') != 'insufficient_data':
            print(f"\\nTrends:")
            print(f"  Crossbar health: {trends.get('crossbar_health', 'unknown')}")
            print(f"  Solver health: {trends.get('solver_health', 'unknown')}")
            print(f"  Memory usage: {trends.get('memory_usage', 'unknown')}")
    
    # Export health data
    health_export_path = Path("health_metrics_export.json")
    if health_monitor.export_metrics(health_export_path):
        print(f"\\nHealth metrics exported to {health_export_path}")
    
    # Cleanup
    print("\\nCleaning up...")
    autoscaler.stop_monitoring()
    optimizer.cleanup()
    autoscaler.cleanup()
    
    print("\\nAdvanced optimization example completed!")
    print("This example demonstrated:")
    print("- Performance optimization with caching and parallelization")
    print("- Adaptive precision solving")
    print("- Auto-scaling with predictive load balancing")
    print("- Comprehensive health monitoring")
    print("- Real-time metrics collection and analysis")


if __name__ == "__main__":
    main()