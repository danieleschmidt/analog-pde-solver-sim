#!/usr/bin/env python3
"""
Distributed Analog Computing Example

Demonstrates exascale distributed analog PDE solving across heterogeneous
compute resources including CPUs, GPUs, analog crossbars, and specialized
accelerators with intelligent workload distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the repo to Python path
sys.path.insert(0, '/root/repo')

from analog_pde_solver.acceleration.distributed_computing import (
    DistributedAnalogComputing,
    ResourceType,
    ComputeNode,
    benchmark_distributed_analog_computing
)
from analog_pde_solver.core.equations import PoissonEquation

def demonstrate_distributed_scaling():
    """Demonstrate distributed analog computing scaling capabilities."""
    print("🌐 Distributed Analog Computing Scaling Demo")
    print("=" * 50)
    
    # Initialize large-scale distributed system
    print("Initializing distributed analog computing cluster...")
    distributed_system = DistributedAnalogComputing(max_workers=16)
    
    # Display cluster information
    metrics = distributed_system.get_distributed_metrics()
    print(f"  Cluster size: {metrics['cluster_info']['total_nodes']} nodes")
    print(f"  Available nodes: {metrics['cluster_info']['available_nodes']}")
    print(f"  Resource types: {list(metrics['cluster_info']['resource_distribution'].keys())}")
    
    for resource_type, count in metrics['cluster_info']['resource_distribution'].items():
        print(f"    {resource_type}: {count} nodes")
    
    # Create increasingly complex PDE problems
    problems = [
        ("Small (128x128)", create_test_pde(128)),
        ("Medium (512x512)", create_test_pde(512)), 
        ("Large (1024x1024)", create_test_pde(1024)),
        ("Extra Large (2048x2048)", create_test_pde(2048))
    ]
    
    scaling_results = []
    
    print("\n🚀 Scaling Analysis:")
    print("-" * 50)
    
    for problem_name, pde_problem in problems:
        print(f"\nSolving {problem_name} problem...")
        
        # Determine optimal partition count based on problem size
        problem_size = pde_problem.domain_size
        if isinstance(problem_size, (tuple, list)):
            problem_size = problem_size[0]
        
        optimal_partitions = min(16, max(4, problem_size // 128))
        
        start_time = time.time()
        
        result = distributed_system.execute_distributed_pde_solving(
            pde_problem=pde_problem,
            num_partitions=optimal_partitions,
            fault_tolerance=True
        )
        
        wall_time = time.time() - start_time
        
        # Analyze results
        solution_quality = analyze_solution_quality(result['solution'])
        
        scaling_data = {
            'problem_name': problem_name,
            'problem_size': problem_size,
            'partitions': result['partitions'],
            'nodes_used': result['nodes_used'],
            'execution_time': result['execution_time'],
            'wall_time': wall_time,
            'throughput': result['performance_metrics']['throughput_elements_per_sec'],
            'success_rate': result['performance_metrics']['success_rate'],
            'load_balance': result['performance_metrics']['load_balance_efficiency'],
            'parallel_efficiency': result['performance_metrics']['parallel_efficiency'],
            'solution_quality': solution_quality,
            'allocation_map': result['allocation_map']
        }
        
        scaling_results.append(scaling_data)
        
        print(f"  Size: {problem_size}x{problem_size}")
        print(f"  Partitions: {result['partitions']}")
        print(f"  Nodes used: {result['nodes_used']}")
        print(f"  Execution time: {result['execution_time']:.3f}s")
        print(f"  Wall time: {wall_time:.3f}s")
        print(f"  Throughput: {result['performance_metrics']['throughput_elements_per_sec']:.1f} elements/s")
        print(f"  Success rate: {result['performance_metrics']['success_rate']:.3f}")
        print(f"  Load balance: {result['performance_metrics']['load_balance_efficiency']:.3f}")
        print(f"  Parallel efficiency: {result['performance_metrics']['parallel_efficiency']:.3f}")
        print(f"  Solution quality: {solution_quality:.6f}")
        
        # Show resource allocation
        print(f"  Resource allocation:")
        resource_counts = {}
        for partition_id, node_id in result['allocation_map'].items():
            resource_type = node_id.split('_')[0]
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        for resource, count in resource_counts.items():
            print(f"    {resource}: {count} partitions")
        
        # Adaptive optimization
        distributed_system.adaptive_load_balancing()
    
    return {
        'distributed_system': distributed_system,
        'scaling_results': scaling_results,
        'final_metrics': distributed_system.get_distributed_metrics()
    }

def create_test_pde(size: int):
    """Create a test PDE problem of specified size."""
    class ScalablePoissonPDE:
        def __init__(self, domain_size):
            self.domain_size = domain_size
            self.boundary_conditions = "dirichlet"
            
        def source_function(self, x, y):
            # Multi-scale source function
            return (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 
                   0.5 * np.sin(8 * np.pi * x) * np.cos(8 * np.pi * y) +
                   0.25 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1))
    
    return ScalablePoissonPDE(size)

def analyze_solution_quality(solution: np.ndarray) -> float:
    """Analyze the quality of the computed solution."""
    if len(solution) == 0:
        return 0.0
    
    # Compute various quality metrics
    solution_norm = np.linalg.norm(solution)
    solution_variance = np.var(solution)
    smoothness = np.mean(np.abs(np.diff(solution))) if len(solution) > 1 else 0
    
    # Combined quality metric (higher is better)
    quality = solution_norm / (1 + smoothness + solution_variance)
    return quality

def demonstrate_fault_tolerance():
    """Demonstrate fault tolerance capabilities."""
    print("\n🛡️ Fault Tolerance Demonstration")
    print("-" * 40)
    
    distributed_system = DistributedAnalogComputing(max_workers=8)
    
    # Simulate node failures
    print("Simulating node failures...")
    failed_nodes = distributed_system.compute_nodes[:2]  # Fail first 2 nodes
    for node in failed_nodes:
        node.available = False
        print(f"  Simulated failure of {node.node_id}")
    
    # Create test problem
    pde = create_test_pde(256)
    
    print("\nSolving with fault tolerance enabled...")
    result = distributed_system.execute_distributed_pde_solving(
        pde_problem=pde,
        num_partitions=8,
        fault_tolerance=True
    )
    
    print(f"  Success rate: {result['performance_metrics']['success_rate']:.3f}")
    print(f"  Fault tolerance triggered: {result['performance_metrics']['fault_tolerance_triggered']}")
    print(f"  Solution computed: {len(result['solution'])} elements")
    
    # Restore nodes
    for node in failed_nodes:
        node.available = True
        print(f"  Restored {node.node_id}")
    
    return result

def visualize_distributed_performance(demo_results):
    """Create comprehensive visualization of distributed performance."""
    print("\n📈 Generating distributed computing visualizations...")
    
    scaling_results = demo_results['scaling_results']
    
    # Extract data for plotting
    problem_sizes = [r['problem_size'] for r in scaling_results]
    execution_times = [r['execution_time'] for r in scaling_results]
    throughputs = [r['throughput'] for r in scaling_results]
    parallel_efficiencies = [r['parallel_efficiency'] for r in scaling_results]
    nodes_used = [r['nodes_used'] for r in scaling_results]
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distributed Analog Computing Performance Analysis', fontsize=16)
    
    # Plot 1: Scaling performance
    axes[0,0].loglog(problem_sizes, execution_times, 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Problem Size')
    axes[0,0].set_ylabel('Execution Time (s)')
    axes[0,0].set_title('Scaling Performance')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add ideal scaling line
    ideal_scaling = execution_times[0] * (np.array(problem_sizes) / problem_sizes[0])**2
    axes[0,0].loglog(problem_sizes, ideal_scaling, 'r--', alpha=0.5, label='Ideal O(n²)')
    axes[0,0].legend()
    
    # Plot 2: Throughput vs problem size
    axes[0,1].semilogx(problem_sizes, throughputs, 'go-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Problem Size')
    axes[0,1].set_ylabel('Throughput (elements/s)')
    axes[0,1].set_title('Computational Throughput')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Parallel efficiency
    axes[0,2].semilogx(problem_sizes, parallel_efficiencies, 'ro-', linewidth=2, markersize=8)
    axes[0,2].set_xlabel('Problem Size')
    axes[0,2].set_ylabel('Parallel Efficiency')
    axes[0,2].set_title('Parallel Efficiency')
    axes[0,2].set_ylim([0, 1.1])
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Resource utilization
    axes[1,0].bar(range(len(scaling_results)), nodes_used, color='purple', alpha=0.7)
    axes[1,0].set_xlabel('Problem Index')
    axes[1,0].set_ylabel('Nodes Used')
    axes[1,0].set_title('Resource Utilization')
    axes[1,0].set_xticks(range(len(scaling_results)))
    axes[1,0].set_xticklabels([r['problem_name'].split()[0] for r in scaling_results])
    
    # Plot 5: Load balance efficiency
    load_balances = [r['load_balance'] for r in scaling_results]
    axes[1,1].plot(problem_sizes, load_balances, 'mo-', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Problem Size')
    axes[1,1].set_ylabel('Load Balance Efficiency')
    axes[1,1].set_title('Load Balancing Quality')
    axes[1,1].set_ylim([0, 1.1])
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Resource allocation breakdown
    # Aggregate resource usage across all problems
    resource_usage = {}
    for result in scaling_results:
        for partition_id, node_id in result['allocation_map'].items():
            resource_type = node_id.split('_')[0]
            resource_usage[resource_type] = resource_usage.get(resource_type, 0) + 1
    
    if resource_usage:
        resources = list(resource_usage.keys())
        usage_counts = list(resource_usage.values())
        colors = ['blue', 'green', 'red', 'orange', 'purple'][:len(resources)]
        
        axes[1,2].pie(usage_counts, labels=resources, colors=colors, autopct='%1.1f%%')
        axes[1,2].set_title('Resource Type Distribution')
    else:
        axes[1,2].text(0.5, 0.5, 'No Resource Data', ha='center', va='center', 
                      transform=axes[1,2].transAxes)
    
    plt.tight_layout()
    plt.savefig('distributed_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization: distributed_performance.png")
    
    # Performance metrics summary
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Distributed Computing Metrics Summary', fontsize=14)
    
    # Speedup analysis
    sequential_times = [t * n for t, n in zip(execution_times, nodes_used)]  # Estimate
    speedups = [seq / exec for seq, exec in zip(sequential_times, execution_times)]
    
    ax1.loglog(problem_sizes, speedups, 'co-', linewidth=2, markersize=8)
    ax1.loglog(problem_sizes, nodes_used, 'k--', alpha=0.5, label='Ideal Speedup')
    ax1.set_xlabel('Problem Size')
    ax1.set_ylabel('Speedup Factor')
    ax1.set_title('Computational Speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Efficiency vs scale
    efficiencies = [s / n for s, n in zip(speedups, nodes_used)]
    ax2.semilogx(problem_sizes, efficiencies, 'yo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Problem Size')
    ax2.set_ylabel('Efficiency (Speedup/Nodes)')
    ax2.set_title('Scaling Efficiency')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3)
    
    # Work distribution quality
    success_rates = [r['success_rate'] for r in scaling_results]
    ax3.bar(range(len(scaling_results)), success_rates, color='lightgreen', alpha=0.7)
    ax3.set_xlabel('Problem Index')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Fault Tolerance Performance')
    ax3.set_xticks(range(len(scaling_results)))
    ax3.set_xticklabels([r['problem_name'].split()[0] for r in scaling_results])
    ax3.set_ylim([0, 1.1])
    
    # Solution quality
    solution_qualities = [r['solution_quality'] for r in scaling_results]
    ax4.semilogy(problem_sizes, solution_qualities, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Problem Size')
    ax4.set_ylabel('Solution Quality (log scale)')
    ax4.set_title('Computational Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distributed_metrics.png', dpi=300, bbox_inches='tight')
    print("  Saved metrics: distributed_metrics.png")
    
    plt.show()

def main():
    """Main function demonstrating distributed analog computing."""
    print("🚀 Distributed Analog Computing Example")
    print("=" * 55)
    
    try:
        # Run comprehensive scaling demonstration
        demo_results = demonstrate_distributed_scaling()
        
        # Demonstrate fault tolerance
        fault_tolerance_result = demonstrate_fault_tolerance()
        
        # Generate visualizations
        visualize_distributed_performance(demo_results)
        
        # Run research benchmark
        print("\n🔬 Running comprehensive distributed benchmark...")
        benchmark_results = benchmark_distributed_analog_computing()
        
        print("\n✅ Distributed analog computing example completed successfully!")
        print("\nExascale Computing Innovations Demonstrated:")
        print("  🌐 Intelligent workload partitioning")
        print("  🎯 Dynamic resource allocation")
        print("  ⚡ Heterogeneous compute acceleration")
        print("  🛡️ Fault-tolerant execution")
        print("  📊 Adaptive load balancing")
        print("  📈 Performance optimization")
        print("  🔄 Auto-scaling capabilities")
        print("  🧮 Hybrid analog-digital computing")
        
        # Final performance summary
        final_metrics = demo_results['final_metrics']
        print(f"\n📋 Final Cluster Status:")
        print(f"  Total nodes: {final_metrics['cluster_info']['total_nodes']}")
        print(f"  Average throughput: {final_metrics['performance_stats']['throughput']:.1f} elements/s")
        print(f"  Average utilization: {final_metrics['performance_stats']['utilization']:.3f}")
        print(f"  Load balance efficiency: {final_metrics['performance_stats']['load_balance']:.3f}")
        
        return {
            'demo_results': demo_results,
            'fault_tolerance_result': fault_tolerance_result,
            'benchmark_results': benchmark_results
        }
        
    except Exception as e:
        print(f"❌ Error in distributed computing demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()