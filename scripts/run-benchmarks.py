#!/usr/bin/env python3
"""
Benchmark execution script for analog PDE solvers.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analog_pde_solver.benchmarks import BenchmarkSuite
from analog_pde_solver.core.solver import AnalogPDESolver


def create_default_solver():
    """Factory function for default solver configuration."""
    return AnalogPDESolver(
        crossbar_size=128,
        conductance_range=(1e-9, 1e-6),
        noise_model="realistic"
    )


def create_fast_solver():
    """Factory function for fast solver configuration (for CI)."""
    return AnalogPDESolver(
        crossbar_size=32,
        conductance_range=(1e-9, 1e-6),
        noise_model="none"  # Disable noise for faster execution
    )


def create_accurate_solver():
    """Factory function for high-accuracy solver configuration."""
    return AnalogPDESolver(
        crossbar_size=256,
        conductance_range=(1e-10, 1e-5),
        noise_model="realistic"
    )


def main():
    parser = argparse.ArgumentParser(description="Run analog PDE solver benchmarks")
    parser.add_argument(
        "--profile", 
        choices=["fast", "default", "accurate", "all"],
        default="default",
        help="Benchmark profile to run"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        help="Specific problems to benchmark (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline file for performance comparison"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in CI mode (fast, essential tests only)"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline"
    )
    
    args = parser.parse_args()
    
    # Setup benchmark suite
    suite = BenchmarkSuite(output_dir=args.output)
    
    # Configure solver profiles
    solver_profiles = {
        "fast": create_fast_solver,
        "default": create_default_solver,
        "accurate": create_accurate_solver
    }
    
    # Select problems for CI mode
    if args.ci:
        # Run only essential problems in CI to save time
        ci_problems = [
            "poisson_2d_sine",
            "heat_1d_transient", 
            "laplace_2d_dirichlet"
        ]
        test_problems = args.problems or ci_problems
        profile = "fast"
    else:
        test_problems = args.problems
        profile = args.profile
    
    print(f"🚀 Running benchmark suite - Profile: {profile}")
    print(f"📋 Problems: {test_problems or 'all available'}")
    
    if profile == "all":
        # Run comparison across all solver profiles
        solvers = {name: factory for name, factory in solver_profiles.items() if name != "all"}
        comparison_results = suite.run_comparison_benchmark(solvers, test_problems)
        
        print("\n📊 Comparison Results:")
        for solver_name, results in comparison_results.items():
            report = suite.generate_performance_report(results)
            success_rate = report["summary"]["success_rate"]
            avg_time = report["performance_metrics"]["execution_time"]["mean"]
            print(f"  {solver_name}: {success_rate:.1%} success, {avg_time:.3f}s avg")
            
        # Save detailed comparison report
        comparison_file = suite.output_dir / "comparison_report.json"
        with open(comparison_file, 'w') as f:
            json.dump({
                "comparison_results": {
                    name: [r.to_dict() for r in results] 
                    for name, results in comparison_results.items()
                },
                "summary": {
                    name: suite.generate_performance_report(results)
                    for name, results in comparison_results.items()
                }
            }, f, indent=2)
        
        print(f"📄 Detailed comparison saved to {comparison_file}")
        
    else:
        # Run single profile
        solver_factory = solver_profiles[profile]
        results = suite.run_full_suite(solver_factory, test_problems)
        
        # Generate and display report
        report = suite.generate_performance_report(results)
        
        print(f"\n📊 Benchmark Results Summary:")
        print(f"  Total Problems: {report['summary']['total_problems']}")
        print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"  Average Time: {report['performance_metrics']['execution_time']['mean']:.3f}s")
        print(f"  Peak Memory: {report['performance_metrics']['memory_usage']['peak']:.1f}MB")
        
        # Save results
        results_file = suite.save_results(f"benchmark_{profile}.json")
        
        # Compare with baseline if provided
        if args.baseline:
            comparison = suite.compare_with_baseline(args.baseline, results)
            if "error" not in comparison:
                print(f"\n📈 Baseline Comparison:")
                summary = comparison["summary"]
                print(f"  Improved: {summary['improved']}")
                print(f"  Degraded: {summary['degraded']}")
                print(f"  Stable: {summary['stable']}")
                
                # Save comparison
                comparison_file = suite.output_dir / "baseline_comparison.json"
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2)
            else:
                print(f"❌ Baseline comparison failed: {comparison['error']}")
        
        # Save as baseline if requested
        if args.save_baseline:
            baseline_file = suite.output_dir / "baseline.json"
            with open(results_file) as src, open(baseline_file, 'w') as dst:
                dst.write(src.read())
            print(f"💾 Saved as baseline: {baseline_file}")
    
    print("✅ Benchmark execution completed")
    
    # Set exit code based on results for CI
    if args.ci:
        # Check if any critical benchmarks failed
        if profile != "all":
            report = suite.generate_performance_report(results)
            if report["summary"]["success_rate"] < 0.8:  # 80% success threshold
                print("❌ CI benchmark failure threshold exceeded")
                sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())