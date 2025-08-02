#!/usr/bin/env python3
"""
Automated metrics collection script for Analog PDE Solver project.

This script collects various metrics from different sources and updates
the project metrics JSON file.
"""

import json
import os
import sys
import subprocess
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MetricsCollector:
    """Collects and aggregates project metrics from multiple sources."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.metrics_file = self.repo_root / ".github" / "project-metrics.json"
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/analog-pde-solver-sim")
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        metrics = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "code_quality": self._collect_code_quality_metrics(),
            "performance": self._collect_performance_metrics(),
            "security": self._collect_security_metrics(),
            "development": self._collect_development_metrics(),
            "research_specific": self._collect_research_metrics(),
        }
        
        return metrics
    
    def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("📊 Collecting code quality metrics...")
        
        metrics = {}
        
        try:
            # Test coverage (from coverage.py)
            coverage_result = subprocess.run(
                ["python", "-m", "coverage", "report", "--format=json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if coverage_result.returncode == 0:
                coverage_data = json.loads(coverage_result.stdout)
                metrics["test_coverage"] = {
                    "current": round(coverage_data["totals"]["percent_covered"], 1),
                    "lines_covered": coverage_data["totals"]["covered_lines"],
                    "lines_total": coverage_data["totals"]["num_statements"],
                    "last_measured": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            print(f"Warning: Could not collect coverage metrics: {e}")
        
        try:
            # Code complexity (using radon)
            complexity_result = subprocess.run(
                ["python", "-m", "radon", "cc", "analog_pde_solver", "--json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if complexity_result.returncode == 0:
                complexity_data = json.loads(complexity_result.stdout)
                total_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item["type"] in ["function", "method"]:
                            total_complexity += item["complexity"]
                            total_functions += 1
                
                if total_functions > 0:
                    metrics["cyclomatic_complexity"] = {
                        "current": round(total_complexity / total_functions, 1),
                        "total_functions": total_functions,
                        "last_measured": datetime.now(timezone.utc).isoformat()
                    }
        except Exception as e:
            print(f"Warning: Could not collect complexity metrics: {e}")
        
        try:
            # Lines of code
            loc_result = subprocess.run(
                ["find", "analog_pde_solver", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if loc_result.returncode == 0:
                lines = loc_result.stdout.strip().split('\n')
                if lines:
                    total_lines = int(lines[-1].split()[0])
                    metrics["lines_of_code"] = {
                        "current": total_lines,
                        "last_measured": datetime.now(timezone.utc).isoformat()
                    }
        except Exception as e:
            print(f"Warning: Could not collect LOC metrics: {e}")
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        print("⚡ Collecting performance metrics...")
        
        metrics = {}
        
        try:
            # Check for recent benchmark results
            benchmark_dir = self.repo_root / "benchmark_results"
            if benchmark_dir.exists():
                benchmark_files = list(benchmark_dir.glob("*.json"))
                if benchmark_files:
                    # Get most recent benchmark file
                    latest_benchmark = max(benchmark_files, key=lambda f: f.stat().st_mtime)
                    
                    with open(latest_benchmark, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    metrics["solver_performance"] = {
                        "last_benchmark": latest_benchmark.name,
                        "timestamp": datetime.fromtimestamp(latest_benchmark.stat().st_mtime).isoformat(),
                        "results": benchmark_data
                    }
        except Exception as e:
            print(f"Warning: Could not collect performance metrics: {e}")
        
        try:
            # Memory usage during tests
            memory_result = subprocess.run(
                ["python", "-m", "memory_profiler", "-c", 
                 "import analog_pde_solver; print('Memory profiling complete')"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if memory_result.returncode == 0:
                # Parse memory usage from output
                lines = memory_result.stdout.split('\n')
                memory_lines = [line for line in lines if 'MiB' in line]
                if memory_lines:
                    # Extract peak memory usage
                    peak_memory = max([float(line.split()[0]) for line in memory_lines if line.split()[0].replace('.', '').isdigit()])
                    metrics["memory_usage"] = {
                        "peak_memory_mb": round(peak_memory, 1),
                        "last_measured": datetime.now(timezone.utc).isoformat()
                    }
        except Exception as e:
            print(f"Warning: Could not collect memory metrics: {e}")
        
        return metrics
    
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        print("🛡️  Collecting security metrics...")
        
        metrics = {}
        
        try:
            # Run bandit security check
            bandit_result = subprocess.run(
                ["python", "-m", "bandit", "-r", "analog_pde_solver", "-f", "json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if bandit_result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                bandit_data = json.loads(bandit_result.stdout)
                
                severity_counts = {}
                for result in bandit_data.get("results", []):
                    severity = result["issue_severity"].lower()
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                metrics["sast_scan"] = {
                    "findings": severity_counts,
                    "total_findings": len(bandit_data.get("results", [])),
                    "last_scan": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            print(f"Warning: Could not collect security metrics: {e}")
        
        try:
            # Check for known vulnerabilities with safety
            safety_result = subprocess.run(
                ["python", "-m", "safety", "check", "--json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if safety_result.returncode in [0, 64]:  # 0 = no vulns, 64 = vulns found
                if safety_result.stdout.strip().startswith('['):
                    safety_data = json.loads(safety_result.stdout)
                    
                    vulnerability_counts = {}
                    for vuln in safety_data:
                        severity = vuln.get("vulnerability_id", "unknown")
                        vulnerability_counts[severity] = vulnerability_counts.get(severity, 0) + 1
                    
                    metrics["dependency_vulnerabilities"] = {
                        "findings": vulnerability_counts,
                        "total_vulnerabilities": len(safety_data),
                        "last_scan": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    metrics["dependency_vulnerabilities"] = {
                        "total_vulnerabilities": 0,
                        "last_scan": datetime.now(timezone.utc).isoformat()
                    }
        except Exception as e:
            print(f"Warning: Could not collect vulnerability metrics: {e}")
        
        return metrics
    
    def _collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development velocity and quality metrics."""
        print("🚀 Collecting development metrics...")
        
        metrics = {}
        
        if self.github_token:
            try:
                # GitHub API calls
                headers = {
                    "Authorization": f"token {self.github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
                
                # Get recent commits (last 30 days)
                commits_url = f"https://api.github.com/repos/{self.repo_name}/commits"
                since_date = (datetime.now() - timedelta(days=30)).isoformat()
                commits_response = requests.get(
                    f"{commits_url}?since={since_date}", 
                    headers=headers, 
                    timeout=10
                )
                
                if commits_response.status_code == 200:
                    commits_data = commits_response.json()
                    metrics["commits_last_30_days"] = len(commits_data)
                
                # Get pull requests
                prs_url = f"https://api.github.com/repos/{self.repo_name}/pulls"
                prs_response = requests.get(
                    f"{prs_url}?state=all&per_page=100", 
                    headers=headers, 
                    timeout=10
                )
                
                if prs_response.status_code == 200:
                    prs_data = prs_response.json()
                    recent_prs = [
                        pr for pr in prs_data 
                        if datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00')) > 
                           datetime.now(timezone.utc) - timedelta(days=30)
                    ]
                    metrics["pull_requests_last_30_days"] = len(recent_prs)
                
                # Get issues
                issues_url = f"https://api.github.com/repos/{self.repo_name}/issues"
                issues_response = requests.get(
                    f"{issues_url}?state=closed&per_page=100", 
                    headers=headers, 
                    timeout=10
                )
                
                if issues_response.status_code == 200:
                    issues_data = issues_response.json()
                    recent_issues = [
                        issue for issue in issues_data 
                        if issue.get("closed_at") and 
                           datetime.fromisoformat(issue["closed_at"].replace('Z', '+00:00')) > 
                           datetime.now(timezone.utc) - timedelta(days=30)
                    ]
                    metrics["issues_closed_last_30_days"] = len(recent_issues)
                
            except Exception as e:
                print(f"Warning: Could not collect GitHub metrics: {e}")
        
        try:
            # Git metrics
            git_stats = subprocess.run(
                ["git", "log", "--oneline", "--since=30.days.ago"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if git_stats.returncode == 0:
                commit_count = len(git_stats.stdout.strip().split('\n')) if git_stats.stdout.strip() else 0
                metrics["local_commits_last_30_days"] = commit_count
        except Exception as e:
            print(f"Warning: Could not collect git metrics: {e}")
        
        return metrics
    
    def _collect_research_metrics(self) -> Dict[str, Any]:
        """Collect research-specific metrics."""
        print("🔬 Collecting research metrics...")
        
        metrics = {}
        
        try:
            # Count algorithm implementations
            algorithm_files = list((self.repo_root / "analog_pde_solver" / "core").glob("*.py"))
            metrics["algorithm_count"] = len(algorithm_files)
            
            # Count example problems
            if (self.repo_root / "examples").exists():
                example_files = list((self.repo_root / "examples").glob("*.py"))
                metrics["example_problems"] = len(example_files)
            
            # Check for research outputs
            docs_dir = self.repo_root / "docs"
            if docs_dir.exists():
                research_files = list(docs_dir.glob("**/research*.md")) + list(docs_dir.glob("**/paper*.md"))
                metrics["research_documents"] = len(research_files)
            
            # Count test cases (research validation)
            test_dirs = ["tests/unit", "tests/integration", "tests/hardware"]
            total_tests = 0
            for test_dir in test_dirs:
                test_path = self.repo_root / test_dir
                if test_path.exists():
                    test_files = list(test_path.glob("test_*.py"))
                    total_tests += len(test_files)
            
            metrics["validation_tests"] = total_tests
            
        except Exception as e:
            print(f"Warning: Could not collect research metrics: {e}")
        
        return metrics
    
    def update_metrics_file(self, new_metrics: Dict[str, Any]) -> None:
        """Update the project metrics JSON file."""
        print("💾 Updating metrics file...")
        
        try:
            # Load existing metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            else:
                existing_metrics = {
                    "project": {
                        "name": "analog-pde-solver-sim",
                        "version": "0.3.0"
                    },
                    "metrics": {}
                }
            
            # Update with new metrics
            for category, data in new_metrics.items():
                if category != "collection_timestamp":
                    if "metrics" not in existing_metrics:
                        existing_metrics["metrics"] = {}
                    existing_metrics["metrics"][category] = data
            
            # Add timestamp
            existing_metrics["last_updated"] = new_metrics["collection_timestamp"]
            
            # Write updated metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            
            print(f"✅ Metrics updated in {self.metrics_file}")
            
        except Exception as e:
            print(f"❌ Error updating metrics file: {e}")
            raise
    
    def generate_report(self, output_format: str = "text") -> str:
        """Generate a metrics report."""
        if not self.metrics_file.exists():
            return "No metrics data available."
        
        with open(self.metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        if output_format == "json":
            return json.dumps(metrics_data, indent=2)
        
        # Text report
        report = []
        report.append("📊 PROJECT METRICS REPORT")
        report.append("=" * 50)
        report.append(f"Project: {metrics_data.get('project', {}).get('name', 'Unknown')}")
        report.append(f"Last Updated: {metrics_data.get('last_updated', 'Unknown')}")
        report.append("")
        
        metrics = metrics_data.get('metrics', {})
        
        for category, data in metrics.items():
            report.append(f"📈 {category.upper().replace('_', ' ')}")
            report.append("-" * 30)
            
            for metric, value in data.items():
                if isinstance(value, dict):
                    if "current" in value:
                        target = value.get("target", "N/A")
                        trend = value.get("trend", "unknown")
                        unit = value.get("unit", "")
                        report.append(f"  {metric}: {value['current']} {unit} (target: {target}, trend: {trend})")
                    else:
                        report.append(f"  {metric}: {json.dumps(value, indent=4)}")
                else:
                    report.append(f"  {metric}: {value}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                       help="Output format for report")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate report, don't collect new metrics")
    parser.add_argument("--repo-root", default=".",
                       help="Repository root directory")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.repo_root)
    
    if not args.report_only:
        # Collect new metrics
        new_metrics = collector.collect_all_metrics()
        collector.update_metrics_file(new_metrics)
    
    # Generate report
    report = collector.generate_report(args.output)
    print("\n" + report)


if __name__ == "__main__":
    from datetime import timedelta
    main()