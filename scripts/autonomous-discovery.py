#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes development work items.
"""

import json
import os
import re
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse


class AutonomousDiscovery:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        self.discovery_rules = self._load_discovery_rules()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        config_path = self.repo_root / ".terragon" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
        
    def _load_metrics(self) -> Dict[str, Any]:
        """Load current value metrics."""
        metrics_path = self.repo_root / ".terragon" / "value-metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {"discovered_value_items": [], "execution_history": []}
        
    def _load_discovery_rules(self) -> Dict[str, Any]:
        """Load discovery pattern rules."""
        rules_path = self.repo_root / ".terragon" / "discovery-rules.yaml"
        if rules_path.exists():
            with open(rules_path) as f:
                return yaml.safe_load(f)
        return {}
        
    def discover_technical_debt(self) -> List[Dict[str, Any]]:
        """Scan for technical debt patterns in code."""
        debt_items = []
        
        # Find TODO, FIXME, HACK patterns
        for pattern_config in self.discovery_rules.get("patterns", {}).get("technical_debt", {}).get("code_smells", []):
            pattern = pattern_config["pattern"]
            weight = pattern_config["weight"]
            
            try:
                result = subprocess.run([
                    "rg", "--json", pattern, str(self.repo_root)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            match_data = json.loads(line)
                            if match_data.get("type") == "match":
                                debt_items.append({
                                    "id": f"td-{len(debt_items)+1:03d}",
                                    "title": f"Address {pattern} in {match_data['data']['path']['text']}",
                                    "category": "technical_debt",
                                    "file": match_data["data"]["path"]["text"],
                                    "line": match_data["data"]["line_number"],
                                    "context": match_data["data"]["lines"]["text"],
                                    "weight": weight,
                                    "estimated_effort": self._estimate_effort(match_data),
                                    "discovered_date": datetime.now(timezone.utc).isoformat()
                                })
            except (subprocess.SubprocessError, json.JSONDecodeError):
                continue
                
        return debt_items
        
    def discover_security_issues(self) -> List[Dict[str, Any]]:
        """Discover security vulnerabilities and issues."""
        security_items = []
        
        # Run pip-audit for dependency vulnerabilities
        try:
            result = subprocess.run([
                "pip-audit", "--format=json", "--desc"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get("vulnerabilities", []):
                    security_items.append({
                        "id": f"sec-{len(security_items)+1:03d}",
                        "title": f"Update {vuln['package']} (CVE: {vuln.get('id', 'Unknown')})",
                        "category": "security",
                        "severity": vuln.get("fix_versions", {}).get("severity", "medium"),
                        "description": vuln.get("description", ""),
                        "estimated_effort": 1 if vuln.get("fix_versions") else 3,
                        "discovered_date": datetime.now(timezone.utc).isoformat()
                    })
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
            
        return security_items
        
    def discover_performance_issues(self) -> List[Dict[str, Any]]:
        """Discover performance optimization opportunities."""
        perf_items = []
        
        # Look for performance anti-patterns
        performance_patterns = self.discovery_rules.get("patterns", {}).get("performance_issues", {})
        
        for pattern_type, patterns in performance_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config["pattern"]
                weight = pattern_config["weight"]
                
                try:
                    result = subprocess.run([
                        "rg", "--json", pattern, str(self.repo_root), "--type", "py"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                match_data = json.loads(line)
                                if match_data.get("type") == "match":
                                    perf_items.append({
                                        "id": f"perf-{len(perf_items)+1:03d}",
                                        "title": f"Optimize {pattern_type} in {match_data['data']['path']['text']}",
                                        "category": "performance",
                                        "file": match_data["data"]["path"]["text"],
                                        "line": match_data["data"]["line_number"],
                                        "weight": weight,
                                        "estimated_effort": self._estimate_effort(match_data),
                                        "discovered_date": datetime.now(timezone.utc).isoformat()
                                    })
                except (subprocess.SubprocessError, json.JSONDecodeError):
                    continue
                    
        return perf_items
        
    def discover_documentation_gaps(self) -> List[Dict[str, Any]]:
        """Find missing documentation."""
        doc_items = []
        
        # Find functions/classes without docstrings
        try:
            result = subprocess.run([
                "rg", "--json", r"def \w+\(.*\):\s*$", str(self.repo_root), "--type", "py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        match_data = json.loads(line)
                        if match_data.get("type") == "match":
                            doc_items.append({
                                "id": f"doc-{len(doc_items)+1:03d}",
                                "title": f"Add docstring to function in {match_data['data']['path']['text']}",
                                "category": "documentation",
                                "file": match_data["data"]["path"]["text"],
                                "line": match_data["data"]["line_number"],
                                "estimated_effort": 0.5,
                                "discovered_date": datetime.now(timezone.utc).isoformat()
                            })
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
            
        return doc_items
        
    def _estimate_effort(self, match_data: Dict[str, Any]) -> float:
        """Estimate effort for a discovered item."""
        # Simple heuristic based on context and patterns
        context = match_data.get("data", {}).get("lines", {}).get("text", "")
        
        if "TODO" in context.upper():
            return 2.0
        elif "FIXME" in context.upper():
            return 4.0
        elif "HACK" in context.upper():
            return 6.0
        else:
            return 1.0
            
    def calculate_composite_score(self, item: Dict[str, Any]) -> float:
        """Calculate WSJF + ICE + Technical Debt composite score."""
        weights = self.config.get("scoring", {}).get("weights", {})
        
        # WSJF components
        business_value = item.get("business_value", 50)
        time_criticality = item.get("time_criticality", 30)
        effort = item.get("estimated_effort", 2)
        wsjf = (business_value + time_criticality) / max(effort, 0.5)
        
        # ICE components  
        impact = item.get("impact", 5)
        confidence = item.get("confidence", 7)
        ease = 10 - min(effort, 10)
        ice = impact * confidence * ease
        
        # Technical debt score
        debt_weight = item.get("weight", 0.5)
        hotspot_multiplier = self._get_hotspot_multiplier(item.get("file", ""))
        tech_debt = debt_weight * hotspot_multiplier * 100
        
        # Security boost
        security_boost = 1.0
        if item.get("category") == "security":
            security_boost = self.config.get("scoring", {}).get("thresholds", {}).get("securityBoost", 2.0)
            
        # Composite score
        composite = (
            weights.get("wsjf", 0.6) * wsjf +
            weights.get("ice", 0.1) * (ice / 100) +
            weights.get("technicalDebt", 0.2) * (tech_debt / 100) +
            weights.get("security", 0.1) * 50
        ) * security_boost
        
        return round(composite, 1)
        
    def _get_hotspot_multiplier(self, file_path: str) -> float:
        """Calculate hotspot multiplier based on file change frequency."""
        if not file_path:
            return 1.0
            
        try:
            result = subprocess.run([
                "git", "log", "--oneline", "--since=1.month.ago", "--", file_path
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            change_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            if change_count > 10:
                return 1.5  # High churn
            elif change_count > 5:
                return 1.2  # Medium churn
            else:
                return 1.0  # Low churn
        except subprocess.SubprocessError:
            return 1.0
            
    def scan_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run comprehensive value discovery scan."""
        discovered_items = {
            "technical_debt": self.discover_technical_debt(),
            "security": self.discover_security_issues(), 
            "performance": self.discover_performance_issues(),
            "documentation": self.discover_documentation_gaps()
        }
        
        # Calculate scores for all items
        all_items = []
        for category, items in discovered_items.items():
            for item in items:
                item["composite_score"] = self.calculate_composite_score(item)
                all_items.append(item)
                
        # Sort by composite score
        all_items.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return {
            "discovered_items": discovered_items,
            "prioritized_items": all_items,
            "discovery_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_items": len(all_items)
        }
        
    def save_discovery_results(self, results: Dict[str, Any]) -> None:
        """Save discovery results to output file."""
        output_path = self.repo_root / ".terragon" / "discovery-output.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Value Discovery Engine")
    parser.add_argument("--scan-all", action="store_true", help="Run comprehensive scan")
    parser.add_argument("--update-backlog", action="store_true", help="Update BACKLOG.md")
    
    args = parser.parse_args()
    
    discovery = AutonomousDiscovery()
    
    if args.scan_all or args.update_backlog:
        results = discovery.scan_all()
        discovery.save_discovery_results(results)
        
        print(f"🔍 Discovery complete: {results['total_items']} items found")
        print(f"📊 Top priority: {results['prioritized_items'][0]['title'] if results['prioritized_items'] else 'None'}")
        
        if args.update_backlog:
            # Update BACKLOG.md with new items
            print("📝 Updating BACKLOG.md...")
            

if __name__ == "__main__":
    main()