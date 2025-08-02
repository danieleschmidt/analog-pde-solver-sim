#!/usr/bin/env python3
"""
Automated dependency update system for Analog PDE Solver.

This script checks for dependency updates, tests compatibility,
and creates pull requests for safe updates.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import requests


class DependencyUpdater:
    """Automated dependency update manager."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/analog-pde-solver-sim")
        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        
    def check_outdated_dependencies(self) -> List[Dict[str, str]]:
        """Check for outdated Python dependencies."""
        print("🔍 Checking for outdated dependencies...")
        
        outdated_deps = []
        
        try:
            # Check pip outdated packages
            result = subprocess.run(
                ["python", "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.returncode == 0:
                outdated_data = json.loads(result.stdout)
                
                for dep in outdated_data:
                    outdated_deps.append({
                        "name": dep["name"],
                        "current_version": dep["version"],
                        "latest_version": dep["latest_version"],
                        "type": "pip"
                    })
                    
            print(f"Found {len(outdated_deps)} outdated dependencies")
            
        except Exception as e:
            print(f"Error checking dependencies: {e}")
        
        return outdated_deps
    
    def check_security_vulnerabilities(self) -> List[Dict[str, str]]:
        """Check for security vulnerabilities in dependencies."""
        print("🛡️  Checking for security vulnerabilities...")
        
        vulnerabilities = []
        
        try:
            # Run pip-audit
            result = subprocess.run(
                ["python", "-m", "pip_audit", "--format=json"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.returncode in [0, 1]:  # 0 = no vulns, 1 = vulns found
                if result.stdout.strip():
                    vuln_data = json.loads(result.stdout)
                    
                    for vuln in vuln_data.get("vulnerabilities", []):
                        vulnerabilities.append({
                            "name": vuln["package"],
                            "current_version": vuln["installed_version"],
                            "vulnerability_id": vuln["id"],
                            "severity": vuln.get("severity", "unknown"),
                            "fixed_versions": vuln.get("fixed_versions", []),
                            "type": "security"
                        })
                        
            print(f"Found {len(vulnerabilities)} security vulnerabilities")
            
        except Exception as e:
            print(f"Error checking vulnerabilities: {e}")
        
        return vulnerabilities
    
    def prioritize_updates(self, updates: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prioritize updates based on security and compatibility."""
        print("📊 Prioritizing updates...")
        
        # Priority order: security vulnerabilities, major versions, minor versions, patches
        def priority_key(update):
            if update["type"] == "security":
                severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
                return (0, severity_order.get(update.get("severity", "unknown"), 4))
            
            # Parse version numbers for regular updates
            try:
                current_parts = [int(x) for x in update["current_version"].split(".")]
                latest_parts = [int(x) for x in update["latest_version"].split(".")]
                
                # Major version change
                if current_parts[0] != latest_parts[0]:
                    return (3, update["name"])
                # Minor version change
                elif len(current_parts) > 1 and len(latest_parts) > 1 and current_parts[1] != latest_parts[1]:
                    return (2, update["name"])
                # Patch version change
                else:
                    return (1, update["name"])
            except (ValueError, IndexError):
                return (4, update["name"])
        
        prioritized = sorted(updates, key=priority_key)
        
        for i, update in enumerate(prioritized):
            update["priority"] = i + 1
        
        return prioritized
    
    def test_update(self, package_name: str, new_version: str) -> bool:
        """Test a dependency update in isolation."""
        print(f"🧪 Testing update: {package_name} → {new_version}")
        
        # Create temporary virtual environment
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_venv"
            
            try:
                # Create virtual environment
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True, cwd=self.repo_root
                )
                
                # Get pip path
                if os.name == 'nt':  # Windows
                    pip_path = venv_path / "Scripts" / "pip"
                    python_path = venv_path / "Scripts" / "python"
                else:  # Unix
                    pip_path = venv_path / "bin" / "pip"
                    python_path = venv_path / "bin" / "python"
                
                # Install current package with dependencies
                subprocess.run(
                    [str(pip_path), "install", "-e", ".[dev]"],
                    check=True, cwd=self.repo_root
                )
                
                # Update specific package
                subprocess.run(
                    [str(pip_path), "install", f"{package_name}=={new_version}"],
                    check=True, cwd=self.repo_root
                )
                
                # Run basic import test
                import_test = subprocess.run(
                    [str(python_path), "-c", "import analog_pde_solver; print('Import successful')"],
                    capture_output=True, text=True, cwd=self.repo_root
                )
                
                if import_test.returncode != 0:
                    print(f"❌ Import test failed for {package_name} {new_version}")
                    return False
                
                # Run quick tests
                test_result = subprocess.run(
                    [str(python_path), "-m", "pytest", "tests/unit/", "-x", "--tb=no"],
                    capture_output=True, text=True, cwd=self.repo_root
                )
                
                if test_result.returncode != 0:
                    print(f"❌ Tests failed for {package_name} {new_version}")
                    return False
                
                print(f"✅ Update test passed for {package_name} {new_version}")
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Update test failed for {package_name} {new_version}: {e}")
                return False
    
    def update_requirements_file(self, package_name: str, new_version: str) -> bool:
        """Update requirements.txt with new version."""
        requirements_file = self.repo_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("⚠️  requirements.txt not found")
            return False
        
        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(package_name + "==") or line.strip().startswith(package_name + ">="):
                    lines[i] = f"{package_name}>={new_version}\n"
                    updated = True
                    break
                elif line.strip().startswith(package_name) and ("==" in line or ">=" in line):
                    lines[i] = f"{package_name}>={new_version}\n"
                    updated = True
                    break
            
            if updated:
                with open(requirements_file, 'w') as f:
                    f.writelines(lines)
                print(f"✅ Updated {package_name} in requirements.txt")
                return True
            else:
                print(f"⚠️  {package_name} not found in requirements.txt")
                return False
                
        except Exception as e:
            print(f"❌ Error updating requirements.txt: {e}")
            return False
    
    def create_update_branch(self, updates: List[Dict[str, str]]) -> str:
        """Create a new branch for dependency updates."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"auto-update-dependencies-{timestamp}"
        
        try:
            # Create and checkout new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                check=True, cwd=self.repo_root
            )
            print(f"✅ Created branch: {branch_name}")
            return branch_name
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create branch: {e}")
            raise
    
    def commit_updates(self, updates: List[Dict[str, str]], branch_name: str) -> bool:
        """Commit dependency updates."""
        try:
            # Add changed files
            subprocess.run(
                ["git", "add", "requirements.txt"],
                check=True, cwd=self.repo_root
            )
            
            # Create commit message
            if len(updates) == 1:
                update = updates[0]
                commit_msg = f"chore: update {update['name']} to {update['latest_version']}"
                if update["type"] == "security":
                    commit_msg += f" (security fix for {update['vulnerability_id']})"
            else:
                commit_msg = f"chore: update {len(updates)} dependencies"
                security_updates = [u for u in updates if u["type"] == "security"]
                if security_updates:
                    commit_msg += f" ({len(security_updates)} security fixes)"
            
            commit_msg += "\n\nAuto-generated dependency update\n"
            commit_msg += "🤖 Generated with [Claude Code](https://claude.ai/code)\n\n"
            commit_msg += "Co-Authored-By: Claude <noreply@anthropic.com>"
            
            # Commit changes
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                check=True, cwd=self.repo_root
            )
            
            print(f"✅ Committed updates to branch: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to commit updates: {e}")
            return False
    
    def create_pull_request(self, updates: List[Dict[str, str]], branch_name: str) -> Optional[str]:
        """Create a pull request for dependency updates."""
        if not self.github_token:
            print("⚠️  No GitHub token available, cannot create PR")
            return None
        
        try:
            # Push branch
            subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                check=True, cwd=self.repo_root
            )
            
            # Create PR via GitHub API
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Generate PR title and body
            if len(updates) == 1:
                update = updates[0]
                title = f"chore: update {update['name']} to {update['latest_version']}"
                if update["type"] == "security":
                    title += " (security fix)"
            else:
                title = f"chore: update {len(updates)} dependencies"
                security_count = len([u for u in updates if u["type"] == "security"])
                if security_count > 0:
                    title += f" ({security_count} security fixes)"
            
            body_lines = ["## Dependency Updates\n"]
            body_lines.append("This PR updates the following dependencies:\n")
            
            for update in updates:
                if update["type"] == "security":
                    body_lines.append(f"- **{update['name']}**: {update['current_version']} → {update['latest_version']} (🛡️ Security fix for {update.get('vulnerability_id', 'unknown')})")
                else:
                    body_lines.append(f"- **{update['name']}**: {update['current_version']} → {update['latest_version']}")
            
            body_lines.append("\n## Testing")
            body_lines.append("- [x] All updates tested in isolation")
            body_lines.append("- [x] Import tests passed")
            body_lines.append("- [x] Unit tests passed")
            body_lines.append("\n## Notes")
            body_lines.append("This PR was automatically generated by the dependency update system.")
            body_lines.append("Please review the changes and run additional tests if needed.")
            body_lines.append("\n🤖 Generated with [Claude Code](https://claude.ai/code)")
            
            pr_data = {
                "title": title,
                "head": branch_name,
                "base": "main",
                "body": "\n".join(body_lines)
            }
            
            pr_url = f"https://api.github.com/repos/{self.repo_name}/pulls"
            response = requests.post(pr_url, headers=headers, json=pr_data, timeout=30)
            
            if response.status_code == 201:
                pr_info = response.json()
                print(f"✅ Created pull request: {pr_info['html_url']}")
                return pr_info['html_url']
            else:
                print(f"❌ Failed to create PR: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to create pull request: {e}")
            return None
    
    def run_update_process(self) -> None:
        """Run the complete dependency update process."""
        print("🚀 Starting automated dependency update process...")
        
        if self.dry_run:
            print("🧪 Running in DRY RUN mode - no changes will be made")
        
        # Check for updates
        outdated = self.check_outdated_dependencies()
        vulnerabilities = self.check_security_vulnerabilities()
        
        all_updates = outdated + vulnerabilities
        
        if not all_updates:
            print("✅ All dependencies are up to date!")
            return
        
        # Prioritize updates
        prioritized_updates = self.prioritize_updates(all_updates)
        
        print(f"\n📋 Found {len(prioritized_updates)} potential updates:")
        for update in prioritized_updates[:10]:  # Show top 10
            status = "🛡️ SECURITY" if update["type"] == "security" else "📦 UPDATE"
            print(f"  {status} {update['name']}: {update['current_version']} → {update['latest_version']}")
        
        if len(prioritized_updates) > 10:
            print(f"  ... and {len(prioritized_updates) - 10} more")
        
        # Process high-priority updates first
        successful_updates = []
        failed_updates = []
        
        # Limit to top 5 updates per run to avoid overwhelming PRs
        updates_to_process = prioritized_updates[:5]
        
        for update in updates_to_process:
            print(f"\n🔄 Processing {update['name']}...")
            
            if self.dry_run:
                print(f"  Would test update: {update['name']} → {update['latest_version']}")
                successful_updates.append(update)
                continue
            
            # Test the update
            if self.test_update(update['name'], update['latest_version']):
                # Update requirements file
                if self.update_requirements_file(update['name'], update['latest_version']):
                    successful_updates.append(update)
                else:
                    failed_updates.append(update)
            else:
                failed_updates.append(update)
        
        if successful_updates and not self.dry_run:
            print(f"\n✅ Successfully processed {len(successful_updates)} updates")
            
            # Create branch and commit
            branch_name = self.create_update_branch(successful_updates)
            
            if self.commit_updates(successful_updates, branch_name):
                # Create pull request
                pr_url = self.create_pull_request(successful_updates, branch_name)
                
                if pr_url:
                    print(f"\n🎉 Dependency update complete!")
                    print(f"   Pull request: {pr_url}")
                else:
                    print("\n⚠️  Updates committed but PR creation failed")
            else:
                print("\n❌ Failed to commit updates")
        
        if failed_updates:
            print(f"\n⚠️  {len(failed_updates)} updates failed:")
            for update in failed_updates:
                print(f"  ❌ {update['name']}: {update['current_version']} → {update['latest_version']}")
        
        print("\n📊 Update Summary:")
        print(f"  Successful: {len(successful_updates)}")
        print(f"  Failed: {len(failed_updates)}")
        print(f"  Remaining: {len(prioritized_updates) - len(updates_to_process)}")


def main():
    """Main entry point."""
    updater = DependencyUpdater()
    updater.run_update_process()


if __name__ == "__main__":
    main()