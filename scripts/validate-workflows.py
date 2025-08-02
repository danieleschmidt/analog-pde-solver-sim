#!/usr/bin/env python3
"""
Workflow validation script for GitHub Actions.

This script validates that all workflow files are properly configured
and provides helpful feedback for manual setup requirements.
"""

import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re


class WorkflowValidator:
    """Validates GitHub Actions workflows and configuration."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.workflows_template_dir = self.repo_root / "docs" / "github-workflows"
        self.workflows_active_dir = self.repo_root / ".github" / "workflows"
        self.errors = []
        self.warnings = []
        self.info = []
    
    def validate_all(self) -> bool:
        """Run all validations and return True if everything is valid."""
        print("🔍 Validating GitHub Actions workflows and configuration...\n")
        
        success = True
        
        # Check if workflows are properly installed
        success &= self._check_workflow_installation()
        
        # Validate YAML syntax
        success &= self._validate_yaml_syntax()
        
        # Check workflow dependencies  
        success &= self._check_workflow_dependencies()
        
        # Validate workflow permissions
        success &= self._validate_permissions()
        
        # Check secret requirements
        success &= self._check_secret_requirements()
        
        # Validate workflow triggers
        success &= self._validate_triggers()
        
        # Check for security best practices
        success &= self._check_security_practices()
        
        # Print summary
        self._print_summary()
        
        return success
    
    def _check_workflow_installation(self) -> bool:
        """Check if workflow files are properly installed."""
        print("📋 Checking workflow installation...")
        
        if not self.workflows_active_dir.exists():
            self.errors.append("❌ .github/workflows/ directory does not exist")
            self.errors.append("   Run: mkdir -p .github/workflows")
            return False
        
        template_files = list(self.workflows_template_dir.glob("*.yml"))
        active_files = list(self.workflows_active_dir.glob("*.yml"))
        
        if not template_files:
            self.warnings.append("⚠️  No workflow templates found in docs/github-workflows/")
            return True
        
        if not active_files:
            self.errors.append("❌ No workflow files found in .github/workflows/")
            self.errors.append("   Run: cp docs/github-workflows/*.yml .github/workflows/")
            return False
        
        # Check if template files are copied
        template_names = {f.name for f in template_files}
        active_names = {f.name for f in active_files}
        
        missing_workflows = template_names - active_names
        if missing_workflows:
            self.warnings.append(f"⚠️  Missing workflows: {', '.join(missing_workflows)}")
            self.warnings.append("   Consider copying missing templates")
        
        extra_workflows = active_names - template_names
        if extra_workflows:
            self.info.append(f"ℹ️  Additional workflows found: {', '.join(extra_workflows)}")
        
        self.info.append(f"✅ Found {len(active_files)} workflow files")
        return True
    
    def _validate_yaml_syntax(self) -> bool:
        """Validate YAML syntax in all workflow files."""
        print("📝 Validating YAML syntax...")
        
        success = True
        
        for workflow_file in self.workflows_active_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    yaml.safe_load(f)
                self.info.append(f"✅ {workflow_file.name}: Valid YAML")
            except yaml.YAMLError as e:
                self.errors.append(f"❌ {workflow_file.name}: YAML syntax error")
                self.errors.append(f"   {str(e)}")
                success = False
            except Exception as e:
                self.errors.append(f"❌ {workflow_file.name}: Error reading file")
                self.errors.append(f"   {str(e)}")
                success = False
        
        return success
    
    def _check_workflow_dependencies(self) -> bool:
        """Check for workflow dependencies and action versions."""
        print("🔗 Checking workflow dependencies...")
        
        success = True
        action_versions = {}
        
        for workflow_file in self.workflows_active_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow = yaml.safe_load(f)
                
                # Extract actions used
                actions = self._extract_actions(workflow)
                for action, version in actions:
                    if action not in action_versions:
                        action_versions[action] = set()
                    action_versions[action].add(version)
                
            except Exception as e:
                self.warnings.append(f"⚠️  Could not analyze {workflow_file.name}: {e}")
        
        # Check for inconsistent action versions
        for action, versions in action_versions.items():
            if len(versions) > 1:
                self.warnings.append(f"⚠️  Inconsistent versions for {action}: {', '.join(versions)}")
        
        # Check for pinned versions
        unpinned_actions = []
        for action, versions in action_versions.items():
            for version in versions:
                if version in ['main', 'master', 'latest']:
                    unpinned_actions.append(f"{action}@{version}")
        
        if unpinned_actions:
            self.warnings.append("⚠️  Unpinned action versions (security risk):")
            for action in unpinned_actions:
                self.warnings.append(f"   {action}")
        
        self.info.append(f"✅ Found {len(action_versions)} unique actions")
        return success
    
    def _extract_actions(self, workflow: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract actions and their versions from a workflow."""
        actions = []
        
        if 'jobs' not in workflow:
            return actions
        
        for job_name, job in workflow['jobs'].items():
            if 'steps' not in job:
                continue
            
            for step in job['steps']:
                if 'uses' in step:
                    uses = step['uses']
                    if '@' in uses:
                        action, version = uses.rsplit('@', 1)
                        actions.append((action, version))
                    else:
                        actions.append((uses, 'latest'))
        
        return actions
    
    def _validate_permissions(self) -> bool:
        """Validate workflow permissions."""
        print("🔐 Validating workflow permissions...")
        
        success = True
        
        for workflow_file in self.workflows_active_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow = yaml.safe_load(f)
                
                # Check for overly broad permissions
                permissions = workflow.get('permissions', {})
                if permissions == 'write-all':
                    self.warnings.append(f"⚠️  {workflow_file.name}: Uses 'write-all' permissions")
                elif isinstance(permissions, dict):
                    broad_permissions = [k for k, v in permissions.items() if v == 'write']
                    if len(broad_permissions) > 3:
                        self.warnings.append(f"⚠️  {workflow_file.name}: Many write permissions: {broad_permissions}")
                
            except Exception as e:
                self.warnings.append(f"⚠️  Could not check permissions in {workflow_file.name}: {e}")
        
        return success
    
    def _check_secret_requirements(self) -> bool:
        """Check for required secrets in workflows."""
        print("🔑 Checking secret requirements...")
        
        required_secrets = set()
        
        for workflow_file in self.workflows_active_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                
                # Find secret references
                secret_pattern = r'\$\{\{\s*secrets\.([A-Z_]+)\s*\}\}'
                secrets_in_file = re.findall(secret_pattern, content)
                required_secrets.update(secrets_in_file)
                
                if secrets_in_file:
                    self.info.append(f"ℹ️  {workflow_file.name} requires secrets: {', '.join(secrets_in_file)}")
                
            except Exception as e:
                self.warnings.append(f"⚠️  Could not analyze secrets in {workflow_file.name}: {e}")
        
        if required_secrets:
            self.info.append("📋 Required repository secrets:")
            for secret in sorted(required_secrets):
                self.info.append(f"   - {secret}")
            self.info.append("   Configure these in: Settings → Secrets and variables → Actions")
        
        return True
    
    def _validate_triggers(self) -> bool:
        """Validate workflow triggers."""
        print("⚡ Validating workflow triggers...")
        
        success = True
        triggers_found = set()
        
        for workflow_file in self.workflows_active_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow = yaml.safe_load(f)
                
                on_config = workflow.get('on', {})
                if isinstance(on_config, str):
                    triggers_found.add(on_config)
                elif isinstance(on_config, list):
                    triggers_found.update(on_config)
                elif isinstance(on_config, dict):
                    triggers_found.update(on_config.keys())
                
            except Exception as e:
                self.warnings.append(f"⚠️  Could not analyze triggers in {workflow_file.name}: {e}")
        
        # Check for common triggers
        recommended_triggers = {'push', 'pull_request', 'schedule', 'workflow_dispatch'}
        missing_triggers = recommended_triggers - triggers_found
        
        if missing_triggers:
            self.info.append(f"ℹ️  Consider adding triggers: {', '.join(missing_triggers)}")
        
        self.info.append(f"✅ Found triggers: {', '.join(sorted(triggers_found))}")
        return success
    
    def _check_security_practices(self) -> bool:
        """Check for security best practices."""
        print("🛡️  Checking security best practices...")
        
        success = True
        
        for workflow_file in self.workflows_active_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                    workflow = yaml.safe_load(f)
                
                # Check for hardcoded secrets
                potential_secrets = re.findall(r'(api[_-]?key|password|token|secret)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', content, re.IGNORECASE)
                if potential_secrets:
                    self.warnings.append(f"⚠️  {workflow_file.name}: Potential hardcoded secrets found")
                
                # Check for pull_request_target usage
                if 'pull_request_target' in content:
                    self.warnings.append(f"⚠️  {workflow_file.name}: Uses pull_request_target (security risk)")
                
                # Check for checkout with token
                if 'actions/checkout' in content and 'token:' in content:
                    if '${{ secrets.GITHUB_TOKEN }}' not in content:
                        self.warnings.append(f"⚠️  {workflow_file.name}: Custom token in checkout (review needed)")
                
            except Exception as e:
                self.warnings.append(f"⚠️  Could not analyze security in {workflow_file.name}: {e}")
        
        return success
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.info:
            print(f"\nℹ️  INFO ({len(self.info)}):")
            for info in self.info:
                print(f"  {info}")
        
        print(f"\n📈 RESULTS:")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Info: {len(self.info)}")
        
        if not self.errors:
            print(f"\n✅ Validation passed! Workflows are properly configured.")
        else:
            print(f"\n❌ Validation failed. Please fix the errors above.")
        
        print("\n📖 For detailed setup instructions, see:")
        print("  - docs/workflows/GITHUB_WORKFLOWS_SETUP.md")
        print("  - SETUP_REQUIRED.md")
        print("="*60)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        repo_root = sys.argv[1]
    else:
        repo_root = "."
    
    validator = WorkflowValidator(repo_root)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()