#!/usr/bin/env python3
"""Security audit script for analog PDE solver codebase."""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
import hashlib
import subprocess


class SecurityAuditor:
    """Comprehensive security auditor for the analog PDE solver."""
    
    def __init__(self, project_root: Path):
        """Initialize security auditor.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.findings: List[Dict[str, Any]] = []
        self.stats = {
            'files_scanned': 0,
            'lines_scanned': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0
        }
        
        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'key\s*=\s*["\'][A-Za-z0-9+/=]{20,}["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'][^"\']*%s[^"\']*["\']',
                r'cursor\.execute\s*\([^)]*%',
                r'query\s*=.*\+.*input'
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\('
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\.\/',
                r'file\s*=.*\.\.\/',
                r'path.*\.\.\/'
            ],
            'weak_crypto': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'random\.random\s*\(',
                r'DES\s*\(',
                r'RC4\s*\('
            ]
        }
    
    def audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit.
        
        Returns:
            Audit results dictionary
        """
        print("Starting security audit...")
        
        # Scan Python files
        self._scan_python_files()
        
        # Check dependencies
        self._check_dependencies()
        
        # Check file permissions
        self._check_file_permissions()
        
        # Check for secrets in git history (if available)
        self._check_git_history()
        
        # Check configuration files
        self._check_configuration_files()
        
        # Generate report
        return self._generate_report()
    
    def _scan_python_files(self):
        """Scan Python files for security vulnerabilities."""
        print("Scanning Python files...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.stats['files_scanned'] += 1
                self.stats['lines_scanned'] += len(content.split('\\n'))
                
                # Check for security patterns
                self._check_security_patterns(py_file, content)
                
                # Parse AST for additional checks
                self._check_ast_security(py_file, content)
                
            except Exception as e:
                self._add_finding(
                    'file_read_error',
                    'low',
                    f"Could not read file {py_file}: {e}",
                    str(py_file)
                )
    
    def _check_security_patterns(self, file_path: Path, content: str):
        """Check file content for security patterns."""
        lines = content.split('\\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_no, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = self._get_pattern_severity(category)
                        self._add_finding(
                            category,
                            severity,
                            f"Potential {category.replace('_', ' ')} detected",
                            str(file_path),
                            line_no,
                            line.strip()
                        )
    
    def _check_ast_security(self, file_path: Path, content: str):
        """Check Python AST for security issues."""
        try:
            tree = ast.parse(content)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self, auditor, file_path):
                    self.auditor = auditor
                    self.file_path = file_path
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Check for eval/exec
                        if func_name in ['eval', 'exec']:
                            self.auditor._add_finding(
                                'dangerous_functions',
                                'high',
                                f"Use of dangerous function: {func_name}",
                                str(self.file_path),
                                getattr(node, 'lineno', 0)
                            )
                        
                        # Check for pickle usage
                        elif func_name in ['pickle', 'loads', 'load'] and self._is_pickle_context(node):
                            self.auditor._add_finding(
                                'pickle_usage',
                                'medium',
                                "Pickle usage detected - can execute arbitrary code",
                                str(self.file_path),
                                getattr(node, 'lineno', 0)
                            )
                    
                    self.generic_visit(node)
                
                def _is_pickle_context(self, node):
                    """Check if this is a pickle-related call."""
                    # Simplified check - in practice, would be more sophisticated
                    return True
                
                def visit_Import(self, node):
                    # Check for imports of dangerous modules
                    dangerous_modules = ['pickle', 'marshal', 'shelve']
                    
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            self.auditor._add_finding(
                                'dangerous_imports',
                                'low',
                                f"Import of potentially dangerous module: {alias.name}",
                                str(self.file_path),
                                getattr(node, 'lineno', 0)
                            )
                    
                    self.generic_visit(node)
            
            visitor = SecurityVisitor(self, file_path)
            visitor.visit(tree)
            
        except SyntaxError:
            # File has syntax errors, skip AST analysis
            pass
        except Exception as e:
            self._add_finding(
                'ast_parse_error',
                'low',
                f"Could not parse AST for {file_path}: {e}",
                str(file_path)
            )
    
    def _check_dependencies(self):
        """Check dependencies for known vulnerabilities."""
        print("Checking dependencies...")
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    requirements = f.read()
                
                # Look for known vulnerable packages (simplified check)
                vulnerable_patterns = [
                    r'pillow\s*[<>=]*\s*[0-8]\.',  # Old Pillow versions
                    r'requests\s*[<>=]*\s*2\.[0-19]\.',  # Old requests
                    r'flask\s*[<>=]*\s*[0-1]\.',  # Very old Flask
                ]
                
                for pattern in vulnerable_patterns:
                    if re.search(pattern, requirements, re.IGNORECASE):
                        self._add_finding(
                            'vulnerable_dependency',
                            'medium',
                            f"Potentially vulnerable dependency detected: {pattern}",
                            str(req_file)
                        )
                        
            except Exception as e:
                self._add_finding(
                    'dependency_check_error',
                    'low',
                    f"Could not check requirements.txt: {e}",
                    str(req_file)
                )
        
        # Check pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            self._check_pyproject_security(pyproject_file)
    
    def _check_pyproject_security(self, pyproject_file: Path):
        """Check pyproject.toml for security issues."""
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Check for insecure configurations
            if 'index-url' in content and 'http://' in content:
                self._add_finding(
                    'insecure_index',
                    'medium',
                    "HTTP index URL found (should use HTTPS)",
                    str(pyproject_file)
                )
            
        except Exception as e:
            self._add_finding(
                'pyproject_check_error',
                'low',
                f"Could not check pyproject.toml: {e}",
                str(pyproject_file)
            )
    
    def _check_file_permissions(self):
        """Check for overly permissive file permissions."""
        print("Checking file permissions...")
        
        # Check for world-writable files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = stat.st_mode
                    
                    # Check if world-writable (002 permission)
                    if mode & 0o002:
                        self._add_finding(
                            'file_permissions',
                            'medium',
                            "World-writable file detected",
                            str(file_path)
                        )
                    
                    # Check for executable Python files (might be intentional)
                    if file_path.suffix == '.py' and mode & 0o111:
                        # This is often fine, just note it
                        self._add_finding(
                            'executable_python',
                            'low',
                            "Executable Python file (verify this is intentional)",
                            str(file_path)
                        )
                        
                except OSError:
                    # Permission denied or file doesn't exist
                    pass
    
    def _check_git_history(self):
        """Check git history for accidentally committed secrets."""
        print("Checking git history...")
        
        if not (self.project_root / ".git").exists():
            return
        
        try:
            # Use git log to search for potential secrets
            result = subprocess.run([
                'git', 'log', '--all', '--full-history', '--grep=password',
                '--grep=secret', '--grep=key', '--oneline'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout.strip():
                self._add_finding(
                    'git_history_secrets',
                    'medium',
                    "Potential secrets found in git commit messages",
                    "git_history",
                    details=result.stdout.strip()
                )
            
            # Check for large files that might contain secrets
            result = subprocess.run([
                'git', 'ls-files', '-z'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                files = result.stdout.split('\\0')
                for file_name in files:
                    if file_name and self._is_suspicious_filename(file_name):
                        file_path = self.project_root / file_name
                        if file_path.exists() and file_path.stat().st_size > 100000:  # > 100KB
                            self._add_finding(
                                'large_suspicious_file',
                                'low',
                                f"Large file with suspicious name: {file_name}",
                                file_name
                            )
            
        except (subprocess.SubprocessError, FileNotFoundError):
            # Git not available or not in a git repo
            pass
    
    def _check_configuration_files(self):
        """Check configuration files for security issues."""
        print("Checking configuration files...")
        
        config_patterns = [
            "*.conf", "*.cfg", "*.ini", "*.yaml", "*.yml", "*.json", "*.env"
        ]
        
        for pattern in config_patterns:
            for config_file in self.project_root.rglob(pattern):
                if self._should_skip_file(config_file):
                    continue
                
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for hardcoded credentials
                    credential_patterns = [
                        r'password\s*[:=]\s*[^\\s]+',
                        r'secret\s*[:=]\s*[^\\s]+',
                        r'token\s*[:=]\s*[^\\s]+',
                        r'api[_-]?key\s*[:=]\s*[^\\s]+',
                    ]
                    
                    for pattern in credential_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            self._add_finding(
                                'config_credentials',
                                'high',
                                f"Hardcoded credentials in config file",
                                str(config_file)
                            )
                            break
                    
                    # Check for debug mode enabled
                    if re.search(r'debug\s*[:=]\s*true', content, re.IGNORECASE):
                        self._add_finding(
                            'debug_mode',
                            'low',
                            "Debug mode enabled in configuration",
                            str(config_file)
                        )
                
                except Exception as e:
                    self._add_finding(
                        'config_read_error',
                        'low',
                        f"Could not read config file {config_file}: {e}",
                        str(config_file)
                    )
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning."""
        skip_patterns = [
            '*/venv/*', '*/.git/*', '*/__pycache__/*', '*/node_modules/*',
            '*.pyc', '*.pyo', '*.egg-info/*', '*/.pytest_cache/*',
            '*/.mypy_cache/*', '*/build/*', '*/dist/*'
        ]
        
        file_str = str(file_path)
        for pattern in skip_patterns:
            if Path(file_str).match(pattern):
                return True
        
        return False
    
    def _is_suspicious_filename(self, filename: str) -> bool:
        """Check if filename is suspicious."""
        suspicious_patterns = [
            'password', 'secret', 'key', 'token', 'credential',
            'backup', 'dump', 'db', 'database'
        ]
        
        filename_lower = filename.lower()
        return any(pattern in filename_lower for pattern in suspicious_patterns)
    
    def _get_pattern_severity(self, category: str) -> str:
        """Get severity level for a pattern category."""
        severity_map = {
            'hardcoded_secrets': 'high',
            'sql_injection': 'high',
            'command_injection': 'high',
            'path_traversal': 'medium',
            'weak_crypto': 'medium'
        }
        return severity_map.get(category, 'low')
    
    def _add_finding(
        self,
        category: str,
        severity: str,
        description: str,
        file_path: str,
        line_no: int = None,
        details: str = None
    ):
        """Add a security finding."""
        finding = {
            'category': category,
            'severity': severity,
            'description': description,
            'file_path': file_path,
            'line_number': line_no,
            'details': details
        }
        
        self.findings.append(finding)
        self.stats[f'{severity}_risk'] += 1
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Sort findings by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_findings = sorted(
            self.findings,
            key=lambda x: (severity_order[x['severity']], x['category'])
        )
        
        report = {
            'timestamp': self._get_timestamp(),
            'project_root': str(self.project_root),
            'statistics': self.stats,
            'findings': sorted_findings,
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of findings."""
        total_findings = len(self.findings)
        
        # Categorize findings
        categories = {}
        for finding in self.findings:
            category = finding['category']
            if category not in categories:
                categories[category] = {'count': 0, 'severity_breakdown': {}}
            categories[category]['count'] += 1
            
            severity = finding['severity']
            if severity not in categories[category]['severity_breakdown']:
                categories[category]['severity_breakdown'][severity] = 0
            categories[category]['severity_breakdown'][severity] += 1
        
        # Calculate risk score
        risk_score = (
            self.stats['high_risk'] * 10 +
            self.stats['medium_risk'] * 5 +
            self.stats['low_risk'] * 1
        )
        
        summary = {
            'total_findings': total_findings,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score, total_findings),
            'categories': categories,
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _get_risk_level(self, risk_score: int, total_findings: int) -> str:
        """Calculate overall risk level."""
        if self.stats['high_risk'] > 0:
            return 'HIGH'
        elif self.stats['medium_risk'] > 5:
            return 'HIGH'
        elif self.stats['medium_risk'] > 0 or self.stats['low_risk'] > 10:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if self.stats['high_risk'] > 0:
            recommendations.append("Address all HIGH severity findings immediately")
        
        if self.stats['medium_risk'] > 0:
            recommendations.append("Review and fix MEDIUM severity findings")
        
        # Category-specific recommendations
        categories = set(f['category'] for f in self.findings)
        
        if 'hardcoded_secrets' in categories:
            recommendations.append("Use environment variables or secure vaults for secrets")
        
        if 'weak_crypto' in categories:
            recommendations.append("Upgrade to stronger cryptographic algorithms")
        
        if 'file_permissions' in categories:
            recommendations.append("Review and fix file permissions")
        
        if 'config_credentials' in categories:
            recommendations.append("Remove hardcoded credentials from configuration files")
        
        recommendations.extend([
            "Implement automated security scanning in CI/CD pipeline",
            "Conduct regular security code reviews",
            "Keep dependencies updated",
            "Use static analysis security testing (SAST) tools"
        ])
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def print_report(report: Dict[str, Any]):
    """Print security report to console."""
    print("\\n" + "="*80)
    print("SECURITY AUDIT REPORT")
    print("="*80)
    
    print(f"\\nProject: {report['project_root']}")
    print(f"Timestamp: {report['timestamp']}")
    
    # Statistics
    stats = report['statistics']
    print(f"\\nScan Statistics:")
    print(f"  Files scanned: {stats['files_scanned']}")
    print(f"  Lines scanned: {stats['lines_scanned']:,}")
    
    # Summary
    summary = report['summary']
    print(f"\\nFindings Summary:")
    print(f"  Total findings: {summary['total_findings']}")
    print(f"  Risk score: {summary['risk_score']}")
    print(f"  Risk level: {summary['risk_level']}")
    print(f"  High risk: {stats['high_risk']}")
    print(f"  Medium risk: {stats['medium_risk']}")
    print(f"  Low risk: {stats['low_risk']}")
    
    # Findings by category
    if summary['categories']:
        print(f"\\nFindings by Category:")
        for category, info in summary['categories'].items():
            print(f"  {category}: {info['count']}")
    
    # Top findings
    if report['findings']:
        print(f"\\nTop Security Findings:")
        high_findings = [f for f in report['findings'] if f['severity'] == 'high']
        medium_findings = [f for f in report['findings'] if f['severity'] == 'medium']
        
        for finding in high_findings[:5]:  # Top 5 high severity
            print(f"  [HIGH] {finding['description']}")
            print(f"    File: {finding['file_path']}")
            if finding.get('line_number'):
                print(f"    Line: {finding['line_number']}")
            print()
        
        for finding in medium_findings[:3]:  # Top 3 medium severity
            print(f"  [MEDIUM] {finding['description']}")
            print(f"    File: {finding['file_path']}")
            print()
    
    # Recommendations
    if summary['recommendations']:
        print(f"\\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
    
    print("\\n" + "="*80)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path.cwd()
    
    if not project_root.exists():
        print(f"Error: Project root {project_root} does not exist")
        sys.exit(1)
    
    # Run security audit
    auditor = SecurityAuditor(project_root)
    report = auditor.audit()
    
    # Print report
    print_report(report)
    
    # Save detailed report
    import json
    report_file = project_root / "security_audit_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\nDetailed report saved to: {report_file}")
    
    # Exit with appropriate code
    risk_level = report['summary']['risk_level']
    if risk_level == 'HIGH':
        sys.exit(2)
    elif risk_level == 'MEDIUM':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()