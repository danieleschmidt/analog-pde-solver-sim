"""Advanced threat detection and security monitoring."""

import time
import logging
import hashlib
import re
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    source: str
    description: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'severity': self.severity,
            'source': self.source,
            'description': self.description,
            'metadata': self.metadata
        }


class ThreatDetector:
    """Real-time threat detection for analog computing environment."""
    
    def __init__(self, log_retention_hours: int = 24):
        self.logger = logging.getLogger(__name__)
        self.log_retention_hours = log_retention_hours
        
        # Event storage
        self.security_events: deque = deque(maxlen=10000)
        self.blocked_operations: Set[str] = set()
        
        # Threat patterns
        self.malicious_patterns = {
            'code_injection': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__\s*\(',
                r'compile\s*\(',
                r'getattr\s*\(',
                r'setattr\s*\(',
                r'delattr\s*\(',
            ],
            'file_access': [
                r'open\s*\(',
                r'file\s*\(',
                r'\.read\s*\(',
                r'\.write\s*\(',
                r'os\.system',
                r'subprocess\.',
            ],
            'network_access': [
                r'socket\.',
                r'urllib\.',
                r'requests\.',
                r'http\.',
                r'ftp\.',
            ],
            'system_modification': [
                r'os\.remove',
                r'os\.unlink',
                r'shutil\.rmtree',
                r'os\.chmod',
                r'os\.chown',
            ]
        }
        
        # Rate limiting
        self.operation_counts = defaultdict(int)
        self.rate_limits = {
            'crossbar_programming': 100,  # per minute
            'solve_operations': 1000,     # per minute
            'file_access': 10,            # per minute
        }
        
        self.logger.info("Threat detector initialized")
    
    def scan_input(self, data: Any, source: str = "unknown") -> bool:
        """Scan input data for potential threats.
        
        Args:
            data: Input data to scan
            source: Source identifier for the data
            
        Returns:
            True if data is safe, False if threat detected
            
        Raises:
            SecurityError: If critical threat detected
        """
        try:
            threats_found = []
            
            # Scan string data
            if isinstance(data, str):
                threats_found.extend(self._scan_string_patterns(data))
            
            # Scan dictionary/object data
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        threats_found.extend(self._scan_string_patterns(value))
            
            # Check for anomalous data patterns
            threats_found.extend(self._detect_anomalous_patterns(data))
            
            # Log and handle threats
            for threat in threats_found:
                self._record_security_event(
                    event_type="threat_detected",
                    severity=threat['severity'],
                    source=source,
                    description=threat['description'],
                    metadata=threat
                )
                
                if threat['severity'] == 'CRITICAL':
                    raise SecurityError(f"Critical threat detected: {threat['description']}")
                elif threat['severity'] == 'HIGH':
                    self.logger.error(f"High severity threat: {threat['description']}")
                    return False
            
            return len(threats_found) == 0
            
        except Exception as e:
            self.logger.error(f"Error during threat scanning: {e}")
            return False
    
    def _scan_string_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Scan string for malicious patterns."""
        threats = []
        
        for category, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    threats.append({
                        'category': category,
                        'pattern': pattern,
                        'match': match.group(),
                        'position': match.span(),
                        'severity': self._get_threat_severity(category),
                        'description': f"Suspicious {category} pattern detected: {match.group()}"
                    })
        
        return threats
    
    def _detect_anomalous_patterns(self, data: Any) -> List[Dict[str, Any]]:
        """Detect anomalous data patterns."""
        threats = []
        
        # Check for extremely large data
        if hasattr(data, '__len__'):
            if len(str(data)) > 100000:  # 100KB limit
                threats.append({
                    'category': 'data_size',
                    'severity': 'MEDIUM',
                    'description': f"Unusually large data input: {len(str(data))} bytes",
                    'size': len(str(data))
                })
        
        # Check for binary data in unexpected contexts
        if isinstance(data, (bytes, bytearray)):
            threats.append({
                'category': 'binary_data',
                'severity': 'LOW',
                'description': "Binary data detected in text context",
                'data_type': type(data).__name__
            })
        
        return threats
    
    def _get_threat_severity(self, category: str) -> str:
        """Get severity level for threat category."""
        severity_map = {
            'code_injection': 'CRITICAL',
            'system_modification': 'CRITICAL',
            'file_access': 'HIGH',
            'network_access': 'MEDIUM',
            'data_size': 'MEDIUM',
            'binary_data': 'LOW'
        }
        return severity_map.get(category, 'MEDIUM')
    
    def check_rate_limits(self, operation: str, source: str = "unknown") -> bool:
        """Check if operation exceeds rate limits.
        
        Args:
            operation: Operation type
            source: Source identifier
            
        Returns:
            True if within limits, False if exceeded
        """
        current_time = time.time()
        minute_window = int(current_time // 60)
        key = f"{operation}_{source}_{minute_window}"
        
        self.operation_counts[key] += 1
        limit = self.rate_limits.get(operation, 1000)
        
        if self.operation_counts[key] > limit:
            self._record_security_event(
                event_type="rate_limit_exceeded",
                severity="HIGH",
                source=source,
                description=f"Rate limit exceeded for {operation}: {self.operation_counts[key]} > {limit}",
                metadata={"operation": operation, "count": self.operation_counts[key], "limit": limit}
            )
            return False
        
        return True
    
    def _record_security_event(self, event_type: str, severity: str, source: str, 
                             description: str, metadata: Dict[str, Any]):
        """Record a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source=source,
            description=description,
            metadata=metadata
        )
        
        self.security_events.append(event)
        
        # Log based on severity
        if severity == 'CRITICAL':
            self.logger.critical(f"SECURITY: {description}")
        elif severity == 'HIGH':
            self.logger.error(f"SECURITY: {description}")
        elif severity == 'MEDIUM':
            self.logger.warning(f"SECURITY: {description}")
        else:
            self.logger.info(f"SECURITY: {description}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        current_time = time.time()
        
        # Filter recent events
        recent_events = [e for e in self.security_events 
                        if current_time - e.timestamp < self.log_retention_hours * 3600]
        
        # Categorize events
        events_by_severity = defaultdict(list)
        events_by_type = defaultdict(list)
        
        for event in recent_events:
            events_by_severity[event.severity].append(event)
            events_by_type[event.event_type].append(event)
        
        return {
            "report_timestamp": current_time,
            "total_events": len(recent_events),
            "events_by_severity": {k: len(v) for k, v in events_by_severity.items()},
            "events_by_type": {k: len(v) for k, v in events_by_type.items()},
            "critical_events": [e.to_dict() for e in events_by_severity.get('CRITICAL', [])],
            "high_severity_events": [e.to_dict() for e in events_by_severity.get('HIGH', [])],
            "blocked_operations": list(self.blocked_operations),
            "rate_limit_status": self._get_rate_limit_status()
        }
    
    def _get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        status = {}
        for operation, limit in self.rate_limits.items():
            current_count = sum(
                count for key, count in self.operation_counts.items()
                if key.startswith(f"{operation}_") and key.endswith(f"_{minute_window}")
            )
            status[operation] = {
                "current": current_count,
                "limit": limit,
                "utilization": current_count / limit
            }
        
        return status


class SecurityAudit:
    """Comprehensive security audit system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def audit_codebase(self, base_path: str = ".") -> Dict[str, Any]:
        """Perform security audit of codebase."""
        audit_results = {
            "timestamp": time.time(),
            "base_path": base_path,
            "files_scanned": 0,
            "vulnerabilities": [],
            "recommendations": []
        }
        
        try:
            # Scan Python files for security issues
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        vulns = self._scan_file_security(filepath)
                        audit_results["vulnerabilities"].extend(vulns)
                        audit_results["files_scanned"] += 1
            
            # Generate recommendations
            audit_results["recommendations"] = self._generate_recommendations(
                audit_results["vulnerabilities"]
            )
            
            self.logger.info(f"Security audit completed: {len(audit_results['vulnerabilities'])} issues found")
            
        except Exception as e:
            self.logger.error(f"Security audit failed: {e}")
            audit_results["error"] = str(e)
        
        return audit_results
    
    def _scan_file_security(self, filepath: str) -> List[Dict[str, Any]]:
        """Scan individual file for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for dangerous function usage
            dangerous_patterns = {
                'eval_usage': r'eval\s*\(',
                'exec_usage': r'exec\s*\(',
                'pickle_usage': r'import\s+pickle|from\s+pickle',
                'shell_injection': r'os\.system|subprocess\.shell\s*=\s*True',
                'hardcoded_secrets': r'password\s*=\s*["\'][^"\']{8,}["\']|api_key\s*=\s*["\'][^"\']{10,}["\']'
            }
            
            for vuln_type, pattern in dangerous_patterns.items():
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    vulnerabilities.append({
                        'file': filepath,
                        'line': line_num,
                        'type': vuln_type,
                        'severity': self._get_vulnerability_severity(vuln_type),
                        'pattern': match.group(),
                        'recommendation': self._get_vulnerability_recommendation(vuln_type)
                    })
            
        except Exception as e:
            self.logger.warning(f"Could not scan file {filepath}: {e}")
        
        return vulnerabilities
    
    def _get_vulnerability_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            'eval_usage': 'CRITICAL',
            'exec_usage': 'CRITICAL', 
            'pickle_usage': 'HIGH',
            'shell_injection': 'CRITICAL',
            'hardcoded_secrets': 'HIGH'
        }
        return severity_map.get(vuln_type, 'MEDIUM')
    
    def _get_vulnerability_recommendation(self, vuln_type: str) -> str:
        """Get recommendation for vulnerability type."""
        recommendations = {
            'eval_usage': 'Replace eval() with safe alternatives like ast.literal_eval()',
            'exec_usage': 'Replace exec() with safer code execution methods',
            'pickle_usage': 'Use safer serialization like JSON for untrusted data',
            'shell_injection': 'Use subprocess with shell=False and validate inputs',
            'hardcoded_secrets': 'Move secrets to environment variables or secure storage'
        }
        return recommendations.get(vuln_type, 'Review and secure this pattern')
    
    def _generate_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on found vulnerabilities."""
        recommendations = []
        
        vuln_counts = defaultdict(int)
        for vuln in vulnerabilities:
            vuln_counts[vuln['type']] += 1
        
        if vuln_counts['eval_usage'] > 0:
            recommendations.append(
                f"Found {vuln_counts['eval_usage']} eval() usage(s). "
                "Replace with ast.literal_eval() or safer alternatives."
            )
        
        if vuln_counts['hardcoded_secrets'] > 0:
            recommendations.append(
                f"Found {vuln_counts['hardcoded_secrets']} potential hardcoded secret(s). "
                "Move to environment variables or secure key management."
            )
        
        if vuln_counts['shell_injection'] > 0:
            recommendations.append(
                f"Found {vuln_counts['shell_injection']} potential shell injection(s). "
                "Use subprocess with shell=False and input validation."
            )
        
        # General recommendations
        if len(vulnerabilities) > 0:
            recommendations.extend([
                "Implement input validation for all external data",
                "Add rate limiting for API endpoints",
                "Enable security logging and monitoring",
                "Regular security audits and dependency updates"
            ])
        
        return recommendations


class SecurityError(Exception):
    """Security-related error."""
    pass