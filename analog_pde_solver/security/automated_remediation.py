"""Automated security remediation and response system."""

import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from .threat_detector import ThreatDetector, SecurityEvent


class RemediationAction(Enum):
    """Available remediation actions."""
    BLOCK_OPERATION = "block_operation"
    RATE_LIMIT = "rate_limit"
    SANITIZE_INPUT = "sanitize_input"
    RESTART_COMPONENT = "restart_component"
    ALERT_ADMIN = "alert_admin"
    QUARANTINE = "quarantine"


@dataclass
class RemediationRule:
    """Security remediation rule."""
    threat_type: str
    severity_threshold: str
    action: RemediationAction
    parameters: Dict[str, Any]
    cooldown_seconds: int = 300  # 5 minutes


class AutomatedRemediation:
    """Automated security incident response system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threat_detector = ThreatDetector()
        
        # Remediation queue and worker
        self.remediation_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        # Remediation rules
        self.rules = self._get_default_rules()
        self.last_action_times = {}
        
        # Response handlers
        self.action_handlers = {
            RemediationAction.BLOCK_OPERATION: self._block_operation,
            RemediationAction.RATE_LIMIT: self._apply_rate_limit,
            RemediationAction.SANITIZE_INPUT: self._sanitize_input,
            RemediationAction.RESTART_COMPONENT: self._restart_component,
            RemediationAction.ALERT_ADMIN: self._alert_admin,
            RemediationAction.QUARANTINE: self._quarantine_threat,
        }
        
        # Statistics
        self.remediation_stats = {
            'total_threats': 0,
            'remediated_threats': 0,
            'blocked_operations': 0,
            'false_positives': 0
        }
    
    def start_monitoring(self):
        """Start automated remediation monitoring."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._remediation_worker, daemon=True)
        self.worker_thread.start()
        self.logger.info("Automated remediation monitoring started")
    
    def stop_monitoring(self):
        """Stop automated remediation monitoring."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("Automated remediation monitoring stopped")
    
    def process_security_event(self, event: SecurityEvent) -> bool:
        """Process security event and trigger remediation if needed.
        
        Args:
            event: Security event to process
            
        Returns:
            True if event was handled, False if ignored
        """
        try:
            self.remediation_stats['total_threats'] += 1
            
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(event)
            
            if not applicable_rules:
                self.logger.debug(f"No remediation rules for event: {event.event_type}")
                return False
            
            # Queue remediation actions
            for rule in applicable_rules:
                if self._check_cooldown(rule):
                    self.remediation_queue.put((event, rule))
                    self.logger.info(f"Queued remediation action: {rule.action.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process security event: {e}")
            return False
    
    def _find_applicable_rules(self, event: SecurityEvent) -> List[RemediationRule]:
        """Find remediation rules applicable to security event."""
        applicable = []
        
        for rule in self.rules:
            # Check threat type match
            if rule.threat_type == event.event_type or rule.threat_type == "*":
                # Check severity threshold
                if self._meets_severity_threshold(event.severity, rule.severity_threshold):
                    applicable.append(rule)
        
        return applicable
    
    def _meets_severity_threshold(self, event_severity: str, threshold: str) -> bool:
        """Check if event severity meets rule threshold."""
        severity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        event_level = severity_levels.index(event_severity)
        threshold_level = severity_levels.index(threshold)
        return event_level >= threshold_level
    
    def _check_cooldown(self, rule: RemediationRule) -> bool:
        """Check if rule is in cooldown period."""
        now = time.time()
        last_action = self.last_action_times.get(rule.action.value, 0)
        return (now - last_action) >= rule.cooldown_seconds
    
    def _remediation_worker(self):
        """Background worker for processing remediation actions."""
        while self.running:
            try:
                # Get remediation task with timeout
                event, rule = self.remediation_queue.get(timeout=1)
                
                # Execute remediation action
                success = self._execute_remediation(event, rule)
                
                if success:
                    self.remediation_stats['remediated_threats'] += 1
                    self.last_action_times[rule.action.value] = time.time()
                    self.logger.info(f"Remediation successful: {rule.action.value}")
                else:
                    self.logger.warning(f"Remediation failed: {rule.action.value}")
                
                self.remediation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Remediation worker error: {e}")
    
    def _execute_remediation(self, event: SecurityEvent, rule: RemediationRule) -> bool:
        """Execute specific remediation action."""
        try:
            handler = self.action_handlers.get(rule.action)
            if handler:
                return handler(event, rule.parameters)
            else:
                self.logger.error(f"No handler for action: {rule.action}")
                return False
        except Exception as e:
            self.logger.error(f"Remediation execution failed: {e}")
            return False
    
    def _block_operation(self, event: SecurityEvent, params: Dict[str, Any]) -> bool:
        """Block a specific operation."""
        operation_id = params.get('operation_id', event.source)
        duration = params.get('duration_seconds', 600)  # 10 minutes default
        
        self.threat_detector.blocked_operations.add(operation_id)
        self.remediation_stats['blocked_operations'] += 1
        
        # Schedule unblock
        def unblock():
            time.sleep(duration)
            self.threat_detector.blocked_operations.discard(operation_id)
            self.logger.info(f"Unblocked operation: {operation_id}")
        
        threading.Thread(target=unblock, daemon=True).start()
        
        self.logger.warning(f"Blocked operation {operation_id} for {duration} seconds")
        return True
    
    def _apply_rate_limit(self, event: SecurityEvent, params: Dict[str, Any]) -> bool:
        """Apply dynamic rate limiting."""
        operation = params.get('operation', event.metadata.get('operation', 'unknown'))
        reduction_factor = params.get('reduction_factor', 0.5)
        
        if operation in self.threat_detector.rate_limits:
            original_limit = self.threat_detector.rate_limits[operation]
            new_limit = int(original_limit * reduction_factor)
            self.threat_detector.rate_limits[operation] = max(1, new_limit)
            
            self.logger.warning(f"Reduced rate limit for {operation}: {original_limit} -> {new_limit}")
            return True
        
        return False
    
    def _sanitize_input(self, event: SecurityEvent, params: Dict[str, Any]) -> bool:
        """Apply input sanitization."""
        # This would typically involve cleaning the specific input
        # For now, just log the action
        self.logger.info(f"Applied input sanitization for threat: {event.description}")
        return True
    
    def _restart_component(self, event: SecurityEvent, params: Dict[str, Any]) -> bool:
        """Restart affected system component."""
        component = params.get('component', 'unknown')
        self.logger.warning(f"Component restart triggered for: {component}")
        
        # In a real system, this would restart the actual component
        # For simulation, we just log the action
        return True
    
    def _alert_admin(self, event: SecurityEvent, params: Dict[str, Any]) -> bool:
        """Send alert to system administrator."""
        urgency = params.get('urgency', 'normal')
        
        alert_message = {
            'timestamp': time.time(),
            'event': event.to_dict(),
            'urgency': urgency,
            'recommended_action': params.get('recommended_action', 'investigate')
        }
        
        # In production, this would send actual alerts (email, slack, etc.)
        self.logger.critical(f"ADMIN ALERT [{urgency}]: {event.description}")
        return True
    
    def _quarantine_threat(self, event: SecurityEvent, params: Dict[str, Any]) -> bool:
        """Quarantine threat source."""
        source = event.source
        quarantine_duration = params.get('duration_hours', 24)
        
        # Add to blocked sources
        self.threat_detector.blocked_operations.add(f"quarantine_{source}")
        
        self.logger.warning(f"Quarantined threat source: {source} for {quarantine_duration} hours")
        return True
    
    def _get_default_rules(self) -> List[RemediationRule]:
        """Get default remediation rules."""
        return [
            # Critical threats - immediate blocking
            RemediationRule(
                threat_type="code_injection",
                severity_threshold="HIGH",
                action=RemediationAction.BLOCK_OPERATION,
                parameters={"duration_seconds": 3600},  # 1 hour
                cooldown_seconds=60
            ),
            
            # Rate limiting for abuse
            RemediationRule(
                threat_type="rate_limit_exceeded", 
                severity_threshold="MEDIUM",
                action=RemediationAction.RATE_LIMIT,
                parameters={"reduction_factor": 0.5},
                cooldown_seconds=300
            ),
            
            # Admin alerts for critical events
            RemediationRule(
                threat_type="*",  # All threat types
                severity_threshold="CRITICAL",
                action=RemediationAction.ALERT_ADMIN,
                parameters={"urgency": "high"},
                cooldown_seconds=600
            ),
            
            # Input sanitization for medium threats
            RemediationRule(
                threat_type="data_anomaly",
                severity_threshold="MEDIUM", 
                action=RemediationAction.SANITIZE_INPUT,
                parameters={},
                cooldown_seconds=30
            ),
            
            # Quarantine for persistent threats
            RemediationRule(
                threat_type="repeated_violation",
                severity_threshold="HIGH",
                action=RemediationAction.QUARANTINE,
                parameters={"duration_hours": 24},
                cooldown_seconds=3600
            )
        ]
    
    def add_custom_rule(self, rule: RemediationRule):
        """Add custom remediation rule."""
        self.rules.append(rule)
        self.logger.info(f"Added custom remediation rule: {rule.threat_type} -> {rule.action.value}")
    
    def get_remediation_stats(self) -> Dict[str, Any]:
        """Get remediation system statistics."""
        return {
            **self.remediation_stats,
            'active_rules': len(self.rules),
            'queue_size': self.remediation_queue.qsize(),
            'blocked_operations': len(self.threat_detector.blocked_operations),
            'worker_running': self.running
        }


class SecurityError(Exception):
    """Security-related error for remediation system."""
    pass