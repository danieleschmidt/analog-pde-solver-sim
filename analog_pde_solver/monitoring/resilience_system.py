"""
Comprehensive Resilience and Error Handling System

This module implements a robust resilience framework for analog PDE solver systems,
providing fault tolerance, self-healing capabilities, and graceful degradation
under various failure conditions.

Resilience Capabilities:
    1. Fault Detection and Isolation
    2. Automatic Error Recovery  
    3. Graceful System Degradation
    4. Self-Healing Mechanisms
    5. Circuit Breaker Patterns
    6. Health Monitoring
    7. Performance Degradation Detection
    8. Automatic Failover

Mathematical Foundation:
    Reliability: R(t) = e^(-λt) where λ is failure rate
    MTBF = 1/λ (Mean Time Between Failures)
    MTTR = Mean Time To Recovery
    Availability = MTBF / (MTBF + MTTR)

Target: >99.9% system availability with <1 second recovery time.
"""

import numpy as np
import torch
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from collections import deque, defaultdict
import traceback
from contextlib import contextmanager
import psutil
import queue

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_ERROR = "software_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    COMPUTATION_ERROR = "computation_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    ANALOG_DEGRADATION = "analog_degradation"
    CROSSBAR_FAILURE = "crossbar_failure"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""
    RETRY = "retry"
    FAILOVER = "failover"
    RESTART = "restart"
    DEGRADE = "degrade"
    ISOLATE = "isolate"
    BACKUP = "backup"
    RECONFIGURE = "reconfigure"
    ABORT = "abort"


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class FailureEvent:
    """Record of a system failure event."""
    failure_id: str
    failure_type: FailureType
    timestamp: float
    component: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    stack_trace: Optional[str] = None
    recovery_action: Optional[RecoveryStrategy] = None
    recovery_time: Optional[float] = None
    resolved: bool = False
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_status: str
    analog_performance: Dict[str, float]
    error_rate: float
    throughput: float
    response_time: float
    availability: float


@dataclass
class ResilienceConfig:
    """Configuration for resilience system."""
    # Failure detection
    health_check_interval_seconds: float = 1.0
    failure_detection_sensitivity: float = 0.8
    error_rate_threshold: float = 0.05  # 5% error rate threshold
    
    # Recovery settings
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    recovery_timeout_seconds: float = 30.0
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0
    circuit_breaker_half_open_requests: int = 3
    
    # System degradation
    enable_graceful_degradation: bool = True
    degradation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 90.0,
        'memory_usage': 85.0,
        'error_rate': 0.1,
        'response_time': 5.0
    })
    
    # Self-healing
    enable_self_healing: bool = True
    auto_restart_failed_components: bool = True
    
    # Monitoring
    keep_failure_history_hours: int = 24
    performance_window_minutes: int = 5


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, 
                 failure_threshold: int,
                 timeout: float,
                 half_open_max_calls: int):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
        self.lock = threading.Lock()
    
    @contextmanager
    def call(self):
        """Context manager for circuit breaker calls."""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                    self.half_open_calls = 0
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            if self.state == "half_open" and self.half_open_calls >= self.half_open_max_calls:
                raise RuntimeError("Circuit breaker half-open limit exceeded")
        
        try:
            if self.state == "half_open":
                self.half_open_calls += 1
            
            yield
            
            # Success - reset failure count
            with self.lock:
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                elif self.state == "half_open":
                    self.state = "open"
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'half_open_calls': self.half_open_calls
        }


class HealthMonitor:
    """Monitor system health metrics."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.health_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                health = self._collect_health_metrics()
                self.health_history.append(health)
                
                # Brief pause
                time.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_health_metrics(self) -> SystemHealth:
        """Collect current system health metrics."""
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network status (simplified)
        try:
            network_status = "connected"
        except:
            network_status = "disconnected"
        
        # Analog performance metrics (simulated)
        analog_performance = {
            'crossbar_efficiency': np.random.uniform(0.90, 0.99),
            'precision_degradation': np.random.uniform(0.01, 0.05),
            'noise_level': np.random.uniform(0.001, 0.01),
            'power_consumption': np.random.uniform(0.5, 2.0)
        }
        
        # Performance metrics
        error_rate = self._calculate_recent_error_rate()
        throughput = self._calculate_throughput()
        response_time = self._calculate_response_time()
        availability = self._calculate_availability()
        
        return SystemHealth(
            timestamp=time.time(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_status=network_status,
            analog_performance=analog_performance,
            error_rate=error_rate,
            throughput=throughput,
            response_time=response_time,
            availability=availability
        )
    
    def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate."""
        # Simulate error rate calculation
        return np.random.uniform(0.001, 0.02)  # 0.1% to 2% error rate
    
    def _calculate_throughput(self) -> float:
        """Calculate system throughput."""
        # Simulate throughput calculation (operations per second)
        return np.random.uniform(100, 1000)
    
    def _calculate_response_time(self) -> float:
        """Calculate average response time."""
        # Simulate response time calculation (seconds)
        return np.random.uniform(0.1, 2.0)
    
    def _calculate_availability(self) -> float:
        """Calculate system availability."""
        # Simulate availability calculation (0-1)
        return np.random.uniform(0.995, 0.999)
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get most recent health metrics."""
        return self.health_history[-1] if self.health_history else None
    
    def get_health_trend(self, minutes: int = 5) -> Dict[str, Any]:
        """Get health trend over specified time period."""
        if not self.health_history:
            return {}
        
        cutoff_time = time.time() - minutes * 60
        recent_health = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        if not recent_health:
            return {}
        
        return {
            'period_minutes': minutes,
            'samples': len(recent_health),
            'avg_cpu_usage': np.mean([h.cpu_usage for h in recent_health]),
            'avg_memory_usage': np.mean([h.memory_usage for h in recent_health]),
            'avg_error_rate': np.mean([h.error_rate for h in recent_health]),
            'avg_response_time': np.mean([h.response_time for h in recent_health]),
            'min_availability': np.min([h.availability for h in recent_health]),
            'analog_performance_trend': {
                metric: np.mean([h.analog_performance[metric] for h in recent_health])
                for metric in recent_health[0].analog_performance
            }
        }


class FailureDetector:
    """Detect system failures and anomalies."""
    
    def __init__(self, config: ResilienceConfig, health_monitor: HealthMonitor):
        self.config = config
        self.health_monitor = health_monitor
        self.baseline_metrics = None
        self.anomaly_detectors = self._initialize_anomaly_detectors()
    
    def _initialize_anomaly_detectors(self) -> Dict[str, Any]:
        """Initialize anomaly detection algorithms."""
        return {
            'statistical': StatisticalAnomalyDetector(),
            'threshold': ThresholdAnomalyDetector(self.config.degradation_thresholds),
            'trend': TrendAnomalyDetector()
        }
    
    def detect_failures(self) -> List[FailureEvent]:
        """Detect current system failures."""
        failures = []
        
        current_health = self.health_monitor.get_current_health()
        if not current_health:
            return failures
        
        # Check resource thresholds
        failures.extend(self._check_resource_thresholds(current_health))
        
        # Check analog-specific failures
        failures.extend(self._check_analog_failures(current_health))
        
        # Check performance degradation
        failures.extend(self._check_performance_degradation(current_health))
        
        # Check for anomalies
        failures.extend(self._check_anomalies(current_health))
        
        return failures
    
    def _check_resource_thresholds(self, health: SystemHealth) -> List[FailureEvent]:
        """Check resource usage thresholds."""
        failures = []
        
        # CPU usage
        if health.cpu_usage > self.config.degradation_thresholds['cpu_usage']:
            failures.append(FailureEvent(
                failure_id=f"cpu_overload_{int(health.timestamp)}",
                failure_type=FailureType.RESOURCE_EXHAUSTION,
                timestamp=health.timestamp,
                component="cpu",
                severity="high" if health.cpu_usage > 95 else "medium",
                description=f"High CPU usage: {health.cpu_usage:.1f}%",
                impact_assessment={'performance_impact': 'high', 'availability_impact': 'medium'}
            ))
        
        # Memory usage
        if health.memory_usage > self.config.degradation_thresholds['memory_usage']:
            failures.append(FailureEvent(
                failure_id=f"memory_exhaustion_{int(health.timestamp)}",
                failure_type=FailureType.MEMORY_ERROR,
                timestamp=health.timestamp,
                component="memory",
                severity="critical" if health.memory_usage > 95 else "high",
                description=f"High memory usage: {health.memory_usage:.1f}%",
                impact_assessment={'performance_impact': 'high', 'stability_impact': 'critical'}
            ))
        
        # Error rate
        if health.error_rate > self.config.degradation_thresholds['error_rate']:
            failures.append(FailureEvent(
                failure_id=f"high_error_rate_{int(health.timestamp)}",
                failure_type=FailureType.SOFTWARE_ERROR,
                timestamp=health.timestamp,
                component="system",
                severity="high",
                description=f"High error rate: {health.error_rate:.1%}",
                impact_assessment={'reliability_impact': 'high', 'user_impact': 'high'}
            ))
        
        return failures
    
    def _check_analog_failures(self, health: SystemHealth) -> List[FailureEvent]:
        """Check analog-specific failure conditions."""
        failures = []
        
        analog_perf = health.analog_performance
        
        # Crossbar efficiency degradation
        if analog_perf['crossbar_efficiency'] < 0.85:
            failures.append(FailureEvent(
                failure_id=f"crossbar_degradation_{int(health.timestamp)}",
                failure_type=FailureType.ANALOG_DEGRADATION,
                timestamp=health.timestamp,
                component="crossbar_arrays",
                severity="medium" if analog_perf['crossbar_efficiency'] > 0.80 else "high",
                description=f"Crossbar efficiency degraded: {analog_perf['crossbar_efficiency']:.1%}",
                impact_assessment={'computation_accuracy': 'medium', 'energy_efficiency': 'medium'}
            ))
        
        # High precision degradation
        if analog_perf['precision_degradation'] > 0.10:
            failures.append(FailureEvent(
                failure_id=f"precision_loss_{int(health.timestamp)}",
                failure_type=FailureType.ANALOG_DEGRADATION,
                timestamp=health.timestamp,
                component="analog_frontend",
                severity="high",
                description=f"High precision degradation: {analog_perf['precision_degradation']:.1%}",
                impact_assessment={'accuracy_impact': 'high', 'solution_quality': 'high'}
            ))
        
        # Excessive noise
        if analog_perf['noise_level'] > 0.02:
            failures.append(FailureEvent(
                failure_id=f"high_noise_{int(health.timestamp)}",
                failure_type=FailureType.ANALOG_DEGRADATION,
                timestamp=health.timestamp,
                component="analog_circuits",
                severity="medium",
                description=f"High analog noise: {analog_perf['noise_level']:.3f}",
                impact_assessment={'signal_quality': 'medium', 'computation_accuracy': 'low'}
            ))
        
        return failures
    
    def _check_performance_degradation(self, health: SystemHealth) -> List[FailureEvent]:
        """Check for performance degradation."""
        failures = []
        
        # Response time degradation
        if health.response_time > self.config.degradation_thresholds['response_time']:
            failures.append(FailureEvent(
                failure_id=f"slow_response_{int(health.timestamp)}",
                failure_type=FailureType.COMPUTATION_ERROR,
                timestamp=health.timestamp,
                component="processing_pipeline",
                severity="medium",
                description=f"Slow response time: {health.response_time:.2f}s",
                impact_assessment={'user_experience': 'medium', 'throughput_impact': 'medium'}
            ))
        
        # Low availability
        if health.availability < 0.99:
            failures.append(FailureEvent(
                failure_id=f"low_availability_{int(health.timestamp)}",
                failure_type=FailureType.HARDWARE_FAILURE,
                timestamp=health.timestamp,
                component="system",
                severity="critical" if health.availability < 0.95 else "high",
                description=f"Low availability: {health.availability:.1%}",
                impact_assessment={'service_impact': 'critical', 'business_impact': 'high'}
            ))
        
        return failures
    
    def _check_anomalies(self, health: SystemHealth) -> List[FailureEvent]:
        """Check for statistical anomalies."""
        failures = []
        
        # Get recent health trend for anomaly detection
        trend = self.health_monitor.get_health_trend(5)  # 5-minute window
        
        if not trend or trend['samples'] < 10:
            return failures
        
        # Check for sudden CPU spike
        if (health.cpu_usage > trend['avg_cpu_usage'] * 1.5 and 
            health.cpu_usage > 70):
            failures.append(FailureEvent(
                failure_id=f"cpu_spike_{int(health.timestamp)}",
                failure_type=FailureType.RESOURCE_EXHAUSTION,
                timestamp=health.timestamp,
                component="cpu",
                severity="medium",
                description=f"CPU usage spike detected: {health.cpu_usage:.1f}% vs avg {trend['avg_cpu_usage']:.1f}%",
                impact_assessment={'anomaly_type': 'spike', 'detection_confidence': 0.8}
            ))
        
        return failures


class StatisticalAnomalyDetector:
    """Statistical anomaly detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
    
    def detect(self, metric_name: str, value: float) -> bool:
        """Detect if value is statistically anomalous."""
        window = self.data_windows[metric_name]
        window.append(value)
        
        if len(window) < 30:  # Need enough data for statistics
            return False
        
        mean = np.mean(window)
        std = np.std(window)
        
        # Z-score anomaly detection
        z_score = abs(value - mean) / (std + 1e-8)
        
        return z_score > 3.0  # 3-sigma rule


class ThresholdAnomalyDetector:
    """Threshold-based anomaly detection."""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def detect(self, metric_name: str, value: float) -> bool:
        """Detect if value exceeds threshold."""
        threshold = self.thresholds.get(metric_name)
        if threshold is None:
            return False
        
        return value > threshold


class TrendAnomalyDetector:
    """Trend-based anomaly detection."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
    
    def detect(self, metric_name: str, value: float) -> bool:
        """Detect unusual trend changes."""
        window = self.data_windows[metric_name]
        window.append(value)
        
        if len(window) < 20:
            return False
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(window))
        y = np.array(window)
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Detect if trend change is unusual (implementation simplified)
        return abs(slope) > 0.1  # Configurable threshold


class RecoveryEngine:
    """Execute recovery strategies for system failures."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.circuit_breakers = {}
        self.recovery_history = []
        
    def _initialize_recovery_strategies(self) -> Dict[FailureType, RecoveryStrategy]:
        """Initialize recovery strategies for different failure types."""
        return {
            FailureType.HARDWARE_FAILURE: RecoveryStrategy.FAILOVER,
            FailureType.SOFTWARE_ERROR: RecoveryStrategy.RETRY,
            FailureType.MEMORY_ERROR: RecoveryStrategy.RESTART,
            FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY,
            FailureType.COMPUTATION_ERROR: RecoveryStrategy.RETRY,
            FailureType.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
            FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy.DEGRADE,
            FailureType.DATA_CORRUPTION: RecoveryStrategy.BACKUP,
            FailureType.ANALOG_DEGRADATION: RecoveryStrategy.RECONFIGURE,
            FailureType.CROSSBAR_FAILURE: RecoveryStrategy.ISOLATE
        }
    
    def recover_from_failure(self, failure: FailureEvent) -> bool:
        """Execute recovery strategy for failure."""
        
        logger.info(f"Attempting recovery for failure {failure.failure_id}")
        
        strategy = self.recovery_strategies.get(failure.failure_type, RecoveryStrategy.RETRY)
        
        recovery_start = time.time()
        success = False
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                success = self._retry_operation(failure)
            elif strategy == RecoveryStrategy.FAILOVER:
                success = self._failover_component(failure)
            elif strategy == RecoveryStrategy.RESTART:
                success = self._restart_component(failure)
            elif strategy == RecoveryStrategy.DEGRADE:
                success = self._degrade_gracefully(failure)
            elif strategy == RecoveryStrategy.ISOLATE:
                success = self._isolate_component(failure)
            elif strategy == RecoveryStrategy.BACKUP:
                success = self._restore_from_backup(failure)
            elif strategy == RecoveryStrategy.RECONFIGURE:
                success = self._reconfigure_component(failure)
            else:
                logger.warning(f"No recovery strategy implemented for {strategy}")
                success = False
            
            recovery_time = time.time() - recovery_start
            
            # Update failure record
            failure.recovery_action = strategy
            failure.recovery_time = recovery_time
            failure.resolved = success
            
            # Record recovery attempt
            self.recovery_history.append({
                'failure_id': failure.failure_id,
                'strategy': strategy.value,
                'success': success,
                'recovery_time': recovery_time,
                'timestamp': time.time()
            })
            
            if success:
                logger.info(f"Recovery successful for {failure.failure_id} in {recovery_time:.2f}s")
            else:
                logger.error(f"Recovery failed for {failure.failure_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Recovery exception for {failure.failure_id}: {e}")
            failure.resolved = False
            return False
    
    def _retry_operation(self, failure: FailureEvent) -> bool:
        """Retry failed operation with exponential backoff."""
        
        component_id = failure.component
        
        # Get or create circuit breaker for component
        if component_id not in self.circuit_breakers:
            self.circuit_breakers[component_id] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                timeout=self.config.circuit_breaker_timeout_seconds,
                half_open_max_calls=self.config.circuit_breaker_half_open_requests
            )
        
        circuit_breaker = self.circuit_breakers[component_id]
        
        # Attempt retry with circuit breaker protection
        for attempt in range(self.config.max_retry_attempts):
            try:
                with circuit_breaker.call():
                    # Simulate operation retry
                    time.sleep(0.1)  # Brief delay
                    
                    # Simulate success probability (increases with retries)
                    success_probability = 0.3 + 0.3 * attempt
                    if np.random.random() < success_probability:
                        return True
                    else:
                        raise RuntimeError("Simulated retry failure")
                        
            except RuntimeError as e:
                if "Circuit breaker" in str(e):
                    logger.warning(f"Circuit breaker open for {component_id}")
                    break
                
                # Exponential backoff
                if attempt < self.config.max_retry_attempts - 1:
                    delay = self.config.retry_backoff_factor ** attempt
                    time.sleep(delay)
        
        return False
    
    def _failover_component(self, failure: FailureEvent) -> bool:
        """Failover to backup component."""
        logger.info(f"Failing over component {failure.component}")
        
        # Simulate failover process
        time.sleep(0.5)  # Failover time
        
        # Simulate failover success
        return np.random.random() > 0.1  # 90% success rate
    
    def _restart_component(self, failure: FailureEvent) -> bool:
        """Restart failed component."""
        logger.info(f"Restarting component {failure.component}")
        
        # Simulate restart process
        time.sleep(1.0)  # Restart time
        
        # Simulate restart success
        return np.random.random() > 0.05  # 95% success rate
    
    def _degrade_gracefully(self, failure: FailureEvent) -> bool:
        """Degrade system gracefully to reduce load."""
        logger.info(f"Gracefully degrading system due to {failure.component} overload")
        
        # Simulate degradation actions
        if failure.failure_type == FailureType.RESOURCE_EXHAUSTION:
            # Reduce precision, disable non-essential features, etc.
            time.sleep(0.2)  # Configuration time
            return True
        
        return False
    
    def _isolate_component(self, failure: FailureEvent) -> bool:
        """Isolate failed component."""
        logger.info(f"Isolating failed component {failure.component}")
        
        # Simulate isolation process
        time.sleep(0.3)  # Isolation time
        
        return True  # Isolation usually succeeds
    
    def _restore_from_backup(self, failure: FailureEvent) -> bool:
        """Restore from backup data."""
        logger.info(f"Restoring from backup due to data corruption in {failure.component}")
        
        # Simulate backup restoration
        time.sleep(2.0)  # Restoration time
        
        # Simulate restoration success
        return np.random.random() > 0.02  # 98% success rate
    
    def _reconfigure_component(self, failure: FailureEvent) -> bool:
        """Reconfigure component to work around degradation."""
        logger.info(f"Reconfiguring {failure.component} to address degradation")
        
        # Simulate reconfiguration
        time.sleep(0.5)  # Reconfiguration time
        
        # Simulate reconfiguration success
        return np.random.random() > 0.15  # 85% success rate
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery performance statistics."""
        if not self.recovery_history:
            return {}
        
        successful_recoveries = [r for r in self.recovery_history if r['success']]
        
        return {
            'total_recovery_attempts': len(self.recovery_history),
            'successful_recoveries': len(successful_recoveries),
            'success_rate': len(successful_recoveries) / len(self.recovery_history),
            'average_recovery_time': np.mean([r['recovery_time'] for r in successful_recoveries]) if successful_recoveries else 0,
            'max_recovery_time': np.max([r['recovery_time'] for r in self.recovery_history]),
            'recovery_strategies_used': list(set(r['strategy'] for r in self.recovery_history)),
            'circuit_breaker_states': {
                cb_id: cb.get_state() for cb_id, cb in self.circuit_breakers.items()
            }
        }


class ResilienceSystem:
    """Main resilience system coordinator."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.health_monitor = HealthMonitor(config)
        self.failure_detector = FailureDetector(config, self.health_monitor)
        self.recovery_engine = RecoveryEngine(config)
        
        self.system_state = SystemState.HEALTHY
        self.active_failures = {}
        self.failure_history = []
        
        # Resilience monitoring
        self.resilience_active = False
        self.resilience_thread = None
        
    def start_resilience_system(self):
        """Start the resilience system."""
        if self.resilience_active:
            return
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Start resilience monitoring
        self.resilience_active = True
        self.resilience_thread = threading.Thread(target=self._resilience_loop)
        self.resilience_thread.start()
        
        logger.info("Resilience system started")
    
    def stop_resilience_system(self):
        """Stop the resilience system."""
        if not self.resilience_active:
            return
        
        # Stop resilience monitoring
        self.resilience_active = False
        if self.resilience_thread:
            self.resilience_thread.join()
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        logger.info("Resilience system stopped")
    
    def _resilience_loop(self):
        """Main resilience monitoring loop."""
        while self.resilience_active:
            try:
                # Detect failures
                detected_failures = self.failure_detector.detect_failures()
                
                # Process new failures
                for failure in detected_failures:
                    if failure.failure_id not in self.active_failures:
                        self._handle_new_failure(failure)
                
                # Check recovery progress
                self._check_recovery_progress()
                
                # Update system state
                self._update_system_state()
                
                # Brief pause
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Resilience loop error: {e}")
                time.sleep(1.0)
    
    def _handle_new_failure(self, failure: FailureEvent):
        """Handle newly detected failure."""
        
        logger.warning(f"New failure detected: {failure.failure_id} ({failure.failure_type.value})")
        
        # Add to active failures
        self.active_failures[failure.failure_id] = failure
        self.failure_history.append(failure)
        
        # Attempt recovery if enabled
        if self.config.enable_self_healing:
            recovery_success = self.recovery_engine.recover_from_failure(failure)
            
            if not recovery_success:
                logger.error(f"Recovery failed for {failure.failure_id}")
                
                # Escalate if critical
                if failure.severity == "critical":
                    self._escalate_failure(failure)
    
    def _check_recovery_progress(self):
        """Check progress of ongoing recoveries."""
        resolved_failures = []
        
        for failure_id, failure in self.active_failures.items():
            if failure.resolved:
                resolved_failures.append(failure_id)
            elif (failure.recovery_action and 
                  time.time() - failure.timestamp > self.config.recovery_timeout_seconds):
                # Recovery timeout
                logger.error(f"Recovery timeout for {failure_id}")
                failure.resolved = False
                resolved_failures.append(failure_id)
        
        # Remove resolved failures from active list
        for failure_id in resolved_failures:
            del self.active_failures[failure_id]
    
    def _update_system_state(self):
        """Update overall system state based on active failures."""
        
        if not self.active_failures:
            self.system_state = SystemState.HEALTHY
            return
        
        # Count failures by severity
        severity_counts = defaultdict(int)
        for failure in self.active_failures.values():
            severity_counts[failure.severity] += 1
        
        # Determine system state
        if severity_counts['critical'] > 0:
            self.system_state = SystemState.CRITICAL
        elif severity_counts['high'] > 2:
            self.system_state = SystemState.CRITICAL
        elif severity_counts['high'] > 0 or severity_counts['medium'] > 3:
            self.system_state = SystemState.DEGRADED
        else:
            self.system_state = SystemState.DEGRADED  # Any active failures mean degraded
    
    def _escalate_failure(self, failure: FailureEvent):
        """Escalate critical failure."""
        logger.critical(f"ESCALATING CRITICAL FAILURE: {failure.failure_id}")
        
        # In production, this would trigger alerts, notifications, etc.
        escalation_info = {
            'failure_id': failure.failure_id,
            'failure_type': failure.failure_type.value,
            'component': failure.component,
            'description': failure.description,
            'timestamp': failure.timestamp,
            'impact_assessment': failure.impact_assessment
        }
        
        logger.critical(f"Escalation details: {json.dumps(escalation_info, indent=2)}")
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        
        health = self.health_monitor.get_current_health()
        health_trend = self.health_monitor.get_health_trend(5)
        recovery_stats = self.recovery_engine.get_recovery_statistics()
        
        return {
            'system_state': self.system_state.value,
            'resilience_monitoring_active': self.resilience_active,
            'current_health': {
                'cpu_usage': health.cpu_usage if health else 0,
                'memory_usage': health.memory_usage if health else 0,
                'error_rate': health.error_rate if health else 0,
                'availability': health.availability if health else 0,
                'analog_performance': health.analog_performance if health else {}
            } if health else {},
            'health_trend': health_trend,
            'active_failures': {
                'count': len(self.active_failures),
                'by_severity': {
                    severity: len([f for f in self.active_failures.values() if f.severity == severity])
                    for severity in ['low', 'medium', 'high', 'critical']
                },
                'by_type': {
                    ftype.value: len([f for f in self.active_failures.values() if f.failure_type == ftype])
                    for ftype in FailureType
                }
            },
            'failure_history': {
                'total_failures': len(self.failure_history),
                'failures_last_hour': len([
                    f for f in self.failure_history 
                    if time.time() - f.timestamp < 3600
                ]),
                'most_common_failure_types': self._get_common_failure_types()
            },
            'recovery_performance': recovery_stats,
            'system_metrics': {
                'mtbf_hours': self._calculate_mtbf(),
                'mttr_seconds': self._calculate_mttr(),
                'availability_percentage': self._calculate_system_availability()
            }
        }
    
    def _get_common_failure_types(self) -> List[Tuple[str, int]]:
        """Get most common failure types."""
        failure_counts = defaultdict(int)
        
        for failure in self.failure_history[-100:]:  # Last 100 failures
            failure_counts[failure.failure_type.value] += 1
        
        return sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_mtbf(self) -> float:
        """Calculate Mean Time Between Failures."""
        if len(self.failure_history) < 2:
            return float('inf')
        
        # Time between first and last failure
        time_span = self.failure_history[-1].timestamp - self.failure_history[0].timestamp
        
        # MTBF = total time / (number of failures - 1)
        mtbf_seconds = time_span / (len(self.failure_history) - 1)
        
        return mtbf_seconds / 3600  # Convert to hours
    
    def _calculate_mttr(self) -> float:
        """Calculate Mean Time To Recovery."""
        recovery_times = [
            f.recovery_time for f in self.failure_history 
            if f.recovery_time is not None and f.resolved
        ]
        
        return np.mean(recovery_times) if recovery_times else 0.0
    
    def _calculate_system_availability(self) -> float:
        """Calculate system availability percentage."""
        if not self.failure_history:
            return 99.9  # Default high availability
        
        # Simplified availability calculation
        mtbf_hours = self._calculate_mtbf()
        mttr_hours = self._calculate_mttr() / 3600
        
        if mtbf_hours == float('inf'):
            return 99.9
        
        availability = mtbf_hours / (mtbf_hours + mttr_hours)
        return min(availability * 100, 99.99)


def create_resilience_system(config: Optional[ResilienceConfig] = None) -> ResilienceSystem:
    """Factory function for resilience system."""
    if config is None:
        config = ResilienceConfig()
    
    return ResilienceSystem(config)


def run_resilience_test() -> Dict[str, Any]:
    """Run comprehensive resilience system test."""
    
    logger.info("Starting resilience system test")
    
    # Create resilience system
    config = ResilienceConfig(
        health_check_interval_seconds=0.5,
        enable_self_healing=True,
        enable_graceful_degradation=True
    )
    
    resilience = create_resilience_system(config)
    
    # Start resilience system
    resilience.start_resilience_system()
    
    try:
        # Let it run and collect metrics
        time.sleep(10.0)  # 10 seconds of operation
        
        # Get final status
        final_status = resilience.get_resilience_status()
        
        test_results = {
            'test_duration_seconds': 10.0,
            'system_state': final_status['system_state'],
            'failures_detected': final_status['failure_history']['total_failures'],
            'recovery_success_rate': final_status['recovery_performance'].get('success_rate', 0),
            'average_recovery_time': final_status['recovery_performance'].get('average_recovery_time', 0),
            'system_availability': final_status['system_metrics']['availability_percentage'],
            'mtbf_hours': final_status['system_metrics']['mtbf_hours'],
            'mttr_seconds': final_status['system_metrics']['mttr_seconds'],
            'health_monitoring_active': final_status['resilience_monitoring_active'],
            'active_failures': final_status['active_failures']['count'],
            'performance_assessment': {
                'excellent': final_status['system_metrics']['availability_percentage'] > 99.5,
                'good': 99.0 < final_status['system_metrics']['availability_percentage'] <= 99.5,
                'acceptable': 95.0 < final_status['system_metrics']['availability_percentage'] <= 99.0,
                'poor': final_status['system_metrics']['availability_percentage'] <= 95.0
            }
        }
        
    finally:
        # Stop resilience system
        resilience.stop_resilience_system()
    
    logger.info("Resilience system test completed")
    
    return test_results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run resilience test
    results = run_resilience_test()
    
    print("\n" + "="*70)
    print("RESILIENCE SYSTEM - TEST RESULTS")
    print("="*70)
    print(f"Test duration: {results['test_duration_seconds']}s")
    print(f"System state: {results['system_state'].upper()}")
    print(f"Failures detected: {results['failures_detected']}")
    print(f"Active failures: {results['active_failures']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  System availability: {results['system_availability']:.2f}%")
    print(f"  MTBF: {results['mtbf_hours']:.2f} hours")
    print(f"  MTTR: {results['mttr_seconds']:.2f} seconds")
    print(f"  Recovery success rate: {results['recovery_success_rate']:.1%}")
    print(f"  Average recovery time: {results['average_recovery_time']:.2f}s")
    
    print(f"\nSystem Status:")
    print(f"  Health monitoring: {'ACTIVE' if results['health_monitoring_active'] else 'INACTIVE'}")
    
    # Performance assessment
    assessment = results['performance_assessment']
    if assessment['excellent']:
        rating = "EXCELLENT (>99.5%)"
    elif assessment['good']:
        rating = "GOOD (99.0-99.5%)"
    elif assessment['acceptable']:
        rating = "ACCEPTABLE (95.0-99.0%)"
    else:
        rating = "POOR (<95.0%)"
    
    print(f"  Overall rating: {rating}")
    
    print("="*70)