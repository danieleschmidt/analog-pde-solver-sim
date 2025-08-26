"""Auto-scaling system for analog PDE solver performance optimization."""

import numpy as np
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc


class ScalingMetric(Enum):
    """Metrics for auto-scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"  
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    throughput: float = 0.0  # operations/second
    average_latency: float = 0.0  # seconds
    error_rate: float = 0.0  # percentage
    queue_length: int = 0
    active_workers: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8
    cooldown_seconds: int = 60
    min_workers: int = 1
    max_workers: int = 16


class AdaptiveAutoScaler:
    """Adaptive auto-scaling system for PDE solver performance."""
    
    def __init__(
        self,
        initial_workers: int = 4,
        monitoring_interval: float = 10.0,
        metrics_history_size: int = 100
    ):
        self.initial_workers = initial_workers
        self.monitoring_interval = monitoring_interval
        self.metrics_history_size = metrics_history_size
        
        # Current state
        self.current_workers = initial_workers
        self.metrics_history: List[PerformanceMetrics] = []
        self.last_scaling_time = 0.0
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = self._create_default_rules()
        
        # Monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.scaling_events: List[Dict[str, Any]] = []
        self.total_operations = 0
        self.total_errors = 0
        
        # Resource limits
        self.max_memory_gb = 8.0
        self.max_cpu_percent = 80.0
    
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            # CPU-based scaling
            ScalingRule(
                metric=ScalingMetric.CPU_USAGE,
                threshold_up=70.0,
                threshold_down=30.0,
                scale_up_factor=1.3,
                scale_down_factor=0.8,
                cooldown_seconds=30
            ),
            # Memory-based scaling
            ScalingRule(
                metric=ScalingMetric.MEMORY_USAGE,
                threshold_up=80.0,
                threshold_down=40.0,
                scale_up_factor=1.2,
                scale_down_factor=0.9,
                cooldown_seconds=60
            ),
            # Throughput-based scaling
            ScalingRule(
                metric=ScalingMetric.THROUGHPUT,
                threshold_up=100.0,  # ops/sec
                threshold_down=20.0,
                scale_up_factor=1.4,
                scale_down_factor=0.7,
                cooldown_seconds=45
            ),
            # Latency-based scaling
            ScalingRule(
                metric=ScalingMetric.LATENCY,
                threshold_up=5.0,  # seconds
                threshold_down=1.0,
                scale_up_factor=1.5,
                scale_down_factor=0.8,
                cooldown_seconds=30
            )
        ]
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        with self._lock:
            if self._monitoring_active:
                return
            
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        with self._lock:
            self._monitoring_active = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                self._monitor_thread = None
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                with self._lock:
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > self.metrics_history_size:
                        self.metrics_history.pop(0)
                
                # Make scaling decisions
                self._evaluate_scaling_rules(metrics)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        # Application metrics (would be provided by solver)
        current_time = time.time()
        
        # Calculate throughput from recent history
        throughput = self._calculate_throughput()
        
        # Calculate average latency
        avg_latency = self._calculate_average_latency()
        
        # Calculate error rate
        error_rate = self._calculate_error_rate()
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            throughput=throughput,
            average_latency=avg_latency,
            error_rate=error_rate,
            queue_length=0,  # Would be provided by task queue
            active_workers=self.current_workers,
            timestamp=current_time
        )
    
    def _calculate_throughput(self) -> float:
        """Calculate operations per second."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        if len(recent_metrics) < 2:
            return 0.0
        
        time_window = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        if time_window <= 0:
            return 0.0
        
        # Simplified throughput calculation
        operations_in_window = len(recent_metrics) * self.current_workers
        return operations_in_window / time_window
    
    def _calculate_average_latency(self) -> float:
        """Calculate average operation latency."""
        if len(self.metrics_history) < 5:
            return 1.0
        
        # Estimate latency based on throughput and worker count
        recent_throughput = self._calculate_throughput()
        if recent_throughput <= 0:
            return 5.0  # High latency when no throughput
        
        # Simple latency model
        estimated_latency = self.current_workers / max(recent_throughput, 1.0)
        return min(estimated_latency, 10.0)  # Cap at 10 seconds
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_operations == 0:
            return 0.0
        
        return (self.total_errors / self.total_operations) * 100.0
    
    def _evaluate_scaling_rules(self, current_metrics: PerformanceMetrics):
        """Evaluate all scaling rules and make decisions."""
        
        current_time = time.time()
        
        for rule in self.scaling_rules:
            # Check cooldown period
            if current_time - self.last_scaling_time < rule.cooldown_seconds:
                continue
            
            # Get metric value
            metric_value = self._get_metric_value(current_metrics, rule.metric)
            
            # Check scaling conditions
            should_scale_up = (
                metric_value > rule.threshold_up and
                self.current_workers < rule.max_workers
            )
            
            should_scale_down = (
                metric_value < rule.threshold_down and
                self.current_workers > rule.min_workers
            )
            
            # Apply scaling
            if should_scale_up:
                new_workers = min(
                    int(self.current_workers * rule.scale_up_factor),
                    rule.max_workers
                )
                self._scale_workers(new_workers, "up", rule.metric, metric_value)
                break  # Only apply one rule at a time
                
            elif should_scale_down:
                new_workers = max(
                    int(self.current_workers * rule.scale_down_factor),
                    rule.min_workers
                )
                self._scale_workers(new_workers, "down", rule.metric, metric_value)
                break
    
    def _get_metric_value(
        self,
        metrics: PerformanceMetrics,
        metric_type: ScalingMetric
    ) -> float:
        """Get specific metric value."""
        
        if metric_type == ScalingMetric.CPU_USAGE:
            return metrics.cpu_percent
        elif metric_type == ScalingMetric.MEMORY_USAGE:
            return metrics.memory_percent
        elif metric_type == ScalingMetric.THROUGHPUT:
            return metrics.throughput
        elif metric_type == ScalingMetric.LATENCY:
            return metrics.average_latency
        elif metric_type == ScalingMetric.ERROR_RATE:
            return metrics.error_rate
        elif metric_type == ScalingMetric.QUEUE_LENGTH:
            return metrics.queue_length
        else:
            return 0.0
    
    def _scale_workers(
        self,
        new_worker_count: int,
        direction: str,
        metric: ScalingMetric,
        metric_value: float
    ):
        """Scale worker count."""
        
        old_count = self.current_workers
        self.current_workers = new_worker_count
        self.last_scaling_time = time.time()
        
        # Record scaling event
        scaling_event = {
            'timestamp': self.last_scaling_time,
            'old_workers': old_count,
            'new_workers': new_worker_count,
            'direction': direction,
            'trigger_metric': metric.value,
            'metric_value': metric_value,
            'reason': f"{metric.value} {direction} scaling: {metric_value:.2f}"
        }
        
        self.scaling_events.append(scaling_event)
        
        # Keep only recent events
        if len(self.scaling_events) > 100:
            self.scaling_events = self.scaling_events[-50:]
        
        print(f"Auto-scaling: {old_count} -> {new_worker_count} workers "
              f"({direction}) due to {metric.value}={metric_value:.2f}")
    
    def get_optimal_worker_count(
        self,
        problem_size: int,
        complexity_factor: float = 1.0
    ) -> int:
        """Get optimal worker count for problem size."""
        
        # Base calculation on problem size and system resources
        cpu_cores = psutil.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate optimal workers based on problem characteristics
        size_factor = min(problem_size / 1000, 10.0)  # Scale with problem size
        memory_factor = min(available_memory_gb / 2.0, 8.0)  # Memory constraint
        cpu_factor = min(cpu_cores, 16)  # CPU constraint
        
        # Combine factors
        optimal = int(min(size_factor * complexity_factor, memory_factor, cpu_factor))
        
        # Apply bounds
        return max(1, min(optimal, 32))
    
    def suggest_batch_size(self, total_items: int, target_latency: float = 1.0) -> int:
        """Suggest optimal batch size for processing."""
        
        if len(self.metrics_history) == 0:
            return min(100, max(10, total_items // 10))
        
        # Use recent performance data
        recent_metrics = self.metrics_history[-5:]
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        
        if avg_throughput <= 0:
            return min(100, max(10, total_items // 10))
        
        # Calculate batch size for target latency
        items_per_second_per_worker = avg_throughput / max(self.current_workers, 1)
        target_batch_size = int(items_per_second_per_worker * target_latency)
        
        # Apply reasonable bounds
        return max(10, min(target_batch_size, 1000))
    
    def trigger_garbage_collection(self):
        """Trigger intelligent garbage collection."""
        
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 75.0:
            # Force garbage collection
            collected = gc.collect()
            print(f"Garbage collection freed {collected} objects (memory: {memory_usage:.1f}%)")
        
        # Clean up metrics history if it's getting too large
        if len(self.metrics_history) > self.metrics_history_size * 1.5:
            self.metrics_history = self.metrics_history[-self.metrics_history_size:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        recent_metrics = self.metrics_history[-10:]
        
        # Calculate statistics
        cpu_stats = [m.cpu_percent for m in recent_metrics]
        memory_stats = [m.memory_percent for m in recent_metrics]
        throughput_stats = [m.throughput for m in recent_metrics]
        latency_stats = [m.average_latency for m in recent_metrics]
        
        return {
            'current_workers': self.current_workers,
            'total_operations': self.total_operations,
            'total_errors': self.total_errors,
            'error_rate_percent': self._calculate_error_rate(),
            'cpu': {
                'current': cpu_stats[-1] if cpu_stats else 0,
                'average': np.mean(cpu_stats),
                'max': np.max(cpu_stats)
            },
            'memory': {
                'current': memory_stats[-1] if memory_stats else 0,
                'average': np.mean(memory_stats),
                'max': np.max(memory_stats)
            },
            'throughput': {
                'current': throughput_stats[-1] if throughput_stats else 0,
                'average': np.mean(throughput_stats),
                'max': np.max(throughput_stats)
            },
            'latency': {
                'current': latency_stats[-1] if latency_stats else 0,
                'average': np.mean(latency_stats),
                'min': np.min(latency_stats) if latency_stats else 0
            },
            'scaling_events': len(self.scaling_events),
            'last_scaling': self.scaling_events[-1] if self.scaling_events else None
        }
    
    def add_operation_result(self, success: bool, duration: float):
        """Record operation result for metrics."""
        self.total_operations += 1
        if not success:
            self.total_errors += 1
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_monitoring()
        except:
            pass