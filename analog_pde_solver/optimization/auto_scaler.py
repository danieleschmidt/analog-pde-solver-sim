"""Auto-scaling capabilities for analog PDE solver systems."""

import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from ..utils.logging_config import get_logger


class ScalingDecision(Enum):
    """Scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: float
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    active_workers: int = 0


@dataclass  
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.7  # 70% utilization
    scale_down_threshold: float = 0.3  # 30% utilization
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    metrics_window: int = 5  # Number of metrics to consider
    enable_predictive: bool = True


class AutoScaler:
    """Intelligent auto-scaling for analog PDE solver workloads."""
    
    def __init__(self, policy: ScalingPolicy = None):
        """Initialize auto-scaler.
        
        Args:
            policy: Scaling policy configuration
        """
        self.policy = policy or ScalingPolicy()
        self.logger = get_logger('autoscaler')
        
        # Metrics tracking
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_events: List[Dict[str, Any]] = []
        
        # Scaling state
        self.current_instances = self.policy.min_instances
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        
        # Resource management
        self.resource_pools: Dict[str, List[Any]] = {
            'solvers': [],
            'crossbars': [],
            'workers': []
        }
        
        # Predictive scaling
        self.demand_predictor = DemandPredictor() if self.policy.enable_predictive else None
        
        # Threading
        self.scaling_lock = threading.Lock()
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger.info(f"Auto-scaler initialized with policy: {self.policy}")
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring and scaling.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring and scaling."""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                
                # Execute scaling if needed
                if decision != ScalingDecision.MAINTAIN:
                    self._execute_scaling_decision(decision, metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)  # Continue monitoring despite errors
    
    def _collect_system_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        metrics = ScalingMetrics(timestamp=time.time())
        
        # CPU and memory utilization
        try:
            import psutil
            metrics.cpu_utilization = psutil.cpu_percent(interval=1.0) / 100.0
            memory_info = psutil.virtual_memory()
            metrics.memory_utilization = memory_info.percent / 100.0
        except ImportError:
            # Fallback metrics
            metrics.cpu_utilization = 0.5  # Assume moderate load
            metrics.memory_utilization = 0.4
        
        # Worker and queue metrics
        metrics.active_workers = self.current_instances
        metrics.queue_length = len(getattr(self, 'pending_tasks', []))
        
        # Performance metrics from recent operations
        if hasattr(self, 'recent_response_times'):
            times = getattr(self, 'recent_response_times', [1.0])
            metrics.response_time = np.mean(times[-10:]) if times else 1.0
        
        if hasattr(self, 'recent_error_counts'):
            errors = getattr(self, 'recent_error_counts', [0])
            total_ops = max(1, len(errors))
            metrics.error_rate = sum(errors) / total_ops if errors else 0.0
        
        # Calculate throughput
        if len(self.metrics_history) > 1:
            prev_metrics = self.metrics_history[-1]
            time_delta = metrics.timestamp - prev_metrics.timestamp
            if time_delta > 0:
                # Simple throughput estimate
                metrics.throughput = 1.0 / max(0.1, metrics.response_time)
        
        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Limit history size
            self.metrics_history = self.metrics_history[-500:]
        
        return metrics
    
    def _make_scaling_decision(self, current_metrics: ScalingMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics."""
        if len(self.metrics_history) < self.policy.metrics_window:
            return ScalingDecision.MAINTAIN
        
        # Get recent metrics for trend analysis
        recent_metrics = self.metrics_history[-self.policy.metrics_window:]
        
        # Calculate average utilization
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_memory = np.mean([m.memory_utilization for m in recent_metrics])
        avg_response_time = np.mean([m.response_time for m in recent_metrics])
        
        # Combined utilization score
        utilization_score = max(avg_cpu, avg_memory)
        
        # Predictive component
        predicted_load = 0.0
        if self.demand_predictor and self.policy.enable_predictive:
            predicted_load = self.demand_predictor.predict_load(recent_metrics)
            utilization_score = max(utilization_score, predicted_load)
        
        # Check cooldown periods
        current_time = time.time()
        scale_up_ready = (current_time - self.last_scale_up_time) > self.policy.scale_up_cooldown
        scale_down_ready = (current_time - self.last_scale_down_time) > self.policy.scale_down_cooldown
        
        # Scaling logic
        if (utilization_score > self.policy.scale_up_threshold and 
            scale_up_ready and 
            self.current_instances < self.policy.max_instances):
            
            # Additional checks for scale up
            if (avg_response_time > 2.0 or  # High response time
                current_metrics.queue_length > 10 or  # Queue backlog
                current_metrics.error_rate > 0.05):  # High error rate
                
                self.logger.info(
                    f"Scale up decision: utilization={utilization_score:.2f}, "
                    f"response_time={avg_response_time:.2f}s"
                )
                return ScalingDecision.SCALE_UP
        
        elif (utilization_score < self.policy.scale_down_threshold and 
              scale_down_ready and 
              self.current_instances > self.policy.min_instances):
            
            # Additional checks for scale down
            if (avg_response_time < 0.5 and  # Low response time
                current_metrics.queue_length == 0 and  # No queue
                current_metrics.error_rate < 0.01):  # Low error rate
                
                self.logger.info(
                    f"Scale down decision: utilization={utilization_score:.2f}, "
                    f"response_time={avg_response_time:.2f}s"
                )
                return ScalingDecision.SCALE_DOWN
        
        return ScalingDecision.MAINTAIN
    
    def _execute_scaling_decision(self, decision: ScalingDecision, metrics: ScalingMetrics):
        """Execute scaling decision."""
        with self.scaling_lock:
            try:
                if decision == ScalingDecision.SCALE_UP:
                    self._scale_up(metrics)
                elif decision == ScalingDecision.SCALE_DOWN:
                    self._scale_down(metrics)
                    
            except Exception as e:
                self.logger.error(f"Failed to execute scaling decision: {e}")
    
    def _scale_up(self, metrics: ScalingMetrics):
        """Add resources to handle increased load."""
        # Calculate how many instances to add
        target_instances = min(
            self.current_instances + 1,  # Conservative scaling
            self.policy.max_instances
        )
        
        if target_instances <= self.current_instances:
            return
        
        instances_to_add = target_instances - self.current_instances
        
        # Add solver instances
        for i in range(instances_to_add):
            try:
                # Create new solver instance
                from ..core.solver_robust import RobustAnalogPDESolver
                new_solver = RobustAnalogPDESolver(
                    crossbar_size=128,
                    conductance_range=(1e-9, 1e-6),
                    noise_model="realistic"
                )
                
                self.resource_pools['solvers'].append(new_solver)
                
            except Exception as e:
                self.logger.error(f"Failed to create solver instance: {e}")
                continue
        
        # Update instance count
        old_count = self.current_instances
        self.current_instances = len(self.resource_pools['solvers'])
        self.last_scale_up_time = time.time()
        
        # Log scaling event
        event = {
            'timestamp': time.time(),
            'type': 'scale_up',
            'old_instances': old_count,
            'new_instances': self.current_instances,
            'trigger_metrics': metrics.__dict__.copy()
        }
        self.scaling_events.append(event)
        
        self.logger.info(f"Scaled up from {old_count} to {self.current_instances} instances")
    
    def _scale_down(self, metrics: ScalingMetrics):
        """Remove resources when load decreases."""
        target_instances = max(
            self.current_instances - 1,  # Conservative scaling
            self.policy.min_instances
        )
        
        if target_instances >= self.current_instances:
            return
        
        instances_to_remove = self.current_instances - target_instances
        
        # Remove solver instances (LIFO - remove newest first)
        for i in range(instances_to_remove):
            if self.resource_pools['solvers']:
                removed_solver = self.resource_pools['solvers'].pop()
                
                # Cleanup removed solver if needed
                if hasattr(removed_solver, 'cleanup'):
                    try:
                        removed_solver.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup removed solver: {e}")
        
        # Update instance count
        old_count = self.current_instances
        self.current_instances = len(self.resource_pools['solvers'])
        self.last_scale_down_time = time.time()
        
        # Log scaling event
        event = {
            'timestamp': time.time(),
            'type': 'scale_down',
            'old_instances': old_count,
            'new_instances': self.current_instances,
            'trigger_metrics': metrics.__dict__.copy()
        }
        self.scaling_events.append(event)
        
        self.logger.info(f"Scaled down from {old_count} to {self.current_instances} instances")
    
    def get_available_solver(self) -> Optional[Any]:
        """Get an available solver instance for processing."""
        with self.scaling_lock:
            if self.resource_pools['solvers']:
                # Simple round-robin selection
                solver = self.resource_pools['solvers'][0]
                # Move to end for round-robin
                self.resource_pools['solvers'] = (
                    self.resource_pools['solvers'][1:] + 
                    [self.resource_pools['solvers'][0]]
                )
                return solver
            return None
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        recent_events = [e for e in self.scaling_events 
                        if time.time() - e['timestamp'] < 86400]  # Last 24h
        
        scale_ups = sum(1 for e in recent_events if e['type'] == 'scale_up')
        scale_downs = sum(1 for e in recent_events if e['type'] == 'scale_down')
        
        stats = {
            'current_instances': self.current_instances,
            'policy': {
                'min_instances': self.policy.min_instances,
                'max_instances': self.policy.max_instances,
                'scale_up_threshold': self.policy.scale_up_threshold,
                'scale_down_threshold': self.policy.scale_down_threshold
            },
            'recent_events': {
                'scale_ups_24h': scale_ups,
                'scale_downs_24h': scale_downs,
                'total_events': len(recent_events)
            },
            'current_metrics': self.metrics_history[-1].__dict__ if self.metrics_history else {},
            'resource_pools': {
                pool: len(resources) for pool, resources in self.resource_pools.items()
            }
        }
        
        # Calculate efficiency metrics
        if len(self.metrics_history) > 10:
            recent_util = [m.cpu_utilization for m in self.metrics_history[-10:]]
            stats['efficiency'] = {
                'avg_utilization': float(np.mean(recent_util)),
                'utilization_variance': float(np.var(recent_util)),
                'target_utilization': (self.policy.scale_up_threshold + self.policy.scale_down_threshold) / 2
            }
        
        return stats
    
    def cleanup(self):
        """Cleanup auto-scaler resources."""
        # Stop monitoring
        self.stop_monitoring()
        
        # Cleanup all solver instances
        for solver in self.resource_pools['solvers']:
            if hasattr(solver, 'cleanup'):
                try:
                    solver.cleanup()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup solver: {e}")
        
        # Clear resource pools
        for pool in self.resource_pools.values():
            pool.clear()
        
        self.logger.info("Auto-scaler cleanup completed")


class DemandPredictor:
    """Simple demand prediction for proactive scaling."""
    
    def __init__(self, window_size: int = 20):
        """Initialize demand predictor.
        
        Args:
            window_size: Size of historical window for predictions
        """
        self.window_size = window_size
        self.logger = get_logger('demand_predictor')
    
    def predict_load(self, metrics_history: List[ScalingMetrics]) -> float:
        """Predict future load based on historical trends.
        
        Args:
            metrics_history: Recent metrics for trend analysis
            
        Returns:
            Predicted load (0-1 scale)
        """
        if len(metrics_history) < 5:
            return 0.0
        
        try:
            # Simple linear trend prediction
            recent_metrics = metrics_history[-min(self.window_size, len(metrics_history)):]
            
            # Extract time series data
            timestamps = [m.timestamp for m in recent_metrics]
            utilizations = [max(m.cpu_utilization, m.memory_utilization) for m in recent_metrics]
            
            # Simple linear regression
            n = len(timestamps)
            if n < 3:
                return utilizations[-1]  # Return current if not enough data
            
            # Calculate trend slope
            x_mean = np.mean(timestamps)
            y_mean = np.mean(utilizations)
            
            numerator = sum((timestamps[i] - x_mean) * (utilizations[i] - y_mean) for i in range(n))
            denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return utilizations[-1]
            
            slope = numerator / denominator
            
            # Predict 5 minutes ahead
            future_time = timestamps[-1] + 300  # 5 minutes
            predicted_util = y_mean + slope * (future_time - x_mean)
            
            # Clamp prediction to reasonable range
            predicted_util = max(0.0, min(1.0, predicted_util))
            
            self.logger.debug(f"Predicted load: {predicted_util:.3f} (slope: {slope:.6f})")
            
            return predicted_util
            
        except Exception as e:
            self.logger.error(f"Load prediction failed: {e}")
            return metrics_history[-1].cpu_utilization if metrics_history else 0.0