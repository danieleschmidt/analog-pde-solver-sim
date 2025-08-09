"""Adaptive scaling and load balancing for analog PDE solver."""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Strategies for adaptive scaling."""
    WORKLOAD_BASED = "workload_based"
    RESOURCE_BASED = "resource_based"
    PERFORMANCE_BASED = "performance_based"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkloadMetrics:
    """Workload performance metrics."""
    throughput: float  # operations per second
    latency: float     # average response time
    queue_length: int  # number of queued tasks
    error_rate: float  # error percentage
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    action: str  # "scale_up", "scale_down", "maintain"
    target_workers: int
    confidence: float
    reasoning: List[str]
    expected_improvement: float
    timestamp: float = field(default_factory=time.time)


class ResourceMonitor:
    """Monitor system resources for scaling decisions."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: Time interval between measurements (seconds)
        """
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 1000
        
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_lock = threading.Lock()
        
        # GPU monitoring (if available)
        self.has_gpu = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            return False
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        with self._monitor_lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(target=self._monitor_loop)
                self._monitor_thread.daemon = True
                self._monitor_thread.start()
                self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        with self._monitor_lock:
            if self._monitoring:
                self._monitoring = False
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5)
                self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._monitor_lock:
                    self.metrics_history.append(metrics)
                    
                    # Limit history size
                    if len(self.metrics_history) > self.max_history:
                        self.metrics_history = self.metrics_history[-self.max_history//2:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and memory usage
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk I/O
        disk_io = 0.0
        try:
            disk_stats = psutil.disk_io_counters()
            if disk_stats:
                disk_io = disk_stats.read_bytes + disk_stats.write_bytes
        except:
            pass
        
        # Network I/O
        network_io = 0.0
        try:
            net_stats = psutil.net_io_counters()
            if net_stats:
                network_io = net_stats.bytes_sent + net_stats.bytes_recv
        except:
            pass
        
        # GPU usage (if available)
        gpu_usage = 0.0
        if self.has_gpu:
            gpu_usage = self._get_gpu_usage()
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            disk_io=disk_io,
            network_io=network_io
        )
    
    def _get_gpu_usage(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0.0
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent resource metrics."""
        with self._monitor_lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_seconds: float = 300) -> List[ResourceMetrics]:
        """Get recent metrics history.
        
        Args:
            duration_seconds: How far back to look (default: 5 minutes)
            
        Returns:
            List of metrics within the specified duration
        """
        cutoff_time = time.time() - duration_seconds
        
        with self._monitor_lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_resource_trends(self, duration_seconds: float = 300) -> Dict[str, float]:
        """Calculate resource usage trends.
        
        Args:
            duration_seconds: Time window for trend calculation
            
        Returns:
            Dictionary of trend values (positive = increasing, negative = decreasing)
        """
        history = self.get_metrics_history(duration_seconds)
        
        if len(history) < 2:
            return {}
        
        # Calculate linear trends
        times = [m.timestamp for m in history]
        cpu_values = [m.cpu_usage for m in history]
        memory_values = [m.memory_usage for m in history]
        gpu_values = [m.gpu_usage for m in history]
        
        trends = {}
        
        # Simple linear trend calculation
        for name, values in [("cpu", cpu_values), ("memory", memory_values), ("gpu", gpu_values)]:
            if len(values) >= 2:
                # Calculate slope
                x = np.array(times)
                y = np.array(values)
                
                # Normalize time to avoid numerical issues
                x = x - x[0]
                
                if len(x) > 1 and np.std(x) > 0:
                    slope = np.polyfit(x, y, 1)[0]
                    trends[f"{name}_trend"] = slope
                else:
                    trends[f"{name}_trend"] = 0.0
        
        return trends


class WorkloadManager:
    """Manage workload distribution and performance tracking."""
    
    def __init__(self, initial_workers: int = 4):
        """Initialize workload manager.
        
        Args:
            initial_workers: Initial number of worker threads
        """
        self.current_workers = initial_workers
        self.max_workers = psutil.cpu_count()
        self.min_workers = 1
        
        self.logger = logging.getLogger(__name__)
        
        # Task tracking
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.active_tasks = {}
        
        # Performance metrics
        self.metrics_history: List[WorkloadMetrics] = []
        self.max_history = 1000
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        
        self._metrics_lock = threading.Lock()
        
    def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            priority: Task priority (higher = more urgent)
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        if task_id is None:
            task_id = f"task_{time.time()}_{id(func)}"
        
        # Record task submission time
        task_info = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submit_time': time.time(),
            'priority': priority
        }
        
        # Submit to executor
        future = self.executor.submit(self._execute_task, task_info)
        
        with self._metrics_lock:
            self.active_tasks[task_id] = {
                'future': future,
                'info': task_info
            }
        
        return task_id
    
    def _execute_task(self, task_info: Dict[str, Any]) -> Any:
        """Execute a single task and track performance."""
        task_id = task_info['id']
        start_time = time.time()
        
        try:
            # Execute the function
            result = task_info['func'](*task_info['args'], **task_info['kwargs'])
            
            # Record successful completion
            end_time = time.time()
            execution_time = end_time - start_time
            
            self._record_task_completion(task_id, execution_time, success=True)
            
            return result
            
        except Exception as e:
            # Record failed completion
            end_time = time.time()
            execution_time = end_time - start_time
            
            self._record_task_completion(task_id, execution_time, success=False, error=str(e))
            
            raise
        finally:
            # Clean up task tracking
            with self._metrics_lock:
                self.active_tasks.pop(task_id, None)
    
    def _record_task_completion(
        self,
        task_id: str,
        execution_time: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record task completion metrics."""
        # This would be used to update workload metrics
        self.logger.debug(f"Task {task_id} completed in {execution_time:.3f}s, success: {success}")
    
    def get_workload_metrics(self) -> WorkloadMetrics:
        """Calculate current workload performance metrics."""
        with self._metrics_lock:
            active_count = len(self.active_tasks)
            
            # Calculate recent throughput and latency
            recent_completions = []  # Would track recent completions
            
            # Simple metrics calculation
            throughput = len(recent_completions)  # tasks per second (simplified)
            avg_latency = 0.1  # simplified
            error_rate = 0.0  # simplified
            
            return WorkloadMetrics(
                throughput=throughput,
                latency=avg_latency,
                queue_length=active_count,
                error_rate=error_rate
            )
    
    def scale_workers(self, target_workers: int):
        """Scale worker pool to target size.
        
        Args:
            target_workers: Target number of worker threads
        """
        target_workers = max(self.min_workers, min(target_workers, self.max_workers))
        
        if target_workers == self.current_workers:
            return
        
        self.logger.info(f"Scaling workers from {self.current_workers} to {target_workers}")
        
        # Create new executor with target size
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=target_workers)
        
        # Shutdown old executor gracefully
        old_executor.shutdown(wait=False)
        
        self.current_workers = target_workers
    
    def get_current_load(self) -> float:
        """Get current workload as a percentage of capacity.
        
        Returns:
            Load percentage (0.0 to 1.0+)
        """
        with self._metrics_lock:
            active_count = len(self.active_tasks)
            return active_count / self.current_workers


class AdaptiveScaler:
    """Adaptive scaling controller for analog PDE solver."""
    
    def __init__(
        self,
        initial_workers: int = 4,
        scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scaling_cooldown: float = 60.0
    ):
        """Initialize adaptive scaler.
        
        Args:
            initial_workers: Initial number of workers
            scaling_strategy: Scaling decision strategy
            scale_up_threshold: Resource utilization threshold for scaling up
            scale_down_threshold: Resource utilization threshold for scaling down
            scaling_cooldown: Minimum time between scaling decisions (seconds)
        """
        self.scaling_strategy = scaling_strategy
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_cooldown = scaling_cooldown
        
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.resource_monitor = ResourceMonitor()
        self.workload_manager = WorkloadManager(initial_workers)
        
        # Scaling history and state
        self.scaling_history: List[ScalingDecision] = []
        self.last_scaling_time = 0.0
        
        # Performance tracking for predictive scaling
        self.performance_history: List[Dict[str, Any]] = []
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
    
    def make_scaling_decision(self) -> ScalingDecision:
        """Make adaptive scaling decision based on current conditions.
        
        Returns:
            Scaling decision with rationale
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return ScalingDecision(
                action="maintain",
                target_workers=self.workload_manager.current_workers,
                confidence=1.0,
                reasoning=["Scaling cooldown period active"],
                expected_improvement=0.0
            )
        
        # Collect current metrics
        resource_metrics = self.resource_monitor.get_current_metrics()
        workload_metrics = self.workload_manager.get_workload_metrics()
        resource_trends = self.resource_monitor.get_resource_trends()
        
        if resource_metrics is None:
            return ScalingDecision(
                action="maintain",
                target_workers=self.workload_manager.current_workers,
                confidence=0.0,
                reasoning=["No resource metrics available"],
                expected_improvement=0.0
            )
        
        # Apply scaling strategy
        if self.scaling_strategy == ScalingStrategy.WORKLOAD_BASED:
            decision = self._workload_based_scaling(workload_metrics)
        elif self.scaling_strategy == ScalingStrategy.RESOURCE_BASED:
            decision = self._resource_based_scaling(resource_metrics, resource_trends)
        elif self.scaling_strategy == ScalingStrategy.PERFORMANCE_BASED:
            decision = self._performance_based_scaling(workload_metrics, resource_metrics)
        elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
            decision = self._predictive_scaling(resource_metrics, workload_metrics, resource_trends)
        else:  # HYBRID
            decision = self._hybrid_scaling(resource_metrics, workload_metrics, resource_trends)
        
        # Record decision
        self.scaling_history.append(decision)
        
        # Limit history size
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-50:]
        
        return decision
    
    def _workload_based_scaling(self, workload_metrics: WorkloadMetrics) -> ScalingDecision:
        """Make scaling decision based on workload characteristics."""
        current_workers = self.workload_manager.current_workers
        current_load = self.workload_manager.get_current_load()
        
        reasoning = [f"Current load: {current_load:.2%}"]
        
        if current_load > self.scale_up_threshold:
            # Scale up
            target_workers = min(current_workers + 2, self.workload_manager.max_workers)
            expected_improvement = (target_workers - current_workers) / current_workers * 0.7
            
            reasoning.append(f"High load ({current_load:.2%}) > threshold ({self.scale_up_threshold:.2%})")
            
            return ScalingDecision(
                action="scale_up",
                target_workers=target_workers,
                confidence=min(1.0, current_load),
                reasoning=reasoning,
                expected_improvement=expected_improvement
            )
        
        elif current_load < self.scale_down_threshold and current_workers > self.workload_manager.min_workers:
            # Scale down
            target_workers = max(current_workers - 1, self.workload_manager.min_workers)
            expected_improvement = 0.1  # Small efficiency gain
            
            reasoning.append(f"Low load ({current_load:.2%}) < threshold ({self.scale_down_threshold:.2%})")
            
            return ScalingDecision(
                action="scale_down",
                target_workers=target_workers,
                confidence=1.0 - current_load,
                reasoning=reasoning,
                expected_improvement=expected_improvement
            )
        
        else:
            return ScalingDecision(
                action="maintain",
                target_workers=current_workers,
                confidence=0.8,
                reasoning=reasoning + ["Load within acceptable range"],
                expected_improvement=0.0
            )
    
    def _resource_based_scaling(
        self,
        resource_metrics: ResourceMetrics,
        resource_trends: Dict[str, float]
    ) -> ScalingDecision:
        """Make scaling decision based on resource utilization."""
        current_workers = self.workload_manager.current_workers
        
        # Weighted resource utilization score
        cpu_weight = 0.4
        memory_weight = 0.3
        gpu_weight = 0.3
        
        resource_score = (
            cpu_weight * resource_metrics.cpu_usage +
            memory_weight * resource_metrics.memory_usage +
            gpu_weight * resource_metrics.gpu_usage
        ) / 100.0
        
        reasoning = [
            f"CPU: {resource_metrics.cpu_usage:.1f}%",
            f"Memory: {resource_metrics.memory_usage:.1f}%",
            f"GPU: {resource_metrics.gpu_usage:.1f}%",
            f"Resource score: {resource_score:.2%}"
        ]
        
        # Consider trends
        cpu_trend = resource_trends.get('cpu_trend', 0.0)
        memory_trend = resource_trends.get('memory_trend', 0.0)
        
        if cpu_trend > 5.0:  # CPU usage increasing
            reasoning.append(f"CPU trend: +{cpu_trend:.1f}%/min")
            resource_score += 0.1  # Boost score for increasing trends
        elif cpu_trend < -5.0:
            reasoning.append(f"CPU trend: {cpu_trend:.1f}%/min")
            resource_score -= 0.1
        
        # Make scaling decision
        if resource_score > self.scale_up_threshold:
            target_workers = min(current_workers + 1, self.workload_manager.max_workers)
            
            return ScalingDecision(
                action="scale_up",
                target_workers=target_workers,
                confidence=resource_score,
                reasoning=reasoning + ["High resource utilization"],
                expected_improvement=0.2
            )
        
        elif resource_score < self.scale_down_threshold and current_workers > self.workload_manager.min_workers:
            target_workers = max(current_workers - 1, self.workload_manager.min_workers)
            
            return ScalingDecision(
                action="scale_down",
                target_workers=target_workers,
                confidence=1.0 - resource_score,
                reasoning=reasoning + ["Low resource utilization"],
                expected_improvement=0.1
            )
        
        else:
            return ScalingDecision(
                action="maintain",
                target_workers=current_workers,
                confidence=0.7,
                reasoning=reasoning + ["Resource utilization acceptable"],
                expected_improvement=0.0
            )
    
    def _performance_based_scaling(
        self,
        workload_metrics: WorkloadMetrics,
        resource_metrics: ResourceMetrics
    ) -> ScalingDecision:
        """Make scaling decision based on performance metrics."""
        current_workers = self.workload_manager.current_workers
        
        # Performance indicators
        high_latency = workload_metrics.latency > 1.0  # 1 second threshold
        low_throughput = workload_metrics.throughput < current_workers * 0.5
        high_queue = workload_metrics.queue_length > current_workers * 2
        high_error_rate = workload_metrics.error_rate > 0.05
        
        performance_issues = sum([high_latency, low_throughput, high_queue, high_error_rate])
        
        reasoning = [
            f"Latency: {workload_metrics.latency:.3f}s",
            f"Throughput: {workload_metrics.throughput:.1f} ops/s",
            f"Queue length: {workload_metrics.queue_length}",
            f"Error rate: {workload_metrics.error_rate:.2%}",
            f"Performance issues: {performance_issues}/4"
        ]
        
        if performance_issues >= 2:
            # Performance degradation - scale up
            target_workers = min(current_workers + 2, self.workload_manager.max_workers)
            
            return ScalingDecision(
                action="scale_up",
                target_workers=target_workers,
                confidence=0.8,
                reasoning=reasoning + ["Multiple performance issues detected"],
                expected_improvement=0.3
            )
        
        elif performance_issues == 0 and resource_metrics.cpu_usage < 20.0:
            # Good performance with low resource usage - possible scale down
            if current_workers > self.workload_manager.min_workers:
                target_workers = max(current_workers - 1, self.workload_manager.min_workers)
                
                return ScalingDecision(
                    action="scale_down",
                    target_workers=target_workers,
                    confidence=0.6,
                    reasoning=reasoning + ["Good performance, low resource usage"],
                    expected_improvement=0.1
                )
        
        return ScalingDecision(
            action="maintain",
            target_workers=current_workers,
            confidence=0.7,
            reasoning=reasoning + ["Performance within acceptable range"],
            expected_improvement=0.0
        )
    
    def _predictive_scaling(
        self,
        resource_metrics: ResourceMetrics,
        workload_metrics: WorkloadMetrics,
        resource_trends: Dict[str, float]
    ) -> ScalingDecision:
        """Make predictive scaling decision based on trends."""
        # This would implement machine learning-based prediction
        # For now, use simple trend extrapolation
        
        current_workers = self.workload_manager.current_workers
        
        # Predict resource usage in next 5 minutes
        cpu_trend = resource_trends.get('cpu_trend', 0.0)
        predicted_cpu = resource_metrics.cpu_usage + cpu_trend * 5  # 5 minute prediction
        
        reasoning = [
            f"Current CPU: {resource_metrics.cpu_usage:.1f}%",
            f"CPU trend: {cpu_trend:.2f}%/min",
            f"Predicted CPU (5min): {predicted_cpu:.1f}%"
        ]
        
        if predicted_cpu > 80.0 and cpu_trend > 2.0:
            # Predict high resource usage
            target_workers = min(current_workers + 1, self.workload_manager.max_workers)
            
            return ScalingDecision(
                action="scale_up",
                target_workers=target_workers,
                confidence=0.7,
                reasoning=reasoning + ["Predicted resource constraint"],
                expected_improvement=0.25
            )
        
        elif predicted_cpu < 20.0 and cpu_trend < -2.0:
            # Predict low resource usage
            if current_workers > self.workload_manager.min_workers:
                target_workers = max(current_workers - 1, self.workload_manager.min_workers)
                
                return ScalingDecision(
                    action="scale_down",
                    target_workers=target_workers,
                    confidence=0.6,
                    reasoning=reasoning + ["Predicted low resource usage"],
                    expected_improvement=0.1
                )
        
        return ScalingDecision(
            action="maintain",
            target_workers=current_workers,
            confidence=0.5,
            reasoning=reasoning + ["No clear predictive signal"],
            expected_improvement=0.0
        )
    
    def _hybrid_scaling(
        self,
        resource_metrics: ResourceMetrics,
        workload_metrics: WorkloadMetrics,
        resource_trends: Dict[str, float]
    ) -> ScalingDecision:
        """Hybrid scaling combining multiple strategies."""
        # Get decisions from different strategies
        workload_decision = self._workload_based_scaling(workload_metrics)
        resource_decision = self._resource_based_scaling(resource_metrics, resource_trends)
        performance_decision = self._performance_based_scaling(workload_metrics, resource_metrics)
        
        # Voting system
        decisions = [workload_decision, resource_decision, performance_decision]
        scale_up_votes = sum(1 for d in decisions if d.action == "scale_up")
        scale_down_votes = sum(1 for d in decisions if d.action == "scale_down")
        maintain_votes = sum(1 for d in decisions if d.action == "maintain")
        
        # Combine reasoning
        all_reasoning = []
        for i, decision in enumerate(decisions):
            strategy_names = ["workload", "resource", "performance"]
            all_reasoning.extend([f"{strategy_names[i]}: {reason}" for reason in decision.reasoning])
        
        current_workers = self.workload_manager.current_workers
        
        if scale_up_votes > scale_down_votes and scale_up_votes > maintain_votes:
            # Majority says scale up
            target_workers = min(current_workers + 1, self.workload_manager.max_workers)
            confidence = scale_up_votes / len(decisions)
            expected_improvement = np.mean([d.expected_improvement for d in decisions if d.action == "scale_up"])
            
            return ScalingDecision(
                action="scale_up",
                target_workers=target_workers,
                confidence=confidence,
                reasoning=all_reasoning + [f"Hybrid decision: {scale_up_votes}/3 vote scale up"],
                expected_improvement=expected_improvement
            )
        
        elif scale_down_votes > maintain_votes and current_workers > self.workload_manager.min_workers:
            # Scale down
            target_workers = max(current_workers - 1, self.workload_manager.min_workers)
            confidence = scale_down_votes / len(decisions)
            expected_improvement = np.mean([d.expected_improvement for d in decisions if d.action == "scale_down"])
            
            return ScalingDecision(
                action="scale_down",
                target_workers=target_workers,
                confidence=confidence,
                reasoning=all_reasoning + [f"Hybrid decision: {scale_down_votes}/3 vote scale down"],
                expected_improvement=expected_improvement
            )
        
        else:
            # Maintain current state
            confidence = maintain_votes / len(decisions) if maintain_votes > 0 else 0.5
            
            return ScalingDecision(
                action="maintain",
                target_workers=current_workers,
                confidence=confidence,
                reasoning=all_reasoning + ["Hybrid decision: maintain current scale"],
                expected_improvement=0.0
            )
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision.
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            True if scaling was successful
        """
        if decision.action == "maintain":
            return True
        
        try:
            if decision.action in ["scale_up", "scale_down"]:
                self.workload_manager.scale_workers(decision.target_workers)
                self.last_scaling_time = time.time()
                
                self.logger.info(
                    f"Executed {decision.action} to {decision.target_workers} workers "
                    f"(confidence: {decision.confidence:.2%})"
                )
                
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
            return False
        
        return False
    
    def get_scaling_report(self) -> str:
        """Generate scaling report with recent decisions and performance."""
        if not self.scaling_history:
            return "No scaling decisions recorded yet."
        
        recent_decisions = self.scaling_history[-10:]  # Last 10 decisions
        
        report_lines = [
            "=" * 50,
            "ADAPTIVE SCALING REPORT",
            "=" * 50,
            f"Current Workers: {self.workload_manager.current_workers}",
            f"Scaling Strategy: {self.scaling_strategy.value}",
            ""
        ]
        
        # Recent decisions
        report_lines.append("Recent Scaling Decisions:")
        for decision in recent_decisions[-5:]:  # Last 5
            timestamp = time.strftime("%H:%M:%S", time.localtime(decision.timestamp))
            report_lines.append(
                f"  {timestamp}: {decision.action} to {decision.target_workers} workers "
                f"({decision.confidence:.1%} confidence)"
            )
        
        # Performance summary
        scale_ups = sum(1 for d in recent_decisions if d.action == "scale_up")
        scale_downs = sum(1 for d in recent_decisions if d.action == "scale_down")
        maintains = sum(1 for d in recent_decisions if d.action == "maintain")
        
        report_lines.extend([
            "",
            f"Decision Summary (last {len(recent_decisions)} decisions):",
            f"  Scale ups: {scale_ups}",
            f"  Scale downs: {scale_downs}",
            f"  Maintained: {maintains}",
            ""
        ])
        
        # Current metrics
        resource_metrics = self.resource_monitor.get_current_metrics()
        if resource_metrics:
            report_lines.extend([
                "Current Resource Usage:",
                f"  CPU: {resource_metrics.cpu_usage:.1f}%",
                f"  Memory: {resource_metrics.memory_usage:.1f}%",
                f"  GPU: {resource_metrics.gpu_usage:.1f}%",
                ""
            ])
        
        report_lines.extend([
            "=" * 50,
            "Report generated by Terragon Labs Adaptive Scaling Suite"
        ])
        
        return "\n".join(report_lines)
    
    def cleanup(self):
        """Cleanup resources."""
        self.resource_monitor.stop_monitoring()
        self.workload_manager.executor.shutdown(wait=True)
        self.logger.info("Adaptive scaler cleanup completed")