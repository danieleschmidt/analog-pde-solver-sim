"""Intelligent auto-scaling system for analog computing workloads."""

import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging
import multiprocessing as mp


@dataclass
class WorkloadMetrics:
    """Workload performance metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    solve_time: float
    queue_depth: int
    error_rate: float
    throughput: float
    
    def get_load_score(self) -> float:
        """Compute overall load score (0-100)."""
        return (
            self.cpu_utilization * 0.3 +
            self.memory_utilization * 0.3 +
            min(100, self.queue_depth * 10) * 0.2 +
            min(100, self.solve_time * 100) * 0.2
        )


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    timestamp: float
    action: str  # "scale_up", "scale_down", "no_action"
    reason: str
    current_workers: int
    target_workers: int
    confidence: float


class AutoScalingSystem:
    """Intelligent auto-scaling for analog PDE solver workloads."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = None):
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.current_workers = min_workers
        
        self.logger = logging.getLogger(__name__)
        
        # Metrics collection
        self.metrics_window_size = 60  # Keep 1 hour of metrics
        self.metrics_history: deque = deque(maxlen=self.metrics_window_size)
        
        # Scaling parameters
        self.scale_up_threshold = 75.0    # Load percentage
        self.scale_down_threshold = 25.0  # Load percentage
        self.scaling_cooldown = 300       # 5 minutes between scaling events
        self.last_scaling_time = 0
        
        # Worker management
        self.worker_pool = []
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # Scaling history
        self.scaling_decisions: List[ScalingDecision] = []
        
        # Performance tracking
        self.performance_baseline = None
        self.adaptive_thresholds = True
        
        self.logger.info(f"Auto-scaling initialized: {min_workers}-{max_workers} workers")
    
    def record_workload_metrics(
        self,
        cpu_util: float,
        memory_util: float,
        solve_time: float,
        queue_depth: int,
        error_rate: float,
        throughput: float
    ):
        """Record current workload metrics."""
        metrics = WorkloadMetrics(
            timestamp=time.time(),
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            solve_time=solve_time,
            queue_depth=queue_depth,
            error_rate=error_rate,
            throughput=throughput
        )
        
        self.metrics_history.append(metrics)
        
        # Trigger scaling decision if enough data
        if len(self.metrics_history) >= 5:
            self._evaluate_scaling_decision()
    
    def _evaluate_scaling_decision(self):
        """Evaluate whether scaling action is needed."""
        try:
            # Check cooldown
            if time.time() - self.last_scaling_time < self.scaling_cooldown:
                return
            
            # Analyze recent metrics
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            
            # Calculate average load
            avg_load = np.mean([m.get_load_score() for m in recent_metrics])
            
            # Calculate load trend
            if len(recent_metrics) >= 5:
                early_load = np.mean([m.get_load_score() for m in recent_metrics[:5]])
                late_load = np.mean([m.get_load_score() for m in recent_metrics[-5:]])
                load_trend = late_load - early_load
            else:
                load_trend = 0
            
            # Adaptive threshold adjustment
            if self.adaptive_thresholds:
                self._adjust_thresholds(recent_metrics)
            
            # Make scaling decision
            decision = self._make_scaling_decision(avg_load, load_trend, recent_metrics)
            
            if decision.action != "no_action":
                self.scaling_decisions.append(decision)
                self._execute_scaling_decision(decision)
            
        except Exception as e:
            self.logger.error(f"Scaling evaluation failed: {e}")
    
    def _make_scaling_decision(
        self, 
        avg_load: float, 
        load_trend: float, 
        metrics: List[WorkloadMetrics]
    ) -> ScalingDecision:
        """Make intelligent scaling decision."""
        
        current_time = time.time()
        
        # Check for scale-up conditions
        if avg_load > self.scale_up_threshold and self.current_workers < self.max_workers:
            confidence = min(0.95, (avg_load - self.scale_up_threshold) / 25.0)
            
            # Higher confidence if load is trending up
            if load_trend > 5:
                confidence += 0.1
            
            # Higher confidence if error rate is increasing
            recent_error_rate = np.mean([m.error_rate for m in metrics[-3:]])
            if recent_error_rate > 0.05:
                confidence += 0.15
            
            target_workers = min(self.max_workers, self.current_workers + 1)
            
            # Intelligent scale-up: more aggressive under high load
            if avg_load > 90:
                target_workers = min(self.max_workers, self.current_workers + 2)
            
            return ScalingDecision(
                timestamp=current_time,
                action="scale_up",
                reason=f"High load: {avg_load:.1f}%, trend: {load_trend:+.1f}%",
                current_workers=self.current_workers,
                target_workers=target_workers,
                confidence=confidence
            )
        
        # Check for scale-down conditions
        elif avg_load < self.scale_down_threshold and self.current_workers > self.min_workers:
            confidence = min(0.9, (self.scale_down_threshold - avg_load) / 25.0)
            
            # Higher confidence if load is trending down
            if load_trend < -5:
                confidence += 0.1
            
            # Only scale down if consistently low load
            if len(metrics) >= 10:
                all_low = all(m.get_load_score() < self.scale_down_threshold * 1.2 for m in metrics[-10:])
                if all_low:
                    confidence += 0.2
            
            target_workers = max(self.min_workers, self.current_workers - 1)
            
            return ScalingDecision(
                timestamp=current_time,
                action="scale_down",
                reason=f"Low load: {avg_load:.1f}%, trend: {load_trend:+.1f}%",
                current_workers=self.current_workers,
                target_workers=target_workers,
                confidence=confidence
            )
        
        # No scaling needed
        return ScalingDecision(
            timestamp=current_time,
            action="no_action",
            reason=f"Load within bounds: {avg_load:.1f}%",
            current_workers=self.current_workers,
            target_workers=self.current_workers,
            confidence=1.0
        )
    
    def _adjust_thresholds(self, metrics: List[WorkloadMetrics]):
        """Adaptively adjust scaling thresholds based on performance."""
        if len(metrics) < 10:
            return
        
        # Analyze performance at different load levels
        loads = [m.get_load_score() for m in metrics]
        solve_times = [m.solve_time for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        
        # Find optimal load range where performance is good
        good_performance_mask = (
            (np.array(solve_times) < np.percentile(solve_times, 75)) &
            (np.array(error_rates) < 0.02)
        )
        
        if np.any(good_performance_mask):
            optimal_loads = np.array(loads)[good_performance_mask]
            
            # Adjust thresholds towards optimal range
            if len(optimal_loads) > 3:
                optimal_max = np.percentile(optimal_loads, 90)
                optimal_min = np.percentile(optimal_loads, 10)
                
                # Gradual adjustment
                adjustment_rate = 0.1
                self.scale_up_threshold = (
                    self.scale_up_threshold * (1 - adjustment_rate) +
                    optimal_max * adjustment_rate
                )
                self.scale_down_threshold = (
                    self.scale_down_threshold * (1 - adjustment_rate) +
                    optimal_min * adjustment_rate
                )
                
                self.logger.debug(
                    f"Adjusted thresholds: up={self.scale_up_threshold:.1f}%, "
                    f"down={self.scale_down_threshold:.1f}%"
                )
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        if decision.confidence < 0.7:
            self.logger.debug(f"Skipping scaling due to low confidence: {decision.confidence:.2f}")
            return
        
        try:
            if decision.action == "scale_up":
                self._scale_up(decision.target_workers)
            elif decision.action == "scale_down":
                self._scale_down(decision.target_workers)
            
            self.last_scaling_time = time.time()
            self.logger.info(f"Scaling executed: {decision.action} to {decision.target_workers} workers")
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
    
    def _scale_up(self, target_workers: int):
        """Scale up worker count."""
        workers_to_add = target_workers - self.current_workers
        
        for _ in range(workers_to_add):
            worker = mp.Process(target=self._worker_process, daemon=True)
            worker.start()
            self.worker_pool.append(worker)
        
        self.current_workers = target_workers
        self.logger.info(f"Scaled up to {self.current_workers} workers")
    
    def _scale_down(self, target_workers: int):
        """Scale down worker count."""
        workers_to_remove = self.current_workers - target_workers
        
        # Gracefully terminate workers
        for _ in range(workers_to_remove):
            if self.worker_pool:
                worker = self.worker_pool.pop()
                worker.terminate()
                worker.join(timeout=5)
        
        self.current_workers = target_workers
        self.logger.info(f"Scaled down to {self.current_workers} workers")
    
    def _worker_process(self):
        """Worker process for handling solve tasks."""
        while True:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1)
                
                # Process task
                result = self._process_task(task)
                
                # Put result
                self.result_queue.put(result)
                
            except Exception as e:
                self.logger.error(f"Worker process error: {e}")
                break
    
    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual solve task."""
        # This would contain the actual task processing logic
        # For now, simulate some work
        start_time = time.time()
        
        # Simulate computation
        time.sleep(np.random.uniform(0.1, 1.0))
        
        processing_time = time.time() - start_time
        
        return {
            'task_id': task.get('id'),
            'result': 'completed',
            'processing_time': processing_time,
            'worker_id': mp.current_process().pid
        }
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'scaling_decisions': len(self.scaling_decisions),
            'last_scaling': self.last_scaling_time,
            'metrics_collected': len(self.metrics_history),
            'adaptive_thresholds': self.adaptive_thresholds
        }


class LoadBalancer:
    """Intelligent load balancer for analog computing tasks."""
    
    def __init__(self, scaling_system: AutoScalingSystem):
        self.scaling_system = scaling_system
        self.logger = logging.getLogger(__name__)
        
        # Load balancing strategies
        self.strategies = {
            'round_robin': self._round_robin_balance,
            'least_loaded': self._least_loaded_balance,
            'performance_aware': self._performance_aware_balance
        }
        
        self.current_strategy = 'performance_aware'
        self.worker_stats = {}
        
    def distribute_task(self, task: Dict[str, Any]) -> str:
        """Distribute task to optimal worker."""
        strategy = self.strategies[self.current_strategy]
        worker_id = strategy(task)
        
        # Update worker stats
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = {
                'tasks_assigned': 0,
                'avg_processing_time': 0.0,
                'last_assignment': 0
            }
        
        self.worker_stats[worker_id]['tasks_assigned'] += 1
        self.worker_stats[worker_id]['last_assignment'] = time.time()
        
        return worker_id
    
    def _round_robin_balance(self, task: Dict[str, Any]) -> str:
        """Simple round-robin load balancing."""
        worker_count = self.scaling_system.current_workers
        worker_id = hash(str(time.time())) % worker_count
        return f"worker_{worker_id}"
    
    def _least_loaded_balance(self, task: Dict[str, Any]) -> str:
        """Balance to least loaded worker."""
        if not self.worker_stats:
            return "worker_0"
        
        # Find worker with least recent tasks
        least_loaded = min(
            self.worker_stats.items(),
            key=lambda x: x[1]['tasks_assigned']
        )
        
        return least_loaded[0]
    
    def _performance_aware_balance(self, task: Dict[str, Any]) -> str:
        """Balance based on worker performance characteristics."""
        if not self.worker_stats:
            return "worker_0"
        
        # Consider both load and performance
        best_worker = None
        best_score = float('inf')
        
        for worker_id, stats in self.worker_stats.items():
            # Score based on recent load and average processing time
            load_score = stats['tasks_assigned'] / max(1, time.time() - stats.get('last_assignment', 0))
            time_score = stats.get('avg_processing_time', 1.0)
            
            combined_score = load_score * 0.6 + time_score * 0.4
            
            if combined_score < best_score:
                best_score = combined_score
                best_worker = worker_id
        
        return best_worker or "worker_0"
    
    def update_worker_performance(self, worker_id: str, processing_time: float):
        """Update worker performance statistics."""
        if worker_id in self.worker_stats:
            current_avg = self.worker_stats[worker_id]['avg_processing_time']
            task_count = self.worker_stats[worker_id]['tasks_assigned']
            
            # Update running average
            new_avg = ((current_avg * (task_count - 1)) + processing_time) / task_count
            self.worker_stats[worker_id]['avg_processing_time'] = new_avg
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'current_strategy': self.current_strategy,
            'worker_count': len(self.worker_stats),
            'total_tasks_distributed': sum(stats['tasks_assigned'] for stats in self.worker_stats.values()),
            'worker_performance': self.worker_stats
        }


class PerformanceOptimizer:
    """AI-driven performance optimization system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Optimization history
        self.optimization_history = []
        
        # Performance models
        self.performance_models = {
            'crossbar_size_optimization': self._optimize_crossbar_size,
            'convergence_optimization': self._optimize_convergence_params,
            'memory_optimization': self._optimize_memory_usage
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        
    def optimize_parameters(
        self, 
        current_params: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize parameters based on performance feedback."""
        
        optimized_params = current_params.copy()
        
        try:
            # Apply optimization models
            for model_name, model_func in self.performance_models.items():
                param_updates = model_func(current_params, performance_metrics)
                optimized_params.update(param_updates)
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': time.time(),
                'original_params': current_params,
                'optimized_params': optimized_params,
                'performance_before': performance_metrics
            })
            
            self.logger.info(f"Parameters optimized using {len(self.performance_models)} models")
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
        
        return optimized_params
    
    def _optimize_crossbar_size(
        self, 
        params: Dict[str, Any], 
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize crossbar size based on performance."""
        current_size = params.get('crossbar_size', 128)
        solve_time = metrics.get('solve_time', 1.0)
        memory_usage = metrics.get('memory_usage_mb', 100)
        accuracy = metrics.get('accuracy', 0.95)
        
        # If solving too slow, consider smaller crossbar
        if solve_time > 5.0 and current_size > 64:
            new_size = max(64, int(current_size * 0.8))
            return {'crossbar_size': new_size}
        
        # If very fast but low accuracy, consider larger crossbar  
        if solve_time < 0.5 and accuracy < 0.9 and current_size < 512:
            new_size = min(512, int(current_size * 1.2))
            return {'crossbar_size': new_size}
        
        return {}
    
    def _optimize_convergence_params(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize convergence parameters."""
        current_threshold = params.get('convergence_threshold', 1e-6)
        current_iterations = params.get('iterations', 100)
        
        solve_time = metrics.get('solve_time', 1.0)
        accuracy = metrics.get('accuracy', 0.95)
        convergence_rate = metrics.get('convergence_rate', 0.01)
        
        updates = {}
        
        # Adjust threshold based on accuracy requirements
        if accuracy > 0.99 and solve_time > 2.0:
            # Relax threshold for speed
            new_threshold = min(1e-4, current_threshold * 2)
            updates['convergence_threshold'] = new_threshold
        elif accuracy < 0.9:
            # Tighten threshold for accuracy
            new_threshold = max(1e-8, current_threshold * 0.5)
            updates['convergence_threshold'] = new_threshold
        
        # Adjust iterations based on convergence behavior
        if convergence_rate > 0.1 and current_iterations > 50:
            # Fast convergence, reduce iterations
            new_iterations = max(50, int(current_iterations * 0.8))
            updates['iterations'] = new_iterations
        elif convergence_rate < 0.01 and current_iterations < 500:
            # Slow convergence, increase iterations
            new_iterations = min(500, int(current_iterations * 1.5))
            updates['iterations'] = new_iterations
        
        return updates
    
    def _optimize_memory_usage(
        self,
        params: Dict[str, Any], 
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize memory usage patterns."""
        memory_usage = metrics.get('memory_usage_mb', 100)
        
        # If memory usage is very high, suggest memory-saving options
        if memory_usage > 1000:  # 1GB
            return {
                'use_memory_optimization': True,
                'batch_size_reduction': 0.5
            }
        
        return {}
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization performance report."""
        if not self.optimization_history:
            return {"status": "no_optimizations_performed"}
        
        recent_optimizations = self.optimization_history[-10:]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': len(recent_optimizations),
            'optimization_models': list(self.performance_models.keys()),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate
        }


# Global instances
_auto_scaling_system = None
_performance_optimizer = None


def get_auto_scaling_system() -> AutoScalingSystem:
    """Get global auto-scaling system."""
    global _auto_scaling_system
    if _auto_scaling_system is None:
        _auto_scaling_system = AutoScalingSystem()
    return _auto_scaling_system


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer