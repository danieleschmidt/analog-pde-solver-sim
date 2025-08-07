"""Health monitoring and diagnostics for analog crossbar systems."""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path
from ..utils.logging_config import get_logger, MemoryMonitor


@dataclass
class HealthMetrics:
    """Container for system health metrics."""
    timestamp: float = field(default_factory=time.time)
    crossbar_health: float = 0.0
    solver_health: float = 0.0
    memory_usage_mb: float = 0.0
    error_rate: float = 0.0
    performance_score: float = 0.0
    convergence_rate: float = 0.0
    device_degradation: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'crossbar_health': self.crossbar_health,
            'solver_health': self.solver_health,
            'memory_usage_mb': self.memory_usage_mb,
            'error_rate': self.error_rate,
            'performance_score': self.performance_score,
            'convergence_rate': self.convergence_rate,
            'device_degradation': self.device_degradation
        }


class SystemHealthMonitor:
    """Comprehensive health monitoring for analog PDE solver systems."""
    
    def __init__(self, history_size: int = 1000, log_file: Optional[Path] = None):
        """Initialize health monitor.
        
        Args:
            history_size: Maximum number of historical metrics to keep
            log_file: Optional file path for health logs
        """
        self.history_size = history_size
        self.log_file = log_file
        self.logger = get_logger('health_monitor')
        
        # Metrics storage
        self.metrics_history: List[HealthMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = self._get_default_thresholds()
        
        # Performance tracking
        self.operation_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        self.logger.info("Health monitor initialized")
    
    def record_metrics(
        self,
        crossbar_stats: Dict[str, Any],
        solver_info: Dict[str, Any],
        performance_data: Optional[Dict[str, Any]] = None
    ) -> HealthMetrics:
        """Record comprehensive system metrics.
        
        Args:
            crossbar_stats: Crossbar device statistics
            solver_info: Solver convergence information
            performance_data: Optional performance measurements
            
        Returns:
            Computed health metrics
        """
        try:
            metrics = HealthMetrics()
            
            # Crossbar health assessment
            metrics.crossbar_health = self._assess_crossbar_health(crossbar_stats)
            
            # Solver health assessment
            metrics.solver_health = self._assess_solver_health(solver_info)
            
            # Memory usage
            try:
                import psutil
                metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            except ImportError:
                metrics.memory_usage_mb = 0.0
            
            # Error rate calculation
            self.operation_count += 1
            metrics.error_rate = self.error_count / self.operation_count if self.operation_count > 0 else 0.0
            
            # Performance score
            if performance_data:
                metrics.performance_score = self._calculate_performance_score(performance_data)
            
            # Convergence rate
            if solver_info.get('convergence_rate'):
                metrics.convergence_rate = solver_info['convergence_rate']
            
            # Device degradation
            metrics.device_degradation = self._estimate_device_degradation(crossbar_stats)
            
            # Store metrics
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.history_size:
                self.metrics_history.pop(0)
            
            # Check for alerts
            self._check_alert_conditions(metrics)
            
            # Log to file if specified
            if self.log_file:
                self._log_metrics_to_file(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to record metrics: {e}")
            self.error_count += 1
            return HealthMetrics()
    
    def _assess_crossbar_health(self, stats: Dict[str, Any]) -> float:
        """Assess crossbar device health (0-100 scale)."""
        if not stats.get('is_programmed', False):
            return 0.0
        
        health_score = 100.0
        
        # Programming errors penalty
        prog_errors = stats.get('programming_errors', 0)
        if prog_errors > 0:
            health_score -= min(50, prog_errors * 5)
        
        # Device degradation penalty
        if 'health_percentage' in stats:
            device_health = stats['health_percentage']
            health_score = min(health_score, device_health)
        
        # Stuck device penalty
        stuck_devices = stats.get('health_stuck_low_devices', 0) + stats.get('health_stuck_high_devices', 0)
        total_devices = stats.get('total_devices', 1)
        stuck_ratio = stuck_devices / total_devices
        health_score -= stuck_ratio * 30
        
        # Conductance variation penalty
        if 'g_positive_std' in stats and 'g_negative_std' in stats:
            variation = (stats['g_positive_std'] + stats['g_negative_std']) / 2
            if variation > 1e-7:  # High variation threshold
                health_score -= 20
        
        return max(0.0, min(100.0, health_score))
    
    def _assess_solver_health(self, info: Dict[str, Any]) -> float:
        """Assess solver health (0-100 scale)."""
        if info.get('status') != 'solved':
            return 0.0
        
        health_score = 100.0
        
        # Convergence penalty
        iterations = info.get('iterations', 0)
        if iterations > 500:  # High iteration count
            health_score -= 30
        elif iterations > 200:
            health_score -= 15
        
        # Error level penalty
        final_error = info.get('final_error', 1.0)
        if final_error > 1e-3:
            health_score -= 40
        elif final_error > 1e-4:
            health_score -= 20
        elif final_error > 1e-5:
            health_score -= 10
        
        # Convergence rate bonus
        conv_rate = info.get('convergence_rate', 0)
        if conv_rate > 0.1:
            health_score += 10
        elif conv_rate < 0.01:
            health_score -= 15
        
        return max(0.0, min(100.0, health_score))
    
    def _calculate_performance_score(self, perf_data: Dict[str, Any]) -> float:
        """Calculate performance score (0-100 scale)."""
        score = 50.0  # Baseline
        
        # Timing performance
        if 'solve_time' in perf_data:
            solve_time = perf_data['solve_time']
            if solve_time < 0.1:
                score += 25
            elif solve_time < 1.0:
                score += 15
            elif solve_time > 10.0:
                score -= 25
        
        # Memory efficiency
        if 'memory_efficiency' in perf_data:
            efficiency = perf_data['memory_efficiency']
            score += (efficiency - 0.5) * 50  # Scale around 50% efficiency
        
        # Accuracy bonus
        if 'accuracy' in perf_data:
            accuracy = perf_data['accuracy']
            if accuracy > 0.99:
                score += 20
            elif accuracy < 0.9:
                score -= 30
        
        return max(0.0, min(100.0, score))
    
    def _estimate_device_degradation(self, stats: Dict[str, Any]) -> float:
        """Estimate device degradation (0-100 scale, 0=new, 100=fully degraded)."""
        if not stats.get('is_programmed', False):
            return 0.0
        
        degradation = 0.0
        
        # Operation count based degradation
        op_count = stats.get('operation_count', 0)
        if op_count > 0:
            # Assume 1% degradation per 100k operations
            degradation += min(50, (op_count / 100000) * 1)
        
        # Stuck devices indicate degradation
        stuck_devices = stats.get('health_stuck_low_devices', 0) + stats.get('health_stuck_high_devices', 0)
        total_devices = stats.get('total_devices', 1)
        stuck_ratio = stuck_devices / total_devices
        degradation += stuck_ratio * 100
        
        # Programming errors indicate wear
        prog_errors = stats.get('programming_errors', 0)
        degradation += min(25, prog_errors * 2)
        
        return min(100.0, degradation)
    
    def _check_alert_conditions(self, metrics: HealthMetrics):
        """Check for alert conditions and generate alerts."""
        alerts_generated = []
        
        # Critical health thresholds
        if metrics.crossbar_health < self.thresholds['crossbar_health_critical']:
            alerts_generated.append({
                'level': 'CRITICAL',
                'type': 'crossbar_health',
                'message': f'Crossbar health critically low: {metrics.crossbar_health:.1f}%',
                'timestamp': metrics.timestamp,
                'value': metrics.crossbar_health
            })
        
        if metrics.solver_health < self.thresholds['solver_health_critical']:
            alerts_generated.append({
                'level': 'CRITICAL',
                'type': 'solver_health',
                'message': f'Solver health critically low: {metrics.solver_health:.1f}%',
                'timestamp': metrics.timestamp,
                'value': metrics.solver_health
            })
        
        # Memory alerts
        if metrics.memory_usage_mb > self.thresholds['memory_warning']:
            level = 'CRITICAL' if metrics.memory_usage_mb > self.thresholds['memory_critical'] else 'WARNING'
            alerts_generated.append({
                'level': level,
                'type': 'memory_usage',
                'message': f'High memory usage: {metrics.memory_usage_mb:.1f} MB',
                'timestamp': metrics.timestamp,
                'value': metrics.memory_usage_mb
            })
        
        # Error rate alerts
        if metrics.error_rate > self.thresholds['error_rate_warning']:
            level = 'CRITICAL' if metrics.error_rate > self.thresholds['error_rate_critical'] else 'WARNING'
            alerts_generated.append({
                'level': level,
                'type': 'error_rate',
                'message': f'High error rate: {metrics.error_rate:.2%}',
                'timestamp': metrics.timestamp,
                'value': metrics.error_rate
            })
        
        # Device degradation alerts
        if metrics.device_degradation > self.thresholds['degradation_warning']:
            level = 'CRITICAL' if metrics.device_degradation > self.thresholds['degradation_critical'] else 'WARNING'
            alerts_generated.append({
                'level': level,
                'type': 'device_degradation',
                'message': f'Device degradation: {metrics.device_degradation:.1f}%',
                'timestamp': metrics.timestamp,
                'value': metrics.device_degradation
            })
        
        # Add to alerts list and log
        for alert in alerts_generated:
            self.alerts.append(alert)
            if alert['level'] == 'CRITICAL':
                self.logger.error(alert['message'])
            else:
                self.logger.warning(alert['message'])
        
        # Limit alerts history
        if len(self.alerts) > self.history_size:
            self.alerts = self.alerts[-self.history_size:]
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds."""
        return {
            'crossbar_health_critical': 30.0,
            'solver_health_critical': 40.0,
            'memory_warning': 1024.0,  # MB
            'memory_critical': 4096.0,  # MB
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.15,  # 15%
            'degradation_warning': 50.0,  # %
            'degradation_critical': 80.0   # %
        }
    
    def _log_metrics_to_file(self, metrics: HealthMetrics):
        """Log metrics to file in JSON format."""
        if not self.log_file:
            return
        
        try:
            # Create directory if needed
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append metrics to file
            with open(self.log_file, 'a') as f:
                json.dump(metrics.to_dict(), f)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics to file: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-1]
        
        # Calculate trends over last 10 measurements
        trend_window = min(10, len(self.metrics_history))
        recent_history = self.metrics_history[-trend_window:]
        
        summary = {
            "timestamp": recent_metrics.timestamp,
            "overall_status": self._get_overall_status(recent_metrics),
            "current_metrics": recent_metrics.to_dict(),
            "trends": self._calculate_trends(recent_history),
            "active_alerts": [a for a in self.alerts if time.time() - a['timestamp'] < 3600],  # Last hour
            "system_uptime": time.time() - self.start_time,
            "total_operations": self.operation_count,
            "total_errors": self.error_count
        }
        
        return summary
    
    def _get_overall_status(self, metrics: HealthMetrics) -> str:
        """Determine overall system status."""
        if metrics.crossbar_health < 40 or metrics.solver_health < 40:
            return "CRITICAL"
        elif metrics.crossbar_health < 70 or metrics.solver_health < 70:
            return "WARNING"
        elif metrics.error_rate > 0.1:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def _calculate_trends(self, history: List[HealthMetrics]) -> Dict[str, str]:
        """Calculate trends for key metrics."""
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        # Simple trend calculation (last vs first in window)
        first = history[0]
        last = history[-1]
        
        trends = {}
        
        # Health trends
        if last.crossbar_health > first.crossbar_health * 1.05:
            trends['crossbar_health'] = 'improving'
        elif last.crossbar_health < first.crossbar_health * 0.95:
            trends['crossbar_health'] = 'degrading'
        else:
            trends['crossbar_health'] = 'stable'
        
        if last.solver_health > first.solver_health * 1.05:
            trends['solver_health'] = 'improving'
        elif last.solver_health < first.solver_health * 0.95:
            trends['solver_health'] = 'degrading'
        else:
            trends['solver_health'] = 'stable'
        
        # Memory trend
        if last.memory_usage_mb > first.memory_usage_mb * 1.1:
            trends['memory_usage'] = 'increasing'
        elif last.memory_usage_mb < first.memory_usage_mb * 0.9:
            trends['memory_usage'] = 'decreasing'
        else:
            trends['memory_usage'] = 'stable'
        
        return trends
    
    def export_metrics(self, filepath: Path, format: str = 'json') -> bool:
        """Export metrics history to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                data = [m.to_dict() for m in self.metrics_history]
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == 'csv':
                import csv
                if self.metrics_history:
                    with open(filepath, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].to_dict().keys())
                        writer.writeheader()
                        for metrics in self.metrics_history:
                            writer.writerow(metrics.to_dict())
            
            self.logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def clear_history(self, keep_recent: int = 100):
        """Clear metrics history, optionally keeping recent entries.
        
        Args:
            keep_recent: Number of recent entries to keep
        """
        if keep_recent > 0:
            self.metrics_history = self.metrics_history[-keep_recent:]
        else:
            self.metrics_history = []
        
        # Clear old alerts too
        current_time = time.time()
        self.alerts = [a for a in self.alerts if current_time - a['timestamp'] < 86400]  # Keep 24h
        
        self.logger.info(f"Cleared history, keeping {len(self.metrics_history)} recent entries")