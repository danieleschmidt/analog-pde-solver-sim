"""Hardware monitoring dashboard for analog PDE solver."""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class HardwareMetrics:
    """Hardware utilization metrics."""
    timestamp: float
    crossbar_utilization: float
    power_consumption: float
    memory_usage: float
    temperature: float
    throughput: float


class HardwareMonitorDashboard:
    """Real-time hardware monitoring dashboard."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize hardware monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics_history: List[HardwareMetrics] = []
        self.logger = logging.getLogger(__name__)
        
        # Check plotting availability
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            self.plt = plt
            self.animation = animation
            self._has_matplotlib = True
        except ImportError:
            self.logger.warning("Matplotlib not available, dashboard disabled")
            self._has_matplotlib = False
    
    def record_metrics(
        self,
        crossbar_utilization: float,
        power_consumption: float,
        memory_usage: float,
        temperature: float = 25.0,
        throughput: float = 0.0
    ):
        """Record new hardware metrics.
        
        Args:
            crossbar_utilization: Crossbar utilization percentage (0-100)
            power_consumption: Power consumption in mW
            memory_usage: Memory usage percentage (0-100)
            temperature: Temperature in Celsius
            throughput: Operations per second
        """
        metrics = HardwareMetrics(
            timestamp=time.time(),
            crossbar_utilization=crossbar_utilization,
            power_consumption=power_consumption,
            memory_usage=memory_usage,
            temperature=temperature,
            throughput=throughput
        )
        
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate hardware utilization report.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.metrics_history:
            return {"error": "No metrics recorded"}
        
        # Extract metric arrays
        crossbar_util = [m.crossbar_utilization for m in self.metrics_history]
        power = [m.power_consumption for m in self.metrics_history]
        memory = [m.memory_usage for m in self.metrics_history]
        temperature = [m.temperature for m in self.metrics_history]
        throughput = [m.throughput for m in self.metrics_history]
        
        report = {
            "recording_duration": self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
            "num_samples": len(self.metrics_history),
            "avg_utilization": np.mean(crossbar_util),
            "max_utilization": np.max(crossbar_util),
            "min_utilization": np.min(crossbar_util),
            "avg_power_mw": np.mean(power),
            "max_power_mw": np.max(power),
            "peak_power_mw": np.max(power),
            "total_energy_mj": np.sum(power) * (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp) / len(power) / 1000,
            "avg_memory_usage": np.mean(memory),
            "max_memory_usage": np.max(memory),
            "avg_temperature": np.mean(temperature),
            "max_temperature": np.max(temperature),
            "avg_throughput": np.mean(throughput),
            "peak_throughput": np.max(throughput)
        }
        
        return report
    
    def plot_real_time_dashboard(
        self,
        save_path: Optional[str] = None,
        window_size: int = 100
    ):
        """Plot real-time hardware dashboard.
        
        Args:
            save_path: Path to save dashboard plot (optional)
            window_size: Number of recent samples to show
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for dashboard")
            return None
        
        if not self.metrics_history:
            self.logger.warning("No metrics to plot")
            return None
        
        # Get recent metrics
        recent_metrics = self.metrics_history[-window_size:]
        
        # Extract data
        timestamps = [(m.timestamp - recent_metrics[0].timestamp) for m in recent_metrics]
        crossbar_util = [m.crossbar_utilization for m in recent_metrics]
        power = [m.power_consumption for m in recent_metrics]
        memory = [m.memory_usage for m in recent_metrics]
        temperature = [m.temperature for m in recent_metrics]
        throughput = [m.throughput for m in recent_metrics]
        
        # Create dashboard with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = self.plt.subplots(
            3, 2, figsize=(15, 12), dpi=100
        )
        
        # Crossbar utilization
        ax1.plot(timestamps, crossbar_util, 'b-', linewidth=2, label='Crossbar Utilization')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('Crossbar Utilization')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Power consumption
        ax2.plot(timestamps, power, 'r-', linewidth=2, label='Power')
        ax2.set_ylabel('Power (mW)')
        ax2.set_title('Power Consumption')
        ax2.grid(True, alpha=0.3)
        
        # Memory usage
        ax3.plot(timestamps, memory, 'g-', linewidth=2, label='Memory')
        ax3.set_ylabel('Memory Usage (%)')
        ax3.set_title('Memory Usage')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Temperature
        ax4.plot(timestamps, temperature, 'orange', linewidth=2, label='Temperature')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Temperature')
        ax4.grid(True, alpha=0.3)
        
        # Throughput
        ax5.plot(timestamps, throughput, 'purple', linewidth=2, label='Throughput')
        ax5.set_ylabel('Operations/sec')
        ax5.set_xlabel('Time (s)')
        ax5.set_title('Throughput')
        ax5.grid(True, alpha=0.3)
        
        # Summary statistics
        ax6.axis('off')
        report = self.generate_report()
        
        summary_text = f"""
        Hardware Utilization Summary
        
        Avg Crossbar Utilization: {report['avg_utilization']:.1f}%
        Max Crossbar Utilization: {report['max_utilization']:.1f}%
        
        Avg Power Consumption: {report['avg_power_mw']:.2f} mW
        Peak Power: {report['peak_power_mw']:.2f} mW
        Total Energy: {report['total_energy_mj']:.4f} mJ
        
        Avg Memory Usage: {report['avg_memory_usage']:.1f}%
        Max Memory Usage: {report['max_memory_usage']:.1f}%
        
        Avg Temperature: {report['avg_temperature']:.1f}°C
        Max Temperature: {report['max_temperature']:.1f}°C
        
        Avg Throughput: {report['avg_throughput']:.1f} ops/s
        Peak Throughput: {report['peak_throughput']:.1f} ops/s
        
        Recording Duration: {report['recording_duration']:.1f}s
        Samples: {report['num_samples']}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        self.plt.tight_layout()
        self.plt.suptitle('Analog PDE Solver - Hardware Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            try:
                fig.savefig(save_path, dpi=200, bbox_inches='tight')
                self.logger.info(f"Dashboard saved to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save dashboard: {e}")
        
        return fig
    
    def create_live_dashboard(self, update_interval: int = 1000):
        """Create live updating dashboard.
        
        Args:
            update_interval: Update interval in milliseconds
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for live dashboard")
            return None
        
        fig, ax = self.plt.subplots(figsize=(12, 8))
        
        def update_plot(frame):
            if not self.metrics_history:
                return
            
            ax.clear()
            
            # Get recent data
            recent = self.metrics_history[-50:]  # Last 50 samples
            timestamps = [(m.timestamp - recent[0].timestamp) for m in recent]
            utilization = [m.crossbar_utilization for m in recent]
            
            ax.plot(timestamps, utilization, 'b-', linewidth=2)
            ax.set_ylabel('Crossbar Utilization (%)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Live Crossbar Utilization')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Add current value annotation
            if utilization:
                current_util = utilization[-1]
                ax.annotate(f'Current: {current_util:.1f}%',
                           xy=(timestamps[-1], current_util),
                           xytext=(timestamps[-1] * 0.7, current_util + 10),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=12, color='red')
        
        anim = self.animation.FuncAnimation(
            fig, update_plot, interval=update_interval, blit=False, repeat=True
        )
        
        return anim, fig