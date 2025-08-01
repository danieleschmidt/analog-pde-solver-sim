"""Performance metrics collection and analysis for benchmarking."""

import psutil
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    io_read_mb: float
    io_write_mb: float


class PerformanceMetrics:
    """Collect and analyze performance metrics during benchmark execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.snapshots: List[PerformanceSnapshot] = []
        self.baseline_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_io_stats(self) -> tuple[float, float]:
        """Get I/O read and write statistics in MB."""
        try:
            io_counters = self.process.io_counters()
            read_mb = io_counters.read_bytes / 1024 / 1024
            write_mb = io_counters.write_bytes / 1024 / 1024
            return read_mb, write_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return 0.0, 0.0
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot at current time."""
        memory_mb = self.get_memory_usage()
        cpu_percent = self.get_cpu_usage()
        io_read, io_write = self.get_io_stats()
        
        # Get system memory percentage
        system_memory = psutil.virtual_memory()
        memory_percent = (memory_mb / (system_memory.total / 1024 / 1024)) * 100
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            io_read_mb=io_read,
            io_write_mb=io_write
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_monitoring(self, interval: float = 0.1) -> 'PerformanceMonitor':
        """Start continuous performance monitoring."""
        return PerformanceMonitor(self, interval)
    
    def analyze_snapshots(self) -> Dict[str, Any]:
        """Analyze collected performance snapshots."""
        if not self.snapshots:
            return {"error": "No snapshots collected"}
        
        # Extract metrics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]
        memory_delta = [s.memory_mb - self.baseline_memory for s in self.snapshots]
        
        # Calculate statistics
        analysis = {
            "snapshot_count": len(self.snapshots),
            "duration": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "cpu_usage": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory_usage": {
                "peak_mb": np.max(memory_values),
                "mean_mb": np.mean(memory_values),
                "delta_mb": np.max(memory_delta),
                "baseline_mb": self.baseline_memory
            },
            "io_statistics": {
                "total_read_mb": self.snapshots[-1].io_read_mb - self.snapshots[0].io_read_mb,
                "total_write_mb": self.snapshots[-1].io_write_mb - self.snapshots[0].io_write_mb
            }
        }
        
        return analysis
    
    def clear_snapshots(self) -> None:
        """Clear collected snapshots."""
        self.snapshots.clear()
        self.baseline_memory = self.get_memory_usage()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            "platform": psutil.sys.platform
        }


class PerformanceMonitor:
    """Context manager for continuous performance monitoring."""
    
    def __init__(self, metrics: PerformanceMetrics, interval: float = 0.1):
        self.metrics = metrics
        self.interval = interval
        self.monitoring = False
        self._monitor_thread = None
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self) -> None:
        """Start continuous monitoring."""
        import threading
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            self.metrics.take_snapshot()
            time.sleep(self.interval)


class BenchmarkProfiler:
    """Advanced profiling for benchmark analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def profile_function(self, func_name: str):
        """Decorator to profile function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss
                
                profile_data = {
                    "function": func_name,
                    "execution_time": end_time - start_time,
                    "memory_delta": (end_memory - start_memory) / 1024 / 1024,  # MB
                    "success": success,
                    "error": error,
                    "timestamp": time.time()
                }
                
                self.profiles[func_name].append(profile_data)
                
                if not success:
                    raise
                    
                return result
            return wrapper
        return decorator
    
    def get_function_stats(self, func_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function."""
        if func_name not in self.profiles:
            return {"error": f"No data for function {func_name}"}
            
        data = self.profiles[func_name]
        successful_calls = [d for d in data if d["success"]]
        
        if not successful_calls:
            return {"error": f"No successful calls for {func_name}"}
            
        execution_times = [d["execution_time"] for d in successful_calls]
        memory_deltas = [d["memory_delta"] for d in successful_calls]
        
        return {
            "function": func_name,
            "call_count": len(data),
            "success_count": len(successful_calls),
            "success_rate": len(successful_calls) / len(data),
            "execution_time": {
                "mean": np.mean(execution_times),
                "median": np.median(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times)
            },
            "memory_delta": {
                "mean": np.mean(memory_deltas),
                "max": np.max(memory_deltas),
                "total": np.sum(memory_deltas)
            }
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all profiled functions."""
        return {
            func_name: self.get_function_stats(func_name)
            for func_name in self.profiles.keys()
        }
    
    def clear_profiles(self) -> None:
        """Clear all collected profiling data."""
        self.profiles.clear()


class MemoryProfiler:
    """Specialized memory usage profiler for large matrix operations."""
    
    def __init__(self):
        self.memory_timeline: List[tuple[float, float]] = []
        self.peak_memory = 0.0
        self.baseline_memory = 0.0
        
    def start_profiling(self) -> None:
        """Start memory profiling."""
        self.baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_timeline.clear()
        self.peak_memory = self.baseline_memory
        
    def record_memory_point(self, label: str = "") -> float:
        """Record a memory measurement point."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        timestamp = time.time()
        
        self.memory_timeline.append((timestamp, current_memory))
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return current_memory
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        if not self.memory_timeline:
            return {"error": "No memory data collected"}
            
        timestamps, memory_values = zip(*self.memory_timeline)
        memory_deltas = [m - self.baseline_memory for m in memory_values]
        
        return {
            "baseline_memory_mb": self.baseline_memory,
            "peak_memory_mb": self.peak_memory,
            "peak_delta_mb": self.peak_memory - self.baseline_memory,
            "final_memory_mb": memory_values[-1],
            "memory_growth_mb": memory_values[-1] - memory_values[0],
            "timeline_points": len(self.memory_timeline),
            "duration_seconds": timestamps[-1] - timestamps[0],
            "average_memory_mb": np.mean(memory_values),
            "memory_volatility": np.std(memory_deltas)
        }