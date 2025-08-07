"""Logging configuration for analog PDE solver."""

import logging
import sys
from typing import Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_colors: bool = True
) -> logging.Logger:
    """Setup logging configuration for the application.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file path for logging output
        enable_colors: Whether to use colored output for console
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('analog_pde_solver')
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Format strings
    detailed_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s'
    )
    simple_format = '%(levelname)s - %(message)s'
    
    if enable_colors and sys.stdout.isatty():
        formatter = ColoredFormatter(simple_format)
    else:
        formatter = logging.Formatter(simple_format)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always detailed for files
        
        file_formatter = logging.Formatter(detailed_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Add performance logger
    perf_logger = logging.getLogger('analog_pde_solver.performance')
    perf_logger.setLevel(logging.INFO)
    
    return logger


class PerformanceMonitor:
    """Context manager for performance monitoring."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """Initialize performance monitor.
        
        Args:
            operation_name: Name of operation being monitored
            logger: Logger instance to use
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger('analog_pde_solver.performance')
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        import time
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            
            if exc_type is None:
                self.logger.info(
                    f"{self.operation_name} completed in {elapsed:.4f}s"
                )
            else:
                self.logger.error(
                    f"{self.operation_name} failed after {elapsed:.4f}s: {exc_val}"
                )


class MemoryMonitor:
    """Memory usage monitoring utilities."""
    
    @staticmethod
    def log_memory_usage(logger: Optional[logging.Logger] = None):
        """Log current memory usage."""
        logger = logger or logging.getLogger('analog_pde_solver.memory')
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
            
            logger.info(f"Memory usage - RSS: {rss_mb:.1f} MB, VMS: {vms_mb:.1f} MB")
            
        except ImportError:
            logger.debug("psutil not available, cannot monitor memory")
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
    
    @staticmethod
    def check_available_memory(required_gb: float, logger: Optional[logging.Logger] = None):
        """Check if sufficient memory is available.
        
        Args:
            required_gb: Required memory in GB
            logger: Logger instance
            
        Returns:
            bool: True if sufficient memory available
        """
        logger = logger or logging.getLogger('analog_pde_solver.memory')
        
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / 1024**3
            
            if available_gb < required_gb:
                logger.warning(
                    f"Low memory: {available_gb:.1f} GB available, "
                    f"{required_gb:.1f} GB required"
                )
                return False
            else:
                logger.debug(
                    f"Memory OK: {available_gb:.1f} GB available, "
                    f"{required_gb:.1f} GB required"
                )
                return True
                
        except ImportError:
            logger.debug("psutil not available, cannot check memory")
            return True  # Assume OK if can't check
        except Exception as e:
            logger.warning(f"Failed to check memory: {e}")
            return True  # Assume OK on error


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the standard configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return logging.getLogger(f'analog_pde_solver.{name}')


# Initialize default logging
_default_logger = setup_logging()