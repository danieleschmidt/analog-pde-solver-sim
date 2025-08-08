"""Structured logging utilities for analog PDE solver."""

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record with structured data."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, 'custom_fields'):
            log_entry.update(record.custom_fields)
            
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._start_times: Dict[str, float] = {}
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self._start_times[operation] = time.perf_counter()
        self.logger.debug(f"Started timing: {operation}")
        
    def end_timer(self, operation: str, extra_data: Optional[Dict] = None) -> float:
        """End timing and log duration."""
        import time
        if operation not in self._start_times:
            self.logger.warning(f"Timer '{operation}' was not started")
            return 0.0
            
        duration = time.perf_counter() - self._start_times[operation]
        del self._start_times[operation]
        
        log_data = {
            'operation': operation,
            'duration_ms': duration * 1000,
            'duration_s': duration
        }
        
        if extra_data:
            log_data.update(extra_data)
            
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Operation '{operation}' completed in {duration:.3f}s",
            args=(),
            exc_info=None
        )
        record.custom_fields = {'performance_metrics': log_data}
        
        self.logger.handle(record)
        return duration


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger instance
    """
    # Get log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
    
    # Get application logger
    logger = logging.getLogger('analog_pde_solver')
    logger.info(f"Logging initialized at level {level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f'analog_pde_solver.{name}')


def log_system_info():
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger = get_logger('system')
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count() if 'psutil' in sys.modules else 'unknown',
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2) if 'psutil' in sys.modules else 'unknown'
    }
    
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="System information",
        args=(),
        exc_info=None
    )
    record.custom_fields = {'system_info': system_info}
    
    logger.handle(record)


# Create default logger
default_logger = setup_logging(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    structured=os.getenv('LOG_STRUCTURED', 'true').lower() == 'true',
    log_file=os.getenv('LOG_FILE')
)