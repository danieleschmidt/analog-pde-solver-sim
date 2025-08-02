#!/usr/bin/env python3
"""
Health check endpoint for Analog PDE Solver.

Provides comprehensive health monitoring including:
- Application status
- Database connectivity
- External service dependencies
- System resources
- SPICE simulator availability
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import psutil
import requests


class HealthChecker:
    """Comprehensive health checker for Analog PDE Solver."""
    
    def __init__(self):
        self.start_time = time.time()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("health_check")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def check_application_health(self) -> Dict[str, Any]:
        """Check basic application health."""
        try:
            # Try to import the main package
            import analog_pde_solver
            
            return {
                "status": "healthy",
                "version": getattr(analog_pde_solver, '__version__', 'unknown'),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self.start_time
            }
        except ImportError as e:
            return {
                "status": "unhealthy",
                "error": f"Failed to import analog_pde_solver: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            return {
                "status": "healthy",
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": round((disk.used / disk.total) * 100, 1)
                },
                "cpu": {
                    "percent_used": cpu_percent,
                    "load_average": {
                        "1min": load_avg[0],
                        "5min": load_avg[1],
                        "15min": load_avg[2]
                    }
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Failed to check system resources: {e}"
            }
    
    def check_spice_simulator(self) -> Dict[str, Any]:
        """Check SPICE simulator availability."""
        try:
            import subprocess
            result = subprocess.run(
                ['ngspice', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0] if result.stdout else ""
                return {
                    "status": "healthy",
                    "version": version_line.strip(),
                    "path": subprocess.which('ngspice')
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "NgSpice command failed",
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "unhealthy",
                "error": "NgSpice command timed out"
            }
        except FileNotFoundError:
            return {
                "status": "unhealthy",
                "error": "NgSpice not found in PATH"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Unexpected error checking NgSpice: {e}"
            }
    
    def check_verilog_tools(self) -> Dict[str, Any]:
        """Check Verilog simulation tools availability."""
        tools = ['iverilog', 'verilator']
        tool_status = {}
        
        for tool in tools:
            try:
                import subprocess
                result = subprocess.run(
                    [tool, '--version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0] if result.stdout else ""
                    tool_status[tool] = {
                        "status": "healthy",
                        "version": version_line.strip(),
                        "path": subprocess.which(tool)
                    }
                else:
                    tool_status[tool] = {
                        "status": "unhealthy",
                        "error": f"{tool} command failed"
                    }
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                tool_status[tool] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        overall_status = "healthy" if all(
            status["status"] == "healthy" for status in tool_status.values()
        ) else "degraded"
        
        return {
            "status": overall_status,
            "tools": tool_status
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical Python package dependencies."""
        critical_packages = [
            'numpy', 'scipy', 'matplotlib', 'torch'
        ]
        
        package_status = {}
        
        for package in critical_packages:
            try:
                __import__(package)
                pkg = sys.modules[package]
                version = getattr(pkg, '__version__', 'unknown')
                package_status[package] = {
                    "status": "healthy",
                    "version": version
                }
            except ImportError as e:
                package_status[package] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        overall_status = "healthy" if all(
            status["status"] == "healthy" for status in package_status.values()
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "packages": package_status
        }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check critical file and directory permissions."""
        paths_to_check = [
            {'path': '/tmp', 'required': 'write'},
            {'path': './temp', 'required': 'write'},
            {'path': './logs', 'required': 'write'},
            {'path': '.', 'required': 'read'},
        ]
        
        permission_status = {}
        
        for path_info in paths_to_check:
            path = Path(path_info['path'])
            required = path_info['required']
            
            try:
                if not path.exists():
                    permission_status[str(path)] = {
                        "status": "unhealthy",
                        "error": "Path does not exist"
                    }
                    continue
                
                # Check read permission
                if required in ['read', 'write'] and not os.access(path, os.R_OK):
                    permission_status[str(path)] = {
                        "status": "unhealthy",
                        "error": "No read permission"
                    }
                    continue
                
                # Check write permission
                if required == 'write' and not os.access(path, os.W_OK):
                    permission_status[str(path)] = {
                        "status": "unhealthy",
                        "error": "No write permission"
                    }
                    continue
                
                permission_status[str(path)] = {
                    "status": "healthy",
                    "permissions": oct(path.stat().st_mode)[-3:]
                }
                
            except Exception as e:
                permission_status[str(path)] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        overall_status = "healthy" if all(
            status["status"] == "healthy" for status in permission_status.values()
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "paths": permission_status
        }
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        checks = {
            "application": self.check_application_health,
            "system_resources": self.check_system_resources,
            "spice_simulator": self.check_spice_simulator,
            "verilog_tools": self.check_verilog_tools,
            "dependencies": self.check_dependencies,
            "file_permissions": self.check_file_permissions,
        }
        
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in checks.items():
            try:
                self.logger.info(f"Running {check_name} check...")
                result = check_func()
                results[check_name] = result
                
                # Update overall status
                if result["status"] == "unhealthy":
                    overall_status = "unhealthy"
                elif result["status"] == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                self.logger.error(f"Error in {check_name} check: {e}")
                results[check_name] = {
                    "status": "unhealthy",
                    "error": f"Check failed with exception: {e}",
                    "traceback": traceback.format_exc()
                }
                overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results,
            "summary": {
                "healthy": sum(1 for r in results.values() if r["status"] == "healthy"),
                "degraded": sum(1 for r in results.values() if r["status"] == "degraded"),
                "unhealthy": sum(1 for r in results.values() if r["status"] == "unhealthy"),
                "total": len(results)
            }
        }


def main():
    """Main entry point for health check."""
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Simple health check (just application)
        result = checker.check_application_health()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "healthy" else 1)
    else:
        # Comprehensive health check
        result = checker.run_comprehensive_check()
        print(json.dumps(result, indent=2))
        
        # Exit with error code if not healthy
        if result["overall_status"] == "unhealthy":
            sys.exit(1)
        elif result["overall_status"] == "degraded":
            sys.exit(2)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()