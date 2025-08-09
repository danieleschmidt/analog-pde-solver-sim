"""Hardware validation and verification tools for analog PDE solver."""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class HardwareTestLevel(Enum):
    """Hardware validation test levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"


@dataclass
class HardwareValidationResult:
    """Result of hardware validation."""
    is_valid: bool
    reliability_score: float  # 0.0 to 1.0
    performance_metrics: Dict[str, float]
    compliance_results: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    test_level: HardwareTestLevel
    tests_passed: int
    tests_total: int
    validation_time: float
    
    def summary(self) -> str:
        """Generate hardware validation summary."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"""
Hardware Validation Summary
{status} (Reliability: {self.reliability_score:.2%})
Tests: {self.tests_passed}/{self.tests_total}
Level: {self.test_level.value.upper()}
Time: {self.validation_time:.2f}s
Errors: {len(self.errors)} | Warnings: {len(self.warnings)}
"""


class HardwareValidator:
    """Hardware validation and compliance checker for analog crossbar arrays."""
    
    def __init__(self, test_level: HardwareTestLevel = HardwareTestLevel.STANDARD):
        """Initialize hardware validator.
        
        Args:
            test_level: Level of hardware testing to perform
        """
        self.test_level = test_level
        self.logger = logging.getLogger(__name__)
        
        # Hardware limits and thresholds
        self.limits = self._get_hardware_limits(test_level)
        
    def _get_hardware_limits(self, level: HardwareTestLevel) -> Dict[str, Any]:
        """Get hardware validation limits for test level."""
        base_limits = {
            "max_power_mw": 1000.0,
            "max_temperature_c": 85.0,
            "min_efficiency": 0.1,
            "max_crossbar_size": 1024,
            "min_conductance_s": 1e-9,
            "max_conductance_s": 1e-3,
            "max_programming_voltage_v": 5.0,
            "min_snr_db": 20.0,
            "max_latency_ms": 100.0,
            "min_accuracy_bits": 8
        }
        
        if level == HardwareTestLevel.PRODUCTION:
            # Stricter limits for production
            base_limits.update({
                "max_power_mw": 500.0,
                "max_temperature_c": 70.0,
                "min_efficiency": 0.3,
                "min_snr_db": 30.0,
                "max_latency_ms": 50.0,
                "min_accuracy_bits": 10
            })
        elif level == HardwareTestLevel.BASIC:
            # Relaxed limits for basic testing
            base_limits.update({
                "max_power_mw": 2000.0,
                "max_temperature_c": 95.0,
                "min_efficiency": 0.05,
                "min_snr_db": 15.0,
                "max_latency_ms": 200.0,
                "min_accuracy_bits": 6
            })
        
        return base_limits
    
    def validate_hardware(
        self,
        crossbar_array,
        power_consumption_mw: Optional[float] = None,
        temperature_c: Optional[float] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        compliance_data: Optional[Dict[str, Any]] = None
    ) -> HardwareValidationResult:
        """Validate hardware implementation comprehensively.
        
        Args:
            crossbar_array: Crossbar array implementation to validate
            power_consumption_mw: Power consumption in milliwatts
            temperature_c: Operating temperature in Celsius
            performance_metrics: Performance metrics dictionary
            compliance_data: Regulatory compliance data
            
        Returns:
            Hardware validation result
        """
        start_time = time.time()
        self.logger.info(f"Starting hardware validation at {self.test_level.value} level")
        
        errors = []
        warnings = []
        perf_metrics = {}
        compliance_results = {}
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Crossbar Array Validation
        tests_total += 1
        if self._validate_crossbar_array(crossbar_array):
            tests_passed += 1
        else:
            errors.append("Crossbar array validation failed")
        
        # Test 2: Power Consumption Analysis
        if power_consumption_mw is not None:
            tests_total += 1
            power_valid, power_metrics = self._validate_power_consumption(power_consumption_mw)
            perf_metrics.update(power_metrics)
            
            if power_valid:
                tests_passed += 1
            else:
                errors.append(f"Power consumption exceeds limit: {power_consumption_mw:.1f}mW")
        
        # Test 3: Thermal Analysis
        if temperature_c is not None:
            tests_total += 1
            thermal_valid = self._validate_thermal_performance(temperature_c)
            perf_metrics["operating_temperature_c"] = temperature_c
            
            if thermal_valid:
                tests_passed += 1
            else:
                errors.append(f"Operating temperature too high: {temperature_c:.1f}°C")
        
        # Test 4: Conductance Range Validation
        tests_total += 1
        conductance_valid = self._validate_conductance_range(crossbar_array)
        if conductance_valid:
            tests_passed += 1
        else:
            warnings.append("Conductance range may be suboptimal")
        
        # Test 5: Signal Integrity
        tests_total += 1
        signal_metrics = self._validate_signal_integrity(crossbar_array)
        perf_metrics.update(signal_metrics)
        
        snr_db = signal_metrics.get("snr_db", 0)
        if snr_db >= self.limits["min_snr_db"]:
            tests_passed += 1
        else:
            errors.append(f"Poor signal-to-noise ratio: {snr_db:.1f}dB")
        
        # Test 6: Latency Performance
        tests_total += 1
        latency_ms = self._measure_latency(crossbar_array)
        perf_metrics["latency_ms"] = latency_ms
        
        if latency_ms <= self.limits["max_latency_ms"]:
            tests_passed += 1
        else:
            warnings.append(f"High latency detected: {latency_ms:.1f}ms")
        
        # Test 7: Accuracy Assessment
        tests_total += 1
        accuracy_bits = self._assess_computation_accuracy(crossbar_array)
        perf_metrics["effective_bits"] = accuracy_bits
        
        if accuracy_bits >= self.limits["min_accuracy_bits"]:
            tests_passed += 1
        else:
            errors.append(f"Insufficient computational accuracy: {accuracy_bits:.1f} bits")
        
        # Test 8: Device Variability
        if self.test_level in [HardwareTestLevel.COMPREHENSIVE, HardwareTestLevel.PRODUCTION]:
            tests_total += 1
            variability_score = self._assess_device_variability(crossbar_array)
            perf_metrics["variability_score"] = variability_score
            
            if variability_score >= 0.7:
                tests_passed += 1
            elif variability_score >= 0.5:
                warnings.append(f"Moderate device variability: {variability_score:.2f}")
            else:
                errors.append(f"High device variability: {variability_score:.2f}")
        
        # Test 9: Endurance Testing
        if self.test_level == HardwareTestLevel.PRODUCTION:
            tests_total += 1
            endurance_cycles = self._test_endurance(crossbar_array)
            perf_metrics["endurance_cycles"] = endurance_cycles
            
            if endurance_cycles >= 1e6:
                tests_passed += 1
            elif endurance_cycles >= 1e4:
                warnings.append(f"Limited endurance: {endurance_cycles:.0e} cycles")
            else:
                errors.append(f"Poor endurance: {endurance_cycles:.0e} cycles")
        
        # Test 10: Compliance Checks
        if compliance_data:
            compliance_results = self._validate_regulatory_compliance(compliance_data)
            tests_total += len(compliance_results)
            tests_passed += sum(compliance_results.values())
            
            failed_compliance = [k for k, v in compliance_results.items() if not v]
            if failed_compliance:
                errors.extend([f"Compliance failure: {std}" for std in failed_compliance])
        
        # Calculate overall validation result
        reliability_score = tests_passed / tests_total if tests_total > 0 else 0.0
        is_valid = len(errors) == 0 and reliability_score >= 0.8
        
        # Apply stricter criteria for production
        if self.test_level == HardwareTestLevel.PRODUCTION and reliability_score < 0.95:
            is_valid = False
        
        validation_time = time.time() - start_time
        
        result = HardwareValidationResult(
            is_valid=is_valid,
            reliability_score=reliability_score,
            performance_metrics=perf_metrics,
            compliance_results=compliance_results,
            warnings=warnings,
            errors=errors,
            test_level=self.test_level,
            tests_passed=tests_passed,
            tests_total=tests_total,
            validation_time=validation_time
        )
        
        self.logger.info(f"Hardware validation completed: {'PASSED' if is_valid else 'FAILED'} "
                        f"({reliability_score:.1%} reliability)")
        
        return result
    
    def _validate_crossbar_array(self, crossbar_array) -> bool:
        """Validate crossbar array structure and parameters."""
        try:
            if not hasattr(crossbar_array, 'rows') or not hasattr(crossbar_array, 'cols'):
                return False
            
            # Check size limits
            if crossbar_array.rows * crossbar_array.cols > self.limits["max_crossbar_size"]:
                return False
            
            # Check conductance arrays exist
            if not hasattr(crossbar_array, 'g_positive') or not hasattr(crossbar_array, 'g_negative'):
                return False
            
            # Validate conductance values
            g_pos = crossbar_array.g_positive
            g_neg = crossbar_array.g_negative
            
            if not isinstance(g_pos, np.ndarray) or not isinstance(g_neg, np.ndarray):
                return False
            
            if g_pos.shape != (crossbar_array.rows, crossbar_array.cols):
                return False
            
            if g_neg.shape != (crossbar_array.rows, crossbar_array.cols):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Crossbar validation failed: {e}")
            return False
    
    def _validate_power_consumption(self, power_mw: float) -> Tuple[bool, Dict[str, float]]:
        """Validate power consumption metrics."""
        metrics = {
            "power_consumption_mw": power_mw,
            "power_efficiency": min(1.0, 1000.0 / max(1.0, power_mw))  # Normalized efficiency
        }
        
        is_valid = power_mw <= self.limits["max_power_mw"]
        
        if power_mw > self.limits["max_power_mw"] * 0.8:
            metrics["power_warning"] = True
        
        return is_valid, metrics
    
    def _validate_thermal_performance(self, temperature_c: float) -> bool:
        """Validate thermal performance."""
        return temperature_c <= self.limits["max_temperature_c"]
    
    def _validate_conductance_range(self, crossbar_array) -> bool:
        """Validate conductance range is within specifications."""
        try:
            g_pos = crossbar_array.g_positive
            g_neg = crossbar_array.g_negative
            
            min_g = min(g_pos.min(), g_neg.min())
            max_g = max(g_pos.max(), g_neg.max())
            
            return (min_g >= self.limits["min_conductance_s"] and 
                   max_g <= self.limits["max_conductance_s"])
        except:
            return False
    
    def _validate_signal_integrity(self, crossbar_array) -> Dict[str, float]:
        """Validate signal integrity and compute SNR."""
        try:
            # Simulate a test pattern
            test_input = np.ones(crossbar_array.rows) * 0.5
            output = crossbar_array.compute_vmm(test_input)
            
            # Estimate SNR (simplified)
            signal_power = np.mean(output**2)
            noise_power = np.var(output) * 0.1  # Assume 10% of variance is noise
            
            if noise_power > 0:
                snr_ratio = signal_power / noise_power
                snr_db = 10 * np.log10(snr_ratio)
            else:
                snr_db = 60.0  # High SNR if no noise detected
            
            return {
                "snr_db": snr_db,
                "signal_power": signal_power,
                "noise_power": noise_power
            }
            
        except Exception as e:
            self.logger.warning(f"Signal integrity validation failed: {e}")
            return {"snr_db": 0.0, "signal_power": 0.0, "noise_power": float('inf')}
    
    def _measure_latency(self, crossbar_array) -> float:
        """Measure computational latency."""
        try:
            test_input = np.random.random(crossbar_array.rows)
            
            # Measure computation time
            start_time = time.time()
            
            # Run multiple iterations for accurate measurement
            for _ in range(100):
                _ = crossbar_array.compute_vmm(test_input)
            
            end_time = time.time()
            
            # Convert to milliseconds per operation
            latency_ms = (end_time - start_time) * 1000 / 100
            
            return latency_ms
            
        except Exception:
            return float('inf')
    
    def _assess_computation_accuracy(self, crossbar_array) -> float:
        """Assess effective computation accuracy in bits."""
        try:
            # Test with known patterns
            test_patterns = [
                np.ones(crossbar_array.rows),
                np.zeros(crossbar_array.rows),
                np.random.random(crossbar_array.rows),
                np.sin(np.linspace(0, 2*np.pi, crossbar_array.rows))
            ]
            
            errors = []
            
            for pattern in test_patterns:
                # Compute expected result (simplified)
                expected = np.dot(crossbar_array.g_positive.T, pattern) - \
                          np.dot(crossbar_array.g_negative.T, pattern)
                
                # Compute actual result
                actual = crossbar_array.compute_vmm(pattern)
                
                # Compute relative error
                rel_error = np.mean(np.abs(actual - expected) / (np.abs(expected) + 1e-12))
                errors.append(rel_error)
            
            # Estimate effective bits from average error
            avg_error = np.mean(errors)
            if avg_error > 0:
                effective_bits = -np.log2(avg_error)
            else:
                effective_bits = 16.0  # High precision
            
            return min(16.0, max(1.0, effective_bits))
            
        except Exception:
            return 1.0
    
    def _assess_device_variability(self, crossbar_array) -> float:
        """Assess device-to-device variability."""
        try:
            g_pos = crossbar_array.g_positive
            g_neg = crossbar_array.g_negative
            
            # Compute coefficient of variation for conductances
            cv_pos = np.std(g_pos) / (np.mean(g_pos) + 1e-12)
            cv_neg = np.std(g_neg) / (np.mean(g_neg) + 1e-12)
            
            avg_cv = (cv_pos + cv_neg) / 2
            
            # Convert to variability score (0 = high variability, 1 = low variability)
            variability_score = np.exp(-avg_cv * 5)  # Exponential decay
            
            return min(1.0, variability_score)
            
        except Exception:
            return 0.0
    
    def _test_endurance(self, crossbar_array) -> float:
        """Test endurance (simplified simulation)."""
        try:
            # Simulate endurance testing
            # In practice, this would involve repeated programming cycles
            
            # Estimate based on conductance stability
            g_pos = crossbar_array.g_positive
            g_stability = 1.0 - np.std(g_pos) / (np.mean(g_pos) + 1e-12)
            
            # Convert stability to estimated endurance cycles
            if g_stability > 0.9:
                endurance = 1e7  # High endurance
            elif g_stability > 0.7:
                endurance = 1e5  # Medium endurance
            else:
                endurance = 1e3   # Low endurance
            
            return endurance
            
        except Exception:
            return 1e2
    
    def _validate_regulatory_compliance(self, compliance_data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate regulatory compliance."""
        results = {}
        
        # EMC compliance
        if "emc_test_report" in compliance_data:
            results["EMC"] = compliance_data["emc_test_report"].get("passed", False)
        
        # Safety compliance
        if "safety_certification" in compliance_data:
            results["Safety"] = compliance_data["safety_certification"].get("certified", False)
        
        # Environmental compliance (RoHS, REACH)
        if "environmental_data" in compliance_data:
            env_data = compliance_data["environmental_data"]
            results["RoHS"] = env_data.get("rohs_compliant", True)
            results["REACH"] = env_data.get("reach_compliant", True)
        
        # Export control
        if "export_control" in compliance_data:
            results["Export_Control"] = compliance_data["export_control"].get("approved", True)
        
        return results
    
    def generate_hardware_report(self, result: HardwareValidationResult) -> str:
        """Generate comprehensive hardware validation report."""
        report_lines = [
            "=" * 60,
            "ANALOG PDE SOLVER - HARDWARE VALIDATION REPORT",
            "=" * 60,
            f"Test Level: {result.test_level.value.upper()}",
            f"Overall Result: {'✅ PASSED' if result.is_valid else '❌ FAILED'}",
            f"Reliability Score: {result.reliability_score:.1%}",
            f"Tests Passed: {result.tests_passed}/{result.tests_total}",
            f"Validation Time: {result.validation_time:.2f}s",
            ""
        ]
        
        # Performance metrics
        if result.performance_metrics:
            report_lines.append("📊 Performance Metrics:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric}: {value:.3f}")
                else:
                    report_lines.append(f"  {metric}: {value}")
            report_lines.append("")
        
        # Compliance results
        if result.compliance_results:
            report_lines.append("📋 Regulatory Compliance:")
            for standard, passed in result.compliance_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                report_lines.append(f"  {standard}: {status}")
            report_lines.append("")
        
        # Errors
        if result.errors:
            report_lines.append("❌ Critical Issues:")
            for error in result.errors:
                report_lines.append(f"  • {error}")
            report_lines.append("")
        
        # Warnings
        if result.warnings:
            report_lines.append("⚠️  Warnings:")
            for warning in result.warnings:
                report_lines.append(f"  • {warning}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("🔧 Recommendations:")
        if result.reliability_score < 0.8:
            report_lines.append("  • Hardware requires significant improvements for production use")
        if "power_consumption_mw" in result.performance_metrics:
            power = result.performance_metrics["power_consumption_mw"]
            if power > 500:
                report_lines.append(f"  • Consider power optimization (current: {power:.1f}mW)")
        if "snr_db" in result.performance_metrics:
            snr = result.performance_metrics["snr_db"]
            if snr < 25:
                report_lines.append(f"  • Improve signal integrity (current SNR: {snr:.1f}dB)")
        
        report_lines.extend([
            "",
            "=" * 60,
            "Report generated by Terragon Labs Hardware Validation Suite",
            "=" * 60
        ])
        
        return "\n".join(report_lines)