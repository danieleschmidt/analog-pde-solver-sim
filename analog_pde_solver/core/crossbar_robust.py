"""Enhanced analog crossbar array with comprehensive device modeling."""

import numpy as np
from typing import Tuple, Optional
import logging
from ..utils.logging_config import get_logger


class RobustAnalogCrossbarArray:
    """Enhanced analog crossbar array with realistic device modeling and error handling."""
    
    def __init__(
        self, 
        rows: int, 
        cols: int, 
        cell_type: str = "1T1R",
        noise_model: str = "realistic"
    ):
        """Initialize enhanced crossbar array.
        
        Args:
            rows: Number of rows (input size)
            cols: Number of columns (output size)  
            cell_type: Crossbar cell type ('1T1R', 'ReRAM', 'PCM')
            noise_model: Noise modeling approach
        """
        self.rows = rows
        self.cols = cols
        self.cell_type = cell_type
        self.noise_model = noise_model
        
        # Initialize conductance matrices
        self.g_positive = np.zeros((rows, cols), dtype=np.float64)
        self.g_negative = np.zeros((rows, cols), dtype=np.float64)
        
        # Device parameters
        self.device_params = self._get_device_parameters()
        
        # Initialize logger
        self.logger = get_logger('crossbar')
        
        # State tracking
        self.is_programmed = False
        self.programming_errors = 0
        self.operation_count = 0
        
        self.logger.info(
            f"Initialized {rows}x{cols} crossbar array "
            f"(cell_type={cell_type}, noise_model={noise_model})"
        )
    
    def program_conductances(self, target_matrix: np.ndarray) -> None:
        """Map target matrix to positive/negative conductance pairs with validation."""
        try:
            self.logger.debug(f"Programming {target_matrix.shape} matrix to crossbar")
            
            # Validate input matrix
            if target_matrix.shape != (self.rows, self.cols):
                raise ValueError(
                    f"Matrix shape {target_matrix.shape} does not match "
                    f"crossbar size {(self.rows, self.cols)}"
                )
            
            if not np.isfinite(target_matrix).all():
                raise ValueError("Target matrix contains NaN or infinity")
            
            g_min, g_max = self.device_params['g_range']
            
            # Decompose into positive and negative components
            pos_matrix = np.maximum(target_matrix, 0)
            neg_matrix = np.maximum(-target_matrix, 0)
            
            # Scale to conductance range with error handling
            self.g_positive = self._scale_to_conductance(pos_matrix, g_min, g_max)
            self.g_negative = self._scale_to_conductance(neg_matrix, g_min, g_max)
            
            # Add programming variations
            self._add_programming_variations()
            
            self.is_programmed = True
            self.programming_errors = 0
            
            self.logger.info(
                f"Successfully programmed crossbar with matrix range "
                f"[{target_matrix.min():.2e}, {target_matrix.max():.2e}]"
            )
            
        except Exception as e:
            self.programming_errors += 1
            self.logger.error(f"Programming failed: {e}")
            raise
    
    def compute_vmm(self, input_vector: np.ndarray) -> np.ndarray:
        """Analog vector-matrix multiplication with comprehensive error handling."""
        try:
            # Validate inputs
            if not self.is_programmed:
                raise RuntimeError("Crossbar not programmed")
            
            if len(input_vector) != self.rows:
                raise ValueError(
                    f"Input vector length {len(input_vector)} does not match "
                    f"crossbar rows {self.rows}"
                )
            
            if not np.isfinite(input_vector).all():
                raise ValueError("Input vector contains NaN or infinity")
            
            # Clamp input voltage range for realistic operation
            input_clamped = np.clip(input_vector, -1.0, 1.0)
            
            # Ohm's law: I = G × V with error handling
            try:
                i_pos = np.dot(self.g_positive.T, input_clamped)
                i_neg = np.dot(self.g_negative.T, input_clamped)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Matrix multiplication failed: {e}")
            
            # Differential current sensing
            output_current = i_pos - i_neg
            
            # Add device noise based on model
            noise = self._compute_noise(output_current)
            output_with_noise = output_current + noise
            
            # Add device non-linearities
            output_final = self._apply_device_nonlinearities(output_with_noise)
            
            # Check for numerical issues
            if not np.isfinite(output_final).all():
                self.logger.warning("VMM output contains non-finite values")
                output_final = np.nan_to_num(output_final, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Update operation counter
            self.operation_count += 1
            
            # Apply device aging effects
            if self.operation_count % 1000 == 0:
                self._apply_aging_effects()
            
            return output_final
            
        except Exception as e:
            self.logger.error(f"VMM computation failed: {e}")
            raise
    
    def _scale_to_conductance(
        self, 
        matrix: np.ndarray, 
        g_min: float, 
        g_max: float
    ) -> np.ndarray:
        """Scale matrix values to conductance range with robust handling."""
        try:
            # Handle zero matrix
            if np.allclose(matrix, 0):
                return np.full_like(matrix, g_min, dtype=np.float64)
            
            # Robust scaling
            matrix_max = np.max(np.abs(matrix))
            if matrix_max == 0:
                return np.full_like(matrix, g_min, dtype=np.float64)
            
            # Linear scaling with bounds checking
            scaled = g_min + (g_max - g_min) * np.abs(matrix) / matrix_max
            
            # Ensure values are within bounds
            scaled = np.clip(scaled, g_min, g_max)
            
            return scaled.astype(np.float64)
            
        except Exception as e:
            self.logger.error(f"Conductance scaling failed: {e}")
            return np.full_like(matrix, g_min, dtype=np.float64)
    
    def _compute_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add realistic device noise based on specified model."""
        if self.noise_model == "none":
            return np.zeros_like(signal)
        
        try:
            noise = np.zeros_like(signal, dtype=np.float64)
            
            if self.noise_model in ["gaussian", "realistic"]:
                # Thermal noise (Johnson-Nyquist)
                thermal_std = self.device_params['thermal_noise'] * np.sqrt(np.abs(signal))
                thermal_noise = np.random.normal(0, thermal_std)
                noise += thermal_noise
            
            if self.noise_model == "realistic":
                # Shot noise (Poissonian)
                shot_std = self.device_params['shot_noise'] * np.sqrt(np.abs(signal))
                shot_noise = np.random.normal(0, shot_std)
                noise += shot_noise
                
                # 1/f (flicker) noise
                flicker_std = self.device_params['flicker_noise'] * np.abs(signal)
                flicker_noise = np.random.normal(0, flicker_std)
                noise += flicker_noise
                
                # Random telegraph signal (RTS) noise for memristors
                if self.cell_type in ['ReRAM', 'PCM']:
                    rts_noise = self._generate_rts_noise(signal)
                    noise += rts_noise
            
            return noise
            
        except Exception as e:
            self.logger.warning(f"Noise generation failed: {e}, using no noise")
            return np.zeros_like(signal)
    
    def _get_device_parameters(self) -> dict:
        """Get device-specific parameters."""
        base_params = {
            'g_range': (1e-9, 1e-6),  # 1nS to 1μS
            'thermal_noise': 0.01,
            'shot_noise': 0.005,
            'flicker_noise': 0.001,
            'programming_variation': 0.05,
            'drift_rate': 1e-6,
            'aging_rate': 1e-8
        }
        
        # Device-specific modifications
        if self.cell_type == "ReRAM":
            base_params['g_range'] = (1e-8, 1e-5)
            base_params['programming_variation'] = 0.1
            base_params['aging_rate'] = 5e-8
        elif self.cell_type == "PCM":
            base_params['g_range'] = (1e-7, 1e-4)
            base_params['drift_rate'] = 1e-5
            base_params['aging_rate'] = 2e-8
        elif self.cell_type == "1T1R":
            base_params['programming_variation'] = 0.02
            base_params['aging_rate'] = 1e-9
        
        return base_params
    
    def _add_programming_variations(self):
        """Add realistic programming variations to conductances."""
        variation = self.device_params['programming_variation']
        
        # Multiplicative variations (log-normal distribution)
        pos_variation = np.random.lognormal(0, variation, self.g_positive.shape)
        neg_variation = np.random.lognormal(0, variation, self.g_negative.shape)
        
        self.g_positive *= pos_variation
        self.g_negative *= neg_variation
        
        # Ensure bounds are maintained
        g_min, g_max = self.device_params['g_range']
        self.g_positive = np.clip(self.g_positive, g_min, g_max)
        self.g_negative = np.clip(self.g_negative, g_min, g_max)
    
    def _generate_rts_noise(self, signal: np.ndarray) -> np.ndarray:
        """Generate random telegraph signal noise for memristive devices."""
        # Simplified RTS model
        rts_amplitude = 0.001 * np.abs(signal)
        rts_probability = 0.1  # 10% chance of switching
        
        rts_noise = np.zeros_like(signal)
        for i in range(len(signal)):
            if np.random.random() < rts_probability:
                rts_noise[i] = np.random.choice([-1, 1]) * rts_amplitude[i]
        
        return rts_noise
    
    def _apply_device_nonlinearities(self, current: np.ndarray) -> np.ndarray:
        """Apply device non-linearities and saturation effects."""
        # Current saturation
        max_current = 1e-3  # 1mA max per column
        current_saturated = np.tanh(current / max_current) * max_current
        
        # Add conductance modulation effects
        if self.cell_type == "ReRAM":
            # ReRAM shows conductance modulation under high currents
            modulation = 1 - 0.1 * np.tanh(np.abs(current_saturated) / 1e-4)
            current_saturated *= modulation
        elif self.cell_type == "PCM":
            # PCM shows threshold switching behavior
            threshold = 1e-5
            mask = np.abs(current_saturated) > threshold
            current_saturated[mask] *= 1.2  # Increased conductance above threshold
        
        return current_saturated
    
    def _apply_aging_effects(self):
        """Apply long-term device aging effects."""
        if not self.is_programmed:
            return
            
        aging_rate = self.device_params['aging_rate']
        
        # Gradual conductance drift
        drift_factor = 1 - aging_rate * self.operation_count
        drift_factor = max(0.8, drift_factor)  # Limit maximum drift
        
        self.g_positive *= drift_factor
        self.g_negative *= drift_factor
        
        # Ensure minimum conductances
        g_min, _ = self.device_params['g_range']
        self.g_positive = np.maximum(self.g_positive, g_min)
        self.g_negative = np.maximum(self.g_negative, g_min)
        
        if self.operation_count % 10000 == 0:
            self.logger.info(f"Applied aging effects after {self.operation_count} operations")
    
    def get_device_stats(self) -> dict:
        """Get comprehensive device statistics and health metrics."""
        stats = {
            "is_programmed": self.is_programmed,
            "programming_errors": self.programming_errors,
            "operation_count": self.operation_count,
            "device_type": self.cell_type,
            "noise_model": self.noise_model,
            "array_size": (self.rows, self.cols),
            "total_devices": self.rows * self.cols
        }
        
        if self.is_programmed:
            # Conductance statistics
            stats.update({
                "g_positive_range": (float(self.g_positive.min()), float(self.g_positive.max())),
                "g_negative_range": (float(self.g_negative.min()), float(self.g_negative.max())),
                "g_positive_mean": float(self.g_positive.mean()),
                "g_negative_mean": float(self.g_negative.mean()),
                "g_positive_std": float(self.g_positive.std()),
                "g_negative_std": float(self.g_negative.std())
            })
            
            # Health metrics
            g_min, g_max = self.device_params['g_range']
            stuck_low = np.sum(self.g_positive <= g_min * 1.1) + np.sum(self.g_negative <= g_min * 1.1)
            stuck_high = np.sum(self.g_positive >= g_max * 0.9) + np.sum(self.g_negative >= g_max * 0.9)
            
            stats.update({
                "health_stuck_low_devices": int(stuck_low),
                "health_stuck_high_devices": int(stuck_high),
                "health_percentage": float(100 * (1 - (stuck_low + stuck_high) / (2 * self.rows * self.cols)))
            })
        
        return stats
    
    def perform_calibration(self) -> dict:
        """Perform device calibration and return calibration data."""
        if not self.is_programmed:
            return {"status": "error", "message": "Crossbar not programmed"}
        
        try:
            # Test with known input patterns
            test_patterns = [
                np.ones(self.rows),
                np.zeros(self.rows),
                np.random.random(self.rows)
            ]
            
            calibration_data = {
                "status": "success",
                "timestamp": np.datetime64('now').item().isoformat(),
                "test_results": []
            }
            
            for i, pattern in enumerate(test_patterns):
                result = self.compute_vmm(pattern)
                calibration_data["test_results"].append({
                    "pattern_id": i,
                    "input_norm": float(np.linalg.norm(pattern)),
                    "output_norm": float(np.linalg.norm(result)),
                    "output_mean": float(np.mean(result)),
                    "output_std": float(np.std(result))
                })
            
            self.logger.info("Device calibration completed successfully")
            return calibration_data
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def reset_device(self):
        """Reset device to initial state."""
        self.g_positive = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.g_negative = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.is_programmed = False
        self.programming_errors = 0
        self.operation_count = 0
        
        self.logger.info("Device reset to initial state")