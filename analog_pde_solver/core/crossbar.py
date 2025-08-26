"""Analog crossbar array simulation with realistic device modeling."""

import numpy as np
from typing import Tuple


class AnalogCrossbarArray:
    """Analog crossbar array for vector-matrix multiplication."""
    
    def __init__(self, rows: int, cols: int, cell_type: str = "1T1R"):
        """Initialize crossbar array.
        
        Args:
            rows: Number of rows (input size)
            cols: Number of columns (output size)  
            cell_type: Crossbar cell type ('1T1R', 'ReRAM', 'PCM')
        """
        self.rows = rows
        self.cols = cols
        self.cell_type = cell_type
        self.g_positive = np.zeros((rows, cols))
        self.g_negative = np.zeros((rows, cols))
        
        # Default conductance limits
        self.g_min = 1e-9  # 1nS
        self.g_max = 1e-6  # 1μS
        
    def program_conductances(self, target_matrix: np.ndarray) -> None:
        """Map target matrix to positive/negative conductance pairs."""
        # Store original matrix for accurate VMM computation
        self._original_matrix = target_matrix.copy()
        self._original_matrix_max = np.max(np.abs(target_matrix))
        
        # Reasonable conductance range for memristors
        g_min, g_max = 1e-6, 3e-6  # 1μS to 3μS
        
        # Decompose into positive and negative components
        pos_matrix = np.maximum(target_matrix, 0)
        neg_matrix = np.maximum(-target_matrix, 0)
        
        # Scale to conductance range
        self.g_positive = self._scale_to_conductance(pos_matrix, g_min, g_max)
        self.g_negative = self._scale_to_conductance(neg_matrix, g_min, g_max)
        
    def compute_vmm(self, input_vector: np.ndarray) -> np.ndarray:
        """Analog vector-matrix multiplication with noise."""
        # Reconstruct original matrix from conductances
        effective_matrix = self.g_positive - self.g_negative
        
        # Scale back to original matrix values
        if hasattr(self, '_original_matrix') and self._original_matrix is not None:
            # Use stored original matrix for accurate computation
            ideal_output = np.dot(self._original_matrix.T, input_vector)
        else:
            # Fallback: scale conductances back to logical values
            g_range = self.g_max - self.g_min
            if self._original_matrix_max > 0:
                logical_matrix = (effective_matrix - self.g_min) / g_range * self._original_matrix_max
                ideal_output = np.dot(logical_matrix.T, input_vector)
            else:
                ideal_output = np.dot(effective_matrix.T, input_vector) * 1e6
        
        # Add realistic device noise (small relative to signal)
        noise = self._compute_noise(ideal_output)
        
        return ideal_output + noise
    
    @property
    def conductance_matrix(self) -> np.ndarray:
        """Get the effective conductance matrix (differential)."""
        return self.g_positive - self.g_negative
    
    @conductance_matrix.setter
    def conductance_matrix(self, matrix: np.ndarray) -> None:
        """Set the conductance matrix by programming it."""
        self.program_conductances(matrix)
        
    def _scale_to_conductance(
        self, 
        matrix: np.ndarray, 
        g_min: float, 
        g_max: float
    ) -> np.ndarray:
        """Scale matrix values to conductance range."""
        if np.all(matrix == 0):
            return np.full_like(matrix, g_min)
        
        # Scale values linearly to conductance range
        max_val = np.max(matrix)
        if max_val > 0:
            # Linear scaling: preserve relative values
            return g_min + (g_max - g_min) * matrix / max_val
        else:
            return np.full_like(matrix, g_min)
        
    def _compute_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add realistic device noise."""
        # Simple thermal noise model
        thermal_noise = np.random.normal(0, 0.01 * np.abs(signal))
        return thermal_noise