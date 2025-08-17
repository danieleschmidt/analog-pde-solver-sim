"""Adaptive precision control for analog computing systems."""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Precision levels for adaptive control."""
    LOW = 4      # 4-bit
    MEDIUM = 8   # 8-bit  
    HIGH = 12    # 12-bit
    ULTRA = 16   # 16-bit


@dataclass
class PrecisionMetrics:
    """Metrics for precision adaptation."""
    current_precision: int
    error_estimate: float
    computation_time: float
    energy_estimate: float
    convergence_rate: float
    stability_score: float


class AdaptivePrecisionController:
    """Adaptive precision control for analog PDE solvers."""
    
    def __init__(self, 
                 initial_precision: int = 8,
                 target_accuracy: float = 1e-6,
                 energy_budget: float = 1.0):
        """Initialize adaptive precision controller.
        
        Args:
            initial_precision: Starting precision in bits
            target_accuracy: Target solution accuracy
            energy_budget: Relative energy budget (1.0 = baseline)
        """
        self.current_precision = initial_precision
        self.target_accuracy = target_accuracy
        self.energy_budget = energy_budget
        self.logger = logger
        
        # Precision history for learning
        self.precision_history: List[PrecisionMetrics] = []
        self.adaptation_count = 0
        
        # Performance models
        self.energy_model = self._build_energy_model()
        self.accuracy_model = self._build_accuracy_model()
        
    def adapt_precision(self, 
                       current_error: float,
                       iteration: int,
                       convergence_history: List[float]) -> int:
        """Adapt precision based on current solver state.
        
        Args:
            current_error: Current solution error
            iteration: Current iteration number
            convergence_history: History of convergence errors
            
        Returns:
            New precision level in bits
        """
        start_time = time.time()
        
        # Compute metrics
        metrics = self._compute_metrics(current_error, iteration, convergence_history)
        
        # Determine if adaptation is needed
        adaptation_needed = self._should_adapt(metrics)
        
        if adaptation_needed:
            new_precision = self._select_precision(metrics)
            
            if new_precision != self.current_precision:
                self.logger.info(f"Adapting precision: {self.current_precision} → {new_precision} bits")
                self.current_precision = new_precision
                self.adaptation_count += 1
        
        # Store metrics for learning
        metrics.computation_time = time.time() - start_time
        self.precision_history.append(metrics)
        
        return self.current_precision
    
    def _compute_metrics(self, 
                        current_error: float,
                        iteration: int, 
                        convergence_history: List[float]) -> PrecisionMetrics:
        """Compute precision adaptation metrics."""
        
        # Estimate convergence rate
        if len(convergence_history) >= 3:
            recent_errors = convergence_history[-3:]
            convergence_rate = np.log(recent_errors[0] / recent_errors[-1]) / len(recent_errors)
        else:
            convergence_rate = 0.1  # Default
        
        # Estimate stability (variance in recent errors)
        if len(convergence_history) >= 5:
            recent_errors = convergence_history[-5:]
            stability_score = 1.0 / (1.0 + np.var(recent_errors))
        else:
            stability_score = 0.5  # Neutral
        
        # Energy estimate based on precision
        energy_estimate = self.energy_model.predict(self.current_precision)
        
        return PrecisionMetrics(
            current_precision=self.current_precision,
            error_estimate=current_error,
            computation_time=0.0,  # Will be filled later
            energy_estimate=energy_estimate,
            convergence_rate=convergence_rate,
            stability_score=stability_score
        )
    
    def _should_adapt(self, metrics: PrecisionMetrics) -> bool:
        """Determine if precision adaptation is needed."""
        
        # Don't adapt too frequently
        if self.adaptation_count > 0 and len(self.precision_history) < 5:
            return False
        
        # Adapt if error is much larger than target
        if metrics.error_estimate > 10 * self.target_accuracy:
            return True
        
        # Adapt if error is much smaller than target (can reduce precision)
        if metrics.error_estimate < 0.1 * self.target_accuracy:
            return True
        
        # Adapt if convergence is slow
        if metrics.convergence_rate < 0.01:
            return True
        
        # Adapt if system is unstable
        if metrics.stability_score < 0.3:
            return True
        
        return False
    
    def _select_precision(self, metrics: PrecisionMetrics) -> int:
        """Select optimal precision level."""
        
        # Get available precision levels
        available_precisions = [level.value for level in PrecisionLevel]
        
        best_precision = self.current_precision
        best_score = float('-inf')
        
        for precision in available_precisions:
            score = self._evaluate_precision(precision, metrics)
            
            if score > best_score:
                best_score = score
                best_precision = precision
        
        return best_precision
    
    def _evaluate_precision(self, precision: int, metrics: PrecisionMetrics) -> float:
        """Evaluate precision option using multi-objective scoring."""
        
        # Predict accuracy improvement
        accuracy_score = self.accuracy_model.predict(precision, metrics.error_estimate)
        
        # Predict energy cost
        energy_cost = self.energy_model.predict(precision)
        
        # Convergence benefit
        convergence_benefit = min(precision / 16.0, 1.0)  # Normalized to max precision
        
        # Stability penalty for very low precision
        stability_penalty = max(0, (6 - precision) * 0.1)
        
        # Multi-objective score
        score = (
            2.0 * accuracy_score +           # Accuracy is most important
            1.0 * convergence_benefit +      # Convergence speed
            -1.5 * energy_cost +             # Energy efficiency (negative = lower cost better)
            -stability_penalty               # Stability penalty
        )
        
        # Apply energy budget constraint
        if energy_cost > self.energy_budget:
            score -= 10.0  # Heavy penalty for exceeding budget
        
        return score
    
    def _build_energy_model(self):
        """Build energy consumption model."""
        return SimpleEnergyModel()
    
    def _build_accuracy_model(self):
        """Build accuracy prediction model."""
        return SimpleAccuracyModel()
    
    def get_precision_stats(self) -> Dict[str, Any]:
        """Get precision adaptation statistics."""
        if not self.precision_history:
            return {}
        
        precisions = [m.current_precision for m in self.precision_history]
        errors = [m.error_estimate for m in self.precision_history]
        energies = [m.energy_estimate for m in self.precision_history]
        
        return {
            'adaptation_count': self.adaptation_count,
            'current_precision': self.current_precision,
            'avg_precision': np.mean(precisions),
            'precision_range': (min(precisions), max(precisions)),
            'avg_error': np.mean(errors),
            'total_energy': sum(energies),
            'convergence_improvement': self._calculate_convergence_improvement()
        }
    
    def _calculate_convergence_improvement(self) -> float:
        """Calculate convergence improvement from adaptation."""
        if len(self.precision_history) < 10:
            return 0.0
        
        # Compare convergence rates before and after adaptations
        early_rates = [m.convergence_rate for m in self.precision_history[:5]]
        recent_rates = [m.convergence_rate for m in self.precision_history[-5:]]
        
        improvement = np.mean(recent_rates) - np.mean(early_rates)
        return improvement


class SimpleEnergyModel:
    """Simple energy consumption model for precision levels."""
    
    def predict(self, precision: int) -> float:
        """Predict relative energy consumption."""
        # Quadratic relationship: higher precision = more energy
        base_energy = 1.0  # 8-bit baseline
        precision_factor = (precision / 8.0) ** 2
        return base_energy * precision_factor


class SimpleAccuracyModel:
    """Simple accuracy prediction model."""
    
    def predict(self, precision: int, current_error: float) -> float:
        """Predict accuracy improvement score."""
        # Higher precision generally improves accuracy
        precision_benefit = precision / 16.0  # Normalized
        
        # But benefits diminish if error is already small
        error_factor = min(current_error * 1e6, 1.0)  # Scale error
        
        return precision_benefit * error_factor


def create_adaptive_controller(solver_config: Dict[str, Any]) -> AdaptivePrecisionController:
    """Create adaptive precision controller with problem-specific tuning.
    
    Args:
        solver_config: Solver configuration dictionary
        
    Returns:
        Configured adaptive precision controller
    """
    # Extract relevant parameters
    target_accuracy = solver_config.get('convergence_threshold', 1e-6)
    crossbar_size = solver_config.get('crossbar_size', 128)
    
    # Adjust initial precision based on problem size
    if crossbar_size <= 32:
        initial_precision = 6
    elif crossbar_size <= 128:
        initial_precision = 8
    else:
        initial_precision = 10
    
    # Adjust energy budget based on problem complexity
    energy_budget = 1.0 + (crossbar_size / 128.0) * 0.5
    
    return AdaptivePrecisionController(
        initial_precision=initial_precision,
        target_accuracy=target_accuracy,
        energy_budget=energy_budget
    )