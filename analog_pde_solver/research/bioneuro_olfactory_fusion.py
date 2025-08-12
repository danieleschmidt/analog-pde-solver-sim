"""
Bio-Neuromorphic Olfactory Fusion Engine

Advanced analog computing framework that merges biological olfactory processing
with neuromorphic analog circuits for next-generation PDE solving with chemical
gradient detection and bio-inspired optimization.

Research Innovation: Combines mammalian olfactory bulb neural dynamics with
analog crossbar arrays for ultra-efficient chemical gradient PDE solving.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.signal import convolve2d
from ..core.solver import AnalogPDESolver
from ..core.equations import PoissonEquation

@dataclass
class OlfactoryReceptorConfig:
    """Configuration for bio-inspired olfactory receptor arrays."""
    num_receptors: int = 256
    sensitivity_range: Tuple[float, float] = (1e-12, 1e-6)  # molar concentration
    response_time: float = 0.001  # seconds
    adaptation_rate: float = 0.1
    noise_level: float = 0.02

@dataclass
class MitralCellNetwork:
    """Mitral cell lateral inhibition network for contrast enhancement."""
    num_cells: int = 64
    inhibition_radius: float = 3.0
    inhibition_strength: float = 0.5
    temporal_dynamics: bool = True
    oscillation_frequency: float = 40.0  # Hz (gamma rhythm)

class BioneuroOlfactoryFusion:
    """
    Bio-neuromorphic olfactory fusion engine combining:
    1. Olfactory receptor analog arrays
    2. Mitral cell lateral inhibition networks  
    3. Glomerular processing layers
    4. Chemical gradient PDE solving
    """
    
    def __init__(
        self,
        receptor_config: OlfactoryReceptorConfig = None,
        mitral_config: MitralCellNetwork = None,
        crossbar_size: int = 128
    ):
        """Initialize bio-neuromorphic olfactory fusion system."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration defaults
        self.receptor_config = receptor_config or OlfactoryReceptorConfig()
        self.mitral_config = mitral_config or MitralCellNetwork()
        
        # Initialize analog PDE solver
        self.pde_solver = AnalogPDESolver(
            crossbar_size=crossbar_size,
            conductance_range=(1e-9, 1e-6),
            noise_model="realistic"
        )
        
        # Initialize olfactory processing layers
        self._initialize_olfactory_layers()
        
        self.logger.info(f"Initialized BioneuroOlfactoryFusion with {self.receptor_config.num_receptors} receptors")
    
    def _initialize_olfactory_layers(self):
        """Initialize bio-inspired olfactory processing layers."""
        # Olfactory receptor layer
        self.receptor_weights = self._generate_receptor_sensitivity_map()
        
        # Glomerular convergence layer  
        self.glomerular_map = self._create_glomerular_convergence()
        
        # Mitral cell lateral inhibition kernel
        self.inhibition_kernel = self._create_lateral_inhibition_kernel()
        
        # Temporal dynamics state
        self.mitral_state = np.zeros(self.mitral_config.num_cells)
        self.receptor_adaptation = np.ones(self.receptor_config.num_receptors)
        
    def _generate_receptor_sensitivity_map(self) -> np.ndarray:
        """Generate biologically-inspired receptor sensitivity patterns."""
        num_receptors = self.receptor_config.num_receptors
        
        # Create log-normal distribution of sensitivities (biological pattern)
        sensitivities = np.random.lognormal(
            mean=np.log(1e-9),
            sigma=2.0,
            size=num_receptors
        )
        
        # Clip to biological range
        min_sens, max_sens = self.receptor_config.sensitivity_range
        sensitivities = np.clip(sensitivities, min_sens, max_sens)
        
        return sensitivities.reshape(int(np.sqrt(num_receptors)), -1)
    
    def _create_glomerular_convergence(self) -> np.ndarray:
        """Create glomerular convergence pattern from receptors to mitral cells."""
        num_receptors = self.receptor_config.num_receptors
        num_mitral = self.mitral_config.num_cells
        
        # Each mitral cell receives from ~20-50 receptors (biological ratio)
        convergence_ratio = num_receptors // num_mitral
        convergence_map = np.zeros((num_mitral, num_receptors))
        
        for mitral_idx in range(num_mitral):
            # Random selection of receptor inputs with distance bias
            receptor_indices = np.random.choice(
                num_receptors,
                size=min(convergence_ratio, num_receptors),
                replace=False
            )
            
            # Gaussian weights for convergence
            weights = np.random.normal(1.0, 0.2, len(receptor_indices))
            weights = np.clip(weights, 0.1, 2.0)
            
            convergence_map[mitral_idx, receptor_indices] = weights
            
        return convergence_map
    
    def _create_lateral_inhibition_kernel(self) -> np.ndarray:
        """Create lateral inhibition kernel for mitral cell interactions."""
        radius = self.mitral_config.inhibition_radius
        strength = self.mitral_config.inhibition_strength
        
        # 2D Gaussian inhibition kernel
        size = int(2 * radius + 1)
        center = size // 2
        y, x = np.ogrid[-center:size-center, -center:size-center]
        
        # Gaussian profile with center excitation
        kernel = np.exp(-(x*x + y*y) / (2 * radius**2))
        kernel = -strength * kernel  # Inhibitory
        kernel[center, center] = 1.0  # Self-excitation
        
        return kernel / np.sum(np.abs(kernel))  # Normalize
    
    def detect_chemical_gradients(self, concentration_field: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect chemical gradients using bio-inspired olfactory processing.
        
        Args:
            concentration_field: 2D array of chemical concentrations
            
        Returns:
            Dictionary containing detected gradients and processed signals
        """
        self.logger.debug("Starting chemical gradient detection")
        
        # Step 1: Olfactory receptor response
        receptor_response = self._olfactory_receptor_response(concentration_field)
        
        # Step 2: Glomerular convergence
        glomerular_output = self._glomerular_processing(receptor_response)
        
        # Step 3: Mitral cell lateral inhibition
        mitral_output = self._mitral_cell_processing(glomerular_output)
        
        # Step 4: Gradient computation using analog PDE solving
        gradients = self._compute_gradients_analog(concentration_field, mitral_output)
        
        return {
            'receptor_response': receptor_response,
            'glomerular_output': glomerular_output,
            'mitral_output': mitral_output,
            'gradients': gradients,
            'gradient_magnitude': np.linalg.norm(gradients, axis=2),
            'gradient_direction': np.arctan2(gradients[:,:,1], gradients[:,:,0])
        }
    
    def _olfactory_receptor_response(self, concentration: np.ndarray) -> np.ndarray:
        """Simulate olfactory receptor response to chemical concentrations."""
        # Resize concentration field to match receptor array
        receptor_grid_size = int(np.sqrt(self.receptor_config.num_receptors))
        if concentration.shape != (receptor_grid_size, receptor_grid_size):
            from scipy.ndimage import zoom
            scale_factor = receptor_grid_size / concentration.shape[0]
            concentration_resized = zoom(concentration, scale_factor)
        else:
            concentration_resized = concentration
        
        # Apply receptor sensitivity with adaptation
        response = (concentration_resized * self.receptor_weights * 
                   self.receptor_adaptation.reshape(receptor_grid_size, -1))
        
        # Hill equation for receptor saturation
        hill_coeff = 2.0
        half_max = 1e-8
        response = (response**hill_coeff) / (response**hill_coeff + half_max**hill_coeff)
        
        # Add receptor noise
        noise = np.random.normal(0, self.receptor_config.noise_level, response.shape)
        response += noise
        
        # Update adaptation (slow negative feedback)
        adaptation_decay = np.exp(-self.receptor_config.adaptation_rate)
        self.receptor_adaptation *= adaptation_decay
        self.receptor_adaptation += (1 - adaptation_decay) * response.flatten()
        
        return np.clip(response, 0, 1)
    
    def _glomerular_processing(self, receptor_response: np.ndarray) -> np.ndarray:
        """Process receptor signals through glomerular convergence."""
        # Flatten receptor response for matrix multiplication
        receptor_flat = receptor_response.flatten()
        
        # Glomerular convergence: many-to-one mapping
        glomerular_output = np.dot(self.glomerular_map, receptor_flat)
        
        # Apply sigmoid activation (glomerular nonlinearity)
        glomerular_output = 1 / (1 + np.exp(-5 * (glomerular_output - 0.5)))
        
        return glomerular_output
    
    def _mitral_cell_processing(self, glomerular_input: np.ndarray) -> np.ndarray:
        """Process signals through mitral cell lateral inhibition network."""
        # Reshape to 2D grid for convolution
        grid_size = int(np.sqrt(self.mitral_config.num_cells))
        if len(glomerular_input) != grid_size * grid_size:
            # Pad or truncate to fit grid
            if len(glomerular_input) < grid_size * grid_size:
                padded = np.zeros(grid_size * grid_size)
                padded[:len(glomerular_input)] = glomerular_input
                glomerular_input = padded
            else:
                glomerular_input = glomerular_input[:grid_size * grid_size]
        
        mitral_grid = glomerular_input.reshape(grid_size, grid_size)
        
        # Apply lateral inhibition via convolution
        inhibited = convolve2d(mitral_grid, self.inhibition_kernel, 
                              mode='same', boundary='symm')
        
        # Temporal dynamics (gamma oscillations)
        if self.mitral_config.temporal_dynamics:
            oscillation = np.sin(2 * np.pi * self.mitral_config.oscillation_frequency * 
                               np.random.random())
            inhibited *= (1 + 0.1 * oscillation)
        
        # Update mitral state with decay
        decay_rate = 0.9
        self.mitral_state = decay_rate * self.mitral_state + (1 - decay_rate) * inhibited.flatten()
        
        return np.clip(self.mitral_state.reshape(grid_size, grid_size), 0, 1)
    
    def _compute_gradients_analog(self, concentration: np.ndarray, 
                                 mitral_output: np.ndarray) -> np.ndarray:
        """Compute chemical gradients using analog PDE solver."""
        # Create modified Poisson equation for gradient computation
        # ∇²φ = -ρ where ρ is the processed concentration field
        
        class GradientPoissonEquation:
            def __init__(self, concentration_field, mitral_weights):
                self.domain_size = concentration_field.shape
                self.source_field = concentration_field * mitral_weights
                
            def source_function(self, x, y):
                i, j = int(x * self.domain_size[0]), int(y * self.domain_size[1])
                i = np.clip(i, 0, self.domain_size[0] - 1)
                j = np.clip(j, 0, self.domain_size[1] - 1)
                return self.source_field[i, j]
        
        # Resize mitral output to match concentration field
        if mitral_output.shape != concentration.shape:
            from scipy.ndimage import zoom
            scale_factor = concentration.shape[0] / mitral_output.shape[0]
            mitral_resized = zoom(mitral_output, scale_factor)
        else:
            mitral_resized = mitral_output
            
        pde = GradientPoissonEquation(concentration, mitral_resized)
        
        # Solve for potential field
        try:
            potential = self.pde_solver.solve(pde, iterations=50, convergence_threshold=1e-4)
            
            # Compute gradients via finite differences
            grad_x = np.gradient(potential.reshape(concentration.shape), axis=1)
            grad_y = np.gradient(potential.reshape(concentration.shape), axis=0)
            
            gradients = np.stack([grad_x, grad_y], axis=2)
            
        except Exception as e:
            self.logger.warning(f"Analog gradient computation failed: {e}")
            # Fallback to simple finite difference
            grad_x = np.gradient(concentration, axis=1)
            grad_y = np.gradient(concentration, axis=0)
            gradients = np.stack([grad_x, grad_y], axis=2)
        
        return gradients
    
    def fuse_multimodal_signals(self, 
                               chemical_signals: List[np.ndarray],
                               signal_weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Fuse multiple chemical signals using bio-inspired processing.
        
        Args:
            chemical_signals: List of 2D chemical concentration fields
            signal_weights: Optional weights for each signal type
            
        Returns:
            Fused chemical gradient map
        """
        if not chemical_signals:
            raise ValueError("No chemical signals provided")
            
        if signal_weights is None:
            signal_weights = [1.0] * len(chemical_signals)
            
        if len(signal_weights) != len(chemical_signals):
            raise ValueError("Number of weights must match number of signals")
        
        # Process each signal through olfactory pathway
        processed_signals = []
        for signal, weight in zip(chemical_signals, signal_weights):
            result = self.detect_chemical_gradients(signal)
            processed_signals.append(weight * result['gradient_magnitude'])
        
        # Fusion strategy: weighted combination with competitive dynamics
        fused_signal = np.zeros_like(processed_signals[0])
        total_weight = 0
        
        for signal, weight in zip(processed_signals, signal_weights):
            # Winner-take-all with soft competition
            competition_factor = np.exp(5 * signal) / (1 + np.exp(5 * signal))
            fused_signal += weight * signal * competition_factor
            total_weight += weight
            
        # Normalize
        if total_weight > 0:
            fused_signal /= total_weight
            
        return fused_signal
    
    def adapt_to_environment(self, 
                           training_signals: List[np.ndarray],
                           learning_rate: float = 0.01) -> None:
        """
        Adapt olfactory processing to environmental statistics.
        
        Args:
            training_signals: List of representative chemical fields
            learning_rate: Adaptation learning rate
        """
        self.logger.info(f"Adapting to environment with {len(training_signals)} training signals")
        
        # Compute signal statistics
        signal_means = []
        signal_stds = []
        
        for signal in training_signals:
            signal_means.append(np.mean(signal))
            signal_stds.append(np.std(signal))
        
        env_mean = np.mean(signal_means)
        env_std = np.mean(signal_stds)
        
        # Adapt receptor sensitivities
        adaptation_factor = learning_rate * (env_std / (env_mean + 1e-8))
        
        self.receptor_weights *= (1 + adaptation_factor * 
                                 np.random.normal(0, 0.1, self.receptor_weights.shape))
        
        # Adapt mitral cell inhibition strength
        self.mitral_config.inhibition_strength *= (1 + adaptation_factor * 0.1)
        self.mitral_config.inhibition_strength = np.clip(
            self.mitral_config.inhibition_strength, 0.1, 1.0
        )
        
        # Update inhibition kernel
        self.inhibition_kernel = self._create_lateral_inhibition_kernel()
        
        self.logger.info(f"Environment adaptation complete. New inhibition strength: "
                        f"{self.mitral_config.inhibition_strength:.3f}")
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about olfactory processing performance."""
        return {
            'receptor_metrics': {
                'num_receptors': self.receptor_config.num_receptors,
                'sensitivity_range': self.receptor_config.sensitivity_range,
                'adaptation_state': {
                    'mean': np.mean(self.receptor_adaptation),
                    'std': np.std(self.receptor_adaptation),
                    'min': np.min(self.receptor_adaptation),
                    'max': np.max(self.receptor_adaptation)
                }
            },
            'mitral_metrics': {
                'num_cells': self.mitral_config.num_cells,
                'inhibition_strength': self.mitral_config.inhibition_strength,
                'current_state': {
                    'mean': np.mean(self.mitral_state),
                    'std': np.std(self.mitral_state),
                    'active_fraction': np.mean(self.mitral_state > 0.1)
                }
            },
            'glomerular_metrics': {
                'convergence_ratio': self.receptor_config.num_receptors / self.mitral_config.num_cells,
                'connection_density': np.mean(self.glomerular_map > 0)
            }
        }

# Research benchmark function
def benchmark_bioneuro_olfactory_performance():
    """Benchmark bio-neuromorphic olfactory fusion performance."""
    print("🧠 Bio-Neuromorphic Olfactory Fusion Benchmark")
    print("=" * 50)
    
    # Initialize system
    fusion_engine = BioneuroOlfactoryFusion(
        receptor_config=OlfactoryReceptorConfig(num_receptors=256),
        mitral_config=MitralCellNetwork(num_cells=64)
    )
    
    # Create test chemical fields
    size = 32
    x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
    
    # Chemical plume with gradient
    chemical_field1 = np.exp(-((x-5)**2 + (y-7)**2) / 4)
    
    # Secondary chemical source
    chemical_field2 = np.exp(-((x-3)**2 + (y-2)**2) / 2)
    
    # Process signals
    print("Processing chemical field 1...")
    result1 = fusion_engine.detect_chemical_gradients(chemical_field1)
    
    print("Processing chemical field 2...")
    result2 = fusion_engine.detect_chemical_gradients(chemical_field2)
    
    # Multi-modal fusion
    print("Performing multi-modal fusion...")
    fused = fusion_engine.fuse_multimodal_signals(
        [chemical_field1, chemical_field2], 
        signal_weights=[0.7, 0.3]
    )
    
    # Environmental adaptation
    print("Adapting to environment...")
    fusion_engine.adapt_to_environment([chemical_field1, chemical_field2])
    
    # Get metrics
    metrics = fusion_engine.get_processing_metrics()
    
    print("\n📊 Performance Metrics:")
    print(f"  Receptors: {metrics['receptor_metrics']['num_receptors']}")
    print(f"  Mitral cells: {metrics['mitral_metrics']['num_cells']}")
    print(f"  Adaptation state: {metrics['receptor_metrics']['adaptation_state']['mean']:.3f}")
    print(f"  Mitral activity: {metrics['mitral_metrics']['current_state']['active_fraction']:.3f}")
    print(f"  Gradient magnitude (field 1): {np.mean(result1['gradient_magnitude']):.6f}")
    print(f"  Gradient magnitude (field 2): {np.mean(result2['gradient_magnitude']):.6f}")
    print(f"  Fused signal strength: {np.mean(fused):.6f}")
    
    print("\n✅ Bio-neuromorphic olfactory fusion benchmark complete!")
    
    return {
        'results': [result1, result2],
        'fused_signal': fused,
        'metrics': metrics,
        'fusion_engine': fusion_engine
    }

if __name__ == "__main__":
    # Run benchmark
    benchmark_results = benchmark_bioneuro_olfactory_performance()