"""Neuromorphic PDE Acceleration (NPA) for ultra-low power sparse PDE solving.

This module implements spike-based neuromorphic architectures for sparse PDE solving
with extreme energy efficiency for sparse problems.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque
from ..core.solver import AnalogPDESolver
from ..core.crossbar import AnalogCrossbarArray
from ..utils.logger import get_logger, PerformanceLogger


class SpikeEncoding(Enum):
    """Spike encoding schemes for neuromorphic computation."""
    RATE = "rate"           # Rate-based encoding
    TEMPORAL = "temporal"   # Temporal encoding
    POPULATION = "population"  # Population encoding
    DELTA = "delta"         # Delta modulation
    RANK_ORDER = "rank_order"  # Rank order encoding


@dataclass
class SpikeEvent:
    """Individual spike event in neuromorphic system."""
    timestamp: float
    neuron_id: int
    spike_value: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NeuronState:
    """State of a neuromorphic neuron."""
    membrane_potential: float
    threshold: float
    leak_rate: float
    refractory_period: float
    last_spike_time: float
    spike_count: int
    accumulated_input: float


class SparseEventBuffer:
    """Event-driven sparse data buffer for neuromorphic processing."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize sparse event buffer.
        
        Args:
            capacity: Maximum number of events to store
        """
        self.capacity = capacity
        self.events = deque(maxlen=capacity)
        self.active_neurons = set()
        self.event_statistics = defaultdict(int)
        
    def add_event(self, event: SpikeEvent) -> None:
        """Add spike event to buffer."""
        self.events.append(event)
        self.active_neurons.add(event.neuron_id)
        self.event_statistics['total_events'] += 1
        self.event_statistics[f'neuron_{event.neuron_id}'] += 1
    
    def get_events_in_window(
        self,
        start_time: float,
        end_time: float,
        neuron_ids: Optional[List[int]] = None
    ) -> List[SpikeEvent]:
        """Get events within time window."""
        filtered_events = []
        
        for event in self.events:
            if start_time <= event.timestamp <= end_time:
                if neuron_ids is None or event.neuron_id in neuron_ids:
                    filtered_events.append(event)
        
        return filtered_events
    
    def get_sparsity_statistics(self) -> Dict[str, Any]:
        """Get sparsity statistics."""
        if not self.events:
            return {'sparsity': 1.0, 'active_fraction': 0.0, 'event_rate': 0.0}
        
        total_possible_neurons = max(self.active_neurons) + 1 if self.active_neurons else 1
        active_fraction = len(self.active_neurons) / total_possible_neurons
        
        time_span = max(event.timestamp for event in self.events) - min(event.timestamp for event in self.events)
        event_rate = len(self.events) / max(time_span, 1e-6)
        
        return {
            'sparsity': 1.0 - active_fraction,
            'active_fraction': active_fraction,
            'event_rate': event_rate,
            'total_events': len(self.events),
            'active_neurons': len(self.active_neurons)
        }


class NeuromorphicSpikeEncoder:
    """Encoder for converting PDE data to spike trains."""
    
    def __init__(
        self,
        encoding_scheme: SpikeEncoding = SpikeEncoding.RATE,
        time_window: float = 1.0,
        spike_threshold: float = 0.5,
        max_spike_rate: float = 1000.0
    ):
        """Initialize spike encoder.
        
        Args:
            encoding_scheme: Spike encoding method
            time_window: Time window for encoding
            spike_threshold: Threshold for spike generation
            max_spike_rate: Maximum spike rate (Hz)
        """
        self.logger = get_logger('spike_encoder')
        
        self.encoding_scheme = encoding_scheme
        self.time_window = time_window
        self.spike_threshold = spike_threshold
        self.max_spike_rate = max_spike_rate
        
        self.encoding_statistics = defaultdict(float)
    
    def encode_data(
        self,
        data: np.ndarray,
        current_time: float
    ) -> List[SpikeEvent]:
        """Encode data array into spike events.
        
        Args:
            data: Input data to encode
            current_time: Current simulation time
            
        Returns:
            List of spike events
        """
        if self.encoding_scheme == SpikeEncoding.RATE:
            return self._rate_encode(data, current_time)
        elif self.encoding_scheme == SpikeEncoding.TEMPORAL:
            return self._temporal_encode(data, current_time)
        elif self.encoding_scheme == SpikeEncoding.POPULATION:
            return self._population_encode(data, current_time)
        elif self.encoding_scheme == SpikeEncoding.DELTA:
            return self._delta_encode(data, current_time)
        elif self.encoding_scheme == SpikeEncoding.RANK_ORDER:
            return self._rank_order_encode(data, current_time)
        else:
            return self._rate_encode(data, current_time)  # Default
    
    def _rate_encode(self, data: np.ndarray, current_time: float) -> List[SpikeEvent]:
        """Rate-based spike encoding."""
        events = []
        flattened_data = data.flatten()
        
        # Normalize data to [0, 1] range
        if np.max(np.abs(flattened_data)) > 0:
            normalized_data = np.abs(flattened_data) / np.max(np.abs(flattened_data))
        else:
            normalized_data = flattened_data
        
        for neuron_id, value in enumerate(normalized_data):
            # Skip very small values (sparsity)
            if value < self.spike_threshold * 0.1:
                continue
            
            # Convert value to spike rate
            spike_rate = value * self.max_spike_rate
            
            # Generate spikes based on Poisson process
            dt = 1.0 / self.max_spike_rate  # Time resolution
            num_spikes = int(spike_rate * self.time_window)
            
            for spike_idx in range(num_spikes):
                # Random spike timing within window
                spike_time = current_time + np.random.random() * self.time_window
                
                event = SpikeEvent(
                    timestamp=spike_time,
                    neuron_id=neuron_id,
                    spike_value=value,
                    metadata={'encoding': 'rate', 'original_value': flattened_data[neuron_id]}
                )
                events.append(event)
        
        self.encoding_statistics['rate_events'] += len(events)
        return events
    
    def _temporal_encode(self, data: np.ndarray, current_time: float) -> List[SpikeEvent]:
        """Temporal spike encoding - timing carries information."""
        events = []
        flattened_data = data.flatten()
        
        # Sort values to determine spike timing order
        sorted_indices = np.argsort(np.abs(flattened_data))[::-1]  # Descending order
        
        for rank, neuron_id in enumerate(sorted_indices):
            value = flattened_data[neuron_id]
            
            # Skip very small values
            if np.abs(value) < self.spike_threshold:
                continue
            
            # Earlier spikes for larger values
            spike_time = current_time + (rank / len(sorted_indices)) * self.time_window
            
            event = SpikeEvent(
                timestamp=spike_time,
                neuron_id=neuron_id,
                spike_value=value,
                metadata={'encoding': 'temporal', 'rank': rank}
            )
            events.append(event)
        
        self.encoding_statistics['temporal_events'] += len(events)
        return events
    
    def _population_encode(self, data: np.ndarray, current_time: float) -> List[SpikeEvent]:
        """Population encoding using multiple neurons per value."""
        events = []
        flattened_data = data.flatten()
        neurons_per_value = 4  # Use 4 neurons to represent each value
        
        for value_idx, value in enumerate(flattened_data):
            if np.abs(value) < self.spike_threshold:
                continue
            
            # Normalize value to [0, 1]
            normalized_value = (value + np.max(np.abs(flattened_data))) / (2 * np.max(np.abs(flattened_data)))
            
            for neuron_offset in range(neurons_per_value):
                neuron_id = value_idx * neurons_per_value + neuron_offset
                
                # Each neuron has different activation profile
                activation_center = neuron_offset / neurons_per_value
                activation_width = 1.0 / neurons_per_value
                
                # Gaussian activation
                activation = np.exp(-((normalized_value - activation_center) / activation_width)**2)
                
                if activation > self.spike_threshold:
                    # Generate spikes based on activation
                    num_spikes = int(activation * 10)  # Max 10 spikes
                    
                    for spike_idx in range(num_spikes):
                        spike_time = current_time + np.random.random() * self.time_window
                        
                        event = SpikeEvent(
                            timestamp=spike_time,
                            neuron_id=neuron_id,
                            spike_value=activation,
                            metadata={'encoding': 'population', 'value_idx': value_idx, 'neuron_offset': neuron_offset}
                        )
                        events.append(event)
        
        self.encoding_statistics['population_events'] += len(events)
        return events
    
    def _delta_encode(self, data: np.ndarray, current_time: float) -> List[SpikeEvent]:
        """Delta encoding - only encode changes."""
        events = []
        flattened_data = data.flatten()
        
        # Store previous data for delta computation
        if not hasattr(self, '_previous_data'):
            self._previous_data = np.zeros_like(flattened_data)
        
        # Compute delta
        delta = flattened_data - self._previous_data
        self._previous_data = flattened_data.copy()
        
        for neuron_id, delta_value in enumerate(delta):
            if np.abs(delta_value) > self.spike_threshold:
                # Positive delta -> positive spike, negative delta -> negative spike
                spike_time = current_time + np.random.random() * self.time_window
                
                event = SpikeEvent(
                    timestamp=spike_time,
                    neuron_id=neuron_id,
                    spike_value=delta_value,
                    metadata={'encoding': 'delta', 'delta_magnitude': np.abs(delta_value)}
                )
                events.append(event)
        
        self.encoding_statistics['delta_events'] += len(events)
        return events
    
    def _rank_order_encode(self, data: np.ndarray, current_time: float) -> List[SpikeEvent]:
        """Rank order encoding - first spike timing indicates magnitude."""
        events = []
        flattened_data = data.flatten()
        
        # Sort by magnitude
        sorted_indices = np.argsort(np.abs(flattened_data))[::-1]
        
        for rank, neuron_id in enumerate(sorted_indices):
            value = flattened_data[neuron_id]
            
            if np.abs(value) < self.spike_threshold:
                break  # Skip remaining smaller values
            
            # First spike time inversely related to magnitude
            spike_delay = (rank / len(sorted_indices)) * self.time_window
            spike_time = current_time + spike_delay
            
            event = SpikeEvent(
                timestamp=spike_time,
                neuron_id=neuron_id,
                spike_value=value,
                metadata={'encoding': 'rank_order', 'rank': rank}
            )
            events.append(event)
        
        self.encoding_statistics['rank_order_events'] += len(events)
        return events


class NeuromorphicSpikeDecoder:
    """Decoder for converting spike trains back to PDE data."""
    
    def __init__(
        self,
        decoding_scheme: SpikeEncoding = SpikeEncoding.RATE,
        time_window: float = 1.0,
        output_size: int = 64
    ):
        """Initialize spike decoder.
        
        Args:
            decoding_scheme: Spike decoding method
            time_window: Time window for decoding
            output_size: Size of output data array
        """
        self.logger = get_logger('spike_decoder')
        
        self.decoding_scheme = decoding_scheme
        self.time_window = time_window
        self.output_size = output_size
        
        self.decoding_statistics = defaultdict(float)
    
    def decode_events(
        self,
        events: List[SpikeEvent],
        current_time: float
    ) -> np.ndarray:
        """Decode spike events into data array.
        
        Args:
            events: List of spike events to decode
            current_time: Current simulation time
            
        Returns:
            Decoded data array
        """
        if self.decoding_scheme == SpikeEncoding.RATE:
            return self._rate_decode(events, current_time)
        elif self.decoding_scheme == SpikeEncoding.TEMPORAL:
            return self._temporal_decode(events, current_time)
        elif self.decoding_scheme == SpikeEncoding.POPULATION:
            return self._population_decode(events, current_time)
        elif self.decoding_scheme == SpikeEncoding.DELTA:
            return self._delta_decode(events, current_time)
        elif self.decoding_scheme == SpikeEncoding.RANK_ORDER:
            return self._rank_order_decode(events, current_time)
        else:
            return self._rate_decode(events, current_time)  # Default
    
    def _rate_decode(self, events: List[SpikeEvent], current_time: float) -> np.ndarray:
        """Rate-based spike decoding."""
        output = np.zeros(self.output_size)
        spike_counts = defaultdict(int)
        
        # Count spikes in time window
        window_start = current_time - self.time_window
        
        for event in events:
            if window_start <= event.timestamp <= current_time:
                if event.neuron_id < self.output_size:
                    spike_counts[event.neuron_id] += 1
        
        # Convert spike counts to values
        for neuron_id, count in spike_counts.items():
            # Normalize by time window
            rate = count / self.time_window
            output[neuron_id] = rate
        
        return output
    
    def _temporal_decode(self, events: List[SpikeEvent], current_time: float) -> np.ndarray:
        """Temporal spike decoding."""
        output = np.zeros(self.output_size)
        
        # Group events by neuron
        neuron_events = defaultdict(list)
        window_start = current_time - self.time_window
        
        for event in events:
            if window_start <= event.timestamp <= current_time and event.neuron_id < self.output_size:
                neuron_events[event.neuron_id].append(event)
        
        # Decode based on first spike timing
        for neuron_id, neuron_event_list in neuron_events.items():
            if neuron_event_list:
                # Find earliest spike
                earliest_event = min(neuron_event_list, key=lambda e: e.timestamp)
                
                # Convert timing to magnitude (earlier = larger)
                relative_time = earliest_event.timestamp - window_start
                normalized_time = relative_time / self.time_window
                
                # Invert: earlier spikes (smaller time) -> larger values
                output[neuron_id] = 1.0 - normalized_time
        
        return output
    
    def _population_decode(self, events: List[SpikeEvent], current_time: float) -> np.ndarray:
        """Population spike decoding."""
        neurons_per_value = 4
        num_values = self.output_size
        output = np.zeros(num_values)
        
        window_start = current_time - self.time_window
        
        # Group spikes by value index
        for value_idx in range(num_values):
            population_activity = []
            
            for neuron_offset in range(neurons_per_value):
                neuron_id = value_idx * neurons_per_value + neuron_offset
                
                # Count spikes for this neuron in time window
                spike_count = sum(1 for event in events 
                                if (window_start <= event.timestamp <= current_time and 
                                    event.neuron_id == neuron_id))
                
                population_activity.append(spike_count)
            
            # Decode population activity to single value
            if population_activity:
                # Weighted average based on neuron position
                weights = np.array([i / neurons_per_value for i in range(neurons_per_value)])
                weighted_activity = np.array(population_activity) * weights
                
                if np.sum(population_activity) > 0:
                    output[value_idx] = np.sum(weighted_activity) / np.sum(population_activity)
        
        return output
    
    def _delta_decode(self, events: List[SpikeEvent], current_time: float) -> np.ndarray:
        """Delta spike decoding."""
        delta_output = np.zeros(self.output_size)
        window_start = current_time - self.time_window
        
        # Accumulate delta values
        for event in events:
            if window_start <= event.timestamp <= current_time and event.neuron_id < self.output_size:
                delta_output[event.neuron_id] += event.spike_value
        
        # Integrate delta to get absolute values
        if not hasattr(self, '_integrated_output'):
            self._integrated_output = np.zeros(self.output_size)
        
        self._integrated_output += delta_output
        
        return self._integrated_output.copy()
    
    def _rank_order_decode(self, events: List[SpikeEvent], current_time: float) -> np.ndarray:
        """Rank order spike decoding."""
        output = np.zeros(self.output_size)
        window_start = current_time - self.time_window
        
        # Find first spike for each neuron
        first_spikes = {}
        
        for event in events:
            if window_start <= event.timestamp <= current_time and event.neuron_id < self.output_size:
                if event.neuron_id not in first_spikes or event.timestamp < first_spikes[event.neuron_id].timestamp:
                    first_spikes[event.neuron_id] = event
        
        # Sort by spike timing to get rank order
        sorted_spikes = sorted(first_spikes.values(), key=lambda e: e.timestamp)
        
        # Assign values based on rank (earlier = higher value)
        for rank, event in enumerate(sorted_spikes):
            # Higher rank (later timing) gets lower value
            value = 1.0 - (rank / len(sorted_spikes)) if sorted_spikes else 0.0
            output[event.neuron_id] = value
        
        return output


class NeuromorphicPDESolver:
    """Neuromorphic PDE Acceleration (NPA) system.
    
    Ultra-low power sparse PDE solving using spike-based neuromorphic architectures
    with extreme energy efficiency for sparse problems.
    """
    
    def __init__(
        self,
        base_solver: AnalogPDESolver,
        spike_encoder: NeuromorphicSpikeEncoder = None,
        spike_decoder: NeuromorphicSpikeDecoder = None,
        sparsity_threshold: float = 0.9,
        max_neurons: int = 1024
    ):
        """Initialize NPA system.
        
        Args:
            base_solver: Base analog PDE solver
            spike_encoder: Spike encoding system
            spike_decoder: Spike decoding system
            sparsity_threshold: Minimum sparsity to activate neuromorphic mode
            max_neurons: Maximum number of neurons
        """
        self.logger = get_logger('npa')
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.base_solver = base_solver
        self.sparsity_threshold = sparsity_threshold
        self.max_neurons = max_neurons
        
        # Initialize encoder/decoder if not provided
        self.spike_encoder = spike_encoder or NeuromorphicSpikeEncoder()
        self.spike_decoder = spike_decoder or NeuromorphicSpikeDecoder()
        
        # Neuromorphic components
        self.event_buffer = SparseEventBuffer()
        self.neuron_states = {i: NeuronState(0.0, 1.0, 0.1, 0.001, 0.0, 0, 0.0) 
                             for i in range(max_neurons)}
        
        # Performance tracking
        self.energy_savings = []
        self.sparsity_levels = []
        self.neuromorphic_activations = 0
        self.analog_fallbacks = 0
        
        self.logger.info(f"Initialized NPA with {max_neurons} neurons, sparsity threshold: {sparsity_threshold}")
    
    def solve_sparse_pde(
        self,
        pde,
        initial_solution: np.ndarray,
        time_span: Tuple[float, float],
        num_time_steps: int,
        adaptive_mode: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve sparse PDE using neuromorphic acceleration.
        
        Args:
            pde: PDE problem to solve
            initial_solution: Initial solution state
            time_span: Time integration span
            num_time_steps: Number of time steps
            adaptive_mode: Whether to adaptively switch between neuromorphic and analog
            
        Returns:
            Tuple of (final_solution, neuromorphic_metrics)
        """
        self.perf_logger.start_timer('neuromorphic_solve')
        
        t_start, t_end = time_span
        dt = (t_end - t_start) / num_time_steps
        current_time = t_start
        current_solution = initial_solution.copy()
        
        neuromorphic_metrics = {
            'neuromorphic_steps': 0,
            'analog_fallback_steps': 0,
            'average_sparsity': 0.0,
            'total_energy_savings': 0.0,
            'spike_statistics': {},
            'adaptation_decisions': []
        }
        
        sparsity_history = []
        energy_savings_history = []
        
        for step in range(num_time_steps):
            # Analyze current solution sparsity
            sparsity_level = self._compute_sparsity(current_solution)
            sparsity_history.append(sparsity_level)
            
            # Decide whether to use neuromorphic or analog solver
            use_neuromorphic = sparsity_level >= self.sparsity_threshold
            
            if adaptive_mode:
                # Additional criteria for neuromorphic activation
                solution_magnitude = np.linalg.norm(current_solution)
                use_neuromorphic = (use_neuromorphic and 
                                  solution_magnitude > 1e-6 and
                                  len(self.event_buffer.active_neurons) < self.max_neurons * 0.8)
            
            neuromorphic_metrics['adaptation_decisions'].append({
                'step': step,
                'sparsity': sparsity_level,
                'use_neuromorphic': use_neuromorphic,
                'active_neurons': len(self.event_buffer.active_neurons)
            })
            
            if use_neuromorphic:
                # Solve using neuromorphic acceleration
                new_solution, step_metrics = self._solve_neuromorphic_step(
                    pde, current_solution, dt, current_time
                )
                neuromorphic_metrics['neuromorphic_steps'] += 1
                self.neuromorphic_activations += 1
                
                # Estimate energy savings
                energy_saving = self._estimate_energy_savings(sparsity_level)
                energy_savings_history.append(energy_saving)
                
            else:
                # Fall back to analog solver
                new_solution = self._solve_analog_step(
                    pde, current_solution, dt, current_time
                )
                step_metrics = {'method': 'analog_fallback'}
                neuromorphic_metrics['analog_fallback_steps'] += 1
                self.analog_fallbacks += 1
                energy_savings_history.append(0.0)
            
            current_solution = new_solution
            current_time += dt
            
            # Update metrics periodically
            if step % max(1, num_time_steps // 10) == 0:
                self.logger.debug(f"NPA step {step}/{num_time_steps}: sparsity={sparsity_level:.3f}, neuromorphic={use_neuromorphic}")
        
        solve_time = self.perf_logger.end_timer('neuromorphic_solve')
        
        # Compute final metrics
        neuromorphic_metrics.update({
            'total_solve_time': solve_time,
            'average_sparsity': np.mean(sparsity_history),
            'total_energy_savings': np.sum(energy_savings_history),
            'neuromorphic_fraction': neuromorphic_metrics['neuromorphic_steps'] / num_time_steps,
            'spike_statistics': self.event_buffer.get_sparsity_statistics()
        })
        
        self.logger.info(f"NPA solve completed: {neuromorphic_metrics['neuromorphic_fraction']:.1%} neuromorphic, {neuromorphic_metrics['total_energy_savings']:.2f} energy savings")
        
        return current_solution, neuromorphic_metrics
    
    def _compute_sparsity(self, solution: np.ndarray) -> float:
        """Compute sparsity level of solution."""
        total_elements = solution.size
        if total_elements == 0:
            return 1.0
        
        # Count near-zero elements
        threshold = np.max(np.abs(solution)) * 0.01 if np.max(np.abs(solution)) > 0 else 1e-10
        near_zero_elements = np.sum(np.abs(solution) < threshold)
        
        sparsity = near_zero_elements / total_elements
        return sparsity
    
    def _solve_neuromorphic_step(
        self,
        pde,
        current_solution: np.ndarray,
        dt: float,
        current_time: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve single time step using neuromorphic acceleration."""
        step_metrics = {'method': 'neuromorphic', 'spike_events': 0, 'active_neurons': 0}
        
        # Encode current solution to spike events
        spike_events = self.spike_encoder.encode_data(current_solution, current_time)
        step_metrics['spike_events'] = len(spike_events)
        
        # Add events to buffer
        for event in spike_events:
            self.event_buffer.add_event(event)
        
        # Process spikes through neuromorphic dynamics
        processed_events = self._process_spike_dynamics(spike_events, dt)
        step_metrics['processed_events'] = len(processed_events)
        
        # Decode spike events back to solution
        new_solution = self.spike_decoder.decode_events(processed_events, current_time + dt)
        
        # Ensure solution maintains correct dimensions
        if len(new_solution) != len(current_solution):
            if len(new_solution) > len(current_solution):
                new_solution = new_solution[:len(current_solution)]
            else:
                padded_solution = np.zeros_like(current_solution)
                padded_solution[:len(new_solution)] = new_solution
                new_solution = padded_solution
        
        step_metrics['active_neurons'] = len(self.event_buffer.active_neurons)
        
        return new_solution, step_metrics
    
    def _solve_analog_step(
        self,
        pde,
        current_solution: np.ndarray,
        dt: float,
        current_time: float
    ) -> np.ndarray:
        """Solve single time step using analog fallback."""
        # Simple forward Euler step (could be enhanced)
        try:
            # Use base analog solver for one iteration
            solution = self.base_solver.solve(pde, iterations=1, convergence_threshold=1e-6)
            
            # If base solver returns wrong dimensions, interpolate
            if len(solution) != len(current_solution):
                if len(solution) > len(current_solution):
                    solution = solution[:len(current_solution)]
                else:
                    padded_solution = np.zeros_like(current_solution)
                    padded_solution[:len(solution)] = solution
                    solution = padded_solution
            
            return solution
            
        except Exception as e:
            self.logger.warning(f"Analog fallback failed: {e}")
            return current_solution  # Return unchanged solution
    
    def _process_spike_dynamics(
        self,
        input_events: List[SpikeEvent],
        dt: float
    ) -> List[SpikeEvent]:
        """Process spike events through neuromorphic dynamics."""
        output_events = []
        
        # Update neuron states based on input spikes
        for event in input_events:
            if event.neuron_id in self.neuron_states:
                neuron = self.neuron_states[event.neuron_id]
                
                # Check refractory period
                time_since_last_spike = event.timestamp - neuron.last_spike_time
                if time_since_last_spike < neuron.refractory_period:
                    continue
                
                # Update membrane potential
                neuron.accumulated_input += event.spike_value
                neuron.membrane_potential += neuron.accumulated_input
                
                # Apply leak
                neuron.membrane_potential *= (1.0 - neuron.leak_rate * dt)
                
                # Check for output spike
                if neuron.membrane_potential > neuron.threshold:
                    # Generate output spike
                    output_event = SpikeEvent(
                        timestamp=event.timestamp + dt * 0.1,  # Small delay
                        neuron_id=event.neuron_id,
                        spike_value=neuron.membrane_potential - neuron.threshold,
                        metadata={
                            'processed': True,
                            'membrane_potential': neuron.membrane_potential
                        }
                    )
                    output_events.append(output_event)
                    
                    # Reset neuron
                    neuron.membrane_potential = 0.0
                    neuron.last_spike_time = event.timestamp
                    neuron.spike_count += 1
                
                # Reset accumulated input after processing
                neuron.accumulated_input = 0.0
        
        return output_events
    
    def _estimate_energy_savings(self, sparsity_level: float) -> float:
        """Estimate energy savings from neuromorphic processing."""
        # Energy model: neuromorphic energy scales with activity, not problem size
        base_analog_energy = 1.0  # Normalized base energy for analog computation
        
        # Neuromorphic energy depends on spike activity
        activity_level = 1.0 - sparsity_level
        neuromorphic_energy = activity_level * 0.01  # Very low energy for sparse activity
        
        # Additional savings from event-driven computation
        event_efficiency = 0.001 if sparsity_level > 0.95 else 0.01
        neuromorphic_energy *= event_efficiency
        
        energy_saving = base_analog_energy - neuromorphic_energy
        return max(0.0, energy_saving)  # Ensure non-negative savings
    
    def get_neuromorphic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive neuromorphic processing statistics."""
        stats = {
            'system_configuration': {
                'max_neurons': self.max_neurons,
                'sparsity_threshold': self.sparsity_threshold,
                'encoding_scheme': self.spike_encoder.encoding_scheme.value,
                'decoding_scheme': self.spike_decoder.decoding_scheme.value
            },
            'activation_statistics': {
                'neuromorphic_activations': self.neuromorphic_activations,
                'analog_fallbacks': self.analog_fallbacks,
                'neuromorphic_fraction': self.neuromorphic_activations / max(1, self.neuromorphic_activations + self.analog_fallbacks)
            },
            'event_buffer_statistics': self.event_buffer.get_sparsity_statistics(),
            'encoding_statistics': dict(self.spike_encoder.encoding_statistics),
            'decoding_statistics': dict(self.spike_decoder.decoding_statistics),
            'neuron_utilization': {
                'total_neurons': self.max_neurons,
                'active_neurons': len(self.event_buffer.active_neurons),
                'utilization_fraction': len(self.event_buffer.active_neurons) / self.max_neurons
            }
        }
        
        # Energy analysis
        if self.energy_savings:
            stats['energy_analysis'] = {
                'total_energy_savings': np.sum(self.energy_savings),
                'average_energy_savings': np.mean(self.energy_savings),
                'peak_energy_savings': np.max(self.energy_savings),
                'energy_efficiency_ratio': np.sum(self.energy_savings) / max(1, len(self.energy_savings))
            }
        
        # Sparsity analysis
        if self.sparsity_levels:
            stats['sparsity_analysis'] = {
                'average_sparsity': np.mean(self.sparsity_levels),
                'peak_sparsity': np.max(self.sparsity_levels),
                'sparsity_std': np.std(self.sparsity_levels),
                'high_sparsity_fraction': np.mean(np.array(self.sparsity_levels) > self.sparsity_threshold)
            }
        
        return stats
    
    def optimize_neuromorphic_parameters(
        self,
        sample_data: List[np.ndarray],
        target_energy_reduction: float = 0.9
    ) -> Dict[str, Any]:
        """Optimize neuromorphic parameters for target performance."""
        optimization_results = {
            'optimal_sparsity_threshold': self.sparsity_threshold,
            'optimal_encoding_scheme': self.spike_encoder.encoding_scheme,
            'optimization_metrics': {}
        }
        
        # Test different sparsity thresholds
        sparsity_thresholds = np.linspace(0.5, 0.99, 10)
        encoding_schemes = list(SpikeEncoding)
        
        best_energy_reduction = 0.0
        best_config = None
        
        for threshold in sparsity_thresholds:
            for encoding_scheme in encoding_schemes:
                # Create test configuration
                test_encoder = NeuromorphicSpikeEncoder(encoding_scheme=encoding_scheme)
                test_decoder = NeuromorphicSpikeDecoder(decoding_scheme=encoding_scheme)
                
                # Simulate processing with sample data
                total_energy_saving = 0.0
                total_samples = len(sample_data)
                
                for sample in sample_data:
                    sample_sparsity = self._compute_sparsity(sample)
                    
                    if sample_sparsity >= threshold:
                        # Would use neuromorphic processing
                        energy_saving = self._estimate_energy_savings(sample_sparsity)
                        total_energy_saving += energy_saving
                
                avg_energy_reduction = total_energy_saving / max(1, total_samples)
                
                # Check if this configuration meets target
                if avg_energy_reduction > best_energy_reduction:
                    best_energy_reduction = avg_energy_reduction
                    best_config = {
                        'sparsity_threshold': threshold,
                        'encoding_scheme': encoding_scheme,
                        'energy_reduction': avg_energy_reduction
                    }
        
        if best_config and best_config['energy_reduction'] >= target_energy_reduction:
            # Update system with optimal parameters
            self.sparsity_threshold = best_config['sparsity_threshold']
            self.spike_encoder.encoding_scheme = best_config['encoding_scheme']
            self.spike_decoder.decoding_scheme = best_config['encoding_scheme']
            
            optimization_results.update({
                'optimal_sparsity_threshold': best_config['sparsity_threshold'],
                'optimal_encoding_scheme': best_config['encoding_scheme'],
                'achieved_energy_reduction': best_config['energy_reduction'],
                'optimization_successful': True
            })
            
            self.logger.info(f"Neuromorphic optimization successful: {best_config['energy_reduction']:.1%} energy reduction")
        else:
            optimization_results['optimization_successful'] = False
            self.logger.warning(f"Failed to achieve target energy reduction of {target_energy_reduction:.1%}")
        
        return optimization_results