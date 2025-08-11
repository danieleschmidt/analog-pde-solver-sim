"""Analog Multi-Physics Coupling (AMPC) for coupled PDE systems.

This module implements direct analog coupling between different physical phenomena
for coupled PDE systems, eliminating digital coupling overhead.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.solver import AnalogPDESolver
from ..core.crossbar import AnalogCrossbarArray
from ..utils.logger import get_logger, PerformanceLogger


class PhysicsDomain(Enum):
    """Types of physics domains for multi-physics coupling."""
    THERMAL = "thermal"
    ELECTROMAGNETIC = "electromagnetic"
    FLUID = "fluid"
    STRUCTURAL = "structural"
    CHEMICAL = "chemical"
    DIFFUSION = "diffusion"
    ACOUSTIC = "acoustic"
    OPTICAL = "optical"


@dataclass
class CouplingInterface:
    """Interface between different physics domains."""
    source_domain: PhysicsDomain
    target_domain: PhysicsDomain
    coupling_type: str  # 'direct', 'source_term', 'boundary', 'material_property'
    coupling_strength: float
    coupling_function: Callable[[np.ndarray], np.ndarray]
    interface_regions: List[Tuple[int, int, int, int]]  # Crossbar regions affected
    conservation_required: bool
    bidirectional: bool


@dataclass
class PhysicsDomainConfig:
    """Configuration for a physics domain."""
    domain_type: PhysicsDomain
    governing_equations: List[str]  # List of PDE names
    crossbar_allocation: Tuple[int, int, int, int]  # (start_row, end_row, start_col, end_col)
    boundary_conditions: Dict[str, Any]
    material_properties: Dict[str, float]
    source_terms: Optional[Callable[[np.ndarray, float], np.ndarray]]
    time_scale: float  # Characteristic time scale
    length_scale: float  # Characteristic length scale


class AnalogMultiPhysicsCoupler:
    """Analog Multi-Physics Coupling (AMPC) system.
    
    Direct analog coupling between different physical phenomena to eliminate
    digital coupling overhead in multi-physics problems.
    """
    
    def __init__(
        self,
        primary_crossbar: AnalogCrossbarArray,
        physics_domains: List[PhysicsDomainConfig],
        coupling_interfaces: List[CouplingInterface],
        conservation_tolerance: float = 1e-8
    ):
        """Initialize AMPC system.
        
        Args:
            primary_crossbar: Primary crossbar for multi-physics computation
            physics_domains: List of physics domain configurations
            coupling_interfaces: List of coupling interfaces between domains
            conservation_tolerance: Tolerance for conservation checking
        """
        self.logger = get_logger('ampc')
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.primary_crossbar = primary_crossbar
        self.physics_domains = {domain.domain_type: domain for domain in physics_domains}
        self.coupling_interfaces = coupling_interfaces
        self.conservation_tolerance = conservation_tolerance
        
        # Domain-specific crossbar regions
        self.domain_crossbars = {}
        self._initialize_domain_crossbars()
        
        # Coupling matrices and interface conductances
        self.coupling_matrices = {}
        self.interface_conductances = {}
        self._initialize_coupling_matrices()
        
        # Conservation tracking
        self.conservation_errors = []
        self.coupling_residuals = {}
        
        # Performance metrics
        self.coupling_overhead = []
        self.interface_utilization = []
        
        self.logger.info(f"Initialized AMPC with {len(physics_domains)} domains and {len(coupling_interfaces)} interfaces")
    
    def _initialize_domain_crossbars(self) -> None:
        """Initialize crossbar regions for each physics domain."""
        for domain_type, domain_config in self.physics_domains.items():
            r1, r2, c1, c2 = domain_config.crossbar_allocation
            
            # Create view of primary crossbar for this domain
            domain_conductances = self.primary_crossbar.conductance_matrix[r1:r2, c1:c2].copy()
            
            # Configure domain-specific properties
            domain_crossbar = AnalogCrossbarArray(
                rows=r2-r1,
                cols=c2-c1,
                cell_type=self.primary_crossbar.cell_type
            )
            domain_crossbar.conductance_matrix = domain_conductances
            
            self.domain_crossbars[domain_type] = {
                'crossbar': domain_crossbar,
                'allocation': (r1, r2, c1, c2),
                'state': np.zeros(r2-r1)  # Current domain state
            }
            
            self.logger.debug(f"Initialized {domain_type.value} domain: region ({r1}:{r2}, {c1}:{c2})")
    
    def _initialize_coupling_matrices(self) -> None:
        """Initialize coupling matrices for interface interactions."""
        for interface in self.coupling_interfaces:
            interface_key = f"{interface.source_domain.value}_to_{interface.target_domain.value}"
            
            # Get source and target domain dimensions
            source_domain = self.physics_domains[interface.source_domain]
            target_domain = self.physics_domains[interface.target_domain]
            
            source_allocation = source_domain.crossbar_allocation
            target_allocation = target_domain.crossbar_allocation
            
            source_size = (source_allocation[1] - source_allocation[0])
            target_size = (target_allocation[1] - target_allocation[0])
            
            # Initialize coupling matrix based on coupling type
            if interface.coupling_type == 'direct':
                # Direct state coupling
                coupling_matrix = np.zeros((target_size, source_size))
                
                # Map interface regions
                for region in interface.interface_regions:
                    r1, r2, c1, c2 = region
                    # Simplified: uniform coupling strength
                    coupling_matrix[r1:r2, c1:c2] = interface.coupling_strength
                    
            elif interface.coupling_type == 'source_term':
                # Coupling appears as source term
                coupling_matrix = interface.coupling_strength * np.eye(
                    min(source_size, target_size)
                )
                
            elif interface.coupling_type == 'boundary':
                # Boundary coupling
                coupling_matrix = np.zeros((target_size, source_size))
                coupling_matrix[0, :] = interface.coupling_strength  # First row
                coupling_matrix[-1, :] = interface.coupling_strength  # Last row
                
            elif interface.coupling_type == 'material_property':
                # Material property dependence
                coupling_matrix = interface.coupling_strength * np.random.normal(
                    1.0, 0.1, (target_size, source_size)
                )
                
            else:
                # Default: identity coupling
                coupling_matrix = interface.coupling_strength * np.eye(
                    min(source_size, target_size)
                )
            
            self.coupling_matrices[interface_key] = coupling_matrix
            
            # Initialize interface conductances for analog implementation
            self._program_interface_conductances(interface, coupling_matrix)
            
            self.logger.debug(f"Initialized coupling matrix: {interface_key} ({coupling_matrix.shape})")
    
    def _program_interface_conductances(
        self,
        interface: CouplingInterface,
        coupling_matrix: np.ndarray
    ) -> None:
        """Program analog conductances for coupling interface."""
        interface_key = f"{interface.source_domain.value}_to_{interface.target_domain.value}"
        
        # Map coupling matrix to conductance values
        g_min, g_max = self.primary_crossbar.g_min, self.primary_crossbar.g_max
        
        # Normalize coupling matrix to conductance range
        matrix_min, matrix_max = np.min(coupling_matrix), np.max(coupling_matrix)
        if matrix_max > matrix_min:
            normalized_matrix = (coupling_matrix - matrix_min) / (matrix_max - matrix_min)
            interface_conductances = g_min + normalized_matrix * (g_max - g_min)
        else:
            interface_conductances = np.full_like(coupling_matrix, (g_min + g_max) / 2)
        
        self.interface_conductances[interface_key] = interface_conductances
        
        # Program interface regions in primary crossbar
        for region in interface.interface_regions:
            r1, r2, c1, c2 = region
            
            # Map interface conductances to crossbar region
            region_height = r2 - r1
            region_width = c2 - c1
            
            if interface_conductances.shape == (region_height, region_width):
                self.primary_crossbar.conductance_matrix[r1:r2, c1:c2] = interface_conductances
            else:
                # Resize interface conductances to fit region
                resized_conductances = np.resize(interface_conductances, (region_height, region_width))
                self.primary_crossbar.conductance_matrix[r1:r2, c1:c2] = resized_conductances
    
    def solve_coupled_system(
        self,
        initial_conditions: Dict[PhysicsDomain, np.ndarray],
        time_span: Tuple[float, float],
        num_time_steps: int,
        coupling_iterations: int = 5
    ) -> Tuple[Dict[PhysicsDomain, np.ndarray], Dict[str, Any]]:
        """Solve coupled multi-physics system.
        
        Args:
            initial_conditions: Initial conditions for each domain
            time_span: Time integration span (start, end)
            num_time_steps: Number of time steps
            coupling_iterations: Number of coupling iterations per time step
            
        Returns:
            Tuple of (final_solutions, coupling_metrics)
        """
        self.perf_logger.start_timer('coupled_solve')
        
        t_start, t_end = time_span
        dt = (t_end - t_start) / num_time_steps
        
        # Initialize domain states
        current_states = {}
        for domain_type, initial_state in initial_conditions.items():
            current_states[domain_type] = initial_state.copy()
            # Update domain crossbar state
            if domain_type in self.domain_crossbars:
                self.domain_crossbars[domain_type]['state'] = initial_state.copy()
        
        coupling_metrics = {
            'time_steps': num_time_steps,
            'coupling_iterations_per_step': coupling_iterations,
            'conservation_errors': [],
            'coupling_residuals': [],
            'interface_activity': {iface: [] for iface in self.coupling_matrices.keys()},
            'domain_evolution': {domain.value: [] for domain in current_states.keys()}
        }
        
        # Time integration loop
        for step in range(num_time_steps):
            current_time = t_start + step * dt
            
            # Coupling iteration loop (sub-iterations for tight coupling)
            for coupling_iter in range(coupling_iterations):
                # Store previous states for convergence check
                previous_states = {k: v.copy() for k, v in current_states.items()}
                
                # Update each domain with coupling effects
                for domain_type in current_states.keys():
                    new_state = self._solve_domain_step(
                        domain_type, current_states, dt, current_time
                    )
                    current_states[domain_type] = new_state
                
                # Check coupling convergence
                coupling_residual = self._compute_coupling_residual(
                    current_states, previous_states
                )
                coupling_metrics['coupling_residuals'].append(coupling_residual)
                
                # Check conservation
                conservation_error = self._check_conservation(current_states)
                coupling_metrics['conservation_errors'].append(conservation_error)
                
                # Early convergence check
                if coupling_residual < self.conservation_tolerance:
                    break
            
            # Record domain evolution
            for domain_type, state in current_states.items():
                coupling_metrics['domain_evolution'][domain_type.value].append(
                    np.linalg.norm(state)
                )
            
            # Record interface activity
            for interface_key in self.coupling_matrices.keys():
                activity = self._compute_interface_activity(interface_key, current_states)
                coupling_metrics['interface_activity'][interface_key].append(activity)
            
            if step % max(1, num_time_steps // 10) == 0:
                self.logger.debug(f"Time step {step}/{num_time_steps}, coupling residual: {coupling_residual:.2e}")
        
        solve_time = self.perf_logger.end_timer('coupled_solve')
        coupling_metrics['total_solve_time'] = solve_time
        coupling_metrics['avg_coupling_residual'] = np.mean(coupling_metrics['coupling_residuals'])
        coupling_metrics['avg_conservation_error'] = np.mean(coupling_metrics['conservation_errors'])
        
        self.logger.info(f"Coupled solve completed: {solve_time:.3f}s, avg coupling residual: {coupling_metrics['avg_coupling_residual']:.2e}")
        
        return current_states, coupling_metrics
    
    def _solve_domain_step(
        self,
        domain_type: PhysicsDomain,
        all_states: Dict[PhysicsDomain, np.ndarray],
        dt: float,
        current_time: float
    ) -> np.ndarray:
        """Solve single domain time step with coupling effects.
        
        Args:
            domain_type: Domain to solve
            all_states: States of all domains
            dt: Time step size
            current_time: Current time
            
        Returns:
            Updated domain state
        """
        domain_config = self.physics_domains[domain_type]
        domain_info = self.domain_crossbars[domain_type]
        current_state = all_states[domain_type]
        
        # Compute coupling contributions from other domains
        coupling_contribution = np.zeros_like(current_state)
        
        for interface in self.coupling_interfaces:
            if interface.target_domain == domain_type:
                # This domain receives coupling from source domain
                source_state = all_states[interface.source_domain]
                interface_key = f"{interface.source_domain.value}_to_{domain_type.value}"
                
                if interface_key in self.coupling_matrices:
                    coupling_matrix = self.coupling_matrices[interface_key]
                    
                    # Apply coupling function if specified
                    if interface.coupling_function:
                        modified_source = interface.coupling_function(source_state)
                    else:
                        modified_source = source_state
                    
                    # Compute coupling contribution
                    if coupling_matrix.shape[1] == len(modified_source):
                        contribution = coupling_matrix @ modified_source
                        
                        # Resize if necessary
                        if len(contribution) != len(coupling_contribution):
                            if len(contribution) > len(coupling_contribution):
                                contribution = contribution[:len(coupling_contribution)]
                            else:
                                padded_contribution = np.zeros_like(coupling_contribution)
                                padded_contribution[:len(contribution)] = contribution
                                contribution = padded_contribution
                        
                        coupling_contribution += contribution
        
        # Apply source terms if specified
        source_term = np.zeros_like(current_state)
        if domain_config.source_terms:
            source_term = domain_config.source_terms(current_state, current_time)
        
        # Solve domain physics using analog crossbar
        domain_crossbar = domain_info['crossbar']
        
        # Combine current state, coupling, and source terms
        rhs = current_state + dt * (coupling_contribution + source_term)
        
        # Solve using crossbar VMM (simplified time integration)
        try:
            new_state = domain_crossbar.compute_vmm(rhs)
            
            # Apply boundary conditions
            new_state = self._apply_domain_boundary_conditions(
                domain_type, new_state, current_time
            )
            
        except Exception as e:
            self.logger.warning(f"Domain {domain_type.value} solve failed: {e}")
            new_state = current_state  # Fallback to previous state
        
        return new_state
    
    def _apply_domain_boundary_conditions(
        self,
        domain_type: PhysicsDomain,
        state: np.ndarray,
        current_time: float
    ) -> np.ndarray:
        """Apply boundary conditions for specific domain."""
        domain_config = self.physics_domains[domain_type]
        boundary_conditions = domain_config.boundary_conditions
        
        modified_state = state.copy()
        
        if 'dirichlet' in boundary_conditions:
            # Dirichlet boundary conditions
            bc_value = boundary_conditions.get('dirichlet_value', 0.0)
            if callable(bc_value):
                bc_value = bc_value(current_time)
            
            # Apply to boundaries (simplified)
            modified_state[0] = bc_value  # First node
            modified_state[-1] = bc_value  # Last node
            
        elif 'neumann' in boundary_conditions:
            # Neumann boundary conditions (flux specified)
            flux_value = boundary_conditions.get('neumann_value', 0.0)
            if callable(flux_value):
                flux_value = flux_value(current_time)
            
            # Apply flux condition (simplified finite difference)
            dx = domain_config.length_scale / len(modified_state)
            modified_state[0] = modified_state[1] - flux_value * dx
            modified_state[-1] = modified_state[-2] + flux_value * dx
        
        return modified_state
    
    def _compute_coupling_residual(
        self,
        current_states: Dict[PhysicsDomain, np.ndarray],
        previous_states: Dict[PhysicsDomain, np.ndarray]
    ) -> float:
        """Compute coupling residual for convergence check."""
        total_residual = 0.0
        num_domains = 0
        
        for domain_type in current_states.keys():
            if domain_type in previous_states:
                current = current_states[domain_type]
                previous = previous_states[domain_type]
                
                # L2 norm of state change
                domain_residual = np.linalg.norm(current - previous)
                total_residual += domain_residual
                num_domains += 1
        
        return total_residual / max(1, num_domains)
    
    def _check_conservation(
        self,
        current_states: Dict[PhysicsDomain, np.ndarray]
    ) -> float:
        """Check conservation properties across coupled system."""
        conservation_error = 0.0
        
        for interface in self.coupling_interfaces:
            if interface.conservation_required:
                source_domain = interface.source_domain
                target_domain = interface.target_domain
                
                if source_domain in current_states and target_domain in current_states:
                    source_state = current_states[source_domain]
                    target_state = current_states[target_domain]
                    
                    # Simple conservation check: total quantity conservation
                    source_total = np.sum(source_state)
                    target_total = np.sum(target_state)
                    
                    # For bidirectional coupling, check balance
                    if interface.bidirectional:
                        # Total should be conserved
                        total_error = abs(source_total + target_total)
                    else:
                        # Source should influence target consistently
                        coupling_strength = interface.coupling_strength
                        expected_target = target_total + coupling_strength * source_total
                        total_error = abs(target_total - expected_target)
                    
                    conservation_error += total_error
        
        return conservation_error
    
    def _compute_interface_activity(
        self,
        interface_key: str,
        current_states: Dict[PhysicsDomain, np.ndarray]
    ) -> float:
        """Compute activity level for coupling interface."""
        if interface_key in self.coupling_matrices:
            coupling_matrix = self.coupling_matrices[interface_key]
            
            # Extract domain types from interface key
            source_name, target_name = interface_key.split('_to_')
            
            try:
                source_domain = PhysicsDomain(source_name)
                target_domain = PhysicsDomain(target_name)
                
                if source_domain in current_states:
                    source_state = current_states[source_domain]
                    
                    # Compute coupling strength as interface activity
                    if coupling_matrix.shape[1] == len(source_state):
                        coupling_output = coupling_matrix @ source_state
                        activity = np.linalg.norm(coupling_output)
                    else:
                        activity = np.linalg.norm(coupling_matrix)
                    
                    return activity
                    
            except ValueError:
                pass  # Invalid domain name
        
        return 0.0
    
    def analyze_coupling_efficiency(self) -> Dict[str, Any]:
        """Analyze coupling efficiency and interface utilization."""
        analysis = {
            'interface_utilization': {},
            'domain_efficiency': {},
            'conservation_quality': {},
            'coupling_overhead_estimate': 0.0
        }
        
        # Interface utilization analysis
        for interface_key, conductances in self.interface_conductances.items():
            # Measure how much of the conductance range is utilized
            g_min, g_max = self.primary_crossbar.g_min, self.primary_crossbar.g_max
            normalized_conductances = (conductances - g_min) / (g_max - g_min)
            
            utilization = {
                'mean_utilization': np.mean(normalized_conductances),
                'std_utilization': np.std(normalized_conductances),
                'dynamic_range': np.ptp(conductances),  # Peak-to-peak range
                'active_elements': np.sum(normalized_conductances > 0.01)
            }
            
            analysis['interface_utilization'][interface_key] = utilization
        
        # Domain efficiency analysis
        for domain_type, domain_info in self.domain_crossbars.items():
            domain_conductances = domain_info['crossbar'].conductance_matrix
            
            efficiency = {
                'conductance_uniformity': 1.0 / (1.0 + np.std(domain_conductances)),
                'active_fraction': np.mean(domain_conductances > self.primary_crossbar.g_min * 1.1),
                'state_stability': np.std(domain_info['state']) if len(domain_info['state']) > 1 else 0.0
            }
            
            analysis['domain_efficiency'][domain_type.value] = efficiency
        
        # Conservation quality analysis
        if self.conservation_errors:
            analysis['conservation_quality'] = {
                'mean_error': np.mean(self.conservation_errors),
                'max_error': np.max(self.conservation_errors),
                'error_trend': np.polyfit(range(len(self.conservation_errors)), self.conservation_errors, 1)[0],
                'violations': sum(1 for err in self.conservation_errors if err > self.conservation_tolerance)
            }
        
        # Coupling overhead estimate
        total_interfaces = len(self.coupling_interfaces)
        active_interfaces = sum(1 for iface in self.coupling_interfaces 
                              if any(region for region in iface.interface_regions))
        
        if total_interfaces > 0:
            analysis['coupling_overhead_estimate'] = 1.0 - (active_interfaces / total_interfaces)
        
        return analysis
    
    def visualize_coupling_network(self) -> Dict[str, Any]:
        """Generate visualization data for coupling network."""
        network_data = {
            'nodes': [],
            'edges': [],
            'node_properties': {},
            'edge_properties': {}
        }
        
        # Add domain nodes
        for domain_type, domain_config in self.physics_domains.items():
            node_data = {
                'id': domain_type.value,
                'type': 'domain',
                'equations': domain_config.governing_equations,
                'allocation': domain_config.crossbar_allocation,
                'time_scale': domain_config.time_scale,
                'length_scale': domain_config.length_scale
            }
            
            network_data['nodes'].append(node_data)
            
            # Add node properties for visualization
            if domain_type in self.domain_crossbars:
                state = self.domain_crossbars[domain_type]['state']
                network_data['node_properties'][domain_type.value] = {
                    'state_magnitude': np.linalg.norm(state),
                    'state_mean': np.mean(state),
                    'state_std': np.std(state)
                }
        
        # Add coupling edges
        for interface in self.coupling_interfaces:
            edge_data = {
                'source': interface.source_domain.value,
                'target': interface.target_domain.value,
                'type': interface.coupling_type,
                'strength': interface.coupling_strength,
                'bidirectional': interface.bidirectional,
                'conservation_required': interface.conservation_required
            }
            
            network_data['edges'].append(edge_data)
            
            # Add edge properties
            interface_key = f"{interface.source_domain.value}_to_{interface.target_domain.value}"
            if interface_key in self.coupling_matrices:
                coupling_matrix = self.coupling_matrices[interface_key]
                network_data['edge_properties'][interface_key] = {
                    'matrix_norm': np.linalg.norm(coupling_matrix),
                    'matrix_rank': np.linalg.matrix_rank(coupling_matrix),
                    'condition_number': np.linalg.cond(coupling_matrix) if coupling_matrix.shape[0] == coupling_matrix.shape[1] else None
                }
        
        return network_data