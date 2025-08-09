"""SPICE simulator interface for crossbar validation."""

import numpy as np
import subprocess
import tempfile
import os
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CircuitComponent:
    """Circuit component definition."""
    name: str
    component_type: str
    nodes: tuple
    params: Dict[str, Any]
    
    def to_spice_line(self) -> str:
        """Convert component to SPICE netlist line."""
        if self.component_type == "resistor":
            return f"R{self.name} {self.nodes[0]} {self.nodes[1]} {self.params.get('resistance', '1k')}"
        elif self.component_type == "memristor":
            ron = self.params.get('ron', '1k')
            roff = self.params.get('roff', '1M')
            rinit = self.params.get('rinit', '10k')
            return f"R{self.name} {self.nodes[0]} {self.nodes[1]} {rinit}"
        elif self.component_type == "voltage_source":
            voltage = self.params.get('voltage', '1V')
            return f"V{self.name} {self.nodes[0]} {self.nodes[1]} DC {voltage}"
        elif self.component_type == "current_source":
            current = self.params.get('current', '1mA')
            return f"I{self.name} {self.nodes[0]} {self.nodes[1]} DC {current}"
        else:
            logger.warning(f"Unknown component type: {self.component_type}")
            return f"* Unknown component {self.name}"


class SPICESimulator:
    """Interface to NgSpice for crossbar circuit simulation."""
    
    def __init__(self, spice_command: str = "ngspice"):
        """Initialize SPICE simulator interface.
        
        Args:
            spice_command: SPICE simulator command (default: ngspice)
        """
        self.spice_command = spice_command
        self.netlist: List[str] = []
        self.components: Dict[str, CircuitComponent] = {}
        self.logger = logging.getLogger(__name__)
        
        # Check SPICE availability
        self._check_spice_availability()
        
    def _check_spice_availability(self) -> bool:
        """Check if SPICE simulator is available."""
        try:
            result = subprocess.run(
                [self.spice_command, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.logger.info(f"SPICE simulator found: {self.spice_command}")
                return True
            else:
                self.logger.warning(f"SPICE simulator not responding: {self.spice_command}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            self.logger.warning(f"SPICE simulator not found: {self.spice_command}")
            return False
    
    def add_component(
        self, 
        name: str, 
        component_type: str, 
        nodes: tuple, 
        params: Dict[str, Any]
    ) -> None:
        """Add component to SPICE netlist.
        
        Args:
            name: Component name
            component_type: Type of component (resistor, memristor, voltage_source, etc.)
            nodes: Tuple of node names/numbers
            params: Component parameters dictionary
        """
        component = CircuitComponent(name, component_type, nodes, params)
        self.components[name] = component
        self.logger.debug(f"Added component {name}: {component_type}")
        
    def add_crossbar_array(
        self,
        rows: int,
        cols: int, 
        resistance_matrix: np.ndarray,
        prefix: str = "X"
    ) -> None:
        """Add crossbar array to netlist.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            resistance_matrix: Resistance values for each crosspoint
            prefix: Component name prefix
        """
        for i in range(rows):
            for j in range(cols):
                component_name = f"{prefix}_{i}_{j}"
                resistance = resistance_matrix[i, j] if resistance_matrix is not None else 10e3
                
                self.add_component(
                    name=component_name,
                    component_type="memristor",
                    nodes=(f"row_{i}", f"col_{j}"),
                    params={
                        "ron": "1k",
                        "roff": "1M", 
                        "rinit": f"{resistance}"
                    }
                )
    
    def add_dac_array(
        self,
        name: str,
        resolution: int,
        voltage_range: float,
        num_channels: int = 1
    ) -> None:
        """Add DAC array to netlist.
        
        Args:
            name: DAC array name
            resolution: DAC resolution in bits
            voltage_range: Full-scale voltage range
            num_channels: Number of DAC channels
        """
        # Simplified DAC model as voltage sources
        for ch in range(num_channels):
            self.add_component(
                name=f"{name}_ch{ch}",
                component_type="voltage_source",
                nodes=(f"dac_out_{ch}", "gnd"),
                params={"voltage": f"{voltage_range/2}V"}
            )
            
        self.logger.debug(f"Added {resolution}-bit DAC array '{name}' with {num_channels} channels")
    
    def add_adc_array(
        self,
        name: str,
        resolution: int,
        sampling_rate: float,
        num_channels: int = 1
    ) -> None:
        """Add ADC array to netlist.
        
        Args:
            name: ADC array name
            resolution: ADC resolution in bits
            sampling_rate: Sampling rate in Hz
            num_channels: Number of ADC channels
        """
        # ADCs are typically measurement points in SPICE
        for ch in range(num_channels):
            # Create a high impedance path to ground for measurement
            self.add_component(
                name=f"{name}_load_{ch}",
                component_type="resistor",
                nodes=(f"adc_in_{ch}", "gnd"),
                params={"resistance": "1G"}  # High impedance
            )
            
        self.logger.debug(f"Added {resolution}-bit ADC array '{name}' with {num_channels} channels")
    
    def generate_netlist(
        self,
        title: str = "Analog Crossbar Simulation",
        analysis_type: str = "DC"
    ) -> List[str]:
        """Generate complete SPICE netlist.
        
        Args:
            title: Simulation title
            analysis_type: Type of analysis (DC, AC, TRAN)
            
        Returns:
            List of netlist lines
        """
        netlist = [f"* {title}"]
        netlist.append("")
        
        # Add components
        for component in self.components.values():
            netlist.append(component.to_spice_line())
        
        netlist.append("")
        
        # Add analysis commands
        if analysis_type == "DC":
            netlist.append(".op")
        elif analysis_type == "TRAN":
            netlist.append(".tran 1u 1m")
        elif analysis_type == "AC":
            netlist.append(".ac dec 10 1 1G")
        
        # Add output commands
        netlist.append(".print all")
        netlist.append(".control")
        netlist.append("run")
        netlist.append("print all")
        netlist.append(".endc")
        netlist.append(".end")
        
        return netlist
    
    def dc_analysis(
        self,
        output_nodes: Optional[List[str]] = None
    ) -> 'SimulationResults':
        """Run DC operating point analysis.
        
        Args:
            output_nodes: Nodes to analyze (optional)
            
        Returns:
            Simulation results
        """
        netlist = self.generate_netlist(analysis_type="DC")
        return self._run_simulation(netlist)
    
    def transient(
        self, 
        stop_time: float, 
        time_step: float,
        initial_conditions: Optional[Dict[str, float]] = None
    ) -> 'SimulationResults':
        """Run transient SPICE simulation.
        
        Args:
            stop_time: Simulation stop time (seconds)
            time_step: Time step (seconds) 
            initial_conditions: Initial node voltages (optional)
            
        Returns:
            Simulation results
        """
        # Generate netlist with transient analysis
        netlist = [f"* Transient Analysis"]
        netlist.append("")
        
        # Add components
        for component in self.components.values():
            netlist.append(component.to_spice_line())
        
        # Add initial conditions
        if initial_conditions:
            for node, voltage in initial_conditions.items():
                netlist.append(f".ic V({node})={voltage}")
        
        netlist.append("")
        netlist.append(f".tran {time_step} {stop_time}")
        netlist.append(".control")
        netlist.append("run")
        netlist.append("print all")
        netlist.append(".endc")
        netlist.append(".end")
        
        return self._run_simulation(netlist)
    
    def _run_simulation(self, netlist: List[str]) -> 'SimulationResults':
        """Execute SPICE simulation.
        
        Args:
            netlist: SPICE netlist lines
            
        Returns:
            Simulation results
        """
        try:
            # Write netlist to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
                f.write('\n'.join(netlist))
                netlist_file = f.name
            
            # Run SPICE simulation
            result = subprocess.run(
                [self.spice_command, "-b", netlist_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temporary file
            os.unlink(netlist_file)
            
            if result.returncode == 0:
                self.logger.info("SPICE simulation completed successfully")
                return SimulationResults(result.stdout, result.stderr)
            else:
                self.logger.error(f"SPICE simulation failed: {result.stderr}")
                return SimulationResults("", result.stderr, success=False)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.error(f"SPICE simulation error: {e}")
            return SimulationResults("", str(e), success=False)
        except Exception as e:
            self.logger.error(f"Unexpected error in SPICE simulation: {e}")
            return SimulationResults("", str(e), success=False)


class SimulationResults:
    """Container for SPICE simulation results."""
    
    def __init__(
        self, 
        stdout: str = "", 
        stderr: str = "",
        success: bool = True
    ):
        """Initialize simulation results.
        
        Args:
            stdout: SPICE standard output
            stderr: SPICE standard error
            success: Whether simulation succeeded
        """
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.logger = logging.getLogger(__name__)
        
        # Parse results
        self.node_voltages: Dict[str, float] = {}
        self.node_currents: Dict[str, float] = {}
        
        if success and stdout:
            self._parse_results()
    
    def _parse_results(self) -> None:
        """Parse SPICE output to extract results."""
        lines = self.stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse voltage results (format: v(node) = value)
            if 'v(' in line.lower() and '=' in line:
                try:
                    parts = line.split('=')
                    if len(parts) == 2:
                        node_part = parts[0].strip()
                        voltage_part = parts[1].strip()
                        
                        # Extract node name
                        if 'v(' in node_part.lower():
                            node_name = node_part.lower().replace('v(', '').replace(')', '')
                            voltage_value = float(voltage_part)
                            self.node_voltages[node_name] = voltage_value
                            
                except (ValueError, IndexError):
                    continue
            
            # Parse current results (format: i(component) = value)  
            elif 'i(' in line.lower() and '=' in line:
                try:
                    parts = line.split('=')
                    if len(parts) == 2:
                        current_part = parts[0].strip()
                        current_value = float(parts[1].strip())
                        
                        # Extract component name
                        if 'i(' in current_part.lower():
                            component_name = current_part.lower().replace('i(', '').replace(')', '')
                            self.node_currents[component_name] = current_value
                            
                except (ValueError, IndexError):
                    continue
    
    def get_node_voltages(self, nodes: Union[str, List[str]]) -> Dict[str, float]:
        """Extract node voltages from simulation.
        
        Args:
            nodes: Node name or list of node names
            
        Returns:
            Dictionary mapping node names to voltages
        """
        if isinstance(nodes, str):
            nodes = [nodes]
        
        result = {}
        for node in nodes:
            if node in self.node_voltages:
                result[node] = self.node_voltages[node]
            else:
                self.logger.warning(f"Node voltage not found: {node}")
                result[node] = 0.0
        
        return result
    
    def get_node_currents(self, components: Union[str, List[str]]) -> Dict[str, float]:
        """Extract component currents from simulation.
        
        Args:
            components: Component name or list of component names
            
        Returns:
            Dictionary mapping component names to currents
        """
        if isinstance(components, str):
            components = [components]
        
        result = {}
        for component in components:
            if component in self.node_currents:
                result[component] = self.node_currents[component]
            else:
                self.logger.warning(f"Component current not found: {component}")
                result[component] = 0.0
                
        return result
    
    def get_power_consumption(self) -> float:
        """Calculate total power consumption.
        
        Returns:
            Total power in watts
        """
        total_power = 0.0
        
        # Calculate P = V * I for each component
        for component, current in self.node_currents.items():
            # This is simplified - would need voltage across component
            if component.startswith('v'):  # Voltage source
                total_power += abs(current * 1.0)  # Assume 1V for simplicity
        
        return total_power
    
    def summary(self) -> str:
        """Generate summary of simulation results.
        
        Returns:
            Formatted summary string
        """
        if not self.success:
            return f"Simulation failed: {self.stderr}"
        
        summary_lines = ["SPICE Simulation Results Summary:"]
        summary_lines.append("=" * 40)
        
        if self.node_voltages:
            summary_lines.append(f"Node Voltages ({len(self.node_voltages)} nodes):")
            for node, voltage in sorted(self.node_voltages.items()):
                summary_lines.append(f"  {node}: {voltage:.6f} V")
        
        if self.node_currents:
            summary_lines.append(f"Component Currents ({len(self.node_currents)} components):")
            for comp, current in sorted(self.node_currents.items()):
                summary_lines.append(f"  {comp}: {current:.6f} A")
        
        total_power = self.get_power_consumption()
        if total_power > 0:
            summary_lines.append(f"Total Power: {total_power:.6f} W")
        
        return '\n'.join(summary_lines)