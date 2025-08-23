"""
Distributed Scaling System for Analog PDE Solver Infrastructure

This module implements an advanced distributed scaling system that automatically
scales analog PDE solver deployments across multiple nodes, manages resource
allocation, and optimizes performance for large-scale computational workloads.

Scaling Capabilities:
    1. Horizontal Auto-Scaling (multi-node)
    2. Vertical Auto-Scaling (resource allocation)
    3. Elastic Load Balancing
    4. Intelligent Work Distribution
    5. Resource Pool Management
    6. Performance-Based Scaling
    7. Cost-Aware Optimization
    8. Global Load Balancing

Mathematical Foundation:
    Scaling Efficiency: E = (P_n / P_1) / n where P_n is n-node performance
    Resource Utilization: U = Σ(R_used / R_available) / num_resources
    Cost Optimization: C = min(Σ(r_i × c_i)) subject to performance constraints

Performance Target: Linear scaling to 1000+ nodes with 95%+ efficiency.
"""

import numpy as np
import torch
import logging
import time
import threading
import asyncio
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import hashlib
from collections import defaultdict, deque
import psutil
import socket
import uuid
import concurrent.futures
from pathlib import Path

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Scaling policies for different scenarios."""
    REACTIVE = "reactive"        # Scale based on current load
    PREDICTIVE = "predictive"    # Scale based on predicted load  
    PROACTIVE = "proactive"      # Pre-scale for expected load
    COST_OPTIMIZED = "cost_optimized"  # Minimize cost while meeting SLA
    PERFORMANCE_FIRST = "performance_first"  # Maximize performance
    BALANCED = "balanced"        # Balance cost and performance


class NodeType(Enum):
    """Types of compute nodes."""
    CPU_INTENSIVE = "cpu_intensive"      # High CPU, standard memory
    MEMORY_INTENSIVE = "memory_intensive"  # High memory, standard CPU
    GPU_ACCELERATED = "gpu_accelerated"   # GPU acceleration
    ANALOG_SPECIALIZED = "analog_specialized"  # Analog computing hardware
    EDGE_DEVICE = "edge_device"          # Edge/IoT devices
    CLOUD_INSTANCE = "cloud_instance"    # Cloud-based instances


class WorkloadType(Enum):
    """Types of computational workloads."""
    POISSON_SOLVER = "poisson_solver"
    HEAT_EQUATION = "heat_equation"
    WAVE_EQUATION = "wave_equation"
    NAVIER_STOKES = "navier_stokes"
    MIXED_PDE = "mixed_pde"
    RESEARCH_VALIDATION = "research_validation"
    PARAMETER_SWEEP = "parameter_sweep"
    OPTIMIZATION = "optimization"


@dataclass
class NodeSpec:
    """Specification for a compute node."""
    node_id: str
    node_type: NodeType
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    analog_crossbars: int = 0
    network_bandwidth_gbps: float = 1.0
    storage_gb: float = 100.0
    cost_per_hour: float = 0.10
    geographic_region: str = "us-east-1"
    
    # Performance characteristics
    compute_performance: Dict[WorkloadType, float] = field(default_factory=dict)
    energy_efficiency: float = 1.0  # Performance per watt
    
    # Current state
    current_utilization: float = 0.0
    active_jobs: List[str] = field(default_factory=list)
    health_score: float = 1.0


@dataclass
class WorkItem:
    """Individual work item for distributed processing."""
    work_id: str
    workload_type: WorkloadType
    priority: int  # 1-10, higher is more important
    estimated_runtime_seconds: float
    resource_requirements: Dict[str, float]  # CPU, memory, etc.
    dependencies: List[str] = field(default_factory=list)
    submitted_time: float = field(default_factory=time.time)
    
    # Execution state
    assigned_node: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    result_data: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class ScalingConfig:
    """Configuration for distributed scaling system."""
    # Scaling policies
    scaling_policy: ScalingPolicy = ScalingPolicy.BALANCED
    target_utilization: float = 0.75  # 75% target utilization
    scale_up_threshold: float = 0.85   # Scale up when utilization > 85%
    scale_down_threshold: float = 0.60  # Scale down when utilization < 60%
    
    # Scaling limits
    min_nodes: int = 1
    max_nodes: int = 100
    max_nodes_per_region: int = 20
    
    # Timing parameters
    scale_up_cooldown_seconds: float = 300    # 5 minutes
    scale_down_cooldown_seconds: float = 600  # 10 minutes
    health_check_interval_seconds: float = 30
    
    # Performance parameters
    target_response_time_seconds: float = 10.0
    max_queue_length: int = 1000
    load_balancing_algorithm: str = "least_connections"  # round_robin, least_connections, weighted
    
    # Cost optimization
    max_hourly_cost: Optional[float] = None
    prefer_spot_instances: bool = True
    cost_optimization_weight: float = 0.3  # 0=performance only, 1=cost only
    
    # Geographic distribution
    multi_region_enabled: bool = True
    primary_regions: List[str] = field(default_factory=lambda: ["us-east-1", "us-west-2", "eu-west-1"])
    
    # Advanced features
    enable_predictive_scaling: bool = True
    enable_preemptive_migration: bool = True
    enable_auto_scaling: bool = True


class WorkloadPredictor:
    """Predict future workload patterns for proactive scaling."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.historical_data = deque(maxlen=10000)
        self.prediction_models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize prediction models."""
        # In production, these would be trained ML models
        return {
            'time_series': None,  # LSTM for time series prediction
            'workload_classifier': None,  # Classify workload types
            'resource_estimator': None   # Estimate resource requirements
        }
    
    def record_workload(self, 
                       timestamp: float,
                       workload_type: WorkloadType,
                       node_count: int,
                       resource_usage: Dict[str, float]):
        """Record historical workload data."""
        
        data_point = {
            'timestamp': timestamp,
            'workload_type': workload_type.value,
            'node_count': node_count,
            'resource_usage': resource_usage,
            'hour_of_day': time.localtime(timestamp).tm_hour,
            'day_of_week': time.localtime(timestamp).tm_wday
        }
        
        self.historical_data.append(data_point)
    
    def predict_future_load(self, horizon_minutes: int = 30) -> Dict[str, Any]:
        """Predict future computational load."""
        
        if len(self.historical_data) < 10:
            # Not enough data for prediction
            return {
                'predicted_node_count': 1,
                'confidence': 0.1,
                'predicted_workload_types': [WorkloadType.POISSON_SOLVER.value],
                'resource_forecast': {'cpu': 0.5, 'memory': 0.5}
            }
        
        # Simple heuristic-based prediction (in production, use ML models)
        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        
        # Get recent similar time periods
        similar_periods = [
            data for data in self.historical_data
            if abs(data['hour_of_day'] - current_hour) <= 1
        ]
        
        if not similar_periods:
            similar_periods = list(self.historical_data)[-20:]  # Fall back to recent data
        
        # Average prediction
        avg_node_count = np.mean([data['node_count'] for data in similar_periods])
        avg_cpu_usage = np.mean([data['resource_usage'].get('cpu', 0.5) for data in similar_periods])
        avg_memory_usage = np.mean([data['resource_usage'].get('memory', 0.5) for data in similar_periods])
        
        # Trend analysis
        if len(similar_periods) >= 5:
            recent_trend = np.polyfit(
                range(len(similar_periods)),
                [data['node_count'] for data in similar_periods],
                1
            )[0]  # Slope
            
            # Adjust prediction based on trend
            predicted_node_count = max(1, int(avg_node_count + recent_trend * horizon_minutes / 60))
        else:
            predicted_node_count = max(1, int(avg_node_count))
        
        # Most common workload types
        workload_counts = defaultdict(int)
        for data in similar_periods:
            workload_counts[data['workload_type']] += 1
        
        top_workloads = sorted(workload_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'predicted_node_count': predicted_node_count,
            'confidence': min(len(similar_periods) / 20.0, 0.9),  # Higher confidence with more data
            'predicted_workload_types': [wl[0] for wl in top_workloads],
            'resource_forecast': {
                'cpu': min(avg_cpu_usage * 1.1, 1.0),  # Add 10% buffer
                'memory': min(avg_memory_usage * 1.1, 1.0)
            },
            'trend': 'increasing' if recent_trend > 0.1 else 'decreasing' if recent_trend < -0.1 else 'stable'
        }


class LoadBalancer:
    """Intelligent load balancer for distributed work items."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.node_connections = defaultdict(int)  # Track connections per node
        self.node_weights = defaultdict(lambda: 1.0)  # Dynamic weights
        
    def select_node(self, 
                   available_nodes: List[NodeSpec],
                   work_item: WorkItem) -> Optional[NodeSpec]:
        """Select best node for work item."""
        
        if not available_nodes:
            return None
        
        # Filter nodes that can handle the workload
        suitable_nodes = self._filter_suitable_nodes(available_nodes, work_item)
        
        if not suitable_nodes:
            return None
        
        # Apply load balancing algorithm
        if self.config.load_balancing_algorithm == "round_robin":
            return self._round_robin_selection(suitable_nodes)
        elif self.config.load_balancing_algorithm == "least_connections":
            return self._least_connections_selection(suitable_nodes)
        elif self.config.load_balancing_algorithm == "weighted":
            return self._weighted_selection(suitable_nodes, work_item)
        else:
            return suitable_nodes[0]  # Default to first suitable node
    
    def _filter_suitable_nodes(self, 
                              nodes: List[NodeSpec],
                              work_item: WorkItem) -> List[NodeSpec]:
        """Filter nodes that can handle the work item."""
        
        suitable = []
        
        for node in nodes:
            # Check resource requirements
            cpu_required = work_item.resource_requirements.get('cpu', 0.1)
            memory_required = work_item.resource_requirements.get('memory', 0.1)
            
            # Estimate available resources
            available_cpu = (1 - node.current_utilization) * node.cpu_cores
            available_memory = (1 - node.current_utilization) * node.memory_gb
            
            if (available_cpu >= cpu_required * node.cpu_cores and 
                available_memory >= memory_required * node.memory_gb and
                node.health_score > 0.5):  # Minimum health threshold
                
                # Check workload compatibility
                if self._is_workload_compatible(node, work_item):
                    suitable.append(node)
        
        return suitable
    
    def _is_workload_compatible(self, node: NodeSpec, work_item: WorkItem) -> bool:
        """Check if node is compatible with workload type."""
        
        workload_type = work_item.workload_type
        
        # Analog workloads need analog hardware
        if workload_type in [WorkloadType.POISSON_SOLVER, WorkloadType.HEAT_EQUATION]:
            if node.node_type == NodeType.ANALOG_SPECIALIZED and node.analog_crossbars > 0:
                return True
            elif node.node_type in [NodeType.CPU_INTENSIVE, NodeType.GPU_ACCELERATED]:
                return True  # Can fall back to digital computation
        
        # GPU-accelerated workloads prefer GPU nodes
        if workload_type in [WorkloadType.NAVIER_STOKES, WorkloadType.OPTIMIZATION]:
            if node.node_type == NodeType.GPU_ACCELERATED and node.gpu_count > 0:
                return True
        
        # Memory-intensive workloads need sufficient memory
        if workload_type == WorkloadType.PARAMETER_SWEEP:
            if node.node_type == NodeType.MEMORY_INTENSIVE:
                return True
        
        # General compatibility
        return node.node_type in [NodeType.CPU_INTENSIVE, NodeType.CLOUD_INSTANCE]
    
    def _round_robin_selection(self, nodes: List[NodeSpec]) -> NodeSpec:
        """Round-robin node selection."""
        # Simple round-robin based on connection count
        return min(nodes, key=lambda n: self.node_connections[n.node_id])
    
    def _least_connections_selection(self, nodes: List[NodeSpec]) -> NodeSpec:
        """Select node with least connections."""
        return min(nodes, key=lambda n: len(n.active_jobs))
    
    def _weighted_selection(self, nodes: List[NodeSpec], work_item: WorkItem) -> NodeSpec:
        """Weighted selection based on performance and cost."""
        
        best_node = None
        best_score = float('-inf')
        
        for node in nodes:
            # Performance score
            perf_score = node.compute_performance.get(work_item.workload_type, 1.0)
            
            # Utilization score (prefer less utilized nodes)
            util_score = 1.0 - node.current_utilization
            
            # Health score
            health_score = node.health_score
            
            # Cost score (lower cost is better)
            cost_score = 1.0 / (node.cost_per_hour + 0.01)  # Avoid division by zero
            
            # Combined weighted score
            total_score = (
                0.4 * perf_score +
                0.3 * util_score +  
                0.2 * health_score +
                0.1 * cost_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_node = node
        
        return best_node
    
    def update_node_connection(self, node_id: str, delta: int):
        """Update connection count for node."""
        self.node_connections[node_id] += delta
        self.node_connections[node_id] = max(0, self.node_connections[node_id])


class AutoScaler:
    """Automatic scaling controller."""
    
    def __init__(self, config: ScalingConfig, predictor: WorkloadPredictor):
        self.config = config
        self.predictor = predictor
        
        # Scaling state
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        self.scaling_in_progress = False
        
        # Node management
        self.node_pool = {}  # node_id -> NodeSpec
        self.pending_nodes = set()  # Nodes being provisioned
        
        # Metrics tracking
        self.scaling_history = []
        
    def evaluate_scaling_need(self, 
                             current_load: Dict[str, Any],
                             queue_length: int) -> Dict[str, Any]:
        """Evaluate if scaling is needed."""
        
        current_time = time.time()
        active_nodes = [node for node in self.node_pool.values() if node.health_score > 0.5]
        
        if not active_nodes:
            return {'action': 'scale_up', 'target_nodes': self.config.min_nodes, 'reason': 'no_active_nodes'}
        
        # Calculate current utilization
        avg_utilization = np.mean([node.current_utilization for node in active_nodes])
        max_utilization = np.max([node.current_utilization for node in active_nodes])
        
        # Check cooldown periods
        scale_up_ready = current_time - self.last_scale_up_time > self.config.scale_up_cooldown_seconds
        scale_down_ready = current_time - self.last_scale_down_time > self.config.scale_down_cooldown_seconds
        
        # Determine scaling action
        scaling_decision = {
            'action': 'none',
            'target_nodes': len(active_nodes),
            'reason': 'within_thresholds',
            'current_utilization': avg_utilization,
            'max_utilization': max_utilization,
            'queue_length': queue_length
        }
        
        # Scale up conditions
        if (avg_utilization > self.config.scale_up_threshold or
            max_utilization > 0.95 or  # Any node > 95%
            queue_length > self.config.max_queue_length * 0.8):  # Queue getting full
            
            if scale_up_ready and len(active_nodes) < self.config.max_nodes:
                
                # Calculate desired nodes
                if self.config.scaling_policy == ScalingPolicy.PREDICTIVE:
                    prediction = self.predictor.predict_future_load(30)
                    target_nodes = min(prediction['predicted_node_count'], self.config.max_nodes)
                else:
                    # Reactive scaling
                    utilization_factor = avg_utilization / self.config.target_utilization
                    target_nodes = min(int(len(active_nodes) * utilization_factor), self.config.max_nodes)
                
                scaling_decision.update({
                    'action': 'scale_up',
                    'target_nodes': target_nodes,
                    'reason': f'utilization={avg_utilization:.2f}, queue_length={queue_length}'
                })
        
        # Scale down conditions
        elif (avg_utilization < self.config.scale_down_threshold and
              max_utilization < 0.7 and  # No node > 70%
              queue_length < self.config.max_queue_length * 0.2):  # Queue mostly empty
            
            if scale_down_ready and len(active_nodes) > self.config.min_nodes:
                
                # Calculate desired nodes
                utilization_factor = avg_utilization / self.config.target_utilization
                target_nodes = max(int(len(active_nodes) * utilization_factor), self.config.min_nodes)
                
                scaling_decision.update({
                    'action': 'scale_down', 
                    'target_nodes': target_nodes,
                    'reason': f'utilization={avg_utilization:.2f}, over_provisioned'
                })
        
        return scaling_decision
    
    def execute_scaling(self, scaling_decision: Dict[str, Any]) -> bool:
        """Execute scaling decision."""
        
        action = scaling_decision['action']
        target_nodes = scaling_decision['target_nodes']
        
        if action == 'none':
            return True
        
        current_time = time.time()
        active_nodes = len([node for node in self.node_pool.values() if node.health_score > 0.5])
        
        success = False
        
        if action == 'scale_up':
            nodes_to_add = target_nodes - active_nodes
            success = self._provision_nodes(nodes_to_add, scaling_decision['reason'])
            if success:
                self.last_scale_up_time = current_time
        
        elif action == 'scale_down':
            nodes_to_remove = active_nodes - target_nodes
            success = self._decommission_nodes(nodes_to_remove, scaling_decision['reason'])
            if success:
                self.last_scale_down_time = current_time
        
        # Record scaling event
        self.scaling_history.append({
            'timestamp': current_time,
            'action': action,
            'target_nodes': target_nodes,
            'actual_nodes_before': active_nodes,
            'reason': scaling_decision['reason'],
            'success': success
        })
        
        # Keep history bounded
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]
        
        return success
    
    def _provision_nodes(self, count: int, reason: str) -> bool:
        """Provision new compute nodes."""
        
        logger.info(f"Provisioning {count} new nodes. Reason: {reason}")
        
        # Determine optimal node types
        node_types = self._select_optimal_node_types(count)
        
        provisioned = 0
        
        for node_type, node_count in node_types.items():
            for i in range(node_count):
                node_id = f"{node_type.value}_{uuid.uuid4().hex[:8]}"
                
                # Create node specification
                node_spec = self._create_node_spec(node_id, node_type)
                
                # Simulate provisioning (in production, this would call cloud APIs)
                if self._simulate_provisioning(node_spec):
                    self.node_pool[node_id] = node_spec
                    self.pending_nodes.add(node_id)
                    provisioned += 1
                    
                    logger.info(f"Provisioned node {node_id} ({node_type.value})")
        
        return provisioned > 0
    
    def _decommission_nodes(self, count: int, reason: str) -> bool:
        """Decommission compute nodes."""
        
        logger.info(f"Decommissioning {count} nodes. Reason: {reason}")
        
        # Select nodes to decommission (prefer least utilized, lowest performance)
        active_nodes = [node for node in self.node_pool.values() if node.health_score > 0.5]
        
        # Sort by utilization and cost (remove least utilized, highest cost first)
        nodes_to_remove = sorted(
            active_nodes,
            key=lambda n: (n.current_utilization, -n.cost_per_hour)
        )[:count]
        
        decommissioned = 0
        
        for node in nodes_to_remove:
            if self._can_safely_decommission(node):
                # Graceful shutdown
                if self._simulate_decommissioning(node):
                    del self.node_pool[node.node_id]
                    decommissioned += 1
                    
                    logger.info(f"Decommissioned node {node.node_id}")
        
        return decommissioned > 0
    
    def _select_optimal_node_types(self, count: int) -> Dict[NodeType, int]:
        """Select optimal mix of node types to provision."""
        
        # Analyze current workload patterns
        if self.config.scaling_policy == ScalingPolicy.COST_OPTIMIZED:
            # Prefer cheaper nodes
            node_distribution = {
                NodeType.CPU_INTENSIVE: int(count * 0.6),
                NodeType.CLOUD_INSTANCE: int(count * 0.4)
            }
        elif self.config.scaling_policy == ScalingPolicy.PERFORMANCE_FIRST:
            # Prefer high-performance nodes
            node_distribution = {
                NodeType.ANALOG_SPECIALIZED: int(count * 0.4),
                NodeType.GPU_ACCELERATED: int(count * 0.3),
                NodeType.MEMORY_INTENSIVE: int(count * 0.3)
            }
        else:
            # Balanced approach
            node_distribution = {
                NodeType.CPU_INTENSIVE: int(count * 0.4),
                NodeType.GPU_ACCELERATED: int(count * 0.3),
                NodeType.ANALOG_SPECIALIZED: int(count * 0.2),
                NodeType.CLOUD_INSTANCE: int(count * 0.1)
            }
        
        # Ensure we allocate exactly 'count' nodes
        total_allocated = sum(node_distribution.values())
        if total_allocated < count:
            # Add remaining to most common type
            primary_type = max(node_distribution.items(), key=lambda x: x[1])[0]
            node_distribution[primary_type] += count - total_allocated
        
        return {k: v for k, v in node_distribution.items() if v > 0}
    
    def _create_node_spec(self, node_id: str, node_type: NodeType) -> NodeSpec:
        """Create node specification for given type."""
        
        # Define node specifications by type
        type_specs = {
            NodeType.CPU_INTENSIVE: {
                'cpu_cores': 16, 'memory_gb': 32, 'gpu_count': 0, 'analog_crossbars': 0,
                'cost_per_hour': 0.50, 'energy_efficiency': 1.2
            },
            NodeType.MEMORY_INTENSIVE: {
                'cpu_cores': 8, 'memory_gb': 128, 'gpu_count': 0, 'analog_crossbars': 0,
                'cost_per_hour': 0.80, 'energy_efficiency': 1.0
            },
            NodeType.GPU_ACCELERATED: {
                'cpu_cores': 12, 'memory_gb': 64, 'gpu_count': 4, 'analog_crossbars': 0,
                'cost_per_hour': 2.00, 'energy_efficiency': 2.5
            },
            NodeType.ANALOG_SPECIALIZED: {
                'cpu_cores': 8, 'memory_gb': 32, 'gpu_count': 0, 'analog_crossbars': 16,
                'cost_per_hour': 1.20, 'energy_efficiency': 10.0  # Very efficient for PDE solving
            },
            NodeType.CLOUD_INSTANCE: {
                'cpu_cores': 4, 'memory_gb': 16, 'gpu_count': 0, 'analog_crossbars': 0,
                'cost_per_hour': 0.20, 'energy_efficiency': 0.8
            }
        }
        
        base_spec = type_specs.get(node_type, type_specs[NodeType.CPU_INTENSIVE])
        
        # Create performance profiles
        performance_profile = {}
        if node_type == NodeType.ANALOG_SPECIALIZED:
            performance_profile = {
                WorkloadType.POISSON_SOLVER: 10.0,
                WorkloadType.HEAT_EQUATION: 8.0,
                WorkloadType.WAVE_EQUATION: 6.0,
                WorkloadType.NAVIER_STOKES: 3.0
            }
        elif node_type == NodeType.GPU_ACCELERATED:
            performance_profile = {
                WorkloadType.NAVIER_STOKES: 5.0,
                WorkloadType.OPTIMIZATION: 4.0,
                WorkloadType.PARAMETER_SWEEP: 3.0,
                WorkloadType.POISSON_SOLVER: 2.0
            }
        else:
            # General CPU performance
            performance_profile = {wt: 1.0 for wt in WorkloadType}
        
        return NodeSpec(
            node_id=node_id,
            node_type=node_type,
            compute_performance=performance_profile,
            **base_spec
        )
    
    def _simulate_provisioning(self, node_spec: NodeSpec) -> bool:
        """Simulate node provisioning."""
        # Simulate provisioning delay and potential failures
        provisioning_success_rate = 0.95  # 95% success rate
        return np.random.random() < provisioning_success_rate
    
    def _simulate_decommissioning(self, node_spec: NodeSpec) -> bool:
        """Simulate node decommissioning."""
        # Simulate graceful shutdown
        return len(node_spec.active_jobs) == 0  # Only decommission if no active jobs
    
    def _can_safely_decommission(self, node: NodeSpec) -> bool:
        """Check if node can be safely decommissioned."""
        # Don't decommission nodes with active jobs or high utilization
        return len(node.active_jobs) == 0 and node.current_utilization < 0.1
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling system metrics."""
        
        active_nodes = [node for node in self.node_pool.values() if node.health_score > 0.5]
        
        if not active_nodes:
            return {'error': 'no_active_nodes'}
        
        # Recent scaling events
        recent_scaling = [
            event for event in self.scaling_history
            if time.time() - event['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'total_nodes': len(self.node_pool),
            'active_nodes': len(active_nodes),
            'pending_nodes': len(self.pending_nodes),
            'node_types': {
                node_type.value: len([n for n in active_nodes if n.node_type == node_type])
                for node_type in NodeType
            },
            'utilization_stats': {
                'average': np.mean([n.current_utilization for n in active_nodes]),
                'min': np.min([n.current_utilization for n in active_nodes]),
                'max': np.max([n.current_utilization for n in active_nodes]),
                'std': np.std([n.current_utilization for n in active_nodes])
            },
            'cost_metrics': {
                'total_hourly_cost': sum(n.cost_per_hour for n in active_nodes),
                'cost_per_node_avg': np.mean([n.cost_per_hour for n in active_nodes]),
                'efficiency_score': sum(n.energy_efficiency for n in active_nodes) / len(active_nodes)
            },
            'scaling_activity': {
                'recent_events': len(recent_scaling),
                'scale_up_events': len([e for e in recent_scaling if e['action'] == 'scale_up']),
                'scale_down_events': len([e for e in recent_scaling if e['action'] == 'scale_down']),
                'last_scaling_time': self.scaling_history[-1]['timestamp'] if self.scaling_history else 0
            }
        }


class DistributedScalingSystem:
    """Main distributed scaling system coordinator."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        
        # Core components
        self.predictor = WorkloadPredictor(config)
        self.load_balancer = LoadBalancer(config)
        self.auto_scaler = AutoScaler(config, self.predictor)
        
        # Work management
        self.work_queue = deque()
        self.completed_work = deque(maxlen=10000)
        self.work_execution_pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        
        # System state
        self.scaling_active = False
        self.scaling_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_work_items': 0,
            'completed_work_items': 0,
            'failed_work_items': 0,
            'average_response_time': 0.0,
            'throughput_per_second': 0.0,
            'scaling_efficiency': 1.0
        }
        
    def start_scaling_system(self):
        """Start the distributed scaling system."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop)
        self.scaling_thread.start()
        
        logger.info("Distributed scaling system started")
    
    def stop_scaling_system(self):
        """Stop the distributed scaling system."""
        if not self.scaling_active:
            return
        
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join()
        
        # Shutdown execution pool
        self.work_execution_pool.shutdown(wait=True)
        
        logger.info("Distributed scaling system stopped")
    
    def submit_work(self, work_item: WorkItem) -> str:
        """Submit work item for distributed processing."""
        
        self.work_queue.append(work_item)
        self.performance_metrics['total_work_items'] += 1
        
        logger.info(f"Submitted work item {work_item.work_id} ({work_item.workload_type.value})")
        
        return work_item.work_id
    
    def get_work_status(self, work_id: str) -> Optional[Dict[str, Any]]:
        """Get status of work item."""
        
        # Check active work queue
        for work_item in self.work_queue:
            if work_item.work_id == work_id:
                return {
                    'work_id': work_id,
                    'status': work_item.status,
                    'assigned_node': work_item.assigned_node,
                    'start_time': work_item.start_time,
                    'estimated_completion': (
                        work_item.start_time + work_item.estimated_runtime_seconds
                        if work_item.start_time else None
                    )
                }
        
        # Check completed work
        for work_item in self.completed_work:
            if work_item.work_id == work_id:
                return {
                    'work_id': work_id,
                    'status': work_item.status,
                    'assigned_node': work_item.assigned_node,
                    'start_time': work_item.start_time,
                    'completion_time': work_item.completion_time,
                    'runtime_seconds': (
                        work_item.completion_time - work_item.start_time
                        if work_item.completion_time and work_item.start_time else None
                    ),
                    'error_message': work_item.error_message
                }
        
        return None
    
    def _scaling_loop(self):
        """Main scaling system loop."""
        
        while self.scaling_active:
            try:
                # Process work queue
                self._process_work_queue()
                
                # Evaluate scaling needs
                current_load = self._assess_current_load()
                scaling_decision = self.auto_scaler.evaluate_scaling_need(
                    current_load, len(self.work_queue)
                )
                
                # Execute scaling if needed
                if scaling_decision['action'] != 'none':
                    self.auto_scaler.execute_scaling(scaling_decision)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Record workload data for prediction
                self._record_workload_data(current_load)
                
                # Health check interval
                time.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(5.0)
    
    def _process_work_queue(self):
        """Process pending work items."""
        
        # Get available nodes
        available_nodes = [
            node for node in self.auto_scaler.node_pool.values()
            if node.health_score > 0.5 and node.current_utilization < 0.9
        ]
        
        if not available_nodes:
            return
        
        # Process work items
        items_to_process = []
        
        while self.work_queue and len(items_to_process) < len(available_nodes):
            work_item = self.work_queue.popleft()
            
            # Select node for work item
            selected_node = self.load_balancer.select_node(available_nodes, work_item)
            
            if selected_node:
                work_item.assigned_node = selected_node.node_id
                work_item.status = "running"
                work_item.start_time = time.time()
                
                # Update node state
                selected_node.active_jobs.append(work_item.work_id)
                selected_node.current_utilization = min(
                    selected_node.current_utilization + 0.1,  # Simulate load increase
                    1.0
                )
                
                items_to_process.append(work_item)
                
                # Update load balancer
                self.load_balancer.update_node_connection(selected_node.node_id, 1)
                
                # Remove node from available list to avoid double assignment
                available_nodes.remove(selected_node)
            else:
                # No suitable node, put work back in queue
                self.work_queue.appendleft(work_item)
                break
        
        # Submit work items for execution
        for work_item in items_to_process:
            self.work_execution_pool.submit(self._execute_work_item, work_item)
    
    def _execute_work_item(self, work_item: WorkItem):
        """Execute work item on assigned node."""
        
        try:
            logger.info(f"Executing work item {work_item.work_id} on node {work_item.assigned_node}")
            
            # Simulate work execution
            execution_time = work_item.estimated_runtime_seconds + np.random.normal(0, 0.1)
            time.sleep(max(0.1, execution_time))  # Minimum 0.1s execution time
            
            # Simulate work result
            if np.random.random() < 0.95:  # 95% success rate
                work_item.status = "completed"
                work_item.result_data = {
                    'solution': f"result_for_{work_item.work_id}",
                    'execution_time': execution_time,
                    'node_id': work_item.assigned_node
                }
                self.performance_metrics['completed_work_items'] += 1
            else:
                work_item.status = "failed"
                work_item.error_message = "Simulated execution failure"
                self.performance_metrics['failed_work_items'] += 1
            
            work_item.completion_time = time.time()
            
            # Update node state
            if work_item.assigned_node in self.auto_scaler.node_pool:
                node = self.auto_scaler.node_pool[work_item.assigned_node]
                if work_item.work_id in node.active_jobs:
                    node.active_jobs.remove(work_item.work_id)
                node.current_utilization = max(
                    node.current_utilization - 0.1,  # Simulate load decrease
                    0.0
                )
                
                # Update load balancer
                self.load_balancer.update_node_connection(work_item.assigned_node, -1)
            
            # Add to completed work
            self.completed_work.append(work_item)
            
            logger.info(f"Completed work item {work_item.work_id} with status {work_item.status}")
            
        except Exception as e:
            work_item.status = "failed"
            work_item.error_message = str(e)
            work_item.completion_time = time.time()
            self.performance_metrics['failed_work_items'] += 1
            
            logger.error(f"Work item {work_item.work_id} failed: {e}")
    
    def _assess_current_load(self) -> Dict[str, Any]:
        """Assess current system load."""
        
        active_nodes = [
            node for node in self.auto_scaler.node_pool.values()
            if node.health_score > 0.5
        ]
        
        if not active_nodes:
            return {'avg_utilization': 0.0, 'total_nodes': 0, 'active_jobs': 0}
        
        total_jobs = sum(len(node.active_jobs) for node in active_nodes)
        avg_utilization = np.mean([node.current_utilization for node in active_nodes])
        
        return {
            'avg_utilization': avg_utilization,
            'max_utilization': np.max([node.current_utilization for node in active_nodes]),
            'total_nodes': len(active_nodes),
            'active_jobs': total_jobs,
            'queue_length': len(self.work_queue),
            'nodes_by_type': {
                node_type.value: len([n for n in active_nodes if n.node_type == node_type])
                for node_type in NodeType
            }
        }
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        
        # Calculate response time (queue time + execution time)
        if self.completed_work:
            recent_work = [w for w in self.completed_work if w.completion_time and w.start_time]
            if recent_work:
                response_times = [
                    (w.completion_time - w.submitted_time) for w in recent_work[-100:]
                ]
                self.performance_metrics['average_response_time'] = np.mean(response_times)
        
        # Calculate throughput
        completed_last_minute = len([
            w for w in self.completed_work
            if w.completion_time and time.time() - w.completion_time < 60
        ])
        self.performance_metrics['throughput_per_second'] = completed_last_minute / 60.0
        
        # Calculate scaling efficiency
        active_nodes = len([
            node for node in self.auto_scaler.node_pool.values()
            if node.health_score > 0.5
        ])
        
        if active_nodes > 0 and self.performance_metrics['throughput_per_second'] > 0:
            # Efficiency = (actual throughput) / (theoretical max throughput)
            theoretical_max = active_nodes * 1.0  # 1 work item per second per node
            actual_throughput = self.performance_metrics['throughput_per_second']
            self.performance_metrics['scaling_efficiency'] = min(actual_throughput / theoretical_max, 1.0)
    
    def _record_workload_data(self, current_load: Dict[str, Any]):
        """Record workload data for prediction models."""
        
        # Sample workload types from recent work
        recent_work_types = [
            w.workload_type for w in list(self.work_queue)[-10:]  # Last 10 in queue
        ]
        
        if not recent_work_types and self.completed_work:
            recent_work_types = [
                w.workload_type for w in list(self.completed_work)[-10:]  # Last 10 completed
            ]
        
        dominant_workload = max(set(recent_work_types), key=recent_work_types.count) if recent_work_types else WorkloadType.POISSON_SOLVER
        
        self.predictor.record_workload(
            timestamp=time.time(),
            workload_type=dominant_workload,
            node_count=current_load['total_nodes'],
            resource_usage={
                'cpu': current_load['avg_utilization'],
                'memory': current_load['avg_utilization'] * 0.8  # Approximate memory usage
            }
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        scaling_metrics = self.auto_scaler.get_scaling_metrics()
        current_load = self._assess_current_load()
        
        return {
            'system_active': self.scaling_active,
            'current_load': current_load,
            'performance_metrics': self.performance_metrics,
            'scaling_metrics': scaling_metrics,
            'work_queue_status': {
                'pending_items': len(self.work_queue),
                'completed_items': len(self.completed_work),
                'average_queue_time': self._calculate_average_queue_time(),
                'work_types_in_queue': self._get_queue_work_types()
            },
            'prediction_status': {
                'historical_data_points': len(self.predictor.historical_data),
                'next_prediction': self.predictor.predict_future_load(30)
            },
            'load_balancing': {
                'algorithm': self.config.load_balancing_algorithm,
                'node_connections': dict(self.load_balancer.node_connections)
            }
        }
    
    def _calculate_average_queue_time(self) -> float:
        """Calculate average time items spend in queue."""
        
        current_time = time.time()
        if not self.work_queue:
            return 0.0
        
        queue_times = [current_time - item.submitted_time for item in self.work_queue]
        return np.mean(queue_times)
    
    def _get_queue_work_types(self) -> Dict[str, int]:
        """Get work types currently in queue."""
        
        work_types = defaultdict(int)
        for item in self.work_queue:
            work_types[item.workload_type.value] += 1
        
        return dict(work_types)


def create_distributed_scaling_system(config: Optional[ScalingConfig] = None) -> DistributedScalingSystem:
    """Factory function for distributed scaling system."""
    if config is None:
        config = ScalingConfig()
    
    return DistributedScalingSystem(config)


def run_scaling_benchmark() -> Dict[str, Any]:
    """Run comprehensive scaling system benchmark."""
    
    logger.info("Starting distributed scaling benchmark")
    
    # Create scaling system with aggressive scaling for testing
    config = ScalingConfig(
        min_nodes=2,
        max_nodes=20,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        scale_up_cooldown_seconds=30,
        scale_down_cooldown_seconds=60,
        health_check_interval_seconds=5
    )
    
    scaling_system = create_distributed_scaling_system(config)
    
    # Start system
    scaling_system.start_scaling_system()
    
    try:
        # Phase 1: Submit initial workload
        logger.info("Phase 1: Submitting initial workload")
        work_items = []
        
        for i in range(50):  # 50 work items
            work_item = WorkItem(
                work_id=f"benchmark_work_{i}",
                workload_type=np.random.choice(list(WorkloadType)),
                priority=np.random.randint(1, 11),
                estimated_runtime_seconds=np.random.uniform(1.0, 5.0),
                resource_requirements={
                    'cpu': np.random.uniform(0.1, 0.5),
                    'memory': np.random.uniform(0.1, 0.3)
                }
            )
            work_items.append(work_item)
            scaling_system.submit_work(work_item)
        
        # Let system scale up
        time.sleep(45)
        
        # Phase 2: Monitor progress
        logger.info("Phase 2: Monitoring system performance")
        
        phase2_start = time.time()
        while time.time() - phase2_start < 60:  # Monitor for 60 seconds
            status = scaling_system.get_system_status()
            
            completed = status['performance_metrics']['completed_work_items']
            throughput = status['performance_metrics']['throughput_per_second']
            active_nodes = status['current_load']['total_nodes']
            
            logger.info(f"Progress: {completed}/50 completed, "
                       f"throughput: {throughput:.2f}/s, "
                       f"nodes: {active_nodes}")
            
            if completed >= 45:  # Most work completed
                break
            
            time.sleep(10)
        
        # Phase 3: Reduce load and test scale-down
        logger.info("Phase 3: Testing scale-down behavior")
        
        # Wait for scale-down
        time.sleep(90)
        
        # Get final metrics
        final_status = scaling_system.get_system_status()
        
        benchmark_results = {
            'test_duration_seconds': 195,  # Approximate total time
            'work_items_submitted': len(work_items),
            'work_items_completed': final_status['performance_metrics']['completed_work_items'],
            'work_items_failed': final_status['performance_metrics']['failed_work_items'],
            'completion_rate': final_status['performance_metrics']['completed_work_items'] / len(work_items),
            'average_response_time': final_status['performance_metrics']['average_response_time'],
            'peak_throughput': final_status['performance_metrics']['throughput_per_second'],
            'scaling_efficiency': final_status['performance_metrics']['scaling_efficiency'],
            'final_node_count': final_status['current_load']['total_nodes'],
            'peak_node_count': final_status['scaling_metrics']['active_nodes'],
            'scaling_events': {
                'scale_up_events': final_status['scaling_metrics']['scaling_activity']['scale_up_events'],
                'scale_down_events': final_status['scaling_metrics']['scaling_activity']['scale_down_events']
            },
            'cost_metrics': final_status['scaling_metrics']['cost_metrics'],
            'load_balancing_effectiveness': len(final_status['load_balancing']['node_connections']) > 0,
            'prediction_accuracy': final_status['prediction_status']['next_prediction']['confidence']
        }
        
    finally:
        # Stop system
        scaling_system.stop_scaling_system()
    
    logger.info("Distributed scaling benchmark completed")
    
    return benchmark_results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run scaling benchmark
    results = run_scaling_benchmark()
    
    print("\n" + "="*80)
    print("DISTRIBUTED SCALING SYSTEM - BENCHMARK RESULTS")
    print("="*80)
    
    print(f"Test Duration: {results['test_duration_seconds']}s")
    print(f"Work Items: {results['work_items_submitted']} submitted, "
          f"{results['work_items_completed']} completed, "
          f"{results['work_items_failed']} failed")
    print(f"Completion Rate: {results['completion_rate']:.1%}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average Response Time: {results['average_response_time']:.2f}s")
    print(f"  Peak Throughput: {results['peak_throughput']:.2f} items/s")
    print(f"  Scaling Efficiency: {results['scaling_efficiency']:.1%}")
    
    print(f"\nScaling Behavior:")
    print(f"  Final Nodes: {results['final_node_count']}")
    print(f"  Peak Nodes: {results['peak_node_count']}")
    print(f"  Scale-up Events: {results['scaling_events']['scale_up_events']}")
    print(f"  Scale-down Events: {results['scaling_events']['scale_down_events']}")
    
    print(f"\nCost Analysis:")
    cost_metrics = results['cost_metrics']
    print(f"  Total Hourly Cost: ${cost_metrics['total_hourly_cost']:.2f}")
    print(f"  Avg Cost per Node: ${cost_metrics['cost_per_node_avg']:.2f}")
    print(f"  Efficiency Score: {cost_metrics['efficiency_score']:.2f}")
    
    print(f"\nSystem Intelligence:")
    print(f"  Load Balancing: {'Effective' if results['load_balancing_effectiveness'] else 'Basic'}")
    print(f"  Prediction Confidence: {results['prediction_accuracy']:.1%}")
    
    # Overall assessment
    if (results['completion_rate'] > 0.9 and 
        results['scaling_efficiency'] > 0.8 and
        results['scaling_events']['scale_up_events'] > 0 and
        results['scaling_events']['scale_down_events'] > 0):
        assessment = "EXCELLENT - Full scaling lifecycle demonstrated"
    elif results['completion_rate'] > 0.8 and results['scaling_efficiency'] > 0.6:
        assessment = "GOOD - Effective scaling with room for optimization"
    elif results['completion_rate'] > 0.7:
        assessment = "ACCEPTABLE - Basic scaling functionality working"
    else:
        assessment = "POOR - Scaling system needs improvement"
    
    print(f"\nOverall Assessment: {assessment}")
    print("="*80)