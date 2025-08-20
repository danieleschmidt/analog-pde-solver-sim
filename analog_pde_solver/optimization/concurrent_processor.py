"""High-performance concurrent processing for analog PDE solving."""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
import logging
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


@dataclass
class ProcessingTask:
    """Task for concurrent processing."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    submitted_time: float = 0.0
    
    def __post_init__(self):
        if self.submitted_time == 0.0:
            self.submitted_time = time.time()


@dataclass
class ProcessingResult:
    """Result from concurrent processing."""
    task_id: str
    success: bool
    result: Any
    error: Optional[str]
    processing_time: float
    worker_id: str


class ResourcePool:
    """Intelligent resource pool management."""
    
    def __init__(self, pool_type: str = "mixed"):
        self.pool_type = pool_type
        self.logger = logging.getLogger(__name__)
        
        # Resource pools
        self.thread_pool = None
        self.process_pool = None
        self.async_pool = None
        
        # Resource utilization tracking
        self.resource_stats = {
            'thread_utilization': 0.0,
            'process_utilization': 0.0,
            'memory_utilization': 0.0,
            'active_tasks': 0
        }
        
        # Dynamic sizing
        self.min_threads = 2
        self.max_threads = mp.cpu_count() * 4
        self.min_processes = 1
        self.max_processes = mp.cpu_count()
        
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize resource pools."""
        try:
            if self.pool_type in ["mixed", "threads"]:
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.max_threads,
                    thread_name_prefix="analog_pde"
                )
            
            if self.pool_type in ["mixed", "processes"]:
                self.process_pool = ProcessPoolExecutor(
                    max_workers=self.max_processes
                )
            
            self.logger.info(f"Resource pools initialized: {self.pool_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize resource pools: {e}")
            raise
    
    def submit_task(
        self, 
        func: Callable, 
        *args, 
        execution_type: str = "auto",
        **kwargs
    ) -> concurrent.futures.Future:
        """Submit task for concurrent execution.
        
        Args:
            func: Function to execute
            args: Positional arguments
            execution_type: "threads", "processes", or "auto"
            kwargs: Keyword arguments
            
        Returns:
            Future object for result retrieval
        """
        try:
            # Determine optimal execution type
            if execution_type == "auto":
                execution_type = self._choose_execution_type(func, args, kwargs)
            
            # Submit to appropriate pool
            if execution_type == "threads" and self.thread_pool:
                future = self.thread_pool.submit(func, *args, **kwargs)
            elif execution_type == "processes" and self.process_pool:
                future = self.process_pool.submit(func, *args, **kwargs)
            else:
                # Fallback to synchronous execution
                result = func(*args, **kwargs)
                future = concurrent.futures.Future()
                future.set_result(result)
            
            self.resource_stats['active_tasks'] += 1
            
            # Add completion callback
            future.add_done_callback(self._task_completed)
            
            return future
            
        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            raise
    
    def _choose_execution_type(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Intelligently choose execution type based on task characteristics."""
        
        # Analyze task characteristics
        is_cpu_intensive = self._is_cpu_intensive(func, args, kwargs)
        is_io_bound = self._is_io_bound(func, args, kwargs)
        has_large_data = self._has_large_data(args, kwargs)
        
        # Decision logic
        if is_cpu_intensive and not has_large_data:
            return "processes"
        elif is_io_bound or has_large_data:
            return "threads"
        else:
            # Default for mixed workloads
            current_thread_util = self.resource_stats['thread_utilization']
            current_process_util = self.resource_stats['process_utilization']
            
            if current_thread_util < current_process_util:
                return "threads"
            else:
                return "processes"
    
    def _is_cpu_intensive(self, func: Callable, args: tuple, kwargs: dict) -> bool:
        """Determine if task is CPU-intensive."""
        # Heuristics for CPU-intensive tasks
        cpu_intensive_indicators = [
            'solve', 'compute', 'optimize', 'iterate',
            'matrix', 'linear_algebra', 'numerical'
        ]
        
        func_name = getattr(func, '__name__', '').lower()
        return any(indicator in func_name for indicator in cpu_intensive_indicators)
    
    def _is_io_bound(self, func: Callable, args: tuple, kwargs: dict) -> bool:
        """Determine if task is I/O bound."""
        io_indicators = [
            'read', 'write', 'load', 'save', 'export', 'import',
            'fetch', 'download', 'upload'
        ]
        
        func_name = getattr(func, '__name__', '').lower()
        return any(indicator in func_name for indicator in io_indicators)
    
    def _has_large_data(self, args: tuple, kwargs: dict) -> bool:
        """Check if task involves large data structures."""
        size_threshold = 10 * 1024 * 1024  # 10MB
        
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.nbytes > size_threshold:
                return True
        
        for value in kwargs.values():
            if isinstance(value, np.ndarray) and value.nbytes > size_threshold:
                return True
        
        return False
    
    def _task_completed(self, future: concurrent.futures.Future):
        """Callback for task completion."""
        self.resource_stats['active_tasks'] -= 1
        
        # Update utilization metrics
        self._update_utilization_metrics()
    
    def _update_utilization_metrics(self):
        """Update resource utilization metrics."""
        try:
            if self.thread_pool:
                # Estimate thread utilization
                active_threads = self.thread_pool._threads.__len__()
                self.resource_stats['thread_utilization'] = (
                    active_threads / self.max_threads * 100
                )
            
            # Memory utilization
            import psutil
            memory = psutil.virtual_memory()
            self.resource_stats['memory_utilization'] = memory.percent
            
        except Exception as e:
            self.logger.debug(f"Failed to update utilization metrics: {e}")
    
    def shutdown(self, wait: bool = True):
        """Shutdown resource pools."""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=wait)
            
            if self.process_pool:
                self.process_pool.shutdown(wait=wait)
            
            self.logger.info("Resource pools shutdown")
            
        except Exception as e:
            self.logger.error(f"Error during pool shutdown: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        return {
            'pool_type': self.pool_type,
            'max_threads': self.max_threads,
            'max_processes': self.max_processes,
            'resource_stats': self.resource_stats
        }


class ConcurrentPDESolver:
    """High-performance concurrent PDE solver."""
    
    def __init__(self, max_concurrent_solves: int = None):
        self.max_concurrent_solves = max_concurrent_solves or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
        
        # Resource management
        self.resource_pool = ResourcePool("mixed")
        
        # Task management
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        
        # Performance optimization
        self.solver_cache = {}
        self.optimization_enabled = True
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.max_concurrent_solves)
        
    async def solve_concurrent(
        self,
        pde_problems: List[Dict[str, Any]],
        solver_config: Dict[str, Any]
    ) -> List[ProcessingResult]:
        """Solve multiple PDE problems concurrently.
        
        Args:
            pde_problems: List of PDE problem configurations
            solver_config: Solver configuration parameters
            
        Returns:
            List of processing results
        """
        try:
            # Create tasks
            tasks = []
            for i, problem in enumerate(pde_problems):
                task = ProcessingTask(
                    task_id=f"pde_solve_{i}",
                    task_type="pde_solve",
                    parameters={**problem, **solver_config},
                    priority=problem.get('priority', 1)
                )
                tasks.append(task)
            
            # Sort by priority
            tasks.sort(key=lambda t: t.priority, reverse=True)
            
            # Execute concurrently
            futures = []
            for task in tasks:
                future = asyncio.create_task(self._solve_single_async(task))
                futures.append(future)
            
            # Wait for completion
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ProcessingResult(
                        task_id=tasks[i].task_id,
                        success=False,
                        result=None,
                        error=str(result),
                        processing_time=0.0,
                        worker_id="unknown"
                    ))
                else:
                    processed_results.append(result)
            
            self.logger.info(f"Completed {len(processed_results)} concurrent solves")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Concurrent solving failed: {e}")
            raise
    
    async def _solve_single_async(self, task: ProcessingTask) -> ProcessingResult:
        """Solve single PDE problem asynchronously."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Check cache first
                cache_key = self._generate_cache_key(task.parameters)
                if cache_key in self.solver_cache:
                    cached_result = self.solver_cache[cache_key]
                    self.logger.debug(f"Using cached result for task {task.task_id}")
                    
                    return ProcessingResult(
                        task_id=task.task_id,
                        success=True,
                        result=cached_result,
                        error=None,
                        processing_time=time.time() - start_time,
                        worker_id="cache"
                    )
                
                # Execute solve
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default executor
                    self._solve_pde_task,
                    task
                )
                
                # Cache result if optimization enabled
                if self.optimization_enabled:
                    self.solver_cache[cache_key] = result
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    error=None,
                    processing_time=processing_time,
                    worker_id=threading.current_thread().name
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task.task_id,
                    success=False,
                    result=None,
                    error=str(e),
                    processing_time=processing_time,
                    worker_id=threading.current_thread().name
                )
    
    def _solve_pde_task(self, task: ProcessingTask) -> Any:
        """Solve PDE task (synchronous execution)."""
        from ..core.solver_robust import RobustAnalogPDESolver
        from ..core.equations import PoissonEquation
        
        # Create solver
        solver = RobustAnalogPDESolver(
            crossbar_size=task.parameters.get('crossbar_size', 64),
            conductance_range=task.parameters.get('conductance_range', (1e-9, 1e-6)),
            noise_model=task.parameters.get('noise_model', 'realistic')
        )
        
        # Create PDE
        pde = PoissonEquation(
            domain_size=task.parameters.get('domain_size', (64,)),
            boundary_conditions=task.parameters.get('boundary_conditions', 'dirichlet')
        )
        
        # Solve
        solution = solver.solve(
            pde,
            iterations=task.parameters.get('iterations', 100),
            convergence_threshold=task.parameters.get('convergence_threshold', 1e-6)
        )
        
        return {
            'solution': solution,
            'convergence_info': solver.get_convergence_info(),
            'health_check': solver.health_check()
        }
    
    def _generate_cache_key(self, parameters: Dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        # Create deterministic hash of parameters
        param_str = str(sorted(parameters.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def solve_batch_parallel(
        self,
        problems: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> List[ProcessingResult]:
        """Solve batch of problems in parallel (synchronous)."""
        max_workers = max_workers or min(len(problems), mp.cpu_count())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for i, problem in enumerate(problems):
                task = ProcessingTask(
                    task_id=f"batch_{i}",
                    task_type="pde_solve",
                    parameters=problem
                )
                future = executor.submit(self._solve_pde_task, task)
                futures[future] = task
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                start_time = task.submitted_time
                
                try:
                    result = future.result()
                    processing_result = ProcessingResult(
                        task_id=task.task_id,
                        success=True,
                        result=result,
                        error=None,
                        processing_time=time.time() - start_time,
                        worker_id=str(threading.get_ident())
                    )
                except Exception as e:
                    processing_result = ProcessingResult(
                        task_id=task.task_id,
                        success=False,
                        result=None,
                        error=str(e),
                        processing_time=time.time() - start_time,
                        worker_id=str(threading.get_ident())
                    )
                
                results.append(processing_result)
            
            self.logger.info(f"Completed batch of {len(problems)} problems")
            return results
    
    def pipeline_processing(
        self,
        data_stream: List[Any],
        processing_stages: List[Callable],
        stage_configs: List[Dict[str, Any]]
    ) -> List[Any]:
        """Process data through pipeline of concurrent stages."""
        results = data_stream.copy()
        
        for stage_idx, (stage_func, config) in enumerate(zip(processing_stages, stage_configs)):
            stage_name = f"stage_{stage_idx}"
            max_workers = config.get('max_workers', mp.cpu_count())
            
            self.logger.debug(f"Processing pipeline {stage_name} with {max_workers} workers")
            
            # Process stage concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                stage_futures = {
                    executor.submit(stage_func, item, **config): i 
                    for i, item in enumerate(results)
                }
                
                stage_results = [None] * len(results)
                for future in concurrent.futures.as_completed(stage_futures):
                    idx = stage_futures[future]
                    try:
                        stage_results[idx] = future.result()
                    except Exception as e:
                        self.logger.error(f"Pipeline stage {stage_name} failed for item {idx}: {e}")
                        stage_results[idx] = None
                
                results = stage_results
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get concurrent processing performance metrics."""
        return {
            'resource_stats': self.resource_stats,
            'cache_size': len(self.solver_cache),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'pool_type': self.pool_type
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.resource_pool:
            self.resource_pool.shutdown()
        
        self.solver_cache.clear()
        self.active_tasks.clear()
        self.completed_tasks.clear()
        
        self.logger.info("Concurrent processor cleaned up")


class DistributedSolverCoordinator:
    """Coordinate distributed analog PDE solving across multiple nodes."""
    
    def __init__(self, node_configs: List[Dict[str, Any]]):
        self.node_configs = node_configs
        self.logger = logging.getLogger(__name__)
        
        # Node management
        self.active_nodes = {}
        self.node_health = {}
        
        # Load balancing
        self.load_balancer = self._create_load_balancer()
        
        # Fault tolerance
        self.replication_factor = 2
        self.failure_recovery = True
        
    def _create_load_balancer(self):
        """Create load balancer for distributed nodes."""
        # Simplified load balancer
        return {
            'strategy': 'least_loaded',
            'health_check_interval': 30,
            'failure_threshold': 3
        }
    
    def submit_distributed_solve(
        self,
        pde_config: Dict[str, Any],
        distribution_strategy: str = "domain_decomposition"
    ) -> Dict[str, Any]:
        """Submit PDE solve for distributed processing."""
        
        try:
            # Decompose problem based on strategy
            if distribution_strategy == "domain_decomposition":
                sub_problems = self._decompose_domain(pde_config)
            elif distribution_strategy == "frequency_decomposition":
                sub_problems = self._decompose_frequency(pde_config)
            else:
                sub_problems = [pde_config]  # Single node processing
            
            # Distribute to nodes
            node_assignments = self._assign_to_nodes(sub_problems)
            
            # Execute on nodes (simulated)
            results = self._execute_on_nodes(node_assignments)
            
            # Combine results
            final_result = self._combine_distributed_results(results, distribution_strategy)
            
            return {
                'success': True,
                'result': final_result,
                'nodes_used': len(node_assignments),
                'execution_time': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Distributed solve failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'nodes_used': 0
            }
    
    def _decompose_domain(self, pde_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose PDE domain for parallel processing."""
        domain_size = pde_config.get('domain_size', (128,))
        
        if len(domain_size) == 1:
            # 1D decomposition
            size = domain_size[0]
            num_parts = min(len(self.node_configs), size // 16)  # At least 16 points per partition
            
            part_size = size // num_parts
            sub_problems = []
            
            for i in range(num_parts):
                start = i * part_size
                end = start + part_size if i < num_parts - 1 else size
                
                sub_config = pde_config.copy()
                sub_config['domain_slice'] = (start, end)
                sub_config['domain_size'] = (end - start,)
                sub_problems.append(sub_config)
            
            return sub_problems
        
        # For higher dimensions, return single problem for now
        return [pde_config]
    
    def _decompose_frequency(self, pde_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose PDE in frequency domain."""
        # Frequency domain decomposition for spectral methods
        # For now, return single problem
        return [pde_config]
    
    def _assign_to_nodes(self, sub_problems: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Assign sub-problems to available nodes."""
        assignments = {}
        
        # Simple round-robin assignment
        for i, problem in enumerate(sub_problems):
            node_idx = i % len(self.node_configs)
            node_id = f"node_{node_idx}"
            
            if node_id not in assignments:
                assignments[node_id] = []
            
            assignments[node_id].append(problem)
        
        return assignments
    
    def _execute_on_nodes(self, assignments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Any]]:
        """Execute assignments on distributed nodes."""
        # For simulation, use local concurrent processing
        processor = ConcurrentPDESolver()
        
        results = {}
        for node_id, problems in assignments.items():
            node_results = processor.solve_batch_parallel(problems)
            results[node_id] = [r.result for r in node_results if r.success]
        
        return results
    
    def _combine_distributed_results(
        self,
        node_results: Dict[str, List[Any]],
        strategy: str
    ) -> Any:
        """Combine results from distributed nodes."""
        
        if strategy == "domain_decomposition":
            # Combine domain slices
            all_solutions = []
            for node_id in sorted(node_results.keys()):
                for result in node_results[node_id]:
                    if result and 'solution' in result:
                        all_solutions.append(result['solution'])
            
            if all_solutions:
                # Concatenate solutions
                combined_solution = np.concatenate(all_solutions)
                return {
                    'solution': combined_solution,
                    'distributed': True,
                    'nodes_used': len(node_results)
                }
        
        # Default: return first successful result
        for node_results_list in node_results.values():
            if node_results_list:
                return node_results_list[0]
        
        return None


# Global instances
_concurrent_processor = None
_distributed_coordinator = None


def get_concurrent_processor() -> ConcurrentPDESolver:
    """Get global concurrent processor."""
    global _concurrent_processor
    if _concurrent_processor is None:
        _concurrent_processor = ConcurrentPDESolver()
    return _concurrent_processor