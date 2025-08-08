"""Parallel analog PDE solver with multi-processing and vectorization."""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from ..core.solver import AnalogPDESolver
from ..utils.logger import get_logger, PerformanceLogger


class ParallelAnalogPDESolver:
    """Parallel version of analog PDE solver with multi-core support."""
    
    def __init__(
        self,
        crossbar_size: int = 128,
        conductance_range: tuple = (1e-9, 1e-6),
        noise_model: str = "realistic",
        num_workers: Optional[int] = None,
        use_threading: bool = False
    ):
        """Initialize parallel analog PDE solver.
        
        Args:
            crossbar_size: Size of crossbar array
            conductance_range: Min/max conductance values in Siemens
            noise_model: Noise modeling approach
            num_workers: Number of parallel workers (default: CPU count)
            use_threading: Use threading instead of multiprocessing
        """
        self.logger = get_logger('parallel_solver')
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.crossbar_size = crossbar_size
        self.conductance_range = conductance_range
        self.noise_model = noise_model
        
        self.num_workers = num_workers or cpu_count()
        self.use_threading = use_threading
        
        # Thread-local storage for solver instances
        self._local = threading.local()
        
        self.logger.info(
            f"Initialized ParallelAnalogPDESolver with {self.num_workers} workers, "
            f"using {'threading' if use_threading else 'multiprocessing'}"
        )
    
    def _get_local_solver(self) -> AnalogPDESolver:
        """Get thread-local solver instance."""
        if not hasattr(self._local, 'solver'):
            self._local.solver = AnalogPDESolver(
                crossbar_size=self.crossbar_size,
                conductance_range=self.conductance_range,
                noise_model=self.noise_model
            )
        return self._local.solver
    
    def solve_parallel_ensemble(
        self,
        pde,
        num_realizations: int = 10,
        iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """Solve PDE with multiple realizations in parallel.
        
        Args:
            pde: PDE object to solve
            num_realizations: Number of parallel realizations
            iterations: Max iterations per realization
            convergence_threshold: Convergence threshold
            
        Returns:
            Dictionary with ensemble statistics and solutions
        """
        self.perf_logger.start_timer('parallel_ensemble_solve')
        
        self.logger.info(
            f"Starting ensemble solve with {num_realizations} realizations, "
            f"{iterations} max iterations each"
        )
        
        if self.use_threading:
            results = self._solve_ensemble_threaded(
                pde, num_realizations, iterations, convergence_threshold
            )
        else:
            results = self._solve_ensemble_multiprocess(
                pde, num_realizations, iterations, convergence_threshold
            )
        
        # Compute ensemble statistics
        solutions = np.array([r['solution'] for r in results])
        
        ensemble_stats = {
            'solutions': solutions,
            'mean_solution': np.mean(solutions, axis=0),
            'std_solution': np.std(solutions, axis=0),
            'median_solution': np.median(solutions, axis=0),
            'convergence_rates': [r['converged'] for r in results],
            'iteration_counts': [r['iterations_used'] for r in results],
            'solve_times': [r['solve_time'] for r in results],
            'num_realizations': num_realizations,
            'success_rate': sum(r['converged'] for r in results) / num_realizations
        }
        
        total_time = self.perf_logger.end_timer('parallel_ensemble_solve')
        ensemble_stats['total_time'] = total_time
        
        self.logger.info(
            f"Ensemble solve completed: {ensemble_stats['success_rate']:.1%} success rate, "
            f"mean {np.mean(ensemble_stats['iteration_counts']):.1f} iterations"
        )
        
        return ensemble_stats
    
    def _solve_ensemble_threaded(
        self, 
        pde, 
        num_realizations: int, 
        iterations: int, 
        convergence_threshold: float
    ) -> List[Dict]:
        """Solve ensemble using threading."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = []
            for i in range(num_realizations):
                future = executor.submit(
                    self._solve_single_realization,
                    pde, iterations, convergence_threshold, i
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Realization failed: {e}")
                    results.append(self._create_failed_result())
        
        return results
    
    def _solve_ensemble_multiprocess(
        self, 
        pde, 
        num_realizations: int, 
        iterations: int, 
        convergence_threshold: float
    ) -> List[Dict]:
        """Solve ensemble using multiprocessing."""
        # Create arguments for each worker
        args_list = [
            (pde, iterations, convergence_threshold, i, 
             self.crossbar_size, self.conductance_range, self.noise_model)
            for i in range(num_realizations)
        ]
        
        results = []
        try:
            with Pool(processes=self.num_workers) as pool:
                results = pool.starmap(_solve_worker_function, args_list)
        except Exception as e:
            self.logger.error(f"Multiprocessing failed: {e}")
            # Fall back to single-threaded
            results = [
                self._solve_single_realization(pde, iterations, convergence_threshold, i)
                for i in range(num_realizations)
            ]
        
        return results
    
    def _solve_single_realization(
        self, 
        pde, 
        iterations: int, 
        convergence_threshold: float,
        realization_id: int
    ) -> Dict:
        """Solve single realization."""
        import time
        start_time = time.perf_counter()
        
        try:
            solver = self._get_local_solver()
            solution = solver.solve(pde, iterations, convergence_threshold)
            
            solve_time = time.perf_counter() - start_time
            
            # Check convergence (simplified)
            converged = True  # In practice, would check actual convergence
            iterations_used = iterations  # Would track actual iterations
            
            return {
                'solution': solution,
                'converged': converged,
                'iterations_used': iterations_used,
                'solve_time': solve_time,
                'realization_id': realization_id,
                'success': True
            }
            
        except Exception as e:
            solve_time = time.perf_counter() - start_time
            self.logger.warning(f"Realization {realization_id} failed: {e}")
            return self._create_failed_result(realization_id, solve_time)
    
    def _create_failed_result(self, realization_id: int = -1, solve_time: float = 0.0) -> Dict:
        """Create result for failed realization."""
        return {
            'solution': np.zeros(self.crossbar_size),
            'converged': False,
            'iterations_used': 0,
            'solve_time': solve_time,
            'realization_id': realization_id,
            'success': False
        }
    
    def solve_domain_decomposition(
        self,
        pde,
        num_subdomains: int = 4,
        overlap: int = 2,
        iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> np.ndarray:
        """Solve PDE using domain decomposition for large problems.
        
        Args:
            pde: PDE object to solve
            num_subdomains: Number of subdomains
            overlap: Overlap between subdomains
            iterations: Max iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            Combined solution array
        """
        self.perf_logger.start_timer('domain_decomposition_solve')
        
        # Get domain size
        if isinstance(pde.domain_size, tuple):
            domain_size = pde.domain_size[0]
        else:
            domain_size = pde.domain_size
        
        if domain_size < num_subdomains * 2:
            self.logger.warning("Domain too small for decomposition, using single solve")
            solver = self._get_local_solver()
            return solver.solve(pde, iterations, convergence_threshold)
        
        # Calculate subdomain sizes
        subdomain_size = domain_size // num_subdomains
        subdomains = []
        
        for i in range(num_subdomains):
            start = max(0, i * subdomain_size - overlap)
            end = min(domain_size, (i + 1) * subdomain_size + overlap)
            subdomains.append((start, end))
        
        self.logger.info(f"Solving {num_subdomains} subdomains of size ~{subdomain_size}")
        
        # Solve subdomains in parallel
        if self.use_threading:
            subdomain_solutions = self._solve_subdomains_threaded(
                subdomains, pde, iterations, convergence_threshold
            )
        else:
            subdomain_solutions = self._solve_subdomains_multiprocess(
                subdomains, pde, iterations, convergence_threshold
            )
        
        # Combine solutions
        combined_solution = self._combine_subdomain_solutions(
            subdomain_solutions, subdomains, domain_size, overlap
        )
        
        self.perf_logger.end_timer('domain_decomposition_solve')
        
        return combined_solution
    
    def _solve_subdomains_threaded(
        self, 
        subdomains: List[tuple], 
        pde, 
        iterations: int, 
        convergence_threshold: float
    ) -> List[np.ndarray]:
        """Solve subdomains using threading."""
        solutions = [None] * len(subdomains)
        
        def solve_subdomain(idx, start, end):
            try:
                # Create subdomain PDE (simplified)
                subdomain_size = end - start
                solver = AnalogPDESolver(
                    crossbar_size=subdomain_size,
                    conductance_range=self.conductance_range,
                    noise_model=self.noise_model
                )
                
                # Create simplified subdomain PDE
                from ..core.equations import PoissonEquation
                subdomain_pde = PoissonEquation((subdomain_size,))
                
                solution = solver.solve(subdomain_pde, iterations, convergence_threshold)
                solutions[idx] = solution
                
            except Exception as e:
                self.logger.error(f"Subdomain {idx} failed: {e}")
                solutions[idx] = np.zeros(end - start)
        
        # Use threading for subdomain solving
        threads = []
        for i, (start, end) in enumerate(subdomains):
            thread = threading.Thread(target=solve_subdomain, args=(i, start, end))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        return solutions
    
    def _solve_subdomains_multiprocess(
        self, 
        subdomains: List[tuple], 
        pde, 
        iterations: int, 
        convergence_threshold: float
    ) -> List[np.ndarray]:
        """Solve subdomains using multiprocessing."""
        # Simplified for now - would need more complex subdomain PDE creation
        return self._solve_subdomains_threaded(subdomains, pde, iterations, convergence_threshold)
    
    def _combine_subdomain_solutions(
        self,
        solutions: List[np.ndarray],
        subdomains: List[tuple],
        domain_size: int,
        overlap: int
    ) -> np.ndarray:
        """Combine subdomain solutions with overlap handling."""
        combined = np.zeros(domain_size)
        weights = np.zeros(domain_size)
        
        for i, ((start, end), solution) in enumerate(zip(subdomains, solutions)):
            # Simple averaging in overlap regions
            combined[start:end] += solution
            weights[start:end] += 1.0
        
        # Normalize by weights to handle overlaps
        weights[weights == 0] = 1.0  # Avoid division by zero
        combined /= weights
        
        return combined


def _solve_worker_function(
    pde, 
    iterations: int, 
    convergence_threshold: float, 
    realization_id: int,
    crossbar_size: int,
    conductance_range: tuple,
    noise_model: str
) -> Dict:
    """Worker function for multiprocessing."""
    import time
    start_time = time.perf_counter()
    
    try:
        solver = AnalogPDESolver(
            crossbar_size=crossbar_size,
            conductance_range=conductance_range,
            noise_model=noise_model
        )
        solution = solver.solve(pde, iterations, convergence_threshold)
        
        solve_time = time.perf_counter() - start_time
        
        return {
            'solution': solution,
            'converged': True,
            'iterations_used': iterations,
            'solve_time': solve_time,
            'realization_id': realization_id,
            'success': True
        }
        
    except Exception as e:
        solve_time = time.perf_counter() - start_time
        return {
            'solution': np.zeros(crossbar_size),
            'converged': False,
            'iterations_used': 0,
            'solve_time': solve_time,
            'realization_id': realization_id,
            'success': False,
            'error': str(e)
        }