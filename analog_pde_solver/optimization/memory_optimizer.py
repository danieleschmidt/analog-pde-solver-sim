"""Memory optimization and caching for analog PDE solver."""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
import threading
import weakref
import gc
import sys
from functools import lru_cache
from ..utils.logger import get_logger


class MemoryPool:
    """Memory pool for reusing NumPy arrays."""
    
    def __init__(self, max_size_mb: int = 512):
        """Initialize memory pool.
        
        Args:
            max_size_mb: Maximum pool size in MB
        """
        self.logger = get_logger('memory_pool')
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self._pools = {}  # shape -> list of arrays
        self._lock = threading.Lock()
        
        self.logger.info(f"Initialized memory pool with {max_size_mb} MB limit")
    
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Get array from pool or allocate new one.
        
        Args:
            shape: Array shape
            dtype: Array data type
            
        Returns:
            NumPy array
        """
        key = (shape, dtype)
        
        with self._lock:
            if key in self._pools and self._pools[key]:
                array = self._pools[key].pop()
                array.fill(0)  # Clear previous data
                return array
        
        # Allocate new array
        return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool.
        
        Args:
            array: Array to return
        """
        if array is None:
            return
            
        key = (array.shape, array.dtype)
        array_size = array.nbytes
        
        with self._lock:
            # Check if we have space
            if self.current_size + array_size <= self.max_size_bytes:
                if key not in self._pools:
                    self._pools[key] = []
                
                self._pools[key].append(array)
                self.current_size += array_size
            else:
                # Pool is full, let array be garbage collected
                pass
    
    def clear(self) -> None:
        """Clear the memory pool."""
        with self._lock:
            self._pools.clear()
            self.current_size = 0
            gc.collect()
        
        self.logger.info("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_arrays = sum(len(arrays) for arrays in self._pools.values())
            return {
                'current_size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'pool_count': len(self._pools),
                'total_arrays': total_arrays,
                'utilization': self.current_size / self.max_size_bytes
            }


class ConductanceCache:
    """Cache for conductance matrices to avoid recomputation."""
    
    def __init__(self, max_entries: int = 100):
        """Initialize conductance cache.
        
        Args:
            max_entries: Maximum cache entries
        """
        self.logger = get_logger('conductance_cache')
        self.max_entries = max_entries
        self._cache = {}
        self._access_order = []
        self._lock = threading.Lock()
        
        self.logger.info(f"Initialized conductance cache with {max_entries} max entries")
    
    def _make_key(
        self, 
        matrix_size: int, 
        conductance_range: Tuple[float, float],
        pde_type: str
    ) -> str:
        """Create cache key."""
        return f"{matrix_size}_{conductance_range[0]}_{conductance_range[1]}_{pde_type}"
    
    def get(
        self, 
        matrix_size: int, 
        conductance_range: Tuple[float, float],
        pde_type: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached conductance matrices.
        
        Args:
            matrix_size: Size of the matrix
            conductance_range: Min/max conductance values
            pde_type: Type of PDE
            
        Returns:
            Tuple of (g_positive, g_negative) or None if not cached
        """
        key = self._make_key(matrix_size, conductance_range, pde_type)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                
                g_pos, g_neg = self._cache[key]
                return g_pos.copy(), g_neg.copy()
        
        return None
    
    def put(
        self, 
        matrix_size: int, 
        conductance_range: Tuple[float, float],
        pde_type: str,
        g_positive: np.ndarray,
        g_negative: np.ndarray
    ) -> None:
        """Store conductance matrices in cache.
        
        Args:
            matrix_size: Size of the matrix
            conductance_range: Min/max conductance values
            pde_type: Type of PDE
            g_positive: Positive conductance matrix
            g_negative: Negative conductance matrix
        """
        key = self._make_key(matrix_size, conductance_range, pde_type)
        
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_entries and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            # Store copies to avoid external modifications
            self._cache[key] = (g_positive.copy(), g_negative.copy())
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
        
        self.logger.info("Conductance cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cache_size_mb = sum(
                (g_pos.nbytes + g_neg.nbytes) 
                for g_pos, g_neg in self._cache.values()
            ) / (1024 * 1024)
            
            return {
                'entries': len(self._cache),
                'max_entries': self.max_entries,
                'cache_size_mb': cache_size_mb,
                'hit_rate': getattr(self, '_hit_rate', 0.0),
                'utilization': len(self._cache) / self.max_entries
            }


class MemoryOptimizedSolver:
    """Memory-optimized analog PDE solver."""
    
    def __init__(
        self,
        crossbar_size: int = 128,
        conductance_range: tuple = (1e-9, 1e-6),
        noise_model: str = "realistic",
        memory_pool_size_mb: int = 256,
        enable_caching: bool = True
    ):
        """Initialize memory-optimized solver.
        
        Args:
            crossbar_size: Size of crossbar array
            conductance_range: Min/max conductance values
            noise_model: Noise modeling approach
            memory_pool_size_mb: Memory pool size in MB
            enable_caching: Enable conductance caching
        """
        self.logger = get_logger('memory_optimized_solver')
        
        self.crossbar_size = crossbar_size
        self.conductance_range = conductance_range
        self.noise_model = noise_model
        
        # Memory management
        self.memory_pool = MemoryPool(memory_pool_size_mb)
        self.conductance_cache = ConductanceCache() if enable_caching else None
        
        # Statistics
        self._solve_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info(
            f"Initialized MemoryOptimizedSolver with {memory_pool_size_mb}MB pool, "
            f"caching {'enabled' if enable_caching else 'disabled'}"
        )
    
    def solve(
        self,
        pde,
        iterations: int = 100,
        convergence_threshold: float = 1e-6,
        in_place: bool = True
    ) -> np.ndarray:
        """Solve PDE with memory optimization.
        
        Args:
            pde: PDE object to solve
            iterations: Maximum iterations
            convergence_threshold: Convergence threshold
            in_place: Use in-place operations where possible
            
        Returns:
            Solution array
        """
        self._solve_count += 1
        pde_type = type(pde).__name__
        
        # Try to get conductances from cache
        cached_conductances = None
        if self.conductance_cache:
            cached_conductances = self.conductance_cache.get(
                self.crossbar_size, self.conductance_range, pde_type
            )
            
            if cached_conductances:
                self._cache_hits += 1
                self.logger.debug("Using cached conductance matrices")
            else:
                self._cache_misses += 1
        
        # Get working arrays from memory pool
        phi = self.memory_pool.get_array((self.crossbar_size,))
        phi_new = self.memory_pool.get_array((self.crossbar_size,))
        source = self.memory_pool.get_array((self.crossbar_size,))
        residual = self.memory_pool.get_array((self.crossbar_size,))
        
        try:
            # Initialize arrays
            np.random.seed(42)  # For reproducible results
            phi[:] = np.random.random(self.crossbar_size) * 0.1
            
            # Create source term
            self._create_source_term(pde, source)
            
            # Get or compute conductances
            if cached_conductances:
                g_positive, g_negative = cached_conductances
            else:
                g_positive, g_negative = self._compute_conductances(pde)
                
                # Cache for future use
                if self.conductance_cache:
                    self.conductance_cache.put(
                        self.crossbar_size, self.conductance_range, pde_type,
                        g_positive, g_negative
                    )
            
            # Iterative solver with memory optimization
            for i in range(iterations):
                # Compute residual using cached matrices
                self._compute_residual_optimized(
                    phi, source, residual, g_positive, g_negative
                )
                
                # Update solution (in-place if requested)
                if in_place:
                    phi -= 0.1 * residual
                    phi_new, phi = phi, phi_new  # Swap references
                else:
                    np.subtract(phi, 0.1 * residual, out=phi_new)
                    phi, phi_new = phi_new, phi  # Swap for next iteration
                
                # Apply boundary conditions
                phi[0] = 0.0
                phi[-1] = 0.0
                
                # Check convergence
                error = np.linalg.norm(residual)
                if error < convergence_threshold:
                    self.logger.debug(f"Converged after {i+1} iterations")
                    break
            
            # Create result array (not from pool, as it's returned)
            result = phi.copy()
            
        finally:
            # Return arrays to pool
            self.memory_pool.return_array(phi)
            self.memory_pool.return_array(phi_new)
            self.memory_pool.return_array(source)
            self.memory_pool.return_array(residual)
        
        return result
    
    def _create_source_term(self, pde, source: np.ndarray) -> None:
        """Create source term efficiently."""
        if hasattr(pde, 'source_function') and pde.source_function:
            x = np.linspace(0, 1, self.crossbar_size)
            for i, xi in enumerate(x):
                source[i] = pde.source_function(xi, 0)
        else:
            source.fill(0.1)
    
    def _compute_conductances(self, pde) -> Tuple[np.ndarray, np.ndarray]:
        """Compute conductance matrices."""
        # Create Laplacian matrix
        laplacian = self._create_laplacian_matrix_optimized()
        
        # Decompose into positive and negative components
        g_min, g_max = self.conductance_range
        
        pos_matrix = np.maximum(laplacian, 0)
        neg_matrix = np.maximum(-laplacian, 0)
        
        # Scale to conductance range
        g_positive = self._scale_to_conductance(pos_matrix, g_min, g_max)
        g_negative = self._scale_to_conductance(neg_matrix, g_min, g_max)
        
        return g_positive, g_negative
    
    def _create_laplacian_matrix_optimized(self) -> np.ndarray:
        """Create Laplacian matrix with memory optimization."""
        size = self.crossbar_size
        
        # Use sparse-like approach for memory efficiency
        laplacian = np.zeros((size, size), dtype=np.float32)  # Use float32 to save memory
        
        # Main diagonal
        np.fill_diagonal(laplacian, -2.0)
        
        # Off-diagonals (vectorized)
        if size > 1:
            laplacian[np.arange(size-1), np.arange(1, size)] = 1.0
            laplacian[np.arange(1, size), np.arange(size-1)] = 1.0
        
        return laplacian.astype(np.float64)  # Convert back if needed
    
    def _scale_to_conductance(
        self, 
        matrix: np.ndarray, 
        g_min: float, 
        g_max: float
    ) -> np.ndarray:
        """Scale matrix to conductance range efficiently."""
        if matrix.max() == 0:
            return np.full_like(matrix, g_min)
        
        # Avoid unnecessary copies
        scaled = matrix / matrix.max()  # Normalize
        scaled *= (g_max - g_min)       # Scale to range
        scaled += g_min                 # Shift to minimum
        
        return scaled
    
    def _compute_residual_optimized(
        self,
        phi: np.ndarray,
        source: np.ndarray,
        residual: np.ndarray,
        g_positive: np.ndarray,
        g_negative: np.ndarray
    ) -> None:
        """Compute residual with optimized memory usage."""
        # Analog vector-matrix multiplication (simplified)
        # In practice, this would use the crossbar compute_vmm method
        
        # Positive contribution
        np.dot(g_positive.T, phi, out=residual)
        
        # Subtract negative contribution (in-place)
        temp = np.dot(g_negative.T, phi)
        residual -= temp
        
        # Add source term
        residual += source
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        pool_stats = self.memory_pool.get_stats()
        
        stats = {
            'memory_pool': pool_stats,
            'solve_count': self._solve_count,
            'system_memory': self._get_system_memory_info()
        }
        
        if self.conductance_cache:
            cache_stats = self.conductance_cache.get_stats()
            cache_stats['hit_rate'] = (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            )
            stats['conductance_cache'] = cache_stats
        
        return stats
    
    def _get_system_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_percent': mem.percent,
                'python_process_mb': psutil.Process().memory_info().rss / (1024**2)
            }
        except ImportError:
            return {'available': False, 'reason': 'psutil not installed'}
    
    def cleanup(self) -> None:
        """Clean up memory resources."""
        self.memory_pool.clear()
        if self.conductance_cache:
            self.conductance_cache.clear()
        gc.collect()
        
        self.logger.info("Memory cleanup completed")