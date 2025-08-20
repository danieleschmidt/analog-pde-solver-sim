"""Intelligent caching system with adaptive patterns."""

import time
import hashlib
import threading
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import logging


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    access_count: int
    last_access: float
    computation_time: float
    size_bytes: int
    
    def __post_init__(self):
        self.score = self._compute_score()
    
    def _compute_score(self) -> float:
        """Compute cache entry importance score."""
        now = time.time()
        age_hours = (now - self.timestamp) / 3600
        access_frequency = self.access_count / max(1, age_hours)
        
        # Higher score = more important to keep
        score = (
            access_frequency * 100 +           # Frequent access bonus
            max(0, 10 - age_hours) * 5 +       # Recency bonus
            min(self.computation_time, 10) * 2  # Computation cost bonus
        )
        
        # Penalty for large entries
        size_mb = self.size_bytes / (1024 * 1024)
        if size_mb > 10:
            score *= 0.5
        
        return score


class IntelligentCache:
    """Adaptive caching system that learns from access patterns."""
    
    def __init__(self, max_size_mb: int = 512, max_entries: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.logger = logging.getLogger(__name__)
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # Access pattern learning
        self.access_patterns = defaultdict(list)
        self.prediction_accuracy = defaultdict(float)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance metrics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'predictions_correct': 0,
            'predictions_total': 0
        }
        
        self.logger.info(f"Intelligent cache initialized: {max_size_mb}MB, {max_entries} entries")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with access pattern learning."""
        with self.lock:
            cache_key = self._hash_key(key)
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.access_count += 1
                entry.last_access = time.time()
                entry.score = entry._compute_score()
                
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                
                # Learn access pattern
                self._learn_access_pattern(key)
                
                self.stats['hits'] += 1
                self.logger.debug(f"Cache hit: {key}")
                return entry.data
            else:
                self.stats['misses'] += 1
                self.logger.debug(f"Cache miss: {key}")
                return default
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> bool:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            cache_key = self._hash_key(key)
            
            try:
                # Estimate size
                size_bytes = self._estimate_size(value)
                
                # Check if it fits
                if size_bytes > self.max_size_bytes:
                    self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                    return False
                
                # Evict if necessary
                self._make_space(size_bytes)
                
                # Create cache entry
                entry = CacheEntry(
                    data=value,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time(),
                    computation_time=computation_time,
                    size_bytes=size_bytes
                )
                
                # Update cache
                if cache_key in self.cache:
                    self.current_size_bytes -= self.cache[cache_key].size_bytes
                
                self.cache[cache_key] = entry
                self.current_size_bytes += size_bytes
                
                self.logger.debug(f"Cached: {key} ({size_bytes} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache item: {e}")
                return False
    
    def _make_space(self, required_bytes: int):
        """Make space in cache using intelligent eviction."""
        # Check if we need to evict
        while (self.current_size_bytes + required_bytes > self.max_size_bytes or 
               len(self.cache) >= self.max_entries):
            
            if not self.cache:
                break
            
            # Find item to evict using scoring
            evict_key = self._select_eviction_candidate()
            if evict_key:
                evicted_entry = self.cache.pop(evict_key)
                self.current_size_bytes -= evicted_entry.size_bytes
                self.stats['evictions'] += 1
                self.logger.debug(f"Evicted cache entry: {evict_key}")
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select best candidate for eviction based on scoring."""
        if not self.cache:
            return None
        
        # Score all entries
        scored_entries = []
        for key, entry in self.cache.items():
            # Update score based on current time
            entry.score = entry._compute_score()
            scored_entries.append((key, entry.score))
        
        # Sort by score (lowest first = best candidate for eviction)
        scored_entries.sort(key=lambda x: x[1])
        
        return scored_entries[0][0]
    
    def _learn_access_pattern(self, key: str):
        """Learn access patterns for predictive caching."""
        now = time.time()
        
        # Record access time
        pattern_key = self._get_pattern_key(key)
        self.access_patterns[pattern_key].append(now)
        
        # Keep only recent accesses (last 24 hours)
        cutoff = now - 86400
        self.access_patterns[pattern_key] = [
            t for t in self.access_patterns[pattern_key] if t > cutoff
        ]
        
        # Predict next access
        if len(self.access_patterns[pattern_key]) >= 3:
            predicted_next = self._predict_next_access(pattern_key)
            if predicted_next:
                self._preload_related_data(key, predicted_next)
    
    def _predict_next_access(self, pattern_key: str) -> Optional[float]:
        """Predict next access time based on historical pattern."""
        accesses = self.access_patterns[pattern_key]
        if len(accesses) < 3:
            return None
        
        # Calculate average interval between accesses
        intervals = []
        for i in range(1, len(accesses)):
            intervals.append(accesses[i] - accesses[i-1])
        
        if not intervals:
            return None
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Predict next access time
        last_access = accesses[-1]
        predicted_next = last_access + avg_interval
        
        # Add prediction to stats
        self.stats['predictions_total'] += 1
        
        return predicted_next if std_interval < avg_interval else None
    
    def _preload_related_data(self, key: str, predicted_time: float):
        """Preload related data based on predictions."""
        # This would implement predictive loading logic
        # For now, just log the prediction
        time_until = predicted_time - time.time()
        if 0 < time_until < 3600:  # Within next hour
            self.logger.debug(f"Predicted access for {key} in {time_until:.1f} seconds")
    
    def _hash_key(self, key: str) -> str:
        """Create hash key for cache storage."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_pattern_key(self, key: str) -> str:
        """Extract pattern key for access learning."""
        # Group similar keys together for pattern learning
        # e.g., "solve_poisson_64x64" -> "solve_poisson"
        parts = key.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:2])
        return key
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Use pickle size as approximation
                return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimate
            return len(str(obj)) * 4  # Rough estimate
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(1, total_requests)
            
            # Pattern learning accuracy
            prediction_accuracy = 0.0
            if self.stats['predictions_total'] > 0:
                prediction_accuracy = self.stats['predictions_correct'] / self.stats['predictions_total']
            
            return {
                'size_mb': self.current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'entries': len(self.cache),
                'max_entries': self.max_entries,
                'hit_rate': hit_rate,
                'prediction_accuracy': prediction_accuracy,
                'evictions': self.stats['evictions'],
                'patterns_learned': len(self.access_patterns),
                **self.stats
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.access_patterns.clear()
            self.logger.info("Cache cleared")
    
    def optimize(self):
        """Optimize cache performance based on learned patterns."""
        with self.lock:
            # Re-score all entries
            for entry in self.cache.values():
                entry.score = entry._compute_score()
            
            # Remove very low-scoring entries proactively
            to_remove = []
            for key, entry in self.cache.items():
                if entry.score < 1.0 and time.time() - entry.last_access > 3600:
                    to_remove.append(key)
            
            for key in to_remove[:len(self.cache)//4]:  # Remove max 25%
                removed_entry = self.cache.pop(key)
                self.current_size_bytes -= removed_entry.size_bytes
            
            if to_remove:
                self.logger.info(f"Proactively evicted {len(to_remove)} low-scoring entries")


class CacheManager:
    """Global cache manager for analog PDE solver system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Multiple cache instances for different data types
        self.caches = {
            'crossbar_configs': IntelligentCache(max_size_mb=128, max_entries=500),
            'pde_solutions': IntelligentCache(max_size_mb=256, max_entries=200), 
            'hardware_simulations': IntelligentCache(max_size_mb=128, max_entries=100),
            'rtl_generations': IntelligentCache(max_size_mb=64, max_entries=50)
        }
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_worker, daemon=True)
        self.optimization_thread.start()
    
    def get_cache(self, cache_type: str) -> IntelligentCache:
        """Get specific cache instance."""
        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}")
        return self.caches[cache_type]
    
    def _optimization_worker(self):
        """Background worker for cache optimization."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                for cache_name, cache in self.caches.items():
                    cache.optimize()
                
                self.logger.debug("Cache optimization completed")
                
            except Exception as e:
                self.logger.error(f"Cache optimization error: {e}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        total_size_mb = 0
        total_entries = 0
        total_hit_rate = 0
        
        for cache_name, cache in self.caches.items():
            cache_stats = cache.get_cache_stats()
            stats[cache_name] = cache_stats
            total_size_mb += cache_stats['size_mb']
            total_entries += cache_stats['entries']
            total_hit_rate += cache_stats['hit_rate']
        
        stats['global'] = {
            'total_size_mb': total_size_mb,
            'total_entries': total_entries,
            'average_hit_rate': total_hit_rate / len(self.caches)
        }
        
        return stats
    
    def clear_all(self):
        """Clear all cache instances."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("All caches cleared")


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager