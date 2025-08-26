"""Performance-oriented caching system for analog PDE solver."""

import numpy as np
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from threading import RLock
import time


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    computation_time: float = 0.0


class AdaptivePerformanceCache:
    """Adaptive caching system with performance optimization."""
    
    def __init__(
        self,
        max_memory_mb: int = 512,
        max_entries: int = 1000,
        ttl_seconds: int = 3600
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._memory_usage = 0
        self._lock = RLock()
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic hash from arguments
        key_data = {
            'args': [self._serialize_arg(arg) for arg in args],
            'kwargs': {k: self._serialize_arg(v) for k, v in sorted(kwargs.items())}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _serialize_arg(self, arg) -> str:
        """Serialize argument for hashing."""
        if isinstance(arg, np.ndarray):
            return f"array_{arg.shape}_{arg.dtype}_{hashlib.md5(arg.tobytes()).hexdigest()[:8]}"
        elif hasattr(arg, '__dict__'):
            # Object with attributes
            attrs = {k: str(v) for k, v in arg.__dict__.items() if not k.startswith('_')}
            return f"obj_{type(arg).__name__}_{hash(str(sorted(attrs.items())))}"
        else:
            return str(arg)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    self._evict(key)
                    self.miss_count += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                self.hit_count += 1
                return entry.data
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Store value in cache."""
        with self._lock:
            # Calculate size
            size = self._estimate_size(value)
            
            # Check if we need to make space
            while (self._memory_usage + size > self.max_memory_bytes or 
                   len(self._cache) >= self.max_entries):
                if not self._evict_lru():
                    break  # No more entries to evict
            
            # Store entry
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                size_bytes=size,
                computation_time=computation_time
            )
            
            # Remove old entry if exists
            if key in self._cache:
                self._memory_usage -= self._cache[key].size_bytes
            
            self._cache[key] = entry
            self._memory_usage += size
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
        elif isinstance(obj, str):
            return len(obj.encode('utf-8'))
        else:
            # Rough estimate
            return 100
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        
        # Find LRU entry (lowest access_count / age ratio)
        lru_key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].access_count / 
                                  max(time.time() - self._cache[k].timestamp, 1.0))
        
        self._evict(lru_key)
        return True
    
    def _evict(self, key: str) -> None:
        """Evict specific entry."""
        if key in self._cache:
            self._memory_usage -= self._cache[key].size_bytes
            del self._cache[key]
            self.eviction_count += 1
    
    def cache_decorator(self, func):
        """Decorator for automatic caching."""
        def wrapper(*args, **kwargs):
            cache_key = self._generate_key(func.__name__, *args, **kwargs)
            
            # Try cache first
            cached_result = self.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            self.put(cache_key, result, computation_time)
            return result
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'eviction_count': self.eviction_count,
            'entries': len(self._cache),
            'memory_usage_mb': self._memory_usage / (1024 * 1024),
            'memory_usage_percent': (self._memory_usage / self.max_memory_bytes) * 100
        }
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0


# Global cache instance
_global_cache = AdaptivePerformanceCache()

def cached(func):
    """Global cache decorator."""
    return _global_cache.cache_decorator(func)

def get_global_cache() -> AdaptivePerformanceCache:
    """Get global cache instance."""
    return _global_cache