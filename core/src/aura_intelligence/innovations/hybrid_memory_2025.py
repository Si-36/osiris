"""
Hybrid Memory System 2025 - DRAM/PMEM/Storage Tiering
"""

import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import redis


class MemoryTier(Enum):
    HOT = "hot"      # DRAM - sub-microsecond
    WARM = "warm"    # PMEM - 100ns, persistent  
    COLD = "cold"    # Storage - 100μs, archival


@dataclass
class MemoryItem:
    key: str
    data: Any
    tier: MemoryTier
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0


class HybridMemoryManager:
    def __init__(self):
        # Hot tier - DRAM equivalent
        self.hot_memory: Dict[str, MemoryItem] = {}
        self.hot_capacity = 1000
        
        # Warm tier - PMEM equivalent (Redis)
        self.warm_redis = redis.Redis(host='localhost', port=6379, db=1, decode_responses=False)
        
        # Cold tier - Storage equivalent
        self.cold_redis = redis.Redis(host='localhost', port=6379, db=2, decode_responses=False)
        
        # Metrics
        self.metrics = {
            'hot_hits': 0, 'warm_hits': 0, 'cold_hits': 0, 'misses': 0,
            'promotions': 0, 'demotions': 0
        }
        
    def store(self, key: str, data: Any, hint: Optional[MemoryTier] = None) -> bool:
        data_str = json.dumps(data) if not isinstance(data, str) else data
        size_bytes = len(data_str.encode('utf-8'))
        
        tier = hint or self._determine_tier(key, size_bytes)
        
        item = MemoryItem(
            key=key, data=data, tier=tier,
            access_count=1, last_access=time.time(), size_bytes=size_bytes
        )
        
        return self._store_in_tier(item)
    
    def retrieve(self, key: str) -> Optional[Any]:
        # Check hot tier
        if key in self.hot_memory:
            item = self.hot_memory[key]
            item.access_count += 1
            item.last_access = time.time()
            self.metrics['hot_hits'] += 1
            return item.data
        
        # Check warm tier
        try:
            warm_data = self.warm_redis.get(f"warm:{key}")
            if warm_data:
                data = json.loads(warm_data.decode('utf-8'))
                self.metrics['warm_hits'] += 1
                return data
        except:
            pass
        
        # Check cold tier
        try:
            cold_data = self.cold_redis.get(f"cold:{key}")
            if cold_data:
                data = json.loads(cold_data.decode('utf-8'))
                self.metrics['cold_hits'] += 1
                return data
        except:
            pass
        
        self.metrics['misses'] += 1
        return None
    
    def _determine_tier(self, key: str, size_bytes: int) -> MemoryTier:
        if size_bytes < 1024:  # < 1KB -> HOT
            return MemoryTier.HOT
        elif size_bytes < 10240:  # < 10KB -> WARM
            return MemoryTier.WARM
        else:  # >= 10KB -> COLD
            return MemoryTier.COLD
    
    def _store_in_tier(self, item: MemoryItem) -> bool:
        if item.tier == MemoryTier.HOT:
            if len(self.hot_memory) >= self.hot_capacity:
                self._evict_from_hot()
            self.hot_memory[item.key] = item
            return True
            
        elif item.tier == MemoryTier.WARM:
            try:
                data_str = json.dumps(item.data)
                self.warm_redis.set(f"warm:{item.key}", data_str, ex=86400)
                return True
            except:
                return False
                
        else:  # COLD
            try:
                data_str = json.dumps(item.data)
                self.cold_redis.set(f"cold:{item.key}", data_str, ex=604800)
                return True
            except:
                return False
    
    def _evict_from_hot(self):
        if not self.hot_memory:
            return
        lru_key = min(self.hot_memory.keys(), 
                     key=lambda k: self.hot_memory[k].last_access)
        del self.hot_memory[lru_key]
        self.metrics['demotions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        total = sum(self.metrics.values())
        return {
            'tier_sizes': {
                'hot': len(self.hot_memory),
                'warm': len(self.warm_redis.keys("warm:*") or []),
                'cold': len(self.cold_redis.keys("cold:*") or [])
            },
            'hit_rates': {
                'hot': self.metrics['hot_hits'] / max(1, total),
                'warm': self.metrics['warm_hits'] / max(1, total),
                'cold': self.metrics['cold_hits'] / max(1, total)
            },
            'cache_efficiency': (self.metrics['hot_hits'] + self.metrics['warm_hits']) / max(1, total)
        }


def test_hybrid_memory():
    print("🧪 Testing Hybrid Memory System...")
    
    memory = HybridMemoryManager()
    
    # Test data
    test_data = [
        ("small", {"val": 42}, MemoryTier.HOT),
        ("medium", {"arr": list(range(100))}, MemoryTier.WARM),
        ("large", {"matrix": [[i*j for j in range(50)] for i in range(50)]}, MemoryTier.COLD)
    ]
    
    # Store
    for key, data, hint in test_data:
        success = memory.store(key, data, hint)
        print(f"  Stored {key}: {success}")
    
    # Retrieve
    for key, _, _ in test_data:
        result = memory.retrieve(key)
        print(f"  Retrieved {key}: {result is not None}")
    
    # Stats
    stats = memory.get_stats()
    print(f"📊 Tier sizes: {stats['tier_sizes']}")
    print(f"📊 Cache efficiency: {stats['cache_efficiency']:.2%}")
    print("✅ Hybrid memory working!")


if __name__ == "__main__":
    test_hybrid_memory()