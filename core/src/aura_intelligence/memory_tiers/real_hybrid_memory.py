#!/usr/bin/env python3
"""
REAL HYBRID MEMORY SYSTEM - CXL 3.0 Memory Tiering
Hot/Warm/Cold memory tiers with real promotion/demotion
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, OrderedDict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class MemoryTier(Enum):
    """Memory tier levels based on CXL 3.0 architecture"""
    HOT = "L0_HBM3"      # 3.2 TB/s - Active tensors
    WARM = "L1_DDR5"     # 100 GB/s - Hot cache  
    COLD = "L2_CXL"      # 64 GB/s - Warm pool
    ARCHIVE = "L3_SSD"   # 10 GB/s - Cold archive

@dataclass
class MemoryItem:
    """Memory item with access tracking"""
    key: str
    data: Any
    tier: MemoryTier
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(str(self.data).encode('utf-8'))

class RealHybridMemoryManager:
    """Real hybrid memory manager with CXL-style tiering"""
    
    def __init__(self, 
                 hot_capacity_mb: int = 512,    # HBM3 capacity
                 warm_capacity_mb: int = 2048,  # DDR5 capacity  
                 cold_capacity_mb: int = 8192,  # CXL capacity
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        # Tier capacities in bytes
        self.tier_capacities = {
            MemoryTier.HOT: hot_capacity_mb * 1024 * 1024,
            MemoryTier.WARM: warm_capacity_mb * 1024 * 1024,
            MemoryTier.COLD: cold_capacity_mb * 1024 * 1024,
            MemoryTier.ARCHIVE: float('inf')  # Unlimited archive
        }
        
        # In-memory storage for hot/warm tiers
        self.hot_storage: OrderedDict[str, MemoryItem] = OrderedDict()
        self.warm_storage: OrderedDict[str, MemoryItem] = OrderedDict()
        self.cold_storage: Dict[str, MemoryItem] = {}
        
        # Redis for archive tier
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None
        
        # Access tracking
        self.access_stats = defaultdict(lambda: {'count': 0, 'last_access': 0})
        self.promotion_threshold = 5  # Access count for promotion
        self.demotion_age_seconds = 300  # 5 minutes for demotion
        
        # Background tasks
        self._running = True
        self._background_task = threading.Thread(target=self._background_manager, daemon=True)
        self._background_task.start()
        
        # Performance metrics
        self.metrics = {
            'total_accesses': 0,
            'hot_hits': 0,
            'warm_hits': 0,
            'cold_hits': 0,
            'archive_hits': 0,
            'promotions': 0,
            'demotions': 0,
            'evictions': 0
        }
    
    def _get_tier_storage(self, tier: MemoryTier):
        """Get storage for specific tier"""
        if tier == MemoryTier.HOT:
            return self.hot_storage
        elif tier == MemoryTier.WARM:
            return self.warm_storage
        elif tier == MemoryTier.COLD:
            return self.cold_storage
        else:  # ARCHIVE
            return None  # Redis storage
    
    def _calculate_tier_usage(self, tier: MemoryTier) -> int:
        """Calculate current usage of a tier in bytes"""
        storage = self._get_tier_storage(tier)
        if storage is None:
            return 0
        return sum(item.size_bytes for item in storage.values())
    
    async def store(self, key: str, data: Any, hint_tier: Optional[MemoryTier] = None) -> Dict[str, Any]:
        """Store data with automatic tier placement"""
        start_time = time.perf_counter()
        
        # Create memory item
        item = MemoryItem(key=key, data=data, tier=hint_tier or MemoryTier.HOT)
        
        # Determine optimal tier
        target_tier = self._determine_optimal_tier(item, hint_tier)
        item.tier = target_tier
        
        # Store in appropriate tier
        success = await self._store_in_tier(item, target_tier)
        
        if success:
            # Update access stats
            self.access_stats[key]['count'] = 1
            self.access_stats[key]['last_access'] = time.time()
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'stored': success,
            'key': key,
            'tier': target_tier.value,
            'size_bytes': item.size_bytes,
            'processing_time_ms': processing_time,
            'hybrid_memory': True
        }
    
    async def retrieve(self, key: str) -> Dict[str, Any]:
        """Retrieve data with automatic promotion"""
        start_time = time.perf_counter()
        self.metrics['total_accesses'] += 1
        
        # Search through tiers (hot -> warm -> cold -> archive)
        item = None
        found_tier = None
        
        # Check hot tier
        if key in self.hot_storage:
            item = self.hot_storage[key]
            found_tier = MemoryTier.HOT
            self.metrics['hot_hits'] += 1
        
        # Check warm tier
        elif key in self.warm_storage:
            item = self.warm_storage[key]
            found_tier = MemoryTier.WARM
            self.metrics['warm_hits'] += 1
        
        # Check cold tier
        elif key in self.cold_storage:
            item = self.cold_storage[key]
            found_tier = MemoryTier.COLD
            self.metrics['cold_hits'] += 1
        
        # Check archive tier (Redis)
        elif self.redis_client:
            try:
                archived_data = self.redis_client.get(f"archive:{key}")
                if archived_data:
                    data = json.loads(archived_data)
                    item = MemoryItem(key=key, data=data, tier=MemoryTier.ARCHIVE)
                    found_tier = MemoryTier.ARCHIVE
                    self.metrics['archive_hits'] += 1
            except Exception:
                pass
        
        if item:
            # Update access tracking
            item.access_count += 1
            item.last_access = time.time()
            self.access_stats[key]['count'] += 1
            self.access_stats[key]['last_access'] = time.time()
            
            # Consider promotion
            await self._consider_promotion(item, found_tier)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'found': True,
                'data': item.data,
                'tier': found_tier.value,
                'access_count': item.access_count,
                'processing_time_ms': processing_time,
                'hybrid_memory': True
            }
        else:
            processing_time = (time.perf_counter() - start_time) * 1000
            return {
                'found': False,
                'processing_time_ms': processing_time,
                'hybrid_memory': True
            }
    
    def _determine_optimal_tier(self, item: MemoryItem, hint: Optional[MemoryTier]) -> MemoryTier:
        """Determine optimal tier for new item"""
        if hint:
            return hint
        
        # Small, frequently accessed items go to hot tier
        if item.size_bytes < 1024:  # < 1KB
            return MemoryTier.HOT
        elif item.size_bytes < 10240:  # < 10KB
            return MemoryTier.WARM
        elif item.size_bytes < 102400:  # < 100KB
            return MemoryTier.COLD
        else:
            return MemoryTier.ARCHIVE
    
    async def _store_in_tier(self, item: MemoryItem, tier: MemoryTier) -> bool:
        """Store item in specific tier"""
        storage = self._get_tier_storage(tier)
        
        if tier == MemoryTier.ARCHIVE:
            # Store in Redis
            if self.redis_client:
                try:
                    self.redis_client.set(
                        f"archive:{item.key}", 
                        json.dumps(item.data),
                        ex=86400  # 24 hour expiry
                    )
                    return True
                except Exception:
                    return False
            return False
        
        # Check capacity
        current_usage = self._calculate_tier_usage(tier)
        if current_usage + item.size_bytes > self.tier_capacities[tier]:
            # Evict items to make space
            await self._evict_from_tier(tier, item.size_bytes)
        
        # Store item
        storage[item.key] = item
        return True
    
    async def _evict_from_tier(self, tier: MemoryTier, needed_bytes: int):
        """Evict items from tier to make space"""
        storage = self._get_tier_storage(tier)
        if not storage:
            return
        
        # Evict least recently used items
        items_to_evict = []
        bytes_to_free = needed_bytes
        
        # Sort by last access time (oldest first)
        sorted_items = sorted(storage.items(), key=lambda x: x[1].last_access)
        
        for key, item in sorted_items:
            if bytes_to_free <= 0:
                break
            items_to_evict.append((key, item))
            bytes_to_free -= item.size_bytes
        
        # Evict items and demote to lower tier
        for key, item in items_to_evict:
            del storage[key]
            self.metrics['evictions'] += 1
            
            # Try to demote to lower tier
            await self._demote_item(item)
    
    async def _consider_promotion(self, item: MemoryItem, current_tier: MemoryTier):
        """Consider promoting item to higher tier"""
        if item.access_count >= self.promotion_threshold:
            target_tier = self._get_promotion_tier(current_tier)
            if target_tier and target_tier != current_tier:
                success = await self._promote_item(item, current_tier, target_tier)
                if success:
                    self.metrics['promotions'] += 1
    
    def _get_promotion_tier(self, current_tier: MemoryTier) -> Optional[MemoryTier]:
        """Get target tier for promotion"""
        if current_tier == MemoryTier.ARCHIVE:
            return MemoryTier.COLD
        elif current_tier == MemoryTier.COLD:
            return MemoryTier.WARM
        elif current_tier == MemoryTier.WARM:
            return MemoryTier.HOT
        return None
    
    async def _promote_item(self, item: MemoryItem, from_tier: MemoryTier, to_tier: MemoryTier) -> bool:
        """Promote item to higher tier"""
        # Remove from current tier
        current_storage = self._get_tier_storage(from_tier)
        if current_storage and item.key in current_storage:
            del current_storage[item.key]
        elif from_tier == MemoryTier.ARCHIVE and self.redis_client:
            self.redis_client.delete(f"archive:{item.key}")
        
        # Store in target tier
        item.tier = to_tier
        return await self._store_in_tier(item, to_tier)
    
    async def _demote_item(self, item: MemoryItem):
        """Demote item to lower tier"""
        current_tier = item.tier
        
        if current_tier == MemoryTier.HOT:
            target_tier = MemoryTier.WARM
        elif current_tier == MemoryTier.WARM:
            target_tier = MemoryTier.COLD
        elif current_tier == MemoryTier.COLD:
            target_tier = MemoryTier.ARCHIVE
        else:
            return  # Already at lowest tier
        
        item.tier = target_tier
        await self._store_in_tier(item, target_tier)
        self.metrics['demotions'] += 1
    
    def _background_manager(self):
        """Background thread for memory management"""
        while self._running:
            try:
                # Age-based demotion
                current_time = time.time()
                
                # Check hot tier for old items
                for key, item in list(self.hot_storage.items()):
                    if current_time - item.last_access > self.demotion_age_seconds:
                        asyncio.run_coroutine_threadsafe(
                            self._demote_item(item), 
                            asyncio.new_event_loop()
                        )
                        del self.hot_storage[key]
                
                # Check warm tier for old items
                for key, item in list(self.warm_storage.items()):
                    if current_time - item.last_access > self.demotion_age_seconds * 2:
                        asyncio.run_coroutine_threadsafe(
                            self._demote_item(item),
                            asyncio.new_event_loop()
                        )
                        del self.warm_storage[key]
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Background manager error: {e}")
                time.sleep(60)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            'tier_usage': {},
            'tier_item_counts': {},
            'tier_capacities': {},
            'performance_metrics': self.metrics.copy()
        }
        
        for tier in MemoryTier:
            if tier == MemoryTier.ARCHIVE:
                stats['tier_usage'][tier.value] = 'unlimited'
                stats['tier_item_counts'][tier.value] = 'redis_managed'
            else:
                usage = self._calculate_tier_usage(tier)
                capacity = self.tier_capacities[tier]
                storage = self._get_tier_storage(tier)
                
                stats['tier_usage'][tier.value] = {
                    'used_bytes': usage,
                    'capacity_bytes': capacity,
                    'utilization_percent': (usage / capacity) * 100 if capacity > 0 else 0
                }
                stats['tier_item_counts'][tier.value] = len(storage) if storage else 0
            
            stats['tier_capacities'][tier.value] = self.tier_capacities[tier]
        
        # Calculate hit rates
        total_hits = sum([
            self.metrics['hot_hits'],
            self.metrics['warm_hits'], 
            self.metrics['cold_hits'],
            self.metrics['archive_hits']
        ])
        
        if total_hits > 0:
            stats['hit_rates'] = {
                'hot_hit_rate': self.metrics['hot_hits'] / total_hits,
                'warm_hit_rate': self.metrics['warm_hits'] / total_hits,
                'cold_hit_rate': self.metrics['cold_hits'] / total_hits,
                'archive_hit_rate': self.metrics['archive_hits'] / total_hits
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown the memory manager"""
        self._running = False
        if self.redis_client:
            self.redis_client.close()

def get_real_hybrid_memory():
    """Factory function to get real hybrid memory system"""
    return RealHybridMemoryManager()