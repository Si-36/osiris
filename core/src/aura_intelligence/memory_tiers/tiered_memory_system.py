"""
Tiered Memory System - 2025 Implementation

Based on latest research:
- CXL 3.0 memory pooling and coherence
- DDR5/HBM3 high-bandwidth memory
- Persistent memory (Intel Optane evolution)
- NVMe/SSD cold storage
- NUMA-aware placement
- Intelligent tiering algorithms

Key features:
- Heterogeneous memory management
- Automatic data movement between tiers
- NUMA-aware allocation
- Memory pooling with CXL
- Persistent memory support
- Cost/performance optimization
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
import hashlib
import json
import structlog
from collections import defaultdict, OrderedDict
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = structlog.get_logger(__name__)


class MemoryTier(IntEnum):
    """Memory tier hierarchy from fastest to slowest"""
    L0_HBM = 0      # High Bandwidth Memory (HBM3) - 1TB/s
    L1_DDR = 1      # DDR5 DRAM - 100GB/s
    L2_CXL = 2      # CXL-attached memory - 50GB/s
    L3_PMEM = 3     # Persistent memory - 10GB/s
    L4_NVME = 4     # NVMe SSD - 5GB/s
    L5_DISK = 5     # Disk storage - 200MB/s


class AccessPattern(str, Enum):
    """Data access patterns"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STREAMING = "streaming"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class PlacementPolicy(str, Enum):
    """Memory placement policies"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    NUMA_LOCAL = "numa_local"
    PERFORMANCE = "performance"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class TierConfig:
    """Configuration for each memory tier"""
    tier: MemoryTier
    capacity_gb: float
    bandwidth_gbps: float
    latency_ns: float
    cost_per_gb: float
    is_persistent: bool = False
    numa_node: Optional[int] = None


@dataclass
class MemoryObject:
    """Object stored in tiered memory"""
    key: str
    data: Any
    size_bytes: int
    
    # Tier placement
    current_tier: MemoryTier
    preferred_tier: Optional[MemoryTier] = None
    
    # Access tracking
    access_count: int = 0
    read_count: int = 0
    write_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    
    # Access pattern
    access_pattern: AccessPattern = AccessPattern.RANDOM
    access_history: List[float] = field(default_factory=list)
    
    # Temperature (hot/cold)
    temperature: float = 1.0  # 0=cold, 1=hot
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False  # If true, don't move between tiers
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()
        self.access_history.append(self.last_access)
        
        # Keep only recent history
        if len(self.access_history) > 100:
            self.access_history = self.access_history[-100:]
        
        # Update temperature based on access frequency
        self._update_temperature()
    
    def _update_temperature(self):
        """Calculate data temperature based on access pattern"""
        current_time = time.time()
        
        # Calculate access frequency
        if len(self.access_history) < 2:
            self.temperature = 1.0
            return
        
        # Recent access rate (last 10 accesses)
        recent_accesses = self.access_history[-10:]
        time_span = current_time - recent_accesses[0]
        
        if time_span > 0:
            access_rate = len(recent_accesses) / time_span
            # Normalize to 0-1 range (assuming max 100 accesses/second is hot)
            self.temperature = min(1.0, access_rate / 100.0)
        
        # Decay temperature over time
        time_since_access = current_time - self.last_access
        decay_factor = np.exp(-time_since_access / 3600.0)  # 1 hour half-life
        self.temperature *= decay_factor


class TierManager:
    """Manages individual memory tier"""
    
    def __init__(self, config: TierConfig):
        self.config = config
        self.tier = config.tier
        self.capacity_bytes = int(config.capacity_gb * 1024 * 1024 * 1024)
        self.used_bytes = 0
        
        # Storage based on tier type
        self.storage: Dict[str, MemoryObject] = OrderedDict()
        
        # Statistics
        self.stats = {
            "reads": 0,
            "writes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # For persistent tiers, use memory-mapped files
        self.mmap_file = None
        if config.is_persistent:
            self._init_persistent_storage()
        
        logger.info(f"Initialized {self.tier.name} tier with {config.capacity_gb}GB")
    
    def _init_persistent_storage(self):
        """Initialize persistent storage using memory-mapped files"""
        try:
            filename = f"/tmp/aura_tier_{self.tier.name}.dat"
            
            # Create file if doesn't exist
            if not os.path.exists(filename):
                with open(filename, 'wb') as f:
                    f.write(b'\0' * self.capacity_bytes)
            
            # Memory map the file
            with open(filename, 'r+b') as f:
                self.mmap_file = mmap.mmap(f.fileno(), self.capacity_bytes)
            
            logger.info(f"Initialized persistent storage for {self.tier.name}")
            
        except Exception as e:
            logger.warning(f"Failed to init persistent storage: {e}")
    
    def can_fit(self, size_bytes: int) -> bool:
        """Check if object can fit in this tier"""
        return self.used_bytes + size_bytes <= self.capacity_bytes
    
    async def store(self, obj: MemoryObject) -> bool:
        """Store object in this tier"""
        if not self.can_fit(obj.size_bytes):
            return False
        
        # Evict if necessary
        while self.used_bytes + obj.size_bytes > self.capacity_bytes:
            await self._evict_coldest()
        
        # Store object
        self.storage[obj.key] = obj
        self.used_bytes += obj.size_bytes
        obj.current_tier = self.tier
        
        self.stats["writes"] += 1
        
        # Simulate tier-specific latency
        await asyncio.sleep(self.config.latency_ns / 1e9)
        
        return True
    
    async def retrieve(self, key: str) -> Optional[MemoryObject]:
        """Retrieve object from this tier"""
        if key in self.storage:
            obj = self.storage[key]
            obj.update_access()
            
            self.stats["reads"] += 1
            self.stats["hits"] += 1
            
            # Move to end (LRU)
            self.storage.move_to_end(key)
            
            # Simulate tier-specific latency
            await asyncio.sleep(self.config.latency_ns / 1e9)
            
            return obj
        
        self.stats["misses"] += 1
        return None
    
    async def remove(self, key: str) -> Optional[MemoryObject]:
        """Remove object from this tier"""
        if key in self.storage:
            obj = self.storage.pop(key)
            self.used_bytes -= obj.size_bytes
            return obj
        return None
    
    async def _evict_coldest(self) -> Optional[MemoryObject]:
        """Evict coldest object from tier"""
        if not self.storage:
            return None
        
        # Find coldest unpinned object
        coldest_key = None
        coldest_temp = float('inf')
        
        for key, obj in self.storage.items():
            if not obj.pinned and obj.temperature < coldest_temp:
                coldest_temp = obj.temperature
                coldest_key = key
        
        if coldest_key:
            self.stats["evictions"] += 1
            return await self.remove(coldest_key)
        
        return None
    
    def get_utilization(self) -> float:
        """Get tier utilization percentage"""
        return self.used_bytes / self.capacity_bytes if self.capacity_bytes > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tier statistics"""
        return {
            **self.stats,
            "utilization": self.get_utilization(),
            "capacity_gb": self.config.capacity_gb,
            "used_gb": self.used_bytes / (1024**3),
            "objects": len(self.storage)
        }


class TieringPolicy:
    """Intelligent data tiering policy"""
    
    def __init__(self):
        # Thresholds for tier promotion/demotion
        self.hot_threshold = 0.7
        self.cold_threshold = 0.3
        
        # Cost/performance weights
        self.performance_weight = 0.6
        self.cost_weight = 0.4
        
        logger.info("Initialized tiering policy")
    
    def determine_tier(self, 
                      obj: MemoryObject,
                      tier_configs: List[TierConfig],
                      current_stats: Dict[MemoryTier, Dict]) -> MemoryTier:
        """Determine optimal tier for object"""
        # Hot data goes to faster tiers
        if obj.temperature > self.hot_threshold:
            # Prefer HBM/DDR for hot data
            if obj.size_bytes < 1024 * 1024:  # Small hot objects to HBM
                return MemoryTier.L0_HBM
            else:
                return MemoryTier.L1_DDR
        
        # Cold data goes to slower tiers
        elif obj.temperature < self.cold_threshold:
            # Large cold data to disk
            if obj.size_bytes > 100 * 1024 * 1024:
                return MemoryTier.L5_DISK
            else:
                return MemoryTier.L4_NVME
        
        # Warm data in middle tiers
        else:
            # Consider access pattern
            if obj.access_pattern == AccessPattern.STREAMING:
                return MemoryTier.L2_CXL  # Good for streaming
            elif obj.access_pattern == AccessPattern.TEMPORAL:
                return MemoryTier.L3_PMEM  # Good for temporal locality
            else:
                return MemoryTier.L1_DDR  # Default to DRAM
    
    def should_promote(self, obj: MemoryObject) -> bool:
        """Check if object should be promoted to faster tier"""
        return obj.temperature > self.hot_threshold and not obj.pinned
    
    def should_demote(self, obj: MemoryObject) -> bool:
        """Check if object should be demoted to slower tier"""
        return obj.temperature < self.cold_threshold and not obj.pinned


class CXLMemoryPool:
    """CXL 3.0 memory pooling and coherence"""
    
    def __init__(self, pool_size_gb: int = 1024):
        self.pool_size_bytes = pool_size_gb * 1024 * 1024 * 1024
        self.allocated_bytes = 0
        
        # CXL devices (simulated)
        self.devices: Dict[str, Dict[str, Any]] = {}
        
        # Coherence directory
        self.coherence_dir: Dict[str, Set[str]] = defaultdict(set)
        
        # Pool statistics
        self.stats = {
            "allocations": 0,
            "deallocations": 0,
            "coherence_updates": 0
        }
        
        logger.info(f"Initialized CXL memory pool with {pool_size_gb}GB")
    
    async def allocate(self, size_bytes: int, device_id: str) -> Optional[int]:
        """Allocate memory from CXL pool"""
        if self.allocated_bytes + size_bytes > self.pool_size_bytes:
            return None
        
        # Simulate CXL allocation
        allocation_addr = self.allocated_bytes
        self.allocated_bytes += size_bytes
        
        # Track allocation
        if device_id not in self.devices:
            self.devices[device_id] = {
                "allocations": {},
                "total_allocated": 0
            }
        
        self.devices[device_id]["allocations"][allocation_addr] = size_bytes
        self.devices[device_id]["total_allocated"] += size_bytes
        
        self.stats["allocations"] += 1
        
        # Simulate CXL latency (50ns)
        await asyncio.sleep(50e-9)
        
        return allocation_addr
    
    async def deallocate(self, addr: int, device_id: str):
        """Deallocate memory back to pool"""
        if device_id in self.devices and addr in self.devices[device_id]["allocations"]:
            size_bytes = self.devices[device_id]["allocations"].pop(addr)
            self.devices[device_id]["total_allocated"] -= size_bytes
            
            self.stats["deallocations"] += 1
            
            # Simulate CXL latency
            await asyncio.sleep(50e-9)
    
    async def update_coherence(self, addr: int, device_ids: Set[str]):
        """Update coherence directory for shared data"""
        self.coherence_dir[str(addr)] = device_ids
        self.stats["coherence_updates"] += 1
        
        # Simulate coherence protocol overhead
        await asyncio.sleep(100e-9)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CXL pool statistics"""
        return {
            **self.stats,
            "pool_size_gb": self.pool_size_bytes / (1024**3),
            "allocated_gb": self.allocated_bytes / (1024**3),
            "utilization": self.allocated_bytes / self.pool_size_bytes,
            "devices": len(self.devices),
            "coherence_entries": len(self.coherence_dir)
        }


class HeterogeneousMemorySystem:
    """
    Complete heterogeneous memory system
    Manages multiple memory tiers with intelligent data placement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize tier configurations
        self.tier_configs = self._create_tier_configs(config)
        
        # Initialize tier managers
        self.tiers: Dict[MemoryTier, TierManager] = {}
        for tier_config in self.tier_configs:
            self.tiers[tier_config.tier] = TierManager(tier_config)
        
        # Tiering policy
        self.policy = TieringPolicy()
        
        # CXL memory pool
        self.cxl_pool = CXLMemoryPool(
            pool_size_gb=config.get("cxl_pool_gb", 1024)
        )
        
        # Migration engine
        self.migration_executor = ThreadPoolExecutor(max_workers=4)
        self.migration_queue: asyncio.Queue = asyncio.Queue()
        
        # Global index
        self.global_index: Dict[str, MemoryTier] = {}
        
        # Statistics
        self.stats = {
            "total_objects": 0,
            "migrations": 0,
            "promotions": 0,
            "demotions": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Heterogeneous memory system initialized")
    
    def _create_tier_configs(self, config: Dict[str, Any]) -> List[TierConfig]:
        """Create tier configurations"""
        return [
            TierConfig(
                tier=MemoryTier.L0_HBM,
                capacity_gb=config.get("hbm_gb", 16),
                bandwidth_gbps=1000,
                latency_ns=20,
                cost_per_gb=100
            ),
            TierConfig(
                tier=MemoryTier.L1_DDR,
                capacity_gb=config.get("ddr_gb", 128),
                bandwidth_gbps=100,
                latency_ns=50,
                cost_per_gb=10
            ),
            TierConfig(
                tier=MemoryTier.L2_CXL,
                capacity_gb=config.get("cxl_gb", 512),
                bandwidth_gbps=50,
                latency_ns=100,
                cost_per_gb=5
            ),
            TierConfig(
                tier=MemoryTier.L3_PMEM,
                capacity_gb=config.get("pmem_gb", 1024),
                bandwidth_gbps=10,
                latency_ns=300,
                cost_per_gb=2,
                is_persistent=True
            ),
            TierConfig(
                tier=MemoryTier.L4_NVME,
                capacity_gb=config.get("nvme_gb", 4096),
                bandwidth_gbps=5,
                latency_ns=10000,
                cost_per_gb=0.5,
                is_persistent=True
            ),
            TierConfig(
                tier=MemoryTier.L5_DISK,
                capacity_gb=config.get("disk_gb", 10000),
                bandwidth_gbps=0.2,
                latency_ns=1000000,
                cost_per_gb=0.1,
                is_persistent=True
            )
        ]
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        asyncio.create_task(self._migration_worker())
        asyncio.create_task(self._tier_optimizer())
        asyncio.create_task(self._stats_collector())
    
    async def store(self,
                   key: str,
                   data: Any,
                   size_bytes: Optional[int] = None,
                   preferred_tier: Optional[MemoryTier] = None,
                   access_pattern: AccessPattern = AccessPattern.RANDOM,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data in tiered memory"""
        # Calculate size if not provided
        if size_bytes is None:
            # Simple estimation
            size_bytes = len(str(data).encode('utf-8'))
        
        # Create memory object
        obj = MemoryObject(
            key=key,
            data=data,
            size_bytes=size_bytes,
            current_tier=preferred_tier or MemoryTier.L1_DDR,
            preferred_tier=preferred_tier,
            access_pattern=access_pattern,
            metadata=metadata or {}
        )
        
        # Determine optimal tier
        if preferred_tier is None:
            target_tier = self.policy.determine_tier(
                obj, self.tier_configs, self._get_tier_stats()
            )
        else:
            target_tier = preferred_tier
        
        # Try to store in target tier
        stored = False
        for tier in range(target_tier, MemoryTier.L5_DISK + 1):
            if tier in self.tiers:
                if await self.tiers[MemoryTier(tier)].store(obj):
                    stored = True
                    self.global_index[key] = MemoryTier(tier)
                    break
        
        if stored:
            self.stats["total_objects"] += 1
            logger.debug(f"Stored {key} in {obj.current_tier.name}")
        else:
            logger.warning(f"Failed to store {key} - no space available")
        
        return stored
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from any tier"""
        if key not in self.global_index:
            return None
        
        # Check current tier
        current_tier = self.global_index[key]
        obj = await self.tiers[current_tier].retrieve(key)
        
        if obj:
            # Check if should migrate
            if self.policy.should_promote(obj):
                await self.migration_queue.put((obj, "promote"))
            elif self.policy.should_demote(obj):
                await self.migration_queue.put((obj, "demote"))
            
            return obj.data
        
        return None
    
    async def delete(self, key: str) -> bool:
        """Delete data from memory"""
        if key not in self.global_index:
            return False
        
        current_tier = self.global_index[key]
        obj = await self.tiers[current_tier].remove(key)
        
        if obj:
            del self.global_index[key]
            self.stats["total_objects"] -= 1
            return True
        
        return False
    
    async def _migration_worker(self):
        """Background worker for data migration"""
        while True:
            try:
                obj, direction = await self.migration_queue.get()
                
                if direction == "promote":
                    await self._promote_object(obj)
                elif direction == "demote":
                    await self._demote_object(obj)
                
                self.stats["migrations"] += 1
                
            except Exception as e:
                logger.error(f"Migration error: {e}")
            
            await asyncio.sleep(0.1)
    
    async def _promote_object(self, obj: MemoryObject):
        """Promote object to faster tier"""
        current_tier_num = obj.current_tier.value
        
        if current_tier_num == 0:  # Already at fastest
            return
        
        # Try to promote to next faster tier
        for tier_num in range(current_tier_num - 1, -1, -1):
            target_tier = MemoryTier(tier_num)
            
            if target_tier in self.tiers:
                if await self.tiers[target_tier].store(obj):
                    # Remove from current tier
                    await self.tiers[obj.current_tier].remove(obj.key)
                    
                    # Update index
                    self.global_index[obj.key] = target_tier
                    
                    self.stats["promotions"] += 1
                    logger.debug(f"Promoted {obj.key} from {obj.current_tier.name} to {target_tier.name}")
                    break
    
    async def _demote_object(self, obj: MemoryObject):
        """Demote object to slower tier"""
        current_tier_num = obj.current_tier.value
        
        if current_tier_num == MemoryTier.L5_DISK:  # Already at slowest
            return
        
        # Try to demote to next slower tier
        for tier_num in range(current_tier_num + 1, MemoryTier.L5_DISK + 1):
            target_tier = MemoryTier(tier_num)
            
            if target_tier in self.tiers:
                if await self.tiers[target_tier].store(obj):
                    # Remove from current tier
                    await self.tiers[obj.current_tier].remove(obj.key)
                    
                    # Update index
                    self.global_index[obj.key] = target_tier
                    
                    self.stats["demotions"] += 1
                    logger.debug(f"Demoted {obj.key} from {obj.current_tier.name} to {target_tier.name}")
                    break
    
    async def _tier_optimizer(self):
        """Periodically optimize tier placement"""
        while True:
            await asyncio.sleep(60)  # Run every minute
            
            try:
                # Check all objects for optimal placement
                for key, current_tier in list(self.global_index.items()):
                    obj = await self.tiers[current_tier].retrieve(key)
                    
                    if obj:
                        optimal_tier = self.policy.determine_tier(
                            obj, self.tier_configs, self._get_tier_stats()
                        )
                        
                        if optimal_tier != current_tier:
                            if optimal_tier < current_tier:
                                await self.migration_queue.put((obj, "promote"))
                            else:
                                await self.migration_queue.put((obj, "demote"))
                
            except Exception as e:
                logger.error(f"Tier optimization error: {e}")
    
    async def _stats_collector(self):
        """Collect and log statistics"""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds
            
            try:
                stats = self.get_stats()
                logger.info("Memory system stats", **stats)
                
            except Exception as e:
                logger.error(f"Stats collection error: {e}")
    
    def _get_tier_stats(self) -> Dict[MemoryTier, Dict]:
        """Get statistics for all tiers"""
        return {
            tier: manager.get_stats()
            for tier, manager in self.tiers.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        tier_stats = self._get_tier_stats()
        
        total_capacity = sum(s["capacity_gb"] for s in tier_stats.values())
        total_used = sum(s["used_gb"] for s in tier_stats.values())
        
        return {
            **self.stats,
            "total_capacity_gb": total_capacity,
            "total_used_gb": total_used,
            "overall_utilization": total_used / total_capacity if total_capacity > 0 else 0,
            "tier_stats": tier_stats,
            "cxl_pool": self.cxl_pool.get_stats()
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        # Get system memory info
        vm = psutil.virtual_memory()
        
        return {
            "system": {
                "total_gb": vm.total / (1024**3),
                "available_gb": vm.available / (1024**3),
                "percent_used": vm.percent
            },
            "tiers": self._get_tier_stats(),
            "policy": {
                "hot_threshold": self.policy.hot_threshold,
                "cold_threshold": self.policy.cold_threshold
            }
        }


# Example usage
async def demonstrate_tiered_memory():
    """Demonstrate tiered memory system"""
    print("🏗️ Tiered Memory System Demonstration")
    print("=" * 60)
    
    # Initialize memory system
    memory_config = {
        "hbm_gb": 16,
        "ddr_gb": 64,
        "cxl_gb": 256,
        "pmem_gb": 512,
        "nvme_gb": 2048,
        "disk_gb": 10000,
        "cxl_pool_gb": 1024
    }
    
    memory_system = HeterogeneousMemorySystem(memory_config)
    
    # Test data placement
    print("\n1️⃣ Testing Data Placement")
    print("-" * 40)
    
    # Store hot data
    hot_data = {"type": "hot", "value": np.random.randn(1000).tolist()}
    await memory_system.store(
        key="hot_data_1",
        data=hot_data,
        size_bytes=8000,
        access_pattern=AccessPattern.RANDOM
    )
    print("✅ Stored hot data")
    
    # Store streaming data
    stream_data = {"type": "stream", "frames": list(range(1000))}
    await memory_system.store(
        key="stream_1",
        data=stream_data,
        size_bytes=4000,
        access_pattern=AccessPattern.STREAMING
    )
    print("✅ Stored streaming data")
    
    # Store cold data
    cold_data = {"type": "archive", "logs": ["log" * 100 for _ in range(100)]}
    await memory_system.store(
        key="cold_data_1",
        data=cold_data,
        size_bytes=100000,
        access_pattern=AccessPattern.SEQUENTIAL
    )
    print("✅ Stored cold data")
    
    # Test retrieval and access patterns
    print("\n2️⃣ Testing Data Access")
    print("-" * 40)
    
    # Access hot data multiple times
    for i in range(10):
        data = await memory_system.retrieve("hot_data_1")
        if i == 0:
            print(f"✅ Retrieved hot data: {data['type']}")
        await asyncio.sleep(0.1)
    
    # Access streaming data
    stream = await memory_system.retrieve("stream_1")
    print(f"✅ Retrieved stream data: {len(stream['frames'])} frames")
    
    # Let migration happen
    print("\n3️⃣ Waiting for Tier Optimization...")
    await asyncio.sleep(2)
    
    # Check tier placement
    print("\n4️⃣ Checking Tier Placement")
    print("-" * 40)
    
    for key in ["hot_data_1", "stream_1", "cold_data_1"]:
        if key in memory_system.global_index:
            tier = memory_system.global_index[key]
            print(f"{key}: {tier.name}")
    
    # Test CXL memory pool
    print("\n5️⃣ Testing CXL Memory Pool")
    print("-" * 40)
    
    # Allocate from CXL pool
    device_id = "gpu_0"
    addr = await memory_system.cxl_pool.allocate(1024 * 1024, device_id)
    if addr is not None:
        print(f"✅ Allocated 1MB from CXL pool at address {addr}")
    
    # Update coherence
    await memory_system.cxl_pool.update_coherence(addr, {"gpu_0", "cpu_0"})
    print("✅ Updated coherence directory")
    
    # Get statistics
    print("\n📊 Memory System Statistics")
    print("-" * 40)
    
    stats = memory_system.get_stats()
    
    print(f"Total objects: {stats['total_objects']}")
    print(f"Migrations: {stats['migrations']}")
    print(f"Promotions: {stats['promotions']}")
    print(f"Demotions: {stats['demotions']}")
    print(f"Total capacity: {stats['total_capacity_gb']:.1f} GB")
    print(f"Total used: {stats['total_used_gb']:.3f} GB")
    print(f"Overall utilization: {stats['overall_utilization']:.1%}")
    
    print("\nTier utilization:")
    for tier, tier_stats in stats['tier_stats'].items():
        print(f"  {tier.name}: {tier_stats['utilization']:.1%} ({tier_stats['objects']} objects)")
    
    print(f"\nCXL Pool:")
    cxl_stats = stats['cxl_pool']
    print(f"  Utilization: {cxl_stats['utilization']:.1%}")
    print(f"  Devices: {cxl_stats['devices']}")
    
    # Get memory info
    print("\n💾 System Memory Info")
    print("-" * 40)
    
    mem_info = memory_system.get_memory_info()
    sys_mem = mem_info['system']
    
    print(f"System RAM:")
    print(f"  Total: {sys_mem['total_gb']:.1f} GB")
    print(f"  Available: {sys_mem['available_gb']:.1f} GB")
    print(f"  Used: {sys_mem['percent_used']:.1f}%")
    
    print("\n✅ Tiered memory demonstration complete")


if __name__ == "__main__":
    asyncio.run(demonstrate_tiered_memory())