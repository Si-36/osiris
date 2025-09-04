"""
🧠 Advanced Hybrid Memory Manager 2025
State-of-the-art memory system with hierarchical tiers, attention-based consolidation,
and real-time access patterns for AURA's 40 memory components.
"""

import asyncio
import time
import json
import hashlib
import msgpack
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Deque
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timezone
import mmap
import os
import threading
import zstandard as zstd
import lmdb
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import structlog

logger = structlog.get_logger(__name__)


class MemoryTier(str, Enum):
    """Memory tier levels with access characteristics."""
    HOT = "hot"        # Redis/RAM - <1ms access, high frequency
    WARM = "warm"      # Compressed RAM - ~5ms access, medium frequency  
    COLD = "cold"      # LMDB/SSD - ~20ms access, low frequency
    ARCHIVE = "archive" # S3/Disk - >100ms access, rare access


@dataclass
class MemorySegment:
    """Individual memory segment with metadata."""
    segment_id: str
    key: str
    data: bytes
    tier: MemoryTier
    component_id: str
    created_at: float
    last_access: float
    access_count: int = 0
    size_bytes: int = 0
    importance_score: float = 0.5
    embedding: Optional[np.ndarray] = None
    access_pattern: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    def update_access(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()
        self.access_pattern.append(self.last_access)


@dataclass
class AccessStatistics:
    """Track access patterns for intelligent tier management."""
    total_accesses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tier_promotions: int = 0
    tier_demotions: int = 0
    avg_access_time_ms: float = 0
    access_frequency: Dict[str, int] = field(default_factory=dict)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


class AttentionMemoryConsolidator(nn.Module):
    """Neural network for attention-based memory consolidation."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process memory segments with attention."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class MemoryPatternAnalyzer:
    """Analyzes access patterns for predictive prefetching."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_sequences = defaultdict(lambda: deque(maxlen=window_size))
        self.pattern_cache = {}
        self.lock = threading.Lock()
    
    def record_access(self, key: str, timestamp: float):
        """Record memory access for pattern analysis."""
        with self.lock:
            self.access_sequences[key].append(timestamp)
    
    def predict_next_access(self, key: str) -> Optional[float]:
        """Predict when key will be accessed next."""
        with self.lock:
            sequence = self.access_sequences.get(key)
            if not sequence or len(sequence) < 3:
                return None
            
            # Simple prediction based on average interval
            intervals = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            avg_interval = np.mean(intervals)
            last_access = sequence[-1]
            
            return last_access + avg_interval
    
    def find_correlated_keys(self, key: str, threshold: float = 0.7) -> List[str]:
        """Find keys that are often accessed together."""
        correlations = []
        target_seq = self.access_sequences.get(key, deque())
        
        if len(target_seq) < 10:
            return []
        
        for other_key, other_seq in self.access_sequences.items():
            if other_key == key or len(other_seq) < 10:
                continue
            
            # Calculate temporal correlation
            correlation = self._calculate_temporal_correlation(
                list(target_seq), list(other_seq)
            )
            
            if correlation > threshold:
                correlations.append((other_key, correlation))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in correlations[:5]]  # Top 5 correlated keys
    
    def _calculate_temporal_correlation(self, seq1: List[float], seq2: List[float]) -> float:
        """Calculate correlation between two access sequences."""
        # Simple approach: check how often accesses occur within same time window
        window = 10.0  # 10 second window
        matches = 0
        
        for t1 in seq1[-20:]:  # Check last 20 accesses
            for t2 in seq2:
                if abs(t1 - t2) < window:
                    matches += 1
                    break
        
        return matches / min(20, len(seq1))


class HybridMemoryManager:
    """
    Advanced Hybrid Memory Manager with:
    - Multi-tier storage (Hot/Warm/Cold/Archive)
    - Attention-based memory consolidation
    - Predictive prefetching
    - Access pattern analysis
    - Automatic tier management
    - Real-time performance monitoring
    """
    
    def __init__(
        self,
        hot_size_mb: int = 512,
        warm_size_mb: int = 2048,
        cold_size_gb: int = 10,
        redis_url: str = "redis://localhost:6379",
        lmdb_path: str = "/tmp/aura_memory"
    ):
        # Initialize storage backends
        self.redis_client = None  # Will be initialized async
        self.redis_url = redis_url
        
        # LMDB for cold storage
        self.lmdb_env = lmdb.open(
            lmdb_path,
            map_size=cold_size_gb * 1024 * 1024 * 1024,
            max_dbs=10,
            writemap=True
        )
        
        # In-memory tiers
        self.hot_storage: Dict[str, MemorySegment] = {}
        self.warm_storage: Dict[str, MemorySegment] = {}
        
        # Tier limits (bytes)
        self.hot_limit = hot_size_mb * 1024 * 1024
        self.warm_limit = warm_size_mb * 1024 * 1024
        
        # Current usage tracking
        self.hot_usage = 0
        self.warm_usage = 0
        
        # Compression
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()
        
        # Neural components
        self.consolidator = AttentionMemoryConsolidator()
        self.pattern_analyzer = MemoryPatternAnalyzer()
        
        # Access tracking
        self.stats = AccessStatistics()
        self.segment_index: Dict[str, MemorySegment] = {}
        
        # Tier management thresholds
        self.promotion_threshold = 5     # Access count for promotion
        self.demotion_age_seconds = 3600 # 1 hour for demotion
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info(
            "Advanced HybridMemoryManager initialized",
            hot_size_mb=hot_size_mb,
            warm_size_mb=warm_size_mb,
            cold_size_gb=cold_size_gb
        )
    
    async def initialize(self):
        """Initialize async components."""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Redis connection established")
        
        # Start background maintenance
        asyncio.create_task(self._maintenance_loop())
        asyncio.create_task(self._prefetch_loop())
    
    async def store(
        self,
        key: str,
        data: Any,
        component_id: str,
        tier_hint: Optional[MemoryTier] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store data with intelligent tier placement.
        
        Args:
            key: Unique identifier
            data: Data to store
            component_id: Component that owns this data
            tier_hint: Optional tier preference
            importance: Importance score (0-1)
        
        Returns:
            Storage result with metrics
        """
        start_time = time.time()
        
        async with self.lock:
            # Serialize data
            serialized = msgpack.packb(data)
            size_bytes = len(serialized)
            
            # Generate embedding for similarity search
            embedding = self._generate_embedding(key, data)
            
            # Determine optimal tier
            if tier_hint:
                target_tier = tier_hint
            else:
                target_tier = await self._determine_tier(
                    key, size_bytes, importance, component_id
                )
            
            # Create segment
            segment = MemorySegment(
                segment_id=hashlib.sha256(f"{component_id}:{key}".encode()).hexdigest()[:16],
                key=key,
                data=serialized,
                tier=target_tier,
                component_id=component_id,
                created_at=time.time(),
                last_access=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                importance_score=importance,
                embedding=embedding
            )
            
            # Store in appropriate tier
            success = await self._store_in_tier(segment)
            
            # Update indexes
            self.segment_index[key] = segment
            
            # Record access pattern
            self.pattern_analyzer.record_access(key, time.time())
            
            storage_time_ms = (time.time() - start_time) * 1000
            self.stats.avg_access_time_ms = (
                self.stats.avg_access_time_ms * 0.9 + storage_time_ms * 0.1
            )
            
            result = {
                "success": success,
                "key": key,
                "tier": target_tier.value,
                "size_bytes": size_bytes,
                "duration_ms": storage_time_ms,
                "segment_id": segment.segment_id
            }
            
            logger.debug(
                "Data stored",
                key=key,
                tier=target_tier.value,
                size_bytes=size_bytes,
                duration_ms=storage_time_ms
            )
            
            return result
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve data from any tier with automatic promotion.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Deserialized data or None
        """
        start_time = time.time()
        
        async with self.lock:
            # Check if segment exists
            segment = self.segment_index.get(key)
            if not segment:
                self.stats.cache_misses += 1
                return None
            
            # Update access tracking
            segment.update_access()
            self.stats.total_accesses += 1
            self.pattern_analyzer.record_access(key, time.time())
            
            # Retrieve from appropriate tier
            data = await self._retrieve_from_tier(segment)
            
            if data is None:
                self.stats.cache_misses += 1
                return None
            
            self.stats.cache_hits += 1
            
            # Check for promotion
            if segment.access_count >= self.promotion_threshold:
                asyncio.create_task(self._promote_segment(segment))
            
            # Deserialize
            result = msgpack.unpackb(data)
            
            retrieval_time_ms = (time.time() - start_time) * 1000
            self.stats.avg_access_time_ms = (
                self.stats.avg_access_time_ms * 0.9 + retrieval_time_ms * 0.1
            )
            
            return result
    
    async def _store_in_tier(self, segment: MemorySegment) -> bool:
        """Store segment in the appropriate tier."""
        try:
            if segment.tier == MemoryTier.HOT:
                # Check space
                if self.hot_usage + segment.size_bytes > self.hot_limit:
                    await self._evict_from_hot()
                
                # Store in hot (in-memory)
                self.hot_storage[segment.key] = segment
                self.hot_usage += segment.size_bytes
                
                # Also store in Redis for persistence
                if self.redis_client:
                    await self.redis_client.setex(
                        f"hot:{segment.key}",
                        3600,  # 1 hour TTL
                        segment.data
                    )
            
            elif segment.tier == MemoryTier.WARM:
                # Compress data
                compressed = self.compressor.compress(segment.data)
                segment.data = compressed
                segment.size_bytes = len(compressed)
                
                # Check space
                if self.warm_usage + segment.size_bytes > self.warm_limit:
                    await self._evict_from_warm()
                
                # Store in warm
                self.warm_storage[segment.key] = segment
                self.warm_usage += segment.size_bytes
            
            elif segment.tier == MemoryTier.COLD:
                # Compress and store in LMDB
                compressed = self.compressor.compress(segment.data)
                
                with self.lmdb_env.begin(write=True) as txn:
                    txn.put(segment.key.encode(), compressed)
            
            else:  # ARCHIVE
                # Would store to S3/object storage in production
                # For now, use LMDB with heavy compression
                compressed = self.compressor.compress(segment.data)
                
                with self.lmdb_env.begin(write=True) as txn:
                    txn.put(f"archive:{segment.key}".encode(), compressed)
            
            return True
            
        except Exception as e:
            logger.error("Failed to store in tier", error=str(e), tier=segment.tier.value)
            return False
    
    async def _retrieve_from_tier(self, segment: MemorySegment) -> Optional[bytes]:
        """Retrieve data from the appropriate tier."""
        try:
            if segment.tier == MemoryTier.HOT:
                # Try in-memory first
                if segment.key in self.hot_storage:
                    return self.hot_storage[segment.key].data
                
                # Try Redis
                if self.redis_client:
                    data = await self.redis_client.get(f"hot:{segment.key}")
                    if data:
                        return data
            
            elif segment.tier == MemoryTier.WARM:
                if segment.key in self.warm_storage:
                    compressed = self.warm_storage[segment.key].data
                    return self.decompressor.decompress(compressed)
            
            elif segment.tier == MemoryTier.COLD:
                with self.lmdb_env.begin() as txn:
                    compressed = txn.get(segment.key.encode())
                    if compressed:
                        return self.decompressor.decompress(compressed)
            
            else:  # ARCHIVE
                with self.lmdb_env.begin() as txn:
                    compressed = txn.get(f"archive:{segment.key}".encode())
                    if compressed:
                        return self.decompressor.decompress(compressed)
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve from tier", error=str(e), tier=segment.tier.value)
            return None
    
    async def _determine_tier(
        self,
        key: str,
        size_bytes: int,
        importance: float,
        component_id: str
    ) -> MemoryTier:
        """Intelligently determine the best tier for data."""
        # Small, important data goes to HOT
        if size_bytes < 1024 * 10 and importance > 0.7:  # <10KB and important
            return MemoryTier.HOT
        
        # Medium data or moderate importance goes to WARM
        if size_bytes < 1024 * 1024 and importance > 0.3:  # <1MB
            return MemoryTier.WARM
        
        # Large data or low importance goes to COLD
        if importance > 0.1:
            return MemoryTier.COLD
        
        # Everything else goes to ARCHIVE
        return MemoryTier.ARCHIVE
    
    async def _promote_segment(self, segment: MemorySegment):
        """Promote segment to a higher tier."""
        old_tier = segment.tier
        
        if segment.tier == MemoryTier.ARCHIVE:
            new_tier = MemoryTier.COLD
        elif segment.tier == MemoryTier.COLD:
            new_tier = MemoryTier.WARM
        elif segment.tier == MemoryTier.WARM:
            new_tier = MemoryTier.HOT
        else:
            return  # Already in HOT
        
        # Retrieve data
        data = await self._retrieve_from_tier(segment)
        if not data:
            return
        
        # Update segment
        segment.tier = new_tier
        segment.data = data
        
        # Store in new tier
        await self._store_in_tier(segment)
        
        self.stats.tier_promotions += 1
        logger.info(
            "Segment promoted",
            key=segment.key,
            from_tier=old_tier.value,
            to_tier=new_tier.value
        )
    
    async def _evict_from_hot(self):
        """Evict least recently used segments from hot tier."""
        # Sort by last access time
        candidates = sorted(
            self.hot_storage.values(),
            key=lambda s: s.last_access
        )
        
        # Evict oldest 20%
        evict_count = max(1, len(candidates) // 5)
        
        for segment in candidates[:evict_count]:
            # Move to warm
            segment.tier = MemoryTier.WARM
            await self._store_in_tier(segment)
            
            # Remove from hot
            del self.hot_storage[segment.key]
            self.hot_usage -= segment.size_bytes
            
            self.stats.tier_demotions += 1
    
    async def _evict_from_warm(self):
        """Evict least recently used segments from warm tier."""
        candidates = sorted(
            self.warm_storage.values(),
            key=lambda s: s.last_access
        )
        
        evict_count = max(1, len(candidates) // 5)
        
        for segment in candidates[:evict_count]:
            # Move to cold
            segment.tier = MemoryTier.COLD
            await self._store_in_tier(segment)
            
            # Remove from warm
            del self.warm_storage[segment.key]
            self.warm_usage -= segment.size_bytes
            
            self.stats.tier_demotions += 1
    
    def _generate_embedding(self, key: str, data: Any) -> np.ndarray:
        """Generate embedding for similarity-based retrieval."""
        # Simple hash-based embedding for demo
        # In production, use a proper encoder
        key_hash = hashlib.sha256(key.encode()).digest()
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).digest()
        
        # Combine hashes into embedding
        embedding = np.frombuffer(key_hash + data_hash, dtype=np.uint8)
        embedding = embedding.astype(np.float32) / 255.0
        
        # Pad or truncate to fixed size
        target_size = 128
        if len(embedding) < target_size:
            embedding = np.pad(embedding, (0, target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return embedding
    
    async def _maintenance_loop(self):
        """Background maintenance for tier management."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                async with self.lock:
                    current_time = time.time()
                    
                    # Demote old segments in hot tier
                    for segment in list(self.hot_storage.values()):
                        age = current_time - segment.last_access
                        if age > self.demotion_age_seconds:
                            segment.tier = MemoryTier.WARM
                            await self._store_in_tier(segment)
                            del self.hot_storage[segment.key]
                            self.hot_usage -= segment.size_bytes
                            self.stats.tier_demotions += 1
                    
                    # Log statistics
                    logger.info(
                        "Memory stats",
                        hot_usage_mb=self.hot_usage / 1024 / 1024,
                        warm_usage_mb=self.warm_usage / 1024 / 1024,
                        hit_rate=self.stats.hit_rate,
                        promotions=self.stats.tier_promotions,
                        demotions=self.stats.tier_demotions
                    )
                    
            except Exception as e:
                logger.error("Maintenance error", error=str(e))
    
    async def _prefetch_loop(self):
        """Background prefetching based on access patterns."""
        while True:
            try:
                await asyncio.sleep(5)  # Run every 5 seconds
                
                # Find candidates for prefetching
                current_time = time.time()
                prefetch_candidates = []
                
                for key, segment in self.segment_index.items():
                    # Predict next access
                    predicted_access = self.pattern_analyzer.predict_next_access(key)
                    
                    if predicted_access and predicted_access - current_time < 10:
                        # Will be accessed soon, consider prefetching
                        if segment.tier in [MemoryTier.COLD, MemoryTier.ARCHIVE]:
                            prefetch_candidates.append(segment)
                
                # Prefetch top candidates
                for segment in prefetch_candidates[:5]:
                    logger.debug("Prefetching segment", key=segment.key)
                    await self._promote_segment(segment)
                    
            except Exception as e:
                logger.error("Prefetch error", error=str(e))
    
    async def search_similar(
        self,
        query_key: str,
        component_id: Optional[str] = None,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar memory segments using embeddings.
        
        Args:
            query_key: Key to search similar to
            component_id: Optional component filter
            k: Number of results
            
        Returns:
            List of (key, similarity_score) tuples
        """
        query_segment = self.segment_index.get(query_key)
        if not query_segment or query_segment.embedding is None:
            return []
        
        query_embedding = query_segment.embedding
        similarities = []
        
        for key, segment in self.segment_index.items():
            if key == query_key:
                continue
                
            if component_id and segment.component_id != component_id:
                continue
                
            if segment.embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, segment.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(segment.embedding)
                )
                similarities.append((key, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    async def consolidate_memories(
        self,
        component_id: str,
        max_segments: int = 100
    ) -> Dict[str, Any]:
        """
        Use attention-based consolidation to merge related memories.
        
        Args:
            component_id: Component to consolidate
            max_segments: Maximum segments to process
            
        Returns:
            Consolidation results
        """
        # Get segments for component
        segments = [
            s for s in self.segment_index.values()
            if s.component_id == component_id
        ][:max_segments]
        
        if len(segments) < 2:
            return {"consolidated": 0, "message": "Not enough segments"}
        
        # Extract embeddings
        embeddings = torch.tensor([
            s.embedding if s.embedding is not None else np.zeros(128)
            for s in segments
        ])
        
        # Apply attention-based consolidation
        with torch.no_grad():
            consolidated = self.consolidator(embeddings.unsqueeze(0))
            consolidated = consolidated.squeeze(0)
        
        # Find segments to merge based on attention scores
        attention_scores = torch.matmul(consolidated, consolidated.T)
        merge_pairs = []
        
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if attention_scores[i, j] > 0.8:  # High attention score
                    merge_pairs.append((i, j))
        
        # Merge segments (simplified for demo)
        merged_count = 0
        for i, j in merge_pairs[:10]:  # Limit merges
            # In production, would actually merge data
            merged_count += 1
        
        return {
            "consolidated": merged_count,
            "total_segments": len(segments),
            "merge_pairs": len(merge_pairs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "tiers": {
                "hot": {
                    "count": len(self.hot_storage),
                    "usage_mb": self.hot_usage / 1024 / 1024,
                    "limit_mb": self.hot_limit / 1024 / 1024
                },
                "warm": {
                    "count": len(self.warm_storage),
                    "usage_mb": self.warm_usage / 1024 / 1024,
                    "limit_mb": self.warm_limit / 1024 / 1024
                },
                "cold": {
                    "count": sum(1 for s in self.segment_index.values() if s.tier == MemoryTier.COLD)
                },
                "archive": {
                    "count": sum(1 for s in self.segment_index.values() if s.tier == MemoryTier.ARCHIVE)
                }
            },
            "performance": {
                "total_accesses": self.stats.total_accesses,
                "hit_rate": self.stats.hit_rate,
                "avg_access_time_ms": self.stats.avg_access_time_ms,
                "promotions": self.stats.tier_promotions,
                "demotions": self.stats.tier_demotions
            },
            "total_segments": len(self.segment_index)
        }
    
    async def close(self):
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()
        self.lmdb_env.close()
        self.executor.shutdown()
        logger.info("HybridMemoryManager closed")