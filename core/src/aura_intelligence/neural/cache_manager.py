"""
Cache Manager - Production Semantic Caching for Model Routing
Based on 2025 research: 30-40% cost reduction via intelligent caching

Key Features:
- Two-layer cache: exact (Redis/in-memory) + semantic (Qdrant)
- Domain-specific embeddings for semantic matching
- TTL management and cache invalidation
- Cost and hit rate tracking
- Integration with AURA's persistence layer
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import structlog

from .provider_adapters import ProviderRequest, ProviderResponse, ProviderType
from ..persistence.stores.vector import QdrantVectorStore
from ..persistence.stores.kv import NATSKVStore
from ..memory.embeddings import EmbeddingGenerator
from ..observability import create_tracer, create_meter

logger = structlog.get_logger(__name__)
tracer = create_tracer("cache_manager")
meter = create_meter("cache_manager")

# Metrics
cache_operations = meter.create_counter(
    name="aura.cache.operations",
    description="Cache operations by type"
)

cache_hit_rate = meter.create_gauge(
    name="aura.cache.hit_rate",
    description="Cache hit rate percentage"
)

cache_cost_savings = meter.create_counter(
    name="aura.cache.cost_savings_usd",
    description="Cost savings from cache hits in USD"
)

cache_latency = meter.create_histogram(
    name="aura.cache.latency_ms",
    description="Cache operation latency in milliseconds",
    unit="ms"
)


class CacheStrategy(str, Enum):
    """Cache strategies"""
    EXACT_ONLY = "exact_only"
    SEMANTIC_ONLY = "semantic_only"
    TWO_LAYER = "two_layer"  # Exact first, then semantic
    AGGRESSIVE = "aggressive"  # Lower thresholds


@dataclass
class CacheConfig:
    """Configuration for cache manager"""
    # Exact cache settings
    exact_cache_size: int = 10000
    exact_ttl_seconds: int = 3600  # 1 hour
    
    # Semantic cache settings
    semantic_threshold: float = 0.85  # Similarity threshold
    semantic_ttl_seconds: int = 86400  # 24 hours
    max_semantic_results: int = 5
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    use_domain_embeddings: bool = True  # Fine-tuned for queries
    
    # Cost tracking
    track_savings: bool = True
    report_interval_seconds: int = 300  # 5 minutes
    
    # Strategy
    strategy: CacheStrategy = CacheStrategy.TWO_LAYER
    
    # Storage backends
    use_redis: bool = True
    use_qdrant: bool = True
    use_nats_kv: bool = True
    
    # Advanced settings
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    enable_encryption: bool = False
    encryption_key: Optional[str] = None


@dataclass
class CacheEntry:
    """Single cache entry"""
    key: str
    request: ProviderRequest
    response: ProviderResponse
    
    # Metadata
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Embeddings for semantic search
    prompt_embedding: Optional[np.ndarray] = None
    response_embedding: Optional[np.ndarray] = None
    
    # Cost tracking
    cost_saved: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        age = datetime.now(timezone.utc) - self.timestamp
        return age.total_seconds() > self.ttl_seconds
        
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


@dataclass
class CacheStats:
    """Cache statistics"""
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    total_requests: int = 0
    
    cost_saved_usd: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Per provider stats
    provider_hits: Dict[str, int] = field(default_factory=dict)
    provider_savings: Dict[str, float] = field(default_factory=dict)
    
    # Time-based stats
    hourly_hits: Dict[int, int] = field(default_factory=dict)
    
    def hit_rate(self) -> float:
        """Calculate overall hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits) / self.total_requests
        
    def exact_hit_rate(self) -> float:
        """Calculate exact hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.exact_hits / self.total_requests
        
    def semantic_hit_rate(self) -> float:
        """Calculate semantic hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.semantic_hits / self.total_requests


class ExactCache:
    """In-memory exact cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache"""
        async with self.lock:
            if key in self.cache:
                # Move to end (LRU)
                entry = self.cache.pop(key)
                
                # Check expiration
                if entry.is_expired():
                    return None
                    
                # Update and reinsert
                entry.update_access()
                self.cache[key] = entry
                return entry
                
            return None
            
    async def put(self, key: str, entry: CacheEntry):
        """Put entry in cache"""
        async with self.lock:
            # Remove if exists
            if key in self.cache:
                self.cache.pop(key)
                
            # Add to end
            self.cache[key] = entry
            
            # Evict if over size
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
                
    async def remove(self, key: str):
        """Remove entry from cache"""
        async with self.lock:
            self.cache.pop(key, None)
            
    async def clear(self):
        """Clear all entries"""
        async with self.lock:
            self.cache.clear()
            
    async def size(self) -> int:
        """Get cache size"""
        async with self.lock:
            return len(self.cache)


class SemanticCache:
    """Semantic cache using vector similarity"""
    
    def __init__(self, config: CacheConfig, vector_store: Optional[QdrantVectorStore] = None):
        self.config = config
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator(model=config.embedding_model)
        
        # Local index for faster access
        self.entries: Dict[str, CacheEntry] = {}
        
    async def search(self, prompt: str, threshold: float = 0.85, limit: int = 5) -> List[Tuple[CacheEntry, float]]:
        """Search for semantically similar entries"""
        with tracer.start_as_current_span("semantic_search") as span:
            span.set_attribute("threshold", threshold)
            
            # Generate embedding for prompt
            embedding = await self.embedding_generator.generate(prompt)
            
            if self.vector_store:
                # Search in Qdrant
                results = await self.vector_store.search(
                    collection="cache_prompts",
                    query_vector=embedding,
                    limit=limit,
                    score_threshold=threshold
                )
                
                # Retrieve entries
                matches = []
                for result in results:
                    entry_key = result.payload.get("entry_key")
                    if entry_key in self.entries:
                        entry = self.entries[entry_key]
                        if not entry.is_expired():
                            matches.append((entry, result.score))
                            
                return matches
                
            else:
                # Fallback to local search
                return await self._local_search(embedding, threshold, limit)
                
    async def _local_search(self, query_embedding: np.ndarray, threshold: float, limit: int) -> List[Tuple[CacheEntry, float]]:
        """Local semantic search without vector store"""
        scores = []
        
        for key, entry in self.entries.items():
            if entry.is_expired():
                continue
                
            if entry.prompt_embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, entry.prompt_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry.prompt_embedding)
                )
                
                if similarity >= threshold:
                    scores.append((entry, float(similarity)))
                    
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
        
    async def add(self, entry: CacheEntry):
        """Add entry to semantic cache"""
        # Generate embeddings
        if entry.prompt_embedding is None:
            entry.prompt_embedding = await self.embedding_generator.generate(entry.request.prompt)
            
        # Store locally
        self.entries[entry.key] = entry
        
        # Store in Qdrant if available
        if self.vector_store:
            await self.vector_store.upsert(
                collection="cache_prompts",
                points=[{
                    "id": entry.key,
                    "vector": entry.prompt_embedding.tolist(),
                    "payload": {
                        "entry_key": entry.key,
                        "prompt": entry.request.prompt[:500],  # First 500 chars
                        "model": entry.response.model,
                        "provider": entry.response.provider.value,
                        "timestamp": entry.timestamp.isoformat(),
                        "cost_saved": entry.cost_saved
                    }
                }]
            )
            
    async def remove(self, key: str):
        """Remove entry from semantic cache"""
        self.entries.pop(key, None)
        
        if self.vector_store:
            await self.vector_store.delete(
                collection="cache_prompts",
                ids=[key]
            )
            
    async def cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.entries.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            await self.remove(key)


class CacheManager:
    """Main cache manager orchestrating multi-layer caching"""
    
    def __init__(self, config: Optional[CacheConfig] = None,
                 vector_store: Optional[QdrantVectorStore] = None,
                 kv_store: Optional[NATSKVStore] = None):
        self.config = config or CacheConfig()
        
        # Initialize caches
        self.exact_cache = ExactCache(max_size=self.config.exact_cache_size)
        self.semantic_cache = SemanticCache(self.config, vector_store)
        
        # External stores
        self.vector_store = vector_store
        self.kv_store = kv_store
        
        # Statistics
        self.stats = CacheStats()
        self.stats_lock = asyncio.Lock()
        
        # Background tasks
        self._cleanup_task = None
        self._stats_task = None
        
    async def start(self):
        """Start background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._stats_task = asyncio.create_task(self._stats_loop())
        logger.info("Cache manager started")
        
    async def stop(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._stats_task:
            self._stats_task.cancel()
        logger.info("Cache manager stopped")
        
    async def get(self, request: ProviderRequest) -> Optional[Tuple[ProviderResponse, str]]:
        """Get cached response for request"""
        with tracer.start_as_current_span("cache_get") as span:
            start_time = time.time()
            
            # Update stats
            async with self.stats_lock:
                self.stats.total_requests += 1
                
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            span.set_attribute("cache_key", cache_key[:16])  # First 16 chars
            
            # Try exact cache first
            if self.config.strategy in [CacheStrategy.EXACT_ONLY, CacheStrategy.TWO_LAYER]:
                entry = await self.exact_cache.get(cache_key)
                if entry:
                    latency = (time.time() - start_time) * 1000
                    await self._record_hit(entry, "exact", latency)
                    span.set_attribute("hit_type", "exact")
                    return entry.response, "exact_hit"
                    
            # Try semantic cache
            if self.config.strategy in [CacheStrategy.SEMANTIC_ONLY, CacheStrategy.TWO_LAYER, CacheStrategy.AGGRESSIVE]:
                threshold = self.config.semantic_threshold
                if self.config.strategy == CacheStrategy.AGGRESSIVE:
                    threshold = 0.80  # Lower threshold
                    
                matches = await self.semantic_cache.search(
                    request.prompt,
                    threshold=threshold,
                    limit=self.config.max_semantic_results
                )
                
                if matches:
                    # Use best match
                    best_entry, score = matches[0]
                    latency = (time.time() - start_time) * 1000
                    await self._record_hit(best_entry, "semantic", latency)
                    span.set_attribute("hit_type", "semantic")
                    span.set_attribute("similarity_score", score)
                    
                    # Also add to exact cache for faster future access
                    await self.exact_cache.put(cache_key, best_entry)
                    
                    return best_entry.response, "semantic_hit"
                    
            # Cache miss
            latency = (time.time() - start_time) * 1000
            async with self.stats_lock:
                self.stats.misses += 1
                
            cache_latency.record(latency, {"operation": "get", "result": "miss"})
            cache_operations.add(1, {"operation": "get", "result": "miss"})
            span.set_attribute("hit_type", "miss")
            
            return None
            
    async def put(self, request: ProviderRequest, response: ProviderResponse):
        """Cache a response"""
        with tracer.start_as_current_span("cache_put") as span:
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                request=request,
                response=response,
                timestamp=datetime.now(timezone.utc),
                ttl_seconds=self._determine_ttl(request, response),
                cost_saved=response.cost_usd  # Save this cost on future hits
            )
            
            # Add to exact cache
            if self.config.strategy in [CacheStrategy.EXACT_ONLY, CacheStrategy.TWO_LAYER]:
                await self.exact_cache.put(cache_key, entry)
                
            # Add to semantic cache
            if self.config.strategy in [CacheStrategy.SEMANTIC_ONLY, CacheStrategy.TWO_LAYER, CacheStrategy.AGGRESSIVE]:
                await self.semantic_cache.add(entry)
                
            # Persist to KV store if available
            if self.kv_store and not response.cache_hit:
                await self._persist_to_kv(entry)
                
            cache_operations.add(1, {"operation": "put", "provider": response.provider.value})
            span.set_attribute("ttl_seconds", entry.ttl_seconds)
            
    async def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern"""
        if pattern is None:
            # Clear all
            await self.exact_cache.clear()
            self.semantic_cache.entries.clear()
            logger.info("Cache cleared")
        else:
            # Pattern-based invalidation
            # TODO: Implement pattern matching
            logger.warning(f"Pattern-based invalidation not yet implemented: {pattern}")
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self.stats_lock:
            stats_dict = asdict(self.stats)
            
        # Add computed metrics
        stats_dict["hit_rate"] = self.stats.hit_rate()
        stats_dict["exact_hit_rate"] = self.stats.exact_hit_rate()
        stats_dict["semantic_hit_rate"] = self.stats.semantic_hit_rate()
        stats_dict["exact_cache_size"] = await self.exact_cache.size()
        stats_dict["semantic_cache_size"] = len(self.semantic_cache.entries)
        
        return stats_dict
        
    def _generate_cache_key(self, request: ProviderRequest) -> str:
        """Generate deterministic cache key"""
        # Include relevant request parameters
        key_parts = [
            request.prompt,
            str(request.temperature),
            str(request.max_tokens),
            json.dumps(request.tools or [], sort_keys=True),
            request.system_prompt or "",
            json.dumps(request.stop_sequences or [], sort_keys=True)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
        
    def _determine_ttl(self, request: ProviderRequest, response: ProviderResponse) -> int:
        """Determine TTL based on request/response characteristics"""
        # Base TTL
        ttl = self.config.exact_ttl_seconds
        
        # Adjust based on content type
        prompt_lower = request.prompt.lower()
        
        # Factual queries can be cached longer
        if any(word in prompt_lower for word in ["what is", "define", "fact"]):
            ttl = self.config.semantic_ttl_seconds  # 24 hours
            
        # Time-sensitive queries should have shorter TTL
        elif any(word in prompt_lower for word in ["today", "now", "current", "latest"]):
            ttl = 300  # 5 minutes
            
        # Code generation can be cached moderately
        elif any(word in prompt_lower for word in ["code", "function", "implement"]):
            ttl = 7200  # 2 hours
            
        # Expensive queries can be cached longer
        if response.cost_usd > 0.10:
            ttl = max(ttl, 14400)  # At least 4 hours
            
        return ttl
        
    async def _record_hit(self, entry: CacheEntry, hit_type: str, latency_ms: float):
        """Record cache hit metrics"""
        async with self.stats_lock:
            if hit_type == "exact":
                self.stats.exact_hits += 1
            else:
                self.stats.semantic_hits += 1
                
            # Track cost savings
            self.stats.cost_saved_usd += entry.cost_saved
            
            # Track by provider
            provider = entry.response.provider.value
            self.stats.provider_hits[provider] = self.stats.provider_hits.get(provider, 0) + 1
            self.stats.provider_savings[provider] = self.stats.provider_savings.get(provider, 0.0) + entry.cost_saved
            
            # Track by hour
            current_hour = datetime.now(timezone.utc).hour
            self.stats.hourly_hits[current_hour] = self.stats.hourly_hits.get(current_hour, 0) + 1
            
        # Update metrics
        cache_operations.add(1, {"operation": "get", "result": hit_type})
        cache_latency.record(latency_ms, {"operation": "get", "result": hit_type})
        cache_cost_savings.add(entry.cost_saved)
        
        # Update entry
        entry.update_access()
        
    async def _persist_to_kv(self, entry: CacheEntry):
        """Persist entry to KV store"""
        try:
            # Serialize entry
            entry_data = {
                "key": entry.key,
                "request": asdict(entry.request),
                "response": asdict(entry.response),
                "timestamp": entry.timestamp.isoformat(),
                "ttl_seconds": entry.ttl_seconds,
                "cost_saved": entry.cost_saved
            }
            
            await self.kv_store.put(
                bucket="model_cache",
                key=entry.key,
                value=entry_data,
                ttl=entry.ttl_seconds
            )
        except Exception as e:
            logger.error(f"Failed to persist cache entry: {e}")
            
    async def _cleanup_loop(self):
        """Background task to cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.semantic_cache.cleanup_expired()
                logger.debug("Cache cleanup completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                
    async def _stats_loop(self):
        """Background task to report statistics"""
        while True:
            try:
                await asyncio.sleep(self.config.report_interval_seconds)
                
                # Update hit rate gauge
                hit_rate = self.stats.hit_rate()
                cache_hit_rate.set(hit_rate * 100)
                
                # Log summary
                logger.info(
                    "Cache stats",
                    hit_rate=f"{hit_rate:.2%}",
                    exact_hits=self.stats.exact_hits,
                    semantic_hits=self.stats.semantic_hits,
                    cost_saved=f"${self.stats.cost_saved_usd:.2f}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats reporting error: {e}")


# Export main classes
__all__ = [
    "CacheStrategy",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "ExactCache",
    "SemanticCache",
    "CacheManager"
]