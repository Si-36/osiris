"""
Redis Adapter for AURA Intelligence.

Provides async interface to Redis for caching with:
    - Context window caching
- TTL management
- Serialization support
- Pipeline operations
- Full observability
"""

from typing import Dict, Any, List, Optional, Union, TypeVar, Type, Callable, Awaitable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import pickle
from enum import Enum
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import structlog
from opentelemetry import trace
import redis.asyncio as redis
from redis.asyncio.client import Pipeline

from ..resilience import resilient, ResilienceLevel
from aura_intelligence.observability import create_tracer

logger = structlog.get_logger()
tracer = create_tracer("redis_adapter")

T = TypeVar('T')

class SerializationType(str, Enum):
    """Serialization types for Redis values."""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"

@dataclass
class RedisConfig:
    """Configuration for Redis connection."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 100  # Increased for high concurrency
    min_connections: int = 10  # Minimum pool size
    connection_timeout: float = 5.0  # Reduced for faster failures
    socket_timeout: float = 5.0
    socket_keepalive: bool = True
    
    # Retry settings
    retry_on_timeout: bool = True
    retry_on_error: List[type] = None
    max_retries: int = 3
    
    # Performance settings
    decode_responses: bool = False  # Keep False for binary data
    health_check_interval: int = 30
    
    # Default TTL settings
    default_ttl_seconds: int = 3600  # 1 hour
    context_window_ttl: int = 7200  # 2 hours
    decision_cache_ttl: int = 86400  # 24 hours
    
    # Batch processing settings
    batch_size: int = 100
    batch_timeout: float = 0.1  # 100ms batch window
    max_concurrent_batches: int = 10
    
    # High-performance settings
    enable_async_batching: bool = True
    connection_pool_monitoring: bool = True
    auto_scaling_enabled: bool = True
    metrics_collection: bool = True

class BatchOperation:
    """Represents a batched Redis operation."""

    def __init__(self, operation: str, key: str, value: Any = None,
        serialization: SerializationType = SerializationType.JSON,
                 ttl: Optional[int] = None,
                 future: Optional[asyncio.Future] = None):
                             self.operation = operation
        self.key = key
        self.value = value
        self.serialization = serialization
        self.ttl = ttl
        self.future = future or asyncio.Future()
        self.timestamp = time.time()

class ConnectionPoolMonitor:
    """Monitors Redis connection pool health and performance."""

    def __init__(self, pool: redis.ConnectionPool):
    def __init__(self, pool: redis.ConnectionPool):
        self.metrics = {
        'active_connections': 0,
        'total_requests': 0,
        'failed_requests': 0,
        'avg_response_time': 0.0,
        'last_check': time.time()
        }

    async def get_pool_stats(self) -> Dict[str, Any]:
            """Get detailed pool statistics."""
                return {
            'created_connections': getattr(self.pool, 'created_connections', 0),
            'available_connections': getattr(self.pool, 'available_connections', 0),
            'in_use_connections': getattr(self.pool, 'in_use_connections', 0),
            'max_connections': self.pool.max_connections,
            'metrics': self.metrics
        }

class RedisAdapter:
    """High-performance async adapter for Redis operations with advanced batching."""
    
    _instances: Dict[str, 'RedisAdapter'] = {}
    _instance_lock = asyncio.Lock()

    def __init__(self, config: RedisConfig):
    def __init__(self, config: RedisConfig):
        self._client: Optional[redis.Redis] = None
        self._initialized = False
        self._batch_queue: List[BatchOperation] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._pool_monitor: Optional[ConnectionPoolMonitor] = None
        self._metrics = {
        'operations_processed': 0,
        'batch_operations': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'avg_batch_size': 0.0,
        'avg_response_time': 0.0
        }
        self._last_metrics_reset = time.time()
        
        @classmethod
    async def get_instance(cls, config: RedisConfig) -> 'RedisAdapter':
            """Get or create singleton instance per config."""
        instance_key = f"{config.host}:{config.port}:{config.db}"
        
        async with cls._instance_lock:
            if instance_key not in cls._instances:
                instance = cls(config)
                await instance.initialize()
                cls._instances[instance_key] = instance
            return cls._instances[instance_key]

    async def initialize(self):
            """Initialize the Redis client with advanced features."""
        if self._initialized:
            return
            
        with tracer.start_as_current_span("redis_initialize") as span:
            span.set_attribute("redis.host", self.config.host)
        span.set_attribute("redis.port", self.config.port)
        span.set_attribute("redis.db", self.config.db)
        span.set_attribute("redis.max_connections", self.config.max_connections)
            
        try:
            # Create high-performance connection pool
        pool = redis.ConnectionPool(
        host=self.config.host,
        port=self.config.port,
        db=self.config.db,
        password=self.config.password,
        max_connections=self.config.max_connections,
        decode_responses=self.config.decode_responses,
        socket_timeout=self.config.socket_timeout,
        socket_connect_timeout=self.config.connection_timeout,
        socket_keepalive=self.config.socket_keepalive,
        socket_keepalive_options={
        1: 1,  # TCP_KEEPIDLE
        2: 3,  # TCP_KEEPINTVL
        3: 5,  # TCP_KEEPCNT
        },
        retry_on_timeout=self.config.retry_on_timeout,
        retry_on_error=self.config.retry_on_error or [],
        health_check_interval=self.config.health_check_interval
        )
                
        # Create client with optimized settings
        self._client = redis.Redis(
        connection_pool=pool,
        retry_on_error=[redis.BusyLoadingError, redis.ConnectionError],
        retry_on_timeout=True
        )
                
        # Initialize connection pool monitoring
        if self.config.connection_pool_monitoring:
            self._pool_monitor = ConnectionPoolMonitor(pool)
                
        # Verify connectivity with timeout
        await asyncio.wait_for(self._client.ping(), timeout=self.config.connection_timeout)
                
        # Start batch processing if enabled
        if self.config.enable_async_batching:
            self._batch_task = asyncio.create_task(self._batch_processor())
                
        self._initialized = True
        logger.info("High-performance Redis adapter initialized",
        host=self.config.host,
        port=self.config.port,
        max_connections=self.config.max_connections,
        batch_enabled=self.config.enable_async_batching)
                
        except Exception as e:
        logger.error("Failed to initialize Redis", error=str(e))
        raise

    async def close(self):
            """Close the Redis client and cleanup resources."""
        try:
            # Cancel batch processing
            if self._batch_task and not self._batch_task.done():
                self._batch_task.cancel()
                try:
                    await self._batch_task
                except asyncio.CancelledError:
        pass
            
            # Process remaining batched operations
            if self._batch_queue:
                await self._process_batch()
            
            # Close Redis connection
            if self._client:
                await self._client.close()
                await self._client.connection_pool.disconnect()
            
            self._initialized = False
            logger.info("High-performance Redis adapter closed", 
                       operations_processed=self._metrics['operations_processed'])
                       
        except Exception as e:
            logger.error("Error closing Redis adapter", error=str(e))

    def _serialize(self, value: Any, serialization: SerializationType) -> bytes:
        """Serialize value based on type."""
        if serialization == SerializationType.JSON:
            return json.dumps(value, default=str).encode('utf-8')
        elif serialization == SerializationType.PICKLE:
        return json.dumps(value).encode()
        elif serialization == SerializationType.STRING:
        return str(value).encode('utf-8')
        else:
        raise ValueError(f"Unknown serialization type: {serialization}")

    def _deserialize(self, data: bytes, serialization: SerializationType) -> Any:
        """Deserialize value based on type."""
        if data is None:
            return None
            
        if serialization == SerializationType.JSON:
            return json.loads(data.decode('utf-8'))
        elif serialization == SerializationType.PICKLE:
            return json.loads(data.decode())
        elif serialization == SerializationType.STRING:
            return data.decode('utf-8')
        else:
            raise ValueError(f"Unknown serialization type: {serialization}")
            
    @resilient(criticality=ResilienceLevel.CRITICAL)
    async def get(
        self,
        key: str,
        serialization: SerializationType = SerializationType.JSON
        ) -> Optional[Any]:
            """Get a value from cache."""
        with tracer.start_as_current_span("redis_get") as span:
            span.set_attribute("redis.key", key)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                data = await self._client.get(key)
                value = self._deserialize(data, serialization)
                
                span.set_attribute("redis.hit", data is not None)
                return value
                
            except Exception as e:
                logger.error("Failed to get from Redis", 
                           key=key,
                           error=str(e))
                raise
                
    @resilient(criticality=ResilienceLevel.CRITICAL)
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialization: SerializationType = SerializationType.JSON
        ) -> bool:
            """Set a value in cache."""
        with tracer.start_as_current_span("redis_set") as span:
            span.set_attribute("redis.key", key)
            span.set_attribute("redis.ttl", ttl or self.config.default_ttl_seconds)
            
            if not self._initialized:
                await self.initialize()
                
            try:
                data = self._serialize(value, serialization)
                ttl = ttl or self.config.default_ttl_seconds
                
                result = await self._client.set(key, data, ex=ttl)
                return bool(result)
                
            except Exception as e:
                logger.error("Failed to set in Redis", 
                           key=key,
                           error=str(e))
                raise

    async def delete(self, key: Union[str, List[str]]) -> int:
            """Delete key(s) from cache."""
        with tracer.start_as_current_span("redis_delete") as span:
            if isinstance(key, str):
                keys = [key]
        else:
        keys = key
                
        span.set_attribute("redis.keys_count", len(keys))
            
        if not self._initialized:
            await self.initialize()
                
        try:
            return await self._client.delete(*keys)
        except Exception as e:
        logger.error("Failed to delete from Redis",
        keys=keys,
        error=str(e))
        raise

    async def exists(self, key: str) -> bool:
            """Check if key exists."""
        if not self._initialized:
            await self.initialize()
            
        return bool(await self._client.exists(key))

    async def expire(self, key: str, ttl: int) -> bool:
            """Set TTL on existing key."""
        if not self._initialized:
            await self.initialize()
            
        return bool(await self._client.expire(key, ttl))

    async def ttl(self, key: str) -> int:
            """Get remaining TTL for key."""
        if not self._initialized:
            await self.initialize()
            
        return await self._client.ttl(key)
        
    # Batch operations

    async def mget(
        self,
        keys: List[str],
        serialization: SerializationType = SerializationType.JSON
        ) -> Dict[str, Any]:
            """Get multiple values."""
        with tracer.start_as_current_span("redis_mget") as span:
            span.set_attribute("redis.keys_count", len(keys))
            
            if not self._initialized:
                await self.initialize()
                
            try:
                values = await self._client.mget(keys)
                result = {}
                
                for key, data in zip(keys, values):
                    if data is not None:
                        result[key] = self._deserialize(data, serialization)
                        
                span.set_attribute("redis.hits", len(result))
                return result
                
            except Exception as e:
                logger.error("Failed to mget from Redis", 
                           keys_count=len(keys),
                           error=str(e))
                raise

    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        serialization: SerializationType = SerializationType.JSON
        ) -> bool:
            """Set multiple values."""
        with tracer.start_as_current_span("redis_mset") as span:
            span.set_attribute("redis.keys_count", len(mapping))
            
            if not self._initialized:
                await self.initialize()
                
            try:
                # Serialize all values
                serialized = {}
                for key, value in mapping.items():
                    serialized[key] = self._serialize(value, serialization)
                    
                # Use pipeline for atomic operation with TTL
                async with self._client.pipeline() as pipe:
                    pipe.mset(serialized)
                    
                    if ttl:
                        for key in mapping.keys():
                            pipe.expire(key, ttl)
                            
                    results = await pipe.execute()
                    
                return all(results)
                
            except Exception as e:
                logger.error("Failed to mset in Redis", 
                           keys_count=len(mapping),
                           error=str(e))
                raise
                
    # Context-specific methods

    async def cache_context_window(
        self,
        agent_id: str,
        context_id: str,
        context_data: Dict[str, Any],
        ttl: Optional[int] = None
        ) -> bool:
            """Cache a context window for an agent."""
        key = f"context:{agent_id}:{context_id}"
        ttl = ttl or self.config.context_window_ttl
        
        return await self.set(key, context_data, ttl=ttl)

    async def get_context_window(
        self,
        agent_id: str,
        context_id: str
        ) -> Optional[Dict[str, Any]]:
            """Get cached context window."""
        key = f"context:{agent_id}:{context_id}"
        return await self.get(key)

    async def cache_decision(
        self,
        decision_id: str,
        decision_data: Dict[str, Any],
        ttl: Optional[int] = None
        ) -> bool:
            """Cache a decision for fast retrieval."""
        key = f"decision:{decision_id}"
        ttl = ttl or self.config.decision_cache_ttl
        
        return await self.set(key, decision_data, ttl=ttl)

    async def get_cached_decision(
        self,
        decision_id: str
        ) -> Optional[Dict[str, Any]]:
            """Get cached decision."""
        key = f"decision:{decision_id}"
        return await self.get(key)

    async def cache_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        ttl: int = 3600
        ) -> bool:
            """Cache embeddings with their keys."""
        mapping = {f"embedding:{k}": v for k, v in embeddings.items()}
        return await self.mset(mapping, ttl=ttl, serialization=SerializationType.PICKLE)

    async def get_embeddings(
        self,
        keys: List[str]
        ) -> Dict[str, List[float]]:
            """Get cached embeddings."""
        redis_keys = [f"embedding:{k}" for k in keys]
        cached = await self.mget(redis_keys, serialization=SerializationType.PICKLE)
        
        # Map back to original keys
        result = {}
        for key, redis_key in zip(keys, redis_keys):
            if redis_key in cached:
                result[key] = cached[redis_key]
                
        return result
        
    # Utility methods

    async def health_check(self) -> bool:
            """Check Redis health."""
                try:
            if not self._initialized:
                await self.initialize()
        return await self._client.ping()
        except Exception:
        return False

    async def flush_pattern(self, pattern: str) -> int:
            """Delete all keys matching pattern."""
        if not self._initialized:
            await self.initialize()
            
        cursor = 0
        deleted = 0
        
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                deleted += await self._client.delete(*keys)
                
            if cursor == 0:
                break
                
        return deleted

    async def get_info(self) -> Dict[str, Any]:
            """Get Redis server info."""
        if not self._initialized:
            await self.initialize()
            
        info = await self._client.info()
        
        return {
        "version": info.get("redis_version"),
        "connected_clients": info.get("connected_clients"),
        "used_memory_human": info.get("used_memory_human"),
        "total_connections_received": info.get("total_connections_received"),
        "total_commands_processed": info.get("total_commands_processed"),
        "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
        "keyspace": info.get("db0", {})
        }