"""
NATS JetStream KV Store
======================
Ultra-low latency key-value store using NATS JetStream KV
with mirrored buckets for <1ms config lookups.

Based on Discord's implementation achieving 50μs p50 latency.
"""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import nats
from nats.js import JetStreamContext
from nats.js.kv import KeyValue

from ..core import (
    AbstractStore,
    StoreType,
    QueryResult,
    WriteResult,
    TransactionContext,
    ConnectionConfig,
    StoreCircuitBreaker
)

logger = logging.getLogger(__name__)


@dataclass
class KVConfig(ConnectionConfig):
    """Configuration for NATS KV store"""
    # NATS settings
    servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    bucket_name: str = "aura-config"
    
    # Replication
    replicas: int = 3
    mirror_buckets: List[str] = field(default_factory=list)
    
    # Performance
    ttl_seconds: Optional[int] = None  # Default no TTL
    max_value_size: int = 1048576  # 1MB
    history: int = 10  # Keep last N versions
    
    # Caching
    enable_local_cache: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: int = 60
    
    # Watch settings
    enable_watch: bool = True
    watch_prefix: Optional[str] = None
    
    # Leader election
    enable_leader_election: bool = False
    leader_ttl_seconds: int = 10


@dataclass
class KVEntry:
    """Key-value entry with metadata"""
    key: str
    value: Any
    revision: int
    created: datetime
    updated: datetime
    operation: str  # PUT, DELETE, PURGE
    delta: Optional[int] = None  # For counters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'key': self.key,
            'value': self.value,
            'revision': self.revision,
            'created': self.created.isoformat(),
            'updated': self.updated.isoformat(),
            'operation': self.operation,
            'delta': self.delta
        }


class LocalCache:
    """Thread-safe local cache with TTL"""
    
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
        self._access_order: List[str] = []
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Check TTL
                if (datetime.utcnow() - timestamp).total_seconds() < self.ttl_seconds:
                    # Move to end (LRU)
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    self._access_order.remove(key)
                    
        return None
        
    async def put(self, key: str, value: Any):
        """Put value in cache"""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
                
            # Update cache
            self._cache[key] = (value, datetime.utcnow())
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
    async def invalidate(self, key: str):
        """Invalidate cache entry"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                
    async def clear(self):
        """Clear entire cache"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()


class NATSKVStore(AbstractStore[str, Any]):
    """
    NATS JetStream KV implementation for ultra-low latency config storage.
    Provides <1ms lookups with multi-region replication.
    """
    
    def __init__(self, config: Optional[KVConfig] = None):
        config = config or KVConfig()
        super().__init__(StoreType.KV, config.__dict__)
        
        self.kv_config = config
        self._nc: Optional[nats.NATS] = None
        self._js: Optional[JetStreamContext] = None
        self._kv: Optional[KeyValue] = None
        
        # Local cache
        self._cache: Optional[LocalCache] = None
        if config.enable_local_cache:
            self._cache = LocalCache(config.cache_size, config.cache_ttl_seconds)
            
        # Watchers
        self._watchers: Dict[str, asyncio.Task] = {}
        self._watch_callbacks: Dict[str, List[callable]] = {}
        
        # Circuit breaker
        self._circuit_breaker = StoreCircuitBreaker(
            f"nats_kv_{config.bucket_name}",
            fallback_func=self._fallback_get
        )
        
        # Metrics
        self._metrics = {
            'gets': 0,
            'puts': 0,
            'deletes': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def initialize(self) -> None:
        """Initialize NATS connection and KV bucket"""
        try:
            # Connect to NATS
            self._nc = await nats.connect(
                servers=self.kv_config.servers,
                max_reconnect_attempts=10,
                reconnect_time_wait=2
            )
            
            # Get JetStream context
            self._js = self._nc.jetstream()
            
            # Create or get KV bucket
            bucket_config = nats.js.api.KeyValueConfig(
                bucket=self.kv_config.bucket_name,
                replicas=self.kv_config.replicas,
                ttl=self.kv_config.ttl_seconds,
                max_value_size=self.kv_config.max_value_size,
                history=self.kv_config.history
            )
            
            try:
                self._kv = await self._js.create_key_value(bucket_config)
            except Exception:
                # Bucket might already exist
                self._kv = await self._js.key_value(self.kv_config.bucket_name)
                
            # Set up mirrors
            for mirror_bucket in self.kv_config.mirror_buckets:
                await self._setup_mirror(mirror_bucket)
                
            # Start watchers if enabled
            if self.kv_config.enable_watch:
                await self._start_watchers()
                
            self._initialized = True
            logger.info(f"NATS KV store initialized: {self.kv_config.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NATS KV: {e}")
            raise
            
    async def _setup_mirror(self, mirror_bucket: str):
        """Set up bucket mirroring for multi-region"""
        try:
            mirror_config = nats.js.api.StreamConfig(
                name=f"{self.kv_config.bucket_name}_mirror_{mirror_bucket}",
                mirror=nats.js.api.StreamSource(
                    name=f"KV_{self.kv_config.bucket_name}",
                    external=nats.js.api.ExternalStream(
                        api_prefix=f"$JS.{mirror_bucket}.API",
                        deliver_prefix=f"$JS.{mirror_bucket}.DELIVER"
                    )
                )
            )
            
            await self._js.add_stream(mirror_config)
            logger.info(f"Set up mirror to {mirror_bucket}")
            
        except Exception as e:
            logger.warning(f"Failed to set up mirror to {mirror_bucket}: {e}")
            
    async def health_check(self) -> Dict[str, Any]:
        """Check NATS connection health"""
        if not self._nc or self._nc.is_closed:
            return {'healthy': False, 'reason': 'Not connected'}
            
        try:
            # Try a simple operation
            await asyncio.wait_for(
                self._kv.status(),
                timeout=1.0
            )
            
            return {
                'healthy': True,
                'connected': not self._nc.is_closed,
                'metrics': self._metrics
            }
            
        except Exception as e:
            return {'healthy': False, 'reason': str(e)}
            
    async def close(self) -> None:
        """Close NATS connection"""
        # Stop watchers
        for task in self._watchers.values():
            task.cancel()
            
        # Close connection
        if self._nc:
            await self._nc.close()
            
        self._initialized = False
        logger.info("NATS KV store closed")
        
    # Core KV operations
    
    async def upsert(self, 
                    key: str,
                    value: Any,
                    context: Optional[TransactionContext] = None) -> WriteResult:
        """Put key-value pair"""
        return await self._circuit_breaker.call(self._upsert, key, value, context)
        
    async def _upsert(self,
                     key: str, 
                     value: Any,
                     context: Optional[TransactionContext] = None) -> WriteResult:
        """Internal upsert implementation"""
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                data = json.dumps(value).encode()
            elif isinstance(value, str):
                data = value.encode()
            elif isinstance(value, bytes):
                data = value
            else:
                data = str(value).encode()
                
            # Put to KV
            revision = await self._kv.put(key, data)
            
            # Update cache
            if self._cache:
                await self._cache.put(key, value)
                
            # Update metrics
            self._metrics['puts'] += 1
            
            return WriteResult(
                success=True,
                id=key,
                version=revision,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to put {key}: {e}")
            return WriteResult(
                success=False,
                error=str(e)
            )
            
    async def get(self,
                  key: str,
                  context: Optional[TransactionContext] = None) -> Optional[Any]:
        """Get value by key"""
        return await self._circuit_breaker.call(self._get, key, context)
        
    async def _get(self,
                   key: str,
                   context: Optional[TransactionContext] = None) -> Optional[Any]:
        """Internal get implementation"""
        # Check cache first
        if self._cache:
            cached = await self._cache.get(key)
            if cached is not None:
                self._metrics['cache_hits'] += 1
                return cached
            else:
                self._metrics['cache_misses'] += 1
                
        try:
            # Get from KV
            entry = await self._kv.get(key)
            
            if entry is None:
                return None
                
            # Deserialize
            value = self._deserialize_value(entry.value)
            
            # Update cache
            if self._cache:
                await self._cache.put(key, value)
                
            # Update metrics
            self._metrics['gets'] += 1
            
            return value
            
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            logger.error(f"Failed to get {key}: {e}")
            raise
            
    async def _fallback_get(self, key: str, context: Optional[TransactionContext] = None) -> Optional[Any]:
        """Fallback when circuit is open - use cache only"""
        if self._cache:
            return await self._cache.get(key)
        return None
        
    async def list(self,
                   filter_dict: Optional[Dict[str, Any]] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None,
                   context: Optional[TransactionContext] = None) -> QueryResult[Any]:
        """List keys with optional prefix filter"""
        try:
            # Get prefix from filter
            prefix = filter_dict.get('prefix', '') if filter_dict else ''
            
            # List keys
            keys = await self._kv.keys(prefix=prefix if prefix else None)
            
            # Apply cursor-based pagination
            start_idx = 0
            if cursor:
                try:
                    start_idx = int(cursor)
                except ValueError:
                    pass
                    
            # Get subset
            key_subset = list(keys)[start_idx:start_idx + limit]
            
            # Fetch values
            data = []
            for key in key_subset:
                value = await self.get(key)
                if value is not None:
                    data.append({
                        'key': key,
                        'value': value
                    })
                    
            # Determine next cursor
            next_cursor = None
            if len(keys) > start_idx + limit:
                next_cursor = str(start_idx + limit)
                
            return QueryResult(
                success=True,
                data=data,
                total_count=len(keys),
                next_cursor=next_cursor
            )
            
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return QueryResult(
                success=False,
                error=str(e)
            )
            
    async def delete(self,
                     key: str,
                     context: Optional[TransactionContext] = None) -> WriteResult:
        """Delete a key"""
        try:
            # Delete from KV
            await self._kv.delete(key)
            
            # Invalidate cache
            if self._cache:
                await self._cache.invalidate(key)
                
            # Update metrics
            self._metrics['deletes'] += 1
            
            return WriteResult(
                success=True,
                id=key,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return WriteResult(
                success=False,
                error=str(e)
            )
            
    # Batch operations
    
    async def batch_upsert(self,
                          items: List[Tuple[str, Any]],
                          context: Optional[TransactionContext] = None) -> List[WriteResult]:
        """Batch put multiple key-value pairs"""
        results = []
        
        # NATS doesn't have native batch put, so we do it sequentially
        # but could parallelize with asyncio.gather
        tasks = [self.upsert(key, value, context) for key, value in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(WriteResult(
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)
                
        return final_results
        
    async def batch_get(self,
                       keys: List[str],
                       context: Optional[TransactionContext] = None) -> Dict[str, Optional[Any]]:
        """Batch get multiple values"""
        # Parallelize gets
        tasks = [self.get(key, context) for key in keys]
        values = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dict
        result = {}
        for key, value in zip(keys, values):
            if isinstance(value, Exception):
                result[key] = None
            else:
                result[key] = value
                
        return result
        
    # Watch operations
    
    async def watch(self, 
                   key_or_prefix: str,
                   callback: callable,
                   is_prefix: bool = False) -> str:
        """Watch for changes to a key or prefix"""
        watch_id = f"watch_{key_or_prefix}_{id(callback)}"
        
        # Register callback
        if key_or_prefix not in self._watch_callbacks:
            self._watch_callbacks[key_or_prefix] = []
        self._watch_callbacks[key_or_prefix].append(callback)
        
        # Start watcher if not already running
        if key_or_prefix not in self._watchers:
            task = asyncio.create_task(
                self._watch_loop(key_or_prefix, is_prefix)
            )
            self._watchers[key_or_prefix] = task
            
        return watch_id
        
    async def _watch_loop(self, key_or_prefix: str, is_prefix: bool):
        """Watch loop for key changes"""
        try:
            # Create watcher
            if is_prefix:
                watcher = await self._kv.watch(key_or_prefix + "*")
            else:
                watcher = await self._kv.watch(key_or_prefix)
                
            # Process updates
            async for update in watcher:
                if update is None:
                    continue
                    
                # Deserialize value
                value = self._deserialize_value(update.value) if update.value else None
                
                # Create entry
                entry = KVEntry(
                    key=update.key,
                    value=value,
                    revision=update.revision,
                    created=datetime.utcnow(),  # Would use actual timestamps
                    updated=datetime.utcnow(),
                    operation=update.operation,
                    delta=update.delta
                )
                
                # Invalidate cache
                if self._cache:
                    await self._cache.invalidate(update.key)
                    
                # Call callbacks
                callbacks = self._watch_callbacks.get(key_or_prefix, [])
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(entry)
                        else:
                            callback(entry)
                    except Exception as e:
                        logger.error(f"Watch callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Watch loop error for {key_or_prefix}: {e}")
            
    async def unwatch(self, watch_id: str):
        """Stop watching"""
        # Extract key from watch_id
        parts = watch_id.split('_')
        if len(parts) >= 2:
            key = parts[1]
            
            # Cancel watcher if no more callbacks
            if key in self._watch_callbacks:
                self._watch_callbacks[key] = []
                
                if not self._watch_callbacks[key] and key in self._watchers:
                    self._watchers[key].cancel()
                    del self._watchers[key]
                    
    # Leader election
    
    async def acquire_leader(self, 
                           election_key: str,
                           leader_id: str,
                           ttl_seconds: Optional[int] = None) -> bool:
        """Try to acquire leadership for a key"""
        ttl = ttl_seconds or self.kv_config.leader_ttl_seconds
        
        try:
            # Try to create key with our ID
            current = await self._kv.get(election_key)
            
            if current is None:
                # No leader, try to become one
                await self._kv.create(election_key, leader_id.encode())
                return True
            else:
                # Check if current leader is us
                current_leader = current.value.decode()
                return current_leader == leader_id
                
        except Exception as e:
            if "key exists" in str(e).lower():
                return False
            raise
            
    async def renew_leader(self, election_key: str, leader_id: str) -> bool:
        """Renew leadership lease"""
        try:
            current = await self._kv.get(election_key)
            
            if current and current.value.decode() == leader_id:
                # We're still the leader, update to renew
                await self._kv.put(election_key, leader_id.encode())
                return True
                
            return False
            
        except Exception:
            return False
            
    async def release_leader(self, election_key: str, leader_id: str) -> bool:
        """Release leadership"""
        try:
            current = await self._kv.get(election_key)
            
            if current and current.value.decode() == leader_id:
                await self._kv.delete(election_key)
                return True
                
            return False
            
        except Exception:
            return False
            
    # Utility methods
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from bytes"""
        try:
            # Try JSON first
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Try string
            try:
                return data.decode()
            except UnicodeDecodeError:
                # Return as bytes
                return data
                
    async def get_metrics(self) -> Dict[str, Any]:
        """Get store metrics"""
        base_metrics = await super().get_metrics()
        
        # Add KV-specific metrics
        base_metrics.update({
            'kv_metrics': self._metrics,
            'cache_enabled': self._cache is not None,
            'watchers_active': len(self._watchers),
            'circuit_breaker': self._circuit_breaker.get_status()
        })
        
        return base_metrics