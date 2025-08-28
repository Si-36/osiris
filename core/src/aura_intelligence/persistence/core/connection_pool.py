"""
Connection Pool Manager - Enterprise-Grade Resource Management
=============================================================
Manages connection pooling with health checks, retry logic,
and automatic recovery for all persistence backends.
"""

import asyncio
from typing import Dict, Any, Optional, Protocol, TypeVar, Generic, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import time
from abc import ABC, abstractmethod
from collections import deque
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ConnectionConfig:
    """Configuration for connection pooling"""
    # Pool settings
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    
    # Health check settings
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    unhealthy_threshold: int = 3
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.1
    retry_backoff: float = 2.0
    
    # Connection settings
    connection_params: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring
    enable_metrics: bool = True
    slow_query_threshold: float = 1.0  # seconds


@dataclass
class ConnectionStats:
    """Statistics for a pooled connection"""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: datetime = field(default_factory=datetime.utcnow)
    total_queries: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    health_check_failures: int = 0
    is_healthy: bool = True
    
    def record_success(self):
        """Record successful operation"""
        self.last_used_at = datetime.utcnow()
        self.total_queries += 1
        self.consecutive_errors = 0
        
    def record_error(self):
        """Record failed operation"""
        self.last_used_at = datetime.utcnow()
        self.total_errors += 1
        self.consecutive_errors += 1
        
    def is_idle(self, idle_timeout: float) -> bool:
        """Check if connection is idle"""
        idle_time = (datetime.utcnow() - self.last_used_at).total_seconds()
        return idle_time > idle_timeout


class Connection(Protocol[T]):
    """Protocol for database connections"""
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> T:
        """Execute a query"""
        ...
        
    async def health_check(self) -> bool:
        """Check connection health"""
        ...
        
    async def close(self) -> None:
        """Close the connection"""
        ...


class PooledConnection(Generic[T]):
    """Wrapper for pooled connections with statistics"""
    
    def __init__(self, connection: Connection[T], pool: 'ConnectionPool', conn_id: str):
        self.connection = connection
        self.pool = pool
        self.conn_id = conn_id
        self.stats = ConnectionStats()
        self._in_use = False
        
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> T:
        """Execute query with stats tracking"""
        start_time = time.perf_counter()
        
        try:
            result = await self.connection.execute(query, params)
            self.stats.record_success()
            
            # Track slow queries
            elapsed = time.perf_counter() - start_time
            if elapsed > self.pool.config.slow_query_threshold:
                logger.warning(f"Slow query detected: {elapsed:.2f}s - {query[:100]}...")
                
            return result
            
        except Exception as e:
            self.stats.record_error()
            
            # Check if connection should be marked unhealthy
            if self.stats.consecutive_errors >= self.pool.config.unhealthy_threshold:
                self.stats.is_healthy = False
                logger.error(f"Connection {self.conn_id} marked unhealthy after {self.stats.consecutive_errors} errors")
                
            raise
            
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            result = await asyncio.wait_for(
                self.connection.health_check(),
                timeout=self.pool.config.health_check_timeout
            )
            
            if result:
                self.stats.health_check_failures = 0
                self.stats.is_healthy = True
            else:
                self.stats.health_check_failures += 1
                
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for connection {self.conn_id}")
            self.stats.health_check_failures += 1
            return False
            
        except Exception as e:
            logger.error(f"Health check error for connection {self.conn_id}: {e}")
            self.stats.health_check_failures += 1
            return False
            
    async def close(self) -> None:
        """Close the connection"""
        try:
            await self.connection.close()
        except Exception as e:
            logger.error(f"Error closing connection {self.conn_id}: {e}")
            
    def acquire(self):
        """Mark connection as in use"""
        self._in_use = True
        
    def release(self):
        """Mark connection as available"""
        self._in_use = False
        
    @property
    def is_available(self) -> bool:
        """Check if connection is available for use"""
        return not self._in_use and self.stats.is_healthy


class ConnectionFactory(ABC, Generic[T]):
    """Factory for creating connections"""
    
    @abstractmethod
    async def create_connection(self, config: ConnectionConfig) -> Connection[T]:
        """Create a new connection"""
        pass


class ConnectionPool(Generic[T]):
    """
    Enterprise-grade connection pool with health checks and auto-recovery.
    Supports any backend that implements the Connection protocol.
    """
    
    def __init__(self, 
                 factory: ConnectionFactory[T],
                 config: ConnectionConfig):
        self.factory = factory
        self.config = config
        
        # Connection tracking
        self._connections: Dict[str, PooledConnection[T]] = {}
        self._available: deque[str] = deque()
        self._conn_counter = 0
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._closing = False
        
        # Metrics
        self._total_acquired = 0
        self._total_released = 0
        self._total_created = 0
        self._total_destroyed = 0
        
        logger.info(f"Connection pool initialized: min={config.min_connections}, max={config.max_connections}")
        
    async def initialize(self) -> None:
        """Initialize the pool with minimum connections"""
        async with self._lock:
            # Create minimum connections
            for _ in range(self.config.min_connections):
                await self._create_connection()
                
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
        logger.info(f"Connection pool initialized with {len(self._connections)} connections")
        
    async def _create_connection(self) -> str:
        """Create a new connection (must be called with lock held)"""
        conn_id = f"conn_{self._conn_counter}"
        self._conn_counter += 1
        
        try:
            raw_conn = await self.factory.create_connection(self.config)
            pooled_conn = PooledConnection(raw_conn, self, conn_id)
            
            self._connections[conn_id] = pooled_conn
            self._available.append(conn_id)
            self._total_created += 1
            
            logger.debug(f"Created connection {conn_id}")
            return conn_id
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
            
    async def _destroy_connection(self, conn_id: str) -> None:
        """Destroy a connection (must be called with lock held)"""
        if conn_id not in self._connections:
            return
            
        conn = self._connections[conn_id]
        
        try:
            await conn.close()
        except Exception as e:
            logger.error(f"Error closing connection {conn_id}: {e}")
            
        del self._connections[conn_id]
        
        # Remove from available queue
        try:
            self._available.remove(conn_id)
        except ValueError:
            pass
            
        self._total_destroyed += 1
        logger.debug(f"Destroyed connection {conn_id}")
        
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        start_time = time.perf_counter()
        conn_id = None
        
        try:
            async with self._not_empty:
                # Wait for available connection
                while True:
                    # Check for available healthy connections
                    async with self._lock:
                        for cid in list(self._available):
                            conn = self._connections[cid]
                            if conn.is_available:
                                self._available.remove(cid)
                                conn.acquire()
                                conn_id = cid
                                self._total_acquired += 1
                                break
                                
                    if conn_id:
                        break
                        
                    # Try to create new connection if under limit
                    async with self._lock:
                        if len(self._connections) < self.config.max_connections:
                            try:
                                conn_id = await self._create_connection()
                                self._available.remove(conn_id)
                                self._connections[conn_id].acquire()
                                self._total_acquired += 1
                                break
                            except Exception as e:
                                logger.error(f"Failed to create new connection: {e}")
                                
                    # Wait for connection to become available
                    wait_time = self.config.connection_timeout - (time.perf_counter() - start_time)
                    if wait_time <= 0:
                        raise asyncio.TimeoutError("Connection pool timeout")
                        
                    try:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=wait_time)
                    except asyncio.TimeoutError:
                        raise asyncio.TimeoutError(f"Timeout acquiring connection after {self.config.connection_timeout}s")
                        
            # Return the connection
            yield self._connections[conn_id]
            
        finally:
            # Release connection back to pool
            if conn_id and conn_id in self._connections:
                async with self._lock:
                    conn = self._connections[conn_id]
                    conn.release()
                    
                    # Only return to pool if healthy
                    if conn.stats.is_healthy and not conn.stats.is_idle(self.config.idle_timeout):
                        self._available.append(conn_id)
                        self._total_released += 1
                    else:
                        # Destroy unhealthy or idle connections
                        await self._destroy_connection(conn_id)
                        
                # Notify waiters
                async with self._not_empty:
                    self._not_empty.notify()
                    
    async def _health_check_loop(self) -> None:
        """Background task to check connection health"""
        while not self._closing:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check all connections
                async with self._lock:
                    for conn_id in list(self._connections.keys()):
                        conn = self._connections[conn_id]
                        
                        # Skip in-use connections
                        if conn._in_use:
                            continue
                            
                        # Check idle timeout
                        if conn.stats.is_idle(self.config.idle_timeout):
                            logger.info(f"Destroying idle connection {conn_id}")
                            await self._destroy_connection(conn_id)
                            continue
                            
                        # Perform health check
                        is_healthy = await conn.health_check()
                        
                        if not is_healthy and conn.stats.health_check_failures >= self.config.unhealthy_threshold:
                            logger.warning(f"Destroying unhealthy connection {conn_id}")
                            await self._destroy_connection(conn_id)
                            
                    # Ensure minimum connections
                    while len(self._connections) < self.config.min_connections:
                        try:
                            await self._create_connection()
                        except Exception as e:
                            logger.error(f"Failed to maintain minimum connections: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                
    async def close(self) -> None:
        """Close all connections and shutdown pool"""
        logger.info("Closing connection pool...")
        self._closing = True
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        # Close all connections
        async with self._lock:
            for conn_id in list(self._connections.keys()):
                await self._destroy_connection(conn_id)
                
        logger.info(f"Connection pool closed. Total created: {self._total_created}, destroyed: {self._total_destroyed}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'total_connections': len(self._connections),
            'available_connections': len(self._available),
            'healthy_connections': sum(1 for c in self._connections.values() if c.stats.is_healthy),
            'total_acquired': self._total_acquired,
            'total_released': self._total_released,
            'total_created': self._total_created,
            'total_destroyed': self._total_destroyed,
            'connection_stats': {
                conn_id: {
                    'is_healthy': conn.stats.is_healthy,
                    'total_queries': conn.stats.total_queries,
                    'total_errors': conn.stats.total_errors,
                    'last_used': conn.stats.last_used_at.isoformat()
                }
                for conn_id, conn in self._connections.items()
            }
        }