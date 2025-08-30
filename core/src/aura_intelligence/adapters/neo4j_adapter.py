"""
Neo4j Adapter for AURA Intelligence - 2025 Best Practices

Features:
- Async/await with proper connection pooling
- Circuit breaker pattern for resilience
- Exponential backoff with jitter
- Comprehensive observability
- Type safety with dataclasses
- Context managers for resource cleanup
"""

from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import random
from enum import Enum

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    # Mock classes for development without Neo4j
    class AsyncDriver:
        pass
    class AsyncSession:
        pass
    class Neo4jError(Exception):
        pass
    class ServiceUnavailable(Exception):
        pass
    class SessionExpired(Exception):
        pass

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = structlog.get_logger(__name__)

# Create tracer with fallback
try:
    from ..observability import create_tracer
    tracer = create_tracer("neo4j_adapter")
except ImportError:
    tracer = trace.get_tracer(__name__)


class RetryStrategy(Enum):
    """Retry strategies for resilience"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection with 2025 best practices"""
    # Connection settings
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    
    # Connection pool settings
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: float = 60.0
    connection_timeout: float = 30.0
    keep_alive: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    retry_jitter: float = 0.2
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Query settings
    query_timeout: float = 30.0
    fetch_size: int = 1000
    
    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = True


@dataclass
class QueryResult:
    """Result container for Neo4j queries"""
    records: List[Dict[str, Any]]
    summary: Dict[str, Any]
    query_time_ms: float
    record_count: int


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, threshold: int, timeout: float):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        
        if self.failure_count >= self.threshold:
            self.state = CircuitBreakerState.OPEN
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = asyncio.get_event_loop().time() - self.last_failure_time
                if elapsed > self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
            return False
        
        # HALF_OPEN - allow one request to test
        return True


class Neo4jAdapter:
    """
    Modern Neo4j adapter with 2025 best practices
    
    Features:
    - Async/await throughout
    - Connection pooling
    - Circuit breaker pattern
    - Exponential backoff with jitter
    - Comprehensive error handling
    - OpenTelemetry integration
    - Type safety
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        self.config = config or Neo4jConfig()
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False
        self._circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        
    async def initialize(self) -> None:
        """Initialize the Neo4j driver with connection verification"""
        if self._initialized:
            return
            
        with tracer.start_as_current_span("neo4j_initialize") as span:
            span.set_attribute("neo4j.uri", self.config.uri)
            span.set_attribute("neo4j.database", self.config.database)
            
            try:
                if not NEO4J_AVAILABLE:
                    logger.warning("Neo4j driver not available, using mock")
                    self._initialized = True
                    return
                
                self._driver = AsyncGraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.username, self.config.password),
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                    connection_timeout=self.config.connection_timeout,
                    keep_alive=self.config.keep_alive
                )
                
                # Verify connectivity
                await self._driver.verify_connectivity()
                self._initialized = True
                
                logger.info(
                    "Neo4j adapter initialized",
                    uri=self.config.uri,
                    database=self.config.database
                )
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error("Failed to initialize Neo4j", error=str(e))
                raise
    
    async def close(self) -> None:
        """Close the Neo4j driver and cleanup resources"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info("Neo4j adapter closed")
    
    @asynccontextmanager
    async def session(self, database: Optional[str] = None) -> AsyncIterator[AsyncSession]:
        """
        Create a Neo4j session with automatic cleanup
        
        Args:
            database: Optional database name (defaults to config)
            
        Yields:
            AsyncSession: Neo4j session for queries
        """
        if not self._initialized:
            await self.initialize()
            
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
            
        session = self._driver.session(
            database=database or self.config.database,
            fetch_size=self.config.fetch_size
        )
        
        try:
            yield session
        finally:
            await session.close()
    
    async def query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> QueryResult:
        """
        Execute a read query with retry and circuit breaker
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            database: Optional database name
            
        Returns:
            QueryResult: Query results with metadata
        """
        if not self._circuit_breaker.can_execute():
            raise ServiceUnavailable("Circuit breaker is open")
            
        with tracer.start_as_current_span("neo4j_query") as span:
            span.set_attribute("neo4j.query", cypher[:100])
            
            try:
                result = await self._execute_with_retry(
                    self._run_read_query,
                    cypher,
                    params or {},
                    database
                )
                
                self._circuit_breaker.record_success()
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                self._circuit_breaker.record_failure()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "Query failed",
                    query=cypher[:100],
                    error=str(e)
                )
                raise
    
    async def execute(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> QueryResult:
        """
        Execute a write query with retry and circuit breaker
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            database: Optional database name
            
        Returns:
            QueryResult: Query results with metadata
        """
        if not self._circuit_breaker.can_execute():
            raise ServiceUnavailable("Circuit breaker is open")
            
        with tracer.start_as_current_span("neo4j_execute") as span:
            span.set_attribute("neo4j.query", cypher[:100])
            
            try:
                result = await self._execute_with_retry(
                    self._run_write_query,
                    cypher,
                    params or {},
                    database
                )
                
                self._circuit_breaker.record_success()
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                self._circuit_breaker.record_failure()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(
                    "Execute failed",
                    query=cypher[:100],
                    error=str(e)
                )
                raise
    
    async def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
                
            except (ServiceUnavailable, SessionExpired) as e:
                last_error = e
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        "Retrying after error",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                    
                    # Reinitialize connection if needed
                    if isinstance(e, SessionExpired):
                        await self.initialize()
                        
            except Exception as e:
                # Non-retryable error
                raise
        
        # All retries exhausted
        raise last_error or Exception("Retry failed")
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with jitter"""
        if self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
            base_delay = self.config.initial_retry_delay * (2 ** attempt)
        elif self.config.retry_strategy == RetryStrategy.LINEAR:
            base_delay = self.config.initial_retry_delay * (attempt + 1)
        else:  # FIXED
            base_delay = self.config.initial_retry_delay
            
        # Apply jitter
        jitter = base_delay * self.config.retry_jitter * (2 * random.random() - 1)
        delay = base_delay + jitter
        
        # Cap at max delay
        return min(delay, self.config.max_retry_delay)
    
    async def _run_read_query(
        self,
        cypher: str,
        params: Dict[str, Any],
        database: Optional[str]
    ) -> QueryResult:
        """Execute read query in a transaction"""
        start_time = asyncio.get_event_loop().time()
        
        async with self.session(database) as session:
            result = await session.execute_read(
                self._execute_transaction,
                cypher,
                params
            )
            
        query_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return QueryResult(
            records=result['records'],
            summary=result['summary'],
            query_time_ms=query_time_ms,
            record_count=len(result['records'])
        )
    
    async def _run_write_query(
        self,
        cypher: str,
        params: Dict[str, Any],
        database: Optional[str]
    ) -> QueryResult:
        """Execute write query in a transaction"""
        start_time = asyncio.get_event_loop().time()
        
        async with self.session(database) as session:
            result = await session.execute_write(
                self._execute_transaction,
                cypher,
                params
            )
            
        query_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return QueryResult(
            records=result['records'],
            summary=result['summary'],
            query_time_ms=query_time_ms,
            record_count=len(result['records'])
        )
    
    @staticmethod
    async def _execute_transaction(tx, cypher: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query within a transaction"""
        result = await tx.run(cypher, **params)
        records = [dict(record) async for record in result]
        summary = await result.consume()
        
        return {
            'records': records,
            'summary': {
                'counters': summary.counters,
                'database': summary.database,
                'query_type': summary.query_type,
                'result_available_after': summary.result_available_after,
                'result_consumed_after': summary.result_consumed_after
            }
        }
    
    # Convenience methods for common operations
    
    async def create_node(
        self,
        labels: Union[str, List[str]],
        properties: Dict[str, Any]
    ) -> str:
        """Create a node and return its ID"""
        if isinstance(labels, str):
            labels = [labels]
            
        label_str = ":".join(labels)
        
        query = f"""
        CREATE (n:{label_str} $props)
        RETURN elementId(n) as id
        """
        
        result = await self.execute(query, {"props": properties})
        return result.records[0]["id"]
    
    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a relationship between nodes"""
        query = """
        MATCH (a) WHERE elementId(a) = $from_id
        MATCH (b) WHERE elementId(b) = $to_id
        CREATE (a)-[r:""" + rel_type + """ $props]->(b)
        RETURN elementId(r) as id
        """
        
        result = await self.execute(
            query,
            {
                "from_id": from_id,
                "to_id": to_id,
                "props": properties or {}
            }
        )
        return result.records[0]["id"]
    
    async def find_node(
        self,
        labels: Optional[Union[str, List[str]]] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Find nodes matching criteria"""
        conditions = []
        
        if labels:
            if isinstance(labels, str):
                labels = [labels]
            label_str = ":".join(labels)
            conditions.append(f"n:{label_str}")
        else:
            conditions.append("n")
            
        if properties:
            prop_conditions = [f"n.{k} = ${k}" for k in properties.keys()]
            conditions.extend(prop_conditions)
            
        query = f"""
        MATCH ({conditions[0]})
        {' WHERE ' + ' AND '.join(conditions[1:]) if len(conditions) > 1 else ''}
        RETURN n
        LIMIT 1000
        """
        
        result = await self.query(query, properties or {})
        return [record["n"] for record in result.records]


# Export the adapter
__all__ = ["Neo4jAdapter", "Neo4jConfig", "QueryResult", "CircuitBreakerState", "RetryStrategy"]