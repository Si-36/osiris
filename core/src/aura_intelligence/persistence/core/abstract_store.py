"""
Abstract Store Protocol - Foundation for All Persistence
========================================================
Defines the unified interface for all storage backends.
Ensures consistent API across vector, graph, time-series, and document stores.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, TypeVar, Generic
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import asyncio
from datetime import datetime
import uuid

# Type variables for generic store
T = TypeVar('T')
K = TypeVar('K')


class StoreType(Enum):
    """Types of specialized stores in AURA"""
    VECTOR = "vector"
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    DOCUMENT = "document"
    EVENT = "event"
    KV = "key_value"
    LAKEHOUSE = "lakehouse"


@dataclass
class WriteResult:
    """Result of a write operation"""
    success: bool
    id: Optional[str] = None
    version: Optional[int] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult(Generic[T]):
    """Result of a query operation"""
    success: bool
    data: List[T] = field(default_factory=list)
    total_count: Optional[int] = None
    next_cursor: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionContext:
    """Context for transactional operations"""
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    temporal_workflow_id: Optional[str] = None
    temporal_run_id: Optional[str] = None
    saga_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    tenant_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def create_child_context(self, operation: str) -> 'TransactionContext':
        """Create a child context for nested operations"""
        return TransactionContext(
            transaction_id=f"{self.transaction_id}:{operation}",
            temporal_workflow_id=self.temporal_workflow_id,
            temporal_run_id=self.temporal_run_id,
            saga_id=self.saga_id,
            idempotency_key=f"{self.idempotency_key}:{operation}" if self.idempotency_key else None,
            tenant_id=self.tenant_id,
            metadata={**self.metadata, 'parent_tx': self.transaction_id}
        )


class AbstractStore(ABC, Generic[K, T]):
    """
    Abstract base class for all storage implementations.
    Provides unified interface with ACID guarantees and idempotency.
    """
    
    def __init__(self, store_type: StoreType, config: Dict[str, Any]):
        self.store_type = store_type
        self.config = config
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize store connection and resources"""
        pass
        
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check store health and return status"""
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Close store connections and cleanup"""
        pass
        
    # Core CRUD Operations
    
    @abstractmethod
    async def upsert(self, 
                    key: K,
                    value: T,
                    context: Optional[TransactionContext] = None) -> WriteResult:
        """
        Idempotent upsert operation.
        Uses context.idempotency_key to ensure exactly-once semantics.
        """
        pass
        
    @abstractmethod
    async def get(self,
                  key: K,
                  context: Optional[TransactionContext] = None) -> Optional[T]:
        """Get a single value by key"""
        pass
        
    @abstractmethod
    async def list(self,
                   filter_dict: Optional[Dict[str, Any]] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None,
                   context: Optional[TransactionContext] = None) -> QueryResult[T]:
        """List values with optional filtering and pagination"""
        pass
        
    @abstractmethod
    async def delete(self,
                     key: K,
                     context: Optional[TransactionContext] = None) -> WriteResult:
        """Delete a value by key"""
        pass
        
    # Batch Operations
    
    @abstractmethod
    async def batch_upsert(self,
                          items: List[tuple[K, T]],
                          context: Optional[TransactionContext] = None) -> List[WriteResult]:
        """Batch upsert with transactional guarantees"""
        pass
        
    @abstractmethod
    async def batch_get(self,
                       keys: List[K],
                       context: Optional[TransactionContext] = None) -> Dict[K, Optional[T]]:
        """Batch get multiple values"""
        pass
        
    # Advanced Query Operations
    
    async def query(self,
                   query_builder: 'QueryBuilder',
                   context: Optional[TransactionContext] = None) -> QueryResult[T]:
        """
        Execute advanced query using QueryBuilder.
        Default implementation delegates to list() - override for optimization.
        """
        # Convert QueryBuilder to filter_dict for basic implementation
        filter_dict = query_builder.build()
        return await self.list(
            filter_dict=filter_dict,
            limit=query_builder.limit,
            cursor=query_builder.cursor,
            context=context
        )
        
    # Transaction Support
    
    async def begin_transaction(self, context: TransactionContext) -> None:
        """Begin a transaction - override if store supports it"""
        pass
        
    async def commit_transaction(self, context: TransactionContext) -> None:
        """Commit a transaction - override if store supports it"""
        pass
        
    async def rollback_transaction(self, context: TransactionContext) -> None:
        """Rollback a transaction - override if store supports it"""
        pass
        
    # Tenant Isolation
    
    def _apply_tenant_filter(self, 
                           filter_dict: Optional[Dict[str, Any]],
                           context: Optional[TransactionContext]) -> Dict[str, Any]:
        """Apply tenant isolation to queries"""
        if not context or not context.tenant_id:
            return filter_dict or {}
            
        result = filter_dict.copy() if filter_dict else {}
        result['tenant_id'] = context.tenant_id
        return result
        
    # Observability
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get store-specific metrics"""
        return {
            'store_type': self.store_type.value,
            'initialized': self._initialized,
            'health': await self.health_check()
        }


class VectorStore(AbstractStore[str, Dict[str, Any]]):
    """Specialized interface for vector stores"""
    
    @abstractmethod
    async def search_similar(self,
                           embedding: List[float],
                           limit: int = 10,
                           filter_dict: Optional[Dict[str, Any]] = None,
                           context: Optional[TransactionContext] = None) -> QueryResult[Dict[str, Any]]:
        """Search for similar vectors"""
        pass
        
    @abstractmethod
    async def create_index(self,
                          index_config: Dict[str, Any],
                          context: Optional[TransactionContext] = None) -> WriteResult:
        """Create or update vector index"""
        pass


class GraphStore(AbstractStore[str, Dict[str, Any]]):
    """Specialized interface for graph stores"""
    
    @abstractmethod
    async def traverse(self,
                      start_node: str,
                      query: str,
                      max_depth: int = 3,
                      context: Optional[TransactionContext] = None) -> QueryResult[Dict[str, Any]]:
        """Traverse graph from start node"""
        pass
        
    @abstractmethod
    async def add_edge(self,
                      from_node: str,
                      to_node: str,
                      edge_type: str,
                      properties: Optional[Dict[str, Any]] = None,
                      context: Optional[TransactionContext] = None) -> WriteResult:
        """Add edge between nodes"""
        pass


class TimeSeriesStore(AbstractStore[str, Dict[str, Any]]):
    """Specialized interface for time-series stores"""
    
    @abstractmethod
    async def write_points(self,
                         series_key: str,
                         points: List[Dict[str, Any]],
                         context: Optional[TransactionContext] = None) -> WriteResult:
        """Write time-series points"""
        pass
        
    @abstractmethod
    async def query_range(self,
                        series_key: str,
                        start_time: datetime,
                        end_time: datetime,
                        aggregation: Optional[str] = None,
                        context: Optional[TransactionContext] = None) -> QueryResult[Dict[str, Any]]:
        """Query time range with optional aggregation"""
        pass
        
    @abstractmethod
    async def downsample(self,
                       series_key: str,
                       retention_policy: Dict[str, Any],
                       context: Optional[TransactionContext] = None) -> WriteResult:
        """Apply downsampling policy"""
        pass