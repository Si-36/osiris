"""
AURA Persistence Core - 2025 Production Data Platform
=====================================================
Unified abstraction for Iceberg lakehouse, Temporal histories,
NATS KV mirrors, and specialized vector/time-series/graph stores.

Based on latest 2025 research:
- Apache Iceberg 1.8.1 with branching/tags
- Temporal.io multi-region HA 
- NATS JetStream KV mirrors
- Qdrant 1.15 with quantization
- InfluxDB 3.0 / QuestDB 9.x
"""

from .abstract_store import (
    AbstractStore,
    StoreType,
    QueryResult,
    WriteResult,
    TransactionContext,
    VectorStore,
    GraphStore,
    TimeSeriesStore
)
from .connection_pool import (
    ConnectionPool,
    PooledConnection,
    ConnectionConfig
)
from .circuit_breaker import (
    StoreCircuitBreaker,
    CircuitState
)
from .query_builder import (
    QueryBuilder,
    FilterOperator,
    SortOrder,
    query,
    vector_query,
    time_query,
    graph_query
)
from .transaction_manager import (
    TransactionManager,
    SagaOrchestrator,
    OutboxPattern
)

__all__ = [
    # Abstract Store
    'AbstractStore',
    'StoreType', 
    'QueryResult',
    'WriteResult',
    'TransactionContext',
    
    # Specialized Stores
    'VectorStore',
    'GraphStore',
    'TimeSeriesStore',
    
    # Connection Management
    'ConnectionPool',
    'PooledConnection',
    'ConnectionConfig',
    
    # Resilience
    'StoreCircuitBreaker',
    'CircuitState',
    
    # Query Building
    'QueryBuilder',
    'FilterOperator',
    'SortOrder',
    'query',
    'vector_query',
    'time_query',
    'graph_query',
    
    # Transactions
    'TransactionManager',
    'SagaOrchestrator',
    'OutboxPattern'
]