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
    TransactionContext
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
    SortOrder
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
    
    # Transactions
    'TransactionManager',
    'SagaOrchestrator',
    'OutboxPattern'
]