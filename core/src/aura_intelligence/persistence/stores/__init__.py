"""
Unified Store Implementations
============================
Production-ready adapters for vector, graph, time-series,
document, and event stores with consistent API.
"""

from .registry import (
    StoreRegistry,
    get_store,
    register_store
)

from .vector import (
    UnifiedVectorStore,
    QdrantVectorStore,
    ClickHouseVectorStore,
    PgVectorStore,
    VectorIndexConfig
)

from .timeseries import (
    UnifiedTimeSeriesStore,
    InfluxDB3Store,
    QuestDBStore,
    TimeSeriesConfig
)

from .graph import (
    UnifiedGraphStore,
    Neo4jGraphStore,
    GraphConfig
)

from .document import (
    UnifiedDocumentStore,
    DocumentConfig
)

from .event import (
    UnifiedEventStore,
    EventStoreConfig
)

from .kv import (
    NATSKVStore,
    KVConfig
)

__all__ = [
    # Registry
    'StoreRegistry',
    'get_store',
    'register_store',
    
    # Vector stores
    'UnifiedVectorStore',
    'QdrantVectorStore',
    'ClickHouseVectorStore',
    'PgVectorStore',
    'VectorIndexConfig',
    
    # Time-series stores
    'UnifiedTimeSeriesStore',
    'InfluxDB3Store',
    'QuestDBStore',
    'TimeSeriesConfig',
    
    # Graph stores
    'UnifiedGraphStore',
    'Neo4jGraphStore',
    'GraphConfig',
    
    # Document stores
    'UnifiedDocumentStore',
    'DocumentConfig',
    
    # Event stores
    'UnifiedEventStore',
    'EventStoreConfig',
    
    # KV stores
    'NATSKVStore',
    'KVConfig'
]