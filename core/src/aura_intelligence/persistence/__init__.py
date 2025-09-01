"""
AURA Persistence Layer - 2025 Production Data Platform
======================================================

A unified persistence abstraction that combines:
- Apache Iceberg lakehouse with branching/tags
- Temporal.io multi-region HA for process durability
- NATS JetStream KV mirrors for control plane
- Specialized vector/time-series/graph stores
- Enterprise security with encryption and WORM compliance

This implementation follows the latest 2025 patterns from:
- Databricks (Iceberg integration)
- Netflix (Temporal at scale)
- Uber (NATS for control plane)
- Discord (Qdrant quantization)
- Grafana (InfluxDB 3.0 architecture)
"""

# Core abstractions
from .core import (
    # Abstract Store
    AbstractStore,
    StoreType,
    QueryResult,
    WriteResult,
    TransactionContext,
    
    # Specialized Stores
    VectorStore,
    GraphStore,
    TimeSeriesStore,
    
    # Connection Management
    ConnectionPool,
    PooledConnection,
    ConnectionConfig,
    
    # Resilience
    StoreCircuitBreaker,
    CircuitState,
    
    # Query Building
    QueryBuilder,
    FilterOperator,
    SortOrder,
    query,
    vector_query,
    time_query,
    graph_query,
    
    # Transactions
    TransactionManager,
    SagaOrchestrator,
    OutboxPattern
)

# Store implementations
from .stores import (
    # Store interfaces
    UnifiedVectorStore,
    UnifiedGraphStore,
    UnifiedTimeSeriesStore,
    UnifiedDocumentStore,
    UnifiedEventStore,
    
    # Store registry
    StoreRegistry,
    get_store
)

# Lakehouse implementation
from .lakehouse import (
    # Catalog management
    IcebergCatalog,
    CatalogType,
    
    # Datasets
    EventsDataset,
    FeaturesDataset,
    EmbeddingsDataset,
    TopologyDataset,
    AuditDataset,
    
    # Branching
    BranchManager,
    TagManager,
    
    # Streaming
    CDCSink,
    StreamingBridge
)

# Security and compliance
from .security import (
    # Encryption
    EnvelopeEncryption,
    FieldLevelEncryption,
    KeyRotationManager,
    
    # Audit
    ImmutableAuditLog,
    AccessMonitor,
    ComplianceReporter,
    
    # Multi-tenancy
    TenantIsolation,
    DataResidencyManager,
    RowLevelSecurity
)

# Note: Backup functionality not implemented in current version

# Version and metadata
__version__ = "2.0.0"
__author__ = "AURA Intelligence Team"

# Initialize default registry
_default_registry = StoreRegistry()

# Convenience functions
def initialize_persistence(config: dict) -> StoreRegistry:
    """Initialize the persistence layer with configuration"""
    registry = StoreRegistry()
    registry.configure(config)
    return registry

def get_default_registry() -> StoreRegistry:
    """Get the default store registry"""
    return _default_registry

__all__ = [
    # Core abstractions
    'AbstractStore',
    'StoreType',
    'QueryResult',
    'WriteResult',
    'TransactionContext',
    'VectorStore',
    'GraphStore', 
    'TimeSeriesStore',
    
    # Connection management
    'ConnectionPool',
    'PooledConnection',
    'ConnectionConfig',
    
    # Resilience
    'StoreCircuitBreaker',
    'CircuitState',
    
    # Query building
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
    'OutboxPattern',
    
    # Store implementations
    'UnifiedVectorStore',
    'UnifiedGraphStore',
    'UnifiedTimeSeriesStore',
    'UnifiedDocumentStore',
    'UnifiedEventStore',
    'StoreRegistry',
    'get_store',
    
    # Lakehouse
    'IcebergCatalog',
    'CatalogType',
    'EventsDataset',
    'FeaturesDataset',
    'EmbeddingsDataset',
    'TopologyDataset',
    'AuditDataset',
    'BranchManager',
    'TagManager',
    'CDCSink',
    'StreamingBridge',
    
    # Security
    'EnvelopeEncryption',
    'FieldLevelEncryption',
    'KeyRotationManager',
    'ImmutableAuditLog',
    'AccessMonitor',
    'ComplianceReporter',
    'TenantIsolation',
    'DataResidencyManager',
    'RowLevelSecurity',
    
    # Convenience
    'initialize_persistence',
    'get_default_registry'
]