"""
Apache Iceberg Lakehouse Implementation
======================================
Provides immutable data storage with time travel, branching,
and ACID guarantees on object storage.
"""

from .catalog import (
    IcebergCatalog,
    CatalogType,
    CatalogConfig
)

from .datasets import (
    EventsDataset,
    FeaturesDataset,
    EmbeddingsDataset,
    TopologyDataset,
    AuditDataset
)

from .branching import (
    BranchManager,
    TagManager,
    BranchConfig,
    PromotionStrategy
)

from .streaming import (
    CDCSink,
    StreamingBridge,
    StreamConfig
)

__all__ = [
    # Catalog
    'IcebergCatalog',
    'CatalogType',
    'CatalogConfig',
    
    # Datasets
    'EventsDataset',
    'FeaturesDataset', 
    'EmbeddingsDataset',
    'TopologyDataset',
    'AuditDataset',
    
    # Branching
    'BranchManager',
    'TagManager',
    'BranchConfig',
    'PromotionStrategy',
    
    # Streaming
    'CDCSink',
    'StreamingBridge',
    'StreamConfig'
]