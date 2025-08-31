"""
Memory module for AURA Intelligence
"""

# Import only the essentials to avoid circular imports
try:
    from .knn_index_real import HybridKNNIndex, KNNConfig, create_knn_index
except (ImportError, NameError):
    # Fallback to simple implementation if dependencies are missing
    from .knn_index_simple import HybridKNNIndex, KNNConfig, create_knn_index

# Import MemorySettings if available
try:
    from .config import MemorySettings
except ImportError:
    # Create a minimal MemorySettings for compatibility
    class MemorySettings:
        """Memory configuration settings"""
        def __init__(self):
            self.max_memory_size = 1000000
            self.ttl_seconds = 3600
            self.compression_enabled = True
            self.persistence_enabled = True
            self.cache_size = 10000

# Import unified memory interface
from .unified_memory_interface import (
    UnifiedMemoryInterface,
    MemoryType,
    ConsistencyLevel,
    SearchType,
    MemoryMetadata,
    MemoryResult
)

# Import Mem0 pipeline
from .mem0_pipeline import (
    Mem0Pipeline,
    ExtractedFact,
    FactType,
    ConfidenceLevel,
    RetrievalContext
)

# Import H-MEM hierarchical routing
from .hierarchical_routing import (
    HMemSystem,
    MemoryLevel,
    HierarchicalRouter
)

# Import Qdrant configuration
from .qdrant_config import (
    QdrantMultitenantManager,
    QdrantCollectionConfig,
    QuantizationType,
    QuantizationPreset
)

# Import HybridMemoryManager and alias for backward compatibility
from .hybrid_manager import HybridMemoryManager
MemoryManager = HybridMemoryManager

__all__ = [
    # KNN Index
    'HybridKNNIndex',
    'KNNConfig',
    'create_knn_index',
    'MemorySettings',
    # Memory Manager
    'HybridMemoryManager',
    'MemoryManager',  # Alias for backward compatibility
    # Unified Interface
    'UnifiedMemoryInterface',
    'MemoryType',
    'ConsistencyLevel',
    'SearchType',
    'MemoryMetadata',
    'MemoryResult',
    # Mem0 Pipeline
    'Mem0Pipeline',
    'ExtractedFact',
    'FactType',
    'ConfidenceLevel',
    'RetrievalContext',
    # H-MEM
    'HMemSystem',
    'MemoryLevel',
    'HierarchicalRouter',
    # Qdrant
    'QdrantMultitenantManager',
    'QdrantCollectionConfig',
    'QuantizationType',
    'QuantizationPreset'
]
