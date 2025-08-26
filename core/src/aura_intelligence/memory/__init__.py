"""
Memory module for AURA Intelligence
"""

# Import only the essentials to avoid circular imports
from .knn_index_real import HybridKNNIndex, KNNConfig, create_knn_index

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

            __all__ = [
            'HybridKNNIndex',
            'KNNConfig',
            'create_knn_index',
            'MemorySettings',
            ]
