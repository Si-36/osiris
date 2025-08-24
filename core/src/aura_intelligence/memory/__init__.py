"""
Memory module for AURA Intelligence
"""

# Import only the essentials to avoid circular imports
from .knn_index_real import HybridKNNIndex, KNNConfig, create_knn_index

__all__ = [
    'HybridKNNIndex',
    'KNNConfig', 
    'create_knn_index',
]