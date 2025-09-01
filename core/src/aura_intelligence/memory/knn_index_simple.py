"""
Simple KNN Index Implementation - No External Dependencies
=========================================================

A basic but functional k-NN index using only numpy and scipy.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class KNNConfig:
    """Configuration for k-NN index."""
    metric: str = 'cosine'
    backend: str = 'numpy'
    initial_capacity: int = 10000


class SimpleKNNIndex:
    """Simple k-NN index using numpy/scipy."""
    
    def __init__(self, embedding_dim: int, config: Optional[KNNConfig] = None):
        self.embedding_dim = embedding_dim
        self.config = config or KNNConfig()
        self._vectors: List[np.ndarray] = []
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to index."""
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {vectors.shape[1]}")
            
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            if id_ in self._id_to_idx:
                # Update existing
                idx = self._id_to_idx[id_]
                self._vectors[idx] = vec
            else:
                # Add new
                self._id_to_idx[id_] = len(self._vectors)
                self._vectors.append(vec)
                self._ids.append(id_)
                
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if not self._vectors:
            return []
            
        if query.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {query.shape[0]}")
            
        # Convert to numpy array
        vectors = np.array(self._vectors)
        
        # Compute distances
        if self.config.metric == 'cosine':
            # Normalize for cosine similarity
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(vectors_norm, query_norm)
            distances = 1 - similarities
        elif self.config.metric == 'euclidean':
            distances = np.linalg.norm(vectors - query, axis=1)
        else:
            # Use scipy for other metrics
            distances = cdist([query], vectors, metric=self.config.metric)[0]
            
        # Get top k
        k = min(k, len(self._vectors))
        indices = np.argpartition(distances, k-1)[:k]
        indices = indices[np.argsort(distances[indices])]
        
        # Return results
        results = []
        for idx in indices:
            score = float(1 - distances[idx]) if self.config.metric == 'cosine' else float(-distances[idx])
            results.append((self._ids[idx], score))
            
        return results
        
    def get(self, id_: str) -> Optional[np.ndarray]:
        """Get vector by ID."""
        if id_ in self._id_to_idx:
            return self._vectors[self._id_to_idx[id_]]
        return None
        
    def remove(self, ids: List[str]) -> List[str]:
        """Remove vectors by ID."""
        removed = []
        for id_ in ids:
            if id_ in self._id_to_idx:
                idx = self._id_to_idx[id_]
                # Mark for removal (don't actually remove to preserve indices)
                self._vectors[idx] = None
                del self._id_to_idx[id_]
                removed.append(id_)
        
        # Periodically compact
        if len(removed) > len(self._vectors) * 0.2:
            self._compact()
            
        return removed
        
    def _compact(self):
        """Compact the index by removing None entries."""
        new_vectors = []
        new_ids = []
        new_id_to_idx = {}
        
        for i, (vec, id_) in enumerate(zip(self._vectors, self._ids)):
            if vec is not None and id_ in self._id_to_idx:
                new_id_to_idx[id_] = len(new_vectors)
                new_vectors.append(vec)
                new_ids.append(id_)
                
        self._vectors = new_vectors
        self._ids = new_ids
        self._id_to_idx = new_id_to_idx
        
    def clear(self):
        """Clear all vectors."""
        self._vectors.clear()
        self._ids.clear()
        self._id_to_idx.clear()
        
    def __len__(self) -> int:
        """Number of vectors in index."""
        return len(self._id_to_idx)
        
    def save(self, path: str) -> None:
        """Save index to disk."""
        import pickle
        data = {
            'vectors': self._vectors,
            'ids': self._ids,
            'id_to_idx': self._id_to_idx,
            'embedding_dim': self.embedding_dim,
            'config': self.config
        }
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, path: str) -> None:
        """Load index from disk."""
        import pickle
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
        self._vectors = data['vectors']
        self._ids = data['ids']
        self._id_to_idx = data['id_to_idx']
        self.embedding_dim = data['embedding_dim']
        self.config = data.get('config', KNNConfig())


# Alias for compatibility
HybridKNNIndex = SimpleKNNIndex


def create_knn_index(
    embedding_dim: int,
    backend: str = 'numpy',
    metric: str = 'cosine',
    **kwargs
) -> SimpleKNNIndex:
    """Create a simple k-NN index."""
    config = KNNConfig(backend=backend, metric=metric, **kwargs)
    return SimpleKNNIndex(embedding_dim, config)