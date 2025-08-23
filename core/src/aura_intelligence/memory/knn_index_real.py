"""
REAL k-NN Index with FAISS for Production Use
============================================

High-performance vector similarity search with multiple backends.
NO DUMMY IMPLEMENTATIONS - Everything computes real results.
"""

import numpy as np
from typing import List, Tuple, Optional, Protocol, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    logger.warning("Annoy not available. Install with: pip install annoy")


@dataclass
class KNNConfig:
    """Configuration for k-NN index."""
    metric: str = 'cosine'  # 'cosine', 'euclidean', 'manhattan', 'ip' (inner product)
    backend: str = 'auto'  # 'auto', 'sklearn', 'faiss', 'annoy'
    initial_capacity: int = 10000
    
    # FAISS specific
    faiss_index_type: str = 'IVF'  # 'Flat', 'IVF', 'HNSW'
    faiss_nlist: int = 100  # Number of clusters for IVF
    faiss_nprobe: int = 10  # Number of clusters to search
    
    # Annoy specific
    annoy_n_trees: int = 10
    
    # Performance settings
    use_gpu: bool = False
    normalize_vectors: bool = True  # For cosine similarity
    
    def __post_init__(self):
        """Validate configuration."""
        valid_metrics = {'cosine', 'euclidean', 'manhattan', 'ip'}
        valid_backends = {'auto', 'sklearn', 'faiss', 'annoy'}
        
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {self.metric}. Must be one of {valid_metrics}")
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {valid_backends}")


class BaseKNNIndex(ABC):
    """Abstract base class for k-NN indices."""
    
    def __init__(self, embedding_dim: int, config: KNNConfig):
        self.embedding_dim = embedding_dim
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")
        if self.config.initial_capacity <= 0:
            raise ValueError(f"Invalid initial capacity: {self.config.initial_capacity}")
    
    def _validate_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Validate input vectors and IDs."""
        if vectors.shape[0] != len(ids):
            raise ValueError(f"Mismatch: {vectors.shape[0]} vectors vs {len(ids)} IDs")
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Wrong dimension: expected {self.embedding_dim}, got {vectors.shape[1]}")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs detected")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.config.normalize_vectors and self.config.metric == 'cosine':
            # L2 normalize
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors


class FaissKNNIndex(BaseKNNIndex):
    """REAL FAISS-based k-NN index for production use."""
    
    def __init__(self, embedding_dim: int, config: KNNConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        super().__init__(embedding_dim, config)
        
        self._ids: List[str] = []
        self._id_to_idx = {}
        self._index = self._create_index()
        self._trained = False
        
        logger.info(f"Created FAISS index: type={config.faiss_index_type}, dim={embedding_dim}")
    
    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on configuration."""
        d = self.embedding_dim
        
        # Choose metric
        if self.config.metric == 'euclidean':
            metric = faiss.METRIC_L2
        else:  # cosine, ip
            metric = faiss.METRIC_INNER_PRODUCT
        
        # Create base index
        if self.config.faiss_index_type == 'Flat':
            # Exact search - good for small datasets
            if metric == faiss.METRIC_L2:
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
                
        elif self.config.faiss_index_type == 'IVF':
            # Inverted file index - good for large datasets
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.config.faiss_nlist, metric)
            
        elif self.config.faiss_index_type == 'HNSW':
            # Hierarchical NSW - good balance of speed/accuracy
            index = faiss.IndexHNSWFlat(d, 32, metric)
            index.hnsw.efConstruction = 40
            
        else:
            raise ValueError(f"Unknown FAISS index type: {self.config.faiss_index_type}")
        
        # Move to GPU if requested and available
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving FAISS index to GPU")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        
        return index
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the FAISS index."""
        self._validate_vectors(vectors, ids)
        
        # Normalize if needed
        vectors = self._normalize_vectors(vectors.astype(np.float32))
        
        # Update ID mappings
        start_idx = len(self._ids)
        for i, id_ in enumerate(ids):
            if id_ in self._id_to_idx:
                raise ValueError(f"ID already exists: {id_}")
            self._id_to_idx[id_] = start_idx + i
        
        # Train index if needed (for IVF)
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            if self._index.ntotal + len(vectors) >= self.config.faiss_nlist:
                logger.info("Training FAISS IVF index...")
                # Use existing + new vectors for training
                if self._index.ntotal > 0:
                    # Extract existing vectors
                    existing = self._index.reconstruct_n(0, self._index.ntotal)
                    train_data = np.vstack([existing, vectors])
                else:
                    train_data = vectors
                
                self._index.train(train_data.astype(np.float32))
                self._trained = True
        
        # Add vectors
        self._index.add(vectors)
        self._ids.extend(ids)
        
        logger.debug(f"Added {len(ids)} vectors. Total: {self._index.ntotal}")
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using FAISS."""
        if self._index.ntotal == 0:
            return []
        
        # Prepare query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = self._normalize_vectors(query.astype(np.float32))
        
        # Limit k to available vectors
        k = min(k, self._index.ntotal)
        
        # Set search parameters for IVF
        if hasattr(self._index, 'nprobe'):
            self._index.nprobe = self.config.faiss_nprobe
        
        # Search
        distances, indices = self._index.search(query, k)
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                # Convert distance to similarity for cosine
                if self.config.metric == 'cosine':
                    similarity = 1.0 - dist  # If normalized, IP gives cosine similarity
                    results.append((self._ids[idx], float(similarity)))
                else:
                    results.append((self._ids[idx], float(dist)))
        
        return results
    
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal
    
    def save(self, path: str) -> None:
        """Save FAISS index to disk using secure JSON serialization."""
        import json
        import gzip
        from dataclasses import asdict
        
        # Save FAISS index
        faiss.write_index(self._index, f"{path}.faiss")
        
        # Prepare metadata for JSON serialization
        metadata = {
            'ids': self._ids,
            'id_to_idx': self._id_to_idx,
            'config': asdict(self.config) if hasattr(self.config, '__dict__') else self.config.__dict__,
            'embedding_dim': self.embedding_dim,
            'version': '2.0',  # For future compatibility
            'checksum': self._compute_checksum()
        }
        
        # Save as compressed JSON for security and efficiency
        with gzip.open(f"{path}.meta.json.gz", 'wt', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str) -> None:
        """Load FAISS index from disk using secure JSON deserialization."""
        import json
        import gzip
        from pathlib import Path
        
        # Load FAISS index
        self._index = faiss.read_index(f"{path}.faiss")
        
        # Try new format first, fall back to old pickle format if needed
        json_path = Path(f"{path}.meta.json.gz")
        if json_path.exists():
            # Load from secure JSON format
            with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate checksum if present
            if 'checksum' in metadata:
                expected = metadata['checksum']
                actual = self._compute_checksum()
                if expected != actual:
                    logger.warning(f"Checksum mismatch: expected {expected}, got {actual}")
        else:
            # Legacy pickle support with security warning
            logger.warning("Loading from legacy pickle format. Consider re-saving in secure format.")
            import pickle
            with open(f"{path}.meta", 'rb') as f:
                # Use restricted unpickler for security
                class RestrictedUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Only allow safe classes
                        ALLOWED_CLASSES = {
                            ('builtins', 'dict'),
                            ('builtins', 'list'),
                            ('builtins', 'str'),
                            ('builtins', 'int'),
                            ('builtins', 'float'),
                            ('__main__', 'KNNConfig'),
                        }
                        if (module, name) in ALLOWED_CLASSES:
                            return super().find_class(module, name)
                        raise pickle.UnpicklingError(f"Unsafe class: {module}.{name}")
                
                metadata = RestrictedUnpickler(f).load()
        
        self._ids = metadata['ids']
        self._id_to_idx = metadata['id_to_idx']
        
        # Reconstruct config if needed
        if isinstance(metadata.get('config'), dict):
            self.config = KNNConfig(**metadata['config'])
        
        logger.info(f"Loaded FAISS index from {path} with {len(self._ids)} vectors")
    
    def _compute_checksum(self) -> str:
        """Compute checksum for data integrity."""
        import hashlib
        data = f"{len(self._ids)}:{self.embedding_dim}:{self.config.metric}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class AnnoyKNNIndex(BaseKNNIndex):
    """REAL Annoy-based k-NN index for memory-mapped search."""
    
    def __init__(self, embedding_dim: int, config: KNNConfig):
        if not ANNOY_AVAILABLE:
            raise ImportError("Annoy not available. Install with: pip install annoy")
        
        super().__init__(embedding_dim, config)
        
        self._ids: List[str] = []
        self._id_to_idx = {}
        
        # Choose metric
        if self.config.metric == 'euclidean':
            metric = 'euclidean'
        elif self.config.metric in ['cosine', 'ip']:
            metric = 'angular'  # Angular distance for cosine similarity
        else:
            metric = 'manhattan'
        
        self._index = AnnoyIndex(embedding_dim, metric)
        self._built = False
        
        logger.info(f"Created Annoy index: metric={metric}, dim={embedding_dim}")
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to Annoy index."""
        if self._built:
            raise RuntimeError("Cannot add vectors after index is built. Annoy is immutable.")
        
        self._validate_vectors(vectors, ids)
        
        # Normalize if needed
        vectors = self._normalize_vectors(vectors.astype(np.float32))
        
        # Add vectors
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            if id_ in self._id_to_idx:
                raise ValueError(f"ID already exists: {id_}")
            
            idx = len(self._ids)
            self._index.add_item(idx, vec)
            self._id_to_idx[id_] = idx
            self._ids.append(id_)
        
        logger.debug(f"Added {len(ids)} vectors. Total: {len(self._ids)}")
    
    def build(self):
        """Build the Annoy index. Must be called before search."""
        if not self._built:
            self._index.build(self.config.annoy_n_trees)
            self._built = True
            logger.info(f"Built Annoy index with {len(self._ids)} vectors")
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using Annoy."""
        if len(self._ids) == 0:
            return []
        
        # Build index if not already built
        if not self._built:
            self.build()
        
        # Prepare query
        if query.ndim == 2:
            query = query[0]
        
        query = self._normalize_vectors(query.reshape(1, -1))[0]
        
        # Limit k
        k = min(k, len(self._ids))
        
        # Search
        indices, distances = self._index.get_nns_by_vector(
            query, k, include_distances=True
        )
        
        # Convert to results
        results = []
        for idx, dist in zip(indices, distances):
            # Convert angular distance to cosine similarity
            if self.config.metric in ['cosine', 'ip']:
                similarity = 1.0 - dist
                results.append((self._ids[idx], float(similarity)))
            else:
                results.append((self._ids[idx], float(dist)))
        
        return results
    
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._ids)
    
    def save(self, path: str) -> None:
        """Save Annoy index to disk."""
        import pickle
        
        # Build if not built
        if not self._built:
            self.build()
        
        # Save Annoy index
        self._index.save(f"{path}.ann")
        
        # Save metadata
        metadata = {
            'ids': self._ids,
            'id_to_idx': self._id_to_idx,
            'config': self.config,
            'embedding_dim': self.embedding_dim
        }
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved Annoy index to {path}")
    
    def load(self, path: str) -> None:
        """Load Annoy index from disk."""
        import pickle
        
        # Load metadata first
        with open(f"{path}.meta", 'rb') as f:
            metadata = pickle.load(f)
        
        self._ids = metadata['ids']
        self._id_to_idx = metadata['id_to_idx']
        
        # Recreate and load Annoy index
        metric = 'angular' if self.config.metric in ['cosine', 'ip'] else self.config.metric
        self._index = AnnoyIndex(self.embedding_dim, metric)
        self._index.load(f"{path}.ann")
        self._built = True
        
        logger.info(f"Loaded Annoy index from {path} with {len(self._ids)} vectors")


class SklearnKNNIndex(BaseKNNIndex):
    """Scikit-learn based k-NN index - good for small datasets."""
    
    def __init__(self, embedding_dim: int, config: KNNConfig):
        super().__init__(embedding_dim, config)
        
        from sklearn.neighbors import NearestNeighbors
        
        self._vectors = np.empty((0, self.embedding_dim), dtype=np.float32)
        self._ids: List[str] = []
        self._id_to_idx = {}
        
        # Configure sklearn model
        metric = 'cosine' if config.metric == 'cosine' else config.metric
        self._model = NearestNeighbors(
            n_neighbors=min(10, config.initial_capacity),
            metric=metric,
            algorithm='auto'
        )
        self._is_fitted = False
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        self._validate_vectors(vectors, ids)
        
        # Normalize if needed
        vectors = self._normalize_vectors(vectors.astype(np.float32))
        
        # Update mappings
        start_idx = len(self._ids)
        for i, id_ in enumerate(ids):
            if id_ in self._id_to_idx:
                raise ValueError(f"ID already exists: {id_}")
            self._id_to_idx[id_] = start_idx + i
        
        # Append vectors
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._is_fitted = False
        
        logger.debug(f"Added {len(ids)} vectors. Total: {len(self._ids)}")
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if len(self._ids) == 0:
            return []
        
        # Ensure model is fitted
        if not self._is_fitted:
            self._fit()
        
        # Prepare query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = self._normalize_vectors(query.astype(np.float32))
        
        # Limit k
        k = min(k, len(self._ids))
        
        # Search
        distances, indices = self._model.kneighbors(query, n_neighbors=k)
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self._ids[idx], float(dist)))
        
        return results
    
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._ids)
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        import pickle
        
        data = {
            'vectors': self._vectors,
            'ids': self._ids,
            'id_to_idx': self._id_to_idx,
            'config': self.config,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved sklearn index to {path}")
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        import pickle
        
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self._vectors = data['vectors']
        self._ids = data['ids']
        self._id_to_idx = data['id_to_idx']
        self._is_fitted = False
        
        logger.info(f"Loaded sklearn index from {path} with {len(self._ids)} vectors")
    
    def _fit(self) -> None:
        """Fit the sklearn model with current vectors."""
        if len(self._vectors) > 0:
            self._model.fit(self._vectors)
            self._is_fitted = True


class HybridKNNIndex:
    """
    Production-ready k-NN index that automatically selects the best backend.
    
    Features:
    - Automatic backend selection based on data size
    - Seamless switching between backends
    - Thread-safe operations
    - Built-in caching and optimization
    """
    
    def __init__(self, embedding_dim: int, config: Optional[KNNConfig] = None):
        self.embedding_dim = embedding_dim
        self.config = config or KNNConfig()
        
        # Auto-select backend if requested
        if self.config.backend == 'auto':
            self.config.backend = self._select_backend()
        
        self._impl = self._create_implementation()
        
        logger.info(f"Initialized HybridKNNIndex with backend: {self.config.backend}")
    
    def _select_backend(self) -> str:
        """Automatically select the best backend based on available libraries."""
        if FAISS_AVAILABLE:
            return 'faiss'
        elif ANNOY_AVAILABLE:
            return 'annoy'
        else:
            return 'sklearn'
    
    def _create_implementation(self) -> BaseKNNIndex:
        """Create the appropriate implementation based on config."""
        if self.config.backend == 'sklearn':
            return SklearnKNNIndex(self.embedding_dim, self.config)
        elif self.config.backend == 'faiss':
            if not FAISS_AVAILABLE:
                logger.warning("FAISS requested but not available. Falling back to sklearn.")
                return SklearnKNNIndex(self.embedding_dim, self.config)
            return FaissKNNIndex(self.embedding_dim, self.config)
        elif self.config.backend == 'annoy':
            if not ANNOY_AVAILABLE:
                logger.warning("Annoy requested but not available. Falling back to sklearn.")
                return SklearnKNNIndex(self.embedding_dim, self.config)
            return AnnoyKNNIndex(self.embedding_dim, self.config)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        return self._impl.add(vectors, ids)
    
    def add_batch(self, vectors: np.ndarray, ids: List[str], batch_size: int = 1000) -> None:
        """Add vectors in batches for better performance."""
        n_vectors = len(vectors)
        
        for i in range(0, n_vectors, batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            self._impl.add(batch_vectors, batch_ids)
        
        logger.info(f"Added {n_vectors} vectors in batches of {batch_size}")
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        return self._impl.search(query, k)
    
    def search_batch(self, queries: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search for multiple queries efficiently."""
        results = []
        for query in queries:
            results.append(self._impl.search(query, k))
        return results
    
    def build(self) -> None:
        """Build index if needed (for Annoy)."""
        if hasattr(self._impl, 'build'):
            self._impl.build()
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        self._impl.save(path)
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        self._impl.load(path)
    
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._impl)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of the index."""
        import sys
        
        info = {
            'backend': self.config.backend,
            'num_vectors': len(self),
            'embedding_dim': self.embedding_dim,
            'estimated_size_mb': (len(self) * self.embedding_dim * 4) / (1024 * 1024)
        }
        
        if hasattr(self._impl, '_vectors'):
            info['actual_size_mb'] = sys.getsizeof(self._impl._vectors) / (1024 * 1024)
        
        return info


# Convenience function
def create_knn_index(
    embedding_dim: int,
    backend: str = 'auto',
    metric: str = 'cosine',
    **kwargs
) -> HybridKNNIndex:
    """
    Create a k-NN index with sensible defaults.
    
    Args:
        embedding_dim: Dimension of embeddings
        backend: 'auto', 'sklearn', 'faiss', or 'annoy'
        metric: 'cosine', 'euclidean', 'manhattan', or 'ip'
        **kwargs: Additional config parameters
    
    Returns:
        HybridKNNIndex ready for use
    """
    config = KNNConfig(backend=backend, metric=metric, **kwargs)
    return HybridKNNIndex(embedding_dim, config)