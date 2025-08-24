"""
MAX Memory - Ultra-Fast Vector Memory 2025
==========================================

State-of-the-art memory system with:
- FAISS GPU acceleration
- Hierarchical indexing
- Continuous learning
- Memory consolidation
"""

import numpy as np
import torch
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import time
from collections import deque
import json
import lmdb
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryConfig:
    """Configuration for MAX Memory"""
    embedding_dim: int = 768
    index_type: str = "IVF4096_HNSW32,PQ64"
    use_gpu: bool = True
    memory_size: int = 10_000_000
    consolidation_interval: int = 3600  # 1 hour
    similarity_threshold: float = 0.85
    batch_size: int = 1000
    
class HierarchicalIndex:
    """Hierarchical FAISS index for multi-scale search"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        
        # Initialize FAISS resources
        if self.device == "cuda":
            self.gpu_res = faiss.StandardGpuResources()
            self.gpu_res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
        
        # Create hierarchical indices
        self._init_indices()
        
    def _init_indices(self):
        """Initialize multi-level indices"""
        d = self.config.embedding_dim
        
        # Level 1: Coarse quantizer (fast)
        self.coarse_index = faiss.IndexFlatL2(d)
        
        # Level 2: Fine-grained index (accurate)
        if self.config.index_type == "IVF4096_HNSW32,PQ64":
            # IVF with HNSW quantizer and PQ compression
            quantizer = faiss.IndexHNSWFlat(d, 32)
            self.fine_index = faiss.IndexIVFPQ(
                quantizer, d, 4096, 64, 8
            )
        else:
            # Default to IVF with flat quantizer
            quantizer = faiss.IndexFlatL2(d)
            self.fine_index = faiss.IndexIVFFlat(quantizer, d, 100)
        
        # Move to GPU if available
        if self.device == "cuda":
            self.coarse_index = faiss.index_cpu_to_gpu(
                self.gpu_res, 0, self.coarse_index
            )
            self.fine_index = faiss.index_cpu_to_gpu(
                self.gpu_res, 0, self.fine_index
            )
        
        self.is_trained = False
        
    def train(self, embeddings: np.ndarray):
        """Train the index on sample data"""
        if not self.is_trained and hasattr(self.fine_index, 'train'):
            print(f"Training index on {len(embeddings)} samples...")
            self.fine_index.train(embeddings)
            self.is_trained = True
            
    def add(self, embeddings: np.ndarray, ids: np.ndarray):
        """Add embeddings to indices"""
        if not self.is_trained:
            self.train(embeddings[:min(100000, len(embeddings))])
        
        # Add to both indices
        self.coarse_index.add(embeddings)
        self.fine_index.add_with_ids(embeddings, ids)
        
    def search(self, queries: np.ndarray, k: int, use_fine: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Hierarchical search"""
        if use_fine and self.fine_index.ntotal > 0:
            # Fine-grained search
            distances, indices = self.fine_index.search(queries, k)
        else:
            # Coarse search for speed
            distances, indices = self.coarse_index.search(queries, k)
        
        return distances, indices

class MemoryConsolidator:
    """Consolidate and compress memories"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def consolidate(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate similar memories"""
        if len(memories) < 2:
            return memories
        
        # Extract embeddings
        texts = [m.get('text', str(m.get('data', ''))) for m in memories]
        embeddings = self.encoder.encode(texts, batch_size=32)
        
        # Cluster similar memories
        clustering = faiss.Kmeans(
            embeddings.shape[1],
            min(len(memories) // 10, 100),
            niter=20,
            gpu=self.config.use_gpu
        )
        clustering.train(embeddings.astype(np.float32))
        _, labels = clustering.index.search(embeddings.astype(np.float32), 1)
        
        # Consolidate clusters
        consolidated = []
        for cluster_id in range(clustering.k):
            cluster_memories = [m for i, m in enumerate(memories) if labels[i][0] == cluster_id]
            if cluster_memories:
                # Create consolidated memory
                consolidated.append({
                    'id': f"consolidated_{cluster_id}_{int(time.time())}",
                    'type': 'consolidated',
                    'source_count': len(cluster_memories),
                    'data': self._merge_memories(cluster_memories),
                    'timestamp': time.time(),
                    'importance': max(m.get('importance', 0.5) for m in cluster_memories)
                })
        
        return consolidated
    
    def _merge_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple memories into one"""
        # Implement sophisticated merging logic
        merged = {
            'texts': [m.get('text', '') for m in memories],
            'entities': list(set(e for m in memories for e in m.get('entities', []))),
            'topics': list(set(t for m in memories for t in m.get('topics', []))),
            'avg_confidence': np.mean([m.get('confidence', 0.5) for m in memories])
        }
        return merged

class MaxMemory:
    """MAX Performance Memory System"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # Components
        self.index = HierarchicalIndex(self.config)
        self.consolidator = MemoryConsolidator(self.config)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage
        self.lmdb_env = lmdb.open('./max_memory_db', map_size=100 * 1024**3)  # 100GB
        self.memory_buffer = deque(maxlen=10000)
        self.id_counter = 0
        
        # Metadata
        self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Start background tasks
        self._start_background_tasks()
        
    def _start_background_tasks(self):
        """Start background consolidation"""
        asyncio.create_task(self._consolidation_loop())
        
    async def _consolidation_loop(self):
        """Periodic memory consolidation"""
        while True:
            await asyncio.sleep(self.config.consolidation_interval)
            await self._consolidate_memories()
            
    async def _consolidate_memories(self):
        """Consolidate buffer memories"""
        if len(self.memory_buffer) < 100:
            return
            
        memories = list(self.memory_buffer)
        consolidated = await self.consolidator.consolidate(memories)
        
        # Store consolidated memories
        for mem in consolidated:
            await self.store(mem)
            
        # Clear buffer
        self.memory_buffer.clear()
        
    async def store(self, data: Dict[str, Any]) -> str:
        """Store memory with embedding"""
        # Generate embedding
        text = data.get('text', str(data))
        embedding = self.encoder.encode([text])[0]
        
        # Generate ID
        mem_id = self.id_counter
        self.id_counter += 1
        
        # Store in LMDB
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(
                f"mem_{mem_id}".encode(),
                json.dumps(data).encode()
            )
        
        # Add to index
        self.index.add(
            embedding.reshape(1, -1).astype(np.float32),
            np.array([mem_id])
        )
        
        # Store metadata
        self.id_to_metadata[mem_id] = {
            'timestamp': time.time(),
            'type': data.get('type', 'generic'),
            'importance': data.get('importance', 0.5)
        }
        
        # Add to buffer for consolidation
        self.memory_buffer.append(data)
        
        return f"mem_{mem_id}"
        
    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar memories"""
        # Generate query embedding
        query_embedding = self.encoder.encode([query])[0]
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            k
        )
        
        # Retrieve memories
        memories = []
        with self.lmdb_env.begin() as txn:
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:  # Valid index
                    data = txn.get(f"mem_{idx}".encode())
                    if data:
                        memory = json.loads(data.decode())
                        memory['similarity'] = 1.0 - dist  # Convert distance to similarity
                        memory['metadata'] = self.id_to_metadata.get(idx, {})
                        memories.append(memory)
        
        return memories
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'total_memories': self.index.fine_index.ntotal,
            'buffer_size': len(self.memory_buffer),
            'index_trained': self.index.is_trained,
            'device': self.index.device,
            'consolidation_interval': self.config.consolidation_interval
        }
