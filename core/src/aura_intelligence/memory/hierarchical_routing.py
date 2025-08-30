"""
H-MEM Hierarchical Routing - Multi-Level Memory Organization
===========================================================

Implements hierarchical memory routing from semantic summaries 
to episodic fragments with positional index encodings.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


@dataclass
class MemoryLevel:
    """Hierarchical memory level"""
    level: int  # 0=most abstract, higher=more specific
    name: str
    summary: str
    embedding: np.ndarray
    child_indices: List[str] = field(default_factory=list)
    parent_index: Optional[str] = None
    position_encoding: Optional[np.ndarray] = None
    access_count: int = 0
    last_access: float = 0.0


@dataclass
class PositionalEncoding:
    """Positional encoding for memory hierarchy"""
    level_encoding: np.ndarray  # Level in hierarchy
    temporal_encoding: np.ndarray  # Temporal position
    semantic_encoding: np.ndarray  # Semantic cluster position
    
    def to_vector(self) -> np.ndarray:
        """Combine encodings into single vector"""
        return np.concatenate([
            self.level_encoding,
            self.temporal_encoding,
            self.semantic_encoding
        ])


class HierarchicalRouter(nn.Module):
    """Neural router for hierarchical memory traversal"""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Level predictor
        self.level_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # Max 5 levels
        )
        
        # Routing scorer
        self.route_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query_embedding: torch.Tensor, level_embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Route query through hierarchy"""
        # Encode query
        query_hidden = self.query_encoder(query_embedding)
        
        # Predict target level
        level_logits = self.level_predictor(query_hidden)
        level_probs = F.softmax(level_logits, dim=-1)
        
        # Score each level embedding
        route_scores = []
        for level_emb in level_embeddings:
            combined = torch.cat([query_hidden, level_emb], dim=-1)
            score = self.route_scorer(combined)
            route_scores.append(score)
            
        route_scores = torch.cat(route_scores, dim=-1)
        route_probs = F.softmax(route_scores, dim=-1)
        
        return {
            "level_probs": level_probs,
            "route_probs": route_probs
        }


class HMemSystem:
    """
    H-MEM Hierarchical Memory System
    
    Features:
    - Multi-level memory organization
    - Positional index encodings
    - Efficient routing to reduce compute
    - Dynamic hierarchy adaptation
    """
    
    def __init__(self, max_levels: int = 5):
        self.max_levels = max_levels
        self.memory_levels: Dict[int, Dict[str, MemoryLevel]] = {
            i: {} for i in range(max_levels)
        }
        
        # Neural router
        self.router = HierarchicalRouter()
        
        # Positional encoding generators
        self.level_encoder = self._create_level_encoder()
        self.temporal_encoder = self._create_temporal_encoder()
        self.semantic_encoder = self._create_semantic_encoder()
        
        # Metrics
        self.metrics = {
            "total_traversals": 0,
            "levels_pruned": 0,
            "avg_hops": 0.0
        }
        
    def _create_level_encoder(self) -> nn.Module:
        """Create level position encoder"""
        class LevelEncoder(nn.Module):
            def __init__(self, max_levels: int, encoding_dim: int = 32):
                super().__init__()
                self.encoding = nn.Embedding(max_levels, encoding_dim)
                
            def forward(self, level: int) -> torch.Tensor:
                return self.encoding(torch.tensor(level))
                
        return LevelEncoder(self.max_levels)
        
    def _create_temporal_encoder(self) -> nn.Module:
        """Create temporal position encoder"""
        class TemporalEncoder(nn.Module):
            def __init__(self, encoding_dim: int = 32):
                super().__init__()
                self.encoding_dim = encoding_dim
                
            def forward(self, timestamp: float) -> torch.Tensor:
                # Sinusoidal encoding
                encoding = torch.zeros(self.encoding_dim)
                for i in range(0, self.encoding_dim, 2):
                    encoding[i] = np.sin(timestamp / (10000 ** (i / self.encoding_dim)))
                    encoding[i + 1] = np.cos(timestamp / (10000 ** (i / self.encoding_dim)))
                return encoding
                
        return TemporalEncoder()
        
    def _create_semantic_encoder(self) -> nn.Module:
        """Create semantic cluster position encoder"""
        class SemanticEncoder(nn.Module):
            def __init__(self, encoding_dim: int = 32):
                super().__init__()
                self.projector = nn.Linear(768, encoding_dim)  # From embedding dim
                
            def forward(self, embedding: torch.Tensor) -> torch.Tensor:
                return self.projector(embedding)
                
        return SemanticEncoder()
        
    async def build_hierarchy(
        self,
        memories: List[Dict[str, Any]],
        tenant_id: str
    ) -> Dict[int, List[str]]:
        """
        Build hierarchical memory structure from flat memories.
        
        Args:
            memories: Flat list of memories with embeddings
            tenant_id: Tenant identifier
            
        Returns:
            Dict mapping level to memory indices
        """
        # Level 0: Most detailed (individual memories)
        level_0_indices = []
        for i, memory in enumerate(memories):
            memory_id = f"{tenant_id}:L0:{i}"
            
            # Create positional encoding
            pos_encoding = PositionalEncoding(
                level_encoding=self.level_encoder(0).numpy(),
                temporal_encoding=self.temporal_encoder(memory["timestamp"]).numpy(),
                semantic_encoding=self.semantic_encoder(
                    torch.tensor(memory["embedding"])
                ).numpy()
            )
            
            level_0 = MemoryLevel(
                level=0,
                name=f"Memory_{i}",
                summary=memory["content"][:100],  # First 100 chars
                embedding=memory["embedding"],
                position_encoding=pos_encoding.to_vector()
            )
            
            self.memory_levels[0][memory_id] = level_0
            level_0_indices.append(memory_id)
            
        # Build higher levels through clustering
        current_indices = level_0_indices
        
        for level in range(1, self.max_levels):
            if len(current_indices) < 2:
                break  # Can't cluster further
                
            # Cluster memories at current level
            clusters = await self._cluster_memories(current_indices, level)
            
            next_indices = []
            for cluster_id, member_indices in clusters.items():
                parent_id = f"{tenant_id}:L{level}:{cluster_id}"
                
                # Create summary for cluster
                summary = await self._generate_cluster_summary(member_indices)
                
                # Compute cluster embedding (centroid)
                embeddings = [
                    self.memory_levels[level-1][idx].embedding
                    for idx in member_indices
                ]
                cluster_embedding = np.mean(embeddings, axis=0)
                
                # Create positional encoding
                pos_encoding = PositionalEncoding(
                    level_encoding=self.level_encoder(level).numpy(),
                    temporal_encoding=self.temporal_encoder(time.time()).numpy(),
                    semantic_encoding=self.semantic_encoder(
                        torch.tensor(cluster_embedding)
                    ).numpy()
                )
                
                parent_level = MemoryLevel(
                    level=level,
                    name=f"Cluster_L{level}_{cluster_id}",
                    summary=summary,
                    embedding=cluster_embedding,
                    child_indices=member_indices,
                    position_encoding=pos_encoding.to_vector()
                )
                
                # Update child->parent links
                for child_idx in member_indices:
                    self.memory_levels[level-1][child_idx].parent_index = parent_id
                    
                self.memory_levels[level][parent_id] = parent_level
                next_indices.append(parent_id)
                
            current_indices = next_indices
            
        # Return level mapping
        level_mapping = {}
        for level in range(self.max_levels):
            level_mapping[level] = list(self.memory_levels[level].keys())
            
        logger.info(
            "Built memory hierarchy",
            tenant_id=tenant_id,
            levels=len(level_mapping),
            total_nodes=sum(len(indices) for indices in level_mapping.values())
        )
        
        return level_mapping
        
    async def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        tenant_id: str,
        k: int = 10,
        pruning_threshold: float = 0.7
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Hierarchical search with intelligent pruning.
        
        Process:
        1. Start at highest level (most abstract)
        2. Use neural router to select promising branches
        3. Prune branches below threshold
        4. Descend to detailed memories
        5. Return top-k results with paths
        
        Args:
            query_embedding: Query vector
            tenant_id: Tenant identifier
            k: Number of results
            pruning_threshold: Minimum score to explore branch
            
        Returns:
            List of (memory_id, score, path) tuples
        """
        self.metrics["total_traversals"] += 1
        
        # Find highest level with memories
        start_level = self.max_levels - 1
        while start_level > 0 and not self.memory_levels[start_level]:
            start_level -= 1
            
        # Convert query to tensor
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        
        # Start traversal
        candidates = []
        paths_explored = []
        nodes_visited = 0
        
        # BFS with pruning
        queue = []
        
        # Initialize with all nodes at start level
        for memory_id in self.memory_levels[start_level]:
            memory = self.memory_levels[start_level][memory_id]
            if memory_id.startswith(tenant_id):
                score = self._compute_similarity(query_embedding, memory.embedding)
                queue.append((memory_id, score, [memory_id]))
                
        # Sort by score
        queue.sort(key=lambda x: x[1], reverse=True)
        
        while queue and len(candidates) < k * 3:  # Get extra for final ranking
            memory_id, score, path = queue.pop(0)
            nodes_visited += 1
            
            # Prune low-scoring branches
            if score < pruning_threshold:
                self.metrics["levels_pruned"] += 1
                continue
                
            memory = self._get_memory_by_id(memory_id)
            
            if memory.level == 0:
                # Reached leaf node
                candidates.append((memory_id, score, path))
                paths_explored.append(path)
            else:
                # Expand children
                for child_id in memory.child_indices:
                    child = self._get_memory_by_id(child_id)
                    
                    # Use neural router for smart routing
                    child_embedding_tensor = torch.tensor(child.embedding)
                    routing_result = self.router(
                        query_tensor.unsqueeze(0),
                        [child_embedding_tensor.unsqueeze(0)]
                    )
                    
                    # Combine similarity and routing score
                    similarity = self._compute_similarity(query_embedding, child.embedding)
                    routing_score = routing_result["route_probs"][0][0].item()
                    combined_score = 0.7 * similarity + 0.3 * routing_score
                    
                    if combined_score >= pruning_threshold:
                        new_path = path + [child_id]
                        queue.append((child_id, combined_score, new_path))
                        
                # Re-sort queue
                queue.sort(key=lambda x: x[1], reverse=True)
                
        # Update metrics
        self.metrics["avg_hops"] = (
            self.metrics["avg_hops"] * 0.9 + 
            np.mean([len(p) for p in paths_explored]) * 0.1
        )
        
        # Final ranking
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            "Hierarchical search complete",
            query_dims=query_embedding.shape,
            candidates=len(candidates),
            nodes_visited=nodes_visited,
            avg_path_length=np.mean([len(p) for p in paths_explored])
        )
        
        return candidates[:k]
        
    async def update_access_patterns(self, accessed_memories: List[str]):
        """Update access counts and patterns for adaptation"""
        for memory_id in accessed_memories:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                memory.access_count += 1
                memory.last_access = time.time()
                
        # Propagate access counts up the hierarchy
        for memory_id in accessed_memories:
            await self._propagate_access_count(memory_id)
            
    async def adapt_hierarchy(self, tenant_id: str):
        """Adapt hierarchy based on access patterns"""
        # Find frequently accessed paths
        hot_paths = await self._find_hot_paths(tenant_id)
        
        # Reorganize hierarchy to optimize hot paths
        for path in hot_paths:
            await self._optimize_path(path)
            
        logger.info(
            "Adapted hierarchy",
            tenant_id=tenant_id,
            hot_paths=len(hot_paths)
        )
        
    # Private methods
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryLevel]:
        """Get memory by ID from any level"""
        parts = memory_id.split(":")
        if len(parts) >= 3 and parts[1].startswith("L"):
            level = int(parts[1][1:])
            if level < self.max_levels:
                return self.memory_levels[level].get(memory_id)
        return None
        
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
        
    async def _cluster_memories(
        self,
        memory_indices: List[str],
        target_level: int
    ) -> Dict[int, List[str]]:
        """Cluster memories for hierarchy building"""
        # Simple clustering - in production use HDBSCAN or similar
        cluster_size = max(2, len(memory_indices) // 10)
        clusters = {}
        
        for i, idx in enumerate(memory_indices):
            cluster_id = i // cluster_size
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(idx)
            
        return clusters
        
    async def _generate_cluster_summary(self, member_indices: List[str]) -> str:
        """Generate summary for memory cluster"""
        # In production, use LLM to summarize
        summaries = []
        for idx in member_indices[:3]:  # First 3
            memory = self._get_memory_by_id(idx)
            if memory:
                summaries.append(memory.summary)
                
        return f"Cluster of {len(member_indices)} memories: {'; '.join(summaries[:2])}..."
        
    async def _propagate_access_count(self, memory_id: str):
        """Propagate access count up the hierarchy"""
        memory = self._get_memory_by_id(memory_id)
        if memory and memory.parent_index:
            parent = self._get_memory_by_id(memory.parent_index)
            if parent:
                parent.access_count += 1
                # Recursively propagate
                await self._propagate_access_count(memory.parent_index)
                
    async def _find_hot_paths(self, tenant_id: str) -> List[List[str]]:
        """Find frequently accessed paths in hierarchy"""
        hot_paths = []
        
        # Find memories with high access counts
        threshold = 10  # Configurable
        
        for level in range(self.max_levels):
            for memory_id, memory in self.memory_levels[level].items():
                if memory_id.startswith(tenant_id) and memory.access_count > threshold:
                    # Build path to root
                    path = [memory_id]
                    current = memory
                    while current.parent_index:
                        path.append(current.parent_index)
                        current = self._get_memory_by_id(current.parent_index)
                        if not current:
                            break
                    hot_paths.append(path)
                    
        return hot_paths
        
    async def _optimize_path(self, path: List[str]):
        """Optimize a hot path for faster access"""
        # In production, could create shortcuts or reorganize levels
        # For now, just log
        logger.debug("Would optimize path", path_length=len(path))
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            **self.metrics,
            "total_memories": sum(
                len(level) for level in self.memory_levels.values()
            ),
            "levels_used": sum(
                1 for level in self.memory_levels.values() if level
            )
        }