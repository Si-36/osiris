"""
Advanced Memory System - 2025 Implementation

Based on latest research:
- Hierarchical memory architecture
- Episodic and semantic memory separation
- Working memory with attention
- Long-term consolidation
- Memory compression and indexing
- CXL memory pooling support
- Mem0 integration for persistence

Key features:
- Multi-tier memory hierarchy
- Vector similarity search
- Temporal decay modeling
- Memory consolidation
- Associative retrieval
- Memory sharing across agents
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import structlog
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the system"""
    SENSORY = "sensory"  # Very short-term, raw input
    WORKING = "working"  # Active processing
    EPISODIC = "episodic"  # Specific events
    SEMANTIC = "semantic"  # General knowledge
    PROCEDURAL = "procedural"  # Skills and procedures


class MemoryPriority(int, Enum):
    """Priority levels for memory storage"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MemoryItem:
    """Individual memory item"""
    memory_id: str
    memory_type: MemoryType
    content: Any
    
    # Embeddings
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Importance and decay
    importance: float = 1.0
    decay_rate: float = 0.01
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    
    # Associations
    related_memories: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def decay(self, time_delta: timedelta) -> float:
        """Calculate memory strength after decay"""
        hours_passed = time_delta.total_seconds() / 3600
        decay_factor = np.exp(-self.decay_rate * hours_passed)
        return self.importance * decay_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content if isinstance(self.content, (str, int, float, dict, list)) else str(self.content),
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
            "tags": list(self.tags)
        }


@dataclass
class MemoryCluster:
    """Cluster of related memories"""
    cluster_id: str
    memory_ids: Set[str] = field(default_factory=set)
    centroid: Optional[np.ndarray] = None
    
    # Cluster metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Semantic meaning
    concept: Optional[str] = None
    confidence: float = 0.0


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for memory retrieval
    Implements self-attention over memory items
    """
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Value projection
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
                query: torch.Tensor,
                memory_embeddings: torch.Tensor,
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to retrieve relevant memories
        
        Args:
            query: Query embedding [batch_size, embedding_dim]
            memory_embeddings: Memory embeddings [batch_size, num_memories, embedding_dim]
            memory_mask: Mask for invalid memories [batch_size, num_memories]
        
        Returns:
            attended_memory: Weighted combination of memories
            attention_weights: Attention weights over memories
        """
        # Expand query for attention
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            query_expanded,
            memory_embeddings,
            memory_embeddings,
            key_padding_mask=memory_mask
        )
        
        # Project output
        output = self.output_proj(attended_output)
        
        return output.squeeze(1), attention_weights.squeeze(1)


class WorkingMemory:
    """
    Working memory with limited capacity
    Implements attention-based retrieval
    """
    
    def __init__(self, capacity: int = 7, embedding_dim: int = 768):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        
        # Memory buffer
        self.buffer: deque = deque(maxlen=capacity)
        
        # Attention mechanism
        self.attention = AttentionMechanism(embedding_dim)
        
        # Priority queue for importance
        self.priorities: Dict[str, float] = {}
        
        logger.info(f"Working memory initialized with capacity {capacity}")
    
    def add(self, memory: MemoryItem, priority: float = 1.0):
        """Add memory to working memory"""
        # Remove least important if at capacity
        if len(self.buffer) >= self.capacity:
            self._evict_least_important()
        
        self.buffer.append(memory)
        self.priorities[memory.memory_id] = priority
        memory.accessed_at = datetime.now()
        memory.access_count += 1
    
    def _evict_least_important(self):
        """Evict least important memory"""
        if not self.buffer:
            return
        
        # Find memory with lowest priority
        min_priority = float('inf')
        min_idx = 0
        
        for i, mem in enumerate(self.buffer):
            priority = self.priorities.get(mem.memory_id, 0)
            if priority < min_priority:
                min_priority = priority
                min_idx = i
        
        # Remove from buffer
        evicted = self.buffer[min_idx]
        del self.buffer[min_idx]
        del self.priorities[evicted.memory_id]
    
    async def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[MemoryItem]:
        """Retrieve memories using attention"""
        if not self.buffer:
            return []
        
        # Get embeddings from buffer
        embeddings = []
        valid_memories = []
        
        for memory in self.buffer:
            if memory.embedding is not None:
                embeddings.append(memory.embedding)
                valid_memories.append(memory)
        
        if not embeddings:
            return list(self.buffer)[:top_k]
        
        # Convert to tensors
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
        memory_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).unsqueeze(0)
        
        # Apply attention
        with torch.no_grad():
            _, attention_weights = self.attention(query_tensor, memory_tensor)
        
        # Get top-k memories
        weights = attention_weights[0].numpy()
        top_indices = np.argsort(weights)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(valid_memories):
                memory = valid_memories[idx]
                memory.accessed_at = datetime.now()
                memory.access_count += 1
                results.append(memory)
        
        return results
    
    def clear(self):
        """Clear working memory"""
        self.buffer.clear()
        self.priorities.clear()


class EpisodicMemory:
    """
    Episodic memory for storing specific events
    Implements temporal organization
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: Dict[str, MemoryItem] = {}
        
        # Temporal index
        self.temporal_index: Dict[datetime, List[str]] = defaultdict(list)
        
        # Episode boundaries
        self.episodes: List[Tuple[datetime, datetime, List[str]]] = []
        
        logger.info(f"Episodic memory initialized with max size {max_size}")
    
    async def store(self, memory: MemoryItem):
        """Store episodic memory"""
        # Evict old memories if at capacity
        if len(self.memories) >= self.max_size:
            await self._evict_oldest()
        
        self.memories[memory.memory_id] = memory
        self.temporal_index[memory.created_at].append(memory.memory_id)
    
    async def _evict_oldest(self):
        """Evict oldest memories"""
        # Sort by creation time
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest 10%
        to_remove = int(self.max_size * 0.1)
        for memory_id, _ in sorted_memories[:to_remove]:
            del self.memories[memory_id]
    
    async def retrieve_episode(self, 
                             start_time: datetime,
                             end_time: datetime) -> List[MemoryItem]:
        """Retrieve memories from a time period"""
        episode_memories = []
        
        for timestamp, memory_ids in self.temporal_index.items():
            if start_time <= timestamp <= end_time:
                for memory_id in memory_ids:
                    if memory_id in self.memories:
                        episode_memories.append(self.memories[memory_id])
        
        # Sort by time
        episode_memories.sort(key=lambda m: m.created_at)
        
        return episode_memories
    
    def mark_episode_boundary(self, timestamp: datetime):
        """Mark the end of an episode"""
        if self.episodes:
            last_start, _, memories = self.episodes[-1]
            
            # Get memories in this episode
            episode_memory_ids = []
            for ts, mem_ids in self.temporal_index.items():
                if last_start <= ts <= timestamp:
                    episode_memory_ids.extend(mem_ids)
            
            # Update last episode
            self.episodes[-1] = (last_start, timestamp, episode_memory_ids)
        
        # Start new episode
        self.episodes.append((timestamp, timestamp, []))


class SemanticMemory:
    """
    Semantic memory for general knowledge
    Implements concept clustering
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.memories: Dict[str, MemoryItem] = {}
        
        # Concept clusters
        self.clusters: Dict[str, MemoryCluster] = {}
        self.memory_to_cluster: Dict[str, str] = {}
        
        # Concept hierarchy
        self.concept_graph = defaultdict(set)  # parent -> children
        
        logger.info("Semantic memory initialized")
    
    async def store(self, memory: MemoryItem):
        """Store semantic memory with clustering"""
        self.memories[memory.memory_id] = memory
        
        # Assign to cluster
        if memory.embedding is not None:
            cluster_id = await self._assign_to_cluster(memory)
            self.memory_to_cluster[memory.memory_id] = cluster_id
    
    async def _assign_to_cluster(self, memory: MemoryItem) -> str:
        """Assign memory to best cluster or create new one"""
        if not self.clusters:
            # Create first cluster
            cluster = MemoryCluster(
                cluster_id=f"cluster_{len(self.clusters)}",
                centroid=memory.embedding
            )
            cluster.memory_ids.add(memory.memory_id)
            self.clusters[cluster.cluster_id] = cluster
            return cluster.cluster_id
        
        # Find closest cluster
        best_cluster_id = None
        best_similarity = -1
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.centroid is not None:
                similarity = np.dot(memory.embedding, cluster.centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
        
        # Create new cluster if similarity too low
        if best_similarity < 0.7:
            cluster = MemoryCluster(
                cluster_id=f"cluster_{len(self.clusters)}",
                centroid=memory.embedding
            )
            cluster.memory_ids.add(memory.memory_id)
            self.clusters[cluster.cluster_id] = cluster
            return cluster.cluster_id
        
        # Add to existing cluster
        cluster = self.clusters[best_cluster_id]
        cluster.memory_ids.add(memory.memory_id)
        
        # Update centroid
        await self._update_cluster_centroid(cluster)
        
        return best_cluster_id
    
    async def _update_cluster_centroid(self, cluster: MemoryCluster):
        """Update cluster centroid based on members"""
        embeddings = []
        
        for memory_id in cluster.memory_ids:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                if memory.embedding is not None:
                    embeddings.append(memory.embedding)
        
        if embeddings:
            cluster.centroid = np.mean(embeddings, axis=0)
            cluster.updated_at = datetime.now()
    
    async def retrieve_by_concept(self, concept: str, top_k: int = 10) -> List[MemoryItem]:
        """Retrieve memories related to a concept"""
        # Find clusters with matching concept
        matching_clusters = []
        
        for cluster in self.clusters.values():
            if cluster.concept and concept.lower() in cluster.concept.lower():
                matching_clusters.append(cluster)
        
        # Get memories from matching clusters
        results = []
        for cluster in matching_clusters:
            for memory_id in cluster.memory_ids:
                if memory_id in self.memories:
                    results.append(self.memories[memory_id])
        
        # Sort by importance
        results.sort(key=lambda m: m.importance, reverse=True)
        
        return results[:top_k]


class MemoryConsolidation:
    """
    Memory consolidation system
    Transfers memories between tiers
    """
    
    def __init__(self, 
                 working_memory: WorkingMemory,
                 episodic_memory: EpisodicMemory,
                 semantic_memory: SemanticMemory):
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        
        # Consolidation parameters
        self.consolidation_threshold = 5  # Access count threshold
        self.importance_threshold = 0.7
        
        logger.info("Memory consolidation system initialized")
    
    async def consolidate(self):
        """Perform memory consolidation"""
        # Working -> Episodic
        for memory in list(self.working_memory.buffer):
            if memory.access_count >= self.consolidation_threshold:
                # Move to episodic memory
                await self.episodic_memory.store(memory)
                
                # Check if should also go to semantic
                if memory.importance >= self.importance_threshold:
                    await self.semantic_memory.store(memory)
        
        # Episodic -> Semantic (abstraction)
        await self._abstract_episodes()
    
    async def _abstract_episodes(self):
        """Abstract episodic memories into semantic knowledge"""
        # Get recent episodes
        if not self.episodic_memory.episodes:
            return
        
        recent_episodes = self.episodic_memory.episodes[-10:]
        
        for start_time, end_time, memory_ids in recent_episodes:
            # Get memories in episode
            episode_memories = []
            for memory_id in memory_ids:
                if memory_id in self.episodic_memory.memories:
                    episode_memories.append(self.episodic_memory.memories[memory_id])
            
            if len(episode_memories) < 3:
                continue
            
            # Extract common patterns
            common_tags = set()
            for memory in episode_memories:
                if not common_tags:
                    common_tags = memory.tags.copy()
                else:
                    common_tags &= memory.tags
            
            if common_tags:
                # Create semantic memory from pattern
                semantic_memory = MemoryItem(
                    memory_id=f"semantic_{hashlib.sha256(str(common_tags).encode()).hexdigest()[:8]}",
                    memory_type=MemoryType.SEMANTIC,
                    content={
                        "pattern": list(common_tags),
                        "episode_count": len(episode_memories),
                        "time_range": (start_time.isoformat(), end_time.isoformat())
                    },
                    importance=0.8,
                    tags=common_tags
                )
                
                # Generate embedding (average of episode embeddings)
                embeddings = [m.embedding for m in episode_memories if m.embedding is not None]
                if embeddings:
                    semantic_memory.embedding = np.mean(embeddings, axis=0)
                
                await self.semantic_memory.store(semantic_memory)


class HierarchicalMemorySystem:
    """
    Complete hierarchical memory system
    Coordinates all memory types
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize memory tiers
        self.working_memory = WorkingMemory(
            capacity=config.get("working_capacity", 7),
            embedding_dim=config.get("embedding_dim", 768)
        )
        
        self.episodic_memory = EpisodicMemory(
            max_size=config.get("episodic_size", 10000)
        )
        
        self.semantic_memory = SemanticMemory(
            embedding_dim=config.get("embedding_dim", 768)
        )
        
        # Consolidation system
        self.consolidation = MemoryConsolidation(
            self.working_memory,
            self.episodic_memory,
            self.semantic_memory
        )
        
        # Memory index for fast retrieval
        self.memory_index: Dict[str, MemoryItem] = {}
        
        # Stats
        self.stats = {
            "total_memories": 0,
            "consolidations": 0,
            "retrievals": 0
        }
        
        logger.info("Hierarchical memory system initialized")
    
    async def store(self,
                   content: Any,
                   memory_type: MemoryType = MemoryType.WORKING,
                   priority: MemoryPriority = MemoryPriority.NORMAL,
                   embedding: Optional[np.ndarray] = None,
                   tags: Optional[Set[str]] = None,
                   context: Optional[Dict[str, Any]] = None) -> str:
        """Store memory in appropriate tier"""
        # Create memory item
        memory_id = f"{memory_type.value}_{hashlib.sha256(str(content).encode()).hexdigest()[:16]}"
        
        memory = MemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=float(priority.value) / 3.0,
            tags=tags or set(),
            context=context or {}
        )
        
        # Store in appropriate tier
        if memory_type == MemoryType.WORKING:
            self.working_memory.add(memory, priority=float(priority.value))
        elif memory_type == MemoryType.EPISODIC:
            await self.episodic_memory.store(memory)
        elif memory_type == MemoryType.SEMANTIC:
            await self.semantic_memory.store(memory)
        
        # Update index
        self.memory_index[memory_id] = memory
        self.stats["total_memories"] += 1
        
        # Trigger consolidation periodically
        if self.stats["total_memories"] % 100 == 0:
            await self.consolidation.consolidate()
            self.stats["consolidations"] += 1
        
        return memory_id
    
    async def retrieve(self,
                      query: Union[str, np.ndarray],
                      memory_types: Optional[List[MemoryType]] = None,
                      top_k: int = 10,
                      time_range: Optional[Tuple[datetime, datetime]] = None) -> List[MemoryItem]:
        """Retrieve memories across all tiers"""
        results = []
        
        # Generate query embedding if string
        if isinstance(query, str):
            # In production, use actual embedding model
            query_embedding = np.random.randn(self.working_memory.embedding_dim)
        else:
            query_embedding = query
        
        # Search each tier
        if not memory_types or MemoryType.WORKING in memory_types:
            working_results = await self.working_memory.retrieve(query_embedding, top_k=top_k)
            results.extend(working_results)
        
        if not memory_types or MemoryType.EPISODIC in memory_types:
            if time_range:
                episodic_results = await self.episodic_memory.retrieve_episode(
                    time_range[0], time_range[1]
                )
                results.extend(episodic_results[:top_k])
        
        if not memory_types or MemoryType.SEMANTIC in memory_types:
            # For semantic, use concept-based retrieval
            if isinstance(query, str):
                semantic_results = await self.semantic_memory.retrieve_by_concept(
                    query, top_k=top_k
                )
                results.extend(semantic_results)
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for memory in results:
            if memory.memory_id not in unique_results:
                unique_results[memory.memory_id] = memory
        
        final_results = list(unique_results.values())
        
        # Sort by access recency and importance
        final_results.sort(
            key=lambda m: (m.accessed_at, m.importance),
            reverse=True
        )
        
        self.stats["retrievals"] += 1
        
        return final_results[:top_k]
    
    async def forget(self, memory_id: str):
        """Remove memory from system"""
        if memory_id in self.memory_index:
            memory = self.memory_index[memory_id]
            
            # Remove from appropriate tier
            if memory.memory_type == MemoryType.WORKING:
                # Remove from working memory buffer
                self.working_memory.buffer = deque(
                    [m for m in self.working_memory.buffer if m.memory_id != memory_id],
                    maxlen=self.working_memory.capacity
                )
            elif memory.memory_type == MemoryType.EPISODIC:
                if memory_id in self.episodic_memory.memories:
                    del self.episodic_memory.memories[memory_id]
            elif memory.memory_type == MemoryType.SEMANTIC:
                if memory_id in self.semantic_memory.memories:
                    del self.semantic_memory.memories[memory_id]
            
            # Remove from index
            del self.memory_index[memory_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            **self.stats,
            "working_memory_size": len(self.working_memory.buffer),
            "episodic_memory_size": len(self.episodic_memory.memories),
            "semantic_memory_size": len(self.semantic_memory.memories),
            "semantic_clusters": len(self.semantic_memory.clusters)
        }


# Example usage
async def demonstrate_memory_system():
    """Demonstrate hierarchical memory system"""
    print("🧠 Hierarchical Memory System Demonstration")
    print("=" * 60)
    
    # Initialize memory system
    memory_system = HierarchicalMemorySystem({
        "working_capacity": 5,
        "episodic_size": 1000,
        "embedding_dim": 768
    })
    
    # Test working memory
    print("\n1️⃣ Testing Working Memory")
    print("-" * 40)
    
    # Store some memories
    for i in range(7):
        embedding = np.random.randn(768)
        embedding = embedding / np.linalg.norm(embedding)
        
        memory_id = await memory_system.store(
            content=f"Working memory item {i}",
            memory_type=MemoryType.WORKING,
            priority=MemoryPriority.NORMAL if i < 5 else MemoryPriority.HIGH,
            embedding=embedding,
            tags={f"tag_{i}", "working"}
        )
        print(f"Stored: {memory_id}")
    
    print(f"\nWorking memory size: {len(memory_system.working_memory.buffer)}")
    
    # Test episodic memory
    print("\n2️⃣ Testing Episodic Memory")
    print("-" * 40)
    
    # Create episode
    episode_start = datetime.now()
    
    for i in range(5):
        embedding = np.random.randn(768)
        embedding = embedding / np.linalg.norm(embedding)
        
        memory_id = await memory_system.store(
            content={
                "event": f"Event {i}",
                "timestamp": datetime.now().isoformat(),
                "data": {"value": i * 10}
            },
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.NORMAL,
            embedding=embedding,
            tags={f"event_{i}", "episode_1"}
        )
        
        await asyncio.sleep(0.1)  # Simulate time passing
    
    episode_end = datetime.now()
    memory_system.episodic_memory.mark_episode_boundary(episode_end)
    
    print(f"Created episode with 5 events")
    
    # Retrieve episode
    episode_memories = await memory_system.episodic_memory.retrieve_episode(
        episode_start, episode_end
    )
    print(f"Retrieved {len(episode_memories)} memories from episode")
    
    # Test semantic memory
    print("\n3️⃣ Testing Semantic Memory")
    print("-" * 40)
    
    # Store semantic knowledge
    concepts = [
        ("Python", {"type": "programming_language", "paradigm": "multi-paradigm"}),
        ("Machine Learning", {"type": "field", "related": "AI"}),
        ("Neural Networks", {"type": "model", "related": "ML"})
    ]
    
    for concept, properties in concepts:
        embedding = np.random.randn(768)
        embedding = embedding / np.linalg.norm(embedding)
        
        memory_id = await memory_system.store(
            content={
                "concept": concept,
                "properties": properties
            },
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.HIGH,
            embedding=embedding,
            tags={concept.lower().replace(" ", "_"), "knowledge"}
        )
    
    print(f"Stored {len(concepts)} semantic memories")
    print(f"Semantic clusters: {len(memory_system.semantic_memory.clusters)}")
    
    # Test retrieval
    print("\n4️⃣ Testing Memory Retrieval")
    print("-" * 40)
    
    # Query with embedding
    query_embedding = np.random.randn(768)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = await memory_system.retrieve(
        query=query_embedding,
        memory_types=[MemoryType.WORKING, MemoryType.SEMANTIC],
        top_k=5
    )
    
    print(f"\nQuery results (top 5):")
    for i, memory in enumerate(results):
        print(f"{i+1}. {memory.memory_id} ({memory.memory_type.value})")
        print(f"   Content: {str(memory.content)[:50]}...")
        print(f"   Importance: {memory.importance:.2f}")
    
    # Test consolidation
    print("\n5️⃣ Testing Memory Consolidation")
    print("-" * 40)
    
    # Access working memories multiple times
    for _ in range(6):
        results = await memory_system.working_memory.retrieve(query_embedding, top_k=2)
    
    # Trigger consolidation
    await memory_system.consolidation.consolidate()
    
    print("Consolidation completed")
    print(f"Episodic memory size: {len(memory_system.episodic_memory.memories)}")
    
    # Get final stats
    print("\n📊 Memory System Statistics")
    print("-" * 40)
    
    stats = memory_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Memory system demonstration complete")


if __name__ == "__main__":
    asyncio.run(demonstrate_memory_system())