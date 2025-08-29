"""
Memory Manager for Collective Intelligence - 2025 Production Implementation

Features:
- Distributed memory with consensus
- Semantic memory clustering
- Temporal memory windows
- Memory consolidation and pruning
- Cross-agent memory sharing
- Causal memory chains
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import structlog
from collections import defaultdict, deque
import hashlib
import json
import heapq
import uuid

logger = structlog.get_logger(__name__)


class MemoryType(Enum):
    """Types of memory in the system"""
    EPISODIC = "episodic"          # Specific events
    SEMANTIC = "semantic"          # General knowledge
    PROCEDURAL = "procedural"      # How-to knowledge
    WORKING = "working"            # Short-term active
    CONSENSUS = "consensus"        # Agreed upon by collective
    CAUSAL = "causal"             # Cause-effect relationships


class MemoryPriority(Enum):
    """Priority levels for memory retention"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    EPHEMERAL = 1


@dataclass
class Memory:
    """Individual memory unit"""
    id: str = field(default_factory=lambda: f"mem_{hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]}")
    type: MemoryType = MemoryType.EPISODIC
    content: Dict[str, Any] = field(default_factory=dict)
    source_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    priority: MemoryPriority = MemoryPriority.MEDIUM
    confidence: float = 1.0
    embeddings: Optional[np.ndarray] = None
    linked_memories: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process memory on creation"""
        # Generate content hash for deduplication
        content_str = json.dumps(self.content, sort_keys=True)
        self.content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        # Initialize decay factor based on priority
        self.decay_factor = 0.99 - (0.1 * (5 - self.priority.value))
    
    def access(self) -> None:
        """Record memory access"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_relevance_score(self, current_time: datetime) -> float:
        """Calculate current relevance score"""
        # Time decay
        time_diff = (current_time - self.last_accessed).total_seconds() / 3600  # Hours
        time_factor = np.exp(-time_diff / 24)  # Decay over days
        
        # Access frequency factor
        access_factor = np.log1p(self.access_count) / 10
        
        # Priority factor
        priority_factor = self.priority.value / 5
        
        # Confidence factor
        confidence_factor = self.confidence
        
        # Combined score
        score = (
            0.3 * time_factor +
            0.2 * access_factor +
            0.3 * priority_factor +
            0.2 * confidence_factor
        )
        
        return float(np.clip(score, 0, 1))


@dataclass
class MemoryCluster:
    """Cluster of related memories"""
    id: str = field(default_factory=lambda: f"cluster_{uuid.uuid4().hex[:8]}")
    memories: Set[str] = field(default_factory=set)
    centroid: Optional[np.ndarray] = None
    type: MemoryType = MemoryType.SEMANTIC
    coherence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_memory(self, memory_id: str) -> None:
        """Add memory to cluster"""
        self.memories.add(memory_id)
        self.last_updated = datetime.now()
    
    def remove_memory(self, memory_id: str) -> None:
        """Remove memory from cluster"""
        self.memories.discard(memory_id)
        self.last_updated = datetime.now()


class CollectiveMemoryManager:
    """
    Advanced memory management for collective intelligence
    
    Key features:
    - Distributed consensus-based memory
    - Semantic clustering and retrieval
    - Temporal memory consolidation
    - Cross-agent memory sharing
    - Causal chain preservation
    """
    
    def __init__(self,
                 max_memories: int = 10000,
                 max_working_memory: int = 100,
                 consolidation_interval: float = 300.0,  # 5 minutes
                 consensus_threshold: float = 0.66,
                 embedding_dim: int = 768):
        self.max_memories = max_memories
        self.max_working_memory = max_working_memory
        self.consolidation_interval = consolidation_interval
        self.consensus_threshold = consensus_threshold
        self.embedding_dim = embedding_dim
        
        # Memory stores by type
        self.memories: Dict[MemoryType, Dict[str, Memory]] = {
            mem_type: {} for mem_type in MemoryType
        }
        
        # Working memory (most recent/relevant)
        self.working_memory: deque[str] = deque(maxlen=max_working_memory)
        
        # Memory clusters for semantic organization
        self.clusters: Dict[str, MemoryCluster] = {}
        self.memory_to_cluster: Dict[str, str] = {}
        
        # Agent memory ownership
        self.agent_memories: Dict[str, Set[str]] = defaultdict(set)
        
        # Consensus tracking
        self.consensus_votes: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Memory index for fast retrieval
        self.memory_index: Dict[str, str] = {}  # memory_id -> type
        
        # Consolidation task
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Collective memory manager initialized",
                   max_memories=max_memories,
                   consolidation_interval=consolidation_interval)
    
    async def start(self):
        """Start memory consolidation process"""
        if self._running:
            return
        
        self._running = True
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        logger.info("Memory consolidation started")
    
    async def stop(self):
        """Stop memory consolidation process"""
        self._running = False
        if self._consolidation_task:
            await self._consolidation_task
        logger.info("Memory consolidation stopped")
    
    async def store(self,
                   content: Dict[str, Any],
                   agent_id: str,
                   memory_type: MemoryType = MemoryType.EPISODIC,
                   priority: MemoryPriority = MemoryPriority.MEDIUM,
                   linked_to: Optional[List[str]] = None) -> str:
        """Store a new memory"""
        # Create memory
        memory = Memory(
            type=memory_type,
            content=content,
            source_agent=agent_id,
            priority=priority,
            linked_memories=set(linked_to or [])
        )
        
        # Check for duplicates
        if self._is_duplicate(memory):
            logger.debug("Duplicate memory rejected", agent=agent_id)
            return ""
        
        # Store memory
        self.memories[memory_type][memory.id] = memory
        self.memory_index[memory.id] = memory_type.value
        self.agent_memories[agent_id].add(memory.id)
        
        # Add to working memory if high priority
        if priority.value >= MemoryPriority.HIGH.value:
            self.working_memory.append(memory.id)
        
        # Trigger clustering for semantic memories
        if memory_type == MemoryType.SEMANTIC:
            await self._cluster_memory(memory)
        
        # Check memory limits
        await self._enforce_memory_limits()
        
        logger.info("Memory stored",
                   id=memory.id,
                   type=memory_type.value,
                   agent=agent_id,
                   priority=priority.name)
        
        return memory.id
    
    async def retrieve(self,
                      query: Dict[str, Any],
                      agent_id: str,
                      memory_types: Optional[List[MemoryType]] = None,
                      max_results: int = 10) -> List[Memory]:
        """Retrieve memories matching query"""
        relevant_memories = []
        current_time = datetime.now()
        
        # Search across specified memory types
        search_types = memory_types or list(MemoryType)
        
        for mem_type in search_types:
            type_memories = self.memories[mem_type]
            
            for memory in type_memories.values():
                # Calculate relevance
                relevance = self._calculate_relevance(memory, query, agent_id)
                
                if relevance > 0.1:  # Threshold
                    relevant_memories.append((relevance, memory))
        
        # Sort by relevance and recency
        relevant_memories.sort(
            key=lambda x: (x[0], x[1].get_relevance_score(current_time)),
            reverse=True
        )
        
        # Get top results
        results = []
        for _, memory in relevant_memories[:max_results]:
            memory.access()  # Update access stats
            results.append(memory)
            
            # Update working memory
            if memory.id not in self.working_memory:
                self.working_memory.append(memory.id)
        
        logger.info("Memories retrieved",
                   agent=agent_id,
                   query_keys=list(query.keys()),
                   results=len(results))
        
        return results
    
    async def share_memory(self,
                          memory_id: str,
                          from_agent: str,
                          to_agents: List[str]) -> bool:
        """Share memory between agents"""
        # Find memory
        memory = self._find_memory(memory_id)
        if not memory:
            logger.warning("Memory not found for sharing", id=memory_id)
            return False
        
        # Verify ownership
        if memory_id not in self.agent_memories[from_agent]:
            logger.warning("Agent doesn't own memory", 
                         agent=from_agent, 
                         memory=memory_id)
            return False
        
        # Share with other agents
        for agent in to_agents:
            self.agent_memories[agent].add(memory_id)
        
        # Update metadata
        memory.metadata["shared_with"] = memory.metadata.get("shared_with", []) + to_agents
        memory.metadata["shared_at"] = datetime.now().isoformat()
        
        logger.info("Memory shared",
                   id=memory_id,
                   from_agent=from_agent,
                   to_agents=to_agents)
        
        return True
    
    async def build_consensus(self,
                            content: Dict[str, Any],
                            voting_agents: Dict[str, float]) -> Optional[str]:
        """Build consensus memory from agent votes"""
        # Create consensus memory candidate
        consensus_memory = Memory(
            type=MemoryType.CONSENSUS,
            content=content,
            source_agent="collective",
            priority=MemoryPriority.HIGH
        )
        
        # Collect votes
        total_weight = sum(voting_agents.values())
        if total_weight == 0:
            return None
        
        support_weight = sum(w for w in voting_agents.values() if w > 0)
        consensus_ratio = support_weight / total_weight
        
        # Check if consensus reached
        if consensus_ratio >= self.consensus_threshold:
            consensus_memory.confidence = consensus_ratio
            
            # Store consensus memory
            memory_id = await self.store(
                content=content,
                agent_id="collective",
                memory_type=MemoryType.CONSENSUS,
                priority=MemoryPriority.HIGH
            )
            
            # Record votes
            self.consensus_votes[memory_id] = voting_agents
            
            logger.info("Consensus memory created",
                       id=memory_id,
                       support=f"{consensus_ratio:.2%}",
                       voters=len(voting_agents))
            
            return memory_id
        
        logger.debug("Consensus not reached",
                    support=f"{consensus_ratio:.2%}",
                    threshold=f"{self.consensus_threshold:.2%}")
        
        return None
    
    async def get_causal_chain(self,
                             memory_id: str,
                             max_depth: int = 5) -> List[Memory]:
        """Get causal chain of memories"""
        chain = []
        visited = set()
        
        def traverse(mid: str, depth: int):
            if depth > max_depth or mid in visited:
                return
            
            visited.add(mid)
            memory = self._find_memory(mid)
            
            if memory:
                chain.append(memory)
                
                # Follow links
                for linked_id in memory.linked_memories:
                    traverse(linked_id, depth + 1)
        
        traverse(memory_id, 0)
        
        # Sort by timestamp (causal order)
        chain.sort(key=lambda m: m.timestamp)
        
        return chain
    
    async def consolidate_memories(self):
        """Consolidate and compress memories"""
        current_time = datetime.now()
        consolidation_candidates = []
        
        # Find memories eligible for consolidation
        for mem_type in [MemoryType.EPISODIC, MemoryType.WORKING]:
            for memory in self.memories[mem_type].values():
                # Old, low-relevance memories
                if memory.get_relevance_score(current_time) < 0.3:
                    time_since_access = (current_time - memory.last_accessed).total_seconds() / 3600
                    if time_since_access > 24:  # Not accessed in 24 hours
                        consolidation_candidates.append(memory)
        
        # Group similar memories
        clusters = self._group_similar_memories(consolidation_candidates)
        
        # Consolidate each cluster
        consolidated_count = 0
        for cluster in clusters:
            if len(cluster) > 1:
                # Create consolidated memory
                consolidated = await self._create_consolidated_memory(cluster)
                
                # Remove original memories
                for memory in cluster:
                    self._remove_memory(memory.id)
                
                # Store consolidated memory
                await self.store(
                    content=consolidated["content"],
                    agent_id="consolidation",
                    memory_type=MemoryType.SEMANTIC,
                    priority=MemoryPriority.MEDIUM
                )
                
                consolidated_count += 1
        
        logger.info("Memory consolidation completed",
                   candidates=len(consolidation_candidates),
                   consolidated=consolidated_count)
    
    def _calculate_relevance(self,
                           memory: Memory,
                           query: Dict[str, Any],
                           agent_id: str) -> float:
        """Calculate memory relevance to query"""
        relevance = 0.0
        
        # Content similarity (simplified - in production use embeddings)
        content_str = json.dumps(memory.content, sort_keys=True).lower()
        query_str = json.dumps(query, sort_keys=True).lower()
        
        # Keyword matching
        query_words = set(query_str.split())
        content_words = set(content_str.split())
        overlap = len(query_words & content_words)
        
        if overlap > 0:
            relevance += overlap / len(query_words)
        
        # Agent affinity
        if memory.source_agent == agent_id:
            relevance *= 1.2  # Boost own memories
        elif memory.id in self.agent_memories[agent_id]:
            relevance *= 1.1  # Boost shared memories
        
        # Type relevance
        if "memory_type" in query and memory.type.value == query["memory_type"]:
            relevance *= 1.5
        
        # Temporal relevance
        if "time_range" in query:
            start, end = query["time_range"]
            if start <= memory.timestamp <= end:
                relevance *= 1.3
        
        return float(np.clip(relevance, 0, 1))
    
    def _is_duplicate(self, memory: Memory) -> bool:
        """Check if memory is duplicate"""
        for existing in self.memories[memory.type].values():
            if existing.content_hash == memory.content_hash:
                # Same content, boost priority if higher
                if memory.priority.value > existing.priority.value:
                    existing.priority = memory.priority
                return True
        return False
    
    def _find_memory(self, memory_id: str) -> Optional[Memory]:
        """Find memory by ID"""
        if memory_id in self.memory_index:
            mem_type = MemoryType(self.memory_index[memory_id])
            return self.memories[mem_type].get(memory_id)
        return None
    
    def _remove_memory(self, memory_id: str):
        """Remove memory from all stores"""
        memory = self._find_memory(memory_id)
        if not memory:
            return
        
        # Remove from type store
        self.memories[memory.type].pop(memory_id, None)
        
        # Remove from indices
        self.memory_index.pop(memory_id, None)
        
        # Remove from agent memories
        for agent_mems in self.agent_memories.values():
            agent_mems.discard(memory_id)
        
        # Remove from working memory
        if memory_id in self.working_memory:
            self.working_memory.remove(memory_id)
        
        # Remove from clusters
        if memory_id in self.memory_to_cluster:
            cluster_id = self.memory_to_cluster[memory_id]
            if cluster_id in self.clusters:
                self.clusters[cluster_id].remove_memory(memory_id)
            self.memory_to_cluster.pop(memory_id)
    
    async def _enforce_memory_limits(self):
        """Enforce memory capacity limits"""
        total_memories = sum(len(mems) for mems in self.memories.values())
        
        if total_memories > self.max_memories:
            # Find least relevant memories
            all_memories = []
            current_time = datetime.now()
            
            for mem_type, mems in self.memories.items():
                # Skip consensus memories (protected)
                if mem_type == MemoryType.CONSENSUS:
                    continue
                
                for memory in mems.values():
                    score = memory.get_relevance_score(current_time)
                    all_memories.append((score, memory.id))
            
            # Sort by relevance (lowest first)
            all_memories.sort(key=lambda x: x[0])
            
            # Remove lowest relevance memories
            to_remove = total_memories - int(self.max_memories * 0.9)  # Keep 90%
            
            for _, memory_id in all_memories[:to_remove]:
                self._remove_memory(memory_id)
            
            logger.info("Memory limits enforced", removed=to_remove)
    
    async def _cluster_memory(self, memory: Memory):
        """Add memory to appropriate cluster"""
        # In production, use embedding similarity
        # For now, simple clustering by content similarity
        
        best_cluster = None
        best_score = 0.0
        
        for cluster in self.clusters.values():
            if cluster.type != memory.type:
                continue
            
            # Calculate similarity to cluster
            similarity = self._calculate_cluster_similarity(memory, cluster)
            
            if similarity > best_score and similarity > 0.5:  # Threshold
                best_score = similarity
                best_cluster = cluster
        
        if best_cluster:
            best_cluster.add_memory(memory.id)
            self.memory_to_cluster[memory.id] = best_cluster.id
        else:
            # Create new cluster
            new_cluster = MemoryCluster(type=memory.type)
            new_cluster.add_memory(memory.id)
            self.clusters[new_cluster.id] = new_cluster
            self.memory_to_cluster[memory.id] = new_cluster.id
    
    def _calculate_cluster_similarity(self,
                                    memory: Memory,
                                    cluster: MemoryCluster) -> float:
        """Calculate similarity between memory and cluster"""
        if not cluster.memories:
            return 0.0
        
        # Get sample of cluster memories
        sample_size = min(5, len(cluster.memories))
        sample_ids = list(cluster.memories)[:sample_size]
        
        similarities = []
        for mem_id in sample_ids:
            other_memory = self._find_memory(mem_id)
            if other_memory:
                # Simple content overlap
                mem_content = json.dumps(memory.content, sort_keys=True)
                other_content = json.dumps(other_memory.content, sort_keys=True)
                
                mem_words = set(mem_content.lower().split())
                other_words = set(other_content.lower().split())
                
                if len(mem_words | other_words) > 0:
                    similarity = len(mem_words & other_words) / len(mem_words | other_words)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _group_similar_memories(self,
                              memories: List[Memory]) -> List[List[Memory]]:
        """Group similar memories for consolidation"""
        groups = []
        used = set()
        
        for i, mem1 in enumerate(memories):
            if i in used:
                continue
            
            group = [mem1]
            used.add(i)
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                # Check similarity
                similarity = self._calculate_memory_similarity(mem1, mem2)
                if similarity > 0.7:  # High similarity threshold
                    group.append(mem2)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_memory_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate similarity between two memories"""
        # Type must match
        if mem1.type != mem2.type:
            return 0.0
        
        # Simple content similarity
        content1 = json.dumps(mem1.content, sort_keys=True).lower()
        content2 = json.dumps(mem2.content, sort_keys=True).lower()
        
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _create_consolidated_memory(self,
                                        memories: List[Memory]) -> Dict[str, Any]:
        """Create consolidated memory from group"""
        # Extract common patterns
        all_content = {}
        
        for memory in memories:
            for key, value in memory.content.items():
                if key not in all_content:
                    all_content[key] = []
                all_content[key].append(value)
        
        # Consolidate content
        consolidated_content = {}
        for key, values in all_content.items():
            # Find most common value or average
            if all(isinstance(v, (int, float)) for v in values):
                consolidated_content[key] = np.mean(values)
            else:
                # Most common value
                value_counts = {}
                for v in values:
                    v_str = str(v)
                    value_counts[v_str] = value_counts.get(v_str, 0) + 1
                
                most_common = max(value_counts.items(), key=lambda x: x[1])
                consolidated_content[key] = most_common[0]
        
        # Metadata
        consolidated_content["_consolidated_from"] = [m.id for m in memories]
        consolidated_content["_consolidation_time"] = datetime.now().isoformat()
        
        return {
            "content": consolidated_content,
            "source_count": len(memories),
            "time_range": (
                min(m.timestamp for m in memories),
                max(m.timestamp for m in memories)
            )
        }
    
    async def _consolidation_loop(self):
        """Background consolidation process"""
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval)
                await self.consolidate_memories()
            except Exception as e:
                logger.error("Consolidation error", error=str(e))


# Example usage
async def example_memory_usage():
    """Example of using collective memory manager"""
    manager = CollectiveMemoryManager()
    await manager.start()
    
    try:
        # Agent 1 stores episodic memory
        mem_id1 = await manager.store(
            content={"event": "sensor_reading", "value": 25.5},
            agent_id="agent_1",
            memory_type=MemoryType.EPISODIC
        )
        
        # Agent 2 stores related memory
        mem_id2 = await manager.store(
            content={"event": "analysis", "sensor_value": 25.5, "status": "normal"},
            agent_id="agent_2",
            memory_type=MemoryType.SEMANTIC,
            linked_to=[mem_id1]
        )
        
        # Build consensus
        consensus_id = await manager.build_consensus(
            content={"agreed_status": "system_normal", "confidence": 0.95},
            voting_agents={
                "agent_1": 1.0,
                "agent_2": 0.9,
                "agent_3": 0.8
            }
        )
        
        # Retrieve memories
        memories = await manager.retrieve(
            query={"event": "sensor_reading"},
            agent_id="agent_1"
        )
        
        print(f"Retrieved {len(memories)} memories")
        
        # Get causal chain
        if consensus_id:
            chain = await manager.get_causal_chain(consensus_id)
            print(f"Causal chain length: {len(chain)}")
        
    finally:
        await manager.stop()
    
    return manager


if __name__ == "__main__":
    asyncio.run(example_memory_usage())