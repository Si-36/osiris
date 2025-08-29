"""
🤝 Memory System Collective Intelligence Extensions
==================================================

Adds consensus and clustering capabilities to AURA Memory System.
"""

from typing import Dict, Any, List, Optional
import time
import structlog

from .memory_api import AURAMemorySystem, MemoryEntry, MemoryTier
from ..enhancements.collective_consensus import (
    CollectiveMemoryConsensus,
    ConsensusType,
    ConsensusResult
)
from ..enhancements.semantic_clustering import (
    SemanticMemoryClustering,
    ClusteringAlgorithm,
    ClusteringResult
)

logger = structlog.get_logger(__name__)


class CollectiveMemoryExtensions:
    """Extensions for collective intelligence in memory system."""
    
    def __init__(self, memory_system: AURAMemorySystem):
        self.memory_system = memory_system
        self.consensus = CollectiveMemoryConsensus()
        self.clustering = SemanticMemoryClustering()
        
    async def build_consensus(self,
                            memory_content: Any,
                            voting_agents: Dict[str, float],
                            consensus_type: str = "weighted") -> Optional[ConsensusResult]:
        """
        Build consensus on memory content across multiple agents.
        
        Args:
            memory_content: Content to build consensus on
            voting_agents: Dict of agent_id -> confidence/weight
            consensus_type: Type of consensus (weighted, byzantine, crdt)
            
        Returns:
            ConsensusResult with consensus details
        """
        # Generate memory ID
        memory_id = f"consensus_{int(time.time() * 1000)}"
        
        # Map string to enum
        consensus_enum = ConsensusType.WEIGHTED
        if consensus_type == "byzantine":
            consensus_enum = ConsensusType.BYZANTINE
        elif consensus_type == "crdt":
            consensus_enum = ConsensusType.CRDT
            
        # Build consensus
        result = await self.consensus.build_consensus(
            memory_id=memory_id,
            content=memory_content,
            voting_agents=voting_agents,
            consensus_type=consensus_enum
        )
        
        # If consensus reached, store the memory
        if result.consensus_reached:
            await self.memory_system.store(
                content=memory_content,
                metadata={
                    "consensus": True,
                    "support_ratio": result.support_ratio,
                    "consensus_type": consensus_type,
                    "voters": len(voting_agents)
                },
                tier_preference=MemoryTier.WARM  # Important consensus memories
            )
            
        return result
    
    async def cluster_memories(self,
                             algorithm: str = "incremental",
                             min_cluster_size: int = 3) -> Optional[ClusteringResult]:
        """
        Cluster all memories using semantic similarity.
        
        Args:
            algorithm: Clustering algorithm (incremental, hierarchical, etc)
            min_cluster_size: Minimum memories per cluster
            
        Returns:
            ClusteringResult with cluster assignments
        """
        # Get all memories for clustering
        memories_to_cluster = {}
        
        # Retrieve from hot tier first
        hot_memories = await self.memory_system.tier_manager.list_tier(MemoryTier.HOT)
        for mem_id in hot_memories[:1000]:  # Limit for performance
            memory = await self.memory_system.tier_manager.retrieve(mem_id)
            if memory:
                memories_to_cluster[mem_id] = memory.content
                
        # Map string to enum
        algo_enum = ClusteringAlgorithm.INCREMENTAL
        if algorithm == "hierarchical":
            algo_enum = ClusteringAlgorithm.HIERARCHICAL
        elif algorithm == "hdbscan":
            algo_enum = ClusteringAlgorithm.HDBSCAN
            
        # Run clustering
        result = await self.clustering.cluster_memories(
            memories=memories_to_cluster,
            algorithm=algo_enum
        )
        
        # Store cluster assignments in metadata
        for cluster in result.clusters:
            for memory_id in cluster.memories:
                if memory_id in self.memory_system.entries:
                    self.memory_system.entries[memory_id].metadata["cluster_id"] = cluster.id
                    self.memory_system.entries[memory_id].metadata["cluster_cohesion"] = cluster.cohesion
                    
        logger.info("Memory clustering completed",
                   num_clusters=len(result.clusters),
                   coverage=result.quality_metrics.get("coverage", 0))
                   
        return result
    
    async def find_consensus_memories(self,
                                    min_support: float = 0.7) -> List[MemoryEntry]:
        """Find memories that have high consensus support."""
        consensus_memories = []
        
        for entry in self.memory_system.entries.values():
            if (entry.metadata.get("consensus", False) and
                entry.metadata.get("support_ratio", 0) >= min_support):
                consensus_memories.append(entry)
                
        # Sort by support ratio
        consensus_memories.sort(
            key=lambda m: m.metadata.get("support_ratio", 0),
            reverse=True
        )
        
        return consensus_memories
    
    async def get_cluster_memories(self, cluster_id: str) -> List[MemoryEntry]:
        """Get all memories in a specific cluster."""
        memory_ids = self.clustering.get_cluster_memories(cluster_id)
        memories = []
        
        for mem_id in memory_ids:
            if mem_id in self.memory_system.entries:
                memories.append(self.memory_system.entries[mem_id])
            else:
                # Try retrieving from storage
                memory = await self.memory_system.tier_manager.retrieve(mem_id)
                if memory:
                    memories.append(memory)
                    
        return memories
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collective intelligence metrics."""
        return {
            "consensus": self.consensus.get_consensus_metrics(),
            "clustering": {
                "num_clusters": len(self.clustering.incremental.micro_clusters),
                "quality_history": self.clustering.quality_history[-10:] if self.clustering.quality_history else []
            }
        }


# Convenience function to add collective features to existing memory system
def add_collective_intelligence(memory_system: AURAMemorySystem) -> CollectiveMemoryExtensions:
    """Add collective intelligence features to memory system."""
    extensions = CollectiveMemoryExtensions(memory_system)
    
    # Add methods to the memory system instance
    memory_system.build_consensus = extensions.build_consensus
    memory_system.cluster_memories = extensions.cluster_memories
    memory_system.find_consensus_memories = extensions.find_consensus_memories
    memory_system.get_cluster_memories = extensions.get_cluster_memories
    
    # Update metrics
    original_get_metrics = memory_system.get_metrics
    def enhanced_get_metrics():
        metrics = original_get_metrics()
        metrics.update(extensions.get_metrics())
        return metrics
    
    memory_system.get_metrics = enhanced_get_metrics
    
    logger.info("Collective intelligence added to memory system")
    return extensions