"""
TDA Mem0 Adapter - Stores Topological Data Analysis results in Mem0 memory system
2025 Best Practices Implementation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import asyncio
import numpy as np

from ..mem0_adapter import Mem0Adapter, Mem0AdapterConfig, MemoryType, MemoryPriority
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TDAMemoryConfig:
    """Configuration for TDA memory storage"""
    memory_type: MemoryType = MemoryType.SEMANTIC
    priority: MemoryPriority = MemoryPriority.HIGH
    max_diagrams_per_agent: int = 100
    compress_large_diagrams: bool = True
    

class TDAMem0Adapter:
    """Adapter for storing TDA results in Mem0 memory system"""
    
    def __init__(self, mem0_config: Optional[Mem0AdapterConfig] = None):
        self.mem0 = Mem0Adapter(mem0_config or Mem0AdapterConfig())
        self.config = TDAMemoryConfig()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the adapter"""
        if not self._initialized:
            await self.mem0.initialize()
            self._initialized = True
            logger.info("TDA Mem0 adapter initialized")
            
    async def store_persistence_diagram(
        self,
        agent_id: str,
        diagram_id: str,
        persistence_pairs: List[tuple],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a persistence diagram as a memory"""
        if not self._initialized:
            await self.initialize()
            
        # Prepare diagram data
        diagram_data = {
            "diagram_id": diagram_id,
            "pairs": [
                {
                    "birth": float(birth),
                    "death": float(death) if death != float('inf') else -1,
                    "persistence": float(death - birth) if death != float('inf') else float('inf')
                }
                for birth, death in persistence_pairs
            ],
            "statistics": {
                "total_pairs": len(persistence_pairs),
                "max_persistence": max(
                    (d - b for b, d in persistence_pairs if d != float('inf')),
                    default=0
                ),
                "avg_persistence": np.mean([
                    d - b for b, d in persistence_pairs 
                    if d != float('inf')
                ]) if persistence_pairs else 0
            },
            **(metadata or {})
        }
        
        # Create memory content
        content = f"Persistence diagram {diagram_id}: {len(persistence_pairs)} topological features"
        
        # Add as memory
        memory = await self.mem0.add_memory(
            agent_id=agent_id,
            content=content,
            memory_type=self.config.memory_type,
            priority=self.config.priority,
            metadata={
                "tda_data": diagram_data,
                "data_type": "persistence_diagram"
            }
        )
        
        logger.info(f"Stored TDA diagram {diagram_id} for agent {agent_id}")
        return memory.id
        
    async def retrieve_persistence_diagrams(
        self,
        agent_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve persistence diagrams from memory"""
        if not self._initialized:
            await self.initialize()
            
        # Search for TDA memories
        from ..mem0_adapter import MemoryQuery
        
        search_query = MemoryQuery(
            query=query or "persistence diagram topological",
            agent_id=agent_id,
            memory_types=[self.config.memory_type],
            limit=limit
        )
        
        memories = await self.mem0.search_memories(search_query)
        
        # Extract TDA data
        diagrams = []
        for memory in memories:
            if "tda_data" in memory.metadata:
                diagrams.append({
                    "memory_id": memory.id,
                    "diagram": memory.metadata["tda_data"],
                    "relevance_score": memory.relevance_score
                })
                
        return diagrams
        
    async def store_topological_features(
        self,
        agent_id: str,
        feature_type: str,
        features: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store extracted topological features"""
        content = f"Topological features ({feature_type}): {json.dumps(features)[:100]}..."
        
        memory = await self.mem0.add_memory(
            agent_id=agent_id,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.MEDIUM,
            metadata={
                "feature_type": feature_type,
                "features": features,
                "data_type": "topological_features",
                **(metadata or {})
            }
        )
        
        return memory.id
        
    async def find_similar_topologies(
        self,
        agent_id: str,
        reference_features: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find memories with similar topological features"""
        # Create query based on features
        feature_str = " ".join(str(v) for v in reference_features.values())
        
        from ..mem0_adapter import MemoryQuery
        
        query = MemoryQuery(
            query=f"topological features {feature_str}",
            agent_id=agent_id,
            threshold=threshold,
            limit=20
        )
        
        memories = await self.mem0.search_memories(query)
        
        # Filter for topological features
        similar = []
        for memory in memories:
            if memory.metadata.get("data_type") == "topological_features":
                similar.append({
                    "memory_id": memory.id,
                    "features": memory.metadata.get("features", {}),
                    "similarity_score": memory.relevance_score
                })
                
        return similar
        
    async def cleanup_old_diagrams(
        self,
        agent_id: str,
        keep_count: Optional[int] = None
    ) -> int:
        """Clean up old persistence diagrams"""
        keep_count = keep_count or self.config.max_diagrams_per_agent
        
        # Get all TDA memories
        memories = await self.mem0.get_agent_memories(
            agent_id=agent_id,
            memory_type=self.config.memory_type,
            limit=1000
        )
        
        # Filter for persistence diagrams
        tda_memories = [
            m for m in memories
            if m.metadata.get("data_type") == "persistence_diagram"
        ]
        
        # Sort by creation time
        tda_memories.sort(key=lambda m: m.created_at, reverse=True)
        
        # Delete old ones
        deleted = 0
        for memory in tda_memories[keep_count:]:
            if await self.mem0.delete_memory(memory.id):
                deleted += 1
                
        logger.info(f"Cleaned up {deleted} old TDA diagrams for agent {agent_id}")
        return deleted
        
    async def close(self) -> None:
        """Close the adapter"""
        await self.mem0.close()
        self._initialized = False


__all__ = ["TDAMem0Adapter", "TDAMemoryConfig"]
