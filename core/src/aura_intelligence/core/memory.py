"""
Advanced Memory System for AURA Intelligence
2025 Best Practices Implementation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import asyncio

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryType(Enum):
    """Types of memory in the system"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """Individual memory entry"""
    id: str
    content: Any
    memory_type: MemoryType
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MemorySystem:
    """
    Advanced Memory System with Multiple Storage Tiers
    
    Features:
    - Multi-tier memory (short-term, long-term, episodic, semantic)
    - Importance-based retention
    - Context-aware retrieval
    - Memory consolidation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_index: Dict[MemoryType, List[str]] = {
            mem_type: [] for mem_type in MemoryType
        }
        self._initialized = False
        
    async def initialize(self):
        """Initialize memory system"""
        if self._initialized:
            return
            
        # Setup memory stores
        self._initialized = True
        logger.info("Memory system initialized")
        
    async def store(
        self,
        content: Any,
        memory_type: MemoryType,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory"""
        memory_id = f"{memory_type.value}_{datetime.utcnow().timestamp()}"
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.utcnow(),
            importance=importance,
            metadata=metadata
        )
        
        self.memories[memory_id] = entry
        self.memory_index[memory_type].append(memory_id)
        
        # Trigger consolidation if needed
        if len(self.memories) > self.config.get("consolidation_threshold", 1000):
            asyncio.create_task(self._consolidate_memories())
            
        return memory_id
        
    async def retrieve(
        self,
        query: Union[str, Dict[str, Any]],
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Retrieve memories based on query"""
        results = []
        
        # Simple retrieval - can be enhanced with semantic search
        for memory_id, memory in self.memories.items():
            if memory_types and memory.memory_type not in memory_types:
                continue
                
            # Simple relevance check
            if isinstance(query, str) and query.lower() in str(memory.content).lower():
                results.append(memory)
                memory.access_count += 1
                
        # Sort by importance and recency
        results.sort(key=lambda m: (m.importance, m.timestamp.timestamp()), reverse=True)
        
        return results[:limit]
        
    async def forget(self, memory_id: str) -> bool:
        """Remove a memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            self.memory_index[memory.memory_type].remove(memory_id)
            del self.memories[memory_id]
            return True
        return False
        
    async def _consolidate_memories(self):
        """Consolidate memories based on importance and access patterns"""
        # Remove low-importance, rarely accessed memories
        threshold_time = datetime.utcnow().timestamp() - 86400  # 24 hours
        
        to_remove = []
        for memory_id, memory in self.memories.items():
            if (memory.importance < 0.3 and 
                memory.access_count < 2 and
                memory.timestamp.timestamp() < threshold_time):
                to_remove.append(memory_id)
                
        for memory_id in to_remove:
            await self.forget(memory_id)
            
        logger.info("Consolidated memories, removed {} entries", len(to_remove))
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            "total_memories": len(self.memories),
            "by_type": {}
        }
        
        for mem_type in MemoryType:
            stats["by_type"][mem_type.value] = len(self.memory_index[mem_type])
            
        return stats


# Export main classes
__all__ = ["MemorySystem", "MemoryEntry", "MemoryType"]
