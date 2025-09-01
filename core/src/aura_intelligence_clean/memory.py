"""
Memory System - Clean Implementation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio
import time
import json
from collections import defaultdict

@dataclass
class MemoryEntry:
    id: str
    content: Any
    metadata: Dict[str, Any]
    timestamp: float
    embeddings: Optional[List[float]] = None
    topology_features: Optional[Dict[str, float]] = None

class AURAMemorySystem:
    """Topological memory system without external dependencies"""
    
    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}
        self.topology_analyzer = None
        self.next_id = 0
        
        # Tiered storage simulation
        self.hot_tier = {}  # Fast access
        self.warm_tier = {}  # Medium access
        self.cold_tier = {}  # Slow access
        
    def set_topology_analyzer(self, analyzer):
        """Set the topology analyzer for advanced features"""
        self.topology_analyzer = analyzer
        
    async def store(self, data: Dict[str, Any]) -> str:
        """Store data in memory"""
        # Generate ID
        entry_id = f"mem_{self.next_id}"
        self.next_id += 1
        
        # Create entry
        entry = MemoryEntry(
            id=entry_id,
            content=data.get("content", data),
            metadata=data.get("metadata", {}),
            timestamp=time.time()
        )
        
        # Add topology features if analyzer available
        if self.topology_analyzer:
            entry.topology_features = await self._compute_topology_features(data)
            
        # Store in hot tier initially
        self.hot_tier[entry_id] = entry
        self.entries[entry_id] = entry
        
        # Schedule tier migration
        asyncio.create_task(self._migrate_tiers())
        
        return entry_id
    
    async def retrieve(self, query: Dict[str, Any]) -> List[MemoryEntry]:
        """Retrieve memories based on query"""
        results = []
        
        # Simple keyword matching for now
        keywords = query.get("keywords", [])
        limit = query.get("limit", 10)
        
        for entry in self.entries.values():
            content_str = json.dumps(entry.content).lower()
            if any(kw.lower() in content_str for kw in keywords):
                results.append(entry)
                
        # Sort by recency
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]
    
    async def _compute_topology_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute topological features for data"""
        # Mock topology features
        return {
            "centrality": 0.5,
            "clustering": 0.3,
            "persistence": 0.8
        }
    
    async def _migrate_tiers(self):
        """Migrate data between tiers based on access patterns"""
        # Simple age-based migration
        current_time = time.time()
        
        # Hot -> Warm after 60 seconds
        for entry_id, entry in list(self.hot_tier.items()):
            if current_time - entry.timestamp > 60:
                self.warm_tier[entry_id] = entry
                del self.hot_tier[entry_id]
                
        # Warm -> Cold after 300 seconds  
        for entry_id, entry in list(self.warm_tier.items()):
            if current_time - entry.timestamp > 300:
                self.cold_tier[entry_id] = entry
                del self.warm_tier[entry_id]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_entries": len(self.entries),
            "hot_tier": len(self.hot_tier),
            "warm_tier": len(self.warm_tier),
            "cold_tier": len(self.cold_tier),
            "has_topology": self.topology_analyzer is not None
        }