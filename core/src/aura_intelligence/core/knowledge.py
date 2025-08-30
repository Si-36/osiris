"""
Knowledge Management System for AURA Intelligence
2025 Best Practices Implementation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import asyncio

from ..utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge in the system"""
    FACT = "fact"
    RULE = "rule"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    PROCEDURE = "procedure"


@dataclass
class KnowledgeNode:
    """Individual knowledge node"""
    id: str
    content: Any
    knowledge_type: KnowledgeType
    confidence: float = 0.8
    source: Optional[str] = None
    created_at: datetime = None
    connections: Set[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.connections is None:
            self.connections = set()


class KnowledgeGraph:
    """
    Advanced Knowledge Graph System
    
    Features:
    - Multi-type knowledge representation
    - Confidence scoring
    - Relationship management
    - Query and inference capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, List[tuple[str, str, float]]] = {}  # node_id -> [(target_id, relation, weight)]
        self._initialized = False
        
    async def initialize(self):
        """Initialize knowledge graph"""
        if self._initialized:
            return
            
        self._initialized = True
        logger.info("Knowledge graph initialized")
        
    async def add_knowledge(
        self,
        content: Any,
        knowledge_type: KnowledgeType,
        confidence: float = 0.8,
        source: Optional[str] = None
    ) -> str:
        """Add knowledge to the graph"""
        node_id = f"{knowledge_type.value}_{datetime.utcnow().timestamp()}"
        
        node = KnowledgeNode(
            id=node_id,
            content=content,
            knowledge_type=knowledge_type,
            confidence=confidence,
            source=source
        )
        
        self.nodes[node_id] = node
        self.edges[node_id] = []
        
        return node_id
        
    async def add_relationship(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        weight: float = 1.0
    ):
        """Add relationship between knowledge nodes"""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges[from_id].append((to_id, relation, weight))
            self.nodes[from_id].connections.add(to_id)
            self.nodes[to_id].connections.add(from_id)
            
    async def query(
        self,
        query: Dict[str, Any],
        max_depth: int = 3
    ) -> List[KnowledgeNode]:
        """Query the knowledge graph"""
        results = []
        
        # Simple query implementation
        query_type = query.get("type")
        content_match = query.get("content", "").lower()
        
        for node in self.nodes.values():
            if query_type and node.knowledge_type != query_type:
                continue
                
            if content_match and content_match not in str(node.content).lower():
                continue
                
            results.append(node)
            
        return results
        
    async def infer(
        self,
        start_node_id: str,
        inference_type: str = "related"
    ) -> List[KnowledgeNode]:
        """Perform inference from a starting node"""
        if start_node_id not in self.nodes:
            return []
            
        visited = set()
        results = []
        
        # Simple breadth-first traversal
        queue = [(start_node_id, 0)]
        
        while queue:
            node_id, depth = queue.pop(0)
            
            if node_id in visited or depth > 3:
                continue
                
            visited.add(node_id)
            results.append(self.nodes[node_id])
            
            # Add connected nodes
            for target_id, _, _ in self.edges.get(node_id, []):
                if target_id not in visited:
                    queue.append((target_id, depth + 1))
                    
        return results[1:]  # Exclude start node
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        total_edges = sum(len(edges) for edges in self.edges.values())
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": total_edges,
            "by_type": {
                kt.value: sum(1 for n in self.nodes.values() if n.knowledge_type == kt)
                for kt in KnowledgeType
            }
        }


# Export main classes
__all__ = ["KnowledgeGraph", "KnowledgeNode", "KnowledgeType"]
