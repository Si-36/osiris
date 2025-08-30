"""
Mem0 ↔ Neo4j Bridge - 2025 Production
Hybrid memory: Semantic (Mem0) + Topological (Neo4j)
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import json

# Import our components
from ..adapters.mem0_adapter import Mem0Adapter, Memory, MemoryType, SearchQuery
from .tda_neo4j_bridge import get_tda_neo4j_bridge, TopologicalSignature

@dataclass
class HybridMemoryResult:
    semantic_memories: List[Memory]
    topological_shapes: List[Dict[str, Any]]
    combined_score: float
    retrieval_time_ms: float

class Mem0Neo4jBridge:
    """Production bridge for hybrid semantic + topological memory"""
    
    def __init__(self):
        self.mem0_adapter = None
        self.tda_bridge = get_tda_neo4j_bridge()
        self.hybrid_cache = {}
        
        async def initialize(self):
            """Initialize both Mem0 and Neo4j connections"""
        pass
        # Initialize Mem0
        from ..adapters.mem0_adapter import Mem0Config
        config = Mem0Config(base_url="http://localhost:8080")
        self.mem0_adapter = Mem0Adapter(config)
        await self.mem0_adapter.initialize()
        
        # Initialize TDA-Neo4j bridge
        await self.tda_bridge.initialize()
    
        async def store_hybrid_memory(
        self, 
        agent_id: str, 
        content: Dict[str, Any], 
        context_data: Optional[List[List[float]]] = None
        ) -> str:
            pass
        """Store memory in both Mem0 and Neo4j with topological analysis"""
        
        # Step 1: Store in Mem0
        memory = Memory(
            id="",  # Will be generated
            agent_id=agent_id,
            memory_type=MemoryType.CONTEXT,
            content=content
        )
        
        memory_id = await self.mem0_adapter.add_memory(memory)
        
        # Step 2: Extract and store topological signature
        if context_data:
            import numpy as np
            data_array = np.array(context_data)
            signature = await self.tda_bridge.extract_and_store_shape(data_array, memory_id)
            
            # Link memory to shape
            await self._link_memory_to_shape(memory_id, signature.shape_hash)
        
        return memory_id
    
        async def _link_memory_to_shape(self, memory_id: str, shape_hash: str):
            pass
        """Create link between Mem0 memory and Neo4j shape"""
        if self.tda_bridge.driver:
            async with self.tda_bridge.driver.session() as session:
                pass
        await session.run("""
        MATCH (s:Shape {betti_hash: $shape_hash})
        MERGE (m:Memory {memory_id: $memory_id})
        MERGE (m)-[:HAS_SHAPE]->(s)
        """, shape_hash=shape_hash, memory_id=memory_id)
    
        async def hybrid_search(
        self, 
        query_text: str, 
        query_context: Optional[List[List[float]]] = None,
        agent_id: Optional[str] = None,
        limit: int = 10
        ) -> HybridMemoryResult:
            pass
        """Search using both semantic and topological similarity"""
        start_time = time.perf_counter()
        
        # Step 1: Semantic search via Mem0
        search_query = SearchQuery(
            query_text=query_text,
            agent_ids=[agent_id] if agent_id else None,
            limit=limit
        )
        
        semantic_memories = await self.mem0_adapter.search_memories(search_query)
        
        # Step 2: Topological search via Neo4j
        topological_shapes = []
        if query_context:
            import numpy as np
            query_signature = await self.tda_bridge._compute_topology(np.array(query_context))
            topological_shapes = await self.tda_bridge.find_similar_shapes(query_signature, limit)
        
        # Step 3: Combine results
        combined_score = self._compute_hybrid_score(semantic_memories, topological_shapes)
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        return HybridMemoryResult(
            semantic_memories=semantic_memories,
            topological_shapes=topological_shapes,
            combined_score=combined_score,
            retrieval_time_ms=retrieval_time
        )
    
    def _compute_hybrid_score(self, semantic_memories: List[Memory], topological_shapes: List[Dict[str, Any]]) -> float:
        """Compute combined semantic + topological relevance score"""
        semantic_score = sum(m.relevance_score for m in semantic_memories) / max(len(semantic_memories), 1)
        topological_score = len(topological_shapes) / 10.0  # Normalize
        
        # Weighted combination
        return 0.7 * semantic_score + 0.3 * topological_score
    
        async def get_context_with_shapes(self, agent_id: str, window_size: int = 50) -> Dict[str, Any]:
            pass
        """Get agent context enriched with topological information"""
        
        # Get recent memories
        memories = await self.mem0_adapter.get_context_window(agent_id, window_size)
        
        # Get associated shapes
        memory_ids = [m.id for m in memories]
        shapes = await self._get_shapes_for_memories(memory_ids)
        
        return {
            'memories': [
                {
                    'id': m.id,
                    'content': m.content,
                    'timestamp': m.timestamp.isoformat(),
                    'relevance_score': m.relevance_score
                }
                for m in memories
            ],
            'topological_context': shapes,
            'context_summary': {
                'memory_count': len(memories),
                'shape_count': len(shapes),
                'time_span_hours': self._calculate_time_span(memories)
            }
        }
    
        async def _get_shapes_for_memories(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
            pass
        """Get topological shapes associated with memories"""
        if not self.tda_bridge.driver:
            return []
            
        shapes = []
        async with self.tda_bridge.driver.session() as session:
            pass
        for memory_id in memory_ids:
            pass
        result = await session.run("""
        MATCH (m:Memory {memory_id: $memory_id})-[:HAS_SHAPE]->(s:Shape)
        RETURN s.betti_numbers, s.complexity_score, s.context_id
        """, memory_id=memory_id)
                
        async for record in result:
            pass
        shapes.append({
        'memory_id': memory_id,
        'betti_numbers': record['s.betti_numbers'],
        'complexity_score': record['s.complexity_score'],
        'context_id': record['s.context_id']
        })
        
        return shapes
    
    def _calculate_time_span(self, memories: List[Memory]) -> float:
        """Calculate time span of memories in hours"""
        if len(memories) < 2:
            return 0.0
            
        timestamps = [m.timestamp for m in memories]
        time_span = max(timestamps) - min(timestamps)
        return time_span.total_seconds() / 3600.0
    
        async def cleanup_old_links(self, days_old: int = 30):
            pass
        """Clean up old memory-shape links"""
        if self.tda_bridge.driver:
            async with self.tda_bridge.driver.session() as session:
                pass
        await session.run("""
        MATCH (m:Memory)-[r:HAS_SHAPE]->(s:Shape)
        WHERE s.created_at < datetime() - duration({days: $days_old})
        DELETE r
        """, days_old=days_old)

    # Global instance
        _mem0_neo4j_bridge = None

    def get_mem0_neo4j_bridge():
        global _mem0_neo4j_bridge
        if _mem0_neo4j_bridge is None:
            pass
        _mem0_neo4j_bridge = Mem0Neo4jBridge()
        return _mem0_neo4j_bridge
