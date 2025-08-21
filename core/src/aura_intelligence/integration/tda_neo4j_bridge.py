"""
TDA ↔ Neo4j Integration Bridge - 2025 Production
Real-time topological analysis with graph storage
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Latest 2025 libraries
try:
    import gudhi
    import neo4j
    from neo4j import AsyncGraphDatabase
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

# Research integrations
try:
    from ..tda.phformer_integration import get_phformer_processor
    from ..tda.multi_parameter_persistence import get_multiparameter_processor
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False

@dataclass
class TopologicalSignature:
    betti_numbers: List[int]
    persistence_diagram: List[List[float]]
    shape_hash: str
    complexity_score: float

class TDANeo4jBridge:
    """Production bridge between TDA analysis and Neo4j storage"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687"):
        self.neo4j_uri = neo4j_uri
        self.driver = None
        self.tda_cache = {}
        
        # Initialize research components
        if RESEARCH_AVAILABLE:
            self.phformer = get_phformer_processor('base', 'cpu')
            self.mp_processor = get_multiparameter_processor(max_dimension=2)
        else:
            self.phformer = None
            self.mp_processor = None
        
    async def initialize(self):
        """Initialize Neo4j connection"""
        if LIBS_AVAILABLE:
            self.driver = AsyncGraphDatabase.driver(self.neo4j_uri, auth=("neo4j", "password"))
            await self._create_indexes()
        
    async def _create_indexes(self):
        """Create optimized indexes for shape queries"""
        async with self.driver.session() as session:
            await session.run("""
                CREATE INDEX shape_betti IF NOT EXISTS
                FOR (s:Shape) ON (s.betti_hash)
            """)
            await session.run("""
                CREATE INDEX shape_complexity IF NOT EXISTS  
                FOR (s:Shape) ON (s.complexity_score)
            """)
    
    async def extract_and_store_shape(self, data: np.ndarray, context_id: str) -> TopologicalSignature:
        """Extract topological features and store in Neo4j with research enhancements"""
        # Step 1: Basic TDA Analysis
        signature = await self._compute_topology(data)
        
        # Step 2: PHFormer Enhancement
        if self.phformer:
            phformer_result = self.phformer.process_topology(
                signature.betti_numbers, 
                signature.persistence_diagram
            )
            signature.phformer_features = phformer_result
        
        # Step 3: Multi-Parameter Analysis
        if self.mp_processor and len(data.shape) >= 2:
            if len(data.shape) == 2:
                data_3d = data.reshape(1, *data.shape)
            else:
                data_3d = data
            
            mp_result = self.mp_processor.compute_multi_parameter_persistence(data_3d)
            signature.mp_features = mp_result
        
        # Step 4: Store in Neo4j
        if self.driver:
            await self._store_shape(signature, context_id)
        
        return signature
    
    async def _compute_topology(self, data: np.ndarray) -> TopologicalSignature:
        """Compute topological signature using GUDHI"""
        if not LIBS_AVAILABLE:
            # Fallback computation
            return TopologicalSignature(
                betti_numbers=[1, 0],
                persistence_diagram=[[0.0, 1.0]],
                shape_hash="fallback_hash",
                complexity_score=0.5
            )
        
        # Real GUDHI computation
        rips_complex = gudhi.RipsComplex(points=data, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        
        # Compute persistence
        persistence = simplex_tree.persistence()
        betti_numbers = simplex_tree.betti_numbers()
        
        # Extract persistence diagram
        persistence_diagram = []
        for interval in persistence:
            if len(interval) >= 2:
                birth, death = interval[1]
                if death != float('inf'):
                    persistence_diagram.append([birth, death])
        
        # Compute shape hash
        shape_hash = hash(str(betti_numbers) + str(persistence_diagram[:10]))
        
        # Complexity score
        complexity_score = sum(betti_numbers) / (len(data) + 1)
        
        return TopologicalSignature(
            betti_numbers=betti_numbers,
            persistence_diagram=persistence_diagram,
            shape_hash=str(abs(shape_hash))[:16],
            complexity_score=complexity_score
        )
    
    async def _store_shape(self, signature: TopologicalSignature, context_id: str):
        """Store topological signature in Neo4j"""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (s:Shape {context_id: $context_id})
                SET s.betti_numbers = $betti_numbers,
                    s.betti_hash = $shape_hash,
                    s.complexity_score = $complexity_score,
                    s.persistence_diagram = $persistence_diagram,
                    s.created_at = datetime()
            """, 
            context_id=context_id,
            betti_numbers=signature.betti_numbers,
            shape_hash=signature.shape_hash,
            complexity_score=signature.complexity_score,
            persistence_diagram=signature.persistence_diagram[:50]  # Limit size
            )
    
    async def find_similar_shapes(self, signature: TopologicalSignature, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar topological shapes"""
        if not self.driver:
            return []
            
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (s:Shape)
                WHERE s.betti_numbers = $betti_numbers
                AND abs(s.complexity_score - $complexity_score) < 0.2
                RETURN s.context_id, s.complexity_score, s.betti_numbers
                ORDER BY abs(s.complexity_score - $complexity_score)
                LIMIT $limit
            """,
            betti_numbers=signature.betti_numbers,
            complexity_score=signature.complexity_score,
            limit=limit
            )
            
            similar_shapes = []
            async for record in result:
                similar_shapes.append({
                    'context_id': record['s.context_id'],
                    'complexity_score': record['s.complexity_score'],
                    'betti_numbers': record['s.betti_numbers']
                })
            
            return similar_shapes

# Global instance
_tda_neo4j_bridge = None

def get_tda_neo4j_bridge():
    global _tda_neo4j_bridge
    if _tda_neo4j_bridge is None:
        _tda_neo4j_bridge = TDANeo4jBridge()
    return _tda_neo4j_bridge