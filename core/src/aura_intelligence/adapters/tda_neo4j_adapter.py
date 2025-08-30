"""
TDA Neo4j Adapter - Connects Topological Data Analysis with Neo4j Graph Database
2025 Best Practices Implementation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
import numpy as np

from ..neo4j_adapter import Neo4jAdapter, Neo4jConfig
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TDAGraphMapping:
    """Mapping between TDA results and Neo4j graph structure"""
    node_labels: List[str] = None
    edge_type: str = "TOPOLOGICALLY_CONNECTED"
    persistence_property: str = "persistence"
    dimension_property: str = "dimension"
    

class TDANeo4jAdapter:
    """Adapter for storing TDA results in Neo4j"""
    
    def __init__(self, neo4j_config: Optional[Neo4jConfig] = None):
        self.neo4j = Neo4jAdapter(neo4j_config or Neo4jConfig())
        self.mapping = TDAGraphMapping()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the adapter"""
        if not self._initialized:
            await self.neo4j.initialize()
            self._initialized = True
            logger.info("TDA Neo4j adapter initialized")
            
    async def store_persistence_diagram(
        self,
        diagram_id: str,
        persistence_pairs: List[tuple],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a persistence diagram in Neo4j"""
        if not self._initialized:
            await self.initialize()
            
        # Create diagram node
        properties = {
            "diagram_id": diagram_id,
            "pair_count": len(persistence_pairs),
            "timestamp": asyncio.get_event_loop().time(),
            **(metadata or {})
        }
        
        diagram_node_id = await self.neo4j.create_node(
            labels=["PersistenceDiagram", "TDA"],
            properties=properties
        )
        
        # Create nodes for each persistence pair
        for i, (birth, death) in enumerate(persistence_pairs):
            pair_props = {
                "birth": float(birth),
                "death": float(death) if death != float('inf') else -1,
                "persistence": float(death - birth) if death != float('inf') else float('inf'),
                "index": i
            }
            
            pair_node_id = await self.neo4j.create_node(
                labels=["PersistencePair"],
                properties=pair_props
            )
            
            # Connect to diagram
            await self.neo4j.create_relationship(
                from_id=diagram_node_id,
                to_id=pair_node_id,
                rel_type="HAS_PAIR"
            )
            
        logger.info(f"Stored persistence diagram {diagram_id} with {len(persistence_pairs)} pairs")
        return diagram_node_id
        
    async def query_persistence_features(
        self,
        min_persistence: float = 0.0,
        max_persistence: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query persistence features by persistence value"""
        if not self._initialized:
            await self.initialize()
            
        query = """
        MATCH (p:PersistencePair)
        WHERE p.persistence >= $min_persistence
        """
        
        if max_persistence is not None:
            query += " AND p.persistence <= $max_persistence"
            
        query += """
        RETURN p
        ORDER BY p.persistence DESC
        LIMIT $limit
        """
        
        result = await self.neo4j.query(
            query,
            params={
                "min_persistence": min_persistence,
                "max_persistence": max_persistence,
                "limit": limit
            }
        )
        
        return result.records
        
    async def find_topological_anomalies(
        self,
        threshold: float = 2.0,
        time_window: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find topological anomalies based on persistence"""
        query = """
        MATCH (d:PersistenceDiagram)-[:HAS_PAIR]->(p:PersistencePair)
        WHERE p.persistence > $threshold
        """
        
        if time_window:
            query += " AND d.timestamp > $min_time"
            
        query += """
        RETURN d, collect(p) as anomalous_pairs
        ORDER BY d.timestamp DESC
        """
        
        params = {"threshold": threshold}
        if time_window:
            params["min_time"] = asyncio.get_event_loop().time() - time_window
            
        result = await self.neo4j.query(query, params)
        return result.records
        
    async def close(self) -> None:
        """Close the adapter"""
        await self.neo4j.close()
        self._initialized = False


__all__ = ["TDANeo4jAdapter", "TDAGraphMapping"]
