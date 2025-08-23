"""Graph Relationship Manager using Neo4j"""

from typing import List, Dict, Any, Optional
import structlog
from neo4j import AsyncGraphDatabase

logger = structlog.get_logger()


class GraphRelationshipManager:
    """Manages graph relationships in Neo4j"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.logger = logger
        
    async def initialize(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            self.logger.info("Neo4j connected successfully")
        except Exception as e:
            self.logger.warning("Neo4j connection failed", error=str(e))
            
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            
    async def create_relationship(self, key1: str, key2: str, 
                                relationship_type: str = "RELATED_TO"):
        """Create relationship between two nodes"""
        if not self.driver:
            return
            
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (a:MemoryNode {key: $key1})
                MERGE (b:MemoryNode {key: $key2})
                MERGE (a)-[r:$rel_type]->(b)
                """,
                key1=key1, key2=key2, rel_type=relationship_type
            )
            
    async def find_related(self, key: str, max_hops: int = 2) -> List[str]:
        """Find related nodes up to max_hops away"""
        if not self.driver:
            return []
            
        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH (n:MemoryNode {{key: $key}})-[*1..{max_hops}]-(m:MemoryNode)
                RETURN DISTINCT m.key as related_key
                LIMIT 100
                """,
                key=key
            )
            return [record["related_key"] async for record in result]