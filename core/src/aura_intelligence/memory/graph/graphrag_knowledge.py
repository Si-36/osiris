"""
🕸️ GraphRAG Knowledge Integration

Extracts the BEST from enterprise/knowledge_graph.py:
- Multi-hop reasoning across knowledge graphs
- Causal chain discovery
- Entity relationship mapping
- Temporal graph analysis
- Pattern prediction with Graph ML

This is Microsoft's GraphRAG approach - connecting information
to synthesize new knowledge!
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import structlog
import networkx as nx
from collections import defaultdict

logger = structlog.get_logger()


# ======================
# Core Types
# ======================

class EntityType(str, Enum):
    """Types of entities in knowledge graph"""
    AGENT = "agent"
    CONCEPT = "concept"
    EVENT = "event"
    LOCATION = "location"
    SYSTEM = "system"
    PATTERN = "pattern"
    OUTCOME = "outcome"


class RelationType(str, Enum):
    """Types of relationships"""
    CAUSES = "causes"
    REQUIRES = "requires"
    PRODUCES = "produces"
    RELATES_TO = "relates_to"
    PART_OF = "part_of"
    TEMPORAL_BEFORE = "before"
    TEMPORAL_AFTER = "after"
    SIMILAR_TO = "similar_to"


@dataclass
class Entity:
    """Knowledge graph entity"""
    entity_id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    source: str = ""
    
    def __hash__(self):
        return hash(self.entity_id)
    
    def __eq__(self, other):
        return isinstance(other, Entity) and self.entity_id == other.entity_id


@dataclass
class Relationship:
    """Relationship between entities"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence: List[str] = field(default_factory=list)


@dataclass
class CausalChain:
    """Chain of causal relationships"""
    chain_id: str
    entities: List[Entity]
    relationships: List[Relationship]
    
    # Analysis
    root_cause: Optional[Entity] = None
    final_outcome: Optional[Entity] = None
    confidence: float = 0.0
    
    def length(self) -> int:
        return len(self.relationships)
    
    def to_path(self) -> List[str]:
        """Convert to entity path"""
        path = []
        for rel in self.relationships:
            if not path:
                path.append(rel.source_id)
            path.append(rel.target_id)
        return path


@dataclass
class KnowledgeSynthesis:
    """Synthesized knowledge from graph analysis"""
    synthesis_id: str
    query: str
    
    # Results
    entities: List[Entity]
    relationships: List[Relationship]
    causal_chains: List[CausalChain]
    
    # Insights
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_path: List[str] = field(default_factory=list)


# ======================
# GraphRAG Engine
# ======================

class GraphRAGEngine:
    """
    GraphRAG engine for knowledge synthesis.
    
    Features:
    - Multi-hop reasoning
    - Causal discovery
    - Pattern recognition
    - Knowledge synthesis
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.entity_index: Dict[EntityType, Set[str]] = defaultdict(set)
        
        # Analysis cache
        self.causal_cache: Dict[str, List[CausalChain]] = {}
        self.pattern_cache: Dict[str, List[Entity]] = {}
        
        logger.info("GraphRAG engine initialized")
    
    # ======================
    # Graph Building
    # ======================
    
    async def add_entity(self, entity: Entity) -> None:
        """Add entity to knowledge graph"""
        self.entities[entity.entity_id] = entity
        self.entity_index[entity.entity_type].add(entity.entity_id)
        
        self.graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            properties=entity.properties,
            confidence=entity.confidence
        )
        
        logger.debug(
            f"Added entity",
            entity_id=entity.entity_id,
            type=entity.entity_type.value
        )
    
    async def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to knowledge graph"""
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            relation_type=relationship.relation_type.value,
            properties=relationship.properties,
            confidence=relationship.confidence,
            timestamp=relationship.timestamp
        )
        
        # Invalidate relevant caches
        self._invalidate_cache(relationship.source_id, relationship.target_id)
        
        logger.debug(
            f"Added relationship",
            source=relationship.source_id,
            target=relationship.target_id,
            type=relationship.relation_type.value
        )
    
    # ======================
    # Multi-Hop Reasoning
    # ======================
    
    async def multi_hop_query(
        self,
        start_entity: str,
        target_type: Optional[EntityType] = None,
        max_hops: int = 3,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[Tuple[Entity, List[str], float]]:
        """
        Perform multi-hop reasoning from start entity.
        
        Args:
            start_entity: Starting entity ID
            target_type: Target entity type to find
            max_hops: Maximum hops to traverse
            relation_types: Allowed relationship types
            
        Returns:
            List of (entity, path, confidence) tuples
        """
        if start_entity not in self.graph:
            return []
        
        results = []
        visited = set()
        
        # BFS with path tracking
        queue = [(start_entity, [start_entity], 1.0)]
        
        while queue:
            current, path, confidence = queue.pop(0)
            
            if len(path) > max_hops:
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check if we found target
            if current in self.entities:
                entity = self.entities[current]
                if target_type is None or entity.entity_type == target_type:
                    if len(path) > 1:  # Not the start entity
                        results.append((entity, path, confidence))
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                
                # Check relation type filter
                if relation_types:
                    valid_edge = False
                    for _, edge_attrs in edge_data.items():
                        if edge_attrs.get("relation_type") in [r.value for r in relation_types]:
                            valid_edge = True
                            break
                    if not valid_edge:
                        continue
                
                # Calculate path confidence
                edge_confidence = max(
                    edge_attrs.get("confidence", 0.5)
                    for edge_attrs in edge_data.values()
                )
                new_confidence = confidence * edge_confidence
                
                if new_confidence > 0.1:  # Confidence threshold
                    queue.append((neighbor, path + [neighbor], new_confidence))
        
        # Sort by confidence
        results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(
            f"Multi-hop query found {len(results)} results",
            start=start_entity,
            max_hops=max_hops,
            target_type=target_type
        )
        
        return results
    
    # ======================
    # Causal Discovery
    # ======================
    
    async def discover_causal_chains(
        self,
        outcome_entity: str,
        max_depth: int = 5
    ) -> List[CausalChain]:
        """
        Discover causal chains leading to an outcome.
        
        Args:
            outcome_entity: Outcome entity ID
            max_depth: Maximum chain depth
            
        Returns:
            List of causal chains
        """
        # Check cache
        cache_key = f"{outcome_entity}:{max_depth}"
        if cache_key in self.causal_cache:
            return self.causal_cache[cache_key]
        
        chains = []
        
        # Find all paths with CAUSES relationships
        causal_paths = await self._find_causal_paths(
            outcome_entity,
            max_depth
        )
        
        # Convert paths to causal chains
        for path, confidence in causal_paths:
            entities = []
            relationships = []
            
            # Build chain
            for i in range(len(path)):
                if path[i] in self.entities:
                    entities.append(self.entities[path[i]])
                
                if i < len(path) - 1:
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        # Get CAUSES relationship
                        for _, attrs in edge_data.items():
                            if attrs.get("relation_type") == RelationType.CAUSES.value:
                                rel = Relationship(
                                    source_id=path[i],
                                    target_id=path[i+1],
                                    relation_type=RelationType.CAUSES,
                                    properties=attrs.get("properties", {}),
                                    confidence=attrs.get("confidence", 1.0)
                                )
                                relationships.append(rel)
                                break
            
            if entities and relationships:
                chain = CausalChain(
                    chain_id=f"chain_{outcome_entity}_{len(chains)}",
                    entities=entities,
                    relationships=relationships,
                    root_cause=entities[0] if entities else None,
                    final_outcome=entities[-1] if entities else None,
                    confidence=confidence
                )
                chains.append(chain)
        
        # Sort by confidence and length
        chains.sort(key=lambda c: (c.confidence, -c.length()), reverse=True)
        
        # Cache results
        self.causal_cache[cache_key] = chains
        
        logger.info(
            f"Discovered {len(chains)} causal chains",
            outcome=outcome_entity,
            max_depth=max_depth
        )
        
        return chains
    
    async def _find_causal_paths(
        self,
        target: str,
        max_depth: int
    ) -> List[Tuple[List[str], float]]:
        """Find all causal paths to target"""
        paths = []
        
        # Reverse BFS (from outcome to causes)
        queue = [([target], 1.0)]
        visited_paths = set()
        
        while queue:
            path, confidence = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            path_key = tuple(path)
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)
            
            # Find predecessors with CAUSES relationship
            current = path[0]
            if current in self.graph:
                for pred in self.graph.predecessors(current):
                    edge_data = self.graph.get_edge_data(pred, current)
                    
                    # Check for CAUSES relationship
                    for _, attrs in edge_data.items():
                        if attrs.get("relation_type") == RelationType.CAUSES.value:
                            edge_conf = attrs.get("confidence", 0.5)
                            new_conf = confidence * edge_conf
                            
                            if new_conf > 0.1:
                                new_path = [pred] + path
                                queue.append((new_path, new_conf))
                                
                                # Add complete path
                                if len(new_path) > 1:
                                    paths.append((new_path, new_conf))
        
        return paths
    
    # ======================
    # Pattern Recognition
    # ======================
    
    async def find_similar_patterns(
        self,
        pattern_entity: Entity,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities with similar patterns.
        
        Args:
            pattern_entity: Pattern to match
            similarity_threshold: Minimum similarity
            
        Returns:
            List of (entity, similarity) tuples
        """
        similar = []
        
        # Get pattern properties
        pattern_props = pattern_entity.properties
        
        # Compare with other entities of same type
        for entity_id in self.entity_index[pattern_entity.entity_type]:
            if entity_id == pattern_entity.entity_id:
                continue
            
            entity = self.entities[entity_id]
            similarity = self._calculate_similarity(
                pattern_props,
                entity.properties
            )
            
            if similarity >= similarity_threshold:
                similar.append((entity, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Found {len(similar)} similar patterns",
            pattern=pattern_entity.entity_id,
            threshold=similarity_threshold
        )
        
        return similar
    
    def _calculate_similarity(
        self,
        props1: Dict[str, Any],
        props2: Dict[str, Any]
    ) -> float:
        """Calculate property similarity"""
        if not props1 or not props2:
            return 0.0
        
        # Simple Jaccard similarity on keys
        keys1 = set(props1.keys())
        keys2 = set(props2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        key_similarity = len(intersection) / len(union)
        
        # Value similarity for common keys
        value_similarity = 0.0
        for key in intersection:
            if props1[key] == props2[key]:
                value_similarity += 1.0
        
        if intersection:
            value_similarity /= len(intersection)
        
        # Combined similarity
        return 0.7 * key_similarity + 0.3 * value_similarity
    
    # ======================
    # Knowledge Synthesis
    # ======================
    
    async def synthesize_knowledge(
        self,
        query: str,
        start_entities: List[str],
        synthesis_depth: int = 3
    ) -> KnowledgeSynthesis:
        """
        Synthesize knowledge from multiple starting points.
        
        Args:
            query: Query describing what to synthesize
            start_entities: Starting entity IDs
            synthesis_depth: Depth of exploration
            
        Returns:
            Synthesized knowledge
        """
        synthesis = KnowledgeSynthesis(
            synthesis_id=f"synthesis_{hash(query)}",
            query=query,
            entities=[],
            relationships=[],
            causal_chains=[]
        )
        
        # Collect all relevant entities and relationships
        all_entities = set()
        all_relationships = []
        
        # Explore from each start point
        for start in start_entities:
            # Multi-hop exploration
            results = await self.multi_hop_query(
                start,
                max_hops=synthesis_depth
            )
            
            for entity, path, confidence in results:
                all_entities.add(entity.entity_id)
                synthesis.reasoning_path.extend(path)
            
            # Find causal chains
            chains = await self.discover_causal_chains(
                start,
                max_depth=synthesis_depth
            )
            synthesis.causal_chains.extend(chains[:3])  # Top 3 chains
        
        # Extract subgraph
        if all_entities:
            subgraph = self.graph.subgraph(all_entities)
            
            # Add entities
            for node in subgraph.nodes():
                if node in self.entities:
                    synthesis.entities.append(self.entities[node])
            
            # Add relationships
            for u, v, data in subgraph.edges(data=True):
                rel = Relationship(
                    source_id=u,
                    target_id=v,
                    relation_type=RelationType(data.get("relation_type", "relates_to")),
                    properties=data.get("properties", {}),
                    confidence=data.get("confidence", 1.0)
                )
                synthesis.relationships.append(rel)
        
        # Generate insights
        synthesis.key_insights = self._generate_insights(synthesis)
        synthesis.confidence = self._calculate_synthesis_confidence(synthesis)
        
        logger.info(
            f"Knowledge synthesis complete",
            query=query[:50],
            entities=len(synthesis.entities),
            relationships=len(synthesis.relationships),
            chains=len(synthesis.causal_chains)
        )
        
        return synthesis
    
    def _generate_insights(self, synthesis: KnowledgeSynthesis) -> List[str]:
        """Generate insights from synthesis"""
        insights = []
        
        # Causal insights
        if synthesis.causal_chains:
            longest_chain = max(synthesis.causal_chains, key=lambda c: c.length())
            insights.append(
                f"Found {len(synthesis.causal_chains)} causal chains, "
                f"longest has {longest_chain.length()} steps"
            )
            
            # Root causes
            root_causes = set()
            for chain in synthesis.causal_chains:
                if chain.root_cause:
                    root_causes.add(chain.root_cause.name)
            
            if root_causes:
                insights.append(
                    f"Root causes identified: {', '.join(list(root_causes)[:3])}"
                )
        
        # Entity insights
        entity_types = defaultdict(int)
        for entity in synthesis.entities:
            entity_types[entity.entity_type.value] += 1
        
        if entity_types:
            dominant_type = max(entity_types.items(), key=lambda x: x[1])
            insights.append(
                f"Synthesis involves {len(synthesis.entities)} entities, "
                f"primarily {dominant_type[0]} ({dominant_type[1]})"
            )
        
        # Relationship insights
        rel_types = defaultdict(int)
        for rel in synthesis.relationships:
            rel_types[rel.relation_type.value] += 1
        
        if rel_types:
            insights.append(
                f"Found {len(synthesis.relationships)} relationships across "
                f"{len(rel_types)} types"
            )
        
        return insights
    
    def _calculate_synthesis_confidence(
        self,
        synthesis: KnowledgeSynthesis
    ) -> float:
        """Calculate overall synthesis confidence"""
        if not synthesis.entities:
            return 0.0
        
        # Average entity confidence
        entity_conf = sum(e.confidence for e in synthesis.entities) / len(synthesis.entities)
        
        # Average relationship confidence
        rel_conf = 1.0
        if synthesis.relationships:
            rel_conf = sum(r.confidence for r in synthesis.relationships) / len(synthesis.relationships)
        
        # Chain confidence
        chain_conf = 1.0
        if synthesis.causal_chains:
            chain_conf = sum(c.confidence for c in synthesis.causal_chains) / len(synthesis.causal_chains)
        
        # Weighted average
        return 0.4 * entity_conf + 0.3 * rel_conf + 0.3 * chain_conf
    
    def _invalidate_cache(self, *entity_ids):
        """Invalidate caches for entities"""
        # Clear causal cache for affected entities
        keys_to_remove = []
        for key in self.causal_cache:
            if any(eid in key for eid in entity_ids):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.causal_cache[key]
    
    # ======================
    # Metrics
    # ======================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get graph metrics"""
        return {
            "entities": {
                "total": len(self.entities),
                "by_type": dict(
                    (t.value, len(ids))
                    for t, ids in self.entity_index.items()
                )
            },
            "relationships": {
                "total": self.graph.number_of_edges()
            },
            "graph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
            },
            "cache": {
                "causal_chains": len(self.causal_cache),
                "patterns": len(self.pattern_cache)
            }
        }


# ======================
# Integration Helper
# ======================

class GraphRAGMemoryIntegration:
    """
    Integrates GraphRAG with our Memory system.
    
    This enables:
    - Knowledge graph from memories
    - Multi-hop memory queries
    - Causal memory analysis
    """
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.graphrag = GraphRAGEngine()
        
    async def build_from_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> None:
        """Build knowledge graph from memories"""
        # Extract entities and relationships
        for mem in memories:
            # Create entity from memory
            entity = Entity(
                entity_id=mem.get("id", f"mem_{hash(mem.get('content', ''))}"),
                entity_type=EntityType.CONCEPT,
                name=mem.get("content", "")[:50],
                properties={
                    "content": mem.get("content"),
                    "type": mem.get("type"),
                    "confidence": mem.get("confidence", 1.0)
                }
            )
            await self.graphrag.add_entity(entity)
        
        # TODO: Extract relationships from content
        
    async def query_with_reasoning(
        self,
        query: str,
        max_hops: int = 3
    ) -> KnowledgeSynthesis:
        """Query memory with multi-hop reasoning"""
        # First, get relevant memories
        memories = await self.memory_system.retrieve(query, limit=5)
        
        # Build graph if needed
        if not self.graphrag.entities:
            await self.build_from_memories(memories)
        
        # Get starting entities
        start_entities = [
            f"mem_{hash(mem.get('content', ''))}"
            for mem in memories[:3]
        ]
        
        # Synthesize knowledge
        return await self.graphrag.synthesize_knowledge(
            query,
            start_entities,
            synthesis_depth=max_hops
        )


# ======================
# Example Usage
# ======================

async def example():
    """Example of GraphRAG"""
    print("\n🕸️ GraphRAG Example\n")
    
    # Create engine
    graphrag = GraphRAGEngine()
    
    # Add entities
    print("1. Building knowledge graph...")
    
    # System entities
    agent1 = Entity("agent_1", EntityType.AGENT, "Observer Agent")
    agent2 = Entity("agent_2", EntityType.AGENT, "Analyzer Agent")
    
    # Event entities
    event1 = Entity("event_1", EntityType.EVENT, "High CPU Usage Detected")
    event2 = Entity("event_2", EntityType.EVENT, "Memory Leak Identified")
    
    # Outcome entities
    outcome1 = Entity("outcome_1", EntityType.OUTCOME, "System Crash")
    outcome2 = Entity("outcome_2", EntityType.OUTCOME, "Performance Degradation")
    
    # Add to graph
    for entity in [agent1, agent2, event1, event2, outcome1, outcome2]:
        await graphrag.add_entity(entity)
    
    # Add relationships
    await graphrag.add_relationship(
        Relationship(agent1.entity_id, event1.entity_id, RelationType.PRODUCES)
    )
    await graphrag.add_relationship(
        Relationship(event1.entity_id, event2.entity_id, RelationType.CAUSES)
    )
    await graphrag.add_relationship(
        Relationship(event2.entity_id, outcome1.entity_id, RelationType.CAUSES)
    )
    await graphrag.add_relationship(
        Relationship(event1.entity_id, outcome2.entity_id, RelationType.CAUSES)
    )
    
    print("   Graph built with 6 entities and 4 relationships")
    
    # Multi-hop query
    print("\n2. Multi-hop reasoning...")
    results = await graphrag.multi_hop_query(
        agent1.entity_id,
        target_type=EntityType.OUTCOME,
        max_hops=3
    )
    
    for entity, path, confidence in results:
        path_str = " → ".join(path)
        print(f"   Found: {entity.name} via {path_str} (conf={confidence:.2f})")
    
    # Causal discovery
    print("\n3. Causal chain discovery...")
    chains = await graphrag.discover_causal_chains(outcome1.entity_id)
    
    for chain in chains:
        path = " → ".join([e.name for e in chain.entities])
        print(f"   Chain: {path} (conf={chain.confidence:.2f})")
    
    # Knowledge synthesis
    print("\n4. Knowledge synthesis...")
    synthesis = await graphrag.synthesize_knowledge(
        "What causes system crashes?",
        [outcome1.entity_id],
        synthesis_depth=3
    )
    
    print(f"   Entities found: {len(synthesis.entities)}")
    print(f"   Relationships: {len(synthesis.relationships)}")
    print(f"   Causal chains: {len(synthesis.causal_chains)}")
    print("   Insights:")
    for insight in synthesis.key_insights:
        print(f"     - {insight}")
    
    # Show metrics
    print("\n5. Graph metrics:")
    metrics = graphrag.get_metrics()
    for category, data in metrics.items():
        print(f"   {category}: {data}")


if __name__ == "__main__":
    asyncio.run(example())