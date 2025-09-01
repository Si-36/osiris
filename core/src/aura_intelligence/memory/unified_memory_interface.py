"""
Unified Memory Interface - Software-Centric Hierarchical Memory Layer
=====================================================================

Based on 2025 research:
- Mem0 for persistent agent memory (26% accuracy gains)
- H-MEM hierarchical routing for efficiency
- Qdrant multitenancy with quantization
- Neo4j 5 GraphRAG for multi-hop reasoning
- Iceberg/Temporal for durability

Tiers:
- L1 Hot: Redis (<1ms) for working/session memory
- L2 Warm: Qdrant ANN (<10ms) with HNSW and quantization
- L3 Semantic: Neo4j 5 GraphRAG with vector indexes
- L4 Cold: Iceberg tables for long-term retention
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import structlog

# Import existing components with error handling
try:
    from ..persistence.stores.kv import NATSKVStore
except ImportError:
    NATSKVStore = None
    
try:
    from ..persistence.stores.vector import QdrantVectorStore
except ImportError:
    QdrantVectorStore = None
    
try:
    from ..persistence.stores.graph import Neo4jGraphStore
except ImportError:
    Neo4jGraphStore = None
    
try:
    from ..persistence.lakehouse.datasets import MemoryDataset
except ImportError:
    MemoryDataset = None
    
from .mem0_integration import Mem0Manager, Memory

try:
    from .shape_memory_v2_prod import ShapeMemoryV2
except ImportError:
    from .shape_memory_v2_clean import ShapeMemoryV2

try:
    from .redis_store import RedisVectorStore
except ImportError:
    RedisVectorStore = None

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the unified system"""
    WORKING = "working"      # Active processing in L1
    EPISODIC = "episodic"    # Specific events in L2
    SEMANTIC = "semantic"    # General knowledge in L3
    PROCEDURAL = "procedural"  # Skills in L3
    ARCHIVAL = "archival"    # Long-term in L4


class ConsistencyLevel(str, Enum):
    """Consistency guarantees for retrieval"""
    EVENTUAL = "eventual"
    STRONG = "strong"
    BOUNDED_STALENESS = "bounded_staleness"


class SearchType(str, Enum):
    """Types of memory search"""
    VECTOR = "vector"        # Dense similarity
    GRAPH = "graph"          # Multi-hop reasoning
    HYBRID = "hybrid"        # Combined vector + graph
    HIERARCHICAL = "hierarchical"  # H-MEM routing
    TOPOLOGICAL = "topological"    # Shape-based


@dataclass
class MemoryMetadata:
    """Standard metadata for all memories"""
    tenant_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    type: MemoryType = MemoryType.EPISODIC
    ttl_seconds: Optional[int] = None
    policy_tags: Set[str] = field(default_factory=set)
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "ttl_seconds": self.ttl_seconds,
            "policy_tags": list(self.policy_tags),
            "provenance": self.provenance
        }


@dataclass
class MemoryResult:
    """Result from memory search"""
    memory_id: str
    content: Any
    metadata: MemoryMetadata
    score: float
    tier: str
    retrieval_path: List[str] = field(default_factory=list)


@dataclass 
class HierarchicalIndex:
    """H-MEM style hierarchical routing index"""
    semantic_summary: str
    episodic_refs: List[str]
    position_encoding: np.ndarray
    level: int
    
    
class UnifiedMemoryInterface:
    """
    Unified interface for all memory operations across AURA.
    
    Implements software-centric hierarchical memory with:
    - L1 Hot cache (Redis)
    - L2 Warm vectors (Qdrant)
    - L3 Semantic graph (Neo4j)
    - L4 Cold archive (Iceberg)
    """
    
    def __init__(self):
        # L1 Hot tier - Redis
        self.redis_store = RedisVectorStore() if RedisVectorStore else None
        
        # L2 Warm tier - Qdrant with quantization
        self.qdrant_store = QdrantVectorStore() if QdrantVectorStore else None
        
        # L3 Semantic tier - Neo4j GraphRAG
        self.neo4j_store = Neo4jGraphStore() if Neo4jGraphStore else None
        
        # L4 Cold tier - Iceberg
        self.iceberg_dataset = MemoryDataset() if MemoryDataset else None
        
        # Mem0 pipeline for extract->update->retrieve
        self.mem0_manager = Mem0Manager()
        
        # Shape Memory V2 for topological search
        try:
            self.shape_memory = ShapeMemoryV2()
        except:
            self.shape_memory = None
        
        # Hierarchical routing indices
        self.hierarchical_indices: Dict[str, HierarchicalIndex] = {}
        
        # Metrics
        self.metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l4_hits": 0,
            "total_requests": 0
        }
        
        # Log available components
        available = []
        if self.redis_store: available.append("Redis")
        if self.qdrant_store: available.append("Qdrant")
        if self.neo4j_store: available.append("Neo4j")
        if self.iceberg_dataset: available.append("Iceberg")
        
        logger.info(
            "UnifiedMemoryInterface initialized",
            available_stores=available
        )
        
    async def initialize(self):
        """Initialize all memory tiers"""
        tasks = []
        
        if self.redis_store:
            tasks.append(self.redis_store.initialize())
        if self.qdrant_store:
            tasks.append(self.qdrant_store.initialize())
        if self.neo4j_store:
            tasks.append(self.neo4j_store.initialize())
        if self.iceberg_dataset:
            tasks.append(self.iceberg_dataset.initialize())
            
        tasks.append(self.mem0_manager.initialize())
        
        if tasks:
            await asyncio.gather(*tasks)
            
        logger.info("Memory tiers initialized")
        
    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType,
        metadata: MemoryMetadata,
        tier_hint: Optional[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Store memory in appropriate tier based on type and hint.
        
        Args:
            key: Memory key
            value: Memory content
            memory_type: Type of memory
            metadata: Memory metadata including tenant
            tier_hint: Optional tier preference
            embedding: Optional precomputed embedding
            
        Returns:
            Memory ID
        """
        memory_id = f"{metadata.tenant_id}:{key}:{int(time.time()*1000000)}"
        
        # Mem0 extract phase for structured memories
        if memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
            extracted = await self._mem0_extract(value, metadata)
            if extracted:
                value = extracted
                
        # Determine target tier
        tier = tier_hint or self._select_tier(memory_type, metadata)
        
        # Store based on tier
        if tier == "L1":
            # Hot cache with TTL
            ttl = metadata.ttl_seconds or 3600  # 1 hour default
            await self.redis_store.set(
                key=memory_id,
                value=json.dumps({
                    "content": value,
                    "metadata": metadata.to_dict()
                }),
                ttl=ttl
            )
            
        elif tier == "L2":
            # Warm vector store
            if embedding is None:
                embedding = await self._compute_embedding(value)
                
            await self.qdrant_store.upsert(
                collection="memories",
                points=[{
                    "id": memory_id,
                    "vector": embedding.tolist(),
                    "payload": {
                        "content": value,
                        "metadata": metadata.to_dict(),
                        "tenant_id": metadata.tenant_id  # For filtering
                    }
                }]
            )
            
        elif tier == "L3":
            # Semantic graph store
            await self._store_to_graph(memory_id, value, metadata, embedding)
            
        else:  # L4
            # Cold archive
            await self.iceberg_dataset.append([{
                "memory_id": memory_id,
                "timestamp": datetime.fromtimestamp(metadata.timestamp, tz=timezone.utc),
                "tenant_id": metadata.tenant_id,
                "user_id": metadata.user_id,
                "type": memory_type.value,
                "content": json.dumps(value),
                "metadata": json.dumps(metadata.to_dict())
            }])
            
        # Update hierarchical index if semantic
        if memory_type == MemoryType.SEMANTIC:
            await self._update_hierarchical_index(memory_id, value)
            
        logger.info(
            "Stored memory",
            memory_id=memory_id,
            tier=tier,
            type=memory_type.value
        )
        
        return memory_id
        
    async def retrieve(
        self,
        key: str,
        metadata: MemoryMetadata,
        consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ) -> Optional[Any]:
        """
        Retrieve memory by key with tiered lookup.
        
        Args:
            key: Memory key
            metadata: Must include tenant_id
            consistency: Consistency requirements
            
        Returns:
            Memory content or None
        """
        self.metrics["total_requests"] += 1
        
        # L1 check first (fastest)
        memory_id = f"{metadata.tenant_id}:{key}:*"
        l1_result = await self.redis_store.get_pattern(memory_id)
        if l1_result:
            self.metrics["l1_hits"] += 1
            return json.loads(l1_result)["content"]
            
        # L2 vector check
        l2_results = await self.qdrant_store.search(
            collection="memories",
            query_filter={"tenant_id": metadata.tenant_id, "key": key},
            limit=1
        )
        if l2_results:
            self.metrics["l2_hits"] += 1
            # Promote to L1 for future access
            await self._promote_to_l1(l2_results[0])
            return l2_results[0]["payload"]["content"]
            
        # L3 graph check
        l3_result = await self._retrieve_from_graph(key, metadata)
        if l3_result:
            self.metrics["l3_hits"] += 1
            return l3_result
            
        # L4 cold storage
        l4_result = await self._retrieve_from_iceberg(key, metadata)
        if l4_result:
            self.metrics["l4_hits"] += 1
            # Promote based on access patterns
            await self._consider_promotion(l4_result)
            return l4_result["content"]
            
        return None
        
    async def search(
        self,
        query: Union[str, np.ndarray],
        search_type: SearchType,
        metadata: MemoryMetadata,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryResult]:
        """
        Search memories using specified strategy.
        
        Args:
            query: Search query (text or embedding)
            search_type: Type of search to perform
            metadata: Must include tenant_id
            k: Number of results
            filters: Additional filters
            
        Returns:
            List of memory results
        """
        results = []
        
        if search_type == SearchType.HIERARCHICAL:
            # H-MEM hierarchical routing
            results = await self._hierarchical_search(query, metadata, k)
            
        elif search_type == SearchType.HYBRID:
            # Combined vector + graph search
            vector_results = await self._vector_search(query, metadata, k)
            graph_results = await self._graph_search(query, metadata, k)
            results = self._merge_results(vector_results, graph_results, k)
            
        elif search_type == SearchType.VECTOR:
            # Pure vector search in Qdrant
            results = await self._vector_search(query, metadata, k)
            
        elif search_type == SearchType.GRAPH:
            # Pure graph search in Neo4j
            results = await self._graph_search(query, metadata, k)
            
        elif search_type == SearchType.TOPOLOGICAL:
            # Shape-based search
            results = await self._topological_search(query, metadata, k)
            
        # Apply Mem0 retrieve enhancement
        if results:
            results = await self._mem0_enhance_results(results, metadata)
            
        return results
        
    async def consolidate(self, tenant_id: str):
        """
        Consolidate memories using H-MEM style hierarchical organization.
        
        Args:
            tenant_id: Tenant to consolidate
        """
        # Get episodic memories from L2
        episodic_memories = await self.qdrant_store.scroll(
            collection="memories",
            scroll_filter={
                "tenant_id": tenant_id,
                "metadata.type": MemoryType.EPISODIC.value
            },
            limit=1000
        )
        
        if not episodic_memories:
            return
            
        # Group by semantic similarity
        clusters = await self._cluster_memories(episodic_memories)
        
        # Create hierarchical summaries
        for cluster_id, memories in clusters.items():
            # Generate semantic summary
            summary = await self._generate_semantic_summary(memories)
            
            # Create hierarchical index
            index = HierarchicalIndex(
                semantic_summary=summary,
                episodic_refs=[m["id"] for m in memories],
                position_encoding=await self._compute_position_encoding(memories),
                level=1
            )
            
            # Store index
            index_id = f"{tenant_id}:hidx:{cluster_id}"
            self.hierarchical_indices[index_id] = index
            
            # Store semantic summary in L3
            await self.store(
                key=f"semantic:{cluster_id}",
                value=summary,
                memory_type=MemoryType.SEMANTIC,
                metadata=MemoryMetadata(tenant_id=tenant_id),
                tier_hint="L3"
            )
            
        logger.info(
            "Memory consolidation complete",
            tenant_id=tenant_id,
            clusters=len(clusters)
        )
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get memory system metrics"""
        total = self.metrics["total_requests"]
        if total == 0:
            return self.metrics
            
        return {
            **self.metrics,
            "l1_hit_rate": self.metrics["l1_hits"] / total,
            "l2_hit_rate": self.metrics["l2_hits"] / total,
            "l3_hit_rate": self.metrics["l3_hits"] / total,
            "l4_hit_rate": self.metrics["l4_hits"] / total
        }
        
    # Private helper methods
    
    def _select_tier(self, memory_type: MemoryType, metadata: MemoryMetadata) -> str:
        """Select appropriate tier based on memory type"""
        if memory_type == MemoryType.WORKING:
            return "L1"
        elif memory_type == MemoryType.EPISODIC:
            return "L2"
        elif memory_type in [MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            return "L3"
        else:
            return "L4"
            
    async def _compute_embedding(self, content: Any) -> np.ndarray:
        """Compute embedding for content"""
        # In production, use sentence transformers or similar
        # For now, return random embedding
        return np.random.randn(768).astype(np.float32)
        
    async def _mem0_extract(self, content: Any, metadata: MemoryMetadata) -> Optional[Dict[str, Any]]:
        """Extract structured information using Mem0"""
        # Implement Mem0 extraction pipeline
        # For now, return None to use original content
        return None
        
    async def _store_to_graph(
        self,
        memory_id: str,
        content: Any,
        metadata: MemoryMetadata,
        embedding: Optional[np.ndarray]
    ):
        """Store memory in Neo4j graph"""
        # Create memory node
        await self.neo4j_store.execute_write(
            """
            MERGE (m:Memory {id: $memory_id})
            SET m.content = $content,
                m.tenant_id = $tenant_id,
                m.type = $type,
                m.timestamp = $timestamp
            """,
            memory_id=memory_id,
            content=json.dumps(content),
            tenant_id=metadata.tenant_id,
            type=metadata.type.value,
            timestamp=metadata.timestamp
        )
        
        # Add vector if provided
        if embedding is not None:
            await self.neo4j_store.create_vector_index("Memory", "embedding", 768)
            await self.neo4j_store.execute_write(
                "MATCH (m:Memory {id: $memory_id}) SET m.embedding = $embedding",
                memory_id=memory_id,
                embedding=embedding.tolist()
            )
            
    async def _hierarchical_search(
        self,
        query: Union[str, np.ndarray],
        metadata: MemoryMetadata,
        k: int
    ) -> List[MemoryResult]:
        """H-MEM hierarchical search implementation"""
        results = []
        
        # First, search semantic summaries
        semantic_candidates = []
        for idx_id, index in self.hierarchical_indices.items():
            if idx_id.startswith(metadata.tenant_id):
                # Simple similarity for now
                score = np.random.random()  # Replace with actual similarity
                semantic_candidates.append((score, index))
                
        # Sort by score and take top candidates
        semantic_candidates.sort(key=lambda x: x[0], reverse=True)
        top_indices = semantic_candidates[:min(3, len(semantic_candidates))]
        
        # Expand to episodic memories
        episodic_ids = []
        for score, index in top_indices:
            episodic_ids.extend(index.episodic_refs)
            
        # Retrieve episodic memories from L2
        if episodic_ids:
            episodic_results = await self.qdrant_store.retrieve_batch(
                collection="memories",
                ids=episodic_ids[:k]
            )
            
            for i, result in enumerate(episodic_results):
                results.append(MemoryResult(
                    memory_id=result["id"],
                    content=result["payload"]["content"],
                    metadata=MemoryMetadata(**result["payload"]["metadata"]),
                    score=1.0 - (i / len(episodic_results)),
                    tier="L2",
                    retrieval_path=["hierarchical", "semantic", "episodic"]
                ))
                
        return results[:k]
        
    async def _vector_search(
        self,
        query: Union[str, np.ndarray],
        metadata: MemoryMetadata,
        k: int
    ) -> List[MemoryResult]:
        """Vector similarity search in Qdrant"""
        # Convert query to embedding if needed
        if isinstance(query, str):
            query_embedding = await self._compute_embedding(query)
        else:
            query_embedding = query
            
        # Search with tenant filter
        results = await self.qdrant_store.search(
            collection="memories",
            query_vector=query_embedding.tolist(),
            query_filter={"tenant_id": metadata.tenant_id},
            limit=k
        )
        
        return [
            MemoryResult(
                memory_id=r["id"],
                content=r["payload"]["content"],
                metadata=MemoryMetadata(**r["payload"]["metadata"]),
                score=r["score"],
                tier="L2",
                retrieval_path=["vector"]
            )
            for r in results
        ]
        
    async def _graph_search(
        self,
        query: Union[str, np.ndarray],
        metadata: MemoryMetadata,
        k: int
    ) -> List[MemoryResult]:
        """Graph-based search in Neo4j with GraphRAG"""
        # For now, simple graph traversal
        # In production, use neo4j-graphrag package
        results = await self.neo4j_store.execute_read(
            """
            MATCH (m:Memory {tenant_id: $tenant_id})
            WHERE m.content CONTAINS $query
            RETURN m.id as memory_id, m.content as content, 
                   m.type as type, m.timestamp as timestamp
            LIMIT $limit
            """,
            tenant_id=metadata.tenant_id,
            query=query if isinstance(query, str) else "",
            limit=k
        )
        
        return [
            MemoryResult(
                memory_id=r["memory_id"],
                content=json.loads(r["content"]),
                metadata=MemoryMetadata(
                    tenant_id=metadata.tenant_id,
                    type=MemoryType(r["type"]),
                    timestamp=r["timestamp"]
                ),
                score=1.0,
                tier="L3",
                retrieval_path=["graph"]
            )
            for r in results
        ]
        
    async def _topological_search(
        self,
        query: Any,
        metadata: MemoryMetadata,
        k: int
    ) -> List[MemoryResult]:
        """Topological search using Shape Memory V2"""
        # Delegate to shape memory
        topo_results = await self.shape_memory.retrieve(query, k=k)
        
        results = []
        for entry, similarity in topo_results:
            # Convert shape memory results to unified format
            results.append(MemoryResult(
                memory_id=entry.id,
                content=entry.content,
                metadata=MemoryMetadata(
                    tenant_id=metadata.tenant_id,
                    type=MemoryType.SEMANTIC
                ),
                score=similarity,
                tier="L2",
                retrieval_path=["topological"]
            ))
            
        return results
        
    def _merge_results(
        self,
        vector_results: List[MemoryResult],
        graph_results: List[MemoryResult],
        k: int
    ) -> List[MemoryResult]:
        """Merge and deduplicate results from multiple sources"""
        # Simple merge for now - in production use reciprocal rank fusion
        seen = set()
        merged = []
        
        for result in vector_results + graph_results:
            if result.memory_id not in seen:
                seen.add(result.memory_id)
                merged.append(result)
                
        # Sort by score and take top k
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:k]
        
    async def _mem0_enhance_results(
        self,
        results: List[MemoryResult],
        metadata: MemoryMetadata
    ) -> List[MemoryResult]:
        """Enhance results using Mem0 retrieval pipeline"""
        # In production, apply Mem0's graph enhancement
        # For now, return as-is
        return results
        
    async def _promote_to_l1(self, memory_data: Dict[str, Any]):
        """Promote memory to L1 cache"""
        await self.redis_store.set(
            key=memory_data["id"],
            value=json.dumps(memory_data["payload"]),
            ttl=3600  # 1 hour
        )
        
    async def _retrieve_from_iceberg(self, key: str, metadata: MemoryMetadata) -> Optional[Dict[str, Any]]:
        """Retrieve from cold storage"""
        # Query Iceberg for memory
        results = await self.iceberg_dataset.query(
            f"SELECT * FROM memories WHERE tenant_id = '{metadata.tenant_id}' AND memory_id LIKE '%:{key}:%' LIMIT 1"
        )
        
        if results:
            return {
                "content": json.loads(results[0]["content"]),
                "metadata": json.loads(results[0]["metadata"])
            }
        return None
        
    async def _cluster_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster memories by similarity"""
        # Simple clustering for now
        # In production use HDBSCAN or similar
        clusters = {}
        for i, memory in enumerate(memories):
            cluster_id = i // 10  # Group every 10 memories
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(memory)
        return clusters
        
    async def _generate_semantic_summary(self, memories: List[Dict[str, Any]]) -> str:
        """Generate semantic summary of episodic memories"""
        # In production, use LLM to summarize
        return f"Summary of {len(memories)} episodic memories"
        
    async def _compute_position_encoding(self, memories: List[Dict[str, Any]]) -> np.ndarray:
        """Compute position encoding for hierarchical index"""
        # Simple encoding for now
        return np.random.randn(128).astype(np.float32)
        
    async def _retrieve_from_graph(self, key: str, metadata: MemoryMetadata) -> Optional[Any]:
        """Retrieve from graph store"""
        results = await self.neo4j_store.execute_read(
            """
            MATCH (m:Memory {tenant_id: $tenant_id})
            WHERE m.id CONTAINS $key
            RETURN m.content as content
            LIMIT 1
            """,
            tenant_id=metadata.tenant_id,
            key=key
        )
        
        if results:
            return json.loads(results[0]["content"])
        return None
        
    async def _consider_promotion(self, memory: Dict[str, Any]):
        """Consider promoting memory to warmer tier based on access patterns"""
        # In production, use access frequency and recency
        # For now, always promote to L2
        await self.store(
            key=memory["memory_id"].split(":")[1],
            value=memory["content"],
            memory_type=MemoryType(memory["metadata"]["type"]),
            metadata=MemoryMetadata(**memory["metadata"]),
            tier_hint="L2"
        )