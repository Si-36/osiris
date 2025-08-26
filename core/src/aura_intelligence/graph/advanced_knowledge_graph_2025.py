"""
🧠 Advanced Knowledge Graph Engine 2025
State-of-the-art graph-based knowledge representation with:
- Multi-modal knowledge fusion
- Causal reasoning
- Temporal dynamics
- Graph neural networks
- Real-time pattern discovery
"""

import asyncio
import time
import json
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Deque
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timezone
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAT, global_mean_pool
from torch_geometric.data import Data, Batch
import structlog

logger = structlog.get_logger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    PATTERN = "pattern"
    DECISION = "decision"
    MEMORY = "memory"
    AGENT = "agent"


class EdgeType(str, Enum):
    """Types of edges/relationships."""
    CAUSES = "causes"
    CORRELATES = "correlates"
    PRECEDES = "precedes"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    DERIVED_FROM = "derived_from"
    INFLUENCES = "influences"


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    data: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, data: Dict[str, Any]):
        """Update node data."""
        self.data.update(data)
        self.updated_at = time.time()
        self.access_count += 1


@dataclass
class KnowledgeEdge:
    """Edge in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalChain:
    """Represents a causal chain in the graph."""
    chain_id: str
    nodes: List[str]  # Ordered list of node IDs
    strength: float
    evidence_count: int
    discovered_at: float
    last_activated: float
    activation_count: int = 0
    
    def activate(self):
        """Mark chain as activated."""
        self.activation_count += 1
        self.last_activated = time.time()


@dataclass
class TemporalPattern:
    """Temporal pattern in the knowledge graph."""
    pattern_id: str
    node_sequence: List[str]
    time_intervals: List[float]
    frequency: float
    confidence: float
    first_seen: float
    last_seen: float
    occurrence_count: int = 1


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for knowledge processing."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.gat1 = GAT(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GAT(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.gat3 = GAT(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass through GAT."""
        x = self.dropout(F.elu(self.gat1(x, edge_index)))
        x = self.dropout(F.elu(self.gat2(x, edge_index)))
        x = self.gat3(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class CausalReasoningEngine:
    """Engine for causal reasoning and discovery."""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.causal_cache = {}
        self.intervention_history = deque(maxlen=1000)
    
    def discover_causality(self, graph: nx.DiGraph, source: str, target: str) -> Optional[CausalChain]:
        """Discover causal relationships between nodes."""
        # Check cache
        cache_key = f"{source}->{target}"
        if cache_key in self.causal_cache:
            return self.causal_cache[cache_key]
        
        # Find all paths
        try:
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        
        if not paths:
            return None
        
        # Analyze paths for causality
        best_chain = None
        max_strength = 0
        
        for path in paths:
            strength = self._calculate_causal_strength(graph, path)
            if strength > max_strength and strength > self.sensitivity:
                max_strength = strength
                best_chain = CausalChain(
                    chain_id=hashlib.md5(f"{path}".encode()).hexdigest()[:8],
                    nodes=path,
                    strength=strength,
                    evidence_count=len(path) - 1,
                    discovered_at=time.time(),
                    last_activated=time.time()
                )
        
        if best_chain:
            self.causal_cache[cache_key] = best_chain
        
        return best_chain
    
    def _calculate_causal_strength(self, graph: nx.DiGraph, path: List[str]) -> float:
        """Calculate strength of causal relationship."""
        if len(path) < 2:
            return 0.0
        
        strengths = []
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1], {})
            weight = edge_data.get('weight', 1.0)
            confidence = edge_data.get('confidence', 1.0)
            strengths.append(weight * confidence)
        
        # Causal strength decreases with path length
        length_penalty = 1.0 / (1.0 + len(path) - 2)
        return np.mean(strengths) * length_penalty
    
    def apply_intervention(self, graph: nx.DiGraph, node_id: str, changes: Dict[str, Any]) -> List[str]:
        """Apply intervention and track causal effects."""
        affected_nodes = []
        
        # Record intervention
        self.intervention_history.append({
            'node_id': node_id,
            'changes': changes,
            'timestamp': time.time()
        })
        
        # Find downstream nodes
        descendants = nx.descendants(graph, node_id)
        
        for desc in descendants:
            # Check causal influence
            chain = self.discover_causality(graph, node_id, desc)
            if chain and chain.strength > self.sensitivity:
                affected_nodes.append(desc)
                chain.activate()
        
        return affected_nodes


class TemporalAnalyzer:
    """Analyzes temporal patterns in the graph."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.event_sequences = defaultdict(lambda: deque(maxlen=window_size))
        self.patterns = {}
    
    def record_event(self, node_id: str, timestamp: float):
        """Record node activation event."""
        self.event_sequences[node_id].append(timestamp)
    
    def discover_patterns(self, min_support: float = 0.1) -> List[TemporalPattern]:
        """Discover temporal patterns."""
        patterns = []
        nodes = list(self.event_sequences.keys())
        
        # Find sequential patterns
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                pattern = self._find_sequential_pattern(node1, node2)
                if pattern and pattern.frequency > min_support:
                    patterns.append(pattern)
        
        return patterns
    
    def _find_sequential_pattern(self, node1: str, node2: str) -> Optional[TemporalPattern]:
        """Find sequential activation pattern between two nodes."""
        seq1 = list(self.event_sequences[node1])
        seq2 = list(self.event_sequences[node2])
        
        if len(seq1) < 2 or len(seq2) < 2:
            return None
        
        intervals = []
        for t1 in seq1:
            # Find closest subsequent activation
            subsequent = [t2 for t2 in seq2 if t2 > t1]
            if subsequent:
                intervals.append(min(subsequent) - t1)
        
        if len(intervals) < 3:
            return None
        
        # Calculate pattern statistics
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Low variance indicates strong pattern
        if std_interval / mean_interval < 0.5:
            return TemporalPattern(
                pattern_id=f"{node1}->{node2}",
                node_sequence=[node1, node2],
                time_intervals=[mean_interval],
                frequency=len(intervals) / len(seq1),
                confidence=1.0 - (std_interval / mean_interval),
                first_seen=min(seq1 + seq2),
                last_seen=max(seq1 + seq2),
                occurrence_count=len(intervals)
            )
        
        return None


class AdvancedKnowledgeGraph:
    """
    Advanced Knowledge Graph Engine with:
    - Graph neural networks for reasoning
    - Causal discovery and inference
    - Temporal pattern analysis
    - Multi-modal knowledge fusion
    - Real-time pattern discovery
    - Distributed graph processing
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        max_nodes: int = 1000000,
        enable_causal: bool = True,
        enable_temporal: bool = True,
        enable_gnn: bool = True
    ):
        # Core graph structure
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        # Configuration
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.enable_causal = enable_causal
        self.enable_temporal = enable_temporal
        self.enable_gnn = enable_gnn
        
        # Components
        self.gnn = GraphAttentionNetwork() if enable_gnn else None
        self.causal_engine = CausalReasoningEngine() if enable_causal else None
        self.temporal_analyzer = TemporalAnalyzer() if enable_temporal else None
        
        # Indexes for fast lookup
        self.type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        self.embedding_index = {}  # For similarity search
        
        # Statistics
        self.stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "causal_chains": 0,
            "patterns_discovered": 0,
            "queries_processed": 0
        }
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info(
            "Advanced Knowledge Graph initialized",
            embedding_dim=embedding_dim,
            max_nodes=max_nodes,
            features={
                "causal": enable_causal,
                "temporal": enable_temporal,
                "gnn": enable_gnn
            }
        )
    
    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        data: Dict[str, Any],
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """Add a node to the knowledge graph."""
        async with self.lock:
            # Check capacity
            if len(self.nodes) >= self.max_nodes:
                await self._evict_least_important()
            
            # Create node
            node = KnowledgeNode(
                node_id=node_id,
                node_type=node_type,
                data=data,
                importance=importance,
                embedding=embedding or self._generate_embedding(data)
            )
            
            # Add to graph
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.data)
            
            # Update indexes
            self.type_index[node_type].add(node_id)
            if node.embedding is not None:
                self.embedding_index[node_id] = node.embedding
            
            # Record temporal event
            if self.enable_temporal:
                self.temporal_analyzer.record_event(node_id, time.time())
            
            self.stats["nodes_created"] += 1
            
            logger.debug(
                "Node added",
                node_id=node_id,
                node_type=node_type.value,
                importance=importance
            )
            
            return True
    
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        confidence: float = 1.0,
        evidence: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add an edge between nodes."""
        async with self.lock:
            # Verify nodes exist
            if source_id not in self.nodes or target_id not in self.nodes:
                return False
            
            # Create edge
            edge_id = f"{source_id}-{edge_type.value}-{target_id}"
            edge = KnowledgeEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                confidence=confidence,
                evidence=evidence or []
            )
            
            # Add to graph
            self.edges[edge_id] = edge
            self.graph.add_edge(
                source_id,
                target_id,
                edge_type=edge_type.value,
                weight=weight,
                confidence=confidence
            )
            
            # Update node importance based on connections
            self.nodes[source_id].importance = min(1.0, self.nodes[source_id].importance + 0.05)
            self.nodes[target_id].importance = min(1.0, self.nodes[target_id].importance + 0.05)
            
            # Check for causal relationships
            if self.enable_causal and edge_type in [EdgeType.CAUSES, EdgeType.INFLUENCES]:
                chain = self.causal_engine.discover_causality(self.graph, source_id, target_id)
                if chain:
                    self.stats["causal_chains"] += 1
            
            self.stats["edges_created"] += 1
            
            return True
    
    async def query(
        self,
        query_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a query on the knowledge graph."""
        start_time = time.time()
        self.stats["queries_processed"] += 1
        
        async with self.lock:
            try:
                if query_type == "find_similar":
                    result = await self._find_similar_nodes(
                        parameters.get("node_id"),
                        parameters.get("k", 5)
                    )
                
                elif query_type == "find_path":
                    result = await self._find_path(
                        parameters.get("source"),
                        parameters.get("target"),
                        parameters.get("max_length", 5)
                    )
                
                elif query_type == "causal_chain":
                    result = await self._find_causal_chain(
                        parameters.get("source"),
                        parameters.get("target")
                    )
                
                elif query_type == "temporal_pattern":
                    result = await self._find_temporal_patterns(
                        parameters.get("min_support", 0.1)
                    )
                
                elif query_type == "subgraph":
                    result = await self._extract_subgraph(
                        parameters.get("center_node"),
                        parameters.get("radius", 2)
                    )
                
                elif query_type == "influence_spread":
                    result = await self._calculate_influence_spread(
                        parameters.get("node_id"),
                        parameters.get("changes", {})
                    )
                
                else:
                    result = {"error": f"Unknown query type: {query_type}"}
                
                duration_ms = (time.time() - start_time) * 1000
                result["query_time_ms"] = duration_ms
                
                return result
                
            except Exception as e:
                logger.error("Query error", error=str(e), query_type=query_type)
                return {"error": str(e)}
    
    async def _find_similar_nodes(self, node_id: str, k: int) -> Dict[str, Any]:
        """Find similar nodes using embeddings."""
        if node_id not in self.nodes or node_id not in self.embedding_index:
            return {"similar_nodes": []}
        
        query_embedding = self.embedding_index[node_id]
        similarities = []
        
        for other_id, other_embedding in self.embedding_index.items():
            if other_id != node_id:
                # Cosine similarity
                similarity = np.dot(query_embedding, other_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((other_id, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        return {
            "query_node": node_id,
            "similar_nodes": [
                {
                    "node_id": nid,
                    "similarity": sim,
                    "type": self.nodes[nid].node_type.value,
                    "data": self.nodes[nid].data
                }
                for nid, sim in top_k
            ]
        }
    
    async def _find_path(self, source: str, target: str, max_length: int) -> Dict[str, Any]:
        """Find path between nodes."""
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source, target)
            
            if len(path) > max_length:
                return {"path": None, "message": "Path too long"}
            
            # Build path details
            path_details = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                path_details.append({
                    "from": path[i],
                    "to": path[i+1],
                    "edge_type": edge_data.get("edge_type", "unknown"),
                    "weight": edge_data.get("weight", 1.0)
                })
            
            return {
                "path": path,
                "length": len(path) - 1,
                "details": path_details
            }
            
        except nx.NetworkXNoPath:
            return {"path": None, "message": "No path found"}
    
    async def _find_causal_chain(self, source: str, target: str) -> Dict[str, Any]:
        """Find causal chain between nodes."""
        if not self.enable_causal:
            return {"error": "Causal reasoning not enabled"}
        
        chain = self.causal_engine.discover_causality(self.graph, source, target)
        
        if not chain:
            return {"causal_chain": None}
        
        return {
            "causal_chain": {
                "chain_id": chain.chain_id,
                "nodes": chain.nodes,
                "strength": chain.strength,
                "evidence_count": chain.evidence_count,
                "activation_count": chain.activation_count
            }
        }
    
    async def _find_temporal_patterns(self, min_support: float) -> Dict[str, Any]:
        """Find temporal patterns in the graph."""
        if not self.enable_temporal:
            return {"error": "Temporal analysis not enabled"}
        
        patterns = self.temporal_analyzer.discover_patterns(min_support)
        self.stats["patterns_discovered"] = len(patterns)
        
        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "sequence": p.node_sequence,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "occurrences": p.occurrence_count
                }
                for p in patterns[:10]  # Top 10 patterns
            ],
            "total_patterns": len(patterns)
        }
    
    async def _extract_subgraph(self, center_node: str, radius: int) -> Dict[str, Any]:
        """Extract subgraph around a node."""
        if center_node not in self.nodes:
            return {"error": "Node not found"}
        
        # Get nodes within radius
        subgraph_nodes = set([center_node])
        current_layer = set([center_node])
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                # Add neighbors
                next_layer.update(self.graph.predecessors(node))
                next_layer.update(self.graph.successors(node))
            
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
        
        # Extract subgraph
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        return {
            "center_node": center_node,
            "radius": radius,
            "nodes": list(subgraph.nodes()),
            "edges": list(subgraph.edges()),
            "node_count": len(subgraph.nodes()),
            "edge_count": len(subgraph.edges())
        }
    
    async def _calculate_influence_spread(self, node_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate influence spread from a node."""
        if not self.enable_causal:
            return {"error": "Causal reasoning not enabled"}
        
        affected = self.causal_engine.apply_intervention(self.graph, node_id, changes)
        
        # Calculate influence scores
        influence_scores = {}
        for affected_node in affected:
            chain = self.causal_engine.discover_causality(self.graph, node_id, affected_node)
            if chain:
                influence_scores[affected_node] = chain.strength
        
        return {
            "intervention_node": node_id,
            "changes": changes,
            "affected_nodes": affected,
            "influence_scores": influence_scores,
            "total_affected": len(affected)
        }
    
    async def apply_gnn_reasoning(self, subgraph_nodes: List[str]) -> np.ndarray:
        """Apply GNN for reasoning on subgraph."""
        if not self.enable_gnn or not subgraph_nodes:
            return np.array([])
        
        # Prepare data for GNN
        node_features = []
        edge_index = []
        node_map = {node: i for i, node in enumerate(subgraph_nodes)}
        
        # Extract features
        for node in subgraph_nodes:
            if node in self.embedding_index:
                node_features.append(self.embedding_index[node])
            else:
                node_features.append(np.zeros(self.embedding_dim))
        
        # Extract edges
        for edge in self.graph.edges(subgraph_nodes):
            if edge[0] in node_map and edge[1] in node_map:
                edge_index.append([node_map[edge[0]], node_map[edge[1]]])
        
        if not edge_index:
            return np.array(node_features)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_idx = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Apply GNN
        with torch.no_grad():
            output = self.gnn(x, edge_idx)
        
        return output.numpy()
    
    def _generate_embedding(self, data: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for node data."""
        # Convert data to string for hashing
        data_str = json.dumps(data, sort_keys=True, default=str)
        
        # Generate hash-based embedding
        hash_obj = hashlib.sha256(data_str.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float array
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        embedding = embedding / 255.0  # Normalize
        
        # Resize to target dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        return embedding
    
    async def _evict_least_important(self):
        """Evict least important nodes when capacity reached."""
        # Sort nodes by importance and access time
        candidates = sorted(
            self.nodes.values(),
            key=lambda n: (n.importance, n.access_count, n.updated_at)
        )
        
        # Evict bottom 10%
        evict_count = max(1, len(candidates) // 10)
        
        for node in candidates[:evict_count]:
            # Remove from graph
            self.graph.remove_node(node.node_id)
            
            # Remove from indexes
            del self.nodes[node.node_id]
            self.type_index[node.node_type].discard(node.node_id)
            if node.node_id in self.embedding_index:
                del self.embedding_index[node.node_id]
    
    async def merge_knowledge(self, other_graph: 'AdvancedKnowledgeGraph') -> Dict[str, Any]:
        """Merge another knowledge graph into this one."""
        merged_nodes = 0
        merged_edges = 0
        conflicts = []
        
        async with self.lock:
            # Merge nodes
            for node_id, node in other_graph.nodes.items():
                if node_id in self.nodes:
                    # Resolve conflict - keep higher importance
                    if node.importance > self.nodes[node_id].importance:
                        self.nodes[node_id] = node
                        conflicts.append(node_id)
                else:
                    await self.add_node(
                        node_id,
                        node.node_type,
                        node.data,
                        node.importance,
                        node.embedding
                    )
                    merged_nodes += 1
            
            # Merge edges
            for edge_id, edge in other_graph.edges.items():
                if edge_id not in self.edges:
                    await self.add_edge(
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type,
                        edge.weight,
                        edge.confidence,
                        edge.evidence
                    )
                    merged_edges += 1
        
        return {
            "merged_nodes": merged_nodes,
            "merged_edges": merged_edges,
            "conflicts": conflicts,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        graph_stats = {
            "basic": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "density": nx.density(self.graph) if len(self.nodes) > 1 else 0
            },
            "by_type": {
                node_type.value: len(nodes)
                for node_type, nodes in self.type_index.items()
            },
            "connectivity": {
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "average_degree": sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0
            },
            "operations": self.stats
        }
        
        if self.enable_causal:
            graph_stats["causal"] = {
                "chains_discovered": self.stats["causal_chains"],
                "interventions": len(self.causal_engine.intervention_history)
            }
        
        if self.enable_temporal:
            patterns = self.temporal_analyzer.discover_patterns()
            graph_stats["temporal"] = {
                "patterns_found": len(patterns),
                "active_sequences": len(self.temporal_analyzer.event_sequences)
            }
        
        return graph_stats
    
    async def save_checkpoint(self, path: str):
        """Save graph checkpoint."""
        checkpoint = {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": {k: v.__dict__ for k, v in self.edges.items()},
            "stats": self.stats,
            "config": {
                "embedding_dim": self.embedding_dim,
                "max_nodes": self.max_nodes,
                "enable_causal": self.enable_causal,
                "enable_temporal": self.enable_temporal,
                "enable_gnn": self.enable_gnn
            }
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, default=str)
        
        logger.info("Graph checkpoint saved", path=path)