"""
🧠 AURA Knowledge Graph 2025 - The Failure Prevention Brain
===========================================================

The most advanced knowledge graph system designed specifically for:
- Predicting and preventing cascading failures in multi-agent systems
- Topological pattern recognition and anomaly detection  
- Causal reasoning about agent behaviors and system states
- Real-time failure risk assessment and mitigation
- GraphRAG-inspired retrieval augmented generation

"We see the shape of failure before it happens"
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
from torch_geometric.nn import GCNConv, GAT, GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import structlog
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

class NodeType(str, Enum):
    """Types of nodes in AURA's knowledge graph."""
    # Agent-related
    AGENT = "agent"
    AGENT_STATE = "agent_state"
    AGENT_DECISION = "agent_decision"
    
    # System components
    WORKFLOW = "workflow"
    SUPERVISOR = "supervisor"
    MEMORY = "memory"
    
    # Failure-related
    FAILURE_PATTERN = "failure_pattern"
    RISK_INDICATOR = "risk_indicator"
    MITIGATION = "mitigation"
    
    # Topological
    TOPOLOGY_SIGNATURE = "topology_signature"
    ANOMALY = "anomaly"
    CRITICAL_POINT = "critical_point"
    
    # Causal
    CAUSE = "cause"
    EFFECT = "effect"
    INTERVENTION = "intervention"


class EdgeType(str, Enum):
    """Types of relationships in the graph."""
    # Causal relationships
    CAUSES = "causes"
    PREVENTS = "prevents"
    MITIGATES = "mitigates"
    
    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CO_OCCURS = "co_occurs"
    
    # Structural relationships
    CONTAINS = "contains"
    PART_OF = "part_of"
    CONNECTED_TO = "connected_to"
    
    # Similarity relationships
    SIMILAR_TO = "similar_to"
    PATTERN_MATCH = "pattern_match"
    
    # Agent relationships
    COMMUNICATES_WITH = "communicates_with"
    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"


@dataclass
class FailureSignature:
    """Represents a failure pattern signature."""
    signature_id: str
    pattern_type: str  # cascade, deadlock, resource_starvation, etc.
    topology_features: np.ndarray
    precursor_events: List[str]
    risk_score: float
    detected_at: float
    occurrences: int = 1
    prevented_count: int = 0
    
    def update_prevention(self):
        """Update when this pattern was successfully prevented."""
        self.prevented_count += 1


@dataclass
class CausalChain:
    """Enhanced causal chain for failure analysis."""
    chain_id: str
    root_cause: str
    effect_sequence: List[str]
    probability: float
    evidence_strength: float
    interventions: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    last_activated: float = field(default_factory=time.time)
    
    def add_intervention(self, intervention: str, success: bool):
        """Track intervention effectiveness."""
        self.interventions.append(intervention)
        if success:
            self.success_rate = (self.success_rate * (len(self.interventions) - 1) + 1.0) / len(self.interventions)
        else:
            self.success_rate = (self.success_rate * (len(self.interventions) - 1)) / len(self.interventions)


@dataclass
class TopologicalAnomaly:
    """Represents a topological anomaly in the system."""
    anomaly_id: str
    topology_snapshot: Dict[str, Any]
    persistence_diagram: np.ndarray
    anomaly_score: float
    affected_agents: List[str]
    risk_level: str  # low, medium, high, critical
    detected_at: float
    resolved: bool = False
    resolution_time: Optional[float] = None


# ==================== Neural Components ====================

class FailurePredictionGNN(nn.Module):
    """Graph Neural Network for failure prediction."""
    
    def __init__(self, node_dim: int = 128, edge_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Failure prediction head
        self.failure_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)  # low, medium, high risk
        )
        
        # Pattern detection head
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 10)  # 10 failure pattern types
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass for failure prediction."""
        # Graph convolutions
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        
        h = self.conv3(h, edge_index)
        
        # Self-attention
        h_attended, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h + h_attended.squeeze(0)
        
        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        
        # Predictions
        risk_pred = self.failure_predictor(h)
        pattern_pred = self.pattern_detector(h)
        
        return {
            'risk_levels': F.softmax(risk_pred, dim=-1),
            'failure_patterns': torch.sigmoid(pattern_pred)
        }


class CausalReasoningEngine:
    """Advanced causal reasoning for failure analysis."""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.intervention_history = deque(maxlen=1000)
        self.causal_strength_cache = {}
        
    def discover_root_causes(self, failure_node: str, graph: nx.DiGraph) -> List[CausalChain]:
        """Discover root causes of a failure."""
        root_causes = []
        
        # Find all ancestors that could be causes
        try:
            ancestors = nx.ancestors(graph, failure_node)
        except:
            return []
        
        for ancestor in ancestors:
            # Check if this is a plausible cause
            if self._is_causal_candidate(ancestor, failure_node, graph):
                # Trace causal path
                paths = list(nx.all_simple_paths(graph, ancestor, failure_node, cutoff=5))
                
                for path in paths:
                    strength = self._calculate_causal_strength(path, graph)
                    if strength > 0.5:  # Threshold for causal relationship
                        chain = CausalChain(
                            chain_id=hashlib.md5(f"{ancestor}->{failure_node}".encode()).hexdigest()[:8],
                            root_cause=ancestor,
                            effect_sequence=path,
                            probability=strength,
                            evidence_strength=self._calculate_evidence_strength(path, graph)
                        )
                        root_causes.append(chain)
        
        # Sort by probability
        root_causes.sort(key=lambda c: c.probability, reverse=True)
        return root_causes[:5]  # Top 5 causes
    
    def predict_cascading_failures(self, initial_failure: str, graph: nx.DiGraph) -> List[Tuple[str, float]]:
        """Predict which nodes might fail as a cascade."""
        at_risk = []
        
        # BFS to find potentially affected nodes
        visited = set()
        queue = [(initial_failure, 1.0)]  # (node, probability)
        
        while queue:
            current, prob = queue.pop(0)
            if current in visited or prob < 0.1:  # Probability threshold
                continue
            
            visited.add(current)
            
            # Check all descendants
            for successor in graph.successors(current):
                if successor not in visited:
                    # Calculate cascade probability
                    edge_data = graph.get_edge_data(current, successor, {})
                    edge_strength = edge_data.get('weight', 0.5)
                    cascade_prob = prob * edge_strength * self._get_node_vulnerability(successor, graph)
                    
                    if cascade_prob > 0.1:
                        at_risk.append((successor, cascade_prob))
                        queue.append((successor, cascade_prob))
        
        # Sort by risk
        at_risk.sort(key=lambda x: x[1], reverse=True)
        return at_risk
    
    def recommend_interventions(self, failure_chain: CausalChain, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Recommend interventions to prevent failure."""
        interventions = []
        
        # Analyze each step in the causal chain
        for i in range(len(failure_chain.effect_sequence) - 1):
            node = failure_chain.effect_sequence[i]
            next_node = failure_chain.effect_sequence[i + 1]
            
            # Find intervention points
            intervention = {
                'target_node': node,
                'action': self._determine_intervention_action(node, next_node, graph),
                'priority': self._calculate_intervention_priority(i, failure_chain),
                'expected_effectiveness': self._estimate_effectiveness(node, failure_chain)
            }
            interventions.append(intervention)
        
        # Sort by priority
        interventions.sort(key=lambda x: x['priority'], reverse=True)
        return interventions[:3]  # Top 3 interventions
    
    def _is_causal_candidate(self, cause: str, effect: str, graph: nx.DiGraph) -> bool:
        """Check if a node could be a cause of another."""
        # Must have a path
        if not nx.has_path(graph, cause, effect):
            return False
        
        # Check temporal ordering if available
        cause_data = graph.nodes.get(cause, {})
        effect_data = graph.nodes.get(effect, {})
        
        cause_time = cause_data.get('timestamp', 0)
        effect_time = effect_data.get('timestamp', float('inf'))
        
        return cause_time < effect_time
    
    def _calculate_causal_strength(self, path: List[str], graph: nx.DiGraph) -> float:
        """Calculate strength of causal relationship."""
        if len(path) < 2:
            return 0.0
        
        # Check cache
        path_key = "->".join(path)
        if path_key in self.causal_strength_cache:
            return self.causal_strength_cache[path_key]
        
        # Calculate based on edge weights and path length
        strength = 1.0
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1], {})
            edge_weight = edge_data.get('weight', 0.5)
            edge_type = edge_data.get('edge_type', '')
            
            # Causal edges have higher weight
            if edge_type in ['causes', 'influences']:
                edge_weight *= 1.2
            
            strength *= edge_weight
        
        # Decay with path length
        strength *= (0.9 ** (len(path) - 2))
        
        self.causal_strength_cache[path_key] = strength
        return strength
    
    def _calculate_evidence_strength(self, path: List[str], graph: nx.DiGraph) -> float:
        """Calculate evidence strength for causal chain."""
        evidence_count = 0
        
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1], {})
            evidence = edge_data.get('evidence', [])
            evidence_count += len(evidence)
        
        # Normalize by path length
        return min(1.0, evidence_count / (len(path) - 1))
    
    def _get_node_vulnerability(self, node: str, graph: nx.DiGraph) -> float:
        """Calculate vulnerability of a node to cascading failure."""
        node_data = graph.nodes.get(node, {})
        
        # Base vulnerability
        vulnerability = 0.5
        
        # Increase based on dependencies
        in_degree = graph.in_degree(node)
        vulnerability += min(0.3, in_degree * 0.05)
        
        # Check historical failures
        failure_history = node_data.get('failure_count', 0)
        vulnerability += min(0.2, failure_history * 0.1)
        
        return min(1.0, vulnerability)
    
    def _determine_intervention_action(self, node: str, next_node: str, graph: nx.DiGraph) -> str:
        """Determine best intervention action."""
        node_type = graph.nodes[node].get('node_type', '')
        
        if node_type == 'agent':
            return 'isolate_agent'
        elif node_type == 'workflow':
            return 'pause_workflow'
        elif node_type == 'memory':
            return 'cache_state'
        else:
            return 'monitor_closely'
    
    def _calculate_intervention_priority(self, position: int, chain: CausalChain) -> float:
        """Calculate priority of intervention point."""
        # Earlier in chain = higher priority
        position_score = 1.0 - (position / len(chain.effect_sequence))
        
        # Weight by chain probability
        return position_score * chain.probability
    
    def _estimate_effectiveness(self, node: str, chain: CausalChain) -> float:
        """Estimate effectiveness of intervention at node."""
        # Base effectiveness
        effectiveness = 0.6
        
        # Adjust based on past success
        if chain.interventions:
            effectiveness = chain.success_rate
        
        return effectiveness


# ==================== Main Knowledge Graph ====================

class AURAKnowledgeGraph:
    """
    AURA's Advanced Knowledge Graph for Failure Prevention
    
    Core capabilities:
    - Topological failure pattern recognition
    - Causal reasoning about agent behaviors
    - Real-time risk assessment
    - GraphRAG-inspired retrieval
    - Predictive failure analysis
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        enable_gnn: bool = True,
        enable_causal: bool = True,
        failure_threshold: float = 0.7
    ):
        # Core graph
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = {}
        
        # Configuration
        self.embedding_dim = embedding_dim
        self.failure_threshold = failure_threshold
        
        # Neural components
        self.failure_gnn = FailurePredictionGNN() if enable_gnn else None
        self.causal_engine = CausalReasoningEngine() if enable_causal else None
        
        # Pattern storage
        self.failure_signatures: Dict[str, FailureSignature] = {}
        self.topology_snapshots: deque = deque(maxlen=1000)
        self.anomaly_history: List[TopologicalAnomaly] = []
        
        # Indexes
        self.embedding_index = {}  # For similarity search
        self.type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        self.risk_index: Dict[str, float] = {}  # Node risk scores
        
        # Statistics
        self.stats = {
            "failures_predicted": 0,
            "failures_prevented": 0,
            "false_positives": 0,
            "patterns_discovered": 0,
            "interventions_successful": 0,
            "total_queries": 0
        }
        
        # Real-time monitoring
        self.active_risks: Dict[str, Dict[str, Any]] = {}
        self.monitoring_queue = asyncio.Queue()
        
        logger.info(
            "AURA Knowledge Graph initialized",
            embedding_dim=embedding_dim,
            enable_gnn=enable_gnn,
            enable_causal=enable_causal
        )
    
    async def ingest_agent_state(
        self,
        agent_id: str,
        state: Dict[str, Any],
        topology_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest agent state and analyze for failure risks."""
        start_time = time.time()
        
        # Create state node
        state_node_id = f"{agent_id}_state_{int(time.time() * 1000)}"
        
        await self._add_node(
            node_id=state_node_id,
            node_type=NodeType.AGENT_STATE,
            data={
                "agent_id": agent_id,
                "state": state,
                "timestamp": time.time(),
                "error_count": state.get("error_count", 0),
                "performance_metrics": state.get("metrics", {})
            }
        )
        
        # Link to agent
        await self._add_edge(
            source_id=agent_id,
            target_id=state_node_id,
            edge_type=EdgeType.CONTAINS,
            weight=1.0
        )
        
        # Analyze for risks
        risk_analysis = await self._analyze_failure_risk(agent_id, state, topology_data)
        
        # Check for patterns
        if risk_analysis["risk_score"] > self.failure_threshold:
            pattern = await self._detect_failure_pattern(agent_id, state, risk_analysis)
            if pattern:
                await self._trigger_prevention_protocol(agent_id, pattern, risk_analysis)
        
        # Update monitoring
        if risk_analysis["risk_score"] > 0.5:
            self.active_risks[agent_id] = {
                "risk_score": risk_analysis["risk_score"],
                "patterns": risk_analysis.get("patterns", []),
                "timestamp": time.time()
            }
        
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "state_node": state_node_id,
            "risk_analysis": risk_analysis,
            "duration_ms": duration_ms
        }
    
    async def predict_failure_cascade(
        self,
        initial_failure: str,
        time_horizon: float = 300.0  # 5 minutes
    ) -> Dict[str, Any]:
        """Predict cascading failures from an initial failure."""
        if not self.causal_engine:
            return {"error": "Causal engine not enabled"}
        
        # Find cascading risks
        cascade_risks = self.causal_engine.predict_cascading_failures(
            initial_failure,
            self.graph
        )
        
        # Analyze each at-risk component
        cascade_analysis = []
        for node_id, probability in cascade_risks[:10]:  # Top 10 risks
            node_data = self.graph.nodes.get(node_id, {})
            
            analysis = {
                "node_id": node_id,
                "node_type": node_data.get("node_type", "unknown"),
                "failure_probability": probability,
                "estimated_time": self._estimate_failure_time(initial_failure, node_id),
                "impact_score": self._calculate_impact_score(node_id),
                "mitigation_available": self._has_mitigation(node_id)
            }
            cascade_analysis.append(analysis)
        
        # Generate prevention plan
        prevention_plan = await self._generate_prevention_plan(
            initial_failure,
            cascade_analysis
        )
        
        self.stats["failures_predicted"] += len(cascade_analysis)
        
        return {
            "initial_failure": initial_failure,
            "cascade_risks": cascade_analysis,
            "total_at_risk": len(cascade_risks),
            "prevention_plan": prevention_plan,
            "time_horizon": time_horizon
        }
    
    async def analyze_topology_snapshot(
        self,
        topology_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze topology snapshot for anomalies."""
        # Store snapshot
        self.topology_snapshots.append({
            "data": topology_data,
            "timestamp": time.time()
        })
        
        # Extract features
        persistence_diagram = topology_data.get("persistence_diagram", [])
        anomaly_score = topology_data.get("anomaly_score", 0.0)
        
        # Check against known patterns
        matching_patterns = []
        for sig_id, signature in self.failure_signatures.items():
            similarity = self._compare_topology_signatures(
                topology_data,
                signature.topology_features
            )
            if similarity > 0.8:
                matching_patterns.append({
                    "pattern_id": sig_id,
                    "pattern_type": signature.pattern_type,
                    "similarity": similarity,
                    "risk_score": signature.risk_score
                })
        
        # Detect new anomalies
        if anomaly_score > 0.7 and not matching_patterns:
            anomaly = TopologicalAnomaly(
                anomaly_id=hashlib.md5(str(topology_data).encode()).hexdigest()[:8],
                topology_snapshot=topology_data,
                persistence_diagram=np.array(persistence_diagram),
                anomaly_score=anomaly_score,
                affected_agents=self._identify_affected_agents(topology_data),
                risk_level=self._classify_risk_level(anomaly_score),
                detected_at=time.time()
            )
            self.anomaly_history.append(anomaly)
            
            # Create anomaly node
            await self._add_node(
                node_id=anomaly.anomaly_id,
                node_type=NodeType.ANOMALY,
                data={
                    "anomaly": anomaly.__dict__,
                    "topology_features": topology_data
                }
            )
        
        return {
            "anomaly_detected": anomaly_score > 0.7,
            "anomaly_score": anomaly_score,
            "matching_patterns": matching_patterns,
            "risk_level": self._classify_risk_level(anomaly_score),
            "affected_agents": self._identify_affected_agents(topology_data)
        }
    
    async def query_similar_failures(
        self,
        query_state: Dict[str, Any],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar past failures using GraphRAG approach."""
        # Generate embedding for query
        query_embedding = self._generate_state_embedding(query_state)
        
        # Find similar failure patterns
        similarities = []
        
        for node_id in self.type_index[NodeType.FAILURE_PATTERN]:
            if node_id in self.embedding_index:
                node_embedding = self.embedding_index[node_id]
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    node_embedding.reshape(1, -1)
                )[0, 0]
                
                similarities.append((node_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results with context
        results = []
        for node_id, similarity in similarities[:k]:
            node_data = self.nodes.get(node_id, {})
            
            # Get causal context
            root_causes = []
            if self.causal_engine:
                causes = self.causal_engine.discover_root_causes(node_id, self.graph)
                root_causes = [
                    {
                        "cause": c.root_cause,
                        "probability": c.probability,
                        "interventions": c.interventions[:3]
                    }
                    for c in causes[:3]
                ]
            
            results.append({
                "failure_id": node_id,
                "similarity": float(similarity),
                "pattern_type": node_data.get("pattern_type", "unknown"),
                "risk_score": node_data.get("risk_score", 0.0),
                "root_causes": root_causes,
                "prevention_success_rate": node_data.get("prevention_rate", 0.0),
                "last_occurrence": node_data.get("last_seen", 0)
            })
        
        return results
    
    async def recommend_preventive_actions(
        self,
        agent_id: str,
        risk_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recommend actions to prevent failure."""
        recommendations = []
        
        # Get agent's causal relationships
        if self.causal_engine and agent_id in self.graph:
            # Find potential failure paths
            failure_nodes = list(self.type_index[NodeType.FAILURE_PATTERN])
            
            for failure_node in failure_nodes[:5]:  # Check top failure patterns
                if nx.has_path(self.graph, agent_id, failure_node):
                    # Get causal chain
                    chains = self.causal_engine.discover_root_causes(failure_node, self.graph)
                    
                    for chain in chains:
                        if agent_id in chain.effect_sequence:
                            # Get interventions
                            interventions = self.causal_engine.recommend_interventions(
                                chain,
                                self.graph
                            )
                            
                            for intervention in interventions:
                                recommendations.append({
                                    "action": intervention["action"],
                                    "target": intervention["target_node"],
                                    "priority": intervention["priority"],
                                    "expected_effectiveness": intervention["expected_effectiveness"],
                                    "failure_prevented": failure_node,
                                    "reasoning": f"Prevents {chain.root_cause} -> {failure_node}"
                                })
        
        # Add risk-based recommendations
        if risk_analysis.get("risk_score", 0) > 0.6:
            recommendations.append({
                "action": "increase_monitoring",
                "target": agent_id,
                "priority": 0.8,
                "expected_effectiveness": 0.7,
                "reasoning": "High risk score detected"
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        return recommendations[:5]  # Top 5 recommendations
    
    async def learn_from_outcome(
        self,
        intervention_id: str,
        outcome: Dict[str, Any]
    ) -> bool:
        """Learn from intervention outcomes to improve predictions."""
        success = outcome.get("success", False)
        
        # Update intervention statistics
        if success:
            self.stats["interventions_successful"] += 1
            self.stats["failures_prevented"] += 1
        else:
            self.stats["false_positives"] += outcome.get("false_positive", 0)
        
        # Update causal chains
        if self.causal_engine and "causal_chain_id" in outcome:
            chain_id = outcome["causal_chain_id"]
            # Find and update the chain
            for chain in self.causal_engine.causal_graph.nodes():
                if hasattr(chain, 'chain_id') and chain.chain_id == chain_id:
                    chain.add_intervention(intervention_id, success)
        
        # Update failure signatures
        if "pattern_id" in outcome:
            pattern_id = outcome["pattern_id"]
            if pattern_id in self.failure_signatures:
                signature = self.failure_signatures[pattern_id]
                if success:
                    signature.update_prevention()
        
        # Store learning outcome
        await self._add_node(
            node_id=f"outcome_{intervention_id}",
            node_type=NodeType.INTERVENTION,
            data={
                "intervention_id": intervention_id,
                "outcome": outcome,
                "success": success,
                "timestamp": time.time()
            }
        )
        
        return True
    
    # ==================== Internal Methods ====================
    
    async def _add_node(
        self,
        node_id: str,
        node_type: NodeType,
        data: Dict[str, Any]
    ) -> bool:
        """Add node to graph."""
        # Add to NetworkX graph
        self.graph.add_node(
            node_id,
            node_type=node_type.value,
            **data
        )
        
        # Store node data
        self.nodes[node_id] = {
            "node_type": node_type,
            "data": data,
            "created_at": time.time()
        }
        
        # Update type index
        self.type_index[node_type].add(node_id)
        
        # Generate embedding
        if node_type in [NodeType.AGENT_STATE, NodeType.FAILURE_PATTERN]:
            embedding = self._generate_state_embedding(data)
            self.embedding_index[node_id] = embedding
        
        return True
    
    async def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        evidence: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add edge to graph."""
        self.graph.add_edge(
            source_id,
            target_id,
            edge_type=edge_type.value,
            weight=weight,
            evidence=evidence or [],
            created_at=time.time()
        )
        
        edge_id = f"{source_id}-{edge_type.value}-{target_id}"
        self.edges[edge_id] = {
            "source": source_id,
            "target": target_id,
            "edge_type": edge_type,
            "weight": weight
        }
        
        return True
    
    async def _analyze_failure_risk(
        self,
        agent_id: str,
        state: Dict[str, Any],
        topology_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze failure risk for an agent."""
        risk_factors = {}
        
        # Error-based risk
        error_count = state.get("error_count", 0)
        error_rate = state.get("error_rate", 0.0)
        risk_factors["error_risk"] = min(1.0, error_count * 0.1 + error_rate)
        
        # Performance-based risk
        latency = state.get("latency_ms", 0)
        risk_factors["performance_risk"] = min(1.0, latency / 5000)  # 5s = max risk
        
        # Resource-based risk
        cpu_usage = state.get("cpu_usage", 0.0)
        memory_usage = state.get("memory_usage", 0.0)
        risk_factors["resource_risk"] = max(cpu_usage, memory_usage)
        
        # Topology-based risk
        if topology_data:
            risk_factors["topology_risk"] = topology_data.get("anomaly_score", 0.0)
        
        # Historical risk
        agent_history = self._get_agent_failure_history(agent_id)
        risk_factors["historical_risk"] = min(1.0, len(agent_history) * 0.2)
        
        # Calculate composite risk
        risk_score = np.mean(list(risk_factors.values()))
        
        # Detect patterns
        patterns = []
        if risk_score > 0.5:
            patterns = self._match_risk_patterns(risk_factors)
        
        return {
            "risk_score": float(risk_score),
            "risk_factors": risk_factors,
            "patterns": patterns,
            "risk_level": self._classify_risk_level(risk_score)
        }
    
    async def _detect_failure_pattern(
        self,
        agent_id: str,
        state: Dict[str, Any],
        risk_analysis: Dict[str, Any]
    ) -> Optional[FailureSignature]:
        """Detect failure pattern from current state."""
        # Generate pattern features
        pattern_features = self._extract_pattern_features(state, risk_analysis)
        
        # Check against known patterns
        for sig_id, signature in self.failure_signatures.items():
            similarity = cosine_similarity(
                pattern_features.reshape(1, -1),
                signature.topology_features.reshape(1, -1)
            )[0, 0]
            
            if similarity > 0.8:
                signature.occurrences += 1
                return signature
        
        # New pattern detected
        if risk_analysis["risk_score"] > 0.7:
            new_signature = FailureSignature(
                signature_id=hashlib.md5(str(pattern_features).encode()).hexdigest()[:8],
                pattern_type=self._classify_pattern_type(risk_analysis),
                topology_features=pattern_features,
                precursor_events=self._extract_precursor_events(agent_id),
                risk_score=risk_analysis["risk_score"],
                detected_at=time.time()
            )
            
            self.failure_signatures[new_signature.signature_id] = new_signature
            self.stats["patterns_discovered"] += 1
            
            return new_signature
        
        return None
    
    async def _trigger_prevention_protocol(
        self,
        agent_id: str,
        pattern: FailureSignature,
        risk_analysis: Dict[str, Any]
    ):
        """Trigger failure prevention protocol."""
        # Create intervention
        intervention = {
            "intervention_id": f"int_{int(time.time() * 1000)}",
            "agent_id": agent_id,
            "pattern": pattern,
            "risk_analysis": risk_analysis,
            "timestamp": time.time(),
            "actions": []
        }
        
        # Determine actions based on pattern type
        if pattern.pattern_type == "cascade":
            intervention["actions"] = [
                {"type": "isolate_agent", "target": agent_id},
                {"type": "checkpoint_state", "target": agent_id},
                {"type": "notify_supervisor", "severity": "high"}
            ]
        elif pattern.pattern_type == "deadlock":
            intervention["actions"] = [
                {"type": "reset_locks", "target": agent_id},
                {"type": "restart_agent", "target": agent_id}
            ]
        else:
            intervention["actions"] = [
                {"type": "increase_monitoring", "target": agent_id},
                {"type": "reduce_load", "target": agent_id}
            ]
        
        # Add to monitoring queue
        await self.monitoring_queue.put(intervention)
        
        logger.warning(
            "Prevention protocol triggered",
            agent_id=agent_id,
            pattern_type=pattern.pattern_type,
            risk_score=risk_analysis["risk_score"],
            actions=len(intervention["actions"])
        )
    
    def _generate_state_embedding(self, state: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for state data."""
        # Extract numerical features
        features = []
        
        # Add numeric values
        for key in ['error_count', 'error_rate', 'latency_ms', 'cpu_usage', 'memory_usage']:
            features.append(state.get(key, 0.0))
        
        # Add categorical features as one-hot
        status = state.get('status', 'unknown')
        status_values = ['running', 'failed', 'completed', 'pending']
        for s in status_values:
            features.append(1.0 if status == s else 0.0)
        
        # Add derived features
        features.append(state.get('retry_count', 0) / 10.0)
        features.append(min(1.0, state.get('duration_ms', 0) / 60000))  # Normalize to minutes
        
        # Pad to embedding dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return np.array(features[:self.embedding_dim], dtype=np.float32)
    
    def _compare_topology_signatures(
        self,
        topology_data: Dict[str, Any],
        signature_features: np.ndarray
    ) -> float:
        """Compare topology signatures."""
        # Extract features from topology data
        current_features = self._extract_topology_features(topology_data)
        
        # Cosine similarity
        similarity = cosine_similarity(
            current_features.reshape(1, -1),
            signature_features.reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def _extract_topology_features(self, topology_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from topology data."""
        features = []
        
        # Persistence diagram features
        persistence = topology_data.get("persistence_diagram", [])
        if persistence:
            features.extend([
                len(persistence),
                np.mean([p[1] - p[0] for p in persistence]) if persistence else 0,
                np.max([p[1] - p[0] for p in persistence]) if persistence else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        # Topology metrics
        features.append(topology_data.get("num_components", 0))
        features.append(topology_data.get("num_loops", 0))
        features.append(topology_data.get("complexity", 0))
        features.append(topology_data.get("anomaly_score", 0))
        
        return np.array(features, dtype=np.float32)
    
    def _identify_affected_agents(self, topology_data: Dict[str, Any]) -> List[str]:
        """Identify agents affected by topology anomaly."""
        affected = []
        
        # Get nodes with high centrality in anomalous region
        if "affected_nodes" in topology_data:
            for node_id in topology_data["affected_nodes"]:
                if node_id in self.graph and self.graph.nodes[node_id].get("node_type") == "agent":
                    affected.append(node_id)
        
        return affected
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level."""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"
    
    def _get_agent_failure_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get agent's failure history."""
        history = []
        
        # Find failure nodes connected to agent
        if agent_id in self.graph:
            for neighbor in self.graph.neighbors(agent_id):
                node_data = self.graph.nodes.get(neighbor, {})
                if node_data.get("node_type") == NodeType.FAILURE_PATTERN.value:
                    history.append(node_data)
        
        return history
    
    def _match_risk_patterns(self, risk_factors: Dict[str, float]) -> List[str]:
        """Match risk factors to known patterns."""
        patterns = []
        
        # High error + high resource = potential cascade
        if risk_factors.get("error_risk", 0) > 0.6 and risk_factors.get("resource_risk", 0) > 0.7:
            patterns.append("cascade_risk")
        
        # High performance risk + topology risk = system degradation
        if risk_factors.get("performance_risk", 0) > 0.7 and risk_factors.get("topology_risk", 0) > 0.5:
            patterns.append("system_degradation")
        
        # Historical risk pattern
        if risk_factors.get("historical_risk", 0) > 0.6:
            patterns.append("recurring_failure")
        
        return patterns
    
    def _extract_pattern_features(
        self,
        state: Dict[str, Any],
        risk_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features for pattern matching."""
        features = []
        
        # Risk scores
        features.extend(list(risk_analysis["risk_factors"].values()))
        
        # State features
        features.append(state.get("error_count", 0) / 10.0)
        features.append(state.get("retry_count", 0) / 5.0)
        features.append(min(1.0, state.get("duration_ms", 0) / 30000))
        
        # Pad to standard size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _classify_pattern_type(self, risk_analysis: Dict[str, Any]) -> str:
        """Classify the type of failure pattern."""
        risk_factors = risk_analysis["risk_factors"]
        
        # Classification logic
        if risk_factors.get("error_risk", 0) > 0.7 and risk_factors.get("resource_risk", 0) > 0.6:
            return "cascade"
        elif risk_factors.get("performance_risk", 0) > 0.8:
            return "performance_degradation"
        elif risk_factors.get("topology_risk", 0) > 0.7:
            return "topology_anomaly"
        else:
            return "general_failure"
    
    def _extract_precursor_events(self, agent_id: str) -> List[str]:
        """Extract events that preceded current state."""
        events = []
        
        # Get recent events for agent
        if agent_id in self.graph:
            # Find recent state nodes
            for neighbor in self.graph.predecessors(agent_id):
                node_data = self.graph.nodes.get(neighbor, {})
                if node_data.get("node_type") == NodeType.AGENT_STATE.value:
                    events.append(neighbor)
        
        return events[-5:]  # Last 5 events
    
    def _estimate_failure_time(self, source: str, target: str) -> float:
        """Estimate time until failure propagates."""
        # Simple estimation based on graph distance
        try:
            distance = nx.shortest_path_length(self.graph, source, target)
            # Assume ~30 seconds per hop
            return distance * 30.0
        except:
            return 300.0  # Default 5 minutes
    
    def _calculate_impact_score(self, node_id: str) -> float:
        """Calculate impact score if node fails."""
        if node_id not in self.graph:
            return 0.0
        
        # Based on node centrality and dependencies
        out_degree = self.graph.out_degree(node_id)
        centrality = nx.degree_centrality(self.graph).get(node_id, 0)
        
        # Normalize
        impact = (out_degree / 10.0) * 0.5 + centrality * 0.5
        return min(1.0, impact)
    
    def _has_mitigation(self, node_id: str) -> bool:
        """Check if mitigation exists for node."""
        # Check if connected to mitigation nodes
        if node_id in self.graph:
            for neighbor in self.graph.neighbors(node_id):
                node_data = self.graph.nodes.get(neighbor, {})
                if node_data.get("node_type") == NodeType.MITIGATION.value:
                    return True
        return False
    
    async def _generate_prevention_plan(
        self,
        initial_failure: str,
        cascade_analysis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive prevention plan."""
        plan = {
            "immediate_actions": [],
            "monitoring_targets": [],
            "resource_allocation": {},
            "estimated_prevention_rate": 0.0
        }
        
        # Immediate actions for high-risk nodes
        for analysis in cascade_analysis:
            if analysis["failure_probability"] > 0.7:
                plan["immediate_actions"].append({
                    "action": "isolate_component",
                    "target": analysis["node_id"],
                    "priority": "critical",
                    "reason": f"High cascade risk ({analysis['failure_probability']:.2f})"
                })
            elif analysis["failure_probability"] > 0.4:
                plan["monitoring_targets"].append({
                    "target": analysis["node_id"],
                    "metrics": ["error_rate", "latency", "resource_usage"],
                    "threshold": 0.8
                })
        
        # Resource allocation
        total_risk = sum(a["failure_probability"] for a in cascade_analysis)
        if total_risk > 2.0:  # High overall risk
            plan["resource_allocation"] = {
                "additional_compute": "20%",
                "monitoring_priority": "high",
                "backup_resources": "standby"
            }
        
        # Estimate prevention rate
        preventable = sum(1 for a in cascade_analysis if a.get("mitigation_available", False))
        plan["estimated_prevention_rate"] = preventable / len(cascade_analysis) if cascade_analysis else 0.0
        
        return plan
    
    # ==================== Public Interface ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "graph_stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": {k.value: len(v) for k, v in self.type_index.items()},
                "active_risks": len(self.active_risks)
            },
            "pattern_stats": {
                "patterns_discovered": len(self.failure_signatures),
                "anomalies_detected": len(self.anomaly_history),
                "topology_snapshots": len(self.topology_snapshots)
            },
            "prevention_stats": {
                "failures_predicted": self.stats["failures_predicted"],
                "failures_prevented": self.stats["failures_prevented"],
                "interventions_successful": self.stats["interventions_successful"],
                "false_positive_rate": (
                    self.stats["false_positives"] / max(1, self.stats["failures_predicted"])
                )
            },
            "performance_stats": {
                "total_queries": self.stats["total_queries"],
                "active_monitoring": self.monitoring_queue.qsize()
            }
        }
    
    async def export_knowledge(self, format: str = "json") -> Dict[str, Any]:
        """Export knowledge graph for analysis."""
        if format == "json":
            return {
                "nodes": [
                    {
                        "id": node_id,
                        "type": data["node_type"].value if hasattr(data["node_type"], 'value') else data["node_type"],
                        "data": data["data"],
                        "created_at": data["created_at"]
                    }
                    for node_id, data in self.nodes.items()
                ],
                "edges": [
                    {
                        "id": edge_id,
                        "source": data["source"],
                        "target": data["target"],
                        "type": data["edge_type"].value if hasattr(data["edge_type"], 'value') else data["edge_type"],
                        "weight": data["weight"]
                    }
                    for edge_id, data in self.edges.items()
                ],
                "patterns": [
                    {
                        "id": sig_id,
                        "type": sig.pattern_type,
                        "risk_score": sig.risk_score,
                        "occurrences": sig.occurrences,
                        "prevented": sig.prevented_count
                    }
                    for sig_id, sig in self.failure_signatures.items()
                ],
                "stats": self.get_stats()
            }
        else:
            raise ValueError(f"Unsupported format: {format}")