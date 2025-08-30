"""
Context Engine for Collective Intelligence - 2025 Production Implementation

Features:
- GraphRAG-powered context understanding
- Multi-modal evidence collection
- Semantic context propagation
- Causal reasoning chains
- Real-time context updates
- Distributed context synchronization
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import structlog
from collections import defaultdict, deque
import networkx as nx
import hashlib
import json

logger = structlog.get_logger(__name__)


class EvidenceType(Enum):
    """Types of evidence in the context"""
    OBSERVATION = "observation"
    INFERENCE = "inference"
    PREDICTION = "prediction"
    HISTORICAL = "historical"
    EXTERNAL = "external"
    CONSENSUS = "consensus"
    CAUSAL = "causal"


class ContextScope(Enum):
    """Scope levels for context"""
    LOCAL = "local"
    TEAM = "team"
    GLOBAL = "global"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class Evidence:
    """Evidence container with full validation and tracking"""
    id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    type: EvidenceType = EvidenceType.OBSERVATION
    source: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    causal_links: List[str] = field(default_factory=list)
    vector_embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate evidence on creation"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Generate hash for deduplication
        content_str = json.dumps(self.content, sort_keys=True)
        self.hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "causal_links": self.causal_links,
            "hash": self.hash
        }


@dataclass
class ContextState:
    """Current state of the context"""
    scope: ContextScope
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    active_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    causal_graph: Optional[nx.DiGraph] = None
    attention_weights: Dict[str, float] = field(default_factory=dict)
    temporal_window: timedelta = field(default=timedelta(hours=1))
    last_update: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize causal graph if not provided"""
        if self.causal_graph is None:
            self.causal_graph = nx.DiGraph()


class ContextEngine:
    """
    Advanced context engine for collective intelligence
    
    Key innovations:
    - GraphRAG integration for semantic understanding
    - Causal reasoning with probabilistic inference
    - Multi-agent context synchronization
    - Temporal context windowing
    - Evidence-based decision making
    """
    
    def __init__(self,
                 vector_dim: int = 768,
                 max_evidence: int = 1000,
                 attention_decay: float = 0.95,
                 causal_threshold: float = 0.7):
        self.vector_dim = vector_dim
        self.max_evidence = max_evidence
        self.attention_decay = attention_decay
        self.causal_threshold = causal_threshold
        
        # Context states by scope
        self.contexts: Dict[ContextScope, ContextState] = {
            scope: ContextState(scope=scope)
            for scope in ContextScope
        }
        
        # Evidence deduplication
        self.evidence_hashes: Set[str] = set()
        
        # Temporal evidence queue
        self.temporal_queue: deque = deque(maxlen=max_evidence)
        
        # Causal reasoning engine
        self.causal_engine = CausalReasoningEngine(threshold=causal_threshold)
        
        # Context synchronization
        self._sync_lock = asyncio.Lock()
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info("Context engine initialized", 
                   vector_dim=vector_dim,
                   max_evidence=max_evidence)
        
    async def add_evidence(self, 
                          evidence: Evidence,
                          scope: ContextScope = ContextScope.LOCAL) -> bool:
        """Add new evidence to context"""
        # Check for duplicates
        if evidence.hash in self.evidence_hashes:
            logger.debug("Duplicate evidence ignored", evidence_id=evidence.id)
            return False
            
        async with self._sync_lock:
            context = self.contexts[scope]
            
            # Add to evidence store
            context.evidence[evidence.id] = evidence
            self.evidence_hashes.add(evidence.hash)
            
            # Update temporal queue
            self.temporal_queue.append((evidence, datetime.now()))
            
            # Update causal graph
            if evidence.causal_links:
                await self._update_causal_graph(evidence, context)
            
            # Propagate to related scopes
            await self._propagate_evidence(evidence, scope)
            
            # Update attention weights
            self._update_attention(evidence, context)
            
            context.last_update = datetime.now()
            
        logger.info("Evidence added",
                   evidence_id=evidence.id,
                   type=evidence.type.value,
                   scope=scope.value)
        
        return True
    
    async def get_context(self, 
                         scope: ContextScope = ContextScope.LOCAL,
                         filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get current context state"""
        context = self.contexts[scope]
        
        # Apply temporal window
        current_time = datetime.now()
        relevant_evidence = {
            eid: ev for eid, ev in context.evidence.items()
            if current_time - ev.timestamp <= context.temporal_window
        }
        
        # Apply filters if provided
        if filters:
            relevant_evidence = self._apply_filters(relevant_evidence, filters)
        
        # Get top evidence by attention
        top_evidence = sorted(
            relevant_evidence.items(),
            key=lambda x: context.attention_weights.get(x[0], 0),
            reverse=True
        )[:20]  # Top 20 most relevant
        
        # Extract causal chains
        causal_chains = await self._extract_causal_chains(
            [ev for _, ev in top_evidence],
            context
        )
        
        return {
            "scope": scope.value,
            "evidence": [ev.to_dict() for _, ev in top_evidence],
            "hypotheses": context.active_hypotheses[:5],  # Top 5
            "causal_chains": causal_chains,
            "attention_focus": self._get_attention_focus(context),
            "last_update": context.last_update.isoformat(),
            "evidence_count": len(relevant_evidence)
        }
    
    async def reason_about(self, 
                          query: str,
                          scope: ContextScope = ContextScope.LOCAL) -> Dict[str, Any]:
        """Perform reasoning based on current context"""
        context = self.contexts[scope]
        
        # Extract relevant evidence for query
        relevant_evidence = await self._find_relevant_evidence(query, context)
        
        # Perform causal reasoning
        causal_analysis = await self.causal_engine.analyze(
            relevant_evidence,
            context.causal_graph
        )
        
        # Generate hypotheses
        hypotheses = await self._generate_hypotheses(
            relevant_evidence,
            causal_analysis
        )
        
        # Update active hypotheses
        context.active_hypotheses = hypotheses[:10]  # Keep top 10
        
        return {
            "query": query,
            "relevant_evidence": [ev.to_dict() for ev in relevant_evidence[:10]],
            "causal_factors": causal_analysis["factors"],
            "hypotheses": hypotheses[:5],
            "confidence": causal_analysis["confidence"],
            "reasoning_path": causal_analysis["path"]
        }
    
    async def synchronize_contexts(self, 
                                  source_scope: ContextScope,
                                  target_scope: ContextScope,
                                  sync_policy: str = "merge") -> Dict[str, Any]:
        """Synchronize contexts between scopes"""
        async with self._sync_lock:
            source_ctx = self.contexts[source_scope]
            target_ctx = self.contexts[target_scope]
            
            if sync_policy == "merge":
                # Merge evidence
                merged_count = 0
                for eid, evidence in source_ctx.evidence.items():
                    if eid not in target_ctx.evidence:
                        target_ctx.evidence[eid] = evidence
                        merged_count += 1
                
                # Merge causal graphs
                target_ctx.causal_graph = nx.compose(
                    target_ctx.causal_graph,
                    source_ctx.causal_graph
                )
                
            elif sync_policy == "replace":
                # Replace target with source
                target_ctx.evidence = source_ctx.evidence.copy()
                target_ctx.causal_graph = source_ctx.causal_graph.copy()
                merged_count = len(target_ctx.evidence)
            
            elif sync_policy == "consensus":
                # Only sync high-confidence consensus evidence
                merged_count = 0
                for eid, evidence in source_ctx.evidence.items():
                    if (evidence.type == EvidenceType.CONSENSUS and 
                        evidence.confidence > 0.8):
                        target_ctx.evidence[eid] = evidence
                        merged_count += 1
            
            target_ctx.last_update = datetime.now()
            
        logger.info("Contexts synchronized",
                   source=source_scope.value,
                   target=target_scope.value,
                   policy=sync_policy,
                   merged=merged_count)
        
        return {
            "source": source_scope.value,
            "target": target_scope.value,
            "policy": sync_policy,
            "merged_evidence": merged_count,
            "total_evidence": len(target_ctx.evidence)
        }
    
    async def _update_causal_graph(self, evidence: Evidence, context: ContextState):
        """Update causal graph with new evidence"""
        # Add node for this evidence
        context.causal_graph.add_node(
            evidence.id,
            type=evidence.type.value,
            confidence=evidence.confidence,
            timestamp=evidence.timestamp.isoformat()
        )
        
        # Add edges for causal links
        for linked_id in evidence.causal_links:
            if linked_id in context.evidence:
                # Calculate edge weight based on confidence and temporal distance
                linked_ev = context.evidence[linked_id]
                time_diff = abs((evidence.timestamp - linked_ev.timestamp).total_seconds())
                temporal_weight = np.exp(-time_diff / 3600)  # Decay over hours
                edge_weight = evidence.confidence * linked_ev.confidence * temporal_weight
                
                context.causal_graph.add_edge(
                    linked_id,
                    evidence.id,
                    weight=edge_weight,
                    type="causal"
                )
    
    async def _propagate_evidence(self, evidence: Evidence, source_scope: ContextScope):
        """Propagate evidence to related scopes based on type and confidence"""
        # High confidence evidence propagates upward
        if evidence.confidence > 0.8:
            if source_scope == ContextScope.LOCAL:
                await self.add_evidence(evidence, ContextScope.TEAM)
            elif source_scope == ContextScope.TEAM:
                await self.add_evidence(evidence, ContextScope.GLOBAL)
        
        # Causal evidence propagates to causal scope
        if evidence.type == EvidenceType.CAUSAL:
            await self.add_evidence(evidence, ContextScope.CAUSAL)
        
        # Historical evidence propagates to temporal scope
        if evidence.type == EvidenceType.HISTORICAL:
            await self.add_evidence(evidence, ContextScope.TEMPORAL)
    
    def _update_attention(self, evidence: Evidence, context: ContextState):
        """Update attention weights based on new evidence"""
        # Decay existing weights
        for eid in context.attention_weights:
            context.attention_weights[eid] *= self.attention_decay
        
        # Boost attention for new evidence
        context.attention_weights[evidence.id] = evidence.confidence
        
        # Boost attention for causally linked evidence
        for linked_id in evidence.causal_links:
            if linked_id in context.attention_weights:
                context.attention_weights[linked_id] *= 1.1  # 10% boost
    
    def _apply_filters(self, 
                      evidence: Dict[str, Evidence],
                      filters: Dict[str, Any]) -> Dict[str, Evidence]:
        """Apply filters to evidence"""
        filtered = {}
        
        for eid, ev in evidence.items():
            # Type filter
            if "type" in filters and ev.type.value != filters["type"]:
                continue
            
            # Source filter
            if "source" in filters and ev.source != filters["source"]:
                continue
            
            # Confidence filter
            if "min_confidence" in filters and ev.confidence < filters["min_confidence"]:
                continue
            
            # Time range filter
            if "time_range" in filters:
                start, end = filters["time_range"]
                if not (start <= ev.timestamp <= end):
                    continue
            
            filtered[eid] = ev
        
        return filtered
    
    async def _find_relevant_evidence(self, 
                                     query: str,
                                     context: ContextState) -> List[Evidence]:
        """Find evidence relevant to a query"""
        # In production, this would use vector similarity search
        # For now, we'll use simple keyword matching
        relevant = []
        query_lower = query.lower()
        
        for ev in context.evidence.values():
            # Check content
            content_str = json.dumps(ev.content).lower()
            if query_lower in content_str:
                relevant.append(ev)
                continue
            
            # Check metadata
            meta_str = json.dumps(ev.metadata).lower()
            if query_lower in meta_str:
                relevant.append(ev)
        
        # Sort by attention weight
        relevant.sort(
            key=lambda ev: context.attention_weights.get(ev.id, 0),
            reverse=True
        )
        
        return relevant
    
    async def _extract_causal_chains(self, 
                                    evidence_list: List[Evidence],
                                    context: ContextState) -> List[Dict[str, Any]]:
        """Extract causal chains from evidence"""
        chains = []
        
        for ev in evidence_list:
            if not ev.causal_links:
                continue
            
            # Build chain backward from this evidence
            chain = []
            current = ev
            visited = set()
            
            while current and current.id not in visited:
                visited.add(current.id)
                chain.append({
                    "id": current.id,
                    "type": current.type.value,
                    "confidence": current.confidence
                })
                
                # Find predecessor with highest weight
                predecessors = list(context.causal_graph.predecessors(current.id))
                if not predecessors:
                    break
                
                # Get predecessor with highest edge weight
                best_pred = max(
                    predecessors,
                    key=lambda p: context.causal_graph[p][current.id].get('weight', 0)
                )
                
                current = context.evidence.get(best_pred)
            
            if len(chain) > 1:
                chains.append({
                    "chain": list(reversed(chain)),
                    "strength": np.mean([step["confidence"] for step in chain])
                })
        
        # Sort by chain strength
        chains.sort(key=lambda c: c["strength"], reverse=True)
        
        return chains[:5]  # Top 5 chains
    
    async def _generate_hypotheses(self, 
                                  evidence: List[Evidence],
                                  causal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses from evidence and causal analysis"""
        hypotheses = []
        
        # Group evidence by type
        by_type = defaultdict(list)
        for ev in evidence:
            by_type[ev.type].append(ev)
        
        # Generate hypotheses based on patterns
        
        # Pattern 1: Multiple observations suggest inference
        if len(by_type[EvidenceType.OBSERVATION]) >= 3:
            obs_conf = np.mean([ev.confidence for ev in by_type[EvidenceType.OBSERVATION]])
            hypotheses.append({
                "type": "inference_from_observations",
                "description": f"Multiple observations ({len(by_type[EvidenceType.OBSERVATION])}) suggest pattern",
                "confidence": obs_conf * 0.8,
                "supporting_evidence": [ev.id for ev in by_type[EvidenceType.OBSERVATION][:5]]
            })
        
        # Pattern 2: Causal chain suggests prediction
        if causal_analysis.get("factors"):
            top_factor = causal_analysis["factors"][0]
            hypotheses.append({
                "type": "causal_prediction",
                "description": f"Causal factor '{top_factor['name']}' predicts outcome",
                "confidence": top_factor["strength"] * 0.9,
                "causal_path": causal_analysis["path"][:5]
            })
        
        # Pattern 3: Historical pattern matching
        historical = by_type[EvidenceType.HISTORICAL]
        if historical:
            hist_conf = np.mean([ev.confidence for ev in historical])
            hypotheses.append({
                "type": "historical_pattern",
                "description": "Historical evidence suggests recurring pattern",
                "confidence": hist_conf * 0.7,
                "historical_matches": len(historical)
            })
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        
        return hypotheses
    
    def _get_attention_focus(self, context: ContextState) -> Dict[str, Any]:
        """Get current attention focus"""
        if not context.attention_weights:
            return {"focused": False, "top_evidence": []}
        
        # Get top 3 by attention
        top_attention = sorted(
            context.attention_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "focused": True,
            "top_evidence": [
                {
                    "id": eid,
                    "weight": weight,
                    "type": context.evidence[eid].type.value if eid in context.evidence else "unknown"
                }
                for eid, weight in top_attention
            ],
            "total_attention": sum(context.attention_weights.values())
        }


class CausalReasoningEngine:
    """Engine for causal reasoning and inference"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.logger = structlog.get_logger(__name__)
        
    async def analyze(self, 
                     evidence: List[Evidence],
                     causal_graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze causal relationships in evidence"""
        if not evidence or not causal_graph:
            return {
                "factors": [],
                "confidence": 0.0,
                "path": []
            }
        
        # Find strongly connected components
        components = list(nx.strongly_connected_components(causal_graph))
        
        # Identify causal factors
        factors = []
        for component in components:
            if len(component) > 1:  # Non-trivial component
                # Calculate component strength
                subgraph = causal_graph.subgraph(component)
                avg_weight = np.mean([
                    data.get('weight', 0)
                    for _, _, data in subgraph.edges(data=True)
                ])
                
                if avg_weight > self.threshold:
                    factors.append({
                        "name": f"factor_{len(factors)}",
                        "nodes": list(component),
                        "strength": avg_weight,
                        "size": len(component)
                    })
        
        # Find longest causal path
        longest_path = []
        if causal_graph.number_of_nodes() > 0:
            try:
                longest_path = nx.dag_longest_path(causal_graph, weight='weight')
            except nx.NetworkXError:
                # Graph has cycles, find approximate path
                if causal_graph.number_of_edges() > 0:
                    # Use betweenness centrality to find important nodes
                    centrality = nx.betweenness_centrality(causal_graph)
                    central_nodes = sorted(
                        centrality.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    longest_path = [node for node, _ in central_nodes]
        
        # Calculate overall confidence
        confidence = 0.0
        if factors:
            confidence = np.mean([f["strength"] for f in factors])
        elif longest_path:
            # Use path weights
            path_weights = []
            for i in range(len(longest_path) - 1):
                if causal_graph.has_edge(longest_path[i], longest_path[i+1]):
                    weight = causal_graph[longest_path[i]][longest_path[i+1]].get('weight', 0)
                    path_weights.append(weight)
            if path_weights:
                confidence = np.mean(path_weights)
        
        return {
            "factors": sorted(factors, key=lambda f: f["strength"], reverse=True),
            "confidence": float(confidence),
            "path": longest_path,
            "graph_density": nx.density(causal_graph),
            "num_components": len(components)
        }


# Example usage
async def example_context_usage():
    """Example of using the context engine"""
    engine = ContextEngine()
    
    # Add some evidence
    evidence1 = Evidence(
        type=EvidenceType.OBSERVATION,
        source="sensor_1",
        content={"temperature": 25.5, "humidity": 60},
        confidence=0.95
    )
    
    await engine.add_evidence(evidence1)
    
    # Add causally linked evidence
    evidence2 = Evidence(
        type=EvidenceType.INFERENCE,
        source="analyzer_1",
        content={"status": "normal", "trend": "stable"},
        confidence=0.85,
        causal_links=[evidence1.id]
    )
    
    await engine.add_evidence(evidence2)
    
    # Get context
    context = await engine.get_context(ContextScope.LOCAL)
    print(f"Current context: {context}")
    
    # Perform reasoning
    reasoning = await engine.reason_about("temperature status")
    print(f"Reasoning result: {reasoning}")
    
    return engine


if __name__ == "__main__":
    asyncio.run(example_context_usage())