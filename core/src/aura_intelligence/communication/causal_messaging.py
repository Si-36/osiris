"""
🔗 Causal Messaging System
==========================

Tracks cause-effect relationships between messages,
enables causal replay, and provides debugging insights.

Features:
- Causal graph construction
- OpenTelemetry span linking
- Message lineage tracking
- Causal pattern detection
- Replay capabilities
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx
import structlog
from collections import defaultdict, deque

try:
    from .protocols import SemanticEnvelope
except ImportError:
    from protocols import SemanticEnvelope

logger = structlog.get_logger(__name__)


# ==================== Causal Structures ====================

@dataclass
class CausalEdge:
    """Edge in causal graph representing message causality"""
    cause_id: str
    effect_id: str
    edge_type: str  # reply, trigger, cascade, correlation
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalChain:
    """Chain of causally related messages"""
    chain_id: str
    root_message_id: str
    messages: List[str]
    total_latency_ms: float
    chain_type: str  # linear, branching, cyclic
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalPattern:
    """Detected pattern in causal relationships"""
    pattern_type: str  # cascade, feedback_loop, fork_join, scatter_gather
    instances: List[CausalChain]
    frequency: int
    avg_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Causal Graph Manager ====================

class CausalGraphManager:
    """
    Manages causal relationships between messages.
    
    Builds and maintains a directed graph of message causality
    for debugging, analysis, and replay.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Causal graph
        self.graph = nx.DiGraph()
        
        # Message metadata
        self.messages: Dict[str, Dict[str, Any]] = {}
        
        # Conversation tracking
        self.conversation_chains: Dict[str, List[str]] = defaultdict(list)
        
        # Pattern detection
        self.detected_patterns: Dict[str, CausalPattern] = {}
        
        # Metrics
        self.metrics = {
            "total_messages": 0,
            "causal_edges": 0,
            "chains_detected": 0,
            "patterns_found": 0
        }
        
        # History management
        self._message_queue = deque(maxlen=max_history)
        
        logger.info("Causal graph manager initialized")
    
    def track_message(
        self,
        envelope: SemanticEnvelope,
        trace_context: Optional[TraceContext] = None,
        caused_by: Optional[List[str]] = None
    ):
        """Track a message and its causal relationships"""
        message_id = envelope.message_id
        
        # Store message metadata
        self.messages[message_id] = {
            "envelope": envelope,
            "trace_context": trace_context,
            "timestamp": envelope.timestamp,
            "performative": envelope.performative.value,
            "sender": envelope.sender,
            "receiver": envelope.receiver,
            "conversation_id": envelope.conversation_id
        }
        
        # Add to graph
        self.graph.add_node(message_id, **self.messages[message_id])
        
        # Track conversation chain
        self.conversation_chains[envelope.conversation_id].append(message_id)
        
        # Add causal edges
        if envelope.in_reply_to:
            self._add_causal_edge(
                envelope.in_reply_to,
                message_id,
                "reply",
                confidence=1.0
            )
        
        if caused_by:
            for cause_id in caused_by:
                self._add_causal_edge(
                    cause_id,
                    message_id,
                    "trigger",
                    confidence=0.8
                )
        
        # Detect correlation edges
        self._detect_correlation_edges(message_id)
        
        # Update metrics
        self.metrics["total_messages"] += 1
        self._message_queue.append(message_id)
        
        # Cleanup old messages
        if len(self.graph) > self.max_history:
            self._cleanup_old_messages()
    
    def _add_causal_edge(
        self,
        cause_id: str,
        effect_id: str,
        edge_type: str,
        confidence: float = 1.0
    ):
        """Add causal edge between messages"""
        if cause_id not in self.messages or effect_id not in self.messages:
            return
        
        # Calculate latency
        cause_time = self.messages[cause_id]["timestamp"]
        effect_time = self.messages[effect_id]["timestamp"]
        latency_ms = (effect_time - cause_time).total_seconds() * 1000
        
        # Add edge
        self.graph.add_edge(
            cause_id,
            effect_id,
            type=edge_type,
            confidence=confidence,
            latency_ms=latency_ms
        )
        
        self.metrics["causal_edges"] += 1
        
        logger.debug(
            "Causal edge added",
            cause=cause_id,
            effect=effect_id,
            type=edge_type,
            latency_ms=latency_ms
        )
    
    def _detect_correlation_edges(self, message_id: str):
        """Detect correlation-based causal edges"""
        if message_id not in self.messages:
            return
        
        message = self.messages[message_id]
        
        # Look for messages in same time window
        time_window = 1.0  # 1 second
        current_time = message["timestamp"]
        
        for other_id, other_msg in self.messages.items():
            if other_id == message_id:
                continue
            
            time_diff = abs((current_time - other_msg["timestamp"]).total_seconds())
            
            if time_diff < time_window:
                # Check for correlation indicators
                confidence = 0.0
                
                # Same conversation
                if message["conversation_id"] == other_msg["conversation_id"]:
                    confidence += 0.3
                
                # Same sender or receiver
                if message["sender"] == other_msg["sender"]:
                    confidence += 0.2
                if message["receiver"] == other_msg["receiver"]:
                    confidence += 0.2
                
                # Related performatives
                if self._are_performatives_related(
                    message["performative"],
                    other_msg["performative"]
                ):
                    confidence += 0.3
                
                if confidence > 0.5:
                    self._add_causal_edge(
                        other_id,
                        message_id,
                        "correlation",
                        confidence
                    )
    
    def _are_performatives_related(self, perf1: str, perf2: str) -> bool:
        """Check if performatives are causally related"""
        causal_pairs = [
            ("request", "agree"),
            ("request", "refuse"),
            ("propose", "accept-proposal"),
            ("propose", "reject-proposal"),
            ("query-if", "confirm"),
            ("query-if", "disconfirm"),
            ("cfp", "propose")
        ]
        
        return (perf1, perf2) in causal_pairs or (perf2, perf1) in causal_pairs
    
    # ==================== Chain Detection ====================
    
    def detect_causal_chains(self) -> List[CausalChain]:
        """Detect causal chains in the graph"""
        chains = []
        visited = set()
        
        # Find root nodes (no incoming edges)
        root_nodes = [
            n for n in self.graph.nodes()
            if self.graph.in_degree(n) == 0
        ]
        
        for root in root_nodes:
            if root in visited:
                continue
            
            # Trace chain from root
            chain = self._trace_chain_from_root(root, visited)
            if len(chain) > 1:
                chains.append(self._create_causal_chain(root, chain))
        
        self.metrics["chains_detected"] = len(chains)
        return chains
    
    def _trace_chain_from_root(
        self,
        root: str,
        visited: Set[str]
    ) -> List[str]:
        """Trace causal chain from root node"""
        chain = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            chain.append(node)
            
            # Add successors
            successors = list(self.graph.successors(node))
            stack.extend(successors)
        
        return chain
    
    def _create_causal_chain(
        self,
        root: str,
        messages: List[str]
    ) -> CausalChain:
        """Create CausalChain object from message list"""
        # Calculate total latency
        total_latency = 0.0
        for i in range(len(messages) - 1):
            if self.graph.has_edge(messages[i], messages[i+1]):
                edge_data = self.graph.edges[messages[i], messages[i+1]]
                total_latency += edge_data.get("latency_ms", 0)
        
        # Determine chain type
        chain_type = "linear"
        if any(self.graph.out_degree(n) > 1 for n in messages):
            chain_type = "branching"
        if nx.is_directed_acyclic_graph(self.graph.subgraph(messages)):
            if any(self.graph.has_edge(messages[-1], messages[0]) for i in range(len(messages))):
                chain_type = "cyclic"
        
        return CausalChain(
            chain_id=f"chain_{root}_{int(datetime.utcnow().timestamp())}",
            root_message_id=root,
            messages=messages,
            total_latency_ms=total_latency,
            chain_type=chain_type
        )
    
    # ==================== Pattern Detection ====================
    
    def detect_patterns(self) -> Dict[str, CausalPattern]:
        """Detect patterns in causal relationships"""
        chains = self.detect_causal_chains()
        patterns = {}
        
        # Detect cascade patterns
        cascade_pattern = self._detect_cascade_pattern(chains)
        if cascade_pattern:
            patterns["cascade"] = cascade_pattern
        
        # Detect feedback loops
        feedback_pattern = self._detect_feedback_loops()
        if feedback_pattern:
            patterns["feedback_loop"] = feedback_pattern
        
        # Detect fork-join patterns
        fork_join_pattern = self._detect_fork_join_pattern()
        if fork_join_pattern:
            patterns["fork_join"] = fork_join_pattern
        
        # Detect scatter-gather patterns
        scatter_gather_pattern = self._detect_scatter_gather_pattern()
        if scatter_gather_pattern:
            patterns["scatter_gather"] = scatter_gather_pattern
        
        self.detected_patterns = patterns
        self.metrics["patterns_found"] = len(patterns)
        
        return patterns
    
    def _detect_cascade_pattern(
        self,
        chains: List[CausalChain]
    ) -> Optional[CausalPattern]:
        """Detect cascade patterns (rapid propagation)"""
        cascade_chains = []
        
        for chain in chains:
            if chain.chain_type == "linear" and len(chain.messages) > 3:
                # Check if messages propagate rapidly
                avg_latency = chain.total_latency_ms / (len(chain.messages) - 1)
                if avg_latency < 100:  # Less than 100ms between messages
                    cascade_chains.append(chain)
        
        if cascade_chains:
            avg_latency = sum(c.total_latency_ms for c in cascade_chains) / len(cascade_chains)
            return CausalPattern(
                pattern_type="cascade",
                instances=cascade_chains,
                frequency=len(cascade_chains),
                avg_latency_ms=avg_latency
            )
        
        return None
    
    def _detect_feedback_loops(self) -> Optional[CausalPattern]:
        """Detect feedback loop patterns"""
        cycles = list(nx.simple_cycles(self.graph))
        
        if cycles:
            # Create chains for cycles
            feedback_chains = []
            for cycle in cycles:
                if len(cycle) > 2:  # Non-trivial cycles
                    chain = CausalChain(
                        chain_id=f"feedback_{int(datetime.utcnow().timestamp())}",
                        root_message_id=cycle[0],
                        messages=cycle,
                        total_latency_ms=0,  # Calculate if needed
                        chain_type="cyclic"
                    )
                    feedback_chains.append(chain)
            
            if feedback_chains:
                return CausalPattern(
                    pattern_type="feedback_loop",
                    instances=feedback_chains,
                    frequency=len(feedback_chains),
                    avg_latency_ms=0
                )
        
        return None
    
    def _detect_fork_join_pattern(self) -> Optional[CausalPattern]:
        """Detect fork-join patterns (parallel execution)"""
        fork_join_instances = []
        
        for node in self.graph.nodes():
            out_degree = self.graph.out_degree(node)
            if out_degree > 2:  # Fork point
                # Check if paths reconverge
                successors = list(self.graph.successors(node))
                
                # Find common descendants
                descendant_sets = []
                for successor in successors:
                    descendants = nx.descendants(self.graph, successor)
                    descendant_sets.append(descendants)
                
                # Find join points
                if descendant_sets:
                    common_descendants = set.intersection(*descendant_sets)
                    if common_descendants:
                        # Create fork-join chain
                        messages = [node] + successors + list(common_descendants)[:1]
                        chain = CausalChain(
                            chain_id=f"fork_join_{node}",
                            root_message_id=node,
                            messages=messages,
                            total_latency_ms=0,
                            chain_type="branching",
                            properties={"fork_degree": out_degree}
                        )
                        fork_join_instances.append(chain)
        
        if fork_join_instances:
            return CausalPattern(
                pattern_type="fork_join",
                instances=fork_join_instances,
                frequency=len(fork_join_instances),
                avg_latency_ms=0
            )
        
        return None
    
    def _detect_scatter_gather_pattern(self) -> Optional[CausalPattern]:
        """Detect scatter-gather patterns (broadcast and collect)"""
        scatter_gather_instances = []
        
        # Look for nodes with high out-degree followed by convergence
        for node in self.graph.nodes():
            out_degree = self.graph.out_degree(node)
            if out_degree > 3:  # Scatter point
                # Check if responses converge
                successors = list(self.graph.successors(node))
                
                # Look for gathering point
                for potential_gatherer in self.graph.nodes():
                    if potential_gatherer == node:
                        continue
                    
                    # Check if many successors lead to gatherer
                    paths_to_gatherer = 0
                    for successor in successors:
                        if nx.has_path(self.graph, successor, potential_gatherer):
                            paths_to_gatherer += 1
                    
                    if paths_to_gatherer > out_degree * 0.5:  # At least half converge
                        chain = CausalChain(
                            chain_id=f"scatter_gather_{node}_{potential_gatherer}",
                            root_message_id=node,
                            messages=[node, potential_gatherer],
                            total_latency_ms=0,
                            chain_type="branching",
                            properties={
                                "scatter_degree": out_degree,
                                "gather_degree": paths_to_gatherer
                            }
                        )
                        scatter_gather_instances.append(chain)
        
        if scatter_gather_instances:
            return CausalPattern(
                pattern_type="scatter_gather",
                instances=scatter_gather_instances,
                frequency=len(scatter_gather_instances),
                avg_latency_ms=0
            )
        
        return None
    
    # ==================== OpenTelemetry Integration ====================
    
    def create_span_links(
        self,
        message_id: str
    ) -> List[Dict[str, Any]]:
        """
        Create OpenTelemetry span links for message.
        
        Returns list of span link configurations.
        """
        links = []
        
        if message_id not in self.messages:
            return links
        
        # Get causal predecessors
        predecessors = list(self.graph.predecessors(message_id))
        
        for pred_id in predecessors:
            pred_msg = self.messages.get(pred_id, {})
            trace_ctx = pred_msg.get("trace_context")
            
            if trace_ctx:
                edge_data = self.graph.edges[pred_id, message_id]
                
                link = {
                    "trace_id": trace_ctx.trace_id,
                    "span_id": trace_ctx.span_id,
                    "attributes": {
                        "causality.type": edge_data.get("type", "unknown"),
                        "causality.confidence": edge_data.get("confidence", 0),
                        "causality.latency_ms": edge_data.get("latency_ms", 0),
                        "message.id": pred_id
                    }
                }
                links.append(link)
        
        return links
    
    # ==================== Replay Capabilities ====================
    
    async def replay_causal_chain(
        self,
        chain_id: str,
        replay_handler: callable,
        speed_multiplier: float = 1.0
    ):
        """
        Replay a causal chain for debugging or testing.
        
        Args:
            chain_id: ID of chain to replay
            replay_handler: Async function to handle each message
            speed_multiplier: Speed up or slow down replay
        """
        # Find chain
        chain = None
        for pattern in self.detected_patterns.values():
            for instance in pattern.instances:
                if instance.chain_id == chain_id:
                    chain = instance
                    break
        
        if not chain:
            logger.error(f"Chain {chain_id} not found")
            return
        
        logger.info(f"Replaying chain {chain_id} with {len(chain.messages)} messages")
        
        # Replay messages in order
        for i, message_id in enumerate(chain.messages):
            if message_id not in self.messages:
                continue
            
            message_data = self.messages[message_id]
            envelope = message_data["envelope"]
            
            # Calculate delay
            if i > 0:
                prev_id = chain.messages[i-1]
                if self.graph.has_edge(prev_id, message_id):
                    edge_data = self.graph.edges[prev_id, message_id]
                    delay = edge_data.get("latency_ms", 0) / 1000.0
                    delay /= speed_multiplier
                    await asyncio.sleep(delay)
            
            # Handle message
            await replay_handler(envelope)
    
    # ==================== Utility Methods ====================
    
    def get_message_lineage(
        self,
        message_id: str,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """Get complete lineage of a message"""
        if message_id not in self.messages:
            return {}
        
        # Get ancestors
        ancestors = nx.ancestors(self.graph, message_id)
        ancestors_limited = list(ancestors)[:max_depth]
        
        # Get descendants
        descendants = nx.descendants(self.graph, message_id)
        descendants_limited = list(descendants)[:max_depth]
        
        # Build lineage tree
        lineage = {
            "message_id": message_id,
            "message": self.messages[message_id],
            "ancestors": [
                self.messages.get(a, {}) for a in ancestors_limited
            ],
            "descendants": [
                self.messages.get(d, {}) for d in descendants_limited
            ],
            "total_ancestors": len(ancestors),
            "total_descendants": len(descendants)
        }
        
        return lineage
    
    def get_conversation_graph(
        self,
        conversation_id: str
    ) -> nx.DiGraph:
        """Get subgraph for a specific conversation"""
        conv_messages = self.conversation_chains.get(conversation_id, [])
        return self.graph.subgraph(conv_messages)
    
    def _cleanup_old_messages(self):
        """Remove old messages to maintain size limit"""
        while len(self.graph) > self.max_history and self._message_queue:
            old_id = self._message_queue.popleft()
            if old_id in self.graph:
                self.graph.remove_node(old_id)
                self.messages.pop(old_id, None)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get causal graph metrics"""
        return {
            **self.metrics,
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "avg_degree": sum(d for n, d in self.graph.degree()) / self.graph.number_of_nodes() if self.graph else 0,
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }


# ==================== Causal Analysis Tools ====================

class CausalAnalyzer:
    """Tools for analyzing causal relationships"""
    
    @staticmethod
    def find_root_cause(
        graph_manager: CausalGraphManager,
        effect_message_id: str
    ) -> Optional[str]:
        """Find root cause of a message"""
        if effect_message_id not in graph_manager.graph:
            return None
        
        # Find all ancestors
        ancestors = nx.ancestors(graph_manager.graph, effect_message_id)
        
        # Find roots (no incoming edges among ancestors)
        subgraph = graph_manager.graph.subgraph(ancestors | {effect_message_id})
        roots = [n for n in subgraph if subgraph.in_degree(n) == 0]
        
        # Return earliest root
        if roots:
            earliest = min(
                roots,
                key=lambda r: graph_manager.messages[r]["timestamp"]
            )
            return earliest
        
        return None
    
    @staticmethod
    def calculate_influence_score(
        graph_manager: CausalGraphManager,
        message_id: str
    ) -> float:
        """Calculate influence score of a message"""
        if message_id not in graph_manager.graph:
            return 0.0
        
        # PageRank-style influence calculation
        try:
            pagerank = nx.pagerank(graph_manager.graph, alpha=0.85)
            return pagerank.get(message_id, 0.0)
        except:
            # Fallback to simple degree-based score
            out_degree = graph_manager.graph.out_degree(message_id)
            total_nodes = graph_manager.graph.number_of_nodes()
            return out_degree / total_nodes if total_nodes > 0 else 0.0
    
    @staticmethod
    def find_critical_path(
        graph_manager: CausalGraphManager,
        start_id: str,
        end_id: str
    ) -> Optional[List[str]]:
        """Find critical path between two messages"""
        if start_id not in graph_manager.graph or end_id not in graph_manager.graph:
            return None
        
        try:
            # Find shortest path weighted by latency
            path = nx.shortest_path(
                graph_manager.graph,
                start_id,
                end_id,
                weight=lambda u, v, d: d.get("latency_ms", 1)
            )
            return path
        except nx.NetworkXNoPath:
            return None