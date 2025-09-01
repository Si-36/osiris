"""
🧠 Enhanced Neural Mesh - Consciousness-Aware Communication
===========================================================

Preserves and enhances the original neural mesh with:
- Consciousness-aware routing
- Dynamic topology optimization
- Fault tolerance and self-healing
- Integration with our communication system
"""

import asyncio
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
import structlog

try:
    from .protocols import MessagePriority, SemanticEnvelope, Performative
    from .causal_messaging import CausalGraphManager
except ImportError:
    # For direct import
    from protocols import MessagePriority, SemanticEnvelope, Performative
    from causal_messaging import CausalGraphManager

logger = structlog.get_logger(__name__)


# ==================== Enhanced Message Types ====================

class MessageType(Enum):
    """Enhanced message types for neural mesh"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    CONSENSUS = "consensus"
    HEARTBEAT = "heartbeat"
    SWARM_SYNC = "swarm_sync"
    PATTERN_DISCOVERY = "pattern_discovery"
    COLLECTIVE_LEARN = "collective_learn"


class NodeStatus(Enum):
    """Neural node status with self-healing states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    HEALING = "healing"
    FAILED = "failed"
    QUARANTINED = "quarantined"


# ==================== Core Data Structures ====================

@dataclass
class NeuralNode:
    """Enhanced neural node with consciousness and health metrics"""
    id: str
    position: np.ndarray = field(default_factory=lambda: np.random.randn(3))
    consciousness_level: float = 0.5
    status: NodeStatus = NodeStatus.HEALTHY
    connections: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    messages_processed: int = 0
    average_latency: float = 0.0
    error_rate: float = 0.0
    
    # Consciousness factors
    attention_focus: Dict[str, float] = field(default_factory=dict)
    emergent_patterns: List[str] = field(default_factory=list)
    
    def distance_to(self, other: 'NeuralNode') -> float:
        """Calculate distance to another node"""
        return float(np.linalg.norm(self.position - other.position))
    
    def update_health(self):
        """Update node health based on metrics"""
        if self.error_rate > 0.5:
            self.status = NodeStatus.FAILED
        elif self.error_rate > 0.2:
            self.status = NodeStatus.DEGRADED
        elif self.status == NodeStatus.DEGRADED and self.error_rate < 0.1:
            self.status = NodeStatus.HEALING
        elif self.status == NodeStatus.HEALING and self.error_rate < 0.05:
            self.status = NodeStatus.HEALTHY


@dataclass
class NeuralPath:
    """Path through neural mesh with quality metrics"""
    nodes: List[str]
    strength: float
    latency: float
    reliability: float = 1.0
    path_type: str = "direct"
    
    @property
    def quality_score(self) -> float:
        """Combined quality score for path selection"""
        return (self.strength * 0.4 + 
                self.reliability * 0.4 + 
                (1.0 / (1.0 + self.latency)) * 0.2)


@dataclass
class MeshMessage:
    """Enhanced mesh message with semantic support"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.DIRECT
    sender_id: str = ""
    target_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: int = 10  # Time to live (hops)
    path: List[str] = field(default_factory=list)
    
    # Semantic fields
    performative: Optional[Performative] = None
    conversation_id: Optional[str] = None
    
    def to_semantic_envelope(self) -> SemanticEnvelope:
        """Convert to semantic envelope for protocol handling"""
        return SemanticEnvelope(
            performative=self.performative or Performative.INFORM,
            sender=self.sender_id,
            receiver=self.target_id or "*",
            content=self.payload,
            conversation_id=self.conversation_id,
            message_id=self.id
        )


# ==================== Consciousness-Aware Router ====================

class ConsciousnessAwareRouter:
    """
    Advanced routing based on consciousness levels and attention.
    
    This is the key innovation - routing decisions based on:
    - Node consciousness levels
    - Attention focus alignment
    - Emergent pattern detection
    - Collective intelligence state
    """
    
    def __init__(self, mesh: 'EnhancedNeuralMesh'):
        self.mesh = mesh
        self.route_cache: Dict[Tuple[str, str], NeuralPath] = {}
        self.cache_ttl = 60.0  # seconds
        self.last_cache_clear = datetime.utcnow()
    
    async def find_best_path(
        self,
        source_id: str,
        target_id: str,
        message: MeshMessage
    ) -> Optional[NeuralPath]:
        """Find optimal path considering consciousness and network state"""
        
        # Check cache
        cache_key = (source_id, target_id)
        if cache_key in self.route_cache:
            cached_path = self.route_cache[cache_key]
            if self._is_path_valid(cached_path):
                return cached_path
        
        # Clear old cache entries
        if (datetime.utcnow() - self.last_cache_clear).total_seconds() > self.cache_ttl:
            self.route_cache.clear()
            self.last_cache_clear = datetime.utcnow()
        
        # Find paths using consciousness-aware algorithm
        paths = await self._find_consciousness_paths(source_id, target_id, message)
        
        if not paths:
            # Fallback to shortest path
            paths = await self._find_shortest_paths(source_id, target_id)
        
        if paths:
            best_path = max(paths, key=lambda p: p.quality_score)
            self.route_cache[cache_key] = best_path
            return best_path
        
        return None
    
    async def _find_consciousness_paths(
        self,
        source_id: str,
        target_id: str,
        message: MeshMessage
    ) -> List[NeuralPath]:
        """Find paths that maximize consciousness alignment"""
        paths = []
        
        try:
            # Get source and target nodes
            source = self.mesh.nodes.get(source_id)
            target = self.mesh.nodes.get(target_id)
            
            if not source or not target:
                return paths
            
            # Use A* with consciousness heuristic
            def consciousness_heuristic(n1: str, n2: str) -> float:
                node1 = self.mesh.nodes.get(n1)
                node2 = self.mesh.nodes.get(n2)
                
                if not node1 or not node2:
                    return float('inf')
                
                # Consider consciousness levels
                consciousness_diff = abs(node1.consciousness_level - node2.consciousness_level)
                
                # Consider attention alignment
                attention_alignment = self._calculate_attention_alignment(node1, node2)
                
                # Consider physical distance
                distance = node1.distance_to(node2)
                
                # Combined heuristic (lower is better)
                return (distance * 0.4 + 
                       consciousness_diff * 0.3 + 
                       (1.0 - attention_alignment) * 0.3)
            
            # Find top k paths
            k = min(3, len(self.mesh.nodes))
            
            for _ in range(k):
                try:
                    path_nodes = nx.astar_path(
                        self.mesh.topology,
                        source_id,
                        target_id,
                        heuristic=consciousness_heuristic
                    )
                    
                    # Calculate path metrics
                    strength = await self._calculate_path_strength(path_nodes)
                    latency = await self._calculate_path_latency(path_nodes)
                    reliability = await self._calculate_path_reliability(path_nodes)
                    
                    paths.append(NeuralPath(
                        nodes=path_nodes,
                        strength=strength,
                        latency=latency,
                        reliability=reliability,
                        path_type="consciousness"
                    ))
                    
                    # Remove edges to find alternative paths
                    if len(path_nodes) > 1:
                        self.mesh.topology.remove_edge(path_nodes[0], path_nodes[1])
                    
                except nx.NetworkXNoPath:
                    break
            
            # Restore removed edges
            for path in paths:
                for i in range(len(path.nodes) - 1):
                    if not self.mesh.topology.has_edge(path.nodes[i], path.nodes[i+1]):
                        node1 = self.mesh.nodes[path.nodes[i]]
                        node2 = self.mesh.nodes[path.nodes[i+1]]
                        strength = 1.0 / (1.0 + node1.distance_to(node2))
                        self.mesh.topology.add_edge(path.nodes[i], path.nodes[i+1], strength=strength)
            
        except Exception as e:
            logger.error(f"Consciousness path finding error: {e}")
        
        return paths
    
    async def _find_shortest_paths(
        self,
        source_id: str,
        target_id: str
    ) -> List[NeuralPath]:
        """Fallback to find shortest paths"""
        paths = []
        
        try:
            if nx.has_path(self.mesh.topology, source_id, target_id):
                # Simple shortest path
                path_nodes = nx.shortest_path(self.mesh.topology, source_id, target_id)
                
                strength = await self._calculate_path_strength(path_nodes)
                latency = await self._calculate_path_latency(path_nodes)
                
                paths.append(NeuralPath(
                    nodes=path_nodes,
                    strength=strength,
                    latency=latency,
                    path_type="shortest"
                ))
        except Exception as e:
            logger.error(f"Shortest path error: {e}")
        
        return paths
    
    def _calculate_attention_alignment(
        self,
        node1: NeuralNode,
        node2: NeuralNode
    ) -> float:
        """Calculate how aligned two nodes' attention focuses are"""
        if not node1.attention_focus or not node2.attention_focus:
            return 0.5
        
        # Cosine similarity of attention vectors
        keys = set(node1.attention_focus.keys()) | set(node2.attention_focus.keys())
        
        if not keys:
            return 0.5
        
        vec1 = np.array([node1.attention_focus.get(k, 0.0) for k in keys])
        vec2 = np.array([node2.attention_focus.get(k, 0.0) for k in keys])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.5
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    async def _calculate_path_strength(self, path_nodes: List[str]) -> float:
        """Calculate overall path strength"""
        if len(path_nodes) < 2:
            return 1.0
        
        strengths = []
        for i in range(len(path_nodes) - 1):
            if self.mesh.topology.has_edge(path_nodes[i], path_nodes[i+1]):
                edge_data = self.mesh.topology[path_nodes[i]][path_nodes[i+1]]
                strengths.append(edge_data.get('strength', 0.5))
        
        return float(np.mean(strengths)) if strengths else 0.0
    
    async def _calculate_path_latency(self, path_nodes: List[str]) -> float:
        """Estimate path latency based on node metrics"""
        if len(path_nodes) < 2:
            return 0.0
        
        total_latency = 0.0
        for node_id in path_nodes:
            node = self.mesh.nodes.get(node_id)
            if node:
                total_latency += node.average_latency
        
        return total_latency
    
    async def _calculate_path_reliability(self, path_nodes: List[str]) -> float:
        """Calculate path reliability based on node health"""
        if not path_nodes:
            return 0.0
        
        reliabilities = []
        for node_id in path_nodes:
            node = self.mesh.nodes.get(node_id)
            if node:
                reliabilities.append(1.0 - node.error_rate)
        
        return float(np.prod(reliabilities)) if reliabilities else 0.0
    
    def _is_path_valid(self, path: NeuralPath) -> bool:
        """Check if cached path is still valid"""
        for node_id in path.nodes:
            node = self.mesh.nodes.get(node_id)
            if not node or node.status == NodeStatus.FAILED:
                return False
        
        for i in range(len(path.nodes) - 1):
            if not self.mesh.topology.has_edge(path.nodes[i], path.nodes[i+1]):
                return False
        
        return True


# ==================== Enhanced Neural Mesh ====================

class EnhancedNeuralMesh:
    """
    Production-ready neural mesh with consciousness-aware routing.
    
    Integrates with AURA communication system while preserving
    the innovative consciousness-based routing algorithms.
    """
    
    def __init__(
        self,
        max_nodes: int = 1000,
        connection_threshold: float = 0.7,
        heartbeat_interval: float = 5.0,
        message_retry_limit: int = 3,
        enable_self_healing: bool = True
    ):
        self.max_nodes = max_nodes
        self.connection_threshold = connection_threshold
        self.heartbeat_interval = heartbeat_interval
        self.message_retry_limit = message_retry_limit
        self.enable_self_healing = enable_self_healing
        
        # Network components
        self.nodes: Dict[str, NeuralNode] = {}
        self.topology: nx.DiGraph = nx.DiGraph()
        self.router = ConsciousnessAwareRouter(self)
        
        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.pending_messages: Dict[str, MeshMessage] = {}
        self.message_history: deque = deque(maxlen=10000)
        
        # Consensus tracking
        self.consensus_groups: Dict[str, Set[str]] = {}
        self.consensus_votes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Integration with other systems
        self.causal_graph: Optional[CausalGraphManager] = None
        
        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "consensus_reached": 0,
            "healings_performed": 0,
            "avg_consciousness": 0.5
        }
        
        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info(
            "Enhanced Neural Mesh initialized",
            max_nodes=max_nodes,
            connection_threshold=connection_threshold
        )
    
    async def start(self):
        """Start the neural mesh network"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._tasks.extend([
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._topology_optimizer()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._consciousness_updater())
        ])
        
        if self.enable_self_healing:
            self._tasks.append(asyncio.create_task(self._self_healing_loop()))
        
        logger.info("Enhanced Neural Mesh started")
    
    async def stop(self):
        """Stop the neural mesh network"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        logger.info("Enhanced Neural Mesh stopped")
    
    # ==================== Node Management ====================
    
    async def add_node(
        self,
        node_id: str,
        position: Optional[np.ndarray] = None,
        consciousness_level: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NeuralNode:
        """Add a new node to the mesh"""
        if len(self.nodes) >= self.max_nodes:
            raise ValueError(f"Maximum nodes ({self.max_nodes}) reached")
        
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        # Create node
        node = NeuralNode(
            id=node_id,
            position=position if position is not None else np.random.randn(3),
            consciousness_level=consciousness_level,
            metadata=metadata or {}
        )
        
        # Add to network
        self.nodes[node_id] = node
        self.topology.add_node(node_id, **node.metadata)
        
        # Connect to nearby nodes
        await self._connect_node(node)
        
        # Register default handlers
        self.register_handler(
            MessageType.DIRECT,
            lambda msg, nid: self._default_handler(msg, nid),
            node_id
        )
        
        logger.info(
            "Node added to mesh",
            node_id=node_id,
            consciousness=consciousness_level,
            connections=len(node.connections)
        )
        
        return node
    
    async def remove_node(self, node_id: str):
        """Remove a node from the mesh"""
        if node_id not in self.nodes:
            return
        
        # Remove from topology
        self.topology.remove_node(node_id)
        
        # Remove from other nodes' connections
        for other_node in self.nodes.values():
            other_node.connections.discard(node_id)
        
        # Remove from consensus groups
        for group_nodes in self.consensus_groups.values():
            group_nodes.discard(node_id)
        
        # Remove node
        del self.nodes[node_id]
        
        logger.info("Node removed from mesh", node_id=node_id)
    
    async def _connect_node(self, node: NeuralNode):
        """Connect node to nearby nodes based on distance and consciousness"""
        candidates = []
        
        for other_id, other_node in self.nodes.items():
            if other_id == node.id:
                continue
            
            # Calculate connection score
            distance = node.distance_to(other_node)
            consciousness_factor = (node.consciousness_level + other_node.consciousness_level) / 2
            
            # Consider node health
            health_factor = 1.0
            if other_node.status == NodeStatus.DEGRADED:
                health_factor = 0.5
            elif other_node.status == NodeStatus.FAILED:
                health_factor = 0.1
            
            # Combined score
            score = (consciousness_factor * health_factor) / (1 + distance)
            
            if score > self.connection_threshold:
                candidates.append((other_id, score))
        
        # Sort by score and connect to top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for other_id, score in candidates[:10]:  # Max 10 connections
            # Create bidirectional connection
            node.connections.add(other_id)
            self.nodes[other_id].connections.add(node.id)
            
            # Add edges to topology
            self.topology.add_edge(node.id, other_id, strength=score)
            self.topology.add_edge(other_id, node.id, strength=score)
    
    # ==================== Message Handling ====================
    
    async def send_message(
        self,
        message: MeshMessage,
        target_id: Optional[str] = None
    ) -> bool:
        """Send a message through the mesh"""
        if target_id:
            message.target_id = target_id
        
        # Add to message history
        self.message_history.append(message)
        
        # Track in causal graph if available
        if self.causal_graph and message.performative:
            envelope = message.to_semantic_envelope()
            self.causal_graph.track_message(envelope)
        
        # Queue for processing
        await self.message_queue.put(message)
        
        self.metrics["messages_sent"] += 1
        
        logger.debug(
            "Message queued",
            message_id=message.id,
            type=message.type.value,
            target=target_id
        )
        
        return True
    
    async def broadcast(
        self,
        sender_id: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude: Optional[Set[str]] = None
    ) -> int:
        """Broadcast message to all connected nodes"""
        message = MeshMessage(
            type=MessageType.BROADCAST,
            sender_id=sender_id,
            payload=payload,
            priority=priority
        )
        
        exclude = exclude or set()
        exclude.add(sender_id)
        
        # Send to all nodes except excluded
        sent_count = 0
        for node_id in self.nodes:
            if node_id not in exclude:
                await self.send_message(message, node_id)
                sent_count += 1
        
        return sent_count
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[MeshMessage, str], Any],
        node_id: Optional[str] = None
    ):
        """Register a message handler"""
        key = f"{message_type.value}:{node_id or '*'}"
        if key not in self.message_handlers:
            self.message_handlers[key] = []
        self.message_handlers[key].append(handler)
    
    async def _default_handler(self, message: MeshMessage, node_id: str):
        """Default message handler"""
        logger.debug(
            "Message received",
            node_id=node_id,
            message_id=message.id,
            type=message.type.value
        )
    
    # ==================== Consensus Mechanisms ====================
    
    async def initiate_consensus(
        self,
        initiator_id: str,
        topic: str,
        proposal: Any,
        participants: Optional[Set[str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Initiate consensus among nodes"""
        consensus_id = f"consensus_{uuid.uuid4().hex[:8]}"
        
        # Determine participants
        if participants is None:
            # Use all healthy connected nodes
            node = self.nodes.get(initiator_id)
            if node:
                participants = {
                    nid for nid in node.connections
                    if self.nodes[nid].status in [NodeStatus.HEALTHY, NodeStatus.HEALING]
                }
            else:
                participants = set()
        
        self.consensus_groups[consensus_id] = participants
        self.consensus_votes[consensus_id] = {}
        
        # Send consensus request
        message = MeshMessage(
            type=MessageType.CONSENSUS,
            sender_id=initiator_id,
            payload={
                "consensus_id": consensus_id,
                "topic": topic,
                "proposal": proposal,
                "timeout": timeout
            },
            priority=MessagePriority.HIGH
        )
        
        for participant in participants:
            await self.send_message(message, participant)
        
        # Wait for votes
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            votes = self.consensus_votes[consensus_id]
            if len(votes) >= len(participants) * 0.67:  # 2/3 majority
                break
            await asyncio.sleep(0.1)
        
        # Calculate result
        result = self._calculate_consensus(consensus_id, self.consensus_votes[consensus_id])
        
        # Clean up
        del self.consensus_groups[consensus_id]
        del self.consensus_votes[consensus_id]
        
        if result["consensus"] is not None:
            self.metrics["consensus_reached"] += 1
        
        return result
    
    def _calculate_consensus(
        self,
        consensus_id: str,
        votes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate consensus from votes"""
        if not votes:
            return {
                "consensus": None,
                "support": 0.0,
                "votes": 0,
                "total": len(self.consensus_groups.get(consensus_id, []))
            }
        
        # Weight votes by consciousness level
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        
        for node_id, vote in votes.items():
            node = self.nodes.get(node_id)
            if node:
                weight = node.consciousness_level
                vote_key = json.dumps(vote, sort_keys=True)
                weighted_votes[vote_key] += weight
                total_weight += weight
        
        # Find majority
        for vote_key, weight in weighted_votes.items():
            if weight > total_weight * 0.5:
                return {
                    "consensus": json.loads(vote_key),
                    "support": weight / total_weight,
                    "votes": len(votes),
                    "total": len(self.consensus_groups.get(consensus_id, []))
                }
        
        # No clear majority
        return {
            "consensus": None,
            "support": max(weighted_votes.values()) / total_weight if weighted_votes else 0.0,
            "votes": len(votes),
            "total": len(self.consensus_groups.get(consensus_id, [])),
            "distribution": {
                json.loads(k): v/total_weight 
                for k, v in weighted_votes.items()
            }
        }
    
    # ==================== Background Tasks ====================
    
    async def _message_processor(self):
        """Process messages from the queue"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                if message.type == MessageType.DIRECT:
                    await self._route_direct_message(message)
                elif message.type == MessageType.BROADCAST:
                    await self._handle_broadcast(message)
                elif message.type == MessageType.CONSENSUS:
                    await self._handle_consensus(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _route_direct_message(self, message: MeshMessage):
        """Route a direct message to target"""
        if not message.target_id or message.target_id not in self.nodes:
            self.metrics["messages_failed"] += 1
            return
        
        # Find best path
        source_id = message.sender_id or message.path[-1] if message.path else None
        if not source_id:
            self.metrics["messages_failed"] += 1
            return
        
        path = await self.router.find_best_path(source_id, message.target_id, message)
        
        if not path:
            self.metrics["messages_failed"] += 1
            logger.warning(
                "No path found",
                source=source_id,
                target=message.target_id
            )
            return
        
        # Update message path
        message.path = path.nodes
        
        # Deliver to target
        await self._deliver_message(message, message.target_id)
        self.metrics["messages_delivered"] += 1
    
    async def _handle_broadcast(self, message: MeshMessage):
        """Handle broadcast message"""
        delivered = 0
        for node_id in self.nodes:
            if node_id != message.sender_id:
                await self._deliver_message(message, node_id)
                delivered += 1
        
        self.metrics["messages_delivered"] += delivered
    
    async def _handle_consensus(self, message: MeshMessage):
        """Handle consensus message"""
        consensus_id = message.payload.get("consensus_id")
        
        if consensus_id and message.target_id:
            # Deliver to participant
            await self._deliver_message(message, message.target_id)
            
            # Auto-vote based on consciousness level (simplified)
            node = self.nodes.get(message.target_id)
            if node and node.consciousness_level > 0.5:
                self.consensus_votes[consensus_id][message.target_id] = {
                    "accept": True,
                    "confidence": node.consciousness_level
                }
    
    async def _deliver_message(self, message: MeshMessage, node_id: str):
        """Deliver message to node handlers"""
        # Find handlers
        handlers = []
        
        # Specific node handlers
        key = f"{message.type.value}:{node_id}"
        handlers.extend(self.message_handlers.get(key, []))
        
        # Wildcard handlers
        key = f"{message.type.value}:*"
        handlers.extend(self.message_handlers.get(key, []))
        
        # Execute handlers
        for handler in handlers:
            try:
                await handler(message, node_id)
            except Exception as e:
                logger.error(f"Handler error: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                for node in list(self.nodes.values()):
                    if node.status == NodeStatus.HEALTHY:
                        # Update heartbeat
                        node.last_heartbeat = datetime.utcnow()
                        
                        # Send heartbeat to connections
                        message = MeshMessage(
                            type=MessageType.HEARTBEAT,
                            sender_id=node.id,
                            payload={
                                "status": node.status.value,
                                "consciousness": node.consciousness_level,
                                "metrics": {
                                    "messages_processed": node.messages_processed,
                                    "avg_latency": node.average_latency,
                                    "error_rate": node.error_rate
                                }
                            },
                            priority=MessagePriority.BACKGROUND
                        )
                        
                        for connected_id in node.connections:
                            await self.send_message(message, connected_id)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _topology_optimizer(self):
        """Optimize network topology periodically"""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Run every 30 seconds
                
                # Recalculate connections for nodes with poor connectivity
                for node in list(self.nodes.values()):
                    if len(node.connections) < 3 or node.status == NodeStatus.HEALING:
                        await self._connect_node(node)
                
                # Remove connections to failed nodes
                for node in list(self.nodes.values()):
                    failed_connections = [
                        nid for nid in node.connections
                        if self.nodes.get(nid, NeuralNode("")).status == NodeStatus.FAILED
                    ]
                    for nid in failed_connections:
                        node.connections.discard(nid)
                        if self.topology.has_edge(node.id, nid):
                            self.topology.remove_edge(node.id, nid)
                
            except Exception as e:
                logger.error(f"Topology optimization error: {e}")
    
    async def _health_monitor(self):
        """Monitor node health"""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                for node in list(self.nodes.values()):
                    # Check heartbeat timeout
                    if (datetime.utcnow() - node.last_heartbeat).total_seconds() > 30:
                        node.status = NodeStatus.FAILED
                    else:
                        # Update health based on metrics
                        node.update_health()
                
                # Calculate average consciousness
                if self.nodes:
                    avg_consciousness = sum(
                        n.consciousness_level for n in self.nodes.values()
                    ) / len(self.nodes)
                    self.metrics["avg_consciousness"] = avg_consciousness
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _consciousness_updater(self):
        """Update consciousness levels based on activity"""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
                for node in list(self.nodes.values()):
                    # Increase consciousness for active nodes
                    if node.messages_processed > 0:
                        activity_factor = min(1.0, node.messages_processed / 100)
                        node.consciousness_level = min(
                            1.0,
                            node.consciousness_level + 0.01 * activity_factor
                        )
                    
                    # Decrease consciousness for idle nodes
                    else:
                        node.consciousness_level = max(
                            0.1,
                            node.consciousness_level * 0.99
                        )
                    
                    # Reset message counter
                    node.messages_processed = 0
                
            except Exception as e:
                logger.error(f"Consciousness update error: {e}")
    
    async def _self_healing_loop(self):
        """Self-healing mechanisms"""
        while self._running:
            try:
                await asyncio.sleep(20.0)  # Run every 20 seconds
                
                # Heal degraded nodes
                for node in list(self.nodes.values()):
                    if node.status == NodeStatus.DEGRADED:
                        # Attempt healing
                        node.status = NodeStatus.HEALING
                        
                        # Reset error rate gradually
                        node.error_rate *= 0.9
                        
                        # Reconnect to healthy nodes
                        await self._connect_node(node)
                        
                        self.metrics["healings_performed"] += 1
                        
                        logger.info(
                            "Node healing initiated",
                            node_id=node.id,
                            error_rate=node.error_rate
                        )
                
                # Quarantine problematic nodes
                for node in list(self.nodes.values()):
                    if node.error_rate > 0.8:
                        node.status = NodeStatus.QUARANTINED
                        
                        # Remove from all connections
                        for other_node in self.nodes.values():
                            other_node.connections.discard(node.id)
                        
                        node.connections.clear()
                        
                        logger.warning(
                            "Node quarantined",
                            node_id=node.id,
                            error_rate=node.error_rate
                        )
                
            except Exception as e:
                logger.error(f"Self-healing error: {e}")
    
    # ==================== Utility Methods ====================
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a node"""
        node = self.nodes.get(node_id)
        if not node:
            return None
        
        return {
            "id": node.id,
            "status": node.status.value,
            "consciousness_level": node.consciousness_level,
            "connections": list(node.connections),
            "position": node.position.tolist(),
            "metrics": {
                "messages_processed": node.messages_processed,
                "average_latency": node.average_latency,
                "error_rate": node.error_rate
            },
            "attention_focus": node.attention_focus,
            "emergent_patterns": node.emergent_patterns
        }
    
    def get_topology_metrics(self) -> Dict[str, Any]:
        """Get network topology metrics"""
        if not self.topology.nodes():
            return {
                "nodes": 0,
                "edges": 0,
                "avg_degree": 0,
                "connectivity": 0,
                "diameter": 0
            }
        
        return {
            "nodes": self.topology.number_of_nodes(),
            "edges": self.topology.number_of_edges(),
            "avg_degree": sum(dict(self.topology.degree()).values()) / self.topology.number_of_nodes(),
            "connectivity": nx.node_connectivity(self.topology) if nx.is_connected(self.topology.to_undirected()) else 0,
            "diameter": nx.diameter(self.topology) if nx.is_strongly_connected(self.topology) else -1,
            "clustering": nx.average_clustering(self.topology.to_undirected())
        }
    
    def visualize_topology(self) -> Dict[str, Any]:
        """Get topology data for visualization"""
        nodes_data = []
        edges_data = []
        
        # Node data
        for node_id, node in self.nodes.items():
            nodes_data.append({
                "id": node_id,
                "x": float(node.position[0]),
                "y": float(node.position[1]),
                "z": float(node.position[2]),
                "consciousness": node.consciousness_level,
                "status": node.status.value,
                "connections": len(node.connections)
            })
        
        # Edge data
        for u, v, data in self.topology.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "strength": data.get("strength", 0.5)
            })
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "metrics": self.get_topology_metrics()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mesh performance metrics"""
        return {
            **self.metrics,
            **self.get_topology_metrics()
        }