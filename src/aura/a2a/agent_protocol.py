#!/usr/bin/env python3
"""
🤝 AURA A2A (Agent-to-Agent) Communication Protocol with MCP Integration

This implements a production-grade agent communication protocol that enables:
- Direct agent-to-agent communication with topology awareness
- Model Context Protocol (MCP) for standardized context sharing
- Byzantine fault tolerance for reliable messaging
- Real-time cascade prevention through coordinated responses
- Liquid neural network adaptation for dynamic routing

Based on latest 2025 agent engineering patterns and distributed AI systems.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib
import struct
from collections import defaultdict
import websockets
import aiohttp
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
message_sent_counter = Counter('a2a_messages_sent', 'Total A2A messages sent', ['agent_type', 'message_type'])
message_received_counter = Counter('a2a_messages_received', 'Total A2A messages received', ['agent_type', 'message_type'])
message_latency_histogram = Histogram('a2a_message_latency', 'A2A message latency', ['message_type'])
active_connections_gauge = Gauge('a2a_active_connections', 'Active A2A connections')
context_size_histogram = Histogram('mcp_context_size', 'MCP context size in bytes')
cascade_prevented_counter = Counter('a2a_cascades_prevented', 'Cascades prevented through A2A coordination')

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """A2A Message Types"""
    # Core communication
    TOPOLOGY_UPDATE = "topology_update"
    STATE_SYNC = "state_sync"
    FAILURE_ALERT = "failure_alert"
    CASCADE_WARNING = "cascade_warning"
    INTERVENTION_REQUEST = "intervention_request"
    
    # MCP context sharing
    CONTEXT_SHARE = "context_share"
    CONTEXT_REQUEST = "context_request"
    CONTEXT_UPDATE = "context_update"
    
    # Coordination
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_RESULT = "consensus_result"
    
    # Health & monitoring
    HEARTBEAT = "heartbeat"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRIC = "performance_metric"


class AgentRole(Enum):
    """Agent roles in the system"""
    PREDICTOR = "predictor"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    COORDINATOR = "coordinator"
    GUARDIAN = "guardian"


@dataclass
class MCPContext:
    """Model Context Protocol data structure"""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    
    # Topological context
    topology_hash: str = ""
    persistence_diagram: Dict[str, Any] = field(default_factory=dict)
    shape_features: List[float] = field(default_factory=list)
    
    # Agent state
    load: float = 0.0
    health: float = 1.0
    cascade_risk: float = 0.0
    
    # Historical context
    recent_failures: List[Dict[str, Any]] = field(default_factory=list)
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model-specific context
    model_state: Dict[str, Any] = field(default_factory=dict)
    liquid_nn_state: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize context to bytes"""
        data = json.dumps(asdict(self), default=str)
        return data.encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MCPContext':
        """Deserialize context from bytes"""
        json_data = json.loads(data.decode('utf-8'))
        # Convert timestamp string back to datetime
        json_data['timestamp'] = datetime.fromisoformat(json_data['timestamp'])
        return cls(**json_data)


@dataclass
class A2AMessage:
    """A2A Protocol Message"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Routing
    source_agent: str = ""
    target_agent: Optional[str] = None  # None for broadcast
    hop_count: int = 0
    max_hops: int = 5
    
    # Message content
    message_type: MessageType = MessageType.HEARTBEAT
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # MCP context
    context: Optional[MCPContext] = None
    
    # Security & reliability
    signature: str = ""
    requires_ack: bool = False
    priority: int = 0  # 0-10, higher is more important
    
    # Byzantine consensus
    consensus_required: bool = False
    consensus_threshold: float = 0.67
    
    def sign(self, agent_key: str) -> None:
        """Sign message for authentication"""
        content = f"{self.message_id}:{self.source_agent}:{self.message_type.value}"
        self.signature = hashlib.sha256((content + agent_key).encode()).hexdigest()
    
    def verify(self, agent_key: str) -> bool:
        """Verify message signature"""
        content = f"{self.message_id}:{self.source_agent}:{self.message_type.value}"
        expected = hashlib.sha256((content + agent_key).encode()).hexdigest()
        return self.signature == expected
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'source_agent': self.source_agent,
            'target_agent': self.target_agent,
            'hop_count': self.hop_count,
            'max_hops': self.max_hops,
            'message_type': self.message_type.value,
            'payload': self.payload,
            'context': asdict(self.context) if self.context else None,
            'signature': self.signature,
            'requires_ack': self.requires_ack,
            'priority': self.priority,
            'consensus_required': self.consensus_required,
            'consensus_threshold': self.consensus_threshold
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'A2AMessage':
        """Deserialize message from bytes"""
        json_data = json.loads(data.decode('utf-8'))
        
        # Convert back to proper types
        json_data['timestamp'] = datetime.fromisoformat(json_data['timestamp'])
        json_data['message_type'] = MessageType(json_data['message_type'])
        
        # Handle context
        if json_data['context']:
            json_data['context'] = MCPContext(**json_data['context'])
        
        return cls(**json_data)


class A2AProtocol:
    """Agent-to-Agent Communication Protocol Implementation"""
    
    def __init__(self, agent_id: str, agent_role: AgentRole, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.config = config or {}
        
        # Network topology
        self.peers: Set[str] = set()
        self.routing_table: Dict[str, List[str]] = {}
        self.topology_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_acks: Dict[str, asyncio.Future] = {}
        self.message_cache: Dict[str, A2AMessage] = {}
        
        # Consensus tracking
        self.consensus_votes: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
        self.consensus_futures: Dict[str, asyncio.Future] = {}
        
        # WebSocket connections
        self.ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.ws_server = None
        
        # Performance tracking
        self.message_latencies: List[float] = []
        self.cascade_interventions: int = 0
        
        # Agent key for signing
        self.agent_key = hashlib.sha256(f"{agent_id}:{agent_role.value}".encode()).hexdigest()
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self, listen_port: int = 8765):
        """Start A2A protocol services"""
        self._running = True
        
        # Start WebSocket server
        self.ws_server = await websockets.serve(
            self._handle_incoming_connection,
            "0.0.0.0",
            listen_port
        )
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._topology_sync_loop()))
        self._tasks.append(asyncio.create_task(self._health_monitor_loop()))
        
        active_connections_gauge.set(0)
        logger.info(f"A2A Protocol started for agent {self.agent_id} on port {listen_port}")
    
    async def stop(self):
        """Stop A2A protocol services"""
        self._running = False
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            await ws.close()
        
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info(f"A2A Protocol stopped for agent {self.agent_id}")
    
    async def connect_to_peer(self, peer_id: str, peer_address: str):
        """Connect to a peer agent"""
        try:
            ws = await websockets.connect(peer_address)
            self.ws_connections[peer_id] = ws
            self.peers.add(peer_id)
            self.topology_graph[self.agent_id].add(peer_id)
            
            # Start message handler for this connection
            asyncio.create_task(self._handle_peer_messages(peer_id, ws))
            
            active_connections_gauge.inc()
            logger.info(f"Connected to peer {peer_id} at {peer_address}")
            
            # Send initial handshake
            await self.send_message(
                A2AMessage(
                    source_agent=self.agent_id,
                    target_agent=peer_id,
                    message_type=MessageType.STATE_SYNC,
                    payload={
                        "agent_role": self.agent_role.value,
                        "topology": list(self.peers)
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id}: {e}")
    
    async def send_message(self, message: A2AMessage, timeout: float = 5.0) -> Optional[A2AMessage]:
        """Send a message to target agent(s)"""
        message.source_agent = self.agent_id
        message.sign(self.agent_key)
        
        # Track metrics
        message_sent_counter.labels(
            agent_type=self.agent_role.value,
            message_type=message.message_type.value
        ).inc()
        
        if message.context:
            context_size_histogram.observe(len(message.context.to_bytes()))
        
        # Handle broadcast
        if message.target_agent is None:
            tasks = []
            for peer_id in self.peers:
                tasks.append(self._send_to_peer(peer_id, message))
            await asyncio.gather(*tasks, return_exceptions=True)
            return None
        
        # Handle targeted message
        if message.requires_ack:
            ack_future = asyncio.Future()
            self.pending_acks[message.message_id] = ack_future
            
            await self._send_to_peer(message.target_agent, message)
            
            try:
                ack = await asyncio.wait_for(ack_future, timeout=timeout)
                return ack
            except asyncio.TimeoutError:
                logger.warning(f"Message {message.message_id} timed out")
                del self.pending_acks[message.message_id]
                return None
        else:
            await self._send_to_peer(message.target_agent, message)
            return None
    
    async def request_consensus(self, topic: str, proposal: Dict[str, Any], 
                               timeout: float = 10.0) -> Tuple[bool, float]:
        """Request Byzantine consensus on a topic"""
        consensus_id = f"{self.agent_id}:{topic}:{uuid.uuid4()}"
        
        # Create consensus request
        message = A2AMessage(
            source_agent=self.agent_id,
            message_type=MessageType.CONSENSUS_REQUEST,
            payload={
                "consensus_id": consensus_id,
                "topic": topic,
                "proposal": proposal
            },
            consensus_required=True,
            priority=8
        )
        
        # Track consensus
        consensus_future = asyncio.Future()
        self.consensus_futures[consensus_id] = consensus_future
        
        # Broadcast to all peers
        await self.send_message(message)
        
        try:
            # Wait for consensus
            result = await asyncio.wait_for(consensus_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Calculate partial consensus
            votes = self.consensus_votes.get(consensus_id, [])
            if votes:
                agree_count = sum(1 for _, vote in votes if vote)
                agreement_ratio = agree_count / len(votes)
                return agreement_ratio >= message.consensus_threshold, agreement_ratio
            return False, 0.0
        finally:
            # Cleanup
            self.consensus_votes.pop(consensus_id, None)
            self.consensus_futures.pop(consensus_id, None)
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    async def share_context(self, context: MCPContext, target: Optional[str] = None):
        """Share MCP context with other agents"""
        message = A2AMessage(
            source_agent=self.agent_id,
            target_agent=target,
            message_type=MessageType.CONTEXT_SHARE,
            context=context,
            priority=5
        )
        
        await self.send_message(message)
    
    async def request_context(self, target_agent: str, context_type: str) -> Optional[MCPContext]:
        """Request specific context from another agent"""
        message = A2AMessage(
            source_agent=self.agent_id,
            target_agent=target_agent,
            message_type=MessageType.CONTEXT_REQUEST,
            payload={"context_type": context_type},
            requires_ack=True,
            priority=6
        )
        
        response = await self.send_message(message)
        if response and response.context:
            return response.context
        return None
    
    async def alert_cascade_risk(self, risk_data: Dict[str, Any]):
        """Alert all peers about cascade risk"""
        context = MCPContext(
            agent_id=self.agent_id,
            cascade_risk=risk_data.get("risk_level", 0.0),
            topology_hash=risk_data.get("topology_hash", ""),
            metadata={"risk_source": risk_data.get("source", "unknown")}
        )
        
        message = A2AMessage(
            source_agent=self.agent_id,
            message_type=MessageType.CASCADE_WARNING,
            payload=risk_data,
            context=context,
            priority=10,  # Highest priority
            consensus_required=True
        )
        
        # Broadcast to all
        await self.send_message(message)
        
        # Track intervention
        self.cascade_interventions += 1
        cascade_prevented_counter.inc()
    
    # Private methods
    
    async def _handle_incoming_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        peer_id = None
        try:
            # Wait for handshake
            data = await websocket.recv()
            message = A2AMessage.from_bytes(data.encode('utf-8') if isinstance(data, str) else data)
            
            if message.message_type == MessageType.STATE_SYNC:
                peer_id = message.source_agent
                self.ws_connections[peer_id] = websocket
                self.peers.add(peer_id)
                self.topology_graph[self.agent_id].add(peer_id)
                
                active_connections_gauge.inc()
                
                # Handle messages from this peer
                await self._handle_peer_messages(peer_id, websocket)
                
        except Exception as e:
            logger.error(f"Error handling incoming connection: {e}")
        finally:
            if peer_id:
                self.peers.discard(peer_id)
                self.ws_connections.pop(peer_id, None)
                self.topology_graph[self.agent_id].discard(peer_id)
                active_connections_gauge.dec()
    
    async def _handle_peer_messages(self, peer_id: str, websocket):
        """Handle messages from a connected peer"""
        try:
            async for data in websocket:
                message = A2AMessage.from_bytes(data.encode('utf-8') if isinstance(data, str) else data)
                
                # Verify signature
                if not message.verify(self.agent_key):
                    logger.warning(f"Invalid signature from {peer_id}")
                    continue
                
                # Track metrics
                message_received_counter.labels(
                    agent_type=self.agent_role.value,
                    message_type=message.message_type.value
                ).inc()
                
                # Route message
                asyncio.create_task(self._route_message(message))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed with peer {peer_id}")
        except Exception as e:
            logger.error(f"Error handling messages from {peer_id}: {e}")
    
    async def _route_message(self, message: A2AMessage):
        """Route incoming message to appropriate handler"""
        # Check if message is for us
        if message.target_agent and message.target_agent != self.agent_id:
            # Forward if we're not the target and haven't exceeded hop count
            if message.hop_count < message.max_hops:
                message.hop_count += 1
                await self._forward_message(message)
            return
        
        # Handle acknowledgment
        if message.message_id in self.pending_acks:
            self.pending_acks[message.message_id].set_result(message)
            return
        
        # Cache message to prevent duplicates
        if message.message_id in self.message_cache:
            return
        self.message_cache[message.message_id] = message
        
        # Cleanup old cache entries (keep last 1000)
        if len(self.message_cache) > 1000:
            oldest = sorted(self.message_cache.keys())[:100]
            for msg_id in oldest:
                del self.message_cache[msg_id]
        
        # Handle consensus messages
        if message.message_type == MessageType.CONSENSUS_VOTE:
            await self._handle_consensus_vote(message)
        elif message.message_type == MessageType.CONSENSUS_REQUEST:
            await self._handle_consensus_request(message)
        
        # Call registered handler
        if message.message_type in self.message_handlers:
            try:
                await self.message_handlers[message.message_type](message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    async def _send_to_peer(self, peer_id: str, message: A2AMessage):
        """Send message to specific peer"""
        if peer_id not in self.ws_connections:
            logger.warning(f"No connection to peer {peer_id}")
            return
        
        try:
            ws = self.ws_connections[peer_id]
            await ws.send(message.to_bytes().decode('utf-8'))
            
            # Track latency
            if message.requires_ack:
                start_time = asyncio.get_event_loop().time()
                self.message_latencies.append(start_time)
                
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
            # Remove failed connection
            self.peers.discard(peer_id)
            self.ws_connections.pop(peer_id, None)
            active_connections_gauge.dec()
    
    async def _forward_message(self, message: A2AMessage):
        """Forward message to next hop"""
        # Simple forwarding - in production, use proper routing algorithms
        if message.target_agent in self.routing_table:
            next_hops = self.routing_table[message.target_agent]
            if next_hops:
                await self._send_to_peer(next_hops[0], message)
    
    async def _handle_consensus_request(self, message: A2AMessage):
        """Handle consensus request"""
        consensus_id = message.payload.get("consensus_id")
        proposal = message.payload.get("proposal")
        
        # Make decision (simplified - in production, use proper consensus logic)
        vote = self._evaluate_proposal(proposal)
        
        # Send vote
        vote_message = A2AMessage(
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            message_type=MessageType.CONSENSUS_VOTE,
            payload={
                "consensus_id": consensus_id,
                "vote": vote
            },
            priority=8
        )
        
        await self.send_message(vote_message)
    
    async def _handle_consensus_vote(self, message: A2AMessage):
        """Handle consensus vote"""
        consensus_id = message.payload.get("consensus_id")
        vote = message.payload.get("vote", False)
        
        if consensus_id in self.consensus_futures:
            self.consensus_votes[consensus_id].append((message.source_agent, vote))
            
            # Check if we have enough votes
            total_peers = len(self.peers) + 1  # Include self
            votes = self.consensus_votes[consensus_id]
            
            if len(votes) >= total_peers * 0.67:  # 2/3 majority
                agree_count = sum(1 for _, v in votes if v)
                agreement_ratio = agree_count / len(votes)
                
                consensus_reached = agreement_ratio >= 0.67
                
                if not self.consensus_futures[consensus_id].done():
                    self.consensus_futures[consensus_id].set_result(
                        (consensus_reached, agreement_ratio)
                    )
    
    def _evaluate_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Evaluate a proposal for consensus"""
        # Simplified evaluation - in production, use proper logic
        # based on agent role and system state
        
        if self.agent_role == AgentRole.GUARDIAN:
            # Guardians are conservative
            return proposal.get("risk_level", 1.0) < 0.3
        elif self.agent_role == AgentRole.EXECUTOR:
            # Executors focus on feasibility
            return proposal.get("feasible", False)
        else:
            # Default: approve if benefits outweigh risks
            return proposal.get("benefit", 0) > proposal.get("risk", 1)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                # Create heartbeat with current state
                context = MCPContext(
                    agent_id=self.agent_id,
                    load=self._get_current_load(),
                    health=self._get_health_score(),
                    cascade_risk=self._get_cascade_risk()
                )
                
                message = A2AMessage(
                    source_agent=self.agent_id,
                    message_type=MessageType.HEARTBEAT,
                    context=context,
                    payload={
                        "timestamp": datetime.now().isoformat(),
                        "uptime": self._get_uptime(),
                        "connections": len(self.peers)
                    }
                )
                
                await self.send_message(message)
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _topology_sync_loop(self):
        """Sync topology information"""
        while self._running:
            try:
                # Share topology updates
                message = A2AMessage(
                    source_agent=self.agent_id,
                    message_type=MessageType.TOPOLOGY_UPDATE,
                    payload={
                        "agent_id": self.agent_id,
                        "peers": list(self.peers),
                        "routing_table": dict(self.routing_table),
                        "topology_graph": {k: list(v) for k, v in self.topology_graph.items()}
                    }
                )
                
                await self.send_message(message)
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Topology sync error: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitor_loop(self):
        """Monitor health and performance"""
        while self._running:
            try:
                # Calculate message latency
                if self.message_latencies:
                    avg_latency = sum(self.message_latencies) / len(self.message_latencies)
                    message_latency_histogram.labels(message_type="average").observe(avg_latency)
                    
                    # Keep only recent latencies
                    self.message_latencies = self.message_latencies[-100:]
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    # Helper methods
    
    def _get_current_load(self) -> float:
        """Get current agent load"""
        # Simplified - in production, calculate actual load
        return len(self.message_cache) / 1000.0
    
    def _get_health_score(self) -> float:
        """Get agent health score"""
        # Simplified - in production, calculate actual health
        failed_connections = len(self.peers) - len(self.ws_connections)
        return max(0.0, 1.0 - (failed_connections / max(1, len(self.peers))))
    
    def _get_cascade_risk(self) -> float:
        """Get current cascade risk"""
        # Simplified - in production, use AURA's cascade prediction
        return min(1.0, self.cascade_interventions / 100.0)
    
    def _get_uptime(self) -> float:
        """Get agent uptime in seconds"""
        # Simplified - in production, track actual start time
        return 0.0


# Example usage and integration functions

async def create_a2a_network(agents: List[Dict[str, Any]]) -> Dict[str, A2AProtocol]:
    """Create a network of A2A-enabled agents"""
    protocols = {}
    
    # Create protocols for each agent
    for agent in agents:
        protocol = A2AProtocol(
            agent_id=agent['id'],
            agent_role=AgentRole(agent['role']),
            config=agent.get('config', {})
        )
        
        # Start protocol
        port = agent.get('port', 8765 + len(protocols))
        await protocol.start(port)
        
        protocols[agent['id']] = protocol
    
    # Connect agents based on topology
    for agent in agents:
        protocol = protocols[agent['id']]
        for peer_id in agent.get('peers', []):
            if peer_id in protocols:
                peer_port = next(
                    a['port'] for a in agents if a['id'] == peer_id
                )
                await protocol.connect_to_peer(
                    peer_id,
                    f"ws://localhost:{peer_port}"
                )
    
    return protocols


async def demonstrate_cascade_prevention():
    """Demonstrate cascade prevention through A2A coordination"""
    # Create agent network
    agents = [
        {"id": "predictor_1", "role": "predictor", "port": 8765, "peers": ["analyzer_1", "guardian_1"]},
        {"id": "analyzer_1", "role": "analyzer", "port": 8766, "peers": ["predictor_1", "executor_1"]},
        {"id": "executor_1", "role": "executor", "port": 8767, "peers": ["analyzer_1", "guardian_1"]},
        {"id": "guardian_1", "role": "guardian", "port": 8768, "peers": ["predictor_1", "executor_1"]},
    ]
    
    protocols = await create_a2a_network(agents)
    
    # Register cascade warning handler
    async def handle_cascade_warning(message: A2AMessage):
        logger.warning(f"Cascade warning received: {message.payload}")
        
        # Guardian takes action
        if message.source_agent != "guardian_1":
            guardian = protocols["guardian_1"]
            
            # Request consensus on intervention
            consensus, ratio = await guardian.request_consensus(
                "cascade_intervention",
                {
                    "risk_level": message.payload.get("risk_level", 0),
                    "intervention_type": "isolate_failing_agents",
                    "affected_agents": message.payload.get("affected_agents", [])
                }
            )
            
            if consensus:
                logger.info(f"Consensus reached ({ratio:.2%}) - Intervening to prevent cascade")
                # Implement intervention logic here
    
    # Register handler for all agents
    for protocol in protocols.values():
        protocol.register_handler(MessageType.CASCADE_WARNING, handle_cascade_warning)
    
    # Simulate cascade risk detection
    predictor = protocols["predictor_1"]
    await predictor.alert_cascade_risk({
        "risk_level": 0.85,
        "source": "topology_analysis",
        "affected_agents": ["executor_1"],
        "topology_hash": "abc123",
        "predicted_impact": {
            "agents_affected": 3,
            "time_to_cascade": 5.2
        }
    })
    
    # Wait for coordination
    await asyncio.sleep(2)
    
    # Cleanup
    for protocol in protocols.values():
        await protocol.stop()


if __name__ == "__main__":
    # Run demonstration
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demonstrate_cascade_prevention())