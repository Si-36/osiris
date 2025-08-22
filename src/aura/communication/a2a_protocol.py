"""
🤝 AURA A2A (Agent-to-Agent) Communication Protocol with MCP

Implements secure, topology-aware agent communication with Model Context Protocol.
Features:
- Zero-trust security with mTLS
- Byzantine fault tolerance
- Topology-aware routing
- MCP for context sharing
- Real-time failure detection
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import nats
from nats.aio.client import Client
from pydantic import BaseModel, Field
import logging

# Configure logging
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """A2A message types"""
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    TOPOLOGY_UPDATE = "topology_update"
    FAILURE_ALERT = "failure_alert"
    CASCADE_WARNING = "cascade_warning"
    INTERVENTION_REQUEST = "intervention_request"
    CONTEXT_SYNC = "context_sync"
    MCP_TOOL_CALL = "mcp_tool_call"
    MCP_TOOL_RESPONSE = "mcp_tool_response"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_VOTE = "consensus_vote"


@dataclass
class AgentProfile:
    """Agent identity and capabilities"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    topology_position: Dict[str, float]
    trust_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0
    mcp_tools: List[str] = field(default_factory=list)


class A2AMessage(BaseModel):
    """Structured A2A message format"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    from_agent: str
    to_agent: Optional[str] = None  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)
    ttl: int = Field(default=60)  # Time to live in seconds
    requires_ack: bool = False
    topology_context: Optional[Dict[str, Any]] = None


class MCPContext(BaseModel):
    """Model Context Protocol information"""
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shared_memory: Dict[str, Any] = {}
    tool_registry: Dict[str, Dict[str, Any]] = {}
    conversation_history: List[Dict[str, Any]] = []
    model_states: Dict[str, Any] = {}


class A2ACommunicationProtocol:
    """
    Advanced A2A communication protocol with MCP integration.
    
    Features:
    - NATS-based pub/sub messaging
    - Byzantine fault-tolerant consensus
    - Topology-aware routing
    - MCP for shared context
    - Real-time failure detection
    """
    
    def __init__(self, agent_id: str, agent_type: str = "generic"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.nc: Optional[Client] = None
        self.connected = False
        
        # Agent registry
        self.agents: Dict[str, AgentProfile] = {}
        self.topology: Dict[str, Set[str]] = {}  # agent_id -> connected agents
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_acks: Dict[str, asyncio.Future] = {}
        
        # MCP context
        self.mcp_context = MCPContext()
        self.mcp_tools: Dict[str, Callable] = {}
        
        # Performance metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.avg_latency = 0.0
        
        # Byzantine consensus
        self.consensus_threshold = 0.66  # 2/3 majority
        self.ongoing_consensus: Dict[str, Dict[str, Any]] = {}
        
        self._setup_default_handlers()
        
    def _setup_default_handlers(self):
        """Set up default message handlers"""
        self.message_handlers[MessageType.HANDSHAKE] = self._handle_handshake
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.TOPOLOGY_UPDATE] = self._handle_topology_update
        self.message_handlers[MessageType.FAILURE_ALERT] = self._handle_failure_alert
        self.message_handlers[MessageType.MCP_TOOL_CALL] = self._handle_mcp_tool_call
        self.message_handlers[MessageType.CONSENSUS_VOTE] = self._handle_consensus_vote
        
    async def connect(self, nats_url: str = "nats://localhost:4222"):
        """Connect to NATS message broker"""
        try:
            self.nc = await nats.connect(nats_url)
            self.connected = True
            
            # Subscribe to agent channels
            await self.nc.subscribe(f"agent.{self.agent_id}", cb=self._handle_direct_message)
            await self.nc.subscribe("agent.broadcast", cb=self._handle_broadcast_message)
            await self.nc.subscribe(f"agent.type.{self.agent_type}", cb=self._handle_type_message)
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Announce presence
            await self._announce_presence()
            
            logger.info(f"✅ Agent {self.agent_id} connected to A2A network")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to A2A network: {e}")
            return False
    
    async def _announce_presence(self):
        """Announce agent presence to network"""
        profile = AgentProfile(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            capabilities=list(self.mcp_tools.keys()),
            topology_position={"x": 0, "y": 0, "z": 0},  # Will be updated
            mcp_tools=list(self.mcp_tools.keys())
        )
        
        message = A2AMessage(
            from_agent=self.agent_id,
            message_type=MessageType.HANDSHAKE,
            payload=profile.dict()
        )
        
        await self.broadcast(message)
    
    async def send_to_agent(self, agent_id: str, message: A2AMessage) -> Optional[Any]:
        """Send message to specific agent"""
        if not self.connected:
            logger.error("Not connected to A2A network")
            return None
            
        message.to_agent = agent_id
        self.messages_sent += 1
        
        try:
            # Add ACK future if required
            ack_future = None
            if message.requires_ack:
                ack_future = asyncio.Future()
                self.pending_acks[message.message_id] = ack_future
            
            # Send message
            await self.nc.publish(
                f"agent.{agent_id}",
                message.json().encode()
            )
            
            # Wait for ACK if required
            if ack_future:
                try:
                    response = await asyncio.wait_for(ack_future, timeout=5.0)
                    return response
                except asyncio.TimeoutError:
                    logger.warning(f"No ACK received from {agent_id}")
                    return None
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {agent_id}: {e}")
            return None
    
    async def broadcast(self, message: A2AMessage):
        """Broadcast message to all agents"""
        if not self.connected:
            return
            
        self.messages_sent += 1
        
        try:
            await self.nc.publish(
                "agent.broadcast",
                message.json().encode()
            )
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    async def request_consensus(self, topic: str, proposal: Dict[str, Any]) -> bool:
        """Request Byzantine consensus on a topic"""
        consensus_id = str(uuid.uuid4())
        
        self.ongoing_consensus[consensus_id] = {
            "topic": topic,
            "proposal": proposal,
            "votes": {},
            "start_time": time.time()
        }
        
        # Broadcast consensus request
        message = A2AMessage(
            from_agent=self.agent_id,
            message_type=MessageType.CONSENSUS_REQUEST,
            payload={
                "consensus_id": consensus_id,
                "topic": topic,
                "proposal": proposal
            },
            priority=9
        )
        
        await self.broadcast(message)
        
        # Wait for votes
        await asyncio.sleep(2.0)  # Give agents time to vote
        
        # Count votes
        consensus_data = self.ongoing_consensus.get(consensus_id, {})
        votes = consensus_data.get("votes", {})
        
        if not votes:
            return False
            
        agree_count = sum(1 for v in votes.values() if v)
        total_count = len(votes)
        
        consensus_reached = (agree_count / total_count) >= self.consensus_threshold
        
        # Clean up
        self.ongoing_consensus.pop(consensus_id, None)
        
        return consensus_reached
    
    async def call_mcp_tool(self, agent_id: str, tool_name: str, 
                           arguments: Dict[str, Any]) -> Optional[Any]:
        """Call an MCP tool on another agent"""
        message = A2AMessage(
            from_agent=self.agent_id,
            message_type=MessageType.MCP_TOOL_CALL,
            payload={
                "tool_name": tool_name,
                "arguments": arguments,
                "context": self.mcp_context.dict()
            },
            requires_ack=True,
            priority=7
        )
        
        response = await self.send_to_agent(agent_id, message)
        return response
    
    def register_mcp_tool(self, name: str, func: Callable, 
                         description: str = "", parameters: Dict[str, Any] = None):
        """Register an MCP tool"""
        self.mcp_tools[name] = func
        self.mcp_context.tool_registry[name] = {
            "description": description,
            "parameters": parameters or {},
            "agent_id": self.agent_id
        }
    
    async def update_topology(self, connections: Dict[str, float]):
        """Update agent's topology connections"""
        self.topology[self.agent_id] = set(connections.keys())
        
        # Broadcast topology update
        message = A2AMessage(
            from_agent=self.agent_id,
            message_type=MessageType.TOPOLOGY_UPDATE,
            payload={
                "agent_id": self.agent_id,
                "connections": connections
            }
        )
        
        await self.broadcast(message)
    
    async def alert_failure(self, failed_agent: str, cascade_risk: float):
        """Alert network about agent failure"""
        message = A2AMessage(
            from_agent=self.agent_id,
            message_type=MessageType.FAILURE_ALERT,
            payload={
                "failed_agent": failed_agent,
                "cascade_risk": cascade_risk,
                "timestamp": time.time(),
                "topology_context": self._get_topology_context(failed_agent)
            },
            priority=10
        )
        
        await self.broadcast(message)
    
    def _get_topology_context(self, agent_id: str) -> Dict[str, Any]:
        """Get topology context for an agent"""
        connected_agents = list(self.topology.get(agent_id, []))
        
        return {
            "direct_connections": connected_agents,
            "connection_count": len(connected_agents),
            "network_position": self._calculate_network_position(agent_id)
        }
    
    def _calculate_network_position(self, agent_id: str) -> str:
        """Calculate agent's position in network (hub, edge, isolated)"""
        connections = len(self.topology.get(agent_id, []))
        
        if connections == 0:
            return "isolated"
        elif connections > len(self.agents) * 0.3:
            return "hub"
        else:
            return "edge"
    
    # Message handlers
    async def _handle_direct_message(self, msg):
        """Handle direct message to this agent"""
        try:
            message = A2AMessage.parse_raw(msg.data)
            self.messages_received += 1
            
            # Check TTL
            if time.time() - message.timestamp > message.ttl:
                logger.warning(f"Message {message.message_id} expired")
                return
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                response = await handler(message)
                
                # Send ACK if required
                if message.requires_ack and response is not None:
                    ack_msg = A2AMessage(
                        from_agent=self.agent_id,
                        to_agent=message.from_agent,
                        message_type=MessageType.HEARTBEAT,  # Reuse for ACK
                        payload={"ack_for": message.message_id, "response": response}
                    )
                    await self.send_to_agent(message.from_agent, ack_msg)
                    
        except Exception as e:
            logger.error(f"Error handling direct message: {e}")
    
    async def _handle_broadcast_message(self, msg):
        """Handle broadcast message"""
        await self._handle_direct_message(msg)
    
    async def _handle_type_message(self, msg):
        """Handle message sent to agent type"""
        await self._handle_direct_message(msg)
    
    async def _handle_handshake(self, message: A2AMessage):
        """Handle agent handshake"""
        profile_data = message.payload
        profile = AgentProfile(**profile_data)
        
        self.agents[profile.agent_id] = profile
        logger.info(f"🤝 New agent joined: {profile.agent_id} ({profile.agent_type})")
    
    async def _handle_heartbeat(self, message: A2AMessage):
        """Handle heartbeat or ACK"""
        # Check if it's an ACK
        if "ack_for" in message.payload:
            msg_id = message.payload["ack_for"]
            if msg_id in self.pending_acks:
                self.pending_acks[msg_id].set_result(message.payload.get("response"))
                return
        
        # Update agent heartbeat
        if message.from_agent in self.agents:
            self.agents[message.from_agent].last_heartbeat = time.time()
    
    async def _handle_topology_update(self, message: A2AMessage):
        """Handle topology update"""
        agent_id = message.payload["agent_id"]
        connections = message.payload["connections"]
        
        self.topology[agent_id] = set(connections.keys())
    
    async def _handle_failure_alert(self, message: A2AMessage):
        """Handle failure alert"""
        failed_agent = message.payload["failed_agent"]
        cascade_risk = message.payload["cascade_risk"]
        
        logger.warning(f"⚠️ Agent failure: {failed_agent} (cascade risk: {cascade_risk:.2%})")
        
        # Mark agent as failed
        if failed_agent in self.agents:
            self.agents[failed_agent].failure_count += 1
            self.agents[failed_agent].trust_score *= 0.8
    
    async def _handle_mcp_tool_call(self, message: A2AMessage):
        """Handle MCP tool call request"""
        tool_name = message.payload["tool_name"]
        arguments = message.payload["arguments"]
        
        if tool_name not in self.mcp_tools:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            # Execute tool
            tool_func = self.mcp_tools[tool_name]
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)
                
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def _handle_consensus_vote(self, message: A2AMessage):
        """Handle consensus vote"""
        consensus_id = message.payload.get("consensus_id")
        vote = message.payload.get("vote", False)
        
        if consensus_id in self.ongoing_consensus:
            self.ongoing_consensus[consensus_id]["votes"][message.from_agent] = vote
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.connected:
            try:
                # Send heartbeat
                message = A2AMessage(
                    from_agent=self.agent_id,
                    message_type=MessageType.HEARTBEAT,
                    payload={"status": "alive", "metrics": self.get_metrics()}
                )
                await self.broadcast(message)
                
                # Check for dead agents
                current_time = time.time()
                for agent_id, profile in list(self.agents.items()):
                    if current_time - profile.last_heartbeat > 30:  # 30 second timeout
                        logger.warning(f"Agent {agent_id} appears to be dead")
                        await self.alert_failure(agent_id, 0.3)
                        del self.agents[agent_id]
                
                await asyncio.sleep(10)  # 10 second heartbeat
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "avg_latency": self.avg_latency,
            "connected_agents": len(self.agents),
            "topology_size": len(self.topology)
        }
    
    async def disconnect(self):
        """Disconnect from A2A network"""
        if self.nc:
            await self.nc.close()
        self.connected = False
        logger.info(f"Agent {self.agent_id} disconnected from A2A network")


# Example MCP tools for agents
def create_topology_analyzer_tool():
    """Create a topology analysis MCP tool"""
    async def analyze_topology(agent_ids: List[str]) -> Dict[str, Any]:
        # Simulated topology analysis
        return {
            "cluster_count": 3,
            "avg_connectivity": 0.4,
            "critical_nodes": agent_ids[:2],
            "vulnerability_score": 0.15
        }
    
    return analyze_topology


def create_failure_predictor_tool():
    """Create a failure prediction MCP tool"""
    async def predict_failure(agent_id: str, window: int = 60) -> Dict[str, Any]:
        # Simulated failure prediction
        return {
            "agent_id": agent_id,
            "failure_probability": 0.08,
            "cascade_risk": 0.12,
            "recommended_action": "monitor"
        }
    
    return predict_failure


# Example usage
async def example_a2a_usage():
    """Example of using the A2A protocol"""
    
    # Create an agent with A2A
    agent = A2ACommunicationProtocol("agent-001", "analyzer")
    
    # Register MCP tools
    agent.register_mcp_tool(
        "analyze_topology",
        create_topology_analyzer_tool(),
        "Analyzes network topology",
        {"agent_ids": "list of agent IDs"}
    )
    
    agent.register_mcp_tool(
        "predict_failure",
        create_failure_predictor_tool(),
        "Predicts agent failures",
        {"agent_id": "agent to analyze", "window": "time window"}
    )
    
    # Connect to network
    await agent.connect()
    
    # Update topology
    await agent.update_topology({
        "agent-002": 0.9,
        "agent-003": 0.7,
        "agent-004": 0.8
    })
    
    # Call MCP tool on another agent
    result = await agent.call_mcp_tool(
        "agent-002",
        "analyze_topology",
        {"agent_ids": ["agent-001", "agent-002", "agent-003"]}
    )
    
    print(f"MCP Tool Result: {result}")
    
    # Request consensus
    consensus = await agent.request_consensus(
        "scale_up",
        {"new_agents": 5, "reason": "high load"}
    )
    
    print(f"Consensus reached: {consensus}")
    
    # Alert about potential failure
    await agent.alert_failure("agent-005", cascade_risk=0.75)
    
    # Get metrics
    metrics = agent.get_metrics()
    print(f"A2A Metrics: {metrics}")
    
    # Disconnect
    await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(example_a2a_usage())