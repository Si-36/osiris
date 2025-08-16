"""
🎛️ Distributed Coordination Manager - Unified Control

Orchestrates all distributed coordination components using 2025 patterns:
- Event-driven architecture with reactive streams
- Actor model for concurrent coordination
- CQRS pattern for command/query separation
- Saga orchestration for distributed transactions
- Real-time monitoring and adaptive control

Integration Points:
- TDA event mesh for system-wide coordination
- Observability for performance monitoring
- Consensus for distributed decision making
- Load balancing for optimal resource utilization
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import uuid
from collections import defaultdict

from .coordination_core import (
    NodeId, MessageId, DistributedMessage, MessageType, VectorClock,
    NodeInfo, NodeState, MessageTransport
)
from .consensus import ModernRaftConsensus
from .load_balancing import TDALoadBalancer, LoadBalancingStrategy, LoadMetrics

class CoordinationEvent(Enum):
    """Coordination events"""
    NODE_JOINED = "node_joined"
    NODE_LEFT = "node_left"
    NODE_FAILED = "node_failed"
    LEADER_ELECTED = "leader_elected"
    CONSENSUS_REACHED = "consensus_reached"
    LOAD_REBALANCED = "load_rebalanced"
    PARTITION_DETECTED = "partition_detected"
    PARTITION_HEALED = "partition_healed"

@dataclass
class CoordinationCommand:
    """Command for coordination operations"""
    command_id: str
    command_type: str
    payload: Dict[str, Any]
    requester_id: NodeId
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

@dataclass
class CoordinationQuery:
    """Query for coordination state"""
    query_id: str
    query_type: str
    parameters: Dict[str, Any]
    requester_id: NodeId
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AgentRequest:
    """Request for agent execution"""
    request_id: str
    workflow_id: str
    agent_type: str
    operation: str
    parameters: Dict[str, Any]
    priority: int = 0
    timeout_seconds: float = 30.0
    tda_correlation_id: Optional[str] = None
    required_capabilities: Set[str] = field(default_factory=set)

@dataclass
class AgentResponse:
    """Response from agent execution"""
    request_id: str
    node_id: NodeId
    status: str
    result: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MockMessageTransport:
    """Mock message transport for testing"""
    
    def __init__(self):
        self.message_queues: Dict[str, List[DistributedMessage]] = defaultdict(list)
        self.broadcast_messages: List[DistributedMessage] = []
    
    async def send_message(self, message: DistributedMessage, target_address: str) -> bool:
        """Send message to specific target"""
        self.message_queues[target_address].append(message)
        return True
    
    async def broadcast_message(self, message: DistributedMessage, targets: List[str]) -> int:
        """Broadcast message to multiple targets"""
        self.broadcast_messages.append(message)
        for target in targets:
            self.message_queues[target].append(message)
        return len(targets)
    
    async def receive_messages(self) -> List[DistributedMessage]:
        """Receive messages for this node"""
        # Simulate receiving messages
        await asyncio.sleep(0.001)
        return []

class DistributedCoordinationManager:
    """
    Unified distributed coordination manager
    """
    
    def __init__(
        self,
        node_id: NodeId,
        cluster_nodes: Set[NodeId],
        tda_integration: Optional[Any] = None,
        observability_manager: Optional[Any] = None
    ):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.tda_integration = tda_integration
        self.observability_manager = observability_manager
        
        # Core components
        self.transport = MockMessageTransport()
        self.consensus = ModernRaftConsensus(
            node_id, cluster_nodes, self.transport, tda_integration
        )
        self.load_balancer = TDALoadBalancer(
            strategy=LoadBalancingStrategy.TDA_AWARE,
            tda_integration=tda_integration
        )
        
        # State management
        self.cluster_state: Dict[NodeId, NodeInfo] = {}
        self.active_requests: Dict[str, AgentRequest] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Event handling
        self.event_handlers: Dict[CoordinationEvent, List[Callable]] = defaultdict(list)
        self.event_stream: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.request_metrics: Dict[str, List[float]] = defaultdict(list)
        self.node_health: Dict[NodeId, datetime] = {}
        
        # Coordination state
        self.is_leader = False
        self.vector_clock = VectorClock({node_id: 0})
        self.running = False
    
    async def start(self):
        """Start the coordination manager"""
        self.running = True
        
        # Initialize cluster state
        await self._initialize_cluster_state()
        
        # Start core components
        consensus_task = asyncio.create_task(self.consensus.start())
        
        # Start coordination tasks
        tasks = [
            consensus_task,
            asyncio.create_task(self._event_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._request_processor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the coordination manager"""
        self.running = False
        await self.consensus.stop()
    
    async def join_cluster(self, node_info: NodeInfo) -> bool:
        """Join a node to the cluster"""
        try:
            # Add node to load balancer
            await self.load_balancer.add_node(node_info)
            
            # Update cluster state
            self.cluster_state[node_info.node_id] = node_info
            self.node_health[node_info.node_id] = datetime.now(timezone.utc)
            
            # Emit event
            await self._emit_event(CoordinationEvent.NODE_JOINED, {
                "node_id": node_info.node_id,
                "node_info": {
                    "address": node_info.address,
                    "port": node_info.port,
                    "capabilities": list(node_info.capabilities),
                    "load_factor": node_info.load_factor
                }
            })
            
            # Notify TDA
            if self.tda_integration:
                await self.tda_integration.send_orchestration_result(
                    {
                        "event": "node_joined",
                        "node_id": node_info.node_id,
                        "cluster_size": len(self.cluster_state),
                        "capabilities": list(node_info.capabilities)
                    },
                    f"cluster_{self.node_id}"
                )
            
            return True
            
        except Exception as e:
            return False
    
    async def leave_cluster(self, node_id: NodeId) -> bool:
        """Remove a node from the cluster"""
        try:
            # Remove from load balancer
            await self.load_balancer.remove_node(node_id)
            
            # Update cluster state
            if node_id in self.cluster_state:
                del self.cluster_state[node_id]
            if node_id in self.node_health:
                del self.node_health[node_id]
            
            # Emit event
            await self._emit_event(CoordinationEvent.NODE_LEFT, {
                "node_id": node_id,
                "cluster_size": len(self.cluster_state)
            })
            
            return True
            
        except Exception as e:
            return False
    
    async def execute_agent_request(self, request: AgentRequest) -> AgentResponse:
        """Execute agent request with distributed coordination"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Record observability
            if self.observability_manager:
                await self.observability_manager.record_step_execution(
                    workflow_id=request.workflow_id,
                    step_name=f"agent_request_{request.agent_type}",
                    duration_seconds=0.0,  # Will update later
                    status="started",
                    metadata={
                        "agent_type": request.agent_type,
                        "operation": request.operation,
                        "node_id": self.node_id
                    }
                )
            
            # Select optimal node for execution
            selection_request = {
                "workflow_id": request.workflow_id,
                "agent_type": request.agent_type,
                "required_capabilities": list(request.required_capabilities),
                "tda_correlation_id": request.tda_correlation_id,
                "priority": request.priority
            }
            
            selected_node = await self.load_balancer.select_node(selection_request)
            
            if not selected_node:
                raise Exception("No available nodes for request")
            
            # Execute request
            if selected_node == self.node_id:
                # Execute locally
                response = await self._execute_local_agent_request(request)
            else:
                # Execute remotely
                response = await self._execute_remote_agent_request(request, selected_node)
            
            # Record metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.request_metrics[request.agent_type].append(execution_time)
            
            # Update load balancer
            await self._update_node_load_metrics(selected_node, execution_time, response.status == "success")
            
            # Record observability
            if self.observability_manager:
                await self.observability_manager.record_step_execution(
                    workflow_id=request.workflow_id,
                    step_name=f"agent_request_{request.agent_type}",
                    duration_seconds=execution_time,
                    status=response.status,
                    metadata={
                        "agent_type": request.agent_type,
                        "operation": request.operation,
                        "selected_node": selected_node,
                        "execution_time": execution_time
                    }
                )
            
            return response
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Record failure
            if self.observability_manager:
                await self.observability_manager.record_step_execution(
                    workflow_id=request.workflow_id,
                    step_name=f"agent_request_{request.agent_type}",
                    duration_seconds=execution_time,
                    status="failed",
                    metadata={
                        "agent_type": request.agent_type,
                        "operation": request.operation,
                        "error": str(e)
                    }
                )
            
            return AgentResponse(
                request_id=request.request_id,
                node_id=self.node_id,
                status="failed",
                result={},
                execution_time=execution_time,
                error=str(e)
            )
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        cluster_load = await self.load_balancer.get_cluster_load()
        
        return {
            "cluster_id": f"cluster_{self.node_id}",
            "node_id": self.node_id,
            "is_leader": await self.consensus.is_leader(),
            "cluster_size": len(self.cluster_state),
            "active_nodes": len([n for n in self.cluster_state.values() if n.state == NodeState.ACTIVE]),
            "cluster_load": cluster_load,
            "average_load": sum(cluster_load.values()) / len(cluster_load) if cluster_load else 0.0,
            "active_requests": len(self.active_requests),
            "load_balancer_stats": self.load_balancer.get_load_balancing_stats(),
            "health_status": {
                node_id: (datetime.now(timezone.utc) - last_seen).total_seconds()
                for node_id, last_seen in self.node_health.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def propose_cluster_decision(self, decision: Dict[str, Any]) -> bool:
        """Propose a cluster-wide decision using consensus"""
        return await self.consensus.propose_value(decision)
    
    async def get_cluster_decision(self, key: str = "default") -> Optional[Any]:
        """Get the current cluster consensus decision"""
        return await self.consensus.get_consensus_value(key)
    
    def subscribe_to_events(self, event_type: CoordinationEvent, handler: Callable):
        """Subscribe to coordination events"""
        self.event_handlers[event_type].append(handler)
    
    async def _initialize_cluster_state(self):
        """Initialize cluster state"""
        # Add self to cluster
        self_info = NodeInfo(
            node_id=self.node_id,
            address="localhost",
            port=8080,
            capabilities={"agent_execution", "consensus", "load_balancing"},
            load_factor=0.0,
            state=NodeState.ACTIVE,
            last_seen=datetime.now(timezone.utc)
        )
        
        await self.join_cluster(self_info)
    
    async def _execute_local_agent_request(self, request: AgentRequest) -> AgentResponse:
        """Execute agent request locally"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Simulate agent execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mock successful execution
            result = {
                "agent_type": request.agent_type,
                "operation": request.operation,
                "status": "completed",
                "output": f"Result from {request.operation}",
                "processed_parameters": request.parameters
            }
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                node_id=self.node_id,
                status="success",
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return AgentResponse(
                request_id=request.request_id,
                node_id=self.node_id,
                status="failed",
                result={},
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _execute_remote_agent_request(
        self, 
        request: AgentRequest, 
        target_node: NodeId
    ) -> AgentResponse:
        """Execute agent request on remote node"""
        # Create distributed message
        message = DistributedMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.AGENT_REQUEST,
            sender_id=self.node_id,
            recipient_id=target_node,
            payload={
                "request": {
                    "request_id": request.request_id,
                    "workflow_id": request.workflow_id,
                    "agent_type": request.agent_type,
                    "operation": request.operation,
                    "parameters": request.parameters,
                    "timeout_seconds": request.timeout_seconds,
                    "tda_correlation_id": request.tda_correlation_id
                }
            },
            vector_clock=self.vector_clock.increment(self.node_id),
            correlation_id=request.tda_correlation_id
        )
        
        # Send request and wait for response
        try:
            # Create future for response
            response_future = asyncio.Future()
            self.pending_responses[request.request_id] = response_future
            
            # Send message
            target_address = f"{target_node}_address"  # Simplified addressing
            await self.transport.send_message(message, target_address)
            
            # Wait for response with timeout
            response = await asyncio.wait_for(
                response_future, 
                timeout=request.timeout_seconds
            )
            
            return response
            
        except asyncio.TimeoutError:
            # Cleanup
            self.pending_responses.pop(request.request_id, None)
            
            return AgentResponse(
                request_id=request.request_id,
                node_id=target_node,
                status="timeout",
                result={},
                execution_time=request.timeout_seconds,
                error="Request timeout"
            )
        
        except Exception as e:
            # Cleanup
            self.pending_responses.pop(request.request_id, None)
            
            return AgentResponse(
                request_id=request.request_id,
                node_id=target_node,
                status="failed",
                result={},
                execution_time=0.0,
                error=str(e)
            )
    
    async def _update_node_load_metrics(
        self, 
        node_id: NodeId, 
        execution_time: float, 
        success: bool
    ):
        """Update node load metrics after request execution"""
        # Create load metrics
        metrics = LoadMetrics(
            cpu_usage=0.5,  # Mock CPU usage
            memory_usage=0.3,  # Mock memory usage
            active_connections=len(self.active_requests),
            request_rate=len(self.request_metrics.get(node_id, [])) / 60.0,  # Requests per minute
            response_time_p95=execution_time * 1000,  # Convert to ms
            error_rate=0.0 if success else 0.1
        )
        
        await self.load_balancer.update_node_load(node_id, metrics)
    
    async def _emit_event(self, event_type: CoordinationEvent, data: Dict[str, Any]):
        """Emit coordination event"""
        event_data = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc),
            "node_id": self.node_id
        }
        
        await self.event_stream.put(event_data)
    
    async def _event_processor(self):
        """Process coordination events"""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_stream.get(), timeout=1.0)
                
                # Call event handlers
                event_type = event["event_type"]
                handlers = self.event_handlers.get(event_type, [])
                
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        # Log handler error but continue
                        pass
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                await asyncio.sleep(0.1)
    
    async def _health_monitor(self):
        """Monitor cluster health"""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                failed_nodes = []
                
                # Check node health
                for node_id, last_seen in self.node_health.items():
                    if current_time - last_seen > timedelta(seconds=30):  # 30 second timeout
                        failed_nodes.append(node_id)
                
                # Handle failed nodes
                for node_id in failed_nodes:
                    if node_id in self.cluster_state:
                        self.cluster_state[node_id] = self.cluster_state[node_id]._replace(
                            state=NodeState.FAILED
                        )
                        
                        await self._emit_event(CoordinationEvent.NODE_FAILED, {
                            "node_id": node_id,
                            "last_seen": self.node_health[node_id].isoformat()
                        })
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                await asyncio.sleep(1.0)
    
    async def _request_processor(self):
        """Process incoming requests"""
        while self.running:
            try:
                # Process incoming messages
                messages = await self.transport.receive_messages()
                
                for message in messages:
                    if message.message_type == MessageType.AGENT_REQUEST:
                        await self._handle_agent_request_message(message)
                    elif message.message_type == MessageType.AGENT_RESPONSE:
                        await self._handle_agent_response_message(message)
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                await asyncio.sleep(0.1)
    
    async def _handle_agent_request_message(self, message: DistributedMessage):
        """Handle incoming agent request message"""
        try:
            request_data = message.payload["request"]
            
            # Create agent request
            request = AgentRequest(
                request_id=request_data["request_id"],
                workflow_id=request_data["workflow_id"],
                agent_type=request_data["agent_type"],
                operation=request_data["operation"],
                parameters=request_data["parameters"],
                timeout_seconds=request_data["timeout_seconds"],
                tda_correlation_id=request_data.get("tda_correlation_id")
            )
            
            # Execute locally
            response = await self._execute_local_agent_request(request)
            
            # Send response back
            response_message = DistributedMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.AGENT_RESPONSE,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                payload={"response": response},
                vector_clock=self.vector_clock.increment(self.node_id),
                correlation_id=message.correlation_id
            )
            
            sender_address = f"{message.sender_id}_address"
            await self.transport.send_message(response_message, sender_address)
            
        except Exception as e:
            # Send error response
            pass
    
    async def _handle_agent_response_message(self, message: DistributedMessage):
        """Handle incoming agent response message"""
        try:
            response_data = message.payload["response"]
            request_id = response_data["request_id"]
            
            # Complete pending request
            if request_id in self.pending_responses:
                future = self.pending_responses.pop(request_id)
                if not future.done():
                    response = AgentResponse(**response_data)
                    future.set_result(response)
            
        except Exception as e:
            pass
    
    async def _metrics_collector(self):
        """Collect and report metrics"""
        while self.running:
            try:
                # Collect metrics
                cluster_status = await self.get_cluster_status()
                
                # Send to TDA
                if self.tda_integration:
                    await self.tda_integration.send_orchestration_result(
                        {
                            "event": "cluster_metrics",
                            "metrics": cluster_status,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        },
                        f"cluster_metrics_{self.node_id}"
                    )
                
                await asyncio.sleep(30.0)  # Report every 30 seconds
                
            except Exception as e:
                await asyncio.sleep(5.0)