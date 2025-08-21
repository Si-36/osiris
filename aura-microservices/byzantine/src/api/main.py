"""
Byzantine Consensus Service API
Fault-tolerant multi-agent coordination and decision making
Based on AURA Intelligence research and 2025 best practices
"""

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time
import json
import uuid
from typing import Optional, List, Dict, Any, Set
import structlog
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from ..models.consensus.byzantine_2025 import (
    ByzantineConsensus, ConsensusConfig, ConsensusProposal, Vote,
    ConsensusPhase, NodeState, MultiAgentByzantineCoordinator
)
from ..services.coordination_service import CoordinationService
from ..services.reputation_service import ReputationService
from ..services.network_manager import NetworkManager
from ..schemas.requests import (
    ProposeRequest, JoinClusterRequest, VoteRequest,
    ConsensusQueryRequest, ReputationUpdateRequest
)
from ..schemas.responses import (
    ProposeResponse, ConsensusStatusResponse, NodeStatusResponse,
    ClusterStatusResponse, ConsensusHistoryResponse
)
from ..middleware.observability import ObservabilityMiddleware
from ..middleware.security import SecurityMiddleware
from ..middleware.circuit_breaker import CircuitBreakerMiddleware
from ..utils.observability import setup_telemetry

# Initialize structured logging
logger = structlog.get_logger()

# Setup telemetry
setup_telemetry()

# WebSocket connections for real-time consensus
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.node_connections: Dict[str, str] = {}  # node_id -> connection_id
        
    async def connect(self, websocket: WebSocket, node_id: str):
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        self.node_connections[node_id] = connection_id
        
    def disconnect(self, node_id: str):
        if node_id in self.node_connections:
            connection_id = self.node_connections[node_id]
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            del self.node_connections[node_id]
            
    async def broadcast_vote(self, vote: Dict[str, Any]):
        """Broadcast vote to all connected nodes"""
        for connection in self.active_connections.values():
            try:
                await connection.send_json({
                    "type": "vote",
                    "data": vote
                })
            except:
                pass
                
    async def send_to_node(self, node_id: str, message: Dict[str, Any]):
        """Send message to specific node"""
        if node_id in self.node_connections:
            connection_id = self.node_connections[node_id]
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_json(message)
                except:
                    pass


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    Handles initialization and cleanup of consensus system
    """
    logger.info("Starting Byzantine Consensus Service")
    
    # Initialize configuration
    num_nodes = 7  # Default cluster size (tolerates 2 Byzantine)
    byzantine_threshold = (num_nodes - 1) // 3
    
    # Initialize primary consensus node
    config = ConsensusConfig(
        node_id="primary",
        total_nodes=num_nodes,
        byzantine_threshold=byzantine_threshold,
        weighted_voting=True,
        edge_optimized=False,
        phase_timeout_ms=5000,
        view_change_timeout_ms=10000
    )
    
    app.state.consensus = ByzantineConsensus(config)
    app.state.coordinator = MultiAgentByzantineCoordinator(num_nodes, byzantine_count=0)
    
    # Initialize services
    app.state.coordination_service = CoordinationService()
    app.state.reputation_service = ReputationService()
    app.state.network_manager = NetworkManager()
    
    # Active proposals tracking
    app.state.active_proposals = {}
    app.state.pending_votes = asyncio.Queue()
    
    # Start background tasks
    app.state.vote_processor = asyncio.create_task(
        process_votes_loop(app.state.consensus, app.state.pending_votes)
    )
    
    logger.info("Byzantine Consensus Service ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Byzantine Consensus Service")
    app.state.vote_processor.cancel()
    await app.state.consensus.shutdown()
    
    # Save final consensus state
    final_status = app.state.consensus.get_node_status()
    logger.info("Final consensus state", **final_status)


async def process_votes_loop(consensus: ByzantineConsensus, vote_queue: asyncio.Queue):
    """Background task to process incoming votes"""
    while True:
        try:
            vote_data = await vote_queue.get()
            vote = Vote(**vote_data['vote'])
            proposal = None
            
            if 'proposal' in vote_data:
                proposal = ConsensusProposal(**vote_data['proposal'])
                
            await consensus.receive_vote(vote, proposal)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error processing vote", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="AURA Byzantine Consensus Service",
    description="Fault-tolerant multi-agent coordination with Byzantine consensus",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CircuitBreakerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Consensus-View", "X-Node-State"]
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AURA Byzantine Consensus Service",
        "version": "2.0.0",
        "features": [
            "Byzantine fault tolerance (3f+1)",
            "HotStuff-inspired 3-phase protocol",
            "Weighted voting with reputation",
            "Real-time leader election",
            "Multi-agent coordination",
            "Edge-optimized consensus",
            "Cryptographic signatures",
            "View changes and recovery"
        ],
        "status": "operational"
    }


@app.get("/api/v1/health", response_model=ConsensusStatusResponse, tags=["health"])
async def health_check(request: Request):
    """
    Comprehensive health check with consensus statistics
    """
    consensus = request.app.state.consensus
    status = consensus.get_node_status()
    
    # Check health criteria
    is_healthy = (
        len(consensus.byzantine_nodes) <= consensus.config.byzantine_threshold and
        consensus.state != NodeState.BYZANTINE
    )
    
    return ConsensusStatusResponse(
        status="healthy" if is_healthy else "degraded",
        node_id=status['node_id'],
        current_view=status['view'],
        current_phase=status['phase'],
        is_leader=status['is_leader'],
        leader_node=status['current_leader'],
        byzantine_nodes=status['byzantine_nodes'],
        total_nodes=consensus.config.total_nodes,
        byzantine_threshold=consensus.config.byzantine_threshold,
        total_decisions=status['total_decisions'],
        consensus_history_size=status['consensus_history']
    )


@app.post("/api/v1/propose", response_model=ProposeResponse, tags=["consensus"])
async def propose_value(
    request: ProposeRequest,
    background_tasks: BackgroundTasks,
    consensus: ByzantineConsensus = Depends(lambda r: r.app.state.consensus)
):
    """
    Propose a value for Byzantine consensus
    
    Features:
    - Automatic leader forwarding
    - 3-phase commit protocol
    - Byzantine fault detection
    - Weighted voting
    """
    try:
        start_time = time.time()
        
        # Validate proposal
        if request.require_unanimous and consensus.config.total_nodes > 10:
            raise HTTPException(
                status_code=400,
                detail="Unanimous consensus not recommended for >10 nodes"
            )
        
        # Create proposal
        proposal_id = await consensus.propose(
            value=request.value,
            metadata={
                "proposer": request.proposer_id or consensus.node_id,
                "category": request.category,
                "priority": request.priority,
                "require_unanimous": request.require_unanimous,
                "timestamp": time.time()
            }
        )
        
        # Track active proposal
        app.state.active_proposals[proposal_id] = {
            "value": request.value,
            "start_time": start_time,
            "status": "pending"
        }
        
        # Background task to monitor consensus
        background_tasks.add_task(
            monitor_proposal,
            proposal_id,
            consensus
        )
        
        return ProposeResponse(
            proposal_id=proposal_id,
            current_view=consensus.current_view,
            leader_node=consensus.leader or consensus.get_leader(consensus.current_view),
            estimated_time_ms=consensus.config.phase_timeout_ms * 3,  # 3 phases
            quorum_required=consensus.config.quorum_size
        )
        
    except Exception as e:
        logger.error("Proposal failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/vote", tags=["consensus"])
async def submit_vote(
    request: VoteRequest,
    pending_votes: asyncio.Queue = Depends(lambda r: r.app.state.pending_votes)
):
    """
    Submit a vote for a consensus proposal
    
    Used by other nodes in the cluster to vote
    """
    try:
        # Queue vote for processing
        await pending_votes.put({
            "vote": request.dict(),
            "timestamp": time.time()
        })
        
        return {
            "status": "queued",
            "voter": request.voter,
            "proposal_id": request.proposal_id
        }
        
    except Exception as e:
        logger.error("Vote submission failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/consensus/{proposal_id}", tags=["consensus"])
async def get_consensus_status(
    proposal_id: str,
    consensus: ByzantineConsensus = Depends(lambda r: r.app.state.consensus)
):
    """Get status of a specific consensus proposal"""
    
    # Check if decided
    if proposal_id in consensus.decided_values:
        # Find result in history
        result = next(
            (r for r in consensus.consensus_history if r.proposal_id == proposal_id),
            None
        )
        
        if result:
            return {
                "proposal_id": proposal_id,
                "status": "decided",
                "decided_value": result.decided_value,
                "duration_ms": result.duration_ms,
                "view": result.view,
                "votes_received": len(result.votes),
                "byzantine_detected": list(result.byzantine_nodes)
            }
    
    # Check if active
    if proposal_id in consensus.proposals:
        proposal = consensus.proposals[proposal_id]
        votes = consensus.votes.get(proposal_id, {})
        
        current_votes = {
            phase.value: len(phase_votes)
            for phase, phase_votes in votes.items()
        }
        
        return {
            "proposal_id": proposal_id,
            "status": "active",
            "current_phase": consensus.phase.value,
            "votes_by_phase": current_votes,
            "view": proposal.view,
            "proposer": proposal.proposer
        }
    
    raise HTTPException(status_code=404, detail="Proposal not found")


@app.post("/api/v1/cluster/join", tags=["cluster"])
async def join_cluster(
    request: JoinClusterRequest,
    coordinator: MultiAgentByzantineCoordinator = Depends(lambda r: r.app.state.coordinator)
):
    """
    Join the Byzantine consensus cluster
    
    For dynamic cluster membership
    """
    try:
        # In production, this would handle node authentication and setup
        # For now, return cluster information
        
        return {
            "status": "joined",
            "node_id": request.node_id,
            "cluster_size": coordinator.num_agents,
            "byzantine_threshold": (coordinator.num_agents - 1) // 3,
            "current_nodes": list(coordinator.nodes.keys()),
            "connection_params": {
                "websocket_url": f"ws://localhost:8002/ws/{request.node_id}",
                "heartbeat_interval_ms": 5000
            }
        }
        
    except Exception as e:
        logger.error("Cluster join failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cluster/status", response_model=ClusterStatusResponse, tags=["cluster"])
async def get_cluster_status(
    coordinator: MultiAgentByzantineCoordinator = Depends(lambda r: r.app.state.coordinator)
):
    """Get overall cluster status"""
    
    node_statuses = []
    for node_id, node in coordinator.nodes.items():
        status = node.get_node_status()
        node_statuses.append({
            "node_id": node_id,
            "state": status['state'],
            "is_leader": status['is_leader'],
            "reputation_score": status['reputation_score'],
            "total_decisions": status['total_decisions']
        })
    
    # Calculate cluster health
    byzantine_count = sum(1 for n in coordinator.nodes.values() if n.state == NodeState.BYZANTINE)
    healthy_count = len(coordinator.nodes) - byzantine_count
    
    return ClusterStatusResponse(
        total_nodes=coordinator.num_agents,
        healthy_nodes=healthy_count,
        byzantine_nodes=byzantine_count,
        byzantine_threshold=(coordinator.num_agents - 1) // 3,
        can_tolerate_more_failures=byzantine_count < (coordinator.num_agents - 1) // 3,
        node_statuses=node_statuses,
        consensus_possible=healthy_count >= (2 * ((coordinator.num_agents - 1) // 3) + 1)
    )


@app.get("/api/v1/history", response_model=ConsensusHistoryResponse, tags=["history"])
async def get_consensus_history(
    limit: int = 10,
    consensus: ByzantineConsensus = Depends(lambda r: r.app.state.consensus)
):
    """Get recent consensus history"""
    
    history = consensus.consensus_history[-limit:]
    
    history_items = []
    for result in history:
        history_items.append({
            "proposal_id": result.proposal_id,
            "decided_value": result.decided_value,
            "view": result.view,
            "duration_ms": result.duration_ms,
            "vote_count": len(result.votes),
            "byzantine_detected": len(result.byzantine_nodes),
            "timestamp": getattr(result, 'timestamp', 0)
        })
    
    return ConsensusHistoryResponse(
        total_decisions=len(consensus.consensus_history),
        recent_decisions=history_items,
        success_rate=1.0 - (len([r for r in history if r.byzantine_nodes]) / max(len(history), 1))
    )


@app.post("/api/v1/reputation/update", tags=["reputation"])
async def update_reputation(
    request: ReputationUpdateRequest,
    consensus: ByzantineConsensus = Depends(lambda r: r.app.state.consensus)
):
    """Update node reputation based on behavior"""
    
    try:
        # Update reputation
        consensus.reputation.update_reputation(
            request.node_id,
            request.behavior_score
        )
        
        # Check if node should be marked Byzantine
        if consensus.reputation.detect_byzantine(request.node_id):
            consensus._mark_byzantine(request.node_id)
            
        return {
            "node_id": request.node_id,
            "new_reputation": consensus.reputation.get_weight(request.node_id),
            "is_byzantine": request.node_id in consensus.byzantine_nodes
        }
        
    except Exception as e:
        logger.error("Reputation update failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{node_id}")
async def websocket_endpoint(websocket: WebSocket, node_id: str):
    """
    WebSocket endpoint for real-time consensus participation
    
    Enables:
    - Real-time vote broadcasting
    - Leader election notifications
    - View change updates
    """
    await manager.connect(websocket, node_id)
    
    try:
        while True:
            # Receive messages from node
            data = await websocket.receive_json()
            
            if data["type"] == "vote":
                # Queue vote for processing
                await app.state.pending_votes.put({
                    "vote": data["vote"],
                    "proposal": data.get("proposal"),
                    "from_node": node_id
                })
                
            elif data["type"] == "heartbeat":
                # Update node liveness
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "timestamp": time.time()
                })
                
            elif data["type"] == "view_change":
                # Handle view change request
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(node_id)
        logger.info("Node disconnected", node_id=node_id)


@app.get("/api/v1/demo/multi-agent", tags=["demo"])
async def demo_multi_agent_consensus(
    num_agents: int = 7,
    byzantine_count: int = 2,
    coordinator: MultiAgentByzantineCoordinator = Depends(lambda r: r.app.state.coordinator)
):
    """
    Demo: Multi-agent Byzantine consensus
    
    Shows fault tolerance with Byzantine nodes
    """
    try:
        # Create test scenario
        test_value = {
            "decision": "update_strategy",
            "parameters": {
                "algorithm": "advanced_planning",
                "risk_tolerance": 0.3
            },
            "timestamp": time.time()
        }
        
        # Run consensus
        results = await coordinator.propose_value(test_value)
        
        # Analyze results
        decided_values = [r.decided_value for r in results.values()]
        consensus_reached = all(v == decided_values[0] for v in decided_values) if decided_values else False
        
        return {
            "scenario": {
                "num_agents": num_agents,
                "byzantine_count": byzantine_count,
                "can_tolerate": (num_agents - 1) // 3
            },
            "consensus_reached": consensus_reached,
            "results_by_node": {
                node_id: {
                    "decided": result.decided_value,
                    "duration_ms": result.duration_ms,
                    "byzantine_detected": len(result.byzantine_nodes)
                }
                for node_id, result in results.items()
            },
            "analysis": {
                "agreement": consensus_reached,
                "total_decisions": len(results),
                "avg_latency_ms": sum(r.duration_ms for r in results.values()) / len(results) if results else 0
            }
        }
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def monitor_proposal(proposal_id: str, consensus: ByzantineConsensus):
    """Monitor proposal progress"""
    timeout = consensus.config.phase_timeout_ms * 4 / 1000  # Total timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if proposal_id in consensus.decided_values:
            app.state.active_proposals[proposal_id]["status"] = "decided"
            break
            
        await asyncio.sleep(0.5)
    else:
        app.state.active_proposals[proposal_id]["status"] = "timeout"


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with correlation ID"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        path=request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": correlation_id,
            "message": str(exc) if request.app.debug else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )