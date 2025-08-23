"""
Response schemas for Byzantine Consensus API
Comprehensive responses with consensus metrics
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime


class ProposeResponse(BaseModel):
    """Response from consensus proposal"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "proposal_id": "agent_1:0:1234567890",
            "current_view": 0,
            "leader_node": "agent_0",
            "estimated_time_ms": 15000,
            "quorum_required": 5
        }
    })
    
    proposal_id: str = Field(..., description="Unique proposal identifier")
    current_view: int = Field(..., description="Current consensus view")
    leader_node: str = Field(..., description="Current leader node")
    estimated_time_ms: int = Field(..., description="Estimated time to consensus")
    quorum_required: int = Field(..., description="Number of votes needed for quorum")


class ConsensusStatusResponse(BaseModel):
    """Byzantine consensus service status"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "node_id": "primary",
            "current_view": 5,
            "current_phase": "commit",
            "is_leader": False,
            "leader_node": "agent_0",
            "byzantine_nodes": ["agent_3"],
            "total_nodes": 7,
            "byzantine_threshold": 2
        }
    })
    
    status: str = Field(..., description="Service health status")
    node_id: str = Field(..., description="This node's ID")
    current_view: int = Field(..., description="Current consensus view")
    current_phase: str = Field(..., description="Current consensus phase")
    is_leader: bool = Field(..., description="Whether this node is leader")
    leader_node: Optional[str] = Field(..., description="Current leader")
    byzantine_nodes: List[str] = Field(..., description="Detected Byzantine nodes")
    total_nodes: int = Field(..., description="Total nodes in cluster")
    byzantine_threshold: int = Field(..., description="Maximum Byzantine nodes tolerated")
    total_decisions: int = Field(..., description="Total consensus decisions made")
    consensus_history_size: int = Field(..., description="Number of decisions in history")


class NodeStatusResponse(BaseModel):
    """Individual node status"""
    node_id: str = Field(..., description="Node identifier")
    state: str = Field(..., description="Node state (follower/leader/byzantine)")
    is_leader: bool = Field(..., description="Whether node is current leader")
    reputation_score: float = Field(..., description="Node reputation (0-1)")
    total_decisions: int = Field(..., description="Decisions participated in")
    last_active: Optional[float] = Field(default=None, description="Last activity timestamp")


class ClusterStatusResponse(BaseModel):
    """Overall cluster status"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_nodes": 7,
            "healthy_nodes": 5,
            "byzantine_nodes": 2,
            "byzantine_threshold": 2,
            "can_tolerate_more_failures": False,
            "consensus_possible": True
        }
    })
    
    total_nodes: int = Field(..., description="Total nodes in cluster")
    healthy_nodes: int = Field(..., description="Number of healthy nodes")
    byzantine_nodes: int = Field(..., description="Number of Byzantine nodes")
    byzantine_threshold: int = Field(..., description="Maximum failures tolerated")
    can_tolerate_more_failures: bool = Field(..., description="Can tolerate more Byzantine nodes")
    consensus_possible: bool = Field(..., description="Whether consensus is still possible")
    node_statuses: List[Dict[str, Any]] = Field(..., description="Individual node statuses")
    current_leader: Optional[str] = Field(default=None, description="Current cluster leader")
    view_changes_count: int = Field(default=0, description="Total view changes")


class ConsensusDecision(BaseModel):
    """A consensus decision"""
    proposal_id: str = Field(..., description="Proposal identifier")
    decided_value: Any = Field(..., description="The decided value")
    view: int = Field(..., description="View when decided")
    duration_ms: float = Field(..., description="Time to reach consensus")
    vote_count: int = Field(..., description="Number of votes received")
    byzantine_detected: int = Field(..., description="Byzantine nodes detected")
    timestamp: float = Field(..., description="Decision timestamp")


class ConsensusHistoryResponse(BaseModel):
    """Consensus decision history"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_decisions": 42,
            "recent_decisions": [
                {
                    "proposal_id": "agent_1:5:123",
                    "decided_value": {"action": "update"},
                    "duration_ms": 1234.5,
                    "vote_count": 5
                }
            ],
            "success_rate": 0.95
        }
    })
    
    total_decisions: int = Field(..., description="Total decisions made")
    recent_decisions: List[Dict[str, Any]] = Field(..., description="Recent consensus decisions")
    success_rate: float = Field(..., description="Success rate (non-Byzantine)")
    avg_duration_ms: Optional[float] = Field(default=None, description="Average consensus duration")
    decisions_per_view: Optional[Dict[str, int]] = Field(default=None, description="Decisions per view")


class VoteInfo(BaseModel):
    """Information about a vote"""
    voter: str = Field(..., description="Voting node ID")
    phase: str = Field(..., description="Vote phase")
    timestamp: float = Field(..., description="Vote timestamp")
    weight: float = Field(..., description="Vote weight")
    signature_valid: bool = Field(..., description="Whether signature is valid")


class ProposalStatusResponse(BaseModel):
    """Status of a specific proposal"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "proposal_id": "agent_1:0:123",
            "status": "active",
            "current_phase": "pre_commit",
            "votes_by_phase": {"prepare": 5, "pre_commit": 3, "commit": 0},
            "proposer": "agent_1",
            "view": 0
        }
    })
    
    proposal_id: str = Field(..., description="Proposal identifier")
    status: str = Field(..., description="Status (active/decided/timeout)")
    decided_value: Optional[Any] = Field(default=None, description="Decided value if complete")
    current_phase: Optional[str] = Field(default=None, description="Current phase if active")
    votes_by_phase: Optional[Dict[str, int]] = Field(default=None, description="Vote counts by phase")
    proposer: str = Field(..., description="Proposing node")
    view: int = Field(..., description="Consensus view")
    duration_ms: Optional[float] = Field(default=None, description="Duration if complete")
    byzantine_detected: Optional[List[str]] = Field(default=None, description="Byzantine nodes detected")


class ReputationResponse(BaseModel):
    """Node reputation information"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "node_id": "agent_3",
            "reputation_score": 0.85,
            "voting_weight": 0.85,
            "behavior_history": [0.9, 0.8, 0.85, 0.85],
            "is_byzantine": False
        }
    })
    
    node_id: str = Field(..., description="Node identifier")
    reputation_score: float = Field(..., description="Current reputation (0-1)")
    voting_weight: float = Field(..., description="Voting weight based on reputation")
    behavior_history: List[float] = Field(..., description="Recent behavior scores")
    is_byzantine: bool = Field(..., description="Whether marked as Byzantine")
    last_updated: float = Field(..., description="Last reputation update timestamp")


class MultiAgentTestResponse(BaseModel):
    """Multi-agent consensus test results"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "test_id": "test_123",
            "configuration": {
                "num_agents": 7,
                "byzantine_count": 2,
                "iterations": 5
            },
            "results": {
                "consensus_achieved": 4,
                "failed": 1,
                "avg_duration_ms": 1500.5
            }
        }
    })
    
    test_id: str = Field(..., description="Test identifier")
    configuration: Dict[str, Any] = Field(..., description="Test configuration")
    results: Dict[str, Any] = Field(..., description="Test results")
    consensus_achieved: int = Field(..., description="Successful consensus rounds")
    failed_rounds: int = Field(..., description="Failed consensus rounds")
    avg_duration_ms: float = Field(..., description="Average consensus duration")
    byzantine_behavior_detected: bool = Field(..., description="Whether Byzantine behavior was detected")
    details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed results per round")