"""
Request schemas for Byzantine Consensus API
Pydantic V2 models with comprehensive validation
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class ConsensusPriority(str, Enum):
    """Priority levels for consensus proposals"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ConsensusCategory(str, Enum):
    """Categories of consensus decisions"""
    MODEL_UPDATE = "model_update"
    STRATEGY_CHANGE = "strategy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    SAFETY_CRITICAL = "safety_critical"
    CONFIGURATION = "configuration"
    GENERAL = "general"


class ProposeRequest(BaseModel):
    """Request to propose a value for consensus"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "value": {"action": "update_model", "parameters": {"lr": 0.01}},
            "category": "model_update",
            "priority": "high",
            "proposer_id": "agent_1",
            "require_unanimous": False,
            "timeout_ms": 5000
        }
    })
    
    value: Any = Field(
        ...,
        description="The value/decision to achieve consensus on"
    )
    category: ConsensusCategory = Field(
        default=ConsensusCategory.GENERAL,
        description="Category of the consensus decision"
    )
    priority: ConsensusPriority = Field(
        default=ConsensusPriority.NORMAL,
        description="Priority of the proposal"
    )
    proposer_id: Optional[str] = Field(
        default=None,
        description="ID of the proposing agent"
    )
    require_unanimous: bool = Field(
        default=False,
        description="Whether unanimous agreement is required"
    )
    timeout_ms: Optional[int] = Field(
        default=None,
        ge=1000,
        le=60000,
        description="Custom timeout for this proposal"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the proposal"
    )


class VoteRequest(BaseModel):
    """Request to submit a vote"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "voter": "agent_2",
            "proposal_id": "agent_1:0:1234567890",
            "vote_type": "commit",
            "phase": "commit",
            "view": 0,
            "signature": "base64_signature"
        }
    })
    
    voter: str = Field(..., description="ID of the voting node")
    proposal_id: str = Field(..., description="ID of the proposal being voted on")
    vote_type: str = Field(..., description="Type of vote")
    phase: str = Field(..., description="Current consensus phase")
    view: int = Field(..., ge=0, description="Current view number")
    signature: Optional[str] = Field(default=None, description="Digital signature of vote")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Vote weight based on reputation")


class JoinClusterRequest(BaseModel):
    """Request to join Byzantine consensus cluster"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "node_id": "agent_7",
            "public_key": "base64_public_key",
            "capabilities": ["edge_deployment", "gpu_acceleration"],
            "initial_reputation": 1.0
        }
    })
    
    node_id: str = Field(
        ...,
        description="Unique identifier for the node",
        pattern="^[a-zA-Z0-9_-]+$"
    )
    public_key: Optional[str] = Field(
        default=None,
        description="Public key for cryptographic operations"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="Node capabilities"
    )
    initial_reputation: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Initial reputation score"
    )


class ConsensusQueryRequest(BaseModel):
    """Request to query consensus status"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "proposal_ids": ["agent_1:0:123", "agent_2:1:456"],
            "include_votes": True,
            "only_decided": False
        }
    })
    
    proposal_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific proposal IDs to query"
    )
    include_votes: bool = Field(
        default=False,
        description="Include detailed vote information"
    )
    only_decided: bool = Field(
        default=False,
        description="Only return decided proposals"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results"
    )


class ReputationUpdateRequest(BaseModel):
    """Request to update node reputation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "node_id": "agent_3",
            "behavior_score": 0.95,
            "reason": "consistent_voting"
        }
    })
    
    node_id: str = Field(..., description="Node to update reputation for")
    behavior_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Behavior score (1.0 = perfect, 0.0 = Byzantine)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for reputation update"
    )


class ViewChangeRequest(BaseModel):
    """Request to initiate view change"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "requesting_node": "agent_2",
            "current_view": 5,
            "reason": "leader_timeout"
        }
    })
    
    requesting_node: str = Field(..., description="Node requesting view change")
    current_view: int = Field(..., ge=0, description="Current view number")
    proposed_view: Optional[int] = Field(default=None, description="Proposed new view")
    reason: str = Field(..., description="Reason for view change")


class MultiAgentTestRequest(BaseModel):
    """Request for multi-agent consensus testing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "num_agents": 7,
            "byzantine_agents": 2,
            "test_values": [
                {"decision": "strategy_a"},
                {"decision": "strategy_b"}
            ],
            "iterations": 5
        }
    })
    
    num_agents: int = Field(
        default=7,
        ge=4,
        le=100,
        description="Total number of agents"
    )
    byzantine_agents: int = Field(
        default=0,
        ge=0,
        description="Number of Byzantine agents"
    )
    test_values: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Values to test consensus on"
    )
    iterations: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of consensus rounds"
    )
    
    @field_validator('byzantine_agents')
    @classmethod
    def validate_byzantine_count(cls, v, info):
        """Ensure Byzantine count doesn't exceed threshold"""
        if 'num_agents' in info.data:
            max_byzantine = (info.data['num_agents'] - 1) // 3
            if v > max_byzantine:
                raise ValueError(f"Byzantine count {v} exceeds maximum {max_byzantine} for {info.data['num_agents']} agents")
        return v