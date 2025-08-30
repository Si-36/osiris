"""
Core types and data structures for consensus protocols - 2025 Implementation

Clean implementation without circular dependencies.
Uses latest consensus algorithm types and patterns.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid


class DecisionType(str, Enum):
    """Types of decisions requiring consensus"""
    OPERATIONAL = "operational"    # Fast, low-stakes (e.g., task assignment)
    TACTICAL = "tactical"          # Medium speed (e.g., resource allocation)
    STRATEGIC = "strategic"        # Slow, high-stakes (e.g., model updates)
    EMERGENCY = "emergency"        # Fast, critical (e.g., safety shutdown)
    COGNITIVE = "cognitive"        # AI consciousness decisions


class VoteType(str, Enum):
    """Types of votes in consensus"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"          # Delegate to another node


class RaftState(str, Enum):
    """Raft consensus node states"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    LEARNER = "learner"            # Non-voting observer


class BFTPhase(str, Enum):
    """Byzantine Fault Tolerant consensus phases"""
    NEW_VIEW = "new_view"
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDE = "decide"
    VIEW_CHANGE = "view_change"    # View change protocol


class ConsensusState(str, Enum):
    """Overall consensus state"""
    IDLE = "idle"
    PROPOSING = "proposing"
    VOTING = "voting"
    DECIDING = "deciding"
    COMMITTED = "committed"
    ABORTED = "aborted"


# Simple AgentState replacement to avoid circular dependency
class AgentState(str, Enum):
    """Agent state for consensus participation"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ConsensusRequest:
    """Request for consensus decision"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType = DecisionType.OPERATIONAL
    proposer_id: str = ""
    proposal: Dict[str, Any] = field(default_factory=dict)
    
    # Timing constraints
    timeout: Optional[timedelta] = None
    deadline: Optional[datetime] = None
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Active inference components
    expected_utility: float = 0.0
    uncertainty: float = 1.0
    
    # Priority and requirements
    priority: int = 0
    required_confidence: float = 0.5
    minimum_votes: Optional[int] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Vote:
    """Individual vote in consensus"""
    vote_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    voter_id: str = ""
    request_id: str = ""
    vote_type: VoteType = VoteType.ABSTAIN
    
    # Vote reasoning (for explainability)
    reasoning: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Cryptographic signature
    signature: str = ""
    
    # Delegation (if vote_type is DELEGATE)
    delegate_to: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Result of consensus decision"""
    request_id: str = ""
    decision: VoteType = VoteType.ABSTAIN
    
    # Voting details
    total_votes: int = 0
    approve_votes: int = 0
    reject_votes: int = 0
    abstain_votes: int = 0
    
    # Decision quality metrics
    confidence: float = 0.0
    consensus_strength: float = 0.0  # How strong the consensus is
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    duration: Optional[timedelta] = None
    
    # Detailed votes (for transparency)
    votes: List[Vote] = field(default_factory=list)
    
    # Byzantine fault detection
    byzantine_nodes: Set[str] = field(default_factory=set)
    
    # Explanation
    explanation: Optional["DecisionExplanation"] = None
    
    def __post_init__(self):
        """Calculate derived fields"""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
        
        # Calculate consensus strength
        if self.total_votes > 0:
            majority = max(self.approve_votes, self.reject_votes)
            self.consensus_strength = majority / self.total_votes


@dataclass
class ConsensusProof:
    """Cryptographic proof of consensus"""
    consensus_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    
    # Proof components
    merkle_root: str = ""                    # Merkle root of votes
    threshold_signature: str = ""            # Aggregated signature
    
    # Consensus details
    decision: VoteType = VoteType.ABSTAIN
    vote_count: int = 0
    threshold: int = 0
    
    # Participating nodes
    participants: List[str] = field(default_factory=list)
    
    # Timestamp and block height (if applicable)
    timestamp: datetime = field(default_factory=datetime.now)
    block_height: Optional[int] = None
    
    # Validity
    is_valid: bool = False
    
    def verify(self, public_keys: Dict[str, str]) -> bool:
        """Verify the consensus proof"""
        # Simplified verification
        # In production, implement proper crypto verification
        return self.vote_count >= self.threshold


@dataclass
class DecisionExplanation:
    """Explainable AI for consensus decisions"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Causal reasoning
    causal_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Decision path
    decision_tree: Dict[str, Any] = field(default_factory=dict)
    
    # Key influences
    primary_factors: List[str] = field(default_factory=list)
    
    # Counterfactual analysis
    alternative_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Confidence breakdown
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    
    # Natural language explanation
    summary: str = ""
    detailed_reasoning: List[str] = field(default_factory=list)
    
    # Active inference metrics
    free_energy: float = 0.0
    expected_free_energy: float = 0.0
    information_gain: float = 0.0


@dataclass
class LogEntry:
    """Raft log entry"""
    index: int
    term: int
    command: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    proposer_id: Optional[str] = None
    consensus_proof: Optional[ConsensusProof] = None


@dataclass
class RaftMessage:
    """Message in Raft protocol"""
    message_type: str  # AppendEntries, RequestVote, etc.
    term: int
    leader_id: Optional[str] = None
    
    # For AppendEntries
    prev_log_index: Optional[int] = None
    prev_log_term: Optional[int] = None
    entries: List[LogEntry] = field(default_factory=list)
    leader_commit: Optional[int] = None
    
    # For RequestVote
    candidate_id: Optional[str] = None
    last_log_index: Optional[int] = None
    last_log_term: Optional[int] = None
    
    # Response fields
    success: Optional[bool] = None
    match_index: Optional[int] = None


@dataclass
class ConsensusConfig:
    """Configuration for consensus protocols"""
    # Protocol selection
    protocol: str = "raft"  # raft, byzantine, pbft, hotstuff
    
    # Node configuration
    node_id: str = ""
    peers: List[str] = field(default_factory=list)
    
    # Timing parameters
    election_timeout_ms: int = 150
    heartbeat_interval_ms: int = 50
    consensus_timeout_ms: int = 5000
    
    # Byzantine fault tolerance
    fault_tolerance: int = 0  # Number of faulty nodes to tolerate
    
    # Active inference
    enable_active_inference: bool = True
    confidence_threshold: float = 0.7
    
    # Storage
    persistent_state_path: Optional[str] = None
    
    # Network
    bind_address: str = "0.0.0.0"
    port: int = 8080
    
    # Security
    enable_tls: bool = True
    cert_path: Optional[str] = None
    key_path: Optional[str] = None


@dataclass
class ConsensusMetrics:
    """Metrics for consensus performance"""
    # Performance metrics
    decisions_total: int = 0
    decisions_approved: int = 0
    decisions_rejected: int = 0
    
    # Timing metrics
    average_decision_time_ms: float = 0.0
    p95_decision_time_ms: float = 0.0
    p99_decision_time_ms: float = 0.0
    
    # Node health
    active_nodes: int = 0
    failed_nodes: int = 0
    byzantine_nodes_detected: int = 0
    
    # Network metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Consensus quality
    average_confidence: float = 0.0
    average_consensus_strength: float = 0.0
    
    # Active inference metrics
    average_free_energy: float = 0.0
    information_gain_total: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)