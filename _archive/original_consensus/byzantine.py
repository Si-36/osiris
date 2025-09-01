"""
Byzantine Fault Tolerant Consensus - 2025 Implementation

Based on:
- HotStuff consensus protocol (latest BFT)
- Active Inference for decision confidence
- Explainable AI with causal reasoning
- Multi-agent consensus mechanisms

Key innovations:
- 3-phase HotStuff protocol (prepare, pre-commit, commit)
- Active inference for confidence estimation
- Cryptographic proofs with threshold signatures
- Byzantine node detection and isolation
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from collections import defaultdict, deque
import json
import uuid

logger = structlog.get_logger(__name__)


class BFTPhase(Enum):
    """HotStuff consensus phases"""
    NEW_VIEW = "new_view"
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDE = "decide"


class VoteType(Enum):
    """Vote types in consensus"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class BFTMessage:
    """Byzantine fault tolerant message"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: BFTPhase = BFTPhase.NEW_VIEW
    view: int = 0
    sequence: int = 0
    proposal: Dict[str, Any] = field(default_factory=dict)
    proposer_id: str = ""
    message_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate message hash if not provided"""
        if not self.message_hash:
            content = f"{self.phase.value}:{self.view}:{self.sequence}:{json.dumps(self.proposal, sort_keys=True)}"
            self.message_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class BFTVote:
    """Vote in BFT consensus"""
    vote_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    voter_id: str = ""
    message_hash: str = ""
    phase: BFTPhase = BFTPhase.PREPARE
    view: int = 0
    sequence: int = 0
    vote_type: VoteType = VoteType.APPROVE
    signature: str = ""  # Cryptographic signature
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Active inference components
    confidence: float = 1.0  # Vote confidence
    free_energy: float = 0.0  # Decision free energy


@dataclass
class BFTProof:
    """Cryptographic proof of consensus"""
    phase: BFTPhase
    view: int
    sequence: int
    message_hash: str
    votes: List[BFTVote] = field(default_factory=list)
    threshold_signature: str = ""  # Aggregate signature
    
    def is_valid(self, threshold: int) -> bool:
        """Check if proof meets threshold"""
        approvals = [v for v in self.votes if v.vote_type == VoteType.APPROVE]
        return len(approvals) >= threshold


@dataclass
class BFTConfig:
    """Configuration for Byzantine consensus"""
    node_id: str
    total_nodes: int = 4
    fault_tolerance: int = 1  # f nodes can be faulty
    timeout_ms: int = 5000
    view_change_timeout_ms: int = 10000
    enable_active_inference: bool = True
    confidence_threshold: float = 0.7


class ByzantineConsensus:
    """
    Byzantine Fault Tolerant Consensus with HotStuff protocol
    Supports f faulty nodes out of 3f+1 total nodes
    """
    
    def __init__(self, config: BFTConfig):
        self.config = config
        self.node_id = config.node_id
        
        # Calculate thresholds
        self.total_nodes = config.total_nodes
        self.fault_tolerance = config.fault_tolerance
        self.threshold = 2 * self.fault_tolerance + 1  # 2f+1 for consensus
        
        # Consensus state
        self.current_view = 0
        self.current_sequence = 0
        self.current_phase = BFTPhase.NEW_VIEW
        self.is_leader = False
        
        # Message and vote tracking
        self.messages: Dict[str, BFTMessage] = {}
        self.phase_votes: Dict[BFTPhase, List[BFTVote]] = defaultdict(list)
        self.vote_history: Dict[str, List[BFTVote]] = defaultdict(list)
        self.proofs: Dict[Tuple[int, int], Dict[BFTPhase, BFTProof]] = defaultdict(dict)
        
        # Byzantine detection
        self.byzantine_nodes: Set[str] = set()
        self.node_reputation: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Active inference for confidence
        self.decision_history = deque(maxlen=100)
        self.confidence_model = ActiveInferenceConfidence() if config.enable_active_inference else None
        
        # Locks and events
        self.vote_lock = asyncio.Lock()
        self.phase_events: Dict[BFTPhase, asyncio.Event] = {
            phase: asyncio.Event() for phase in BFTPhase
        }
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info(f"Byzantine consensus initialized for node {self.node_id}")
    
    async def start(self):
        """Start consensus protocol"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._view_change_timer()))
        self._tasks.append(asyncio.create_task(self._byzantine_detector()))
        
        logger.info(f"Byzantine consensus started for node {self.node_id}")
    
    async def stop(self):
        """Stop consensus protocol"""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info(f"Byzantine consensus stopped for node {self.node_id}")
    
    async def propose(self, proposal: Dict[str, Any]) -> BFTMessage:
        """Propose a new message for consensus"""
        if not self.is_leader:
            raise Exception("Only leader can propose")
        
        # Create message
        message = BFTMessage(
            phase=BFTPhase.PREPARE,
            view=self.current_view,
            sequence=self.current_sequence + 1,
            proposal=proposal,
            proposer_id=self.node_id
        )
        
        self.messages[message.message_hash] = message
        
        # Start HotStuff protocol
        await self._execute_hotstuff_round(message)
        
        return message
    
    async def handle_message(self, message: BFTMessage) -> Optional[BFTVote]:
        """Handle incoming BFT message"""
        # Validate message
        if not self._validate_message(message):
            logger.warning(f"Invalid message from {message.proposer_id}")
            return None
        
        # Store message
        self.messages[message.message_hash] = message
        
        # Create vote based on evaluation
        vote = await self._create_vote(message)
        
        # Process vote
        await self.handle_vote(vote)
        
        return vote
    
    async def handle_vote(self, vote: BFTVote) -> bool:
        """Handle incoming vote"""
        async with self.vote_lock:
            # Check for Byzantine behavior
            if self._is_duplicate_or_conflicting(vote):
                logger.warning(f"Byzantine behavior detected from {vote.voter_id}")
                self.byzantine_nodes.add(vote.voter_id)
                self._update_reputation(vote.voter_id, -0.1)
                return False
            
            # Record vote
            self.phase_votes[vote.phase].append(vote)
            self.vote_history[vote.voter_id].append(vote)
            
            # Check if we have threshold
            if self._check_threshold(vote):
                # Create proof
                proof = self._create_proof(vote.phase, vote.view, vote.sequence)
                self.proofs[(vote.view, vote.sequence)][vote.phase] = proof
                
                # Signal phase completion
                self.phase_events[vote.phase].set()
                
                # Move to next phase
                await self._advance_phase(vote.phase, proof)
                
                return True
        
        return False
    
    def _validate_message(self, message: BFTMessage) -> bool:
        """Validate BFT message"""
        # Check view
        if message.view < self.current_view:
            return False
        
        # Check sequence
        if message.sequence <= self.current_sequence and message.phase != BFTPhase.NEW_VIEW:
            return False
        
        # Check if proposer is Byzantine
        if message.proposer_id in self.byzantine_nodes:
            return False
        
        # Verify message hash
        expected_hash = self._calculate_message_hash(message)
        if message.message_hash != expected_hash:
            return False
        
        return True
    
    def _calculate_message_hash(self, message: BFTMessage) -> str:
        """Calculate message hash"""
        content = f"{message.phase.value}:{message.view}:{message.sequence}:{json.dumps(message.proposal, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _create_vote(self, message: BFTMessage) -> BFTVote:
        """Create vote for message"""
        # Evaluate proposal
        vote_type, confidence = await self._evaluate_proposal(message.proposal)
        
        # Calculate free energy if using active inference
        free_energy = 0.0
        if self.confidence_model:
            free_energy = await self.confidence_model.calculate_free_energy(
                message.proposal,
                self.decision_history
            )
        
        # Create vote
        vote = BFTVote(
            voter_id=self.node_id,
            message_hash=message.message_hash,
            phase=message.phase,
            view=message.view,
            sequence=message.sequence,
            vote_type=vote_type,
            confidence=confidence,
            free_energy=free_energy
        )
        
        # Sign vote
        vote.signature = self._sign_vote(vote)
        
        return vote
    
    async def _evaluate_proposal(self, proposal: Dict[str, Any]) -> Tuple[VoteType, float]:
        """Evaluate proposal and determine vote"""
        # Use active inference if enabled
        if self.confidence_model:
            decision = await self.confidence_model.evaluate(proposal)
            
            if decision.confidence < self.config.confidence_threshold:
                return VoteType.ABSTAIN, decision.confidence
            
            return VoteType.APPROVE if decision.approve else VoteType.REJECT, decision.confidence
        
        # Simple evaluation fallback
        # Check proposal validity
        if "action" not in proposal or "value" not in proposal:
            return VoteType.REJECT, 0.9
        
        # Check value bounds
        value = proposal.get("value", 0)
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            return VoteType.APPROVE, 0.8
        
        return VoteType.REJECT, 0.7
    
    def _sign_vote(self, vote: BFTVote) -> str:
        """Create cryptographic signature for vote"""
        # Simplified signature (in production, use real crypto)
        content = f"{vote.voter_id}:{vote.message_hash}:{vote.phase.value}:{vote.vote_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _is_duplicate_or_conflicting(self, vote: BFTVote) -> bool:
        """Check for duplicate or conflicting votes"""
        for prev_vote in self.vote_history.get(vote.voter_id, []):
            if (prev_vote.phase == vote.phase and 
                prev_vote.view == vote.view and
                prev_vote.sequence == vote.sequence):
                # Check if voting differently on same proposal
                return prev_vote.message_hash != vote.message_hash
        return False
    
    def _check_threshold(self, vote: BFTVote) -> bool:
        """Check if we have threshold votes for phase"""
        phase_votes = [v for v in self.phase_votes[vote.phase] 
                      if v.view == vote.view and v.sequence == vote.sequence 
                      and v.message_hash == vote.message_hash
                      and v.vote_type == VoteType.APPROVE]
        
        return len(phase_votes) >= self.threshold
    
    def _create_proof(self, phase: BFTPhase, view: int, sequence: int) -> BFTProof:
        """Create cryptographic proof of consensus"""
        # Get relevant votes
        votes = [v for v in self.phase_votes[phase]
                if v.view == view and v.sequence == sequence]
        
        # Filter approvals
        approvals = [v for v in votes if v.vote_type == VoteType.APPROVE]
        
        # Get message hash (should be same for all approvals)
        message_hash = approvals[0].message_hash if approvals else ""
        
        # Create proof
        proof = BFTProof(
            phase=phase,
            view=view,
            sequence=sequence,
            message_hash=message_hash,
            votes=approvals
        )
        
        # Create threshold signature (simplified)
        signatures = [v.signature for v in approvals]
        proof.threshold_signature = hashlib.sha256("".join(signatures).encode()).hexdigest()[:32]
        
        return proof
    
    async def _advance_phase(self, current_phase: BFTPhase, proof: BFTProof):
        """Advance to next phase in HotStuff protocol"""
        # Determine next phase
        next_phase = None
        
        if current_phase == BFTPhase.PREPARE:
            next_phase = BFTPhase.PRE_COMMIT
        elif current_phase == BFTPhase.PRE_COMMIT:
            next_phase = BFTPhase.COMMIT
        elif current_phase == BFTPhase.COMMIT:
            next_phase = BFTPhase.DECIDE
            # Consensus reached!
            await self._finalize_decision(proof)
        
        if next_phase and self.is_leader:
            # Leader broadcasts next phase message
            message = BFTMessage(
                phase=next_phase,
                view=proof.view,
                sequence=proof.sequence,
                proposal={"previous_proof": proof.threshold_signature},
                proposer_id=self.node_id
            )
            
            # This would be broadcast to all nodes
            await self.handle_message(message)
    
    async def _finalize_decision(self, proof: BFTProof):
        """Finalize consensus decision"""
        # Update sequence number
        self.current_sequence = proof.sequence
        
        # Record decision
        if self.confidence_model:
            await self.confidence_model.record_decision(
                proof.message_hash,
                True,  # Consensus reached
                proof.votes[0].confidence if proof.votes else 0.5
            )
        
        # Update reputation for voters
        for vote in proof.votes:
            self._update_reputation(vote.voter_id, 0.05)
        
        logger.info(f"Consensus reached for sequence {proof.sequence}")
    
    def _update_reputation(self, node_id: str, delta: float):
        """Update node reputation"""
        self.node_reputation[node_id] = max(0, min(1, 
            self.node_reputation[node_id] + delta
        ))
    
    async def _execute_hotstuff_round(self, message: BFTMessage):
        """Execute a full HotStuff consensus round"""
        try:
            # Phase 1: Prepare
            message.phase = BFTPhase.PREPARE
            await self.handle_message(message)
            
            # Wait for prepare votes
            await asyncio.wait_for(
                self.phase_events[BFTPhase.PREPARE].wait(),
                timeout=self.config.timeout_ms / 1000
            )
            
            # Phase 2: Pre-commit
            prepare_proof = self.proofs[(message.view, message.sequence)][BFTPhase.PREPARE]
            if prepare_proof.is_valid(self.threshold):
                await self._advance_phase(BFTPhase.PREPARE, prepare_proof)
                
                # Wait for pre-commit votes
                await asyncio.wait_for(
                    self.phase_events[BFTPhase.PRE_COMMIT].wait(),
                    timeout=self.config.timeout_ms / 1000
                )
                
                # Phase 3: Commit
                precommit_proof = self.proofs[(message.view, message.sequence)][BFTPhase.PRE_COMMIT]
                if precommit_proof.is_valid(self.threshold):
                    await self._advance_phase(BFTPhase.PRE_COMMIT, precommit_proof)
                    
                    # Wait for commit votes
                    await asyncio.wait_for(
                        self.phase_events[BFTPhase.COMMIT].wait(),
                        timeout=self.config.timeout_ms / 1000
                    )
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout in HotStuff round for sequence {message.sequence}")
            # Trigger view change
            await self._trigger_view_change()
    
    async def _trigger_view_change(self):
        """Trigger view change when timeout occurs"""
        self.current_view += 1
        self.current_phase = BFTPhase.NEW_VIEW
        
        # Clear phase events
        for event in self.phase_events.values():
            event.clear()
        
        # Clear votes for new view
        self.phase_votes.clear()
        
        # Determine new leader (simple round-robin)
        leader_index = self.current_view % self.total_nodes
        self.is_leader = (leader_index == int(self.node_id.split("_")[-1]))
        
        logger.info(f"View change to {self.current_view}, leader: {self.is_leader}")
    
    async def _view_change_timer(self):
        """Monitor for view change timeouts"""
        while self._running:
            try:
                # Check if we need view change
                last_decision_time = max(
                    (vote.timestamp for votes in self.vote_history.values() 
                     for vote in votes),
                    default=datetime.now()
                )
                
                if (datetime.now() - last_decision_time).total_seconds() * 1000 > self.config.view_change_timeout_ms:
                    await self._trigger_view_change()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"View change timer error: {e}")
    
    async def _byzantine_detector(self):
        """Detect Byzantine nodes based on behavior"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Analyze vote patterns
                for node_id, votes in self.vote_history.items():
                    if len(votes) < 10:
                        continue
                    
                    # Check for inconsistent voting patterns
                    recent_votes = list(votes)[-20:]
                    
                    # Count conflicting votes
                    conflicts = 0
                    for i, vote1 in enumerate(recent_votes):
                        for vote2 in recent_votes[i+1:]:
                            if (vote1.view == vote2.view and 
                                vote1.sequence == vote2.sequence and
                                vote1.phase == vote2.phase and
                                vote1.message_hash != vote2.message_hash):
                                conflicts += 1
                    
                    # Mark as Byzantine if too many conflicts
                    if conflicts > 3:
                        self.byzantine_nodes.add(node_id)
                        self.node_reputation[node_id] = 0
                        logger.warning(f"Node {node_id} marked as Byzantine")
                
            except Exception as e:
                logger.error(f"Byzantine detector error: {e}")
    
    def get_byzantine_nodes(self) -> Set[str]:
        """Get detected Byzantine nodes"""
        return self.byzantine_nodes.copy()
    
    def get_consensus_state(self) -> Dict[str, Any]:
        """Get current consensus state"""
        return {
            "node_id": self.node_id,
            "view": self.current_view,
            "sequence": self.current_sequence,
            "phase": self.current_phase.value,
            "is_leader": self.is_leader,
            "byzantine_nodes": list(self.byzantine_nodes),
            "reputation": dict(self.node_reputation),
            "total_votes": sum(len(votes) for votes in self.vote_history.values()),
            "threshold": self.threshold
        }


class ActiveInferenceConfidence:
    """Active inference model for consensus confidence"""
    
    def __init__(self):
        self.belief_buffer = deque(maxlen=50)
        self.prediction_errors = deque(maxlen=50)
        self.learning_rate = 0.05
    
    async def evaluate(self, proposal: Dict[str, Any]) -> Any:
        """Evaluate proposal using active inference"""
        # Calculate expected utility
        expected_utility = self._calculate_expected_utility(proposal)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(proposal)
        
        # Decision based on free energy minimization
        confidence = 1.0 / (1.0 + uncertainty)
        approve = expected_utility > 0.5
        
        @dataclass
        class Decision:
            approve: bool
            confidence: float
        
        return Decision(approve=approve, confidence=confidence)
    
    async def calculate_free_energy(self, proposal: Dict[str, Any], history: deque) -> float:
        """Calculate free energy for decision"""
        # Surprisal based on proposal novelty
        surprisal = self._calculate_surprisal(proposal, history)
        
        # Complexity based on proposal structure
        complexity = len(str(proposal)) / 100.0  # Normalized
        
        return surprisal + complexity
    
    def _calculate_expected_utility(self, proposal: Dict[str, Any]) -> float:
        """Calculate expected utility of proposal"""
        # Simple utility based on proposal features
        value = proposal.get("value", 0.5)
        if isinstance(value, (int, float)):
            return max(0, min(1, value))
        return 0.5
    
    def _calculate_uncertainty(self, proposal: Dict[str, Any]) -> float:
        """Calculate uncertainty about proposal"""
        # Base uncertainty
        uncertainty = 0.5
        
        # Reduce uncertainty if we've seen similar proposals
        for belief in self.belief_buffer:
            similarity = self._calculate_similarity(proposal, belief)
            uncertainty *= (1 - similarity * 0.1)
        
        return uncertainty
    
    def _calculate_surprisal(self, proposal: Dict[str, Any], history: deque) -> float:
        """Calculate surprisal (unexpectedness) of proposal"""
        if not history:
            return 1.0
        
        # Check how different from recent proposals
        total_diff = 0
        for past in list(history)[-10:]:
            diff = 1 - self._calculate_similarity(proposal, past)
            total_diff += diff
        
        return total_diff / min(10, len(history))
    
    def _calculate_similarity(self, prop1: Dict[str, Any], prop2: Dict[str, Any]) -> float:
        """Calculate similarity between proposals"""
        # Simple key overlap similarity
        keys1 = set(prop1.keys())
        keys2 = set(prop2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union)
    
    async def record_decision(self, decision_id: str, success: bool, confidence: float):
        """Record decision outcome for learning"""
        self.belief_buffer.append({
            "decision_id": decision_id,
            "success": success,
            "confidence": confidence,
            "timestamp": datetime.now()
        })
        
        # Update prediction error
        prediction_error = abs(confidence - (1.0 if success else 0.0))
        self.prediction_errors.append(prediction_error)


# HotStuff consensus variant
class HotStuffConsensus(ByzantineConsensus):
    """
    HotStuff consensus implementation
    Linear communication complexity in the optimistic case
    """
    
    def __init__(self, config: BFTConfig):
        super().__init__(config)
        
        # HotStuff specific state
        self.generic_qc: Optional[BFTProof] = None  # Generic quorum certificate
        self.locked_qc: Optional[BFTProof] = None   # Locked quorum certificate
        self.prepare_qc: Optional[BFTProof] = None  # Prepare quorum certificate
        
        logger.info("HotStuff consensus initialized")
    
    async def handle_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Handle proposal in HotStuff protocol"""
        # Create new-view message with highest QC
        if self.is_leader:
            message = BFTMessage(
                phase=BFTPhase.NEW_VIEW,
                view=self.current_view,
                sequence=self.current_sequence + 1,
                proposal=proposal,
                proposer_id=self.node_id
            )
            
            # Attach highest QC
            if self.generic_qc:
                message.proposal["qc"] = self.generic_qc.threshold_signature
            
            # Execute consensus round
            await self._execute_hotstuff_round(message)
            return True
        
        return False
    
    def _update_qc(self, phase: BFTPhase, proof: BFTProof):
        """Update quorum certificates based on phase"""
        if phase == BFTPhase.PREPARE:
            self.prepare_qc = proof
        elif phase == BFTPhase.PRE_COMMIT and proof.is_valid(self.threshold):
            self.locked_qc = proof
        elif phase == BFTPhase.COMMIT and proof.is_valid(self.threshold):
            self.generic_qc = proof


# Example usage
async def example_byzantine_consensus():
    """Example of Byzantine consensus in action"""
    # Configure nodes
    configs = []
    for i in range(4):
        config = BFTConfig(
            node_id=f"node_{i}",
            total_nodes=4,
            fault_tolerance=1  # Tolerates 1 Byzantine node
        )
        configs.append(config)
    
    # Create consensus instances
    nodes = [ByzantineConsensus(config) for config in configs]
    
    # Start all nodes
    for node in nodes:
        await node.start()
    
    # Set first node as leader
    nodes[0].is_leader = True
    
    # Propose something
    proposal = {
        "action": "update_parameter",
        "value": 0.75,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Leader proposes
        message = await nodes[0].propose(proposal)
        print(f"Proposed: {message.message_hash[:8]}...")
        
        # Simulate other nodes receiving and voting
        for i in range(1, 4):
            vote = await nodes[i].handle_message(message)
            if vote:
                # Broadcast vote to all nodes
                for node in nodes:
                    await node.handle_vote(vote)
        
        # Check consensus state
        await asyncio.sleep(1)
        for node in nodes:
            state = node.get_consensus_state()
            print(f"Node {node.node_id}: sequence={state['sequence']}, phase={state['phase']}")
        
    finally:
        # Stop all nodes
        for node in nodes:
            await node.stop()


if __name__ == "__main__":
    asyncio.run(example_byzantine_consensus())