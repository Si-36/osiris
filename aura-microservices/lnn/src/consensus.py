"""
Byzantine Consensus for LNN
Extracted from AURA Intelligence consensus algorithms
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json


class ConsensusLevel(Enum):
    """Consensus levels for decision-making."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


@dataclass
class Decision:
    """A decision to be agreed upon."""
    id: str
    type: str
    data: Dict[str, Any]
    proposer: str
    timestamp: datetime
    
    def to_hash(self) -> str:
        """Generate hash of the decision for verification."""
        content = f"{self.id}:{self.type}:{json.dumps(self.data, sort_keys=True)}:{self.proposer}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ConsensusResult:
    """Result of a consensus round."""
    accepted: bool
    decision: Decision
    votes: Dict[str, bool]  # node_id -> vote
    confidence: float
    consensus_time_ms: float
    metadata: Dict[str, Any]


class ByzantineConsensus:
    """
    Byzantine Fault Tolerant Consensus
    Tolerates up to f faulty nodes in a network of 3f+1 nodes
    """
    
    def __init__(self, node_id: str, byzantine_tolerance: int = 1):
        self.node_id = node_id
        self.f = byzantine_tolerance  # Number of faulty nodes tolerated
        self.min_nodes = 3 * self.f + 1
        
        # Voting state
        self.pending_decisions: Dict[str, Decision] = {}
        self.votes: Dict[str, Dict[str, bool]] = {}  # decision_id -> {node_id: vote}
        self.finalized_decisions: Dict[str, ConsensusResult] = {}
    
    async def propose_decision(
        self,
        decision: Decision,
        timeout: timedelta = timedelta(seconds=5)
    ) -> ConsensusResult:
        """
        Propose a decision for Byzantine consensus
        
        Args:
            decision: The decision to propose
            timeout: Maximum time to wait for consensus
            
        Returns:
            ConsensusResult with the outcome
        """
        start_time = datetime.now(timezone.utc)
        
        # Store decision
        self.pending_decisions[decision.id] = decision
        self.votes[decision.id] = {self.node_id: True}  # Vote for own proposal
        
        # Broadcast to peers (simulated for now)
        await self._broadcast_decision(decision)
        
        # Wait for votes
        deadline = datetime.now(timezone.utc) + timeout
        while datetime.now(timezone.utc) < deadline:
            votes = self.votes.get(decision.id, {})
            
            # Check if we have enough votes
            if len(votes) >= self.min_nodes:
                yes_votes = sum(1 for vote in votes.values() if vote)
                no_votes = len(votes) - yes_votes
                
                # Byzantine agreement requires > 2f votes
                if yes_votes > 2 * self.f:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    result = ConsensusResult(
                        accepted=True,
                        decision=decision,
                        votes=votes.copy(),
                        confidence=yes_votes / len(votes),
                        consensus_time_ms=elapsed,
                        metadata={"byzantine_threshold": 2 * self.f + 1}
                    )
                    self.finalized_decisions[decision.id] = result
                    return result
                
                elif no_votes > 2 * self.f:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    result = ConsensusResult(
                        accepted=False,
                        decision=decision,
                        votes=votes.copy(),
                        confidence=no_votes / len(votes),
                        consensus_time_ms=elapsed,
                        metadata={"rejection_votes": no_votes}
                    )
                    self.finalized_decisions[decision.id] = result
                    return result
            
            # Wait before checking again
            await asyncio.sleep(0.1)
        
        # Timeout - no consensus
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        votes = self.votes.get(decision.id, {})
        
        return ConsensusResult(
            accepted=False,
            decision=decision,
            votes=votes.copy(),
            confidence=0.0,
            consensus_time_ms=elapsed,
            metadata={"reason": "timeout", "votes_received": len(votes)}
        )
    
    async def vote_on_decision(self, decision_id: str, vote: bool):
        """Submit a vote for a decision."""
        if decision_id not in self.votes:
            self.votes[decision_id] = {}
        self.votes[decision_id][self.node_id] = vote
    
    async def _broadcast_decision(self, decision: Decision):
        """Broadcast decision to peer nodes (placeholder for real implementation)."""
        # In production, this would use actual network communication
        # For now, we simulate local voting
        pass
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get statistics about consensus operations."""
        total_decisions = len(self.finalized_decisions)
        accepted = sum(1 for r in self.finalized_decisions.values() if r.accepted)
        
        avg_time = 0
        avg_confidence = 0
        if total_decisions > 0:
            avg_time = sum(r.consensus_time_ms for r in self.finalized_decisions.values()) / total_decisions
            avg_confidence = sum(r.confidence for r in self.finalized_decisions.values()) / total_decisions
        
        return {
            "total_decisions": total_decisions,
            "accepted_decisions": accepted,
            "acceptance_rate": accepted / total_decisions if total_decisions > 0 else 0,
            "average_consensus_time_ms": avg_time,
            "average_confidence": avg_confidence,
            "byzantine_tolerance": self.f,
            "minimum_nodes": self.min_nodes
        }