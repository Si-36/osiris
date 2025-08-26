"""
Consensus Orchestration for Multi-Agent Decisions

Production-ready distributed consensus with TDA context integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class VoteType(Enum):
    """Types of votes in consensus"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"

class ConsensusAlgorithm(Enum):
    """Consensus algorithms"""
    MAJORITY = "majority"          # Simple majority
    SUPERMAJORITY = "supermajority"  # 2/3 majority
    UNANIMOUS = "unanimous"        # All must agree
    WEIGHTED = "weighted"          # TDA-weighted voting

@dataclass
class ConsensusProposal:
    """Consensus proposal for multi-agent decision"""
    proposal_id: str
    proposer_id: str
    decision_type: str
    context: Dict[str, Any]
    timeout_seconds: int = 300
    algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Vote:
    """Individual vote in consensus"""
    voter_id: str
    proposal_id: str
    vote_type: VoteType
    confidence: float = 1.0
    reasoning: Optional[str] = None
    tda_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConsensusResult:
    """Result of consensus process"""
    proposal_id: str
    decision: str  # "approved", "rejected", "timeout"
    votes: List[Vote]
    final_tally: Dict[str, int]
    confidence_score: float
    execution_time_seconds: float
    tda_correlation: Optional[str] = None

class ConsensusOrchestrator:
    """
    Distributed consensus orchestrator for multi-agent decisions.
    
    Integrates with TDA for context-aware voting and conflict resolution.
    """
    
    def __init__(self, tda_integration: Optional[Any] = None):
        self.tda_integration = tda_integration
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.votes: Dict[str, List[Vote]] = {}
        self.consensus_results: List[ConsensusResult] = []
        
        # Agent registry
        self.registered_agents: Set[str] = set()
        self.agent_weights: Dict[str, float] = {}
        self.agent_tda_scores: Dict[str, float] = {}
        
        # Performance metrics
        self.total_proposals = 0
        self.successful_consensus = 0
        self.avg_consensus_time = 0.0
        
        logger.info("Consensus Orchestrator initialized")
    
    def register_agent(self, agent_id: str, weight: float = 1.0, 
        tda_score: float = 0.5) -> None:
        """Register agent for consensus participation"""
        self.registered_agents.add(agent_id)
        self.agent_weights[agent_id] = weight
        self.agent_tda_scores[agent_id] = tda_score
        
        logger.info(f"Registered agent {agent_id} with weight {weight}")
    
        async def submit_proposal(self, proposal: ConsensusProposal) -> str:
        """Submit proposal for consensus"""
        self.active_proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = []
        self.total_proposals += 1
        
        logger.info(f"Submitted proposal {proposal.proposal_id} for consensus")
        
        # Start consensus process
        asyncio.create_task(self._process_consensus(proposal))
        
        return proposal.proposal_id
    
        async def submit_vote(self, vote: Vote) -> bool:
        """Submit vote for active proposal"""
        if vote.proposal_id not in self.active_proposals:
            logger.warning(f"Vote for unknown proposal {vote.proposal_id}")
            return False
        
        if vote.voter_id not in self.registered_agents:
            logger.warning(f"Vote from unregistered agent {vote.voter_id}")
            return False
        
        # Check for duplicate votes
        existing_votes = self.votes[vote.proposal_id]
        if any(v.voter_id == vote.voter_id for v in existing_votes):
            logger.warning(f"Duplicate vote from {vote.voter_id}")
            return False
        
        # Add TDA context if available
        if self.tda_integration and not vote.tda_context:
            vote.tda_context = self._get_tda_context(vote.voter_id)
        
        self.votes[vote.proposal_id].append(vote)
        
        logger.debug(f"Recorded vote from {vote.voter_id}: {vote.vote_type.value}")
        return True
    
        async def _process_consensus(self, proposal: ConsensusProposal) -> ConsensusResult:
        """Process consensus for proposal"""
        start_time = datetime.utcnow()
        
        # Wait for votes or timeout
        timeout_time = start_time + timedelta(seconds=proposal.timeout_seconds)
        
        while datetime.utcnow() < timeout_time:
            # Check if we have enough votes for decision
            result = self._evaluate_consensus(proposal)
            if result:
                # Consensus reached
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                result.execution_time_seconds = execution_time
                
                self.consensus_results.append(result)
                self.successful_consensus += 1
                self._update_consensus_metrics(execution_time)
                
                # Cleanup
                del self.active_proposals[proposal.proposal_id]
                
                logger.info(f"Consensus reached for {proposal.proposal_id}: {result.decision}")
                return result
            
            # Wait before checking again
            await asyncio.sleep(1.0)
        
        # Timeout - create timeout result
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        timeout_result = ConsensusResult(
            proposal_id=proposal.proposal_id,
            decision="timeout",
            votes=self.votes[proposal.proposal_id].copy(),
            final_tally=self._count_votes(proposal.proposal_id),
            confidence_score=0.0,
            execution_time_seconds=execution_time
        )
        
        self.consensus_results.append(timeout_result)
        self._update_consensus_metrics(execution_time)
        
        # Cleanup
        del self.active_proposals[proposal.proposal_id]
        
        logger.warning(f"Consensus timeout for {proposal.proposal_id}")
        return timeout_result
    
    def _evaluate_consensus(self, proposal: ConsensusProposal) -> Optional[ConsensusResult]:
        """Evaluate if consensus has been reached"""
        votes = self.votes[proposal.proposal_id]
        
        if not votes:
            return None
        
        if proposal.algorithm == ConsensusAlgorithm.MAJORITY:
            return self._evaluate_majority(proposal, votes)
        elif proposal.algorithm == ConsensusAlgorithm.SUPERMAJORITY:
            return self._evaluate_supermajority(proposal, votes)
        elif proposal.algorithm == ConsensusAlgorithm.UNANIMOUS:
            return self._evaluate_unanimous(proposal, votes)
        elif proposal.algorithm == ConsensusAlgorithm.WEIGHTED:
            return self._evaluate_weighted(proposal, votes)
        
        return None
    
    def _evaluate_majority(self, proposal: ConsensusProposal, 
        votes: List[Vote]) -> Optional[ConsensusResult]:
        """Evaluate simple majority consensus"""
        total_agents = len(self.registered_agents)
        required_votes = (total_agents // 2) + 1
        
        if len(votes) < required_votes:
            return None
        
        tally = self._count_votes(proposal.proposal_id)
        approve_count = tally.get('approve', 0)
        reject_count = tally.get('reject', 0)
        
        if approve_count > reject_count and approve_count >= required_votes:
            decision = "approved"
            confidence = approve_count / len(votes)
        elif reject_count > approve_count and reject_count >= required_votes:
            decision = "rejected"
            confidence = reject_count / len(votes)
        else:
            return None  # No clear majority yet
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            decision=decision,
            votes=votes.copy(),
            final_tally=tally,
            confidence_score=confidence,
            execution_time_seconds=0.0  # Will be set by caller
        )
    
    def _evaluate_supermajority(self, proposal: ConsensusProposal,
        votes: List[Vote]) -> Optional[ConsensusResult]:
        """Evaluate 2/3 supermajority consensus"""
        total_agents = len(self.registered_agents)
        required_votes = int(total_agents * 2 / 3) + 1
        
        if len(votes) < required_votes:
            return None
        
        tally = self._count_votes(proposal.proposal_id)
        approve_count = tally.get('approve', 0)
        reject_count = tally.get('reject', 0)
        
        if approve_count >= required_votes:
            decision = "approved"
            confidence = approve_count / len(votes)
        elif reject_count >= required_votes:
            decision = "rejected"
            confidence = reject_count / len(votes)
        else:
            return None
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            decision=decision,
            votes=votes.copy(),
            final_tally=tally,
            confidence_score=confidence,
            execution_time_seconds=0.0
        )
    
    def _evaluate_unanimous(self, proposal: ConsensusProposal,
        votes: List[Vote]) -> Optional[ConsensusResult]:
        """Evaluate unanimous consensus"""
        total_agents = len(self.registered_agents)
        
        if len(votes) < total_agents:
            return None
        
        tally = self._count_votes(proposal.proposal_id)
        
        if tally.get('approve', 0) == total_agents:
            decision = "approved"
            confidence = 1.0
        elif tally.get('reject', 0) == total_agents:
            decision = "rejected"
            confidence = 1.0
        else:
            decision = "rejected"  # Not unanimous
            confidence = 0.0
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            decision=decision,
            votes=votes.copy(),
            final_tally=tally,
            confidence_score=confidence,
            execution_time_seconds=0.0
        )
    
    def _evaluate_weighted(self, proposal: ConsensusProposal,
        votes: List[Vote]) -> Optional[ConsensusResult]:
        """Evaluate TDA-weighted consensus"""
        # Calculate weighted votes
        weighted_approve = 0.0
        weighted_reject = 0.0
        total_weight = 0.0
        
        for vote in votes:
            agent_weight = self.agent_weights.get(vote.voter_id, 1.0)
            tda_score = self.agent_tda_scores.get(vote.voter_id, 0.5)
            
            # TDA-enhanced weight
            enhanced_weight = agent_weight * (1.0 + tda_score * 0.5)
            total_weight += enhanced_weight
            
            if vote.vote_type == VoteType.APPROVE:
                weighted_approve += enhanced_weight
            elif vote.vote_type == VoteType.REJECT:
                weighted_reject += enhanced_weight
        
        # Need at least 50% of total possible weight
        total_possible_weight = sum(
            self.agent_weights.get(agent_id, 1.0) * 
            (1.0 + self.agent_tda_scores.get(agent_id, 0.5) * 0.5)
            for agent_id in self.registered_agents
        )
        
        if total_weight < total_possible_weight * 0.5:
            return None  # Not enough participation
        
        if weighted_approve > weighted_reject:
            decision = "approved"
            confidence = weighted_approve / total_weight
        else:
            decision = "rejected"
            confidence = weighted_reject / total_weight
        
        return ConsensusResult(
            proposal_id=proposal.proposal_id,
            decision=decision,
            votes=votes.copy(),
            final_tally=self._count_votes(proposal.proposal_id),
            confidence_score=confidence,
            execution_time_seconds=0.0
        )
    
    def _count_votes(self, proposal_id: str) -> Dict[str, int]:
        """Count votes for proposal"""
        votes = self.votes.get(proposal_id, [])
        tally = {'approve': 0, 'reject': 0, 'abstain': 0}
        
        for vote in votes:
            tally[vote.vote_type.value] += 1
        
        return tally
    
    def _get_tda_context(self, agent_id: str) -> Dict[str, Any]:
        """Get TDA context for agent (mock implementation)"""
        # In production, this would query actual TDA system
        return {
            'agent_id': agent_id,
            'tda_score': self.agent_tda_scores.get(agent_id, 0.5),
            'anomaly_detected': False,
            'context_timestamp': datetime.utcnow().isoformat()
        }
    
    def _update_consensus_metrics(self, execution_time: float) -> None:
        """Update consensus performance metrics"""
        if self.successful_consensus == 1:
            self.avg_consensus_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_consensus_time = (
                alpha * execution_time + 
                (1 - alpha) * self.avg_consensus_time
            )
    
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get consensus orchestrator status"""
        pass
        success_rate = (
            self.successful_consensus / max(1, self.total_proposals) * 100
        )
        
        return {
            'total_proposals': self.total_proposals,
            'successful_consensus': self.successful_consensus,
            'success_rate': f"{success_rate:.1f}%",
            'avg_consensus_time': f"{self.avg_consensus_time:.2f}s",
            'active_proposals': len(self.active_proposals),
            'registered_agents': len(self.registered_agents),
            'tda_integration': self.tda_integration is not None
        }

# Factory function
    def create_consensus_orchestrator(tda_integration: Optional[Any] = None) -> ConsensusOrchestrator:
        """Create consensus orchestrator with optional TDA integration"""
        return ConsensusOrchestrator(tda_integration=tda_integration)
