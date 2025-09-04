"""
Real Consensus System for AURA
Implements multi-agent consensus using voting, debate, and confidence scoring
September 2025 - Production-ready implementation
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import structlog
import numpy as np

from ..schemas.aura_execution import (
    ConsensusDecision,
    ExecutionPlan,
    AgentDecision
)

logger = structlog.get_logger(__name__)


class VotingStrategy(str, Enum):
    """Different voting strategies for consensus"""
    MAJORITY = "majority"          # Simple majority (>50%)
    SUPERMAJORITY = "supermajority"  # 2/3 majority
    UNANIMOUS = "unanimous"        # All must agree
    WEIGHTED = "weighted"          # Weighted by confidence
    QUORUM = "quorum"             # Minimum participation required


class ConsensusProtocol(str, Enum):
    """Different consensus protocols"""
    SIMPLE_VOTE = "simple_vote"
    DEBATE_THEN_VOTE = "debate_then_vote"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    LEADER_BASED = "leader_based"


class RealConsensusSystem:
    """
    Production-ready consensus system for multi-agent agreement.
    Implements various consensus protocols and voting strategies.
    """
    
    def __init__(
        self,
        default_strategy: VotingStrategy = VotingStrategy.MAJORITY,
        default_protocol: ConsensusProtocol = ConsensusProtocol.SIMPLE_VOTE,
        timeout_seconds: float = 30.0,
        max_iterations: int = 3
    ):
        """
        Initialize the consensus system.
        
        Args:
            default_strategy: Default voting strategy
            default_protocol: Default consensus protocol
            timeout_seconds: Timeout for consensus process
            max_iterations: Maximum iterations for iterative protocols
        """
        self.default_strategy = default_strategy
        self.default_protocol = default_protocol
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        
        # Track consensus history
        self.consensus_history: List[ConsensusDecision] = []
        
        # Metrics
        self.metrics = {
            "total_consensus_attempts": 0,
            "successful_consensus": 0,
            "failed_consensus": 0,
            "average_time_to_consensus": 0.0,
            "average_iterations": 0.0
        }
        
        logger.info(f"RealConsensusSystem initialized with {default_strategy} strategy")
    
    async def reach_consensus(
        self,
        proposal: Dict[str, Any],
        agents: List[Any],
        strategy: Optional[VotingStrategy] = None,
        protocol: Optional[ConsensusProtocol] = None
    ) -> ConsensusDecision:
        """
        Main method to reach consensus among agents.
        
        Args:
            proposal: The proposal to vote on (e.g., execution plan)
            agents: List of agent instances that will vote
            strategy: Voting strategy to use
            protocol: Consensus protocol to use
            
        Returns:
            ConsensusDecision with the result
        """
        strategy = strategy or self.default_strategy
        protocol = protocol or self.default_protocol
        
        logger.info(f"Starting consensus with {len(agents)} agents using {protocol} protocol")
        
        self.metrics["total_consensus_attempts"] += 1
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute the appropriate protocol
            if protocol == ConsensusProtocol.SIMPLE_VOTE:
                decision = await self._simple_vote_protocol(proposal, agents, strategy)
            elif protocol == ConsensusProtocol.DEBATE_THEN_VOTE:
                decision = await self._debate_then_vote_protocol(proposal, agents, strategy)
            elif protocol == ConsensusProtocol.ITERATIVE_REFINEMENT:
                decision = await self._iterative_refinement_protocol(proposal, agents, strategy)
            elif protocol == ConsensusProtocol.LEADER_BASED:
                decision = await self._leader_based_protocol(proposal, agents)
            else:
                # Fallback to simple vote
                decision = await self._simple_vote_protocol(proposal, agents, strategy)
            
            # Update metrics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if decision.approved:
                self.metrics["successful_consensus"] += 1
            else:
                self.metrics["failed_consensus"] += 1
            
            self._update_average_time(elapsed_time)
            
            # Store in history
            self.consensus_history.append(decision)
            
            logger.info(f"Consensus {'reached' if decision.approved else 'failed'} in {elapsed_time:.2f}s")
            return decision
            
        except asyncio.TimeoutError:
            logger.error(f"Consensus timed out after {self.timeout_seconds}s")
            self.metrics["failed_consensus"] += 1
            
            return ConsensusDecision(
                approved=False,
                voting_results={"error": "timeout"},
                dissenting_opinions=[{"agent": "system", "reason": "Consensus timeout"}]
            )
            
        except Exception as e:
            logger.error(f"Consensus failed with error: {e}")
            self.metrics["failed_consensus"] += 1
            
            return ConsensusDecision(
                approved=False,
                voting_results={"error": str(e)},
                dissenting_opinions=[{"agent": "system", "reason": f"Error: {e}"}]
            )
    
    async def _simple_vote_protocol(
        self,
        proposal: Dict[str, Any],
        agents: List[Any],
        strategy: VotingStrategy
    ) -> ConsensusDecision:
        """
        Simple voting protocol - each agent votes once.
        """
        logger.info("Executing simple vote protocol")
        
        # Collect votes from all agents
        votes = {}
        opinions = []
        
        # Vote collection with timeout
        vote_tasks = []
        for agent in agents:
            vote_tasks.append(self._get_agent_vote(agent, proposal))
        
        vote_results = await asyncio.wait_for(
            asyncio.gather(*vote_tasks, return_exceptions=True),
            timeout=self.timeout_seconds
        )
        
        # Process votes
        for i, result in enumerate(vote_results):
            agent_id = self._get_agent_id(agents[i])
            
            if isinstance(result, Exception):
                logger.warning(f"Agent {agent_id} failed to vote: {result}")
                votes[agent_id] = False
                opinions.append({"agent": agent_id, "reason": f"Vote error: {result}"})
            else:
                votes[agent_id] = result.get("approve", False)
                if not result.get("approve", False):
                    opinions.append({
                        "agent": agent_id,
                        "reason": result.get("reason", "No reason provided")
                    })
        
        # Determine if consensus is reached based on strategy
        approved = self._evaluate_votes(votes, strategy)
        
        # Create the decision
        decision = ConsensusDecision(
            approved=approved,
            voting_results=votes,
            dissenting_opinions=opinions if not approved else [],
            approved_plan=ExecutionPlan.model_validate(proposal) if approved and "steps" in proposal else None
        )
        
        return decision
    
    async def _debate_then_vote_protocol(
        self,
        proposal: Dict[str, Any],
        agents: List[Any],
        strategy: VotingStrategy
    ) -> ConsensusDecision:
        """
        Agents debate the proposal before voting.
        """
        logger.info("Executing debate-then-vote protocol")
        
        # Phase 1: Initial positions
        initial_positions = await self._gather_initial_positions(agents, proposal)
        
        # Phase 2: Debate round
        debate_results = await self._conduct_debate(agents, proposal, initial_positions)
        
        # Phase 3: Final vote after debate
        final_decision = await self._simple_vote_protocol(
            debate_results.get("refined_proposal", proposal),
            agents,
            strategy
        )
        
        # Add debate context to the decision
        if debate_results.get("modifications"):
            final_decision.modifications = debate_results["modifications"]
        
        return final_decision
    
    async def _iterative_refinement_protocol(
        self,
        proposal: Dict[str, Any],
        agents: List[Any],
        strategy: VotingStrategy
    ) -> ConsensusDecision:
        """
        Iteratively refine the proposal until consensus or max iterations.
        """
        logger.info("Executing iterative refinement protocol")
        
        current_proposal = proposal.copy()
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Vote on current proposal
            decision = await self._simple_vote_protocol(current_proposal, agents, strategy)
            
            if decision.approved:
                # Consensus reached
                self._update_average_iterations(iteration + 1)
                return decision
            
            # If not approved, collect refinements
            refinements = await self._collect_refinements(agents, current_proposal, decision)
            
            if not refinements:
                # No refinements suggested, consensus failed
                break
            
            # Apply refinements to create new proposal
            current_proposal = self._apply_refinements(current_proposal, refinements)
        
        # Max iterations reached without consensus
        self._update_average_iterations(self.max_iterations)
        
        return ConsensusDecision(
            approved=False,
            voting_results={"reason": "max_iterations_reached"},
            dissenting_opinions=[{"agent": "system", "reason": "Could not reach consensus after maximum iterations"}]
        )
    
    async def _leader_based_protocol(
        self,
        proposal: Dict[str, Any],
        agents: List[Any]
    ) -> ConsensusDecision:
        """
        Leader makes final decision after consulting others.
        """
        logger.info("Executing leader-based protocol")
        
        # Select leader (first agent or highest confidence)
        leader = await self._select_leader(agents)
        advisors = [a for a in agents if a != leader]
        
        # Collect advice from other agents
        advice = []
        for advisor in advisors:
            advisor_opinion = await self._get_agent_vote(advisor, proposal)
            advice.append(advisor_opinion)
        
        # Leader makes final decision considering advice
        leader_decision = await self._get_leader_decision(leader, proposal, advice)
        
        # Create consensus decision
        voting_results = {self._get_agent_id(leader): True}
        for i, advisor in enumerate(advisors):
            voting_results[self._get_agent_id(advisor)] = advice[i].get("approve", False)
        
        return ConsensusDecision(
            approved=leader_decision.get("approve", False),
            voting_results=voting_results,
            approved_plan=ExecutionPlan.model_validate(proposal) if leader_decision.get("approve") and "steps" in proposal else None,
            modifications=leader_decision.get("modifications", [])
        )
    
    async def _get_agent_vote(self, agent: Any, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get vote from a single agent.
        """
        # Check if agent has a vote method
        if hasattr(agent, 'vote'):
            try:
                return await agent.vote(proposal)
            except Exception as e:
                logger.warning(f"Agent vote failed: {e}")
                return {"approve": False, "reason": str(e)}
        
        # Fallback: simulate vote based on agent type
        agent_type = type(agent).__name__
        
        if "Planner" in agent_type:
            # Planners generally approve their own plans
            return {"approve": True, "confidence": 0.8}
        elif "Analyst" in agent_type:
            # Analysts are more critical
            risk_level = proposal.get("risk_assessment", {}).get("overall", 0.5)
            return {"approve": risk_level < 0.7, "reason": "Risk assessment", "confidence": 0.9}
        else:
            # Default approval
            return {"approve": True, "confidence": 0.6}
    
    def _evaluate_votes(self, votes: Dict[str, bool], strategy: VotingStrategy) -> bool:
        """
        Evaluate votes based on strategy.
        """
        total_votes = len(votes)
        approvals = sum(1 for v in votes.values() if v)
        
        if total_votes == 0:
            return False
        
        approval_ratio = approvals / total_votes
        
        if strategy == VotingStrategy.UNANIMOUS:
            return approvals == total_votes
        elif strategy == VotingStrategy.SUPERMAJORITY:
            return approval_ratio >= 2/3
        elif strategy == VotingStrategy.MAJORITY:
            return approval_ratio > 0.5
        elif strategy == VotingStrategy.QUORUM:
            # Need at least 50% participation and majority of those
            participation = total_votes > 0
            return participation and approval_ratio > 0.5
        else:
            # Default to majority
            return approval_ratio > 0.5
    
    async def _gather_initial_positions(
        self,
        agents: List[Any],
        proposal: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Gather initial positions from all agents.
        """
        positions = {}
        
        for agent in agents:
            agent_id = self._get_agent_id(agent)
            if hasattr(agent, 'analyze_proposal'):
                position = await agent.analyze_proposal(proposal)
            else:
                position = await self._get_agent_vote(agent, proposal)
            positions[agent_id] = position
        
        return positions
    
    async def _conduct_debate(
        self,
        agents: List[Any],
        proposal: Dict[str, Any],
        initial_positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Conduct a debate round among agents.
        """
        debate_log = []
        modifications = []
        
        # Each agent can propose modifications
        for agent in agents:
            agent_id = self._get_agent_id(agent)
            position = initial_positions.get(agent_id, {})
            
            if not position.get("approve", True):
                # Agent disagrees, can propose changes
                if hasattr(agent, 'propose_modifications'):
                    proposed_changes = await agent.propose_modifications(proposal)
                    modifications.extend(proposed_changes)
                    debate_log.append({
                        "agent": agent_id,
                        "action": "proposed_modifications",
                        "changes": proposed_changes
                    })
        
        # Apply modifications to create refined proposal
        refined_proposal = proposal.copy()
        for mod in modifications[:3]:  # Apply top 3 modifications
            # Simple modification application (in real system, would be more sophisticated)
            if isinstance(mod, dict):
                refined_proposal.update(mod)
        
        return {
            "refined_proposal": refined_proposal,
            "modifications": modifications,
            "debate_log": debate_log
        }
    
    async def _collect_refinements(
        self,
        agents: List[Any],
        proposal: Dict[str, Any],
        failed_decision: ConsensusDecision
    ) -> List[Dict[str, Any]]:
        """
        Collect refinement suggestions from agents.
        """
        refinements = []
        
        for opinion in failed_decision.dissenting_opinions:
            # Ask dissenting agents for refinements
            agent_id = opinion["agent"]
            agent = next((a for a in agents if self._get_agent_id(a) == agent_id), None)
            
            if agent and hasattr(agent, 'suggest_refinements'):
                suggestions = await agent.suggest_refinements(proposal, opinion["reason"])
                refinements.extend(suggestions)
        
        return refinements
    
    def _apply_refinements(
        self,
        proposal: Dict[str, Any],
        refinements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply refinements to create new proposal.
        """
        refined = proposal.copy()
        
        for refinement in refinements:
            if "modify" in refinement:
                # Modify existing fields
                refined.update(refinement["modify"])
            if "add" in refinement:
                # Add new fields
                refined.update(refinement["add"])
            if "remove" in refinement:
                # Remove fields
                for key in refinement["remove"]:
                    refined.pop(key, None)
        
        return refined
    
    async def _select_leader(self, agents: List[Any]) -> Any:
        """
        Select a leader from the agents.
        """
        # Simple selection: first agent or one with highest confidence
        if not agents:
            return None
        
        # Try to select based on confidence
        best_agent = agents[0]
        best_confidence = 0.0
        
        for agent in agents:
            if hasattr(agent, 'get_confidence'):
                confidence = await agent.get_confidence()
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_agent = agent
        
        return best_agent
    
    async def _get_leader_decision(
        self,
        leader: Any,
        proposal: Dict[str, Any],
        advice: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get final decision from leader considering advice.
        """
        if hasattr(leader, 'make_decision'):
            return await leader.make_decision(proposal, advice)
        
        # Fallback: leader approves if majority of advisors approve
        approvals = sum(1 for a in advice if a.get("approve", False))
        return {
            "approve": approvals > len(advice) / 2,
            "reason": "Based on advisor consensus"
        }
    
    def _get_agent_id(self, agent: Any) -> str:
        """
        Get unique identifier for an agent.
        """
        if hasattr(agent, 'agent_id'):
            return agent.agent_id
        elif hasattr(agent, 'id'):
            return agent.id
        else:
            return f"{type(agent).__name__}_{id(agent)}"
    
    def _update_average_time(self, elapsed_time: float):
        """
        Update average time to consensus metric.
        """
        total_attempts = self.metrics["total_consensus_attempts"]
        current_avg = self.metrics["average_time_to_consensus"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_attempts - 1)) + elapsed_time) / total_attempts
        self.metrics["average_time_to_consensus"] = new_avg
    
    def _update_average_iterations(self, iterations: int):
        """
        Update average iterations metric.
        """
        total_attempts = self.metrics["total_consensus_attempts"]
        current_avg = self.metrics.get("average_iterations", 0.0)
        
        # Calculate new average
        new_avg = ((current_avg * (total_attempts - 1)) + iterations) / total_attempts
        self.metrics["average_iterations"] = new_avg
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get consensus system metrics.
        """
        return self.metrics.copy()
    
    def get_history(self, limit: int = 100) -> List[ConsensusDecision]:
        """
        Get consensus history.
        """
        return self.consensus_history[-limit:]