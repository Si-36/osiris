"""
🐝 Swarm Byzantine Consensus - Optimized for Multi-Agent Systems
===============================================================

Specialized Byzantine consensus for swarm intelligence with:
- Locality-aware voting (nearby agents have more weight)
- Task-specific consensus groups
- Dynamic swarm topology adaptation
- Capability-based trust scores

Perfect integration point for SwarmCoordinator.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import structlog
import numpy as np
from collections import defaultdict

from .enhanced_byzantine import EnhancedByzantineConsensus, EnhancedConfig, ConsensusProtocol
from .cabinet.weighted_consensus import WeightingScheme

logger = structlog.get_logger(__name__)


class SwarmTopology(Enum):
    """Swarm network topologies"""
    MESH = "mesh"              # Fully connected
    HIERARCHICAL = "hierarchical"  # Tree structure
    GEOGRAPHIC = "geographic"   # Location-based
    DYNAMIC = "dynamic"        # Adaptive topology


@dataclass
class SwarmAgent:
    """Agent in the swarm"""
    agent_id: str
    capabilities: List[str]
    location: Optional[Tuple[float, float]] = None  # (x, y) coordinates
    trust_score: float = 1.0
    specialization: Optional[str] = None
    last_seen: float = field(default_factory=time.time)


@dataclass  
class ConsensusGroup:
    """Task-specific consensus group"""
    group_id: str
    task_type: str
    members: Set[str]
    required_capabilities: List[str]
    leader: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class SwarmDecision:
    """Decision made by swarm consensus"""
    decision_id: str
    proposal: Any
    consensus_group: ConsensusGroup
    result: Any
    confidence: float
    participants: List[str]
    dissenting_agents: List[str]
    metadata: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class LocalityManager:
    """Manage locality-aware voting weights"""
    
    def __init__(self, distance_threshold: float = 10.0):
        self.distance_threshold = distance_threshold
        self.agent_locations: Dict[str, Tuple[float, float]] = {}
        
    def update_location(self, agent_id: str, location: Tuple[float, float]):
        """Update agent location"""
        self.agent_locations[agent_id] = location
        
    def calculate_distance(self, agent1: str, agent2: str) -> float:
        """Calculate Euclidean distance between agents"""
        if agent1 not in self.agent_locations or agent2 not in self.agent_locations:
            return float('inf')
            
        loc1 = self.agent_locations[agent1]
        loc2 = self.agent_locations[agent2]
        
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
        
    def get_locality_weight(self, agent1: str, agent2: str) -> float:
        """Get weight based on proximity"""
        distance = self.calculate_distance(agent1, agent2)
        
        if distance <= self.distance_threshold:
            # Inverse distance weighting
            return 1.0 + (1.0 - distance / self.distance_threshold)
        else:
            # Too far, minimal weight
            return 0.5


class CapabilityMatcher:
    """Match agents to tasks based on capabilities"""
    
    def __init__(self):
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.capability_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register agent capabilities"""
        self.agent_capabilities[agent_id] = set(capabilities)
        
    def update_capability_score(self, agent_id: str, capability: str, score: float):
        """Update agent's score for specific capability"""
        self.capability_scores[agent_id][capability] = score
        
    def match_agents_to_task(self, 
                           required_capabilities: List[str],
                           available_agents: List[str],
                           min_agents: int) -> List[str]:
        """Find best agents for task requirements"""
        scored_agents = []
        
        for agent_id in available_agents:
            if agent_id not in self.agent_capabilities:
                continue
                
            agent_caps = self.agent_capabilities[agent_id]
            
            # Calculate match score
            match_score = 0.0
            for req_cap in required_capabilities:
                if req_cap in agent_caps:
                    # Base score for having capability
                    match_score += 1.0
                    
                    # Bonus for proven performance
                    if req_cap in self.capability_scores.get(agent_id, {}):
                        match_score += self.capability_scores[agent_id][req_cap]
                        
            if match_score > 0:
                scored_agents.append((agent_id, match_score))
                
        # Sort by score and return top agents
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return [agent_id for agent_id, _ in scored_agents[:min_agents]]


class SwarmByzantineConsensus:
    """
    Byzantine consensus optimized for swarm intelligence.
    Adds swarm-specific features on top of enhanced consensus.
    """
    
    def __init__(self, swarm_config: Optional[Dict[str, Any]] = None):
        self.swarm_config = swarm_config or {}
        
        # Swarm agents
        self.agents: Dict[str, SwarmAgent] = {}
        self.active_agents: Set[str] = set()
        
        # Consensus groups
        self.consensus_groups: Dict[str, ConsensusGroup] = {}
        self.agent_to_groups: Dict[str, Set[str]] = defaultdict(set)
        
        # Swarm features
        self.locality_manager = LocalityManager()
        self.capability_matcher = CapabilityMatcher()
        
        # Enhanced consensus (will be initialized per group)
        self.consensus_instances: Dict[str, EnhancedByzantineConsensus] = {}
        
        # Topology
        self.topology = SwarmTopology.DYNAMIC
        
        # Decision history
        self.decisions: List[SwarmDecision] = []
        
        # Metrics
        self.total_decisions = 0
        self.successful_decisions = 0
        self.average_confidence = 0.0
        
        logger.info("Swarm Byzantine consensus initialized")
        
    def register_agent(self, agent: SwarmAgent):
        """Register an agent in the swarm"""
        self.agents[agent.agent_id] = agent
        self.active_agents.add(agent.agent_id)
        
        # Register capabilities
        self.capability_matcher.register_agent(agent.agent_id, agent.capabilities)
        
        # Register location if provided
        if agent.location:
            self.locality_manager.update_location(agent.agent_id, agent.location)
            
        logger.info(f"Agent {agent.agent_id} registered with capabilities: {agent.capabilities}")
        
    def update_agent_location(self, agent_id: str, location: Tuple[float, float]):
        """Update agent's physical location"""
        if agent_id in self.agents:
            self.agents[agent_id].location = location
            self.locality_manager.update_location(agent_id, location)
            
    def update_agent_trust(self, agent_id: str, trust_delta: float):
        """Update agent's trust score based on behavior"""
        if agent_id in self.agents:
            self.agents[agent_id].trust_score = max(0.0, min(1.0,
                self.agents[agent_id].trust_score + trust_delta
            ))
            
    async def create_consensus_group(self,
                                   task_type: str,
                                   required_capabilities: List[str],
                                   min_size: int = 3) -> ConsensusGroup:
        """Create task-specific consensus group"""
        group_id = f"group_{task_type}_{int(time.time() * 1000)}"
        
        # Find suitable agents
        available = list(self.active_agents)
        matched_agents = self.capability_matcher.match_agents_to_task(
            required_capabilities,
            available,
            min_size * 2  # Get extra for redundancy
        )
        
        if len(matched_agents) < min_size:
            raise ValueError(f"Not enough capable agents for task {task_type}")
            
        # Create group
        group = ConsensusGroup(
            group_id=group_id,
            task_type=task_type,
            members=set(matched_agents[:min_size * 2]),
            required_capabilities=required_capabilities
        )
        
        # Select leader (highest trust score)
        leader_candidates = [(aid, self.agents[aid].trust_score) 
                           for aid in group.members if aid in self.agents]
        if leader_candidates:
            group.leader = max(leader_candidates, key=lambda x: x[1])[0]
            
        self.consensus_groups[group_id] = group
        
        # Track membership
        for agent_id in group.members:
            self.agent_to_groups[agent_id].add(group_id)
            
        # Initialize consensus instance for group
        await self._init_group_consensus(group)
        
        logger.info(f"Created consensus group {group_id} with {len(group.members)} members")
        
        return group
        
    async def _init_group_consensus(self, group: ConsensusGroup):
        """Initialize consensus instance for a group"""
        # Determine fault tolerance based on group size
        fault_tolerance = (len(group.members) - 1) // 3
        
        config = EnhancedConfig(
            node_id=group.leader or list(group.members)[0],
            validators=list(group.members),
            protocol=ConsensusProtocol.HYBRID,
            fault_tolerance=max(1, fault_tolerance),
            enable_dag=True,
            enable_weighting=True,
            weighting_scheme=WeightingScheme.HYBRID
        )
        
        consensus = EnhancedByzantineConsensus(config)
        await consensus.initialize()
        
        self.consensus_instances[group.group_id] = consensus
        
    async def swarm_consensus(self,
                            agents: List[str],
                            proposal: Any,
                            task_type: Optional[str] = None,
                            required_capabilities: Optional[List[str]] = None) -> SwarmDecision:
        """
        Execute swarm consensus with all optimizations.
        
        Args:
            agents: List of participating agents
            proposal: The proposal to decide on
            task_type: Type of task (for group creation)
            required_capabilities: Required agent capabilities
            
        Returns:
            SwarmDecision with results
        """
        decision_id = f"decision_{int(time.time() * 1000)}"
        
        # Create or find consensus group
        if task_type and required_capabilities:
            group = await self.create_consensus_group(
                task_type, 
                required_capabilities,
                min_size=max(3, len(agents) // 3)
            )
        else:
            # Use default group with provided agents
            group = ConsensusGroup(
                group_id=f"adhoc_{decision_id}",
                task_type="general",
                members=set(agents),
                required_capabilities=[]
            )
            await self._init_group_consensus(group)
            
        # Apply swarm-specific weights
        weighted_proposal = self._apply_swarm_weights(proposal, group)
        
        # Execute consensus
        consensus = self.consensus_instances.get(group.group_id)
        if not consensus:
            raise ValueError(f"No consensus instance for group {group.group_id}")
            
        success, result, metadata = await consensus.consensus(
            weighted_proposal,
            proposal_type="weighted_decision" if self.topology == SwarmTopology.GEOGRAPHIC else "generic"
        )
        
        # Calculate confidence based on participation and agreement
        participants = [a for a in group.members if a in agents]
        confidence = len(participants) / len(group.members) if success else 0.0
        
        # Identify dissenting agents (simplified)
        dissenting = [a for a in agents if a not in participants] if success else agents
        
        # Create decision record
        decision = SwarmDecision(
            decision_id=decision_id,
            proposal=proposal,
            consensus_group=group,
            result=result,
            confidence=confidence,
            participants=participants,
            dissenting_agents=dissenting,
            metadata=metadata
        )
        
        self.decisions.append(decision)
        
        # Update metrics
        self.total_decisions += 1
        if success:
            self.successful_decisions += 1
            
        self.average_confidence = (
            (self.average_confidence * (self.total_decisions - 1) + confidence) /
            self.total_decisions
        )
        
        # Update agent trust scores based on participation
        for agent_id in participants:
            self.update_agent_trust(agent_id, 0.01)  # Small trust increase
            
        for agent_id in dissenting:
            if len(dissenting) / len(agents) < 0.3:  # Minority dissent
                self.update_agent_trust(agent_id, -0.02)  # Small trust decrease
                
        logger.info(f"Swarm consensus completed: {decision_id}, "
                   f"success={success}, confidence={confidence:.2f}")
                   
        return decision
        
    def _apply_swarm_weights(self, proposal: Any, group: ConsensusGroup) -> Any:
        """Apply swarm-specific weights to proposal"""
        if self.topology != SwarmTopology.GEOGRAPHIC:
            return proposal  # No modification needed
            
        # For geographic topology, add location weights
        if isinstance(proposal, dict):
            weighted_proposal = proposal.copy()
        else:
            weighted_proposal = {"value": proposal}
            
        # Add agent weights based on proximity
        agent_weights = {}
        if group.leader and group.leader in self.agents:
            leader_loc = self.agents[group.leader].location
            
            for agent_id in group.members:
                if agent_id in self.agents:
                    # Calculate weight based on distance to leader
                    weight = self.locality_manager.get_locality_weight(
                        group.leader, agent_id
                    )
                    
                    # Factor in trust score
                    weight *= self.agents[agent_id].trust_score
                    
                    agent_weights[agent_id] = weight
                    
        weighted_proposal["agent_weights"] = agent_weights
        
        return weighted_proposal
        
    async def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get comprehensive swarm statistics"""
        stats = {
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents),
            "consensus_groups": len(self.consensus_groups),
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate": self.successful_decisions / max(1, self.total_decisions),
            "average_confidence": self.average_confidence,
            "topology": self.topology.value
        }
        
        # Add per-group statistics
        group_stats = {}
        for group_id, group in self.consensus_groups.items():
            if group_id in self.consensus_instances:
                instance = self.consensus_instances[group_id]
                perf_stats = await instance.get_performance_stats()
                group_stats[group_id] = {
                    "members": len(group.members),
                    "task_type": group.task_type,
                    "performance": perf_stats
                }
                
        stats["group_statistics"] = group_stats
        
        # Add agent trust distribution
        trust_scores = [agent.trust_score for agent in self.agents.values()]
        if trust_scores:
            stats["trust_distribution"] = {
                "min": min(trust_scores),
                "max": max(trust_scores),
                "mean": np.mean(trust_scores),
                "std": np.std(trust_scores)
            }
            
        return stats
        
    async def shutdown(self):
        """Shutdown all consensus instances"""
        for instance in self.consensus_instances.values():
            await instance.shutdown()
            
        self.consensus_instances.clear()
        logger.info("Swarm consensus shutdown complete")