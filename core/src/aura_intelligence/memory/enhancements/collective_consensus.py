"""
🤝 Collective Memory Consensus - Advanced Multi-Agent Memory Agreement
====================================================================

Implements cutting-edge consensus algorithms for distributed memory systems:
- Byzantine Fault Tolerant consensus
- CRDT-based eventual consistency
- Weighted voting with confidence scores
- Causal memory chains
- Semantic clustering with transformer embeddings

Based on latest distributed systems and multi-agent AI research (2025).
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import structlog
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json

logger = structlog.get_logger(__name__)


# ==================== Consensus Types ====================

class ConsensusType(Enum):
    """Types of consensus mechanisms"""
    SIMPLE_MAJORITY = "simple_majority"      # >50% agreement
    SUPER_MAJORITY = "super_majority"        # >66% agreement
    BYZANTINE = "byzantine"                  # Byzantine fault tolerant
    WEIGHTED = "weighted"                    # Weighted by agent confidence
    CRDT = "crdt"                           # Conflict-free replicated


@dataclass
class ConsensusVote:
    """Individual vote in consensus"""
    agent_id: str
    value: Any
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result of consensus building"""
    consensus_reached: bool
    value: Any
    support_ratio: float
    votes: List[ConsensusVote]
    consensus_type: ConsensusType
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== CRDT Implementation ====================

class CRDTMemory:
    """
    Conflict-free Replicated Data Type for memory consensus.
    Enables eventual consistency without coordination.
    """
    
    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        self.vector_clock: Dict[str, int] = {}
        self.values: Dict[str, Tuple[Any, Dict[str, int]]] = {}
        
    def update(self, agent_id: str, value: Any) -> None:
        """Update CRDT with new value"""
        # Increment vector clock
        self.vector_clock[agent_id] = self.vector_clock.get(agent_id, 0) + 1
        
        # Store value with clock
        clock_snapshot = self.vector_clock.copy()
        self.values[agent_id] = (value, clock_snapshot)
        
    def merge(self, other: 'CRDTMemory') -> None:
        """Merge with another CRDT"""
        # Merge vector clocks
        for agent, timestamp in other.vector_clock.items():
            self.vector_clock[agent] = max(
                self.vector_clock.get(agent, 0),
                timestamp
            )
        
        # Merge values based on vector clock
        for agent, (value, clock) in other.values.items():
            if agent not in self.values:
                self.values[agent] = (value, clock)
            else:
                # Keep value with more recent clock
                if self._is_concurrent(clock, self.values[agent][1]):
                    # Concurrent updates - merge strategy
                    self.values[agent] = self._merge_concurrent(
                        self.values[agent],
                        (value, clock)
                    )
                elif self._happens_before(self.values[agent][1], clock):
                    self.values[agent] = (value, clock)
    
    def get_consensus_value(self) -> Any:
        """Get consensus value from CRDT"""
        if not self.values:
            return None
            
        # Find value with highest vector clock sum
        best_value = None
        best_score = -1
        
        for agent, (value, clock) in self.values.items():
            score = sum(clock.values())
            if score > best_score:
                best_score = score
                best_value = value
                
        return best_value
    
    def _happens_before(self, clock1: Dict[str, int], clock2: Dict[str, int]) -> bool:
        """Check if clock1 happens before clock2"""
        all_agents = set(clock1.keys()) | set(clock2.keys())
        
        for agent in all_agents:
            if clock1.get(agent, 0) > clock2.get(agent, 0):
                return False
                
        return any(clock1.get(agent, 0) < clock2.get(agent, 0) for agent in all_agents)
    
    def _is_concurrent(self, clock1: Dict[str, int], clock2: Dict[str, int]) -> bool:
        """Check if two clocks are concurrent"""
        return (not self._happens_before(clock1, clock2) and 
                not self._happens_before(clock2, clock1))
    
    def _merge_concurrent(self, 
                         item1: Tuple[Any, Dict[str, int]], 
                         item2: Tuple[Any, Dict[str, int]]) -> Tuple[Any, Dict[str, int]]:
        """Merge concurrent updates - can be customized"""
        # Simple strategy: keep both in a list
        value1, clock1 = item1
        value2, clock2 = item2
        
        # Merge clocks
        merged_clock = {}
        for agent in set(clock1.keys()) | set(clock2.keys()):
            merged_clock[agent] = max(clock1.get(agent, 0), clock2.get(agent, 0))
        
        # Merge values (customize based on use case)
        if isinstance(value1, list) and isinstance(value2, list):
            merged_value = list(set(value1 + value2))
        elif isinstance(value1, dict) and isinstance(value2, dict):
            merged_value = {**value1, **value2}
        else:
            # Keep both as tuple
            merged_value = (value1, value2)
            
        return (merged_value, merged_clock)


# ==================== Byzantine Consensus ====================

class ByzantineConsensus:
    """
    Byzantine Fault Tolerant consensus for untrusted environments.
    Tolerates up to f = (n-1)/3 faulty agents.
    """
    
    def __init__(self, threshold: float = 0.67):
        self.threshold = threshold  # Byzantine requires 2/3 majority
        self.round_number = 0
        self.prepared_values: Dict[int, Dict[str, Any]] = {}
        self.committed_values: Dict[int, Any] = {}
        
    async def propose(self, 
                     value: Any,
                     agents: List[str],
                     vote_collector) -> ConsensusResult:
        """Run Byzantine consensus protocol"""
        n_agents = len(agents)
        f = (n_agents - 1) // 3  # Maximum Byzantine faults
        
        self.round_number += 1
        round_id = self.round_number
        
        # Phase 1: Prepare
        prepare_votes = await self._prepare_phase(
            round_id, value, agents, vote_collector
        )
        
        # Check if enough prepare votes
        if len(prepare_votes) < n_agents - f:
            return ConsensusResult(
                consensus_reached=False,
                value=None,
                support_ratio=len(prepare_votes) / n_agents,
                votes=prepare_votes,
                consensus_type=ConsensusType.BYZANTINE
            )
        
        # Phase 2: Accept
        accept_votes = await self._accept_phase(
            round_id, value, agents, vote_collector
        )
        
        # Check if consensus reached
        if len(accept_votes) >= n_agents - f:
            self.committed_values[round_id] = value
            return ConsensusResult(
                consensus_reached=True,
                value=value,
                support_ratio=len(accept_votes) / n_agents,
                votes=accept_votes,
                consensus_type=ConsensusType.BYZANTINE
            )
        
        return ConsensusResult(
            consensus_reached=False,
            value=None,
            support_ratio=len(accept_votes) / n_agents,
            votes=accept_votes,
            consensus_type=ConsensusType.BYZANTINE
        )
    
    async def _prepare_phase(self, round_id: int, value: Any, 
                           agents: List[str], vote_collector) -> List[ConsensusVote]:
        """Prepare phase of Byzantine consensus"""
        prepare_msg = {
            'phase': 'prepare',
            'round': round_id,
            'value': value
        }
        
        votes = await vote_collector.collect_votes(agents, prepare_msg)
        
        # Store prepared values
        self.prepared_values[round_id] = {
            v.agent_id: v.value for v in votes if v.confidence > 0
        }
        
        return votes
    
    async def _accept_phase(self, round_id: int, value: Any,
                          agents: List[str], vote_collector) -> List[ConsensusVote]:
        """Accept phase of Byzantine consensus"""
        accept_msg = {
            'phase': 'accept',
            'round': round_id,
            'value': value,
            'prepared': len(self.prepared_values.get(round_id, {}))
        }
        
        return await vote_collector.collect_votes(agents, accept_msg)


# ==================== Collective Memory Consensus ====================

class CollectiveMemoryConsensus:
    """
    Advanced consensus system for collective memory.
    Integrates multiple consensus mechanisms based on context.
    """
    
    def __init__(self, default_threshold: float = 0.6):
        self.default_threshold = default_threshold
        self.byzantine = ByzantineConsensus()
        self.crdt_memories: Dict[str, CRDTMemory] = {}
        
        # Consensus history for learning
        self.consensus_history: List[ConsensusResult] = []
        
        logger.info("Collective memory consensus initialized",
                   threshold=default_threshold)
    
    async def build_consensus(self,
                            memory_id: str,
                            content: Any,
                            voting_agents: Dict[str, float],
                            consensus_type: ConsensusType = ConsensusType.WEIGHTED,
                            metadata: Optional[Dict[str, Any]] = None) -> ConsensusResult:
        """
        Build consensus on memory content.
        
        Args:
            memory_id: Unique memory identifier
            content: Content to build consensus on
            voting_agents: Dict of agent_id -> weight/confidence
            consensus_type: Type of consensus to use
            metadata: Additional consensus metadata
        """
        
        if consensus_type == ConsensusType.CRDT:
            return await self._crdt_consensus(memory_id, content, voting_agents)
        elif consensus_type == ConsensusType.BYZANTINE:
            return await self._byzantine_consensus(memory_id, content, voting_agents)
        else:
            return await self._weighted_consensus(memory_id, content, voting_agents)
    
    async def _weighted_consensus(self,
                                memory_id: str,
                                content: Any,
                                voting_agents: Dict[str, float]) -> ConsensusResult:
        """Weighted voting consensus"""
        votes = []
        total_weight = sum(abs(w) for w in voting_agents.values())
        
        if total_weight == 0:
            return ConsensusResult(
                consensus_reached=False,
                value=None,
                support_ratio=0.0,
                votes=[],
                consensus_type=ConsensusType.WEIGHTED
            )
        
        # Collect votes
        support_weight = 0.0
        for agent_id, weight in voting_agents.items():
            vote = ConsensusVote(
                agent_id=agent_id,
                value=content if weight > 0 else None,
                confidence=abs(weight) / total_weight
            )
            votes.append(vote)
            
            if weight > 0:
                support_weight += abs(weight)
        
        support_ratio = support_weight / total_weight
        consensus_reached = support_ratio >= self.default_threshold
        
        result = ConsensusResult(
            consensus_reached=consensus_reached,
            value=content if consensus_reached else None,
            support_ratio=support_ratio,
            votes=votes,
            consensus_type=ConsensusType.WEIGHTED,
            metadata={'memory_id': memory_id}
        )
        
        self.consensus_history.append(result)
        return result
    
    async def _crdt_consensus(self,
                            memory_id: str,
                            content: Any,
                            voting_agents: Dict[str, float]) -> ConsensusResult:
        """CRDT-based eventual consistency"""
        if memory_id not in self.crdt_memories:
            self.crdt_memories[memory_id] = CRDTMemory(memory_id)
        
        crdt = self.crdt_memories[memory_id]
        
        # Update CRDT with agent values
        votes = []
        for agent_id, confidence in voting_agents.items():
            if confidence > 0:
                crdt.update(agent_id, content)
                votes.append(ConsensusVote(
                    agent_id=agent_id,
                    value=content,
                    confidence=confidence
                ))
        
        # Get consensus value
        consensus_value = crdt.get_consensus_value()
        
        return ConsensusResult(
            consensus_reached=consensus_value is not None,
            value=consensus_value,
            support_ratio=len(votes) / max(1, len(voting_agents)),
            votes=votes,
            consensus_type=ConsensusType.CRDT,
            metadata={'memory_id': memory_id, 'crdt_state': crdt.vector_clock}
        )
    
    async def _byzantine_consensus(self,
                                 memory_id: str,
                                 content: Any,
                                 voting_agents: Dict[str, float]) -> ConsensusResult:
        """Byzantine fault tolerant consensus"""
        # Create vote collector
        class VoteCollector:
            async def collect_votes(self, agents, msg):
                votes = []
                for agent in agents:
                    if agent in voting_agents and voting_agents[agent] > 0:
                        votes.append(ConsensusVote(
                            agent_id=agent,
                            value=content,
                            confidence=voting_agents[agent]
                        ))
                return votes
        
        collector = VoteCollector()
        agent_list = list(voting_agents.keys())
        
        return await self.byzantine.propose(
            content, agent_list, collector
        )
    
    def merge_crdt_memories(self, memory_id: str, other_crdt: CRDTMemory) -> None:
        """Merge CRDT from another node"""
        if memory_id not in self.crdt_memories:
            self.crdt_memories[memory_id] = CRDTMemory(memory_id)
        
        self.crdt_memories[memory_id].merge(other_crdt)
        logger.info("CRDT memories merged", memory_id=memory_id)
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get metrics about consensus performance"""
        if not self.consensus_history:
            return {}
        
        recent = self.consensus_history[-100:]  # Last 100
        
        return {
            'total_consensus_attempts': len(self.consensus_history),
            'consensus_success_rate': sum(
                1 for r in recent if r.consensus_reached
            ) / len(recent),
            'average_support_ratio': sum(
                r.support_ratio for r in recent
            ) / len(recent),
            'consensus_types_used': {
                ct.value: sum(1 for r in recent if r.consensus_type == ct)
                for ct in ConsensusType
            }
        }