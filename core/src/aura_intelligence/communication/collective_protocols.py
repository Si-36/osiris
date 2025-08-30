"""
👥 Collective Communication Protocols
=====================================

Higher-level patterns for multi-agent coordination, consensus,
and emergent behaviors. Builds on semantic protocols for
collective intelligence.

Features:
- Swarm synchronization
- Pattern discovery
- Collective learning
- Emergent behavior detection
- Consciousness updates
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import structlog

from .unified_communication import UnifiedCommunication, SemanticEnvelope, Performative
from .semantic_protocols import InteractionProtocol, ConversationManager
from ..swarm_intelligence import SwarmCoordinator, SwarmAlgorithm

logger = structlog.get_logger(__name__)


# ==================== Collective Patterns ====================

@dataclass
class CollectivePattern:
    """Detected pattern in collective behavior"""
    pattern_id: str
    pattern_type: str
    participants: Set[str]
    confidence: float
    emergence_time: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def involves_agent(self, agent_id: str) -> bool:
        return agent_id in self.participants


@dataclass
class SwarmState:
    """Current state of agent swarm"""
    swarm_id: str
    members: Set[str]
    topology: str  # mesh, star, ring, hierarchical
    sync_level: float  # 0.0 to 1.0
    collective_goal: Optional[Dict[str, Any]] = None
    pheromones: Dict[str, float] = field(default_factory=dict)
    last_sync: datetime = field(default_factory=datetime.utcnow)


# ==================== Collective Protocols Manager ====================

class CollectiveProtocolsManager:
    """
    Manages collective communication patterns and emergent behaviors.
    
    Integrates with Neural Mesh and Swarm Intelligence.
    """
    
    def __init__(
        self,
        comm: UnifiedCommunication,
        swarm_coordinator: Optional[SwarmCoordinator] = None
    ):
        self.comm = comm
        self.swarm = swarm_coordinator
        
        # Collective state
        self.swarms: Dict[str, SwarmState] = {}
        self.patterns: Dict[str, CollectivePattern] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Pattern detection
        self.pattern_detectors: List[PatternDetector] = [
            FormationPatternDetector(),
            ConsensusPatternDetector(),
            CascadePatternDetector(),
            SynchronizationPatternDetector()
        ]
        
        # Metrics
        self.metrics = {
            "patterns_detected": 0,
            "swarms_active": 0,
            "sync_operations": 0,
            "collective_decisions": 0
        }
        
        # Register handlers
        self._register_handlers()
        
        logger.info("Collective protocols manager initialized")
    
    def _register_handlers(self):
        """Register collective protocol handlers"""
        # Swarm sync
        self.comm.register_handler(
            Performative.INFORM,
            self._handle_swarm_sync,
            protocol="aura-swarm-sync"
        )
        
        # Pattern discovery
        self.comm.register_handler(
            Performative.INFORM,
            self._handle_pattern_report,
            protocol="aura-pattern-discovery"
        )
        
        # Collective learning
        self.comm.register_handler(
            Performative.PROPOSE,
            self._handle_learning_proposal,
            protocol="aura-collective-learn"
        )
    
    # ==================== Swarm Management ====================
    
    async def create_swarm(
        self,
        swarm_id: str,
        initial_members: List[str],
        topology: str = "mesh",
        goal: Optional[Dict[str, Any]] = None
    ) -> SwarmState:
        """Create a new swarm"""
        swarm = SwarmState(
            swarm_id=swarm_id,
            members=set(initial_members),
            topology=topology,
            collective_goal=goal,
            sync_level=0.0
        )
        
        self.swarms[swarm_id] = swarm
        self.metrics["swarms_active"] += 1
        
        # Notify members
        for member in initial_members:
            await self.comm.send(
                SemanticEnvelope(
                    performative=Performative.INFORM,
                    sender=self.comm.agent_id,
                    receiver=member,
                    content={
                        "action": "join_swarm",
                        "swarm_id": swarm_id,
                        "topology": topology,
                        "goal": goal
                    },
                    protocol="aura-swarm-sync"
                )
            )
        
        logger.info(
            "Swarm created",
            swarm_id=swarm_id,
            members=len(initial_members),
            topology=topology
        )
        
        return swarm
    
    async def synchronize_swarm(
        self,
        swarm_id: str,
        sync_data: Dict[str, Any]
    ) -> float:
        """
        Synchronize swarm members.
        
        Returns synchronization level (0.0 to 1.0).
        """
        if swarm_id not in self.swarms:
            return 0.0
        
        swarm = self.swarms[swarm_id]
        
        # Broadcast sync data
        await self.comm.broadcast(
            content={
                "swarm_id": swarm_id,
                "sync_data": sync_data,
                "timestamp": datetime.utcnow().isoformat()
            },
            performative=Performative.INFORM,
            topic=f"swarm.{swarm_id}",
            protocol="aura-swarm-sync"
        )
        
        # If swarm coordinator available, use it for optimization
        if self.swarm:
            # Define synchronization as optimization problem
            async def sync_objective(params):
                # Measure sync level based on agent responses
                responses = 0
                for member in swarm.members:
                    # In real implementation, check actual sync
                    responses += 1
                return responses / len(swarm.members)
            
            # Optimize synchronization parameters
            result = await self.swarm.optimize_parameters(
                algorithm=SwarmAlgorithm.PSO,
                objective_function=sync_objective,
                search_space=sync_data
            )
            
            swarm.sync_level = result.best_fitness
        else:
            # Simple sync level calculation
            swarm.sync_level = min(swarm.sync_level + 0.1, 1.0)
        
        swarm.last_sync = datetime.utcnow()
        self.metrics["sync_operations"] += 1
        
        return swarm.sync_level
    
    async def update_pheromone(
        self,
        swarm_id: str,
        pheromone_type: str,
        value: float,
        decay_rate: float = 0.1
    ):
        """Update swarm pheromone levels"""
        if swarm_id not in self.swarms:
            return
        
        swarm = self.swarms[swarm_id]
        
        # Apply decay to all pheromones
        for p_type in swarm.pheromones:
            swarm.pheromones[p_type] *= (1 - decay_rate)
        
        # Update specific pheromone
        swarm.pheromones[pheromone_type] = value
        
        # Broadcast pheromone update
        await self.comm.broadcast(
            content={
                "swarm_id": swarm_id,
                "pheromone_type": pheromone_type,
                "value": value,
                "decay_rate": decay_rate
            },
            performative=Performative.INFORM,
            topic=f"swarm.{swarm_id}.pheromones",
            protocol="aura-swarm-sync"
        )
    
    # ==================== Pattern Detection ====================
    
    async def detect_patterns(self) -> List[CollectivePattern]:
        """Run pattern detection across all agent interactions"""
        detected_patterns = []
        
        # Get recent agent states
        recent_states = self._get_recent_agent_states()
        
        # Run each detector
        for detector in self.pattern_detectors:
            patterns = detector.detect(recent_states, self.agent_states)
            detected_patterns.extend(patterns)
        
        # Store new patterns
        for pattern in detected_patterns:
            if pattern.pattern_id not in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
                self.metrics["patterns_detected"] += 1
                
                # Notify involved agents
                for agent in pattern.participants:
                    await self.comm.send(
                        SemanticEnvelope(
                            performative=Performative.INFORM,
                            sender=self.comm.agent_id,
                            receiver=agent,
                            content={
                                "pattern_detected": pattern.pattern_type,
                                "pattern_id": pattern.pattern_id,
                                "confidence": pattern.confidence,
                                "properties": pattern.properties
                            },
                            protocol="aura-pattern-discovery"
                        )
                    )
        
        return detected_patterns
    
    def _get_recent_agent_states(self) -> List[Dict[str, Any]]:
        """Get recent agent states from conversations"""
        recent_states = []
        
        # Get from communication history
        for conv_id, messages in self.comm.conversations.items():
            for msg in messages[-10:]:  # Last 10 messages
                recent_states.append({
                    "agent_id": msg.sender,
                    "timestamp": msg.timestamp,
                    "performative": msg.performative.value,
                    "content": msg.content
                })
        
        return recent_states
    
    # ==================== Collective Learning ====================
    
    async def propose_collective_learning(
        self,
        learning_task: Dict[str, Any],
        participants: List[str],
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Propose collective learning task.
        
        Agents collaborate to learn from shared experiences.
        """
        conversation_id = f"learn_{int(datetime.utcnow().timestamp())}"
        
        # Broadcast learning proposal
        await self.comm.broadcast(
            content={
                "task": learning_task,
                "participants": participants,
                "deadline": (datetime.utcnow() + timedelta(seconds=timeout)).isoformat()
            },
            performative=Performative.PROPOSE,
            topic="collective.learning",
            protocol="aura-collective-learn",
            conversation_id=conversation_id
        )
        
        # Collect responses
        responses = {}
        deadline = datetime.utcnow() + timedelta(seconds=timeout)
        
        # In real implementation, would collect actual responses
        # For now, simulate
        await asyncio.sleep(1)
        
        # Aggregate learning results
        aggregated = self._aggregate_learning_results(responses)
        
        # Broadcast aggregated knowledge
        await self.comm.broadcast(
            content={
                "task": learning_task,
                "aggregated_knowledge": aggregated,
                "contributors": list(responses.keys())
            },
            performative=Performative.INFORM,
            topic="collective.learning",
            protocol="aura-collective-learn",
            conversation_id=conversation_id
        )
        
        self.metrics["collective_decisions"] += 1
        
        return aggregated
    
    def _aggregate_learning_results(
        self,
        responses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate learning results from multiple agents"""
        if not responses:
            return {}
        
        # Simple aggregation - can be enhanced
        aggregated = {
            "consensus_features": [],
            "confidence_scores": {},
            "learned_patterns": []
        }
        
        # Find consensus
        all_features = []
        for agent_id, result in responses.items():
            if "features" in result:
                all_features.extend(result["features"])
        
        # Count occurrences
        feature_counts = defaultdict(int)
        for feature in all_features:
            feature_counts[str(feature)] += 1
        
        # Consensus features appear in >50% of responses
        threshold = len(responses) / 2
        aggregated["consensus_features"] = [
            feature for feature, count in feature_counts.items()
            if count > threshold
        ]
        
        return aggregated
    
    # ==================== Message Handlers ====================
    
    async def _handle_swarm_sync(self, envelope: SemanticEnvelope):
        """Handle swarm synchronization messages"""
        content = envelope.content
        swarm_id = content.get("swarm_id")
        
        if swarm_id and swarm_id in self.swarms:
            # Update agent state
            self.agent_states[envelope.sender]["last_sync"] = envelope.timestamp
            self.agent_states[envelope.sender]["swarm_id"] = swarm_id
            
            # Check if sync threshold reached
            swarm = self.swarms[swarm_id]
            synced_agents = sum(
                1 for agent_id in swarm.members
                if self.agent_states.get(agent_id, {}).get("last_sync", datetime.min) > swarm.last_sync
            )
            
            swarm.sync_level = synced_agents / len(swarm.members)
    
    async def _handle_pattern_report(self, envelope: SemanticEnvelope):
        """Handle pattern detection reports from agents"""
        content = envelope.content
        
        # Update agent state with pattern info
        self.agent_states[envelope.sender]["detected_patterns"] = content.get("patterns", [])
        self.agent_states[envelope.sender]["pattern_confidence"] = content.get("confidence", 0.0)
    
    async def _handle_learning_proposal(self, envelope: SemanticEnvelope):
        """Handle collective learning proposals"""
        content = envelope.content
        
        # Store learning intent
        self.agent_states[envelope.sender]["learning_task"] = content.get("task")
        self.agent_states[envelope.sender]["learning_ready"] = True
    
    # ==================== Consciousness Integration ====================
    
    async def update_collective_consciousness(
        self,
        consciousness_data: Dict[str, Any]
    ):
        """
        Update collective consciousness across all agents.
        
        Integrates with ConsciousnessAwareRouter from neural_mesh.
        """
        # Broadcast consciousness update
        await self.comm.broadcast(
            content={
                "consciousness_level": consciousness_data.get("level", 0.5),
                "attention_focus": consciousness_data.get("focus", {}),
                "emergent_properties": consciousness_data.get("emergent", []),
                "timestamp": datetime.utcnow().isoformat()
            },
            performative=Performative.INFORM,
            topic="collective.consciousness",
            protocol="aura-consciousness-update"
        )
        
        # Update all swarms
        for swarm_id, swarm in self.swarms.items():
            # Adjust swarm behavior based on consciousness
            if consciousness_data.get("level", 0) > 0.8:
                # High consciousness - more exploration
                swarm.pheromones["exploration"] = 0.9
            else:
                # Lower consciousness - more exploitation
                swarm.pheromones["exploitation"] = 0.9
    
    # ==================== Utility Methods ====================
    
    def get_swarm_info(self, swarm_id: str) -> Optional[SwarmState]:
        """Get information about a swarm"""
        return self.swarms.get(swarm_id)
    
    def get_agent_swarms(self, agent_id: str) -> List[str]:
        """Get all swarms an agent belongs to"""
        agent_swarms = []
        for swarm_id, swarm in self.swarms.items():
            if agent_id in swarm.members:
                agent_swarms.append(swarm_id)
        return agent_swarms
    
    def get_active_patterns(self) -> List[CollectivePattern]:
        """Get currently active patterns"""
        # Patterns active in last 5 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        return [
            p for p in self.patterns.values()
            if p.emergence_time > cutoff
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collective protocol metrics"""
        return {
            **self.metrics,
            "active_swarms": len(self.swarms),
            "active_patterns": len(self.get_active_patterns()),
            "total_agents": len(self.agent_states)
        }


# ==================== Pattern Detectors ====================

class PatternDetector:
    """Base class for collective pattern detection"""
    
    def detect(
        self,
        recent_states: List[Dict[str, Any]],
        agent_states: Dict[str, Dict[str, Any]]
    ) -> List[CollectivePattern]:
        raise NotImplementedError


class FormationPatternDetector(PatternDetector):
    """Detects formation patterns (line, circle, cluster)"""
    
    def detect(
        self,
        recent_states: List[Dict[str, Any]],
        agent_states: Dict[str, Dict[str, Any]]
    ) -> List[CollectivePattern]:
        patterns = []
        
        # Group agents by similar behavior
        behavior_groups = defaultdict(list)
        for state in recent_states:
            key = f"{state.get('performative')}_{state.get('content', {}).get('action', '')}"
            behavior_groups[key].append(state['agent_id'])
        
        # Check for formation patterns
        for behavior, agents in behavior_groups.items():
            if len(set(agents)) >= 3:  # At least 3 unique agents
                pattern = CollectivePattern(
                    pattern_id=f"formation_{int(datetime.utcnow().timestamp())}",
                    pattern_type="formation",
                    participants=set(agents),
                    confidence=len(set(agents)) / len(agent_states) if agent_states else 0,
                    emergence_time=datetime.utcnow(),
                    properties={"behavior": behavior}
                )
                patterns.append(pattern)
        
        return patterns


class ConsensusPatternDetector(PatternDetector):
    """Detects consensus formation patterns"""
    
    def detect(
        self,
        recent_states: List[Dict[str, Any]],
        agent_states: Dict[str, Dict[str, Any]]
    ) -> List[CollectivePattern]:
        patterns = []
        
        # Look for voting patterns
        votes = defaultdict(lambda: defaultdict(int))
        for state in recent_states:
            if state.get('performative') in ['accept-proposal', 'reject-proposal', 'agree', 'refuse']:
                conv_id = state.get('content', {}).get('conversation_id', 'unknown')
                choice = state['performative']
                votes[conv_id][choice] += 1
        
        # Check for consensus
        for conv_id, vote_counts in votes.items():
            total_votes = sum(vote_counts.values())
            if total_votes >= 3:
                max_votes = max(vote_counts.values())
                consensus_ratio = max_votes / total_votes
                
                if consensus_ratio > 0.6:  # 60% agreement
                    pattern = CollectivePattern(
                        pattern_id=f"consensus_{conv_id}",
                        pattern_type="consensus",
                        participants=set(),  # Would track actual voters
                        confidence=consensus_ratio,
                        emergence_time=datetime.utcnow(),
                        properties={
                            "conversation_id": conv_id,
                            "vote_distribution": dict(vote_counts)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns


class CascadePatternDetector(PatternDetector):
    """Detects cascade/avalanche patterns"""
    
    def detect(
        self,
        recent_states: List[Dict[str, Any]],
        agent_states: Dict[str, Dict[str, Any]]
    ) -> List[CollectivePattern]:
        patterns = []
        
        # Look for rapid sequential similar actions
        time_window = timedelta(seconds=10)
        action_sequences = defaultdict(list)
        
        for state in sorted(recent_states, key=lambda x: x['timestamp']):
            action = f"{state['performative']}_{state.get('content', {}).get('action', '')}"
            action_sequences[action].append({
                'agent': state['agent_id'],
                'time': state['timestamp']
            })
        
        # Check for cascades
        for action, sequence in action_sequences.items():
            if len(sequence) >= 3:
                # Check if actions happened in quick succession
                cascade = True
                for i in range(1, len(sequence)):
                    if sequence[i]['time'] - sequence[i-1]['time'] > time_window:
                        cascade = False
                        break
                
                if cascade:
                    pattern = CollectivePattern(
                        pattern_id=f"cascade_{action}_{int(datetime.utcnow().timestamp())}",
                        pattern_type="cascade",
                        participants={s['agent'] for s in sequence},
                        confidence=0.8,
                        emergence_time=sequence[0]['time'],
                        properties={
                            "action": action,
                            "cascade_length": len(sequence),
                            "duration": (sequence[-1]['time'] - sequence[0]['time']).total_seconds()
                        }
                    )
                    patterns.append(pattern)
        
        return patterns


class SynchronizationPatternDetector(PatternDetector):
    """Detects synchronization patterns"""
    
    def detect(
        self,
        recent_states: List[Dict[str, Any]],
        agent_states: Dict[str, Dict[str, Any]]
    ) -> List[CollectivePattern]:
        patterns = []
        
        # Look for agents acting simultaneously
        time_buckets = defaultdict(list)
        bucket_size = 5  # 5 second buckets
        
        for state in recent_states:
            bucket = int(state['timestamp'].timestamp() / bucket_size)
            time_buckets[bucket].append(state['agent_id'])
        
        # Check for synchronization
        for bucket, agents in time_buckets.items():
            unique_agents = set(agents)
            if len(unique_agents) >= 3:
                pattern = CollectivePattern(
                    pattern_id=f"sync_{bucket}",
                    pattern_type="synchronization",
                    participants=unique_agents,
                    confidence=len(unique_agents) / len(agent_states) if agent_states else 0,
                    emergence_time=datetime.fromtimestamp(bucket * bucket_size),
                    properties={
                        "sync_window": bucket_size,
                        "agent_count": len(unique_agents)
                    }
                )
                patterns.append(pattern)
        
        return patterns