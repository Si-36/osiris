"""
🐜 Real Swarm Intelligence 2025 - Collective Failure Detection
==============================================================

Swarm Intelligence for AURA uses multiple autonomous agents that:
- Collectively explore the system state space
- Share information through digital pheromones
- Self-organize to detect complex failure patterns
- Emerge intelligent behavior from simple rules

"The wisdom of the swarm sees what individuals cannot"
"""

import asyncio
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import json
import structlog

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

class PheromoneType(str, Enum):
    """Types of digital pheromones."""
    ERROR_TRAIL = "error_trail"          # Marks error paths
    SUCCESS_PATH = "success_path"        # Marks successful routes
    DANGER_ZONE = "danger_zone"          # High-risk areas
    RESOURCE_RICH = "resource_rich"      # Good performance areas
    EXPLORATION = "exploration"          # Unexplored territories
    CONVERGENCE = "convergence"          # Where agents meet


@dataclass
class Pheromone:
    """Digital pheromone in the system."""
    pheromone_id: str
    pheromone_type: PheromoneType
    location: str  # Component/node ID
    strength: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    deposited_at: float = field(default_factory=time.time)
    deposited_by: str = ""  # Agent ID
    
    def decay(self, rate: float = 0.95):
        """Apply pheromone decay."""
        self.strength *= rate
        
    def is_expired(self, threshold: float = 0.01) -> bool:
        """Check if pheromone is too weak."""
        return self.strength < threshold


@dataclass
class SwarmAgent:
    """Individual agent in the swarm."""
    agent_id: str
    position: str  # Current component/node
    energy: float = 1.0  # Agent energy level
    memory: deque = field(default_factory=lambda: deque(maxlen=50))
    visited: Set[str] = field(default_factory=set)
    carrying: Optional[Dict[str, Any]] = None  # Information being carried
    role: str = "explorer"  # explorer, scout, worker, sentinel
    performance: float = 0.0
    
    def can_move(self) -> bool:
        """Check if agent has energy to move."""
        return self.energy > 0.1
        
    def consume_energy(self, amount: float = 0.05):
        """Consume energy for actions."""
        self.energy = max(0, self.energy - amount)
        
    def recharge(self, amount: float = 0.1):
        """Recharge energy."""
        self.energy = min(1.0, self.energy + amount)


@dataclass
class SwarmObservation:
    """Observation made by the swarm."""
    observation_id: str
    observation_type: str  # pattern, anomaly, convergence
    confidence: float  # 0-1
    location: str
    agents_involved: List[str]
    evidence: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


# ==================== Swarm Behaviors ====================

class SwarmBehavior:
    """Base class for swarm behaviors."""
    
    def __init__(self, name: str):
        self.name = name
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: Dict[str, List[Pheromone]]
    ) -> Dict[str, Any]:
        """Execute behavior and return action."""
        raise NotImplementedError


class ExplorationBehavior(SwarmBehavior):
    """Explore unknown areas of the system."""
    
    def __init__(self):
        super().__init__("exploration")
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: Dict[str, List[Pheromone]]
    ) -> Dict[str, Any]:
        """Explore with bias toward unexplored areas."""
        current_node = environment.get("nodes", {}).get(agent.position, {})
        neighbors = current_node.get("neighbors", [])
        
        if not neighbors:
            return {"action": "wait", "reason": "no_neighbors"}
            
        # Find unexplored neighbors
        unexplored = [n for n in neighbors if n not in agent.visited]
        
        if unexplored:
            # Move to unexplored
            next_pos = random.choice(unexplored)
            return {
                "action": "move",
                "target": next_pos,
                "deposit_pheromone": {
                    "type": PheromoneType.EXPLORATION,
                    "strength": 0.5
                }
            }
        else:
            # Follow exploration pheromones
            exploration_trails = pheromones.get(PheromoneType.EXPLORATION, [])
            if exploration_trails:
                # Go to strongest exploration trail
                strongest = max(exploration_trails, key=lambda p: p.strength)
                return {
                    "action": "move",
                    "target": strongest.location,
                    "reason": "following_exploration"
                }
            
            # Random walk
            return {
                "action": "move",
                "target": random.choice(neighbors),
                "reason": "random_exploration"
            }


class ForagingBehavior(SwarmBehavior):
    """Forage for errors and anomalies."""
    
    def __init__(self):
        super().__init__("foraging")
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: Dict[str, List[Pheromone]]
    ) -> Dict[str, Any]:
        """Follow error trails and collect information."""
        current_node = environment.get("nodes", {}).get(agent.position, {})
        
        # Check if current node has errors
        if current_node.get("error_rate", 0) > 0.5:
            # Collect error information
            error_info = {
                "node": agent.position,
                "error_rate": current_node["error_rate"],
                "patterns": current_node.get("patterns", [])
            }
            
            return {
                "action": "collect",
                "data": error_info,
                "deposit_pheromone": {
                    "type": PheromoneType.ERROR_TRAIL,
                    "strength": current_node["error_rate"],
                    "metadata": {"severity": "high"}
                }
            }
        
        # Follow error pheromones
        error_trails = pheromones.get(PheromoneType.ERROR_TRAIL, [])
        if error_trails:
            # Go to strongest error trail
            strongest = max(error_trails, key=lambda p: p.strength)
            
            # Avoid if already visited recently
            if strongest.location not in list(agent.memory)[-5:]:
                return {
                    "action": "move",
                    "target": strongest.location,
                    "reason": "following_error_trail"
                }
        
        # No errors found, explore
        return await ExplorationBehavior().execute(agent, environment, pheromones)


class RecruitmentBehavior(SwarmBehavior):
    """Recruit other agents to interesting locations."""
    
    def __init__(self):
        super().__init__("recruitment")
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: Dict[str, List[Pheromone]]
    ) -> Dict[str, Any]:
        """Deposit strong pheromones to recruit others."""
        # If carrying important information, recruit
        if agent.carrying and agent.carrying.get("importance", 0) > 0.7:
            return {
                "action": "recruit",
                "deposit_pheromone": {
                    "type": PheromoneType.CONVERGENCE,
                    "strength": 0.9,
                    "metadata": {
                        "reason": "important_finding",
                        "data_type": agent.carrying.get("type")
                    }
                }
            }
        
        # Check convergence pheromones
        convergence = pheromones.get(PheromoneType.CONVERGENCE, [])
        if convergence:
            strongest = max(convergence, key=lambda p: p.strength)
            
            # Move to convergence point
            if strongest.location != agent.position:
                return {
                    "action": "move",
                    "target": strongest.location,
                    "reason": "joining_convergence"
                }
        
        return {"action": "continue", "reason": "no_recruitment_needed"}


# ==================== Swarm Coordinator ====================

class SwarmCoordinator:
    """Coordinates the swarm agents and emergence of collective intelligence."""
    
    def __init__(
        self,
        num_agents: int = 50,
        pheromone_decay_rate: float = 0.95,
        convergence_threshold: int = 5
    ):
        self.num_agents = num_agents
        self.pheromone_decay_rate = pheromone_decay_rate
        self.convergence_threshold = convergence_threshold
        
        # Swarm agents
        self.agents: Dict[str, SwarmAgent] = {}
        self._initialize_agents()
        
        # Pheromone trails
        self.pheromone_map: Dict[str, List[Pheromone]] = defaultdict(list)
        
        # Observations
        self.observations: List[SwarmObservation] = []
        self.observation_history = deque(maxlen=1000)
        
        # Behaviors
        self.behaviors = {
            "exploration": ExplorationBehavior(),
            "foraging": ForagingBehavior(),
            "recruitment": RecruitmentBehavior()
        }
        
        # Statistics
        self.stats = {
            "total_moves": 0,
            "errors_found": 0,
            "patterns_detected": 0,
            "convergences": 0
        }
        
    def _initialize_agents(self):
        """Initialize swarm agents."""
        roles = ["explorer"] * (self.num_agents // 2) + \
                ["scout"] * (self.num_agents // 4) + \
                ["worker"] * (self.num_agents // 4)
        
        random.shuffle(roles)
        
        for i in range(self.num_agents):
            agent_id = f"ant_{i:03d}"
            self.agents[agent_id] = SwarmAgent(
                agent_id=agent_id,
                position="start",  # All start at same position
                role=roles[i] if i < len(roles) else "explorer"
            )
    
    async def swarm_explore(
        self,
        environment: Dict[str, Any],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Let the swarm explore the environment.
        
        Args:
            environment: System topology and state
            iterations: Number of swarm iterations
            
        Returns:
            Swarm findings and observations
        """
        logger.info(
            "Swarm exploration starting",
            num_agents=self.num_agents,
            iterations=iterations
        )
        
        for iteration in range(iterations):
            # Decay pheromones
            self._decay_pheromones()
            
            # Move all agents
            agent_actions = []
            
            for agent_id, agent in self.agents.items():
                if not agent.can_move():
                    agent.recharge(0.2)  # Rest to recharge
                    continue
                
                # Select behavior based on role and state
                behavior = self._select_behavior(agent)
                
                # Execute behavior
                action = await behavior.execute(
                    agent,
                    environment,
                    self._get_local_pheromones(agent.position)
                )
                
                agent_actions.append((agent, action))
            
            # Process actions
            await self._process_actions(agent_actions, environment)
            
            # Check for convergence patterns
            convergences = self._detect_convergence()
            if convergences:
                for conv in convergences:
                    await self._analyze_convergence(conv, environment)
            
            # Periodic analysis
            if iteration % 10 == 0:
                self._analyze_swarm_state()
        
        # Final analysis
        findings = self._compile_findings()
        
        logger.info(
            "Swarm exploration complete",
            errors_found=self.stats["errors_found"],
            patterns_detected=self.stats["patterns_detected"]
        )
        
        return findings
    
    def _select_behavior(self, agent: SwarmAgent) -> SwarmBehavior:
        """Select behavior based on agent role and state."""
        if agent.role == "explorer":
            return self.behaviors["exploration"]
        elif agent.role == "scout" and agent.carrying:
            return self.behaviors["recruitment"]
        else:
            return self.behaviors["foraging"]
    
    async def _process_actions(
        self,
        actions: List[Tuple[SwarmAgent, Dict[str, Any]]],
        environment: Dict[str, Any]
    ):
        """Process agent actions."""
        for agent, action in actions:
            action_type = action.get("action")
            
            if action_type == "move":
                # Move agent
                target = action.get("target")
                if target:
                    agent.position = target
                    agent.visited.add(target)
                    agent.memory.append(target)
                    agent.consume_energy(0.05)
                    self.stats["total_moves"] += 1
            
            elif action_type == "collect":
                # Collect information
                data = action.get("data")
                if data:
                    agent.carrying = data
                    agent.consume_energy(0.1)
                    
                    if data.get("error_rate", 0) > 0:
                        self.stats["errors_found"] += 1
            
            elif action_type == "recruit":
                # Recruiting other agents
                agent.consume_energy(0.15)
                self.stats["convergences"] += 1
            
            # Deposit pheromone if specified
            if "deposit_pheromone" in action:
                pheromone_info = action["deposit_pheromone"]
                self._deposit_pheromone(
                    agent.position,
                    pheromone_info["type"],
                    pheromone_info.get("strength", 0.5),
                    pheromone_info.get("metadata", {}),
                    agent.agent_id
                )
    
    def _deposit_pheromone(
        self,
        location: str,
        pheromone_type: PheromoneType,
        strength: float,
        metadata: Dict[str, Any],
        agent_id: str
    ):
        """Deposit pheromone at location."""
        pheromone = Pheromone(
            pheromone_id=f"ph_{int(time.time()*1000)}_{agent_id}",
            pheromone_type=pheromone_type,
            location=location,
            strength=min(1.0, strength),
            metadata=metadata,
            deposited_by=agent_id
        )
        
        self.pheromone_map[location].append(pheromone)
    
    def _get_local_pheromones(self, location: str) -> Dict[PheromoneType, List[Pheromone]]:
        """Get pheromones at a location grouped by type."""
        local_pheromones = defaultdict(list)
        
        for pheromone in self.pheromone_map.get(location, []):
            if not pheromone.is_expired():
                local_pheromones[pheromone.pheromone_type].append(pheromone)
        
        return dict(local_pheromones)
    
    def _decay_pheromones(self):
        """Apply decay to all pheromones."""
        for location, pheromones in list(self.pheromone_map.items()):
            active_pheromones = []
            
            for pheromone in pheromones:
                pheromone.decay(self.pheromone_decay_rate)
                
                if not pheromone.is_expired():
                    active_pheromones.append(pheromone)
            
            if active_pheromones:
                self.pheromone_map[location] = active_pheromones
            else:
                del self.pheromone_map[location]
    
    def _detect_convergence(self) -> List[Dict[str, Any]]:
        """Detect when multiple agents converge on a location."""
        location_counts = defaultdict(list)
        
        for agent_id, agent in self.agents.items():
            location_counts[agent.position].append(agent_id)
        
        convergences = []
        for location, agents in location_counts.items():
            if len(agents) >= self.convergence_threshold:
                convergences.append({
                    "location": location,
                    "agents": agents,
                    "count": len(agents)
                })
        
        return convergences
    
    async def _analyze_convergence(
        self,
        convergence: Dict[str, Any],
        environment: Dict[str, Any]
    ):
        """Analyze why agents converged."""
        location = convergence["location"]
        agents = convergence["agents"]
        
        # Check what attracted agents
        local_pheromones = self._get_local_pheromones(location)
        node_data = environment.get("nodes", {}).get(location, {})
        
        # Determine convergence reason
        if PheromoneType.ERROR_TRAIL in local_pheromones:
            observation_type = "error_convergence"
            confidence = 0.8
        elif PheromoneType.CONVERGENCE in local_pheromones:
            observation_type = "recruitment_success"
            confidence = 0.9
        else:
            observation_type = "spontaneous_convergence"
            confidence = 0.6
        
        # Create observation
        observation = SwarmObservation(
            observation_id=f"obs_{int(time.time()*1000)}",
            observation_type=observation_type,
            confidence=confidence,
            location=location,
            agents_involved=agents,
            evidence={
                "node_data": node_data,
                "pheromone_types": list(local_pheromones.keys()),
                "agent_count": len(agents)
            }
        )
        
        self.observations.append(observation)
        self.observation_history.append(observation)
        
        # Pattern detection
        if observation_type == "error_convergence":
            self.stats["patterns_detected"] += 1
            
            logger.info(
                "Swarm detected error pattern",
                location=location,
                agents=len(agents),
                error_rate=node_data.get("error_rate", 0)
            )
    
    def _analyze_swarm_state(self):
        """Analyze overall swarm state."""
        # Agent distribution
        positions = [agent.position for agent in self.agents.values()]
        unique_positions = len(set(positions))
        
        # Pheromone distribution
        total_pheromones = sum(len(p) for p in self.pheromone_map.values())
        
        # Energy levels
        avg_energy = np.mean([agent.energy for agent in self.agents.values()])
        
        logger.debug(
            "Swarm state",
            coverage=unique_positions,
            pheromones=total_pheromones,
            avg_energy=avg_energy
        )
    
    def _compile_findings(self) -> Dict[str, Any]:
        """Compile swarm findings into report."""
        # Group observations by type
        obs_by_type = defaultdict(list)
        for obs in self.observations:
            obs_by_type[obs.observation_type].append(obs)
        
        # Find high-confidence patterns
        high_confidence = [
            obs for obs in self.observations
            if obs.confidence > 0.7
        ]
        
        # Error hotspots
        error_locations = set()
        for location, pheromones in self.pheromone_map.items():
            error_pheromones = [
                p for p in pheromones
                if p.pheromone_type == PheromoneType.ERROR_TRAIL
            ]
            if error_pheromones:
                avg_strength = np.mean([p.strength for p in error_pheromones])
                if avg_strength > 0.5:
                    error_locations.add(location)
        
        return {
            "summary": {
                "total_observations": len(self.observations),
                "high_confidence_patterns": len(high_confidence),
                "error_hotspots": list(error_locations),
                "convergence_events": self.stats["convergences"]
            },
            "observations": [
                {
                    "type": obs.observation_type,
                    "location": obs.location,
                    "confidence": obs.confidence,
                    "agent_count": len(obs.agents_involved)
                }
                for obs in high_confidence
            ],
            "statistics": self.stats,
            "swarm_coverage": len(set(agent.position for agent in self.agents.values())),
            "recommendations": self._generate_recommendations(obs_by_type, error_locations)
        }
    
    def _generate_recommendations(
        self,
        obs_by_type: Dict[str, List[SwarmObservation]],
        error_locations: Set[str]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on swarm findings."""
        recommendations = []
        
        # Error convergence recommendations
        if "error_convergence" in obs_by_type:
            error_obs = obs_by_type["error_convergence"]
            if len(error_obs) > 3:
                recommendations.append({
                    "action": "investigate_error_pattern",
                    "priority": "high",
                    "locations": [obs.location for obs in error_obs[:5]],
                    "reason": f"Swarm detected {len(error_obs)} error convergence events"
                })
        
        # Hotspot recommendations
        if len(error_locations) > 0:
            recommendations.append({
                "action": "monitor_error_hotspots",
                "priority": "medium",
                "locations": list(error_locations)[:10],
                "reason": f"Persistent error pheromones at {len(error_locations)} locations"
            })
        
        # Coverage recommendations
        coverage = len(set(agent.position for agent in self.agents.values()))
        if coverage < 10:
            recommendations.append({
                "action": "expand_exploration",
                "priority": "low",
                "reason": f"Limited swarm coverage: only {coverage} unique locations explored"
            })
        
        return recommendations


# ==================== Main Swarm Intelligence System ====================

class RealSwarmIntelligence:
    """
    Real Swarm Intelligence System for AURA.
    
    Features:
    - Autonomous agent exploration
    - Digital pheromone communication
    - Emergent pattern detection
    - Collective decision making
    - Self-organizing failure detection
    """
    
    def __init__(
        self,
        num_agents: int = 50,
        enable_visualization: bool = False
    ):
        self.num_agents = num_agents
        self.enable_visualization = enable_visualization
        
        # Swarm coordinator
        self.coordinator = SwarmCoordinator(num_agents=num_agents)
        
        # Pattern memory
        self.pattern_memory = deque(maxlen=100)
        self.known_patterns = {}
        
        # Statistics
        self.stats = {
            "explorations_run": 0,
            "patterns_discovered": 0,
            "errors_detected": 0,
            "avg_convergence_time": 0
        }
        
        logger.info(
            "Real Swarm Intelligence initialized",
            num_agents=num_agents
        )
    
    async def explore_system(
        self,
        system_state: Dict[str, Any],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Deploy swarm to explore system state.
        
        Args:
            system_state: Current system topology and state
            iterations: Number of swarm iterations
            
        Returns:
            Swarm findings and recommendations
        """
        start_time = time.time()
        
        # Convert system state to swarm environment
        environment = self._create_environment(system_state)
        
        # Deploy swarm
        findings = await self.coordinator.swarm_explore(environment, iterations)
        
        # Learn patterns
        self._learn_patterns(findings)
        
        # Update statistics
        self.stats["explorations_run"] += 1
        self.stats["errors_detected"] += findings["statistics"]["errors_found"]
        
        duration = time.time() - start_time
        
        return {
            "findings": findings,
            "duration_seconds": duration,
            "swarm_efficiency": self._calculate_efficiency(findings, duration),
            "learned_patterns": len(self.known_patterns),
            "immediate_risks": self._identify_immediate_risks(findings)
        }
    
    def _create_environment(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert system state to swarm environment."""
        nodes = {}
        
        # Create nodes from agents/components
        for agent_id, state in system_state.items():
            nodes[agent_id] = {
                "error_rate": state.get("error_rate", 0),
                "latency": state.get("latency_ms", 0),
                "cpu_usage": state.get("cpu_usage", 0),
                "patterns": state.get("patterns", []),
                "neighbors": self._get_neighbors(agent_id, system_state)
            }
        
        # Add start node
        all_nodes = list(nodes.keys())
        nodes["start"] = {
            "error_rate": 0,
            "neighbors": all_nodes[:5] if all_nodes else []
        }
        
        return {
            "nodes": nodes,
            "edges": self._create_edges(nodes)
        }
    
    def _get_neighbors(self, agent_id: str, system_state: Dict[str, Any]) -> List[str]:
        """Get neighboring nodes (simplified topology)."""
        all_agents = list(system_state.keys())
        
        # Simple strategy: connect to nearby agents by ID
        agent_num = int(agent_id.split("_")[-1]) if "_" in agent_id else 0
        neighbors = []
        
        for other in all_agents:
            if other != agent_id:
                other_num = int(other.split("_")[-1]) if "_" in other else 0
                if abs(agent_num - other_num) <= 2:  # Close by ID
                    neighbors.append(other)
        
        return neighbors[:5]  # Limit connections
    
    def _create_edges(self, nodes: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Create edges from node neighbors."""
        edges = []
        
        for node_id, node_data in nodes.items():
            for neighbor in node_data.get("neighbors", []):
                if neighbor in nodes:
                    edges.append((node_id, neighbor))
        
        return edges
    
    def _learn_patterns(self, findings: Dict[str, Any]):
        """Learn from swarm findings."""
        observations = findings.get("observations", [])
        
        for obs in observations:
            if obs["confidence"] > 0.7:
                # Create pattern signature
                pattern_sig = f"{obs['type']}_{obs['location']}"
                
                if pattern_sig not in self.known_patterns:
                    self.known_patterns[pattern_sig] = {
                        "type": obs["type"],
                        "occurrences": 1,
                        "avg_confidence": obs["confidence"]
                    }
                    self.stats["patterns_discovered"] += 1
                else:
                    # Update pattern
                    pattern = self.known_patterns[pattern_sig]
                    pattern["occurrences"] += 1
                    pattern["avg_confidence"] = (
                        pattern["avg_confidence"] * (pattern["occurrences"] - 1) + 
                        obs["confidence"]
                    ) / pattern["occurrences"]
        
        # Store in pattern memory
        self.pattern_memory.append({
            "timestamp": time.time(),
            "patterns": list(self.known_patterns.keys()),
            "findings": findings["summary"]
        })
    
    def _calculate_efficiency(self, findings: Dict[str, Any], duration: float) -> float:
        """Calculate swarm efficiency score."""
        # Factors: coverage, findings, speed
        coverage_score = min(1.0, findings["swarm_coverage"] / 20)
        findings_score = min(1.0, len(findings["observations"]) / 10)
        speed_score = max(0, 1.0 - duration / 60)  # Faster is better
        
        return (coverage_score + findings_score + speed_score) / 3
    
    def _identify_immediate_risks(self, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify immediate risks from findings."""
        risks = []
        
        # High-confidence error convergences
        for obs in findings.get("observations", []):
            if obs["type"] == "error_convergence" and obs["confidence"] > 0.8:
                risks.append({
                    "type": "error_cluster",
                    "location": obs["location"],
                    "severity": obs["confidence"],
                    "agent_consensus": obs["agent_count"]
                })
        
        # Error hotspots
        hotspots = findings["summary"].get("error_hotspots", [])
        for hotspot in hotspots[:3]:  # Top 3
            risks.append({
                "type": "persistent_error",
                "location": hotspot,
                "severity": 0.7,
                "agent_consensus": "pheromone_based"
            })
        
        return risks
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return {
            "num_agents": self.num_agents,
            "active_agents": sum(1 for a in self.coordinator.agents.values() if a.energy > 0.1),
            "total_pheromones": sum(len(p) for p in self.coordinator.pheromone_map.values()),
            "known_patterns": len(self.known_patterns),
            "statistics": self.stats,
            "recent_observations": len(self.coordinator.observation_history)
        }
    
    async def guided_exploration(
        self,
        system_state: Dict[str, Any],
        target_areas: List[str],
        iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Perform guided exploration of specific areas.
        
        Args:
            system_state: Current system state
            target_areas: Specific nodes/components to focus on
            iterations: Number of iterations
            
        Returns:
            Targeted findings
        """
        # Bias initial agent positions toward targets
        for i, agent in enumerate(self.coordinator.agents.values()):
            if i < len(target_areas):
                agent.position = target_areas[i]
            
        # Add attraction pheromones to targets
        for target in target_areas:
            self.coordinator._deposit_pheromone(
                target,
                PheromoneType.CONVERGENCE,
                0.8,
                {"reason": "targeted_exploration"},
                "system"
            )
        
        # Run exploration
        environment = self._create_environment(system_state)
        findings = await self.coordinator.swarm_explore(environment, iterations)
        
        return {
            "findings": findings,
            "target_coverage": self._calculate_target_coverage(target_areas),
            "focused_observations": [
                obs for obs in findings.get("observations", [])
                if obs["location"] in target_areas
            ]
        }
    
    def _calculate_target_coverage(self, targets: List[str]) -> float:
        """Calculate how well swarm covered target areas."""
        visited = set()
        
        for agent in self.coordinator.agents.values():
            visited.update(agent.visited)
        
        covered = sum(1 for t in targets if t in visited)
        
        return covered / len(targets) if targets else 0.0