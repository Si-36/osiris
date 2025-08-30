"""
🐜 AURA Swarm Intelligence Coordinator
=====================================

Consolidates the best swarm algorithms for multi-agent coordination.

Features:
- Digital pheromone system for stigmergic communication
- Multiple swarm algorithms (PSO, ACO, Bee, Firefly)
- Neural swarm control with attention
- Collective failure detection
- Self-organizing behaviors
- Energy-based realistic agents

Based on research from:
- Particle Swarm Optimization (Kennedy & Eberhart, 1995)
- Ant Colony Optimization (Dorigo, 1992)
- Artificial Bee Colony (Karaboga, 2005)
- Neural Swarm Control (2025 research)
"""

import asyncio
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import json
import structlog
import math
import networkx as nx

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

class PheromoneType(str, Enum):
    """Types of digital pheromones for communication"""
    ERROR_TRAIL = "error_trail"          # Marks error paths
    SUCCESS_PATH = "success_path"        # Marks successful routes
    DANGER_ZONE = "danger_zone"          # High-risk areas
    RESOURCE_RICH = "resource_rich"      # Good performance areas
    EXPLORATION = "exploration"          # Unexplored territories
    CONVERGENCE = "convergence"          # Where agents meet


class SwarmAlgorithm(str, Enum):
    """Available swarm algorithms"""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_ALGORITHM = "bee_algorithm"
    FIREFLY = "firefly"
    WOLF_PACK = "wolf_pack"
    FISH_SCHOOL = "fish_school"
    HYBRID = "hybrid"


class AgentRole(str, Enum):
    """Dynamic agent roles"""
    EXPLORER = "explorer"
    FORAGER = "forager"
    SCOUT = "scout"
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SENTINEL = "sentinel"


@dataclass
class Pheromone:
    """Digital pheromone in the system"""
    pheromone_id: str
    pheromone_type: PheromoneType
    location: str  # Component/node ID
    strength: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    deposited_at: float = field(default_factory=time.time)
    deposited_by: str = ""  # Agent ID
    
    def decay(self, rate: float = 0.95):
        """Decay pheromone strength over time"""
        self.strength *= rate
    
    def is_expired(self, threshold: float = 0.01) -> bool:
        """Check if pheromone is too weak"""
        return self.strength < threshold


@dataclass
class SwarmAgent:
    """Individual agent in the swarm"""
    agent_id: str
    position: np.ndarray  # Current position in search space
    velocity: np.ndarray  # For PSO
    best_position: np.ndarray  # Personal best
    role: AgentRole = AgentRole.WORKER
    energy: float = 1.0
    memory: List[Any] = field(default_factory=list)
    
    def can_move(self) -> bool:
        """Check if agent has energy to move"""
        return self.energy > 0.1
    
    def consume_energy(self, amount: float = 0.05):
        """Consume energy for action"""
        self.energy = max(0, self.energy - amount)
    
    def recharge(self, amount: float = 0.1):
        """Recharge agent energy"""
        self.energy = min(1.0, self.energy + amount)


# ==================== Digital Pheromone System ====================

class DigitalPheromoneSystem:
    """
    Manages digital pheromones for stigmergic communication.
    
    Unlike traditional pheromones, these are typed and carry metadata.
    """
    
    def __init__(self, decay_rate: float = 0.97):
        self.pheromones: Dict[str, List[Pheromone]] = defaultdict(list)
        self.decay_rate = decay_rate
        self.pheromone_limit = 1000  # Max pheromones per location
        
    def deposit(
        self,
        location: str,
        pheromone_type: PheromoneType,
        strength: float,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Pheromone:
        """Deposit pheromone at location"""
        pheromone = Pheromone(
            pheromone_id=f"{location}_{pheromone_type.value}_{time.time()}",
            pheromone_type=pheromone_type,
            location=location,
            strength=min(1.0, strength),
            metadata=metadata or {},
            deposited_by=agent_id
        )
        
        self.pheromones[location].append(pheromone)
        
        # Limit pheromones per location
        if len(self.pheromones[location]) > self.pheromone_limit:
            # Remove weakest
            self.pheromones[location].sort(key=lambda p: p.strength)
            self.pheromones[location] = self.pheromones[location][-self.pheromone_limit:]
        
        logger.debug(
            f"Pheromone deposited",
            location=location,
            type=pheromone_type.value,
            strength=strength
        )
        
        return pheromone
    
    def sense(
        self,
        location: str,
        pheromone_type: Optional[PheromoneType] = None,
        radius: int = 1
    ) -> List[Pheromone]:
        """Sense pheromones at or near location"""
        sensed = []
        
        # Get pheromones at exact location
        local_pheromones = self.pheromones.get(location, [])
        
        for pheromone in local_pheromones:
            if pheromone_type is None or pheromone.pheromone_type == pheromone_type:
                if not pheromone.is_expired():
                    sensed.append(pheromone)
        
        # TODO: Add radius-based sensing for nearby locations
        
        return sensed
    
    def decay_all(self):
        """Decay all pheromones"""
        expired_locations = []
        
        for location, pheromone_list in self.pheromones.items():
            # Decay each pheromone
            for pheromone in pheromone_list:
                pheromone.decay(self.decay_rate)
            
            # Remove expired
            pheromone_list[:] = [p for p in pheromone_list if not p.is_expired()]
            
            if not pheromone_list:
                expired_locations.append(location)
        
        # Clean up empty locations
        for location in expired_locations:
            del self.pheromones[location]
    
    def get_pheromone_map(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated pheromone strengths by type"""
        pheromone_map = defaultdict(lambda: defaultdict(float))
        
        for location, pheromone_list in self.pheromones.items():
            for pheromone in pheromone_list:
                pheromone_map[location][pheromone.pheromone_type.value] += pheromone.strength
        
        return dict(pheromone_map)


# ==================== Neural Swarm Controller ====================

class NeuralSwarmController(nn.Module):
    """
    Neural network for intelligent swarm control.
    
    Uses attention mechanisms to coordinate agents.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output heads
        self.role_head = nn.Linear(hidden_dim, len(AgentRole))
        self.action_head = nn.Linear(hidden_dim, 64)  # Action embedding
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, agent_states: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass through controller.
        
        Args:
            agent_states: [batch, num_agents, features]
            mask: Optional attention mask
            
        Returns:
            roles: [batch, num_agents, num_roles]
            actions: [batch, num_agents, action_dim]
            values: [batch, num_agents, 1]
        """
        x = self.input_projection(agent_states)
        
        # Apply attention layers
        for i in range(len(self.attention_layers)):
            # Self-attention
            attn_out, _ = self.attention_layers[i](x, x, x, attn_mask=mask)
            x = self.layer_norms1[i](x + attn_out)
            
            # Feed-forward
            ffn_out = self.ffn_layers[i](x)
            x = self.layer_norms2[i](x + ffn_out)
        
        # Generate outputs
        roles = self.role_head(x)
        actions = self.action_head(x)
        values = self.value_head(x)
        
        return roles, actions, values


# ==================== Swarm Algorithms ====================

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for continuous optimization.
    
    Each particle represents a potential solution.
    """
    
    def __init__(
        self,
        num_particles: int = 50,
        dimensions: int = 10,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive parameter
        c2: float = 1.5   # Social parameter
    ):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize particles
        self.particles: List[SwarmAgent] = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
    def initialize(self, bounds: Tuple[np.ndarray, np.ndarray]):
        """Initialize particle positions and velocities"""
        lower_bounds, upper_bounds = bounds
        
        self.particles = []
        for i in range(self.num_particles):
            position = np.random.uniform(lower_bounds, upper_bounds)
            velocity = np.random.uniform(-1, 1, self.dimensions)
            
            agent = SwarmAgent(
                agent_id=f"particle_{i}",
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                role=AgentRole.EXPLORER
            )
            self.particles.append(agent)
    
    async def optimize_step(
        self,
        fitness_func,
        iteration: int
    ) -> Tuple[np.ndarray, float]:
        """Perform one optimization step"""
        # Evaluate fitness for all particles
        fitness_values = []
        for particle in self.particles:
            if particle.can_move():
                fitness = await fitness_func(particle.position)
                fitness_values.append(fitness)
                
                # Update personal best
                personal_best_fitness = await fitness_func(particle.best_position)
                if fitness > personal_best_fitness:
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                
                particle.consume_energy(0.05)
            else:
                particle.recharge(0.2)
        
        # Update velocities and positions
        for particle in self.particles:
            if particle.can_move():
                # Random factors
                r1 = np.random.random(self.dimensions)
                r2 = np.random.random(self.dimensions)
                
                # Velocity update
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                
                # Position update
                particle.position += particle.velocity
        
        logger.info(
            f"PSO iteration {iteration}",
            best_fitness=self.global_best_fitness,
            avg_fitness=np.mean(fitness_values) if fitness_values else 0
        )
        
        return self.global_best_position, self.global_best_fitness


class AntColonyOptimizer:
    """
    Ant Colony Optimization for discrete optimization and pathfinding.
    
    Uses pheromone trails to find optimal paths.
    """
    
    def __init__(
        self,
        num_ants: int = 30,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,   # Heuristic importance
        evaporation_rate: float = 0.1,
        pheromone_deposit: float = 1.0
    ):
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        
        self.pheromone_matrix = None
        self.distance_matrix = None
        
    def initialize(self, distance_matrix: np.ndarray):
        """Initialize ACO with problem graph"""
        self.distance_matrix = distance_matrix
        n = len(distance_matrix)
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((n, n)) * 0.1
        
    async def find_path(
        self,
        start: int,
        end: int,
        iterations: int = 100
    ) -> Tuple[List[int], float]:
        """Find optimal path using ACO"""
        best_path = None
        best_distance = float('inf')
        
        n = len(self.distance_matrix)
        
        for iteration in range(iterations):
            paths = []
            distances = []
            
            # Each ant builds a solution
            for ant_id in range(self.num_ants):
                path = await self._construct_path(start, end)
                distance = self._calculate_path_distance(path)
                
                paths.append(path)
                distances.append(distance)
                
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
            
            # Update pheromones
            self._update_pheromones(paths, distances)
            
            logger.debug(
                f"ACO iteration {iteration}",
                best_distance=best_distance,
                avg_distance=np.mean(distances)
            )
        
        return best_path, best_distance
    
    async def _construct_path(self, start: int, end: int) -> List[int]:
        """Construct path for single ant"""
        current = start
        path = [current]
        visited = {current}
        
        while current != end:
            # Calculate probabilities
            probabilities = []
            next_nodes = []
            
            for next_node in range(len(self.distance_matrix)):
                if next_node not in visited:
                    pheromone = self.pheromone_matrix[current][next_node] ** self.alpha
                    heuristic = (1.0 / self.distance_matrix[current][next_node]) ** self.beta
                    probability = pheromone * heuristic
                    
                    probabilities.append(probability)
                    next_nodes.append(next_node)
            
            if not next_nodes:
                break
            
            # Select next node
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            
            next_node = np.random.choice(next_nodes, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path
    
    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calculate total distance of path"""
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i + 1]]
        return distance
    
    def _update_pheromones(self, paths: List[List[int]], distances: List[float]):
        """Update pheromone trails"""
        # Evaporation
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Deposit new pheromones
        for path, distance in zip(paths, distances):
            deposit = self.pheromone_deposit / distance
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i]][path[i + 1]] += deposit
                self.pheromone_matrix[path[i + 1]][path[i]] += deposit  # Symmetric


class BeeAlgorithm:
    """
    Artificial Bee Colony algorithm for resource allocation.
    
    Employs scout bees, worker bees, and onlooker bees.
    """
    
    def __init__(
        self,
        num_bees: int = 40,
        num_scouts: int = 10,
        num_sites: int = 5,
        elite_sites: int = 2
    ):
        self.num_bees = num_bees
        self.num_scouts = num_scouts
        self.num_sites = num_sites
        self.elite_sites = elite_sites
        
        self.food_sources = []
        self.best_source = None
        self.best_fitness = float('-inf')
        
    async def forage(
        self,
        fitness_func,
        bounds: Tuple[np.ndarray, np.ndarray],
        iterations: int = 100
    ) -> Tuple[np.ndarray, float]:
        """Main foraging loop"""
        lower_bounds, upper_bounds = bounds
        dimensions = len(lower_bounds)
        
        # Initialize food sources with scouts
        self.food_sources = []
        for i in range(self.num_scouts):
            position = np.random.uniform(lower_bounds, upper_bounds)
            fitness = await fitness_func(position)
            
            self.food_sources.append({
                'position': position,
                'fitness': fitness,
                'trials': 0
            })
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_source = position.copy()
        
        # Main foraging iterations
        for iteration in range(iterations):
            # Sort food sources by fitness
            self.food_sources.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Worker bee phase
            for i in range(min(self.num_sites, len(self.food_sources))):
                source = self.food_sources[i]
                
                # More bees for elite sites
                if i < self.elite_sites:
                    num_workers = self.num_bees // self.elite_sites
                else:
                    num_workers = (self.num_bees - self.elite_sites * (self.num_bees // self.elite_sites)) // (self.num_sites - self.elite_sites)
                
                # Workers explore neighborhood
                for _ in range(num_workers):
                    # Generate neighbor
                    neighbor = source['position'] + np.random.uniform(-1, 1, dimensions) * 0.1
                    neighbor = np.clip(neighbor, lower_bounds, upper_bounds)
                    
                    neighbor_fitness = await fitness_func(neighbor)
                    
                    # Greedy selection
                    if neighbor_fitness > source['fitness']:
                        source['position'] = neighbor
                        source['fitness'] = neighbor_fitness
                        source['trials'] = 0
                        
                        if neighbor_fitness > self.best_fitness:
                            self.best_fitness = neighbor_fitness
                            self.best_source = neighbor.copy()
                    else:
                        source['trials'] += 1
            
            # Scout bee phase - abandon exhausted sources
            for source in self.food_sources:
                if source['trials'] > 10:  # Abandonment threshold
                    source['position'] = np.random.uniform(lower_bounds, upper_bounds)
                    source['fitness'] = await fitness_func(source['position'])
                    source['trials'] = 0
            
            logger.debug(
                f"Bee iteration {iteration}",
                best_fitness=self.best_fitness,
                active_sources=len(self.food_sources)
            )
        
        return self.best_source, self.best_fitness


# ==================== Swarm Behaviors ====================

class SwarmBehavior:
    """Base class for swarm behaviors"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: DigitalPheromoneSystem
    ) -> Dict[str, Any]:
        """Execute behavior and return action"""
        raise NotImplementedError


class ExplorationBehavior(SwarmBehavior):
    """Explore unknown areas of the search space"""
    
    def __init__(self):
        super().__init__("exploration")
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: DigitalPheromoneSystem
    ) -> Dict[str, Any]:
        """Explore by avoiding well-visited areas"""
        current_location = environment.get('current_location', agent.agent_id)
        
        # Sense exploration pheromones
        exploration_pheromones = pheromones.sense(
            current_location,
            PheromoneType.EXPLORATION
        )
        
        # Choose direction with least exploration
        neighbors = environment.get('neighbors', [])
        if not neighbors:
            return {'action': 'wait'}
        
        # Score neighbors by exploration level
        neighbor_scores = []
        for neighbor in neighbors:
            neighbor_pheromones = pheromones.sense(neighbor, PheromoneType.EXPLORATION)
            exploration_level = sum(p.strength for p in neighbor_pheromones)
            neighbor_scores.append((neighbor, -exploration_level))  # Negative for least explored
        
        # Sort and pick least explored
        neighbor_scores.sort(key=lambda x: x[1])
        target = neighbor_scores[0][0]
        
        # Deposit exploration pheromone
        pheromones.deposit(
            target,
            PheromoneType.EXPLORATION,
            strength=0.8,
            agent_id=agent.agent_id,
            metadata={'timestamp': time.time()}
        )
        
        agent.consume_energy(0.05)
        
        return {
            'action': 'move',
            'target': target,
            'reason': 'exploring_new_area'
        }


class ForagingBehavior(SwarmBehavior):
    """Forage for resources following pheromone trails"""
    
    def __init__(self):
        super().__init__("foraging")
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: DigitalPheromoneSystem
    ) -> Dict[str, Any]:
        """Follow resource pheromones"""
        current_location = environment.get('current_location', agent.agent_id)
        
        # Sense resource pheromones
        resource_pheromones = pheromones.sense(
            current_location,
            PheromoneType.RESOURCE_RICH
        )
        
        if resource_pheromones:
            # Exploit known resource
            agent.recharge(0.2)
            return {
                'action': 'harvest',
                'location': current_location,
                'resource_level': sum(p.strength for p in resource_pheromones)
            }
        
        # Look for resource trails in neighbors
        neighbors = environment.get('neighbors', [])
        best_neighbor = None
        best_strength = 0
        
        for neighbor in neighbors:
            neighbor_pheromones = pheromones.sense(neighbor, PheromoneType.RESOURCE_RICH)
            strength = sum(p.strength for p in neighbor_pheromones)
            
            if strength > best_strength:
                best_strength = strength
                best_neighbor = neighbor
        
        if best_neighbor:
            agent.consume_energy(0.03)
            return {
                'action': 'move',
                'target': best_neighbor,
                'reason': 'following_resource_trail'
            }
        
        # No resources found, explore
        return await ExplorationBehavior().execute(agent, environment, pheromones)


class RecruitmentBehavior(SwarmBehavior):
    """Recruit other agents to important locations"""
    
    def __init__(self):
        super().__init__("recruitment")
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: DigitalPheromoneSystem
    ) -> Dict[str, Any]:
        """Deposit convergence pheromones to recruit others"""
        current_location = environment.get('current_location', agent.agent_id)
        importance = environment.get('location_importance', 0.5)
        
        if importance > 0.7:
            # Important location, recruit others
            pheromones.deposit(
                current_location,
                PheromoneType.CONVERGENCE,
                strength=importance,
                agent_id=agent.agent_id,
                metadata={'reason': 'high_importance', 'timestamp': time.time()}
            )
            
            agent.consume_energy(0.1)
            
            return {
                'action': 'signal',
                'signal_type': 'recruitment',
                'strength': importance
            }
        
        # Check if we should respond to recruitment
        convergence_pheromones = pheromones.sense(
            current_location,
            PheromoneType.CONVERGENCE
        )
        
        if convergence_pheromones:
            strongest = max(convergence_pheromones, key=lambda p: p.strength)
            if strongest.strength > 0.6:
                return {
                    'action': 'converge',
                    'location': strongest.location,
                    'strength': strongest.strength
                }
        
        return {'action': 'continue'}


class FlockingBehavior(SwarmBehavior):
    """Emergent flocking behavior for coordinated movement"""
    
    def __init__(self, separation_weight: float = 1.0, alignment_weight: float = 1.0, cohesion_weight: float = 1.0):
        super().__init__("flocking")
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        
    async def execute(
        self,
        agent: SwarmAgent,
        environment: Dict[str, Any],
        pheromones: DigitalPheromoneSystem
    ) -> Dict[str, Any]:
        """Calculate flocking forces"""
        nearby_agents = environment.get('nearby_agents', [])
        
        if not nearby_agents:
            return {'action': 'continue'}
        
        # Separation - avoid crowding
        separation_force = np.zeros_like(agent.position)
        for other in nearby_agents:
            diff = agent.position - other.position
            distance = np.linalg.norm(diff)
            if distance > 0 and distance < 2.0:  # Separation radius
                separation_force += diff / (distance ** 2)
        
        # Alignment - match velocity
        avg_velocity = np.mean([a.velocity for a in nearby_agents], axis=0)
        alignment_force = avg_velocity - agent.velocity
        
        # Cohesion - move toward center
        center_of_mass = np.mean([a.position for a in nearby_agents], axis=0)
        cohesion_force = center_of_mass - agent.position
        
        # Combine forces
        total_force = (
            self.separation_weight * separation_force +
            self.alignment_weight * alignment_force +
            self.cohesion_weight * cohesion_force
        )
        
        # Update velocity
        new_velocity = agent.velocity + total_force * 0.1
        new_velocity = np.clip(new_velocity, -1, 1)  # Limit speed
        
        agent.consume_energy(0.02)
        
        return {
            'action': 'flock',
            'new_velocity': new_velocity.tolist(),
            'force_components': {
                'separation': separation_force.tolist(),
                'alignment': alignment_force.tolist(),
                'cohesion': cohesion_force.tolist()
            }
        }


# ==================== Main Swarm Coordinator ====================

class SwarmCoordinator:
    """
    Unified swarm intelligence coordinator for AURA.
    
    Integrates multiple swarm algorithms and behaviors for:
    - Multi-agent coordination
    - Collective optimization
    - Emergent intelligence
    - Distributed problem solving
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.pheromone_system = DigitalPheromoneSystem(
            decay_rate=self.config.get('pheromone_decay', 0.97)
        )
        
        # Swarm algorithms
        self.pso = ParticleSwarmOptimizer(
            num_particles=self.config.get('num_particles', 50)
        )
        self.aco = AntColonyOptimizer(
            num_ants=self.config.get('num_ants', 30)
        )
        self.bee = BeeAlgorithm(
            num_bees=self.config.get('num_bees', 40)
        )
        
        # Neural controller
        self.neural_controller = NeuralSwarmController(
            input_dim=self.config.get('input_dim', 128),
            hidden_dim=self.config.get('hidden_dim', 256)
        )
        
        # Behaviors
        self.behaviors = {
            'explore': ExplorationBehavior(),
            'forage': ForagingBehavior(),
            'recruit': RecruitmentBehavior(),
            'flock': FlockingBehavior()
        }
        
        # Agent management
        self.agents: Dict[str, SwarmAgent] = {}
        self.agent_locations: Dict[str, str] = {}
        
        # Metrics
        self.convergence_history = []
        self.optimization_history = []
        
        logger.info(
            "SwarmCoordinator initialized",
            algorithms=['PSO', 'ACO', 'Bee'],
            behaviors=list(self.behaviors.keys()),
            neural_control=True
        )
    
    async def coordinate_agents(
        self,
        agents: List[str],
        objective: Dict[str, Any],
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Main coordination method for multi-agent swarm.
        
        Args:
            agents: List of agent IDs
            objective: Task objective with type and parameters
            max_iterations: Maximum coordination rounds
            
        Returns:
            Coordination results with paths, assignments, metrics
        """
        # Initialize agents
        for agent_id in agents:
            if agent_id not in self.agents:
                self.agents[agent_id] = SwarmAgent(
                    agent_id=agent_id,
                    position=np.random.randn(10),  # Default 10D space
                    velocity=np.zeros(10),
                    best_position=np.random.randn(10)
                )
        
        objective_type = objective.get('type', 'exploration')
        
        if objective_type == 'optimization':
            return await self._coordinate_optimization(agents, objective, max_iterations)
        elif objective_type == 'pathfinding':
            return await self._coordinate_pathfinding(agents, objective, max_iterations)
        elif objective_type == 'resource_allocation':
            return await self._coordinate_resource_allocation(agents, objective, max_iterations)
        else:
            return await self._coordinate_exploration(agents, objective, max_iterations)
    
    async def _coordinate_optimization(
        self,
        agents: List[str],
        objective: Dict[str, Any],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Coordinate agents for optimization task using PSO"""
        fitness_func = objective.get('fitness_func')
        bounds = objective.get('bounds', (np.array([-10] * 10), np.array([10] * 10)))
        
        # Initialize PSO with agents
        self.pso.initialize(bounds)
        
        best_position = None
        best_fitness = float('-inf')
        
        for iteration in range(max_iterations):
            # PSO step
            position, fitness = await self.pso.optimize_step(fitness_func, iteration)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_position = position
            
            # Decay pheromones
            self.pheromone_system.decay_all()
            
            # Check convergence
            if iteration > 10:
                recent_history = self.optimization_history[-10:]
                if all(abs(h - best_fitness) < 0.001 for h in recent_history):
                    logger.info("Swarm converged early", iteration=iteration)
                    break
            
            self.optimization_history.append(best_fitness)
        
        return {
            'type': 'optimization',
            'best_position': best_position.tolist() if best_position is not None else None,
            'best_fitness': best_fitness,
            'iterations': iteration + 1,
            'convergence_history': self.optimization_history[-50:],
            'algorithm': 'PSO'
        }
    
    async def _coordinate_pathfinding(
        self,
        agents: List[str],
        objective: Dict[str, Any],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Coordinate agents for pathfinding using ACO"""
        graph = objective.get('graph')  # NetworkX graph or distance matrix
        start = objective.get('start')
        end = objective.get('end')
        
        # Convert graph to distance matrix if needed
        if isinstance(graph, nx.Graph):
            nodes = list(graph.nodes())
            n = len(nodes)
            distance_matrix = np.full((n, n), np.inf)
            
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if graph.has_edge(u, v):
                        distance_matrix[i][j] = graph[u][v].get('weight', 1)
                    elif i == j:
                        distance_matrix[i][j] = 0
            
            # Map start/end to indices
            start_idx = nodes.index(start)
            end_idx = nodes.index(end)
        else:
            distance_matrix = graph
            start_idx = start
            end_idx = end
            nodes = list(range(len(distance_matrix)))
        
        # Initialize ACO
        self.aco.initialize(distance_matrix)
        
        # Find path
        best_path, best_distance = await self.aco.find_path(
            start_idx,
            end_idx,
            max_iterations
        )
        
        # Convert indices back to nodes
        if isinstance(graph, nx.Graph):
            best_path = [nodes[i] for i in best_path]
        
        return {
            'type': 'pathfinding',
            'best_path': best_path,
            'path_distance': best_distance,
            'algorithm': 'ACO',
            'pheromone_map': self.pheromone_system.get_pheromone_map()
        }
    
    async def _coordinate_resource_allocation(
        self,
        agents: List[str],
        objective: Dict[str, Any],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Coordinate resource allocation using Bee algorithm"""
        resources = objective.get('resources', [])
        fitness_func = objective.get('allocation_fitness')
        
        # Define bounds based on resource dimensions
        num_resources = len(resources)
        num_agents = len(agents)
        dimensions = num_resources * num_agents  # Allocation matrix
        
        bounds = (
            np.zeros(dimensions),  # Min allocation
            np.ones(dimensions)    # Max allocation (normalized)
        )
        
        # Run bee algorithm
        best_allocation, best_fitness = await self.bee.forage(
            fitness_func,
            bounds,
            max_iterations
        )
        
        # Reshape to allocation matrix
        allocation_matrix = best_allocation.reshape(num_agents, num_resources)
        
        # Normalize rows to ensure valid allocation
        allocation_matrix = allocation_matrix / allocation_matrix.sum(axis=1, keepdims=True)
        
        # Create allocation map
        allocation_map = {}
        for i, agent_id in enumerate(agents):
            allocation_map[agent_id] = {
                resources[j]: float(allocation_matrix[i, j])
                for j in range(num_resources)
            }
        
        return {
            'type': 'resource_allocation',
            'allocation_map': allocation_map,
            'fitness': best_fitness,
            'algorithm': 'Bee',
            'food_sources': len(self.bee.food_sources)
        }
    
    async def _coordinate_exploration(
        self,
        agents: List[str],
        objective: Dict[str, Any],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Coordinate collective exploration"""
        environment = objective.get('environment', {})
        
        # Initialize agent locations
        for agent_id in agents:
            if agent_id not in self.agent_locations:
                self.agent_locations[agent_id] = f"node_{hash(agent_id) % 10}"
        
        exploration_coverage = set()
        convergence_points = []
        
        for iteration in range(max_iterations):
            # Each agent selects behavior
            agent_actions = {}
            
            for agent_id in agents:
                agent = self.agents[agent_id]
                
                # Get agent environment
                agent_env = {
                    'current_location': self.agent_locations[agent_id],
                    'neighbors': self._get_neighbors(self.agent_locations[agent_id]),
                    'nearby_agents': self._get_nearby_agents(agent_id, agents)
                }
                
                # Select behavior based on energy and role
                if agent.energy < 0.3:
                    behavior = self.behaviors['forage']
                elif agent.role == AgentRole.SCOUT:
                    behavior = self.behaviors['explore']
                elif agent.role == AgentRole.COORDINATOR:
                    behavior = self.behaviors['recruit']
                else:
                    behavior = self.behaviors['explore']
                
                # Execute behavior
                action = await behavior.execute(agent, agent_env, self.pheromone_system)
                agent_actions[agent_id] = action
                
                # Update location if moving
                if action.get('action') == 'move':
                    self.agent_locations[agent_id] = action.get('target')
                    exploration_coverage.add(action.get('target'))
            
            # Decay pheromones
            self.pheromone_system.decay_all()
            
            # Check for convergence
            convergence = self._detect_convergence(agents)
            if convergence:
                convergence_points.extend(convergence)
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(
                    f"Exploration iteration {iteration}",
                    coverage=len(exploration_coverage),
                    convergence_points=len(convergence_points)
                )
        
        return {
            'type': 'exploration',
            'coverage': list(exploration_coverage),
            'convergence_points': convergence_points,
            'pheromone_map': self.pheromone_system.get_pheromone_map(),
            'agent_locations': dict(self.agent_locations),
            'iterations': max_iterations
        }
    
    def _get_neighbors(self, location: str) -> List[str]:
        """Get neighboring locations (simplified)"""
        # In real implementation, this would use actual topology
        base = int(location.split('_')[1]) if '_' in location else 0
        neighbors = []
        
        for offset in [-1, 1]:
            neighbor_id = (base + offset) % 10
            neighbors.append(f"node_{neighbor_id}")
        
        return neighbors
    
    def _get_nearby_agents(self, agent_id: str, all_agents: List[str]) -> List[SwarmAgent]:
        """Get agents in same or neighboring locations"""
        current_location = self.agent_locations.get(agent_id)
        nearby = []
        
        for other_id in all_agents:
            if other_id != agent_id:
                other_location = self.agent_locations.get(other_id)
                if other_location == current_location or other_location in self._get_neighbors(current_location):
                    nearby.append(self.agents[other_id])
        
        return nearby
    
    def _detect_convergence(self, agents: List[str]) -> List[Dict[str, Any]]:
        """Detect convergence points where many agents gather"""
        location_counts = defaultdict(int)
        
        for agent_id in agents:
            location = self.agent_locations.get(agent_id)
            if location:
                location_counts[location] += 1
        
        convergence_points = []
        threshold = len(agents) * 0.3  # 30% of agents
        
        for location, count in location_counts.items():
            if count >= threshold:
                convergence_points.append({
                    'location': location,
                    'agent_count': count,
                    'percentage': count / len(agents)
                })
        
        return convergence_points
    
    async def optimize_parameters(
        self,
        search_space: Dict[str, Tuple[float, float]],
        objective_function,
        algorithm: SwarmAlgorithm = SwarmAlgorithm.PARTICLE_SWARM,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Use swarm intelligence to optimize parameters.
        
        Args:
            search_space: Parameter names and bounds
            objective_function: Function to optimize
            algorithm: Which swarm algorithm to use
            iterations: Number of iterations
            
        Returns:
            Optimization results
        """
        param_names = list(search_space.keys())
        lower_bounds = np.array([bounds[0] for bounds in search_space.values()])
        upper_bounds = np.array([bounds[1] for bounds in search_space.values()])
        bounds = (lower_bounds, upper_bounds)
        
        # Wrap objective function to work with arrays
        async def array_objective(x):
            params = {name: x[i] for i, name in enumerate(param_names)}
            return await objective_function(params)
        
        if algorithm == SwarmAlgorithm.PARTICLE_SWARM:
            self.pso.dimensions = len(param_names)
            self.pso.initialize(bounds)
            
            best_position = None
            best_fitness = float('-inf')
            
            for i in range(iterations):
                position, fitness = await self.pso.optimize_step(array_objective, i)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_position = position
            
            best_params = {name: best_position[i] for i, name in enumerate(param_names)}
            
        elif algorithm == SwarmAlgorithm.BEE_ALGORITHM:
            best_position, best_fitness = await self.bee.forage(
                array_objective,
                bounds,
                iterations
            )
            best_params = {name: best_position[i] for i, name in enumerate(param_names)}
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'algorithm': algorithm.value,
            'iterations': iterations,
            'search_space': search_space
        }
    
    async def detect_collective_failures(
        self,
        system_state: Dict[str, Any],
        num_agents: int = 30,
        rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Use swarm intelligence to detect complex failure patterns.
        
        Args:
            system_state: Current system state with component statuses
            num_agents: Number of swarm agents to deploy
            rounds: Number of exploration rounds
            
        Returns:
            Detected failure patterns and risk areas
        """
        # Initialize failure detection agents
        components = list(system_state.get('components', {}).keys())
        
        if not components:
            return {'error': 'No components in system state'}
        
        # Reset agent locations to components
        for i in range(num_agents):
            agent_id = f"detector_{i}"
            if agent_id not in self.agents:
                self.agents[agent_id] = SwarmAgent(
                    agent_id=agent_id,
                    position=np.random.randn(10),
                    velocity=np.zeros(10),
                    best_position=np.random.randn(10),
                    role=AgentRole.SENTINEL
                )
            
            # Assign to random component
            self.agent_locations[agent_id] = random.choice(components)
        
        error_patterns = defaultdict(int)
        danger_zones = set()
        
        for round_num in range(rounds):
            # Each agent explores and marks findings
            for i in range(num_agents):
                agent_id = f"detector_{i}"
                agent = self.agents[agent_id]
                current_component = self.agent_locations[agent_id]
                
                # Check component health
                component_data = system_state['components'].get(current_component, {})
                error_rate = component_data.get('error_rate', 0)
                latency = component_data.get('latency', 0)
                
                # Deposit pheromones based on findings
                if error_rate > 0.1:
                    self.pheromone_system.deposit(
                        current_component,
                        PheromoneType.ERROR_TRAIL,
                        strength=error_rate,
                        agent_id=agent_id,
                        metadata={'error_type': component_data.get('last_error')}
                    )
                    error_patterns[component_data.get('last_error', 'unknown')] += 1
                
                if latency > 100:  # High latency threshold
                    self.pheromone_system.deposit(
                        current_component,
                        PheromoneType.DANGER_ZONE,
                        strength=min(1.0, latency / 1000),
                        agent_id=agent_id,
                        metadata={'latency_ms': latency}
                    )
                    danger_zones.add(current_component)
                
                # Move to connected component
                connections = component_data.get('connections', [])
                if connections and agent.can_move():
                    # Follow error trails or explore
                    next_component = None
                    max_error_pheromone = 0
                    
                    for conn in connections:
                        error_pheromones = self.pheromone_system.sense(
                            conn,
                            PheromoneType.ERROR_TRAIL
                        )
                        total_strength = sum(p.strength for p in error_pheromones)
                        
                        if total_strength > max_error_pheromone:
                            max_error_pheromone = total_strength
                            next_component = conn
                    
                    if not next_component or random.random() < 0.3:  # Exploration
                        next_component = random.choice(connections)
                    
                    self.agent_locations[agent_id] = next_component
                    agent.consume_energy(0.05)
                else:
                    agent.recharge(0.1)
            
            # Decay pheromones
            self.pheromone_system.decay_all()
        
        # Analyze findings
        pheromone_map = self.pheromone_system.get_pheromone_map()
        
        # Identify critical patterns
        critical_components = []
        for component, pheromones in pheromone_map.items():
            error_level = pheromones.get(PheromoneType.ERROR_TRAIL.value, 0)
            danger_level = pheromones.get(PheromoneType.DANGER_ZONE.value, 0)
            
            if error_level > 0.5 or danger_level > 0.5:
                critical_components.append({
                    'component': component,
                    'error_level': error_level,
                    'danger_level': danger_level,
                    'risk_score': error_level + danger_level
                })
        
        # Sort by risk
        critical_components.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {
            'critical_components': critical_components[:10],
            'error_patterns': dict(error_patterns),
            'danger_zones': list(danger_zones),
            'exploration_coverage': len(set(self.agent_locations.values())) / len(components),
            'convergence_detected': len(self._detect_convergence(list(range(num_agents)))) > 0
        }
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status and metrics"""
        active_agents = len([a for a in self.agents.values() if a.energy > 0.1])
        
        return {
            'total_agents': len(self.agents),
            'active_agents': active_agents,
            'agent_locations': dict(self.agent_locations),
            'pheromone_summary': {
                ptype.value: sum(
                    len(pheromones)
                    for pheromones in self.pheromone_system.pheromones.values()
                    if any(p.pheromone_type == ptype for p in pheromones)
                )
                for ptype in PheromoneType
            },
            'convergence_history': self.convergence_history[-20:],
            'optimization_history': self.optimization_history[-20:]
        }


# ==================== Example Usage ====================

async def example():
    """Example usage of SwarmCoordinator"""
    print("\n🐜 AURA Swarm Intelligence Example\n")
    
    # Initialize coordinator
    coordinator = SwarmCoordinator({
        'num_particles': 30,
        'num_ants': 20,
        'num_bees': 25,
        'pheromone_decay': 0.95
    })
    
    # Example 1: Parameter optimization
    print("1. Parameter Optimization with PSO...")
    
    search_space = {
        'learning_rate': (0.0001, 0.1),
        'batch_size': (16, 128),
        'hidden_dim': (64, 512)
    }
    
    async def dummy_objective(params):
        # Simulate model performance
        lr = params['learning_rate']
        bs = params['batch_size']
        hd = params['hidden_dim']
        return -((lr - 0.01)**2 + (bs - 64)**2 + (hd - 256)**2)
    
    result = await coordinator.optimize_parameters(
        search_space,
        dummy_objective,
        algorithm=SwarmAlgorithm.PARTICLE_SWARM,
        iterations=50
    )
    
    print(f"   Best parameters: {result['best_parameters']}")
    print(f"   Best fitness: {result['best_fitness']:.4f}")
    
    # Example 2: Multi-agent coordination
    print("\n2. Multi-agent Exploration...")
    
    agents = [f"agent_{i}" for i in range(10)]
    
    exploration_result = await coordinator.coordinate_agents(
        agents,
        {
            'type': 'exploration',
            'environment': {
                'size': 20,
                'obstacles': []
            }
        },
        max_iterations=30
    )
    
    print(f"   Coverage: {len(exploration_result['coverage'])} locations")
    print(f"   Convergence points: {len(exploration_result['convergence_points'])}")
    
    # Example 3: Failure detection
    print("\n3. Collective Failure Detection...")
    
    # Simulate system with some failing components
    system_state = {
        'components': {
            f'comp_{i}': {
                'error_rate': random.random() * 0.2 if i % 5 == 0 else 0,
                'latency': random.randint(10, 200) if i % 3 == 0 else 20,
                'connections': [f'comp_{j}' for j in range(max(0, i-2), min(20, i+3)) if j != i],
                'last_error': random.choice(['timeout', 'connection_failed', 'invalid_response'])
            }
            for i in range(20)
        }
    }
    
    failure_result = await coordinator.detect_collective_failures(
        system_state,
        num_agents=15,
        rounds=20
    )
    
    print(f"   Critical components: {len(failure_result['critical_components'])}")
    if failure_result['critical_components']:
        print(f"   Highest risk: {failure_result['critical_components'][0]}")
    print(f"   Error patterns: {failure_result['error_patterns']}")
    print(f"   Coverage: {failure_result['exploration_coverage']:.1%}")
    
    # Show final status
    print("\n4. Swarm Status:")
    status = coordinator.get_swarm_status()
    print(f"   Active agents: {status['active_agents']}/{status['total_agents']}")
    print(f"   Pheromone summary: {status['pheromone_summary']}")


if __name__ == "__main__":
    asyncio.run(example())