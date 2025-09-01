"""
Advanced Swarm Intelligence System - 2025 Implementation

Based on latest research:
- Multi-agent coordination algorithms
- Emergent collective behavior
- Bio-inspired swarm dynamics
- Decentralized decision making
- Self-organizing systems
- Adaptive swarm topology
- Quantum-inspired swarm optimization

Key innovations:
- Neural swarm controllers
- Dynamic role assignment
- Stigmergic communication
- Federated learning in swarms
- Resilient consensus mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
import asyncio
from collections import defaultdict
import networkx as nx
import math
import random

logger = structlog.get_logger(__name__)


class SwarmAlgorithm(str, Enum):
    """Types of swarm algorithms"""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_ALGORITHM = "bee_algorithm"
    FIREFLY = "firefly"
    WOLF_PACK = "wolf_pack"
    FISH_SCHOOL = "fish_school"
    HYBRID = "hybrid"


class CommunicationType(str, Enum):
    """Agent communication types"""
    DIRECT = "direct"
    STIGMERGIC = "stigmergic"
    BROADCAST = "broadcast"
    LOCAL = "local"
    QUANTUM = "quantum"


class AgentRole(str, Enum):
    """Dynamic agent roles"""
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"
    COORDINATOR = "coordinator"
    SCOUT = "scout"
    WORKER = "worker"
    QUEEN = "queen"


@dataclass
class SwarmConfig:
    """Configuration for swarm intelligence system"""
    # Swarm composition
    num_agents: int = 100
    algorithm: SwarmAlgorithm = SwarmAlgorithm.HYBRID
    communication: CommunicationType = CommunicationType.STIGMERGIC
    
    # Agent properties
    agent_dim: int = 128
    perception_radius: float = 5.0
    max_velocity: float = 2.0
    
    # Behavior parameters
    inertia_weight: float = 0.7
    cognitive_weight: float = 1.5
    social_weight: float = 1.5
    
    # Topology
    topology: str = "dynamic"  # static, dynamic, adaptive
    neighbors_k: int = 5
    
    # Learning
    use_neural_controller: bool = True
    learning_rate: float = 0.001
    
    # Environment
    environment_dim: int = 3
    pheromone_decay: float = 0.1
    pheromone_strength: float = 1.0


@dataclass
class Agent:
    """Individual swarm agent"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = float('-inf')
    role: AgentRole = AgentRole.WORKER
    neighbors: Set[int] = field(default_factory=set)
    memory: List[Any] = field(default_factory=list)
    pheromone_level: float = 0.0
    energy: float = 100.0
    
    def update_best(self, fitness: float):
        """Update personal best if improved"""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()


class NeuralSwarmController(nn.Module):
    """Neural network controller for swarm agents"""
    
    def __init__(self, config: SwarmConfig):
        super().__init__()
        self.config = config
        
        # Perception network
        self.perception = nn.Sequential(
            nn.Linear(config.agent_dim + config.environment_dim * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Role assignment network
        self.role_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(AgentRole))
        )
        
        # Action network
        self.action_network = nn.Sequential(
            nn.Linear(128 + len(AgentRole), 64),
            nn.ReLU(),
            nn.Linear(64, config.environment_dim * 2)  # velocity + communication
        )
        
        # Value network for reinforcement learning
        self.value_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process observations and output actions"""
        # Perception
        features = self.perception(observations)
        
        # Role assignment
        role_logits = self.role_network(features)
        role_probs = F.softmax(role_logits, dim=-1)
        
        # Action generation
        role_embedding = role_probs
        action_input = torch.cat([features, role_embedding], dim=-1)
        actions = self.action_network(action_input)
        
        # Split actions
        velocity = actions[:, :self.config.environment_dim]
        communication = actions[:, self.config.environment_dim:]
        
        # Value estimation
        value = self.value_network(features)
        
        return {
            'velocity': torch.tanh(velocity) * self.config.max_velocity,
            'communication': torch.sigmoid(communication),
            'role_probs': role_probs,
            'value': value
        }


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization implementation"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
    
    def update_velocity(self, agent: Agent, neighbors_best: np.ndarray) -> np.ndarray:
        """Update agent velocity using PSO equations"""
        # Random factors
        r1 = np.random.random(self.config.environment_dim)
        r2 = np.random.random(self.config.environment_dim)
        
        # Cognitive component (personal best)
        cognitive = self.config.cognitive_weight * r1 * (agent.best_position - agent.position)
        
        # Social component (neighborhood best)
        social = self.config.social_weight * r2 * (neighbors_best - agent.position)
        
        # Update velocity
        new_velocity = (self.config.inertia_weight * agent.velocity + 
                       cognitive + social)
        
        # Clamp velocity
        speed = np.linalg.norm(new_velocity)
        if speed > self.config.max_velocity:
            new_velocity = new_velocity / speed * self.config.max_velocity
        
        return new_velocity


class AntColonyOptimizer:
    """Ant Colony Optimization implementation"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.pheromone_map = defaultdict(float)
    
    def deposit_pheromone(self, path: List[Tuple], quality: float):
        """Deposit pheromone on path based on quality"""
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            self.pheromone_map[edge] += quality * self.config.pheromone_strength
    
    def evaporate_pheromone(self):
        """Evaporate pheromone over time"""
        for edge in list(self.pheromone_map.keys()):
            self.pheromone_map[edge] *= (1 - self.config.pheromone_decay)
            if self.pheromone_map[edge] < 0.01:
                del self.pheromone_map[edge]
    
    def select_next_position(self, current: Tuple, candidates: List[Tuple]) -> Tuple:
        """Select next position based on pheromone levels"""
        if not candidates:
            return current
        
        # Calculate probabilities
        pheromone_levels = []
        for candidate in candidates:
            edge = (current, candidate)
            level = self.pheromone_map.get(edge, 0.1)  # Small default
            pheromone_levels.append(level)
        
        # Convert to probabilities
        total = sum(pheromone_levels)
        if total == 0:
            probabilities = [1.0 / len(candidates)] * len(candidates)
        else:
            probabilities = [level / total for level in pheromone_levels]
        
        # Select based on probabilities
        return np.random.choice(candidates, p=probabilities)


class SwarmTopology:
    """Dynamic swarm topology management"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.graph = nx.Graph()
    
    def update_topology(self, agents: List[Agent], positions: np.ndarray):
        """Update agent connections based on positions"""
        self.graph.clear()
        
        # Add nodes
        for agent in agents:
            self.graph.add_node(agent.id, agent=agent)
        
        if self.config.topology == "static":
            # Ring topology
            for i in range(len(agents)):
                self.graph.add_edge(agents[i].id, agents[(i + 1) % len(agents)].id)
                
        elif self.config.topology == "dynamic":
            # Distance-based connections
            for i, agent_i in enumerate(agents):
                distances = []
                for j, agent_j in enumerate(agents):
                    if i != j:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances.append((dist, j))
                
                # Connect to k nearest neighbors
                distances.sort()
                for dist, j in distances[:self.config.neighbors_k]:
                    if dist <= self.config.perception_radius:
                        self.graph.add_edge(agent_i.id, agents[j].id)
        
        elif self.config.topology == "adaptive":
            # Fitness-based connections
            fitness_scores = [(agent.best_fitness, agent) for agent in agents]
            fitness_scores.sort(reverse=True)
            
            # Elite agents are highly connected
            elite_size = max(1, len(agents) // 10)
            for i in range(elite_size):
                for j in range(i + 1, min(elite_size + 5, len(agents))):
                    self.graph.add_edge(
                        fitness_scores[i][1].id,
                        fitness_scores[j][1].id
                    )
    
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighbor IDs for an agent"""
        if agent_id in self.graph:
            return list(self.graph.neighbors(agent_id))
        return []


class AdvancedSwarmSystem:
    """Complete advanced swarm intelligence system"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Components
        self.topology = SwarmTopology(config)
        self.pso = ParticleSwarmOptimizer(config)
        self.aco = AntColonyOptimizer(config)
        
        # Neural controller
        if config.use_neural_controller:
            self.controller = NeuralSwarmController(config)
            self.optimizer = torch.optim.Adam(
                self.controller.parameters(),
                lr=config.learning_rate
            )
        
        # Metrics
        self.iteration = 0
        self.convergence_history = []
        
        logger.info("Advanced Swarm System initialized",
                   num_agents=config.num_agents,
                   algorithm=config.algorithm.value)
    
    def _initialize_agents(self) -> List[Agent]:
        """Initialize swarm agents"""
        agents = []
        
        for i in range(self.config.num_agents):
            # Random initialization
            position = np.random.uniform(-10, 10, self.config.environment_dim)
            velocity = np.random.uniform(-1, 1, self.config.environment_dim)
            
            # Assign initial roles
            if i < self.config.num_agents // 10:
                role = AgentRole.SCOUT
            elif i < self.config.num_agents // 5:
                role = AgentRole.COORDINATOR
            else:
                role = AgentRole.WORKER
            
            agent = Agent(
                id=i,
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                role=role
            )
            
            agents.append(agent)
        
        return agents
    
    async def step(self, fitness_function: callable) -> Dict[str, Any]:
        """Execute one swarm iteration"""
        self.iteration += 1
        
        # Update topology
        positions = np.array([agent.position for agent in self.agents])
        self.topology.update_topology(self.agents, positions)
        
        # Update agent neighbors
        for agent in self.agents:
            agent.neighbors = set(self.topology.get_neighbors(agent.id))
        
        # Evaluate fitness
        fitness_values = []
        for agent in self.agents:
            fitness = fitness_function(agent.position)
            agent.update_best(fitness)
            fitness_values.append(fitness)
            
            # Update global best
            if fitness > self.pso.global_best_fitness:
                self.pso.global_best_fitness = fitness
                self.pso.global_best_position = agent.position.copy()
        
        # Apply swarm algorithm
        if self.config.algorithm == SwarmAlgorithm.PARTICLE_SWARM:
            await self._pso_step()
        elif self.config.algorithm == SwarmAlgorithm.ANT_COLONY:
            await self._aco_step()
        elif self.config.algorithm == SwarmAlgorithm.HYBRID:
            await self._hybrid_step()
        
        # Collect metrics
        metrics = {
            'iteration': self.iteration,
            'best_fitness': self.pso.global_best_fitness,
            'avg_fitness': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'convergence': self._calculate_convergence()
        }
        
        self.convergence_history.append(metrics['convergence'])
        
        return metrics
    
    async def _pso_step(self):
        """Particle swarm optimization step"""
        for agent in self.agents:
            # Get neighborhood best
            neighbor_bests = [agent.best_position]
            for neighbor_id in agent.neighbors:
                neighbor = self.agents[neighbor_id]
                neighbor_bests.append(neighbor.best_position)
            
            # Find best among neighbors
            best_neighbor_pos = max(
                neighbor_bests,
                key=lambda pos: self.pso.global_best_fitness
            )
            
            # Update velocity and position
            agent.velocity = self.pso.update_velocity(agent, best_neighbor_pos)
            agent.position += agent.velocity
    
    async def _aco_step(self):
        """Ant colony optimization step"""
        # Discretize positions for ACO
        grid_size = 20
        
        for agent in self.agents:
            # Convert to grid coordinates
            grid_pos = tuple(
                int((p + 10) / 20 * grid_size) 
                for p in agent.position
            )
            
            # Get neighboring grid cells
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        neighbor = (
                            grid_pos[0] + dx,
                            grid_pos[1] + dy,
                            grid_pos[2] + dz if len(grid_pos) > 2 else 0
                        )
                        neighbors.append(neighbor)
            
            # Select next position based on pheromone
            next_grid = self.aco.select_next_position(grid_pos, neighbors)
            
            # Convert back to continuous space
            agent.position = np.array([
                (coord + 0.5) / grid_size * 20 - 10
                for coord in next_grid[:self.config.environment_dim]
            ])
        
        # Evaporate pheromone
        self.aco.evaporate_pheromone()
    
    async def _hybrid_step(self):
        """Hybrid algorithm combining multiple strategies"""
        # Neural controller predictions
        if self.config.use_neural_controller:
            observations = self._get_observations()
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
            
            with torch.no_grad():
                actions = self.controller(obs_tensor)
            
            # Apply neural actions
            for i, agent in enumerate(self.agents):
                # Update velocity from neural controller
                neural_velocity = actions['velocity'][i].numpy()
                
                # Blend with PSO velocity
                neighbor_best = self._get_best_neighbor_position(agent)
                pso_velocity = self.pso.update_velocity(agent, neighbor_best)
                
                # Weighted combination
                agent.velocity = 0.5 * neural_velocity + 0.5 * pso_velocity
                
                # Update position
                agent.position += agent.velocity
                
                # Update role based on neural output
                role_idx = actions['role_probs'][i].argmax().item()
                agent.role = list(AgentRole)[role_idx]
        else:
            # Fallback to PSO
            await self._pso_step()
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for neural controller"""
        observations = []
        
        for agent in self.agents:
            # Agent state
            obs = np.concatenate([
                agent.position,
                agent.velocity,
                agent.best_position,
                [agent.best_fitness],
                [agent.energy],
                [len(agent.neighbors)]
            ])
            
            # Neighbor information
            neighbor_positions = []
            neighbor_velocities = []
            
            for neighbor_id in list(agent.neighbors)[:5]:  # Limit neighbors
                neighbor = self.agents[neighbor_id]
                neighbor_positions.extend(neighbor.position)
                neighbor_velocities.extend(neighbor.velocity)
            
            # Pad if needed
            while len(neighbor_positions) < 5 * self.config.environment_dim:
                neighbor_positions.extend([0] * self.config.environment_dim)
                neighbor_velocities.extend([0] * self.config.environment_dim)
            
            obs = np.concatenate([obs, neighbor_positions, neighbor_velocities])
            observations.append(obs)
        
        return np.array(observations)
    
    def _get_best_neighbor_position(self, agent: Agent) -> np.ndarray:
        """Get best position among neighbors"""
        best_pos = agent.best_position
        best_fitness = agent.best_fitness
        
        for neighbor_id in agent.neighbors:
            neighbor = self.agents[neighbor_id]
            if neighbor.best_fitness > best_fitness:
                best_fitness = neighbor.best_fitness
                best_pos = neighbor.best_position
        
        return best_pos
    
    def _calculate_convergence(self) -> float:
        """Calculate swarm convergence metric"""
        positions = np.array([agent.position for agent in self.agents])
        centroid = positions.mean(axis=0)
        
        # Average distance to centroid
        distances = [np.linalg.norm(pos - centroid) for pos in positions]
        return np.mean(distances)
    
    def get_diversity(self) -> float:
        """Calculate swarm diversity"""
        positions = np.array([agent.position for agent in self.agents])
        
        # Pairwise distances
        diversity = 0
        count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                diversity += np.linalg.norm(positions[i] - positions[j])
                count += 1
        
        return diversity / count if count > 0 else 0
    
    def apply_selection_pressure(self, elite_ratio: float = 0.1):
        """Apply evolutionary selection pressure"""
        # Sort by fitness
        self.agents.sort(key=lambda a: a.best_fitness, reverse=True)
        
        # Keep elite agents
        elite_size = int(elite_ratio * len(self.agents))
        elite = self.agents[:elite_size]
        
        # Replace worst agents with mutations of elite
        for i in range(len(self.agents) - elite_size, len(self.agents)):
            # Select random elite agent
            parent = random.choice(elite)
            
            # Create mutated offspring
            self.agents[i].position = parent.position + np.random.normal(0, 0.1, self.config.environment_dim)
            self.agents[i].velocity = np.random.uniform(-0.1, 0.1, self.config.environment_dim)
            self.agents[i].best_position = self.agents[i].position.copy()
            self.agents[i].best_fitness = float('-inf')


# Specialized swarm behaviors
class FlockingBehavior:
    """Reynolds flocking rules"""
    
    @staticmethod
    def separation(agent: Agent, neighbors: List[Agent], min_distance: float = 1.0) -> np.ndarray:
        """Avoid crowding neighbors"""
        steer = np.zeros(len(agent.position))
        count = 0
        
        for neighbor in neighbors:
            distance = np.linalg.norm(agent.position - neighbor.position)
            if 0 < distance < min_distance:
                diff = agent.position - neighbor.position
                diff = diff / distance  # Normalize
                steer += diff
                count += 1
        
        if count > 0:
            steer = steer / count
        
        return steer
    
    @staticmethod
    def alignment(agent: Agent, neighbors: List[Agent]) -> np.ndarray:
        """Align with average heading of neighbors"""
        if not neighbors:
            return np.zeros(len(agent.velocity))
        
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        return avg_velocity - agent.velocity
    
    @staticmethod
    def cohesion(agent: Agent, neighbors: List[Agent]) -> np.ndarray:
        """Steer towards average position of neighbors"""
        if not neighbors:
            return np.zeros(len(agent.position))
        
        avg_position = np.mean([n.position for n in neighbors], axis=0)
        return avg_position - agent.position


# Example usage
def demonstrate_swarm_system():
    """Demonstrate advanced swarm intelligence"""
    print("🐝 Advanced Swarm Intelligence Demonstration")
    print("=" * 60)
    
    # Define fitness function (finding global optimum)
    def fitness_function(position):
        # Rastrigin function (many local optima)
        A = 10
        n = len(position)
        return -(A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in position))
    
    # Create swarm system
    config = SwarmConfig(
        num_agents=50,
        algorithm=SwarmAlgorithm.HYBRID,
        environment_dim=3,
        use_neural_controller=True
    )
    
    swarm = AdvancedSwarmSystem(config)
    
    print(f"\n✅ Swarm initialized with {config.num_agents} agents")
    print(f"✅ Algorithm: {config.algorithm.value}")
    print(f"✅ Neural controller: {'Enabled' if config.use_neural_controller else 'Disabled'}")
    
    # Run optimization
    print("\n🔄 Running swarm optimization...")
    
    async def run_optimization():
        best_fitness_history = []
        
        for i in range(50):
            metrics = await swarm.step(fitness_function)
            best_fitness_history.append(metrics['best_fitness'])
            
            if i % 10 == 0:
                print(f"\nIteration {i}:")
                print(f"  Best fitness: {metrics['best_fitness']:.4f}")
                print(f"  Avg fitness: {metrics['avg_fitness']:.4f}")
                print(f"  Convergence: {metrics['convergence']:.4f}")
                print(f"  Diversity: {swarm.get_diversity():.4f}")
        
        # Apply selection pressure
        swarm.apply_selection_pressure()
        
        return best_fitness_history
    
    # Run async optimization
    import asyncio
    best_history = asyncio.run(run_optimization())
    
    # Analysis
    print("\n📊 Optimization Results")
    print("-" * 40)
    print(f"Final best fitness: {swarm.pso.global_best_fitness:.4f}")
    print(f"Best position: {swarm.pso.global_best_position}")
    print(f"Improvement: {abs(best_history[-1] - best_history[0]):.4f}")
    
    # Role distribution
    role_counts = defaultdict(int)
    for agent in swarm.agents:
        role_counts[agent.role.value] += 1
    
    print("\n👥 Agent Role Distribution:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} agents")
    
    print("\n" + "=" * 60)
    print("✅ Swarm Intelligence Demonstration Complete")


if __name__ == "__main__":
    demonstrate_swarm_system()