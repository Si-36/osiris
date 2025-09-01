"""
Swarm Intelligence - Clean Implementation
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random
import asyncio
import numpy as np

class SwarmAlgorithm(Enum):
    PSO = "pso"  # Particle Swarm Optimization
    ACO = "aco"  # Ant Colony Optimization
    BEE = "bee"  # Bee Algorithm

@dataclass
class Particle:
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float

class SwarmCoordinator:
    """Swarm intelligence coordinator"""
    
    def __init__(self, algorithm: str = "pso", n_particles: int = 20):
        self.algorithm = SwarmAlgorithm(algorithm)
        self.n_particles = n_particles
        self.particles: List[Particle] = []
        self.global_best_position: Optional[List[float]] = None
        self.global_best_fitness: float = float('-inf')
        
    async def optimize(self, 
                      fitness_func: Callable,
                      dimensions: int,
                      iterations: int = 100) -> Dict[str, Any]:
        """Run swarm optimization"""
        
        if self.algorithm == SwarmAlgorithm.PSO:
            return await self._run_pso(fitness_func, dimensions, iterations)
        elif self.algorithm == SwarmAlgorithm.ACO:
            return await self._run_aco(fitness_func, dimensions, iterations)
        else:
            return await self._run_bee(fitness_func, dimensions, iterations)
            
    async def _run_pso(self, fitness_func: Callable, dimensions: int, iterations: int) -> Dict[str, Any]:
        """Particle Swarm Optimization"""
        # Initialize particles
        self.particles = []
        for _ in range(self.n_particles):
            position = [random.uniform(-10, 10) for _ in range(dimensions)]
            velocity = [random.uniform(-1, 1) for _ in range(dimensions)]
            
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('-inf')
            )
            self.particles.append(particle)
            
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Run iterations
        for iteration in range(iterations):
            for particle in self.particles:
                # Evaluate fitness
                if asyncio.iscoroutinefunction(fitness_func):
                    fitness = await fitness_func(particle.position)
                else:
                    fitness = fitness_func(particle.position)
                    
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                    
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    
            # Update velocities and positions
            for particle in self.particles:
                for i in range(dimensions):
                    r1, r2 = random.random(), random.random()
                    
                    # Velocity update
                    particle.velocity[i] = (
                        w * particle.velocity[i] +
                        c1 * r1 * (particle.best_position[i] - particle.position[i]) +
                        c2 * r2 * (self.global_best_position[i] - particle.position[i])
                    )
                    
                    # Position update
                    particle.position[i] += particle.velocity[i]
                    
        return {
            "algorithm": "PSO",
            "best_position": self.global_best_position,
            "best_fitness": self.global_best_fitness,
            "iterations": iterations
        }
        
    async def _run_aco(self, fitness_func: Callable, dimensions: int, iterations: int) -> Dict[str, Any]:
        """Ant Colony Optimization (simplified)"""
        # Mock ACO implementation
        best_solution = [random.uniform(-10, 10) for _ in range(dimensions)]
        
        if asyncio.iscoroutinefunction(fitness_func):
            best_fitness = await fitness_func(best_solution)
        else:
            best_fitness = fitness_func(best_solution)
            
        return {
            "algorithm": "ACO",
            "best_position": best_solution,
            "best_fitness": best_fitness,
            "iterations": iterations
        }
        
    async def _run_bee(self, fitness_func: Callable, dimensions: int, iterations: int) -> Dict[str, Any]:
        """Bee Algorithm (simplified)"""
        # Mock Bee implementation
        best_solution = [random.uniform(-10, 10) for _ in range(dimensions)]
        
        if asyncio.iscoroutinefunction(fitness_func):
            best_fitness = await fitness_func(best_solution)
        else:
            best_fitness = fitness_func(best_solution)
            
        return {
            "algorithm": "BEE",
            "best_position": best_solution,
            "best_fitness": best_fitness,
            "iterations": iterations
        }
        
    def coordinate_agents(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, List[str]]:
        """Coordinate agents for a task using swarm principles"""
        # Simple task allocation based on swarm principles
        n_subtasks = task.get("subtasks", 3)
        
        assignments = {}
        for i in range(n_subtasks):
            # Assign agents in round-robin with some randomness
            assigned = []
            for j, agent in enumerate(agents):
                if random.random() > 0.3:  # 70% chance of assignment
                    assigned.append(agent)
                    
            assignments[f"subtask_{i}"] = assigned
            
        return assignments