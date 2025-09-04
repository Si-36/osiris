"""
🎯 Coordinator Agent - Swarm Intelligence Orchestration
======================================================

Specializes in:
- Multi-agent orchestration with GPU acceleration
- Byzantine-robust consensus
- Task decomposition and distribution
- Resource allocation and monitoring
- Swarm intelligence coordination
"""

import asyncio
import time
import json
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import structlog

from .test_agents import TestAgentBase, TestAgentConfig, Tool, AgentRole
from ..swarm_intelligence.swarm_coordinator import SwarmCoordinator
from ..consensus.swarm_consensus import SwarmConsensusProtocol
from ..adapters.swarm_adapter_gpu import GPUSwarmAdapter

logger = structlog.get_logger(__name__)


@dataclass
class TaskDecomposition:
    """Decomposed task structure"""
    task_id: str
    original_task: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]  # task_id -> [dependency_ids]
    agent_assignments: Dict[str, str]  # task_id -> agent_id
    estimated_time_ms: float
    priority: int = 1


@dataclass
class ConsensusResult:
    """Result of consensus operation"""
    consensus_id: str
    selected_result: Any
    consensus_quality: float
    byzantine_detection: List[str]  # List of potentially byzantine agents
    voting_details: Dict[str, Any]
    time_to_consensus_ms: float


@dataclass  
class CoordinationResult:
    """Result of coordination operation"""
    execution_plan: Dict[str, Any]
    results: Any
    performance_metrics: Dict[str, float]
    consensus_quality: float
    agents_involved: List[str]


class TaskDecomposer:
    """Decompose complex tasks into subtasks"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network for task complexity estimation
        self.complexity_estimator = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to(self.device)
        
    async def decompose(self, task: Dict[str, Any]) -> TaskDecomposition:
        """Decompose task into subtasks"""
        task_id = task.get("id", f"task_{int(time.time())}")
        task_type = task.get("type", "general")
        
        # Extract task features
        features = self._extract_task_features(task)
        
        # Estimate complexity
        complexity = await self._estimate_complexity(features)
        
        # Decompose based on type and complexity
        if task_type == "analysis":
            subtasks = self._decompose_analysis_task(task, complexity)
        elif task_type == "generation":
            subtasks = self._decompose_generation_task(task, complexity)
        elif task_type == "optimization":
            subtasks = self._decompose_optimization_task(task, complexity)
        else:
            subtasks = self._decompose_general_task(task, complexity)
            
        # Identify dependencies
        dependencies = self._identify_dependencies(subtasks)
        
        # Estimate time
        estimated_time = complexity * 100  # Base estimate
        
        return TaskDecomposition(
            task_id=task_id,
            original_task=task,
            subtasks=subtasks,
            dependencies=dependencies,
            agent_assignments={},
            estimated_time_ms=estimated_time,
            priority=task.get("priority", 1)
        )
        
    def _extract_task_features(self, task: Dict[str, Any]) -> torch.Tensor:
        """Extract features from task"""
        features = []
        
        # Task type features
        task_types = ["analysis", "generation", "optimization", "general"]
        task_type = task.get("type", "general")
        type_encoding = [1.0 if t == task_type else 0.0 for t in task_types]
        features.extend(type_encoding)
        
        # Size features
        data_size = len(str(task.get("data", "")))
        features.append(min(data_size / 10000, 1.0))  # Normalized size
        
        # Complexity indicators
        features.append(float(task.get("requires_gpu", False)))
        features.append(float(task.get("requires_consensus", False)))
        features.append(float(task.get("parallel", True)))
        
        # Resource requirements
        features.append(task.get("memory_gb", 1.0) / 10.0)
        features.append(task.get("timeout_seconds", 60) / 300.0)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
            
        return torch.tensor(features[:20], dtype=torch.float32, device=self.device)
        
    async def _estimate_complexity(self, features: torch.Tensor) -> float:
        """Estimate task complexity"""
        with torch.no_grad():
            complexity = self.complexity_estimator(features.unsqueeze(0)).item()
            
        # Normalize to 0-10 range
        return min(max(complexity, 0.1), 10.0)
        
    def _decompose_analysis_task(self, task: Dict[str, Any], complexity: float) -> List[Dict[str, Any]]:
        """Decompose analysis task"""
        subtasks = []
        
        # Data preparation
        subtasks.append({
            "id": f"{task.get('id', 'task')}_prep",
            "type": "data_preparation",
            "description": "Prepare and validate data",
            "agent_type": "data",
            "estimated_time_ms": 50
        })
        
        # Core analysis
        if complexity > 5:
            # Split into parallel analyses
            for i in range(int(complexity / 2)):
                subtasks.append({
                    "id": f"{task.get('id', 'task')}_analysis_{i}",
                    "type": "parallel_analysis",
                    "description": f"Analyze subset {i}",
                    "agent_type": "data",
                    "estimated_time_ms": 100
                })
        else:
            subtasks.append({
                "id": f"{task.get('id', 'task')}_analysis",
                "type": "analysis",
                "description": "Perform core analysis",
                "agent_type": "data",
                "estimated_time_ms": 150
            })
            
        # Aggregation
        subtasks.append({
            "id": f"{task.get('id', 'task')}_aggregate",
            "type": "aggregation",
            "description": "Aggregate results",
            "agent_type": "architect",
            "estimated_time_ms": 50
        })
        
        return subtasks
        
    def _decompose_generation_task(self, task: Dict[str, Any], complexity: float) -> List[Dict[str, Any]]:
        """Decompose generation task"""
        subtasks = []
        
        # Planning
        subtasks.append({
            "id": f"{task.get('id', 'task')}_plan",
            "type": "planning",
            "description": "Plan generation approach",
            "agent_type": "architect",
            "estimated_time_ms": 75
        })
        
        # Generation variants
        num_variants = max(3, int(complexity))
        for i in range(num_variants):
            subtasks.append({
                "id": f"{task.get('id', 'task')}_gen_{i}",
                "type": "generation",
                "description": f"Generate variant {i}",
                "agent_type": "creative",
                "estimated_time_ms": 200
            })
            
        # Quality assessment
        subtasks.append({
            "id": f"{task.get('id', 'task')}_quality",
            "type": "quality_check",
            "description": "Assess quality and select best",
            "agent_type": "code",
            "estimated_time_ms": 100
        })
        
        return subtasks
        
    def _decompose_optimization_task(self, task: Dict[str, Any], complexity: float) -> List[Dict[str, Any]]:
        """Decompose optimization task"""
        subtasks = []
        
        # Baseline measurement
        subtasks.append({
            "id": f"{task.get('id', 'task')}_baseline",
            "type": "measurement",
            "description": "Measure baseline performance",
            "agent_type": "code",
            "estimated_time_ms": 100
        })
        
        # Optimization strategies
        strategies = ["algorithmic", "architectural", "resource"]
        for strategy in strategies[:int(complexity / 3) + 1]:
            subtasks.append({
                "id": f"{task.get('id', 'task')}_opt_{strategy}",
                "type": "optimization",
                "description": f"Apply {strategy} optimization",
                "agent_type": "architect" if strategy == "architectural" else "code",
                "estimated_time_ms": 150
            })
            
        # Validation
        subtasks.append({
            "id": f"{task.get('id', 'task')}_validate",
            "type": "validation",
            "description": "Validate optimizations",
            "agent_type": "data",
            "estimated_time_ms": 100
        })
        
        return subtasks
        
    def _decompose_general_task(self, task: Dict[str, Any], complexity: float) -> List[Dict[str, Any]]:
        """Decompose general task"""
        # Simple sequential decomposition
        num_steps = max(2, min(5, int(complexity / 2)))
        
        subtasks = []
        for i in range(num_steps):
            subtasks.append({
                "id": f"{task.get('id', 'task')}_step_{i}",
                "type": "processing",
                "description": f"Process step {i}",
                "agent_type": "general",
                "estimated_time_ms": 100
            })
            
        return subtasks
        
    def _identify_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Identify task dependencies"""
        dependencies = {}
        
        for i, task in enumerate(subtasks):
            task_id = task["id"]
            dependencies[task_id] = []
            
            # Sequential dependencies by default
            if i > 0:
                prev_task = subtasks[i-1]
                
                # Parallel tasks don't depend on each other
                if "parallel" not in task["type"] or "parallel" not in prev_task["type"]:
                    dependencies[task_id].append(prev_task["id"])
                    
            # Special dependencies
            if "aggregate" in task["type"] or "quality" in task["type"]:
                # Depends on all previous parallel tasks
                for prev_task in subtasks[:i]:
                    if "parallel" in prev_task["type"] or "gen_" in prev_task["id"]:
                        if prev_task["id"] not in dependencies[task_id]:
                            dependencies[task_id].append(prev_task["id"])
                            
        return dependencies


class ByzantineConsensusEngine:
    """Byzantine-robust consensus for multi-agent decisions"""
    
    def __init__(self, byzantine_threshold: int = 2):
        self.byzantine_threshold = byzantine_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network for vote quality assessment
        self.vote_assessor = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
    async def coordinate_execution(self,
                                 assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate execution with Byzantine consensus"""
        consensus_id = f"consensus_{int(time.time())}"
        start_time = time.perf_counter()
        
        # Collect votes from agents
        votes = await self._collect_votes(assignments)
        
        # Assess vote quality
        vote_qualities = await self._assess_vote_quality(votes)
        
        # Detect Byzantine behavior
        byzantine_agents = self._detect_byzantine_behavior(votes, vote_qualities)
        
        # Aggregate votes (excluding Byzantine)
        final_result = self._aggregate_votes(votes, byzantine_agents)
        
        # Calculate consensus quality
        consensus_quality = self._calculate_consensus_quality(votes, byzantine_agents)
        
        time_to_consensus = (time.perf_counter() - start_time) * 1000
        
        return {
            "consensus_id": consensus_id,
            "result": final_result,
            "byzantine_agents": byzantine_agents,
            "consensus_quality": consensus_quality,
            "time_to_consensus_ms": time_to_consensus,
            "vote_count": len(votes),
            "valid_votes": len(votes) - len(byzantine_agents)
        }
        
    async def _collect_votes(self, assignments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect votes from assigned agents"""
        # In practice, would collect via NATS
        # For now, simulate votes
        votes = []
        
        for agent_id, task in assignments.items():
            vote = {
                "agent_id": agent_id,
                "result": f"Result from {agent_id}",
                "confidence": np.random.uniform(0.5, 1.0),
                "features": np.random.randn(10)  # Vote features
            }
            votes.append(vote)
            
        return votes
        
    async def _assess_vote_quality(self, votes: List[Dict[str, Any]]) -> List[float]:
        """Assess quality of each vote"""
        qualities = []
        
        for vote in votes:
            # Extract features
            features = torch.tensor(vote["features"], dtype=torch.float32, device=self.device)
            
            # Assess quality
            with torch.no_grad():
                quality = self.vote_assessor(features.unsqueeze(0)).item()
                
            qualities.append(quality * vote["confidence"])
            
        return qualities
        
    def _detect_byzantine_behavior(self,
                                  votes: List[Dict[str, Any]],
                                  qualities: List[float]) -> List[str]:
        """Detect potentially Byzantine agents"""
        byzantine_agents = []
        
        if len(votes) < 3:
            return byzantine_agents
            
        # Statistical outlier detection
        quality_array = np.array(qualities)
        mean_quality = np.mean(quality_array)
        std_quality = np.std(quality_array)
        
        # Z-score based detection
        for i, (vote, quality) in enumerate(zip(votes, qualities)):
            z_score = abs(quality - mean_quality) / (std_quality + 1e-6)
            
            if z_score > 2.5:  # Outlier threshold
                byzantine_agents.append(vote["agent_id"])
                
        # Consistency check
        # In practice, would check vote consistency across multiple rounds
        
        return byzantine_agents[:self.byzantine_threshold]  # Limit detections
        
    def _aggregate_votes(self,
                        votes: List[Dict[str, Any]],
                        byzantine_agents: List[str]) -> Any:
        """Aggregate votes excluding Byzantine agents"""
        valid_votes = [v for v in votes if v["agent_id"] not in byzantine_agents]
        
        if not valid_votes:
            # All Byzantine - use all votes
            valid_votes = votes
            
        # Simple majority for demonstration
        # In practice, would use more sophisticated aggregation
        results = {}
        for vote in valid_votes:
            result = vote["result"]
            if result not in results:
                results[result] = 0
            results[result] += vote["confidence"]
            
        # Return highest confidence result
        if results:
            return max(results.items(), key=lambda x: x[1])[0]
        else:
            return None
            
    def _calculate_consensus_quality(self,
                                   votes: List[Dict[str, Any]],
                                   byzantine_agents: List[str]) -> float:
        """Calculate quality of consensus"""
        if not votes:
            return 0.0
            
        valid_votes = [v for v in votes if v["agent_id"] not in byzantine_agents]
        
        if not valid_votes:
            return 0.0
            
        # Agreement level
        confidences = [v["confidence"] for v in valid_votes]
        avg_confidence = np.mean(confidences)
        
        # Byzantine ratio
        byzantine_ratio = len(byzantine_agents) / len(votes)
        
        # Consensus quality combines agreement and Byzantine detection
        quality = avg_confidence * (1 - byzantine_ratio)
        
        return float(quality)
        
    def get_quality_score(self) -> float:
        """Get overall consensus quality score"""
        # Would track historical quality
        return 0.85  # Placeholder


class TopologyAwareLoadBalancer:
    """Load balancer that considers system topology"""
    
    def __init__(self):
        self.agent_loads = defaultdict(float)
        self.agent_capabilities = {}
        
    def assign_tasks(self,
                    subtasks: List[Dict[str, Any]],
                    agents: List[Any]) -> Dict[str, Any]:
        """Assign tasks to agents based on topology and load"""
        assignments = {}
        
        # Group agents by type
        agent_groups = defaultdict(list)
        for agent in agents:
            agent_type = getattr(agent, 'specialty', 'general')
            agent_groups[agent_type].append(agent)
            
        # Assign tasks
        for task in subtasks:
            required_type = task.get("agent_type", "general")
            
            # Find suitable agents
            suitable_agents = agent_groups.get(required_type, [])
            if not suitable_agents:
                suitable_agents = agent_groups.get("general", agents)
                
            # Select least loaded agent
            if suitable_agents:
                selected_agent = min(
                    suitable_agents,
                    key=lambda a: self.agent_loads[a.state.agent_id]
                )
                
                assignments[task["id"]] = {
                    "agent_id": selected_agent.state.agent_id,
                    "agent": selected_agent,
                    "task": task
                }
                
                # Update load
                self.agent_loads[selected_agent.state.agent_id] += task.get("estimated_time_ms", 100)
                
        return assignments
        
    def update_load(self, agent_id: str, completed_time_ms: float):
        """Update agent load after task completion"""
        self.agent_loads[agent_id] = max(0, self.agent_loads[agent_id] - completed_time_ms)


class CoordinatorAgent(TestAgentBase):
    """
    Specialized agent for multi-agent coordination.
    
    Capabilities:
    - Task decomposition
    - Agent assignment
    - Byzantine consensus
    - Progress monitoring
    - Resource optimization
    """
    
    def __init__(self, agent_id: str = "coordinator_agent_001", **kwargs):
        config = TestAgentConfig(
            agent_role=AgentRole.COORDINATOR,
            specialty="coordinator",
            target_latency_ms=50.0,  # Low latency for coordination
            consensus_threshold=0.7,
            byzantine_tolerance=2,
            **kwargs
        )
        
        super().__init__(agent_id=agent_id, config=config, **kwargs)
        
        # Initialize specialized components
        self.task_decomposer = TaskDecomposer()
        self.consensus_engine = ByzantineConsensusEngine(
            byzantine_threshold=config.byzantine_tolerance
        )
        self.load_balancer = TopologyAwareLoadBalancer()
        
        # Swarm components
        self.swarm_coordinator = SwarmCoordinator()
        self.swarm_consensus = SwarmConsensusProtocol()
        
        # Agent registry
        self.registered_agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, str] = {}  # agent_id -> status
        
        # Task tracking
        self.active_tasks: Dict[str, TaskDecomposition] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Initialize tools
        self._init_coordinator_tools()
        
        logger.info("Coordinator Agent initialized",
                   agent_id=agent_id,
                   capabilities=["task_decomposition", "byzantine_consensus", 
                               "swarm_coordination", "load_balancing"])
                   
    def _init_coordinator_tools(self):
        """Initialize coordinator-specific tools"""
        self.tools = {
            "decompose_task": Tool(
                name="decompose_task",
                description="Decompose complex task",
                func=self._tool_decompose_task
            ),
            "assign_agents": Tool(
                name="assign_agents",
                description="Assign tasks to agents",
                func=self._tool_assign_agents
            ),
            "coordinate_consensus": Tool(
                name="coordinate_consensus",
                description="Coordinate Byzantine consensus",
                func=self._tool_coordinate_consensus
            ),
            "monitor_progress": Tool(
                name="monitor_progress",
                description="Monitor task progress",
                func=self._tool_monitor_progress
            ),
            "optimize_resources": Tool(
                name="optimize_resources",
                description="Optimize resource allocation",
                func=self._tool_optimize_resources
            )
        }
        
    def register_agent(self, agent: Any):
        """Register an agent for coordination"""
        agent_id = agent.state.agent_id
        self.registered_agents[agent_id] = agent
        self.agent_status[agent_id] = "available"
        
        logger.info(f"Registered agent: {agent_id}",
                   specialty=getattr(agent, 'specialty', 'general'))
                   
    async def _handle_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coordination analysis requests"""
        task = context.get("original", {})
        
        # Analyze task complexity
        decomposition = await self.task_decomposer.decompose(task)
        
        # Analyze agent availability
        available_agents = [
            agent_id for agent_id, status in self.agent_status.items()
            if status == "available"
        ]
        
        # Analyze system load
        system_load = {
            "active_tasks": len(self.active_tasks),
            "available_agents": len(available_agents),
            "total_agents": len(self.registered_agents),
            "avg_load": np.mean(list(self.load_balancer.agent_loads.values())) if self.load_balancer.agent_loads else 0
        }
        
        return {
            "task_complexity": len(decomposition.subtasks),
            "estimated_time_ms": decomposition.estimated_time_ms,
            "required_agents": len(set(t["agent_type"] for t in decomposition.subtasks)),
            "system_load": system_load,
            "feasibility": "high" if len(available_agents) >= len(decomposition.subtasks) else "medium"
        }
        
    async def _handle_generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coordination plan generation"""
        task = context.get("original", {})
        
        # Generate execution plan
        plan = await self._generate_execution_plan(task)
        
        return plan
        
    async def coordinate_swarm(self,
                             agents: List[Any],
                             task: Dict[str, Any]) -> CoordinationResult:
        """Orchestrate multi-agent swarm execution"""
        start_time = time.perf_counter()
        
        # Decompose task
        decomposition = await self.task_decomposer.decompose(task)
        
        # Store active task
        self.active_tasks[decomposition.task_id] = decomposition
        
        # Assign subtasks to agents
        assignments = self.load_balancer.assign_tasks(
            decomposition.subtasks,
            agents
        )
        
        # Update decomposition with assignments
        for task_id, assignment in assignments.items():
            decomposition.agent_assignments[task_id] = assignment["agent_id"]
            
        # Execute with monitoring
        results = await self._execute_with_monitoring(assignments, decomposition)
        
        # Aggregate results
        final_result = await self._aggregate_results(results, decomposition)
        
        # Run consensus if needed
        if task.get("requires_consensus", False):
            consensus_result = await self.consensus_engine.coordinate_execution(assignments)
            consensus_quality = consensus_result["consensus_quality"]
        else:
            consensus_quality = 1.0
            
        # Calculate metrics
        execution_time = (time.perf_counter() - start_time) * 1000
        
        performance_metrics = {
            "total_time_ms": execution_time,
            "subtask_count": len(decomposition.subtasks),
            "parallel_efficiency": self._calculate_parallel_efficiency(decomposition, execution_time),
            "resource_utilization": self._calculate_resource_utilization()
        }
        
        # Clean up
        del self.active_tasks[decomposition.task_id]
        
        return CoordinationResult(
            execution_plan=assignments,
            results=final_result,
            performance_metrics=performance_metrics,
            consensus_quality=consensus_quality,
            agents_involved=list(decomposition.agent_assignments.values())
        )
        
    async def _generate_execution_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed execution plan"""
        # Decompose task
        decomposition = await self.task_decomposer.decompose(task)
        
        # Simulate agent assignment
        agent_types = ["code", "data", "creative", "architect"]
        assignments = {}
        
        for subtask in decomposition.subtasks:
            required_type = subtask.get("agent_type", "general")
            
            # Find matching agent type
            if required_type in agent_types:
                agent_id = f"{required_type}_agent_001"
            else:
                agent_id = f"general_agent_{np.random.randint(1, 5):03d}"
                
            assignments[subtask["id"]] = agent_id
            
        # Create execution timeline
        timeline = self._create_execution_timeline(decomposition, assignments)
        
        return {
            "task_id": decomposition.task_id,
            "subtasks": decomposition.subtasks,
            "dependencies": decomposition.dependencies,
            "assignments": assignments,
            "timeline": timeline,
            "estimated_duration_ms": decomposition.estimated_time_ms,
            "parallelism_factor": self._calculate_parallelism(decomposition)
        }
        
    def _create_execution_timeline(self,
                                 decomposition: TaskDecomposition,
                                 assignments: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create execution timeline respecting dependencies"""
        timeline = []
        completed = set()
        time = 0
        
        while len(completed) < len(decomposition.subtasks):
            # Find ready tasks
            ready_tasks = []
            for subtask in decomposition.subtasks:
                task_id = subtask["id"]
                
                if task_id in completed:
                    continue
                    
                # Check dependencies
                deps = decomposition.dependencies.get(task_id, [])
                if all(dep in completed for dep in deps):
                    ready_tasks.append(subtask)
                    
            if not ready_tasks:
                break  # Circular dependency
                
            # Schedule ready tasks
            for task in ready_tasks:
                timeline.append({
                    "task_id": task["id"],
                    "agent_id": assignments.get(task["id"]),
                    "start_time_ms": time,
                    "duration_ms": task.get("estimated_time_ms", 100)
                })
                completed.add(task["id"])
                
            # Advance time by the longest task in this batch
            if ready_tasks:
                max_duration = max(t.get("estimated_time_ms", 100) for t in ready_tasks)
                time += max_duration
                
        return timeline
        
    def _calculate_parallelism(self, decomposition: TaskDecomposition) -> float:
        """Calculate potential parallelism in task decomposition"""
        if not decomposition.subtasks:
            return 1.0
            
        # Count maximum parallel tasks at each level
        levels = self._get_dependency_levels(decomposition)
        
        if not levels:
            return 1.0
            
        max_parallel = max(len(level) for level in levels.values())
        
        return min(max_parallel / len(decomposition.subtasks), 1.0)
        
    def _get_dependency_levels(self, decomposition: TaskDecomposition) -> Dict[int, List[str]]:
        """Get tasks organized by dependency level"""
        levels = defaultdict(list)
        task_levels = {}
        
        # Topological sort to find levels
        visited = set()
        
        def get_level(task_id):
            if task_id in task_levels:
                return task_levels[task_id]
                
            deps = decomposition.dependencies.get(task_id, [])
            
            if not deps:
                level = 0
            else:
                level = max(get_level(dep) for dep in deps) + 1
                
            task_levels[task_id] = level
            levels[level].append(task_id)
            
            return level
            
        for subtask in decomposition.subtasks:
            get_level(subtask["id"])
            
        return dict(levels)
        
    async def _execute_with_monitoring(self,
                                     assignments: Dict[str, Any],
                                     decomposition: TaskDecomposition) -> Dict[str, Any]:
        """Execute tasks with progress monitoring"""
        results = {}
        
        # Group by dependency level for parallel execution
        levels = self._get_dependency_levels(decomposition)
        
        for level, task_ids in sorted(levels.items()):
            # Execute tasks at this level in parallel
            level_tasks = []
            
            for task_id in task_ids:
                if task_id in assignments:
                    assignment = assignments[task_id]
                    agent = assignment["agent"]
                    task = assignment["task"]
                    
                    # Mark agent as busy
                    self.agent_status[assignment["agent_id"]] = "busy"
                    
                    # Execute task
                    level_tasks.append(self._execute_single_task(agent, task))
                    
            # Wait for level completion
            if level_tasks:
                level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
                
                for i, task_id in enumerate(task_ids):
                    if i < len(level_results):
                        results[task_id] = level_results[i]
                        
                        # Mark agent as available
                        if task_id in assignments:
                            agent_id = assignments[task_id]["agent_id"]
                            self.agent_status[agent_id] = "available"
                            
                            # Update load
                            task_time = decomposition.subtasks[i].get("estimated_time_ms", 100)
                            self.load_balancer.update_load(agent_id, task_time)
                            
        return results
        
    async def _execute_single_task(self, agent: Any, task: Dict[str, Any]) -> Any:
        """Execute single task on agent"""
        try:
            # Send task to agent
            result = await agent.process_message({
                "type": task.get("type", "general"),
                "payload": task
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}",
                        agent_id=agent.state.agent_id,
                        task_id=task.get("id"))
            return {"error": str(e)}
            
    async def _aggregate_results(self,
                               results: Dict[str, Any],
                               decomposition: TaskDecomposition) -> Any:
        """Aggregate results from subtasks"""
        # Simple aggregation - in practice would be task-specific
        
        if not results:
            return {"status": "no_results"}
            
        # Check for errors
        errors = [r for r in results.values() if isinstance(r, dict) and "error" in r]
        
        if errors:
            return {
                "status": "partial_failure",
                "errors": errors,
                "successful": len(results) - len(errors)
            }
            
        # Aggregate based on task type
        task_type = decomposition.original_task.get("type", "general")
        
        if task_type == "analysis":
            # Combine analysis results
            combined = {
                "combined_results": list(results.values()),
                "subtask_count": len(results)
            }
        elif task_type == "generation":
            # Select best generation
            # In practice, would use quality metrics
            combined = {
                "selected_result": list(results.values())[0],
                "alternatives": list(results.values())[1:]
            }
        else:
            # Generic aggregation
            combined = {
                "results": results,
                "summary": f"Completed {len(results)} subtasks"
            }
            
        return combined
        
    def _calculate_parallel_efficiency(self,
                                     decomposition: TaskDecomposition,
                                     actual_time: float) -> float:
        """Calculate parallel execution efficiency"""
        # Sequential time estimate
        sequential_time = sum(
            task.get("estimated_time_ms", 100)
            for task in decomposition.subtasks
        )
        
        # Parallel efficiency
        if actual_time > 0:
            efficiency = sequential_time / actual_time
        else:
            efficiency = 1.0
            
        # Normalize to 0-1
        return min(efficiency / len(self.registered_agents), 1.0)
        
    def _calculate_resource_utilization(self) -> float:
        """Calculate overall resource utilization"""
        if not self.agent_status:
            return 0.0
            
        busy_agents = sum(1 for status in self.agent_status.values() if status == "busy")
        
        return busy_agents / len(self.agent_status)
        
    # Tool implementations
    async def _tool_decompose_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose task tool"""
        decomposition = await self.task_decomposer.decompose(task)
        
        return {
            "task_id": decomposition.task_id,
            "subtask_count": len(decomposition.subtasks),
            "subtasks": decomposition.subtasks,
            "dependencies": decomposition.dependencies,
            "estimated_time_ms": decomposition.estimated_time_ms
        }
        
    async def _tool_assign_agents(self,
                                subtasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assign agents tool"""
        # Get available agents
        available_agents = [
            self.registered_agents[agent_id]
            for agent_id, status in self.agent_status.items()
            if status == "available"
        ]
        
        if not available_agents:
            return {"error": "No available agents"}
            
        # Assign tasks
        assignments = self.load_balancer.assign_tasks(subtasks, available_agents)
        
        # Extract agent assignments
        agent_assignments = {
            task_id: assignment["agent_id"]
            for task_id, assignment in assignments.items()
        }
        
        return agent_assignments
        
    async def _tool_coordinate_consensus(self,
                                       votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate consensus tool"""
        # Create dummy assignments for consensus
        assignments = {
            vote["agent_id"]: vote
            for vote in votes
        }
        
        consensus_result = await self.consensus_engine.coordinate_execution(assignments)
        
        return consensus_result
        
    async def _tool_monitor_progress(self) -> Dict[str, Any]:
        """Monitor progress tool"""
        progress = {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_results),
            "agent_status": dict(self.agent_status),
            "system_load": {
                agent_id: load
                for agent_id, load in self.load_balancer.agent_loads.items()
                if load > 0
            }
        }
        
        # Task details
        task_progress = []
        for task_id, decomposition in self.active_tasks.items():
            completed_subtasks = sum(
                1 for st_id in decomposition.agent_assignments
                if st_id in self.task_results
            )
            
            task_progress.append({
                "task_id": task_id,
                "progress": completed_subtasks / len(decomposition.subtasks),
                "completed": completed_subtasks,
                "total": len(decomposition.subtasks)
            })
            
        progress["task_progress"] = task_progress
        
        return progress
        
    async def _tool_optimize_resources(self) -> Dict[str, Any]:
        """Optimize resources tool"""
        # Analyze current allocation
        total_load = sum(self.load_balancer.agent_loads.values())
        avg_load = total_load / max(len(self.registered_agents), 1)
        
        # Find imbalances
        overloaded = []
        underutilized = []
        
        for agent_id, load in self.load_balancer.agent_loads.items():
            if load > avg_load * 1.5:
                overloaded.append({
                    "agent_id": agent_id,
                    "load": load,
                    "excess": load - avg_load
                })
            elif load < avg_load * 0.5:
                underutilized.append({
                    "agent_id": agent_id,
                    "load": load,
                    "capacity": avg_load - load
                })
                
        recommendations = []
        
        if overloaded:
            recommendations.append({
                "type": "rebalance",
                "description": f"Rebalance load from {len(overloaded)} overloaded agents",
                "agents": overloaded
            })
            
        if underutilized and self.active_tasks:
            recommendations.append({
                "type": "utilize",
                "description": f"Utilize {len(underutilized)} underused agents",
                "agents": underutilized
            })
            
        return {
            "total_load": total_load,
            "average_load": avg_load,
            "load_variance": np.var(list(self.load_balancer.agent_loads.values())),
            "recommendations": recommendations
        }


# Factory function
def create_coordinator_agent(agent_id: Optional[str] = None, **kwargs) -> CoordinatorAgent:
    """Create a Coordinator Agent instance"""
    if agent_id is None:
        agent_id = f"coordinator_agent_{int(time.time())}"
        
    return CoordinatorAgent(agent_id=agent_id, **kwargs)