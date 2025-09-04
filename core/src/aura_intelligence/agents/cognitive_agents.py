"""
Cognitive Agents for AURA - Real implementations
Perception, Planning, Analysis agents with memory integration
September 2025 - Production-ready agents
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
import numpy as np
from abc import ABC, abstractmethod

from ..schemas.aura_execution import (
    ExecutionPlan,
    ExecutionStep,
    ObservationResult,
    AgentDecision,
    MemoryContext
)
from ..memory.unified_cognitive_memory import UnifiedCognitiveMemory

logger = structlog.get_logger(__name__)


class BaseCognitiveAgent(ABC):
    """
    Base class for all cognitive agents in AURA.
    Provides common functionality and memory integration.
    """
    
    def __init__(
        self,
        agent_id: str,
        memory: Optional[UnifiedCognitiveMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base cognitive agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory: Memory system for context and learning
            config: Agent-specific configuration
        """
        self.agent_id = agent_id
        self.memory = memory
        self.config = config or {}
        
        # Agent state
        self.confidence = 0.8  # Default confidence
        self.experience_count = 0
        self.last_decision = None
        
        # Metrics
        self.metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "average_confidence": 0.8,
            "processing_time": 0.0
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {agent_id}")
    
    async def perceive(self, input_data: Any, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive and understand input.
        """
        return await self._perceive_impl(input_data, environment)
    
    @abstractmethod
    async def _perceive_impl(self, input_data: Any, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of perception logic.
        """
        pass
    
    async def vote(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vote on a proposal during consensus.
        """
        # Default implementation
        analysis = await self.analyze_proposal(proposal)
        
        return {
            "approve": analysis.get("risk_score", 0.5) < 0.7,
            "confidence": self.confidence,
            "reason": analysis.get("concerns", ""),
            "agent_id": self.agent_id
        }
    
    async def analyze_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a proposal for voting.
        """
        # Default analysis
        return {
            "risk_score": 0.5,
            "concerns": "",
            "suggestions": []
        }
    
    async def get_confidence(self) -> float:
        """
        Get current confidence level.
        """
        return self.confidence
    
    def update_confidence(self, success: bool):
        """
        Update confidence based on outcome.
        """
        if success:
            self.confidence = min(1.0, self.confidence * 1.1)
        else:
            self.confidence = max(0.1, self.confidence * 0.9)
        
        # Update average
        self.metrics["average_confidence"] = (
            self.metrics["average_confidence"] * 0.9 + self.confidence * 0.1
        )
    
    async def query_memory(self, query: str) -> Optional[MemoryContext]:
        """
        Query memory for relevant context.
        """
        if not self.memory:
            return None
        
        try:
            result = await self.memory.query(query)
            return MemoryContext(
                episodic_memories=result.get("episodic", []),
                semantic_concepts=result.get("semantic", []),
                causal_patterns=result.get("patterns", []),
                synthesis=result.get("synthesis", "")
            )
        except Exception as e:
            logger.warning(f"Memory query failed: {e}")
            return None


class PerceptionAgent(BaseCognitiveAgent):
    """
    Agent responsible for understanding tasks and gathering initial context.
    Specializes in task decomposition and requirement analysis.
    """
    
    def __init__(
        self,
        agent_id: str = "perception_agent_001",
        memory: Optional[UnifiedCognitiveMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id, memory, config)
        
        # Perception-specific configuration
        self.analysis_depth = config.get("analysis_depth", "medium") if config else "medium"
        self.context_window = config.get("context_window", 10) if config else 10
    
    async def _perceive_impl(self, input_data: Any, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement perception logic.
        """
        logger.info(f"PerceptionAgent analyzing: {input_data}")
        
        perception_result = {
            "understanding": "",
            "key_concepts": [],
            "complexity": "medium",
            "requirements": [],
            "constraints": [],
            "success_criteria": [],
            "context": {}
        }
        
        # Analyze the input (task description)
        if isinstance(input_data, str):
            task_text = input_data.lower()
            
            # Extract key concepts
            perception_result["key_concepts"] = self._extract_concepts(task_text)
            
            # Assess complexity
            perception_result["complexity"] = self._assess_complexity(task_text)
            
            # Extract requirements
            perception_result["requirements"] = self._extract_requirements(task_text)
            
            # Identify constraints from environment
            perception_result["constraints"] = self._identify_constraints(environment)
            
            # Define success criteria
            perception_result["success_criteria"] = self._define_success_criteria(
                task_text,
                perception_result["requirements"]
            )
            
            # Generate understanding
            perception_result["understanding"] = self._generate_understanding(
                task_text,
                perception_result["key_concepts"],
                perception_result["complexity"]
            )
        
        # Query memory for similar tasks
        if self.memory:
            memory_context = await self.query_memory(str(input_data))
            if memory_context:
                perception_result["context"]["similar_tasks"] = memory_context.episodic_memories[:5]
                perception_result["context"]["relevant_patterns"] = memory_context.causal_patterns
        
        # Update metrics
        self.metrics["total_decisions"] += 1
        self.experience_count += 1
        
        return perception_result
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.
        """
        concepts = []
        
        # Technical concepts
        technical_terms = [
            "performance", "memory", "cpu", "latency", "throughput",
            "error", "failure", "anomaly", "pattern", "topology",
            "optimization", "scaling", "monitoring", "analysis"
        ]
        
        # Action concepts
        action_terms = [
            "analyze", "detect", "prevent", "optimize", "monitor",
            "investigate", "diagnose", "fix", "improve", "measure"
        ]
        
        # Domain concepts
        domain_terms = [
            "system", "service", "database", "network", "application",
            "infrastructure", "cluster", "node", "container", "process"
        ]
        
        # Find present concepts
        for term in technical_terms + action_terms + domain_terms:
            if term in text:
                concepts.append(term)
        
        return concepts
    
    def _assess_complexity(self, text: str) -> str:
        """
        Assess task complexity.
        """
        # Count indicators of complexity
        complexity_score = 0
        
        # Complex keywords
        if any(word in text for word in ["complex", "advanced", "sophisticated", "detailed"]):
            complexity_score += 2
        
        # Multiple requirements
        if text.count("and") > 2:
            complexity_score += 1
        
        # Technical depth
        technical_count = sum(1 for word in ["topology", "causal", "pattern", "anomaly"] if word in text)
        complexity_score += technical_count
        
        # Determine complexity level
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _extract_requirements(self, text: str) -> List[str]:
        """
        Extract requirements from task description.
        """
        requirements = []
        
        # Look for requirement patterns
        if "analyze" in text:
            requirements.append("Perform analysis on target system")
        if "detect" in text or "identify" in text:
            requirements.append("Detect patterns or anomalies")
        if "prevent" in text:
            requirements.append("Implement preventive measures")
        if "optimize" in text:
            requirements.append("Optimize system performance")
        if "monitor" in text:
            requirements.append("Continuous monitoring required")
        
        # Look for specific metrics
        if "memory" in text:
            requirements.append("Monitor memory metrics")
        if "performance" in text:
            requirements.append("Measure performance indicators")
        if "error" in text or "failure" in text:
            requirements.append("Track error rates and failures")
        
        return requirements
    
    def _identify_constraints(self, environment: Dict[str, Any]) -> List[str]:
        """
        Identify constraints from environment.
        """
        constraints = []
        
        # Time constraints
        if environment.get("timeout"):
            constraints.append(f"Must complete within {environment['timeout']} seconds")
        
        # Resource constraints
        if environment.get("max_memory"):
            constraints.append(f"Memory limit: {environment['max_memory']}")
        
        # Access constraints
        if environment.get("read_only"):
            constraints.append("Read-only access to system")
        
        # Target constraints
        if environment.get("target"):
            constraints.append(f"Limited to target: {environment['target']}")
        
        return constraints
    
    def _define_success_criteria(self, text: str, requirements: List[str]) -> List[str]:
        """
        Define success criteria for the task.
        """
        criteria = []
        
        # Based on task type
        if "analyze" in text:
            criteria.append("Complete analysis with findings documented")
        if "detect" in text:
            criteria.append("Successfully identify any anomalies or patterns")
        if "prevent" in text:
            criteria.append("Implement measures to prevent identified issues")
        if "optimize" in text:
            criteria.append("Achieve measurable performance improvement")
        
        # Based on requirements
        if any("memory" in req for req in requirements):
            criteria.append("Memory usage patterns identified")
        if any("error" in req for req in requirements):
            criteria.append("Error sources identified and documented")
        
        # Default criteria
        if not criteria:
            criteria.append("Task completed without errors")
        
        return criteria
    
    def _generate_understanding(
        self,
        text: str,
        concepts: List[str],
        complexity: str
    ) -> str:
        """
        Generate a natural language understanding of the task.
        """
        understanding_parts = []
        
        # Task type
        if "analyze" in text:
            understanding_parts.append("This is an analysis task")
        elif "monitor" in text:
            understanding_parts.append("This is a monitoring task")
        elif "optimize" in text:
            understanding_parts.append("This is an optimization task")
        else:
            understanding_parts.append("This is a system operation task")
        
        # Complexity
        understanding_parts.append(f"with {complexity} complexity")
        
        # Focus areas
        if concepts:
            focus = ", ".join(concepts[:3])
            understanding_parts.append(f"focusing on {focus}")
        
        return " ".join(understanding_parts) + "."


class PlannerAgent(BaseCognitiveAgent):
    """
    Agent responsible for creating execution plans.
    Specializes in strategy formulation and resource allocation.
    """
    
    def __init__(
        self,
        agent_id: str = "planner_agent_001",
        memory: Optional[UnifiedCognitiveMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id, memory, config)
        
        # Planning-specific configuration
        self.max_steps = config.get("max_steps", 10) if config else 10
        self.risk_tolerance = config.get("risk_tolerance", 0.5) if config else 0.5
    
    async def _perceive_impl(self, input_data: Any, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement perception for planner (focuses on planning aspects).
        """
        return {
            "understanding": "Planning task",
            "key_concepts": ["plan", "strategy", "execution"],
            "complexity": "medium"
        }
    
    async def create_plan(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Create an execution plan for the goal.
        """
        logger.info(f"PlannerAgent creating plan for: {goal}")
        
        # Extract context components
        perception = context.get("perception", {})
        memory_context = context.get("memory", {})
        environment = context.get("environment", {})
        
        # Generate plan steps
        steps = await self._generate_steps(goal, perception, memory_context, environment)
        
        # Assess risks
        risk_assessment = self._assess_risks(steps, perception)
        
        # Determine if parallelization is possible
        can_parallelize = self._check_parallelization(steps)
        
        # Create the plan
        plan = ExecutionPlan(
            objective=goal,
            steps=steps,
            risk_assessment=risk_assessment,
            parallelization_possible=can_parallelize,
            estimated_duration=len(steps) * 10.0  # Simple estimate
        )
        
        # Update metrics
        self.metrics["total_decisions"] += 1
        
        return plan
    
    async def _generate_steps(
        self,
        goal: str,
        perception: Dict[str, Any],
        memory_context: Dict[str, Any],
        environment: Dict[str, Any]
    ) -> List[ExecutionStep]:
        """
        Generate execution steps for the plan.
        """
        steps = []
        
        # Always start with observation if it's an analysis task
        if any(word in goal.lower() for word in ["analyze", "monitor", "detect", "investigate"]):
            steps.append(ExecutionStep(
                tool="SystemObservationTool",
                params={
                    "target": environment.get("target", "system"),
                    "params": {
                        "duration": "15m",
                        "include_logs": True,
                        "include_events": True
                    }
                }
            ))
        
        # Add analysis step if patterns need to be found
        if "pattern" in goal.lower() or "anomaly" in goal.lower():
            if steps:
                # Depends on observation
                steps.append(ExecutionStep(
                    tool="PatternAnalysisTool",
                    params={"input": "observations"},
                    dependencies=[steps[0].step_id] if steps else []
                ))
        
        # Add optimization step if needed
        if "optimize" in goal.lower():
            steps.append(ExecutionStep(
                tool="OptimizationTool",
                params={
                    "target": environment.get("target", "system"),
                    "metrics": ["performance", "memory"]
                },
                dependencies=[s.step_id for s in steps]  # Depends on all previous
            ))
        
        # If no specific steps generated, create a basic observation
        if not steps:
            steps.append(ExecutionStep(
                tool="SystemObservationTool",
                params={
                    "target": "default",
                    "params": {"duration": "5m"}
                }
            ))
        
        return steps
    
    def _assess_risks(
        self,
        steps: List[ExecutionStep],
        perception: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Assess risks in the plan.
        """
        risks = {}
        
        # Base risk based on complexity
        complexity = perception.get("complexity", "medium")
        if complexity == "high":
            risks["complexity_risk"] = 0.7
        elif complexity == "medium":
            risks["complexity_risk"] = 0.4
        else:
            risks["complexity_risk"] = 0.2
        
        # Risk based on number of steps
        if len(steps) > 5:
            risks["execution_risk"] = 0.6
        elif len(steps) > 3:
            risks["execution_risk"] = 0.4
        else:
            risks["execution_risk"] = 0.2
        
        # Risk based on dependencies
        has_dependencies = any(step.dependencies for step in steps)
        if has_dependencies:
            risks["dependency_risk"] = 0.5
        else:
            risks["dependency_risk"] = 0.1
        
        # Overall risk
        risks["overall"] = np.mean(list(risks.values()))
        
        return risks
    
    def _check_parallelization(self, steps: List[ExecutionStep]) -> bool:
        """
        Check if steps can be parallelized.
        """
        # Can parallelize if no dependencies between steps
        for step in steps:
            if step.dependencies:
                return False
        return len(steps) > 1
    
    async def analyze_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a proposal for voting.
        """
        # Planner agents generally approve plans they created
        if proposal.get("plan_id"):
            # Check if this is our plan (simplified check)
            risk = proposal.get("risk_assessment", {}).get("overall", 0.5)
            
            return {
                "risk_score": risk,
                "concerns": "High risk detected" if risk > 0.7 else "",
                "suggestions": ["Consider risk mitigation"] if risk > 0.7 else []
            }
        
        return await super().analyze_proposal(proposal)


class AnalystAgent(BaseCognitiveAgent):
    """
    Agent responsible for analyzing patterns and results.
    Specializes in pattern recognition and insight generation.
    """
    
    def __init__(
        self,
        agent_id: str = "analyst_agent_001",
        memory: Optional[UnifiedCognitiveMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id, memory, config)
        
        # Analysis-specific configuration
        self.pattern_threshold = config.get("pattern_threshold", 0.7) if config else 0.7
        self.anomaly_sensitivity = config.get("anomaly_sensitivity", 0.5) if config else 0.5
    
    async def _perceive_impl(self, input_data: Any, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement perception for analyst (focuses on analysis aspects).
        """
        return {
            "understanding": "Analysis task",
            "key_concepts": ["analyze", "pattern", "insight"],
            "complexity": "high"
        }
    
    async def analyze_patterns(
        self,
        observations: List[ObservationResult]
    ) -> List[Dict[str, Any]]:
        """
        Analyze observations for patterns.
        """
        logger.info(f"AnalystAgent analyzing {len(observations)} observations")
        
        patterns = []
        
        # Analyze topological patterns
        topo_patterns = self._analyze_topology(observations)
        patterns.extend(topo_patterns)
        
        # Analyze anomalies
        anomaly_patterns = self._analyze_anomalies(observations)
        patterns.extend(anomaly_patterns)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal(observations)
        patterns.extend(temporal_patterns)
        
        # Query memory for similar patterns
        if self.memory and patterns:
            try:
                similar = await self.memory.query(f"patterns similar to {patterns[0]}")
                if similar:
                    patterns.append({
                        "type": "historical_match",
                        "description": "Similar patterns found in history",
                        "historical_context": similar
                    })
            except Exception as e:
                logger.warning(f"Memory query failed: {e}")
        
        # Update metrics
        self.metrics["total_decisions"] += 1
        
        return patterns
    
    def _analyze_topology(self, observations: List[ObservationResult]) -> List[Dict[str, Any]]:
        """
        Analyze topological patterns.
        """
        patterns = []
        
        for obs in observations:
            if obs.topology:
                # Check Betti numbers
                betti = obs.topology.betti_numbers
                
                if len(betti) > 0 and betti[0] > 5:
                    patterns.append({
                        "type": "topological_complexity",
                        "severity": "high",
                        "description": f"High connectivity detected (Betti-0: {betti[0]})",
                        "observation_id": obs.observation_id,
                        "recommendation": "Review system dependencies"
                    })
                
                if len(betti) > 1 and betti[1] > 3:
                    patterns.append({
                        "type": "cyclical_pattern",
                        "severity": "medium",
                        "description": f"Multiple loops detected (Betti-1: {betti[1]})",
                        "observation_id": obs.observation_id,
                        "recommendation": "Check for circular dependencies or resource cycles"
                    })
                
                # Check Wasserstein distance
                if obs.topology.wasserstein_distance_from_norm > 2.0:
                    patterns.append({
                        "type": "topology_deviation",
                        "severity": "medium",
                        "description": f"Significant deviation from baseline (distance: {obs.topology.wasserstein_distance_from_norm:.2f})",
                        "observation_id": obs.observation_id,
                        "recommendation": "Investigate recent changes"
                    })
        
        return patterns
    
    def _analyze_anomalies(self, observations: List[ObservationResult]) -> List[Dict[str, Any]]:
        """
        Analyze anomaly patterns.
        """
        patterns = []
        
        # Collect all anomalies
        all_anomalies = []
        for obs in observations:
            all_anomalies.extend(obs.anomalies)
        
        if not all_anomalies:
            return patterns
        
        # Group anomalies by type
        anomaly_types = {}
        for anomaly in all_anomalies:
            atype = anomaly.get("type", "unknown")
            if atype not in anomaly_types:
                anomaly_types[atype] = []
            anomaly_types[atype].append(anomaly)
        
        # Create patterns for grouped anomalies
        for atype, anomalies in anomaly_types.items():
            if len(anomalies) >= 2:
                patterns.append({
                    "type": "recurring_anomaly",
                    "anomaly_type": atype,
                    "count": len(anomalies),
                    "severity": "high" if len(anomalies) > 5 else "medium",
                    "description": f"Recurring {atype} anomalies detected",
                    "instances": anomalies[:5]  # First 5 instances
                })
        
        # Check for anomaly clusters
        if len(all_anomalies) > 10:
            patterns.append({
                "type": "anomaly_cluster",
                "severity": "critical",
                "description": f"Large cluster of {len(all_anomalies)} anomalies detected",
                "recommendation": "Immediate investigation required"
            })
        
        return patterns
    
    def _analyze_temporal(self, observations: List[ObservationResult]) -> List[Dict[str, Any]]:
        """
        Analyze temporal patterns.
        """
        patterns = []
        
        if len(observations) < 2:
            return patterns
        
        # Check for increasing trend in anomalies
        anomaly_counts = [len(obs.anomalies) for obs in observations]
        if len(anomaly_counts) > 2:
            # Simple trend detection
            first_half = np.mean(anomaly_counts[:len(anomaly_counts)//2])
            second_half = np.mean(anomaly_counts[len(anomaly_counts)//2:])
            
            if second_half > first_half * 1.5:
                patterns.append({
                    "type": "increasing_anomaly_trend",
                    "severity": "high",
                    "description": "Anomalies are increasing over time",
                    "trend_ratio": second_half / max(first_half, 0.1),
                    "recommendation": "Investigate escalating issues"
                })
        
        return patterns
    
    async def analyze_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a proposal for voting (Analysts are more critical).
        """
        risk_assessment = proposal.get("risk_assessment", {})
        overall_risk = risk_assessment.get("overall", 0.5)
        
        # Analysts are more conservative
        concerns = []
        if overall_risk > 0.6:
            concerns.append("Risk level exceeds acceptable threshold")
        
        if proposal.get("parallelization_possible"):
            concerns.append("Parallel execution may miss sequential dependencies")
        
        steps = proposal.get("steps", [])
        if len(steps) > 5:
            concerns.append("Plan complexity may lead to execution failures")
        
        return {
            "risk_score": overall_risk * 1.2,  # Analysts see more risk
            "concerns": "; ".join(concerns) if concerns else "",
            "suggestions": ["Add monitoring steps", "Include rollback plan"] if overall_risk > 0.5 else []
        }