"""
Cognitive Agent Base Class - Memory-Aware Agent Implementation
==============================================================

This is the production implementation that connects agents to the
UnifiedCognitiveMemory system, enabling true learning and reasoning.

Based on September 2025 best practices:
- Full perceive-think-act cognitive loop
- Continuous learning through experience
- Causal pattern tracking for failure prevention
- Memory consolidation during operation
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import hashlib
import numpy as np
import structlog
from abc import ABC, abstractmethod

# Import the complete memory system
from ..memory.unified_cognitive_memory import UnifiedCognitiveMemory, MemoryContext
from ..memory.episodic_memory import Episode
from ..memory.semantic_memory import Concept

# Import agent components
from .agent_core import AURAAgentState, AURAAgentCore as BaseAURAAgent

# Import supporting systems
from ..memory.core.causal_tracker import CausalPatternTracker, CausalPattern, OutcomeType

logger = structlog.get_logger(__name__)


# ==================== Data Structures ====================

@dataclass
class AgentExperience:
    """
    Represents an agent's experience to be stored in memory
    """
    agent_id: str
    experience_type: str  # perception, action, outcome, thought
    content: Any
    
    # Context for episodic storage
    importance: float = 0.5
    emotional_valence: float = 0.0  # -1 to 1
    surprise_score: float = 0.0
    
    # Causal tracking
    causal_trigger: Optional[str] = None
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def to_memory_format(self) -> Dict[str, Any]:
        """Convert to format expected by memory system"""
        return {
            'content': self.content,
            'context': {
                'importance': self.importance,
                'surprise': self.surprise_score,
                'emotional': {
                    'valence': self.emotional_valence,
                    'arousal': 0.5,
                    'dominance': 0.5
                },
                'spatial': {
                    'location_id': f'agent_{self.agent_id}_workspace'
                },
                'social': {
                    'participants': [self.agent_id]
                },
                'causal': {
                    'trigger_episodes': [self.causal_trigger] if self.causal_trigger else []
                }
            }
        }


@dataclass
class AgentDecision:
    """
    Represents a decision made by the agent
    """
    decision_id: str
    decision_type: str
    action: Dict[str, Any]
    reasoning: str
    confidence: float
    
    # Memory context that informed this decision
    memory_sources: List[str] = field(default_factory=list)
    episodic_support: List[Episode] = field(default_factory=list)
    semantic_support: List[Concept] = field(default_factory=list)
    
    # Predicted outcome
    predicted_outcome: Optional[str] = None
    predicted_success_probability: float = 0.5


# ==================== Cognitive Agent Base Class ====================

class CognitiveAgent(BaseAURAAgent):
    """
    Advanced agent base class with full cognitive memory integration
    
    This implements the complete perceive-think-act loop with:
    - Experience recording in episodic memory
    - Knowledge extraction to semantic memory
    - Causal pattern learning for failure prevention
    - Continuous consolidation during operation
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        memory_system: UnifiedCognitiveMemory,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cognitive agent with memory system
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (planner, executor, analyst, etc.)
            memory_system: The unified cognitive memory system
            config: Additional configuration
        """
        # Initialize base agent
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            config=config or {}
        )
        
        # Validate memory system
        if not isinstance(memory_system, UnifiedCognitiveMemory):
            raise TypeError(
                f"CognitiveAgent requires UnifiedCognitiveMemory, got {type(memory_system)}"
            )
        
        # Connect to memory system
        self.memory = memory_system
        
        # Update agent state with memory reference
        self.state.memory_system = memory_system
        
        # Initialize causal tracker for this agent
        self.causal_tracker = CausalPatternTracker()
        
        # Track agent's experiences
        self.experience_history: List[AgentExperience] = []
        self.decision_history: List[AgentDecision] = []
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        
        # Performance metrics
        self.successful_decisions = 0
        self.failed_decisions = 0
        self.total_experiences = 0
        
        logger.info(
            "CognitiveAgent initialized",
            agent_id=agent_id,
            agent_type=agent_type,
            memory_connected=True
        )
    
    # ==================== Perception Phase ====================
    
    async def perceive(
        self,
        observation: Any,
        importance: Optional[float] = None
    ) -> AgentExperience:
        """
        Process and store a new observation/experience
        
        This is the PERCEIVE phase of the cognitive loop.
        The agent observes something and stores it in memory.
        
        Args:
            observation: The observed data/event
            importance: Override importance score (0-1)
        
        Returns:
            The processed experience
        """
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(observation)
        
        # Calculate surprise based on expectations
        surprise = await self._calculate_surprise(observation)
        
        # Create experience record
        experience = AgentExperience(
            agent_id=self.agent_id,
            experience_type='perception',
            content=observation,
            importance=importance,
            surprise_score=surprise,
            emotional_valence=self._assess_emotional_impact(observation)
        )
        
        # Store in memory system
        memory_result = await self.memory.process_experience(
            content=experience.to_memory_format()['content'],
            context=experience.to_memory_format()['context']
        )
        
        if memory_result['success']:
            experience.tags.append(f"memory_id:{memory_result.get('working_memory_id')}")
            
            # Track for causal patterns
            if surprise > 0.7:
                await self._track_causal_event(experience, 'high_surprise_observation')
        
        # Update history
        self.experience_history.append(experience)
        self.total_experiences += 1
        
        # Log perception
        logger.debug(
            "Agent perceived",
            agent_id=self.agent_id,
            importance=importance,
            surprise=surprise,
            stored_in_memory=memory_result['success']
        )
        
        return experience
    
    # ==================== Thinking Phase ====================
    
    async def think(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_memory: bool = True
    ) -> MemoryContext:
        """
        Think about a query using memory and reasoning
        
        This is the THINK phase of the cognitive loop.
        The agent retrieves relevant memories and reasons about them.
        
        Args:
            query: What to think about
            context: Additional context for the query
            use_memory: Whether to use memory system (vs pure reasoning)
        
        Returns:
            Memory context with retrieved information and synthesis
        """
        if not use_memory:
            # Pure reasoning without memory (for baseline comparison)
            return await self._think_without_memory(query, context)
        
        # Query the memory system
        memory_context = await self.memory.query(
            query_text=query,
            context=context,
            timeout=5.0
        )
        
        # Store in agent state for decision phase
        self.state.last_memory_context = memory_context
        
        # Analyze retrieved memories for patterns
        if memory_context.episodes:
            patterns = await self._analyze_memory_patterns(memory_context.episodes)
            memory_context.reasoning_path.append({
                'step': 'pattern_analysis',
                'patterns_found': len(patterns),
                'content': patterns
            })
        
        # Check for relevant causal patterns
        if memory_context.causal_chains:
            failure_risk = await self._assess_failure_risk(memory_context.causal_chains)
            if failure_risk > 0.5:
                logger.warning(
                    "High failure risk detected",
                    agent_id=self.agent_id,
                    query=query,
                    risk_level=failure_risk
                )
                memory_context.reasoning_path.append({
                    'step': 'risk_assessment',
                    'failure_risk': failure_risk,
                    'recommendation': 'proceed_with_caution'
                })
        
        # Log thinking
        logger.debug(
            "Agent thought",
            agent_id=self.agent_id,
            query=query[:100],
            sources_retrieved=memory_context.total_sources,
            confidence=memory_context.confidence
        )
        
        return memory_context
    
    # ==================== Action Phase ====================
    
    async def act(
        self,
        memory_context: Optional[MemoryContext] = None,
        exploration_override: Optional[float] = None
    ) -> AgentDecision:
        """
        Make a decision and take action based on memory and reasoning
        
        This is the ACT phase of the cognitive loop.
        The agent decides what to do and executes the action.
        
        Args:
            memory_context: Context from thinking phase
            exploration_override: Override exploration rate for this decision
        
        Returns:
            The decision made and action taken
        """
        # Use last memory context if not provided
        if memory_context is None:
            memory_context = self.state.last_memory_context
        
        # Determine exploration vs exploitation
        exploration_rate = exploration_override or self.exploration_rate
        should_explore = np.random.random() < exploration_rate
        
        # Generate decision based on memory
        if memory_context and not should_explore:
            # Exploitation: Use memory to make informed decision
            decision = await self._make_informed_decision(memory_context)
        else:
            # Exploration: Try something new
            decision = await self._make_exploratory_decision()
        
        # Predict outcome before acting
        predicted_outcome = await self._predict_outcome(decision)
        decision.predicted_outcome = predicted_outcome
        
        # Execute the action
        actual_outcome = await self._execute_action(decision.action)
        
        # Record the experience
        action_experience = AgentExperience(
            agent_id=self.agent_id,
            experience_type='action',
            content={
                'decision': decision.decision_id,
                'action': decision.action,
                'predicted': predicted_outcome,
                'actual': actual_outcome
            },
            importance=0.7,  # Actions are generally important
            causal_trigger=memory_context.query if memory_context else None,
            expected_outcome=predicted_outcome,
            actual_outcome=actual_outcome
        )
        
        # Store action and outcome in memory
        await self.memory.process_experience(
            content=action_experience.to_memory_format()['content'],
            context=action_experience.to_memory_format()['context']
        )
        
        # Learn from the outcome
        await self._learn_from_outcome(decision, actual_outcome)
        
        # Update history
        self.decision_history.append(decision)
        
        # Update metrics
        if self._is_successful_outcome(actual_outcome):
            self.successful_decisions += 1
        else:
            self.failed_decisions += 1
        
        # Log action
        logger.info(
            "Agent acted",
            agent_id=self.agent_id,
            decision_type=decision.decision_type,
            confidence=decision.confidence,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome
        )
        
        return decision
    
    # ==================== Learning Methods ====================
    
    async def _learn_from_outcome(
        self,
        decision: AgentDecision,
        actual_outcome: str
    ):
        """
        Learn from the outcome of a decision
        
        This updates the agent's understanding and improves future decisions
        """
        # Check if prediction was correct
        prediction_correct = decision.predicted_outcome == actual_outcome
        
        # Update confidence based on prediction accuracy
        if prediction_correct:
            self.learning_rate *= 1.01  # Slightly increase learning rate
        else:
            self.learning_rate *= 0.99  # Slightly decrease learning rate
            
            # Store surprising outcome for consolidation
            surprise_experience = AgentExperience(
                agent_id=self.agent_id,
                experience_type='surprise',
                content={
                    'expected': decision.predicted_outcome,
                    'actual': actual_outcome,
                    'decision': decision.decision_id
                },
                importance=0.9,
                surprise_score=0.9
            )
            
            await self.memory.process_experience(
                content=surprise_experience.to_memory_format()['content'],
                context=surprise_experience.to_memory_format()['context']
            )
        
        # Track causal pattern
        if decision.memory_sources:
            event = CausalEvent(
                event_id=decision.decision_id,
                event_type='agent_decision',
                topology_signature=np.random.randn(128),  # Would use real topology
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                context={'decision': decision.action},
                embeddings=np.random.randn(384),  # Would use real embeddings
                confidence=decision.confidence
            )
            
            # Determine outcome type
            if self._is_successful_outcome(actual_outcome):
                outcome_type = OutcomeType.SUCCESS
            else:
                outcome_type = OutcomeType.FAILURE
            
            # Track pattern
            await self.causal_tracker.track_event(event)
            
            # If we have enough events, record outcome
            if len(self.decision_history) % 10 == 0:
                chain_id = f"agent_{self.agent_id}_chain_{len(self.decision_history)}"
                pattern = await self.causal_tracker.record_outcome(chain_id, outcome_type)
                
                logger.info(
                    "Causal pattern learned",
                    agent_id=self.agent_id,
                    pattern_id=pattern.pattern_id,
                    confidence=pattern.compute_confidence()
                )
    
    async def consolidate_learning(self):
        """
        Trigger memory consolidation for this agent's experiences
        
        This should be called periodically or after important experiences
        """
        logger.info(f"Agent {self.agent_id} triggering consolidation")
        
        # Run awake consolidation
        await self.memory.run_awake_consolidation()
        
        # Extract knowledge from recent experiences
        if len(self.experience_history) >= 10:
            recent_episodes = await self.memory.episodic_memory.get_recent(limit=20)
            
            # Filter for this agent's episodes
            agent_episodes = [
                ep for ep in recent_episodes
                if hasattr(ep, 'social_context') and 
                self.agent_id in ep.social_context.participants
            ]
            
            if agent_episodes:
                # Extract semantic knowledge
                concepts = await self.memory.semantic_memory.extract_knowledge_from_episodes(
                    agent_episodes
                )
                
                logger.info(
                    "Knowledge extracted",
                    agent_id=self.agent_id,
                    episodes_processed=len(agent_episodes),
                    concepts_learned=len(concepts)
                )
    
    # ==================== Helper Methods ====================
    
    def _calculate_importance(self, observation: Any) -> float:
        """Calculate importance score for an observation"""
        # Simple heuristic - override in subclasses
        importance = 0.5
        
        # Increase importance for certain types
        if isinstance(observation, dict):
            if observation.get('error') or observation.get('failure'):
                importance = 0.9
            elif observation.get('success') or observation.get('completed'):
                importance = 0.8
            elif observation.get('warning'):
                importance = 0.7
        
        return importance
    
    async def _calculate_surprise(self, observation: Any) -> float:
        """Calculate surprise score based on expectations"""
        # Query memory for similar past observations
        if isinstance(observation, (str, dict)):
            query = str(observation)[:100]
            similar = await self.memory.query(
                query_text=f"Similar to: {query}",
                timeout=1.0
            )
            
            # High surprise if nothing similar found
            if similar.total_sources == 0:
                return 0.8
            
            # Low surprise if many similar
            return max(0.1, 1.0 - (similar.total_sources / 10.0))
        
        return 0.5
    
    def _assess_emotional_impact(self, observation: Any) -> float:
        """Assess emotional valence of observation"""
        # Simple sentiment analysis - override in subclasses
        if isinstance(observation, (str, dict)):
            content = str(observation).lower()
            
            # Positive indicators
            positive_words = ['success', 'complete', 'good', 'great', 'excellent']
            # Negative indicators
            negative_words = ['error', 'fail', 'bad', 'wrong', 'problem']
            
            positive_count = sum(1 for word in positive_words if word in content)
            negative_count = sum(1 for word in negative_words if word in content)
            
            if positive_count > negative_count:
                return min(1.0, positive_count * 0.3)
            elif negative_count > positive_count:
                return max(-1.0, -negative_count * 0.3)
        
        return 0.0
    
    async def _track_causal_event(
        self,
        experience: AgentExperience,
        event_type: str
    ):
        """Track experience as causal event"""
        event = CausalEvent(
            event_id=hashlib.md5(str(experience.content).encode()).hexdigest()[:8],
            event_type=event_type,
            topology_signature=np.random.randn(128),  # Would use real topology
            timestamp=experience.timestamp,
            agent_id=self.agent_id,
            context={'experience': experience.experience_type},
            embeddings=np.random.randn(384),  # Would use real embeddings
            confidence=experience.importance
        )
        
        await self.causal_tracker.track_event(event)
    
    async def _think_without_memory(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> MemoryContext:
        """Fallback thinking without memory access"""
        # Simple reasoning without memory
        return MemoryContext(
            query=query,
            timestamp=datetime.now(),
            synthesized_answer="Reasoning without memory access",
            confidence=0.3
        )
    
    async def _analyze_memory_patterns(
        self,
        episodes: List[Episode]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in retrieved episodes"""
        patterns = []
        
        # Look for temporal patterns
        if len(episodes) >= 3:
            timestamps = [ep.timestamp for ep in episodes if hasattr(ep, 'timestamp')]
            if timestamps:
                # Check for periodicity
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                if intervals:
                    avg_interval = sum(intervals, 0) / len(intervals)
                    patterns.append({
                        'type': 'temporal',
                        'pattern': 'periodic',
                        'interval': avg_interval
                    })
        
        # Look for causal patterns
        causal_chains = [
            ep.causal_context for ep in episodes 
            if hasattr(ep, 'causal_context')
        ]
        if causal_chains:
            patterns.append({
                'type': 'causal',
                'pattern': 'chain_detected',
                'length': len(causal_chains)
            })
        
        return patterns
    
    async def _assess_failure_risk(
        self,
        causal_chains: List[Any]
    ) -> float:
        """Assess risk of failure based on causal patterns"""
        if not causal_chains:
            return 0.0
        
        # Simple risk assessment
        failure_indicators = 0
        total_indicators = 0
        
        for chain in causal_chains:
            if isinstance(chain, dict):
                if chain.get('outcome') == 'FAILURE':
                    failure_indicators += 1
                total_indicators += 1
        
        if total_indicators > 0:
            return failure_indicators / total_indicators
        
        return 0.0
    
    async def _make_informed_decision(
        self,
        memory_context: MemoryContext
    ) -> AgentDecision:
        """Make decision based on memory context"""
        # Extract relevant information from memory
        episodic_support = memory_context.episodes[:3] if memory_context.episodes else []
        semantic_support = memory_context.concepts[:3] if memory_context.concepts else []
        
        # Generate decision (would use LLM or policy network in production)
        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            decision_type='memory_informed',
            action={
                'type': 'execute',
                'based_on': memory_context.synthesized_answer[:100] if memory_context.synthesized_answer else 'no_synthesis'
            },
            reasoning=f"Decision based on {memory_context.total_sources} memory sources",
            confidence=memory_context.confidence,
            episodic_support=episodic_support,
            semantic_support=semantic_support
        )
        
        return decision
    
    async def _make_exploratory_decision(self) -> AgentDecision:
        """Make exploratory decision to try something new"""
        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            decision_type='exploratory',
            action={
                'type': 'explore',
                'strategy': 'random'
            },
            reasoning="Exploring new possibilities",
            confidence=0.3
        )
        
        return decision
    
    async def _predict_outcome(self, decision: AgentDecision) -> str:
        """Predict the outcome of a decision"""
        # Use causal tracker for prediction
        if decision.episodic_support:
            # Create event sequence from support
            events = []  # Would convert episodes to events
            
            failure_prob, pattern_id = self.causal_tracker.get_failure_prediction(events)
            
            if failure_prob > 0.7:
                return "likely_failure"
            elif failure_prob < 0.3:
                return "likely_success"
        
        return "uncertain"
    
    @abstractmethod
    async def _execute_action(self, action: Dict[str, Any]) -> str:
        """
        Execute the actual action
        
        This must be implemented by concrete agent classes
        """
        pass
    
    def _is_successful_outcome(self, outcome: str) -> bool:
        """Determine if outcome represents success"""
        success_indicators = ['success', 'complete', 'done', 'achieved']
        return any(indicator in str(outcome).lower() for indicator in success_indicators)
    
    # ==================== Lifecycle Methods ====================
    
    async def startup(self):
        """Initialize agent and connect to systems"""
        await super().startup()
        
        logger.info(
            "CognitiveAgent started",
            agent_id=self.agent_id,
            memory_connected=True,
            causal_tracking=True
        )
    
    async def shutdown(self):
        """Cleanup and save agent state"""
        # Trigger final consolidation
        await self.consolidate_learning()
        
        # Log final metrics
        success_rate = (
            self.successful_decisions / max(1, self.successful_decisions + self.failed_decisions)
        )
        
        logger.info(
            "CognitiveAgent shutting down",
            agent_id=self.agent_id,
            total_experiences=self.total_experiences,
            total_decisions=len(self.decision_history),
            success_rate=success_rate
        )
        
        await super().shutdown()
    
    # ==================== Metrics and Monitoring ====================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_decisions = self.successful_decisions + self.failed_decisions
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'total_experiences': self.total_experiences,
            'total_decisions': total_decisions,
            'successful_decisions': self.successful_decisions,
            'failed_decisions': self.failed_decisions,
            'success_rate': self.successful_decisions / max(1, total_decisions),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'memory_queries': len([h for h in self.experience_history if h.experience_type == 'perception']),
            'causal_patterns_tracked': len(self.causal_tracker.patterns)
        }


# ==================== Concrete Agent Implementations ====================

class CognitivePlannerAgent(CognitiveAgent):
    """
    Planner agent that uses memory to create better plans
    """
    
    def __init__(self, agent_id: str, memory_system: UnifiedCognitiveMemory, config: Optional[Dict] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type='planner',
            memory_system=memory_system,
            config=config
        )
    
    async def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute planning action"""
        if action.get('type') == 'create_plan':
            # Simulate plan creation
            await asyncio.sleep(0.1)
            return "plan_created_successfully"
        
        return "action_executed"
    
    async def create_plan(self, goal: str) -> Dict[str, Any]:
        """
        Create a plan using memory of past successful plans
        """
        # Perceive the goal
        await self.perceive(
            observation={'goal': goal, 'type': 'planning_request'},
            importance=0.8
        )
        
        # Think about similar past plans
        memory_context = await self.think(
            query=f"What plans have succeeded for goals similar to: {goal}",
            context={'planning': True}
        )
        
        # Act to create the plan
        decision = await self.act(memory_context)
        
        # Generate plan based on memory
        plan = {
            'goal': goal,
            'steps': [],
            'confidence': decision.confidence,
            'based_on_memories': len(memory_context.episodes),
            'risk_assessment': memory_context.grounding_strength
        }
        
        # Add steps from successful past plans
        for episode in memory_context.episodes[:5]:
            if hasattr(episode, 'content') and isinstance(episode.content, dict):
                if 'plan' in episode.content:
                    plan['steps'].extend(episode.content['plan'].get('steps', []))
        
        # Store the created plan
        await self.perceive(
            observation={'created_plan': plan, 'type': 'plan_output'},
            importance=0.7
        )
        
        return plan


class CognitiveExecutorAgent(CognitiveAgent):
    """
    Executor agent that learns from execution outcomes
    """
    
    def __init__(self, agent_id: str, memory_system: UnifiedCognitiveMemory, config: Optional[Dict] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type='executor',
            memory_system=memory_system,
            config=config
        )
    
    async def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute actual task"""
        task_type = action.get('task_type', 'unknown')
        
        # Simulate execution with different outcomes
        if task_type == 'data_processing':
            await asyncio.sleep(0.2)
            return "data_processed_successfully"
        elif task_type == 'api_call':
            await asyncio.sleep(0.3)
            # Sometimes fail to learn from failures
            if np.random.random() > 0.8:
                return "api_call_failed_timeout"
            return "api_call_successful"
        
        return "task_executed"
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using memory of past executions
        """
        # Perceive the task
        await self.perceive(
            observation={'task': task, 'type': 'execution_request'},
            importance=0.7
        )
        
        # Think about similar past executions
        task_description = task.get('description', str(task))
        memory_context = await self.think(
            query=f"How to best execute task: {task_description}",
            context={'execution': True}
        )
        
        # Check for failure patterns
        if memory_context.causal_chains:
            for chain in memory_context.causal_chains:
                if isinstance(chain, dict) and chain.get('pattern_id'):
                    logger.warning(
                        "Potential failure pattern detected",
                        agent_id=self.agent_id,
                        pattern=chain['pattern_id']
                    )
        
        # Act to execute the task
        decision = await self.act(memory_context)
        
        # Execute with the decided approach
        result = await self._execute_action({
            'task_type': task.get('type', 'general'),
            'approach': decision.action
        })
        
        # Store execution result
        await self.perceive(
            observation={
                'task': task,
                'result': result,
                'type': 'execution_result'
            },
            importance=0.6
        )
        
        return {
            'task': task,
            'result': result,
            'success': self._is_successful_outcome(result),
            'confidence': decision.confidence,
            'memory_informed': len(memory_context.episodes) > 0
        }


class CognitiveAnalystAgent(CognitiveAgent):
    """
    Analyst agent that builds knowledge from observations
    """
    
    def __init__(self, agent_id: str, memory_system: UnifiedCognitiveMemory, config: Optional[Dict] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type='analyst',
            memory_system=memory_system,
            config=config
        )
    
    async def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute analysis action"""
        if action.get('type') == 'analyze':
            await asyncio.sleep(0.15)
            return "analysis_completed"
        
        return "action_executed"
    
    async def analyze_data(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data using accumulated knowledge
        """
        # Perceive the data
        await self.perceive(
            observation={'data': data, 'type': 'analysis_request'},
            importance=0.6
        )
        
        # Think about patterns and insights
        memory_context = await self.think(
            query=f"What patterns exist in data like: {str(data)[:100]}",
            context={'analysis': True}
        )
        
        # Act to perform analysis
        decision = await self.act(memory_context)
        
        # Generate analysis based on memory
        analysis = {
            'data_summary': str(data)[:200],
            'patterns_found': [],
            'insights': [],
            'confidence': decision.confidence
        }
        
        # Extract patterns from semantic memory
        if memory_context.concepts:
            for concept in memory_context.concepts[:3]:
                if hasattr(concept, 'label'):
                    analysis['patterns_found'].append(concept.label)
        
        # Extract insights from episodic memory
        if memory_context.episodes:
            for episode in memory_context.episodes[:3]:
                if hasattr(episode, 'content'):
                    analysis['insights'].append(str(episode.content)[:100])
        
        # Store analysis result
        await self.perceive(
            observation={'analysis': analysis, 'type': 'analysis_result'},
            importance=0.5
        )
        
        # Trigger consolidation to extract knowledge
        if len(self.experience_history) % 20 == 0:
            await self.consolidate_learning()
        
        return analysis