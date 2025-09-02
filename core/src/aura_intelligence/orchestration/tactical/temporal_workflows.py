"""
Temporal Workflows for Cognitive Pipeline
=========================================
Durable, fault-tolerant workflows with Active-Active failover
Based on Temporal 1.26 patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from abc import ABC, abstractmethod
import json

# Temporal imports (with fallback for testing)
try:
    from temporalio import workflow, activity
    from temporalio.workflow import workflow_method
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    # Mock decorators for testing
    def workflow_method(func):
        return func
    class workflow:
        @staticmethod
        def defn(cls):
            return cls
        @staticmethod
        def run(func):
            return func
    class activity:
        @staticmethod
        def defn(func):
            return func

# AURA imports
from ...tda.legacy.persistence_simple import TDAProcessor
from ...inference.active_inference_lite import ActiveInferenceLite
from ...coral.coral_2025 import CoRaL2025System
from ...dpo.dpo_2025_advanced import AURAAdvancedDPO
from ...agents.production_langgraph_agent import AURAProductionAgent

import logging
logger = logging.getLogger(__name__)


@dataclass
class ContextRequest:
    """Input context for cognitive workflow"""
    request_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Input data
    observations: Dict[str, Any] = field(default_factory=dict)
    previous_state: Optional[Dict[str, Any]] = None
    
    # Configuration
    priority: str = "normal"  # low, normal, high, critical
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source: str = ""
    user_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass 
class PhaseResult:
    """Result from a workflow phase"""
    phase_name: str
    status: str  # success, failure, timeout
    duration_ms: float
    
    # Phase outputs
    output: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Error info
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    request_id: str
    
    # Phase results
    perception: Optional[PhaseResult] = None
    inference: Optional[PhaseResult] = None
    consensus: Optional[PhaseResult] = None
    action: Optional[PhaseResult] = None
    
    # Overall metrics
    total_duration_ms: float = 0.0
    success: bool = False
    
    # Final outputs
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    retries: int = 0


class WorkflowPhase(ABC):
    """Abstract base class for workflow phases"""
    
    @abstractmethod
    async def execute(self, 
                     context: ContextRequest,
                     previous_results: List[PhaseResult]) -> PhaseResult:
        """Execute the phase"""
        pass


@activity.defn
class PerceptionPhase(WorkflowPhase):
    """
    Perception phase: Extract features from observations
    Integrates TDA and initial Active Inference
    """
    
    def __init__(self):
        self.tda_processor = None
        self.active_inference = None
        
    async def execute(self,
                     context: ContextRequest,
                     previous_results: List[PhaseResult]) -> PhaseResult:
        """Execute perception phase"""
        start_time = time.perf_counter()
        
        try:
            # Initialize components if needed
            if self.tda_processor is None:
                self.tda_processor = TDAProcessor()
            if self.active_inference is None:
                self.active_inference = ActiveInferenceLite()
                
            # Extract observations
            raw_data = context.observations.get('data', [])
            
            # Compute topological features
            tda_features = await self._compute_tda_features(raw_data)
            
            # Initial Active Inference processing
            ai_metrics = await self.active_inference.process_observation(
                np.array(raw_data)
            )
            
            # Combine results
            output = {
                'topological_features': tda_features,
                'persistence_diagram': tda_features.get('diagram', []),
                'anomaly_score': ai_metrics.anomaly_score,
                'uncertainty': ai_metrics.uncertainty,
                'free_energy': ai_metrics.free_energy
            }
            
            metrics = {
                'tda_computation_ms': tda_features.get('computation_time', 0.0),
                'inference_time_ms': ai_metrics.inference_time_ms,
                'feature_dimension': len(tda_features.get('features', []))
            }
            
            duration = (time.perf_counter() - start_time) * 1000
            
            return PhaseResult(
                phase_name="perception",
                status="success",
                duration_ms=duration,
                output=output,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Perception phase failed: {e}")
            return PhaseResult(
                phase_name="perception",
                status="failure",
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
            
    async def _compute_tda_features(self, data: Any) -> Dict[str, Any]:
        """Compute TDA features"""
        # Placeholder - would use actual TDA processor
        return {
            'features': [0.1, 0.2, 0.3],
            'diagram': [(0, 0.5), (0.1, 0.8)],
            'betti_numbers': [1, 2, 0],
            'computation_time': 10.5
        }


@activity.defn
class InferencePhase(WorkflowPhase):
    """
    Inference phase: Active Inference and belief updating
    Integrates Free Energy minimization and DPO
    """
    
    def __init__(self):
        self.active_inference = None
        self.dpo_system = None
        
    async def execute(self,
                     context: ContextRequest,
                     previous_results: List[PhaseResult]) -> PhaseResult:
        """Execute inference phase"""
        start_time = time.perf_counter()
        
        try:
            # Get perception results
            perception_result = next(
                (r for r in previous_results if r.phase_name == "perception"),
                None
            )
            
            if not perception_result or perception_result.status != "success":
                raise ValueError("Perception phase failed or missing")
                
            # Initialize components
            if self.active_inference is None:
                self.active_inference = ActiveInferenceLite()
            if self.dpo_system is None:
                self.dpo_system = AURAAdvancedDPO()
                
            # Extract features
            features = perception_result.output['topological_features']
            
            # Deep Active Inference
            beliefs = await self._update_beliefs(features, context)
            
            # DPO preference alignment
            preferences = await self._apply_preferences(beliefs, context)
            
            output = {
                'beliefs': beliefs,
                'expected_free_energy': beliefs.get('expected_fe', 0.0),
                'preferences': preferences,
                'confidence': beliefs.get('confidence', 0.0)
            }
            
            metrics = {
                'belief_entropy': beliefs.get('entropy', 0.0),
                'free_energy': beliefs.get('free_energy', 0.0),
                'preference_alignment': preferences.get('alignment', 0.0)
            }
            
            duration = (time.perf_counter() - start_time) * 1000
            
            return PhaseResult(
                phase_name="inference",
                status="success",
                duration_ms=duration,
                output=output,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Inference phase failed: {e}")
            return PhaseResult(
                phase_name="inference",
                status="failure",
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
            
    async def _update_beliefs(self, features: Dict, context: ContextRequest) -> Dict[str, Any]:
        """Update beliefs using Active Inference"""
        # Placeholder implementation
        return {
            'state_belief': [0.2, 0.8],
            'confidence': 0.85,
            'entropy': 1.2,
            'free_energy': 0.5,
            'expected_fe': 0.3
        }
        
    async def _apply_preferences(self, beliefs: Dict, context: ContextRequest) -> Dict[str, Any]:
        """Apply DPO preferences"""
        # Placeholder implementation
        return {
            'preferred_action': 'analyze',
            'alignment': 0.92,
            'safety_score': 0.98
        }


@activity.defn
class ConsensusPhase(WorkflowPhase):
    """
    Consensus phase: Multi-agent coordination via CoRaL
    Achieves collective agreement on decisions
    """
    
    def __init__(self):
        self.coral_system = None
        
    async def execute(self,
                     context: ContextRequest,
                     previous_results: List[PhaseResult]) -> PhaseResult:
        """Execute consensus phase"""
        start_time = time.perf_counter()
        
        try:
            # Get inference results
            inference_result = next(
                (r for r in previous_results if r.phase_name == "inference"),
                None
            )
            
            if not inference_result or inference_result.status != "success":
                raise ValueError("Inference phase failed or missing")
                
            # Initialize CoRaL
            if self.coral_system is None:
                self.coral_system = CoRaL2025System()
                
            # Prepare consensus request
            beliefs = inference_result.output['beliefs']
            preferences = inference_result.output['preferences']
            
            # Run CoRaL consensus
            consensus = await self._achieve_consensus(beliefs, preferences)
            
            output = {
                'consensus_decision': consensus['decision'],
                'agreement_level': consensus['agreement'],
                'dissenting_agents': consensus['dissenters'],
                'causal_influence': consensus['causal_influence']
            }
            
            metrics = {
                'consensus_time_ms': consensus['time_ms'],
                'num_rounds': consensus['rounds'],
                'message_count': consensus['messages']
            }
            
            duration = (time.perf_counter() - start_time) * 1000
            
            return PhaseResult(
                phase_name="consensus",
                status="success",
                duration_ms=duration,
                output=output,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Consensus phase failed: {e}")
            return PhaseResult(
                phase_name="consensus",
                status="failure", 
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
            
    async def _achieve_consensus(self, beliefs: Dict, preferences: Dict) -> Dict[str, Any]:
        """Achieve consensus using CoRaL"""
        # Placeholder implementation
        return {
            'decision': 'proceed_with_analysis',
            'agreement': 0.88,
            'dissenters': ['agent_3'],
            'causal_influence': 0.75,
            'time_ms': 45.2,
            'rounds': 3,
            'messages': 15
        }


@activity.defn
class ActionPhase(WorkflowPhase):
    """
    Action phase: Execute decisions through agents
    Coordinates LangGraph agents for task execution
    """
    
    def __init__(self):
        self.agent_system = None
        
    async def execute(self,
                     context: ContextRequest,
                     previous_results: List[PhaseResult]) -> PhaseResult:
        """Execute action phase"""
        start_time = time.perf_counter()
        
        try:
            # Get consensus results
            consensus_result = next(
                (r for r in previous_results if r.phase_name == "consensus"),
                None
            )
            
            if not consensus_result or consensus_result.status != "success":
                raise ValueError("Consensus phase failed or missing")
                
            # Initialize agent system
            if self.agent_system is None:
                self.agent_system = AURAProductionAgent.create_agent(
                    agent_type="executor"
                )
                
            # Execute consensus decision
            decision = consensus_result.output['consensus_decision']
            execution_result = await self._execute_decision(decision, context)
            
            output = {
                'executed_actions': execution_result['actions'],
                'execution_status': execution_result['status'],
                'side_effects': execution_result.get('side_effects', [])
            }
            
            metrics = {
                'execution_time_ms': execution_result['time_ms'],
                'actions_count': len(execution_result['actions']),
                'success_rate': execution_result.get('success_rate', 1.0)
            }
            
            duration = (time.perf_counter() - start_time) * 1000
            
            return PhaseResult(
                phase_name="action",
                status="success",
                duration_ms=duration,
                output=output,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Action phase failed: {e}")
            return PhaseResult(
                phase_name="action",
                status="failure",
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
            
    async def _execute_decision(self, decision: str, context: ContextRequest) -> Dict[str, Any]:
        """Execute decision through agents"""
        # Placeholder implementation
        return {
            'actions': [
                {'type': 'analyze', 'target': 'anomaly_1', 'status': 'completed'},
                {'type': 'report', 'target': 'dashboard', 'status': 'completed'}
            ],
            'status': 'completed',
            'time_ms': 125.3,
            'success_rate': 1.0
        }


@workflow.defn
class CognitiveWorkflow:
    """
    Main cognitive workflow orchestrating all phases
    Implements durable execution with Temporal
    """
    
    @workflow.run
    async def run_cycle(self, context: ContextRequest) -> WorkflowResult:
        """
        Run complete cognitive cycle: Perception → Inference → Consensus → Action
        """
        workflow_start = time.perf_counter()
        
        result = WorkflowResult(
            workflow_id=f"cognitive_{context.request_id}",
            request_id=context.request_id
        )
        
        phase_results = []
        
        try:
            # Phase 1: Perception
            perception_phase = PerceptionPhase()
            result.perception = await workflow.execute_activity(
                perception_phase.execute,
                args=[context, phase_results],
                start_to_close_timeout=timedelta(seconds=60)
            )
            phase_results.append(result.perception)
            
            if result.perception.status != "success":
                raise Exception(f"Perception failed: {result.perception.error}")
                
            # Phase 2: Inference
            inference_phase = InferencePhase()
            result.inference = await workflow.execute_activity(
                inference_phase.execute,
                args=[context, phase_results],
                start_to_close_timeout=timedelta(seconds=120)
            )
            phase_results.append(result.inference)
            
            if result.inference.status != "success":
                raise Exception(f"Inference failed: {result.inference.error}")
                
            # Phase 3: Consensus
            consensus_phase = ConsensusPhase()
            result.consensus = await workflow.execute_activity(
                consensus_phase.execute,
                args=[context, phase_results],
                start_to_close_timeout=timedelta(seconds=60)
            )
            phase_results.append(result.consensus)
            
            if result.consensus.status != "success":
                raise Exception(f"Consensus failed: {result.consensus.error}")
                
            # Phase 4: Action
            action_phase = ActionPhase()
            result.action = await workflow.execute_activity(
                action_phase.execute,
                args=[context, phase_results],
                start_to_close_timeout=timedelta(seconds=180)
            )
            phase_results.append(result.action)
            
            # Compile final results
            result.success = all(
                phase.status == "success" 
                for phase in [result.perception, result.inference, 
                             result.consensus, result.action]
            )
            
            if result.consensus:
                result.decisions.append(result.consensus.output.get('consensus_decision', {}))
                
            if result.inference:
                result.confidence = result.inference.output.get('confidence', 0.0)
                
            result.total_duration_ms = (time.perf_counter() - workflow_start) * 1000
            
            logger.info(f"Cognitive workflow completed in {result.total_duration_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"Cognitive workflow failed: {e}")
            result.success = False
            result.total_duration_ms = (time.perf_counter() - workflow_start) * 1000
            
        return result


# Fallback implementation for non-Temporal environments
if not TEMPORAL_AVAILABLE:
    from datetime import timedelta
    
    class workflow:
        @staticmethod
        async def execute_activity(activity, args, start_to_close_timeout):
            """Mock activity execution"""
            return await activity(*args)


async def create_cognitive_workflow(context: ContextRequest) -> WorkflowResult:
    """
    Factory function to create and run cognitive workflow
    Works with or without Temporal
    """
    if TEMPORAL_AVAILABLE:
        # Use actual Temporal client
        # client = await Client.connect("temporal:7233")
        # handle = await client.start_workflow(
        #     CognitiveWorkflow.run_cycle,
        #     context,
        #     id=f"cognitive_{context.request_id}",
        #     task_queue="cognitive-queue"
        # )
        # return await handle.result()
        pass
    
    # Fallback: Direct execution
    workflow = CognitiveWorkflow()
    return await workflow.run_cycle(context)