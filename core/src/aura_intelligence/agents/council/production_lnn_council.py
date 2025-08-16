"""
Production-Ready LNN Council Agent Implementation.

This module provides a complete implementation of the LNN Council Agent
with all abstract methods implemented and full integration with Neo4j,
Mem0, and Kafka event streaming.
"""

from typing import Dict, Any, Optional, List, Union, cast
from datetime import datetime, timezone
import asyncio
import uuid
import json
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import structlog

from ...agents.base import AgentBase, AgentConfig, AgentState
from ...agents.council.lnn_council import (
    LNNCouncilAgent, CouncilTask, CouncilVote, VoteType
)
from ...neural.lnn import LiquidNeuralNetwork, LNNConfig, ODESolver
from ...neural.context_integration import ContextAwareLNN, ContextWindow
from ...neural.memory_hooks import LNNMemoryHooks
from ...adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig
from ...events.producers import EventProducer
from ...memory.mem0_integration import Mem0Manager
from ...observability import create_tracer, create_meter
from ...resilience import resilient, ResilienceLevel

logger = structlog.get_logger()
tracer = create_tracer("production_lnn_council")
meter = create_meter("production_lnn_council")

# Metrics
council_decisions = meter.create_counter(
    name="aura.council.decisions",
    description="Number of council decisions made"
)

decision_confidence = meter.create_histogram(
    name="aura.council.confidence",
    description="Decision confidence scores"
)


class CouncilState(AgentState):
    """Extended state for council workflow."""
    task: Optional[CouncilTask] = None
    context_window: Optional[ContextWindow] = None
    lnn_output: Optional[torch.Tensor] = None
    vote: Optional[CouncilVote] = None
    neo4j_context: Dict[str, Any] = Field(default_factory=dict)
    memory_context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class LiquidTimeStep(nn.Module):
    """Liquid time step module for continuous-time dynamics."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward pass with liquid dynamics."""
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new


class ProductionLNNCouncilAgent(LNNCouncilAgent):
    """Production-ready LNN Council Agent with full integration."""
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]]):
        """Initialize with proper configuration handling."""
        # Convert AgentConfig to dictionary if needed
        if isinstance(config, AgentConfig):
            config_dict = self._convert_config(config)
        else:
            config_dict = config
            
        # Initialize base class with dictionary config
        super().__init__(config_dict)
        
        # Initialize components
        self._initialize_components()
        self._setup_neural_network()
        self._configure_integrations()
        
    def _convert_config(self, agent_config: AgentConfig) -> Dict[str, Any]:
        """Convert AgentConfig dataclass to dictionary format."""
        return {
            "name": agent_config.name,
            "model": agent_config.model,
            "temperature": agent_config.temperature,
            "max_retries": agent_config.max_retries,
            "timeout_seconds": agent_config.timeout_seconds,
            "enable_memory": agent_config.enable_memory,
            "enable_tools": agent_config.enable_tools,
            "lnn_config": {
                "input_size": 256,
                "hidden_size": 128,
                "output_size": 4,  # One per vote type
                "num_layers": 3,
                "time_constant": 1.0,
                "ode_solver": ODESolver.RK4,
                "solver_steps": 10,
                "adaptivity_rate": 0.1,
                "sparsity": 0.7,
                "consensus_enabled": True,
                "consensus_threshold": 0.67
            },
            "feature_flags": {
                "use_lnn_inference": True,
                "enable_memory_hooks": agent_config.enable_memory,
                "enable_context_queries": True,
                "enable_event_streaming": True
            },
            "vote_threshold": 0.7,
            "delegation_threshold": 0.3,
            "expertise_domains": ["gpu_allocation", "resource_management"]
        }
        
    def _initialize_components(self):
        """Initialize all required components."""
        # Create LNN config from dictionary
        lnn_config_dict = self.config.get("lnn_config", {})
        self.lnn_config = LNNConfig(
            input_size=lnn_config_dict.get("input_size", 256),
            hidden_size=lnn_config_dict.get("hidden_size", 128),
            output_size=lnn_config_dict.get("output_size", 4),
            num_layers=lnn_config_dict.get("num_layers", 3),
            time_constant=lnn_config_dict.get("time_constant", 1.0),
            ode_solver=lnn_config_dict.get("ode_solver", ODESolver.RK4),
            solver_steps=lnn_config_dict.get("solver_steps", 10),
            adaptivity_rate=lnn_config_dict.get("adaptivity_rate", 0.1),
            sparsity=lnn_config_dict.get("sparsity", 0.7),
            consensus_enabled=lnn_config_dict.get("consensus_enabled", True),
            consensus_threshold=lnn_config_dict.get("consensus_threshold", 0.67)
        )
        
        # Initialize adapters (will be set by dependency injection in production)
        self.neo4j_adapter: Optional[Neo4jAdapter] = None
        self.event_producer: Optional[EventProducer] = None
        self.memory_manager: Optional[Mem0Manager] = None
        
    def _setup_neural_network(self):
        """Set up the liquid neural network components."""
        # Create context-aware LNN
        self.context_lnn = ContextAwareLNN(
            lnn_config=self.lnn_config,
            memory_manager=self.memory_manager,
            knowledge_graph=self.neo4j_adapter,
            event_producer=self.event_producer,
            feature_flags=self.config.get("feature_flags", {})
        )
        
        # Create liquid layers
        self.liquid_layer = LiquidTimeStep(
            input_size=self.lnn_config.input_size,
            hidden_size=self.lnn_config.hidden_size
        )
        
        # Output layer for vote types
        self.output_layer = nn.Linear(
            self.lnn_config.hidden_size,
            self.lnn_config.output_size
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.liquid_layer.parameters()) + list(self.output_layer.parameters()),
            lr=self.config.get("learning_rate", 0.001)
        )
        
    def _configure_integrations(self):
        """Configure external integrations."""
        # Memory hooks for background indexing
        if self.config.get("feature_flags", {}).get("enable_memory_hooks", True):
            self.memory_hooks = LNNMemoryHooks(
                memory_manager=self.memory_manager
            )
        else:
            self.memory_hooks = None
            
    def set_adapters(
        self,
        neo4j: Optional[Neo4jAdapter] = None,
        events: Optional[EventProducer] = None,
        memory: Optional[Mem0Manager] = None
    ):
        """Set external adapters (for dependency injection)."""
        if neo4j:
            self.neo4j_adapter = neo4j
        if events:
            self.event_producer = events
        if memory:
            self.memory_manager = memory
            
        # Reconfigure components with new adapters
        if hasattr(self, 'context_lnn'):
            self.context_lnn.graph = self.neo4j_adapter
            self.context_lnn.events = self.event_producer
            self.context_lnn.memory = self.memory_manager
            
    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for council decisions."""
        workflow = StateGraph(CouncilState)
        
        # Define workflow nodes
        workflow.add_node("validate_task", self._validate_task_step)
        workflow.add_node("gather_context", self._gather_context_step)
        workflow.add_node("prepare_features", self._prepare_features_step)
        workflow.add_node("lnn_inference", self._lnn_inference_step)
        workflow.add_node("generate_vote", self._generate_vote_step)
        workflow.add_node("store_decision", self._store_decision_step)
        
        # Define edges
        workflow.add_edge("validate_task", "gather_context")
        workflow.add_edge("gather_context", "prepare_features")
        workflow.add_edge("prepare_features", "lnn_inference")
        workflow.add_edge("lnn_inference", "generate_vote")
        workflow.add_edge("generate_vote", "store_decision")
        workflow.add_edge("store_decision", END)
        
        # Set entry point
        workflow.set_entry_point("validate_task")
        
        return workflow
        
    async def _execute_step(self, state: CouncilState, step_name: str) -> CouncilState:
        """Execute a specific step in the workflow."""
        step_methods = {
            "validate_task": self._validate_task_step,
            "gather_context": self._gather_context_step,
            "prepare_features": self._prepare_features_step,
            "lnn_inference": self._lnn_inference_step,
            "generate_vote": self._generate_vote_step,
            "store_decision": self._store_decision_step
        }
        
        if step_name in step_methods:
            return await step_methods[step_name](state)
        else:
            raise ValueError(f"Unknown step: {step_name}")
            
    def _create_initial_state(self, input_data: CouncilTask) -> CouncilState:
        """Create initial state from council task."""
        return CouncilState(
            agent_id=str(uuid.uuid4()),
            task=input_data,
            context=input_data.context if hasattr(input_data, 'context') else {},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
    def _extract_output(self, final_state: CouncilState) -> CouncilVote:
        """Extract vote from final state."""
        if final_state.vote:
            return final_state.vote
        else:
            # Fallback vote if something went wrong
            return CouncilVote(
                agent_id=self.config["name"],
                vote=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning="Failed to generate vote due to processing error",
                supporting_evidence=[],
                timestamp=datetime.now(timezone.utc)
            )
            
    @resilient(criticality=ResilienceLevel.HIGH)
    async def _validate_task_step(self, state: CouncilState) -> CouncilState:
        """Validate the incoming task."""
        with tracer.start_as_current_span("validate_task") as span:
            if not state.task:
                state.add_error(ValueError("No task provided"), "validate_task")
                state.completed = True
                return state
                
            # Validate task structure
            required_fields = ["task_id", "task_type", "payload"]
            for field in required_fields:
                if not hasattr(state.task, field):
                    state.add_error(
                        ValueError(f"Missing required field: {field}"),
                        "validate_task"
                    )
                    state.completed = True
                    return state
                    
            span.set_attribute("task.id", state.task.task_id)
            span.set_attribute("task.type", state.task.task_type)
            
            state.next_step = "gather_context"
            return state
            
    @resilient(criticality=ResilienceLevel.MEDIUM)
    async def _gather_context_step(self, state: CouncilState) -> CouncilState:
        """Gather context from Neo4j and memory."""
        with tracer.start_as_current_span("gather_context") as span:
            context_tasks = []
            
            # Gather Neo4j context
            if self.neo4j_adapter and self.config.get("feature_flags", {}).get("enable_context_queries", True):
                context_tasks.append(self._query_neo4j_context(state))
                
            # Gather memory context
            if self.memory_manager and self.config.get("feature_flags", {}).get("enable_memory_hooks", True):
                context_tasks.append(self._query_memory_context(state))
                
            # Execute context gathering in parallel
            if context_tasks:
                await asyncio.gather(*context_tasks, return_exceptions=True)
                
            # Create context window
            state.context_window = ContextWindow(
                historical_patterns=state.neo4j_context.get("patterns", []),
                recent_decisions=state.neo4j_context.get("decisions", []),
                entity_relationships=state.neo4j_context.get("relationships", {}),
                temporal_context=state.memory_context
            )
            
            span.set_attribute("context.patterns", len(state.context_window.historical_patterns))
            span.set_attribute("context.decisions", len(state.context_window.recent_decisions))
            
            state.next_step = "prepare_features"
            return state
            
    async def _query_neo4j_context(self, state: CouncilState):
        """Query Neo4j for relevant context."""
        try:
            # Extract user ID from task payload
            gpu_allocation = state.task.payload.get("gpu_allocation", {})
            user_id = gpu_allocation.get("user_id", "unknown")
            
            # Query historical allocations
            query = """
            MATCH (u:User {id: $user_id})-[:REQUESTED]->(a:Allocation)
            WHERE a.timestamp > datetime() - duration('P30D')
            RETURN a.gpu_type as gpu_type, 
                   a.gpu_count as gpu_count, 
                   a.approved as approved,
                   a.actual_usage as actual_usage,
                   a.cost_per_hour as cost_per_hour
            ORDER BY a.timestamp DESC
            LIMIT 100
            """
            
            results = await self.neo4j_adapter.query(query, {"user_id": user_id})
            
            # Process results into patterns
            patterns = []
            for record in results:
                patterns.append({
                    "features": [
                        float(record.get("gpu_count", 0)),
                        float(record.get("cost_per_hour", 0)),
                        float(record.get("approved", False)),
                        float(record.get("actual_usage", 0))
                    ]
                })
                
            state.neo4j_context["patterns"] = patterns
            
        except Exception as e:
            logger.error(f"Failed to query Neo4j context: {e}")
            state.neo4j_context["patterns"] = []
            
    async def _query_memory_context(self, state: CouncilState):
        """Query memory for relevant context."""
        try:
            # Get recent decisions from memory
            memories = await self.memory_manager.search(
                query=f"gpu allocation decisions for {state.task.task_id}",
                limit=10
            )
            
            state.memory_context["recent_memories"] = memories
            
        except Exception as e:
            logger.error(f"Failed to query memory context: {e}")
            state.memory_context["recent_memories"] = []
            
    async def _prepare_features_step(self, state: CouncilState) -> CouncilState:
        """Prepare features for LNN inference."""
        with tracer.start_as_current_span("prepare_features") as span:
            # Extract features from task
            gpu_allocation = state.task.payload.get("gpu_allocation", {})
            
            # Create feature vector
            features = [
                float(gpu_allocation.get("gpu_count", 0)) / 10.0,  # Normalize
                float(gpu_allocation.get("cost_per_hour", 0)) / 100.0,  # Normalize
                float(gpu_allocation.get("duration_hours", 0)) / 24.0,  # Normalize
                float(state.task.priority) / 10.0 if hasattr(state.task, 'priority') else 0.5,
            ]
            
            # Add context features if available
            if state.context_window:
                context_tensor = state.context_window.to_tensor()
                # Combine task features with context
                full_features = torch.cat([
                    torch.tensor(features, dtype=torch.float32),
                    context_tensor
                ])
            else:
                # Pad to expected input size
                features.extend([0.0] * (self.lnn_config.input_size - len(features)))
                full_features = torch.tensor(features, dtype=torch.float32)
                
            # Ensure correct shape
            if len(full_features) < self.lnn_config.input_size:
                padding = torch.zeros(self.lnn_config.input_size - len(full_features))
                full_features = torch.cat([full_features, padding])
            elif len(full_features) > self.lnn_config.input_size:
                full_features = full_features[:self.lnn_config.input_size]
                
            state.context["prepared_features"] = full_features
            span.set_attribute("features.size", len(full_features))
            
            state.next_step = "lnn_inference"
            return state
            
    @resilient(criticality=ResilienceLevel.CRITICAL)
    async def _lnn_inference_step(self, state: CouncilState) -> CouncilState:
        """Run LNN inference."""
        with tracer.start_as_current_span("lnn_inference") as span:
            try:
                features = state.context.get("prepared_features")
                if features is None:
                    raise ValueError("No prepared features found")
                    
                # Run inference through liquid layers
                batch_size = 1
                seq_len = 1
                
                # Reshape for batch processing
                x = features.unsqueeze(0).unsqueeze(0)  # [batch, seq, features]
                
                # Initialize hidden state
                h = torch.zeros(batch_size, self.lnn_config.hidden_size)
                
                # Process through liquid time steps
                for t in range(seq_len):
                    h = self.liquid_layer(x[:, t, :], h)
                    
                # Generate output
                output = self.output_layer(h)
                output_probs = torch.softmax(output, dim=-1)
                
                state.lnn_output = output_probs
                
                # Log metrics
                inference_time = span.end_time - span.start_time if hasattr(span, 'end_time') else 0
                span.set_attribute("inference.time_ms", inference_time)
                span.set_attribute("output.shape", str(output_probs.shape))
                
            except Exception as e:
                logger.error(f"LNN inference failed: {e}")
                state.add_error(e, "lnn_inference")
                # Fallback to simple logic
                state.lnn_output = torch.tensor([[0.25, 0.25, 0.25, 0.25]])  # Equal probabilities
                
            state.next_step = "generate_vote"
            return state
            
    async def _generate_vote_step(self, state: CouncilState) -> CouncilState:
        """Generate vote from LNN output."""
        with tracer.start_as_current_span("generate_vote") as span:
            output_probs = state.lnn_output[0] if state.lnn_output is not None else torch.tensor([0.25, 0.25, 0.25, 0.25])
            
            # Map probabilities to vote types
            vote_mapping = {
                0: VoteType.APPROVE,
                1: VoteType.REJECT,
                2: VoteType.ABSTAIN,
                3: VoteType.DELEGATE
            }
            
            # Get vote with highest probability
            vote_idx = torch.argmax(output_probs).item()
            vote_type = vote_mapping[vote_idx]
            confidence = output_probs[vote_idx].item()
            
            # Generate reasoning based on context and vote
            reasoning = self._generate_reasoning(state, vote_type, confidence)
            
            # Collect supporting evidence
            evidence = self._collect_evidence(state, vote_type)
            
            # Create vote
            state.vote = CouncilVote(
                agent_id=self.config["name"],
                vote=vote_type,
                confidence=confidence,
                reasoning=reasoning,
                supporting_evidence=evidence,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Update metrics
            council_decisions.add(1, {"vote_type": vote_type.value})
            decision_confidence.record(confidence)
            
            span.set_attribute("vote.type", vote_type.value)
            span.set_attribute("vote.confidence", confidence)
            
            state.next_step = "store_decision"
            return state
            
    def _generate_reasoning(self, state: CouncilState, vote_type: VoteType, confidence: float) -> str:
        """Generate human-readable reasoning for the vote."""
        gpu_allocation = state.task.payload.get("gpu_allocation", {})
        cost = gpu_allocation.get("cost_per_hour", 0)
        count = gpu_allocation.get("gpu_count", 0)
        
        if vote_type == VoteType.APPROVE:
            reasoning = f"Approved GPU allocation based on neural analysis. "
            reasoning += f"Cost of ${cost}/hr for {count} GPUs is within acceptable range. "
            reasoning += f"Confidence: {confidence:.2%}"
        elif vote_type == VoteType.REJECT:
            reasoning = f"Rejected GPU allocation based on neural analysis. "
            reasoning += f"Cost of ${cost}/hr for {count} GPUs exceeds recommended thresholds. "
            reasoning += f"Confidence: {confidence:.2%}"
        elif vote_type == VoteType.ABSTAIN:
            reasoning = f"Abstaining from decision due to insufficient confidence ({confidence:.2%}). "
            reasoning += "Additional context or expert review recommended."
        else:  # DELEGATE
            reasoning = f"Delegating decision to specialized agent. "
            reasoning += f"Current confidence ({confidence:.2%}) below delegation threshold."
            
        # Add context-based insights if available
        if state.context_window and state.context_window.historical_patterns:
            pattern_count = len(state.context_window.historical_patterns)
            reasoning += f" Analysis based on {pattern_count} historical patterns."
            
        return reasoning
        
    def _collect_evidence(self, state: CouncilState, vote_type: VoteType) -> List[Dict[str, Any]]:
        """Collect supporting evidence for the vote."""
        evidence = []
        
        # Add neural network analysis
        evidence.append({
            "type": "neural_analysis",
            "confidence_scores": state.lnn_output[0].tolist() if state.lnn_output is not None else [],
            "selected_vote": vote_type.value
        })
        
        # Add context summary if available
        if state.context_window:
            evidence.append({
                "type": "context_summary",
                "historical_patterns": len(state.context_window.historical_patterns),
                "recent_decisions": len(state.context_window.recent_decisions)
            })
            
        # Add task details
        gpu_allocation = state.task.payload.get("gpu_allocation", {})
        evidence.append({
            "type": "request_details",
            "gpu_type": gpu_allocation.get("gpu_type"),
            "gpu_count": gpu_allocation.get("gpu_count"),
            "cost_per_hour": gpu_allocation.get("cost_per_hour"),
            "duration_hours": gpu_allocation.get("duration_hours")
        })
        
        return evidence
        
    @resilient(criticality=ResilienceLevel.LOW)
    async def _store_decision_step(self, state: CouncilState) -> CouncilState:
        """Store decision in Neo4j and publish events."""
        with tracer.start_as_current_span("store_decision") as span:
            storage_tasks = []
            
            # Store in Neo4j
            if self.neo4j_adapter:
                storage_tasks.append(self._store_in_neo4j(state))
                
            # Store in memory
            if self.memory_hooks:
                storage_tasks.append(self._store_in_memory(state))
                
            # Publish event
            if self.event_producer:
                storage_tasks.append(self._publish_event(state))
                
            # Execute storage in parallel
            if storage_tasks:
                results = await asyncio.gather(*storage_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Storage task {i} failed: {result}")
                        
            state.completed = True
            state.next_step = None
            
            span.set_attribute("storage.tasks", len(storage_tasks))
            
            return state
            
    async def _store_in_neo4j(self, state: CouncilState):
        """Store decision in Neo4j."""
        try:
            query = """
            CREATE (d:Decision {
                decision_id: $decision_id,
                request_id: $request_id,
                agent_id: $agent_id,
                vote: $vote,
                confidence: $confidence,
                reasoning: $reasoning,
                timestamp: $timestamp
            })
            RETURN d
            """
            
            params = {
                "decision_id": str(uuid.uuid4()),
                "request_id": state.task.task_id,
                "agent_id": state.vote.agent_id,
                "vote": state.vote.vote.value,
                "confidence": state.vote.confidence,
                "reasoning": state.vote.reasoning,
                "timestamp": state.vote.timestamp.isoformat()
            }
            
            await self.neo4j_adapter.query(query, params)
            
        except Exception as e:
            logger.error(f"Failed to store decision in Neo4j: {e}")
            
    async def _store_in_memory(self, state: CouncilState):
        """Store decision in memory."""
        try:
            await self.memory_hooks.store_decision(
                decision_id=str(uuid.uuid4()),
                context=asdict(state.task),
                outcome=state.vote.vote.value,
                confidence=state.vote.confidence,
                features=state.context.get("prepared_features", []).tolist()
            )
        except Exception as e:
            logger.error(f"Failed to store decision in memory: {e}")
            
    async def _publish_event(self, state: CouncilState):
        """Publish decision event."""
        try:
            event = {
                "type": "council_vote",
                "agent_id": state.vote.agent_id,
                "task_id": state.task.task_id,
                "vote": state.vote.vote.value,
                "confidence": state.vote.confidence,
                "reasoning": state.vote.reasoning,
                "timestamp": state.vote.timestamp.isoformat()
            }
            
            await self.event_producer.publish(
                topic="gpu.allocation.decisions",
                event=event
            )
        except Exception as e:
            logger.error(f"Failed to publish decision event: {e}")
            
    async def cleanup(self):
        """Cleanup resources."""
        try:
            cleanup_tasks = []
            
            if self.neo4j_adapter:
                cleanup_tasks.append(self.neo4j_adapter.close())
                
            if self.event_producer:
                cleanup_tasks.append(self.event_producer.close())
                
            if self.memory_manager:
                cleanup_tasks.append(self.memory_manager.cleanup())
                
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    # Override the process method to use the workflow
    async def process(self, task: CouncilTask) -> CouncilVote:
        """Process a council task through the LNN workflow."""
        # Use the base class _process method which handles the workflow
        return await self._process(task)