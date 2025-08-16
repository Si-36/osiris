"""
LNN Council Agent - Real Neural Network Decision Making

Senior-level implementation leveraging 2025 state-of-the-art:
- Closed-form LNN solutions for 1-5 orders of magnitude speedup
- Type-safe configuration with runtime validation
- Built-in observability and telemetry
- Modular, composable architecture
- Graceful degradation and fallback mechanisms
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import uuid
import torch
import numpy as np

from pydantic import BaseModel, Field, validator
import structlog

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from ..base import AgentBase, AgentState, AgentConfig
from ...lnn.core import LiquidNeuralNetwork, LiquidConfig, ActivationType, TimeConstants, WiringConfig
from ...config.agent import AgentSettings

# LangGraph imports for workflow building
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for when LangGraph is not available
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"

# Get tracer and meter for observability
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create LNN-specific metrics
lnn_inference_counter = meter.create_counter(
    name="lnn.inference.total",
    description="Total LNN inference calls",
    unit="1"
)

lnn_inference_duration = meter.create_histogram(
    name="lnn.inference.duration",
    description="LNN inference duration",
    unit="ms"
)

lnn_confidence_histogram = meter.create_histogram(
    name="lnn.confidence.score",
    description="LNN decision confidence scores",
    unit="1"
)

logger = structlog.get_logger()


@dataclass
class LNNCouncilConfig:
    """
    Configuration for LNN Council Agent.
    
    Follows 2025 best practices for type-safe, hierarchical configuration
    with runtime validation and sensible defaults.
    """
    
    # Agent identification
    name: str = "lnn_council_agent"
    version: str = "1.0.0"
    
    # LNN Neural Network Configuration
    input_size: int = 256
    output_size: int = 64
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 96, 64])
    
    # LNN-specific settings
    activation_type: ActivationType = ActivationType.LIQUID
    solver_type: str = "rk4"  # rk4 for stability, euler for speed
    dt: float = 0.01
    use_adaptive_dt: bool = True
    
    # Time constants for liquid dynamics
    tau_min: float = 0.1
    tau_max: float = 10.0
    tau_init: str = "log_uniform"  # Better for diverse dynamics
    adaptive_tau: bool = True
    
    # Sparse wiring configuration
    sparsity: float = 0.8  # 80% sparse for efficiency
    wiring_type: str = "small_world"  # Better for hierarchical decisions
    enable_self_connections: bool = True
    learnable_wiring: bool = True
    prune_threshold: float = 0.01
    
    # Decision-making parameters
    confidence_threshold: float = 0.7
    max_inference_time: float = 2.0  # 2 second SLA
    enable_fallback: bool = True
    fallback_threshold: float = 0.5
    
    # Memory and context configuration
    context_cache_size: int = 1000
    decision_history_limit: int = 10000
    enable_memory_learning: bool = True
    memory_update_frequency: int = 10  # Update every 10 decisions
    
    # Knowledge graph integration
    max_context_nodes: int = 100
    context_query_timeout: float = 1.0
    enable_context_caching: bool = True
    context_relevance_threshold: float = 0.6
    
    # Performance optimization
    batch_size: int = 32
    use_gpu: bool = True
    mixed_precision: bool = True
    compile_mode: Optional[str] = "reduce-overhead"  # torch.compile optimization
    
    # Observability and monitoring
    enable_detailed_logging: bool = True
    log_decision_reasoning: bool = True
    emit_performance_metrics: bool = True
    trace_neural_dynamics: bool = False  # Expensive, only for debugging
    
    def validate(self) -> None:
        """Validate configuration with comprehensive checks."""
        if not self.name:
            raise ValueError("Agent name is required")
        
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("Input and output sizes must be positive")
        
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("Hidden sizes must be positive")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.max_inference_time <= 0:
            raise ValueError("Max inference time must be positive")
        
        if not (0.0 <= self.sparsity <= 1.0):
            raise ValueError("Sparsity must be between 0 and 1")
        
        if self.tau_min <= 0 or self.tau_max <= self.tau_min:
            raise ValueError("Invalid time constant range")
    
    def to_liquid_config(self) -> LiquidConfig:
        """Convert to LiquidConfig for neural network initialization."""
        time_constants = TimeConstants(
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            tau_init=self.tau_init,
            adaptive=self.adaptive_tau
        )
        
        wiring = WiringConfig(
            sparsity=self.sparsity,
            wiring_type=self.wiring_type,
            self_connections=self.enable_self_connections,
            learnable_wiring=self.learnable_wiring,
            prune_threshold=self.prune_threshold
        )
        
        return LiquidConfig(
            time_constants=time_constants,
            activation=self.activation_type,
            use_bias=True,
            wiring=wiring,
            hidden_sizes=self.hidden_sizes,
            solver_type=self.solver_type,
            dt=self.dt,
            adaptive_dt=self.use_adaptive_dt,
            liquid_reg=0.01,
            sparsity_reg=0.001,
            stability_reg=0.001,
            use_cuda=self.use_gpu,
            mixed_precision=self.mixed_precision,
            compile_mode=self.compile_mode
        )
    
    def to_agent_config(self) -> AgentConfig:
        """Convert to base AgentConfig for compatibility."""
        return AgentConfig(
            name=self.name,
            model="lnn-council",  # Custom model identifier
            temperature=0.0,  # Not applicable for LNN
            max_retries=3,
            timeout_seconds=int(self.max_inference_time),
            enable_memory=self.enable_memory_learning,
            enable_tools=True
        )


class GPUAllocationRequest(BaseModel):
    """GPU allocation request model with comprehensive validation."""
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., min_length=1)
    project_id: str = Field(..., min_length=1)
    
    # GPU requirements
    gpu_type: str = Field(..., pattern=r'^(A100|H100|V100|RTX4090|RTX3090)$')
    gpu_count: int = Field(..., ge=1, le=8)
    memory_gb: int = Field(..., ge=1, le=80)
    compute_hours: float = Field(..., ge=0.1, le=168.0)  # Max 1 week
    
    # Priority and scheduling
    priority: int = Field(default=5, ge=1, le=10)
    deadline: Optional[datetime] = None
    flexible_scheduling: bool = Field(default=True)
    
    # Special requirements
    special_requirements: List[str] = Field(default_factory=list)
    requires_infiniband: bool = Field(default=False)
    requires_nvlink: bool = Field(default=False)
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('special_requirements')
    def validate_special_requirements(cls, v):
        """Validate special requirements."""
        allowed = {'high_memory', 'low_latency', 'multi_gpu', 'distributed', 'inference_only'}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid special requirements: {invalid}")
        return v


class GPUAllocationDecision(BaseModel):
    """GPU allocation decision output with detailed reasoning."""
    
    request_id: str
    decision: str = Field(..., pattern=r'^(approve|deny|defer)$')
    
    # Allocation details (if approved)
    allocated_gpu_ids: Optional[List[str]] = None
    allocated_gpu_type: Optional[str] = None
    allocated_memory_gb: Optional[int] = None
    time_slot_start: Optional[datetime] = None
    time_slot_end: Optional[datetime] = None
    estimated_cost: Optional[float] = None
    
    # Decision metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_path: List[str] = Field(default_factory=list)
    fallback_used: bool = Field(default=False)
    inference_time_ms: float = Field(..., ge=0.0)
    
    # Context used in decision
    context_nodes_used: int = Field(default=0)
    historical_decisions_considered: int = Field(default=0)
    
    # Timestamps
    decision_made_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_reasoning(self, step: str, explanation: str) -> None:
        """Add reasoning step to decision path."""
        self.reasoning_path.append(f"{step}: {explanation}")


class LNNCouncilState(AgentState):
    """Extended state for LNN Council Agent with neural network state."""
    
    # Neural network state
    lnn_hidden_state: Optional[torch.Tensor] = None
    neural_dynamics: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision context
    current_request: Optional[GPUAllocationRequest] = None
    context_cache: Dict[str, Any] = Field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance tracking
    inference_start_time: Optional[float] = None
    context_retrieval_time: float = 0.0
    neural_inference_time: float = 0.0
    
    # Confidence and validation
    confidence_score: float = 0.0
    validation_passed: bool = False
    fallback_triggered: bool = False
    
    class Config:
        arbitrary_types_allowed = True  # Allow torch.Tensor


class LNNCouncilAgent(AgentBase[GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState]):
    """
    LNN Council Agent for GPU allocation decisions.
    
    Features 2025 state-of-the-art architecture:
    - Real liquid neural network inference
    - Context-aware decision making
    - Memory integration and learning
    - Comprehensive observability
    - Graceful fallback mechanisms
    """
    
    def __init__(self, config: Union[Dict[str, Any], LNNCouncilConfig, AgentConfig]):
        """Initialize LNN Council Agent with flexible configuration."""
        
        # Handle different configuration types
        if isinstance(config, dict):
            self.lnn_config = LNNCouncilConfig(**config)
        elif isinstance(config, LNNCouncilConfig):
            self.lnn_config = config
        elif isinstance(config, AgentConfig):
            # Convert from base AgentConfig with defaults
            self.lnn_config = LNNCouncilConfig(name=config.name)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
        
        # Validate configuration
        self.lnn_config.validate()
        
        # Initialize base agent
        super().__init__(self.lnn_config.to_agent_config())
        
        # Initialize neural network components
        self.lnn_engine: Optional[LiquidNeuralNetwork] = None
        self.memory_integration = None  # Will be initialized in _initialize_subsystems
        self.knowledge_context = None  # Will be initialized in _initialize_subsystems
        self.fallback_engine = None   # Will be initialized in _initialize_subsystems
        
        # Performance tracking
        self._total_decisions = 0
        self._successful_decisions = 0
        self._fallback_decisions = 0
        self._avg_inference_time = 0.0
        
        self.logger = logger.bind(
            agent=self.lnn_config.name,
            version=self.lnn_config.version
        )
        
        self.logger.info(
            "LNN Council Agent initialized",
            config=self.lnn_config.__dict__
        )
    
    def _initialize_lnn_engine(self) -> LiquidNeuralNetwork:
        """Initialize the liquid neural network engine."""
        liquid_config = self.lnn_config.to_liquid_config()
        
        lnn = LiquidNeuralNetwork(
            input_size=self.lnn_config.input_size,
            output_size=self.lnn_config.output_size,
            config=liquid_config
        )
        
        # Move to GPU if available and configured
        if self.lnn_config.use_gpu and torch.cuda.is_available():
            lnn = lnn.cuda()
            self.logger.info("LNN moved to GPU")
        
        # Enable mixed precision if configured
        if self.lnn_config.mixed_precision:
            lnn = lnn.half()
            self.logger.info("Mixed precision enabled")
        
        return lnn
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for GPU allocation decisions.
        
        Creates a 5-step workflow:
        1. analyze_request -> Analyze GPU allocation request complexity
        2. gather_context -> Retrieve context from knowledge graphs and memory
        3. neural_inference -> Run LNN inference for decision making
        4. validate_decision -> Validate decision against constraints
        5. finalize_output -> Finalize decision output
        
        Returns:
            StateGraph defining the complete decision workflow
        """
        if not LANGGRAPH_AVAILABLE:
            # Return a mock graph when LangGraph is not available
            self.logger.warning("LangGraph not available, using fallback workflow")
            return None
        
        # Create the workflow graph
        workflow = StateGraph(LNNCouncilState)
        
        # Add nodes for each step
        workflow.add_node("analyze_request", self._analyze_request_node)
        workflow.add_node("gather_context", self._gather_context_node)
        workflow.add_node("neural_inference", self._neural_inference_node)
        workflow.add_node("validate_decision", self._validate_decision_node)
        workflow.add_node("finalize_output", self._finalize_output_node)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze_request")
        
        # Conditional routing based on request complexity
        workflow.add_conditional_edges(
            "analyze_request",
            self._route_after_analysis,
            {
                "gather_context": "gather_context",
                "neural_inference": "neural_inference"
            }
        )
        
        workflow.add_edge("gather_context", "neural_inference")
        
        # Conditional routing based on confidence
        workflow.add_conditional_edges(
            "neural_inference",
            self._route_after_inference,
            {
                "validate_decision": "validate_decision",
                "fallback": "validate_decision"  # Both go to validation
            }
        )
        
        workflow.add_edge("validate_decision", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        self.logger.info("LangGraph workflow built successfully")
        return workflow
    
    def _route_after_analysis(self, state: LNNCouncilState) -> str:
        """Route after request analysis based on complexity."""
        complexity = state.context.get("request_complexity", 0.5)
        
        if complexity > 0.8:
            return "gather_context"  # High complexity needs more context
        else:
            return "neural_inference"  # Simple request can go straight to inference
    
    def _route_after_inference(self, state: LNNCouncilState) -> str:
        """Route after neural inference based on confidence."""
        if state.fallback_triggered:
            return "fallback"
        else:
            return "validate_decision"
    
    # LangGraph node wrappers (these call our existing methods)
    async def _analyze_request_node(self, state: LNNCouncilState) -> LNNCouncilState:
        """LangGraph node wrapper for analyze_request."""
        return await self._analyze_request(state)
    
    async def _gather_context_node(self, state: LNNCouncilState) -> LNNCouncilState:
        """LangGraph node wrapper for gather_context."""
        return await self._gather_context(state)
    
    async def _neural_inference_node(self, state: LNNCouncilState) -> LNNCouncilState:
        """LangGraph node wrapper for neural_inference."""
        return await self._neural_inference(state)
    
    async def _validate_decision_node(self, state: LNNCouncilState) -> LNNCouncilState:
        """LangGraph node wrapper for validate_decision."""
        return await self._validate_decision(state)
    
    async def _finalize_output_node(self, state: LNNCouncilState) -> LNNCouncilState:
        """LangGraph node wrapper for finalize_output."""
        return await self._finalize_output(state)

    def _create_initial_state(self, input_data: GPUAllocationRequest) -> LNNCouncilState:
        """Create initial state for LNN processing."""
        return LNNCouncilState(
            current_request=input_data,
            current_step="analyze_request",
            next_step="gather_context",
            inference_start_time=asyncio.get_event_loop().time()
        )
    
    async def _execute_step(self, state: LNNCouncilState, step_name: str) -> LNNCouncilState:
        """
        Execute workflow step using real LNN inference and business logic.
        
        This is the core method that implements the abstract _execute_step from AgentBase.
        It provides comprehensive step execution with error handling, fallback mechanisms,
        and detailed observability.
        
        Args:
            state: Current LNN Council state
            step_name: Name of the step to execute
            
        Returns:
            Updated state after step execution
            
        Raises:
            ValueError: If step_name is unknown
            RuntimeError: If step execution fails and fallback is not available
        """
        
        with tracer.start_as_current_span(f"lnn_council.step.{step_name}") as span:
            span.set_attributes({
                "step.name": step_name,
                "request.id": state.current_request.request_id if state.current_request else "unknown",
                "agent.name": self.name,
                "fallback.triggered": state.fallback_triggered
            })
            
            step_start_time = asyncio.get_event_loop().time()
            
            try:
                # Route to appropriate step implementation
                if step_name == "analyze_request":
                    result_state = await self._analyze_request(state)
                elif step_name == "gather_context":
                    result_state = await self._gather_context(state)
                elif step_name == "neural_inference":
                    result_state = await self._neural_inference(state)
                elif step_name == "validate_decision":
                    result_state = await self._validate_decision(state)
                elif step_name == "finalize_output":
                    result_state = await self._finalize_output(state)
                else:
                    # Handle unknown steps gracefully
                    self.logger.error(f"Unknown step requested: {step_name}")
                    raise ValueError(f"Unknown step: {step_name}")
                
                # Record successful step execution
                step_duration = (asyncio.get_event_loop().time() - step_start_time) * 1000
                span.set_attributes({
                    "step.duration_ms": step_duration,
                    "step.success": True,
                    "step.next": result_state.next_step
                })
                
                self.logger.info(
                    f"Step {step_name} completed successfully",
                    duration_ms=step_duration,
                    next_step=result_state.next_step,
                    confidence=getattr(result_state, 'confidence_score', None)
                )
                
                return result_state
                    
            except Exception as e:
                step_duration = (asyncio.get_event_loop().time() - step_start_time) * 1000
                
                # Record error in telemetry
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attributes({
                    "step.duration_ms": step_duration,
                    "step.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                # Log detailed error information
                self.logger.error(
                    f"Step {step_name} failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=step_duration,
                    fallback_available=self.lnn_config.enable_fallback and not state.fallback_triggered
                )
                
                # Attempt fallback if available and not already triggered
                if self.lnn_config.enable_fallback and not state.fallback_triggered:
                    self.logger.warning(
                        f"Triggering fallback for failed step: {step_name}",
                        original_error=str(e)
                    )
                    
                    try:
                        state.fallback_triggered = True
                        fallback_state = await self._fallback_decision(state)
                        
                        # Record successful fallback
                        span.add_event("fallback_triggered", {
                            "fallback.success": True,
                            "fallback.decision": fallback_state.context.get("neural_decision", "unknown")
                        })
                        
                        return fallback_state
                        
                    except Exception as fallback_error:
                        # Fallback also failed
                        self.logger.error(
                            f"Fallback also failed for step {step_name}",
                            fallback_error=str(fallback_error),
                            original_error=str(e)
                        )
                        
                        span.add_event("fallback_failed", {
                            "fallback.error": str(fallback_error)
                        })
                        
                        # Re-raise original error
                        raise RuntimeError(
                            f"Step {step_name} failed and fallback also failed. "
                            f"Original error: {str(e)}, Fallback error: {str(fallback_error)}"
                        )
                
                # No fallback available or already used, re-raise the error
                raise RuntimeError(f"Step {step_name} failed: {str(e)}")
    
    async def _analyze_request(self, state: LNNCouncilState) -> LNNCouncilState:
        """Analyze the incoming GPU allocation request."""
        
        request = state.current_request
        if not request:
            raise ValueError("No request to analyze")
        
        # Basic request validation and preprocessing
        state.add_message("system", f"Analyzing GPU allocation request {request.request_id}")
        
        # Calculate request complexity score
        complexity_factors = [
            request.gpu_count / 8.0,  # Normalized by max GPUs
            request.memory_gb / 80.0,  # Normalized by max memory
            request.compute_hours / 168.0,  # Normalized by max hours
            len(request.special_requirements) / 5.0,  # Normalized by max requirements
            (10 - request.priority) / 9.0  # Inverted priority (higher priority = lower complexity)
        ]
        
        complexity_score = np.mean(complexity_factors)
        state.context["request_complexity"] = complexity_score
        
        # Determine next step based on complexity
        if complexity_score > 0.8:
            state.next_step = "gather_context"  # High complexity needs more context
        else:
            state.next_step = "neural_inference"  # Simple request can go straight to inference
        
        self.logger.info(
            "Request analyzed",
            request_id=request.request_id,
            complexity_score=complexity_score,
            next_step=state.next_step
        )
        
        return state
    
    async def _gather_context(self, state: LNNCouncilState) -> LNNCouncilState:
        """Gather context from knowledge graphs and memory systems."""
        
        context_start = asyncio.get_event_loop().time()
        
        # TODO: Integrate with actual Neo4j and Mem0 adapters
        # For now, simulate context gathering
        await asyncio.sleep(0.1)  # Simulate I/O
        
        # Simulated context data
        state.context_cache.update({
            "current_utilization": {"gpu_usage": 0.75, "queue_length": 12},
            "user_history": {"successful_allocations": 15, "avg_usage": 0.85},
            "project_context": {"budget_remaining": 5000.0, "priority_tier": "high"},
            "system_constraints": {"maintenance_window": None, "capacity_limit": 0.9}
        })
        
        state.context_retrieval_time = asyncio.get_event_loop().time() - context_start
        state.next_step = "neural_inference"
        
        self.logger.info(
            "Context gathered",
            retrieval_time_ms=state.context_retrieval_time * 1000,
            context_keys=list(state.context_cache.keys())
        )
        
        return state
    
    async def _neural_inference(self, state: LNNCouncilState) -> LNNCouncilState:
        """Perform neural network inference for decision making with enhanced confidence scoring."""
        
        inference_start = asyncio.get_event_loop().time()
        
        # Initialize LNN if not already done
        if self.lnn_engine is None:
            self.lnn_engine = self._initialize_lnn_engine()
        
        # Initialize confidence scorer if not already done
        if not hasattr(self, 'confidence_scorer'):
            from .confidence_scoring import ConfidenceScorer
            self.confidence_scorer = ConfidenceScorer(self.lnn_config.__dict__)
        
        with tracer.start_as_current_span("lnn.inference") as span:
            try:
                # Encode request and context into neural network input
                input_tensor = self._encode_input(state)
                
                # Run LNN inference
                with torch.no_grad():
                    if self.lnn_config.trace_neural_dynamics:
                        output, dynamics = self.lnn_engine(input_tensor, return_dynamics=True)
                        state.neural_dynamics = dynamics
                    else:
                        output = self.lnn_engine(input_tensor)
                
                # Decode neural output into decision
                decision_logits = output.squeeze()
                decision_idx = torch.argmax(decision_logits).item()
                decisions = ["deny", "defer", "approve"]
                decision = decisions[min(decision_idx, len(decisions) - 1)]
                
                # Enhanced confidence scoring using the dedicated confidence scorer
                confidence_metrics = self.confidence_scorer.calculate_confidence(
                    neural_output=output,
                    state=state,
                    decision=decision
                )
                
                # Use the comprehensive confidence score
                confidence_score = confidence_metrics.overall_confidence
                
                # Store detailed confidence information
                state.confidence_score = confidence_score
                state.context["neural_decision"] = decision
                state.context["decision_logits"] = decision_logits.tolist()
                state.context["confidence_metrics"] = confidence_metrics
                state.context["confidence_breakdown"] = confidence_metrics.confidence_breakdown
                
                # Record metrics
                lnn_inference_counter.add(1, {"decision": decision})
                lnn_confidence_histogram.record(confidence_score)
                
                inference_time = asyncio.get_event_loop().time() - inference_start
                state.neural_inference_time = inference_time
                lnn_inference_duration.record(inference_time * 1000)  # Convert to ms
                
                # Enhanced confidence threshold checking
                if confidence_score >= self.lnn_config.confidence_threshold:
                    state.next_step = "validate_decision"
                    
                    self.logger.info(
                        "High confidence neural inference",
                        decision=decision,
                        confidence=confidence_score,
                        neural_conf=confidence_metrics.neural_confidence,
                        context_quality=confidence_metrics.context_quality,
                        constraint_satisfaction=confidence_metrics.constraint_satisfaction
                    )
                else:
                    # Low confidence, trigger fallback
                    self.logger.warning(
                        "Low confidence decision, triggering fallback",
                        confidence=confidence_score,
                        threshold=self.lnn_config.confidence_threshold,
                        neural_conf=confidence_metrics.neural_confidence,
                        context_quality=confidence_metrics.context_quality,
                        entropy=confidence_metrics.output_entropy
                    )
                    state.fallback_triggered = True
                    return await self._fallback_decision(state)
                
                span.set_attributes({
                    "inference.decision": decision,
                    "inference.confidence": confidence_score,
                    "inference.neural_confidence": confidence_metrics.neural_confidence,
                    "inference.context_quality": confidence_metrics.context_quality,
                    "inference.time_ms": inference_time * 1000
                })
                
                self.logger.info(
                    "Enhanced neural inference completed",
                    decision=decision,
                    confidence=confidence_score,
                    inference_time_ms=inference_time * 1000,
                    confidence_breakdown=confidence_metrics.confidence_breakdown
                )
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise RuntimeError(f"Neural inference failed: {str(e)}")
        
        return state
    
    def _encode_input(self, state: LNNCouncilState) -> torch.Tensor:
        """Encode request and context into neural network input tensor."""
        
        request = state.current_request
        if not request:
            raise ValueError("No request to encode")
        
        # Create feature vector (simplified for now)
        features = []
        
        # Request features
        gpu_type_encoding = {"A100": 1.0, "H100": 0.9, "V100": 0.8, "RTX4090": 0.7, "RTX3090": 0.6}
        features.extend([
            gpu_type_encoding.get(request.gpu_type, 0.5),
            request.gpu_count / 8.0,  # Normalized
            request.memory_gb / 80.0,  # Normalized
            request.compute_hours / 168.0,  # Normalized
            request.priority / 10.0,  # Normalized
            float(request.requires_infiniband),
            float(request.requires_nvlink),
            len(request.special_requirements) / 5.0  # Normalized
        ])
        
        # Context features
        context = state.context_cache
        if context:
            utilization = context.get("current_utilization", {})
            features.extend([
                utilization.get("gpu_usage", 0.5),
                utilization.get("queue_length", 0) / 20.0  # Normalized
            ])
            
            user_history = context.get("user_history", {})
            features.extend([
                user_history.get("successful_allocations", 0) / 50.0,  # Normalized
                user_history.get("avg_usage", 0.5)
            ])
        
        # Pad or truncate to input_size
        while len(features) < self.lnn_config.input_size:
            features.append(0.0)
        features = features[:self.lnn_config.input_size]
        
        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        if self.lnn_config.use_gpu and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        if self.lnn_config.mixed_precision:
            tensor = tensor.half()
        
        return tensor
    
    async def _validate_decision(self, state: LNNCouncilState) -> LNNCouncilState:
        """Validate the neural network decision against constraints using DecisionValidator."""
        
        decision = state.context.get("neural_decision")
        request = state.current_request
        
        if not decision or not request:
            raise ValueError("Missing decision or request for validation")
        
        # Initialize decision validator if not already done
        if not hasattr(self, 'decision_validator'):
            from .confidence_scoring import DecisionValidator
            self.decision_validator = DecisionValidator(self.lnn_config.__dict__)
        
        # Perform comprehensive validation
        validation_result = self.decision_validator.validate_decision(
            decision=decision,
            request=request,
            state=state
        )
        
        # Store validation results
        state.validation_passed = validation_result.is_valid
        state.context["validation_result"] = validation_result
        state.context["validation_score"] = validation_result.validation_score
        state.context["validation_violations"] = validation_result.violations
        state.context["validation_warnings"] = validation_result.warnings
        state.context["constraints_checked"] = validation_result.constraints_checked
        
        # Determine next action based on validation
        if validation_result.is_valid:
            state.next_step = "finalize_output"
            
            self.logger.info(
                "Decision validation passed",
                decision=decision,
                validation_score=validation_result.validation_score,
                constraints_checked=len(validation_result.constraints_checked),
                warnings=len(validation_result.warnings)
            )
        else:
            # Override decision if validation failed
            original_decision = decision
            state.context["neural_decision"] = "deny"
            state.context["override_reason"] = "validation_failed"
            state.context["original_decision"] = original_decision
            state.next_step = "finalize_output"
            
            self.logger.warning(
                "Decision validation failed - overriding to deny",
                original_decision=original_decision,
                violations=len(validation_result.violations),
                validation_score=validation_result.validation_score,
                violation_details=validation_result.violations[:3]  # Log first 3 violations
            )
        
        # Log warnings even if validation passed
        if validation_result.warnings:
            self.logger.warning(
                "Decision validation warnings",
                decision=decision,
                warnings=len(validation_result.warnings),
                warning_details=validation_result.warnings[:2]  # Log first 2 warnings
            )
        
        return state
    
    async def _finalize_output(self, state: LNNCouncilState) -> LNNCouncilState:
        """Finalize the decision output."""
        
        state.completed = True
        state.next_step = None
        
        # Update performance metrics
        self._total_decisions += 1
        if state.validation_passed and not state.fallback_triggered:
            self._successful_decisions += 1
        if state.fallback_triggered:
            self._fallback_decisions += 1
        
        total_time = asyncio.get_event_loop().time() - (state.inference_start_time or 0)
        self._avg_inference_time = (self._avg_inference_time * (self._total_decisions - 1) + total_time) / self._total_decisions
        
        self.logger.info(
            "Decision finalized",
            total_time_ms=total_time * 1000,
            fallback_used=state.fallback_triggered
        )
        
        return state
    
    async def _fallback_decision(self, state: LNNCouncilState) -> LNNCouncilState:
        """Make fallback decision using rule-based logic."""
        
        request = state.current_request
        if not request:
            raise ValueError("No request for fallback decision")
        
        # Simple rule-based decision logic
        if request.priority >= 8:
            decision = "approve"
            confidence = 0.6
        elif request.gpu_count > 4:
            decision = "defer"
            confidence = 0.7
        else:
            decision = "approve"
            confidence = 0.5
        
        state.context["neural_decision"] = decision
        state.confidence_score = confidence
        state.fallback_triggered = True
        state.next_step = "validate_decision"
        
        self.logger.info(
            "Fallback decision made",
            decision=decision,
            confidence=confidence
        )
        
        return state
    
    def _extract_output(self, final_state: LNNCouncilState) -> GPUAllocationDecision:
        """Extract final decision output from state with comprehensive reasoning."""
        
        request = final_state.current_request
        if not request:
            raise ValueError("No request in final state")
        
        decision = final_state.context.get("neural_decision", "deny")
        
        # Initialize reasoning path generator if not already done
        if not hasattr(self, 'reasoning_generator'):
            from .confidence_scoring import ReasoningPathGenerator
            self.reasoning_generator = ReasoningPathGenerator({
                "include_technical_details": self.lnn_config.log_decision_reasoning,
                "max_reasoning_steps": 12
            })
        
        # Create decision output
        output = GPUAllocationDecision(
            request_id=request.request_id,
            decision=decision,
            confidence_score=final_state.confidence_score,
            fallback_used=final_state.fallback_triggered,
            inference_time_ms=(final_state.neural_inference_time or 0) * 1000,
            context_nodes_used=len(final_state.context_cache),
            historical_decisions_considered=len(final_state.decision_history)
        )
        
        # Generate comprehensive reasoning path
        confidence_metrics = final_state.context.get("confidence_metrics")
        validation_result = final_state.context.get("validation_result")
        
        if confidence_metrics and validation_result:
            try:
                reasoning_path = self.reasoning_generator.generate_reasoning_path(
                    decision=decision,
                    request=request,
                    confidence_metrics=confidence_metrics,
                    validation_result=validation_result,
                    state=final_state
                )
                
                # Add reasoning steps to output
                for step in reasoning_path:
                    output.add_reasoning("reasoning", step)
                    
                self.logger.debug(
                    "Generated comprehensive reasoning path",
                    request_id=request.request_id,
                    reasoning_steps=len(reasoning_path)
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to generate reasoning path: {e}")
                # Fallback to basic reasoning
                self._add_basic_reasoning(output, final_state)
        else:
            # Fallback to basic reasoning if detailed metrics not available
            self._add_basic_reasoning(output, final_state)
        
        # Add allocation details if approved
        if decision == "approve":
            output.allocated_gpu_type = request.gpu_type
            output.allocated_memory_gb = request.memory_gb
            
            # Calculate cost based on GPU type
            gpu_costs = {"A100": 3.0, "H100": 4.0, "V100": 2.0, "RTX4090": 1.5, "RTX3090": 1.0}
            cost_per_hour = gpu_costs.get(request.gpu_type, 2.5)
            output.estimated_cost = request.gpu_count * request.compute_hours * cost_per_hour
            
            # Add time slot information (simulated for now)
            from datetime import timedelta
            output.time_slot_start = datetime.now(timezone.utc) + timedelta(minutes=30)
            output.time_slot_end = output.time_slot_start + timedelta(hours=request.compute_hours)
            
            # Add GPU IDs (simulated for now)
            output.allocated_gpu_ids = [f"gpu-{i:02d}" for i in range(request.gpu_count)]
        
        return output
    
    def _add_basic_reasoning(self, output: GPUAllocationDecision, final_state: LNNCouncilState):
        """Add basic reasoning when detailed reasoning generation fails."""
        
        # Basic fallback reasoning
        if final_state.fallback_triggered:
            output.add_reasoning("fallback", "Used rule-based fallback due to low neural confidence")
        
        # Validation reasoning
        validation_violations = final_state.context.get("validation_violations", [])
        validation_warnings = final_state.context.get("validation_warnings", [])
        
        if validation_violations:
            output.add_reasoning("validation", f"Found {len(validation_violations)} constraint violations")
            for violation in validation_violations[:2]:  # Add first 2 violations
                output.add_reasoning("constraint", violation)
        
        if validation_warnings:
            output.add_reasoning("validation", f"Found {len(validation_warnings)} warnings")
            for warning in validation_warnings[:1]:  # Add first warning
                output.add_reasoning("warning", warning)
        
        # Confidence reasoning
        confidence = final_state.confidence_score
        if confidence >= 0.8:
            output.add_reasoning("confidence", f"High confidence decision ({confidence:.2f})")
        elif confidence >= 0.6:
            output.add_reasoning("confidence", f"Moderate confidence decision ({confidence:.2f})")
        else:
            output.add_reasoning("confidence", f"Low confidence decision ({confidence:.2f})")
        
        # Context reasoning
        context_quality = len(final_state.context_cache) / 4.0  # Rough quality estimate
        if context_quality >= 0.8:
            output.add_reasoning("context", "Rich context available for decision")
        elif context_quality >= 0.5:
            output.add_reasoning("context", "Moderate context available")
        else:
            output.add_reasoning("context", "Limited context available")
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with LNN-specific metrics."""
        
        base_health = await super().health_check()
        
        # Add LNN-specific health metrics
        lnn_health = {
            "lnn_engine_initialized": self.lnn_engine is not None,
            "total_decisions": self._total_decisions,
            "successful_decisions": self._successful_decisions,
            "fallback_decisions": self._fallback_decisions,
            "success_rate": self._successful_decisions / max(1, self._total_decisions),
            "fallback_rate": self._fallback_decisions / max(1, self._total_decisions),
            "avg_inference_time_ms": self._avg_inference_time * 1000,
            "gpu_available": torch.cuda.is_available() if self.lnn_config.use_gpu else False
        }
        
        # Check neural network health
        if self.lnn_engine is not None:
            try:
                # Test inference with dummy input
                dummy_input = torch.randn(1, self.lnn_config.input_size)
                if self.lnn_config.use_gpu and torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                
                with torch.no_grad():
                    _ = self.lnn_engine(dummy_input)
                
                lnn_health["neural_network_status"] = "healthy"
            except Exception as e:
                lnn_health["neural_network_status"] = f"error: {str(e)}"
        else:
            lnn_health["neural_network_status"] = "not_initialized"
        
        base_health.update(lnn_health)
        return base_health
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        base_capabilities = super().get_capabilities()
        lnn_capabilities = [
            "liquid_neural_networks",
            "gpu_allocation_decisions",
            "context_aware_inference",
            "fallback_mechanisms",
            "performance_monitoring"
        ]
        return base_capabilities + lnn_capabilities