"""
AURA Execution Schemas - Professional Pydantic V2 Models for State Management
September 2025 - State-of-the-art type-safe data structures
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field, ConfigDict
import uuid
from datetime import datetime
from enum import Enum


# --- Enums for Type Safety ---
class TaskStatus(str, Enum):
    """Status of a task in the execution pipeline"""
    PENDING = "pending"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    CONSENSUS = "consensus"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(str, Enum):
    """Roles for agents in the system"""
    PERCEPTION = "perception"
    PLANNER = "planner"
    EXECUTOR = "executor"
    ANALYST = "analyst"
    CONSENSUS = "consensus"


# --- Core Data Structures ---
class TopologicalSignature(BaseModel):
    """The mathematical fingerprint of an observation using TDA"""
    model_config = ConfigDict(frozen=False)
    
    betti_numbers: List[int] = Field(
        ..., 
        description="Betti numbers [b0, b1, b2] representing connected components, loops, and voids"
    )
    persistence_entropy: float = Field(
        ..., 
        ge=0.0,
        description="A measure of the complexity of the topological features"
    )
    wasserstein_distance_from_norm: float = Field(
        ..., 
        ge=0.0,
        description="How far the current shape is from a 'normal' baseline"
    )
    persistence_diagram: Optional[List[List[float]]] = Field(
        default=None,
        description="Full persistence diagram for detailed analysis"
    )
    motif_cost_index: Optional[float] = Field(
        default=None,
        description="MotifCost index from AURA's TDA engine"
    )


class MemoryContext(BaseModel):
    """Context retrieved from AURA's UnifiedCognitiveMemory"""
    model_config = ConfigDict(frozen=False)
    
    episodic_memories: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant episodic memories"
    )
    semantic_concepts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Related semantic graph nodes"
    )
    causal_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified causal relationships"
    )
    working_memory_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Current working memory state"
    )
    synthesis: Optional[str] = Field(
        default=None,
        description="Synthesized understanding from memory"
    )


class ObservationResult(BaseModel):
    """A validated data structure for observation tool outputs"""
    model_config = ConfigDict(frozen=False)
    
    observation_id: str = Field(
        default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}"
    )
    source: str = Field(..., description="Data source (e.g., 'prometheus:user-db')")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(..., description="Raw observation data")
    topology: TopologicalSignature = Field(..., description="Topological analysis")
    anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected anomalies with confidence scores"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the observation"
    )


class ExecutionStep(BaseModel):
    """A single step in an execution plan"""
    model_config = ConfigDict(frozen=False)
    
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    tool: str = Field(..., description="Tool to execute")
    params: Dict[str, Any] = Field(..., description="Parameters for the tool")
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of steps this depends on"
    )
    expected_output_type: Optional[str] = Field(
        default=None,
        description="Expected type of output"
    )


class ExecutionPlan(BaseModel):
    """A complete execution plan from the planner agent"""
    model_config = ConfigDict(frozen=False)
    
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    objective: str = Field(..., description="What this plan aims to achieve")
    steps: List[ExecutionStep] = Field(..., description="Ordered execution steps")
    estimated_duration: Optional[float] = Field(
        default=None,
        description="Estimated time in seconds"
    )
    risk_assessment: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk scores for different failure modes"
    )
    parallelization_possible: bool = Field(
        default=False,
        description="Whether steps can be executed in parallel"
    )


class ConsensusDecision(BaseModel):
    """Result of agent consensus process"""
    model_config = ConfigDict(frozen=False)
    
    decision_id: str = Field(default_factory=lambda: f"consensus_{uuid.uuid4().hex[:8]}")
    approved: bool = Field(..., description="Whether consensus was reached")
    approved_plan: Optional[ExecutionPlan] = Field(
        default=None,
        description="The approved plan (may be modified)"
    )
    voting_results: Dict[str, bool] = Field(
        default_factory=dict,
        description="How each agent voted"
    )
    modifications: List[str] = Field(
        default_factory=list,
        description="Modifications made to original plan"
    )
    dissenting_opinions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Concerns raised by agents"
    )


class AuraTask(BaseModel):
    """Defines a single, traceable cognitive task for the AURA system"""
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    objective: str = Field(..., description="What the task aims to accomplish")
    environment: Dict[str, Any] = Field(..., description="Environmental context")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    perception_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output from perception phase"
    )
    memory_context: Optional[MemoryContext] = Field(
        default=None,
        description="Context from memory system"
    )
    final_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final task result"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AuraWorkflowState(BaseModel):
    """The complete state that flows through the LangGraph workflow"""
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    # Core task information
    task: AuraTask = Field(..., description="The task being executed")
    
    # Workflow state
    plan: Optional[ExecutionPlan] = Field(
        default=None,
        description="The execution plan"
    )
    consensus: Optional[ConsensusDecision] = Field(
        default=None,
        description="Consensus decision on the plan"
    )
    observations: List[ObservationResult] = Field(
        default_factory=list,
        description="Collected observations"
    )
    patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified patterns from analysis"
    )
    
    # Execution tracking
    execution_trace: List[str] = Field(
        default_factory=list,
        description="Log of execution steps"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    
    # This field allows nodes to access the executor and its components
    # We exclude it from serialization to avoid circular references
    executor_instance: Optional[Any] = Field(
        default=None,
        exclude=True,
        description="Reference to the UnifiedWorkflowExecutor"
    )
    
    def add_trace(self, message: str) -> None:
        """Add a message to the execution trace"""
        timestamp = datetime.utcnow().isoformat()
        self.execution_trace.append(f"[{timestamp}] {message}")
    
    def set_metric(self, key: str, value: float) -> None:
        """Set a performance metric"""
        self.metrics[key] = value


# --- Agent Communication Models ---
class AgentMessage(BaseModel):
    """Message passed between agents"""
    model_config = ConfigDict(frozen=False)
    
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    from_agent: str = Field(..., description="Sending agent ID")
    to_agent: str = Field(..., description="Receiving agent ID")
    content: Dict[str, Any] = Field(..., description="Message content")
    message_type: str = Field(..., description="Type of message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentDecision(BaseModel):
    """A decision made by an agent"""
    model_config = ConfigDict(frozen=False)
    
    agent_id: str = Field(..., description="Agent making the decision")
    decision_type: str = Field(..., description="Type of decision")
    rationale: str = Field(..., description="Reasoning behind decision")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in decision"
    )
    alternatives_considered: List[str] = Field(
        default_factory=list,
        description="Other options considered"
    )