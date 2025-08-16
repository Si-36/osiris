"""
LNN Council Data Models - Essential Models Only

WHY: Our core_agent.py imports GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState
but they don't exist, breaking the system. This fixes that with minimal, focused models.
"""


# Fallback imports for missing dependencies
try:
    from opentelemetry.exporter import jaeger
    from opentelemetry import trace
except ImportError:
    print("Warning: OpenTelemetry not available, using fallback")
    # Create mock objects
    class MockExporter:
        def __init__(self, *args, **kwargs): pass
        def export(self, *args, **kwargs): return True
    
    class MockTrace:
        def get_tracer(self, *args, **kwargs): 
            return type('tracer', (), {
                'start_span': lambda *a, **k: type('span', (), {
                    '__enter__': lambda s: s, 
                    '__exit__': lambda *a: None
                })()
            })()
    
    jaeger = type('jaeger', (), {'JaegerExporter': MockExporter})
    trace = MockTrace()

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid

from pydantic import BaseModel, Field, field_validator
from ..base import AgentState


class GPUAllocationRequest(BaseModel):
    """GPU allocation request - what users ask for."""
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., min_length=1)
    project_id: str = Field(..., min_length=1)
    
    # What they want
    gpu_type: str = Field(..., pattern=r'^(A100|H100|V100|RTX4090|RTX3090)$')
    gpu_count: int = Field(..., ge=1, le=8)
    memory_gb: int = Field(..., ge=1, le=80)
    compute_hours: float = Field(..., ge=0.1, le=168.0)
    
    # Metadata
    priority: int = Field(default=5, ge=1, le=10)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('gpu_type')
    @classmethod
    def validate_gpu_type(cls, v: str) -> str:
        """Validate GPU type."""
        valid_types = {'A100', 'H100', 'V100', 'RTX4090', 'RTX3090'}
        if v not in valid_types:
            raise ValueError(f'Invalid GPU type: {v}')
        return v


class GPUAllocationDecision(BaseModel):
    """GPU allocation decision - what the system decides."""
    
    request_id: str
    decision: str = Field(..., pattern=r'^(approve|deny|defer)$')
    
    # Decision metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_path: List[str] = Field(default_factory=list)
    fallback_used: bool = Field(default=False)
    inference_time_ms: float = Field(..., ge=0.0)
    
    # Allocation details (if approved)
    allocated_resources: Optional[Dict[str, Any]] = None
    estimated_cost: Optional[float] = None
    
    # Timestamps
    decision_made_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_reasoning(self, step: str, explanation: str) -> None:
        """Add reasoning step."""
        self.reasoning_path.append(f"{step}: {explanation}")


class DecisionContext(BaseModel):
    """Context for making decisions."""
    
    # System state
    current_utilization: Dict[str, float] = Field(default_factory=dict)
    available_resources: Dict[str, int] = Field(default_factory=dict)
    queue_depth: int = Field(default=0, ge=0)
    
    # Historical data
    similar_requests: List[Dict[str, Any]] = Field(default_factory=list)
    user_history: Dict[str, Any] = Field(default_factory=dict)


class HistoricalDecision(BaseModel):
    """Past decision for learning."""
    
    decision_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    decision_made: str
    outcome_success: bool
    lessons_learned: List[str] = Field(default_factory=list)
    decision_timestamp: datetime


class LNNCouncilState(AgentState):
    """State for LNN Council Agent - extends base AgentState."""
    
    # Request context
    current_request: Optional[GPUAllocationRequest] = None
    context_cache: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision tracking
    confidence_score: float = 0.0
    fallback_triggered: bool = False
    
    # Performance tracking
    inference_start_time: Optional[float] = None
    neural_inference_time: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True