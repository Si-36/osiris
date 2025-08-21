"""
Request schemas for Liquid Neural Network API
Pydantic V2 models with comprehensive validation
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class LiquidMode(str, Enum):
    """LNN operating modes"""
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    EDGE = "edge"
    DISTRIBUTED = "distributed"


class ODESolverType(str, Enum):
    """ODE solver types"""
    EULER = "euler"
    RK4 = "rk4"
    DOPRI5 = "dopri5"
    SEMI_IMPLICIT = "semi_implicit"
    ADJOINT = "adjoint"


class InferenceRequest(BaseModel):
    """Request for LNN inference"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "adaptive",
            "input_data": [0.1, 0.2, 0.3] * 42,  # 126 dimensions
            "session_id": "user_123",
            "return_dynamics": True,
            "batch_mode": False
        }
    })
    
    model_id: str = Field(
        default="standard",
        description="ID of the LNN model to use"
    )
    input_data: List[float] = Field(
        ...,
        description="Input data for inference",
        min_length=1,
        max_length=10000
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for stateful processing"
    )
    return_dynamics: bool = Field(
        default=False,
        description="Return internal dynamics information"
    )
    batch_mode: bool = Field(
        default=False,
        description="Process as batch (adds batch dimension)"
    )
    time_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of time steps for continuous dynamics"
    )


class StreamingInferenceRequest(BaseModel):
    """Request for streaming inference via WebSocket"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "inference",
            "input": [0.5] * 128,
            "stream_dynamics": True
        }
    })
    
    type: Literal["inference", "adapt"] = Field(..., description="Request type")
    input: List[float] = Field(..., description="Input data")
    stream_dynamics: bool = Field(default=False, description="Stream dynamics info")
    feedback: Optional[float] = Field(default=None, description="Feedback for adaptation")


class AdaptRequest(BaseModel):
    """Request to adapt model parameters"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "adaptive",
            "feedback_signal": 0.8,
            "adaptation_strength": 0.1,
            "target_metrics": {"accuracy": 0.95, "latency": 50}
        }
    })
    
    model_id: str = Field(..., description="Model to adapt")
    feedback_signal: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Feedback signal (-1 to 1)"
    )
    adaptation_strength: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Strength of adaptation"
    )
    target_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target performance metrics"
    )


class TrainRequest(BaseModel):
    """Request for continuous training"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "standard",
            "training_data": [[0.1, 0.2] * 64, [0.3, 0.4] * 64],
            "training_labels": [[0, 1], [1, 0]],
            "learning_rate": 0.001,
            "epochs": 10
        }
    })
    
    model_id: str = Field(..., description="Model to train")
    training_data: List[List[float]] = Field(
        ...,
        description="Training data samples",
        min_length=1,
        max_length=10000
    )
    training_labels: Optional[List[List[float]]] = Field(
        default=None,
        description="Training labels (for supervised learning)"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Learning rate"
    )
    epochs: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    validation_split: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Validation data split"
    )


class ModelConfigRequest(BaseModel):
    """Request to create a new LNN model"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "custom_lnn",
            "mode": "adaptive",
            "input_size": 128,
            "hidden_size": 256,
            "output_size": 64,
            "num_layers": 3,
            "enable_growth": True,
            "ode_solver": "dopri5"
        }
    })
    
    model_id: Optional[str] = Field(
        default=None,
        description="Custom model ID",
        pattern="^[a-zA-Z0-9_-]+$"
    )
    mode: LiquidMode = Field(
        default=LiquidMode.STANDARD,
        description="Operating mode"
    )
    input_size: int = Field(
        default=128,
        ge=1,
        le=10000,
        description="Input dimension"
    )
    hidden_size: int = Field(
        default=256,
        ge=16,
        le=2048,
        description="Hidden layer size"
    )
    output_size: int = Field(
        default=64,
        ge=1,
        le=1000,
        description="Output dimension"
    )
    num_layers: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of liquid layers"
    )
    enable_growth: bool = Field(
        default=False,
        description="Enable dynamic neuron growth"
    )
    max_neurons: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum neurons (for adaptive mode)"
    )
    ode_solver: Optional[ODESolverType] = Field(
        default=None,
        description="ODE solver type"
    )
    time_constant: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Time constant for liquid dynamics"
    )
    sparsity: float = Field(
        default=0.8,
        ge=0.0,
        le=0.99,
        description="Connection sparsity"
    )
    
    @field_validator('max_neurons')
    @classmethod
    def validate_max_neurons(cls, v, info):
        """Ensure max_neurons >= hidden_size"""
        if 'hidden_size' in info.data and v < info.data['hidden_size']:
            raise ValueError(f"max_neurons ({v}) must be >= hidden_size ({info.data['hidden_size']})")
        return v


class ConsensusInferenceRequest(BaseModel):
    """Request for consensus-based inference"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_ids": ["standard", "adaptive", "edge"],
            "input_data": [0.5] * 128,
            "consensus_method": "weighted_average",
            "require_quorum": True
        }
    })
    
    model_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Models to use for consensus"
    )
    input_data: List[float] = Field(
        ...,
        description="Input data for all models"
    )
    consensus_method: Literal["average", "weighted_average", "byzantine"] = Field(
        default="weighted_average",
        description="Consensus method to use"
    )
    require_quorum: bool = Field(
        default=True,
        description="Require minimum number of models"
    )
    min_agreement: float = Field(
        default=0.67,
        ge=0.5,
        le=1.0,
        description="Minimum agreement threshold"
    )