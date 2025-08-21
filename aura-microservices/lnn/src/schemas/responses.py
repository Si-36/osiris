"""
Response schemas for Liquid Neural Network API
Comprehensive response models with metrics and dynamics
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class InferenceResponse(BaseModel):
    """Response from LNN inference"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "output": [0.1, 0.2, 0.3],
            "model_id": "adaptive",
            "session_id": "user_123",
            "latency_ms": 12.5,
            "inference_count": 42,
            "adaptations": {"layer_0": {"grew_neurons": 5}},
            "dynamics": {
                "mean_activation": 0.45,
                "std_activation": 0.12,
                "sparsity": 0.78,
                "max_activation": 0.98
            }
        }
    })
    
    output: List[float] = Field(..., description="Model output")
    model_id: str = Field(..., description="Model that produced output")
    session_id: str = Field(..., description="Session ID for stateful processing")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    inference_count: int = Field(..., description="Total inferences for this model")
    adaptations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Adaptations that occurred during inference"
    )
    dynamics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Internal dynamics information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Inference timestamp"
    )


class DynamicsResponse(BaseModel):
    """Detailed dynamics information"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "mean_activation": 0.45,
            "std_activation": 0.12,
            "sparsity": 0.78,
            "max_activation": 0.98,
            "energy_estimate": 0.023,
            "trajectory_length": 1.56,
            "lyapunov_estimate": -0.34
        }
    })
    
    mean_activation: float = Field(..., description="Mean activation level")
    std_activation: float = Field(..., description="Activation standard deviation")
    sparsity: float = Field(..., description="Fraction of near-zero activations")
    max_activation: float = Field(..., description="Maximum activation magnitude")
    energy_estimate: Optional[float] = Field(None, description="Energy consumption estimate")
    trajectory_length: Optional[float] = Field(None, description="State space trajectory length")
    lyapunov_estimate: Optional[float] = Field(None, description="Lyapunov exponent estimate")


class AdaptationResponse(BaseModel):
    """Response from model adaptation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "adaptive",
            "adaptation_type": "parameter_update",
            "parameters_changed": ["tau", "sigma"],
            "old_performance": {"accuracy": 0.85, "latency": 15.2},
            "new_performance": {"accuracy": 0.89, "latency": 14.8},
            "success": True,
            "message": "Model adapted successfully"
        }
    })
    
    model_id: str = Field(..., description="Model that was adapted")
    adaptation_type: str = Field(..., description="Type of adaptation performed")
    parameters_changed: List[str] = Field(..., description="Parameters that were modified")
    old_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance before adaptation"
    )
    new_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance after adaptation"
    )
    success: bool = Field(..., description="Whether adaptation succeeded")
    message: Optional[str] = Field(None, description="Additional information")


class TrainingResponse(BaseModel):
    """Response from continuous training"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "standard",
            "samples_processed": 1000,
            "loss_before": 0.532,
            "loss_after": 0.234,
            "training_time_ms": 523.4,
            "adaptations_triggered": 3,
            "validation_metrics": {"accuracy": 0.92, "f1_score": 0.89}
        }
    })
    
    model_id: str = Field(..., description="Model that was trained")
    samples_processed: int = Field(..., description="Number of samples processed")
    loss_before: float = Field(..., description="Loss before training")
    loss_after: float = Field(..., description="Loss after training")
    training_time_ms: float = Field(..., description="Training time in milliseconds")
    adaptations_triggered: int = Field(default=0, description="Adaptations during training")
    validation_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Validation metrics"
    )


class ModelStatusResponse(BaseModel):
    """Health and status response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "total_models": 3,
            "model_stats": {
                "standard": {"parameters": 98304, "inference_count": 1523, "adaptations": 0},
                "adaptive": {"parameters": 131072, "inference_count": 892, "adaptations": 12},
                "edge": {"parameters": 32768, "inference_count": 3421, "adaptations": 0}
            },
            "total_inferences": 5836,
            "total_adaptations": 12,
            "uptime_seconds": 3600.5
        }
    })
    
    status: str = Field(..., description="Service health status")
    total_models: int = Field(..., description="Number of loaded models")
    model_stats: Dict[str, Dict[str, int]] = Field(..., description="Statistics per model")
    total_inferences: int = Field(..., description="Total inferences across all models")
    total_adaptations: int = Field(..., description="Total adaptations")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    gpu_available: Optional[bool] = Field(None, description="GPU availability")


class ModelInfoResponse(BaseModel):
    """Detailed model information"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_id": "adaptive",
            "implementation": "adaptive",
            "config": {
                "mode": "adaptive",
                "hidden_size": 256,
                "num_layers": 3,
                "ode_solver": "dopri5"
            },
            "parameters": 131072,
            "inference_count": 892,
            "total_adaptations": 12,
            "current_neurons": 287,
            "mode": "adaptive",
            "features": {
                "continuous_time": True,
                "adaptive": True,
                "edge_optimized": False,
                "distributed": False,
                "ode_solver": "dopri5"
            }
        }
    })
    
    model_id: str = Field(..., description="Model identifier")
    implementation: str = Field(..., description="Implementation type")
    config: Dict[str, Any] = Field(..., description="Model configuration")
    parameters: int = Field(..., description="Total parameters")
    inference_count: int = Field(..., description="Total inferences")
    total_adaptations: int = Field(..., description="Total adaptations")
    current_neurons: int = Field(..., description="Current active neurons")
    mode: str = Field(..., description="Operating mode")
    features: Dict[str, Union[bool, str]] = Field(..., description="Model features")
    creation_time: Optional[datetime] = Field(None, description="Model creation time")
    last_inference_time: Optional[datetime] = Field(None, description="Last inference time")


class ConsensusResponse(BaseModel):
    """Response from consensus inference"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "consensus_output": [0.15, 0.25, 0.35],
            "individual_predictions": {
                "standard": {"output": [0.1, 0.2, 0.3], "confidence": 0.9},
                "adaptive": {"output": [0.2, 0.3, 0.4], "confidence": 0.95},
                "edge": {"output": [0.15, 0.25, 0.35], "confidence": 0.85}
            },
            "participants": 3,
            "consensus_method": "weighted_average",
            "agreement_score": 0.92
        }
    })
    
    consensus_output: List[float] = Field(..., description="Consensus prediction")
    individual_predictions: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Individual model predictions"
    )
    participants: int = Field(..., description="Number of participating models")
    consensus_method: str = Field(..., description="Method used for consensus")
    agreement_score: Optional[float] = Field(
        None,
        description="Agreement score between models"
    )


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "result",
            "output": [0.1, 0.2, 0.3],
            "latency_ms": 8.5,
            "adaptations": {},
            "timestamp": "2025-08-20T12:34:56Z"
        }
    })
    
    type: str = Field(..., description="Message type")
    output: Optional[List[float]] = Field(None, description="Model output")
    latency_ms: Optional[float] = Field(None, description="Processing latency")
    adaptations: Optional[Dict[str, Any]] = Field(None, description="Adaptation info")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AdaptationEvent(BaseModel):
    """Real-time adaptation event"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "adaptation",
            "model_id": "adaptive",
            "data": {
                "layer": 0,
                "action": "grow",
                "neurons_added": 5,
                "current_total": 261,
                "trigger": "high_stress"
            }
        }
    })
    
    type: str = Field(default="adaptation", description="Event type")
    model_id: str = Field(..., description="Model that adapted")
    data: Dict[str, Any] = Field(..., description="Adaptation details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)