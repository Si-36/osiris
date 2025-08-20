"""
Request schemas for Neuromorphic API
Pydantic V2 models with comprehensive validation
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Literal, Union
from enum import Enum


class ModelType(str, Enum):
    """Available neuromorphic models"""
    LIF = "lif"
    LSM = "lsm"
    GNN = "gnn"


class HardwareTarget(str, Enum):
    """Supported neuromorphic hardware"""
    LOIHI2 = "loihi2"
    BRAINSCALES2 = "brainscales2"
    TRUENORTH = "truenorth"
    SPINNAKER2 = "spinnaker2"
    CPU = "cpu"
    GPU = "gpu"


class SpikeProcessRequest(BaseModel):
    """Request for spike train processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "spike_data": [[0, 1, 0, 1, 0] * 25],  # 125 dimensional sparse spike train
            "time_steps": 10,
            "reward_signal": 0.5
        }
    })
    
    spike_data: List[List[float]] = Field(
        ...,
        description="Spike train data [batch_size, input_dim]",
        min_length=1,
        max_length=1000
    )
    time_steps: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of time steps to simulate"
    )
    reward_signal: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Reward signal for neuromodulation"
    )
    
    @field_validator('spike_data')
    @classmethod
    def validate_spike_data(cls, v):
        """Ensure spike data is binary or in [0, 1]"""
        for batch in v:
            for spike in batch:
                if not 0 <= spike <= 1:
                    raise ValueError("Spike values must be in range [0, 1]")
        return v


class LSMProcessRequest(BaseModel):
    """Request for Liquid State Machine processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "input_data": [[0.5, -0.3, 0.8] * 20],  # 60 dimensional input
            "time_steps": 20,
            "reservoir_size": 500
        }
    })
    
    input_data: List[List[float]] = Field(
        ...,
        description="Input data for LSM [batch_size, input_dim]",
        min_length=1,
        max_length=100
    )
    time_steps: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Number of time steps for temporal processing"
    )
    reservoir_size: Optional[int] = Field(
        default=500,
        ge=100,
        le=5000,
        description="Size of the liquid reservoir"
    )


class GNNProcessRequest(BaseModel):
    """Request for Spiking Graph Neural Network processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "node_features": [[0.1, 0.2, 0.3] * 10],  # 30 dimensional features for each node
            "edge_index": [[0, 1, 2], [1, 2, 0]],  # Source and target nodes
            "edge_attr": [[0.5], [0.3], [0.8]],  # Edge attributes
            "time_steps": 10
        }
    })
    
    node_features: List[List[float]] = Field(
        ...,
        description="Node feature matrix [batch_size, num_nodes, features]",
        min_length=1
    )
    edge_index: List[List[int]] = Field(
        ...,
        description="Edge connectivity [2, num_edges]",
        min_length=2,
        max_length=2
    )
    edge_attr: Optional[List[List[float]]] = Field(
        default=None,
        description="Edge attributes [num_edges, edge_features]"
    )
    time_steps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of time steps for graph propagation"
    )
    
    @field_validator('edge_index')
    @classmethod
    def validate_edge_index(cls, v):
        """Ensure edge_index has exactly 2 rows"""
        if len(v) != 2:
            raise ValueError("edge_index must have exactly 2 rows (source and target)")
        if len(v[0]) != len(v[1]):
            raise ValueError("Source and target arrays must have same length")
        return v


class BenchmarkRequest(BaseModel):
    """Request for running benchmarks"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_type": "lif",
            "input_size": 128,
            "batch_size": 32,
            "time_steps": 100,
            "iterations": 1000
        }
    })
    
    model_type: ModelType = Field(
        default=ModelType.LIF,
        description="Which model to benchmark"
    )
    input_size: int = Field(
        default=128,
        ge=10,
        le=10000,
        description="Input dimension size"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size for throughput testing"
    )
    time_steps: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Time steps per iteration"
    )
    iterations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of benchmark iterations"
    )


class ConversionRequest(BaseModel):
    """Request for ANN to SNN conversion"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_path": "/models/trained_ann.pt",
            "calibration_samples": 1000,
            "target_hardware": "loihi2",
            "accuracy_threshold": 0.95
        }
    })
    
    model_path: str = Field(
        ...,
        description="Path to trained ANN model"
    )
    calibration_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of samples for threshold calibration"
    )
    target_hardware: HardwareTarget = Field(
        default=HardwareTarget.CPU,
        description="Target hardware for optimization"
    )
    accuracy_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy to maintain after conversion"
    )
    quantization_bits: Optional[int] = Field(
        default=8,
        ge=1,
        le=32,
        description="Bit width for weight quantization"
    )


class EnergyQueryRequest(BaseModel):
    """Request for energy consumption queries"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "time_range": "last_hour",
            "aggregation": "mean"
        }
    })
    
    time_range: Literal["last_minute", "last_hour", "last_day", "all_time"] = Field(
        default="last_hour",
        description="Time range for energy query"
    )
    aggregation: Literal["sum", "mean", "max", "min"] = Field(
        default="mean",
        description="How to aggregate energy data"
    )
    breakdown_by: Optional[Literal["model", "operation", "component"]] = Field(
        default=None,
        description="Break down energy by category"
    )