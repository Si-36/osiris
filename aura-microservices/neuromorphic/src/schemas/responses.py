"""
Response schemas for Neuromorphic API
Comprehensive responses with energy and performance metrics
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime


class SpikeProcessResponse(BaseModel):
    """Response from spike processing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "output": [[0.8, 0.2, 0.1, 0.9]],
            "spike_count": 47,
            "energy_consumed_pj": 47.5,
            "latency_us": 523.7,
            "spike_rate": 0.047,
            "energy_per_spike_pj": 1.01
        }
    })
    
    output: List[List[float]] = Field(
        ...,
        description="Processed output from neuromorphic model"
    )
    spike_count: int = Field(
        ...,
        description="Total number of spikes generated"
    )
    energy_consumed_pj: float = Field(
        ...,
        description="Total energy consumed in picojoules"
    )
    latency_us: float = Field(
        ...,
        description="Processing latency in microseconds"
    )
    spike_rate: float = Field(
        default=0.0,
        description="Average spike rate (spikes per neuron per timestep)"
    )
    membrane_potential: Optional[float] = Field(
        default=None,
        description="Average membrane potential (for LIF models)"
    )
    adaptive_threshold: Optional[float] = Field(
        default=None,
        description="Current adaptive threshold (for homeostasis)"
    )
    sparsity: Optional[float] = Field(
        default=None,
        description="Sparsity of activity (1 - spike_rate)"
    )
    energy_per_spike_pj: float = Field(
        ...,
        description="Energy per spike in picojoules"
    )


class EnergyReport(BaseModel):
    """Comprehensive energy efficiency report"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_energy_consumed_pj": 1234567.8,
            "total_energy_consumed_joules": 1.2345678e-6,
            "total_operations": 10000,
            "avg_energy_per_operation_pj": 123.45,
            "current_power_draw_watts": 0.0025,
            "energy_efficiency_vs_ann": 987.6,
            "projected_battery_life_hours": 4800,
            "carbon_footprint_grams": 0.001
        }
    })
    
    total_energy_consumed_pj: float = Field(
        ...,
        description="Total energy consumed in picojoules since startup"
    )
    total_energy_consumed_joules: float = Field(
        ...,
        description="Total energy consumed in joules (for readability)"
    )
    total_operations: int = Field(
        ...,
        description="Total number of inference operations"
    )
    avg_energy_per_operation_pj: float = Field(
        ...,
        description="Average energy per operation in picojoules"
    )
    current_power_draw_watts: float = Field(
        ...,
        description="Current power draw in watts"
    )
    energy_efficiency_vs_ann: float = Field(
        ...,
        description="Energy efficiency improvement vs traditional ANN (e.g., 1000x)"
    )
    projected_battery_life_hours: float = Field(
        ...,
        description="Projected battery life in hours (assuming 10Wh battery)"
    )
    carbon_footprint_grams: float = Field(
        ...,
        description="Estimated CO2 emissions in grams"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Energy breakdown by component"
    )


class SystemStatus(BaseModel):
    """System health and status"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "uptime_seconds": 3600.5,
            "total_operations": 50000,
            "total_energy_consumed_pj": 5000000,
            "models_loaded": ["lif", "lsm", "gnn"]
        }
    })
    
    status: str = Field(
        ...,
        description="System status: healthy, degraded, or unhealthy"
    )
    uptime_seconds: float = Field(
        ...,
        description="System uptime in seconds"
    )
    total_operations: int = Field(
        ...,
        description="Total operations processed"
    )
    total_energy_consumed_pj: float = Field(
        ...,
        description="Total energy consumed in picojoules"
    )
    avg_energy_per_operation_pj: float = Field(
        ...,
        description="Average energy per operation"
    )
    current_power_draw_watts: float = Field(
        ...,
        description="Current power consumption"
    )
    models_loaded: List[str] = Field(
        ...,
        description="List of loaded neuromorphic models"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current configuration"
    )


class BenchmarkResults(BaseModel):
    """Results from benchmark runs"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_type": "lif",
            "avg_latency_us": 487.3,
            "p50_latency_us": 465.2,
            "p95_latency_us": 612.8,
            "p99_latency_us": 891.3,
            "throughput_ops_per_sec": 2053.4,
            "total_energy_pj": 4873000,
            "energy_per_op_pj": 487.3,
            "energy_efficiency_ratio": 1024.7,
            "accuracy": 0.968
        }
    })
    
    model_type: str = Field(..., description="Model type benchmarked")
    avg_latency_us: float = Field(..., description="Average latency in microseconds")
    p50_latency_us: float = Field(..., description="50th percentile latency")
    p95_latency_us: float = Field(..., description="95th percentile latency")
    p99_latency_us: float = Field(..., description="99th percentile latency")
    throughput_ops_per_sec: float = Field(..., description="Operations per second")
    total_energy_pj: float = Field(..., description="Total energy consumed")
    energy_per_op_pj: float = Field(..., description="Energy per operation")
    energy_efficiency_ratio: float = Field(..., description="Efficiency vs traditional NN")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy if tested")
    iterations: int = Field(default=0, description="Number of iterations run")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    hardware_info: Dict[str, Any] = Field(default_factory=dict, description="Hardware details")


class HardwareOptimizationResponse(BaseModel):
    """Response for hardware optimization queries"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "original_model_size_mb": 45.3,
            "optimized_model_size_mb": 5.7,
            "compression_ratio": 7.95,
            "estimated_speedup": 12.3,
            "estimated_energy_reduction": 156.7,
            "optimization_techniques": ["quantization", "pruning", "threshold_balancing"]
        }
    })
    
    original_model_size_mb: float = Field(..., description="Original model size")
    optimized_model_size_mb: float = Field(..., description="Optimized model size")
    compression_ratio: float = Field(..., description="Size reduction ratio")
    estimated_speedup: float = Field(..., description="Expected performance improvement")
    estimated_energy_reduction: float = Field(..., description="Expected energy reduction factor")
    optimization_techniques: List[str] = Field(..., description="Applied optimizations")
    hardware_specific_notes: Optional[str] = Field(default=None, description="Hardware-specific information")


class EventStreamResponse(BaseModel):
    """Response for event-driven processing streams"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "event_id": "evt_123456",
            "timestamp_us": 1234567890123,
            "spike_events": [
                {"neuron_id": 42, "time_us": 100},
                {"neuron_id": 87, "time_us": 105}
            ],
            "energy_accumulated_pj": 42.7
        }
    })
    
    event_id: str = Field(..., description="Unique event identifier")
    timestamp_us: int = Field(..., description="Event timestamp in microseconds")
    spike_events: List[Dict[str, Any]] = Field(..., description="Individual spike events")
    energy_accumulated_pj: float = Field(..., description="Energy consumed since last event")
    processing_latency_us: Optional[float] = Field(default=None, description="Event processing latency")