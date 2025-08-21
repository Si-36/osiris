"""
Request schemas for MoE Router API
Pydantic V2 models for routing requests
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class RoutingStrategy(str, Enum):
    """Available routing strategies"""
    SWITCH_TRANSFORMER = "switch_transformer"
    TOP_K = "top_k"
    SEMANTIC = "semantic"
    TDA_AWARE = "tda_aware"
    CONSISTENT_HASH = "consistent_hash"
    POWER_OF_TWO = "power_of_two"
    ADAPTIVE = "adaptive"


class ServiceType(str, Enum):
    """Service types"""
    NEUROMORPHIC = "neuromorphic"
    MEMORY = "memory"
    BYZANTINE = "byzantine"
    LNN = "lnn"
    TDA = "tda"
    CUSTOM = "custom"


class RouteRequest(BaseModel):
    """Request for routing to services"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data": {
                "type": "inference",
                "data": [0.1, 0.2, 0.3],
                "priority": 0.8,
                "complexity": 0.5
            },
            "routing_strategy": "adaptive",
            "proxy_request": True,
            "endpoint": "process"
        }
    })
    
    data: Dict[str, Any] = Field(
        ...,
        description="Request data to route"
    )
    routing_strategy: Optional[RoutingStrategy] = Field(
        default=None,
        description="Force specific routing strategy"
    )
    proxy_request: bool = Field(
        default=False,
        description="Proxy request to selected service"
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="Target endpoint when proxying"
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout"
    )


class BatchRouteRequest(BaseModel):
    """Batch routing request"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "requests": [
                {"type": "inference", "data": [0.1, 0.2]},
                {"type": "storage", "data": {"key": "value"}},
                {"type": "consensus", "data": {"votes": [1, 0, 1]}}
            ],
            "routing_strategy": "top_k"
        }
    })
    
    requests: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch of requests to route"
    )
    routing_strategy: Optional[RoutingStrategy] = Field(
        default=None,
        description="Apply same strategy to all"
    )


class ServiceRegistrationRequest(BaseModel):
    """Register a new service"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "service_id": "custom_processor",
            "service_type": "custom",
            "endpoint": "http://localhost:8006",
            "capabilities": ["processing", "analysis"],
            "specializations": ["nlp", "vision"],
            "max_capacity": 100
        }
    })
    
    service_id: str = Field(
        ...,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Unique service identifier"
    )
    service_type: ServiceType = Field(
        ...,
        description="Type of service"
    )
    endpoint: str = Field(
        ...,
        description="Service endpoint URL"
    )
    capabilities: List[str] = Field(
        ...,
        min_length=1,
        description="Service capabilities"
    )
    specializations: List[str] = Field(
        default_factory=list,
        description="Service specializations"
    )
    max_capacity: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum concurrent requests"
    )
    performance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Initial performance score"
    )
    
    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        """Ensure endpoint is valid URL"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Endpoint must be a valid HTTP(S) URL')
        return v


class StrategyOverrideRequest(BaseModel):
    """Override routing strategy"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "strategy": "semantic",
            "duration_seconds": 300,
            "apply_to_services": ["neuromorphic", "lnn"]
        }
    })
    
    strategy: RoutingStrategy = Field(
        ...,
        description="Strategy to use"
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        le=3600,
        description="Override duration"
    )
    apply_to_services: Optional[List[str]] = Field(
        default=None,
        description="Specific services to apply to"
    )


class LoadBalanceRequest(BaseModel):
    """Request for load rebalancing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "trigger": "manual",
            "target_distribution": {
                "neuromorphic": 0.3,
                "memory": 0.2,
                "byzantine": 0.2,
                "lnn": 0.3
            }
        }
    })
    
    trigger: Literal["manual", "automatic"] = Field(
        default="manual",
        description="Rebalancing trigger"
    )
    target_distribution: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target load distribution"
    )
    max_migrations: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum requests to migrate"
    )
    
    @field_validator('target_distribution')
    @classmethod
    def validate_distribution(cls, v):
        """Ensure distribution sums to ~1.0"""
        if v:
            total = sum(v.values())
            if not 0.95 <= total <= 1.05:
                raise ValueError(f'Distribution must sum to ~1.0, got {total}')
        return v


class CircuitBreakerRequest(BaseModel):
    """Circuit breaker management"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "service_id": "neuromorphic",
            "action": "reset",
            "failure_threshold": 5
        }
    })
    
    service_id: str = Field(
        ...,
        description="Service to manage"
    )
    action: Literal["reset", "open", "close"] = Field(
        ...,
        description="Circuit breaker action"
    )
    failure_threshold: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="New failure threshold"
    )


class ServiceHealthReport(BaseModel):
    """Health report from service"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "service_id": "memory",
            "latency_ms": 12.5,
            "success": True,
            "current_load": 45,
            "error_rate": 0.02
        }
    })
    
    service_id: str = Field(..., description="Reporting service")
    latency_ms: float = Field(..., ge=0, description="Request latency")
    success: bool = Field(..., description="Request success")
    current_load: Optional[int] = Field(default=None, ge=0, description="Current load")
    error_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Error rate")
    timestamp: Optional[float] = Field(default=None, description="Report timestamp")