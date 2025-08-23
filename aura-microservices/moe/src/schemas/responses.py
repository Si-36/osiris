"""
Response schemas for MoE Router API
Comprehensive routing responses with metrics
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class RouteResponse(BaseModel):
    """Response from routing decision"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "selected_services": ["neuromorphic", "lnn"],
            "routing_strategy": "adaptive",
            "confidence_scores": [0.85, 0.72],
            "reasoning": "Strategy: adaptive | Selected: neuromorphic, lnn | Confidence: 78.5%",
            "proxy_response": {"result": "processed"},
            "latency_ms": 8.5,
            "success": True
        }
    })
    
    selected_services: List[str] = Field(
        ...,
        description="Services selected for routing"
    )
    routing_strategy: str = Field(
        ...,
        description="Strategy used for routing"
    )
    confidence_scores: List[float] = Field(
        ...,
        description="Confidence in each selection"
    )
    reasoning: str = Field(
        ...,
        description="Human-readable routing reasoning"
    )
    proxy_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response from proxied service"
    )
    latency_ms: float = Field(
        ...,
        description="Routing decision latency"
    )
    success: bool = Field(
        ...,
        description="Whether routing succeeded"
    )
    fallback_services: Optional[List[str]] = Field(
        default=None,
        description="Fallback services if primary fail"
    )
    estimated_total_latency_ms: Optional[float] = Field(
        default=None,
        description="Estimated end-to-end latency"
    )
    load_balance_score: Optional[float] = Field(
        default=None,
        description="Load balance coefficient (0-1)"
    )


class BatchRouteResponse(BaseModel):
    """Response for batch routing"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_requests": 3,
            "successful_routes": 3,
            "results": [
                {
                    "index": 0,
                    "success": True,
                    "selected_services": ["neuromorphic"],
                    "routing_strategy": "switch_transformer",
                    "confidence_scores": [0.92]
                }
            ]
        }
    })
    
    total_requests: int = Field(..., description="Total requests in batch")
    successful_routes: int = Field(..., description="Successfully routed requests")
    results: List[Dict[str, Any]] = Field(..., description="Individual routing results")
    batch_latency_ms: Optional[float] = Field(None, description="Total batch processing time")


class ServiceStatusResponse(BaseModel):
    """Service health status"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "available_services": ["neuromorphic", "memory", "lnn"],
            "total_services": 4,
            "circuit_breaker_states": {
                "neuromorphic": "closed",
                "memory": "closed",
                "byzantine": "open",
                "lnn": "closed"
            },
            "health_score": 0.75,
            "uptime_seconds": 3600.5
        }
    })
    
    status: str = Field(..., description="Overall health status")
    available_services: List[str] = Field(..., description="Currently available services")
    total_services: int = Field(..., description="Total registered services")
    circuit_breaker_states: Dict[str, str] = Field(..., description="Circuit breaker states")
    health_score: float = Field(..., description="Overall health score (0-1)")
    uptime_seconds: float = Field(..., description="Service uptime")


class RouterMetricsResponse(BaseModel):
    """Comprehensive router metrics"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_routes": 15234,
            "strategy_distribution": {
                "adaptive": 8234,
                "switch_transformer": 4521,
                "semantic": 2479
            },
            "service_utilization": {
                "neuromorphic": 4821,
                "memory": 3654,
                "byzantine": 2341,
                "lnn": 4418
            },
            "load_balance_coefficient": 0.12,
            "avg_routing_time_ms": 2.3,
            "circuit_breaker_states": {
                "neuromorphic": "closed",
                "memory": "closed",
                "byzantine": "open",
                "lnn": "closed"
            },
            "active_services": 3,
            "total_services": 4,
            "timestamp": 1692547200.0
        }
    })
    
    total_routes: int = Field(..., description="Total routing decisions made")
    strategy_distribution: Dict[str, int] = Field(..., description="Routes per strategy")
    service_utilization: Dict[str, int] = Field(..., description="Routes per service")
    load_balance_coefficient: float = Field(..., description="Load distribution variance")
    avg_routing_time_ms: float = Field(..., description="Average routing latency")
    circuit_breaker_states: Dict[str, str] = Field(..., description="Current breaker states")
    active_services: int = Field(..., description="Currently active services")
    total_services: int = Field(..., description="Total registered services")
    timestamp: float = Field(..., description="Metrics timestamp")
    error_rate: Optional[float] = Field(None, description="Overall error rate")
    p95_latency_ms: Optional[float] = Field(None, description="95th percentile latency")
    p99_latency_ms: Optional[float] = Field(None, description="99th percentile latency")


class LoadBalanceResponse(BaseModel):
    """Load balancing operation result"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "before_distribution": {
                "neuromorphic": 120,
                "memory": 45,
                "byzantine": 15,
                "lnn": 80
            },
            "after_distribution": {
                "neuromorphic": 65,
                "memory": 65,
                "byzantine": 65,
                "lnn": 65
            },
            "migrations_performed": 42,
            "success": True,
            "message": "Load rebalancing completed"
        }
    })
    
    before_distribution: Dict[str, float] = Field(..., description="Load before rebalancing")
    after_distribution: Dict[str, float] = Field(..., description="Load after rebalancing")
    migrations_performed: int = Field(..., description="Number of requests migrated")
    success: bool = Field(..., description="Whether rebalancing succeeded")
    message: str = Field(..., description="Operation message")
    duration_ms: Optional[float] = Field(None, description="Rebalancing duration")


class ServiceInfo(BaseModel):
    """Detailed service information"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "service_id": "neuromorphic",
            "service_type": "neuromorphic",
            "endpoint": "http://localhost:8000",
            "capabilities": ["spiking", "energy_efficient", "real_time"],
            "available": True,
            "circuit_breaker_state": "closed",
            "current_load": 45,
            "max_capacity": 1000,
            "performance_score": 0.92,
            "latency_p95_ms": 5.2,
            "error_rate": 0.001
        }
    })
    
    service_id: str = Field(..., description="Service identifier")
    service_type: str = Field(..., description="Type of service")
    endpoint: str = Field(..., description="Service endpoint")
    capabilities: List[str] = Field(..., description="Service capabilities")
    available: bool = Field(..., description="Service availability")
    circuit_breaker_state: str = Field(..., description="Circuit breaker state")
    current_load: float = Field(..., description="Current load level")
    max_capacity: int = Field(..., description="Maximum capacity")
    performance_score: float = Field(..., description="Performance score")
    latency_p95_ms: float = Field(..., description="95th percentile latency")
    error_rate: float = Field(..., description="Error rate")
    last_health_check: Optional[datetime] = Field(None, description="Last health check time")


class RoutingDemoResponse(BaseModel):
    """Demo routing response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "request_type": "inference",
            "complexity": "complex",
            "routing_results": [
                {
                    "strategy": "adaptive",
                    "selected_services": ["neuromorphic", "lnn"],
                    "confidence_scores": [0.85, 0.72],
                    "reasoning": "Complex inference benefits from dual processing",
                    "estimated_latency_ms": 15.3,
                    "load_balance_score": 0.82
                }
            ],
            "recommended_strategy": "adaptive",
            "explanation": "For complex inference requests, adaptive provides optimal routing with 15.3ms latency"
        }
    })
    
    request_type: str = Field(..., description="Type of request")
    complexity: str = Field(..., description="Request complexity level")
    routing_results: List[Dict[str, Any]] = Field(..., description="Results per strategy")
    recommended_strategy: str = Field(..., description="Best strategy for this scenario")
    explanation: str = Field(..., description="Detailed explanation")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "routing_update",
            "timestamp": 1692547200.0,
            "selected_services": ["neuromorphic", "lnn"],
            "strategy": "adaptive",
            "load_balance_score": 0.85
        }
    })
    
    type: str = Field(..., description="Message type")
    timestamp: float = Field(..., description="Message timestamp")
    selected_services: Optional[List[str]] = Field(None, description="Routed services")
    strategy: Optional[str] = Field(None, description="Routing strategy used")
    load_balance_score: Optional[float] = Field(None, description="Load balance metric")
    error: Optional[str] = Field(None, description="Error message if any")