"""
MoE Router Service API
Intelligent routing across AURA microservices using advanced MoE strategies
"""

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time
import httpx
import json
from typing import Optional, List, Dict, Any
import structlog
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from ..models.routing.moe_router_2025 import (
    MoERouterSystem, ServiceProfile, ServiceType, RoutingStrategy,
    RoutingDecision, create_default_service_profiles
)
from ..services.request_processor import RequestProcessor
from ..services.load_monitor import LoadMonitorService
from ..services.performance_tracker import PerformanceTracker
from ..schemas.requests import (
    RouteRequest, BatchRouteRequest, ServiceRegistrationRequest,
    StrategyOverrideRequest, LoadBalanceRequest
)
from ..schemas.responses import (
    RouteResponse, ServiceStatusResponse, RouterMetricsResponse,
    BatchRouteResponse, LoadBalanceResponse
)
from ..middleware.observability import ObservabilityMiddleware
from ..middleware.security import SecurityMiddleware
from ..middleware.circuit_breaker import CircuitBreakerMiddleware
from ..utils.observability import setup_telemetry

# Initialize structured logging
logger = structlog.get_logger()

# Setup telemetry
setup_telemetry()

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.service_subscriptions: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        for service_id, connections in self.service_subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
                
    async def broadcast_routing_update(self, routing_decision: RoutingDecision):
        """Broadcast routing decisions to connected clients"""
        message = {
            "type": "routing_update",
            "timestamp": time.time(),
            "selected_services": routing_decision.selected_services,
            "strategy": routing_decision.routing_strategy.value,
            "load_balance_score": routing_decision.load_balance_score
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    Handles initialization and cleanup of router system
    """
    logger.info("Starting MoE Router Service")
    
    # Initialize router with default services
    default_services = create_default_service_profiles()
    app.state.router = MoERouterSystem(default_services)
    
    # Initialize support services
    app.state.request_processor = RequestProcessor()
    app.state.load_monitor = LoadMonitorService(app.state.router)
    app.state.performance_tracker = PerformanceTracker()
    
    # HTTP client for proxying requests
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    
    # Start background tasks
    app.state.load_monitor_task = asyncio.create_task(
        app.state.load_monitor.monitor_services()
    )
    
    logger.info("MoE Router Service ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down MoE Router Service")
    app.state.load_monitor_task.cancel()
    await app.state.http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="AURA MoE Router Service",
    description="Intelligent routing using Mixture of Experts strategies",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CircuitBreakerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Selected-Services", "X-Routing-Strategy"]
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AURA MoE Router Service",
        "version": "2.0.0",
        "features": [
            "Google Switch Transformer routing",
            "Top-K multi-expert selection",
            "Semantic capability matching",
            "TDA-aware anomaly routing",
            "Consistent hashing distribution",
            "Circuit breaker fault isolation",
            "Real-time load balancing",
            "Adaptive strategy selection"
        ],
        "available_strategies": [s.value for s in RoutingStrategy],
        "status": "operational"
    }


@app.get("/api/v1/health", response_model=ServiceStatusResponse, tags=["health"])
async def health_check(request: Request):
    """
    Comprehensive health check with service status
    """
    router = request.app.state.router
    
    # Get available services
    available_services = router._get_available_services()
    
    # Get circuit breaker states
    breaker_states = {
        sid: breaker.state 
        for sid, breaker in router.circuit_breakers.items()
    }
    
    # Calculate health score
    health_score = len(available_services) / len(router.services)
    
    return ServiceStatusResponse(
        status="healthy" if health_score > 0.5 else "degraded",
        available_services=list(available_services),
        total_services=len(router.services),
        circuit_breaker_states=breaker_states,
        health_score=health_score,
        uptime_seconds=time.time() - request.app.state.get("start_time", time.time())
    )


@app.post("/api/v1/route", response_model=RouteResponse, tags=["routing"])
async def route_request(
    request: RouteRequest,
    background_tasks: BackgroundTasks,
    router: MoERouterSystem = Depends(lambda r: r.app.state.router),
    http_client: httpx.AsyncClient = Depends(lambda r: r.app.state.http_client)
):
    """
    Route a request to the optimal service(s)
    
    Features:
    - Adaptive strategy selection
    - Load-aware routing
    - Circuit breaker protection
    - Real-time performance tracking
    """
    try:
        # Process request
        start_time = time.perf_counter()
        
        # Route request
        decision = await router.route_request(
            request.data,
            request.routing_strategy
        )
        
        # Broadcast update
        background_tasks.add_task(manager.broadcast_routing_update, decision)
        
        # If proxy enabled, forward request
        if request.proxy_request and decision.selected_services:
            service_id = decision.selected_services[0]
            service = router.services.get(service_id)
            
            if service:
                try:
                    # Forward request to selected service
                    response = await http_client.post(
                        f"{service.endpoint}/api/v1/{request.endpoint or 'process'}",
                        json=request.data,
                        headers={"X-Routed-By": "moe-router"}
                    )
                    
                    # Report success
                    latency = (time.perf_counter() - start_time) * 1000
                    await router.report_result(service_id, True, latency)
                    
                    return RouteResponse(
                        selected_services=decision.selected_services,
                        routing_strategy=decision.routing_strategy.value,
                        confidence_scores=decision.confidence_scores,
                        reasoning=decision.reasoning,
                        proxy_response=response.json() if response.status_code == 200 else None,
                        latency_ms=latency,
                        success=True
                    )
                    
                except Exception as e:
                    # Report failure
                    await router.report_result(service_id, False, 0, str(e))
                    raise
        
        # Return routing decision
        return RouteResponse(
            selected_services=decision.selected_services,
            routing_strategy=decision.routing_strategy.value,
            confidence_scores=decision.confidence_scores,
            reasoning=decision.reasoning,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            success=True
        )
        
    except Exception as e:
        logger.error("Routing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/route/batch", response_model=BatchRouteResponse, tags=["routing"])
async def batch_route_requests(
    request: BatchRouteRequest,
    router: MoERouterSystem = Depends(lambda r: r.app.state.router)
):
    """
    Route multiple requests in batch for efficiency
    """
    try:
        results = []
        
        # Process requests in parallel
        tasks = []
        for req_data in request.requests:
            task = router.route_request(req_data, request.routing_strategy)
            tasks.append(task)
        
        decisions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, decision in enumerate(decisions):
            if isinstance(decision, Exception):
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(decision)
                })
            else:
                results.append({
                    "index": i,
                    "success": True,
                    "selected_services": decision.selected_services,
                    "routing_strategy": decision.routing_strategy.value,
                    "confidence_scores": decision.confidence_scores
                })
        
        return BatchRouteResponse(
            total_requests=len(request.requests),
            successful_routes=sum(1 for r in results if r["success"]),
            results=results
        )
        
    except Exception as e:
        logger.error("Batch routing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/services/register", tags=["services"])
async def register_service(
    request: ServiceRegistrationRequest,
    router: MoERouterSystem = Depends(lambda r: r.app.state.router)
):
    """
    Register a new service with the router
    """
    try:
        # Create service profile
        profile = ServiceProfile(
            service_id=request.service_id,
            service_type=ServiceType(request.service_type),
            endpoint=request.endpoint,
            capabilities=set(request.capabilities),
            specializations=request.specializations,
            max_capacity=request.max_capacity
        )
        
        # Add to router
        router.services[profile.service_id] = profile
        router.hash_router.add_service(profile.service_id, profile.performance_score)
        
        # Update neural router (would need retraining in production)
        router.num_services = len(router.services)
        
        logger.info(f"Registered service: {profile.service_id}")
        
        return {
            "status": "registered",
            "service_id": profile.service_id,
            "total_services": len(router.services)
        }
        
    except Exception as e:
        logger.error("Service registration failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/services/{service_id}", tags=["services"])
async def unregister_service(
    service_id: str,
    router: MoERouterSystem = Depends(lambda r: r.app.state.router)
):
    """
    Unregister a service from the router
    """
    if service_id not in router.services:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Remove from router
    del router.services[service_id]
    router.hash_router.remove_service(service_id)
    
    # Remove circuit breaker
    if service_id in router.circuit_breakers:
        del router.circuit_breakers[service_id]
    
    return {"status": "unregistered", "service_id": service_id}


@app.get("/api/v1/services", tags=["services"])
async def list_services(
    router: MoERouterSystem = Depends(lambda r: r.app.state.router)
):
    """
    List all registered services with their status
    """
    services = []
    
    for service_id, profile in router.services.items():
        breaker = router.circuit_breakers.get(service_id)
        
        services.append({
            "service_id": service_id,
            "service_type": profile.service_type.value,
            "endpoint": profile.endpoint,
            "capabilities": list(profile.capabilities),
            "available": service_id in router._get_available_services(),
            "circuit_breaker_state": breaker.state if breaker else "unknown",
            "current_load": profile.current_load,
            "max_capacity": profile.max_capacity,
            "performance_score": profile.performance_score,
            "latency_p95_ms": profile.latency_p95_ms,
            "error_rate": profile.error_rate
        })
    
    return {"services": services, "total": len(services)}


@app.get("/api/v1/metrics", response_model=RouterMetricsResponse, tags=["monitoring"])
async def get_metrics(
    router: MoERouterSystem = Depends(lambda r: r.app.state.router),
    performance_tracker: PerformanceTracker = Depends(lambda r: r.app.state.performance_tracker)
):
    """
    Get comprehensive router metrics
    """
    router_metrics = router.get_metrics()
    perf_metrics = performance_tracker.get_metrics()
    
    return RouterMetricsResponse(
        **router_metrics,
        **perf_metrics,
        timestamp=time.time()
    )


@app.post("/api/v1/strategy/override", tags=["configuration"])
async def override_strategy(
    request: StrategyOverrideRequest,
    router: MoERouterSystem = Depends(lambda r: r.app.state.router)
):
    """
    Override routing strategy temporarily
    """
    # This would update router configuration
    # For demo, just return confirmation
    return {
        "status": "strategy_overridden",
        "strategy": request.strategy,
        "duration_seconds": request.duration_seconds
    }


@app.post("/api/v1/load/rebalance", response_model=LoadBalanceResponse, tags=["operations"])
async def trigger_rebalance(
    request: LoadBalanceRequest,
    router: MoERouterSystem = Depends(lambda r: r.app.state.router),
    load_monitor: LoadMonitorService = Depends(lambda r: r.app.state.load_monitor)
):
    """
    Trigger manual load rebalancing
    """
    try:
        # Get current load distribution
        before_distribution = {
            sid: profile.current_load 
            for sid, profile in router.services.items()
        }
        
        # Perform rebalancing
        rebalance_result = await load_monitor.rebalance_load(
            request.target_distribution
        )
        
        # Get after distribution
        after_distribution = {
            sid: profile.current_load 
            for sid, profile in router.services.items()
        }
        
        return LoadBalanceResponse(
            before_distribution=before_distribution,
            after_distribution=after_distribution,
            migrations_performed=rebalance_result.get("migrations", 0),
            success=True,
            message="Load rebalancing completed"
        )
        
    except Exception as e:
        logger.error("Load rebalancing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/routing")
async def websocket_routing_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time routing updates
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
            elif data.startswith("subscribe:"):
                service_id = data.split(":")[1]
                if service_id not in manager.service_subscriptions:
                    manager.service_subscriptions[service_id] = []
                manager.service_subscriptions[service_id].append(websocket)
                await websocket.send_json({
                    "type": "subscribed",
                    "service_id": service_id
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/v1/demo/smart_routing", tags=["demo"])
async def demo_smart_routing(
    complexity: str = "simple",
    router: MoERouterSystem = Depends(lambda r: r.app.state.router)
):
    """
    Demo: See intelligent routing in action
    """
    try:
        # Create demo requests
        demo_requests = {
            "simple": {
                "type": "inference",
                "data": [0.5] * 100,
                "priority": 0.5,
                "complexity": 0.2
            },
            "complex": {
                "type": "consensus",
                "data": [[0.1, 0.2] * 50 for _ in range(10)],
                "priority": 0.9,
                "complexity": 0.8,
                "requires": ["fault_tolerance", "voting"]
            },
            "adaptive": {
                "type": "training",
                "data": list(range(1000)),
                "priority": 0.7,
                "complexity": 0.6,
                "requires": ["continuous_learning", "adaptive"]
            }
        }
        
        request_data = demo_requests.get(complexity, demo_requests["simple"])
        
        # Try different strategies
        strategies = [
            RoutingStrategy.SWITCH_TRANSFORMER,
            RoutingStrategy.TOP_K,
            RoutingStrategy.SEMANTIC,
            RoutingStrategy.ADAPTIVE
        ]
        
        results = []
        
        for strategy in strategies:
            decision = await router.route_request(request_data, strategy)
            
            results.append({
                "strategy": strategy.value,
                "selected_services": decision.selected_services,
                "confidence_scores": decision.confidence_scores,
                "reasoning": decision.reasoning,
                "estimated_latency_ms": decision.estimated_latency_ms,
                "load_balance_score": decision.load_balance_score
            })
        
        # Get best strategy
        best_result = min(results, key=lambda r: r["estimated_latency_ms"])
        
        return {
            "request_type": request_data["type"],
            "complexity": complexity,
            "routing_results": results,
            "recommended_strategy": best_result["strategy"],
            "explanation": f"For {complexity} {request_data['type']} requests, "
                          f"{best_result['strategy']} provides optimal routing with "
                          f"{best_result['estimated_latency_ms']:.1f}ms latency"
        }
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with correlation ID"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        path=request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": correlation_id,
            "message": str(exc) if request.app.debug else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )