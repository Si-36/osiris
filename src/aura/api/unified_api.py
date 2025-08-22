"""
AURA Intelligence Unified API
Production-grade FastAPI application with advanced features
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
import logging
import time
import json
import uuid
from datetime import datetime
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import opentelemetry
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Import AURA components
from ..core.system import AURASystem
from ..core.config import AURAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter('aura_requests_total', 'Total requests')
request_duration = Histogram('aura_request_duration_seconds', 'Request duration')
active_agents = Gauge('aura_active_agents', 'Number of active agents')
cascade_risk = Gauge('aura_cascade_risk', 'Current cascade risk level')

# Models
class AgentData(BaseModel):
    """Agent network data model"""
    agents: List[Dict[str, Any]]
    connections: List[List[int]]
    metadata: Optional[Dict[str, Any]] = {}

class TopologyRequest(BaseModel):
    """Topology analysis request"""
    agent_data: AgentData
    analysis_depth: int = Field(default=2, ge=1, le=5)
    include_predictions: bool = True

class InterventionRequest(BaseModel):
    """Intervention request model"""
    agent_id: str
    intervention_type: str = Field(default="isolate", pattern="^(isolate|heal|reroute)$")
    parameters: Optional[Dict[str, Any]] = {}

class HealthResponse(BaseModel):
    """System health response"""
    status: str
    uptime: float
    component_status: Dict[str, str]
    metrics: Dict[str, float]

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AURA Intelligence API...")
    
    # Initialize AURA system
    config = AURAConfig()
    app.state.aura_system = AURASystem(config)
    
    # Initialize Redis for caching
    try:
        app.state.redis = await aioredis.create_redis_pool('redis://localhost')
    except:
        logger.warning("Redis not available, caching disabled")
        app.state.redis = None
    
    # Start background tasks
    app.state.monitoring_task = asyncio.create_task(monitor_system(app))
    
    logger.info("AURA API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AURA API...")
    
    # Cancel background tasks
    app.state.monitoring_task.cancel()
    
    # Close Redis
    if app.state.redis:
        app.state.redis.close()
        await app.state.redis.wait_closed()

# Create FastAPI app
app = FastAPI(
    title="AURA Intelligence API",
    description="Advanced multi-agent failure prevention system",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Background monitoring
async def monitor_system(app: FastAPI):
    """Background system monitoring"""
    while True:
        try:
            # Update metrics
            system = app.state.aura_system
            stats = system.get_system_status()
            
            active_agents.set(len(system.agents))
            cascade_risk.set(stats.get('risk_level', 0))
            
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")

# Dependency injection
async def get_aura_system():
    """Get AURA system instance"""
    return app.state.aura_system

# Routes

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system info"""
    return {
        "name": "AURA Intelligence",
        "version": "2.0.0",
        "status": "operational",
        "hypothesis": "Prevent agent failures through topological context intelligence",
        "components": 213,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "predict": "/predict",
            "intervene": "/intervene",
            "stream": "/stream",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(system: AURASystem = Depends(get_aura_system)):
    """Comprehensive health check"""
    start_time = time.time()
    
    # Check all components
    component_status = {}
    try:
        # TDA Engine
        component_status["tda_engine"] = "healthy" if len(system.tda_algorithms) == 112 else "degraded"
        
        # Neural Networks
        component_status["neural_networks"] = "healthy" if len(system.neural_networks) == 10 else "degraded"
        
        # Memory Systems
        component_status["memory"] = "healthy" if len(system.memory_components) == 40 else "degraded"
        
        # Agents
        component_status["agents"] = "healthy" if len(system.agents) == 100 else "degraded"
        
        # Infrastructure
        component_status["infrastructure"] = "healthy" if len(system.infrastructure) == 51 else "degraded"
    except Exception as e:
        logger.error(f"Health check error: {e}")
        component_status["error"] = str(e)
    
    # Calculate uptime
    uptime = time.time() - start_time
    
    # Get current metrics
    metrics = {
        "active_agents": len(system.agents),
        "cascade_risk": 0.0,  # Would calculate from system
        "memory_usage_mb": 0.0,  # Would get from psutil
        "cpu_percent": 0.0,  # Would get from psutil
    }
    
    # Determine overall status
    unhealthy_components = [k for k, v in component_status.items() if v != "healthy"]
    status = "degraded" if unhealthy_components else "healthy"
    
    return HealthResponse(
        status=status,
        uptime=uptime,
        component_status=component_status,
        metrics=metrics
    )

@app.post("/analyze")
async def analyze_topology(
    request: TopologyRequest,
    background_tasks: BackgroundTasks,
    system: AURASystem = Depends(get_aura_system)
):
    """Analyze agent network topology"""
    request_count.inc()
    
    with request_duration.time():
        try:
            # Convert to system format
            agent_data = {
                "agents": request.agent_data.agents,
                "connections": request.agent_data.connections,
                "metadata": request.agent_data.metadata
            }
            
            # Analyze topology
            analysis = await system.analyze_topology(agent_data)
            
            # Add predictions if requested
            if request.include_predictions:
                prediction = await system.predict_failure(analysis)
                analysis["prediction"] = prediction
            
            # Cache result
            if app.state.redis:
                cache_key = f"analysis:{hash(str(agent_data))}"
                await app.state.redis.setex(cache_key, 300, json.dumps(analysis))
            
            # Background task to update metrics
            background_tasks.add_task(update_analysis_metrics, analysis)
            
            return {
                "status": "success",
                "analysis": analysis,
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_failure(
    request: TopologyRequest,
    system: AURASystem = Depends(get_aura_system)
):
    """Predict cascade failures"""
    try:
        # First analyze topology
        agent_data = {
            "agents": request.agent_data.agents,
            "connections": request.agent_data.connections
        }
        
        topology = await system.analyze_topology(agent_data)
        
        # Predict failures
        prediction = await system.predict_failure(topology)
        
        return {
            "status": "success",
            "prediction": prediction,
            "risk_level": prediction.get("risk_score", 0),
            "at_risk_agents": prediction.get("at_risk_nodes", []),
            "recommended_interventions": prediction.get("interventions", [])
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intervene")
async def intervene(
    request: InterventionRequest,
    system: AURASystem = Depends(get_aura_system)
):
    """Execute intervention on agent"""
    try:
        # Execute intervention
        result = await system.prevent_cascade(
            agent_id=request.agent_id,
            intervention_type=request.intervention_type,
            parameters=request.parameters
        )
        
        return {
            "status": "success",
            "intervention": {
                "agent_id": request.agent_id,
                "type": request.intervention_type,
                "result": result
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Intervention error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
async def stream_updates():
    """Stream real-time system updates"""
    async def generate():
        while True:
            try:
                # Get current system state
                system = app.state.aura_system
                status = system.get_system_status()
                
                # Create update event
                event = {
                    "event": "system_update",
                    "data": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_agents": len(system.agents),
                        "cascade_risk": status.get("risk_level", 0),
                        "interventions": status.get("interventions_count", 0)
                    }
                }
                
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time bidirectional communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process command
            if data.get("type") == "analyze":
                system = app.state.aura_system
                result = await system.analyze_topology(data.get("payload", {}))
                await websocket.send_json({
                    "type": "analysis_result",
                    "data": result
                })
            
            elif data.get("type") == "subscribe":
                # Start sending updates
                while True:
                    status = app.state.aura_system.get_system_status()
                    await websocket.send_json({
                        "type": "status_update",
                        "data": status
                    })
                    await asyncio.sleep(1)
                    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type="text/plain"
    )

# Helper functions
async def update_analysis_metrics(analysis: Dict[str, Any]):
    """Update metrics after analysis"""
    # Extract metrics from analysis
    if "risk_score" in analysis:
        cascade_risk.set(analysis["risk_score"])

# Advanced endpoints

@app.get("/topology/visualize")
async def visualize_topology(
    format: str = "cytoscape",
    system: AURASystem = Depends(get_aura_system)
):
    """Get topology visualization data"""
    try:
        # Get current topology
        agents = system.agents
        
        # Convert to requested format
        if format == "cytoscape":
            nodes = [{"id": id, "label": f"Agent {id}"} for id in agents.keys()]
            edges = []  # Would extract from agent connections
            
            return {
                "elements": {
                    "nodes": nodes,
                    "edges": edges
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {format}")
            
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/analyze")
async def batch_analyze(
    requests: List[TopologyRequest],
    system: AURASystem = Depends(get_aura_system)
):
    """Batch topology analysis for multiple networks"""
    results = []
    
    for req in requests:
        try:
            agent_data = {
                "agents": req.agent_data.agents,
                "connections": req.agent_data.connections
            }
            analysis = await system.analyze_topology(agent_data)
            results.append({"status": "success", "analysis": analysis})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})
    
    return {
        "total": len(requests),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }

@app.get("/debug/components")
async def debug_components(system: AURASystem = Depends(get_aura_system)):
    """Debug endpoint to inspect all components"""
    components = system.get_all_components()
    
    return {
        "tda_algorithms": len(components["tda_algorithms"]),
        "neural_networks": len(components["neural_networks"]),
        "memory_components": len(components["memory_components"]),
        "agents": len(components["agents"]),
        "infrastructure": len(components["infrastructure"]),
        "samples": {
            "tda": components["tda_algorithms"][:5],
            "agents": components["agents"][:5]
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if app.debug else None
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)