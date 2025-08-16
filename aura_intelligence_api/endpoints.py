"""
AURA Intelligence API Endpoints
==============================

Clean REST API endpoints that expose your incredible system capabilities
through simple, easy-to-use interfaces.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

from .core_api import AURAIntelligence

logger = logging.getLogger(__name__)

# Initialize AURA Intelligence
aura = AURAIntelligence()

# FastAPI app
app = FastAPI(
    title="AURA Intelligence API",
    description="Complete AI system with neural networks, consciousness, agents, and advanced capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class IntelligenceRequest(BaseModel):
    data: Dict[str, Any]
    query: Optional[str] = None
    requirements: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

class IntelligenceResponse(BaseModel):
    status: str
    request_id: str
    timestamp: str
    summary: str
    insights: Dict[str, Any]
    confidence_score: float
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]

# Main endpoints
@app.post("/intelligence", response_model=IntelligenceResponse)
async def process_intelligence(request: IntelligenceRequest):
    """
    Process intelligence request through ALL AURA capabilities
    
    This endpoint gives you access to:
    - Neural processing (5,514+ parameters)
    - Advanced memory systems (Mem0, episodic, vector)
    - Multi-model AI (Gemini, others)
    - Deep agent orchestration (LangGraph)
    - Consciousness and reasoning
    - Topological data analysis
    - And 100+ other advanced features
    """
    try:
        result = await aura.process_intelligence_request(request.dict())
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return IntelligenceResponse(**result)
        
    except Exception as e:
        logger.error(f"Intelligence processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all AURA components"""
    try:
        health = await aura.health_check()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def system_status():
    """Get detailed system status"""
    try:
        return aura.get_system_status()
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Specialized endpoints for different capabilities
@app.post("/neural")
async def neural_endpoint(request: IntelligenceRequest):
    """Direct access to neural processing capabilities"""
    try:
        result = await aura._process_neural(request.dict())
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory")
async def memory_endpoint(request: IntelligenceRequest):
    """Direct access to memory systems (Mem0, episodic, vector search)"""
    try:
        result = await aura._process_memory(request.dict())
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai")
async def ai_endpoint(request: IntelligenceRequest):
    """Direct access to AI systems (Gemini, multi-model)"""
    try:
        result = await aura._process_ai(request.dict())
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consciousness")
async def consciousness_endpoint(request: IntelligenceRequest):
    """Direct access to consciousness and reasoning systems"""
    try:
        # Need insights for consciousness processing
        insights = {}
        result = await aura._process_consciousness(request.dict(), insights)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents")
async def agent_endpoint(request: IntelligenceRequest):
    """Direct access to agent orchestration (LangGraph, Council, etc.)"""
    try:
        insights = {}
        result = await aura._process_agents(request.dict(), insights)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orchestration")
async def orchestration_endpoint(request: IntelligenceRequest):
    """Advanced orchestration of multiple systems"""
    try:
        # This runs a simplified version of the full pipeline
        result = await aura.process_intelligence_request(request.dict())
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/components")
async def list_components():
    """List all available components and their status"""
    try:
        status = aura.get_system_status()
        return {
            "available_components": status["components"],
            "total_categories": len(status["components"]),
            "initialized": status["initialized"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_process(requests: List[IntelligenceRequest]):
    """Process multiple intelligence requests in batch"""
    try:
        results = []
        for req in requests:
            result = await aura.process_intelligence_request(req.dict())
            results.append(result)
        
        return {
            "status": "success",
            "batch_size": len(requests),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time processing
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time intelligence processing"""
    await websocket.accept()
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            # Process through AURA
            result = await aura.process_intelligence_request(data)
            
            # Send response
            await websocket.send_json(result)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)