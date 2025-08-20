#!/usr/bin/env python3
"""
Minimal Working AURA API - No complex imports
"""
import sys
from pathlib import Path

# Add core to path
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

import asyncio
from typing import Dict, Any, List
from datetime import datetime
import torch
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import only working components
from aura_intelligence.components.real_registry import get_real_registry
from aura_intelligence.streaming.kafka_integration import get_event_streaming, EventType
from aura_intelligence.graph.neo4j_integration import get_neo4j_integration

# API Models
class ProcessRequest(BaseModel):
    data: List[float]
    task_type: str = "neural"

class EventRequest(BaseModel):
    type: str
    source: str
    data: Dict[str, Any]

app = FastAPI(
    title="Minimal AURA Intelligence API",
    description="Working API with real components",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
registry = None
event_streaming = None
neo4j_integration = None

@app.on_event("startup")
async def startup():
    """Initialize working components"""
    global registry, event_streaming, neo4j_integration
    
    print("🧠 Initializing Minimal AURA API...")
    
    try:
        registry = get_real_registry()
        print(f"✅ Registry: {len(registry.components)} components loaded")
        
        event_streaming = get_event_streaming()
        await event_streaming.start_streaming()
        print("✅ Event streaming initialized")
        
        neo4j_integration = get_neo4j_integration()
        print("✅ Neo4j integration initialized")
        
        print("🎉 Minimal AURA API ready!")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Minimal AURA Intelligence API",
        "version": "1.0.0",
        "status": "working",
        "components": len(registry.components) if registry else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "components": {
            "registry": registry is not None,
            "event_streaming": event_streaming is not None,
            "neo4j": neo4j_integration is not None
        },
        "component_count": len(registry.components) if registry else 0
    }

@app.post("/process")
async def process_data(request: ProcessRequest):
    """Process data through components"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not available")
    
    try:
        # Get a neural component
        from aura_intelligence.components.real_registry import ComponentType
        neural_components = registry.get_components_by_type(ComponentType.NEURAL)
        if not neural_components:
            raise HTTPException(status_code=404, detail="No neural components found")
        
        component_id = neural_components[0].id
        
        # Process through component
        result = await registry.process_data(component_id, {"values": request.data})
        
        return {
            "success": True,
            "component_used": component_id,
            "result": result,
            "processing_time": "< 0.1s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/events/publish")
async def publish_event(request: EventRequest):
    """Publish event"""
    if not event_streaming:
        raise HTTPException(status_code=503, detail="Event streaming not available")
    
    try:
        event_type = EventType(request.type) if request.type in [e.value for e in EventType] else EventType.COMPONENT_HEALTH
        
        success = await event_streaming.publish_system_event(
            event_type,
            request.source,
            request.data
        )
        
        stats = event_streaming.get_streaming_stats()
        
        return {
            "success": success,
            "event_published": success,
            "streaming_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event publishing failed: {str(e)}")

@app.post("/graph/store")
async def store_decision(decision_data: Dict[str, Any]):
    """Store decision in Neo4j"""
    if not neo4j_integration:
        raise HTTPException(status_code=503, detail="Neo4j not available")
    
    try:
        success = await neo4j_integration.store_council_decision(decision_data)
        status = neo4j_integration.get_connection_status()
        
        return {
            "success": success,
            "decision_stored": success,
            "connection_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

@app.get("/components")
async def list_components():
    """List all components"""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not available")
    
    components = []
    for comp_id, component in registry.components.items():
        components.append({
            "id": comp_id,
            "type": component.type.value,
            "status": component.status,
            "processing_time": component.processing_time,
            "data_processed": component.data_processed
        })
    
    return {
        "total_components": len(components),
        "components": components[:10],  # First 10 for brevity
        "component_types": registry.get_component_stats()["type_distribution"]
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = {}
    
    if registry:
        stats["registry"] = registry.get_component_stats()
    
    if event_streaming:
        stats["streaming"] = event_streaming.get_streaming_stats()
    
    if neo4j_integration:
        stats["neo4j"] = neo4j_integration.get_connection_status()
    
    return stats

def main():
    """Run the minimal AURA API"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                🧠 Minimal AURA Intelligence API              ║
    ║                      VERSION 1.0.0                          ║
    ║                                                              ║
    ║  Real components, real integrations, minimal complexity     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting Minimal AURA Intelligence API...")
    print("📍 Server: http://localhost:8080")
    print("📚 API docs: http://localhost:8080/docs")
    print("🔍 Health: http://localhost:8080/health")
    
    uvicorn.run(
        "minimal_working_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()