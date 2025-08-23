#!/usr/bin/env python3
"""
AURA Production API 2025
Complete Shape-Aware Context Intelligence Platform
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import time
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

# Import AURA system
from aura_intelligence.integration.complete_system_2025 import (
    get_complete_aura_system, 
    SystemRequest, 
    process_aura_request
)

app = FastAPI(
    title="AURA Intelligence Platform 2025",
    description="Shape-Aware Context Intelligence with TDA, Mem0, Neo4j, LNN Council, and MCP",
    version="2025.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AURARequest(BaseModel):
    agent_id: str
    request_type: str  # analysis, decision, memory_query, coordination
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    priority: int = 5

class TopologyAnalysisRequest(BaseModel):
    agent_id: str
    data_points: List[List[float]]
    analysis_type: str = "persistent_homology"

class CouncilDecisionRequest(BaseModel):
    agent_id: str
    decision_context: Dict[str, Any]
    require_consensus: bool = True

class MemorySearchRequest(BaseModel):
    agent_id: str
    query: str
    context_data: Optional[List[List[float]]] = None
    limit: int = 10

# Global system instance
aura_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize AURA system on startup"""
    global aura_system
    print("🚀 Starting AURA Production API 2025...")
    
    aura_system = get_complete_aura_system()
    await aura_system.initialize()
    
    print("✅ AURA system initialized and ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown AURA system"""
    global aura_system
    if aura_system:
        await aura_system.shutdown()
    print("🔄 AURA system shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "system": "AURA Intelligence Platform 2025",
        "description": "Shape-Aware Context Intelligence",
        "version": "2025.1.0",
        "components": [
            "TDA-Neo4j Bridge (Topological Analysis)",
            "Mem0-Neo4j Bridge (Hybrid Memory)",
            "MCP Communication Hub (Agent Coordination)",
            "LNN Council System (Byzantine Consensus)",
            "Complete System Integration"
        ],
        "status": "operational",
        "endpoints": {
            "/health": "System health check",
            "/process": "Process AURA request",
            "/topology/analyze": "Topological analysis",
            "/council/decide": "Council decision making",
            "/memory/search": "Hybrid memory search",
            "/system/status": "Detailed system status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not aura_system or not aura_system.initialized:
        raise HTTPException(status_code=503, detail="AURA system not initialized")
    
    status = await aura_system.get_system_status()
    
    return {
        "status": "healthy" if status["system_health"] == "healthy" else "degraded",
        "system_initialized": status["system_initialized"],
        "success_rate": status["performance_metrics"]["success_rate"],
        "avg_processing_time_ms": status["performance_metrics"]["avg_processing_time_ms"],
        "requests_processed": status["performance_metrics"]["requests_processed"],
        "timestamp": time.time()
    }

@app.post("/process")
async def process_request(request: AURARequest):
    """Process general AURA request"""
    if not aura_system:
        raise HTTPException(status_code=503, detail="AURA system not available")
    
    try:
        response = await process_aura_request(
            agent_id=request.agent_id,
            request_type=request.request_type,
            data=request.data,
            context=request.context
        )
        
        return {
            "success": response.success,
            "request_id": response.request_id,
            "processing_time_ms": response.processing_time_ms,
            "components_used": response.components_used,
            "result": response.result,
            "topological_analysis": response.topological_analysis,
            "council_decision": response.council_decision
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/topology/analyze")
async def analyze_topology(request: TopologyAnalysisRequest):
    """Analyze topological structure of data"""
    if not aura_system:
        raise HTTPException(status_code=503, detail="AURA system not available")
    
    try:
        response = await process_aura_request(
            agent_id=request.agent_id,
            request_type="analysis",
            data={
                "data_points": request.data_points,
                "analysis_type": request.analysis_type
            }
        )
        
        if response.topological_analysis:
            return {
                "success": True,
                "betti_numbers": response.topological_analysis["betti_numbers"],
                "complexity_score": response.topological_analysis["complexity_score"],
                "shape_hash": response.topological_analysis["shape_hash"],
                "persistence_features": response.topological_analysis["persistence_features"],
                "processing_time_ms": response.processing_time_ms
            }
        else:
            return {
                "success": False,
                "error": "Topological analysis not available in response"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topology analysis failed: {str(e)}")

@app.post("/council/decide")
async def council_decision(request: CouncilDecisionRequest):
    """Make decision using LNN council with Byzantine consensus"""
    if not aura_system:
        raise HTTPException(status_code=503, detail="AURA system not available")
    
    try:
        response = await process_aura_request(
            agent_id=request.agent_id,
            request_type="decision",
            data={
                "decision_context": request.decision_context,
                "require_consensus": request.require_consensus
            }
        )
        
        if response.council_decision:
            return {
                "success": True,
                "decision": response.council_decision["decision"],
                "confidence": response.council_decision["confidence"],
                "reasoning": response.council_decision["reasoning"],
                "processing_time_ms": response.processing_time_ms
            }
        else:
            return {
                "success": False,
                "error": "Council decision not available in response"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Council decision failed: {str(e)}")

@app.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search hybrid memory (semantic + topological)"""
    if not aura_system:
        raise HTTPException(status_code=503, detail="AURA system not available")
    
    try:
        response = await process_aura_request(
            agent_id=request.agent_id,
            request_type="memory_query",
            data={
                "query": request.query,
                "data_points": request.context_data,
                "limit": request.limit
            }
        )
        
        return {
            "success": response.success,
            "memories_found": response.result.get("memory_context_count", 0),
            "topological_matches": len(response.result.get("topological_features", {})),
            "processing_time_ms": response.processing_time_ms,
            "result": response.result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")

@app.get("/system/status")
async def system_status():
    """Get detailed system status"""
    if not aura_system:
        raise HTTPException(status_code=503, detail="AURA system not available")
    
    status = await aura_system.get_system_status()
    return status

@app.post("/system/benchmark")
async def run_benchmark(background_tasks: BackgroundTasks):
    """Run system benchmark"""
    if not aura_system:
        raise HTTPException(status_code=503, detail="AURA system not available")
    
    background_tasks.add_task(run_background_benchmark)
    
    return {
        "message": "Benchmark started in background",
        "check_status": "/system/status"
    }

async def run_background_benchmark():
    """Run benchmark in background"""
    import numpy as np
    
    print("🏃 Running AURA system benchmark...")
    
    # Create test requests
    test_requests = []
    for i in range(20):
        test_data = np.random.randn(15, 3).tolist()
        
        request = SystemRequest(
            request_id=f"benchmark_{i}",
            agent_id=f"benchmark_agent_{i}",
            request_type="analysis",
            data={
                "data_points": test_data,
                "query": f"benchmark test {i}"
            }
        )
        test_requests.append(request)
    
    # Process requests
    start_time = time.time()
    tasks = [aura_system.process_request(req) for req in test_requests]
    responses = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Calculate metrics
    successful = sum(1 for r in responses if r.success)
    avg_time = sum(r.processing_time_ms for r in responses) / len(responses)
    throughput = len(responses) / total_time
    
    print(f"📊 Benchmark Results:")
    print(f"  - Requests: {len(responses)}")
    print(f"  - Successful: {successful}")
    print(f"  - Success rate: {successful/len(responses)*100:.1f}%")
    print(f"  - Average processing time: {avg_time:.2f}ms")
    print(f"  - Throughput: {throughput:.1f} requests/second")
    print(f"  - Total time: {total_time:.2f}s")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AURA Production API 2025")
    print("📍 API will be available at: http://localhost:8087")
    print("📚 Documentation at: http://localhost:8087/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8087,
        log_level="info",
        access_log=True
    )