#!/usr/bin/env python3
"""
AURA Intelligence Real API - Direct integration with YOUR components
Bypasses circular imports, uses direct component imports
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import logging
import json
from datetime import datetime
import asyncio
import sys
import os

# Add core to Python path
core_path = os.path.join(os.path.dirname(__file__), 'core', 'src')
sys.path.insert(0, core_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components directly without config circular imports
components_status = {
    "tda": False,
    "lnn": False,
    "memory": False,
    "redis": False,
    "real_components": False
}

# Try to import individual working components
try:
    from aura_intelligence.tda.real_tda import RealTDA
    tda_engine = RealTDA()
    components_status["tda"] = True
    logger.info("✅ TDA engine loaded")
except Exception as e:
    logger.warning(f"⚠️  TDA engine not available: {e}")
    tda_engine = None

try:
    from aura_intelligence.lnn.real_mit_lnn import get_real_mit_lnn
    lnn_model = get_real_mit_lnn(input_size=10, hidden_size=64, output_size=1)
    components_status["lnn"] = True
    logger.info("✅ LNN model loaded")
except Exception as e:
    logger.warning(f"⚠️  LNN model not available: {e}")
    lnn_model = None

try:
    from aura_intelligence.components.real_components import RedisConnectionPool
    redis_pool = RedisConnectionPool()
    components_status["redis"] = True
    logger.info("✅ Redis pool loaded")
except Exception as e:
    logger.warning(f"⚠️  Redis pool not available: {e}")
    redis_pool = None

try:
    from aura_intelligence.examples.real_demo_system import RealDemoSystem
    demo_system = RealDemoSystem()
    components_status["real_components"] = True
    logger.info("✅ Demo system loaded")
except Exception as e:
    logger.warning(f"⚠️  Demo system not available: {e}")
    demo_system = None

try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    components_status["memory"] = True
    logger.info("✅ Redis memory available")
except Exception as e:
    logger.warning(f"⚠️  Redis memory not available: {e}")
    redis_client = None

# Data models
class AURAAnalysisRequest(BaseModel):
    """AURA analysis request"""
    data: Dict[str, Any] = Field(..., description="Input data")
    components: List[str] = Field(default=["tda"], description="Components to use")
    options: Dict[str, Any] = Field(default={}, description="Options")

class AURAAnalysisResponse(BaseModel):
    """AURA analysis response"""
    success: bool
    processing_time_ms: float
    results: Dict[str, Any] = {}
    components_used: List[str] = []
    error: Optional[str] = None

# Initialize FastAPI
app = FastAPI(
    title="AURA Intelligence - Real Component API",
    description="Direct integration with YOUR actual AURA components",
    version="2.0.0"
)

# System metrics
class AURAMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.requests = 0
        self.successes = 0
        self.times = []
    
    def record(self, time_ms: float, success: bool):
        self.requests += 1
        if success:
            self.successes += 1
            self.times.append(time_ms)
        if len(self.times) > 100:
            self.times = self.times[-100:]

metrics = AURAMetrics()

@app.get("/")
async def root():
    """Root endpoint"""
    working_components = sum(components_status.values())
    return {
        "service": "AURA Intelligence Real API",
        "status": "operational",
        "version": "2.0.0",
        "description": "Direct integration with YOUR AURA components",
        "components_available": components_status,
        "working_components": f"{working_components}/{len(components_status)}",
        "endpoints": {
            "analyze": "/analyze",
            "demo": "/demo/{name}",
            "components": "/components/status",
            "memory": "/memory/stats"
        }
    }

@app.post("/analyze", response_model=AURAAnalysisResponse)
async def analyze(request: AURAAnalysisRequest):
    """Analyze using YOUR real AURA components"""
    start_time = time.time()
    results = {}
    components_used = []
    
    try:
        # TDA Analysis with YOUR real TDA engine
        if "tda" in request.components and tda_engine:
            try:
                # Convert data for TDA
                if "points" in request.data:
                    points = request.data["points"]
                elif "data" in request.data and isinstance(request.data["data"], list):
                    points = request.data["data"]
                else:
                    points = list(request.data.values())[:20]  # Use first 20 values
                
                import numpy as np
                points_array = np.array(points).reshape(-1, 1) if len(np.array(points).shape) == 1 else np.array(points)
                
                tda_result = tda_engine.compute_persistence(points_array)
                results["tda"] = tda_result
                components_used.append("tda")
                logger.info(f"TDA analysis completed: {tda_result.get('library', 'unknown')} library")
                
            except Exception as e:
                results["tda"] = {"error": str(e)}
                logger.error(f"TDA analysis failed: {e}")
        
        # LNN Analysis with YOUR real LNN
        if "lnn" in request.components and lnn_model:
            try:
                import torch
                
                # Prepare LNN input
                if "features" in request.data:
                    features = request.data["features"]
                elif "data" in request.data and isinstance(request.data["data"], list):
                    features = request.data["data"][:10]  # Take first 10 for LNN
                else:
                    features = [float(v) if isinstance(v, (int, float)) else 0.0 for v in list(request.data.values())[:10]]
                
                # Pad to 10 dimensions
                while len(features) < 10:
                    features.append(0.0)
                features = features[:10]
                
                lnn_input = torch.tensor([features], dtype=torch.float32)
                
                with torch.no_grad():
                    lnn_output = lnn_model(lnn_input)
                    if isinstance(lnn_output, tuple):
                        lnn_output = lnn_output[0]
                
                results["lnn"] = {
                    "output": lnn_output.squeeze().tolist(),
                    "input_shape": list(lnn_input.shape),
                    "model_info": lnn_model.get_info()
                }
                components_used.append("lnn")
                logger.info("LNN analysis completed")
                
            except Exception as e:
                results["lnn"] = {"error": str(e)}
                logger.error(f"LNN analysis failed: {e}")
        
        # Memory operations with YOUR Redis
        if "memory" in request.components and redis_client:
            try:
                # Store analysis results
                timestamp = int(time.time())
                key = f"aura:analysis:{timestamp}"
                redis_client.set(key, json.dumps(request.data), ex=3600)
                
                # Get memory stats
                memory_info = redis_client.info('memory')
                results["memory"] = {
                    "stored_key": key,
                    "used_memory": memory_info.get('used_memory_human', 'unknown'),
                    "total_keys": redis_client.dbsize(),
                    "operation": "store_successful"
                }
                components_used.append("memory")
                logger.info(f"Memory operation completed: {key}")
                
            except Exception as e:
                results["memory"] = {"error": str(e)}
                logger.error(f"Memory operation failed: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        success = len(components_used) > 0
        metrics.record(processing_time, success)
        
        return AURAAnalysisResponse(
            success=success,
            processing_time_ms=processing_time,
            results=results,
            components_used=components_used
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        metrics.record(processing_time, False)
        
        return AURAAnalysisResponse(
            success=False,
            processing_time_ms=processing_time,
            error=str(e)
        )

@app.get("/demo/{demo_name}")
async def run_demo(demo_name: str):
    """Run YOUR real demo system"""
    if not demo_system:
        raise HTTPException(status_code=503, detail="Demo system not available")
    
    try:
        start_time = time.time()
        result = await demo_system.run_demo(demo_name)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "demo": demo_name,
            "result": result,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

@app.get("/components/status")
async def get_component_status():
    """Get status of YOUR AURA components"""
    detailed_status = {}
    
    # Test each component
    if tda_engine:
        detailed_status["tda"] = {
            "available": True,
            "libraries": tda_engine.available_libraries,
            "type": type(tda_engine).__name__
        }
    else:
        detailed_status["tda"] = {"available": False}
    
    if lnn_model:
        detailed_status["lnn"] = {
            "available": True,
            "info": lnn_model.get_info(),
            "type": type(lnn_model).__name__
        }
    else:
        detailed_status["lnn"] = {"available": False}
    
    if redis_client:
        try:
            redis_client.ping()
            detailed_status["redis"] = {
                "available": True,
                "connected": True,
                "dbsize": redis_client.dbsize()
            }
        except:
            detailed_status["redis"] = {"available": False, "connected": False}
    else:
        detailed_status["redis"] = {"available": False}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": components_status,
        "details": detailed_status,
        "metrics": {
            "uptime_seconds": time.time() - metrics.start_time,
            "requests": metrics.requests,
            "success_rate": metrics.successes / max(1, metrics.requests)
        }
    }

@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        info = redis_client.info()
        
        # Get AURA-specific keys
        aura_keys = []
        for key in redis_client.scan_iter(match="aura:*"):
            aura_keys.append(key)
        
        return {
            "redis_info": {
                "version": info.get('redis_version'),
                "uptime": info.get('uptime_in_seconds'),
                "connected_clients": info.get('connected_clients'),
                "used_memory": info.get('used_memory_human'),
                "total_keys": redis_client.dbsize()
            },
            "aura_data": {
                "aura_keys": len(aura_keys),
                "sample_keys": aura_keys[:10]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory stats failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🧠 AURA Intelligence - Real Component API")
    print("=" * 50)
    print("🚀 Direct integration with YOUR AURA components")
    
    working = sum(components_status.values())
    total = len(components_status)
    print(f"✅ Components working: {working}/{total}")
    
    for component, status in components_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {component}")
    
    print("=" * 50)
    print("\n📡 Endpoints:")
    print("  - Analysis: http://localhost:8080/analyze")
    print("  - Component Status: http://localhost:8080/components/status")
    if components_status["real_components"]:
        print("  - LNN Demo: http://localhost:8080/demo/lnn_demo")
        print("  - TDA Demo: http://localhost:8080/demo/tda_demo")
    if components_status["memory"]:
        print("  - Memory Stats: http://localhost:8080/memory/stats")
    
    print("\n🎯 Test analysis:")
    print('curl -X POST http://localhost:8080/analyze -H "Content-Type: application/json" -d \'{"data":{"points":[1,2,3,10,11,12]}, "components":["tda","lnn","memory"]}\'')
    
    uvicorn.run(app, host="0.0.0.0", port=8080)