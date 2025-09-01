#!/usr/bin/env python3
"""
REAL AURA Intelligence API - Using YOUR actual system components
Integrates with core/src/aura_intelligence/ - NO MORE GENERIC FILES
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

# Add your core system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

# Import YOUR real system components
try:
    from aura_intelligence.core.unified_system import get_unified_system, create_unified_system
    from aura_intelligence.core.unified_config import get_config
    from aura_intelligence.unified_brain import UnifiedAURABrain
    from aura_intelligence.components.real_components import RedisConnectionPool
    from aura_intelligence.tda.real_tda import RealTDA
    from aura_intelligence.lnn.real_mit_lnn import get_real_mit_lnn
    from aura_intelligence.examples.real_demo_system import RealDemoSystem
    from aura_intelligence.integration.real_system_2025 import RealAURASystem
    AURA_CORE_AVAILABLE = True
    print("✅ AURA Intelligence core system loaded successfully")
except ImportError as e:
    AURA_CORE_AVAILABLE = False
    print(f"❌ Could not import AURA core system: {e}")
    print("💡 Ensure you're running from project root with correct PYTHONPATH")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models for API
class UnifiedAnalysisRequest(BaseModel):
    """Request for unified AURA analysis"""
    data: Dict[str, Any] = Field(..., description="Input data for analysis")
    components: List[str] = Field(default=["tda", "lnn"], description="Components to use")
    options: Dict[str, Any] = Field(default={}, description="Analysis options")

class UnifiedAnalysisResponse(BaseModel):
    """Response from unified AURA analysis"""
    success: bool
    processing_time_ms: float
    results: Dict[str, Any] = {}
    system_info: Dict[str, Any] = {}
    error: Optional[str] = None

class SystemStatusResponse(BaseModel):
    """System status and capabilities"""
    status: str
    uptime_seconds: float
    core_system_available: bool
    active_components: List[str]
    system_metrics: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="AURA Intelligence - Real System API",
    description="Using YOUR actual AURA Intelligence system - no mocks",
    version="2.0.0"
)

# Global system instances
unified_system = None
aura_brain = None
redis_pool = None
demo_system = None
real_aura_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize AURA system on startup"""
    global unified_system, aura_brain, redis_pool, demo_system, real_aura_system
    
    if not AURA_CORE_AVAILABLE:
        logger.error("AURA core system not available - running in degraded mode")
        return
    
    try:
        # Initialize your unified system
        config = get_config()
        unified_system = await create_unified_system(config)
        logger.info("✅ Unified system initialized")
        
        # Initialize AURA brain
        aura_brain = UnifiedAURABrain()
        logger.info("✅ AURA brain initialized")
        
        # Initialize Redis connection pool
        redis_pool = RedisConnectionPool()
        await redis_pool.initialize()
        logger.info("✅ Redis connection pool initialized")
        
        # Initialize demo system
        demo_system = RealDemoSystem()
        logger.info("✅ Demo system initialized")
        
        # Initialize real AURA system
        real_aura_system = RealAURASystem()
        logger.info("✅ Real AURA system initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize AURA system: {e}")
        # Continue anyway for basic functionality

# Track system metrics
class SystemMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.successful_requests = 0
        self.processing_times = []
    
    def record_request(self, processing_time: float, success: bool):
        self.request_count += 1
        if success:
            self.successful_requests += 1
            self.processing_times.append(processing_time)
        
        # Keep only recent times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def get_stats(self):
        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": self.request_count,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / max(1, self.request_count),
            "avg_processing_time_ms": sum(self.processing_times) / max(1, len(self.processing_times))
        }

metrics = SystemMetrics()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "service": "AURA Intelligence Real System API",
        "status": "operational" if AURA_CORE_AVAILABLE else "degraded",
        "version": "2.0.0",
        "description": "Using YOUR actual AURA Intelligence system",
        "core_system_available": AURA_CORE_AVAILABLE,
        "endpoints": {
            "status": "/status",
            "analyze": "/analyze/unified",
            "demo": "/demo/{demo_name}",
            "system": "/system/info"
        }
    }

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    
    active_components = []
    if AURA_CORE_AVAILABLE:
        if unified_system:
            active_components.append("unified_system")
        if aura_brain:
            active_components.append("aura_brain")
        if redis_pool:
            active_components.append("redis_pool")
        if demo_system:
            active_components.append("demo_system")
    
    return SystemStatusResponse(
        status="operational" if AURA_CORE_AVAILABLE else "degraded",
        uptime_seconds=time.time() - metrics.start_time,
        core_system_available=AURA_CORE_AVAILABLE,
        active_components=active_components,
        system_metrics=metrics.get_stats()
    )

@app.post("/analyze/unified", response_model=UnifiedAnalysisResponse)
async def analyze_with_unified_system(request: UnifiedAnalysisRequest):
    """Use YOUR real unified system for analysis"""
    
    if not AURA_CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="AURA core system not available")
    
    start_time = time.time()
    
    try:
        logger.info(f"Unified analysis request: components={request.components}")
        
        results = {}
        
        # Process with your unified system if available
        if unified_system:
            try:
                # Use your actual unified system
                unified_result = await unified_system.process(request.data)
                results["unified_system"] = unified_result
            except Exception as e:
                results["unified_system"] = {"error": str(e)}
        
        # Process with AURA brain if available
        if aura_brain and "brain" in request.components:
            try:
                brain_result = await aura_brain.analyze(request.data)
                results["aura_brain"] = brain_result.__dict__ if hasattr(brain_result, '__dict__') else str(brain_result)
            except Exception as e:
                results["aura_brain"] = {"error": str(e)}
        
        # Process with real AURA system
        if real_aura_system:
            try:
                aura_result = await real_aura_system.process_real_data(request.data)
                results["real_aura_system"] = aura_result
            except Exception as e:
                results["real_aura_system"] = {"error": str(e)}
        
        processing_time = (time.time() - start_time) * 1000
        success = len([r for r in results.values() if not isinstance(r, dict) or "error" not in r]) > 0
        
        metrics.record_request(processing_time, success)
        
        return UnifiedAnalysisResponse(
            success=success,
            processing_time_ms=processing_time,
            results=results,
            system_info={
                "components_used": request.components,
                "unified_system_active": unified_system is not None,
                "aura_brain_active": aura_brain is not None,
                "real_system_active": real_aura_system is not None
            }
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        metrics.record_request(processing_time, False)
        logger.error(f"Unified analysis failed: {e}")
        
        return UnifiedAnalysisResponse(
            success=False,
            processing_time_ms=processing_time,
            error=f"Analysis failed: {str(e)}"
        )

@app.get("/demo/{demo_name}")
async def run_demo(demo_name: str):
    """Run YOUR real demo system"""
    
    if not AURA_CORE_AVAILABLE or not demo_system:
        raise HTTPException(status_code=503, detail="Demo system not available")
    
    try:
        start_time = time.time()
        result = await demo_system.run_demo(demo_name)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "demo_name": demo_name,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

@app.get("/system/info")
async def get_system_info():
    """Get detailed system information"""
    
    info = {
        "aura_core_available": AURA_CORE_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "python_path": sys.path[:3],  # Show first 3 entries
        "components": {}
    }
    
    if AURA_CORE_AVAILABLE:
        if unified_system:
            try:
                info["components"]["unified_system"] = {
                    "status": "active",
                    "type": type(unified_system).__name__
                }
            except:
                info["components"]["unified_system"] = {"status": "error"}
        
        if aura_brain:
            try:
                info["components"]["aura_brain"] = {
                    "status": "active",
                    "type": type(aura_brain).__name__
                }
            except:
                info["components"]["aura_brain"] = {"status": "error"}
    
    return info

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy" if AURA_CORE_AVAILABLE else "degraded",
        "timestamp": datetime.now().isoformat(),
        "core_system": AURA_CORE_AVAILABLE
    }

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    print("🧠 AURA Intelligence - Real System API")
    print("=" * 50)
    print("🚀 Using YOUR actual AURA system components")
    print(f"✅ Core system available: {AURA_CORE_AVAILABLE}")
    print("=" * 50)
    print()
    print("📡 Endpoints:")
    print("  - System Status: http://localhost:8080/status")
    print("  - Unified Analysis: http://localhost:8080/analyze/unified")
    print("  - Demo System: http://localhost:8080/demo/{demo_name}")
    print("  - System Info: http://localhost:8080/system/info")
    print()
    
    if AURA_CORE_AVAILABLE:
        print("🎯 Test unified analysis:")
        print('curl -X POST http://localhost:8080/analyze/unified -H "Content-Type: application/json" -d \'{"data":{"input":"test"}}\'')
        print("🎯 Test LNN demo:")
        print("curl http://localhost:8080/demo/lnn_demo")
    else:
        print("⚠️  Running in degraded mode - fix PYTHONPATH and imports to enable full functionality")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)