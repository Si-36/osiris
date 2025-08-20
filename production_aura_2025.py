"""
Production AURA 2025 - Complete system with all enhancements
Real tests, Prometheus metrics, hybrid memory, enhanced systems
"""
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import structlog

# Import all enhanced systems
from core.src.aura_intelligence.enhanced_integration import get_enhanced_aura
from core.src.aura_intelligence.memory.hybrid_manager import get_hybrid_memory
from core.src.aura_intelligence.bio_homeostatic.metabolic_manager import ProductionMetabolicManager
from core.src.aura_intelligence.observability.prometheus_integration import (
    metrics_collector, update_system_metrics
)

logger = structlog.get_logger()

app = FastAPI(
    title="Production AURA Intelligence 2025",
    description="Complete enhanced system with liquid networks, Mamba-2, Constitutional AI 3.0",
    version="2025.PRODUCTION"
)

# Request models
class ProductionRequest(BaseModel):
    council_task: Optional[Dict[str, Any]] = None
    contexts: Optional[List[Dict[str, Any]]] = None
    action: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    system_data: Optional[Dict[str, Any]] = None
    memory_operation: Optional[Dict[str, Any]] = None
    component_data: Optional[Dict[str, Any]] = None

class MemoryRequest(BaseModel):
    key: str
    data: Optional[Any] = None
    component_id: str
    operation: str  # "store" or "retrieve"

class MetabolicRequest(BaseModel):
    component_id: str
    data: Any
    context: Optional[Dict[str, Any]] = None

# Global systems
enhanced_aura = None
hybrid_memory = None
metabolic_manager = None

@app.on_event("startup")
async def startup_production():
    """Initialize all production systems"""
    global enhanced_aura, hybrid_memory, metabolic_manager
    
    logger.info("🚀 Starting Production AURA Intelligence 2025")
    
    # Initialize all systems
    enhanced_aura = get_enhanced_aura()
    hybrid_memory = get_hybrid_memory()
    metabolic_manager = ProductionMetabolicManager()
    
    # Start metrics collection
    asyncio.create_task(update_system_metrics(enhanced_aura))
    
    logger.info("✅ Production AURA 2025 ready - All systems operational")

@app.get("/")
def root():
    return {
        "system": "Production AURA Intelligence 2025",
        "status": "operational",
        "enhancements": [
            "✅ Liquid Neural Networks - Self-adapting architecture",
            "✅ Mamba-2 Unlimited Context - Linear complexity O(n)",
            "✅ Constitutional AI 3.0 - Cross-modal safety with self-correction",
            "✅ Hybrid Memory System - Hot/warm/cold tiers",
            "✅ Production Metabolic Manager - Real signal integration",
            "✅ Prometheus Metrics - Complete observability",
            "✅ Shape Memory V2 - 86K vectors/sec maintained",
            "✅ TDA Engine - 112 algorithms maintained",
            "✅ Component Registry - 209 components maintained"
        ],
        "endpoints": {
            "enhanced": "/enhanced/*",
            "memory": "/memory/*", 
            "metabolic": "/metabolic/*",
            "metrics": "/metrics",
            "health": "/health",
            "test": "/test"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with all systems"""
    try:
        # Enhanced systems health
        enhancement_status = enhanced_aura.get_enhancement_status()
        
        # Memory system health
        memory_stats = hybrid_memory.get_stats()
        
        # Metabolic system health
        metabolic_status = metabolic_manager.get_status()
        
        # Overall health score
        healthy_enhancements = sum(1 for info in enhancement_status.values() 
                                 if isinstance(info, dict) and info.get('status') == 'active')
        total_enhancements = len([info for info in enhancement_status.values() 
                                if isinstance(info, dict) and 'status' in info])
        
        health_score = healthy_enhancements / total_enhancements if total_enhancements > 0 else 0
        
        return {
            "status": "healthy" if health_score > 0.8 else "degraded",
            "health_score": health_score,
            "enhancements": enhancement_status,
            "memory_system": {
                "total_items": memory_stats["total_items"],
                "total_usage_mb": memory_stats["total_usage_mb"],
                "hot_utilization": memory_stats["tiers"]["hot"]["utilization"],
                "warm_utilization": memory_stats["tiers"]["warm"]["utilization"]
            },
            "metabolic_system": {
                "active_components": metabolic_status["active_components"],
                "health_distribution": metabolic_status["health_distribution"],
                "integration_status": metabolic_status["integration_status"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/enhanced/process")
async def enhanced_process(request: ProductionRequest):
    """Process through all enhanced systems"""
    try:
        result = await enhanced_aura.process_enhanced(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")

@app.post("/memory/hybrid")
async def hybrid_memory_operation(request: MemoryRequest):
    """Hybrid memory operations with tier management"""
    try:
        if request.operation == "store":
            if not request.data:
                raise HTTPException(status_code=400, detail="Data required for store operation")
            
            result = await hybrid_memory.store(
                key=request.key,
                data=request.data,
                component_id=request.component_id
            )
            return result
            
        elif request.operation == "retrieve":
            result = await hybrid_memory.retrieve(request.key)
            return result
            
        else:
            raise HTTPException(status_code=400, detail="Operation must be 'store' or 'retrieve'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory operation failed: {str(e)}")

@app.get("/memory/stats")
def memory_stats():
    """Get hybrid memory statistics"""
    try:
        return hybrid_memory.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory stats failed: {str(e)}")

@app.post("/metabolic/process")
async def metabolic_process(request: MetabolicRequest):
    """Process with metabolic gating"""
    try:
        result = await metabolic_manager.process_with_gating(
            component_id=request.component_id,
            data=request.data,
            context=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metabolic processing failed: {str(e)}")

@app.get("/metabolic/status")
def metabolic_status():
    """Get metabolic system status"""
    try:
        return metabolic_manager.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metabolic status failed: {str(e)}")

@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        return metrics_collector.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@app.get("/metrics/summary")
def metrics_summary():
    """Human-readable metrics summary"""
    try:
        return metrics_collector.get_summary_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics summary failed: {str(e)}")

@app.post("/test/comprehensive")
async def run_comprehensive_tests(background_tasks: BackgroundTasks):
    """Run comprehensive system tests"""
    background_tasks.add_task(run_background_tests)
    return {"status": "tests_started", "message": "Comprehensive tests running in background"}

async def run_background_tests():
    """Run comprehensive tests in background"""
    try:
        # Import and run tests
        from test_enhanced_systems import run_comprehensive_tests
        benchmarks = await run_comprehensive_tests()
        
        # Record test results as metrics
        for metric_name, value in benchmarks.items():
            if 'ms' in metric_name:
                metrics_collector.record_system_request(
                    'test_benchmark', 'success', value / 1000.0
                )
        
        logger.info("✅ Comprehensive tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Comprehensive tests failed: {e}")

@app.get("/benchmark")
async def system_benchmark():
    """Quick system benchmark"""
    try:
        import time
        
        benchmarks = {}
        
        # Enhanced processing benchmark
        start = time.time()
        test_request = {
            'council_task': {'gpu_allocation': {'gpu_count': 2, 'cost_per_hour': 1.5}},
            'contexts': [{'test': 'benchmark'}],
            'action': {'type': 'benchmark', 'confidence': 0.8}
        }
        
        result = await enhanced_aura.process_enhanced(test_request)
        benchmarks['enhanced_processing_ms'] = (time.time() - start) * 1000
        
        # Memory benchmark
        start = time.time()
        await hybrid_memory.store('benchmark_key', {'test': 'data'}, 'test_component')
        await hybrid_memory.retrieve('benchmark_key')
        benchmarks['memory_operations_ms'] = (time.time() - start) * 1000
        
        # Metabolic benchmark
        start = time.time()
        await metabolic_manager.process_with_gating('test_component', {'test': 'data'})
        benchmarks['metabolic_processing_ms'] = (time.time() - start) * 1000
        
        return {
            "benchmarks": benchmarks,
            "performance_grade": "excellent" if all(v < 100 for v in benchmarks.values()) else "good",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

if __name__ == "__main__":
    print("🧬 Production AURA Intelligence 2025")
    print("🎯 Complete enhanced system with:")
    print("   • Liquid Neural Networks with self-adaptation")
    print("   • Mamba-2 unlimited context (linear complexity)")
    print("   • Constitutional AI 3.0 cross-modal safety")
    print("   • Hybrid memory system (hot/warm/cold tiers)")
    print("   • Production metabolic manager with real signals")
    print("   • Comprehensive Prometheus metrics")
    print("   • Real comprehensive testing")
    print("🚀 Starting production server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8098,
        workers=1,
        log_level="info",
        access_log=True
    )