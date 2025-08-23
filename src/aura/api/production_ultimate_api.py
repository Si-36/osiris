"""
Production Ultimate API 2025
Real 200+ components with MoE routing and data flow
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import time
import uvicorn

from core.src.aura_intelligence.components.real_registry import get_real_registry
from core.src.aura_intelligence.moe.mixture_of_experts import get_moe_system
from core.src.aura_intelligence.coral.best_coral import get_best_coral


app = FastAPI(
    title="AURA Intelligence Production API 2025",
    description="200+ Real Components with MoE Routing",
    version="2025.8.16"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    data: Dict[str, Any]
    component_ids: Optional[List[str]] = None
    use_moe: bool = True


@app.get("/")
async def root():
    registry = get_real_registry()
    moe = get_moe_system()
    coral = get_best_coral()
    
    stats = registry.get_component_stats()
    moe_stats = moe.get_moe_stats()
    coral_stats = coral.get_coral_stats()
    
    return {
        "message": "AURA Intelligence Production System 2025",
        "version": "2025.8.16",
        "components": {
            "total": stats['total_components'],
            "active": stats['active_components'],
            "types": stats['type_distribution']
        },
        "moe_routing": {
            "experts_tracked": moe_stats['tracked_experts'],
            "avg_success_rate": moe_stats['avg_expert_success_rate']
        },
        "capabilities": [
            "203 Real Components",
            "Mixture of Experts Routing", 
            "CoRaL Emergent Communication",
            "Causal Influence Learning",
            "Real Data Processing",
            "Component Performance Tracking",
            "Intelligent Load Balancing"
        ],
        "coral_system": {
            "information_agents": coral_stats['system_overview']['information_agents'],
            "control_agents": coral_stats['system_overview']['control_agents'],
            "avg_causal_influence": coral_stats['communication_stats']['avg_causal_influence']
        },
        "endpoints": {
            "process": "/process - Main processing with MoE + CoRaL",
            "coral": "/coral/communicate - CoRaL communication round",
            "component": "/component/{id} - Single component processing",
            "pipeline": "/pipeline - Multi-component pipeline",
            "health": "/health - System health check",
            "stats": "/stats - Detailed statistics"
        }
    }


@app.post("/process")
async def process_request(request: ProcessRequest):
    """Main processing endpoint with MoE routing + CoRaL communication"""
    start_time = time.time()
    
    try:
        # Step 1: CoRaL communication round
        coral = get_coral_system()
        coral_result = await coral.communication_round(request.data)
        
        # Step 2: MoE or direct processing
        if request.use_moe:
            # Use Mixture of Experts routing
            moe = get_moe_system()
            moe_result = await moe.process_with_experts(request.data)
            result = moe_result
        else:
            # Direct component processing
            registry = get_real_registry()
            if request.component_ids:
                result = await registry.process_pipeline(request.data, request.component_ids)
            else:
                # Default pipeline
                default_components = [
                    "neural_000_lnn_processor",
                    "memory_000_redis_store", 
                    "agent_000_council_agent",
                    "tda_000_persistence_computer"
                ]
                result = await registry.process_pipeline(request.data, default_components)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "result": result,
            "coral_communication": coral_result,
            "processing_time_ms": processing_time * 1000,
            "routing_method": "moe+coral" if request.use_moe else "direct+coral",
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000
        }


@app.post("/component/{component_id}")
async def process_single_component(component_id: str, data: Dict[str, Any]):
    """Process data through single component"""
    try:
        registry = get_real_registry()
        result = await registry.process_data(component_id, data)
        
        return {
            "success": True,
            "component_id": component_id,
            "result": result,
            "component_info": {
                "type": registry.components[component_id].type.value,
                "status": registry.components[component_id].status,
                "data_processed": registry.components[component_id].data_processed
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Component processing failed: {str(e)}")


@app.post("/coral/communicate")
async def coral_communication_round(context: Dict[str, Any]):
    """Execute CoRaL communication round"""
    try:
        coral = get_coral_system()
        result = await coral.communication_round(context)
        
        return {
            "success": True,
            "coral_result": result,
            "causal_influence": result['average_causal_influence'],
            "communication_efficiency": result['communication_efficiency'],
            "messages_generated": result['messages_generated'],
            "decisions_made": result['decisions_made']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CoRaL communication failed: {str(e)}")


@app.post("/pipeline")
async def process_pipeline(data: Dict[str, Any], component_ids: List[str]):
    """Process data through component pipeline"""
    try:
        registry = get_real_registry()
        result = await registry.process_pipeline(data, component_ids)
        
        return {
            "success": True,
            "pipeline_result": result,
            "components_used": len(component_ids),
            "pipeline_components": component_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """System health check"""
    try:
        registry = get_real_registry()
        moe = get_moe_system()
        
        stats = registry.get_component_stats()
        moe_stats = moe.get_moe_stats()
        
        health_score = (
            0.6 * stats['health_score'] +
            0.4 * min(1.0, moe_stats['avg_expert_success_rate'])
        )
        
        return {
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.6 else "unhealthy",
            "health_score": health_score,
            "timestamp": time.time(),
            "components": {
                "total": stats['total_components'],
                "active": stats['active_components'],
                "health_score": stats['health_score']
            },
            "moe_system": {
                "experts_tracked": moe_stats['tracked_experts'],
                "success_rate": moe_stats['avg_expert_success_rate'],
                "avg_processing_time_ms": moe_stats['avg_processing_time_ms']
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/stats")
async def get_detailed_stats():
    """Get detailed system statistics"""
    try:
        registry = get_real_registry()
        moe = get_moe_system()
        coral = get_coral_system()
        
        component_stats = registry.get_component_stats()
        moe_stats = moe.get_moe_stats()
        coral_stats = coral.get_coral_stats()
        top_performers = registry.get_top_performers(10)
        
        return {
            "system_overview": {
                "total_components": component_stats['total_components'],
                "active_components": component_stats['active_components'],
                "type_distribution": component_stats['type_distribution'],
                "avg_processing_time_ms": component_stats['avg_processing_time_ms'],
                "total_data_processed": component_stats['total_data_processed']
            },
            "moe_system": moe_stats,
            "coral_system": coral_stats,
            "top_performers": [
                {
                    "component_id": comp.id,
                    "type": comp.type.value,
                    "data_processed": comp.data_processed,
                    "avg_processing_time_ms": comp.processing_time * 1000
                }
                for comp in top_performers
            ],
            "performance_metrics": {
                "system_health_score": component_stats['health_score'],
                "moe_routing_efficiency": moe_stats.get('routing_efficiency', 0.0),
                "expert_success_rate": moe_stats['avg_expert_success_rate'],
                "coral_causal_influence": coral_stats['communication_stats']['avg_causal_influence'],
                "coral_communication_efficiency": coral_stats['performance_metrics']['communication_efficiency']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats query failed: {str(e)}")


@app.get("/components")
async def list_components():
    """List all available components"""
    try:
        registry = get_real_registry()
        
        components_by_type = {}
        for component in registry.components.values():
            comp_type = component.type.value
            if comp_type not in components_by_type:
                components_by_type[comp_type] = []
            
            components_by_type[comp_type].append({
                "id": component.id,
                "status": component.status,
                "data_processed": component.data_processed,
                "processing_time_ms": component.processing_time * 1000
            })
        
        return {
            "total_components": len(registry.components),
            "components_by_type": components_by_type,
            "type_counts": {
                comp_type: len(comps) 
                for comp_type, comps in components_by_type.items()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Component listing failed: {str(e)}")


@app.get("/benchmark")
async def run_benchmark():
    """Run system benchmark"""
    try:
        registry = get_real_registry()
        moe = get_moe_system()
        
        # Test data
        test_requests = [
            {"type": "neural", "data": [1, 2, 3, 4, 5]},
            {"type": "memory", "operation": "store", "key": "test", "value": "benchmark"},
            {"type": "agent", "task": "coordination", "priority": "high"},
            {"type": "tda", "analysis": "topology", "data_points": 100}
        ]
        
        benchmark_results = []
        total_start = time.time()
        
        for i, test_request in enumerate(test_requests):
            start_time = time.time()
            
            # Test MoE routing
            moe_result = await moe.process_with_experts(test_request)
            
            processing_time = time.time() - start_time
            
            benchmark_results.append({
                "test_id": i,
                "request_type": test_request.get("type", "unknown"),
                "processing_time_ms": processing_time * 1000,
                "experts_used": moe_result.get("experts_used", 0),
                "routing_efficiency": moe_result.get("routing_efficiency", 0.0),
                "success": "error" not in moe_result
            })
        
        total_time = time.time() - total_start
        
        return {
            "benchmark_complete": True,
            "total_time_ms": total_time * 1000,
            "tests_run": len(test_requests),
            "results": benchmark_results,
            "performance_summary": {
                "avg_processing_time_ms": sum(r["processing_time_ms"] for r in benchmark_results) / len(benchmark_results),
                "success_rate": sum(1 for r in benchmark_results if r["success"]) / len(benchmark_results),
                "avg_experts_per_request": sum(r["experts_used"] for r in benchmark_results) / len(benchmark_results)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting AURA Intelligence Production API 2025...")
    print("📊 200+ Real Components with MoE Routing")
    print("🌐 Server will be available at: http://localhost:8091")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8091,
        log_level="info"
    )