#!/usr/bin/env python3
"""
🚀 ENHANCED WORKING AURA API
Your existing working API + Real advanced features (MoE, Spiking GNN, Multimodal)
"""
import sys
import os
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

# Import WORKING components from AURA core
from aura_intelligence.core.unified_system import UnifiedSystem
from aura_intelligence.unified_brain import UnifiedAURABrain, UnifiedConfig
from aura_intelligence.lnn.core import LiquidNeuralNetwork
from aura_intelligence.consciousness.global_workspace import GlobalWorkspace
from aura_intelligence.memory.shape_memory_v2_prod import ShapeMemoryV2

# Import REAL advanced features
from aura_intelligence.real_components.real_moe_router import get_real_moe_router
from aura_intelligence.real_components.real_spiking_gnn import get_spiking_coordinator
from aura_intelligence.real_components.real_multimodal import get_multimodal_processor

# API Models
class ProcessRequest(BaseModel):
    data: List[float]
    task_type: str = "neural"

class MoERequest(BaseModel):
    task_data: Dict[str, Any]
    complexity: float = 0.5
    priority: float = 0.5

class SpikingRequest(BaseModel):
    task_data: Dict[str, Any]
    energy_budget: float = 1000.0  # pJ

class MultiModalRequest(BaseModel):
    text: str = None
    image_data: List[float] = None
    task_type: str = "fusion"

app = FastAPI(
    title="Enhanced AURA Intelligence API",
    description="Working API + Real MoE, Spiking GNN, Multimodal",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
unified_system = None
unified_brain = None
lnn_model = None
consciousness = None
memory_system = None

# Real advanced features
moe_router = None
spiking_coordinator = None
multimodal_processor = None

@app.on_event("startup")
async def startup():
    """Initialize all AURA components"""
    global unified_system, unified_brain, lnn_model, consciousness, memory_system
    global moe_router, spiking_coordinator, multimodal_processor
    
    print("🧠 Initializing Enhanced AURA Intelligence...")
    
    # Initialize working components
    try:
        unified_system = UnifiedSystem()
        print("✅ UnifiedSystem initialized")
        
        config = UnifiedConfig()
        unified_brain = UnifiedAURABrain(config)
        print("✅ UnifiedAURABrain initialized")
        
        lnn_model = LiquidNeuralNetwork(
            input_size=128,
            hidden_size=256,
            output_size=64,
            num_layers=3
        )
        print("✅ LiquidNeuralNetwork initialized")
        
        consciousness = GlobalWorkspace()
        print("✅ GlobalWorkspace initialized")
        
        memory_system = ShapeMemoryV2()
        print("✅ ShapeMemoryV2 initialized")
        
    except Exception as e:
        print(f"⚠️  Working components error: {e}")
    
    # Initialize real advanced features
    try:
        moe_router = get_real_moe_router()
        print("✅ Real MoE Router initialized")
        
        spiking_coordinator = get_spiking_coordinator()
        print("✅ Real Spiking GNN initialized")
        
        multimodal_processor = get_multimodal_processor()
        print("✅ Real Multimodal Processor initialized")
        
    except Exception as e:
        print(f"⚠️  Advanced features error: {e}")
    
    print("🎉 Enhanced AURA system ready!")

@app.get("/")
async def root():
    """Root endpoint with enhanced status"""
    return {
        "service": "Enhanced AURA Intelligence API",
        "version": "3.0.0",
        "status": "enhanced",
        "working_components": {
            "unified_system": unified_system is not None,
            "unified_brain": unified_brain is not None,
            "lnn_model": lnn_model is not None,
            "consciousness": consciousness is not None,
            "memory_system": memory_system is not None
        },
        "advanced_features": {
            "moe_router": moe_router is not None,
            "spiking_gnn": spiking_coordinator is not None,
            "multimodal": multimodal_processor is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Enhanced health check"""
    working_count = sum([
        unified_system is not None,
        unified_brain is not None,
        lnn_model is not None,
        consciousness is not None,
        memory_system is not None
    ])
    
    advanced_count = sum([
        moe_router is not None,
        spiking_coordinator is not None,
        multimodal_processor is not None
    ])
    
    return {
        "status": "healthy" if working_count >= 3 else "degraded",
        "working_components": f"{working_count}/5",
        "advanced_features": f"{advanced_count}/3",
        "total_capabilities": working_count + advanced_count,
        "enhancement_level": "production" if advanced_count == 3 else "basic"
    }

# Original working endpoints
@app.post("/neural/process")
async def neural_process(request: ProcessRequest):
    """Process through LNN (original working endpoint)"""
    if not lnn_model:
        raise HTTPException(status_code=503, detail="LNN model not available")
    
    try:
        input_tensor = torch.tensor(request.data, dtype=torch.float32)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        if input_tensor.shape[-1] != 128:
            if input_tensor.shape[-1] < 128:
                padding = torch.zeros(input_tensor.shape[0], 128 - input_tensor.shape[-1])
                input_tensor = torch.cat([input_tensor, padding], dim=1)
            else:
                input_tensor = input_tensor[:, :128]
        
        with torch.no_grad():
            output = lnn_model(input_tensor)
        
        return {
            "success": True,
            "output": output.tolist(),
            "processing_method": "liquid_neural_network"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural processing failed: {str(e)}")

# NEW: Real MoE endpoint
@app.post("/moe/route")
async def moe_route(request: MoERequest):
    """Route task through real MoE system"""
    if not moe_router:
        raise HTTPException(status_code=503, detail="MoE router not available")
    
    try:
        task_data = request.task_data
        task_data['complexity'] = request.complexity
        task_data['priority'] = request.priority
        
        result = await moe_router.route(task_data)
        stats = moe_router.get_stats()
        
        return {
            "success": True,
            "moe_result": result,
            "router_stats": stats,
            "routing_method": "switch_transformer"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MoE routing failed: {str(e)}")

# NEW: Real Spiking GNN endpoint
@app.post("/spiking/process")
async def spiking_process(request: SpikingRequest):
    """Process through real spiking GNN"""
    if not spiking_coordinator:
        raise HTTPException(status_code=503, detail="Spiking coordinator not available")
    
    try:
        result = await spiking_coordinator.process_with_spiking(request.task_data)
        stats = spiking_coordinator.get_stats()
        
        return {
            "success": True,
            "spiking_result": result,
            "neuromorphic_stats": stats,
            "energy_budget": request.energy_budget
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spiking processing failed: {str(e)}")

# NEW: Real Multimodal endpoint
@app.post("/multimodal/process")
async def multimodal_process(request: MultiModalRequest):
    """Process through real multimodal system"""
    if not multimodal_processor:
        raise HTTPException(status_code=503, detail="Multimodal processor not available")
    
    try:
        inputs = {}
        if request.text:
            inputs['text'] = request.text
        if request.image_data:
            inputs['image'] = torch.tensor(request.image_data).view(1, 3, 224, 224)
        
        result = await multimodal_processor.process_multimodal(inputs)
        stats = multimodal_processor.get_stats()
        
        return {
            "success": True,
            "multimodal_result": result,
            "processor_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal processing failed: {str(e)}")

@app.get("/system/enhanced_status")
async def enhanced_status():
    """Get detailed enhanced system status"""
    status = {
        "system": "Enhanced AURA Intelligence",
        "version": "3.0.0",
        "enhancement_level": "production",
        "components": {}
    }
    
    # Working components status
    if lnn_model:
        status["components"]["lnn_model"] = {
            "status": "working",
            "parameters": sum(p.numel() for p in lnn_model.parameters()),
            "type": "liquid_neural_network"
        }
    
    # Advanced features status
    if moe_router:
        moe_stats = moe_router.get_stats()
        status["components"]["moe_router"] = {
            "status": "working",
            "type": "switch_transformer",
            "stats": moe_stats
        }
    
    if spiking_coordinator:
        spiking_stats = spiking_coordinator.get_stats()
        status["components"]["spiking_gnn"] = {
            "status": "working",
            "type": "neuromorphic",
            "stats": spiking_stats
        }
    
    if multimodal_processor:
        multimodal_stats = multimodal_processor.get_stats()
        status["components"]["multimodal"] = {
            "status": "working",
            "type": "clip_based",
            "stats": multimodal_stats
        }
    
    return status

def main():
    """Run the enhanced AURA API"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                🧠 Enhanced AURA Intelligence API             ║
    ║                      VERSION 3.0.0                          ║
    ║                                                              ║
    ║  Working Components + Real Advanced Features:               ║
    ║  • MoE Router (Switch Transformer)                          ║
    ║  • Spiking GNN (Neuromorphic)                               ║
    ║  • Multimodal (CLIP-based)                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting Enhanced AURA Intelligence API...")
    print("📍 Server: http://localhost:8081")
    print("📚 API docs: http://localhost:8081/docs")
    print("🔍 Health: http://localhost:8081/health")
    print("🧠 MoE: http://localhost:8081/moe/route")
    print("⚡ Spiking: http://localhost:8081/spiking/process")
    print("🎭 Multimodal: http://localhost:8081/multimodal/process")
    
    uvicorn.run(
        "enhanced_working_api:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()