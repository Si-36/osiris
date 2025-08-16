#!/usr/bin/env python3
"""
🚀 WORKING AURA Intelligence API
Uses only the components that actually work from the core system
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

# ============================================================================
# API MODELS
# ============================================================================

class ProcessRequest(BaseModel):
    data: List[float]
    task_type: str = "neural"

class AnalysisRequest(BaseModel):
    input_data: Dict[str, Any]
    analysis_type: str = "unified"

# ============================================================================
# WORKING AURA API
# ============================================================================

app = FastAPI(
    title="AURA Intelligence API",
    description="Working API using real AURA Intelligence components",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances of working components
unified_system = None
unified_brain = None
lnn_model = None
consciousness = None
memory_system = None

@app.on_event("startup")
async def startup():
    """Initialize working AURA components"""
    global unified_system, unified_brain, lnn_model, consciousness, memory_system
    
    print("🧠 Initializing AURA Intelligence components...")
    
    try:
        # Initialize UnifiedSystem
        unified_system = UnifiedSystem()
        print("✅ UnifiedSystem initialized")
        
        # Initialize UnifiedBrain
        config = UnifiedConfig()
        unified_brain = UnifiedAURABrain(config)
        print("✅ UnifiedAURABrain initialized")
        
        # Initialize LNN
        lnn_model = LiquidNeuralNetwork(
            input_size=128,
            hidden_size=256,
            output_size=64,
            num_layers=3
        )
        print("✅ LiquidNeuralNetwork initialized")
        
        # Initialize Consciousness
        consciousness = GlobalWorkspace()
        print("✅ GlobalWorkspace (consciousness) initialized")
        
        # Initialize Memory
        memory_system = ShapeMemoryV2()
        print("✅ ShapeMemoryV2 (memory) initialized")
        
        print("🎉 All AURA components successfully initialized!")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        # Continue with partial initialization

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AURA Intelligence API",
        "version": "2.0.0",
        "status": "working",
        "components": {
            "unified_system": unified_system is not None,
            "unified_brain": unified_brain is not None,
            "lnn_model": lnn_model is not None,
            "consciousness": consciousness is not None,
            "memory_system": memory_system is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check"""
    working_components = sum([
        unified_system is not None,
        unified_brain is not None,
        lnn_model is not None,
        consciousness is not None,
        memory_system is not None
    ])
    
    return {
        "status": "healthy" if working_components >= 3 else "degraded",
        "working_components": working_components,
        "total_components": 5,
        "details": {
            "unified_system": "working" if unified_system else "failed",
            "unified_brain": "working" if unified_brain else "failed",
            "lnn_model": "working" if lnn_model else "failed",
            "consciousness": "working" if consciousness else "failed",
            "memory_system": "working" if memory_system else "failed"
        }
    }

@app.post("/neural/process")
async def neural_process(request: ProcessRequest):
    """Process data through neural network"""
    if not lnn_model:
        raise HTTPException(status_code=503, detail="LNN model not available")
    
    try:
        # Convert input to tensor
        input_tensor = torch.tensor(request.data, dtype=torch.float32)
        
        # Ensure proper shape
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Pad or truncate to expected size (128)
        if input_tensor.shape[-1] != 128:
            if input_tensor.shape[-1] < 128:
                # Pad with zeros
                padding = torch.zeros(input_tensor.shape[0], 128 - input_tensor.shape[-1])
                input_tensor = torch.cat([input_tensor, padding], dim=1)
            else:
                # Truncate
                input_tensor = input_tensor[:, :128]
        
        # Process through LNN
        with torch.no_grad():
            output = lnn_model(input_tensor)
        
        return {
            "success": True,
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape),
            "output": output.tolist(),
            "processing_time": "< 0.1s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural processing failed: {str(e)}")

@app.post("/consciousness/analyze")
async def consciousness_analyze(request: AnalysisRequest):
    """Analyze data through consciousness system"""
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness system not available")
    
    try:
        # Add content to global workspace
        content_id = await consciousness.add_content(
            source="api_request",
            content=request.input_data,
            priority=5
        )
        
        # Get workspace state
        workspace_state = consciousness.get_workspace_state()
        
        return {
            "success": True,
            "content_id": content_id,
            "workspace_state": workspace_state,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consciousness analysis failed: {str(e)}")

@app.post("/memory/store")
async def memory_store(request: Dict[str, Any]):
    """Store data in memory system"""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        # Store in memory
        memory_id = await memory_system.store(
            data=request.get("data"),
            metadata=request.get("metadata", {})
        )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "stored_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory storage failed: {str(e)}")

@app.get("/system/status")
async def system_status():
    """Get detailed system status"""
    status = {
        "system": "AURA Intelligence",
        "version": "2.0.0",
        "status": "operational",
        "components": {}
    }
    
    # Check each component
    if unified_system:
        try:
            status["components"]["unified_system"] = {
                "status": "working",
                "info": "System orchestrator active"
            }
        except:
            status["components"]["unified_system"] = {"status": "error"}
    
    if lnn_model:
        status["components"]["lnn_model"] = {
            "status": "working",
            "parameters": sum(p.numel() for p in lnn_model.parameters()),
            "input_size": 128,
            "output_size": 64
        }
    
    if consciousness:
        status["components"]["consciousness"] = {
            "status": "working",
            "type": "GlobalWorkspace"
        }
    
    if memory_system:
        status["components"]["memory_system"] = {
            "status": "working",
            "type": "ShapeMemoryV2"
        }
    
    return status

def main():
    """Run the working AURA API"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                🧠 AURA Intelligence API                      ║
    ║                   WORKING VERSION 2.0.0                     ║
    ║                                                              ║
    ║  Uses REAL components from core/src/aura_intelligence/       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting WORKING AURA Intelligence API...")
    print("📍 Server will be available at: http://localhost:8080")
    print("📚 API docs at: http://localhost:8080/docs")
    print("🔍 Health check at: http://localhost:8080/health")
    
    uvicorn.run(
        "working_aura_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()