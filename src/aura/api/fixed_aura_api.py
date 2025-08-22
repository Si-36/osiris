#!/usr/bin/env python3
"""
🔧 FIXED AURA Intelligence API
=============================

Working API using the REAL AURA Intelligence components with CORRECT APIs.
"""

import sys
from pathlib import Path

# Add core to path
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import uvicorn

from aura_intelligence.core.unified_system import UnifiedSystem
from aura_intelligence.lnn.core import LiquidNeuralNetwork, LiquidConfig
from aura_intelligence.consciousness.global_workspace import GlobalWorkspace
from aura_intelligence.memory.shape_memory_v2_prod import ShapeMemoryV2

app = FastAPI(title="FIXED AURA Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize working components
working_components = {}

@app.on_event("startup")
async def startup():
    global working_components
    
    print("🔧 Initializing FIXED AURA Intelligence components...")
    
    try:
        working_components['unified_system'] = UnifiedSystem()
        print("✅ UnifiedSystem initialized")
    except Exception as e:
        print(f"❌ UnifiedSystem: {e}")
    
    try:
        config = LiquidConfig(hidden_sizes=[64, 32])
        working_components['lnn'] = LiquidNeuralNetwork(input_size=128, output_size=64, config=config)
        print("✅ LNN initialized")
    except Exception as e:
        print(f"❌ LNN: {e}")
    
    try:
        working_components['consciousness'] = GlobalWorkspace()
        print("✅ Consciousness initialized")
    except Exception as e:
        print(f"❌ Consciousness: {e}")
    
    try:
        working_components['memory'] = ShapeMemoryV2()
        print("✅ Memory initialized")
    except Exception as e:
        print(f"❌ Memory: {e}")
    
    print(f"🎉 {len(working_components)} components successfully initialized!")

class ProcessRequest(BaseModel):
    data: List[float]
    task_type: str = "neural"

@app.get("/")
def root():
    return {
        "service": "FIXED AURA Intelligence API",
        "version": "2.0.0",
        "status": "working",
        "components": {name: True for name in working_components.keys()},
        "total_components": len(working_components)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "working_components": len(working_components),
        "components": list(working_components.keys())
    }

@app.post("/neural/process")
def neural_process(request: ProcessRequest):
    if 'lnn' not in working_components:
        return {"error": "LNN not available"}
    
    try:
        # Convert input
        data = torch.tensor(request.data, dtype=torch.float32)
        
        # Ensure correct input size (128)
        if len(data) != 128:
            if len(data) < 128:
                data = torch.cat([data, torch.zeros(128 - len(data))])
            else:
                data = data[:128]
        
        # Add batch dimension
        data = data.unsqueeze(0)
        
        # Process through LNN
        with torch.no_grad():
            output = working_components['lnn'](data)
        
        return {
            "success": True,
            "input_shape": list(data.shape),
            "output_shape": list(output.shape),
            "output": output.squeeze().tolist()[:10]
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/system/status")
def system_status():
    return {
        "system": "FIXED AURA Intelligence",
        "version": "2.0.0",
        "components": {
            name: {
                "type": type(component).__name__,
                "status": "working"
            }
            for name, component in working_components.items()
        }
    }

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                🔧 FIXED AURA Intelligence API                ║
    ║                                                              ║
    ║  Uses REAL components with CORRECT APIs                      ║
    ║  No more broken imports or wrong method calls!               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting FIXED AURA Intelligence API...")
    print("📍 http://localhost:8080")
    print("📚 http://localhost:8080/docs")
    
    uvicorn.run(
        "fixed_aura_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    )

if __name__ == "__main__":
    main()
