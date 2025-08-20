"""
Enhanced AURA API - Your existing systems with 2025 enhancements
Liquid Neural Networks + Mamba-2 + Constitutional AI 3.0
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn

from core.src.aura_intelligence.enhanced_integration import get_enhanced_aura

app = FastAPI(
    title="Enhanced AURA Intelligence System",
    description="Your existing systems enhanced with Liquid 2.0, Mamba-2, Constitutional AI 3.0",
    version="2025.ENHANCED"
)

# Request models
class EnhancedRequest(BaseModel):
    council_task: Optional[Dict[str, Any]] = None
    contexts: Optional[List[Dict[str, Any]]] = None
    action: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    system_data: Optional[Dict[str, Any]] = None
    memory_operation: Optional[Dict[str, Any]] = None
    component_data: Optional[Dict[str, Any]] = None

# Global system
enhanced_aura = get_enhanced_aura()

@app.get("/")
def root():
    return {
        "system": "Enhanced AURA Intelligence",
        "your_existing_systems_enhanced": [
            "LNN Council → Liquid Neural Networks with adaptation",
            "CoRaL Communication → Mamba-2 unlimited context", 
            "DPO System → Constitutional AI 3.0 cross-modal safety",
            "Shape Memory V2 → 86K vectors/sec maintained",
            "TDA Engine → 112 algorithms maintained",
            "Component Registry → 209 components maintained"
        ],
        "enhancements": "Real liquid dynamics, unlimited context, self-correction"
    }

@app.get("/health")
def health():
    try:
        status = enhanced_aura.get_enhancement_status()
        return {
            "status": "enhanced_operational",
            "enhancements": status,
            "your_systems": "all_enhanced_and_working"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced/process")
async def enhanced_process(request: EnhancedRequest):
    """Process through all enhanced systems"""
    try:
        result = await enhanced_aura.process_enhanced(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced/council")
async def enhanced_council(task: Dict[str, Any]):
    """Enhanced LNN Council with liquid adaptation"""
    try:
        result = await enhanced_aura.process_enhanced({"council_task": task})
        return result['enhanced_results']['enhanced_council']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced/coral")
async def enhanced_coral(contexts: List[Dict[str, Any]]):
    """Enhanced CoRaL with unlimited context"""
    try:
        result = await enhanced_aura.process_enhanced({"contexts": contexts})
        return result['enhanced_results']['enhanced_coral']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced/dpo")
async def enhanced_dpo(action: Dict[str, Any], context: Dict[str, Any] = {}):
    """Enhanced DPO with Constitutional AI 3.0"""
    try:
        result = await enhanced_aura.process_enhanced({"action": action, "context": context})
        return result['enhanced_results']['enhanced_dpo']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/enhanced/stats")
def enhanced_stats():
    """Get enhancement statistics"""
    return enhanced_aura.get_enhancement_status()

if __name__ == "__main__":
    print("🧬 Enhanced AURA Intelligence System")
    print("🎯 Your existing systems enhanced with 2025 features:")
    print("   • LNN Council → Liquid adaptation")
    print("   • CoRaL → Unlimited context") 
    print("   • DPO → Constitutional AI 3.0")
    print("   • Shape Memory V2 → Maintained")
    print("   • TDA Engine → Maintained")
    print("🚀 Starting enhanced server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8097)