"""TDA API Routes - Expose your 112-algorithm TDA engine"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ..tda.unified_engine_2025 import get_unified_tda_engine

router = APIRouter(prefix="/tda", tags=["TDA"])
tda_engine = get_unified_tda_engine()

class SystemAnalysisRequest(BaseModel):
    system_id: str
    agents: list
    communications: list = []
    metrics: Dict[str, Any] = {}

@router.post("/analyze")
async def analyze_system(request: SystemAnalysisRequest):
    """Analyze agentic system topology and health"""
    try:
        system_data = request.dict()
        health = await tda_engine.analyze_agentic_system(system_data)
        
        return {
            "system_id": health.system_id,
            "topology_score": health.topology_score,
            "risk_level": health.risk_level,
            "bottlenecks": health.bottlenecks,
            "recommendations": health.recommendations,
            "persistence_diagram": health.persistence_diagram.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{system_id}")
async def get_recommendations(system_id: str):
    """Get detailed recommendations for system"""
    try:
        return await tda_engine.get_system_recommendations(system_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_dashboard():
    """Get TDA dashboard data"""
    try:
        return await tda_engine.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))