"""CoRaL API Routes - Expose your 54K items/sec CoRaL system"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from ..coral.best_coral import get_best_coral

router = APIRouter(prefix="/coral", tags=["CoRaL"])
coral_system = get_best_coral()

class CommunicationRequest(BaseModel):
    contexts: List[Dict[str, Any]]

@router.post("/communicate")
async def communicate(request: CommunicationRequest):
    """Execute CoRaL communication round"""
    try:
        result = await coral_system.communicate(request.contexts)
        return {
            "communication_result": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_coral_stats():
    """Get CoRaL system statistics"""
    try:
        return coral_system.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components")
async def get_components():
    """Get CoRaL component assignment"""
    try:
        return {
            "information_agents": len(coral_system.ia_ids),
            "control_agents": len(coral_system.ca_ids),
            "total_components": len(coral_system.ia_ids) + len(coral_system.ca_ids),
            "ia_sample": coral_system.ia_ids[:5],
            "ca_sample": coral_system.ca_ids[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))