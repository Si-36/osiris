"""DPO API Routes - Expose your Constitutional AI DPO system"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..dpo.preference_optimizer import get_dpo_optimizer

router = APIRouter(prefix="/dpo", tags=["DPO"])
dpo_system = get_dpo_optimizer()

class ActionEvaluationRequest(BaseModel):
    action: Dict[str, Any]
    context: Dict[str, Any] = {}

class PreferencePairRequest(BaseModel):
    preferred_action: Dict[str, Any]
    rejected_action: Dict[str, Any]
    context: Dict[str, Any] = {}

@router.post("/evaluate")
async def evaluate_action(request: ActionEvaluationRequest):
    """Evaluate action using DPO + Constitutional AI"""
    try:
        result = await dpo_system.evaluate_action_preference(request.action, request.context)
        return {
            "evaluation": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preference")
async def add_preference_pair(request: PreferencePairRequest):
    """Add preference pair for training"""
    try:
        dpo_system.collect_preference_pair(
            request.preferred_action,
            request.rejected_action, 
            request.context
        )
        return {"status": "preference_pair_added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_dpo(batch_size: int = 32):
    """Train DPO system"""
    try:
        result = await dpo_system.train_batch(batch_size)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_dpo_stats():
    """Get DPO system statistics"""
    try:
        return dpo_system.get_dpo_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/self-improve")
async def self_improve():
    """Trigger Constitutional AI self-improvement"""
    try:
        result = await dpo_system.self_improve_system()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))