"""Memory API Routes - Expose hybrid memory system"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..memory.hybrid_manager import get_hybrid_memory

router = APIRouter(prefix="/memory", tags=["Memory"])
memory_manager = get_hybrid_memory()

class StoreRequest(BaseModel):
    key: str
    data: Any
    component_id: str

@router.post("/store")
async def store_data(request: StoreRequest):
    """Store data in hybrid memory tiers"""
    try:
        result = await memory_manager.store(request.key, request.data, request.component_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retrieve/{key}")
async def retrieve_data(key: str):
    """Retrieve data from hybrid memory tiers"""
    try:
        result = await memory_manager.retrieve(key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_memory_stats():
    """Get hybrid memory tier statistics"""
    try:
        return memory_manager.get_tier_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_memory():
    """Cleanup old data and optimize tiers"""
    try:
        await memory_manager.cleanup_old_data()
        return {"status": "cleanup_completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))