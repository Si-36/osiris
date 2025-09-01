
# TDA Memory Integration Wrapper
# ==============================

from typing import Dict, Any, Optional
import asyncio

class TDAMemoryWrapper:
    """Wrapper that stores TDA analysis in Memory"""
    
    def __init__(self, tda_analyzer, memory_system=None):
        self.tda = tda_analyzer
        self.memory = memory_system
        
    async def analyze_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]):
        """Analyze and store in memory"""
        # Run original analysis
        result = await self.tda.analyze_workflow(workflow_id, workflow_data)
        
        # Store in memory if available
        if self.memory:
            try:
                await self.memory.store(
                    content={
                        "workflow_id": workflow_id,
                        "analysis": result.to_dict(),
                        "bottleneck_score": result.bottleneck_score,
                        "has_cycles": result.has_cycles
                    },
                    memory_type="TOPOLOGICAL",
                    workflow_data=workflow_data,
                    metadata={"component": "tda", "action": "analysis"}
                )
            except Exception as e:
                print(f"Failed to store TDA analysis: {e}")
                
        return result
