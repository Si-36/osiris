"""
Orchestration - Clean Implementation
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Pipeline:
    id: str
    steps: List[Dict[str, Any]]
    status: PipelineStatus = PipelineStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

class UnifiedOrchestrationEngine:
    """Orchestration engine without LangGraph dependencies"""
    
    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.memory = None
        self.next_id = 0
        
    def set_memory(self, memory):
        """Set memory system for state persistence"""
        self.memory = memory
        
    async def create_pipeline(self, steps: List[Dict[str, Any]]) -> str:
        """Create a new pipeline"""
        pipeline_id = f"pipeline_{self.next_id}"
        self.next_id += 1
        
        pipeline = Pipeline(
            id=pipeline_id,
            steps=steps
        )
        
        self.pipelines[pipeline_id] = pipeline
        
        # Store in memory if available
        if self.memory:
            await self.memory.store({
                "type": "pipeline_created",
                "pipeline_id": pipeline_id,
                "steps": steps
            })
            
        return pipeline_id
    
    async def execute(self, data: Any) -> Dict[str, Any]:
        """Execute a pipeline or single operation"""
        # If data is a routing decision, create simple pipeline
        if hasattr(data, 'provider'):
            pipeline_id = await self.create_pipeline([
                {"type": "route", "provider": data.provider, "model": data.model},
                {"type": "process", "input": data}
            ])
        else:
            # Create default pipeline
            pipeline_id = await self.create_pipeline([
                {"type": "process", "input": data}
            ])
            
        return await self.run_pipeline(pipeline_id)
    
    async def run_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Run a pipeline"""
        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}
            
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.RUNNING
        
        try:
            result = None
            for step in pipeline.steps:
                result = await self._execute_step(step, result)
                
                # Store intermediate results
                if self.memory:
                    await self.memory.store({
                        "type": "pipeline_step",
                        "pipeline_id": pipeline_id,
                        "step": step,
                        "result": result
                    })
                    
            pipeline.status = PipelineStatus.COMPLETED
            pipeline.result = result
            
            return {
                "pipeline_id": pipeline_id,
                "status": "completed",
                "result": result
            }
            
        except Exception as e:
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = str(e)
            
            return {
                "pipeline_id": pipeline_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_step(self, step: Dict[str, Any], previous_result: Any) -> Any:
        """Execute a single step"""
        step_type = step.get("type", "process")
        
        if step_type == "route":
            # Routing step
            return {
                "routed_to": step.get("provider"),
                "model": step.get("model")
            }
            
        elif step_type == "process":
            # Processing step
            input_data = step.get("input", previous_result)
            
            # Simulate processing
            await asyncio.sleep(0.1)
            
            return {
                "processed": True,
                "input": input_data,
                "timestamp": time.time()
            }
            
        elif step_type == "parallel":
            # Parallel execution
            tasks = []
            for sub_step in step.get("steps", []):
                tasks.append(self._execute_step(sub_step, previous_result))
                
            results = await asyncio.gather(*tasks)
            return {"parallel_results": results}
            
        else:
            # Unknown step type
            return {"step_type": step_type, "input": previous_result}
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status"""
        if pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            return {
                "id": pipeline.id,
                "status": pipeline.status.value,
                "result": pipeline.result,
                "error": pipeline.error
            }
        return None