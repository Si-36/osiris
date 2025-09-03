"""
AURA Unified Workflow Executor - The Core Execution Engine
Professional implementation using LangGraph and 2025 best practices
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
import uuid

from langgraph.graph import StateGraph
from pydantic import ValidationError

# Import our schemas
from ..schemas.aura_execution import (
    AuraTask,
    AuraWorkflowState,
    TaskStatus,
    ExecutionPlan,
    ConsensusDecision,
    ObservationResult
)

# Import real AURA components
from ..memory.unified_cognitive_memory import UnifiedCognitiveMemory
from ..memory.working_memory import WorkingMemory
from ..memory.episodic_memory import EpisodicMemory
from ..memory.semantic_memory import SemanticMemory
from ..memory.consolidation import SleepConsolidation as MemoryConsolidation
from ..memory.routing.hierarchical_router_2025 import HierarchicalMemoryRouter2025
from ..memory.shape_memory_v2 import ShapeAwareMemoryV2
from ..memory.core.causal_tracker import CausalPatternTracker
from ..tools.tool_registry import ToolRegistry, ToolExecutor

logger = structlog.get_logger(__name__)


class UnifiedWorkflowExecutor:
    """
    The single entry point and REAL execution engine for AURA.
    This class orchestrates the entire cognitive workflow using LangGraph.
    """
    
    def __init__(
        self,
        memory: Optional[UnifiedCognitiveMemory] = None,
        agents: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolRegistry] = None,
        consensus: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the executor with real components.
        
        Args:
            memory: The unified cognitive memory system
            agents: Dictionary of agent instances
            tools: Tool registry with all available tools
            consensus: Consensus system for agent agreement
            config: Additional configuration
        """
        self.config = config or {}
        
        # Initialize or use provided memory system
        if memory:
            self.memory = memory
        else:
            # Create real memory system with all components
            self.memory = self._create_memory_system()
        
        # Initialize or use provided agents
        self.agents = agents or {}
        if not self.agents:
            # We'll create agents later when we implement them
            logger.warning("No agents provided, workflow will have limited functionality")
        
        # Initialize or use provided tool registry
        if tools:
            self.tools = tools
        else:
            self.tools = self._create_tool_registry()
        
        # Tool executor for advanced execution patterns
        self.tool_executor = ToolExecutor(self.tools)
        
        # Consensus system (will be implemented)
        self.consensus = consensus
        
        # Performance metrics
        self.metrics = {
            "tasks_executed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
        # Create the LangGraph workflow
        self.workflow = self._create_workflow()
        
        logger.info("✅ UnifiedWorkflowExecutor initialized with real components")
    
    def _create_memory_system(self) -> UnifiedCognitiveMemory:
        """Create a fully configured memory system"""
        logger.info("Creating UnifiedCognitiveMemory with all subsystems...")
        
        # Create memory services
        services = {
            'working_memory': WorkingMemory(),
            'episodic_memory': EpisodicMemory(config=self.config.get('episodic', {})),
            'semantic_memory': SemanticMemory(config=self.config.get('semantic', {})),
            'memory_consolidation': MemoryConsolidation(config=self.config.get('consolidation', {})),
            'hierarchical_router': HierarchicalMemoryRouter2025(config=self.config.get('router', {})),
            'shape_memory': ShapeAwareMemoryV2(config=self.config.get('shape', {})),
            'causal_tracker': CausalPatternTracker(config=self.config.get('causal', {}))
        }
        
        return UnifiedCognitiveMemory(services)
    
    def _create_tool_registry(self) -> ToolRegistry:
        """Create and populate the tool registry"""
        from ..tools.implementations.observation_tool import SystemObservationTool
        from ..tools.tool_registry import ToolMetadata, ToolCategory
        
        registry = ToolRegistry()
        
        # Register the observation tool
        obs_tool = SystemObservationTool(
            topology_adapter=self.memory.shape_mem if hasattr(self.memory, 'shape_mem') else None,
            memory_system=self.memory,
            causal_tracker=self.memory.causal_tracker if hasattr(self.memory, 'causal_tracker') else None
        )
        
        registry.register(
            "SystemObservationTool",
            obs_tool,
            ToolMetadata(
                name="SystemObservationTool",
                category=ToolCategory.OBSERVATION,
                description="Observes system state and computes topological signature",
                output_schema=ObservationResult,
                timeout_seconds=30.0
            )
        )
        
        logger.info("Tool registry created with real tools")
        return registry
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow implementing the 5-step cognitive process.
        This is imported from our workflow definition.
        """
        from .aura_cognitive_workflow import create_aura_workflow
        return create_aura_workflow(self)
    
    async def execute_task(
        self,
        task_description: str,
        environment: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete cognitive task through the workflow.
        
        This is the main entry point that:
        1. Creates a task
        2. Runs it through the workflow
        3. Consolidates learning
        4. Returns results
        
        Args:
            task_description: What the task aims to accomplish
            environment: Environmental context and parameters
            
        Returns:
            The final result from the workflow execution
        """
        logger.info(f"🚀 Starting AURA task: {task_description}")
        
        # Track execution time
        start_time = asyncio.get_event_loop().time()
        
        # Create the task
        task = AuraTask(
            objective=task_description,
            environment=environment or {},
            status=TaskStatus.PENDING
        )
        
        # Create initial workflow state
        initial_state = AuraWorkflowState(
            task=task,
            executor_instance=self  # Pass reference to self for nodes to access
        )
        
        try:
            # Add trace entry
            initial_state.add_trace(f"Task {task.task_id} created")
            
            # Convert to dict for LangGraph (it doesn't work directly with Pydantic)
            state_dict = initial_state.model_dump(exclude={'executor_instance'})
            state_dict['executor_instance'] = self
            
            # Run the workflow
            logger.info(f"Invoking workflow for task {task.task_id}")
            final_state_dict = await self.workflow.ainvoke(state_dict)
            
            # Validate the final state
            final_state = AuraWorkflowState.model_validate(final_state_dict)
            
            # Perform final memory consolidation
            if self.memory:
                logger.info("Consolidating learning from task execution")
                await self.memory.run_offline_consolidation()
            
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(success=True, duration=execution_time)
            
            # Extract and return the result
            result = final_state.task.final_result or {
                "status": "completed",
                "task_id": task.task_id,
                "observations": [obs.model_dump() for obs in final_state.observations],
                "patterns": final_state.patterns,
                "execution_time": execution_time,
                "trace": final_state.execution_trace[-10:]  # Last 10 trace entries
            }
            
            logger.info(f"✅ Task {task.task_id} completed successfully in {execution_time:.2f}s")
            return result
            
        except ValidationError as e:
            logger.error(f"Validation error in workflow: {e}")
            self._update_metrics(success=False)
            return {
                "status": "error",
                "task_id": task.task_id,
                "error": "Validation error in workflow state",
                "details": str(e)
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            self._update_metrics(success=False)
            return {
                "status": "error",
                "task_id": task.task_id,
                "error": str(e),
                "type": type(e).__name__
            }
    
    async def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks, optionally in parallel.
        
        Args:
            tasks: List of task definitions
            parallel: Whether to execute in parallel
            
        Returns:
            List of results
        """
        if parallel:
            # Execute all tasks concurrently
            coroutines = [
                self.execute_task(
                    task["objective"],
                    task.get("environment")
                )
                for task in tasks
            ]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Process results
            processed = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed.append({
                        "status": "error",
                        "task": tasks[i]["objective"],
                        "error": str(result)
                    })
                else:
                    processed.append(result)
            
            return processed
        else:
            # Execute tasks sequentially
            results = []
            for task in tasks:
                result = await self.execute_task(
                    task["objective"],
                    task.get("environment")
                )
                results.append(result)
            
            return results
    
    def _update_metrics(self, success: bool, duration: float = 0.0):
        """Update execution metrics"""
        self.metrics["tasks_executed"] += 1
        
        if success:
            self.metrics["tasks_succeeded"] += 1
        else:
            self.metrics["tasks_failed"] += 1
        
        if duration > 0:
            self.metrics["total_execution_time"] += duration
            self.metrics["average_execution_time"] = (
                self.metrics["total_execution_time"] / self.metrics["tasks_executed"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return self.metrics.copy()
    
    async def shutdown(self):
        """Gracefully shutdown the executor"""
        logger.info("Shutting down UnifiedWorkflowExecutor...")
        
        # Save any pending memory consolidation
        if self.memory:
            await self.memory.run_offline_consolidation()
        
        # Clean up resources
        # (Add any cleanup needed for agents, tools, etc.)
        
        logger.info("UnifiedWorkflowExecutor shutdown complete")


class WorkflowMonitor:
    """
    Monitors workflow execution and provides observability.
    This could integrate with Prometheus, Grafana, etc.
    """
    
    def __init__(self, executor: UnifiedWorkflowExecutor):
        self.executor = executor
        self.execution_history: List[Dict[str, Any]] = []
        
    async def monitor_task(self, task_id: str):
        """Monitor a specific task execution"""
        # This would track the task through the workflow
        # and collect metrics, logs, traces
        pass
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics in Prometheus format"""
        metrics = self.executor.get_metrics()
        
        # Convert to Prometheus format
        prometheus_metrics = {
            "aura_tasks_total": metrics["tasks_executed"],
            "aura_tasks_succeeded_total": metrics["tasks_succeeded"],
            "aura_tasks_failed_total": metrics["tasks_failed"],
            "aura_task_duration_seconds_total": metrics["total_execution_time"],
            "aura_task_duration_seconds_avg": metrics["average_execution_time"]
        }
        
        return prometheus_metrics