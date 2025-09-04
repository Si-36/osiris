"""
Tool Registry - Central management for all AURA tools
Provides discovery, validation, and execution capabilities
"""

from typing import Dict, Any, Optional, List, Callable, Type
from pydantic import BaseModel
import asyncio
import structlog
from enum import Enum

from ..schemas.aura_execution import ObservationResult

logger = structlog.get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools available in the system"""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    PLANNING = "planning"


class ToolMetadata(BaseModel):
    """Metadata about a registered tool"""
    name: str
    category: ToolCategory
    description: str
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None
    requires_auth: bool = False
    async_execution: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 3


class ToolRegistry:
    """
    Central registry for all AURA tools.
    Manages tool lifecycle, discovery, and execution.
    """
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ToolRegistry initialized")
    
    def register(
        self,
        name: str,
        tool_instance: Any,
        metadata: Optional[ToolMetadata] = None
    ) -> None:
        """Register a new tool with the registry"""
        if name in self._tools:
            logger.warning(f"Tool {name} already registered, overwriting")
        
        self._tools[name] = tool_instance
        
        if metadata:
            self._metadata[name] = metadata
        else:
            # Create default metadata
            self._metadata[name] = ToolMetadata(
                name=name,
                category=ToolCategory.EXECUTION,
                description=f"Tool: {name}",
                async_execution=asyncio.iscoroutinefunction(
                    getattr(tool_instance, 'execute', None)
                )
            )
        
        self._execution_stats[name] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_duration": 0.0
        }
        
        logger.info(f"Registered tool: {name} (category: {self._metadata[name].category})")
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """Execute a tool with the given parameters"""
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name} not found in registry")
        
        tool = self._tools[tool_name]
        metadata = self._metadata[tool_name]
        timeout = timeout or metadata.timeout_seconds
        
        logger.info(f"Executing tool: {tool_name} with params: {params}")
        
        # Update execution stats
        self._execution_stats[tool_name]["total_executions"] += 1
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute with timeout
            if metadata.async_execution:
                result = await asyncio.wait_for(
                    tool.execute(**params),
                    timeout=timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    tool.execute,
                    **params
                )
            
            # Validate output if schema is defined
            if metadata.output_schema and isinstance(result, dict):
                result = metadata.output_schema(**result)
            
            # Update success stats
            self._execution_stats[tool_name]["successful_executions"] += 1
            duration = asyncio.get_event_loop().time() - start_time
            self._execution_stats[tool_name]["total_duration"] += duration
            
            logger.info(f"Tool {tool_name} executed successfully in {duration:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            self._execution_stats[tool_name]["failed_executions"] += 1
            logger.error(f"Tool {tool_name} execution timed out after {timeout}s")
            raise
            
        except Exception as e:
            self._execution_stats[tool_name]["failed_executions"] += 1
            logger.error(f"Tool {tool_name} execution failed: {e}")
            
            # Retry logic
            if metadata.retry_count > 0:
                logger.info(f"Retrying tool {tool_name} (retries left: {metadata.retry_count})")
                metadata.retry_count -= 1
                return await self.execute(tool_name, params, timeout)
            
            raise
    
    def get_tool(self, name: str) -> Any:
        """Get a tool instance by name"""
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found in registry")
        return self._tools[name]
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all registered tools, optionally filtered by category"""
        if category:
            return [
                name for name, meta in self._metadata.items()
                if meta.category == category
            ]
        return list(self._tools.keys())
    
    def get_metadata(self, name: str) -> ToolMetadata:
        """Get metadata for a tool"""
        if name not in self._metadata:
            raise ValueError(f"Tool {name} not found in registry")
        return self._metadata[name]
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics for tools"""
        if name:
            if name not in self._execution_stats:
                raise ValueError(f"Tool {name} not found in registry")
            return self._execution_stats[name]
        return self._execution_stats
    
    def discover_tools_for_objective(self, objective: str) -> List[str]:
        """
        Intelligently discover tools that might be useful for an objective.
        This is a simple implementation - could be enhanced with ML.
        """
        relevant_tools = []
        objective_lower = objective.lower()
        
        # Simple keyword matching for now
        tool_keywords = {
            "observe": ["monitor", "watch", "check", "analyze", "inspect"],
            "analyze": ["pattern", "trend", "anomaly", "investigate"],
            "execute": ["run", "perform", "deploy", "apply"],
            "communicate": ["notify", "alert", "report", "send"],
            "memory": ["remember", "recall", "history", "past"],
            "plan": ["strategy", "approach", "design", "architect"]
        }
        
        for tool_name, metadata in self._metadata.items():
            # Check category keywords
            category_key = metadata.category.value
            if category_key in tool_keywords:
                for keyword in tool_keywords[category_key]:
                    if keyword in objective_lower:
                        relevant_tools.append(tool_name)
                        break
            
            # Check tool description
            if any(word in metadata.description.lower() for word in objective_lower.split()):
                if tool_name not in relevant_tools:
                    relevant_tools.append(tool_name)
        
        logger.info(f"Discovered {len(relevant_tools)} relevant tools for objective")
        return relevant_tools


class ToolExecutor:
    """
    Advanced tool executor with features like parallel execution,
    dependency resolution, and result caching.
    """
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._result_cache: Dict[str, Any] = {}
        
    async def execute_parallel(
        self,
        tool_executions: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute multiple tools in parallel"""
        tasks = []
        for execution in tool_executions:
            task = self.registry.execute(
                tool_name=execution["tool"],
                params=execution["params"]
            )
            tasks.append(task)
        
        logger.info(f"Executing {len(tasks)} tools in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_executions[i]['tool']} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_with_dependencies(
        self,
        executions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute tools respecting dependencies.
        Each execution can specify dependencies on other executions.
        """
        results = {}
        completed = set()
        
        while len(completed) < len(executions):
            # Find executions ready to run
            ready = []
            for execution in executions:
                exec_id = execution.get("id", execution["tool"])
                if exec_id in completed:
                    continue
                    
                dependencies = execution.get("dependencies", [])
                if all(dep in completed for dep in dependencies):
                    ready.append(execution)
            
            if not ready:
                raise ValueError("Circular dependency detected in tool executions")
            
            # Execute ready tools in parallel
            ready_results = await self.execute_parallel(ready)
            
            # Store results
            for execution, result in zip(ready, ready_results):
                exec_id = execution.get("id", execution["tool"])
                results[exec_id] = result
                completed.add(exec_id)
                
                # Update params for dependent executions
                for exec in executions:
                    if exec_id in exec.get("dependencies", []):
                        if "dependency_results" not in exec["params"]:
                            exec["params"]["dependency_results"] = {}
                        exec["params"]["dependency_results"][exec_id] = result
        
        return results
    
    def cache_result(self, key: str, result: Any) -> None:
        """Cache a tool execution result"""
        self._result_cache[key] = result
        
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result if available"""
        return self._result_cache.get(key)