"""
AURA Intelligence Component Registry - 2025 Production Implementation

Features:
- Dynamic component discovery and loading
- Plugin-based architecture
- Dependency injection
- Health monitoring
- Hot-reloading support
- Version management
"""

import asyncio
from typing import Dict, List, Any, Optional, Type, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import structlog
from abc import ABC, abstractmethod
import importlib
import inspect
import weakref
from pathlib import Path

logger = structlog.get_logger(__name__)


class ComponentRole(Enum):
    """Component roles in the system"""
    INFORMATION_AGENT = "information"
    CONTROL_AGENT = "control"
    HYBRID_AGENT = "hybrid"
    ORCHESTRATOR = "orchestrator"
    PROCESSOR = "processor"
    ADAPTER = "adapter"
    MONITOR = "monitor"


class ComponentCategory(Enum):
    """Component categories"""
    NEURAL = "neural"
    MEMORY = "memory"
    AGENT = "agent"
    ORCHESTRATION = "orchestration"
    COMMUNICATION = "communication"
    CONSCIOUSNESS = "consciousness"
    TDA = "tda"
    GOVERNANCE = "governance"
    OBSERVABILITY = "observability"
    INFRASTRUCTURE = "infrastructure"
    PROCESSING = "processing"
    INTEGRATION = "integration"


class ComponentStatus(Enum):
    """Component lifecycle status"""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    FAILED = "failed"
    UNREGISTERED = "unregistered"


@dataclass
class ComponentMetadata:
    """Metadata for component"""
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComponentHealth:
    """Component health information"""
    status: ComponentStatus = ComponentStatus.REGISTERED
    health_score: float = 1.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentInfo:
    """Complete component information"""
    id: str
    name: str
    module_path: str
    category: ComponentCategory
    role: ComponentRole
    capabilities: List[str]
    dependencies: List[str] = field(default_factory=list)
    metadata: ComponentMetadata = field(default_factory=ComponentMetadata)
    health: ComponentHealth = field(default_factory=ComponentHealth)
    config: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
    

class Component(ABC):
    """Base class for all components"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the component"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the component"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get component capabilities"""
        pass


class AURAComponentRegistry:
    """
    Advanced component registry for AURA system
    
    Key features:
    - Dynamic component discovery
    - Dependency resolution
    - Health monitoring
    - Hot-reloading
    - Plugin architecture
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.role_assignments: Dict[ComponentRole, List[str]] = {
            role: [] for role in ComponentRole
        }
        self.category_index: Dict[ComponentCategory, List[str]] = {
            cat: [] for cat in ComponentCategory
        }
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.health_monitors: Dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()
        
        logger.info("Component registry initialized")
    
    async def register(self,
                      component_id: str,
                      name: str,
                      module_path: str,
                      category: ComponentCategory,
                      role: ComponentRole,
                      capabilities: List[str],
                      dependencies: Optional[List[str]] = None,
                      metadata: Optional[ComponentMetadata] = None,
                      config: Optional[Dict[str, Any]] = None) -> bool:
        """Register a component"""
        async with self._lock:
            if component_id in self.components:
                logger.warning("Component already registered", component_id=component_id)
                return False
            
            # Create component info
            info = ComponentInfo(
                id=component_id,
                name=name,
                module_path=module_path,
                category=category,
                role=role,
                capabilities=capabilities,
                dependencies=dependencies or [],
                metadata=metadata or ComponentMetadata(),
                config=config or {}
            )
            
            # Store component
            self.components[component_id] = info
            
            # Update indices
            self.role_assignments[role].append(component_id)
            self.category_index[category].append(component_id)
            
            # Update dependency graph
            self.dependency_graph[component_id] = set(dependencies or [])
            
            logger.info("Component registered",
                       component_id=component_id,
                       name=name,
                       category=category.value,
                       role=role.value)
            
            return True
    
    async def unregister(self, component_id: str) -> bool:
        """Unregister a component"""
        async with self._lock:
            if component_id not in self.components:
                return False
            
            info = self.components[component_id]
            
            # Stop health monitoring
            if component_id in self.health_monitors:
                self.health_monitors[component_id].cancel()
                del self.health_monitors[component_id]
            
            # Stop component if running
            if info.instance and info.health.status == ComponentStatus.ACTIVE:
                try:
                    await info.instance.stop()
                except Exception as e:
                    logger.error("Error stopping component",
                               component_id=component_id,
                               error=str(e))
            
            # Remove from indices
            self.role_assignments[info.role].remove(component_id)
            self.category_index[info.category].remove(component_id)
            del self.dependency_graph[component_id]
            
            # Remove component
            del self.components[component_id]
            
            logger.info("Component unregistered", component_id=component_id)
            
            return True
    
    async def load_component(self, component_id: str) -> Optional[Component]:
        """Load and instantiate a component"""
        if component_id not in self.components:
            logger.error("Component not found", component_id=component_id)
            return None
        
        info = self.components[component_id]
        
        if info.instance:
            logger.debug("Component already loaded", component_id=component_id)
            return info.instance
        
        try:
            # Check dependencies
            for dep_id in info.dependencies:
                if dep_id not in self.components:
                    logger.error("Missing dependency",
                               component_id=component_id,
                               dependency=dep_id)
                    return None
                
                # Load dependency first
                dep_info = self.components[dep_id]
                if not dep_info.instance:
                    await self.load_component(dep_id)
            
            # Import module
            module = importlib.import_module(info.module_path)
            
            # Find component class
            component_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Component) and 
                    obj != Component):
                    component_class = obj
                    break
            
            if not component_class:
                logger.error("No component class found",
                           component_id=component_id,
                           module_path=info.module_path)
                return None
            
            # Create instance
            instance = component_class()
            info.instance = instance
            
            # Initialize
            info.health.status = ComponentStatus.INITIALIZING
            await instance.initialize(info.config)
            
            # Start
            await instance.start()
            info.health.status = ComponentStatus.ACTIVE
            
            # Start health monitoring
            if self._running:
                self.health_monitors[component_id] = asyncio.create_task(
                    self._monitor_health(component_id)
                )
            
            logger.info("Component loaded",
                       component_id=component_id,
                       class_name=component_class.__name__)
            
            return instance
            
        except Exception as e:
            logger.error("Failed to load component",
                        component_id=component_id,
                        error=str(e))
            info.health.status = ComponentStatus.FAILED
            info.health.last_error = str(e)
            info.health.error_count += 1
            return None
    
    async def reload_component(self, component_id: str) -> bool:
        """Reload a component (hot-reload)"""
        if component_id not in self.components:
            return False
        
        info = self.components[component_id]
        
        # Stop existing instance
        if info.instance:
            try:
                await info.instance.stop()
            except Exception as e:
                logger.error("Error stopping component for reload",
                           component_id=component_id,
                           error=str(e))
        
        # Clear instance
        info.instance = None
        info.health.status = ComponentStatus.REGISTERED
        
        # Reload module
        try:
            module = importlib.import_module(info.module_path)
            importlib.reload(module)
        except Exception as e:
            logger.error("Failed to reload module",
                        component_id=component_id,
                        error=str(e))
            return False
        
        # Load component again
        instance = await self.load_component(component_id)
        
        return instance is not None
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get component instance"""
        if component_id not in self.components:
            return None
        return self.components[component_id].instance
    
    def get_components_by_role(self, role: ComponentRole) -> List[ComponentInfo]:
        """Get all components with specific role"""
        return [
            self.components[cid]
            for cid in self.role_assignments[role]
            if cid in self.components
        ]
    
    def get_components_by_category(self, category: ComponentCategory) -> List[ComponentInfo]:
        """Get all components in category"""
        return [
            self.components[cid]
            for cid in self.category_index[category]
            if cid in self.components
        ]
    
    def get_components_by_capability(self, capability: str) -> List[ComponentInfo]:
        """Get components with specific capability"""
        return [
            info for info in self.components.values()
            if capability in info.capabilities
        ]
    
    def get_dependency_order(self) -> List[str]:
        """Get component loading order based on dependencies"""
        # Topological sort
        visited = set()
        order = []
        
        def visit(component_id: str):
            if component_id in visited:
                return
            
            visited.add(component_id)
            
            # Visit dependencies first
            for dep_id in self.dependency_graph.get(component_id, set()):
                if dep_id in self.components:
                    visit(dep_id)
            
            order.append(component_id)
        
        # Visit all components
        for component_id in self.components:
            visit(component_id)
        
        return order
    
    async def start_all(self):
        """Start all registered components"""
        self._running = True
        
        # Load components in dependency order
        order = self.get_dependency_order()
        
        for component_id in order:
            await self.load_component(component_id)
        
        logger.info("All components started", count=len(order))
    
    async def stop_all(self):
        """Stop all components"""
        self._running = False
        
        # Stop health monitors
        for task in self.health_monitors.values():
            task.cancel()
        
        await asyncio.gather(*self.health_monitors.values(), return_exceptions=True)
        self.health_monitors.clear()
        
        # Stop components in reverse dependency order
        order = reversed(self.get_dependency_order())
        
        for component_id in order:
            info = self.components[component_id]
            if info.instance and info.health.status == ComponentStatus.ACTIVE:
                try:
                    await info.instance.stop()
                    info.health.status = ComponentStatus.SUSPENDED
                except Exception as e:
                    logger.error("Error stopping component",
                               component_id=component_id,
                               error=str(e))
        
        logger.info("All components stopped")
    
    async def _monitor_health(self, component_id: str):
        """Monitor component health"""
        while self._running and component_id in self.components:
            try:
                info = self.components[component_id]
                
                if info.instance and info.health.status == ComponentStatus.ACTIVE:
                    # Perform health check
                    health_data = await info.instance.health_check()
                    
                    # Update health info
                    info.health.last_heartbeat = datetime.now()
                    info.health.metrics = health_data
                    
                    # Calculate health score
                    if "healthy" in health_data and not health_data["healthy"]:
                        info.health.status = ComponentStatus.DEGRADED
                        info.health.health_score = 0.5
                    else:
                        info.health.health_score = health_data.get("score", 1.0)
                
                # Wait before next check
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error("Health check failed",
                           component_id=component_id,
                           error=str(e))
                info = self.components[component_id]
                info.health.error_count += 1
                info.health.last_error = str(e)
                
                # Mark as failed after too many errors
                if info.health.error_count > 5:
                    info.health.status = ComponentStatus.FAILED
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        total_components = len(self.components)
        active_components = sum(
            1 for info in self.components.values()
            if info.health.status == ComponentStatus.ACTIVE
        )
        
        avg_health_score = sum(
            info.health.health_score for info in self.components.values()
        ) / total_components if total_components > 0 else 0
        
        return {
            "total_components": total_components,
            "active_components": active_components,
            "average_health_score": avg_health_score,
            "components_by_status": {
                status.value: sum(
                    1 for info in self.components.values()
                    if info.health.status == status
                )
                for status in ComponentStatus
            },
            "unhealthy_components": [
                {
                    "id": info.id,
                    "name": info.name,
                    "status": info.health.status.value,
                    "last_error": info.health.last_error
                }
                for info in self.components.values()
                if info.health.status in [ComponentStatus.DEGRADED, ComponentStatus.FAILED]
            ]
        }


# Singleton registry instance
_registry: Optional[AURAComponentRegistry] = None


def get_registry() -> AURAComponentRegistry:
    """Get singleton registry instance"""
    global _registry
    if _registry is None:
        _registry = AURAComponentRegistry()
    return _registry


# Decorator for auto-registration
def register_component(
    component_id: str,
    name: str,
    category: ComponentCategory,
    role: ComponentRole,
    capabilities: List[str],
    dependencies: Optional[List[str]] = None
):
    """Decorator to auto-register components"""
    def decorator(cls: Type[Component]):
        # Register on import
        asyncio.create_task(
            get_registry().register(
                component_id=component_id,
                name=name,
                module_path=cls.__module__,
                category=category,
                role=role,
                capabilities=capabilities,
                dependencies=dependencies
            )
        )
        return cls
    return decorator


# Example component
class ExampleComponent(Component):
    """Example component implementation"""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component"""
        self.config = config
        logger.info("Example component initialized", config=config)
    
    async def start(self) -> None:
        """Start the component"""
        logger.info("Example component started")
    
    async def stop(self) -> None:
        """Stop the component"""
        logger.info("Example component stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "healthy": True,
            "score": 1.0,
            "uptime": 100,
            "metrics": {
                "requests": 0,
                "errors": 0
            }
        }
    
    def get_capabilities(self) -> List[str]:
        """Get component capabilities"""
        return ["example", "test"]


# Example usage
async def example_registry_usage():
    """Example of using the component registry"""
    registry = get_registry()
    
    # Register a component
    await registry.register(
        component_id="example_1",
        name="Example Component",
        module_path="aura_intelligence.components.registry",
        category=ComponentCategory.PROCESSING,
        role=ComponentRole.PROCESSOR,
        capabilities=["example", "test"]
    )
    
    # Start all components
    await registry.start_all()
    
    # Get component
    component = registry.get_component("example_1")
    if component:
        health = await component.health_check()
        print(f"Component health: {health}")
    
    # Get system health
    system_health = registry.get_system_health()
    print(f"System health: {system_health}")
    
    # Stop all
    await registry.stop_all()
    
    return registry


if __name__ == "__main__":
    asyncio.run(example_registry_usage())