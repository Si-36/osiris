#!/usr/bin/env python3
"""
Unified Core System Interfaces
Consolidated interfaces for all AURA Intelligence components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime
import uuid

# Type variables
T = TypeVar('T')
R = TypeVar('R')

# ============================================================================
# CORE ENUMS AND STATUS TYPES
# ============================================================================

class ComponentStatus(Enum):
    """Universal component status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class Priority(Enum):
    """Universal priority levels."""
    LOW = 0.2
    NORMAL = 0.5
    HIGH = 0.8
    CRITICAL = 1.0

class HealthLevel(Enum):
    """System health levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class ComponentMetrics:
    """Universal component metrics."""
    component_id: str
    status: ComponentStatus
    health_score: float  # 0.0 to 1.0
    uptime_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        pass
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    @property
    def health_level(self) -> HealthLevel:
        """Get health level based on score."""
        pass
        if self.health_score >= 0.9:
            return HealthLevel.EXCELLENT
        elif self.health_score >= 0.7:
            return HealthLevel.GOOD
        elif self.health_score >= 0.5:
            return HealthLevel.FAIR
        elif self.health_score >= 0.3:
            return HealthLevel.POOR
        else:
            return HealthLevel.CRITICAL

@dataclass
class SystemEvent:
    """Universal system event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    component_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    tags: List[str] = field(default_factory=list)

@dataclass
class ConfigurationUpdate:
    """Configuration update event."""
    component_id: str
    config_key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    applied_by: str = "system"

# ============================================================================
# UNIFIED COMPONENT INTERFACE
# ============================================================================

class UnifiedComponent(ABC):
    """
    Unified base interface for all AURA Intelligence components.
    
    This replaces all scattered component interfaces with a single,
    consistent interface that all components must implement.
    """
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        self.component_id = component_id
        self.config = config
        self.status = ComponentStatus.INITIALIZING
        self.start_time = time.time()
        self.metrics = ComponentMetrics(
            component_id=component_id,
            status=ComponentStatus.INITIALIZING,
            health_score=1.0,
            uptime_seconds=0.0,
            total_operations=0,
            successful_operations=0,
            failed_operations=0,
            average_response_time_ms=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0
        )
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._health_callbacks: List[Callable] = []
    
    # ========================================================================
    # LIFECYCLE METHODS (Required)
    # ========================================================================
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component. Returns True if successful."""
        pass
    
    @abstractmethod
        async def start(self) -> bool:
        """Start the component. Returns True if successful."""
        pass
    
    @abstractmethod
        async def stop(self) -> bool:
        """Stop the component. Returns True if successful."""
        pass
    
    @abstractmethod
        async def health_check(self) -> ComponentMetrics:
        """Perform health check and return current metrics."""
        pass
    
    # ========================================================================
    # CONFIGURATION METHODS (Required)
    # ========================================================================
    
    @abstractmethod
        async def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update component configuration. Returns True if successful."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration. Returns True if valid."""
        pass
    
    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for validation."""
        pass
    
    # ========================================================================
    # PROCESSING METHODS (Required)
    # ========================================================================
    
    @abstractmethod
        async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process input data and return result."""
        pass
    
    # ========================================================================
    # OBSERVABILITY METHODS (Implemented)
    # ========================================================================
    
        async def get_metrics(self) -> ComponentMetrics:
        """Get current component metrics."""
        pass
        self.metrics.uptime_seconds = time.time() - self.start_time
        self.metrics.status = self.status
        return self.metrics
    
        async def emit_event(self, event_type: str, data: Dict[str, Any],
        priority: Priority = Priority.NORMAL) -> None:
        """Emit a system event."""
        event = SystemEvent(
            event_type=event_type,
            component_id=self.component_id,
            data=data,
            priority=priority
        )
        
        # Call event handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                    print(f"Event handler error: {e}")
    
    def subscribe_to_events(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def add_health_callback(self, callback: Callable) -> None:
        """Add health status callback."""
        self._health_callbacks.append(callback)
    
        async def _notify_health_callbacks(self) -> None:
        """Notify all health callbacks."""
        pass
        metrics = await self.get_metrics()
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                    print(f"Health callback error: {e}")
    
    # ========================================================================
    # UTILITY METHODS (Implemented)
    # ========================================================================
    
        async def restart(self) -> bool:
        """Restart the component."""
        pass
        try:
            await self.stop()
            return await self.start()
        except Exception as e:
            await self.emit_event("restart_failed", {"error": str(e)}, Priority.HIGH)
            return False
    
    def _update_operation_metrics(self, success: bool, response_time_ms: float) -> None:
        """Update operation metrics."""
        self.metrics.total_operations += 1
        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.failed_operations += 1
        
        # Update running average
        current_avg = self.metrics.average_response_time_ms
        total_ops = self.metrics.total_operations
        self.metrics.average_response_time_ms = (
            (current_avg * (total_ops - 1) + response_time_ms) / total_ops
        )
        
        # Update health score based on success rate
        self.metrics.health_score = self.metrics.success_rate
        self.metrics.last_updated = datetime.now()

# ============================================================================
# SPECIALIZED COMPONENT INTERFACES
# ============================================================================

class AgentComponent(UnifiedComponent):
    """Unified interface for all agent types (council, bio, etc.)."""
    
    @abstractmethod
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on context."""
        pass
    
    @abstractmethod
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Learn from feedback."""
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Get the type of agent (council, bio, etc.)."""
        pass

class MemoryComponent(UnifiedComponent):
    """Unified interface for all memory systems."""
    
    @abstractmethod
    async def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data in memory."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from memory."""
        pass
    
    @abstractmethod
    async def search(self, query: Any, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory for similar items."""
        pass
    
    @abstractmethod
    async def consolidate(self) -> Dict[str, Any]:
        """Consolidate memory (cleanup, optimization, etc.)."""
        pass

class NeuralComponent(UnifiedComponent):
    """Unified interface for all neural network systems."""
    
    @abstractmethod
    async def forward(self, input_tensor: Any) -> Any:
        """Forward pass through the network."""
        pass
    
    @abstractmethod
    async def train_step(self, batch_data: Any) -> Dict[str, float]:
        """Perform one training step."""
        pass
    
    @abstractmethod
    async def save_model(self, path: str) -> bool:
        """Save model to disk."""
        pass
    
    @abstractmethod
    async def load_model(self, path: str) -> bool:
        """Load model from disk."""
        pass

class OrchestrationComponent(UnifiedComponent):
    """Unified interface for orchestration systems."""
    
    @abstractmethod
    async def orchestrate(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a workflow."""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow."""
        pass
    
    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        pass

class ObservabilityComponent(UnifiedComponent):
    """Unified interface for observability systems."""
    
    @abstractmethod
    async def collect_metrics(self, component_id: str) -> ComponentMetrics:
        """Collect metrics from a component."""
        pass
    
    @abstractmethod
    async def log_event(self, event: SystemEvent) -> bool:
        """Log a system event."""
        pass
    
    @abstractmethod
    async def create_alert(self, condition: str, severity: Priority) -> str:
        """Create an alert condition."""
        pass

# ============================================================================
# COMPONENT REGISTRY
# ============================================================================

class ComponentRegistry:
    """
    Centralized registry for all system components.
    
    Provides service discovery, health monitoring, and lifecycle management
    for all components in the system.
    """
    
    def __init__(self):
        self._components: Dict[str, UnifiedComponent] = {}
        self._component_types: Dict[str, str] = {}
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30  # seconds
    
    def register_component(self, component: UnifiedComponent, component_type: str) -> None:
        """Register a component with the registry."""
        self._components[component.component_id] = component
        self._component_types[component.component_id] = component_type
        
        # Subscribe to component events
        component.subscribe_to_events("error", self._handle_component_error)
        component.subscribe_to_events("status_change", self._handle_status_change)
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component."""
        if component_id in self._components:
            del self._components[component_id]
            del self._component_types[component_id]
            return True
        return False
    
    def get_component(self, component_id: str) -> Optional[UnifiedComponent]:
        """Get a component by ID."""
        return self._components.get(component_id)
    
    def get_components_by_type(self, component_type: str) -> List[UnifiedComponent]:
        """Get all components of a specific type."""
        return [
            component for component_id, component in self._components.items()
            if self._component_types[component_id] == component_type
        ]
    
    def list_components(self) -> Dict[str, str]:
        """List all registered components and their types."""
        pass
        return self._component_types.copy()
    
        async def health_check_all(self) -> Dict[str, ComponentMetrics]:
        """Perform health check on all components."""
        pass
        results = {}
        for component_id, component in self._components.items():
            try:
                metrics = await component.health_check()
                results[component_id] = metrics
            except Exception as e:
                    # Create error metrics
                results[component_id] = ComponentMetrics(
                    component_id=component_id,
                    status=ComponentStatus.ERROR,
                    health_score=0.0,
                    uptime_seconds=0.0,
                    total_operations=0,
                    successful_operations=0,
                    failed_operations=1,
                    average_response_time_ms=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0
                )
        return results
    
        async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        pass
        if self._health_monitor_task is None:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
    
        async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        pass
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None
    
        async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        pass
        while True:
            try:
                health_results = await self.health_check_all()
                
                # Check for unhealthy components
                for component_id, metrics in health_results.items():
                    if metrics.health_score < 0.5:
                        print(f"⚠️ Component {component_id} health degraded: {metrics.health_score:.2f}")
                    elif metrics.status == ComponentStatus.ERROR:
                        print(f"❌ Component {component_id} in error state")
                
                await asyncio.sleep(self._health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitoring error: {e}")
                await asyncio.sleep(self._health_check_interval)
    
        async def _handle_component_error(self, event: SystemEvent) -> None:
        """Handle component error events."""
        print(f"🚨 Component error: {event.component_id} - {event.data}")
    
        async def _handle_status_change(self, event: SystemEvent) -> None:
        """Handle component status change events."""
        print(f"📊 Status change: {event.component_id} - {event.data}")

# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

# Global component registry
_global_registry: Optional[ComponentRegistry] = None

    def get_component_registry() -> ComponentRegistry:
        """Get the global component registry."""
        global _global_registry
        if _global_registry is None:
        _global_registry = ComponentRegistry()
        return _global_registry

    def register_component(component: UnifiedComponent, component_type: str) -> None:
        """Register a component with the global registry."""
        registry = get_component_registry()
        registry.register_component(component, component_type)

    def get_component(component_id: str) -> Optional[UnifiedComponent]:
        """Get a component from the global registry."""
        registry = get_component_registry()
        return registry.get_component(component_id)
