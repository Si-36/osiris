"""
🌟 AURA Main System
===================

The unified main system that connects all AURA components.

This consolidates the best of system.py and unified_system.py,
using OUR refactored components instead of the "Ultimate" versions.

Features:
- Clean component architecture
- Event-driven communication
- Self-healing capabilities
- Component registry
- Unified monitoring
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import structlog

# Import OUR components
from ..memory.core.memory_api import AURAMemorySystem
from ..neural.model_router import AURAModelRouter
from ..tda.agent_topology import AgentTopologyAnalyzer
from ..swarm_intelligence.swarm_coordinator import SwarmCoordinator
from ..orchestration.unified_orchestration_engine import UnifiedOrchestrationEngine
from ..agents.agent_core import AURAAgentCore as AURAAgent
from ..agents.lnn_council import LNNCouncilOrchestrator

# Import extracted CORE components
from .self_healing_engine import SelfHealingEngine
from .executive_controller import ExecutiveController

# Import interfaces
try:
    from .unified_interfaces import ComponentStatus, SystemEvent, Priority
except ImportError:
    # Define minimal interfaces if not available
    class ComponentStatus(str, Enum):
        INITIALIZING = "initializing"
        READY = "ready"
        RUNNING = "running"
        DEGRADED = "degraded"
        FAILED = "failed"
        STOPPED = "stopped"
    
    @dataclass
    class SystemEvent:
        event_id: str
        event_type: str
        component: str
        data: Dict[str, Any]
        timestamp: float = field(default_factory=time.time)
        priority: str = "normal"

logger = structlog.get_logger(__name__)


# ==================== System State ====================

@dataclass
class SystemMetrics:
    """System-wide metrics"""
    system_id: str
    start_time: float
    uptime_seconds: float = 0.0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    component_health: Dict[str, float] = field(default_factory=dict)
    resilience_score: float = 0.5
    last_updated: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    @property
    def overall_health(self) -> float:
        if not self.component_health:
            return 1.0
        return sum(self.component_health.values()) / len(self.component_health)


@dataclass
class SystemConfig:
    """System configuration"""
    enable_self_healing: bool = True
    enable_chaos_engineering: bool = False
    enable_swarm_coordination: bool = True
    enable_lnn_council: bool = True
    max_memory_gb: float = 16.0
    component_timeout_seconds: float = 30.0
    health_check_interval: float = 10.0


# ==================== Component Registry ====================

class ComponentRegistry:
    """Registry of all system components"""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.status: Dict[str, ComponentStatus] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
    def register(
        self,
        name: str,
        component: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a component"""
        self.components[name] = component
        self.status[name] = ComponentStatus.INITIALIZING
        self.metadata[name] = metadata or {}
        logger.info(f"Registered component: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a component by name"""
        return self.components.get(name)
    
    def set_status(self, name: str, status: ComponentStatus):
        """Update component status"""
        if name in self.components:
            self.status[name] = status
            logger.info(f"Component {name} status: {status}")
    
    def get_all_status(self) -> Dict[str, ComponentStatus]:
        """Get status of all components"""
        return self.status.copy()
    
    def get_healthy_components(self) -> List[str]:
        """Get list of healthy components"""
        return [
            name for name, status in self.status.items()
            if status in [ComponentStatus.READY, ComponentStatus.RUNNING]
        ]


# ==================== Main System ====================

class AURAMainSystem:
    """
    The main AURA system that orchestrates all components.
    
    This is the central hub that:
    - Initializes all components
    - Manages component lifecycle
    - Handles inter-component communication
    - Monitors system health
    - Applies self-healing when needed
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.system_id = f"aura_{int(time.time())}"
        
        # System state
        self.running = False
        self.metrics = SystemMetrics(
            system_id=self.system_id,
            start_time=time.time()
        )
        
        # Component registry
        self.registry = ComponentRegistry()
        
        # Event queue for async communication
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # Initialize core components
        self._initialize_components()
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        logger.info(
            "AURA Main System initialized",
            system_id=self.system_id,
            config=self.config.__dict__
        )
    
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing AURA components...")
        
        # Memory System (central hub)
        self.memory = AURAMemorySystem({
            "enable_topology": True,
            "enable_mem0": True,
            "enable_graphrag": True,
            "enable_lakehouse": True
        })
        self.registry.register("memory", self.memory, {"type": "storage", "priority": "critical"})
        
        # Neural Router (model selection)
        self.neural_router = AURAModelRouter({
            "enable_cache": True,
            "enable_lnn_council": self.config.enable_lnn_council
        })
        self.registry.register("neural_router", self.neural_router, {"type": "routing", "priority": "high"})
        
        # TDA Analyzer (topology analysis)
        self.tda = AgentTopologyAnalyzer()
        self.registry.register("tda", self.tda, {"type": "analysis", "priority": "medium"})
        
        # Swarm Coordinator (multi-agent coordination)
        if self.config.enable_swarm_coordination:
            self.swarm = SwarmCoordinator({
                "num_particles": 50,
                "num_ants": 30,
                "num_bees": 40
            })
            self.registry.register("swarm", self.swarm, {"type": "coordination", "priority": "medium"})
        else:
            self.swarm = None
        
        # Orchestration Engine (workflow management)
        self.orchestrator = UnifiedOrchestrationEngine()
        self.registry.register("orchestrator", self.orchestrator, {"type": "workflow", "priority": "high"})
        
        # LNN Council (if enabled)
        if self.config.enable_lnn_council:
            self.lnn_council = LNNCouncilOrchestrator(min_agents=3, max_agents=5)
            self.registry.register("lnn_council", self.lnn_council, {"type": "decision", "priority": "medium"})
        else:
            self.lnn_council = None
        
        # Self-Healing Engine
        if self.config.enable_self_healing:
            self.self_healing = SelfHealingEngine(
                memory_system=self.memory,
                tda_analyzer=self.tda
            )
            self.registry.register("self_healing", self.self_healing, {"type": "resilience", "priority": "critical"})
        else:
            self.self_healing = None
        
        # Executive Controller (from consciousness)
        self.executive = ExecutiveController(
            memory_system=self.memory,
            orchestrator=self.orchestrator
        )
        self.registry.register("executive", self.executive, {"type": "control", "priority": "critical"})
        
        logger.info(f"Initialized {len(self.registry.components)} components")
    
    async def start(self):
        """Start the AURA system"""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting AURA system...")
        self.running = True
        
        # Initialize all components
        await self._initialize_all_components()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._event_handler()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        # Update system status
        for name in self.registry.components:
            self.registry.set_status(name, ComponentStatus.RUNNING)
        
        logger.info("AURA system started successfully")
    
    async def stop(self):
        """Stop the AURA system"""
        if not self.running:
            return
        
        logger.info("Stopping AURA system...")
        self.running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Shutdown components gracefully
        await self._shutdown_all_components()
        
        # Update status
        for name in self.registry.components:
            self.registry.set_status(name, ComponentStatus.STOPPED)
        
        logger.info("AURA system stopped")
    
    async def _initialize_all_components(self):
        """Initialize all components that need async init"""
        # Memory system
        if hasattr(self.memory, 'initialize'):
            await self.memory.initialize()
        
        # Orchestrator
        if hasattr(self.orchestrator, 'initialize'):
            await self.orchestrator.initialize()
        
        # Executive controller
        if hasattr(self.executive, 'initialize'):
            await self.executive.initialize()
    
    async def _shutdown_all_components(self):
        """Shutdown all components gracefully"""
        # Save memory state
        if self.memory:
            try:
                await self.memory.create_memory_branch("shutdown_backup")
            except:
                pass
    
    async def _event_handler(self):
        """Handle system events"""
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error handling event: {e}")
    
    async def _process_event(self, event: SystemEvent):
        """Process a system event"""
        logger.debug(f"Processing event: {event.event_type} from {event.component}")
        
        # Route event to appropriate handler
        if event.event_type == "component_failure":
            await self._handle_component_failure(event)
        elif event.event_type == "performance_degradation":
            await self._handle_performance_issue(event)
        elif event.event_type == "memory_threshold":
            await self._handle_memory_issue(event)
        else:
            # Store in memory for analysis
            await self.memory.store({
                "type": "system_event",
                "event": event.__dict__
            })
    
    async def _handle_component_failure(self, event: SystemEvent):
        """Handle component failure"""
        component_name = event.data.get("component_name")
        
        if self.self_healing and component_name:
            # Apply self-healing
            result = await self.self_healing.heal_component(
                component_name,
                {
                    "type": "component_failure",
                    "severity": 0.8,
                    "details": event.data
                }
            )
            
            logger.info(f"Self-healing result for {component_name}: {result}")
    
    async def _handle_performance_issue(self, event: SystemEvent):
        """Handle performance degradation"""
        if self.swarm:
            # Use swarm to optimize
            result = await self.swarm.optimize_parameters(
                search_space=event.data.get("parameters", {}),
                objective_function=lambda x: -event.data.get("latency", 0)
            )
            
            logger.info(f"Swarm optimization result: {result}")
    
    async def _handle_memory_issue(self, event: SystemEvent):
        """Handle memory threshold"""
        # Trigger memory cleanup
        if hasattr(self.memory, 'cleanup'):
            await self.memory.cleanup()
    
    async def _health_monitor(self):
        """Monitor component health"""
        while self.running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check each component
                for name, component in self.registry.components.items():
                    health = await self._check_component_health(name, component)
                    self.metrics.component_health[name] = health
                    
                    # Trigger healing if needed
                    if health < 0.5 and self.self_healing:
                        await self.event_queue.put(SystemEvent(
                            event_id=f"health_{time.time()}",
                            event_type="component_failure",
                            component="health_monitor",
                            data={"component_name": name, "health": health}
                        ))
                
                # Update resilience score
                if self.self_healing:
                    self.metrics.resilience_score = await self.self_healing.get_system_resilience_score()
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _check_component_health(self, name: str, component: Any) -> float:
        """Check health of a component"""
        # Generic health check - can be customized per component
        try:
            if hasattr(component, 'get_health'):
                return await component.get_health()
            elif hasattr(component, 'status'):
                status = component.status
                if status == ComponentStatus.RUNNING:
                    return 1.0
                elif status == ComponentStatus.DEGRADED:
                    return 0.5
                else:
                    return 0.0
            else:
                # Assume healthy if no health method
                return 1.0
        except:
            return 0.0
    
    async def _metrics_collector(self):
        """Collect system metrics"""
        while self.running:
            try:
                await asyncio.sleep(10)  # Collect every 10 seconds
                
                # Update metrics
                self.metrics.uptime_seconds = time.time() - self.metrics.start_time
                self.metrics.last_updated = time.time()
                
                # Store metrics in memory
                await self.memory.store({
                    "type": "system_metrics",
                    "metrics": self.metrics.__dict__
                })
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a user request through the system"""
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Route through neural router for model selection
            routing_result = await self.neural_router.route_request(request)
            
            # Create workflow
            workflow = await self.orchestrator.create_workflow({
                "request_id": request_id,
                "routing": routing_result,
                "tasks": request.get("tasks", [])
            })
            
            # Execute workflow
            result = await self.orchestrator.execute_workflow(workflow)
            
            # Update metrics
            self.metrics.total_operations += 1
            self.metrics.successful_operations += 1
            
            return {
                "request_id": request_id,
                "result": result,
                "duration": time.time() - start_time,
                "model_used": routing_result.model,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            self.metrics.total_operations += 1
            self.metrics.failed_operations += 1
            
            # Trigger self-healing if available
            if self.self_healing:
                await self.self_healing.heal_component(
                    "request_handler",
                    {"type": "request_failure", "error": str(e)}
                )
            
            return {
                "request_id": request_id,
                "error": str(e),
                "duration": time.time() - start_time,
                "status": "failed"
            }
    
    async def run_chaos_experiment(
        self,
        target_components: List[str],
        failure_type: str = "latency",
        intensity: float = 0.3
    ) -> Dict[str, Any]:
        """Run a chaos experiment (if enabled)"""
        if not self.config.enable_chaos_engineering:
            return {"error": "Chaos engineering not enabled"}
        
        if not self.self_healing:
            return {"error": "Self-healing not available"}
        
        # Map string to FailureType
        from .self_healing_engine import FailureType
        failure_map = {
            "latency": FailureType.LATENCY_INJECTION,
            "errors": FailureType.ERROR_INJECTION,
            "resource": FailureType.RESOURCE_EXHAUSTION
        }
        
        return await self.self_healing.run_chaos_experiment(
            target_components=target_components,
            failure_type=failure_map.get(failure_type, FailureType.LATENCY_INJECTION),
            intensity=intensity
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system_id": self.system_id,
            "running": self.running,
            "uptime": self.metrics.uptime_seconds,
            "health": self.metrics.overall_health,
            "resilience": self.metrics.resilience_score,
            "success_rate": self.metrics.success_rate,
            "components": self.registry.get_all_status(),
            "healthy_components": self.registry.get_healthy_components(),
            "metrics": self.metrics.__dict__
        }
    
    async def emergency_shutdown(self, reason: str):
        """Emergency shutdown of the system"""
        logger.warning(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Store shutdown reason
        await self.memory.store({
            "type": "emergency_shutdown",
            "reason": reason,
            "timestamp": time.time(),
            "system_state": self.get_system_status()
        })
        
        # Stop system
        await self.stop()


# ==================== Example Usage ====================

async def example():
    """Example of using the AURA Main System"""
    print("\n🌟 AURA Main System Example\n")
    
    # Initialize system
    config = SystemConfig(
        enable_self_healing=True,
        enable_chaos_engineering=False,  # Disabled for example
        enable_swarm_coordination=True,
        enable_lnn_council=True
    )
    
    system = AURAMainSystem(config)
    
    # Start system
    await system.start()
    
    # Get status
    status = system.get_system_status()
    print(f"System Status: {status['health']:.2f} health, {status['resilience']:.2f} resilience")
    print(f"Components: {len(status['healthy_components'])} healthy")
    
    # Execute a request
    result = await system.execute_request({
        "type": "analysis",
        "query": "Analyze system performance",
        "tasks": ["collect_metrics", "analyze_patterns", "generate_report"]
    })
    
    print(f"\nRequest result: {result['status']}")
    print(f"Duration: {result['duration']:.2f}s")
    
    # Wait a bit
    await asyncio.sleep(5)
    
    # Stop system
    await system.stop()
    
    print("\n✅ AURA Main System example completed!")


if __name__ == "__main__":
    asyncio.run(example())