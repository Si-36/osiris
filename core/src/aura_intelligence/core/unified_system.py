#!/usr/bin/env python3
"""
Unified System Orchestrator
Central system that coordinates all components with unified interfaces
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .unified_interfaces import (
    UnifiedComponent, ComponentStatus, ComponentMetrics, SystemEvent, Priority,
    AgentComponent, MemoryComponent, NeuralComponent, OrchestrationComponent, 
    ObservabilityComponent, ComponentRegistry, get_component_registry
)
from .unified_config import UnifiedConfig, get_config

# ============================================================================
# SYSTEM STATE AND METRICS
# ============================================================================

@dataclass
class SystemMetrics:
    """Unified system metrics."""
    system_id: str
    start_time: datetime
    uptime_seconds: float = 0.0
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    average_cycle_time_ms: float = 0.0
    overall_health_score: float = 1.0
    active_components: int = 0
    total_components: int = 0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        pass
        if self.total_cycles == 0:
            return 1.0
        return self.successful_cycles / self.total_cycles
    
    @property
    def component_health_rate(self) -> float:
        """Calculate component health rate."""
        pass
        if self.total_components == 0:
            return 1.0
        return self.active_components / self.total_components

@dataclass
class SystemState:
    """Current system state."""
    status: ComponentStatus = ComponentStatus.INITIALIZING
    consciousness_level: float = 0.5
    collective_intelligence: float = 0.5
    system_load: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_workflows: int = 0
    pending_events: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

# ============================================================================
# UNIFIED SYSTEM CLASS
# ============================================================================

class UnifiedSystem:
    """
    Unified AURA Intelligence System.
    
    Central orchestrator that manages all components through unified interfaces,
    providing a clean, consistent way to coordinate the entire system.
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None, system_id: Optional[str] = None):
        self.config = config or get_config()
        self.system_id = system_id or f"aura-system-{int(time.time())}"
        
        # System state
        self.state = SystemState()
        self.metrics = SystemMetrics(
            system_id=self.system_id,
            start_time=datetime.now()
        )
        
        # Component management
        self.registry = get_component_registry()
        self.running = False
        self._main_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        # Event handling
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_handlers: Dict[str, List[callable]] = {}
        
        # Lifecycle hooks
        self._startup_hooks: List[callable] = []
        self._shutdown_hooks: List[callable] = []
        
        print(f"🌟 Unified AURA Intelligence System initialized: {self.system_id}")
    
    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================
    
        async def initialize(self) -> bool:
        """Initialize the unified system."""
        pass
        try:
            print("🔧 Initializing Unified AURA Intelligence System...")
            self.state.status = ComponentStatus.INITIALIZING
            
            # Validate configuration
            config_issues = self.config.validate()
            if config_issues:
                print(f"⚠️ Configuration issues: {'; '.join(config_issues)}")
                if self.config.is_production():
                    raise ValueError(f"Configuration validation failed: {'; '.join(config_issues)}")
            
            # Start component registry health monitoring
            await self.registry.start_health_monitoring()
            
            # Initialize all registered components
            await self._initialize_components()
            
            # Run startup hooks
            await self._run_startup_hooks()
            
            # Start event processing
            asyncio.create_task(self._process_events())
            
            self.state.status = ComponentStatus.ACTIVE
            await self._emit_system_event("system_initialized", {"system_id": self.system_id})
            
            print("✅ Unified AURA Intelligence System initialized successfully")
            return True
            
        except Exception as e:
            self.state.status = ComponentStatus.ERROR
            self.state.last_error = str(e)
            self.state.last_error_time = datetime.now()
            print(f"❌ System initialization failed: {e}")
            return False
    
        async def start(self) -> bool:
        """Start the unified system."""
        pass
        try:
            if self.running:
                print("⚠️ System is already running")
                return True
            
            print("🚀 Starting Unified AURA Intelligence System...")
            
            # Initialize if not already done
            if self.state.status == ComponentStatus.INITIALIZING:
                if not await self.initialize():
                    return False
            
            self.running = True
            
            # Start main system loop
            self._main_task = asyncio.create_task(self._main_system_loop())
            
            # Start health monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            await self._emit_system_event("system_started", {"system_id": self.system_id})
            print("✅ Unified AURA Intelligence System started successfully")
            return True
            
        except Exception as e:
            self.state.status = ComponentStatus.ERROR
            self.state.last_error = str(e)
            self.state.last_error_time = datetime.now()
            print(f"❌ System start failed: {e}")
            return False
    
        async def stop(self) -> bool:
        """Stop the unified system."""
        pass
        try:
            print("🛑 Stopping Unified AURA Intelligence System...")
            self.running = False
            
            # Cancel main tasks
            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
            
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all components
            await self._stop_components()
            
            # Run shutdown hooks
            await self._run_shutdown_hooks()
            
            # Stop registry health monitoring
            await self.registry.stop_health_monitoring()
            
            self.state.status = ComponentStatus.INACTIVE
            await self._emit_system_event("system_stopped", {"system_id": self.system_id})
            
            print("✅ Unified AURA Intelligence System stopped successfully")
            return True
            
        except Exception as e:
            self.state.status = ComponentStatus.ERROR
            self.state.last_error = str(e)
            self.state.last_error_time = datetime.now()
            print(f"❌ System stop failed: {e}")
            return False
    
        async def restart(self) -> bool:
        """Restart the unified system."""
        pass
        print("🔄 Restarting Unified AURA Intelligence System...")
        if await self.stop():
            return await self.start()
        return False
    
    # ========================================================================
    # COMPONENT MANAGEMENT
    # ========================================================================
    
    def register_component(self, component: UnifiedComponent, component_type: str) -> None:
        """Register a component with the system."""
        self.registry.register_component(component, component_type)
        
        # Subscribe to component events
        component.subscribe_to_events("error", self._handle_component_error)
        component.subscribe_to_events("status_change", self._handle_component_status_change)
        
        print(f"📝 Registered component: {component.component_id} ({component_type})")
    
    def get_component(self, component_id: str) -> Optional[UnifiedComponent]:
        """Get a component by ID."""
        return self.registry.get_component(component_id)
    
    def get_components_by_type(self, component_type: str) -> List[UnifiedComponent]:
        """Get all components of a specific type."""
        return self.registry.get_components_by_type(component_type)
    
        async def _initialize_components(self) -> None:
        """Initialize all registered components."""
        pass
        components = self.registry.list_components()
        
        for component_id, component_type in components.items():
            component = self.registry.get_component(component_id)
            if component:
                try:
                    print(f"🔧 Initializing {component_type}: {component_id}")
                    success = await component.initialize()
                    if success:
                        success = await component.start()
                    
                    if not success:
                        print(f"⚠️ Failed to initialize component: {component_id}")
                        
                except Exception as e:
                    print(f"❌ Component initialization error {component_id}: {e}")
    
        async def _stop_components(self) -> None:
        """Stop all registered components."""
        pass
        components = self.registry.list_components()
        
        for component_id, component_type in components.items():
            component = self.registry.get_component(component_id)
            if component:
                try:
                    print(f"🛑 Stopping {component_type}: {component_id}")
                    await component.stop()
                except Exception as e:
                    print(f"⚠️ Component stop error {component_id}: {e}")
    
    # ========================================================================
    # SYSTEM ORCHESTRATION
    # ========================================================================
    
        async def _main_system_loop(self) -> None:
        """Main system orchestration loop."""
        pass
        cycle_count = 0
        
        while self.running:
            cycle_start = time.time()
            
            try:
                # Run system cycle
                cycle_result = await self._run_system_cycle(cycle_count)
                
                # Update metrics
                cycle_time_ms = (time.time() - cycle_start) * 1000
                self._update_cycle_metrics(cycle_time_ms, cycle_result['success'])
                
                # Log cycle completion
                if cycle_result['success']:
                    if cycle_count % 10 == 0:  # Log every 10th cycle
                        print(f"✅ System cycle {cycle_count}: {cycle_time_ms:.1f}ms, "
                              f"health={self.metrics.overall_health_score:.3f}")
                else:
                    print(f"❌ System cycle {cycle_count} failed: {cycle_result.get('error')}")
                
                cycle_count += 1
                
                # Adaptive wait based on system load
                wait_time = self._calculate_cycle_wait_time()
                await asyncio.sleep(wait_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ System loop error: {e}")
                cycle_time_ms = (time.time() - cycle_start) * 1000
                self._update_cycle_metrics(cycle_time_ms, False)
                await asyncio.sleep(1.0)  # Error recovery wait
    
        async def _run_system_cycle(self, cycle_number: int) -> Dict[str, Any]:
        """Run a single system cycle."""
        try:
            cycle_data = {
                'cycle_number': cycle_number,
                'timestamp': datetime.now(),
                'system_state': self.state
            }
            
            # Get all active components
            agent_components = self.get_components_by_type('agent')
            memory_components = self.get_components_by_type('memory')
            neural_components = self.get_components_by_type('neural')
            orchestration_components = self.get_components_by_type('orchestration')
            
            # Orchestrate agent decisions
            agent_results = {}
            for agent in agent_components:
                if isinstance(agent, AgentComponent):
                    try:
                        result = await agent.make_decision(cycle_data)
                        agent_results[agent.component_id] = result
                    except Exception as e:
                        print(f"⚠️ Agent {agent.component_id} decision error: {e}")
            
            # Process memory operations
            memory_results = {}
            for memory in memory_components:
                if isinstance(memory, MemoryComponent):
                    try:
                        result = await memory.consolidate()
                        memory_results[memory.component_id] = result
                    except Exception as e:
                        print(f"⚠️ Memory {memory.component_id} consolidation error: {e}")
            
            # Process neural computations
            neural_results = {}
            for neural in neural_components:
                if isinstance(neural, NeuralComponent):
                    try:
                        # Simple forward pass with cycle data
                        result = await neural.process(cycle_data)
                        neural_results[neural.component_id] = result
                    except Exception as e:
                        print(f"⚠️ Neural {neural.component_id} processing error: {e}")
            
            # Execute orchestration workflows
            orchestration_results = {}
            for orchestrator in orchestration_components:
                if isinstance(orchestrator, OrchestrationComponent):
                    try:
                        workflow = {
                            'type': 'system_cycle',
                            'data': {
                                'agents': agent_results,
                                'memory': memory_results,
                                'neural': neural_results
                            }
                        }
                        result = await orchestrator.orchestrate(workflow)
                        orchestration_results[orchestrator.component_id] = result
                    except Exception as e:
                        print(f"⚠️ Orchestrator {orchestrator.component_id} error: {e}")
            
            # Update system state based on results
            await self._update_system_state(agent_results, memory_results, neural_results, orchestration_results)
            
            return {
                'success': True,
                'cycle_number': cycle_number,
                'agent_results': agent_results,
                'memory_results': memory_results,
                'neural_results': neural_results,
                'orchestration_results': orchestration_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'cycle_number': cycle_number,
                'error': str(e)
            }
    
        async def _update_system_state(self, agent_results: Dict, memory_results: Dict,
        neural_results: Dict, orchestration_results: Dict) -> None:
        """Update system state based on cycle results."""
        # Calculate consciousness level based on agent performance
        if agent_results:
            consciousness_scores = []
            for result in agent_results.values():
                if isinstance(result, dict) and 'consciousness_score' in result:
                    consciousness_scores.append(result['consciousness_score'])
            
            if consciousness_scores:
                self.state.consciousness_level = sum(consciousness_scores) / len(consciousness_scores)
        
        # Calculate collective intelligence based on all results
        all_results = [agent_results, memory_results, neural_results, orchestration_results]
        success_count = sum(1 for results in all_results if results)
        self.state.collective_intelligence = success_count / len(all_results)
        
        # Update active workflows count
        self.state.active_workflows = len(orchestration_results)
        
        # Update system load (simplified calculation)
        component_count = len(self.registry.list_components())
        self.state.system_load = min(1.0, component_count / 10.0)  # Normalize to 0-1
    
    # ========================================================================
    # HEALTH MONITORING
    # ========================================================================
    
        async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        pass
        while self.running:
            try:
                # Get health status from all components
                health_results = await self.registry.health_check_all()
                
                # Calculate overall system health
                if health_results:
                    health_scores = [metrics.health_score for metrics in health_results.values()]
                    self.metrics.overall_health_score = sum(health_scores) / len(health_scores)
                    
                    # Count active components
                    active_count = sum(1 for metrics in health_results.values() 
                                     if metrics.status == ComponentStatus.ACTIVE)
                    self.metrics.active_components = active_count
                    self.metrics.total_components = len(health_results)
                
                # Update system metrics
                self.metrics.uptime_seconds = time.time() - self.metrics.start_time.timestamp()
                self.metrics.last_updated = datetime.now()
                
                # Check for critical health issues
                if self.metrics.overall_health_score < 0.3:
                    await self._emit_system_event("critical_health_alert", {
                        "health_score": self.metrics.overall_health_score,
                        "active_components": self.metrics.active_components,
                        "total_components": self.metrics.total_components
                    }, Priority.CRITICAL)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    # ========================================================================
    # EVENT HANDLING
    # ========================================================================
    
        async def _emit_system_event(self, event_type: str, data: Dict[str, Any],
        priority: Priority = Priority.NORMAL) -> None:
        """Emit a system-level event."""
        event = SystemEvent(
            event_type=event_type,
            component_id=self.system_id,
            data=data,
            priority=priority
        )
        await self._event_queue.put(event)
    
        async def _process_events(self) -> None:
        """Process system events."""
        pass
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Process event
                handlers = self._event_handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        print(f"Event handler error: {e}")
                
                # Mark event as done
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Event processing error: {e}")
    
    def subscribe_to_system_events(self, event_type: str, handler: callable) -> None:
        """Subscribe to system events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
        async def _handle_component_error(self, event: SystemEvent) -> None:
        """Handle component error events."""
        print(f"🚨 Component error: {event.component_id} - {event.data}")
        
        # Emit system-level error event
        await self._emit_system_event("component_error", {
            "component_id": event.component_id,
            "error_data": event.data
        }, Priority.HIGH)
    
        async def _handle_component_status_change(self, event: SystemEvent) -> None:
        """Handle component status change events."""
        print(f"📊 Component status change: {event.component_id} - {event.data}")
    
    # ========================================================================
    # LIFECYCLE HOOKS
    # ========================================================================
    
    def add_startup_hook(self, hook: callable) -> None:
        """Add startup hook."""
        self._startup_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: callable) -> None:
        """Add shutdown hook."""
        self._shutdown_hooks.append(hook)
    
        async def _run_startup_hooks(self) -> None:
        """Run all startup hooks."""
        pass
        for hook in self._startup_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            except Exception as e:
                    print(f"Startup hook error: {e}")
    
        async def _run_shutdown_hooks(self) -> None:
        """Run all shutdown hooks."""
        pass
        for hook in self._shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            except Exception as e:
                    print(f"Shutdown hook error: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _update_cycle_metrics(self, cycle_time_ms: float, success: bool) -> None:
        """Update cycle metrics."""
        self.metrics.total_cycles += 1
        
        if success:
            self.metrics.successful_cycles += 1
            
            # Update running average
            current_avg = self.metrics.average_cycle_time_ms
            count = self.metrics.successful_cycles
            self.metrics.average_cycle_time_ms = (
                (current_avg * (count - 1) + cycle_time_ms) / count
            )
        else:
            self.metrics.failed_cycles += 1
    
    def _calculate_cycle_wait_time(self) -> float:
        """Calculate adaptive wait time between cycles."""
        pass
        base_interval = self.config.agents.cycle_interval
        
        # Adjust based on system load
        load_factor = 1.0 + (self.state.system_load * 0.5)
        
        # Adjust based on health
        health_factor = 2.0 - self.metrics.overall_health_score
        
        return base_interval * load_factor * health_factor
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        pass
        return {
            'system_id': self.system_id,
            'status': self.state.status.value,
            'running': self.running,
            'state': {
                'consciousness_level': self.state.consciousness_level,
                'collective_intelligence': self.state.collective_intelligence,
                'system_load': self.state.system_load,
                'active_workflows': self.state.active_workflows,
                'pending_events': self._event_queue.qsize()
            },
            'metrics': {
                'uptime_seconds': self.metrics.uptime_seconds,
                'total_cycles': self.metrics.total_cycles,
                'success_rate': self.metrics.success_rate,
                'average_cycle_time_ms': self.metrics.average_cycle_time_ms,
                'overall_health_score': self.metrics.overall_health_score,
                'active_components': self.metrics.active_components,
                'total_components': self.metrics.total_components
            },
            'components': self.registry.list_components(),
            'configuration': self.config.to_dict(include_secrets=False)
        }

# ============================================================================
# GLOBAL SYSTEM INSTANCE
# ============================================================================

# Global system instance
_global_system: Optional[UnifiedSystem] = None

    def get_unified_system() -> UnifiedSystem:
        """Get the global unified system instance."""
        global _global_system
        if _global_system is None:
        _global_system = UnifiedSystem()
        return _global_system

def create_unified_system(config: Optional[UnifiedConfig] = None, 
                         system_id: Optional[str] = None) -> UnifiedSystem:
    """Create a new unified system instance."""
    return UnifiedSystem(config=config, system_id=system_id)