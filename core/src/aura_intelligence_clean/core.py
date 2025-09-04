"""
Core System - Clean Implementation
"""

from typing import Dict, Any, Optional
import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field

class ComponentStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class SystemMetrics:
    uptime: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    component_health: Dict[str, float] = field(default_factory=dict)

class AURACore:
    """Core system manager"""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.status: Dict[str, ComponentStatus] = {}
        self.metrics = SystemMetrics()
        self.start_time = time.time()
        self.running = False
        
    def register_component(self, name: str, component: Any):
        """Register a component"""
        self.components[name] = component
        self.status[name] = ComponentStatus.INITIALIZING
        
    async def start(self):
        """Start the system"""
        self.running = True
        
        # Initialize all components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'initialize'):
                    await component.initialize()
                self.status[name] = ComponentStatus.READY
                self.metrics.component_health[name] = 1.0
            except Exception as e:
                self.status[name] = ComponentStatus.FAILED
                self.metrics.component_health[name] = 0.0
                print(f"Failed to initialize {name}: {e}")
                
        # Start monitoring
        asyncio.create_task(self._monitor_health())
        
    async def stop(self):
        """Stop the system"""
        self.running = False
        
        # Stop all components
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                await component.shutdown()
            self.status[name] = ComponentStatus.STOPPED
            
    async def _monitor_health(self):
        """Monitor system health"""
        while self.running:
            self.metrics.uptime = time.time() - self.start_time
            
            # Check component health
            for name, component in self.components.items():
                if hasattr(component, 'health_check'):
                    try:
                        health = await component.health_check()
                        self.metrics.component_health[name] = health
                    except:
                        self.metrics.component_health[name] = 0.0
                        
            await asyncio.sleep(10)  # Check every 10 seconds
            
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "running": self.running,
            "uptime": self.metrics.uptime,
            "components": self.status,
            "health": self.metrics.component_health,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.successful_requests / max(1, self.metrics.total_requests)
            }
        }

class SelfHealingEngine:
    """Self-healing capabilities"""
    
    def __init__(self, core: AURACore):
        self.core = core
        self.healing_strategies = {
            "restart": self._restart_component,
            "reset": self._reset_component,
            "failover": self._failover_component
        }
        
    async def heal_component(self, component_name: str, strategy: str = "restart"):
        """Heal a failed component"""
        if strategy in self.healing_strategies:
            return await self.healing_strategies[strategy](component_name)
        return False
        
    async def _restart_component(self, name: str) -> bool:
        """Restart a component"""
        if name in self.core.components:
            component = self.core.components[name]
            
            # Stop if running
            if hasattr(component, 'shutdown'):
                await component.shutdown()
                
            # Restart
            if hasattr(component, 'initialize'):
                await component.initialize()
                
            self.core.status[name] = ComponentStatus.READY
            return True
            
        return False
        
    async def _reset_component(self, name: str) -> bool:
        """Reset component state"""
        if name in self.core.components:
            component = self.core.components[name]
            if hasattr(component, 'reset'):
                await component.reset()
                return True
        return False
        
    async def _failover_component(self, name: str) -> bool:
        """Failover to backup (mock)"""
        # In real system, would switch to backup instance
        return await self._restart_component(name)