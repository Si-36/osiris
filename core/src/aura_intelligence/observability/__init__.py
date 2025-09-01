"""
Observability Module - Enhanced with GPU Monitoring
"""

import structlog
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)

# Import enhanced observability components
try:
    from .enhanced_observability import (
        EnhancedObservabilitySystem,
        get_observability_system,
        PerformanceProfile,
        SystemHealth
    )
    from .gpu_monitoring import (
        GPUMonitor,
        get_gpu_monitor,
        GPUMetrics,
        get_gpu_utilization,
        get_gpu_memory_usage
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logger.warning("Enhanced observability not available, using simplified version")

# Simple implementations for fallback
def create_tracer(name: str):
    """Create a simple tracer."""
    if ENHANCED_AVAILABLE:
        obs = get_observability_system()
        return obs.tracer
    return SimpleTracer(name)

def create_meter(name: str):
    """Create a simple meter."""
    return SimpleMeter(name)

class SimpleTracer:
    def __init__(self, name: str):
        self.name = name
    
    def start_span(self, name: str, **kwargs):
        return SimpleSpan(name)

class SimpleSpan:
    def __init__(self, name: str):
        self.name = name
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def set_attribute(self, key: str, value: Any):
        pass

class SimpleMeter:
    def __init__(self, name: str):
        self.name = name
        
    def create_counter(self, name: str, **kwargs):
        return SimpleCounter(name)
        
    def create_histogram(self, name: str, **kwargs):
        return SimpleHistogram(name)
    
    def create_gauge(self, name: str, **kwargs):
        return SimpleGauge(name)

class SimpleCounter:
    def __init__(self, name: str):
        self.name = name
        self.value = 0
        
    def add(self, value: int = 1, **kwargs):
        self.value += value

class SimpleHistogram:
    def __init__(self, name: str):
        self.name = name
        
    def record(self, value: float, **kwargs):
        pass

class SimpleGauge:
    def __init__(self, name: str):
        self.name = name
        self.value = 0.0
        
    def set(self, value: float, **kwargs):
        self.value = value
    
    def get(self):
        return self.value

# Simple observability core
class NeuralObservabilityCore:
    def __init__(self):
        self.initialized = False
        self._enhanced_system = None
        
    async def initialize(self):
        self.initialized = True
        
        # Try to use enhanced system if available
        if ENHANCED_AVAILABLE:
            try:
                self._enhanced_system = get_observability_system()
                await self._enhanced_system.initialize()
                logger.info("Observability initialized (enhanced mode with GPU monitoring)")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced observability: {e}")
                self._enhanced_system = None
        else:
            logger.info("Observability initialized (simplified mode)")
        
    @asynccontextmanager
    async def observe_workflow(self, state: Any, workflow_type: str = "default"):
        if self._enhanced_system:
            async with self._enhanced_system.trace_operation(
                component="workflow",
                operation=workflow_type,
                adapter=getattr(state, 'adapter', None)
            ) as span:
                yield {"workflow_id": str(span.get_span_context().span_id)}
        else:
            yield {"workflow_id": "test"}
        
    @asynccontextmanager  
    async def observe_agent_call(self, agent_name: str, tool_name: str, inputs: Dict[str, Any] = None):
        if self._enhanced_system:
            async with self._enhanced_system.trace_operation(
                component=agent_name,
                operation=tool_name,
                adapter="agent"
            ) as span:
                yield {"agent": agent_name, "tool": tool_name, "span": span}
        else:
            yield {"agent": agent_name, "tool": tool_name}
            
    async def get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current GPU metrics if available"""
        if self._enhanced_system and self._enhanced_system.gpu_monitor:
            return await self._enhanced_system.gpu_monitor.get_metrics_summary()
        return None
        
    async def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Get performance summary if available"""
        if self._enhanced_system:
            return await self._enhanced_system.get_performance_summary()
        return None

# Export
__all__ = [
    'create_tracer',
    'create_meter',
    'NeuralObservabilityCore',
    'logger',
]

# Export enhanced components if available
if ENHANCED_AVAILABLE:
    __all__.extend([
        'EnhancedObservabilitySystem',
        'get_observability_system',
        'GPUMonitor',
        'get_gpu_monitor',
        'GPUMetrics',
        'get_gpu_utilization',
        'get_gpu_memory_usage',
        'PerformanceProfile',
        'SystemHealth'
    ])