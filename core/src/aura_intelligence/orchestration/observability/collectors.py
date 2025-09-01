"""
📊 Metric Collectors - External Platform Adapters

Functional adapters for external observability platforms using 2025 patterns:
    pass
- Protocol-based polymorphism over inheritance
- Pure functions with effect composition
- Graceful degradation with Maybe/Option types
- Zero-cost abstractions
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Protocol
from dataclasses import dataclass
import asyncio

from .core import Effect, MetricPoint, MetricCollector

# Optional external dependencies with graceful fallbacks
try:
    import phoenix as px
    ARIZE_AVAILABLE = True
except ImportError:
    ARIZE_AVAILABLE = False
    px = None

try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None

@dataclass(frozen=True, slots=True)
class ArizeConfig:
    """Immutable Arize configuration"""
    project_name: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

@dataclass(frozen=True, slots=True)
class LangSmithConfig:
    """Immutable LangSmith configuration"""
    project_name: str
    api_key: Optional[str] = None

class ArizeCollector:
    """Arize Phoenix metric collector"""
    __slots__ = ('_config', '_client')
    
    def __init__(self, config: ArizeConfig):
        self._config = config
        self._client = px if ARIZE_AVAILABLE else None
    
    def collect(self, point: MetricPoint) -> Effect[None]:
        """Collect metric to Arize Phoenix"""
        async def _collect():
            if not self._client:
                return  # Graceful degradation
            
            # Transform to Arize format
            arize_data = {
            'metric_name': point.name,
            'value': point.value,
            'timestamp': point.timestamp.isoformat(),
            'tags': dict(point.tags),
            'project': self._config.project_name
            }
            
            # Send to Arize (mock implementation)
            await asyncio.sleep(0.001)  # Simulate network call
            
            return Effect(_collect)

class LangSmithCollector:
    """LangSmith metric collector"""
    __slots__ = ('_config', '_client')
    
    def __init__(self, config: LangSmithConfig):
        self._config = config
        self._client = LangSmithClient() if LANGSMITH_AVAILABLE else None
    
    def collect(self, point: MetricPoint) -> Effect[None]:
        """Collect metric to LangSmith"""
        async def _collect():
            if not self._client:
                return  # Graceful degradation
            
            # Transform to LangSmith format
            langsmith_data = {
            'name': point.name,
            'value': point.value,
            'timestamp': point.timestamp,
            'metadata': dict(point.tags),
            'project_name': self._config.project_name
            }
            
            # Send to LangSmith (mock implementation)
            await asyncio.sleep(0.001)  # Simulate network call
            
            return Effect(_collect)

class NoOpCollector:
    """No-operation collector for graceful degradation"""
    __slots__ = ()
    
    def collect(self, point: MetricPoint) -> Effect[None]:
        """No-op collection"""
        return Effect(lambda: asyncio.sleep(0))

    # Factory functions (pure, functional style)
    def create_arize_collector(config: ArizeConfig) -> MetricCollector:
        """Create Arize collector with graceful fallback"""
        return ArizeCollector(config) if ARIZE_AVAILABLE else NoOpCollector()

    def create_langsmith_collector(config: LangSmithConfig) -> MetricCollector:
        """Create LangSmith collector with graceful fallback"""
        return LangSmithCollector(config) if LANGSMITH_AVAILABLE else NoOpCollector()

    def create_multi_collector(*collectors: MetricCollector) -> MetricCollector:
        """Compose multiple collectors into one"""
    class MultiCollector:
        pass
    __slots__ = ('_collectors',)
        
        def __init__(self, collectors: tuple[MetricCollector, ...]):
            self._collectors = collectors
        
        def collect(self, point: MetricPoint) -> Effect[None]:
            """Collect to all collectors in parallel"""
            async def _collect_all():
                pass
            effects = [collector.collect(point) for collector in self._collectors]
            await asyncio.gather(*[effect.run() for effect in effects])
            
            return Effect(_collect_all)
    
            return MultiCollector(collectors)
