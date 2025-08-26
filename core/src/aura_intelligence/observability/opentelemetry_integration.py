"""
OpenTelemetry integration for AURA Intelligence observability.
Provides tracing, metrics, and logging integration.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union, AsyncContextManager
from contextlib import asynccontextmanager
import asyncio

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# AURA Core imports with fallbacks
try:
    from aura_intelligence.config import ObservabilityConfig
    from .context_managers import ObservabilityContext
    AURA_CORE_AVAILABLE = True
except ImportError:
    # Fallback implementations
    class ObservabilityConfig:
        def __init__(self, **kwargs):
            self.enabled = kwargs.get('enabled', False)
            self.jaeger_endpoint = kwargs.get('jaeger_endpoint', 'http://localhost:14268/api/traces')
            self.prometheus_port = kwargs.get('prometheus_port', 8090)
            self.service_name = kwargs.get('service_name', 'aura-intelligence')
    
    class ObservabilityContext:
        def __init__(self, **kwargs):
            self.workflow_type = kwargs.get('workflow_type', 'unknown')
            self.workflow_id = kwargs.get('workflow_id', 'unknown')
    
            AURA_CORE_AVAILABLE = False

class MockTracer:
    """Mock tracer for testing when OpenTelemetry not available."""
    def start_span(self, name, **kwargs):
        return MockSpan(name)

class MockSpan:
    """Mock span for testing when OpenTelemetry not available."""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_attribute(self, key, value):
        pass
    
    def set_status(self, status):
        pass

class OpenTelemetryManager:
    """Manages OpenTelemetry tracing, metrics, and logging for AURA Intelligence."""
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tracer = None
        self.meter = None
        self.initialized = False
        
        if OPENTELEMETRY_AVAILABLE and self.config.enabled:
            self._setup_opentelemetry()
        else:
            self._setup_mocks()
    
    def _setup_opentelemetry(self):
        """Set up real OpenTelemetry components."""
        try:
            # Set up tracing
            trace.set_tracer_provider(TracerProvider())
            
            # Jaeger exporter
            jaeger_exporter = JaegerExporter(
                endpoint=self.config.jaeger_endpoint
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(self.config.service_name)
            
            # Set up metrics
            metrics.set_meter_provider(MeterProvider())
            self.meter = metrics.get_meter(self.config.service_name)
            
            # Instrument asyncio
            AsyncioInstrumentor().instrument()
            LoggingInstrumentor().instrument()
            
            self.initialized = True
            self.logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self._setup_mocks()
    
    def _setup_mocks(self):
        """Set up mock components when OpenTelemetry is not available."""
        self.tracer = MockTracer()
        self.meter = None
        self.initialized = False
        self.logger.warning("Using mock OpenTelemetry components")
    
    @asynccontextmanager
        async def trace_workflow(self, workflow_name: str, context: ObservabilityContext):
        """Context manager for tracing workflow execution."""
        span_name = f"workflow.{workflow_name}"
        
        if self.tracer:
            with self.tracer.start_span(span_name) as span:
                if hasattr(span, 'set_attribute'):
                    span.set_attribute("workflow.type", context.workflow_type)
                    span.set_attribute("workflow.id", context.workflow_id)
                
                try:
                    yield span
                except Exception as e:
                    if hasattr(span, 'set_status'):
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        else:
            yield None
    
    def record_metric(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a metric value."""
        if self.meter:
            try:
                counter = self.meter.create_counter(name)
                counter.add(value, tags or {})
            except Exception as e:
                self.logger.error(f"Failed to record metric {name}: {e}")
    
        async def shutdown(self):
        """Shutdown OpenTelemetry components."""
        if self.initialized:
            try:
                # Shutdown span processors
                if hasattr(trace.get_tracer_provider(), 'shutdown'):
                    trace.get_tracer_provider().shutdown()
                
                self.logger.info("OpenTelemetry shutdown completed")
            except Exception as e:
                self.logger.error(f"Error during OpenTelemetry shutdown: {e}")