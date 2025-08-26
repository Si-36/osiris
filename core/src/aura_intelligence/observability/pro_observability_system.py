"""
Professional Observability System
=================================

Enterprise-grade observability with:
- OpenTelemetry for distributed tracing
- Prometheus metrics with custom collectors
- Jaeger integration for trace visualization
- Structured logging with context propagation
- Real-time dashboards and alerts
- Performance profiling
- Error tracking and analysis
- SLO/SLI monitoring
"""

import asyncio
import time
import json
import logging
import sys
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, TypeVar, Tuple, Set
import functools
import inspect
from collections import defaultdict, deque
import traceback
import uuid

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.metrics import MeterProvider, Histogram, Counter, UpDownCounter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.context import attach, detach
from opentelemetry.metrics import CallbackOptions, Observation

# Prometheus imports
from prometheus_client import (
    Counter as PromCounter,
    Histogram as PromHistogram,
    Gauge as PromGauge,
    Summary as PromSummary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    start_http_server
)

# Structured logging
import structlog
from pythonjsonlogger import jsonlogger

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Global tracer and meter instances
tracer: Optional[trace.Tracer] = None
meter: Optional[metrics.Meter] = None
logger: Optional[structlog.BoundLogger] = None


class LogLevel(Enum):
    """Log levels with numeric values"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class SpanAttributes:
    """Common span attributes"""
    SERVICE_NAME: str = "aura.intelligence"
    SERVICE_VERSION: str = "2.0.0"
    DEPLOYMENT_ENV: str = "production"
    
    # HTTP attributes
    HTTP_METHOD: str = "http.method"
    HTTP_URL: str = "http.url"
    HTTP_STATUS_CODE: str = "http.status_code"
    HTTP_ROUTE: str = "http.route"
    
    # Database attributes
    DB_SYSTEM: str = "db.system"
    DB_OPERATION: str = "db.operation"
    DB_STATEMENT: str = "db.statement"
    
    # Message queue attributes
    MESSAGING_SYSTEM: str = "messaging.system"
    MESSAGING_OPERATION: str = "messaging.operation"
    MESSAGING_DESTINATION: str = "messaging.destination"
    
    # Custom AURA attributes
    AURA_COMPONENT: str = "aura.component"
    AURA_OPERATION: str = "aura.operation"
    AURA_AGENT_ID: str = "aura.agent.id"
    AURA_WORKFLOW_ID: str = "aura.workflow.id"
    AURA_TOPOLOGY_SCORE: str = "aura.topology.score"
    AURA_PREDICTION_CONFIDENCE: str = "aura.prediction.confidence"


@dataclass
class MetricDefinition:
    """Metric definition with metadata"""
    name: str
    description: str
    unit: str = "1"
    value_type: str = "float"
    tags: List[str] = field(default_factory=list)


class ObservabilitySystem:
    """Comprehensive observability system for AURA"""
    
    def __init__(
        self,
        service_name: str = "aura-intelligence",
        service_version: str = "2.0.0",
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        prometheus_port: int = 8000,
        enable_console_export: bool = False,
        log_level: LogLevel = LogLevel.INFO
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.attributes = SpanAttributes()
        
        # Initialize components
        self._init_tracing(jaeger_endpoint, enable_console_export)
        self._init_metrics(prometheus_port)
        self._init_logging(log_level)
        
        # Performance profiling
        self.profile_data: Dict[str, List[float]] = defaultdict(list)
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_samples: deque = deque(maxlen=100)
        
        # SLO/SLI tracking
        self.slo_targets: Dict[str, float] = {}
        self.sli_measurements: Dict[str, List[float]] = defaultdict(list)
        
        # Custom collectors
        self._setup_custom_collectors()
        
        global tracer, meter, logger
        tracer = self.tracer
        meter = self.meter
        logger = self.logger
        
        self.logger.info(
            "Observability system initialized",
            service=service_name,
            version=service_version
        )
    
    def _init_tracing(self, jaeger_endpoint: str, enable_console: bool):
        """Initialize distributed tracing"""
        # Create resource
        resource = Resource.create({
        SERVICE_NAME: self.service_name,
        SERVICE_VERSION: self.service_version,
        "deployment.environment": self.attributes.DEPLOYMENT_ENV,
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "host.name": self._get_hostname(),
        })
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
        collector_endpoint=jaeger_endpoint,
        max_tag_value_length=2048
        )
        self.tracer_provider.add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
        )
        
        # Add console exporter for debugging
        if enable_console:
            self.tracer_provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
        )
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(
        self.service_name,
        self.service_version
        )
    
    def _init_metrics(self, prometheus_port: int):
            """Initialize metrics collection"""
        # Create meter provider with Prometheus exporter
        reader = PrometheusMetricReader()
        self.meter_provider = MeterProvider(
            resource=Resource.create({
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
            }),
            metric_readers=[reader]
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(
            self.service_name,
            self.service_version
        )
        
        # Create standard metrics
        self._create_standard_metrics()
        
        # Start Prometheus HTTP server
        start_http_server(prometheus_port)
        
        # Create Prometheus registry for custom metrics
        self.prom_registry = CollectorRegistry()
        self._create_prometheus_metrics()
    
    def _init_logging(self, log_level: LogLevel):
        """Initialize structured logging"""
        # Configure structured logging
        structlog.configure(
        processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        self._add_trace_context,
        structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
        )
        
        # Set up JSON formatter for standard logging
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        
        # Configure root logger
        logging.root.handlers = [logHandler]
        logging.root.setLevel(log_level.value)
        
        # Get structured logger
        self.logger = structlog.get_logger()
        
        # Instrument logging with OpenTelemetry
        LoggingInstrumentor().instrument(set_logging_format=True)
    
    def _add_trace_context(self, logger, method_name, event_dict):
            """Add trace context to logs"""
        pass
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            event_dict['trace_id'] = format(span_context.trace_id, '032x')
            event_dict['span_id'] = format(span_context.span_id, '016x')
            event_dict['trace_flags'] = format(span_context.trace_flags, '02x')
        return event_dict
    
    def _create_standard_metrics(self):
        """Create standard application metrics"""
        pass
        # Request metrics
        self.request_counter = self.meter.create_counter(
        name="requests_total",
        description="Total number of requests",
        unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
        name="request_duration_seconds",
        description="Request duration in seconds",
        unit="s"
        )
        
        # Error metrics
        self.error_counter = self.meter.create_counter(
        name="errors_total",
        description="Total number of errors",
        unit="1"
        )
        
        # Business metrics
        self.cascade_predictions = self.meter.create_counter(
        name="cascade_predictions_total",
        description="Total cascade failure predictions",
        unit="1"
        )
        
        self.topology_computations = self.meter.create_histogram(
        name="topology_computation_duration_seconds",
        description="Topology computation duration",
        unit="s"
        )
        
        # System metrics
        self.active_agents = self.meter.create_up_down_counter(
        name="active_agents",
        description="Number of active agents",
        unit="1"
        )
        
        # Async gauge for system health
        self.meter.create_observable_gauge(
        name="system_health_score",
        callbacks=[self._observe_system_health],
        description="Overall system health score (0-100)",
        unit="1"
        )
    
    def _create_prometheus_metrics(self):
            """Create Prometheus-specific metrics"""
        pass
        self.prom_request_duration = PromHistogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'status'],
            registry=self.prom_registry
        )
        
        self.prom_active_spans = PromGauge(
            'active_spans_total',
            'Number of active spans',
            ['operation'],
            registry=self.prom_registry
        )
        
        self.prom_error_rate = PromGauge(
            'error_rate_per_minute',
            'Error rate per minute',
            ['error_type'],
            registry=self.prom_registry
        )
    
    def _observe_system_health(self, options: CallbackOptions) -> List[Observation]:
        """Callback for system health metric"""
        # Calculate health score based on various factors
        health_score = 100.0
        
        # Factor in error rate
        recent_errors = sum(1 for e in self.error_samples if 
        time.time() - e.get('timestamp', 0) < 300)
        if recent_errors > 0:
            health_score -= min(recent_errors * 5, 50)
        
        # Factor in SLO compliance
        for slo_name, target in self.slo_targets.items():
        measurements = self.sli_measurements.get(slo_name, [])
        if measurements:
            compliance = sum(1 for m in measurements if m >= target) / len(measurements)
        if compliance < 0.95:  # 95% SLO target
        health_score -= (1 - compliance) * 20
        
        return [Observation(value=max(0, health_score))]
    
    def _setup_custom_collectors(self):
            """Setup custom metric collectors"""
        pass
        # Topology metrics collector
        self.meter.create_observable_counter(
            name="topology_changes_total",
            callbacks=[self._collect_topology_changes],
            description="Total topology changes detected",
            unit="1"
        )
        
        # Agent metrics collector
        self.meter.create_observable_gauge(
            name="agent_health_scores",
            callbacks=[self._collect_agent_health],
            description="Health scores of individual agents",
            unit="1"
        )
    
    def _collect_topology_changes(self, options: CallbackOptions) -> List[Observation]:
        """Collect topology change metrics"""
        # This would connect to actual topology monitoring
        # For now, return simulated data
        return [Observation(value=42, attributes={"change_type": "node_addition"})]
    
    def _collect_agent_health(self, options: CallbackOptions) -> List[Observation]:
        """Collect agent health metrics"""
        # This would connect to actual agent monitoring
        # For now, return simulated data
        observations = []
        for i in range(5):
            observations.append(
                Observation(
                    value=85 + i,
                    attributes={"agent_id": f"agent_{i}"}
                )
            )
        return observations
    
    def _get_hostname(self) -> str:
        """Get hostname for resource attributes"""
        pass
        import socket
        return socket.gethostname()
    
        # Decorator for tracing functions
        def trace(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
        ):
        """Decorator to trace function execution"""
    def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
                
            with self.tracer.start_as_current_span(
            span_name,
            kind=kind,
            attributes=attributes or {}
            ) as span:
            try:
                # Add function metadata
            span.set_attribute("function.name", func.__name__)
            span.set_attribute("function.module", func.__module__)
                        
            # Execute function
            result = await func(*args, **kwargs)
                        
            # Mark success
            span.set_status(Status(StatusCode.OK))
                        
            return result
                        
            except Exception as e:
            if record_exception:
                span.record_exception(e)
            span.set_status(
            Status(StatusCode.ERROR, str(e))
            )
            self._record_error(e, span_name)
            raise
            
            @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
                
            with self.tracer.start_as_current_span(
            span_name,
            kind=kind,
            attributes=attributes or {}
            ) as span:
            try:
                # Add function metadata
            span.set_attribute("function.name", func.__name__)
            span.set_attribute("function.module", func.__module__)
                        
            # Execute function
            result = func(*args, **kwargs)
                        
            # Mark success
            span.set_status(Status(StatusCode.OK))
                        
            return result
                        
            except Exception as e:
            if record_exception:
                span.record_exception(e)
            span.set_status(
            Status(StatusCode.ERROR, str(e))
            )
            self._record_error(e, span_name)
            raise
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
            return sync_wrapper
        
            return decorator
    
            # Context manager for tracing
            @contextmanager
            def trace_operation(
            self,
            name: str,
            kind: SpanKind = SpanKind.INTERNAL,
            attributes: Optional[Dict[str, Any]] = None
            ):
            """Context manager for tracing operations"""
            with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {}
            ) as span:
            start_time = time.time()
            try:
                yield span
            span.set_status(Status(StatusCode.OK))
            except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            self._record_error(e, name)
            raise
            finally:
            # Record duration
            duration = time.time() - start_time
            self.profile_data[name].append(duration)
                
            # Update metrics
            self.request_duration.record(
            duration,
            {"operation": name}
            )
    
    def _record_error(self, error: Exception, operation: str):
                """Record error for tracking"""
            error_type = type(error).__name__
            self.error_counts[error_type] += 1
        
            # Sample error details
            self.error_samples.append({
            "timestamp": time.time(),
            "operation": operation,
            "error_type": error_type,
            "error_message": str(error),
            "traceback": traceback.format_exc()
            })
        
            # Update metrics
            self.error_counter.add(1, {"error_type": error_type})
        
            # Log error
            self.logger.error(
            "Operation failed",
            operation=operation,
            error_type=error_type,
            error_message=str(error),
            exc_info=True
            )
    
            # SLO/SLI monitoring
    def define_slo(self, name: str, target: float, unit: str = "ratio"):
                """Define a Service Level Objective"""
            self.slo_targets[name] = target
            self.logger.info(
            "SLO defined",
            slo_name=name,
            target=target,
            unit=unit
            )
    
    def record_sli(self, name: str, value: float):
                """Record a Service Level Indicator measurement"""
            if name not in self.slo_targets:
                self.logger.warning(f"SLI recorded for undefined SLO: {name}")
        
            self.sli_measurements[name].append(value)
        
            # Keep only recent measurements (last 1000)
            if len(self.sli_measurements[name]) > 1000:
                self.sli_measurements[name] = self.sli_measurements[name][-1000:]
        
            # Check SLO compliance
            target = self.slo_targets.get(name, 0)
            if value < target:
                self.logger.warning(
            "SLO violation",
            slo_name=name,
            target=target,
            actual=value
            )
    
    def get_slo_report(self) -> Dict[str, Dict[str, Any]]:
            """Get SLO compliance report"""
            pass
            report = {}
        
            for slo_name, target in self.slo_targets.items():
            measurements = self.sli_measurements.get(slo_name, [])
            
            if measurements:
                compliance = sum(1 for m in measurements if m >= target) / len(measurements)
            report[slo_name] = {
            "target": target,
            "compliance_rate": compliance,
            "measurements": len(measurements),
            "current_value": measurements[-1] if measurements else None,
            "min_value": min(measurements),
            "max_value": max(measurements),
            "avg_value": sum(measurements) / len(measurements)
            }
            else:
            report[slo_name] = {
            "target": target,
            "compliance_rate": None,
            "measurements": 0
            }
        
            return report
    
            # Performance profiling
    def get_performance_report(self, top_n: int = 10) -> Dict[str, Dict[str, float]]:
            """Get performance profiling report"""
            report = {}
        
            for operation, durations in self.profile_data.items():
            if durations:
                report[operation] = {
            "count": len(durations),
            "total": sum(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "p50": self._percentile(durations, 50),
            "p95": self._percentile(durations, 95),
            "p99": self._percentile(durations, 99)
            }
        
            # Sort by total time and return top N
            sorted_ops = sorted(
            report.items(),
            key=lambda x: x[1]["total"],
            reverse=True
            )
        
            return dict(sorted_ops[:top_n])
    
    def _percentile(self, data: List[float], percentile: float) -> float:
            """Calculate percentile"""
            if not data:
                return 0.0
        
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
    
            # Error analysis
    def get_error_report(self) -> Dict[str, Any]:
            """Get error analysis report"""
            pass
            total_errors = sum(self.error_counts.values())
        
            report = {
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "error_rate": total_errors / max(1, sum(len(d) for d in self.profile_data.values())),
            "recent_errors": list(self.error_samples)[-10:],  # Last 10 errors
            "error_distribution": {
            error_type: count / total_errors
            for error_type, count in self.error_counts.items()
            } if total_errors > 0 else {}
            }
        
            return report
    
            # Baggage propagation
    def set_baggage(self, key: str, value: str):
                """Set baggage for context propagation"""
            baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
            """Get baggage value"""
            return baggage.get_baggage(key)
    
            # Shutdown
    def shutdown(self):
                """Shutdown observability system"""
            pass
            self.logger.info("Shutting down observability system")
        
            # Flush and shutdown tracing
            if hasattr(self, 'tracer_provider'):
                self.tracer_provider.shutdown()
        
            # Shutdown metrics
            if hasattr(self, 'meter_provider'):
                self.meter_provider.shutdown()
        
            # Log final reports
            self.logger.info("Performance report", **self.get_performance_report())
            self.logger.info("Error report", **self.get_error_report())
            self.logger.info("SLO report", **self.get_slo_report())


    # Convenience functions for global access
    def get_tracer() -> Optional[trace.Tracer]:
        """Get global tracer instance"""
        return tracer


    def get_meter() -> Optional[metrics.Meter]:
        """Get global meter instance"""
        return meter


    def get_logger() -> Optional[structlog.BoundLogger]:
        """Get global logger instance"""
        return logger


    # Example instrumentation
class InstrumentedAuraComponent:
    """Example of an instrumented AURA component"""
    
    def __init__(self, component_name: str, observability: ObservabilitySystem):
        self.component_name = component_name
        self.obs = observability
        
        # Component-specific metrics
        self.operation_counter = self.obs.meter.create_counter(
        f"{component_name}_operations_total",
        description=f"Total operations for {component_name}"
        )
    
        @property
    def logger(self):
            """Get logger bound with component context"""
        pass
        return self.obs.logger.bind(component=self.component_name)
    
        async def process_topology(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Example instrumented method"""
        with self.obs.trace_operation(
        "process_topology",
        kind=SpanKind.INTERNAL,
        attributes={
        self.obs.attributes.AURA_COMPONENT: self.component_name,
        self.obs.attributes.AURA_OPERATION: "topology_analysis"
        }
        ) as span:
        # Log start
        self.logger.info("Processing topology", data_size=len(data))
            
        # Set span attributes
        span.set_attribute("topology.nodes", data.get("nodes", 0))
        span.set_attribute("topology.edges", data.get("edges", 0))
            
        # Simulate processing
        await asyncio.sleep(0.1)
            
        # Record metrics
        self.operation_counter.add(1, {"operation": "topology_analysis"})
            
        # Simulate result
        result = {
        "betti_numbers": {"b0": 1, "b1": 5, "b2": 0},
        "risk_score": 0.7
        }
            
        # Record SLI
        self.obs.record_sli("topology_processing_accuracy", 0.95)
            
        # Log completion
        self.logger.info(
        "Topology processed",
        risk_score=result["risk_score"],
        betti_numbers=result["betti_numbers"]
        )
            
        return result


    # Example usage
async def test_observability():
        """Test observability system"""
    # Initialize system
        obs = ObservabilitySystem(
        service_name="aura-test",
        enable_console_export=True,
        log_level=LogLevel.INFO
        )
    
    # Define SLOs
        obs.define_slo("topology_processing_accuracy", 0.9)
        obs.define_slo("prediction_latency_ms", 100, "ms")
    
    # Create instrumented component
        component = InstrumentedAuraComponent("tda_analyzer", obs)
    
    # Simulate operations
        for i in range(5):
        try:
        # Process with tracing
        result = await component.process_topology({
        "nodes": 50 + i * 10,
        "edges": 100 + i * 20
        })
            
    # Record latency SLI
        obs.record_sli("prediction_latency_ms", 50 + i * 10)
            
        except Exception as e:
        obs.logger.error("Test operation failed", error=str(e))
    
    # Simulate an error
        try:
        raise ValueError("Simulated error for testing")
        except Exception as e:
        obs._record_error(e, "test_operation")
    
    # Get reports
        print("\n📊 Performance Report:")
        print(json.dumps(obs.get_performance_report(), indent=2))
    
        print("\n❌ Error Report:")
        print(json.dumps(obs.get_error_report(), indent=2))
    
        print("\n📈 SLO Report:")
        print(json.dumps(obs.get_slo_report(), indent=2))
    
    # Shutdown
        obs.shutdown()


        if __name__ == "__main__":
        asyncio.run(test_observability())