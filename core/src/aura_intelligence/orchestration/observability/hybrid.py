"""
🔄 Hybrid Observability - Best of Both Worlds

Combines custom orchestration monitoring with external platform integration:
- Functional composition of observability strategies
- Effect-based side effect management
- Protocol-based extensibility
- Zero-cost abstractions with compile-time optimization
"""

from __future__ import annotations
from typing import Protocol, Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio

from .core import Effect, MetricPoint, SpanContext, MetricCollector, SpanTracer
from .collectors import ArizeConfig, LangSmithConfig, create_arize_collector, create_langsmith_collector, create_multi_collector
from .tracers import JaegerConfig, create_jaeger_tracer, create_span_manager

@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Immutable observability configuration"""
    arize: Optional[ArizeConfig] = None
    langsmith: Optional[LangSmithConfig] = None
    jaeger: Optional[JaegerConfig] = None
    custom_collectors: tuple[MetricCollector, ...] = ()
    custom_tracers: tuple[SpanTracer, ...] = ()

@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """Immutable workflow event"""
    workflow_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

class ObservabilityStrategy(Protocol):
    """Protocol for observability strategies"""
    async def record_event(self, event: WorkflowEvent) -> None: ...
    async def record_metric(self, metric: MetricPoint) -> None: ...
    async def start_trace(self, name: str) -> SpanContext: ...

class HybridObservability:
    """Hybrid observability combining multiple strategies"""
    __slots__ = ('_collector', '_tracer', '_event_handlers')
    
    def __init__(
        self,
        collector: MetricCollector,
        tracer: SpanTracer,
        event_handlers: tuple[Callable[[WorkflowEvent], Effect[None]], ...] = ()
    ):
        self._collector = collector
        self._tracer = tracer
        self._event_handlers = event_handlers
    
    async def record_workflow_start(
        self,
        workflow_id: str,
        operation_name: str,
        correlation_id: Optional[str] = None
    ) -> SpanContext:
        """Record workflow start with hybrid observability"""
        # Create workflow event
        event = WorkflowEvent(
            workflow_id=workflow_id,
            event_type="workflow_start",
            timestamp=datetime.now(timezone.utc),
            data={"operation": operation_name},
            correlation_id=correlation_id
        )
        
        # Start distributed trace
        span_effect = self._tracer.start_span(operation_name)
        span_context = await span_effect.run()
        
        # Record metric
        metric = MetricPoint(
            name="workflow_started",
            value=1.0,
            timestamp=event.timestamp
        ).with_tag("operation", operation_name).with_tag("workflow_id", workflow_id)
        
        metric_effect = self._collector.collect(metric)
        
        # Process event handlers
        handler_effects = [handler(event) for handler in self._event_handlers]
        
        # Execute all effects in parallel
        await asyncio.gather(
            metric_effect.run(),
            *[effect.run() for effect in handler_effects]
        )
        
        return span_context
    
    async def record_workflow_end(
        self,
        workflow_id: str,
        span_context: SpanContext,
        status: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record workflow completion with hybrid observability"""
        # Create workflow event
        event = WorkflowEvent(
            workflow_id=workflow_id,
            event_type="workflow_end",
            timestamp=datetime.now(timezone.utc),
            data={
                "status": status,
                "duration_ms": duration_ms,
                "span_id": span_context.span_id,
                **(metadata or {})
            }
        )
        
        # Record metrics
        duration_metric = MetricPoint(
            name="workflow_duration",
            value=duration_ms / 1000,  # Convert to seconds
            timestamp=event.timestamp
        ).with_tag("status", status).with_tag("workflow_id", workflow_id)
        
        status_metric = MetricPoint(
            name=f"workflow_{status}",
            value=1.0,
            timestamp=event.timestamp
        ).with_tag("workflow_id", workflow_id)
        
        # Execute effects
        await asyncio.gather(
            self._collector.collect(duration_metric).run(),
            self._collector.collect(status_metric).run(),
            *[handler(event).run() for handler in self._event_handlers]
        )
    
    async def record_llm_call(
        self,
        workflow_id: str,
        model: str,
        tokens: int,
        latency_ms: float,
        cost: float = 0.0
    ) -> None:
        """Record LLM call optimized for Arize"""
        event = WorkflowEvent(
            workflow_id=workflow_id,
            event_type="llm_call",
            timestamp=datetime.now(timezone.utc),
            data={
                "model": model,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "cost": cost
            }
        )
        
        # Create LLM-specific metrics
        metrics = [
            MetricPoint("llm_tokens", float(tokens), event.timestamp)
            .with_tag("model", model).with_tag("workflow_id", workflow_id),
            
            MetricPoint("llm_latency", latency_ms / 1000, event.timestamp)
            .with_tag("model", model).with_tag("workflow_id", workflow_id),
            
            MetricPoint("llm_cost", cost, event.timestamp)
            .with_tag("model", model).with_tag("workflow_id", workflow_id)
        ]
        
        # Execute in parallel
        await asyncio.gather(
            *[self._collector.collect(metric).run() for metric in metrics],
            *[handler(event).run() for handler in self._event_handlers]
        )
    
    def with_event_handler(
        self,
        handler: Callable[[WorkflowEvent], Effect[None]]
    ) -> HybridObservability:
        """Add event handler (immutable update)"""
        return HybridObservability(
            self._collector,
            self._tracer,
            (*self._event_handlers, handler)
        )

# Factory function for creating hybrid observability stack
    def create_hybrid_stack(config: ObservabilityConfig) -> HybridObservability:
        """Create hybrid observability stack from configuration"""
        collectors = []
    
    # Add external collectors
        if config.arize:
        collectors.append(create_arize_collector(config.arize))
    
        if config.langsmith:
        collectors.append(create_langsmith_collector(config.langsmith))
    
    # Add custom collectors
        collectors.extend(config.custom_collectors)
    
    # Create multi-collector
        collector = create_multi_collector(*collectors) if collectors else create_multi_collector()
    
    # Create tracer (prefer Jaeger, fallback to first custom tracer)
        tracer = (
        create_jaeger_tracer(config.jaeger) if config.jaeger
        else config.custom_tracers[0] if config.custom_tracers
        else create_jaeger_tracer(JaegerConfig("aura-orchestration"))  # Default
        )
    
        return HybridObservability(collector, tracer)

    # Functional composition helpers
    def compose_event_handlers(*handlers: Callable[[WorkflowEvent], Effect[None]]) -> Callable[[WorkflowEvent], Effect[None]]:
        """Compose multiple event handlers into one"""
    def composed_handler(event: WorkflowEvent) -> Effect[None]:
        async def _handle_all():
        effects = [handler(event) for handler in handlers]
        await asyncio.gather(*[effect.run() for effect in effects])
        
        return Effect(_handle_all)
    
        return composed_handler

    def create_tda_event_handler(tda_client) -> Callable[[WorkflowEvent], Effect[None]]:
        """Create TDA-specific event handler"""
    def tda_handler(event: WorkflowEvent) -> Effect[None]:
        async def _send_to_tda():
        if hasattr(tda_client, 'send_orchestration_result'):
        await tda_client.send_orchestration_result(
        {
        "event_type": event.event_type,
        "workflow_id": event.workflow_id,
        "timestamp": event.timestamp.isoformat(),
        "data": event.data
        },
        event.correlation_id
        )
        
        return Effect(_send_to_tda)
    
        return tda_handler

    # Example usage with functional composition
async def example_hybrid_usage():
        """Example of hybrid observability usage"""
    # Configuration
        config = ObservabilityConfig(
        arize=ArizeConfig(project_name="aura-production"),
        langsmith=LangSmithConfig(project_name="aura-workflows"),
        jaeger=JaegerConfig(service_name="aura-orchestration")
        )
    
    # Create hybrid stack
        observability = create_hybrid_stack(config)
    
    # Add TDA integration
    # observability = observability.with_event_handler(create_tda_event_handler(tda_client))
    
    # Use hybrid observability
        span_context = await observability.record_workflow_start(
        workflow_id="example_workflow",
        operation_name="multi_agent_processing",
        correlation_id="example-correlation-123"
        )
    
    # Record LLM call
        await observability.record_llm_call(
        workflow_id="example_workflow",
        model="gpt-4",
        tokens=150,
        latency_ms=1200,
        cost=0.003
        )
    
    # End workflow
        await observability.record_workflow_end(
        workflow_id="example_workflow",
        span_context=span_context,
        status="success",
        duration_ms=5000,
        metadata={"steps_completed": 4}
        )
