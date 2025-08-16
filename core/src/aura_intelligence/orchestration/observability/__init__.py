"""
🎯 Observability Module - 2025 Architecture

Modular, functional-first observability system with:
- Pure functional composition
- Effect-based side effect management  
- Protocol-based polymorphism
- Immutable data structures
- Zero-dependency core with optional integrations
"""

from .core import Effect, MetricPoint, SpanContext, pure_metric
from .collectors import create_arize_collector, create_langsmith_collector
from .tracers import create_jaeger_tracer, create_otel_tracer
from .hybrid import HybridObservability, create_hybrid_stack

__all__ = [
    'Effect', 'MetricPoint', 'SpanContext', 'pure_metric',
    'create_arize_collector', 'create_langsmith_collector',
    'create_jaeger_tracer', 'create_otel_tracer',
    'HybridObservability', 'create_hybrid_stack'
]