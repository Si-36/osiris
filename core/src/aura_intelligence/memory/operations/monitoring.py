"""
Memory System Monitoring - OTEL Spans and Prometheus Metrics
==========================================================

Production monitoring for our revolutionary memory system with:
- OpenTelemetry spans for distributed tracing
- Prometheus metrics for performance monitoring
- Grafana-ready dashboards
- Real-time alerting
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import structlog
from collections import deque
import numpy as np

# OpenTelemetry imports (mocked if not available)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
    tracer = trace.get_tracer("aura.memory")
except ImportError:
    OTEL_AVAILABLE = False
    tracer = None

# Prometheus imports (mocked if not available)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
    # Mock Prometheus metrics
    class MockMetric:
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
    
    Counter = Histogram = Gauge = Summary = lambda *args, **kwargs: MockMetric()

logger = structlog.get_logger(__name__)


# ==================== Prometheus Metrics ====================

# Counters
memory_operations_total = Counter(
    'aura_memory_operations_total',
    'Total number of memory operations',
    ['operation', 'memory_type', 'tier', 'status']
)

topology_extractions_total = Counter(
    'aura_memory_topology_extractions_total',
    'Total topology extractions performed',
    ['workflow_type', 'status']
)

predictions_total = Counter(
    'aura_memory_predictions_total',
    'Total predictions made',
    ['prediction_type', 'outcome']
)

# Histograms
operation_duration_seconds = Histogram(
    'aura_memory_operation_duration_seconds',
    'Duration of memory operations',
    ['operation', 'tier'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

topology_complexity = Histogram(
    'aura_memory_topology_complexity',
    'Complexity of topological signatures',
    ['metric_type'],
    buckets=(0, 1, 2, 5, 10, 20, 50, 100, 200)
)

# Gauges
active_memories = Gauge(
    'aura_memory_active_memories',
    'Number of active memories',
    ['tier', 'memory_type']
)

tier_capacity_bytes = Gauge(
    'aura_memory_tier_capacity_bytes',
    'Capacity of each tier in bytes',
    ['tier']
)

bottleneck_severity = Gauge(
    'aura_memory_bottleneck_severity',
    'Current bottleneck severity score',
    ['workflow_id']
)

# Summaries
retrieval_latency = Summary(
    'aura_memory_retrieval_latency_ms',
    'Memory retrieval latency in milliseconds',
    ['retrieval_mode', 'tier']
)

fastrp_embedding_time = Summary(
    'aura_memory_fastrp_embedding_seconds',
    'Time to compute FastRP embeddings'
)


# ==================== Performance Tracking ====================

@dataclass
class PerformanceWindow:
    """Rolling window for performance metrics"""
    window_size: int = 1000
    
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughput: deque = field(default_factory=lambda: deque(maxlen=1000))
    errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_operation(self, latency_ms: float, success: bool = True):
        """Add operation to window"""
        self.latencies.append(latency_ms)
        self.throughput.append(1)
        self.errors.append(0 if success else 1)
    
    def get_stats(self) -> Dict[str, float]:
        """Get window statistics"""
        if not self.latencies:
            return {
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "throughput_ops": 0.0,
                "error_rate": 0.0
            }
        
        latencies = list(self.latencies)
        return {
            "p50_latency": float(np.percentile(latencies, 50)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
            "throughput_ops": len(self.throughput),
            "error_rate": sum(self.errors) / len(self.errors) if self.errors else 0.0
        }


# ==================== Memory Metrics ====================

class MemoryMetrics:
    """
    Central metrics collection for memory system
    
    Tracks all operations and provides OTEL spans and Prometheus metrics
    """
    
    def __init__(self):
        # Performance windows by operation
        self.perf_windows: Dict[str, PerformanceWindow] = {
            "store": PerformanceWindow(),
            "retrieve": PerformanceWindow(),
            "topology_extract": PerformanceWindow(),
            "prediction": PerformanceWindow()
        }
        
        # Topology tracking
        self.topology_stats = {
            "total_extractions": 0,
            "avg_betti_numbers": [0, 0, 0],
            "max_bottleneck_score": 0.0,
            "total_failures_predicted": 0
        }
        
        logger.info(
            "Memory metrics initialized",
            otel_available=OTEL_AVAILABLE,
            prometheus_available=PROMETHEUS_AVAILABLE
        )
    
    # ==================== Recording Methods ====================
    
    @contextmanager
    def span(self, operation: str, **attributes):
        """Create OTEL span for operation"""
        if OTEL_AVAILABLE and tracer:
            with tracer.start_as_current_span(operation) as span:
                # Add attributes
                for key, value in attributes.items():
                    span.set_attribute(f"aura.memory.{key}", value)
                
                try:
                    yield span
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        else:
            # No-op context manager
            yield None
    
    def record_store(self, 
                    memory_type: str,
                    tier: Any,
                    duration_ms: float,
                    success: bool = True):
        """Record memory store operation"""
        # Prometheus
        memory_operations_total.labels(
            operation="store",
            memory_type=memory_type,
            tier=tier.name if hasattr(tier, 'name') else str(tier),
            status="success" if success else "failure"
        ).inc()
        
        operation_duration_seconds.labels(
            operation="store",
            tier=tier.name if hasattr(tier, 'name') else str(tier)
        ).observe(duration_ms / 1000.0)
        
        # Performance window
        self.perf_windows["store"].add_operation(duration_ms, success)
    
    def record_retrieval(self,
                        mode: Any,
                        results: int,
                        duration_ms: float,
                        tier_hits: Dict[str, int]):
        """Record memory retrieval operation"""
        # Prometheus
        for tier, hits in tier_hits.items():
            memory_operations_total.labels(
                operation="retrieve",
                memory_type="mixed",
                tier=tier,
                status="hit"
            ).inc(hits)
        
        retrieval_latency.labels(
            retrieval_mode=mode.value if hasattr(mode, 'value') else str(mode),
            tier=list(tier_hits.keys())[0] if tier_hits else "unknown"
        ).observe(duration_ms)
        
        # Performance window
        self.perf_windows["retrieve"].add_operation(duration_ms)
    
    def record_topology_extraction(self,
                                 workflow_id: str,
                                 betti_numbers: Tuple[int, int, int],
                                 bottleneck_score: float,
                                 duration_ms: float):
        """Record topology extraction"""
        # Prometheus
        topology_extractions_total.labels(
            workflow_type="agent_workflow",
            status="success"
        ).inc()
        
        topology_complexity.labels(metric_type="betti_0").observe(betti_numbers[0])
        topology_complexity.labels(metric_type="betti_1").observe(betti_numbers[1])
        topology_complexity.labels(metric_type="bottleneck_nodes").observe(
            int(bottleneck_score * 10)  # Scale to node count approximation
        )
        
        bottleneck_severity.labels(workflow_id=workflow_id).set(bottleneck_score)
        
        # Update stats
        self.topology_stats["total_extractions"] += 1
        self.topology_stats["max_bottleneck_score"] = max(
            self.topology_stats["max_bottleneck_score"],
            bottleneck_score
        )
        
        # Performance window
        self.perf_windows["topology_extract"].add_operation(duration_ms)
    
    def record_prediction(self,
                         prediction_type: str,
                         predicted_outcome: str,
                         confidence: float,
                         duration_ms: float):
        """Record prediction operation"""
        # Prometheus
        predictions_total.labels(
            prediction_type=prediction_type,
            outcome=predicted_outcome
        ).inc()
        
        # Performance window
        self.perf_windows["prediction"].add_operation(duration_ms)
        
        if predicted_outcome == "failure":
            self.topology_stats["total_failures_predicted"] += 1
    
    def record_fastrp_embedding(self, duration_seconds: float):
        """Record FastRP embedding computation"""
        fastrp_embedding_time.observe(duration_seconds)
    
    # ==================== Monitoring Queries ====================
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        summary = {}
        
        for operation, window in self.perf_windows.items():
            stats = window.get_stats()
            summary[operation] = {
                "p50_latency_ms": stats["p50_latency"],
                "p95_latency_ms": stats["p95_latency"],
                "p99_latency_ms": stats["p99_latency"],
                "throughput_per_window": stats["throughput_ops"],
                "error_rate": stats["error_rate"]
            }
        
        return summary
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """Get topology analysis summary"""
        return {
            "total_extractions": self.topology_stats["total_extractions"],
            "max_bottleneck_severity": self.topology_stats["max_bottleneck_score"],
            "total_failures_predicted": self.topology_stats["total_failures_predicted"],
            "extractions_per_minute": self.topology_stats["total_extractions"] / max(
                (time.time() - self._start_time) / 60, 1
            ) if hasattr(self, '_start_time') else 0
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        # In production, this would use prometheus_client.generate_latest()
        return "# Prometheus metrics would be exported here"
    
    # ==================== Alerting ====================
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        # High latency alert
        for operation, window in self.perf_windows.items():
            stats = window.get_stats()
            if stats["p95_latency"] > 100:  # 100ms threshold
                alerts.append({
                    "severity": "warning",
                    "operation": operation,
                    "message": f"High p95 latency: {stats['p95_latency']:.1f}ms",
                    "threshold": 100
                })
        
        # High error rate alert
        for operation, window in self.perf_windows.items():
            stats = window.get_stats()
            if stats["error_rate"] > 0.05:  # 5% error rate
                alerts.append({
                    "severity": "critical",
                    "operation": operation,
                    "message": f"High error rate: {stats['error_rate']*100:.1f}%",
                    "threshold": 5
                })
        
        # Bottleneck severity alert
        if self.topology_stats["max_bottleneck_score"] > 0.8:
            alerts.append({
                "severity": "warning",
                "component": "topology",
                "message": f"Severe bottleneck detected: {self.topology_stats['max_bottleneck_score']:.2f}",
                "threshold": 0.8
            })
        
        return alerts


# ==================== Grafana Dashboard Config ====================

GRAFANA_DASHBOARD = {
    "title": "AURA Memory System",
    "panels": [
        {
            "title": "Operation Latency (p95)",
            "query": "histogram_quantile(0.95, aura_memory_operation_duration_seconds)",
            "type": "graph"
        },
        {
            "title": "Topology Extractions/min",
            "query": "rate(aura_memory_topology_extractions_total[1m])",
            "type": "graph"
        },
        {
            "title": "Memory Tier Distribution",
            "query": "aura_memory_active_memories",
            "type": "pie"
        },
        {
            "title": "Bottleneck Severity",
            "query": "aura_memory_bottleneck_severity",
            "type": "heatmap"
        },
        {
            "title": "Failure Predictions",
            "query": "rate(aura_memory_predictions_total{outcome='failure'}[5m])",
            "type": "graph"
        },
        {
            "title": "FastRP Embedding Time",
            "query": "aura_memory_fastrp_embedding_seconds",
            "type": "histogram"
        }
    ],
    "alerts": [
        {
            "name": "HighMemoryLatency",
            "expr": "histogram_quantile(0.95, aura_memory_operation_duration_seconds) > 0.1",
            "duration": "5m",
            "severity": "warning"
        },
        {
            "name": "HighBottleneckScore", 
            "expr": "aura_memory_bottleneck_severity > 0.8",
            "duration": "2m",
            "severity": "critical"
        }
    ]
}


# ==================== Public API ====================

# Global metrics instance
_metrics = MemoryMetrics()

def get_metrics() -> MemoryMetrics:
    """Get global metrics instance"""
    return _metrics


__all__ = [
    "MemoryMetrics",
    "get_metrics",
    "PerformanceWindow",
    "GRAFANA_DASHBOARD"
]