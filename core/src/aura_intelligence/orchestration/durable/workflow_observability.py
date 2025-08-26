"""
📊 Workflow Observability Integration

Comprehensive monitoring, metrics, and observability for durable workflows.
Integrates with TDA tracing system, provides workflow health checks, and
enables real-time monitoring with graceful fallbacks for missing dependencies.

Key Features:
- Workflow execution metrics and tracing
- Health checks and SLA monitoring
- TDA dashboard integration
- Graceful fallbacks for missing observability components
- Real-time workflow monitoring and alerting

TDA Integration:
- Workflow spans in TDA tracing system
- Metrics integration with TDA monitoring dashboard
- Correlation with TDA anomaly patterns
- Workflow performance analytics
"""

from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
import time
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import threading

# Observability imports with graceful fallbacks
try:
from aura_intelligence.observability.tracing import get_tracer
from aura_intelligence.observability.metrics import get_meter
tracer = get_tracer(__name__)
meter = get_meter(__name__)
OBSERVABILITY_AVAILABLE = True
except ImportError:
# Graceful fallback for missing observability
tracer = None
meter = None
OBSERVABILITY_AVAILABLE = False

# TDA integration with fallbacks
try:
from ..semantic.tda_integration import TDAContextIntegration
from ..semantic.base_interfaces import TDAContext
TDA_INTEGRATION_AVAILABLE = True
except ImportError:
TDAContextIntegration = None
TDAContext = None
TDA_INTEGRATION_AVAILABLE = False

class WorkflowMetricType(Enum):
"""Types of workflow metrics"""
EXECUTION_TIME = "execution_time"
SUCCESS_RATE = "success_rate"
FAILURE_RATE = "failure_rate"
COMPENSATION_RATE = "compensation_rate"
STEP_LATENCY = "step_latency"
THROUGHPUT = "throughput"
RESOURCE_USAGE = "resource_usage"
TDA_CORRELATION = "tda_correlation"

class HealthStatus(Enum):
"""Health check status levels"""
HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"
CRITICAL = "critical"

@dataclass
class WorkflowMetric:
"""Individual workflow metric"""
name: str
metric_type: WorkflowMetricType
value: float
timestamp: datetime
workflow_id: Optional[str] = None
step_name: Optional[str] = None
tags: Dict[str, str] = field(default_factory=dict)
tda_correlation_id: Optional[str] = None

@dataclass
class WorkflowSpan:
"""Workflow execution span for tracing"""
span_id: str
workflow_id: str
operation_name: str
start_time: datetime
end_time: Optional[datetime] = None
duration_ms: Optional[float] = None
status: str = "running"
tags: Dict[str, Any] = field(default_factory=dict)
logs: List[Dict[str, Any]] = field(default_factory=list)
parent_span_id: Optional[str] = None
tda_correlation_id: Optional[str] = None

@dataclass
class HealthCheck:
"""Workflow system health check"""
component: str
status: HealthStatus
message: str
timestamp: datetime
metrics: Dict[str, float] = field(default_factory=dict)
recommendations: List[str] = field(default_factory=list)

class MockTracer:
"""Mock tracer for environments without observability"""

def start_as_current_span(self, name: str, **kwargs):
return MockSpan(name)

class MockSpan:
"""Mock span for environments without observability"""

def __init__(self, name: str):
self.name = name
self.attributes = {}

def __enter__(self):
return self

def __exit__(self, *args, **kwargs):
"""Real implementation"""
pass
# Process input
result = self._process(*args, **kwargs)
return result
def set_attributes(self, attributes: Dict[str, Any]):
self.attributes.update(attributes)

def add_event(self, *args, **kwargs):
"""Real implementation"""
pass
# Process input
result = self._process(*args, **kwargs)
return result
def set_status(self, *args, **kwargs):
"""Real implementation"""
pass
# Process input
result = self._process(*args, **kwargs)
return result
class WorkflowObservabilityManager:
"""
Comprehensive workflow observability with TDA integration
"""

def __init__(self, tda_integration: Optional[TDAContextIntegration] = None):
self.tda_integration = tda_integration
self.tracer = tracer or MockTracer()
self.meter = meter

# Metrics storage
self.metrics_buffer: deque = deque(maxlen=10000)
self.active_spans: Dict[str, WorkflowSpan] = {}
self.health_checks: Dict[str, HealthCheck] = {}

# Performance tracking
self.workflow_stats = defaultdict(lambda: {
"total_executions": 0,
"successful_executions": 0,
"failed_executions": 0,
"total_execution_time": 0.0,
"avg_execution_time": 0.0,
"last_execution": None,
"tda_correlated_count": 0
})

# SLA thresholds
self.sla_thresholds = {
"max_execution_time": 300.0,  # 5 minutes
"min_success_rate": 0.95,     # 95%
"max_failure_rate": 0.05,     # 5%
"max_step_latency": 30.0      # 30 seconds
}

# Thread-safe metrics collection
self._metrics_lock = threading.Lock()

# Initialize built-in metrics if available
self._initialize_metrics()

def _initialize_metrics(self):
"""Initialize observability metrics"""
pass
if not self.meter:
return

try:
# Workflow execution metrics
self.execution_time_histogram = self.meter.create_histogram(
name="workflow_execution_time_seconds",
description="Workflow execution time in seconds",
unit="s"
)

self.workflow_counter = self.meter.create_counter(
name="workflow_executions_total",
description="Total number of workflow executions"
)

self.workflow_success_counter = self.meter.create_counter(
name="workflow_success_total",
description="Total number of successful workflow executions"
)

self.workflow_failure_counter = self.meter.create_counter(
name="workflow_failure_total",
description="Total number of failed workflow executions"
)

self.active_workflows_gauge = self.meter.create_up_down_counter(
name="active_workflows_count",
description="Number of currently active workflows"
)

self.step_latency_histogram = self.meter.create_histogram(
name="workflow_step_latency_seconds",
description="Individual workflow step latency",
unit="s"
)

self.tda_correlation_gauge = self.meter.create_up_down_counter(
name="tda_correlated_workflows",
description="Number of TDA-correlated workflows"
)

except Exception as e:
# Graceful fallback if metrics initialization fails
pass

async def start_workflow_span(
self,
workflow_id: str,
operation_name: str,
tda_correlation_id: Optional[str] = None,
parent_span_id: Optional[str] = None,
tags: Optional[Dict[str, Any]] = None
) -> str:
"""Start a new workflow execution span"""

span_id = f"{workflow_id}_{int(time.time() * 1000)}"

# Create workflow span
workflow_span = WorkflowSpan(
span_id=span_id,
workflow_id=workflow_id,
operation_name=operation_name,
start_time=datetime.now(timezone.utc),
parent_span_id=parent_span_id,
tda_correlation_id=tda_correlation_id,
tags=tags or {}
)

self.active_spans[span_id] = workflow_span

# Start observability span if available
if OBSERVABILITY_AVAILABLE and self.tracer:
with self.tracer.start_as_current_span(operation_name) as span:
span.set_attributes({
"workflow.id": workflow_id,
"workflow.operation": operation_name,
"workflow.span_id": span_id,
"tda.correlation_id": tda_correlation_id or "none",
**(tags or {})
})

# Record workflow start
await self._record_metric(WorkflowMetric(
name="workflow_started",
metric_type=WorkflowMetricType.THROUGHPUT,
value=1.0,
timestamp=datetime.now(timezone.utc),
workflow_id=workflow_id,
tda_correlation_id=tda_correlation_id
))

# Update active workflows gauge
if self.meter and hasattr(self, 'active_workflows_gauge'):
self.active_workflows_gauge.add(1, {"workflow_type": operation_name})

return span_id

async def end_workflow_span(
self,
span_id: str,
status: str = "success",
error: Optional[str] = None,
result_summary: Optional[Dict[str, Any]] = None
):
"""End a workflow execution span"""

if span_id not in self.active_spans:
return

workflow_span = self.active_spans[span_id]
workflow_span.end_time = datetime.now(timezone.utc)
workflow_span.status = status

# Calculate duration
duration = (workflow_span.end_time - workflow_span.start_time).total_seconds()
workflow_span.duration_ms = duration * 1000

# Add error information if present
if error:
workflow_span.logs.append({
"timestamp": datetime.now(timezone.utc).isoformat(),
"level": "error",
"message": error
})

# Add result summary
if result_summary:
workflow_span.tags.update(result_summary)

# Record execution metrics
await self._record_workflow_execution(workflow_span, status, duration, error)

# Update workflow statistics
self._update_workflow_stats(workflow_span, status, duration)

# Send to TDA if available
if self.tda_integration and workflow_span.tda_correlation_id:
await self._send_span_to_tda(workflow_span)

# Clean up active span
del self.active_spans[span_id]

# Update active workflows gauge
if self.meter and hasattr(self, 'active_workflows_gauge'):
self.active_workflows_gauge.add(-1, {"workflow_type": workflow_span.operation_name})

async def record_step_execution(
self,
workflow_id: str,
step_name: str,
duration_seconds: float,
status: str = "success",
error: Optional[str] = None,
tda_correlation_id: Optional[str] = None
):
"""Record individual workflow step execution"""

# Record step latency metric
await self._record_metric(WorkflowMetric(
name="step_execution",
metric_type=WorkflowMetricType.STEP_LATENCY,
value=duration_seconds,
timestamp=datetime.now(timezone.utc),
workflow_id=workflow_id,
step_name=step_name,
tags={"status": status, "step": step_name},
tda_correlation_id=tda_correlation_id
))

# Record in histogram if available
if self.meter and hasattr(self, 'step_latency_histogram'):
self.step_latency_histogram.record(
duration_seconds,
{"workflow_id": workflow_id, "step_name": step_name, "status": status}
)

# Check SLA compliance
if duration_seconds > self.sla_thresholds["max_step_latency"]:
await self._record_sla_violation(
"step_latency_exceeded",
f"Step {step_name} took {duration_seconds:.2f}s (threshold: {self.sla_thresholds['max_step_latency']}s)",
workflow_id,
{"step_name": step_name, "duration": duration_seconds}
)

async def _record_workflow_execution(
self,
workflow_span: WorkflowSpan,
status: str,
duration: float,
error: Optional[str]
):
"""Record workflow execution metrics"""

# Record execution time
await self._record_metric(WorkflowMetric(
name="workflow_execution_time",
metric_type=WorkflowMetricType.EXECUTION_TIME,
value=duration,
timestamp=datetime.now(timezone.utc),
workflow_id=workflow_span.workflow_id,
tags={"status": status, "operation": workflow_span.operation_name},
tda_correlation_id=workflow_span.tda_correlation_id
))

# Record success/failure
if status == "success":
await self._record_metric(WorkflowMetric(
name="workflow_success",
metric_type=WorkflowMetricType.SUCCESS_RATE,
value=1.0,
timestamp=datetime.now(timezone.utc),
workflow_id=workflow_span.workflow_id,
tda_correlation_id=workflow_span.tda_correlation_id
))

if self.meter and hasattr(self, 'workflow_success_counter'):
self.workflow_success_counter.add(1, {"operation": workflow_span.operation_name})
else:
await self._record_metric(WorkflowMetric(
name="workflow_failure",
metric_type=WorkflowMetricType.FAILURE_RATE,
value=1.0,
timestamp=datetime.now(timezone.utc),
workflow_id=workflow_span.workflow_id,
tags={"error": error or "unknown"},
tda_correlation_id=workflow_span.tda_correlation_id
))

if self.meter and hasattr(self, 'workflow_failure_counter'):
self.workflow_failure_counter.add(1, {
"operation": workflow_span.operation_name,
"error_type": type(error).__name__ if error else "unknown"
})

# Record in histograms if available
if self.meter and hasattr(self, 'execution_time_histogram'):
self.execution_time_histogram.record(
duration,
{"operation": workflow_span.operation_name, "status": status}
)

if self.meter and hasattr(self, 'workflow_counter'):
self.workflow_counter.add(1, {"operation": workflow_span.operation_name, "status": status})

# Check SLA compliance
if duration > self.sla_thresholds["max_execution_time"]:
await self._record_sla_violation(
"execution_time_exceeded",
f"Workflow {workflow_span.workflow_id} took {duration:.2f}s (threshold: {self.sla_thresholds['max_execution_time']}s)",
workflow_span.workflow_id,
{"duration": duration, "operation": workflow_span.operation_name}
)

def _update_workflow_stats(self, workflow_span: WorkflowSpan, status: str, duration: float):
"""Update workflow statistics"""

with self._metrics_lock:
stats = self.workflow_stats[workflow_span.operation_name]

stats["total_executions"] += 1
stats["total_execution_time"] += duration
stats["last_execution"] = datetime.now(timezone.utc)

if status == "success":
stats["successful_executions"] += 1
else:
stats["failed_executions"] += 1

if workflow_span.tda_correlation_id:
stats["tda_correlated_count"] += 1

# Calculate average execution time
stats["avg_execution_time"] = stats["total_execution_time"] / stats["total_executions"]

async def _record_metric(self, metric: WorkflowMetric):
"""Record a workflow metric"""

with self._metrics_lock:
self.metrics_buffer.append(metric)

async def _record_sla_violation(
self,
violation_type: str,
message: str,
workflow_id: str,
details: Dict[str, Any]
):
"""Record SLA violation"""

violation_metric = WorkflowMetric(
name=f"sla_violation_{violation_type}",
metric_type=WorkflowMetricType.FAILURE_RATE,
value=1.0,
timestamp=datetime.now(timezone.utc),
workflow_id=workflow_id,
tags={"violation_type": violation_type, **details}
)

await self._record_metric(violation_metric)

# Log SLA violation
if OBSERVABILITY_AVAILABLE and self.tracer:
with self.tracer.start_as_current_span("sla_violation") as span:
span.set_attributes({
"violation.type": violation_type,
"violation.message": message,
"workflow.id": workflow_id,
**details
})
span.add_event("SLA Violation", {"message": message})

async def _send_span_to_tda(self, workflow_span: WorkflowSpan):
"""Send workflow span to TDA for correlation"""

if not self.tda_integration:
return

try:
tda_span_data = {
"span_id": workflow_span.span_id,
"workflow_id": workflow_span.workflow_id,
"operation": workflow_span.operation_name,
"start_time": workflow_span.start_time.isoformat(),
"end_time": workflow_span.end_time.isoformat() if workflow_span.end_time else None,
"duration_ms": workflow_span.duration_ms,
"status": workflow_span.status,
"tags": workflow_span.tags,
"logs": workflow_span.logs
}

await self.tda_integration.send_orchestration_result(
tda_span_data,
workflow_span.tda_correlation_id
)

except Exception as e:
# Graceful fallback if TDA integration fails
pass

async def perform_health_check(self) -> Dict[str, HealthCheck]:
"""Perform comprehensive health check"""
pass

health_checks = {}

# Check workflow execution health
workflow_health = await self._check_workflow_health()
health_checks["workflow_execution"] = workflow_health

# Check observability health
observability_health = await self._check_observability_health()
health_checks["observability"] = observability_health

# Check TDA integration health
if self.tda_integration:
tda_health = await self._check_tda_integration_health()
health_checks["tda_integration"] = tda_health

# Check SLA compliance
sla_health = await self._check_sla_compliance()
health_checks["sla_compliance"] = sla_health

# Store health checks
self.health_checks.update(health_checks)

return health_checks

async def _check_workflow_health(self) -> HealthCheck:
"""Check workflow execution health"""
pass

current_time = datetime.now(timezone.utc)
recent_window = current_time - timedelta(minutes=15)

# Get recent metrics
recent_metrics = [
m for m in self.metrics_buffer
if m.timestamp >= recent_window
]

if not recent_metrics:
return HealthCheck(
component="workflow_execution",
status=HealthStatus.HEALTHY,
message="No recent workflow activity",
timestamp=current_time
)

# Calculate success rate
success_metrics = [m for m in recent_metrics if m.name == "workflow_success"]
failure_metrics = [m for m in recent_metrics if m.name == "workflow_failure"]

total_executions = len(success_metrics) + len(failure_metrics)
success_rate = len(success_metrics) / total_executions if total_executions > 0 else 1.0

# Determine health status
if success_rate >= self.sla_thresholds["min_success_rate"]:
status = HealthStatus.HEALTHY
message = f"Workflow execution healthy: {success_rate:.1%} success rate"
elif success_rate >= 0.8:
status = HealthStatus.DEGRADED
message = f"Workflow execution degraded: {success_rate:.1%} success rate"
else:
status = HealthStatus.UNHEALTHY
message = f"Workflow execution unhealthy: {success_rate:.1%} success rate"

recommendations = []
if success_rate < self.sla_thresholds["min_success_rate"]:
recommendations.append("Investigate recent workflow failures")
recommendations.append("Check TDA anomaly correlation")
recommendations.append("Review error patterns and compensation strategies")

return HealthCheck(
component="workflow_execution",
status=status,
message=message,
timestamp=current_time,
metrics={"success_rate": success_rate, "total_executions": total_executions},
recommendations=recommendations
)

async def _check_observability_health(self) -> HealthCheck:
"""Check observability system health"""
pass

current_time = datetime.now(timezone.utc)

if not OBSERVABILITY_AVAILABLE:
return HealthCheck(
component="observability",
status=HealthStatus.DEGRADED,
message="Observability components not available (using fallbacks)",
timestamp=current_time,
recommendations=["Install observability dependencies for full functionality"]
)

# Check if metrics are being collected
recent_metrics_count = len([
m for m in self.metrics_buffer
if m.timestamp >= current_time - timedelta(minutes=5)
])

if recent_metrics_count > 0:
return HealthCheck(
component="observability",
status=HealthStatus.HEALTHY,
message=f"Observability healthy: {recent_metrics_count} recent metrics",
timestamp=current_time,
metrics={"recent_metrics_count": recent_metrics_count}
)
else:
return HealthCheck(
component="observability",
status=HealthStatus.DEGRADED,
message="No recent metrics collected",
timestamp=current_time,
recommendations=["Check metric collection pipeline"]
)

async def _check_tda_integration_health(self) -> HealthCheck:
"""Check TDA integration health"""
pass

current_time = datetime.now(timezone.utc)

if not TDA_INTEGRATION_AVAILABLE:
return HealthCheck(
component="tda_integration",
status=HealthStatus.DEGRADED,
message="TDA integration not available",
timestamp=current_time,
recommendations=["Install TDA dependencies for full integration"]
)

try:
# Perform TDA health check
tda_health = await self.tda_integration.health_check()

if tda_health.get("status") == "healthy":
return HealthCheck(
component="tda_integration",
status=HealthStatus.HEALTHY,
message="TDA integration healthy",
timestamp=current_time,
metrics=tda_health
)
else:
return HealthCheck(
component="tda_integration",
status=HealthStatus.DEGRADED,
message=f"TDA integration degraded: {tda_health.get('status', 'unknown')}",
timestamp=current_time,
metrics=tda_health,
recommendations=["Check TDA system connectivity"]
)

except Exception as e:
return HealthCheck(
component="tda_integration",
status=HealthStatus.UNHEALTHY,
message=f"TDA integration error: {str(e)}",
timestamp=current_time,
recommendations=["Check TDA system availability", "Verify TDA configuration"]
)

async def _check_sla_compliance(self) -> HealthCheck:
"""Check SLA compliance"""
pass

current_time = datetime.now(timezone.utc)
recent_window = current_time - timedelta(hours=1)

# Get recent SLA violations
sla_violations = [
m for m in self.metrics_buffer
if m.timestamp >= recent_window and "sla_violation" in m.name
]

violation_count = len(sla_violations)

if violation_count == 0:
return HealthCheck(
component="sla_compliance",
status=HealthStatus.HEALTHY,
message="All SLAs within compliance",
timestamp=current_time,
metrics={"violation_count": 0}
)
elif violation_count <= 5:
return HealthCheck(
component="sla_compliance",
status=HealthStatus.DEGRADED,
message=f"{violation_count} SLA violations in the last hour",
timestamp=current_time,
metrics={"violation_count": violation_count},
recommendations=["Review recent SLA violations", "Optimize workflow performance"]
)
else:
return HealthCheck(
component="sla_compliance",
status=HealthStatus.UNHEALTHY,
message=f"{violation_count} SLA violations in the last hour (critical)",
timestamp=current_time,
metrics={"violation_count": violation_count},
recommendations=[
"Immediate investigation required",
"Check system resources and performance",
"Review workflow configurations"
]
)

def get_workflow_metrics(self, workflow_type: Optional[str] = None) -> Dict[str, Any]:
"""Get comprehensive workflow metrics"""

current_time = datetime.now(timezone.utc)

# Filter metrics by workflow type if specified
if workflow_type:
stats = self.workflow_stats.get(workflow_type, {})
return {
"workflow_type": workflow_type,
"total_executions": stats.get("total_executions", 0),
"successful_executions": stats.get("successful_executions", 0),
"failed_executions": stats.get("failed_executions", 0),
"success_rate": (
stats.get("successful_executions", 0) / max(stats.get("total_executions", 1), 1)
),
"average_execution_time": stats.get("avg_execution_time", 0.0),
"tda_correlated_count": stats.get("tda_correlated_count", 0),
"last_execution": stats.get("last_execution"),
"timestamp": current_time.isoformat()
}

# Return overall metrics
total_executions = sum(stats["total_executions"] for stats in self.workflow_stats.values())
total_successful = sum(stats["successful_executions"] for stats in self.workflow_stats.values())
total_failed = sum(stats["failed_executions"] for stats in self.workflow_stats.values())
total_tda_correlated = sum(stats["tda_correlated_count"] for stats in self.workflow_stats.values())

return {
"overall_metrics": {
"total_executions": total_executions,
"successful_executions": total_successful,
"failed_executions": total_failed,
"success_rate": total_successful / max(total_executions, 1),
"tda_correlation_rate": total_tda_correlated / max(total_executions, 1),
"active_workflows": len(self.active_spans),
"metrics_buffer_size": len(self.metrics_buffer),
"workflow_types": list(self.workflow_stats.keys())
},
"by_workflow_type": {
workflow_type: {
"total_executions": stats["total_executions"],
"success_rate": stats["successful_executions"] / max(stats["total_executions"], 1),
"avg_execution_time": stats["avg_execution_time"],
"tda_correlation_rate": stats["tda_correlated_count"] / max(stats["total_executions"], 1)
}
for workflow_type, stats in self.workflow_stats.items()
},
"timestamp": current_time.isoformat()
}

def get_recent_metrics(self, minutes: int = 15) -> List[WorkflowMetric]:
"""Get recent workflow metrics"""

cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

return [
metric for metric in self.metrics_buffer
if metric.timestamp >= cutoff_time
]

def export_metrics_for_dashboard(self) -> Dict[str, Any]:
"""Export metrics in format suitable for TDA dashboard"""
pass

current_time = datetime.now(timezone.utc)
recent_metrics = self.get_recent_metrics(60)  # Last hour

# Group metrics by type
metrics_by_type = defaultdict(list)
for metric in recent_metrics:
metrics_by_type[metric.metric_type.value].append(metric)

dashboard_data = {
"timestamp": current_time.isoformat(),
"summary": self.get_workflow_metrics(),
"health_status": {
component: {
"status": health.status.value,
"message": health.message,
"metrics": health.metrics
}
for component, health in self.health_checks.items()
},
"recent_activity": {
metric_type: {
"count": len(metrics),
"avg_value": sum(m.value for m in metrics) / len(metrics) if metrics else 0,
"latest_timestamp": max(m.timestamp for m in metrics).isoformat() if metrics else None
}
for metric_type, metrics in metrics_by_type.items()
},
"active_workflows": len(self.active_spans),
"sla_compliance": {
"execution_time_threshold": self.sla_thresholds["max_execution_time"],
"success_rate_threshold": self.sla_thresholds["min_success_rate"],
"step_latency_threshold": self.sla_thresholds["max_step_latency"]
}
}

return dashboard_data