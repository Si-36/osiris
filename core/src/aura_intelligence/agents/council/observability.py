"""
Performance Monitoring and Observability for LNN Council Agent (2025 Architecture)

Comprehensive observability system implementing Requirements 6.1-6.5:
- Performance metrics collection for LNN inference
- Detailed logging for decision making process
- Observability hooks for monitoring decision quality
- Error tracking with detailed context
- Performance degradation alerts
"""

import time
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import structlog

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics we collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class DecisionTrace:
    """Detailed trace of a decision process"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: Optional[float] = None
    reasoning_path: List[str] = field(default_factory=list)
    final_decision: Optional[str] = None
    fallback_triggered: bool = False
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class SystemAlert:
    """System performance alert"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    actionable_info: List[str] = field(default_factory=list)


@dataclass
class ComponentMetrics:
    """Metrics for a specific component"""
    component_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    error_rate: float = 0.0
    last_error: Optional[str] = None
    latency_samples: List[float] = field(default_factory=list)


class ObservabilityEngine:
    """
    Comprehensive observability engine for LNN Council Agent.
    
    2025 Pattern:
    - Real-time metrics collection
    - Structured logging
    - Performance monitoring
    - Alert generation
    - Decision tracing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.decision_traces: Dict[str, DecisionTrace] = {}
        self.alerts: List[SystemAlert] = []
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        
        # Performance thresholds
        self.latency_threshold = self.config.get("latency_threshold", 2.0)  # 2 seconds
        self.error_rate_threshold = self.config.get("error_rate_threshold", 0.05)  # 5%
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)  # 70%
        
        # Metrics retention
        self.max_traces = self.config.get("max_traces", 1000)
        self.max_alerts = self.config.get("max_alerts", 100)
        self.max_latency_samples = self.config.get("max_latency_samples", 100)
        
        logger.info("ObservabilityEngine initialized", config=self.config)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     labels: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a performance metric (Requirement 6.1)"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels or {},
            unit=unit
        )
        
        self.metrics[f"{name}_{int(metric.timestamp)}"] = metric
        
        logger.debug(
            "Metric recorded",
            name=name,
            value=value,
            type=metric_type.value,
            labels=labels,
            unit=unit
        )
    
    def start_decision_trace(self, request_id: str) -> DecisionTrace:
        """Start tracing a decision process (Requirement 6.2)"""
        trace = DecisionTrace(
            request_id=request_id,
            start_time=time.time()
        )
        
        self.decision_traces[request_id] = trace
        
        # Cleanup old traces if needed
        if len(self.decision_traces) > self.max_traces:
            oldest_key = min(self.decision_traces.keys(), 
                           key=lambda k: self.decision_traces[k].start_time)
            del self.decision_traces[oldest_key]
        
        logger.info("Decision trace started", request_id=request_id)
        return trace
    
    def add_trace_step(self, request_id: str, step_name: str, 
                      step_data: Dict[str, Any]) -> None:
        """Add a step to the decision trace"""
        if request_id in self.decision_traces:
            step_info = {
                "step_name": step_name,
                "timestamp": time.time(),
                "data": step_data
            }
            self.decision_traces[request_id].steps.append(step_info)
            
            logger.debug(
                "Trace step added",
                request_id=request_id,
                step=step_name,
                data=step_data
            )
    
    def complete_decision_trace(self, request_id: str, final_decision: str,
                              confidence_score: float, reasoning_path: List[str],
                              fallback_triggered: bool = False) -> None:
        """Complete a decision trace (Requirement 6.2)"""
        if request_id in self.decision_traces:
            trace = self.decision_traces[request_id]
            trace.end_time = time.time()
            trace.final_decision = final_decision
            trace.confidence_score = confidence_score
            trace.reasoning_path = reasoning_path
            trace.fallback_triggered = fallback_triggered
            
            # Calculate total decision time
            decision_time = trace.end_time - trace.start_time
            
            # Record decision metrics
            self.record_metric("decision_latency", decision_time, MetricType.TIMER, 
                             {"decision": final_decision}, "seconds")
            self.record_metric("decision_confidence", confidence_score, MetricType.GAUGE,
                             {"decision": final_decision})
            
            logger.info(
                "Decision trace completed",
                request_id=request_id,
                decision=final_decision,
                confidence=confidence_score,
                latency=decision_time,
                fallback=fallback_triggered,
                reasoning_steps=len(reasoning_path)
            )
            
            # Check for performance issues
            self._check_decision_performance(trace)
    
    @contextmanager
    def monitor_component(self, component_name: str, operation: str = ""):
        """Context manager for monitoring component performance (Requirement 6.3)"""
        if component_name not in self.component_metrics:
            self.component_metrics[component_name] = ComponentMetrics(component_name)
        
        metrics = self.component_metrics[component_name]
        start_time = time.time()
        
        try:
            metrics.total_calls += 1
            yield
            
            # Success case
            latency = time.time() - start_time
            metrics.successful_calls += 1
            self._update_latency_metrics(metrics, latency)
            
            # Record component metric
            labels = {"component": component_name}
            if operation:
                labels["operation"] = operation
            
            self.record_metric(f"{component_name}_latency", latency, 
                             MetricType.TIMER, labels, "seconds")
            
            logger.debug(
                "Component operation completed",
                component=component_name,
                operation=operation,
                latency=latency
            )
            
        except Exception as e:
            # Error case
            latency = time.time() - start_time
            metrics.failed_calls += 1
            metrics.last_error = str(e)
            
            # Record error
            self.record_error(component_name, e, {
                "operation": operation,
                "latency": latency
            })
            
            logger.error(
                "Component operation failed",
                component=component_name,
                operation=operation,
                error=str(e),
                latency=latency
            )
            
            raise
        
        finally:
            # Update error rate
            if metrics.total_calls > 0:
                metrics.error_rate = metrics.failed_calls / metrics.total_calls
    
    def record_error(self, component: str, error: Exception, 
                    context: Optional[Dict[str, Any]] = None) -> None:
        """Record detailed error information (Requirement 6.4)"""
        error_info = {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "timestamp": time.time(),
            "context": context or {}
        }
        
        # Record error metric
        self.record_metric("error_count", 1, MetricType.COUNTER, 
                         {"component": component, "error_type": type(error).__name__})
        
        logger.error(
            "Error recorded",
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        # Check if this should trigger an alert
        self._check_error_patterns(component, error_info)
    
    def generate_alert(self, level: AlertLevel, message: str, 
                      context: Optional[Dict[str, Any]] = None,
                      actionable_info: Optional[List[str]] = None) -> SystemAlert:
        """Generate a system alert (Requirement 6.5)"""
        alert = SystemAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            level=level,
            message=message,
            timestamp=time.time(),
            context=context or {},
            actionable_info=actionable_info or []
        )
        
        self.alerts.append(alert)
        
        # Cleanup old alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
        
        logger.warning(
            "System alert generated",
            alert_id=alert.alert_id,
            level=level.value,
            message=message,
            context=context,
            actionable_info=actionable_info
        )
        
        return alert
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]) -> None:
        """Add a callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_time = time.time()
        
        # Calculate overall metrics
        total_decisions = len([t for t in self.decision_traces.values() if t.end_time])
        avg_decision_time = 0.0
        avg_confidence = 0.0
        fallback_rate = 0.0
        
        if total_decisions > 0:
            completed_traces = [t for t in self.decision_traces.values() if t.end_time]
            decision_times = [t.end_time - t.start_time for t in completed_traces]
            confidences = [t.confidence_score for t in completed_traces if t.confidence_score]
            fallbacks = [t for t in completed_traces if t.fallback_triggered]
            
            avg_decision_time = sum(decision_times) / len(decision_times)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            fallback_rate = len(fallbacks) / total_decisions
        
        # Component summaries
        component_summary = {}
        for name, metrics in self.component_metrics.items():
            component_summary[name] = {
                "total_calls": metrics.total_calls,
                "success_rate": (metrics.successful_calls / metrics.total_calls) if metrics.total_calls > 0 else 0.0,
                "error_rate": metrics.error_rate,
                "average_latency": metrics.average_latency,
                "p95_latency": metrics.p95_latency,
                "last_error": metrics.last_error
            }
        
        return {
            "timestamp": current_time,
            "overall_metrics": {
                "total_decisions": total_decisions,
                "average_decision_time": avg_decision_time,
                "average_confidence": avg_confidence,
                "fallback_rate": fallback_rate
            },
            "component_metrics": component_summary,
            "active_traces": len([t for t in self.decision_traces.values() if not t.end_time]),
            "recent_alerts": len([a for a in self.alerts if current_time - a.timestamp < 3600]),  # Last hour
            "health_status": self._calculate_health_status()
        }
    
    def get_decision_trace(self, request_id: str) -> Optional[DecisionTrace]:
        """Get detailed trace for a specific decision"""
        return self.decision_traces.get(request_id)
    
    def get_recent_alerts(self, hours: int = 24) -> List[SystemAlert]:
        """Get recent alerts within specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def _update_latency_metrics(self, metrics: ComponentMetrics, latency: float) -> None:
        """Update latency statistics for a component"""
        # Add to samples
        metrics.latency_samples.append(latency)
        
        # Keep only recent samples
        if len(metrics.latency_samples) > self.max_latency_samples:
            metrics.latency_samples = metrics.latency_samples[-self.max_latency_samples:]
        
        # Calculate statistics
        if metrics.latency_samples:
            metrics.average_latency = sum(metrics.latency_samples) / len(metrics.latency_samples)
            sorted_samples = sorted(metrics.latency_samples)
            p95_index = int(0.95 * len(sorted_samples))
            p99_index = int(0.99 * len(sorted_samples))
            metrics.p95_latency = sorted_samples[min(p95_index, len(sorted_samples) - 1)]
            metrics.p99_latency = sorted_samples[min(p99_index, len(sorted_samples) - 1)]
    
    def _check_decision_performance(self, trace: DecisionTrace) -> None:
        """Check decision performance and generate alerts if needed"""
        if not trace.end_time:
            return
        
        decision_time = trace.end_time - trace.start_time
        
        # Check latency threshold
        if decision_time > self.latency_threshold:
            self.generate_alert(
                AlertLevel.WARNING,
                f"Decision latency exceeded threshold: {decision_time:.2f}s > {self.latency_threshold}s",
                {
                    "request_id": trace.request_id,
                    "decision_time": decision_time,
                    "threshold": self.latency_threshold,
                    "fallback_triggered": trace.fallback_triggered
                },
                [
                    "Check system resource utilization",
                    "Review neural network performance",
                    "Consider scaling compute resources"
                ]
            )
        
        # Check confidence threshold
        if trace.confidence_score and trace.confidence_score < self.confidence_threshold:
            self.generate_alert(
                AlertLevel.INFO,
                f"Low confidence decision: {trace.confidence_score:.2f} < {self.confidence_threshold}",
                {
                    "request_id": trace.request_id,
                    "confidence": trace.confidence_score,
                    "threshold": self.confidence_threshold,
                    "decision": trace.final_decision
                },
                [
                    "Review training data quality",
                    "Check input data completeness",
                    "Consider model retraining"
                ]
            )
    
    def _check_error_patterns(self, component: str, error_info: Dict[str, Any]) -> None:
        """Check for error patterns that should trigger alerts"""
        if component in self.component_metrics:
            metrics = self.component_metrics[component]
            
            # Check error rate threshold
            if metrics.error_rate > self.error_rate_threshold:
                self.generate_alert(
                    AlertLevel.ERROR,
                    f"High error rate in {component}: {metrics.error_rate:.2%} > {self.error_rate_threshold:.2%}",
                    {
                        "component": component,
                        "error_rate": metrics.error_rate,
                        "threshold": self.error_rate_threshold,
                        "total_calls": metrics.total_calls,
                        "failed_calls": metrics.failed_calls
                    },
                    [
                        f"Investigate {component} component health",
                        "Check external dependencies",
                        "Review recent configuration changes",
                        "Consider enabling fallback mechanisms"
                    ]
                )
    
    def _calculate_health_status(self) -> str:
        """Calculate overall system health status"""
        recent_alerts = self.get_recent_alerts(1)  # Last hour
        critical_alerts = [a for a in recent_alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in recent_alerts if a.level == AlertLevel.ERROR]
        
        if critical_alerts:
            return "CRITICAL"
        elif error_alerts:
            return "DEGRADED"
        elif len(recent_alerts) > 10:  # Many warnings
            return "WARNING"
        else:
            return "HEALTHY"
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)"""
        self.metrics.clear()
        self.component_metrics.clear()
        self.decision_traces.clear()
        self.alerts.clear()
        logger.info("All metrics reset")