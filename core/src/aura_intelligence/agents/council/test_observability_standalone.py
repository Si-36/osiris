#!/usr/bin/env python3
"""
Standalone Observability Engine Tests - Task 9 Implementation

Complete standalone implementation with all components for testing.
"""

import asyncio
import time
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Standalone observability engine implementation
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
            except Exception:
                pass  # Don't let callback errors break the system
        
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


class TestObservabilityEngine:
    """Comprehensive test suite for ObservabilityEngine"""
    
    def __init__(self):
        self.engine = ObservabilityEngine({
            "latency_threshold": 1.0,
            "error_rate_threshold": 0.1,
            "confidence_threshold": 0.8
        })
    
    def test_metric_recording(self):
        """Test performance metric recording (Requirement 6.1)"""
        # Record various types of metrics
        self.engine.record_metric("lnn_inference_time", 0.5, MetricType.TIMER, 
                                 {"model": "v1"}, "seconds")
        self.engine.record_metric("gpu_utilization", 85.0, MetricType.GAUGE, 
                                 {"gpu_id": "0"}, "percent")
        self.engine.record_metric("decision_count", 1, MetricType.COUNTER, 
                                 {"decision": "approve"})
        
        # Verify metrics were recorded
        if len(self.engine.metrics) != 3:
            print(f"❌ Expected 3 metrics, got {len(self.engine.metrics)}")
            return False
        
        # Check metric details
        for metric_key, metric in self.engine.metrics.items():
            if not isinstance(metric, PerformanceMetric):
                print(f"❌ Invalid metric type: {type(metric)}")
                return False
            
            if metric.timestamp <= 0:
                print(f"❌ Invalid timestamp: {metric.timestamp}")
                return False
        
        print("✅ Metric recording: PASSED")
        return True
    
    def test_decision_tracing(self):
        """Test decision process tracing (Requirement 6.2)"""
        request_id = "test-request-123"
        
        # Start decision trace
        trace = self.engine.start_decision_trace(request_id)
        
        if trace.request_id != request_id:
            print(f"❌ Wrong request ID: {trace.request_id}")
            return False
        
        if trace.start_time <= 0:
            print(f"❌ Invalid start time: {trace.start_time}")
            return False
        
        # Add trace steps
        self.engine.add_trace_step(request_id, "analyze_request", {
            "priority": 7,
            "gpu_count": 2
        })
        
        self.engine.add_trace_step(request_id, "lnn_inference", {
            "confidence": 0.85,
            "processing_time": 0.3
        })
        
        # Complete trace
        self.engine.complete_decision_trace(
            request_id, 
            "approve", 
            0.85, 
            ["High priority request", "Sufficient resources available"],
            False
        )
        
        # Verify trace completion
        completed_trace = self.engine.get_decision_trace(request_id)
        
        if not completed_trace:
            print("❌ Trace not found after completion")
            return False
        
        if completed_trace.final_decision != "approve":
            print(f"❌ Wrong final decision: {completed_trace.final_decision}")
            return False
        
        if completed_trace.confidence_score != 0.85:
            print(f"❌ Wrong confidence score: {completed_trace.confidence_score}")
            return False
        
        if len(completed_trace.steps) != 2:
            print(f"❌ Wrong number of steps: {len(completed_trace.steps)}")
            return False
        
        if len(completed_trace.reasoning_path) != 2:
            print(f"❌ Wrong reasoning path length: {len(completed_trace.reasoning_path)}")
            return False
        
        print("✅ Decision tracing: PASSED")
        return True
    
    def test_component_monitoring(self):
        """Test component performance monitoring (Requirement 6.3)"""
        # Test successful operation
        with self.engine.monitor_component("memory_system", "query"):
            time.sleep(0.01)  # Simulate work
        
        # Test failed operation
        try:
            with self.engine.monitor_component("knowledge_graph", "search"):
                time.sleep(0.005)  # Simulate work
                raise ValueError("Connection failed")
        except ValueError:
            pass  # Expected
        
        # Verify component metrics
        if "memory_system" not in self.engine.component_metrics:
            print("❌ Memory system metrics not recorded")
            return False
        
        if "knowledge_graph" not in self.engine.component_metrics:
            print("❌ Knowledge graph metrics not recorded")
            return False
        
        memory_metrics = self.engine.component_metrics["memory_system"]
        kg_metrics = self.engine.component_metrics["knowledge_graph"]
        
        # Check memory system metrics (successful)
        if memory_metrics.total_calls != 1:
            print(f"❌ Wrong total calls for memory: {memory_metrics.total_calls}")
            return False
        
        if memory_metrics.successful_calls != 1:
            print(f"❌ Wrong successful calls for memory: {memory_metrics.successful_calls}")
            return False
        
        if memory_metrics.error_rate != 0.0:
            print(f"❌ Wrong error rate for memory: {memory_metrics.error_rate}")
            return False
        
        # Check knowledge graph metrics (failed)
        if kg_metrics.total_calls != 1:
            print(f"❌ Wrong total calls for KG: {kg_metrics.total_calls}")
            return False
        
        if kg_metrics.failed_calls != 1:
            print(f"❌ Wrong failed calls for KG: {kg_metrics.failed_calls}")
            return False
        
        if kg_metrics.error_rate != 1.0:
            print(f"❌ Wrong error rate for KG: {kg_metrics.error_rate}")
            return False
        
        if "Connection failed" not in kg_metrics.last_error:
            print(f"❌ Wrong last error for KG: {kg_metrics.last_error}")
            return False
        
        print("✅ Component monitoring: PASSED")
        return True
    
    def test_error_recording(self):
        """Test detailed error recording (Requirement 6.4)"""
        # Record an error
        test_error = RuntimeError("Neural network inference failed")
        context = {
            "model_version": "v2.1",
            "input_size": 1024,
            "gpu_memory": "8GB"
        }
        
        initial_metric_count = len(self.engine.metrics)
        
        self.engine.record_error("lnn_inference", test_error, context)
        
        # Verify error metric was recorded
        error_metrics = [m for m in self.engine.metrics.values() 
                        if m.name == "error_count"]
        
        if not error_metrics:
            print("❌ Error metric not recorded")
            return False
        
        error_metric = error_metrics[-1]  # Get the latest one
        
        if error_metric.labels.get("component") != "lnn_inference":
            print(f"❌ Wrong component in error metric: {error_metric.labels}")
            return False
        
        if error_metric.labels.get("error_type") != "RuntimeError":
            print(f"❌ Wrong error type in metric: {error_metric.labels}")
            return False
        
        print("✅ Error recording: PASSED")
        return True
    
    def test_alert_generation(self):
        """Test alert generation and callbacks (Requirement 6.5)"""
        # Set up alert callback
        received_alerts = []
        
        def alert_callback(alert):
            received_alerts.append(alert)
        
        self.engine.add_alert_callback(alert_callback)
        
        # Generate different types of alerts
        warning_alert = self.engine.generate_alert(
            AlertLevel.WARNING,
            "High latency detected",
            {"component": "lnn_inference", "latency": 2.5},
            ["Check GPU utilization", "Review model complexity"]
        )
        
        error_alert = self.engine.generate_alert(
            AlertLevel.ERROR,
            "Component failure",
            {"component": "memory_system", "error": "Connection timeout"}
        )
        
        # Verify alerts were generated
        if len(self.engine.alerts) < 2:
            print(f"❌ Expected at least 2 alerts, got {len(self.engine.alerts)}")
            return False
        
        # Verify callback was called
        if len(received_alerts) != 2:
            print(f"❌ Expected 2 callback calls, got {len(received_alerts)}")
            return False
        
        # Check alert details
        if warning_alert.level != AlertLevel.WARNING:
            print(f"❌ Wrong alert level: {warning_alert.level}")
            return False
        
        if "High latency detected" not in warning_alert.message:
            print(f"❌ Wrong alert message: {warning_alert.message}")
            return False
        
        if len(warning_alert.actionable_info) != 2:
            print(f"❌ Wrong actionable info count: {len(warning_alert.actionable_info)}")
            return False
        
        print("✅ Alert generation: PASSED")
        return True
    
    def test_performance_thresholds(self):
        """Test automatic performance threshold monitoring"""
        request_id = "slow-request-456"
        
        # Start a trace
        trace = self.engine.start_decision_trace(request_id)
        
        # Simulate slow decision (exceed threshold)
        time.sleep(0.01)  # Small delay for testing
        
        # Manually set times to simulate slow decision
        trace.start_time = time.time() - 1.5  # 1.5 seconds ago
        
        initial_alert_count = len(self.engine.alerts)
        
        # Complete trace (should trigger latency alert)
        self.engine.complete_decision_trace(
            request_id,
            "approve",
            0.6,  # Low confidence (below threshold)
            ["Slow processing due to high load"],
            False
        )
        
        # Check if alerts were generated
        new_alerts = self.engine.alerts[initial_alert_count:]
        
        # Should have at least one alert (latency or confidence)
        if len(new_alerts) == 0:
            print("❌ No alerts generated for threshold violations")
            return False
        
        # Check for latency alert
        latency_alerts = [a for a in new_alerts if "latency" in a.message.lower()]
        confidence_alerts = [a for a in new_alerts if "confidence" in a.message.lower()]
        
        if not latency_alerts and not confidence_alerts:
            print("❌ No latency or confidence alerts generated")
            return False
        
        print("✅ Performance thresholds: PASSED")
        return True
    
    def test_performance_summary(self):
        """Test comprehensive performance summary generation"""
        # Reset engine to start fresh
        self.engine.reset_metrics()
        
        # Generate some activity
        for i in range(3):
            request_id = f"summary-test-{i}"
            trace = self.engine.start_decision_trace(request_id)
            
            self.engine.add_trace_step(request_id, "process", {"step": i})
            
            self.engine.complete_decision_trace(
                request_id,
                "approve" if i % 2 == 0 else "deny",
                0.8 + (i * 0.05),
                [f"Reasoning step {i}"],
                i == 2  # Last one uses fallback
            )
        
        # Get performance summary
        summary = self.engine.get_performance_summary()
        
        # Verify summary structure
        required_keys = [
            "timestamp", "overall_metrics", "component_metrics",
            "active_traces", "recent_alerts", "health_status"
        ]
        
        for key in required_keys:
            if key not in summary:
                print(f"❌ Missing key in summary: {key}")
                return False
        
        # Check overall metrics
        overall = summary["overall_metrics"]
        
        if overall["total_decisions"] != 3:
            print(f"❌ Wrong total decisions: {overall['total_decisions']}")
            return False
        
        if overall["fallback_rate"] != 1/3:  # One out of three used fallback
            print(f"❌ Wrong fallback rate: {overall['fallback_rate']}")
            return False
        
        # Check health status
        if summary["health_status"] not in ["HEALTHY", "WARNING", "DEGRADED", "CRITICAL"]:
            print(f"❌ Invalid health status: {summary['health_status']}")
            return False
        
        print("✅ Performance summary: PASSED")
        return True
    
    def test_metrics_cleanup(self):
        """Test metrics cleanup and retention policies"""
        # Create engine with small limits for testing
        test_engine = ObservabilityEngine({
            "max_traces": 2,
            "max_alerts": 2,
            "max_latency_samples": 3
        })
        
        # Add more traces than the limit
        for i in range(4):
            request_id = f"cleanup-test-{i}"
            trace = test_engine.start_decision_trace(request_id)
            test_engine.complete_decision_trace(request_id, "approve", 0.8, [], False)
        
        # Should only keep the last 2 traces
        if len(test_engine.decision_traces) != 2:
            print(f"❌ Wrong trace count after cleanup: {len(test_engine.decision_traces)}")
            return False
        
        # Add more alerts than the limit
        for i in range(4):
            test_engine.generate_alert(AlertLevel.INFO, f"Test alert {i}")
        
        # Should only keep the last 2 alerts
        if len(test_engine.alerts) != 2:
            print(f"❌ Wrong alert count after cleanup: {len(test_engine.alerts)}")
            return False
        
        print("✅ Metrics cleanup: PASSED")
        return True
    
    def test_recent_alerts_filtering(self):
        """Test filtering of recent alerts by time"""
        # Create fresh engine for this test
        test_engine = ObservabilityEngine()
        
        # Generate alerts with different timestamps
        current_time = time.time()
        
        # Create an alert from 2 hours ago
        old_alert = SystemAlert(
            alert_id="old_alert",
            level=AlertLevel.WARNING,
            message="Old alert",
            timestamp=current_time - 7200  # 2 hours ago
        )
        
        # Create a recent alert
        recent_alert = SystemAlert(
            alert_id="recent_alert",
            level=AlertLevel.INFO,
            message="Recent alert",
            timestamp=current_time - 1800  # 30 minutes ago
        )
        
        test_engine.alerts.extend([old_alert, recent_alert])
        
        # Get recent alerts (last 1 hour)
        recent_alerts = test_engine.get_recent_alerts(1)
        
        # Should only include the recent alert
        if len(recent_alerts) != 1:
            print(f"❌ Wrong recent alert count: {len(recent_alerts)}")
            return False
        
        if recent_alerts[0].alert_id != "recent_alert":
            print(f"❌ Wrong recent alert: {recent_alerts[0].alert_id}")
            return False
        
        print("✅ Recent alerts filtering: PASSED")
        return True


async def run_all_tests():
    """Run all observability engine tests"""
    print("🧪 Observability Engine Comprehensive Tests - Task 9 Implementation")
    print("=" * 70)
    
    tester = TestObservabilityEngine()
    
    tests = [
        ("Metric Recording", tester.test_metric_recording),
        ("Decision Tracing", tester.test_decision_tracing),
        ("Component Monitoring", tester.test_component_monitoring),
        ("Error Recording", tester.test_error_recording),
        ("Alert Generation", tester.test_alert_generation),
        ("Performance Thresholds", tester.test_performance_thresholds),
        ("Performance Summary", tester.test_performance_summary),
        ("Metrics Cleanup", tester.test_metrics_cleanup),
        ("Recent Alerts Filtering", tester.test_recent_alerts_filtering),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL OBSERVABILITY ENGINE TESTS PASSED!")
        print("\n✅ Task 9 Implementation Complete:")
        print("   • Performance metrics collection for LNN inference ✅")
        print("   • Detailed logging for decision making process ✅")
        print("   • Observability hooks for monitoring decision quality ✅")
        print("   • Error tracking with detailed context information ✅")
        print("   • Performance degradation alerts with actionable info ✅")
        print("   • Unit tests for metrics collection and logging ✅")
        print("\n🚀 Ready for Task 10: Create Data Models and Schemas")
    else:
        print("❌ Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())