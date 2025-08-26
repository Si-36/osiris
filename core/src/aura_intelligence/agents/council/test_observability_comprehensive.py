#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Observability Engine - Task 9 Implementation

Tests all observability features including metrics, tracing, alerting, and monitoring.
"""

import asyncio
import time
from typing import Dict, Any, List

# Import the observability components
try:
    from aura_intelligence.observability import (
        ObservabilityEngine, MetricType, AlertLevel, PerformanceMetric,
        DecisionTrace, SystemAlert, ComponentMetrics
    )
    observability_available = True
except ImportError as e:
    print(f"Observability engine not available: {e}")
    observability_available = False


class TestObservabilityEngine:
    """Comprehensive test suite for ObservabilityEngine"""
    
    def __init__(self):
        if observability_available:
            self.engine = ObservabilityEngine({
        "latency_threshold": 1.0,
        "error_rate_threshold": 0.1,
        "confidence_threshold": 0.8
        })
    
    def test_metric_recording(self):
            """Test performance metric recording (Requirement 6.1)"""
        pass
        if not observability_available:
            return False
        
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
        pass
        if not observability_available:
            return False
        
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
        pass
        if not observability_available:
            return False
        
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
        pass
        if not observability_available:
            return False
        
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
        pass
        if not observability_available:
            return False
        
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
            pass
            if not observability_available:
                return False
        
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
            pass
            if not observability_available:
                return False
        
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
            pass
            if not observability_available:
                return False
        
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
            pass
            if not observability_available:
                return False
        
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
        
            self.engine.alerts.extend([old_alert, recent_alert])
        
            # Get recent alerts (last 1 hour)
            recent_alerts = self.engine.get_recent_alerts(1)
        
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
    
        if not observability_available:
        print("❌ Observability engine not available, skipping tests")
        return
    
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