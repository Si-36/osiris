o"""
🧪 Workflow Observability Integration Tests

Comprehensive test suite for workflow observability, monitoring, and health checks.
Tests cover metrics collection, health monitoring, SLA compliance, and TDA integration
with graceful fallbacks for missing dependencies.

Tests cover:
- Workflow span tracking and metrics collection
- Health checks and SLA monitoring
- TDA integration and correlation
- Dashboard data export
- Graceful fallbacks for missing dependencies
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from .workflow_observability import (
    WorkflowObservabilityManager, WorkflowMetric, WorkflowMetricType,
    WorkflowSpan, HealthCheck, HealthStatus
)


class MockTDAIntegration:
    """Mock TDA integration for testing"""
    
    def __init__(self):
        self.results = []
        self.health_status = {"status": "healthy", "tda_available": True}
    
    async def send_orchestration_result(self, result: Dict[str, Any], correlation_id: str) -> bool:
        self.results.append((result, correlation_id))
        return True
    
    async def health_check(self) -> Dict[r, Any]:
        return self.health_status


class TestWorkflowObservabilityManager:
    """Test workflow observability manager"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration for testing"""
        return MockTDAIntegration()
    
    @pytest.fixture
    def observability_manager(self, mock_tda_integration):
        """Create observability manager with mock TDA integration"""
        return WorkflowObservabilityManager(tda_integration=mock_tda_integration)
    
    @pytest.mark.asyncio
    async def test_workflow_span_lifecycle(self, observability_manager):
        """Test complete workflow span lifecycle"""
        
        # Start workflow span
        span_id = await observability_manager.start_workflow_span(
            workflow_id="test_workflow_001",
            operation_name="multi_agent_processing",
            tda_correlation_id="test-correlation-123",
            tags={"environment": "test", "priority": "high"}
        )
        
        assert span_id is not None
        assert span_id in observability_manager.active_spans
        
        # Verify span details
        span = observability_manager.active_spans[span_id]
        assert span.workflow_id == "test_workflow_001"
        assert span.operation_name == "multi_agent_processing"
        assert span.tda_correlation_id == "test-correlation-123"
        assert span.tags["environment"] == "test"
        assert span.status == "running"
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # End workflow span
        await observability_manager.end_workflow_span(
            span_id=span_id,
            status="success",
            result_summary={"steps_completed": 4, "total_time": 0.1}
        )
        
        # Verify span is completed and removed from active spans
        assert span_id not in observability_manager.active_spans
        assert span.status == "success"
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_workflow_span_with_error(self, observability_manager):
        """Test workflow span with error handling"""
        
        span_id = await observability_manager.start_workflow_span(
            workflow_id="test_workflow_error",
            operation_name="failing_workflow"
        )
        
        # End span with error
        await observability_manager.end_workflow_span(
            span_id=span_id,
            status="failed",
            error="Simulated workflow failure for testing"
        )
        
        # Verify error was recorded
        span = observability_manager.active_spans.get(span_id)
        assert span is None  # Should be removed from active spans
        
        # Check metrics were recorded
        recent_metrics = observability_manager.get_recent_metrics(1)
        failure_metrics = [m for m in recent_metrics if m.name == "workflow_failure"]
        assert len(failure_metrics) > 0
        
        failure_metric = failure_metrics[0]
        assert failure_metric.workflow_id == "test_workflow_error"
        assert failure_metric.metric_type == WorkflowMetricType.FAILURE_RATE
    
    @pytest.mark.asyncio
    async def test_step_execution_recording(self, observability_manager):
        """Test individual step execution recording"""
        
        # Record successful step
        await observability_manager.record_step_execution(
            workflow_id="test_workflow_steps",
            step_name="data_processing",
            duration_seconds=0.5,
            status="success",
            tda_correlation_id="test-correlation-456"
        )
        
        # Record failed step
        await observability_manager.record_step_execution(
            workflow_id="test_workflow_steps",
            step_name="model_inference",
            duration_seconds=2.0,
            status="failed",
            error="GPU memory exhausted"
        )
        
        # Verify metrics were recorded
        recent_metrics = observability_manager.get_recent_metrics(1)
        step_metrics = [m for m in recent_metrics if m.metric_type == WorkflowMetricType.STEP_LATENCY]
        
        assert len(step_metrics) == 2
        
        # Check successful step metric
        success_metric = next(m for m in step_metrics if m.step_name == "data_processing")
        assert success_metric.value == 0.5
        assert success_metric.tags["status"] == "success"
        assert success_metric.tda_correlation_id == "test-correlation-456"
        
        # Check failed step metric
        failed_metric = next(m for m in step_metrics if m.step_name == "model_inference")
        assert failed_metric.value == 2.0
        assert failed_metric.tags["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_sla_violation_detection(self, observability_manager):
        """Test SLA violation detection and recording"""
        
        # Set lower SLA threshold for testing
        observability_manager.sla_thresholds["max_step_latency"] = 1.0
        
        # Record step that exceeds SLA
        await observability_manager.record_step_execution(
            workflow_id="test_sla_violation",
            step_name="slow_step",
            duration_seconds=2.5,  # Exceeds 1.0s threshold
            status="success"
        )
        
        # Check for SLA violation metrics
        recent_metrics = observability_manager.get_recent_metrics(1)
        sla_violations = [m for m in recent_metrics if "sla_violation" in m.name]
        
        assert len(sla_violations) > 0
        
        violation_metric = sla_violations[0]
        assert violation_metric.workflow_id == "test_sla_violation"
        assert violation_metric.metric_type == WorkflowMetricType.FAILURE_RATE
        assert "step_latency_exceeded" in violation_metric.name
    
    @pytest.mark.asyncio
    async def test_health_check_workflow_execution(self, observability_manager):
        """Test workflow execution health check"""
        
        # Simulate successful workflows
        for i in range(8):
            await observability_manager._record_metric(WorkflowMetric(
                name="workflow_success",
                metric_type=WorkflowMetricType.SUCCESS_RATE,
                value=1.0,
                timestamp=datetime.now(timezone.utc),
                workflow_id=f"success_workflow_{i}"
            ))
        
        # Simulate failed workflows
        for i in range(2):
            await observability_manager._record_metric(WorkflowMetric(
                name="workflow_failure",
                metric_type=WorkflowMetricType.FAILURE_RATE,
                value=1.0,
                timestamp=datetime.now(timezone.utc),
                workflow_id=f"failed_workflow_{i}"
            ))
        
        # Perform health check
        health_check = await observability_manager._check_workflow_health()
        
        assert health_check.component == "workflow_execution"
        assert health_check.status == HealthStatus.HEALTHY  # 80% success rate
        assert health_check.metrics["success_rate"] == 0.8
        assert health_check.metrics["total_executions"] == 10
    
    @pytest.mark.asyncio
    async def test_health_check_degraded_performance(self, observability_manager):
        """Test health check with degraded performance"""
        
        # Simulate mostly failed workflows
        for i in range(7):
            await observability_manager._record_metric(WorkflowMetric(
                name="workflow_failure",
                metric_type=WorkflowMetricType.FAILURE_RATE,
                value=1.0,
                timestamp=datetime.now(timezone.utc),
                workflow_id=f"failed_workflow_{i}"
            ))
        
        for i in range(3):
            await observability_manager._record_metric(WorkflowMetric(
                name="workflow_success",
                metric_type=WorkflowMetricType.SUCCESS_RATE,
                value=1.0,
                timestamp=datetime.now(timezone.utc),
                workflow_id=f"success_workflow_{i}"
            ))
        
        # Perform health check
        health_check = await observability_manager._check_workflow_health()
        
        assert health_check.component == "workflow_execution"
        assert health_check.status == HealthStatus.UNHEALTHY  # 30% success rate
        assert health_check.metrics["success_rate"] == 0.3
        assert len(health_check.recommendations) > 0
        assert "Investigate recent workflow failures" in health_check.recommendations
    
    @pytest.mark.asyncio
    async def test_health_check_observability(self, observability_manager):
        """Test observability system health check"""
        
        # Add some recent metrics
        await observability_manager._record_metric(WorkflowMetric(
            name="test_metric",
            metric_type=WorkflowMetricType.THROUGHPUT,
            value=1.0,
            timestamp=datetime.now(timezone.utc)
        ))
        
        health_check = await observability_manager._check_observability_health()
        
        assert health_check.component == "observability"
        # Status depends on whether observability components are available
        assert health_check.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        if health_check.status == HealthStatus.HEALTHY:
            assert "recent metrics" in health_check.message.lower()
        else:
            assert "not available" in health_check.message.lower()
    
    @pytest.mark.asyncio
    async def test_health_check_tda_integration(self, observability_manager, mock_tda_integration):
        """Test TDA integration health check"""
        
        # Test healthy TDA integration
        health_check = await observability_manager._check_tda_integration_health()
        
        assert health_check.component == "tda_integration"
        assert health_check.status == HealthStatus.HEALTHY
        assert "healthy" in health_check.message.lower()
        
        # Test degraded TDA integration
        mock_tda_integration.health_status = {"status": "degraded", "tda_available": False}
        health_check = await observability_manager._check_tda_integration_health()
        
        assert health_check.status == HealthStatus.DEGRADED
        assert "degraded" in health_check.message.lower()
    
    @pytest.mark.asyncio
    async def test_health_check_sla_compliance(self, observability_manager):
        """Test SLA compliance health check"""
        
        # Test with no violations (healthy)
        health_check = await observability_manager._check_sla_compliance()
        
        assert health_check.component == "sla_compliance"
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.metrics["violation_count"] == 0
        
        # Add some SLA violations
        for i in range(3):
            await observability_manager._record_metric(WorkflowMetric(
                name="sla_violation_test",
                metric_type=WorkflowMetricType.FAILURE_RATE,
                value=1.0,
                timestamp=datetime.now(timezone.utc)
            ))
        
        health_check = await observability_manager._check_sla_compliance()
        
        assert health_check.status == HealthStatus.DEGRADED
        assert health_check.metrics["violation_count"] == 3
        assert len(health_check.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, observability_manager):
        """Test comprehensive health check of all components"""
        
        health_checks = await observability_manager.perform_health_check()
        
        # Verify all expected components are checked
        expected_components = ["workflow_execution", "observability", "sla_compliance"]
        if observability_manager.tda_integration:
            expected_components.append("tda_integration")
        
        for component in expected_components:
            assert component in health_checks
            assert isinstance(health_checks[component], HealthCheck)
            assert health_checks[component].component == component
            assert health_checks[component].status in [
                HealthStatus.HEALTHY, HealthStatus.DEGRADED, 
                HealthStatus.UNHEALTHY, HealthStatus.CRITICAL
            ]
    
    def test_workflow_metrics_collection(self, observability_manager):
        """Test workflow metrics collection and aggregation"""
        
        # Simulate workflow executions
        observability_manager._update_workflow_stats(
            WorkflowSpan(
                span_id="span_1",
                workflow_id="workflow_1",
                operation_name="data_processing",
                start_time=datetime.now(timezone.utc)
            ),
            "success",
            1.5
        )
        
        observability_manager._update_workflow_stats(
            WorkflowSpan(
                span_id="span_2",
                workflow_id="workflow_2",
                operation_name="data_processing",
                start_time=datetime.now(timezone.utc),
                tda_correlation_id="test-correlation"
            ),
            "failed",
            2.0
        )
        
        # Get metrics
        metrics = observability_manager.get_workflow_metrics("data_processing")
        
        assert metrics["workflow_type"] == "data_processing"
        assert metrics["total_executions"] == 2
        assert metrics["successful_executions"] == 1
        assert metrics["failed_executions"] == 1
        assert metrics["success_rate"] == 0.5
        assert metrics["average_execution_time"] == 1.75  # (1.5 + 2.0) / 2
        assert metrics["tda_correlated_count"] == 1
    
    def test_overall_workflow_metrics(self, observability_manager):
        """Test overall workflow metrics across all workflow types"""
        
        # Simulate different workflow types
        for workflow_type in ["data_processing", "model_inference", "result_aggregation"]:
            for i in range(3):
                observability_manager._update_workflow_stats(
                    WorkflowSpan(
                        span_id=f"span_{workflow_type}_{i}",
                        workflow_id=f"workflow_{workflow_type}_{i}",
                        operation_name=workflow_type,
                        start_time=datetime.now(timezone.utc)
                    ),
                    "success" if i < 2 else "failed",  # 2 success, 1 failure per type
                    1.0 + i * 0.5
                )
        
        # Get overall metrics
        metrics = observability_manager.get_workflow_metrics()
        
        overall = metrics["overall_metrics"]
        assert overall["total_executions"] == 9  # 3 types × 3 executions
        assert overall["successful_executions"] == 6  # 3 types × 2 successes
        assert overall["failed_executions"] == 3  # 3 types × 1 failure
        assert overall["success_rate"] == 6/9  # 2/3
        assert len(overall["workflow_types"]) == 3
        
        # Check by workflow type
        by_type = metrics["by_workflow_type"]
        for workflow_type in ["data_processing", "model_inference", "result_aggregation"]:
            assert workflow_type in by_type
            assert by_type[workflow_type]["total_executions"] == 3
            assert by_type[workflow_type]["success_rate"] == 2/3
    
    def test_recent_metrics_filtering(self, observability_manager):
        """Test recent metrics filtering by time window"""
        
        current_time = datetime.now(timezone.utc)
        
        # Add metrics at different times
        old_metric = WorkflowMetric(
            name="old_metric",
            metric_type=WorkflowMetricType.THROUGHPUT,
            value=1.0,
            timestamp=current_time - timedelta(hours=1)  # 1 hour ago
        )
        
        recent_metric = WorkflowMetric(
            name="recent_metric",
            metric_type=WorkflowMetricType.THROUGHPUT,
            value=2.0,
            timestamp=current_time - timedelta(minutes=5)  # 5 minutes ago
        )
        
        observability_manager.metrics_buffer.extend([old_metric, recent_metric])
        
        # Get recent metrics (last 15 minutes)
        recent_metrics = observability_manager.get_recent_metrics(15)
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].name == "recent_metric"
        
        # Get recent metrics (last 2 hours)
        recent_metrics = observability_manager.get_recent_metrics(120)
        
        assert len(recent_metrics) == 2
    
    def test_dashboard_data_export(self, observability_manager):
        """Test dashboard data export functionality"""
        
        # Add some test data
        observability_manager._update_workflow_stats(
            WorkflowSpan(
                span_id="dashboard_test_span",
                workflow_id="dashboard_test_workflow",
                operation_name="dashboard_test",
                start_time=datetime.now(timezone.utc)
            ),
            "success",
            1.0
        )
        
        # Add a health check
        observability_manager.health_checks["test_component"] = HealthCheck(
            component="test_component",
            status=HealthStatus.HEALTHY,
            message="Test component healthy",
            timestamp=datetime.now(timezone.utc),
            metrics={"test_metric": 1.0}
        )
        
        # Export dashboard data
        dashboard_data = observability_manager.export_metrics_for_dashboard()
        
        # Verify dashboard data structure
        assert "timestamp" in dashboard_data
        assert "summary" in dashboard_data
        assert "health_status" in dashboard_data
        assert "recent_activity" in dashboard_data
        assert "active_workflows" in dashboard_data
        assert "sla_compliance" in dashboard_data
        
        # Verify health status export
        assert "test_component" in dashboard_data["health_status"]
        health_status = dashboard_data["health_status"]["test_component"]
        assert health_status["status"] == "healthy"
        assert health_status["message"] == "Test component healthy"
        assert health_status["metrics"]["test_metric"] == 1.0
        
        # Verify SLA compliance export
        sla_compliance = dashboard_data["sla_compliance"]
        assert "execution_time_threshold" in sla_compliance
        assert "success_rate_threshold" in sla_compliance
        assert "step_latency_threshold" in sla_compliance
    
    @pytest.mark.asyncio
    async def test_tda_integration_span_sending(self, observability_manager, mock_tda_integration):
        """Test sending workflow spans to TDA integration"""
        
        # Start and end a workflow span with TDA correlation
        span_id = await observability_manager.start_workflow_span(
            workflow_id="tda_test_workflow",
            operation_name="tda_test_operation",
            tda_correlation_id="tda-test-correlation-789"
        )
        
        await observability_manager.end_workflow_span(
            span_id=span_id,
            status="success",
            result_summary={"test": "data"}
        )
        
        # Verify TDA integration received the data
        assert len(mock_tda_integration.results) > 0
        
        # Check the span data sent to TDA
        span_data = None
        for result, correlation_id in mock_tda_integration.results:
            if "span_id" in result:
                span_data = result
                break
        
        assert span_data is not None
        assert span_data["workflow_id"] == "tda_test_workflow"
        assert span_data["operation"] == "tda_test_operation"
        assert span_data["status"] == "success"
        assert "start_time" in span_data
        assert "end_time" in span_data
        assert "duration_ms" in span_data
    
    @pytest.mark.asyncio
    async def test_graceful_fallback_without_dependencies(self):
        """Test graceful fallback when dependencies are not available"""
        
        # Create observability manager without TDA integration
        observability_manager = WorkflowObservabilityManager(tda_integration=None)
        
        # Should still work without TDA integration
        span_id = await observability_manager.start_workflow_span(
            workflow_id="fallback_test_workflow",
            operation_name="fallback_test_operation"
        )
        
        await observability_manager.end_workflow_span(
            span_id=span_id,
            status="success"
        )
        
        # Should still collect metrics
        metrics = observability_manager.get_workflow_metrics("fallback_test_operation")
        assert metrics["total_executions"] == 1
        assert metrics["success_rate"] == 1.0
        
        # Health check should work with degraded status
        health_checks = await observability_manager.perform_health_check()
        assert "workflow_execution" in health_checks
        assert "observability" in health_checks
        assert "sla_compliance" in health_checks
        # TDA integration health check should not be present
        assert "tda_integration" not in health_checks


class TestWorkflowObservabilityIntegration:
    """Integration tests for workflow observability"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_monitoring(self):
        """Test end-to-end workflow monitoring scenario"""
        
        mock_tda = MockTDAIntegration()
        observability_manager = WorkflowObservabilityManager(tda_integration=mock_tda)
        
        # Simulate a complete multi-step workflow
        workflow_id = "e2e_test_workflow"
        correlation_id = "e2e-test-correlation-123"
        
        # Start workflow
        span_id = await observability_manager.start_workflow_span(
            workflow_id=workflow_id,
            operation_name="multi_step_processing",
            tda_correlation_id=correlation_id,
            tags={"environment": "test", "priority": "high"}
        )
        
        # Simulate workflow steps
        steps = [
            ("data_preprocessing", 0.5, "success"),
            ("feature_engineering", 0.8, "success"),
            ("model_inference", 1.2, "success"),
            ("result_aggregation", 0.3, "success")
        ]
        
        for step_name, duration, status in steps:
            await observability_manager.record_step_execution(
                workflow_id=workflow_id,
                step_name=step_name,
                duration_seconds=duration,
                status=status,
                tda_correlation_id=correlation_id
            )
            
            # Simulate processing time
            await asyncio.sleep(0.01)
        
        # End workflow
        await observability_manager.end_workflow_span(
            span_id=span_id,
            status="success",
            result_summary={
                "total_steps": len(steps),
                "total_duration": sum(step[1] for step in steps),
                "all_steps_successful": True
            }
        )
        
        # Verify comprehensive monitoring
        metrics = observability_manager.get_workflow_metrics("multi_step_processing")
        assert metrics["total_executions"] == 1
        assert metrics["success_rate"] == 1.0
        assert metrics["tda_correlated_count"] == 1
        
        # Verify step metrics were recorded
        recent_metrics = observability_manager.get_recent_metrics(1)
        step_metrics = [m for m in recent_metrics if m.metric_type == WorkflowMetricType.STEP_LATENCY]
        assert len(step_metrics) == 4  # One for each step
        
        # Verify TDA integration
        assert len(mock_tda.results) > 0
        
        # Verify health checks
        health_checks = await observability_manager.perform_health_check()
        assert health_checks["workflow_execution"].status == HealthStatus.HEALTHY
        
        # Verify dashboard export
        dashboard_data = observability_manager.export_metrics_for_dashboard()
        assert dashboard_data["summary"]["overall_metrics"]["total_executions"] == 1
        assert dashboard_data["active_workflows"] == 0  # Workflow completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])