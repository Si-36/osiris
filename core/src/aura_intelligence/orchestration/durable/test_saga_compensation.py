"""
🧪 Enhanced Saga Pattern Compensation Tests

Comprehensive test suite for the enhanced saga pattern compensation system
with TDA-aware error correlation and intelligent recovery strategies.

Tests cover:
- TDA-correlated failure analysis
- Intelligent compensation strategy selection
- Error correlation and recovery recommendations
- Circuit breaker and timeout safeguards
- Comprehensive metrics and monitoring
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from .saga_patterns import (
    SagaOrchestrator, SagaStep, SagaStepStatus, CompensationType,
    CompensationHandler
)
from ..semantic.base_interfaces import TDAContext
from ..semantic.tda_integration import MockTDAIntegration


class TestEnhancedSagaCompensation:
    """Test enhanced saga compensation with TDA integration"""
    
    @pytest.fixture
    def mock_tda_integration(self):
        """Mock TDA integration for testing"""
        return MockTDAIntegration()
    
    @pytest.fixture
    def saga_orchestrator(self, mock_tda_integration):
        """Create saga orchestrator with mock TDA integration"""
        return SagaOrchestrator(tda_integration=mock_tda_integration)
    
    @pytest.fixture
    def sample_tda_context(self):
        """Sample TDA context for testing"""
        return TDAContext(
            correlation_id="test-correlation-123",
            pattern_confidence=0.8,
            anomaly_severity=0.6,
            current_patterns={"test_pattern": 0.7},
            temporal_window="1h",
            metadata={"source": "test", "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    
    @pytest.fixture
    def failing_saga_steps(self):
        """Create saga steps that will fail for testing compensation"""
        async def successful_action(input_data):
            return {"status": "success", "data": "test_result"}
        
        async def failing_action(input_data):
            raise Exception("Simulated step failure")
        
        async def compensation_action(input_data):
            return {"status": "compensated", "original_result": input_data.get("original_result")}
        
        return [
            SagaStep(
                step_id="step_1",
                name="successful_step",
                action=successful_action,
                compensation_action=compensation_action,
                parameters={"param1": "value1"}
            ),
            SagaStep(
                step_id="step_2",
                name="failing_step",
                action=failing_action,
                compensation_action=compensation_action,
                parameters={"param2": "value2"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_failure_pattern_analysis(self, saga_orchestrator, sample_tda_context):
        """Test TDA-aware failure pattern analysis"""
        # Mock TDA integration to return specific anomaly correlation
        saga_orchestrator.tda_integration.correlate_with_anomalies = AsyncMock(
            return_value={
                "correlation_strength": 0.8,
                "anomalies": [{"type": "resource_contention", "severity": 0.7}]
            }
        )
        
        executed_steps = [
            SagaStep("step_1", "test_step", Mock(), Mock(), {}, status=SagaStepStatus.FAILED)
        ]
        
        analysis = await saga_orchestrator._analyze_failure_patterns(
            "test_saga", executed_steps, sample_tda_context, "test-correlation-123"
        )
        
        assert analysis["saga_id"] == "test_saga"
        assert analysis["failed_step_count"] == 1
        assert analysis["anomaly_correlation"] == 0.8
        assert analysis["pattern_confidence"] == 0.8
        assert "failure_category" in analysis
    
    @pytest.mark.asyncio
    async def test_compensation_strategy_determination(self, saga_orchestrator, sample_tda_context):
        """Test intelligent compensation strategy selection"""
        # Test TDA-correlated anomaly strategy
        failure_analysis = {
            "failure_category": "tda_correlated_anomaly",
            "anomaly_correlation": 0.8
        }
        
        strategy = await saga_orchestrator._determine_compensation_strategy(
            failure_analysis, sample_tda_context
        )
        
        assert strategy["type"] == CompensationType.FORWARD_RECOVERY
        assert strategy["priority"] == "high"
        assert strategy["retry_compensation"] is True
        assert strategy["timeout_multiplier"] == 2.0
        
        # Test transient failure strategy
        failure_analysis["failure_category"] = "transient_failure"
        strategy = await saga_orchestrator._determine_compensation_strategy(
            failure_analysis, sample_tda_context
        )
        
        assert strategy["type"] == CompensationType.ROLLBACK
        assert strategy["parallel_compensation"] is True
        assert strategy["retry_compensation"] is True
    
    @pytest.mark.asyncio
    async def test_compensation_with_safeguards(self, saga_orchestrator):
        """Test compensation execution with circuit breaker and timeout"""
        compensation_calls = []
        
        async def mock_compensation(input_data):
            compensation_calls.append(input_data)
            return {"status": "compensated"}
        
        step = SagaStep(
            "test_step", "test", Mock(), mock_compensation, {}
        )
        
        strategy = {
            "timeout_multiplier": 1.0,
            "retry_compensation": False
        }
        
        result = await saga_orchestrator._execute_compensation_with_safeguards(
            step, {"test": "data"}, strategy
        )
        
        assert result["status"] == "compensated"
        assert len(compensation_calls) == 1
        assert compensation_calls[0]["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_compensation_timeout_handling(self, saga_orchestrator):
        """Test compensation timeout and retry logic"""
        retry_count = 0
        
        async def slow_compensation(input_data):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 2:
                await asyncio.sleep(2)  # Simulate slow operation
                raise asyncio.TimeoutError("Timeout")
            return {"status": "compensated"}
        
        step = SagaStep(
            "test_step", "test", Mock(), slow_compensation, {}
        )
        
        strategy = {
            "timeout_multiplier": 0.001,  # Very short timeout for testing
            "retry_compensation": True
        }
        
        with pytest.raises(Exception, match="Compensation timeout"):
            await saga_orchestrator._execute_compensation_with_safeguards(
                step, {"test": "data"}, strategy
            )
    
    @pytest.mark.asyncio
    async def test_error_correlation_analysis(self, saga_orchestrator, sample_tda_context):
        """Test compensation error correlation with TDA patterns"""
        # Mock TDA integration
        saga_orchestrator.tda_integration.get_current_patterns = AsyncMock(
            return_value={"patterns": {"resource_pattern": 0.8}}
        )
        
        step = SagaStep("test_step", "test_step", Mock(), Mock(), {})
        
        # Test timeout error correlation
        timeout_error = Exception("Connection timeout occurred")
        correlation = await saga_orchestrator._correlate_compensation_error(
            timeout_error, step, sample_tda_context, "test-correlation"
        )
        
        assert correlation["error_type"] == "Exception"
        assert "timeout" in correlation["error_message"]
        assert correlation["severity"] == 0.6
        assert "increase_timeout" in correlation["recommended_actions"]
        assert correlation["error_category"] == "resource_contention"
        
        # Test connection error correlation
        connection_error = Exception("Connection refused")
        correlation = await saga_orchestrator._correlate_compensation_error(
            connection_error, step, sample_tda_context, "test-correlation"
        )
        
        assert correlation["severity"] == 0.8
        assert "check_network_connectivity" in correlation["recommended_actions"]
        assert correlation["error_category"] == "connectivity_issue"
    
    @pytest.mark.asyncio
    async def test_enhanced_saga_execution_with_compensation(
        self, saga_orchestrator, failing_saga_steps, sample_tda_context
    ):
        """Test complete saga execution with enhanced compensation"""
        # Set up TDA integration mocks
        saga_orchestrator.tda_integration.get_context = AsyncMock(
            return_value=sample_tda_context
        )
        saga_orchestrator.tda_integration.correlate_with_anomalies = AsyncMock(
            return_value={"correlation_strength": 0.7, "anomalies": []}
        )
        saga_orchestrator.tda_integration.send_orchestration_result = AsyncMock(
            return_value=True
        )
        
        result = await saga_orchestrator.execute_saga(
            "test_saga", failing_saga_steps, "test-correlation-123"
        )
        
        # Verify saga failed and compensation was triggered
        assert result["status"] == "failed"
        assert result["saga_id"] == "test_saga"
        assert result["steps_executed"] == 1  # Only first step succeeded
        assert "compensated_steps" in result
        
        # Verify TDA integration was called
        assert saga_orchestrator.tda_integration.send_orchestration_result.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_compensation_handler_registration(self, saga_orchestrator):
        """Test compensation handler registration and execution"""
        handler_calls = []
        
        async def custom_handler(input_data):
            handler_calls.append(input_data)
            return {"status": "custom_compensated"}
        
        handler = CompensationHandler(
            handler_id="test_handler",
            step_name="test_step",
            compensation_type=CompensationType.FORWARD_RECOVERY,
            handler_function=custom_handler,
            parameters={"custom_param": "value"},
            priority=10
        )
        
        saga_orchestrator.register_compensation_handler("test_step", handler)
        
        result = await saga_orchestrator.execute_custom_compensation(
            "test_saga", "test_step", CompensationType.FORWARD_RECOVERY,
            {"test_param": "test_value"}, "test-correlation"
        )
        
        assert result["status"] == "completed"
        assert result["handler_id"] == "test_handler"
        assert len(handler_calls) == 1
        assert handler_calls[0]["step_name"] == "test_step"
        assert handler_calls[0]["parameters"]["test_param"] == "test_value"
    
    def test_enhanced_saga_metrics(self, saga_orchestrator):
        """Test enhanced saga metrics with TDA correlation"""
        # Add some test saga history
        saga_orchestrator.saga_history = [
            {
                "saga_id": "saga_1",
                "status": "completed",
                "execution_time": 1.5,
                "tda_correlation_id": "corr_1"
            },
            {
                "saga_id": "saga_2", 
                "status": "failed",
                "execution_time": 2.0,
                "compensated_steps": ["step_1"],
                "tda_correlation_id": "corr_2"
            },
            {
                "saga_id": "saga_3",
                "status": "completed", 
                "execution_time": 1.0
            }
        ]
        
        metrics = saga_orchestrator.get_saga_metrics()
        
        assert metrics["total_sagas"] == 3
        assert metrics["success_rate"] == 2/3  # 2 successful out of 3
        assert metrics["average_execution_time"] == 1.5  # (1.5 + 2.0 + 1.0) / 3
        assert metrics["compensation_rate"] == 1/3  # 1 compensated out of 3
        assert metrics["tda_correlated_failures"] == 2/3  # 2 with TDA correlation
        assert "compensation_handlers" in metrics
    
    @pytest.mark.asyncio
    async def test_saga_status_tracking(self, saga_orchestrator, failing_saga_steps):
        """Test saga status tracking during execution"""
        # Start saga execution in background
        saga_task = asyncio.create_task(
            saga_orchestrator.execute_saga("status_test_saga", failing_saga_steps)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Check status while running
        status = saga_orchestrator.get_saga_status("status_test_saga")
        if status:  # May complete too quickly in tests
            assert status["saga_id"] == "status_test_saga"
            assert "total_steps" in status
            assert "completed_steps" in status
        
        # Wait for completion
        result = await saga_task
        
        # Check final status
        final_status = saga_orchestrator.get_saga_status("status_test_saga")
        assert final_status["saga_id"] == "status_test_saga"
        assert final_status["status"] == "failed"


class TestSagaCompensationIntegration:
    """Integration tests for saga compensation with real TDA components"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_tda_integration(self):
        """Test saga compensation with real TDA integration (if available)"""
        try:
            from ..semantic.tda_integration import TDAContextIntegration
            
            tda_integration = TDAContextIntegration()
            orchestrator = SagaOrchestrator(tda_integration=tda_integration)
            
            # Test health check
            health = await tda_integration.health_check()
            assert "status" in health
            assert "tda_available" in health
            
            # Test metrics
            metrics = await tda_integration.get_orchestration_metrics()
            assert "cache_size" in metrics
            assert "tda_available" in metrics
            
        except ImportError:
            pytest.skip("TDA integration not available for integration test")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_compensation_performance(self):
        """Test compensation performance under load"""
        orchestrator = SagaOrchestrator(tda_integration=MockTDAIntegration())
        
        async def fast_action(input_data):
            return {"result": "success"}
        
        async def fast_compensation(input_data):
            return {"compensated": True}
        
        # Create multiple saga steps
        steps = [
            SagaStep(
                step_id=f"step_{i}",
                name=f"step_{i}",
                action=fast_action,
                compensation_action=fast_compensation,
                parameters={"step_num": i}
            )
            for i in range(10)
        ]
        
        # Make the last step fail to trigger compensation
        async def failing_action(input_data):
            raise Exception("Performance test failure")
        
        steps[-1].action = failing_action
        
        start_time = datetime.now(timezone.utc)
        
        result = await orchestrator.execute_saga("perf_test_saga", steps, "perf-correlation")
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        assert result["status"] == "failed"
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(result["compensated_steps"]) == 9  # All successful steps compensated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])