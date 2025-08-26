#!/usr/bin/env python3
"""
Standalone Fallback Engine Tests - Task 8 Implementation

Complete standalone implementation with all components for testing.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from enum import Enum

# Standalone fallback engine implementation
class FallbackTrigger(Enum):
    """Types of fallback triggers"""
    LNN_INFERENCE_FAILURE = "lnn_inference_failure"
    MEMORY_SYSTEM_FAILURE = "memory_system_failure"
    KNOWLEDGE_GRAPH_FAILURE = "knowledge_graph_failure"
    CONFIDENCE_SCORING_FAILURE = "confidence_scoring_failure"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_FAILURE = "validation_failure"


class DegradationLevel(Enum):
    """Levels of system degradation"""
    FULL_FUNCTIONALITY = "full"
    REDUCED_AI = "reduced_ai"
    RULE_BASED_ONLY = "rule_based"
    EMERGENCY_MODE = "emergency"


@dataclass
class FallbackMetrics:
    """Metrics for fallback system performance"""
    total_fallbacks: int = 0
    fallbacks_by_trigger: Dict[str, int] = field(default_factory=dict)
    fallbacks_by_level: Dict[str, int] = field(default_factory=dict)
    average_fallback_time: float = 0.0
    success_rate: float = 1.0


@dataclass
class MockGPURequest:
    """Mock GPU request for testing"""
    request_id: str = "test-123"
    user_id: str = "user-456"
    project_id: str = "project-789"
    gpu_count: int = 2
    priority: int = 5
    compute_hours: float = 8.0


@dataclass
class MockLNNCouncilState:
    """Mock state for testing"""
    current_request: Optional[MockGPURequest] = None
    context: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    fallback_triggered: bool = False
    next_step: str = ""


@dataclass
class MockLNNCouncilConfig:
    """Mock config for testing"""
    enable_fallback: bool = True
    fallback_timeout: float = 5.0
    max_recovery_attempts: int = 3


class FallbackEngine:
    """
    Comprehensive fallback decision engine.
    
    2025 Pattern:
    - Multi-level degradation
    - Trigger-specific responses
    - Performance monitoring
    - Graceful recovery
    """
    
    def __init__(self, config: MockLNNCouncilConfig):
        self.config = config
        self.metrics = FallbackMetrics()
        self.current_degradation = DegradationLevel.FULL_FUNCTIONALITY
        self.failed_subsystems: Set[str] = set()
        self.recovery_attempts: Dict[str, int] = {}
        self.last_fallback_time = None
    
    def _classify_failure(self, failed_step: str, error: Exception) -> FallbackTrigger:
        """Classify the type of failure to determine appropriate response."""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return FallbackTrigger.TIMEOUT_EXCEEDED
        elif "memory" in error_str or failed_step == "memory_integration":
            return FallbackTrigger.MEMORY_SYSTEM_FAILURE
        elif "knowledge" in error_str or "neo4j" in error_str or failed_step == "knowledge_context":
            return FallbackTrigger.KNOWLEDGE_GRAPH_FAILURE
        elif "confidence" in error_str or failed_step == "confidence_scoring":
            return FallbackTrigger.CONFIDENCE_SCORING_FAILURE
        elif "resource" in error_str or "memory" in error_str:
            return FallbackTrigger.RESOURCE_EXHAUSTION
        elif "validation" in error_str or failed_step == "validate_decision":
            return FallbackTrigger.VALIDATION_FAILURE
        else:
            return FallbackTrigger.LNN_INFERENCE_FAILURE
    
    def _calculate_degradation_level(self) -> DegradationLevel:
        """Calculate appropriate degradation level based on failed subsystems."""
        pass
        if not self.failed_subsystems:
            return DegradationLevel.FULL_FUNCTIONALITY
        
        critical_systems = {"lnn_inference", "neural_engine"}
        ai_systems = {"memory_integration", "knowledge_context", "confidence_scoring"}
        
        if any(sys in self.failed_subsystems for sys in critical_systems):
            if len(self.failed_subsystems) >= 3:
                return DegradationLevel.EMERGENCY_MODE
        else:
        return DegradationLevel.RULE_BASED_ONLY
        elif any(sys in self.failed_subsystems for sys in ai_systems):
        return DegradationLevel.REDUCED_AI
        else:
        return DegradationLevel.FULL_FUNCTIONALITY
    
    def _emergency_mode_decision(self, state: MockLNNCouncilState) -> Dict[str, Any]:
        """Emergency mode: minimal resource allocation logic."""
        request = state.current_request
        if not request:
            return {
                "neural_decision": "deny",
                "confidence_score": 0.3,
                "fallback_reason": "emergency_mode_no_request",
                "degradation_level": "emergency"
            }
        
        # Ultra-conservative emergency logic
        if request.priority >= 9:  # Only highest priority
            decision = "approve"
            confidence = 0.4
        else:
            decision = "deny"
            confidence = 0.6
        
        return {
            "neural_decision": decision,
            "confidence_score": confidence,
            "fallback_reason": "emergency_mode_conservative",
            "degradation_level": "emergency"
        }
    
    def _rule_based_decision(self, state: MockLNNCouncilState) -> Dict[str, Any]:
        """Rule-based decision when AI systems are down."""
        request = state.current_request
        if not request:
            return {
        "neural_decision": "deny",
        "confidence_score": 0.5,
        "fallback_reason": "rule_based_no_request",
        "degradation_level": "rule_based"
        }
        
        # Comprehensive rule-based logic
        decision_factors = []
        score = 0
        
        # Priority factor (0-40 points)
        priority_score = min(request.priority * 4, 40)
        score += priority_score
        decision_factors.append(f"priority_{request.priority}={priority_score}")
        
        # Resource efficiency factor (0-30 points)
        if request.gpu_count <= 2:
            resource_score = 30
        elif request.gpu_count <= 4:
        resource_score = 20
        elif request.gpu_count <= 8:
        resource_score = 10
        else:
        resource_score = 0
        score += resource_score
        decision_factors.append(f"resource_efficiency={resource_score}")
        
        # Time factor (0-20 points)
        if hasattr(request, 'compute_hours') and request.compute_hours:
            if request.compute_hours <= 4:
                time_score = 20
        elif request.compute_hours <= 12:
        time_score = 15
        elif request.compute_hours <= 24:
        time_score = 10
        else:
        time_score = 5
        else:
        time_score = 10  # Default
        score += time_score
        decision_factors.append(f"time_efficiency={time_score}")
        
        # Budget factor (0-10 points) - simplified
        budget_score = 10 if request.priority >= 5 else 5
        score += budget_score
        decision_factors.append(f"budget_check={budget_score}")
        
        # Decision logic
        if score >= 70:
            decision = "approve"
        confidence = min(0.8, score / 100)
        elif score >= 50:
        decision = "defer"
        confidence = 0.6
        else:
        decision = "deny"
        confidence = 0.7
        
        return {
        "neural_decision": decision,
        "confidence_score": confidence,
        "fallback_reason": "rule_based_comprehensive",
        "degradation_level": "rule_based",
        "decision_factors": decision_factors,
        "rule_score": score
        }
    
        async def _reduced_ai_decision(self, state: MockLNNCouncilState, trigger: FallbackTrigger) -> Dict[str, Any]:
        """Reduced AI mode: use available AI components."""
        request = state.current_request
        if not request:
            return {
                "neural_decision": "deny",
                "confidence_score": 0.5,
                "fallback_reason": "reduced_ai_no_request",
                "degradation_level": "reduced_ai"
            }
        
        # Start with rule-based decision
        base_decision = self._rule_based_decision(state)
        
        # Try to enhance with available AI components
        enhancements = []
        
        # If memory system is working, try to use it
        if "memory_integration" not in self.failed_subsystems:
            try:
                # Simplified memory lookup
                memory_context = await self._get_memory_context(request)
                if memory_context:
                    base_decision["confidence_score"] *= 1.1  # Slight boost
                    enhancements.append("memory_context")
            except Exception:
        pass
        
        # If knowledge graph is working, try to use it
        if "knowledge_context" not in self.failed_subsystems:
            try:
                # Simplified knowledge lookup
                knowledge_boost = self._get_knowledge_boost(request)
                base_decision["confidence_score"] *= knowledge_boost
                enhancements.append("knowledge_context")
            except Exception:
        pass
        
        base_decision.update({
            "fallback_reason": "reduced_ai_enhanced",
            "degradation_level": "reduced_ai",
            "ai_enhancements": enhancements
        })
        
        return base_decision
    
    def _minimal_fallback_decision(self, state: MockLNNCouncilState) -> Dict[str, Any]:
        """Minimal fallback for single component failures."""
        request = state.current_request
        if not request:
            return {
        "neural_decision": "deny",
        "confidence_score": 0.5,
        "fallback_reason": "minimal_fallback_no_request",
        "degradation_level": "full"
        }
        
        # Simple priority-based decision
        if request.priority >= 7:
            decision = "approve"
        confidence = 0.75
        elif request.priority >= 4:
        decision = "defer"
        confidence = 0.65
        else:
        decision = "deny"
        confidence = 0.7
        
        return {
        "neural_decision": decision,
        "confidence_score": confidence,
        "fallback_reason": "minimal_priority_based",
        "degradation_level": "full"
        }
    
        async def _get_memory_context(self, request) -> Optional[Dict[str, Any]]:
        """Simplified memory context retrieval."""
        pass
        return {
            "similar_requests": 0,
            "success_rate": 0.8,
            "avg_satisfaction": 0.7
        }
    
    def _get_knowledge_boost(self, request) -> float:
        """Simple knowledge-based confidence boost."""
        pass
        if hasattr(request, 'project_id') and request.project_id:
            return 1.05  # 5% boost for known projects
        return 1.0
    
    def _update_timing_metrics(self, fallback_time: float):
            """Update timing metrics for fallback operations."""
        if self.metrics.total_fallbacks == 1:
            self.metrics.average_fallback_time = fallback_time
        else:
            # Running average
            self.metrics.average_fallback_time = (
                (self.metrics.average_fallback_time * (self.metrics.total_fallbacks - 1) + fallback_time) 
                / self.metrics.total_fallbacks
            )
    
        async def _execute_fallback_strategy(
        self, 
        state: MockLNNCouncilState, 
        trigger: FallbackTrigger, 
        degradation: DegradationLevel
        ) -> MockLNNCouncilState:
        """Execute appropriate fallback strategy based on trigger and degradation level."""
        
        # Mark fallback triggered
        state.fallback_triggered = True
        
        # Update metrics
        self.metrics.fallbacks_by_level[degradation.value] = (
            self.metrics.fallbacks_by_level.get(degradation.value, 0) + 1
        )
        
        if degradation == DegradationLevel.EMERGENCY_MODE:
            decision_result = self._emergency_mode_decision(state)
        elif degradation == DegradationLevel.RULE_BASED_ONLY:
            decision_result = self._rule_based_decision(state)
        elif degradation == DegradationLevel.REDUCED_AI:
            decision_result = await self._reduced_ai_decision(state, trigger)
        else:
            decision_result = self._minimal_fallback_decision(state)
        
        # Update state
        state.context.update(decision_result)
        state.confidence_score = decision_result["confidence_score"]
        
        # Determine next step based on what's still working
        if "validate_decision" not in self.failed_subsystems:
            state.next_step = "validate_decision"
        else:
            state.next_step = "extract_output"
        
        return state
    
        async def handle_failure(self, state: MockLNNCouncilState, failed_step: str, error: Exception) -> MockLNNCouncilState:
        """Handle step failure with comprehensive fallback logic."""
        start_time = time.time()
        
        # Determine fallback trigger
        trigger = self._classify_failure(failed_step, error)
        
        # Update metrics
        self.metrics.total_fallbacks += 1
        self.metrics.fallbacks_by_trigger[trigger.value] = (
        self.metrics.fallbacks_by_trigger.get(trigger.value, 0) + 1
        )
        
        # Mark subsystem as failed
        self.failed_subsystems.add(failed_step)
        
        # Determine degradation level
        new_degradation = self._calculate_degradation_level()
        if new_degradation != self.current_degradation:
            self.current_degradation = new_degradation
        
        # Execute fallback strategy
        state = await self._execute_fallback_strategy(state, trigger, new_degradation)
        
        # Update timing metrics
        fallback_time = time.time() - start_time
        self._update_timing_metrics(fallback_time)
        
        self.last_fallback_time = time.time()
        
        return state
    
        async def attempt_recovery(self, subsystem: str) -> bool:
        """Attempt to recover a failed subsystem."""
        if subsystem not in self.failed_subsystems:
            return True
        
        # Track recovery attempts
        if subsystem not in self.recovery_attempts:
            self.recovery_attempts[subsystem] = 0
        
        self.recovery_attempts[subsystem] += 1
        
        # Simple recovery logic
        if self.recovery_attempts[subsystem] <= 3:
            # Simulate recovery attempt - first attempt succeeds
            recovery_success = self.recovery_attempts[subsystem] == 1
            
            if recovery_success:
                self.failed_subsystems.discard(subsystem)
                
                # Recalculate degradation level
                new_degradation = self._calculate_degradation_level()
                if new_degradation != self.current_degradation:
                    self.current_degradation = new_degradation
                
                return True
        
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the fallback system."""
        pass
        return {
        "degradation_level": self.current_degradation.value,
        "failed_subsystems": list(self.failed_subsystems),
        "recovery_attempts": dict(self.recovery_attempts),
        "metrics": {
        "total_fallbacks": self.metrics.total_fallbacks,
        "fallbacks_by_trigger": dict(self.metrics.fallbacks_by_trigger),
        "fallbacks_by_level": dict(self.metrics.fallbacks_by_level),
        "average_fallback_time": self.metrics.average_fallback_time,
        "success_rate": self.metrics.success_rate
        },
        "last_fallback_time": self.last_fallback_time,
        "enabled": self.config.enable_fallback
        }
    
    def reset_metrics(self):
            """Reset fallback metrics (useful for testing)."""
        pass
        self.metrics = FallbackMetrics()
        self.failed_subsystems.clear()
        self.recovery_attempts.clear()
        self.current_degradation = DegradationLevel.FULL_FUNCTIONALITY
        self.last_fallback_time = None


class TestFallbackEngine:
    """Comprehensive test suite for FallbackEngine"""
    
    def __init__(self):
        self.config = MockLNNCouncilConfig()
        self.engine = FallbackEngine(self.config)
    
    def test_fallback_trigger_classification(self):
            """Test failure classification into appropriate triggers"""
        pass
        test_cases = [
            ("timeout error", TimeoutError("Operation timed out"), FallbackTrigger.TIMEOUT_EXCEEDED),
            ("memory_integration", ValueError("Memory system failed"), FallbackTrigger.MEMORY_SYSTEM_FAILURE),
            ("knowledge_context", ConnectionError("Neo4j connection failed"), FallbackTrigger.KNOWLEDGE_GRAPH_FAILURE),
            ("confidence_scoring", RuntimeError("Confidence calculation failed"), FallbackTrigger.CONFIDENCE_SCORING_FAILURE),
            ("lnn_inference", Exception("Neural network error"), FallbackTrigger.LNN_INFERENCE_FAILURE),
        ]
        
        for step, error, expected_trigger in test_cases:
            trigger = self.engine._classify_failure(step, error)
            if trigger != expected_trigger:
                print(f"❌ Trigger classification failed: {step} -> {trigger.value}, expected {expected_trigger.value}")
                return False
        
        print("✅ Fallback trigger classification: PASSED")
        return True
    
    def test_degradation_level_calculation(self):
        """Test degradation level calculation based on failed subsystems"""
        pass
        test_cases = [
        (set(), DegradationLevel.FULL_FUNCTIONALITY),
        ({"confidence_scoring"}, DegradationLevel.REDUCED_AI),
        ({"memory_integration", "knowledge_context"}, DegradationLevel.REDUCED_AI),
        ({"lnn_inference"}, DegradationLevel.RULE_BASED_ONLY),
        ({"neural_engine"}, DegradationLevel.RULE_BASED_ONLY),
        ({"lnn_inference", "memory_integration", "knowledge_context"}, DegradationLevel.EMERGENCY_MODE),
        ]
        
        for failed_systems, expected_level in test_cases:
        self.engine.failed_subsystems = failed_systems
        level = self.engine._calculate_degradation_level()
        if level != expected_level:
            print(f"❌ Degradation calculation failed: {failed_systems} -> {level.value}, expected {expected_level.value}")
        return False
        
        print("✅ Degradation level calculation: PASSED")
        return True
    
        async def test_emergency_mode_decision(self):
            """Test emergency mode decision making"""
        pass
        # Test with high priority request
        state = MockLNNCouncilState(
            current_request=MockGPURequest(priority=9)
        )
        
        result = self.engine._emergency_mode_decision(state)
        
        if result["neural_decision"] != "approve":
            print(f"❌ Emergency mode high priority failed: {result['neural_decision']}")
            return False
        
        # Test with low priority request
        state.current_request.priority = 3
        result = self.engine._emergency_mode_decision(state)
        
        if result["neural_decision"] != "deny":
            print(f"❌ Emergency mode low priority failed: {result['neural_decision']}")
            return False
        
        # Test with no request
        state.current_request = None
        result = self.engine._emergency_mode_decision(state)
        
        if result["neural_decision"] != "deny":
            print(f"❌ Emergency mode no request failed: {result['neural_decision']}")
            return False
        
        print("✅ Emergency mode decision: PASSED")
        return True
    
    def test_rule_based_decision(self):
        """Test comprehensive rule-based decision logic"""
        pass
        # Test high-score request (should approve)
        state = MockLNNCouncilState(
        current_request=MockGPURequest(
        priority=8,  # 32 points
        gpu_count=2,  # 30 points
        compute_hours=2.0  # 20 points
        )
        )
        
        result = self.engine._rule_based_decision(state)
        
        if result["neural_decision"] != "approve":
            print(f"❌ Rule-based high score failed: {result['neural_decision']}, score: {result.get('rule_score')}")
        return False
        
        # Test medium-score request (should defer)
        state.current_request.priority = 5  # 20 points
        state.current_request.gpu_count = 4  # 20 points
        state.current_request.compute_hours = 8.0  # 15 points
        
        result = self.engine._rule_based_decision(state)
        
        if result["neural_decision"] != "defer":
            print(f"❌ Rule-based medium score failed: {result['neural_decision']}, score: {result.get('rule_score')}")
        return False
        
        # Test low-score request (should deny)
        state.current_request.priority = 2  # 8 points
        state.current_request.gpu_count = 10  # 0 points
        state.current_request.compute_hours = 48.0  # 5 points
        
        result = self.engine._rule_based_decision(state)
        
        if result["neural_decision"] != "deny":
            print(f"❌ Rule-based low score failed: {result['neural_decision']}, score: {result.get('rule_score')}")
        return False
        
        print("✅ Rule-based decision logic: PASSED")
        return True
    
        async def test_reduced_ai_decision(self):
            """Test reduced AI mode with partial system availability"""
        pass
        state = MockLNNCouncilState(
            current_request=MockGPURequest(priority=6)
        )
        
        # Test with memory system available
        self.engine.failed_subsystems = {"knowledge_context"}
        result = await self.engine._reduced_ai_decision(state, FallbackTrigger.KNOWLEDGE_GRAPH_FAILURE)
        
        if "reduced_ai" not in result["fallback_reason"]:
            print(f"❌ Reduced AI mode failed: {result['fallback_reason']}")
            return False
        
        if "ai_enhancements" not in result:
            print(f"❌ Reduced AI enhancements missing: {result}")
            return False
        
        print("✅ Reduced AI decision: PASSED")
        return True
    
        async def test_full_fallback_workflow(self):
        """Test complete fallback workflow from failure to recovery"""
        pass
        state = MockLNNCouncilState(
        current_request=MockGPURequest(priority=7)
        )
        
        # Simulate LNN inference failure
        error = RuntimeError("Neural network inference failed")
        
        # Handle the failure
        result_state = await self.engine.handle_failure(state, "lnn_inference", error)
        
        # Verify fallback was triggered
        if not result_state.fallback_triggered:
            print("❌ Fallback not triggered")
        return False
        
        # Verify metrics were updated
        if self.engine.metrics.total_fallbacks == 0:
            print("❌ Fallback metrics not updated")
        return False
        
        # Verify degradation level changed
        if self.engine.current_degradation == DegradationLevel.FULL_FUNCTIONALITY:
            print("❌ Degradation level not updated")
        return False
        
        # Verify decision was made
        if "neural_decision" not in result_state.context:
            print("❌ No fallback decision made")
        return False
        
        print("✅ Full fallback workflow: PASSED")
        return True
    
        async def test_recovery_mechanism(self):
            """Test subsystem recovery mechanism"""
        pass
        # Add a failed subsystem
        self.engine.failed_subsystems.add("memory_integration")
        
        # Attempt recovery
        recovery_success = await self.engine.attempt_recovery("memory_integration")
        
        if not recovery_success:
            print("❌ Recovery attempt failed")
            return False
        
        # Verify subsystem was removed from failed list
        if "memory_integration" in self.engine.failed_subsystems:
            print("❌ Failed subsystem not removed after recovery")
            return False
        
        print("✅ Recovery mechanism: PASSED")
        return True
    
    def test_metrics_tracking(self):
        """Test comprehensive metrics tracking"""
        pass
        # Reset metrics
        self.engine.reset_metrics()
        
        # Simulate some fallbacks
        self.engine.metrics.total_fallbacks = 5
        self.engine.metrics.fallbacks_by_trigger = {
        "lnn_inference_failure": 2,
        "memory_system_failure": 2,
        "timeout_exceeded": 1
        }
        self.engine.metrics.fallbacks_by_level = {
        "rule_based": 3,
        "reduced_ai": 2
        }
        
        # Test timing metrics update
        # First, set total_fallbacks to simulate the state
        self.engine.metrics.total_fallbacks = 1
        self.engine._update_timing_metrics(0.5)
        self.engine.metrics.total_fallbacks = 2
        self.engine._update_timing_metrics(1.0)
        
        expected_avg = (0.5 + 1.0) / 2
        if abs(self.engine.metrics.average_fallback_time - expected_avg) > 0.01:
            print(f"❌ Timing metrics incorrect: {self.engine.metrics.average_fallback_time}, expected {expected_avg}")
        return False
        
        # Test health status
        health = self.engine.get_health_status()
        
        required_fields = [
        "degradation_level", "failed_subsystems", "recovery_attempts",
        "metrics", "last_fallback_time", "enabled"
        ]
        
        for field in required_fields:
        if field not in health:
            print(f"❌ Health status missing field: {field}")
        return False
        
        print("✅ Metrics tracking: PASSED")
        return True
    
        async def test_performance_under_load(self):
            """Test fallback performance under multiple failures"""
        pass
        state = MockLNNCouncilState(
            current_request=MockGPURequest(priority=6)
        )
        
        # Simulate multiple rapid failures
        start_time = time.time()
        
        for i in range(10):
            error = RuntimeError(f"Failure {i}")
            await self.engine.handle_failure(state, f"step_{i}", error)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 10 failures in under 1 second
        if total_time > 1.0:
            print(f"❌ Performance test failed: {total_time:.3f}s for 10 failures")
            return False
        
        # Verify all failures were tracked
        if self.engine.metrics.total_fallbacks < 10:
            print(f"❌ Not all failures tracked: {self.engine.metrics.total_fallbacks}")
            return False
        
        print(f"✅ Performance under load: PASSED ({total_time:.3f}s for 10 failures)")
        return True


async def run_all_tests():
        """Run all fallback engine tests"""
        print("🧪 Fallback Engine Comprehensive Tests - Task 8 Implementation")
        print("=" * 70)
    
        tester = TestFallbackEngine()
    
        tests = [
        ("Fallback Trigger Classification", tester.test_fallback_trigger_classification),
        ("Degradation Level Calculation", tester.test_degradation_level_calculation),
        ("Emergency Mode Decision", tester.test_emergency_mode_decision),
        ("Rule-Based Decision Logic", tester.test_rule_based_decision),
        ("Reduced AI Decision", tester.test_reduced_ai_decision),
        ("Full Fallback Workflow", tester.test_full_fallback_workflow),
        ("Recovery Mechanism", tester.test_recovery_mechanism),
        ("Metrics Tracking", tester.test_metrics_tracking),
        ("Performance Under Load", tester.test_performance_under_load),
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
        print("🎉 ALL FALLBACK ENGINE TESTS PASSED!")
        print("\n✅ Task 8 Implementation Complete:")
        print("   • Multi-level degradation system ✅")
        print("   • Trigger-specific fallback responses ✅")
        print("   • Comprehensive rule-based decision logic ✅")
        print("   • Graceful recovery mechanisms ✅")
        print("   • Performance monitoring and metrics ✅")
        print("   • Unit tests for all fallback scenarios ✅")
        print("\n🚀 Ready for Task 9: Performance Monitoring and Observability")
        else:
        print("❌ Some tests failed")
    
        return passed == total


        if __name__ == "__main__":
        asyncio.run(run_all_tests())