#!/usr/bin/env python3
"""
Simple Model Tests - Task 10 Implementation

Tests the essential models we actually need.
"""

import time
from datetime import datetime, timezone

# Test the models we just created
try:
    from .models import (
        GPUAllocationRequest, GPUAllocationDecision, 
        DecisionContext, HistoricalDecision, LNNCouncilState
    )
    models_available = True
except ImportError as e:
    print(f"Models not available: {e}")
    models_available = False


def test_gpu_allocation_request():
    """Test GPU allocation request model."""
    if not models_available:
        return False
    
    # Create a valid request
    request = GPUAllocationRequest(
        user_id="test_user",
        project_id="test_project",
        gpu_type="A100",
        gpu_count=2,
        memory_gb=40,
        compute_hours=8.0,
        priority=7
    )
    
    # Verify basic fields
    if not request.request_id:
        print("❌ Request ID not generated")
        return False
    
    if request.user_id != "test_user":
        print(f"❌ Wrong user_id: {request.user_id}")
        return False
    
    if request.gpu_type != "A100":
        print(f"❌ Wrong gpu_type: {request.gpu_type}")
        return False
    
    # Test validation
    try:
        invalid_request = GPUAllocationRequest(
            user_id="test_user",
            project_id="test_project",
            gpu_type="INVALID_GPU",  # Should fail
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0
        )
        print("❌ Validation should have failed for invalid GPU type")
        return False
    except ValueError:
        pass  # Expected
    
    print("✅ GPU allocation request: PASSED")
    return True


def test_gpu_allocation_decision():
    """Test GPU allocation decision model."""
    if not models_available:
        return False
    
    # Create a decision
    decision = GPUAllocationDecision(
        request_id="test-123",
        decision="approve",
        confidence_score=0.85,
        inference_time_ms=150.0
    )
    
    # Test adding reasoning
    decision.add_reasoning("priority_check", "High priority request")
    decision.add_reasoning("resource_check", "Sufficient resources available")
    
    if len(decision.reasoning_path) != 2:
        print(f"❌ Wrong reasoning path length: {len(decision.reasoning_path)}")
        return False
    
    if "priority_check: High priority request" not in decision.reasoning_path[0]:
        print(f"❌ Wrong reasoning format: {decision.reasoning_path[0]}")
        return False
    
    # Test validation
    try:
        invalid_decision = GPUAllocationDecision(
            request_id="test-123",
            decision="invalid_decision",  # Should fail
            confidence_score=0.85,
            inference_time_ms=150.0
        )
        print("❌ Validation should have failed for invalid decision")
        return False
    except ValueError:
        pass  # Expected
    
    print("✅ GPU allocation decision: PASSED")
    return True


def test_decision_context():
    """Test decision context model."""
    if not models_available:
        return False
    
    context = DecisionContext(
        current_utilization={"gpu": 0.7, "memory": 0.6},
        available_resources={"A100": 4, "H100": 2},
        queue_depth=3
    )
    
    if context.current_utilization["gpu"] != 0.7:
        print(f"❌ Wrong utilization: {context.current_utilization}")
        return False
    
    if context.queue_depth != 3:
        print(f"❌ Wrong queue depth: {context.queue_depth}")
        return False
    
    print("✅ Decision context: PASSED")
    return True


def test_historical_decision():
    """Test historical decision model."""
    if not models_available:
        return False
    
    historical = HistoricalDecision(
        decision_id="hist-123",
        similarity_score=0.8,
        decision_made="approve",
        outcome_success=True,
        lessons_learned=["User satisfied", "Resources used efficiently"],
        decision_timestamp=datetime.now(timezone.utc)
    )
    
    if historical.similarity_score != 0.8:
        print(f"❌ Wrong similarity score: {historical.similarity_score}")
        return False
    
    if len(historical.lessons_learned) != 2:
        print(f"❌ Wrong lessons count: {len(historical.lessons_learned)}")
        return False
    
    print("✅ Historical decision: PASSED")
    return True


def test_lnn_council_state():
    """Test LNN council state model."""
    if not models_available:
        return False
    
    # Create a request first
    request = GPUAllocationRequest(
        user_id="test_user",
        project_id="test_project",
        gpu_type="A100",
        gpu_count=2,
        memory_gb=40,
        compute_hours=8.0
    )
    
    # Create state
    state = LNNCouncilState(
        current_request=request,
        confidence_score=0.75,
        fallback_triggered=False,
        inference_start_time=time.time(),
        neural_inference_time=0.2
    )
    
    if state.confidence_score != 0.75:
        print(f"❌ Wrong confidence score: {state.confidence_score}")
        return False
    
    if state.current_request.user_id != "test_user":
        print(f"❌ Wrong request user: {state.current_request.user_id}")
        return False
    
    print("✅ LNN council state: PASSED")
    return True


def test_model_integration():
    """Test that models work together."""
    if not models_available:
        return False
    
    # Create a complete workflow
    request = GPUAllocationRequest(
        user_id="integration_user",
        project_id="integration_project",
        gpu_type="H100",
        gpu_count=1,
        memory_gb=80,
        compute_hours=4.0,
        priority=8
    )
    
    context = DecisionContext(
        current_utilization={"gpu": 0.5},
        available_resources={"H100": 3},
        queue_depth=1
    )
    
    state = LNNCouncilState(
        current_request=request,
        confidence_score=0.9,
        fallback_triggered=False
    )
    
    decision = GPUAllocationDecision(
        request_id=request.request_id,
        decision="approve",
        confidence_score=0.9,
        inference_time_ms=120.0,
        allocated_resources={
            "gpu_ids": ["h100_001"],
            "gpu_type": "H100",
            "memory_gb": 80
        }
    )
    
    decision.add_reasoning("priority", "High priority request (8/10)")
    decision.add_reasoning("resources", "H100 available")
    
    # Verify integration
    if decision.request_id != request.request_id:
        print("❌ Request ID mismatch in integration")
        return False
    
    if state.current_request.gpu_type != "H100":
        print("❌ GPU type mismatch in integration")
        return False
    
    print("✅ Model integration: PASSED")
    return True


def run_all_tests():
    """Run all model tests."""
    print("🧪 Essential Data Models Tests - Task 10 Implementation")
    print("=" * 60)
    
    if not models_available:
        print("❌ Models not available, skipping tests")
        return
    
    tests = [
        ("GPU Allocation Request", test_gpu_allocation_request),
        ("GPU Allocation Decision", test_gpu_allocation_decision),
        ("Decision Context", test_decision_context),
        ("Historical Decision", test_historical_decision),
        ("LNN Council State", test_lnn_council_state),
        ("Model Integration", test_model_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL ESSENTIAL MODEL TESTS PASSED!")
        print("\n✅ Task 10 Implementation Complete:")
        print("   • GPUAllocationRequest model ✅")
        print("   • GPUAllocationDecision model ✅") 
        print("   • DecisionContext model ✅")
        print("   • HistoricalDecision model ✅")
        print("   • LNNCouncilState model ✅")
        print("   • Pydantic validation ✅")
        print("   • Model integration tests ✅")
        print("\n🚀 System is now unbroken and ready for Task 11!")
    else:
        print("❌ Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()