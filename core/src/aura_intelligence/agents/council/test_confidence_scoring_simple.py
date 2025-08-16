#!/usr/bin/env python3
"""
Simple Unit Tests for Confidence Scoring Components

Focused tests for Task 7 that don't require full system dependencies.
Tests the core confidence scoring logic in isolation.
"""

import torch
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# Mock models for testing (simplified versions)
@dataclass
class MockGPUAllocationRequest:
    request_id: str
    user_id: str
    project_id: str
    gpu_type: str
    gpu_count: int
    memory_gb: int
    compute_hours: float
    priority: int
    special_requirements: List[str] = field(default_factory=list)
    requires_infiniband: bool = False
    requires_nvlink: bool = False
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MockLNNCouncilState:
    current_request: Optional[MockGPUAllocationRequest] = None
    context_cache: Dict[str, Any] = field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_passed: bool = False
    fallback_triggered: bool = False


# Import the actual confidence scoring components
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from confidence_scoring import (
        ConfidenceScorer,
        DecisionValidator,
        ReasoningPathGenerator,
        ConfidenceMetrics,
        ValidationResult
    )
    CONFIDENCE_SCORING_AVAILABLE = True
except ImportError as e:
    print(f"Confidence scoring not available: {e}")
    CONFIDENCE_SCORING_AVAILABLE = False


def test_confidence_scorer_basic():
    """Test basic confidence scorer functionality."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        config = {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2
        }
        
        scorer = ConfidenceScorer(config)
        
        # Test neural confidence calculation
        high_conf_output = torch.tensor([5.0, 1.0, 0.5])
        confidence = scorer._calculate_neural_confidence(high_conf_output)
        
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
        assert confidence > 0.7, f"High confidence output should yield high confidence: {confidence}"
        
        print(f"✅ Neural confidence calculation: {confidence:.3f}")
        
        # Test entropy calculation
        entropy = scorer._calculate_output_entropy(high_conf_output)
        assert 0.0 <= entropy <= 1.0, f"Entropy out of range: {entropy}"
        assert entropy < 0.5, f"Sharp distribution should have low entropy: {entropy}"
        
        print(f"✅ Entropy calculation: {entropy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence scorer basic test failed: {e}")
        return False


def test_confidence_metrics():
    """Test ConfidenceMetrics dataclass."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        metrics = ConfidenceMetrics()
        
        # Test default values
        assert metrics.neural_confidence == 0.0
        assert metrics.overall_confidence == 0.0
        assert isinstance(metrics.confidence_breakdown, dict)
        
        # Test with custom values
        metrics.neural_confidence = 0.85
        metrics.context_quality = 0.75
        metrics.overall_confidence = 0.8
        
        assert metrics.neural_confidence == 0.85
        assert metrics.context_quality == 0.75
        assert metrics.overall_confidence == 0.8
        
        print("✅ ConfidenceMetrics dataclass works correctly")
        return True
        
    except Exception as e:
        print(f"❌ ConfidenceMetrics test failed: {e}")
        return False


def test_validation_result():
    """Test ValidationResult dataclass."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        result = ValidationResult()
        
        # Test default values
        assert result.is_valid is True
        assert result.validation_score == 1.0
        assert len(result.violations) == 0
        assert len(result.warnings) == 0
        
        # Test adding violations
        result.add_violation("budget", "Insufficient funds")
        assert result.is_valid is False
        assert len(result.violations) == 1
        assert "budget: Insufficient funds" in result.violations
        
        # Test adding warnings
        result.add_warning("utilization", "High system load")
        assert len(result.warnings) == 1
        assert "utilization: High system load" in result.warnings
        
        print("✅ ValidationResult dataclass works correctly")
        return True
        
    except Exception as e:
        print(f"❌ ValidationResult test failed: {e}")
        return False


def test_decision_validator_basic():
    """Test basic decision validator functionality."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        config = {
            "max_gpu_allocation": 8,
            "max_duration_hours": 168,
            "budget_check_enabled": True,
            "policy_check_enabled": True
        }
        
        validator = DecisionValidator(config)
        
        # Create test data
        request = MockGPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        state = MockLNNCouncilState(current_request=request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"current_allocations": 1},
            "project_context": {"budget_remaining": 1000.0}
        }
        
        # Test approve decision validation
        result = validator.validate_decision("approve", request, state)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True  # Should pass with good parameters
        assert len(result.constraints_checked) > 0
        
        print(f"✅ Decision validation: valid={result.is_valid}, score={result.validation_score:.2f}")
        
        # Test with constraint violation
        large_request = MockGPUAllocationRequest(
            request_id="large_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=16,  # Exceeds limit
            memory_gb=80,
            compute_hours=8.0,
            priority=7
        )
        
        result_violation = validator.validate_decision("approve", large_request, state)
        assert result_violation.is_valid is False
        assert len(result_violation.violations) > 0
        
        print(f"✅ Constraint violation detection: violations={len(result_violation.violations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision validator basic test failed: {e}")
        return False


def test_reasoning_path_generator_basic():
    """Test basic reasoning path generator functionality."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        config = {
            "include_technical_details": True,
            "max_reasoning_steps": 10
        }
        
        generator = ReasoningPathGenerator(config)
        
        # Create test data
        request = MockGPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        confidence_metrics = ConfidenceMetrics()
        confidence_metrics.neural_confidence = 0.85
        confidence_metrics.context_quality = 0.8
        confidence_metrics.overall_confidence = 0.82
        
        validation_result = ValidationResult()
        validation_result.is_valid = True
        validation_result.validation_score = 0.95
        
        state = MockLNNCouncilState(current_request=request)
        
        # Generate reasoning path
        reasoning_path = generator.generate_reasoning_path(
            decision="approve",
            request=request,
            confidence_metrics=confidence_metrics,
            validation_result=validation_result,
            state=state
        )
        
        assert isinstance(reasoning_path, list)
        assert len(reasoning_path) > 0
        
        # Check for key elements
        reasoning_text = " ".join(reasoning_path)
        assert "2x A100" in reasoning_text
        assert "8.0h" in reasoning_text
        assert "APPROVED" in reasoning_text
        
        print(f"✅ Reasoning path generation: {len(reasoning_path)} steps")
        print(f"   Sample: {reasoning_path[0][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning path generator test failed: {e}")
        return False


def test_full_confidence_calculation():
    """Test full confidence calculation pipeline."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        config = {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2
        }
        
        scorer = ConfidenceScorer(config)
        
        # Create test data
        request = MockGPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        state = MockLNNCouncilState(current_request=request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.6, "queue_length": 5},
            "user_history": {"successful_allocations": 10, "avg_usage": 0.8},
            "project_context": {"budget_remaining": 1000.0, "priority_tier": "high"}
        }
        state.decision_history = [{"decision": "approve"}, {"decision": "approve"}]
        
        # Create realistic neural output
        neural_output = torch.tensor([2.5, 1.0, 0.5])
        
        # Calculate confidence
        metrics = scorer.calculate_confidence(
            neural_output=neural_output,
            state=state,
            decision="approve"
        )
        
        # Verify all metrics are calculated
        assert isinstance(metrics, ConfidenceMetrics)
        assert 0.0 <= metrics.neural_confidence <= 1.0
        assert 0.0 <= metrics.overall_confidence <= 1.0
        assert isinstance(metrics.confidence_breakdown, dict)
        assert len(metrics.confidence_breakdown) > 0
        
        print(f"✅ Full confidence calculation:")
        print(f"   Neural: {metrics.neural_confidence:.3f}")
        print(f"   Context: {metrics.context_quality:.3f}")
        print(f"   Overall: {metrics.overall_confidence:.3f}")
        print(f"   Breakdown: {len(metrics.confidence_breakdown)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ Full confidence calculation test failed: {e}")
        return False


def test_integration_scenario():
    """Test integration scenario with all components."""
    if not CONFIDENCE_SCORING_AVAILABLE:
        print("❌ Confidence scoring not available, skipping test")
        return False
    
    try:
        # Initialize all components
        config = {
            "confidence_threshold": 0.7,
            "entropy_weight": 0.2,
            "context_weight": 0.3,
            "constraint_weight": 0.3,
            "neural_weight": 0.2,
            "max_gpu_allocation": 8,
            "max_duration_hours": 168,
            "budget_check_enabled": True,
            "policy_check_enabled": True,
            "include_technical_details": True,
            "max_reasoning_steps": 12
        }
        
        scorer = ConfidenceScorer(config)
        validator = DecisionValidator(config)
        generator = ReasoningPathGenerator(config)
        
        # Create test scenario
        request = MockGPUAllocationRequest(
            request_id="integration_001",
            user_id="integration_user",
            project_id="integration_project",
            gpu_type="A100",
            gpu_count=4,
            memory_gb=80,
            compute_hours=12.0,
            priority=8,
            special_requirements=["high_memory"]
        )
        
        state = MockLNNCouncilState(current_request=request)
        state.context_cache = {
            "current_utilization": {"gpu_usage": 0.7, "queue_length": 8},
            "user_history": {"successful_allocations": 15, "current_allocations": 1},
            "project_context": {"budget_remaining": 2000.0, "security_level": "standard"},
            "system_constraints": {"maintenance_window": None}
        }
        
        neural_output = torch.tensor([3.5, 1.5, 0.8])
        decision = "approve"
        
        # Step 1: Calculate confidence
        confidence_metrics = scorer.calculate_confidence(
            neural_output=neural_output,
            state=state,
            decision=decision
        )
        
        # Step 2: Validate decision
        validation_result = validator.validate_decision(decision, request, state)
        
        # Step 3: Generate reasoning
        reasoning_path = generator.generate_reasoning_path(
            decision=decision,
            request=request,
            confidence_metrics=confidence_metrics,
            validation_result=validation_result,
            state=state
        )
        
        # Verify integration
        assert isinstance(confidence_metrics, ConfidenceMetrics)
        assert isinstance(validation_result, ValidationResult)
        assert isinstance(reasoning_path, list)
        
        assert confidence_metrics.overall_confidence > 0.0
        assert validation_result.validation_score > 0.0
        assert len(reasoning_path) > 5
        
        print(f"✅ Integration scenario:")
        print(f"   Confidence: {confidence_metrics.overall_confidence:.3f}")
        print(f"   Validation: {validation_result.validation_score:.3f}")
        print(f"   Reasoning: {len(reasoning_path)} steps")
        print(f"   Valid: {validation_result.is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration scenario test failed: {e}")
        return False


def main():
    """Run all confidence scoring tests."""
    print("🧪 Confidence Scoring Tests - Task 7 Implementation")
    print("=" * 60)
    
    tests = [
        ("Confidence Scorer Basic", test_confidence_scorer_basic),
        ("Confidence Metrics", test_confidence_metrics),
        ("Validation Result", test_validation_result),
        ("Decision Validator Basic", test_decision_validator_basic),
        ("Reasoning Path Generator", test_reasoning_path_generator_basic),
        ("Full Confidence Calculation", test_full_confidence_calculation),
        ("Integration Scenario", test_integration_scenario)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 ALL CONFIDENCE SCORING TESTS PASSED!")
        print("\n✅ Task 7 Implementation Complete:")
        print("   • Confidence scoring based on neural network outputs ✅")
        print("   • Decision validation against system constraints ✅")
        print("   • Reasoning path generation for explainable decisions ✅")
        print("   • Unit tests for confidence calculation and validation logic ✅")
        print("\n🚀 Ready for Task 8: Implement Fallback Mechanisms")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())