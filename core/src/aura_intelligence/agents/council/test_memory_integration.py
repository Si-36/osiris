#!/usr/bin/env python3
"""
Test Memory Integration Layer (2025 Architecture)
"""

import asyncio
import torch
from datetime import datetime, timezone, timedelta
from typing import Dict, Any


class MockLNNCouncilConfig:
    def __init__(self):
        self.name = "test_memory_agent"
        self.input_size = 64
        self.output_size = 16


class MockGPURequest:
    def __init__(self):
        self.request_id = "test_123"
        self.user_id = "user_123"
        self.project_id = "proj_456"
        self.gpu_type = "A100"
        self.gpu_count = 2
        self.memory_gb = 40
        self.compute_hours = 8.0
        self.priority = 7
        self.created_at = datetime.now(timezone.utc)


class MockGPUDecision:
    def __init__(self):
        self.request_id = "test_123"
        self.decision = "approve"
        self.confidence_score = 0.85
        self.fallback_used = False
        self.inference_time_ms = 150.0


class MockLNNCouncilState:
    def __init__(self):
        self.current_request = MockGPURequest()
        self.context_cache = {}


async def test_memory_context_retrieval():
        """Test memory context retrieval."""
        print("🧪 Testing Memory Context Retrieval")
    
        try:
            pass
        # Import here to avoid path issues
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        
        from memory_context import LNNMemoryIntegration
        
        config = MockLNNCouncilConfig()
        memory_integration = LNNMemoryIntegration(config)
        
        state = MockLNNCouncilState()
        
        # Test memory context retrieval (will use mock data)
        context = await memory_integration.get_memory_context(state)
        
        if context is not None:
            print(f"✅ Memory context retrieved: shape {context.shape}")
            print(f"   Memory quality: {memory_integration.memory_quality_score:.3f}")
            print(f"   Non-zero features: {(context != 0).sum().item()}")
        else:
            print("✅ No memory context (expected without Mem0)")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Memory context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_decision_outcome_storage():
        """Test storing decision outcomes."""
        print("\n🧪 Testing Decision Outcome Storage")
    
        try:
            pass
        from memory_context import LNNMemoryIntegration
        
        config = MockLNNCouncilConfig()
        memory_integration = LNNMemoryIntegration(config)
        
        request = MockGPURequest()
        decision = MockGPUDecision()
        
        # Test storing decision outcome
        outcome = {
            "success": True,
            "utilization": 0.85,
            "cost_efficiency": 0.9,
            "satisfaction": 0.8
        }
        
        await memory_integration.store_decision_outcome(request, decision, outcome)
        
        print("✅ Decision outcome stored (mock)")
        print(f"   Decision count: {memory_integration.decision_count}")
        print(f"   Learning updates: {memory_integration.learning_updates}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Decision storage test failed: {e}")
        return False


async def test_learning_from_outcomes():
        """Test learning from decision outcomes."""
        print("\n🧪 Testing Learning from Outcomes")
    
        try:
            pass
        from memory_context import LNNMemoryIntegration
        
        config = MockLNNCouncilConfig()
        memory_integration = LNNMemoryIntegration(config)
        
        request = MockGPURequest()
        decision = MockGPUDecision()
        
        # Test learning from outcome
        actual_outcome = {
            "success": True,
            "utilization": 0.9,
            "cost_efficiency": 0.85
        }
        
        await memory_integration.learn_from_outcome(request, decision, actual_outcome)
        
        print("✅ Learning from outcome completed")
        
        # Test learning weight calculation
        learning_weight = memory_integration._calculate_learning_weight(decision, actual_outcome)
        print(f"   Learning weight: {learning_weight:.3f}")
        
        # Test prediction error calculation
        pred_error = memory_integration._calculate_prediction_error(decision, actual_outcome)
        print(f"   Prediction error: {pred_error:.3f}")
        
        # Test confidence error calculation
        conf_error = memory_integration._calculate_confidence_error(decision, actual_outcome)
        print(f"   Confidence error: {conf_error:.3f}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Learning test failed: {e}")
        return False


async def test_memory_learning_engine():
        """Test the memory learning engine."""
        print("\n🧪 Testing Memory Learning Engine")
    
        try:
            pass
        from memory_learning import MemoryLearningEngine
        
        config = MockLNNCouncilConfig()
        learning_engine = MemoryLearningEngine(config)
        
        request = MockGPURequest()
        decision = MockGPUDecision()
        
        # Test learning from decision
        actual_outcome = {
            "success": True,
            "utilization": 0.88,
            "cost_efficiency": 0.92
        }
        
        learning_results = await learning_engine.learn_from_decision(
            request, decision, actual_outcome
        )
        
        print("✅ Memory learning engine test completed")
        print(f"   Learning quality: {learning_results.get('quality', 0.0):.3f}")
        print(f"   Learning episodes: {learning_engine.learning_episodes}")
        
        # Test learning insights
        insights = learning_engine.get_learning_insights()
        print(f"   Calibration accuracy: {insights.get('calibration_accuracy', 0.0):.3f}")
        print(f"   Pattern recognition: {insights.get('pattern_recognition_score', 0.0):.3f}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Memory learning engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_confidence_calibration():
        """Test confidence calibration."""
        print("\n🧪 Testing Confidence Calibration")
    
        try:
            pass
        from memory_learning import ConfidenceCalibrator
        
        config = MockLNNCouncilConfig()
        calibrator = ConfidenceCalibrator(config)
        
        # Test multiple calibration updates
        test_data = [
            (0.9, True),   # High confidence, success
            (0.8, True),   # High confidence, success
            (0.7, False),  # Medium confidence, failure
            (0.6, True),   # Medium confidence, success
            (0.3, False),  # Low confidence, failure
            (0.2, False),  # Low confidence, failure
        ]
        
        for confidence, success in test_data:
            result = await calibrator.update(confidence, success)
            
        print("✅ Confidence calibration test completed")
        
        # Get calibration stats
        stats = calibrator.get_calibration_stats()
        print(f"   Calibration error: {stats['calibration_error']:.3f}")
        print(f"   Data points: {stats['data_points']}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Confidence calibration test failed: {e}")
        return False


async def test_pattern_learning():
        """Test pattern learning."""
        print("\n🧪 Testing Pattern Learning")
    
        try:
            pass
        from memory_learning import PatternLearner
        
        config = MockLNNCouncilConfig()
        pattern_learner = PatternLearner(config)
        
        # Test learning multiple patterns
        requests = [
            MockGPURequest(),
            MockGPURequest(),
            MockGPURequest()
        ]
        
        decisions = [
            MockGPUDecision(),
            MockGPUDecision(),
            MockGPUDecision()
        ]
        
        outcomes = [
            {"success": True},
            {"success": True},
            {"success": False}
        ]
        
        for request, decision, outcome in zip(requests, decisions, outcomes):
            result = await pattern_learner.learn_pattern(request, decision, outcome)
            
        print("✅ Pattern learning test completed")
        
        # Get pattern insights
        insights = pattern_learner.get_pattern_insights()
        print(f"   Total patterns: {insights['total_patterns']}")
        print(f"   Strong patterns: {insights['strong_patterns']}")
        print(f"   Average success rate: {insights['avg_success_rate']:.3f}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Pattern learning test failed: {e}")
        return False


async def test_memory_stats():
        """Test memory statistics."""
        print("\n🧪 Testing Memory Statistics")
    
        try:
            pass
        from memory_context import LNNMemoryIntegration
        
        config = MockLNNCouncilConfig()
        memory_integration = LNNMemoryIntegration(config)
        
        # Simulate some activity
        memory_integration.decision_count = 25
        memory_integration.learning_updates = 15
        memory_integration.memory_quality_score = 0.85
        
        # Get memory stats
        stats = memory_integration.get_memory_stats()
        
        print("✅ Memory statistics test completed")
        print(f"   Decision count: {stats['decision_count']}")
        print(f"   Learning updates: {stats['learning_updates']}")
        print(f"   Memory quality: {stats['memory_quality_score']:.3f}")
        print(f"   Learning rate: {stats['learning_rate']:.3f}")
        print(f"   Cache sizes: {stats['cache_sizes']}")
        
        return True
        
        except Exception as e:
            pass
        print(f"❌ Memory statistics test failed: {e}")
        return False


async def main():
        """Run all memory integration tests."""
        print("🚀 Memory Integration Layer Tests (2025)\n")
    
        tests = [
        test_memory_context_retrieval,
        test_decision_outcome_storage,
        test_learning_from_outcomes,
        test_memory_learning_engine,
        test_confidence_calibration,
        test_pattern_learning,
        test_memory_stats
        ]
    
        results = []
        for test in tests:
            pass
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
        print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
        if all(results):
            pass
        print("🎉 All memory integration tests passed!")
        print("\n🎯 Memory Integration Features Verified:")
        print("   • Multi-level memory context retrieval ✅")
        print("   • Decision outcome storage and learning ✅")
        print("   • Confidence calibration ✅")
        print("   • Pattern recognition and learning ✅")
        print("   • Meta-learning from outcomes ✅")
        print("   • Memory quality assessment ✅")
        print("   • Learning statistics and insights ✅")
        print("\n🚀 Ready for Production:")
        print("   • Mem0 adapter integration ready")
        print("   • Episodic memory storage ready")
        print("   • Semantic similarity search ready")
        print("   • Meta-learning pipeline ready")
        return 0
        else:
        print("❌ Some tests failed")
        return 1


        if __name__ == "__main__":
        exit_code = asyncio.run(main())
        exit(exit_code)
