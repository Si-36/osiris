#!/usr/bin/env python3
"""
REAL System Integration Test - No Mocks, No Fakes

Tests our ACTUAL LNN Council Agent system to see if it really works.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_real_system():
    """Test if our real system actually works."""
    print("🔥 REAL System Integration Test - No Mocks, No Lies")
    print("=" * 60)
    
    test_results = {
        "imports": False,
        "agent_creation": False,
        "request_creation": False,
        "decision_processing": False,
        "end_to_end": False
    }
    
    # Test 1: Can we import our real modules?
    print("\n🔍 Test 1: Real Module Imports")
    try:
        from aura_intelligence.agents.council.models import GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState
        from aura_intelligence.agents.council.core_agent import LNNCouncilAgent
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        
        print("   ✅ Successfully imported real modules")
        test_results["imports"] = True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        print(f"   📋 Traceback: {traceback.format_exc()}")
        return test_results
    
    # Test 2: Can we create a real agent?
    print("\n🔍 Test 2: Real Agent Creation")
    try:
        config = {
            "name": "real_integration_test",
            "enable_fallback": True
        }
        
        agent = LNNCouncilAgent(config)
        print("   ✅ Successfully created real LNN Council Agent")
        test_results["agent_creation"] = True
        
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        print(f"   📋 Traceback: {traceback.format_exc()}")
        return test_results
    
    # Test 3: Can we create a real request?
    print("\n🔍 Test 3: Real Request Creation")
    try:
        request = GPUAllocationRequest(
            user_id="real_test_user",
            project_id="real_test_project", 
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7
        )
        
        print(f"   ✅ Created real request: {request.request_id}")
        print(f"   ✅ Request validation passed")
        test_results["request_creation"] = True
        
    except Exception as e:
        print(f"   ❌ Request creation failed: {e}")
        print(f"   📋 Traceback: {traceback.format_exc()}")
        return test_results
    
    # Test 4: Can we actually process a decision?
    print("\n🔍 Test 4: Real Decision Processing")
    try:
        # This is the REAL test - does our system actually work?
        decision = asyncio.run(agent.process(request))
        
        print(f"   ✅ Got real decision: {decision.decision}")
        print(f"   ✅ Confidence: {decision.confidence_score}")
        print(f"   ✅ Reasoning steps: {len(decision.reasoning_path)}")
        print(f"   ✅ Inference time: {decision.inference_time_ms}ms")
        
        test_results["decision_processing"] = True
        
    except Exception as e:
        print(f"   ❌ Decision processing failed: {e}")
        print(f"   📋 Traceback: {traceback.format_exc()}")
        return test_results
    
    # Test 5: End-to-end validation
    print("\n🔍 Test 5: End-to-End Validation")
    try:
        # Validate the decision makes sense
        if decision.decision not in ['approve', 'deny', 'defer']:
            raise ValueError(f"Invalid decision: {decision.decision}")
        
        if not (0.0 <= decision.confidence_score <= 1.0):
            raise ValueError(f"Invalid confidence: {decision.confidence_score}")
        
        if decision.request_id != request.request_id:
            raise ValueError("Request ID mismatch")
        
        if len(decision.reasoning_path) == 0:
            raise ValueError("No reasoning provided")
        
        print("   ✅ Decision validation passed")
        print("   ✅ All fields properly populated")
        print("   ✅ End-to-end workflow successful")
        
        test_results["end_to_end"] = True
        
    except Exception as e:
        print(f"   ❌ End-to-end validation failed: {e}")
        return test_results
    
    return test_results


def main():
    """Run the real system test."""
    results = test_real_system()
    
    # Calculate results
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 REAL Integration Test Results: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 REAL SYSTEM WORKS! All tests passed!")
        print("\n✅ Our LNN Council Agent actually works:")
        print("   • Real imports ✅")
        print("   • Real agent creation ✅") 
        print("   • Real request processing ✅")
        print("   • Real decision making ✅")
        print("   • Real end-to-end workflow ✅")
        print("\n🚀 Task 11 ACTUALLY completed!")
    else:
        print("❌ REAL SYSTEM IS BROKEN!")
        print("\n🔧 What's broken:")
        for test_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        print("\n💡 We need to fix these issues before claiming Task 11 is done.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)