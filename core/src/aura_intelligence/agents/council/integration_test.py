#!/usr/bin/env python3
"""
Simple integration test for LNN Council Agent.

Tests basic functionality without complex dependencies.
"""

import asyncio
import sys
import os

# Add the core src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lnn_council_agent import (
    LNNCouncilAgent,
    LNNCouncilConfig,
    GPUAllocationRequest,
    GPUAllocationDecision
)


async def test_basic_functionality():
    """Test basic agent functionality."""
    print("🧪 Testing LNN Council Agent Basic Functionality")
    
    # Create configuration
    config = LNNCouncilConfig(
        name="test_agent",
        input_size=64,  # Smaller for testing
        output_size=32,
        hidden_sizes=[32],
        use_gpu=False,  # Disable GPU for testing
        mixed_precision=False,
        compile_mode=None,
        enable_detailed_logging=True
    )
    
    print(f"✅ Configuration created: {config.name}")
    
    # Create agent
    try:
        agent = LNNCouncilAgent(config)
        print(f"✅ Agent initialized: {agent.name}")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Create test request
    request = GPUAllocationRequest(
        user_id="test_user",
        project_id="test_project", 
        gpu_type="A100",
        gpu_count=1,
        memory_gb=20,
        compute_hours=8.0,
        priority=5
    )
    
    print(f"✅ Request created: {request.request_id}")
    
    # Test configuration validation
    try:
        config.validate()
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test state creation
    try:
        state = agent._create_initial_state(request)
        print(f"✅ Initial state created: {state.current_step}")
    except Exception as e:
        print(f"❌ State creation failed: {e}")
        return False
    
    # Test individual steps
    try:
        # Test analyze request
        state = await agent._analyze_request(state)
        print(f"✅ Analyze request completed: complexity = {state.context.get('request_complexity', 'N/A')}")
        
        # Test gather context
        state = await agent._gather_context(state)
        print(f"✅ Context gathering completed: {len(state.context_cache)} items")
        
        # Test neural inference (this will initialize the LNN)
        state = await agent._neural_inference(state)
        print(f"✅ Neural inference completed: decision = {state.context.get('neural_decision')}, confidence = {state.confidence_score:.3f}")
        
        # Test validation
        state = await agent._validate_decision(state)
        print(f"✅ Decision validation completed: passed = {state.validation_passed}")
        
        # Test output extraction
        state.completed = True
        output = agent._extract_output(state)
        print(f"✅ Output extraction completed: {output.decision} with {output.confidence_score:.3f} confidence")
        
    except Exception as e:
        print(f"❌ Step execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test health check
    try:
        health = await agent.health_check()
        print(f"✅ Health check completed: status = {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True


async def test_full_workflow():
    """Test the complete workflow."""
    print("\n🧪 Testing Complete Workflow")
    
    config = LNNCouncilConfig(
        name="workflow_test_agent",
        input_size=64,
        output_size=32,
        hidden_sizes=[32],
        use_gpu=False,
        mixed_precision=False,
        compile_mode=None
    )
    
    agent = LNNCouncilAgent(config)
    
    request = GPUAllocationRequest(
        user_id="workflow_user",
        project_id="workflow_project",
        gpu_type="H100",
        gpu_count=2,
        memory_gb=40,
        compute_hours=12.0,
        priority=8
    )
    
    try:
        # This should run the complete workflow
        result = await agent.process(request)
        
        print(f"✅ Workflow completed successfully!")
        print(f"   Request ID: {result.request_id}")
        print(f"   Decision: {result.decision}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Inference Time: {result.inference_time_ms:.1f}ms")
        print(f"   Fallback Used: {result.fallback_used}")
        print(f"   Reasoning Steps: {len(result.reasoning_path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_scenarios():
    """Test error handling scenarios."""
    print("\n🧪 Testing Error Scenarios")
    
    # Test invalid configuration
    try:
        invalid_config = LNNCouncilConfig(
            name="",  # Invalid empty name
            confidence_threshold=1.5  # Invalid threshold
        )
        invalid_config.validate()
        print("❌ Should have failed validation")
        return False
    except ValueError:
        print("✅ Invalid configuration properly rejected")
    
    # Test invalid request
    try:
        invalid_request = GPUAllocationRequest(
            user_id="test",
            project_id="test",
            gpu_type="INVALID_GPU",  # Invalid GPU type
            gpu_count=1,
            memory_gb=10,
            compute_hours=1.0
        )
        print("❌ Should have failed request validation")
        return False
    except ValueError:
        print("✅ Invalid request properly rejected")
    
    print("✅ Error scenarios handled correctly")
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting LNN Council Agent Integration Tests\n")
    
    tests = [
        test_basic_functionality,
        test_full_workflow,
        test_error_scenarios
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)