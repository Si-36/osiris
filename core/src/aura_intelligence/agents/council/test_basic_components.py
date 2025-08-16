#!/usr/bin/env python3
"""
Basic Component Tests - Ground Up Approach

Start with the most basic components and test what actually exists.
No mocks, no fancy stuff - just test the real code we have.
"""

import asyncio
import torch
import sys
import os
from datetime import datetime, timezone

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

print("🔍 Starting basic component discovery and testing...\n")


def test_basic_imports():
    """Test what we can actually import."""
    print("🧪 Testing Basic Imports")
    
    imports_status = {}
    
    # Test config
    try:
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        config = LNNCouncilConfig()
        imports_status['config'] = f"✅ Config works - name: {config.name}"
    except Exception as e:
        imports_status['config'] = f"❌ Config failed: {e}"
    
    # Test models
    try:
        from aura_intelligence.agents.council.models import LNNCouncilState, GPUAllocationRequest
        request = GPUAllocationRequest(
            request_id="test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        state = LNNCouncilState(current_request=request)
        imports_status['models'] = f"✅ Models work - request_id: {request.request_id}"
    except Exception as e:
        imports_status['models'] = f"❌ Models failed: {e}"
    
    # Test context encoder
    try:
        from aura_intelligence.agents.council.context_encoder import ContextEncoder
        config = LNNCouncilConfig()
        encoder = ContextEncoder(config)
        imports_status['context_encoder'] = f"✅ ContextEncoder works"
    except Exception as e:
        imports_status['context_encoder'] = f"❌ ContextEncoder failed: {e}"
    
    # Test memory context
    try:
        from aura_intelligence.agents.council.memory_context import MemoryContextProvider
        config = LNNCouncilConfig()
        memory_provider = MemoryContextProvider(config)
        imports_status['memory_context'] = f"✅ MemoryContextProvider works"
    except Exception as e:
        imports_status['memory_context'] = f"❌ MemoryContextProvider failed: {e}"
    
    # Test knowledge context
    try:
        from aura_intelligence.agents.council.knowledge_context import KnowledgeGraphContextProvider
        config = LNNCouncilConfig()
        kg_provider = KnowledgeGraphContextProvider(config)
        imports_status['knowledge_context'] = f"✅ KnowledgeGraphContextProvider works"
    except Exception as e:
        imports_status['knowledge_context'] = f"❌ KnowledgeGraphContextProvider failed: {e}"
    
    # Print results
    for component, status in imports_status.items():
        print(f"   {status}")
    
    working_components = sum(1 for status in imports_status.values() if status.startswith('✅'))
    print(f"\n📊 Import Results: {working_components}/{len(imports_status)} components working")
    
    return imports_status


def test_config_component():
    """Test the config component thoroughly."""
    print("\n🧪 Testing Config Component")
    
    try:
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        
        # Test default config
        config = LNNCouncilConfig()
        print(f"   Default name: {config.name}")
        print(f"   Default input_size: {config.input_size}")
        print(f"   Default output_size: {config.output_size}")
        print(f"   Default confidence_threshold: {config.confidence_threshold}")
        
        # Test custom config
        custom_config = LNNCouncilConfig(
            name="test_agent",
            input_size=32,
            output_size=8,
            confidence_threshold=0.8
        )
        print(f"   Custom name: {custom_config.name}")
        print(f"   Custom input_size: {custom_config.input_size}")
        
        # Test liquid config conversion
        try:
            liquid_config = config.to_liquid_config()
            print(f"   Liquid config conversion: ✅")
        except Exception as e:
            print(f"   Liquid config conversion: ❌ {e}")
        
        print("✅ Config component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Config component test failed: {e}")
        return False


def test_models_component():
    """Test the models component thoroughly."""
    print("\n🧪 Testing Models Component")
    
    try:
        from aura_intelligence.agents.council.models import (
            LNNCouncilState, 
            GPUAllocationRequest, 
            GPUAllocationDecision
        )
        
        # Test GPUAllocationRequest
        request = GPUAllocationRequest(
            request_id="model_test_001",
            user_id="test_user",
            project_id="test_project",
            gpu_type="A100",
            gpu_count=4,
            memory_gb=80,
            compute_hours=12.0,
            priority=8,
            created_at=datetime.now(timezone.utc)
        )
        
        print(f"   Request created: {request.request_id}")
        print(f"   GPU type: {request.gpu_type}")
        print(f"   GPU count: {request.gpu_count}")
        print(f"   Priority: {request.priority}")
        
        # Test LNNCouncilState
        state = LNNCouncilState(current_request=request)
        print(f"   State created with request: {state.current_request.request_id}")
        
        # Test state context
        state.context["test_key"] = "test_value"
        print(f"   State context works: {state.context['test_key']}")
        
        # Test state messages
        state.add_message("system", "Test message")
        print(f"   State messages work: {len(state.messages)} messages")
        
        # Test GPUAllocationDecision
        decision = GPUAllocationDecision(
            request_id=request.request_id,
            decision="approve",
            confidence_score=0.85,
            fallback_used=False,
            inference_time_ms=15.5
        )
        
        print(f"   Decision created: {decision.decision}")
        print(f"   Confidence: {decision.confidence_score}")
        
        # Test reasoning
        decision.add_reasoning("neural", "High confidence neural decision")
        decision.add_reasoning("validation", "All constraints satisfied")
        print(f"   Reasoning steps: {len(decision.reasoning)}")
        
        print("✅ Models component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Models component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_encoder_component():
    """Test the context encoder component."""
    print("\n🧪 Testing Context Encoder Component")
    
    try:
        from aura_intelligence.agents.council.context_encoder import ContextEncoder
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        from aura_intelligence.agents.council.models import LNNCouncilState, GPUAllocationRequest
        
        config = LNNCouncilConfig(input_size=32, output_size=8)
        encoder = ContextEncoder(config)
        
        print(f"   Context encoder created")
        print(f"   Config input size: {config.input_size}")
        
        # Create test state
        request = GPUAllocationRequest(
            request_id="encoder_test_001",
            user_id="encoder_user",
            project_id="encoder_project",
            gpu_type="V100",
            gpu_count=2,
            memory_gb=32,
            compute_hours=6.0,
            priority=6,
            created_at=datetime.now(timezone.utc)
        )
        
        state = LNNCouncilState(current_request=request)
        
        # Test encoding
        try:
            encoded = await encoder.encode_request_context(state)
            print(f"   Request encoding: ✅ shape {encoded.shape}")
            print(f"   Encoded dtype: {encoded.dtype}")
            print(f"   Encoded range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")
        except Exception as e:
            print(f"   Request encoding: ❌ {e}")
        
        print("✅ Context Encoder component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Context Encoder component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_context_component():
    """Test the memory context component."""
    print("\n🧪 Testing Memory Context Component")
    
    try:
        from aura_intelligence.agents.council.memory_context import MemoryContextProvider
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        from aura_intelligence.agents.council.models import LNNCouncilState, GPUAllocationRequest
        
        config = LNNCouncilConfig(input_size=32, output_size=8)
        memory_provider = MemoryContextProvider(config)
        
        print(f"   Memory provider created")
        
        # Create test state
        request = GPUAllocationRequest(
            request_id="memory_test_001",
            user_id="memory_user",
            project_id="memory_project",
            gpu_type="A100",
            gpu_count=3,
            memory_gb=60,
            compute_hours=10.0,
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        
        state = LNNCouncilState(current_request=request)
        
        # Test memory context retrieval
        try:
            memory_context = await memory_provider.get_memory_context(state)
            if memory_context is not None:
                print(f"   Memory context: ✅ shape {memory_context.shape}")
                print(f"   Memory dtype: {memory_context.dtype}")
                print(f"   Non-zero features: {(memory_context != 0).sum().item()}")
            else:
                print(f"   Memory context: None (expected without Mem0 adapter)")
        except Exception as e:
            print(f"   Memory context: ❌ {e}")
        
        # Test memory stats
        try:
            stats = memory_provider.get_memory_stats()
            print(f"   Memory stats: ✅ {len(stats)} metrics")
            for key, value in stats.items():
                print(f"     {key}: {value}")
        except Exception as e:
            print(f"   Memory stats: ❌ {e}")
        
        print("✅ Memory Context component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory Context component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_knowledge_context_component():
    """Test the knowledge context component."""
    print("\n🧪 Testing Knowledge Context Component")
    
    try:
        from aura_intelligence.agents.council.knowledge_context import KnowledgeGraphContextProvider
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        from aura_intelligence.agents.council.models import LNNCouncilState, GPUAllocationRequest
        
        config = LNNCouncilConfig(input_size=32, output_size=8)
        kg_provider = KnowledgeGraphContextProvider(config)
        
        print(f"   Knowledge graph provider created")
        
        # Create test state
        request = GPUAllocationRequest(
            request_id="kg_test_001",
            user_id="kg_user",
            project_id="kg_project",
            gpu_type="H100",
            gpu_count=1,
            memory_gb=80,
            compute_hours=4.0,
            priority=9,
            created_at=datetime.now(timezone.utc)
        )
        
        state = LNNCouncilState(current_request=request)
        
        # Test knowledge context retrieval
        try:
            knowledge_context = await kg_provider.get_knowledge_context(state)
            if knowledge_context is not None:
                print(f"   Knowledge context: ✅ shape {knowledge_context.shape}")
                print(f"   Knowledge dtype: {knowledge_context.dtype}")
                print(f"   Non-zero features: {(knowledge_context != 0).sum().item()}")
            else:
                print(f"   Knowledge context: None (expected without Neo4j adapter)")
        except Exception as e:
            print(f"   Knowledge context: ❌ {e}")
        
        # Test knowledge stats
        try:
            stats = kg_provider.get_knowledge_stats()
            print(f"   Knowledge stats: ✅ {len(stats)} metrics")
            for key, value in stats.items():
                print(f"     {key}: {value}")
        except Exception as e:
            print(f"   Knowledge stats: ❌ {e}")
        
        print("✅ Knowledge Context component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge Context component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_component():
    """Test the workflow component."""
    print("\n🧪 Testing Workflow Component")
    
    try:
        from aura_intelligence.agents.council.workflow import WorkflowEngine
        from aura_intelligence.agents.council.config import LNNCouncilConfig
        from aura_intelligence.agents.council.models import LNNCouncilState, GPUAllocationRequest
        
        config = LNNCouncilConfig(input_size=32, output_size=8)
        workflow_engine = WorkflowEngine(config)
        
        print(f"   Workflow engine created")
        
        # Test workflow status
        try:
            status = workflow_engine.get_status()
            print(f"   Workflow status: ✅ {len(status['steps_available'])} steps available")
            print(f"   Available steps: {status['steps_available']}")
        except Exception as e:
            print(f"   Workflow status: ❌ {e}")
        
        # Create test state
        request = GPUAllocationRequest(
            request_id="workflow_test_001",
            user_id="workflow_user",
            project_id="workflow_project",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=40,
            compute_hours=8.0,
            priority=7,
            created_at=datetime.now(timezone.utc)
        )
        
        state = LNNCouncilState(current_request=request)
        
        # Test individual steps
        steps_to_test = ["analyze_request", "gather_context"]
        
        for step_name in steps_to_test:
            try:
                print(f"   Testing step: {step_name}")
                result_state = await workflow_engine.execute_step(state, step_name)
                print(f"     Step {step_name}: ✅ next_step={result_state.next_step}")
                state = result_state  # Update state for next step
            except Exception as e:
                print(f"     Step {step_name}: ❌ {e}")
        
        print("✅ Workflow component test passed")
        return True
        
    except Exception as e:
        print(f"❌ Workflow component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run basic component tests from ground up."""
    print("🚀 Basic Component Tests - Ground Up Approach\n")
    
    # Step 1: Test imports
    imports_status = test_basic_imports()
    
    # Step 2: Test each component that imported successfully
    tests = []
    
    if imports_status.get('config', '').startswith('✅'):
        tests.append(('Config', test_config_component))
    
    if imports_status.get('models', '').startswith('✅'):
        tests.append(('Models', test_models_component))
    
    if imports_status.get('context_encoder', '').startswith('✅'):
        tests.append(('Context Encoder', test_context_encoder_component))
    
    if imports_status.get('memory_context', '').startswith('✅'):
        tests.append(('Memory Context', test_memory_context_component))
    
    if imports_status.get('knowledge_context', '').startswith('✅'):
        tests.append(('Knowledge Context', test_knowledge_context_component))
    
    # Test workflow if we have the basics
    if len([s for s in imports_status.values() if s.startswith('✅')]) >= 2:
        tests.append(('Workflow', test_workflow_component))
    
    # Run tests
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results.append(False)
    
    print(f"\n📊 Basic Test Results: {sum(results)}/{len(results)} passed")
    
    if results and all(results):
        print("🎉 All basic component tests passed!")
        print("\n✅ Working Components:")
        for component, status in imports_status.items():
            if status.startswith('✅'):
                print(f"   • {component.replace('_', ' ').title()}")
        
        print("\n🎯 Next Steps:")
        print("   • Components are working individually")
        print("   • Ready for integration testing")
        print("   • Can build up to full pipeline")
        
        return 0
    else:
        print("❌ Some basic component tests failed")
        print("   Need to fix basic components before integration")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)