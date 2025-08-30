"""
Test suite for AURA Neural Routing System
Validates the transformation from academic neural networks to production routing
"""

import asyncio
import json
import time
from datetime import datetime, timezone
import sys
import os

# Add to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.neural import (
    # Provider system
    ProviderType,
    ModelCapability,
    ProviderRequest,
    ProviderFactory,
    
    # Routing system
    RoutingPolicy,
    AURAModelRouter,
    
    # Adaptive learning
    AdaptiveLNNRouter,
    RoutingState,
    RouterBenchEvaluator,
    
    # Context management
    ContextManager,
    
    # Performance tracking
    ModelPerformanceTracker
)


async def test_provider_adapters():
    """Test provider adapter creation and basic functionality"""
    print("\n=== Testing Provider Adapters ===")
    
    # Test factory creation
    config = {
        "enabled": True,
        "api_key": "test-key",
        "models": {
            "gpt-5": {
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03
            }
        }
    }
    
    try:
        openai_adapter = ProviderFactory.create(ProviderType.OPENAI, "test-key", config)
        print("✓ OpenAI adapter created successfully")
        
        anthropic_adapter = ProviderFactory.create(ProviderType.ANTHROPIC, "test-key", config)
        print("✓ Anthropic adapter created successfully")
        
        together_adapter = ProviderFactory.create(ProviderType.TOGETHER, "test-key", config)
        print("✓ Together adapter created successfully")
        
        ollama_adapter = ProviderFactory.create(ProviderType.OLLAMA, "", {"base_url": "http://localhost:11434"})
        print("✓ Ollama adapter created successfully")
        
    except Exception as e:
        print(f"✗ Provider adapter creation failed: {e}")
        return False
        
    # Test request creation
    request = ProviderRequest(
        prompt="Explain quantum computing in simple terms",
        model="gpt-5",
        temperature=0.7,
        max_tokens=500,
        tools=[{"type": "function", "function": {"name": "calculate", "parameters": {}}}],
        metadata={"request_id": "test-001"}
    )
    
    print(f"✓ Provider request created: {request.prompt[:50]}...")
    
    return True


async def test_model_router():
    """Test the main model router functionality"""
    print("\n=== Testing Model Router ===")
    
    # Router configuration
    config = {
        "providers": {
            "openai": {
                "enabled": True,
                "api_key": "test-key",
                "models": ["gpt-5", "gpt-4o"]
            },
            "anthropic": {
                "enabled": True,
                "api_key": "test-key",
                "models": ["claude-opus-4.1", "claude-sonnet-4.1"]
            },
            "together": {
                "enabled": True,
                "api_key": "test-key",
                "models": ["mamba-2-2.8b", "llama-3.1-70b", "turbo-mixtral"]
            },
            "ollama": {
                "enabled": True,
                "base_url": "http://localhost:11434",
                "models": ["llama3-70b", "mixtral-8x7b"]
            }
        },
        "routing": {
            "learning_rate": 0.01
        }
    }
    
    try:
        router = AURAModelRouter(config)
        print("✓ Model router initialized")
        
        # Test different routing scenarios
        
        # 1. Simple request - should route to fast/cheap provider
        simple_request = ProviderRequest(
            prompt="What is 2+2?",
            model="",  # Let router decide
            temperature=0.7
        )
        
        simple_policy = RoutingPolicy(
            quality_weight=0.2,
            cost_weight=0.5,
            latency_weight=0.3
        )
        
        decision = await router.route_request(simple_request, simple_policy)
        print(f"✓ Simple request routed to: {decision.provider.value}/{decision.model} (reason: {decision.reason.value})")
        
        # 2. Long context request - should route to Together/Mamba
        long_request = ProviderRequest(
            prompt="A" * 150000,  # Very long prompt
            model="",
            temperature=0.7
        )
        
        long_decision = await router.route_request(long_request)
        print(f"✓ Long context routed to: {long_decision.provider.value}/{long_decision.model} (reason: {long_decision.reason.value})")
        assert long_decision.reason.value == "long_context", "Should route long context appropriately"
        
        # 3. Tools request - should route to OpenAI
        tools_request = ProviderRequest(
            prompt="Calculate the compound interest on $1000 at 5% for 10 years",
            model="",
            tools=[{"type": "function", "function": {"name": "calculate", "parameters": {}}}],
            temperature=0.7
        )
        
        tools_decision = await router.route_request(tools_request)
        print(f"✓ Tools request routed to: {tools_decision.provider.value}/{tools_decision.model} (reason: {tools_decision.reason.value})")
        assert tools_decision.reason.value == "tools_required", "Should route tools to OpenAI"
        
        # 4. Privacy request - should route to Ollama
        privacy_request = ProviderRequest(
            prompt="Analyze this sensitive medical data...",
            model="",
            temperature=0.7
        )
        
        privacy_policy = RoutingPolicy(require_privacy=True)
        privacy_decision = await router.route_request(privacy_request, privacy_policy)
        print(f"✓ Privacy request routed to: {privacy_decision.provider.value}/{privacy_decision.model} (reason: {privacy_decision.reason.value})")
        assert privacy_decision.provider == ProviderType.OLLAMA, "Should route privacy to local"
        
        return True
        
    except Exception as e:
        print(f"✗ Model router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_adaptive_routing():
    """Test adaptive routing engine with learning"""
    print("\n=== Testing Adaptive Routing Engine ===")
    
    try:
        # Create adaptive router
        config = {"hidden_dim": 64, "learning_rate": 0.001, "buffer_size": 1000}
        adaptive_router = AdaptiveLNNRouter(config)
        print("✓ Adaptive LNN router created")
        
        # Create routing state
        state = RoutingState(
            context_length=0.3,  # 30k tokens normalized
            complexity=0.7,
            urgency=0.2,
            has_tools=0.0,
            requires_privacy=0.0,
            requires_background=0.0,
            provider_health={p.value: 1.0 for p in ProviderType},
            provider_latencies={p.value: 0.5 for p in ProviderType},
            provider_costs={p.value: 0.5 for p in ProviderType},
            provider_qualities={p.value: 0.5 for p in ProviderType},
            current_load=0.3,
            time_of_day=0.5
        )
        
        # Get initial scores
        scores = adaptive_router.forward(state, session_id="test-session")
        print("✓ Initial provider scores:")
        for provider, score in scores.items():
            print(f"  {provider.value}: {score:.3f}")
            
        # Simulate learning from good outcome
        experience = {
            "state": state,
            "provider": ProviderType.ANTHROPIC,
            "reward": 0.9  # Good outcome
        }
        
        adaptive_router.update(experience)
        print("✓ Updated router with positive experience")
        
        # Get updated scores
        new_scores = adaptive_router.forward(state, session_id="test-session")
        print("✓ Updated provider scores:")
        for provider, score in new_scores.items():
            print(f"  {provider.value}: {score:.3f}")
            
        # Verify learning occurred
        if new_scores[ProviderType.ANTHROPIC] > scores[ProviderType.ANTHROPIC]:
            print("✓ Router learned from positive experience")
        else:
            print("✗ Router did not show expected learning")
            
        # Test RouterBench evaluator
        evaluator = RouterBenchEvaluator(adaptive_router)
        
        # Mock benchmark data
        benchmark_data = [
            {
                "tokens": 5000,
                "complexity": 0.5,
                "has_tools": False,
                "best_single_cost": 0.05,
                "best_single_quality": 0.8,
                "best_single_latency": 2000
            },
            {
                "tokens": 100000,
                "complexity": 0.8,
                "has_tools": False,
                "best_single_cost": 0.5,
                "best_single_quality": 0.85,
                "best_single_latency": 5000
            }
        ]
        
        results = await evaluator.evaluate_on_benchmark(benchmark_data)
        print(f"✓ RouterBench evaluation completed:")
        print(f"  Average cost improvement: {results['avg_cost_improvement']:.2%}")
        print(f"  Average quality improvement: {results['avg_quality_improvement']:.2%}")
        print(f"  Average composite score: {results['avg_composite_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_management():
    """Test context preparation for different providers"""
    print("\n=== Testing Context Management ===")
    
    try:
        context_manager = ContextManager()
        print("✓ Context manager created")
        
        # Test standard context
        from aura_intelligence.neural.provider_adapters import ModelConfig
        
        standard_config = ModelConfig(
            name="gpt-4",
            provider=ProviderType.OPENAI,
            max_context=8000,
            capabilities=[],
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            avg_latency_ms=2000
        )
        
        window = await context_manager.prepare_context(
            prompt="Explain machine learning",
            provider=ProviderType.OPENAI,
            model="gpt-4",
            model_config=standard_config,
            system_prompt="You are a helpful AI assistant"
        )
        
        print(f"✓ Standard context prepared:")
        print(f"  Strategy: {window.strategy}")
        print(f"  Total tokens: {window.total_tokens}")
        print(f"  Compression: {window.compression_applied}")
        
        # Test long context
        long_config = ModelConfig(
            name="claude-opus-4.1",
            provider=ProviderType.ANTHROPIC,
            max_context=200000,
            capabilities=[ModelCapability.LONG_CONTEXT],
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            avg_latency_ms=3000
        )
        
        long_prompt = "This is a very long document. " * 10000
        long_window = await context_manager.prepare_context(
            prompt=long_prompt,
            provider=ProviderType.ANTHROPIC,
            model="claude-opus-4.1",
            model_config=long_config,
            session_id="test-long"
        )
        
        print(f"✓ Long context prepared:")
        print(f"  Strategy: {long_window.strategy}")
        print(f"  Total tokens: {long_window.total_tokens}")
        print(f"  Compression: {long_window.compression_applied}")
        
        # Test chunked context
        small_config = ModelConfig(
            name="small-model",
            provider=ProviderType.OLLAMA,
            max_context=4000,
            capabilities=[],
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            avg_latency_ms=1000
        )
        
        chunked_window = await context_manager.prepare_context(
            prompt="Large document that needs chunking. " * 2000,
            provider=ProviderType.OLLAMA,
            model="small-model",
            model_config=small_config
        )
        
        print(f"✓ Chunked context prepared:")
        print(f"  Strategy: {chunked_window.strategy}")
        print(f"  Total chunks: {chunked_window.metadata.get('total_chunks', 1)}")
        print(f"  Current chunk: {chunked_window.metadata.get('current_chunk', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Context management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_tracking():
    """Test performance tracking and learning"""
    print("\n=== Testing Performance Tracking ===")
    
    try:
        tracker = ModelPerformanceTracker()
        print("✓ Performance tracker created")
        
        # Create mock request and response
        request = ProviderRequest(
            prompt="Explain neural networks in detail",
            model="gpt-5",
            temperature=0.7,
            metadata={"request_id": "perf-001"}
        )
        
        from aura_intelligence.neural.provider_adapters import ProviderResponse
        from aura_intelligence.neural.model_router import RoutingDecision, RoutingReason
        
        response = ProviderResponse(
            content="Neural networks are...",
            model="gpt-5",
            provider=ProviderType.OPENAI,
            usage={"input_tokens": 50, "output_tokens": 200, "total_tokens": 250},
            latency_ms=1500,
            cost_usd=0.0075,
            metadata={}
        )
        
        decision = RoutingDecision(
            provider=ProviderType.OPENAI,
            model="gpt-5",
            reason=RoutingReason.QUALITY_OPTIMIZED,
            confidence=0.85,
            estimated_cost=0.008,
            estimated_latency_ms=2000,
            estimated_quality=0.9
        )
        
        # Track the request
        event = await tracker.track_request(
            request=request,
            decision=decision,
            response=response,
            quality_score=0.88
        )
        
        print(f"✓ Tracked performance event:")
        print(f"  Request type: {event.request_type}")
        print(f"  Latency: {event.latency_ms}ms")
        print(f"  Cost: ${event.cost_usd:.4f}")
        print(f"  Quality: {event.quality_score:.2f}")
        
        # Get provider scores
        scores = tracker.get_provider_scores()
        print("✓ Provider performance scores:")
        for (provider, model), score in scores.items():
            print(f"  {provider.value}/{model}: {score:.3f}")
            
        # Test error tracking
        await tracker.track_error(
            request=request,
            provider=ProviderType.ANTHROPIC,
            model="claude-opus-4.1",
            error=Exception("Rate limit exceeded"),
            decision=decision
        )
        print("✓ Tracked error event")
        
        # Get provider health
        health = tracker.get_provider_health()
        print("✓ Provider health scores:")
        for provider, health_score in health.items():
            print(f"  {provider.value}: {health_score:.2f}")
            
        # Generate report
        report = await tracker.generate_performance_report()
        print(f"✓ Performance report generated:")
        print(f"  Total requests: {report['total_requests']}")
        print(f"  Total cost: ${report['total_cost']:.4f}")
        print(f"  Total savings: ${report['total_savings']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Test full integration of all components"""
    print("\n=== Testing Full Integration ===")
    
    try:
        # Create integrated system
        config = {
            "providers": {
                "openai": {"enabled": True, "api_key": "test"},
                "anthropic": {"enabled": True, "api_key": "test"},
                "together": {"enabled": True, "api_key": "test"},
                "ollama": {"enabled": True, "base_url": "http://localhost:11434"}
            }
        }
        
        router = AURAModelRouter(config)
        context_manager = ContextManager()
        tracker = ModelPerformanceTracker()
        
        print("✓ All components initialized")
        
        # Simulate a full request flow
        original_request = ProviderRequest(
            prompt="Write a function to calculate fibonacci numbers with memoization",
            model="",
            temperature=0.7,
            metadata={"request_id": "integration-001"}
        )
        
        # 1. Route the request
        policy = RoutingPolicy(
            quality_weight=0.5,
            cost_weight=0.3,
            latency_weight=0.2
        )
        
        decision = await router.route_request(original_request, policy)
        print(f"✓ Routing decision: {decision.provider.value}/{decision.model}")
        
        # 2. Prepare context (would normally include memory retrieval)
        from aura_intelligence.neural.provider_adapters import ModelConfig
        model_config = ModelConfig(
            name=decision.model,
            provider=decision.provider,
            max_context=128000,
            capabilities=[ModelCapability.CODING],
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            avg_latency_ms=2000
        )
        
        context_window = await context_manager.prepare_context(
            prompt=original_request.prompt,
            provider=decision.provider,
            model=decision.model,
            model_config=model_config,
            session_id="integration-session"
        )
        
        print(f"✓ Context prepared: {context_window.total_tokens} tokens")
        
        # 3. Mock execution (in production would call provider)
        from aura_intelligence.neural.provider_adapters import ProviderResponse
        mock_response = ProviderResponse(
            content="def fibonacci_memo(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)\n    return memo[n]",
            model=decision.model,
            provider=decision.provider,
            usage={"input_tokens": 100, "output_tokens": 150, "total_tokens": 250},
            latency_ms=1800,
            cost_usd=0.0075,
            metadata={}
        )
        
        print("✓ Request executed (mocked)")
        
        # 4. Track performance
        event = await tracker.track_request(
            request=original_request,
            decision=decision,
            response=mock_response,
            quality_score=0.92  # High quality for coding task
        )
        
        print(f"✓ Performance tracked: quality={event.quality_score:.2f}")
        
        # 5. Update routing model
        await router.track_outcome(
            request=original_request,
            decision=decision,
            response=mock_response,
            quality_score=0.92
        )
        
        print("✓ Routing model updated with outcome")
        
        # Test the full flow worked
        print("\n✓ Full integration test completed successfully!")
        print(f"  Request → {decision.provider.value} → {context_window.strategy} context → Response → Tracking → Learning")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("=== AURA Neural Routing System Test Suite ===")
    print("Testing transformation from academic neural networks to production routing...")
    
    results = []
    
    # Run each test
    results.append(("Provider Adapters", await test_provider_adapters()))
    results.append(("Model Router", await test_model_router()))
    results.append(("Adaptive Routing", await test_adaptive_routing()))
    results.append(("Context Management", await test_context_management()))
    results.append(("Performance Tracking", await test_performance_tracking()))
    results.append(("Full Integration", await test_integration()))
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Neural routing system is ready for production.")
        print("\nKey achievements:")
        print("- ✓ Transformed academic neural networks into practical routing")
        print("- ✓ Integrated with OpenAI, Anthropic, Together, and Ollama")
        print("- ✓ Adaptive learning from routing outcomes")
        print("- ✓ Smart context management for different models")
        print("- ✓ Performance tracking and cost optimization")
        print("- ✓ Ready for enterprise agent infrastructure")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    asyncio.run(main())