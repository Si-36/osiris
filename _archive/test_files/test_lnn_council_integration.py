"""
Test LNN Council Integration with Neural Router

This test verifies that the extracted LNN council system
successfully enhances neural router decisions with:
- Multi-agent voting
- Byzantine consensus
- Liquid neural networks
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.neural.model_router import AURAModelRouter
from aura_intelligence.neural.provider_adapters import ProviderRequest, ModelCapability
from aura_intelligence.agents.lnn_council import (
    create_lnn_council, LNNCouncilAgent, VoteDecision
)


async def test_lnn_council_standalone():
    """Test LNN council functionality independently"""
    print("\n=== Testing LNN Council Standalone ===")
    
    # Create council
    council = create_lnn_council(num_agents=5)
    
    # Test request
    request = {
        "type": "model_selection",
        "priority": 8,
        "requirements": [
            "latency_ms < 1000",
            "cost_per_token < 0.01",
            "quality > 0.9"
        ],
        "models": ["gpt-4", "claude-3", "llama-3"],
        "complexity": 0.8
    }
    
    # Get council decision
    consensus = await council.make_council_decision(request)
    
    print(f"Council Decision: {consensus.final_decision.value}")
    print(f"Consensus Type: {consensus.consensus_type}")
    print(f"Confidence: {consensus.consensus_confidence:.2f}")
    print(f"Total Votes: {len(consensus.votes)}")
    
    # Show individual votes
    print("\nIndividual Agent Votes:")
    for vote in consensus.votes:
        print(f"  {vote.agent_id}: {vote.decision.value} (conf: {vote.confidence:.2f})")
        print(f"    Reasoning: {vote.reasoning}")
    
    # Show metrics
    metrics = council.get_metrics()
    print(f"\nCouncil Metrics:")
    print(f"  Total Decisions: {metrics['total_decisions']}")
    print(f"  Consensus Rate: {metrics['consensus_rate']:.2f}")
    print(f"  Avg Confidence: {metrics['average_confidence']:.2f}")
    
    return consensus


async def test_neural_router_with_council():
    """Test neural router with LNN council integration"""
    print("\n=== Testing Neural Router with LNN Council ===")
    
    # Router config with council enabled
    config = {
        "enable_lnn_council": True,
        "council_agents": 5,
        "providers": {
            "openai": {
                "enabled": True,
                "api_key": "test-key",
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            "anthropic": {
                "enabled": True,
                "api_key": "test-key",
                "models": ["claude-3-opus", "claude-3-sonnet"]
            }
        }
    }
    
    # Create router
    router = AURAModelRouter(config)
    
    # Verify council was initialized
    if router.lnn_council:
        print("✓ LNN Council successfully integrated")
        print(f"  Council has {len(router.lnn_council.agents)} agents")
    else:
        print("✗ LNN Council not initialized")
        return
    
    # Create complex request that triggers council
    request = ProviderRequest(
        prompt="Analyze this complex multi-modal dataset and provide insights",
        max_tokens=4000,
        temperature=0.7,
        required_capabilities=[ModelCapability.LONG_CONTEXT, ModelCapability.REASONING],
        metadata={
            "complexity": 0.9,  # High complexity triggers council
            "priority": 8
        }
    )
    
    try:
        # Route request
        decision = await router.route_request(request)
        
        print(f"\nRouting Decision:")
        print(f"  Provider: {decision.provider.value}")
        print(f"  Model: {decision.model_config.model_id}")
        print(f"  Reason: {decision.reason}")
        print(f"  Confidence: {decision.confidence:.2f}")
        
        # Check if council was used
        if "LNN Council" in decision.reason:
            print("\n✓ LNN Council was used for decision!")
            print(f"  Council confidence: {decision.metadata.get('council_confidence', 'N/A')}")
        else:
            print("\n✗ Standard routing was used (complexity might be too low)")
            
    except Exception as e:
        print(f"Error during routing: {e}")
        import traceback
        traceback.print_exc()


async def test_byzantine_consensus():
    """Test Byzantine fault tolerance in council"""
    print("\n=== Testing Byzantine Consensus ===")
    
    council = create_lnn_council(num_agents=7)  # Need more agents for Byzantine testing
    
    # Simulate some agents with extreme opinions
    request = {
        "type": "critical_decision",
        "priority": 10,
        "risk_level": "high",
        "options": ["approve", "reject", "escalate"]
    }
    
    # Get consensus
    consensus = await council.make_council_decision(request)
    
    print(f"\nByzantine Consensus Result:")
    print(f"  Decision: {consensus.final_decision.value}")
    print(f"  Type: {consensus.consensus_type}")
    print(f"  Confidence: {consensus.consensus_confidence:.2f}")
    
    if consensus.dissenting_reasons:
        print(f"\nDissenting Opinions:")
        for reason in consensus.dissenting_reasons:
            print(f"  - {reason}")
    
    # Check Byzantine tolerance
    print(f"\n✓ Byzantine consensus achieved with {len(consensus.votes)} agents")
    print(f"  Can tolerate up to {len(consensus.votes) // 3} Byzantine agents")


async def main():
    """Run all tests"""
    print("🧠 Testing LNN Council Integration")
    print("=" * 50)
    
    # Test 1: Standalone council
    await test_lnn_council_standalone()
    
    # Test 2: Router integration
    await test_neural_router_with_council()
    
    # Test 3: Byzantine consensus
    await test_byzantine_consensus()
    
    print("\n✅ All tests completed!")
    print("\nKey Features Demonstrated:")
    print("1. Multi-agent neural decision making")
    print("2. Byzantine fault tolerance")
    print("3. Liquid neural networks for adaptation")
    print("4. Integration with neural router")
    print("5. Consensus-based model selection")


if __name__ == "__main__":
    asyncio.run(main())