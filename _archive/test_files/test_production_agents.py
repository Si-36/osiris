#!/usr/bin/env python3
"""
Test Production LangGraph Agents Integration

This tests the new production agents with latest 2025 patterns.
"""

import asyncio
import sys
import os

# Add AURA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.agents import (
    create_production_agent,
    PRODUCTION_AGENTS_AVAILABLE,
    ProductionAgentConfig
)


async def test_production_agent():
    """Test the production agent implementation"""
    print("=== Testing Production LangGraph Agent ===\n")
    
    if not PRODUCTION_AGENTS_AVAILABLE:
        print("❌ Production agents not available (missing dependencies)")
        return False
        
    try:
        # Create a production agent
        agent = create_production_agent(
            name="test-agent-001",
            role="general",
            temperature=0.7,
            enable_memory=False,  # Disable for simple test
            enable_tools=True
        )
        
        print(f"✅ Created production agent: {agent.config.name}")
        print(f"   Role: {agent.config.role}")
        print(f"   Model: {agent.config.model_name}")
        print(f"   Tools enabled: {agent.config.enable_tools}")
        
        # Test simple execution
        print("\n📝 Testing agent execution...")
        result = await agent.execute(
            task="What are the key features of the AURA Intelligence System?",
            context={"source": "test"}
        )
        
        if result["success"]:
            print(f"✅ Agent execution successful!")
            print(f"   Thread ID: {result['thread_id']}")
            print(f"   Execution time: {result['execution_time']:.2f}s")
            print(f"   Response preview: {str(result['result'])[:200]}...")
            
            # Show metrics
            metrics = agent.get_metrics()
            print(f"\n📊 Agent Metrics:")
            print(f"   Total executions: {metrics['total_executions']}")
            print(f"   Success rate: {metrics['successful_executions'] / max(1, metrics['total_executions']) * 100:.1f}%")
            print(f"   Average execution time: {metrics['average_execution_time']:.2f}s")
        else:
            print(f"❌ Agent execution failed: {result.get('error', 'Unknown error')}")
            return False
            
        # Test with tools
        print("\n🔧 Testing with tools...")
        result2 = await agent.execute(
            task="Analyze this data and provide insights: [1, 2, 3, 4, 5]",
            context={"analysis_required": True}
        )
        
        if result2["success"]:
            print("✅ Tool execution successful!")
        else:
            print(f"⚠️  Tool execution failed: {result2.get('error', 'Unknown error')}")
            
        # Cleanup
        await agent.cleanup()
        print("\n✅ Agent cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing production agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration_layer():
    """Test the ultimate integration layer"""
    print("\n=== Testing LangGraph Integration Layer ===\n")
    
    try:
        from aura_intelligence.integrations import UltimateLangGraphIntegration
        
        # Create integration with minimal config
        integration = UltimateLangGraphIntegration(
            config=type('Config', (), {
                'name': 'test-integration',
                'checkpoint_dir': '/tmp/aura-checkpoints'
            })()
        )
        
        print(f"✅ Created integration layer")
        print(f"   Fallback mode: {integration.fallback_mode}")
        
        # Initialize
        await integration.initialize()
        print("✅ Integration initialized")
        
        # Check registered agents
        print(f"\n📋 Registered agents: {len(integration.agent_registry.agents)}")
        for agent_id, agent_info in integration.agent_registry.agents.items():
            print(f"   - {agent_id}: {agent_info['capabilities']}")
            
        # Test workflow execution
        if not integration.fallback_mode:
            print("\n🔄 Testing workflow execution...")
            workflow_result = await integration.execute_advanced_workflows(
                workflow_type="analysis",
                task="Analyze AURA system architecture",
                config={}
            )
            
            if workflow_result.get("success"):
                print("✅ Workflow execution successful!")
            else:
                print(f"⚠️  Workflow execution failed: {workflow_result.get('error')}")
        else:
            print("\n⚠️  In fallback mode - skipping workflow test")
            
        # Check health
        health = await integration.get_health_status()
        print(f"\n🏥 System Health:")
        print(f"   Status: {health['status']}")
        print(f"   Initialized: {health['initialized']}")
        print(f"   Components: {len(health['components'])}")
        
        # Cleanup
        await integration.cleanup()
        print("\n✅ Integration cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing integration layer: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🚀 AURA Production Agent Tests\n")
    
    # Test production agent
    agent_success = await test_production_agent()
    
    # Test integration layer
    integration_success = await test_integration_layer()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print(f"   Production Agent: {'✅ PASSED' if agent_success else '❌ FAILED'}")
    print(f"   Integration Layer: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"   Overall: {'✅ ALL TESTS PASSED' if agent_success and integration_success else '❌ SOME TESTS FAILED'}")
    

if __name__ == "__main__":
    asyncio.run(main())