"""
Test the AURA Agent System with agent_core.py and templates
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.agents.agent_templates import (
    ObserverAgent, AnalystAgent, ExecutorAgent, CoordinatorAgent,
    create_agent
)


async def test_individual_agents():
    """Test each agent type individually"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL AGENTS")
    print("="*60)
    
    # Test Observer Agent
    print("\n1. Testing Observer Agent...")
    observer = create_agent("observer", "obs_001")
    result = await observer.run(
        task="Monitor system health and detect anomalies",
        context={"environment": "production"}
    )
    print(f"Observer result: {result['success']}")
    print(f"Metrics: {observer.get_metrics()}")
    
    # Test Analyst Agent
    print("\n2. Testing Analyst Agent...")
    analyst = create_agent("analyst", "ana_001")
    result = await analyst.run(
        task="Analyze trends in user growth over the past week",
        context={"data_source": "user_metrics"}
    )
    print(f"Analyst result: {result['success']}")
    print(f"Insights: {result.get('results', {}).get('tool_output', 'No insights')}")
    
    # Test Executor Agent
    print("\n3. Testing Executor Agent...")
    executor = create_agent("executor", "exe_001")
    result = await executor.run(
        task="Execute API call to update user preferences",
        context={"endpoint": "/api/v1/users/preferences", "method": "POST"}
    )
    print(f"Executor result: {result['success']}")
    print(f"Action executed: {result.get('results', {}).get('execution', {})}")
    
    # Test Coordinator Agent
    print("\n4. Testing Coordinator Agent...")
    coordinator = create_agent("coordinator", "coord_001")
    result = await coordinator.run(
        task="Coordinate multi-agent workflow for system optimization",
        context={"priority": "high"}
    )
    print(f"Coordinator result: {result['success']}")
    print(f"Coordination: {result.get('results', {})}")


async def test_multi_agent_coordination():
    """Test agents working together"""
    print("\n" + "="*60)
    print("TESTING MULTI-AGENT COORDINATION")
    print("="*60)
    
    # Create team of agents
    observer = ObserverAgent("obs_team_001")
    analyst = AnalystAgent("ana_team_001") 
    executor = ExecutorAgent("exe_team_001")
    coordinator = CoordinatorAgent("coord_team_001")
    
    # Register agents with coordinator
    coordinator.register_agent(observer)
    coordinator.register_agent(analyst)
    coordinator.register_agent(executor)
    
    # Complex coordinated task
    print("\nRunning coordinated workflow...")
    result = await coordinator.run(
        task="Monitor system, analyze performance issues, and execute optimizations",
        context={
            "workflow_type": "performance_optimization",
            "scope": "full_system",
            "auto_execute": True
        }
    )
    
    print(f"\nCoordinated workflow result: {result['success']}")
    print(f"Total subtasks: {len(result.get('results', {}).get('coordination', {}).get('subtasks', []))}")
    
    # Show all agent metrics
    print("\nAgent Team Metrics:")
    for agent in [observer, analyst, executor, coordinator]:
        metrics = agent.get_metrics()
        print(f"- {agent.agent_type.capitalize()}: {metrics['total_tasks']} tasks, "
              f"{metrics['success_rate']:.0%} success rate")


async def test_component_integration():
    """Test integration with AURA components"""
    print("\n" + "="*60)
    print("TESTING COMPONENT INTEGRATION")
    print("="*60)
    
    # Create agent with full config
    config = {
        "neural_router": {
            "enable_cache": True,
            "enable_lnn_council": True,
            "providers": {
                "openai": {"api_key": "mock_key"},
                "anthropic": {"api_key": "mock_key"}
            }
        },
        "checkpoint": {
            "type": "memory"  # Use memory for testing
        }
    }
    
    analyst = create_agent("analyst", "ana_integrated_001", config)
    
    # Test with context that triggers component usage
    result = await analyst.run(
        task="Analyze complex data patterns requiring advanced model routing",
        context={
            "complexity": "high",
            "require_memory": True,
            "require_topology_analysis": True
        }
    )
    
    print(f"\nIntegrated analysis result: {result['success']}")
    print(f"Components used:")
    components = analyst.get_metrics()['components_available']
    for comp, available in components.items():
        print(f"- {comp}: {'✓' if available else '✗'}")


async def test_error_handling():
    """Test error handling and resilience"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    executor = create_agent("executor", "exe_error_001")
    
    # Test with risky task
    result = await executor.run(
        task="Delete critical production data",  # Should trigger high-risk warning
        context={"environment": "production"}
    )
    
    print(f"\nHigh-risk task result: {result}")
    print(f"Risk assessment worked: {'escalate' in str(result.get('results', {}))}")


async def test_performance():
    """Test performance with multiple concurrent agents"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE")
    print("="*60)
    
    # Create multiple agents
    agents = [
        create_agent("observer", f"obs_perf_{i}")
        for i in range(5)
    ]
    
    # Run concurrent tasks
    start_time = asyncio.get_event_loop().time()
    
    tasks = [
        agent.run(f"Monitor subsystem {i}", {"subsystem_id": i})
        for i, agent in enumerate(agents)
    ]
    
    results = await asyncio.gather(*tasks)
    
    duration = asyncio.get_event_loop().time() - start_time
    
    print(f"\nConcurrent execution of {len(agents)} agents:")
    print(f"- Total time: {duration:.2f}s")
    print(f"- Average time per agent: {duration/len(agents):.2f}s")
    print(f"- All successful: {all(r['success'] for r in results)}")


async def main():
    """Run all tests"""
    print("\n🚀 AURA AGENT SYSTEM TEST SUITE")
    
    try:
        await test_individual_agents()
        await test_multi_agent_coordination()
        await test_component_integration()
        await test_error_handling()
        await test_performance()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())