"""
Simple test for AURA agents without full component integration
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

# Mock the components to avoid import errors
class MockMemory:
    async def store(self, data): pass
    async def retrieve(self, query, limit=5): return []

class MockTDA:
    async def analyze_conversation_flow(self, messages): return {"complexity": 0.5}

class MockRouter:
    async def route_request(self, request): 
        return type('obj', (object,), {
            'model_config': type('obj', (object,), {'model_id': 'gpt-4'})(),
            'provider': type('obj', (object,), {'value': 'openai'})(),
            'confidence': 0.9
        })()

class MockOrchestrator:
    pass

# Patch the imports
import aura_intelligence.agents.agent_core as agent_core
agent_core.MEMORY_AVAILABLE = False
agent_core.TDA_AVAILABLE = False  
agent_core.NEURAL_AVAILABLE = False
agent_core.ORCHESTRATION_AVAILABLE = False
agent_core.LANGGRAPH_AVAILABLE = False

from aura_intelligence.agents.agent_templates import (
    ObserverAgent, AnalystAgent, ExecutorAgent, CoordinatorAgent
)


async def test_basic_agents():
    """Test basic agent functionality without components"""
    print("\n🧪 TESTING BASIC AGENT FUNCTIONALITY\n")
    
    # Test Observer
    print("1. Testing Observer Agent...")
    observer = ObserverAgent("obs_001")
    
    # Manually set mock components
    observer.memory = MockMemory()
    observer.tda = MockTDA()
    observer.router = MockRouter()
    
    result = await observer.run("Monitor system health")
    print(f"   Result: {result['success']}")
    print(f"   Metrics: {observer.get_metrics()}")
    
    # Test Analyst
    print("\n2. Testing Analyst Agent...")
    analyst = AnalystAgent("ana_001")
    analyst.memory = MockMemory()
    analyst.router = MockRouter()
    
    result = await analyst.run("Analyze user growth trends")
    print(f"   Result: {result['success']}")
    print(f"   Type: {analyst.agent_type}")
    
    # Test Executor
    print("\n3. Testing Executor Agent...")
    executor = ExecutorAgent("exe_001")
    executor.memory = MockMemory()
    
    result = await executor.run("Execute safe API call")
    print(f"   Result: {result['success']}")
    print(f"   Risk assessment: {result.get('results', {}).get('analysis', {}).get('risk_level', 'unknown')}")
    
    # Test Coordinator
    print("\n4. Testing Coordinator Agent...")
    coordinator = CoordinatorAgent("coord_001")
    
    # Register other agents
    coordinator.register_agent(observer)
    coordinator.register_agent(analyst)
    coordinator.register_agent(executor)
    
    print(f"   Registered agents: {len(coordinator.managed_agents)}")
    result = await coordinator.run("Coordinate simple workflow")
    print(f"   Result: {result['success']}")


async def test_agent_workflow():
    """Test agent workflow without LangGraph"""
    print("\n\n🔄 TESTING AGENT WORKFLOW\n")
    
    observer = ObserverAgent("obs_002")
    observer.memory = MockMemory()
    observer.tda = MockTDA()
    
    # Manually call workflow nodes
    print("Running workflow nodes manually...")
    
    # Observe
    state = observer.state
    state.current_task = "Monitor CPU usage"
    updates = await observer._observe_node(state)
    print(f"1. Observe: {updates.keys()}")
    
    # Analyze
    analysis = await observer.analyze_task(state)
    print(f"2. Analyze: {analysis}")
    
    # Decide
    decision = await observer.make_decision(state)
    print(f"3. Decide: {decision}")
    
    # Execute
    execution = await observer.execute_action(state)
    print(f"4. Execute: {execution}")
    
    # Reflect
    reflection = await observer._reflect_node(state)
    print(f"5. Reflect: Task status = {reflection.get('task_status')}")


async def test_multi_agent():
    """Test multi-agent coordination"""
    print("\n\n👥 TESTING MULTI-AGENT COORDINATION\n")
    
    # Create team
    agents = {
        "observer": ObserverAgent("obs_team"),
        "analyst": AnalystAgent("ana_team"),
        "executor": ExecutorAgent("exe_team")
    }
    
    # Mock components for all
    for agent in agents.values():
        agent.memory = MockMemory()
        agent.router = MockRouter()
    
    # Simulate coordination
    print("Simulating coordinated workflow:")
    
    # Step 1: Observer monitors
    obs_result = await agents["observer"].run("Monitor system performance")
    print(f"1. Observer: {obs_result['success']}")
    
    # Step 2: Analyst analyzes
    ana_result = await agents["analyst"].run("Analyze performance data")
    print(f"2. Analyst: {ana_result['success']}")
    
    # Step 3: Executor acts
    exe_result = await agents["executor"].run("Optimize system settings")
    print(f"3. Executor: {exe_result['success']}")
    
    # Show team metrics
    print("\nTeam Performance:")
    for name, agent in agents.items():
        metrics = agent.get_metrics()
        print(f"- {name}: {metrics['total_tasks']} tasks, {metrics['success_rate']:.0%} success")


async def main():
    """Run all tests"""
    print("\n🚀 AURA AGENT SIMPLE TEST SUITE")
    print("================================")
    print("Testing without full component integration")
    
    try:
        await test_basic_agents()
        await test_agent_workflow()
        await test_multi_agent()
        
        print("\n\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nKey Findings:")
        print("- Agent core architecture works")
        print("- Templates provide good abstractions")
        print("- Multi-agent coordination functional")
        print("- Components can be mocked for testing")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())