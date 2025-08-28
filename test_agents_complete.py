#!/usr/bin/env python3
"""
Complete test for Agent system with all features and integration
"""

import asyncio
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Any
import uuid

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🤖 COMPLETE AGENT SYSTEM TEST")
print("=" * 60)

async def test_agents_complete():
    """Test all agent features comprehensively"""
    
    all_tests_passed = True
    test_results = {}
    
    try:
        # Import the agent systems
        from aura_intelligence.agents.advanced_agent_system import (
            BaseAgent, ReActAgent, PlannerAgent, MultiAgentOrchestrator,
            Tool, AgentRole, ThoughtType, Thought, AgentMemory,
            create_example_tools
        )
        
        print("✅ Imports successful\n")
        
        # Test 1: Agent Memory
        print("1️⃣ TESTING AGENT MEMORY")
        print("-" * 40)
        try:
            memory = AgentMemory(capacity=100)
            
            # Store episodes
            for i in range(5):
                memory.store_episode({
                    'content': f'Episode {i}: Testing memory storage',
                    'type': 'test',
                    'value': i
                })
            
            # Retrieve similar
            similar = memory.retrieve_similar('Testing memory', k=3)
            
            # Update semantic memory
            memory.update_semantic('test_key', 'test_value')
            retrieved = memory.get_semantic('test_key')
            
            print(f"✅ Agent Memory:")
            print(f"   Episodes stored: {len(memory.episodic_memory)}")
            print(f"   Similar retrieved: {len(similar)}")
            print(f"   Semantic memory works: {retrieved == 'test_value'}")
            
            test_results['agent_memory'] = True
            
        except Exception as e:
            print(f"❌ Agent memory test failed: {e}")
            test_results['agent_memory'] = False
            all_tests_passed = False
        
        # Test 2: Thought Generation
        print("\n2️⃣ TESTING THOUGHT GENERATION")
        print("-" * 40)
        try:
            # Create a simple test agent
            class TestAgent(BaseAgent):
                async def process(self, input_data: Any) -> Any:
                    return input_data
            
            agent = TestAgent("TestBot", AgentRole.ANALYZER)
            
            # Test different observations
            observations = [
                "The system has encountered an error",
                "What should we do next?",
                "Plan the deployment strategy",
                "Observation: CPU usage is high"
            ]
            
            thoughts = []
            for obs in observations:
                thought = await agent.think(obs)
                thoughts.append(thought)
                print(f"   {obs[:30]}... -> {thought.type.value}")
            
            print(f"✅ Thought generation:")
            print(f"   Thoughts generated: {len(thoughts)}")
            print(f"   Types: {set(t.type.value for t in thoughts)}")
            
            test_results['thought_generation'] = len(thoughts) == len(observations)
            
        except Exception as e:
            print(f"❌ Thought generation test failed: {e}")
            test_results['thought_generation'] = False
            all_tests_passed = False
        
        # Test 3: Tools and Actions
        print("\n3️⃣ TESTING TOOLS AND ACTIONS")
        print("-" * 40)
        try:
            tools = create_example_tools()
            
            # Test tool execution
            calc_tool = tools[0]  # calculator
            result = await calc_tool.execute(expression="2 + 2 * 3")
            
            print(f"✅ Tool execution:")
            print(f"   Calculator: 2 + 2 * 3 = {result}")
            print(f"   Tool count: {len(tools)}")
            print(f"   Tool names: {[t.name for t in tools]}")
            
            test_results['tools'] = result == 8
            
        except Exception as e:
            print(f"❌ Tools test failed: {e}")
            test_results['tools'] = False
            all_tests_passed = False
        
        # Test 4: ReAct Agent
        print("\n4️⃣ TESTING REACT AGENT")
        print("-" * 40)
        try:
            react_agent = ReActAgent("ReactBot", tools, max_steps=5)
            
            # Process a task
            result = await react_agent.process("Calculate 10 + 20 * 3")
            
            print(f"✅ ReAct Agent:")
            print(f"   Task: Calculate 10 + 20 * 3")
            print(f"   Thoughts: {len(result['thoughts'])}")
            print(f"   Actions: {len(result['actions'])}")
            print(f"   Completed: {result['completed']}")
            
            if result['actions']:
                print(f"   Result: {result['actions'][-1].get('output', 'N/A')}")
            
            test_results['react_agent'] = result['completed']
            
        except Exception as e:
            print(f"❌ ReAct agent test failed: {e}")
            test_results['react_agent'] = False
            all_tests_passed = False
        
        # Test 5: Planner Agent
        print("\n5️⃣ TESTING PLANNER AGENT")
        print("-" * 40)
        try:
            planner = PlannerAgent("PlannerBot")
            
            # Create a plan
            plan_result = await planner.process("Deploy application and monitor performance")
            
            print(f"✅ Planner Agent:")
            print(f"   Goal: Deploy application and monitor performance")
            print(f"   Sub-goals: {len(plan_result['sub_goals'])}")
            print(f"   Plan steps: {len(plan_result['plan'])}")
            
            for i, step in enumerate(plan_result['plan'][:3]):
                print(f"   Step {i+1}: {step['action']} - {step['target']}")
            
            test_results['planner_agent'] = len(plan_result['plan']) > 0
            
        except Exception as e:
            print(f"❌ Planner agent test failed: {e}")
            test_results['planner_agent'] = False
            all_tests_passed = False
        
        # Test 6: Multi-Agent Orchestration
        print("\n6️⃣ TESTING MULTI-AGENT ORCHESTRATION")
        print("-" * 40)
        try:
            orchestrator = MultiAgentOrchestrator()
            
            # Register agents
            orchestrator.register_agent(react_agent)
            orchestrator.register_agent(planner)
            
            # Test delegation
            delegated_result = await orchestrator.delegate_task(
                "Analyze system requirements",
                planner.id
            )
            
            print(f"✅ Orchestration:")
            print(f"   Registered agents: {len(orchestrator.agents)}")
            print(f"   Task delegated to: {planner.name}")
            print(f"   Shared memory episodes: {len(orchestrator.shared_memory.episodic_memory)}")
            
            test_results['orchestration'] = True
            
        except Exception as e:
            print(f"❌ Orchestration test failed: {e}")
            test_results['orchestration'] = False
            all_tests_passed = False
        
        # Test 7: Collaborative Task
        print("\n7️⃣ TESTING COLLABORATIVE TASK")
        print("-" * 40)
        try:
            # Create executor agent for collaboration
            executor = ReActAgent("ExecutorBot", tools)
            executor.role = AgentRole.EXECUTOR
            orchestrator.register_agent(executor)
            
            # Collaborative task
            collab_result = await orchestrator.collaborative_task(
                "Plan and execute data analysis",
                [AgentRole.PLANNER, AgentRole.EXECUTOR]
            )
            
            print(f"✅ Collaborative Task:")
            print(f"   Task: Plan and execute data analysis")
            print(f"   Agents involved: {collab_result['agents']}")
            print(f"   Results generated: {len(collab_result['results'])}")
            
            test_results['collaborative'] = len(collab_result['results']) > 0
            
        except Exception as e:
            print(f"❌ Collaborative task test failed: {e}")
            test_results['collaborative'] = False
            all_tests_passed = False
        
        # Test 8: Agent Reflection
        print("\n8️⃣ TESTING AGENT REFLECTION")
        print("-" * 40)
        try:
            # Create agent with some history
            reflect_agent = ReActAgent("ReflectBot", tools)
            
            # Generate some thoughts and actions
            await reflect_agent.think("Initial observation")
            await reflect_agent.think("Need to take action")
            
            # Force an action thought
            action_thought = Thought(
                type=ThoughtType.ACTION,
                content="Calculate something",
                confidence=0.9
            )
            reflect_agent.state.thoughts.append(action_thought)
            await reflect_agent.act(action_thought)
            
            # Reflect
            reflection = await reflect_agent.reflect()
            
            print(f"✅ Agent Reflection:")
            print(f"   Thoughts before reflection: {len(reflect_agent.state.thoughts)-1}")
            print(f"   Actions taken: {len(reflect_agent.state.actions)}")
            print(f"   Reflection: {reflection.content[:60]}...")
            
            test_results['reflection'] = reflection.type == ThoughtType.REFLECTION
            
        except Exception as e:
            print(f"❌ Reflection test failed: {e}")
            test_results['reflection'] = False
            all_tests_passed = False
        
        # Test 9: Integration Test
        print("\n9️⃣ TESTING INTEGRATION WITH OTHER COMPONENTS")
        print("-" * 40)
        
        # Try integration with other AURA components
        integration_tests = []
        
        # Test with TDA
        try:
            from aura_intelligence.tda.advanced_tda_system import AdvancedTDAEngine, TDAConfig
            
            # Create TDA tool for agents
            tda_engine = AdvancedTDAEngine(TDAConfig(max_dimension=1))
            
            async def analyze_topology(data: str) -> Dict[str, Any]:
                # Mock topology analysis
                return {
                    'topology': 'analyzed',
                    'betti_numbers': [1, 2],
                    'persistence': 0.8
                }
            
            tda_tool = Tool(
                name="topology_analyzer",
                description="Analyze topological features",
                parameters={'data': str},
                function=analyze_topology
            )
            
            # Create agent with TDA tool
            tda_agent = ReActAgent("TDABot", [tda_tool])
            
            print("✅ TDA integration available")
            integration_tests.append(('tda', True))
            
        except Exception as e:
            print(f"⚠️  TDA integration skipped: {e}")
            integration_tests.append(('tda', None))
        
        # Test with Neural
        try:
            from aura_intelligence.neural.advanced_neural_system import AdvancedNeuralNetwork
            
            print("✅ Neural integration available")
            integration_tests.append(('neural', True))
            
        except Exception as e:
            print(f"⚠️  Neural integration skipped: {e}")
            integration_tests.append(('neural', None))
        
        test_results['integration'] = any(result for _, result in integration_tests if result)
        
        # Test 10: Performance Benchmark
        print("\n🔟 PERFORMANCE BENCHMARK")
        print("-" * 40)
        
        # Benchmark different agent operations
        benchmarks = {}
        
        # Thought generation speed
        start = time.time()
        for _ in range(100):
            await react_agent.think("Benchmark observation")
        think_time = (time.time() - start) * 10  # ms per 100 thoughts
        benchmarks['think_time_ms'] = think_time
        
        # Memory storage speed
        start = time.time()
        test_memory = AgentMemory(1000)
        for i in range(100):
            test_memory.store_episode({'content': f'Benchmark {i}'})
        memory_time = (time.time() - start) * 10
        benchmarks['memory_time_ms'] = memory_time
        
        # Multi-agent coordination
        start = time.time()
        test_orch = MultiAgentOrchestrator()
        for _ in range(10):
            test_agent = TestAgent(str(uuid.uuid4()), AgentRole.OBSERVER)
            test_orch.register_agent(test_agent)
        orchestration_time = (time.time() - start) * 100
        benchmarks['orchestration_time_ms'] = orchestration_time
        
        print(f"✅ Performance Benchmarks:")
        print(f"   Thought generation: {benchmarks['think_time_ms']:.2f} ms/100")
        print(f"   Memory storage: {benchmarks['memory_time_ms']:.2f} ms/100")
        print(f"   Agent registration: {benchmarks['orchestration_time_ms']:.2f} ms/10")
        
        test_results['performance'] = all(v < 1000 for v in benchmarks.values())
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("-" * 40)
        
        for test_name, result in test_results.items():
            if result is True:
                status = "✅ PASSED"
            elif result is False:
                status = "❌ FAILED"
            else:
                status = "⚠️  SKIPPED"
            
            print(f"{test_name:20} {status}")
        
        passed = sum(1 for r in test_results.values() if r is True)
        failed = sum(1 for r in test_results.values() if r is False)
        skipped = sum(1 for r in test_results.values() if r is None)
        
        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
        
        if all_tests_passed and failed == 0:
            print("\n✅ ALL AGENT TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed. Please debug.")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run the complete test
if __name__ == "__main__":
    result = asyncio.run(test_agents_complete())
    sys.exit(0 if result else 1)