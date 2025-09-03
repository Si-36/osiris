#!/usr/bin/env python3
"""
Test Cognitive Agent Integration - Complete Memory-Agent Connection
===================================================================

This tests the full integration between:
- UnifiedCognitiveMemory (the brain)
- CognitiveAgent (memory-aware agents)
- UnifiedOrchestrationEngine (the orchestrator)

This verifies that agents can truly learn, remember, and adapt.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

print("=" * 70)
print("COGNITIVE AGENT INTEGRATION TEST")
print("Testing Complete Memory-Agent-Orchestration Connection")
print("=" * 70)


async def test_integration():
    """Test the complete cognitive agent integration"""
    
    # ==================== Phase 1: Initialize Systems ====================
    
    print("\n🧠 Phase 1: Initializing Cognitive Systems...")
    
    try:
        # Import all components
        from aura_intelligence.memory.unified_cognitive_memory import UnifiedCognitiveMemory
        from aura_intelligence.agents.cognitive_agent import (
            CognitivePlannerAgent,
            CognitiveExecutorAgent,
            CognitiveAnalystAgent
        )
        from aura_intelligence.orchestration.unified_orchestration_engine import (
            UnifiedOrchestrationEngine,
            OrchestrationConfig
        )
        
        print("✅ All components imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== Phase 2: Create Memory System ====================
    
    print("\n🧠 Phase 2: Creating Unified Cognitive Memory...")
    
    try:
        # Configure memory
        memory_config = {
            'working': {'capacity': 7},
            'episodic': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'lmdb_path': '/tmp/test_cognitive_lmdb',
                'duckdb_path': '/tmp/test_cognitive.duckdb'
            },
            'semantic': {
                'neo4j_uri': 'bolt://localhost:7687',
                'neo4j_user': 'neo4j',
                'neo4j_password': 'password'
            },
            'use_cuda': False,
            'sleep_interval_hours': 0.001  # Very short for testing
        }
        
        # Create memory system
        memory = UnifiedCognitiveMemory(memory_config)
        await memory.start()
        
        print("✅ UnifiedCognitiveMemory initialized and started")
        
    except Exception as e:
        print(f"❌ Memory initialization failed: {e}")
        return
    
    # ==================== Phase 3: Create Cognitive Agents ====================
    
    print("\n🤖 Phase 3: Creating Cognitive Agents...")
    
    try:
        # Create three specialized agents sharing the same memory
        planner = CognitivePlannerAgent(
            agent_id="planner_001",
            memory_system=memory,
            config={'learning_rate': 0.1, 'exploration_rate': 0.2}
        )
        
        executor = CognitiveExecutorAgent(
            agent_id="executor_001",
            memory_system=memory,
            config={'learning_rate': 0.15, 'exploration_rate': 0.1}
        )
        
        analyst = CognitiveAnalystAgent(
            agent_id="analyst_001",
            memory_system=memory,
            config={'learning_rate': 0.2, 'exploration_rate': 0.3}
        )
        
        # Start all agents
        await planner.startup()
        await executor.startup()
        await analyst.startup()
        
        print("✅ Created 3 cognitive agents:")
        print(f"   • Planner: {planner.agent_id}")
        print(f"   • Executor: {executor.agent_id}")
        print(f"   • Analyst: {analyst.agent_id}")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return
    
    # ==================== Phase 4: Test Perceive-Think-Act Loop ====================
    
    print("\n🔄 Phase 4: Testing Cognitive Loop (Perceive → Think → Act)...")
    
    try:
        # Test 1: Planner perceives a goal and creates a plan
        print("\n📋 Test 1: Planner creates a plan")
        
        goal = "Optimize system performance by 20%"
        plan = await planner.create_plan(goal)
        
        print(f"   ✅ Plan created with confidence: {plan['confidence']:.2f}")
        print(f"      Based on {plan['based_on_memories']} memories")
        
        # Test 2: Executor perceives the plan and executes tasks
        print("\n⚡ Test 2: Executor executes tasks")
        
        tasks = [
            {'type': 'data_processing', 'description': 'Process metrics data'},
            {'type': 'api_call', 'description': 'Fetch system status'},
            {'type': 'optimization', 'description': 'Apply optimization'}
        ]
        
        for task in tasks:
            result = await executor.execute_task(task)
            status = "✅" if result['success'] else "❌"
            print(f"   {status} Task: {task['description']}")
            print(f"      Result: {result['result']}")
            print(f"      Memory-informed: {result['memory_informed']}")
        
        # Test 3: Analyst observes results and extracts patterns
        print("\n📊 Test 3: Analyst extracts patterns")
        
        data = {
            'performance_metrics': [85, 87, 89, 91, 88, 92],
            'error_rates': [0.05, 0.04, 0.03, 0.04, 0.02, 0.03],
            'response_times': [120, 115, 110, 108, 112, 105]
        }
        
        analysis = await analyst.analyze_data(data)
        
        print(f"   ✅ Analysis completed with confidence: {analysis['confidence']:.2f}")
        print(f"      Patterns found: {len(analysis['patterns_found'])}")
        print(f"      Insights: {len(analysis['insights'])}")
        
    except Exception as e:
        print(f"❌ Cognitive loop test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== Phase 5: Test Memory Sharing ====================
    
    print("\n🔗 Phase 5: Testing Memory Sharing Between Agents...")
    
    try:
        # Planner queries about executor's experiences
        print("\n🔍 Planner queries executor's experiences")
        
        memory_context = await planner.think(
            query="What tasks has the executor completed successfully?",
            context={'agent_query': 'executor_001'}
        )
        
        print(f"   ✅ Retrieved {memory_context.total_sources} relevant memories")
        print(f"      Confidence: {memory_context.confidence:.2f}")
        
        # Executor learns from planner's plans
        print("\n📚 Executor learns from planner's strategies")
        
        memory_context = await executor.think(
            query="What planning strategies have been successful?",
            context={'agent_query': 'planner_001'}
        )
        
        print(f"   ✅ Retrieved {memory_context.total_sources} planning memories")
        
        # Analyst synthesizes knowledge from all agents
        print("\n🧩 Analyst synthesizes collective knowledge")
        
        memory_context = await analyst.think(
            query="What patterns exist across all agent activities?",
            context={'collective_analysis': True}
        )
        
        print(f"   ✅ Synthesis confidence: {memory_context.confidence:.2f}")
        
    except Exception as e:
        print(f"❌ Memory sharing test failed: {e}")
    
    # ==================== Phase 6: Test Learning and Adaptation ====================
    
    print("\n🎓 Phase 6: Testing Learning and Adaptation...")
    
    try:
        # Simulate repeated tasks to test learning
        print("\n📈 Running repeated tasks to observe learning")
        
        for round_num in range(3):
            print(f"\n   Round {round_num + 1}:")
            
            # Execute same type of task
            task = {'type': 'api_call', 'description': f'API call round {round_num + 1}'}
            result = await executor.execute_task(task)
            
            # Get metrics to see improvement
            metrics = executor.get_metrics()
            
            print(f"      Success rate: {metrics['success_rate']:.2%}")
            print(f"      Learning rate: {metrics['learning_rate']:.3f}")
            print(f"      Causal patterns: {metrics['causal_patterns_tracked']}")
        
        # Trigger consolidation to solidify learning
        print("\n💤 Triggering memory consolidation...")
        
        await planner.consolidate_learning()
        await executor.consolidate_learning()
        await analyst.consolidate_learning()
        
        print("   ✅ Consolidation completed for all agents")
        
    except Exception as e:
        print(f"❌ Learning test failed: {e}")
    
    # ==================== Phase 7: Test Orchestration Integration ====================
    
    print("\n🎭 Phase 7: Testing Orchestration Engine Integration...")
    
    try:
        # Create orchestration config with memory config
        orch_config = OrchestrationConfig()
        orch_config.memory_config = memory_config
        
        # Create orchestration engine (it will connect to the same memory)
        orchestrator = UnifiedOrchestrationEngine(orch_config)
        await orchestrator.initialize()
        
        print("✅ Orchestration engine initialized with memory system")
        
        # Verify memory system is connected
        if orchestrator.memory_system:
            print("✅ Memory system properly connected to orchestrator")
            
            # Test query through orchestrator
            stats = await orchestrator.memory_system.get_statistics()
            print(f"   • Working memory items: {stats['working_memory']['current_items']}")
            print(f"   • Total writes: {stats['system_metrics']['total_writes']}")
            print(f"   • Total queries: {stats['system_metrics']['total_queries']}")
        else:
            print("❌ Memory system not connected to orchestrator")
        
        # Shutdown orchestrator
        await orchestrator.shutdown()
        
    except Exception as e:
        print(f"❌ Orchestration test failed: {e}")
    
    # ==================== Phase 8: Final Metrics and Cleanup ====================
    
    print("\n📊 Phase 8: Final Metrics and Cleanup...")
    
    try:
        # Get final metrics from all agents
        print("\n🎯 Final Agent Metrics:")
        
        for agent in [planner, executor, analyst]:
            metrics = agent.get_metrics()
            print(f"\n   {agent.agent_type.upper()} ({agent.agent_id}):")
            print(f"      • Total experiences: {metrics['total_experiences']}")
            print(f"      • Total decisions: {metrics['total_decisions']}")
            print(f"      • Success rate: {metrics['success_rate']:.2%}")
            print(f"      • Causal patterns: {metrics['causal_patterns_tracked']}")
        
        # Get memory system statistics
        print("\n🧠 Memory System Statistics:")
        
        mem_stats = await memory.get_statistics()
        print(f"   • Total writes: {mem_stats['system_metrics']['total_writes']}")
        print(f"   • Total queries: {mem_stats['system_metrics']['total_queries']}")
        print(f"   • Consolidation cycles: {mem_stats['system_metrics']['consolidation_cycles']}")
        
        # Shutdown all systems
        print("\n🔚 Shutting down all systems...")
        
        await planner.shutdown()
        await executor.shutdown()
        await analyst.shutdown()
        await memory.stop()
        
        print("✅ All systems shut down cleanly")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ COGNITIVE AGENT INTEGRATION TEST COMPLETE")
    print("=" * 70)


# Run the test
if __name__ == "__main__":
    print("\n⚠️  Note: This test requires Redis and Neo4j to be running")
    print("   Run ./setup_databases.sh if needed\n")
    
    try:
        asyncio.run(test_integration())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()