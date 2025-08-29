#!/usr/bin/env python3
"""
Comprehensive test for collective intelligence components
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧠 TESTING COLLECTIVE INTELLIGENCE SYSTEM")
print("=" * 60)

async def test_collective():
    """Test all collective intelligence components"""
    
    # Test 1: Import all modules
    print("\n📦 Testing imports...")
    try:
        from aura_intelligence.collective.context_engine import (
            ContextEngine, Evidence, EvidenceType, ContextScope
        )
        from aura_intelligence.collective.graph_builder import (
            CollectiveGraphBuilder, GraphType, NodeType,
            create_supervisor_graph, create_map_reduce_graph
        )
        from aura_intelligence.collective.memory_manager import (
            CollectiveMemoryManager, Memory, MemoryType, MemoryPriority
        )
        from aura_intelligence.collective.orchestrator import (
            CollectiveOrchestrator, OrchestratorMode, CollectiveInsight
        )
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Context Engine
    print("\n🎯 Testing Context Engine...")
    try:
        context_engine = ContextEngine()
        
        # Add evidence
        evidence1 = Evidence(
            type=EvidenceType.OBSERVATION,
            source="sensor_1",
            content={"temperature": 25.5, "humidity": 60},
            confidence=0.95
        )
        
        success = await context_engine.add_evidence(evidence1, ContextScope.LOCAL)
        print(f"✅ Evidence added: {success}")
        
        # Add causal evidence
        evidence2 = Evidence(
            type=EvidenceType.INFERENCE,
            source="analyzer_1",
            content={"status": "normal", "trend": "stable"},
            confidence=0.85,
            causal_links=[evidence1.id]
        )
        
        await context_engine.add_evidence(evidence2, ContextScope.LOCAL)
        
        # Get context
        context = await context_engine.get_context(ContextScope.LOCAL)
        print(f"✅ Context retrieved: {len(context['evidence'])} evidence items")
        print(f"   - Attention focus: {context['attention_focus']['focused']}")
        
        # Test reasoning
        reasoning = await context_engine.reason_about("temperature status")
        print(f"✅ Reasoning completed: {len(reasoning['hypotheses'])} hypotheses")
        print(f"   - Confidence: {reasoning['confidence']:.2%}")
        
        # Test context synchronization
        sync_result = await context_engine.synchronize_contexts(
            ContextScope.LOCAL,
            ContextScope.GLOBAL,
            sync_policy="merge"
        )
        print(f"✅ Context synchronized: {sync_result['merged_evidence']} items merged")
    except Exception as e:
        print(f"❌ Context engine test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Graph Builder
    print("\n📊 Testing Graph Builder...")
    try:
        # Create workflow graph
        builder = CollectiveGraphBuilder(GraphType.WORKFLOW)
        
        # Define node functions
        async def analyze(state):
            state["analysis"] = "Data analyzed"
            return state
        
        async def decide(state):
            state["decision"] = "proceed" if state.get("analysis") else "stop"
            return state
        
        async def execute(state):
            state["result"] = "Action executed"
            return state
        
        # Build graph
        builder.add_node("analyze", analyze, NodeType.AGENT)
        builder.add_node("decide", decide, NodeType.DECISION)
        builder.add_node("execute", execute, NodeType.AGENT)
        
        builder.set_entry_point("analyze")
        builder.add_edge("analyze", "decide")
        builder.add_edge("decide", "execute", 
                        condition=lambda s: s.get("decision") == "proceed")
        
        print("✅ Graph built with 3 nodes")
        
        # Compile and execute
        graph = builder.compile()
        result = await graph.ainvoke({"input": "test data"})
        print(f"✅ Graph executed: {result.get('result', 'No result')}")
        
        # Test supervisor pattern
        supervisor_graph = create_supervisor_graph(["agent1", "agent2", "agent3"])
        print("✅ Supervisor graph created")
        
        # Test map-reduce pattern
        def mapper(data):
            return sum(data)
        
        def reducer(results):
            return sum(results)
        
        mapreduce_graph = create_map_reduce_graph(3, mapper, reducer)
        print("✅ Map-reduce graph created")
        
        # Visualize
        viz_data = builder.visualize()
        print(f"✅ Graph visualization: {len(viz_data['nodes'])} nodes, {len(viz_data['edges'])} edges")
    except Exception as e:
        print(f"❌ Graph builder test failed: {e}")
    
    # Test 4: Memory Manager
    print("\n💾 Testing Memory Manager...")
    try:
        memory_manager = CollectiveMemoryManager()
        await memory_manager.start()
        
        # Store memories
        mem_id1 = await memory_manager.store(
            content={"event": "sensor_reading", "value": 25.5},
            agent_id="agent_1",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.HIGH
        )
        print(f"✅ Memory 1 stored: {mem_id1}")
        
        # Store linked memory
        mem_id2 = await memory_manager.store(
            content={"analysis": "temperature normal", "sensor_value": 25.5},
            agent_id="agent_2",
            memory_type=MemoryType.SEMANTIC,
            linked_to=[mem_id1]
        )
        print(f"✅ Memory 2 stored (linked to memory 1)")
        
        # Build consensus
        consensus_id = await memory_manager.build_consensus(
            content={"agreed_status": "system_normal", "confidence": 0.95},
            voting_agents={
                "agent_1": 1.0,
                "agent_2": 0.9,
                "agent_3": 0.8
            }
        )
        print(f"✅ Consensus memory created: {consensus_id}")
        
        # Retrieve memories
        memories = await memory_manager.retrieve(
            query={"event": "sensor_reading"},
            agent_id="agent_1"
        )
        print(f"✅ Retrieved {len(memories)} memories")
        
        # Share memory
        shared = await memory_manager.share_memory(
            mem_id1, "agent_1", ["agent_2", "agent_3"]
        )
        print(f"✅ Memory shared: {shared}")
        
        # Get causal chain
        if consensus_id:
            chain = await memory_manager.get_causal_chain(consensus_id)
            print(f"✅ Causal chain: {len(chain)} memories")
        
        await memory_manager.stop()
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
    
    # Test 5: Orchestrator
    print("\n🎭 Testing Orchestrator...")
    try:
        orchestrator = CollectiveOrchestrator(mode=OrchestratorMode.HYBRID)
        
        # Register agents
        orchestrator.register_agent(
            "analyzer_1",
            capabilities={"analysis", "pattern_recognition"},
            specializations=["anomaly_detection"]
        )
        print("✅ Agent 1 registered")
        
        orchestrator.register_agent(
            "predictor_1",
            capabilities={"prediction", "forecasting"},
            specializations=["time_series"]
        )
        print("✅ Agent 2 registered")
        
        orchestrator.register_agent(
            "optimizer_1",
            capabilities={"optimization", "planning"},
            specializations=["resource_allocation"]
        )
        print("✅ Agent 3 registered")
        
        # Initialize
        await orchestrator.initialize()
        print("✅ Orchestrator initialized")
        
        # Gather insights
        context = {
            "task_type": "analysis",
            "data": {"temperature": 25.5, "trend": "rising"},
            "urgency": 0.6,
            "complexity": 0.7
        }
        
        insight = await orchestrator.gather_insights(context)
        print(f"✅ Collective insight gathered:")
        print(f"   - Type: {insight.insight_type}")
        print(f"   - Confidence: {insight.confidence:.2%}")
        print(f"   - Contributing agents: {len(insight.contributing_agents)}")
        
        # Allocate task
        task = {
            "type": "anomaly_detection",
            "priority": 0.8,
            "data": context["data"]
        }
        
        assigned_agent = await orchestrator.allocate_task(task)
        print(f"✅ Task allocated to: {assigned_agent}")
        
        # Update performance
        await orchestrator.update_agent_performance(
            assigned_agent,
            "task_123",
            success=True,
            metrics={"quality": 0.95, "efficiency": 0.88}
        )
        print("✅ Agent performance updated")
        
        # Test different orchestration modes
        for mode in [OrchestratorMode.CENTRALIZED, OrchestratorMode.DISTRIBUTED, OrchestratorMode.SWARM]:
            orchestrator.mode = mode
            insight = await orchestrator.gather_insights(context)
            print(f"✅ {mode.value} mode insight: confidence={insight.confidence:.2%}")
        
        await orchestrator.shutdown()
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
    
    # Test 6: Integration Test
    print("\n🔗 Testing Component Integration...")
    try:
        # Create integrated system
        orchestrator = CollectiveOrchestrator()
        
        # Register multiple agents
        for i in range(5):
            orchestrator.register_agent(
                f"agent_{i}",
                capabilities={"analysis", "processing"},
                specializations=[f"task_type_{i%3}"]
            )
        
        await orchestrator.initialize()
        
        # Simulate complex scenario
        print("   Simulating multi-agent scenario...")
        
        # Phase 1: Data collection
        for i in range(3):
            evidence = Evidence(
                type=EvidenceType.OBSERVATION,
                source=f"agent_{i}",
                content={"sensor": i, "value": 20 + i * 5},
                confidence=0.9 - i * 0.1
            )
            await orchestrator.context_engine.add_evidence(evidence)
        
        # Phase 2: Analysis
        context = {
            "task_type": "analysis",
            "urgency": 0.7,
            "complexity": 0.8
        }
        
        insight = await orchestrator.gather_insights(context)
        
        # Phase 3: Memory consolidation
        await orchestrator.memory_manager.consolidate_memories()
        
        print("✅ Integration test completed successfully")
        
        # Generate final report
        context_state = await orchestrator.context_engine.get_context(ContextScope.GLOBAL)
        
        print(f"\n📋 FINAL SYSTEM STATE:")
        print(f"   - Evidence items: {context_state['evidence_count']}")
        print(f"   - Active hypotheses: {len(context_state['hypotheses'])}")
        print(f"   - Causal chains: {len(context_state['causal_chains'])}")
        print(f"   - Registered agents: {len(orchestrator.agents)}")
        print(f"   - Active tasks: {len(orchestrator.active_tasks)}")
        
        await orchestrator.shutdown()
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    print("\n" + "=" * 60)
    print("COLLECTIVE INTELLIGENCE TEST COMPLETE")
    
# Run the test
if __name__ == "__main__":
    asyncio.run(test_collective())