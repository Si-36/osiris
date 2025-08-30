"""
Simple Connection Test
=====================

Quick test to verify component connections are working.
"""

import asyncio
from aura_system import create_aura_system


async def test_connections():
    print("\n🧪 TESTING COMPONENT CONNECTIONS\n")
    
    # Initialize system
    print("Initializing AURA system...")
    system = await create_aura_system()
    
    # Test 1: Can we create and retrieve a workflow?
    print("\n1️⃣ Test: Orchestration → Memory")
    
    if system.orchestration and system.memory:
        from aura_intelligence.orchestration.unified_orchestration_engine import WorkflowDefinition
        
        workflow = WorkflowDefinition(
            workflow_id="test-123",
            name="Test",
            version="1.0",
            graph_definition={"nodes": ["a", "b"], "edges": [{"source": "a", "target": "b"}]}
        )
        
        # Create workflow
        wf_id = await system.orchestration.create_workflow(workflow)
        print(f"   Created workflow: {wf_id}")
        
        # Query memory
        await asyncio.sleep(0.1)  # Let async storage complete
        
        from aura_intelligence.memory.core.memory_api import MemoryQuery, RetrievalMode
        query = MemoryQuery(mode=RetrievalMode.SEMANTIC_SEARCH, query_text="test-123", k=5)
        results = await system.memory.retrieve(query)
        
        if results.memories:
            print(f"   ✅ Found in memory! ({len(results.memories)} records)")
        else:
            print(f"   ❌ NOT found in memory")
    
    # Test 2: TDA analysis stored?
    print("\n2️⃣ Test: TDA → Memory")
    
    if system.tda and system.memory:
        workflow_data = {
            "workflow_id": "tda-test",
            "agents": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "dependencies": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"}
            ]
        }
        
        # Analyze
        analysis = await system.tda.analyze_workflow("tda-test", workflow_data)
        print(f"   Analyzed: bottleneck_score={analysis.bottleneck_score:.3f}")
        
        # Check memory
        await asyncio.sleep(0.1)
        query = MemoryQuery(mode=RetrievalMode.SEMANTIC_SEARCH, query_text="tda-test", k=5)
        results = await system.memory.retrieve(query)
        
        if results.memories:
            print(f"   ✅ Analysis stored! ({len(results.memories)} records)")
        else:
            print(f"   ❌ Analysis NOT stored")
    
    print("\n✅ Connection test complete!")


if __name__ == "__main__":
    asyncio.run(test_connections())