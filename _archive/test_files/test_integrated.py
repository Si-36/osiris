
"""
Integrated Component Test
========================

Tests all components working together with proper connections.
"""

import asyncio
from tda_memory_wrapper import TDAMemoryWrapper
from neural_memory_wrapper import NeuralMemoryWrapper

async def test_integrated_workflow():
    """Test complete workflow through all components"""
    
    print("\n🔄 INTEGRATED WORKFLOW TEST\n")
    
    # 1. Create components with wrappers
    from aura_intelligence.memory.core.memory_api import AURAMemorySystem
    from aura_intelligence.tda import AgentTopologyAnalyzer
    
    # Initialize memory first
    memory = AURAMemorySystem()
    print("✅ Memory initialized")
    
    # Wrap TDA with memory integration
    tda_base = AgentTopologyAnalyzer()
    tda = TDAMemoryWrapper(tda_base, memory)
    print("✅ TDA wrapped with memory integration")
    
    # 2. Create test workflow
    workflow_data = {
        "workflow_id": "integrated-test-1",
        "agents": [
            {"id": "collector"},
            {"id": "analyzer"},
            {"id": "reporter"}
        ],
        "dependencies": [
            {"source": "collector", "target": "analyzer"},
            {"source": "analyzer", "target": "reporter"}
        ]
    }
    
    # 3. Analyze with TDA (auto-stores in memory)
    print("\n📊 Analyzing topology...")
    analysis = await tda.analyze_workflow(
        workflow_data["workflow_id"],
        workflow_data
    )
    print(f"   Bottleneck score: {analysis.bottleneck_score:.3f}")
    
    # 4. Query memory for the stored analysis
    print("\n🔍 Querying memory for topology...")
    from aura_intelligence.memory.core.memory_api import MemoryQuery, RetrievalMode
    
    query = MemoryQuery(
        mode=RetrievalMode.SEMANTIC_SEARCH,
        query_text="workflow integrated-test-1",
        k=5
    )
    
    results = await memory.retrieve(query)
    print(f"   Found {len(results.memories)} memories")
    
    # 5. Success!
    if results.memories:
        print("\n✅ INTEGRATION SUCCESS!")
        print("   - Workflow analyzed by TDA")
        print("   - Analysis stored in Memory")
        print("   - Memory retrieval working")
        return True
    else:
        print("\n❌ Integration failed - memory not storing")
        return False


if __name__ == "__main__":
    asyncio.run(test_integrated_workflow())
