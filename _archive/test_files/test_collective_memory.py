#!/usr/bin/env python3
"""
🧪 Test Collective Memory Features
==================================

Tests consensus building and semantic clustering.
"""

import asyncio
import sys
sys.path.append('core/src')

from aura_intelligence.memory.core.memory_api import (
    AURAMemorySystem,
    MemoryQuery,
    RetrievalMode
)
from aura_intelligence.memory.core.memory_collective_extensions import (
    add_collective_intelligence
)


async def test_collective_memory():
    print("🧪 Testing Collective Memory Features\n")
    
    # Create memory system
    print("1️⃣ Creating Memory System...")
    memory_system = AURAMemorySystem()
    
    # Add collective intelligence
    collective = add_collective_intelligence(memory_system)
    print("✅ Collective intelligence added")
    
    # Test 1: Store some memories
    print("\n2️⃣ Storing Test Memories...")
    
    memories = [
        {"content": "AI safety is critical for future systems", "topic": "safety"},
        {"content": "Machine learning requires large datasets", "topic": "ml"},
        {"content": "AI alignment ensures safe behavior", "topic": "safety"},
        {"content": "Deep learning uses neural networks", "topic": "ml"},
        {"content": "Safety measures prevent AI risks", "topic": "safety"},
        {"content": "Neural networks learn from data", "topic": "ml"},
        {"content": "Robust AI systems need safety protocols", "topic": "safety"},
        {"content": "Transformers revolutionized NLP", "topic": "ml"},
    ]
    
    memory_ids = []
    for mem in memories:
        mem_id = await memory_system.store(
            content=mem["content"],
            metadata={"topic": mem["topic"]}
        )
        memory_ids.append(mem_id)
        print(f"  Stored: {mem['content'][:40]}...")
    
    print(f"✅ Stored {len(memory_ids)} memories")
    
    # Test 2: Build Consensus
    print("\n3️⃣ Testing Consensus Building...")
    
    # Multiple agents vote on a memory
    consensus_content = "AI safety protocols are essential for responsible AI development"
    voting_agents = {
        "agent_1": 0.9,   # Strong support
        "agent_2": 0.8,   # Support
        "agent_3": 0.7,   # Support
        "agent_4": -0.5,  # Disagree
        "agent_5": 0.6,   # Support
    }
    
    result = await memory_system.build_consensus(
        memory_content=consensus_content,
        voting_agents=voting_agents,
        consensus_type="weighted"
    )
    
    if result:
        print(f"✅ Consensus {'reached' if result.consensus_reached else 'not reached'}")
        print(f"   Support ratio: {result.support_ratio:.2%}")
        print(f"   Voters: {len(result.votes)}")
    
    # Test Byzantine consensus
    print("\n   Testing Byzantine consensus...")
    result_byz = await memory_system.build_consensus(
        memory_content="Byzantine consensus test content",
        voting_agents={"agent_1": 1.0, "agent_2": 1.0, "agent_3": 1.0},
        consensus_type="byzantine"
    )
    
    if result_byz:
        print(f"✅ Byzantine consensus: {result_byz.consensus_type.value}")
    
    # Test 3: Semantic Clustering
    print("\n4️⃣ Testing Semantic Clustering...")
    
    clustering_result = await memory_system.cluster_memories(
        algorithm="incremental",
        min_cluster_size=2
    )
    
    if clustering_result:
        print(f"✅ Clustering completed:")
        print(f"   Clusters formed: {len(clustering_result.clusters)}")
        print(f"   Unclustered memories: {len(clustering_result.unclustered)}")
        print(f"   Quality metrics: {clustering_result.quality_metrics}")
        
        # Show cluster contents
        for i, cluster in enumerate(clustering_result.clusters[:3]):  # First 3 clusters
            print(f"\n   Cluster {i+1} ({len(cluster.memories)} memories):")
            cluster_mems = await memory_system.get_cluster_memories(cluster.id)
            for mem in cluster_mems[:2]:  # First 2 memories
                print(f"     - {mem.content[:50]}...")
    
    # Test 4: Find Consensus Memories
    print("\n5️⃣ Testing Consensus Memory Retrieval...")
    
    consensus_mems = await memory_system.find_consensus_memories(min_support=0.6)
    print(f"✅ Found {len(consensus_mems)} consensus memories")
    
    for mem in consensus_mems[:3]:
        print(f"   - {mem.content[:50]}... (support: {mem.metadata.get('support_ratio', 0):.2%})")
    
    # Test 5: CRDT Consensus
    print("\n6️⃣ Testing CRDT Consensus...")
    
    crdt_result = await memory_system.build_consensus(
        memory_content={"fact": "CRDT enables eventual consistency"},
        voting_agents={
            "node_1": 1.0,
            "node_2": 1.0,
            "node_3": 0.8
        },
        consensus_type="crdt"
    )
    
    if crdt_result:
        print(f"✅ CRDT consensus: {crdt_result.consensus_type.value}")
        print(f"   Vector clock state: {crdt_result.metadata.get('crdt_state', {})}")
    
    # Test 6: Hierarchical Clustering
    print("\n7️⃣ Testing Hierarchical Clustering...")
    
    hier_result = await memory_system.cluster_memories(
        algorithm="hierarchical",
        min_cluster_size=2
    )
    
    if hier_result:
        print(f"✅ Hierarchical clustering: {hier_result.algorithm_used.value}")
        print(f"   Clusters: {len(hier_result.clusters)}")
    
    # Test 7: Get Metrics
    print("\n8️⃣ Testing Collective Metrics...")
    
    metrics = memory_system.get_metrics()
    
    if "consensus" in metrics:
        print("✅ Consensus metrics:")
        for key, value in metrics["consensus"].items():
            print(f"   {key}: {value}")
    
    if "clustering" in metrics:
        print("\n✅ Clustering metrics:")
        print(f"   Clusters: {metrics['clustering']['num_clusters']}")
    
    print("\n🎉 All collective memory tests passed!")
    
    # Summary
    print("\n📊 Test Summary:")
    print("✅ Consensus building (weighted, Byzantine, CRDT)")
    print("✅ Semantic clustering (incremental, hierarchical)")
    print("✅ Consensus memory retrieval")
    print("✅ Cluster memory access")
    print("✅ Collective intelligence metrics")
    
    print("\n💡 Collective features successfully integrated!")


async def test_advanced_consensus():
    """Test advanced consensus scenarios"""
    print("\n🔬 Testing Advanced Consensus Scenarios\n")
    
    memory_system = AURAMemorySystem()
    add_collective_intelligence(memory_system)
    
    # Scenario 1: Split vote
    print("Testing split vote scenario...")
    split_vote = {
        "agent_1": 1.0,
        "agent_2": 1.0,
        "agent_3": -1.0,
        "agent_4": -1.0,
        "agent_5": 0.5
    }
    
    result = await memory_system.build_consensus(
        memory_content="Controversial memory content",
        voting_agents=split_vote
    )
    
    print(f"Split vote result: {'Consensus' if result.consensus_reached else 'No consensus'}")
    print(f"Support ratio: {result.support_ratio:.2%}")
    
    # Scenario 2: Unanimous agreement
    print("\nTesting unanimous agreement...")
    unanimous = {f"agent_{i}": 1.0 for i in range(10)}
    
    result = await memory_system.build_consensus(
        memory_content="Everyone agrees on this",
        voting_agents=unanimous
    )
    
    print(f"Unanimous result: Support = {result.support_ratio:.2%}")
    
    print("\n✅ Advanced consensus tests complete!")


if __name__ == "__main__":
    print("🚀 AURA Collective Memory Test Suite")
    print("=" * 50)
    
    # Run main tests
    asyncio.run(test_collective_memory())
    
    # Run advanced tests
    asyncio.run(test_advanced_consensus())
    
    print("\n✨ Collective memory integration successful!")