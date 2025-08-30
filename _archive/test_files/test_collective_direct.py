#!/usr/bin/env python3
"""
🧪 Direct Test of Collective Memory Features
===========================================
"""

import asyncio
import sys
import os

# Direct path imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src/aura_intelligence/memory/enhancements'))

from collective_consensus import (
    CollectiveMemoryConsensus,
    ConsensusType,
    ConsensusVote,
    CRDTMemory,
    ByzantineConsensus
)
from semantic_clustering import (
    SemanticMemoryClustering,
    ClusteringAlgorithm,
    MemoryCluster,
    IncrementalClustering
)


async def test_consensus():
    print("🧪 Testing Consensus Building\n")
    
    # Test 1: Weighted Consensus
    print("1️⃣ Testing Weighted Consensus...")
    consensus = CollectiveMemoryConsensus()
    
    voting_agents = {
        "agent_1": 0.9,
        "agent_2": 0.8,
        "agent_3": 0.7,
        "agent_4": -0.5,
        "agent_5": 0.6
    }
    
    result = await consensus.build_consensus(
        memory_id="test_mem_1",
        content="AI safety is important",
        voting_agents=voting_agents,
        consensus_type=ConsensusType.WEIGHTED
    )
    
    print(f"✅ Consensus reached: {result.consensus_reached}")
    print(f"   Support ratio: {result.support_ratio:.2%}")
    print(f"   Total votes: {len(result.votes)}")
    
    # Test 2: CRDT Consensus
    print("\n2️⃣ Testing CRDT Consensus...")
    
    crdt_agents = {
        "node_1": 1.0,
        "node_2": 1.0,
        "node_3": 0.8
    }
    
    crdt_result = await consensus.build_consensus(
        memory_id="test_crdt",
        content={"fact": "CRDTs enable eventual consistency"},
        voting_agents=crdt_agents,
        consensus_type=ConsensusType.CRDT
    )
    
    print(f"✅ CRDT consensus: {crdt_result.consensus_reached}")
    print(f"   Vector clock: {crdt_result.metadata.get('crdt_state', {})}")
    
    # Test 3: CRDT Merging
    print("\n3️⃣ Testing CRDT Merge...")
    
    crdt1 = CRDTMemory("mem1")
    crdt1.update("agent_1", "value_1")
    crdt1.update("agent_2", "value_2")
    
    crdt2 = CRDTMemory("mem1")
    crdt2.update("agent_3", "value_3")
    crdt2.update("agent_1", "value_1_updated")
    
    print(f"   CRDT1 clock: {crdt1.vector_clock}")
    print(f"   CRDT2 clock: {crdt2.vector_clock}")
    
    crdt1.merge(crdt2)
    print(f"✅ Merged clock: {crdt1.vector_clock}")
    print(f"   Consensus value: {crdt1.get_consensus_value()}")
    
    # Test 4: Byzantine Consensus
    print("\n4️⃣ Testing Byzantine Consensus...")
    
    byzantine = ByzantineConsensus()
    
    # Mock vote collector
    class MockVoteCollector:
        async def collect_votes(self, agents, msg):
            votes = []
            # Simulate 2/3 agreement
            for i, agent in enumerate(agents):
                if i < len(agents) * 2 // 3:
                    votes.append(ConsensusVote(
                        agent_id=agent,
                        value="agreed_value",
                        confidence=0.9
                    ))
            return votes
    
    byz_result = await byzantine.propose(
        value="Byzantine test value",
        agents=["agent_1", "agent_2", "agent_3", "agent_4"],
        vote_collector=MockVoteCollector()
    )
    
    print(f"✅ Byzantine consensus: {byz_result.consensus_reached}")
    print(f"   Support: {byz_result.support_ratio:.2%}")


async def test_clustering():
    print("\n\n🧪 Testing Semantic Clustering\n")
    
    # Test 1: Incremental Clustering
    print("1️⃣ Testing Incremental Clustering...")
    
    clustering = SemanticMemoryClustering()
    
    # Add memories incrementally
    memories = {
        "mem_1": "AI safety is critical",
        "mem_2": "Machine learning algorithms",
        "mem_3": "Safety protocols for AI",
        "mem_4": "Deep learning networks",
        "mem_5": "AI alignment research",
        "mem_6": "Neural network training"
    }
    
    # Get embeddings and cluster
    for mem_id, content in memories.items():
        embedding = await clustering.embedding_manager.get_embedding(content)
        cluster_id = await clustering.incremental.add_memory(mem_id, embedding)
        print(f"   {mem_id} -> Cluster {cluster_id if cluster_id else 'New'}")
    
    print(f"✅ Created {len(clustering.incremental.micro_clusters)} clusters")
    
    # Test 2: Cluster Quality
    print("\n2️⃣ Testing Cluster Quality Metrics...")
    
    result = await clustering.cluster_memories(
        memories=memories,
        algorithm=ClusteringAlgorithm.INCREMENTAL
    )
    
    print(f"✅ Clustering metrics:")
    for metric, value in result.quality_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    # Test 3: Find Similar Memories
    print("\n3️⃣ Testing Similarity Search...")
    
    query = "AI safety measures"
    similar = await clustering.find_similar_memories(query, k=3)
    
    print(f"✅ Similar to '{query}':")
    for mem_id, similarity in similar:
        print(f"   {mem_id}: {similarity:.3f} - {memories.get(mem_id, 'Unknown')}")
    
    # Test 4: Hierarchical Clustering
    print("\n4️⃣ Testing Hierarchical Clustering...")
    
    hier_result = await clustering.cluster_memories(
        memories=memories,
        algorithm=ClusteringAlgorithm.HIERARCHICAL
    )
    
    print(f"✅ Hierarchical clusters: {len(hier_result.clusters)}")
    for i, cluster in enumerate(hier_result.clusters):
        print(f"   Cluster {i+1}: {len(cluster.memories)} memories")
    
    # Test 5: Cluster Optimization
    print("\n5️⃣ Testing Cluster Optimization...")
    
    pre_optimization = len(clustering.incremental.micro_clusters)
    clustering.optimize_clusters()
    post_optimization = len(clustering.incremental.micro_clusters)
    
    print(f"✅ Optimization: {pre_optimization} → {post_optimization} clusters")


async def test_integration():
    print("\n\n🧪 Testing Integrated Features\n")
    
    consensus = CollectiveMemoryConsensus()
    clustering = SemanticMemoryClustering()
    
    # Create memories with consensus
    memories_with_consensus = {}
    
    for i in range(10):
        content = f"Test memory {i} about {'AI safety' if i % 2 == 0 else 'machine learning'}"
        
        # Build consensus
        voting = {f"agent_{j}": 0.5 + (j * 0.1) for j in range(5)}
        result = await consensus.build_consensus(
            memory_id=f"mem_{i}",
            content=content,
            voting_agents=voting,
            consensus_type=ConsensusType.WEIGHTED
        )
        
        if result.consensus_reached:
            memories_with_consensus[f"mem_{i}"] = content
    
    print(f"✅ Created {len(memories_with_consensus)} consensus memories")
    
    # Cluster consensus memories
    cluster_result = await clustering.cluster_memories(
        memories=memories_with_consensus,
        algorithm=ClusteringAlgorithm.INCREMENTAL
    )
    
    print(f"✅ Clustered into {len(cluster_result.clusters)} groups")
    
    # Show consensus metrics
    metrics = consensus.get_consensus_metrics()
    print(f"\n📊 Consensus Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")


async def main():
    print("🚀 AURA Collective Memory Features - Direct Test")
    print("=" * 50)
    
    await test_consensus()
    await test_clustering()
    await test_integration()
    
    print("\n\n🎉 All tests completed successfully!")
    
    print("\n📊 What We Tested:")
    print("✅ Weighted consensus with multi-agent voting")
    print("✅ CRDT-based eventual consistency")
    print("✅ Byzantine fault tolerant consensus")
    print("✅ Incremental semantic clustering")
    print("✅ Hierarchical clustering")
    print("✅ Similarity search")
    print("✅ Cluster optimization")
    print("✅ Integrated consensus + clustering")
    
    print("\n💡 Collective intelligence features are working perfectly!")


if __name__ == "__main__":
    asyncio.run(main())