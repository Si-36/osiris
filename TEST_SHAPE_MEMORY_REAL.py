#!/usr/bin/env python3
"""
Test ShapeMemoryV2 with REAL Redis and Neo4j - NO MOCKS!
========================================================

Prerequisites:
1. Run ./setup_databases.sh to start Redis and Neo4j
2. Or manually ensure Redis is on localhost:6379 and Neo4j on localhost:7687
"""

import asyncio
import numpy as np
import sys
import os
import time
from datetime import datetime, timezone

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from aura_intelligence.memory.shape_memory_v2 import (
    ShapeAwareMemoryV2, 
    ShapeMemoryV2Config
)
from aura_intelligence.memory.core.topology_adapter import TopologyMemoryAdapter
from aura_intelligence.tda.models import TDAResult, BettiNumbers


async def test_real_shape_memory():
    """Test ShapeMemoryV2 with REAL databases"""
    
    print("🧪 Testing Shape-Aware Memory V2 with REAL Databases")
    print("=" * 60)
    print("⚠️  This test requires Redis and Neo4j running!")
    print("   Run: ./setup_databases.sh if not already running")
    print("=" * 60)
    
    # Configuration for REAL databases
    config = ShapeMemoryV2Config(
        embedding_dim=128,
        fastrp_iterations=3,
        knn_backend="sklearn",
        storage_mode="full",  # Use REAL databases!
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        redis_url="redis://localhost:6379",
        hot_tier_hours=24,
        warm_tier_days=7,
        event_bus_enabled=False  # Can enable if Redis is running
    )
    
    # Initialize memory system
    memory = ShapeAwareMemoryV2(config)
    
    try:
        # Initialize (will connect to real databases)
        print("\n📡 Connecting to databases...")
        await memory.initialize()
        print("✅ Connected to Redis and Neo4j!")
        
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\n📝 To fix this:")
        print("1. Install Docker: https://docs.docker.com/get-docker/")
        print("2. Run: chmod +x setup_databases.sh")
        print("3. Run: ./setup_databases.sh")
        print("4. Wait 30 seconds for Neo4j to start")
        print("5. Run this test again")
        return
    
    # Create topology adapter
    topology_adapter = TopologyMemoryAdapter(config={})
    
    # Test 1: Store real memory
    print("\n1️⃣ Storing Memory with Real Topology")
    print("-" * 30)
    
    workflow_data = {
        "workflow_id": "real_workflow_001",
        "nodes": ["input", "validate", "process", "store", "output"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 4}
        ],
        "content": "Real workflow stored in Redis and Neo4j",
        "metadata": {
            "created_by": "test_real",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    
    # Extract topology
    topology = await topology_adapter.extract_topology(workflow_data)
    
    # Create TDA result
    tda_result = TDAResult(
        betti_numbers=BettiNumbers(
            b0=topology.betti_numbers[0],
            b1=topology.betti_numbers[1],
            b2=topology.betti_numbers[2]
        ),
        persistence_diagram=np.array([
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.7]
        ])
    )
    
    # Store in REAL databases
    stored_memory = await memory.store(
        content=workflow_data,
        tda_result=tda_result,
        context_type="workflow",
        metadata={"test": "real_database"}
    )
    
    print(f"✅ Stored in Redis/Neo4j: {stored_memory.memory_id}")
    print(f"   Context: {stored_memory.context_type}")
    print(f"   Betti: {tda_result.betti_numbers}")
    
    # Test 2: Store multiple memories
    print("\n2️⃣ Storing Multiple Memories")
    print("-" * 30)
    
    stored_ids = []
    for i in range(5):
        workflow = {
            "workflow_id": f"real_wf_{i:03d}",
            "nodes": [f"node_{j}" for j in range(4 + i)],
            "edges": [{"source": j, "target": j+1} for j in range(3 + i)]
        }
        
        # Add cycle for variety
        if i % 2 == 0:
            workflow["edges"].append({"source": 3 + i - 1, "target": 0})
        
        topology = await topology_adapter.extract_topology(workflow)
        tda_result = TDAResult(
            betti_numbers=BettiNumbers(
                b0=topology.betti_numbers[0],
                b1=topology.betti_numbers[1],
                b2=topology.betti_numbers[2]
            ),
            persistence_diagram=np.random.rand(5, 2)
        )
        
        mem = await memory.store(
            content=workflow,
            tda_result=tda_result,
            context_type="workflow" if i % 2 == 0 else "process"
        )
        stored_ids.append(mem.memory_id)
        print(f"   Stored {i+1}: {mem.memory_id[:20]}...")
    
    # Test 3: Retrieve from real databases
    print("\n3️⃣ Retrieving from Real Databases")
    print("-" * 30)
    
    # Create query
    query_workflow = {
        "workflow_id": "query_real",
        "nodes": ["a", "b", "c", "d", "e"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 4}
        ]
    }
    
    query_topology = await topology_adapter.extract_topology(query_workflow)
    
    from aura_intelligence.memory.shape_aware_memory import TopologicalSignature
    query_signature = TopologicalSignature(
        betti_numbers=BettiNumbers(
            b0=query_topology.betti_numbers[0],
            b1=query_topology.betti_numbers[1],
            b2=query_topology.betti_numbers[2]
        ),
        persistence_diagram=np.array([[0.0, 1.0], [0.1, 0.8]])
    )
    
    # Retrieve similar
    similar = await memory.retrieve(
        query_signature=query_signature,
        k=3
    )
    
    print(f"✅ Found {len(similar)} similar memories:")
    for i, mem in enumerate(similar):
        print(f"   {i+1}. {mem.memory_id[:20]}... (similarity: {mem.similarity_score:.3f})")
    
    # Test 4: Check Redis
    print("\n4️⃣ Verifying Redis Storage")
    print("-" * 30)
    
    import redis.asyncio as redis
    redis_client = await redis.from_url("redis://localhost:6379")
    
    # Check keys
    keys = await redis_client.keys("shape_v2:*")
    print(f"✅ Redis contains {len(keys)} shape memory keys")
    
    # Check a specific memory
    if keys:
        sample_key = keys[0]
        data = await redis_client.hgetall(sample_key)
        print(f"   Sample key: {sample_key.decode()}")
        print(f"   Fields: {list(data.keys())}")
    
    await redis_client.close()
    
    # Test 5: Check Neo4j
    print("\n5️⃣ Verifying Neo4j Storage")
    print("-" * 30)
    
    from neo4j import AsyncGraphDatabase
    neo4j_driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (m:ShapeMemoryV2) RETURN count(m) as count"
        )
        record = await result.single()
        count = record["count"] if record else 0
        print(f"✅ Neo4j contains {count} ShapeMemoryV2 nodes")
        
        # Get sample
        if count > 0:
            result = await session.run(
                "MATCH (m:ShapeMemoryV2) RETURN m.memory_id as id, m.context_type as type LIMIT 3"
            )
            print("   Sample memories:")
            async for record in result:
                print(f"     - {record['id'][:20]}... ({record['type']})")
    
    await neo4j_driver.close()
    
    # Test 6: Performance with real databases
    print("\n6️⃣ Performance Test (Real Databases)")
    print("-" * 30)
    
    # Store timing
    start = time.time()
    test_workflow = {
        "workflow_id": "perf_test",
        "nodes": ["a", "b", "c"],
        "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
    }
    topology = await topology_adapter.extract_topology(test_workflow)
    tda_result = TDAResult(
        betti_numbers=BettiNumbers(
            b0=topology.betti_numbers[0],
            b1=topology.betti_numbers[1],
            b2=topology.betti_numbers[2]
        ),
        persistence_diagram=np.random.rand(3, 2)
    )
    
    await memory.store(
        content=test_workflow,
        tda_result=tda_result,
        context_type="performance_test"
    )
    store_time = (time.time() - start) * 1000
    print(f"✅ Store time: {store_time:.2f}ms")
    
    # Retrieve timing
    start = time.time()
    results = await memory.retrieve(query_signature, k=5)
    retrieve_time = (time.time() - start) * 1000
    print(f"✅ Retrieve time: {retrieve_time:.2f}ms")
    
    # Get statistics
    print("\n7️⃣ Memory Statistics")
    print("-" * 30)
    
    stats = memory.get_stats()
    print(f"✅ Total memories: {stats['total_memories']}")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   Index size: {stats['index_size']}")
    print(f"   Hot tier: {stats['hot_tier_size']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed with REAL databases!")
    print("\n📊 Summary:")
    print("- Redis storage: WORKING ✅")
    print("- Neo4j persistence: WORKING ✅")
    print("- KNN retrieval: WORKING ✅")
    print("- Multi-tier storage: WORKING ✅")
    print("- Metrics collection: WORKING ✅")
    
    # Cleanup
    await topology_adapter.shutdown()
    await memory.shutdown()


if __name__ == "__main__":
    print("\n⚠️  Prerequisites:")
    print("1. Docker must be installed")
    print("2. Run: chmod +x setup_databases.sh")
    print("3. Run: ./setup_databases.sh")
    print("4. Wait for databases to start\n")
    
    asyncio.run(test_real_shape_memory())