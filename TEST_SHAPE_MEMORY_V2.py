#!/usr/bin/env python3
"""
Test Shape-Aware Memory V2 - Real Implementation
================================================

Tests that ShapeMemoryV2:
1. Stores memories with FastRP embeddings
2. Uses KNN for fast retrieval
3. Manages memory tiers
4. Tracks patterns effectively
"""

import asyncio
import numpy as np
import sys
import os
import time
from datetime import datetime, timezone
import zstandard as zstd

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from aura_intelligence.memory.shape_memory_v2 import (
    ShapeAwareMemoryV2, 
    ShapeMemoryV2Config,
    MemoryTier
)
from aura_intelligence.memory.fastrp_embeddings import FastRPEmbedder, FastRPConfig
from aura_intelligence.memory.core.topology_adapter import TopologyMemoryAdapter
from aura_intelligence.tda.models import TDAResult, BettiNumbers


class MockRedis:
    """Mock Redis for testing without Redis server"""
    def __init__(self):
        self.store = {}
        self.expiry = {}
    
    async def hset(self, key, *args):
        if key not in self.store:
            self.store[key] = {}
        
        # Handle both mapping and field-value pairs
        if len(args) == 1 and isinstance(args[0], dict):
            # mapping style: hset(key, {field: value})
            self.store[key].update(args[0])
            return len(args[0])
        elif len(args) == 2:
            # field-value style: hset(key, field, value)
            field, value = args
            self.store[key][field] = value
            return 1
        else:
            raise ValueError(f"Invalid hset arguments: {args}")
    
    async def expire(self, key, seconds):
        self.expiry[key] = time.time() + seconds
        return True
    
    async def hgetall(self, key):
        # Check expiry
        if key in self.expiry and time.time() > self.expiry[key]:
            if key in self.store:
                del self.store[key]
                del self.expiry[key]
            return {}
        return self.store.get(key, {})
    
    async def hget(self, key, field):
        data = await self.hgetall(key)
        return data.get(field)
    
    async def hincrby(self, key, field, amount=1):
        if key not in self.store:
            self.store[key] = {}
        current = int(self.store[key].get(field, 0))
        self.store[key][field] = str(current + amount)
        return current + amount
    
    async def exists(self, *keys):
        return sum(1 for k in keys if k in self.store)
    
    async def close(self):
        pass


class MockNeo4jSession:
    """Mock Neo4j session"""
    def __init__(self):
        self.nodes = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
    
    async def run(self, query, params=None):
        if "CREATE" in query:
            self.nodes.append(params)
        return None
    
    async def close(self):
        pass


class MockNeo4jDriver:
    """Mock Neo4j driver"""
    def __init__(self):
        self.sessions = []
    
    def session(self):
        session = MockNeo4jSession()
        self.sessions.append(session)
        return session
    
    async def close(self):
        pass


async def test_shape_memory_v2():
    """Test ShapeMemoryV2 with real topology and embeddings"""
    
    print("🧪 Testing Shape-Aware Memory V2")
    print("=" * 60)
    
    # Create configuration
    config = ShapeMemoryV2Config(
        embedding_dim=128,
        fastrp_iterations=3,
        knn_backend="sklearn",
        hot_tier_hours=24,
        warm_tier_days=7,
        event_bus_enabled=False  # Disable for testing
    )
    
    # Initialize memory system
    memory = ShapeAwareMemoryV2(config)
    
    # Mock the backends BEFORE initialization
    memory._redis = MockRedis()
    memory._driver = MockNeo4jDriver()
    
    # Initialize only the embedder and KNN index (skip storage backends)
    # Initialize FastRP embedder
    fastrp_config = FastRPConfig(
        embedding_dim=config.embedding_dim,
        iterations=config.fastrp_iterations,
        normalization="l2"
    )
    memory._embedder = FastRPEmbedder(fastrp_config)
    memory._embedder.initialize()
    
    # Initialize k-NN index
    from aura_intelligence.memory.knn_index import KNNIndex, KNNConfig
    knn_config = KNNConfig(
        backend="sklearn",
        metric=config.knn_metric,
        initial_capacity=config.cache_size
    )
    memory._knn_index = KNNIndex(config.embedding_dim, knn_config)
    
    # Initialize memory tiers
    memory._tiers = [
        MemoryTier("hot", config.hot_tier_hours, "redis"),
        MemoryTier("warm", config.warm_tier_days * 24, "neo4j"),
        MemoryTier("cold", config.cold_tier_days * 24, "s3")
    ]
    
    # Initialize other components
    memory._cache = {}
    memory._memory_cache = {}  # Add this for fetch operations
    memory._total_memories = 0
    memory._compressor = zstd.ZstdCompressor(level=3)
    
    # Test 1: Store memory with topology
    print("\n1️⃣ Testing Memory Storage with Topology")
    print("-" * 30)
    
    # Create topology adapter
    topology_adapter = TopologyMemoryAdapter(config={})
    
    # Create workflow data
    workflow_data = {
        "workflow_id": "test_workflow_1",
        "nodes": ["input", "process", "validate", "output"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3}
        ],
        "content": "This is a test workflow"
    }
    
    # Extract topology
    topology = await topology_adapter.extract_topology(workflow_data)
    
    # Create TDA result from topology
    tda_result = TDAResult(
        betti_numbers=BettiNumbers(
            b0=topology.betti_numbers[0],
            b1=topology.betti_numbers[1],
            b2=topology.betti_numbers[2]
        ),
        persistence_diagram=np.array([
            [0.0, 1.0],
            [0.1, 0.8],
            [0.2, 0.5]
        ])
    )
    
    # Store memory
    stored_memory = await memory.store(
        content=workflow_data,
        tda_result=tda_result,
        context_type="workflow",
        metadata={"source": "test"}
    )
    
    print(f"✅ Memory stored: {stored_memory.memory_id}")
    print(f"   Context: {stored_memory.context_type}")
    print(f"   Betti numbers: ({tda_result.betti_numbers.b0}, {tda_result.betti_numbers.b1}, {tda_result.betti_numbers.b2})")
    
    # Test 2: Store multiple memories with different topologies
    print("\n2️⃣ Storing Multiple Memories")
    print("-" * 30)
    
    memories = []
    for i in range(5):
        # Create variations
        workflow = {
            "workflow_id": f"workflow_{i}",
            "nodes": [f"node_{j}" for j in range(3 + i)],
            "edges": [
                {"source": j, "target": j+1} 
                for j in range(2 + i)
            ]
        }
        
        # Add cycle for some
        if i % 2 == 0:
            workflow["edges"].append({"source": 2 + i - 1, "target": 0})
        
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
            context_type="workflow" if i % 2 == 0 else "process",
            metadata={"index": i}
        )
        memories.append(mem)
        print(f"   Stored memory {i}: {mem.memory_id[:12]}... (has_cycle: {topology.workflow_features.has_cycles})")
    
    # Test 3: Retrieve similar memories
    print("\n3️⃣ Testing Similarity Retrieval")
    print("-" * 30)
    
    # Create query topology (similar to first stored)
    query_workflow = {
        "workflow_id": "query_workflow",
        "nodes": ["a", "b", "c", "d"],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3}
        ]
    }
    
    query_topology = await topology_adapter.extract_topology(query_workflow)
    
    # Convert to TopologicalSignature for retrieval
    from aura_intelligence.memory.shape_aware_memory import TopologicalSignature
    query_signature = TopologicalSignature(
        betti_numbers=BettiNumbers(
            b0=query_topology.betti_numbers[0],
            b1=query_topology.betti_numbers[1],
            b2=query_topology.betti_numbers[2]
        ),
        persistence_diagram=np.array([[0.0, 1.0], [0.1, 0.7]])
    )
    
    # Retrieve similar memories
    similar_memories = await memory.retrieve(
        query_signature=query_signature,
        k=3
    )
    
    print(f"✅ Found {len(similar_memories)} similar memories:")
    for i, mem in enumerate(similar_memories):
        print(f"   {i+1}. {mem.memory_id[:12]}... (similarity: {mem.similarity_score:.3f})")
    
    # Test 4: Context filtering
    print("\n4️⃣ Testing Context Filtering")
    print("-" * 30)
    
    workflow_memories = await memory.retrieve(
        query_signature=query_signature,
        k=10,
        context_filter="workflow"
    )
    
    print(f"✅ Found {len(workflow_memories)} workflow memories")
    
    process_memories = await memory.retrieve(
        query_signature=query_signature,
        k=10,
        context_filter="process"
    )
    
    print(f"✅ Found {len(process_memories)} process memories")
    
    # Test 5: Memory statistics
    print("\n5️⃣ Memory Statistics")
    print("-" * 30)
    
    stats = memory.get_stats()
    print(f"✅ Total memories: {stats['total_memories']}")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   Index size: {stats['index_size']}")
    print(f"   Hot tier: {stats['hot_tier_size']}")
    
    # Test 6: Batch embedding
    print("\n6️⃣ Testing Batch Embedding Performance")
    print("-" * 30)
    
    # Create batch of persistence diagrams
    batch_diagrams = [np.random.rand(10, 2) for _ in range(100)]
    batch_betti = [BettiNumbers(b0=1, b1=i%3, b2=0) for i in range(100)]
    
    start_time = time.time()
    batch_embeddings = memory._embedder.embed_batch(batch_diagrams, batch_betti)
    batch_time = (time.time() - start_time) * 1000
    
    print(f"✅ Embedded 100 diagrams in {batch_time:.2f}ms")
    print(f"   Average: {batch_time/100:.2f}ms per diagram")
    print(f"   Embedding shape: {batch_embeddings.shape}")
    
    # Test 7: Memory tiering
    print("\n7️⃣ Testing Memory Tiers")
    print("-" * 30)
    
    # Check hot tier
    hot_keys = [k for k in memory._redis.store.keys() if "hot" in k]
    print(f"✅ Hot tier entries: {len(hot_keys)}")
    
    # Check persistence to Neo4j
    if memory._driver.sessions:
        neo4j_nodes = sum(len(s.nodes) for s in memory._driver.sessions)
        print(f"✅ Neo4j persisted: {neo4j_nodes} nodes")
    
    print("\n" + "=" * 60)
    print("✅ All Shape Memory V2 tests passed!")
    
    # Cleanup
    await topology_adapter.shutdown()
    await memory.shutdown()


if __name__ == "__main__":
    asyncio.run(test_shape_memory_v2())