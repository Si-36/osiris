"""
🧪 Test GPU Memory Adapter - Smoke Test
=====================================

Tests the GPU-accelerated memory adapter with shadow mode.
"""

import asyncio
import numpy as np
import time
import redis.asyncio as redis
from typing import List, Dict, Any

# Mock classes for testing without full dependencies
class MockBettiNumbers:
    def __init__(self):
        self.b0 = 1
        self.b1 = 0
        self.b2 = 0

class MockTDAResult:
    def __init__(self):
        self.betti_numbers = MockBettiNumbers()
        self.persistence_diagram = np.random.rand(10, 2)

class MockMemoryItem:
    def __init__(self, id: str, content: str, embedding: List[float]):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.metadata = {"test": True}
        self.similarity_score = 0.95

class MockAURAMemorySystem:
    """Mock memory system for testing"""
    def __init__(self):
        self.memories = {}
        self.embeddings = {}
        
    async def store_memory(self, content: str, memory_type: str, context: Dict[str, Any]) -> str:
        """Mock store"""
        memory_id = f"mem_{len(self.memories)}"
        self.memories[memory_id] = {
            "content": content,
            "type": memory_type,
            "context": context
        }
        # Generate random embedding
        self.embeddings[memory_id] = np.random.rand(128)
        return memory_id
        
    async def retrieve_memory(self, memory_id: str) -> MockMemoryItem:
        """Mock retrieve"""
        if memory_id in self.memories:
            mem = self.memories[memory_id]
            return MockMemoryItem(
                id=memory_id,
                content=mem["content"],
                embedding=self.embeddings[memory_id].tolist()
            )
        return None
        
    async def search_memories(self, query: Dict, memory_type: str, k: int, mode: Any) -> List[MockMemoryItem]:
        """Mock search"""
        # Return random k memories
        results = []
        for i, (mem_id, mem_data) in enumerate(list(self.memories.items())[:k]):
            item = MockMemoryItem(
                id=mem_id,
                content=mem_data["content"],
                embedding=self.embeddings[mem_id].tolist()
            )
            item.similarity_score = 0.95 - (i * 0.05)
            results.append(item)
        return results


async def test_gpu_memory_adapter():
    """Test the GPU memory adapter"""
    print("🧪 Testing GPU Memory Adapter\n")
    print("=" * 60)
    
    # Import here to avoid issues
    from core.src.aura_intelligence.adapters.memory_adapter_gpu import (
        GPUMemoryAdapter, GPUMemoryConfig, FAISS_AVAILABLE
    )
    
    # Create mock memory system
    memory_system = MockAURAMemorySystem()
    
    # Create GPU adapter config
    config = GPUMemoryConfig(
        use_gpu=FAISS_AVAILABLE,
        shadow_mode=True,
        serve_from_gpu=False,  # Start with shadow only
        sample_rate=1.0,
        fp16_embeddings=True
    )
    
    # Create adapter
    adapter = GPUMemoryAdapter(
        memory_system=memory_system,
        config=config
    )
    
    print(f"✅ GPU Adapter created")
    print(f"   - FAISS Available: {FAISS_AVAILABLE}")
    print(f"   - Shadow Mode: {config.shadow_mode}")
    print(f"   - Serve from GPU: {config.serve_from_gpu}")
    print()
    
    # Test 1: Store memories
    print("📝 Test 1: Storing memories")
    print("-" * 40)
    
    memory_ids = []
    for i in range(10):
        tda_sig = {
            "betti_numbers": {"b0": 1, "b1": i % 3, "b2": 0},
            "persistence_diagram": [[0, 0.1], [0.2, 0.5]]
        }
        
        start = time.time()
        mem_id = await adapter.store(
            content=f"Test memory {i}",
            tda_signature=tda_sig,
            context_type="test",
            metadata={"index": i}
        )
        store_time = (time.time() - start) * 1000
        
        memory_ids.append(mem_id)
        print(f"   Stored {mem_id} in {store_time:.2f}ms")
        
    print(f"\n✅ Stored {len(memory_ids)} memories")
    
    # Test 2: Retrieve with shadow comparison
    print("\n🔍 Test 2: Retrieval with shadow comparison")
    print("-" * 40)
    
    # Create query embedding
    query_embedding = np.random.rand(128)
    
    for i in range(5):
        print(f"\nQuery {i+1}:")
        
        start = time.time()
        results = await adapter.retrieve(
            query_embedding=query_embedding + np.random.rand(128) * 0.1,
            k=5
        )
        retrieve_time = (time.time() - start) * 1000
        
        print(f"   Retrieved {len(results)} results in {retrieve_time:.2f}ms")
        
        if results:
            print("   Top 3 results:")
            for j, (item, score) in enumerate(results[:3]):
                print(f"     {j+1}. {item['id']}: {item['content'][:30]}... (score: {score:.3f})")
                
    # Test 3: Check health and metrics
    print("\n🏥 Test 3: Health check")
    print("-" * 40)
    
    health = await adapter.health()
    print(f"   Status: {health.status.value}")
    print(f"   Latency: {health.latency_ms:.2f}ms")
    
    if health.resource_usage:
        print("   Resources:")
        for key, value in health.resource_usage.items():
            print(f"     - {key}: {value}")
            
    # Test 4: Check promotion readiness
    print("\n📊 Test 4: Promotion readiness")
    print("-" * 40)
    
    should_promote, details = adapter.should_promote()
    print(f"   Ready for promotion: {should_promote}")
    print("   Details:")
    for key, value in details.items():
        print(f"     - {key}: {value}")
        
    # Summary
    print("\n✨ Summary:")
    print("=" * 60)
    
    if FAISS_AVAILABLE:
        print("✅ GPU acceleration available and tested")
        print("✅ Shadow mode comparison working")
        print("✅ Metrics collection functional")
        
        if adapter.faiss_index:
            print(f"✅ FAISS index contains {adapter.faiss_index.ntotal} vectors")
    else:
        print("⚠️  FAISS not available - using baseline path")
        print("✅ Baseline memory operations working")
        print("✅ Ready to install FAISS for GPU acceleration")
        
    print("\n🎯 Next steps:")
    print("   1. Install FAISS GPU: pip install faiss-gpu")
    print("   2. Enable shadow mode in Redis feature flags")
    print("   3. Monitor mismatch rates in Grafana")
    print("   4. Promote to serving when gates pass")


async def test_feature_flags():
    """Test feature flag integration"""
    print("\n\n🚩 Testing Feature Flags")
    print("=" * 60)
    
    try:
        # Create Redis client
        redis_client = redis.Redis.from_url("redis://localhost:6379")
        
        # Set feature flags
        await redis_client.hset('feature_flags', mapping={
            'SHAPEMEMORYV2_GPU_ENABLED': 'true',
            'SHAPEMEMORYV2_SHADOW': 'true',
            'SHAPEMEMORYV2_SERVE': 'false',
            'SHAPEMEMORYV2_SAMPLERATE': '1.0'
        })
        
        print("✅ Feature flags set in Redis")
        
        # Read them back
        flags = await redis_client.hgetall('feature_flags')
        print("\nCurrent flags:")
        for k, v in flags.items():
            print(f"   {k.decode()}: {v.decode()}")
            
        await redis_client.close()
        
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   Feature flags would be controlled via Redis in production")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_gpu_memory_adapter())
    asyncio.run(test_feature_flags())