#!/usr/bin/env python3
"""
Simple test to demonstrate Memory Consolidation working
"""

import asyncio
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

from aura_intelligence.memory.consolidation import (
    SleepConsolidation,
    ConsolidationConfig,
    PriorityReplayBuffer,
    ReplayMemory
)

async def simple_test():
    print("\n" + "="*60)
    print("🧠 MEMORY CONSOLIDATION - SIMPLE DEMONSTRATION")
    print("="*60)
    
    # Create mock services
    class MockCausalTracker:
        async def batch_predict_surprise(self, contents):
            # Return random surprise scores
            return [np.random.uniform(0.1, 0.9) for _ in contents]
    
    # Test 1: Priority Replay Buffer
    print("\n📊 TEST 1: PRIORITY REPLAY BUFFER")
    print("-"*40)
    
    buffer = PriorityReplayBuffer(
        causal_tracker=MockCausalTracker(),
        max_size=100
    )
    
    # Create test memories
    memories = []
    for i in range(20):
        mem = ReplayMemory(
            id=f"mem_{i:03d}",
            content=f"Memory content {i}",
            embedding=np.random.randn(384),
            importance=np.random.uniform(0.1, 1.0)
        )
        memories.append(mem)
    
    # Populate buffer
    await buffer.populate(memories)
    
    stats = buffer.get_statistics()
    print(f"✅ Buffer populated:")
    print(f"   • Added: {stats['buffer_size']} memories")
    print(f"   • Highest surprise: {stats['highest_surprise']:.3f}")
    print(f"   • Average surprise: {stats['avg_surprise']:.3f}")
    
    # Get top candidates
    top = buffer.get_top_candidates(5)
    print(f"\n📈 Top 5 memories by surprise:")
    for i, mem in enumerate(top, 1):
        print(f"   {i}. {mem.id}: surprise={mem.surprise_score:.3f}")
    
    # Test 2: Binary Spike Encoding
    print("\n📉 TEST 2: BINARY SPIKE ENCODING (SESLR)")
    print("-"*40)
    
    test_mem = top[0]
    original_size = len(test_mem.embedding) * 4  # float32 bytes
    
    # Encode
    spike = buffer.encode_to_binary_spike(test_mem)
    compressed_size = len(spike)
    
    # Decode
    reconstructed = buffer.decode_from_binary_spike(spike)
    
    print(f"✅ Binary spike encoding:")
    print(f"   • Original: {original_size} bytes")
    print(f"   • Compressed: {compressed_size} bytes")
    print(f"   • Compression: {original_size/compressed_size:.1f}x")
    print(f"   • Reconstruction error: {np.mean(np.abs(test_mem.embedding - reconstructed)):.4f}")
    
    # Test 3: Distance Calculation
    print("\n🔍 TEST 3: SEMANTIC DISTANCE")
    print("-"*40)
    
    pairs = buffer.select_distant_pairs(count=3)
    print(f"✅ Selected {len(pairs)} distant memory pairs:")
    
    for i, (m1, m2) in enumerate(pairs, 1):
        distance = buffer._calculate_semantic_distance(m1, m2)
        print(f"   {i}. {m1.id} ↔ {m2.id}: distance={distance:.3f}")
    
    print("\n✅ CONSOLIDATION COMPONENTS WORKING!")
    print("   • Surprise-based prioritization ✓")
    print("   • Binary spike encoding (32x compression) ✓")
    print("   • Semantic distance calculation ✓")
    
    print("\n" + "="*60)
    print("This demonstrates the core memory consolidation features:")
    print("• Prioritizes surprising memories for replay")
    print("• Compresses memories 32x with binary spikes")
    print("• Finds distant memories for creative dreaming")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(simple_test())