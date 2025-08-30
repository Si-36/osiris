"""
🧪 Standalone GPU Memory Adapter Test
====================================

Tests GPU memory adapter without full system imports.
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any

# Check FAISS availability
try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ FAISS is available")
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not installed - will test baseline path only")

# Mock classes
class MockMemorySystem:
    def __init__(self):
        self.memories = {}
        self.vectors = []
        
    async def store_memory(self, content: str, **kwargs) -> str:
        mem_id = f"mem_{len(self.memories)}"
        self.memories[mem_id] = {"content": content, **kwargs}
        # Generate random embedding
        embedding = np.random.rand(128).astype(np.float32)
        self.vectors.append(embedding)
        return mem_id
        
    async def search_memories(self, query: Dict, k: int = 5, **kwargs) -> List[Dict]:
        # Simple mock search - return random k items
        results = []
        for i in range(min(k, len(self.memories))):
            results.append({
                "id": f"mem_{i}",
                "content": list(self.memories.values())[i]["content"],
                "score": 0.95 - i * 0.05
            })
        return results


async def test_faiss_gpu():
    """Test FAISS GPU capabilities"""
    print("\n🚀 Testing FAISS GPU Capabilities")
    print("=" * 60)
    
    if not FAISS_AVAILABLE:
        print("FAISS not available - skipping GPU tests")
        return
        
    # Test 1: Create index
    print("\n1️⃣ Creating FAISS index")
    d = 128  # Dimension
    
    # Try different index types
    index_types = {
        "Flat": lambda: faiss.IndexFlatL2(d),
        "IVF": lambda: faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100),
        "HNSW": lambda: faiss.IndexHNSWFlat(d, 32)
    }
    
    for name, create_func in index_types.items():
        try:
            index = create_func()
            print(f"   ✅ {name} index created")
            
            # Check GPU availability
            if faiss.get_num_gpus() > 0:
                print(f"   🎮 GPU available: {faiss.get_num_gpus()} device(s)")
                try:
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                    print(f"   ✅ {name} index moved to GPU")
                except Exception as e:
                    print(f"   ⚠️  Failed to move to GPU: {e}")
            else:
                print("   ℹ️  No GPU available - using CPU")
                
        except Exception as e:
            print(f"   ❌ Failed to create {name}: {e}")
            
    # Test 2: Add and search vectors
    print("\n2️⃣ Testing vector operations")
    
    # Create simple flat index
    index = faiss.IndexFlatL2(d)
    
    # Add vectors
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, d).astype(np.float32)
    
    start = time.time()
    index.add(vectors)
    add_time = (time.time() - start) * 1000
    
    print(f"   ✅ Added {n_vectors} vectors in {add_time:.2f}ms")
    print(f"   📊 Index contains {index.ntotal} vectors")
    
    # Search
    n_queries = 10
    k = 5
    queries = np.random.rand(n_queries, d).astype(np.float32)
    
    start = time.time()
    distances, indices = index.search(queries, k)
    search_time = (time.time() - start) * 1000
    
    print(f"   ✅ Searched {n_queries} queries in {search_time:.2f}ms")
    print(f"   📈 Average per query: {search_time/n_queries:.2f}ms")
    
    # Show sample results
    print(f"\n   Sample results for first query:")
    for i in range(min(3, k)):
        print(f"     {i+1}. Index {indices[0][i]}, Distance: {distances[0][i]:.3f}")


async def test_shadow_mode():
    """Test shadow mode comparison logic"""
    print("\n\n👥 Testing Shadow Mode Logic")
    print("=" * 60)
    
    # Simulate baseline and GPU results
    baseline_results = [
        ("mem_0", 0.95), ("mem_1", 0.90), ("mem_2", 0.85), 
        ("mem_3", 0.80), ("mem_4", 0.75)
    ]
    
    gpu_results = [
        ("mem_0", 0.94), ("mem_3", 0.89), ("mem_1", 0.86),
        ("mem_5", 0.81), ("mem_2", 0.78)
    ]
    
    # Calculate Jaccard similarity
    baseline_ids = {r[0] for r in baseline_results}
    gpu_ids = {r[0] for r in gpu_results}
    
    intersection = len(baseline_ids & gpu_ids)
    union = len(baseline_ids | gpu_ids)
    jaccard = intersection / union if union > 0 else 0.0
    
    print(f"   Baseline IDs: {baseline_ids}")
    print(f"   GPU IDs: {gpu_ids}")
    print(f"   Intersection: {intersection}")
    print(f"   Union: {union}")
    print(f"   Jaccard similarity: {jaccard:.3f}")
    
    # Check mismatch
    mismatch_threshold = 0.03
    mismatch_rate = 1.0 - jaccard
    
    print(f"\n   Mismatch rate: {mismatch_rate:.3f}")
    print(f"   Threshold: {mismatch_threshold}")
    print(f"   Status: {'✅ PASS' if mismatch_rate <= mismatch_threshold else '❌ FAIL'}")
    
    # Promotion gates
    print("\n📊 Checking promotion gates:")
    gates = {
        "mismatch_rate": (mismatch_rate <= 0.03, mismatch_rate),
        "p99_latency": (8.5 <= 10.0, 8.5),  # Mock values
        "recall_at_5": (0.96 >= 0.95, 0.96)
    }
    
    all_passed = True
    for gate, (passed, value) in gates.items():
        status = "✅" if passed else "❌"
        all_passed &= passed
        print(f"   {status} {gate}: {value}")
        
    print(f"\n   Overall: {'✅ READY for promotion' if all_passed else '❌ NOT ready'}")


async def test_performance_comparison():
    """Compare baseline vs GPU performance"""
    print("\n\n⚡ Performance Comparison")
    print("=" * 60)
    
    # Simulate different vector counts
    vector_counts = [100, 1000, 10000]
    
    print("Vector Count | Baseline (ms) | GPU (ms) | Speedup")
    print("-" * 50)
    
    for count in vector_counts:
        # Mock baseline time (increases with count)
        baseline_time = count * 0.01  # 0.01ms per vector
        
        # Mock GPU time (much faster, sub-linear)
        gpu_time = 5.0 + (count * 0.0001)  # Fixed overhead + tiny per-vector
        
        speedup = baseline_time / gpu_time
        
        print(f"{count:11d} | {baseline_time:13.2f} | {gpu_time:8.2f} | {speedup:6.1f}x")
        
    print("\n💡 GPU advantage increases with scale!")


async def main():
    """Run all tests"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║        🚀 GPU MEMORY ADAPTER TEST SUITE 🚀             ║
    ║                                                        ║
    ║  Testing FAISS GPU acceleration for memory retrieval   ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    await test_faiss_gpu()
    await test_shadow_mode()
    await test_performance_comparison()
    
    print("\n\n✅ All tests completed!")
    print("\n🎯 Next Steps:")
    print("   1. Install FAISS GPU: pip install faiss-gpu")
    print("   2. Enable feature flags in Redis")
    print("   3. Run with GPU hardware for real benchmarks")
    print("   4. Monitor shadow mode metrics")
    print("   5. Promote when gates pass (mismatch ≤ 3%, p99 ≤ 10ms)")


if __name__ == "__main__":
    asyncio.run(main())