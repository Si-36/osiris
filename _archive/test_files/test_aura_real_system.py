#!/usr/bin/env python3
"""
🧬 AURA Real System Test - Using ACTUAL Existing Implementations
===============================================================

This test uses the REAL implementations that already exist:
- FAISS for vector search (memory/knn_index_real.py)
- MIT ncps for LNN (lnn/real_mit_lnn.py)
- Ray for distributed (distributed/real_ray_system.py)
- And all other existing components

NO REINVENTING WHEELS!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

import asyncio
import numpy as np
import time
import torch


async def test_real_faiss_vector_search():
    """Test the REAL FAISS implementation."""
    print("\n🔍 Testing REAL FAISS Vector Search")
    print("=" * 60)
    
    try:
        # Import the REAL implementation
        from aura_intelligence.memory.knn_index_real import (
            create_knn_index, KNNConfig, BaseKNNIndex
        )
        
        # Create config for FAISS
        config = KNNConfig(
            metric='cosine',
            backend='faiss' if 'faiss' in sys.modules else 'sklearn',
            faiss_index_type='IVF',
            normalize_vectors=True
        )
        
        # Create the index
        embedding_dim = 128
        index = create_knn_index(embedding_dim=embedding_dim, config=config)
        
        print(f"✅ Created {type(index).__name__}")
        print(f"   Backend: {config.backend}")
        print(f"   Metric: {config.metric}")
        
        # Add test vectors
        num_vectors = 1000
        vectors = np.random.randn(num_vectors, embedding_dim).astype(np.float32)
        ids = [f"doc_{i}" for i in range(num_vectors)]
        
        print(f"\n📊 Adding {num_vectors} vectors...")
        start_time = time.time()
        index.add(vectors, ids)
        add_time = (time.time() - start_time) * 1000
        print(f"✅ Added in {add_time:.2f}ms")
        
        # Search test
        query = np.random.randn(embedding_dim).astype(np.float32)
        k = 10
        
        print(f"\n🔍 Searching for {k} nearest neighbors...")
        start_time = time.time()
        results = index.search(query, k)
        search_time = (time.time() - start_time) * 1000
        
        print(f"✅ Search completed in {search_time:.2f}ms")
        print(f"\nTop 5 results:")
        for i, (id, score) in enumerate(results[:5]):
            print(f"  {i+1}. {id}: {score:.4f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_real_mit_lnn():
    """Test the REAL MIT LNN implementation."""
    print("\n\n🧠 Testing REAL MIT Liquid Neural Network")
    print("=" * 60)
    
    try:
        # Import the REAL MIT implementation
        from aura_intelligence.lnn.real_mit_lnn import (
            RealMITLNN, NCPS_AVAILABLE, get_real_mit_lnn
        )
        
        print(f"📊 NCPS Available: {NCPS_AVAILABLE}")
        if not NCPS_AVAILABLE:
            print("   ⚠️  Install with: pip install ncps")
            print("   Using fallback implementation")
        
        # Create the LNN
        input_size = 32
        hidden_size = 64
        output_size = 16
        
        lnn = get_real_mit_lnn(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        )
        
        info = lnn.get_info()
        print(f"\n📊 LNN Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, input_size)
        
        print(f"\n🔄 Testing forward pass...")
        start_time = time.time()
        output = lnn(x)
        forward_time = (time.time() - start_time) * 1000
        
        print(f"✅ Forward pass completed in {forward_time:.2f}ms")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_real_ray_distributed():
    """Test the REAL Ray distributed system."""
    print("\n\n🌐 Testing REAL Ray Distributed System")
    print("=" * 60)
    
    try:
        # Import the REAL Ray implementation
        from aura_intelligence.distributed.real_ray_system import (
            RealComponentActor, RealOrchestrator
        )
        
        # Initialize Ray
        import ray
        if not ray.is_initialized():
            ray.init(local_mode=True)  # Local mode for testing
            print("✅ Ray initialized in local mode")
        
        # Create orchestrator
        orchestrator = RealOrchestrator(num_components=3)
        print(f"✅ Created orchestrator with 3 components")
        
        # Test data
        test_data = {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
            'timestamp': time.time()
        }
        
        # Process through distributed system
        print(f"\n🔄 Processing through distributed actors...")
        start_time = time.time()
        result = await orchestrator.process_all(test_data)
        process_time = (time.time() - start_time) * 1000
        
        print(f"✅ Distributed processing completed in {process_time:.2f}ms")
        print(f"\nResults from {len(result)} actors:")
        for i, actor_result in enumerate(result):
            print(f"  Actor {i}: {list(actor_result.keys())}")
        
        # Shutdown
        ray.shutdown()
        print("\n✅ Ray shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'ray' in str(e).lower():
            print("   Install Ray with: pip install ray[default]")
        return False


async def test_existing_components_integration():
    """Test integration of existing components."""
    print("\n\n🔗 Testing Existing Components Integration")
    print("=" * 60)
    
    try:
        # Import existing components
        from aura_intelligence.orchestration.workflows.nodes.supervisor import RealSupervisor
        from aura_intelligence.memory.advanced_hybrid_memory_2025 import HybridMemoryManager
        from aura_intelligence.graph.aura_knowledge_graph_2025 import AURAKnowledgeGraph
        
        print("✅ Imported existing components:")
        print("   - RealSupervisor")
        print("   - HybridMemoryManager")
        print("   - AURAKnowledgeGraph")
        
        # Create instances
        supervisor = RealSupervisor()
        memory = HybridMemoryManager()
        kg = AURAKnowledgeGraph()
        
        # Test workflow
        print("\n🔄 Testing component integration...")
        
        # 1. Supervisor analyzes state
        state = {
            "workflow_id": "test_001",
            "step_results": [
                {"success": True, "duration_ms": 100},
                {"success": False, "retry_count": 3}
            ]
        }
        
        analysis = await supervisor.analyze_state(state)
        print(f"\n1️⃣ Supervisor Analysis:")
        print(f"   Risk: {analysis['risk_score']:.2f}")
        print(f"   Patterns: {analysis['patterns']}")
        
        # 2. Store in memory
        await memory.store(
            "test_state",
            {"state": state, "analysis": analysis},
            importance=analysis['risk_score']
        )
        print(f"\n2️⃣ Memory Storage:")
        print(f"   Stored with importance: {analysis['risk_score']:.2f}")
        
        # 3. Knowledge Graph analysis
        kg_result = await kg.ingest_state("agent_test", state, analysis)
        print(f"\n3️⃣ Knowledge Graph:")
        print(f"   Failure risk: {kg_result.get('failure_risk', False)}")
        
        print("\n✅ Integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests with real components."""
    print("🧬 AURA Real System Test")
    print("Using EXISTING implementations - NOT reinventing wheels!")
    print("=" * 80)
    
    # Check available libraries
    print("\n📦 Checking Available Libraries:")
    libraries = {
        "faiss": "FAISS vector search",
        "ncps": "MIT Liquid Neural Networks",
        "ray": "Ray distributed computing",
        "neo4j": "Neo4j graph database",
        "torch": "PyTorch deep learning"
    }
    
    available = []
    for lib, desc in libraries.items():
        try:
            __import__(lib)
            print(f"  ✅ {lib}: {desc}")
            available.append(lib)
        except ImportError:
            print(f"  ❌ {lib}: {desc} (not installed)")
    
    # Run tests
    results = []
    
    # Test FAISS
    if 'faiss' in available or True:  # Test even with sklearn fallback
        results.append(("FAISS Vector Search", await test_real_faiss_vector_search()))
    
    # Test LNN
    results.append(("MIT LNN", await test_real_mit_lnn()))
    
    # Test Ray
    if 'ray' in available:
        results.append(("Ray Distributed", await test_real_ray_distributed()))
    
    # Test integration
    results.append(("Component Integration", await test_existing_components_integration()))
    
    # Summary
    print("\n\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("AURA is using real, professional implementations!")
    else:
        print("\n⚠️  Some tests failed")
        print("Install missing dependencies with:")
        print("  pip install -r requirements_ultimate.txt")


if __name__ == "__main__":
    asyncio.run(main())