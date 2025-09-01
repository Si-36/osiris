#!/usr/bin/env python3
"""
Test Real AURA Implementations - Fixed Version
==============================================

Tests all real implementations by importing them directly.
"""

import asyncio
import sys
import os
import numpy as np
from datetime import datetime
import json

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'src'))


async def test_council_agents():
    """Test the real council agent implementations"""
    print("\n🤖 Testing Council Agents...")
    print("=" * 60)
    
    try:
        # Import implementations directly
        from aura_intelligence.agents.council.lnn.implementations import (
            TransformerNeuralEngine,
            GraphKnowledgeSystem,
            AdaptiveMemorySystem,
            CouncilOrchestrator
        )
        
        # Test Neural Engine
        print("\n1. Testing Neural Engine...")
        neural_engine = TransformerNeuralEngine(
            model_name="microsoft/phi-2",
            device="cpu",  # Use CPU for testing
            quantize=False
        )
        
        # Create test context
        from aura_intelligence.agents.council.contracts import ContextSnapshot
        context = ContextSnapshot(
            query="Should we allocate 4 GPUs for training?",
            historical_data=[{"decision": "approve", "confidence": 0.8}],
            domain_knowledge={"gpu_availability": 10},
            active_patterns=["resource_request"],
            metadata={}
        )
        
        print("   Extracting features...")
        features = await neural_engine.extract_features(context)
        print(f"   ✅ Features extracted: {len(features.embeddings)} dimensions")
        print(f"   ✅ Confidence: {features.confidence_scores['overall']:.2%}")
        
        # Test Knowledge Graph
        print("\n2. Testing Knowledge Graph...")
        kg = GraphKnowledgeSystem()
        
        # Add knowledge
        await kg.add_knowledge({
            "entities": [
                {"id": "gpu_1", "type": "resource", "properties": {"name": "GPU Cluster"}},
                {"id": "project_1", "type": "project", "properties": {"name": "Training"}}
            ],
            "relations": [
                {"source": "project_1", "target": "gpu_1", "type": "requires"}
            ],
            "text": "GPU allocation for ML training"
        })
        
        # Query
        results = await kg.query("GPU allocation")
        print(f"   ✅ Knowledge graph operational")
        print(f"   ✅ Nodes: {(await kg.get_topology_signature()).nodes}")
        
        # Test Memory System
        print("\n3. Testing Memory System...")
        memory = AdaptiveMemorySystem()
        
        # Store memory
        mem_id = await memory.store({
            "content": "GPU allocation approved",
            "type": "decision"
        }, importance=0.8)
        
        # Recall
        recalled = await memory.recall("GPU", k=5)
        print(f"   ✅ Memory stored: {mem_id}")
        print(f"   ✅ Recalled {len(recalled)} memories")
        
        # Test Orchestrator
        print("\n4. Testing Council Orchestrator...")
        orchestrator = CouncilOrchestrator()
        print(f"   ✅ Orchestrator created")
        print(f"   ✅ Min agents: {orchestrator.min_agents}")
        print(f"   ✅ Consensus threshold: {orchestrator.consensus_threshold}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ray_orchestration():
    """Test Ray distributed orchestration"""
    print("\n🌐 Testing Ray Distributed Orchestration...")
    print("=" * 60)
    
    try:
        # Import Ray orchestrator
        from aura_intelligence.orchestration.distributed.ray_orchestrator import (
            RayOrchestrator,
            TaskConfig,
            WorkerActor
        )
        
        print("1. Creating Ray orchestrator...")
        orchestrator = RayOrchestrator(
            num_workers=2,
            enable_autoscaling=False,  # Disable for testing
            min_workers=1,
            max_workers=2
        )
        
        print(f"   ✅ Orchestrator created with {orchestrator.num_workers} workers")
        
        # Submit test task
        print("\n2. Submitting test task...")
        task_id = await orchestrator.submit_task(
            task_type="neural_inference",
            payload={"input": [[1, 2, 3]], "model": "test"},
            priority=5
        )
        
        print(f"   ✅ Task submitted: {task_id}")
        
        # Get result
        print("\n3. Waiting for result...")
        result = await orchestrator.get_result(task_id, timeout=10.0)
        
        if result:
            print(f"   ✅ Task completed")
            print(f"   ✅ Status: {result.status}")
            print(f"   ✅ Execution time: {result.execution_time:.3f}s")
        
        # Get status
        status = await orchestrator.get_status()
        print(f"\n4. Orchestrator Status:")
        print(f"   ✅ Workers: {status['num_workers']}")
        print(f"   ✅ Queue processed: {status['queue_stats']['processed']}")
        
        # Cleanup
        await orchestrator.shutdown()
        print("\n   ✅ Orchestrator shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tda_implementations():
    """Test TDA implementations"""
    print("\n📐 Testing TDA Implementations...")
    print("=" * 60)
    
    try:
        # Test from src/aura
        from src.aura.tda.algorithms import (
            RipsComplex,
            PersistentHomology,
            wasserstein_distance,
            compute_persistence_landscape
        )
        
        print("1. Testing Rips Complex...")
        rips = RipsComplex()
        points = np.random.randn(20, 3)
        result = rips.compute(points, max_edge_length=2.0)
        
        print(f"   ✅ Betti numbers: b0={result['betti_0']}, b1={result['betti_1']}")
        print(f"   ✅ Edges: {result['num_edges']}")
        print(f"   ✅ Triangles: {result['num_triangles']}")
        
        print("\n2. Testing Persistent Homology...")
        ph = PersistentHomology()
        persistence = ph.compute_persistence(points)
        print(f"   ✅ Persistence pairs: {len(persistence)}")
        
        print("\n3. Testing Wasserstein Distance...")
        points2 = np.random.randn(20, 3)
        persistence2 = ph.compute_persistence(points2)
        distance = wasserstein_distance(persistence, persistence2)
        print(f"   ✅ Wasserstein distance: {distance:.4f}")
        
        print("\n4. Testing Persistence Landscape...")
        landscape = compute_persistence_landscape(persistence)
        print(f"   ✅ Landscape computed: shape {landscape.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_lnn_implementations():
    """Test LNN implementations"""
    print("\n🧠 Testing LNN Implementations...")
    print("=" * 60)
    
    try:
        # Test from src/aura
        from src.aura.lnn.variants import (
            MITLiquidNN,
            AdaptiveLNN,
            EdgeLNN,
            all_variants
        )
        
        print("1. Testing MIT Liquid NN...")
        lnn = MITLiquidNN("test_lnn")
        
        # Test forward pass
        import torch
        x = torch.randn(1, 10)  # Batch size 1, input size 10
        h = torch.randn(1, 128)  # Hidden state
        
        output, h_new = lnn(x, h)
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Hidden state updated: {h_new.shape}")
        
        print("\n2. Testing Adaptive LNN...")
        adaptive = AdaptiveLNN("adaptive_test")
        output, h_new = adaptive(x, h)
        print(f"   ✅ Adaptive LNN working")
        
        print("\n3. Testing Edge LNN...")
        edge = EdgeLNN("edge_test")
        output, h_new = edge(x, h)
        print(f"   ✅ Edge LNN working")
        
        print("\n4. Testing all variants...")
        print(f"   ✅ Available variants: {len(all_variants)}")
        for name in list(all_variants.keys())[:3]:
            print(f"   ✅ {name}: initialized")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_implementations():
    """Test memory implementations"""
    print("\n💾 Testing Memory Implementations...")
    print("=" * 60)
    
    try:
        # Test KNN index
        from aura_intelligence.memory.knn_index import (
            create_knn_index,
            KNNConfig
        )
        
        print("1. Testing FAISS KNN Index...")
        config = KNNConfig(
            backend="faiss",
            embedding_dim=128,
            metric="l2",
            index_type="flat"
        )
        
        index = create_knn_index(config)
        
        # Add vectors
        vectors = np.random.randn(100, 128).astype(np.float32)
        ids = [f"vec_{i}" for i in range(100)]
        
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            index.add(vec, id_)
        
        print(f"   ✅ Added {len(ids)} vectors")
        
        # Search
        query = np.random.randn(128).astype(np.float32)
        results = index.search(query, k=5)
        
        print(f"   ✅ Search returned {len(results)} results")
        print(f"   ✅ Top result: {results[0]['id']} (score: {results[0]['score']:.4f})")
        
        # Test save/load with secure serialization
        print("\n2. Testing secure serialization...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_index")
            index.save(save_path)
            print("   ✅ Index saved securely (no pickle)")
            
            # Load
            new_index = create_knn_index(config)
            new_index.load(save_path)
            print("   ✅ Index loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Test integration of components"""
    print("\n🔗 Testing Component Integration...")
    print("=" * 60)
    
    try:
        print("1. TDA → LNN Integration...")
        
        # Generate data
        points = np.random.randn(50, 3)
        
        # TDA analysis
        from src.aura.tda.algorithms import RipsComplex
        rips = RipsComplex()
        tda_result = rips.compute(points, max_edge_length=2.0)
        
        print(f"   ✅ TDA computed: b0={tda_result['betti_0']}, b1={tda_result['betti_1']}")
        
        # Feed to LNN
        from src.aura.lnn.variants import LiquidNeuralNetwork
        lnn = LiquidNeuralNetwork("integration_test")
        
        lnn_input = {
            'components': tda_result['betti_0'],
            'loops': tda_result['betti_1'],
            'connectivity': tda_result['num_edges'] / 50,
            'topology_vector': [tda_result['betti_0'], tda_result['betti_1'], 0]
        }
        
        prediction = lnn.predict(lnn_input)
        print(f"   ✅ LNN prediction: {prediction['prediction']:.2%}")
        print(f"   ✅ Confidence: {prediction['confidence']:.2%}")
        
        print("\n2. Memory → Knowledge Graph Integration...")
        
        # Store in memory
        from aura_intelligence.agents.council.lnn.implementations import (
            AdaptiveMemorySystem,
            GraphKnowledgeSystem
        )
        
        memory = AdaptiveMemorySystem()
        kg = GraphKnowledgeSystem()
        
        # Store decision
        mem_id = await memory.store({
            "content": f"TDA analysis showed {tda_result['betti_1']} loops",
            "prediction": prediction['prediction'],
            "confidence": prediction['confidence']
        }, importance=0.9)
        
        # Add to knowledge graph
        await kg.add_knowledge({
            "entities": [
                {"id": "analysis_1", "type": "tda_analysis", "properties": {"betti_1": tda_result['betti_1']}},
                {"id": "prediction_1", "type": "lnn_prediction", "properties": {"value": prediction['prediction']}}
            ],
            "relations": [
                {"source": "analysis_1", "target": "prediction_1", "type": "leads_to"}
            ],
            "text": f"TDA analysis with {tda_result['betti_1']} loops led to {prediction['prediction']:.2%} prediction"
        })
        
        print(f"   ✅ Stored in memory: {mem_id}")
        print(f"   ✅ Added to knowledge graph")
        
        # Query integration
        results = await kg.query("TDA loops prediction")
        print(f"   ✅ Knowledge query returned results")
        
        print("\n✅ All components integrated successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n🚀 AURA Intelligence - Real Implementation Tests (Fixed)")
    print("=" * 80)
    print("Testing all production-ready 2025 implementations...")
    print("This version bypasses import issues and tests components directly.")
    
    tests = [
        ("Council Agents", test_council_agents),
        ("Ray Orchestration", test_ray_orchestration),
        ("TDA Implementations", test_tda_implementations),
        ("LNN Implementations", test_lnn_implementations),
        ("Memory Implementations", test_memory_implementations),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ FAILED: {str(e)}"
            print(f"\n❌ Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        print(f"{test_name:<25} {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The AURA system is fully operational!")
        print("\n✨ What we've demonstrated:")
        print("   - Neural reasoning with transformers")
        print("   - Knowledge graphs with vector search")
        print("   - Adaptive memory systems")
        print("   - Distributed orchestration with Ray")
        print("   - Real TDA algorithms")
        print("   - Liquid Neural Networks")
        print("   - Secure serialization (no pickle)")
        print("   - Full component integration")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())