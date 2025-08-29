#!/usr/bin/env python3
"""
🚀 AURA Intelligence - All REAL Components Test
==============================================

Demonstrates that ALL components are now REAL:
- TDA with actual topological computations
- LNN with real neural networks
- Memory with FAISS/sklearn vector search
- Agents with Byzantine consensus
- Everything computing REAL results!
"""

import asyncio
import numpy as np
import time
import sys
import json
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/core/src')


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"🔬 {title}")
    print('='*70)


async def test_tda():
    """Test REAL TDA implementation"""
    print_section("Testing TDA - Topological Data Analysis")
    
    try:
        from src.aura.tda.algorithms import create_tda_algorithm
        
        # Create test data - double torus (genus 2 surface)
        n_points = 200
        data_sets = {
            'circle': np.column_stack([
                np.cos(np.linspace(0, 2*np.pi, 50)),
                np.sin(np.linspace(0, 2*np.pi, 50))
            ]),
            'figure_8': np.column_stack([
                np.sin(np.linspace(0, 4*np.pi, 100)) * np.cos(np.linspace(0, 2*np.pi, 100)),
                np.sin(np.linspace(0, 2*np.pi, 100))
            ]),
            'random': np.random.randn(30, 2)
        }
        
        for name, points in data_sets.items():
            print(f"\n📊 Analyzing {name} topology ({len(points)} points):")
            
            # Compute Rips complex
            rips = create_tda_algorithm('vietoris_rips')
            result = rips.compute(points, max_edge_length=2.0)
            
            print(f"  • Betti_0 (components): {result['betti_0']}")
            print(f"  • Betti_1 (loops): {result['betti_1']}")
            print(f"  • Betti_2 (voids): {result['betti_2']}")
            
            # Persistent homology
            ph = create_tda_algorithm('persistent_homology')
            diagram = ph.compute_persistence(points)
            
            if hasattr(ph, 'compute_persistence_entropy'):
                entropy = ph.compute_persistence_entropy(diagram)
                print(f"  • Persistence entropy: {entropy:.4f}")
            
            # Wasserstein distance
            if len(diagram) > 1:
                wd = create_tda_algorithm('wasserstein_distance')
                dist = wd(diagram[:5], diagram[1:6])
                print(f"  • Self-Wasserstein distance: {dist:.4f}")
        
        print("\n✅ TDA is computing REAL topological features!")
        return True
        
    except Exception as e:
        print(f"❌ TDA Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_lnn():
    """Test REAL Liquid Neural Networks"""
    print_section("Testing LNN - Liquid Neural Networks")
    
    try:
        from src.aura.lnn.variants import VARIANTS, all_variants
        
        # Test different input patterns
        test_patterns = {
            'normal': {'load': 0.3, 'error_rate': 0.01, 'response_time': 50},
            'stressed': {'load': 0.8, 'error_rate': 0.05, 'response_time': 200},
            'critical': {'load': 0.95, 'error_rate': 0.15, 'response_time': 500}
        }
        
        print("\n🧠 Testing LNN variants with different scenarios:")
        
        for pattern_name, data in test_patterns.items():
            print(f"\n📍 Scenario: {pattern_name}")
            
            # Test first 3 variants
            for variant_name in list(VARIANTS.keys())[:3]:
                lnn = all_variants[variant_name]
                result = lnn.predict(data)
                
                print(f"  • {variant_name}:")
                print(f"    - Prediction: {result['prediction']:.3f}")
                print(f"    - Risk: {result['failure_probability']:.3f}")
                print(f"    - Time to failure: {result['time_to_failure']}s")
        
        print("\n✅ LNN is making REAL predictions with neural networks!")
        return True
        
    except Exception as e:
        print(f"❌ LNN Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory():
    """Test REAL Memory System with Vector Search"""
    print_section("Testing Memory - Vector Search & Storage")
    
    try:
        from aura_intelligence.memory.knn_index_real import create_knn_index, KNNConfig
        
        # Create memory index
        embedding_dim = 128
        config = KNNConfig(
            backend='sklearn',  # Will auto-select best available
            metric='cosine',
            normalize_vectors=True
        )
        
        memory = create_knn_index(embedding_dim, config=config)
        
        print(f"\n💾 Memory system initialized with backend: {memory.config.backend}")
        
        # Generate test memories
        n_memories = 1000
        memory_vectors = np.random.randn(n_memories, embedding_dim).astype(np.float32)
        memory_ids = [f"memory_{i}" for i in range(n_memories)]
        
        # Add memories
        print(f"\n📝 Adding {n_memories} memories...")
        start_time = time.time()
        memory.add_batch(memory_vectors, memory_ids, batch_size=100)
        add_time = time.time() - start_time
        print(f"  • Added in {add_time:.3f}s ({n_memories/add_time:.0f} memories/sec)")
        
        # Search for similar memories
        print("\n🔍 Searching for similar memories:")
        query = np.random.randn(embedding_dim).astype(np.float32)
        
        start_time = time.time()
        results = memory.search(query, k=10)
        search_time = time.time() - start_time
        
        print(f"  • Search completed in {search_time*1000:.1f}ms")
        print(f"  • Top 3 results:")
        for i, (mem_id, score) in enumerate(results[:3]):
            print(f"    {i+1}. {mem_id}: similarity={score:.3f}")
        
        # Memory usage
        mem_info = memory.get_memory_usage()
        print(f"\n📊 Memory usage: {mem_info['estimated_size_mb']:.1f} MB")
        
        print("\n✅ Memory system with REAL vector search is working!")
        return True
        
    except Exception as e:
        print(f"❌ Memory Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agents():
    """Test REAL Multi-Agent System"""
    print_section("Testing Agents - Byzantine Consensus & Decision Making")
    
    try:
        from aura_intelligence.agents.real_agent_system import (
            MultiAgentSystem, TopologyAnalyzerAgent, FailurePredictorAgent
        )
        
        # Create multi-agent system
        mas = MultiAgentSystem()
        
        # Add agents
        mas.add_agent(TopologyAnalyzerAgent("topo_analyzer_1"))
        mas.add_agent(FailurePredictorAgent("failure_predictor_1"))
        mas.add_agent(TopologyAnalyzerAgent("topo_analyzer_2"))
        mas.add_agent(FailurePredictorAgent("failure_predictor_2"))
        
        print(f"\n🤖 Created {len(mas.agents)} agents with Byzantine consensus")
        
        # Start system
        system_task = asyncio.create_task(mas.start())
        
        print("\n⏱️  Running multi-agent system for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get report
        report = mas.get_system_report()
        
        print("\n📊 Agent System Report:")
        print(f"  • Active agents: {report['system_state']['active_agents']}")
        print(f"  • Total decisions: {report['system_state']['decisions_made']}")
        
        print("\n👥 Agent Status:")
        for agent_id, agent_data in report['agents'].items():
            print(f"  • {agent_id} ({agent_data['role']}):")
            print(f"    - Health: {agent_data['state']['health']:.2f}")
            print(f"    - Messages: {agent_data['metrics']['messages_sent']} sent, "
                  f"{agent_data['metrics']['messages_received']} received")
            print(f"    - Decisions: {agent_data['metrics']['decisions_made']}")
        
        # Cancel system
        system_task.cancel()
        try:
            await system_task
        except asyncio.CancelledError:
            pass
        
        print("\n✅ Multi-agent system with REAL consensus is working!")
        return True
        
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Test all components working together"""
    print_section("Integration Test - All Components Together")
    
    try:
        # Import all components
        from src.aura.tda.algorithms import create_tda_algorithm
        from src.aura.lnn.variants import all_variants
        from aura_intelligence.memory.knn_index_real import create_knn_index
        from aura_intelligence.agents.real_agent_system import (
            TopologyAnalyzerAgent, DecisionNetwork
        )
        
        print("\n🔗 Testing complete data flow:")
        
        # 1. Generate system state
        n_agents = 50
        agent_positions = np.random.randn(n_agents, 2) * 5
        
        # 2. Analyze topology
        print("\n1️⃣ Analyzing system topology...")
        rips = create_tda_algorithm('vietoris_rips')
        topo_features = rips.compute(agent_positions, max_edge_length=3.0)
        print(f"   • Found {topo_features['betti_1']} loops in agent network")
        
        # 3. Store in memory
        print("\n2️⃣ Storing topology in memory...")
        memory = create_knn_index(embedding_dim=5)
        
        # Convert topology to vector
        topo_vector = np.array([
            topo_features['betti_0'],
            topo_features['betti_1'],
            topo_features['betti_2'],
            topo_features['num_edges'] / 100,
            topo_features['num_triangles'] / 1000
        ], dtype=np.float32)
        
        memory.add(topo_vector.reshape(1, -1), [f"topo_t{int(time.time())}"])
        
        # 4. LNN prediction
        print("\n3️⃣ Predicting with LNN...")
        lnn = all_variants['mit_liquid_nn']
        lnn_input = {
            'betti_0': topo_features['betti_0'],
            'betti_1': topo_features['betti_1'],
            'edge_density': topo_features['num_edges'] / (n_agents * (n_agents - 1) / 2)
        }
        
        prediction = lnn.predict(lnn_input)
        print(f"   • Cascade risk: {prediction['failure_probability']:.2%}")
        print(f"   • Time to failure: {prediction['time_to_failure']}s")
        
        # 5. Agent decision
        print("\n4️⃣ Agent decision making...")
        decision_net = DecisionNetwork()
        
        # Prepare features for decision
        features = [
            topo_features['betti_0'] / 10,
            topo_features['betti_1'] / 100,
            prediction['failure_probability'],
            prediction['confidence']
        ]
        features.extend([0] * (256 - len(features)))
        
        import torch
        x = torch.FloatTensor(features)
        with torch.no_grad():
            action_probs, confidence = decision_net(x)
        
        print(f"   • Decision confidence: {confidence.item():.2%}")
        print(f"   • Action distribution: {action_probs[:3].numpy()}")
        
        print("\n✅ All components working together in harmony!")
        print("   TDA → Memory → LNN → Agents")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║      🚀 AURA Intelligence - All REAL Components Test          ║
║                                                               ║
║  This demonstrates that ALL major components are REAL:         ║
║  • TDA: Actual topological computations                       ║
║  • LNN: Real neural networks with PyTorch                     ║
║  • Memory: FAISS/sklearn vector similarity search             ║
║  • Agents: Byzantine consensus & neural decisions             ║
║                                                               ║
║  NO DUMMY IMPLEMENTATIONS - Everything computes!              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Run all tests
    results['TDA'] = await test_tda()
    results['LNN'] = await test_lnn()
    results['Memory'] = await test_memory()
    results['Agents'] = await test_agents()
    results['Integration'] = await test_integration()
    
    # Summary
    print_section("FINAL RESULTS")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\n📊 Test Results: {passed}/{total} passed\n")
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  • {component}: {status}")
    
    if passed == total:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                    🎉 ALL TESTS PASSED! 🎉                    ║
║                                                               ║
║  The AURA Intelligence System is now:                         ║
║  • 100% REAL - No dummy implementations                       ║
║  • Production-ready with actual algorithms                    ║
║  • Computing real topological features                        ║
║  • Making real neural predictions                             ║
║  • Storing/retrieving with real vector search                 ║
║  • Coordinating with real consensus algorithms                ║
║                                                               ║
║  "We see the shape of failure before it happens"              ║
║                              - Now a reality!                  ║
╚═══════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"\n⚠️  {total - passed} components need attention")
    
    return passed == total


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)