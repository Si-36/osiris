#!/usr/bin/env python3
"""
🧪 Test Enhanced Communication System
=====================================

Tests the REAL implementation with consciousness-aware routing
"""

import asyncio
import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.communication.enhanced_neural_mesh import (
    EnhancedNeuralMesh, NeuralNode, NodeStatus, MessageType,
    MeshMessage, MessagePriority, ConsciousnessAwareRouter
)
from aura_intelligence.communication.protocols import Performative
from aura_intelligence.communication.causal_messaging import CausalGraphManager


async def test_neural_mesh():
    """Test the enhanced neural mesh"""
    print("🚀 Testing Enhanced Neural Mesh")
    print("=" * 50)
    
    # Create mesh
    mesh = EnhancedNeuralMesh(
        max_nodes=100,
        connection_threshold=0.3,
        heartbeat_interval=2.0
    )
    
    # Add causal tracking
    mesh.causal_graph = CausalGraphManager()
    
    await mesh.start()
    print("✅ Mesh started")
    
    # ==================== Test 1: Node Management ====================
    print("\n🧪 Test 1: Node Management")
    
    # Add nodes with different consciousness levels
    nodes = []
    for i in range(10):
        consciousness = 0.1 + (i * 0.1)  # 0.1 to 1.0
        node = await mesh.add_node(
            f"node_{i}",
            position=np.random.randn(3) * 5,
            consciousness_level=consciousness,
            metadata={"role": "worker" if i < 8 else "supervisor"}
        )
        nodes.append(node)
        print(f"  Added {node.id} - consciousness: {consciousness:.1f}, connections: {len(node.connections)}")
    
    # Check topology
    metrics = mesh.get_topology_metrics()
    print(f"\n📊 Topology Metrics:")
    print(f"  - Nodes: {metrics['nodes']}")
    print(f"  - Edges: {metrics['edges']}")
    print(f"  - Avg Degree: {metrics['avg_degree']:.2f}")
    print(f"  - Clustering: {metrics['clustering']:.3f}")
    
    # ==================== Test 2: Consciousness-Aware Routing ====================
    print("\n🧪 Test 2: Consciousness-Aware Routing")
    
    # Send message from low to high consciousness node
    message = MeshMessage(
        type=MessageType.DIRECT,
        sender_id="node_0",
        target_id="node_9",
        payload={"data": "important_task"},
        priority=MessagePriority.HIGH,
        performative=Performative.REQUEST
    )
    
    # Test routing
    path = await mesh.router.find_best_path("node_0", "node_9", message)
    if path:
        print(f"  Found path: {' -> '.join(path.nodes)}")
        print(f"  Path quality: {path.quality_score:.3f}")
        print(f"  Path type: {path.path_type}")
    
    # Actually send the message
    await mesh.send_message(message)
    await asyncio.sleep(0.1)  # Let it process
    
    print(f"  Messages sent: {mesh.metrics['messages_sent']}")
    print(f"  Messages delivered: {mesh.metrics['messages_delivered']}")
    
    # ==================== Test 3: Broadcast & Consensus ====================
    print("\n🧪 Test 3: Broadcast & Consensus")
    
    # Broadcast from high consciousness node
    broadcast_count = await mesh.broadcast(
        sender_id="node_9",
        payload={"announcement": "system_update"},
        priority=MessagePriority.HIGH
    )
    print(f"  Broadcast to {broadcast_count} nodes")
    
    # Test consensus
    print("\n  Testing consensus mechanism...")
    result = await mesh.initiate_consensus(
        initiator_id="node_8",
        topic="upgrade_decision",
        proposal={"version": "2.0", "features": ["neural_upgrade", "quantum_routing"]},
        timeout=2.0
    )
    
    print(f"  Consensus result:")
    print(f"    - Reached: {result['consensus'] is not None}")
    print(f"    - Support: {result.get('support', 0):.1%}")
    print(f"    - Votes: {result['votes']}/{result['total']}")
    
    # ==================== Test 4: Self-Healing ====================
    print("\n🧪 Test 4: Self-Healing & Fault Tolerance")
    
    # Simulate node failure
    failed_node = mesh.nodes["node_3"]
    failed_node.status = NodeStatus.FAILED
    failed_node.error_rate = 0.9
    print(f"  Simulated failure: {failed_node.id}")
    
    # Trigger health monitor
    await mesh._health_monitor()
    
    # Try routing around failed node
    message2 = MeshMessage(
        type=MessageType.DIRECT,
        sender_id="node_2",
        target_id="node_4",
        payload={"data": "reroute_test"},
        priority=MessagePriority.NORMAL
    )
    
    path2 = await mesh.router.find_best_path("node_2", "node_4", message2)
    if path2:
        print(f"  Rerouted path: {' -> '.join(path2.nodes)}")
        print(f"  Avoided failed node: {'node_3' not in path2.nodes}")
    
    # ==================== Test 5: Consciousness Evolution ====================
    print("\n🧪 Test 5: Consciousness Evolution")
    
    # Simulate activity
    active_nodes = ["node_5", "node_6", "node_7"]
    for _ in range(50):
        for node_id in active_nodes:
            node = mesh.nodes[node_id]
            node.messages_processed += 1
    
    # Update consciousness
    await mesh._consciousness_updater()
    
    print("  Consciousness levels after activity:")
    for node_id in active_nodes:
        node = mesh.nodes[node_id]
        print(f"    {node_id}: {node.consciousness_level:.3f}")
    
    # ==================== Test 6: Performance Metrics ====================
    print("\n🧪 Test 6: Performance Benchmark")
    
    start_time = time.time()
    message_count = 100
    
    # Send burst of messages
    tasks = []
    for i in range(message_count):
        sender = f"node_{i % 10}"
        target = f"node_{(i + 5) % 10}"
        
        msg = MeshMessage(
            type=MessageType.DIRECT,
            sender_id=sender,
            target_id=target,
            payload={"index": i},
            priority=MessagePriority.NORMAL
        )
        tasks.append(mesh.send_message(msg))
    
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.5)  # Let them process
    
    duration = time.time() - start_time
    throughput = message_count / duration
    
    print(f"  Throughput: {throughput:.0f} msg/sec")
    print(f"  Delivery rate: {mesh.metrics['messages_delivered'] / mesh.metrics['messages_sent']:.1%}")
    
    # ==================== Test 7: Causal Tracking ====================
    print("\n🧪 Test 7: Causal Message Tracking")
    
    if mesh.causal_graph:
        chains = mesh.causal_graph.detect_causal_chains()
        patterns = mesh.causal_graph.detect_patterns()
        
        print(f"  Causal chains detected: {len(chains)}")
        print(f"  Patterns found: {list(patterns.keys())}")
        
        graph_metrics = mesh.causal_graph.get_metrics()
        print(f"  Graph nodes: {graph_metrics['graph_nodes']}")
        print(f"  Causal edges: {graph_metrics['causal_edges']}")
    
    # ==================== Final Metrics ====================
    print("\n📊 Final Mesh Metrics:")
    final_metrics = mesh.get_metrics()
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Visualization data
    viz_data = mesh.visualize_topology()
    print(f"\n🎨 Visualization ready:")
    print(f"  - {len(viz_data['nodes'])} nodes")
    print(f"  - {len(viz_data['edges'])} edges")
    
    # Stop mesh
    await mesh.stop()
    print("\n✅ All tests completed successfully!")


async def main():
    """Run all tests"""
    try:
        await test_neural_mesh()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())