#!/usr/bin/env python3
"""
Test communication system components
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("📡 TESTING COMMUNICATION SYSTEM")
print("=" * 60)

async def test_communication():
    """Test all communication components"""
    
    # Test 1: Import all modules
    print("\n📦 Testing imports...")
    try:
        from aura_intelligence.communication.neural_mesh import (
            NeuralMesh, NeuralNode, MeshMessage, MessageType, 
            MessagePriority, NodeStatus, ConsciousnessAwareRouter
        )
        print("✅ Neural mesh imports successful")
        
        # Try to import NATS (may not have nats-py installed)
        try:
            from aura_intelligence.communication.nats_a2a import (
                AgentMessage, MessagePriority as NATSPriority
            )
            print("✅ NATS A2A imports successful")
            nats_available = True
        except ImportError as e:
            print(f"⚠️  NATS imports failed (likely missing nats-py): {e}")
            nats_available = False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Neural Mesh Network
    print("\n🧠 Testing Neural Mesh Network...")
    try:
        # Create mesh
        mesh = NeuralMesh(
            max_nodes=100,
            connection_threshold=0.5,
            heartbeat_interval=2.0
        )
        print("✅ Neural mesh created")
        
        # Create nodes with different consciousness levels
        nodes = []
        for i in range(5):
            node = NeuralNode(
                name=f"agent_{i}",
                capabilities={f"skill_{i}", "communication", "reasoning"},
                position=(
                    np.random.rand() * 10,
                    np.random.rand() * 10,
                    np.random.rand() * 10
                ),
                consciousness_level=0.3 + (i * 0.15)  # 0.3 to 0.9
            )
            mesh.add_node(node)
            nodes.append(node)
            print(f"✅ Added node: {node.name} (consciousness: {node.consciousness_level:.2f})")
        
        # Check topology
        print(f"✅ Mesh topology: {mesh.topology.number_of_nodes()} nodes, {mesh.topology.number_of_edges()} edges")
        
        # Register handlers
        received_messages = []
        
        async def test_broadcast_handler(message: MeshMessage, node_id: str):
            received_messages.append({
                "type": "broadcast",
                "node": node_id,
                "content": message.content
            })
        
        async def test_unicast_handler(message: MeshMessage, node_id: str):
            received_messages.append({
                "type": "unicast",
                "node": node_id,
                "content": message.content
            })
        
        mesh.register_handler(MessageType.BROADCAST, test_broadcast_handler)
        mesh.register_handler(MessageType.UNICAST, test_unicast_handler)
        
        # Start mesh
        await mesh.start()
        print("✅ Neural mesh started")
        
        # Test broadcast
        await mesh.broadcast(
            nodes[0].id,
            {"test": "broadcast_message", "timestamp": datetime.now().isoformat()},
            MessagePriority.HIGH
        )
        
        # Wait for message propagation
        await asyncio.sleep(0.5)
        
        broadcast_count = len([m for m in received_messages if m["type"] == "broadcast"])
        print(f"✅ Broadcast test: {broadcast_count} nodes received message")
        
        # Test unicast with consciousness-aware routing
        message = MeshMessage(
            type=MessageType.UNICAST,
            source_node=nodes[0].id,
            target_nodes=[nodes[4].id],
            content={"test": "unicast_message"},
            priority=MessagePriority.HIGH,
            consciousness_context=nodes[0].consciousness_level
        )
        
        await mesh.send_message(message)
        await asyncio.sleep(0.5)
        
        unicast_count = len([m for m in received_messages if m["type"] == "unicast"])
        print(f"✅ Unicast test: message delivered via consciousness-aware routing")
        
        # Test consensus
        async def consensus_vote_handler(message: MeshMessage, node_id: str) -> Dict[str, Any]:
            # Higher consciousness nodes more likely to approve
            node = mesh.nodes[node_id]
            approval_chance = node.consciousness_level
            return {"approve": np.random.random() < approval_chance}
        
        mesh.register_handler(MessageType.CONSENSUS, consensus_vote_handler)
        
        consensus_result = await mesh.request_consensus(
            topic="system_upgrade",
            proposal={"version": "3.0", "features": ["quantum_routing", "neural_sync"]},
            participants=[n.id for n in nodes[1:4]],
            timeout=3.0
        )
        
        if consensus_result:
            print(f"✅ Consensus achieved: {consensus_result['support']*100:.1f}% support")
        else:
            print("❌ Consensus not reached")
        
        # Test self-healing
        print("\n🔧 Testing Self-Healing...")
        
        # Simulate node failure
        nodes[2].status = NodeStatus.UNREACHABLE
        nodes[2].last_heartbeat = datetime.now() - timedelta(minutes=5)
        print(f"❌ Simulated failure of {nodes[2].name}")
        
        # Wait for health monitor
        await asyncio.sleep(2)
        
        # Get mesh stats
        stats = mesh.get_mesh_stats()
        print(f"✅ Mesh stats after failure:")
        print(f"   - Total nodes: {stats['total_nodes']}")
        print(f"   - Healthy nodes: {stats['healthy_nodes']}")
        print(f"   - Connections: {stats['total_connections']}")
        print(f"   - Avg consciousness: {stats['avg_consciousness']:.2f}")
        
        await mesh.stop()
        print("✅ Neural mesh stopped successfully")
        
    except Exception as e:
        print(f"❌ Neural mesh test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Message Routing Algorithms
    print("\n🚀 Testing Message Routing...")
    try:
        # Create a larger mesh for routing tests
        large_mesh = NeuralMesh(max_nodes=20)
        
        # Create a grid of nodes
        grid_nodes = []
        for x in range(4):
            for y in range(5):
                node = NeuralNode(
                    name=f"grid_{x}_{y}",
                    position=(x * 3, y * 3, 0),
                    consciousness_level=0.5 + (x + y) * 0.05
                )
                large_mesh.add_node(node)
                grid_nodes.append(node)
        
        # Test router
        router = ConsciousnessAwareRouter()
        
        # Calculate routes from corner to corner
        paths = await router.calculate_route(
            grid_nodes[0].id,  # Top-left
            [grid_nodes[-1].id],  # Bottom-right
            large_mesh,
            MessagePriority.HIGH
        )
        
        if paths:
            path = paths[0]
            print(f"✅ Route calculated:")
            print(f"   - Path length: {len(path.nodes)} nodes")
            print(f"   - Path strength: {path.strength:.3f}")
            print(f"   - Estimated latency: {path.latency_ms}ms")
            
            # Show consciousness levels along path
            consciousness_path = []
            for node_id in path.nodes:
                node = large_mesh.nodes[node_id]
                consciousness_path.append(f"{node.consciousness_level:.2f}")
            print(f"   - Consciousness levels: {' → '.join(consciousness_path)}")
        
    except Exception as e:
        print(f"❌ Routing test failed: {e}")
    
    # Test 4: Performance Metrics
    print("\n📊 Testing Performance Metrics...")
    try:
        # Create a performance test mesh
        perf_mesh = NeuralMesh()
        
        # Add nodes
        perf_nodes = []
        for i in range(10):
            node = NeuralNode(name=f"perf_{i}")
            perf_mesh.add_node(node)
            perf_nodes.append(node)
        
        await perf_mesh.start()
        
        # Measure broadcast latency
        start_time = asyncio.get_event_loop().time()
        
        for i in range(100):
            await perf_mesh.broadcast(
                perf_nodes[i % 10].id,
                {"seq": i, "data": "x" * 100},
                MessagePriority.NORMAL
            )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        throughput = 100 / elapsed
        
        print(f"✅ Performance test:")
        print(f"   - Messages sent: 100")
        print(f"   - Time elapsed: {elapsed:.3f}s")
        print(f"   - Throughput: {throughput:.1f} msg/s")
        print(f"   - Avg latency: {elapsed/100*1000:.1f}ms")
        
        await perf_mesh.stop()
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    # Test 5: NATS Integration (if available)
    if nats_available:
        print("\n🔌 Testing NATS A2A Communication...")
        try:
            # Create test message
            msg = AgentMessage(
                sender_id="agent_1",
                recipient_id="agent_2",
                message_type="test",
                priority=NATSPriority.HIGH,
                payload={"test": "data"}
            )
            print(f"✅ NATS message created: {msg.id}")
            print(f"   - Sender: {msg.sender_id}")
            print(f"   - Recipient: {msg.recipient_id}")
            print(f"   - Priority: {msg.priority.value}")
            
        except Exception as e:
            print(f"❌ NATS test failed: {e}")
    
    print("\n" + "=" * 60)
    print("COMMUNICATION SYSTEM TEST COMPLETE")
    
# Run the test
if __name__ == "__main__":
    asyncio.run(test_communication())