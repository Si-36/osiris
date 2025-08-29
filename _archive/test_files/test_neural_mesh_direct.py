#!/usr/bin/env python3
"""
Test Neural Mesh directly without circular imports
"""

import asyncio
import sys
import os
import time

# Add path but import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src/aura_intelligence/communication'))

# Import enhanced neural mesh directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enhanced_neural_mesh",
    "core/src/aura_intelligence/communication/enhanced_neural_mesh.py"
)
enhanced_neural_mesh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_neural_mesh)

# Import protocols directly
spec2 = importlib.util.spec_from_file_location(
    "protocols",
    "core/src/aura_intelligence/communication/protocols.py"
)
protocols = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(protocols)

# Now we can use the classes
EnhancedNeuralMesh = enhanced_neural_mesh.EnhancedNeuralMesh
MessagePriority = protocols.MessagePriority
Performative = protocols.Performative


async def test_consciousness_routing():
    """Test consciousness-aware routing"""
    print("🧠 Testing Consciousness-Aware Neural Mesh")
    print("=" * 50)
    
    mesh = EnhancedNeuralMesh(
        max_nodes=20,
        connection_threshold=0.3
    )
    
    await mesh.start()
    
    # Create nodes with varying consciousness
    print("\n📊 Creating consciousness gradient...")
    for i in range(10):
        await mesh.add_node(
            f"agent_{i}",
            consciousness_level=i * 0.1 + 0.1
        )
    
    # Show connections
    print("\n🔗 Node connections:")
    for node_id, node in mesh.nodes.items():
        print(f"  {node_id} (consciousness={node.consciousness_level:.1f}): {len(node.connections)} connections")
    
    # Test routing
    print("\n🚀 Testing consciousness-aware routing...")
    
    msg = enhanced_neural_mesh.MeshMessage(
        type=enhanced_neural_mesh.MessageType.DIRECT,
        sender_id="agent_0",
        target_id="agent_9",
        payload={"mission": "critical"},
        priority=MessagePriority.HIGH
    )
    
    path = await mesh.router.find_best_path("agent_0", "agent_9", msg)
    
    if path:
        print(f"  Path found: {' -> '.join(path.nodes)}")
        print(f"  Path type: {path.path_type}")
        print(f"  Quality score: {path.quality_score:.3f}")
        
        # Show consciousness along path
        print("  Consciousness levels along path:")
        for node_id in path.nodes:
            node = mesh.nodes[node_id]
            print(f"    {node_id}: {node.consciousness_level:.2f}")
    
    # Test consensus
    print("\n🤝 Testing consciousness-weighted consensus...")
    
    result = await mesh.initiate_consensus(
        initiator_id="agent_5",
        topic="system_upgrade",
        proposal={"upgrade": True, "version": "2.0"},
        timeout=1.0
    )
    
    print(f"  Consensus: {result['consensus']}")
    print(f"  Support: {result['support']:.1%}")
    
    # Show metrics
    print("\n📈 Mesh Metrics:")
    metrics = mesh.get_metrics()
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    await mesh.stop()
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(test_consciousness_routing())