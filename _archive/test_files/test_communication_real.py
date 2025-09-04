#!/usr/bin/env python3
"""
Test REAL communication components - what actually works
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

# Test what we can actually import
print("🧪 Testing imports...")

try:
    from aura_intelligence.communication import (
        Performative, MessagePriority, SemanticEnvelope,
        ConversationManager, CausalGraphManager, SecureChannel
    )
    print("✅ Core imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test collective components
try:
    from aura_intelligence.communication.collective.memory_manager import CollectiveMemoryManager
    from aura_intelligence.communication.collective.supervisor import CollectiveSupervisor
    from aura_intelligence.communication.collective.context_engine import ContextEngine
    from aura_intelligence.communication.collective.graph_builder import GraphBuilder
    print("✅ Collective imports successful!")
except Exception as e:
    print(f"❌ Collective import failed: {e}")

# Test actual functionality
async def test_real_features():
    print("\n🧪 Testing real features...")
    
    # Test conversation manager
    print("\n1️⃣ Testing Conversation Manager...")
    conv_mgr = ConversationManager()
    await conv_mgr.start()
    
    conv_id = conv_mgr.start_conversation(
        protocol="fipa-request",
        initiator="test_agent",
        participants=["agent1", "agent2"]
    )
    print(f"✅ Created conversation: {conv_id}")
    
    # Test causal graph
    print("\n2️⃣ Testing Causal Graph...")
    causal = CausalGraphManager()
    
    # Create mock envelope
    env1 = type('Envelope', (), {
        'message_id': 'msg1',
        'sender': 'agent1',
        'receiver': 'agent2',
        'performative': Performative.REQUEST,
        'conversation_id': 'conv1',
        'timestamp': asyncio.get_event_loop().time(),
        'in_reply_to': None,
        'reply_with': None,
        'content': {}
    })()
    
    causal.track_message(env1)
    print(f"✅ Tracked message in causal graph")
    
    # Test security
    print("\n3️⃣ Testing Security...")
    secure = SecureChannel("test_agent", "test_tenant")
    
    # Test encryption
    data = b"secret data"
    key_id, encrypted = secure.encryption.encrypt(data)
    decrypted = secure.encryption.decrypt(key_id, encrypted)
    assert decrypted == data
    print("✅ Encryption working!")
    
    # Test data masking
    sensitive = {
        "username": "john",
        "password": "secret123",
        "safe": "public"
    }
    masked = secure.protection.mask_sensitive_data(sensitive)
    print(f"✅ Masking: {masked}")
    
    # Test collective components
    print("\n4️⃣ Testing Collective Components...")
    
    # Memory manager
    mem_config = {"use_langmem": False}
    mem_mgr = CollectiveMemoryManager(mem_config)
    print("✅ Memory manager created")
    
    # Context engine
    ctx_config = {"max_context_entries": 5}
    ctx_engine = ContextEngine(ctx_config)
    print("✅ Context engine created")
    
    # Supervisor
    supervisor = CollectiveSupervisor(mem_mgr, ctx_engine)
    print("✅ Supervisor created")
    
    # Test neural mesh capabilities
    print("\n5️⃣ Checking Neural Mesh...")
    try:
        from aura_intelligence.communication.neural_mesh import (
            NeuralMesh, NeuralNode, NodeStatus, MessageType,
            ConsciousnessAwareRouter
        )
        
        mesh = NeuralMesh()
        print("✅ Neural mesh imports work!")
        
        # Check methods
        methods = [m for m in dir(mesh) if not m.startswith('_')]
        print(f"📊 Neural mesh has {len(methods)} public methods")
        print(f"   Key methods: {methods[:5]}...")
        
    except Exception as e:
        print(f"⚠️  Neural mesh import issue: {e}")
    
    await conv_mgr.stop()
    print("\n✅ All tests complete!")

# Run tests
if __name__ == "__main__":
    asyncio.run(test_real_features())