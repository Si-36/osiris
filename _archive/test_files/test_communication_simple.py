#!/usr/bin/env python3
"""
🧪 Simple Communication System Test (No External Dependencies)
==============================================================

Tests core communication features without NATS/external deps.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

# Import only the parts that don't require external dependencies
from aura_intelligence.communication.semantic_protocols import (
    Performative, InteractionProtocol, ConversationManager,
    ProtocolValidator, ProtocolTemplates, AURA_CORE_ONTOLOGY
)
from aura_intelligence.communication.causal_messaging import (
    CausalGraphManager, CausalAnalyzer, CausalEdge, CausalChain
)
from aura_intelligence.communication.secure_channels import (
    SecureChannel, SecurityConfig, EncryptionManager,
    AuthenticationManager, SubjectAuthorization, DataProtection,
    MessageChunker, SubjectBuilder
)


def test_performatives():
    """Test FIPA ACL performatives"""
    print("\n🧪 Testing Performatives...")
    
    # Test all performative types
    assert Performative.REQUEST.value == "request"
    assert Performative.INFORM.value == "inform"
    assert Performative.PROPOSE.value == "propose"
    
    print(f"✅ Found {len(Performative)} performatives")
    

def test_protocol_validation():
    """Test protocol sequence validation"""
    print("\n🧪 Testing Protocol Validation...")
    
    # Test REQUEST protocol
    valid = ProtocolValidator.validate_sequence(
        InteractionProtocol.REQUEST,
        "init",
        Performative.REQUEST
    )
    assert valid
    
    valid = ProtocolValidator.validate_sequence(
        InteractionProtocol.REQUEST,
        "REQUEST",
        Performative.AGREE
    )
    assert valid
    
    # Test invalid sequence
    valid = ProtocolValidator.validate_sequence(
        InteractionProtocol.REQUEST,
        "REQUEST",
        Performative.CFP  # CFP not valid after REQUEST
    )
    assert not valid
    
    # Test final states
    is_final = ProtocolValidator.is_final_state(
        InteractionProtocol.REQUEST,
        Performative.INFORM
    )
    assert is_final
    
    print("✅ Protocol validation passed!")


async def test_conversation_management():
    """Test conversation state management"""
    print("\n🧪 Testing Conversation Management...")
    
    manager = ConversationManager()
    await manager.start()
    
    # Start conversation
    conv_id = manager.start_conversation(
        protocol=InteractionProtocol.CONTRACT_NET,
        initiator="buyer",
        participants=["seller1", "seller2"],
        timeout=60.0
    )
    
    print(f"📋 Started conversation: {conv_id}")
    
    # Get state
    state = manager.get_conversation_state(conv_id)
    assert state is not None
    assert state.protocol == InteractionProtocol.CONTRACT_NET
    assert state.current_step == "init"
    assert not state.completed
    
    # Update conversation
    success = manager.update_conversation(conv_id, Performative.CFP, "buyer")
    assert success
    
    state = manager.get_conversation_state(conv_id)
    assert state.current_step == "CFP"
    
    await manager.stop()
    print("✅ Conversation management passed!")


def test_causal_graph():
    """Test causal message tracking"""
    print("\n🧪 Testing Causal Graph...")
    
    graph = CausalGraphManager(max_history=100)
    
    # Create mock envelopes
    class MockEnvelope:
        def __init__(self, msg_id, sender, performative, in_reply_to=None):
            self.message_id = msg_id
            self.sender = sender
            self.receiver = "receiver"
            self.performative = performative
            self.in_reply_to = in_reply_to
            self.conversation_id = "conv123"
            self.timestamp = asyncio.get_event_loop().time()
            self.reply_with = None
            self.content = {}
    
    # Create message chain
    msg1 = MockEnvelope("msg1", "agent1", Performative.REQUEST)
    msg2 = MockEnvelope("msg2", "agent2", Performative.AGREE, in_reply_to="msg1")
    msg3 = MockEnvelope("msg3", "agent2", Performative.INFORM, in_reply_to="msg1")
    
    # Track messages
    graph.track_message(msg1)
    graph.track_message(msg2)
    graph.track_message(msg3)
    
    # Check graph
    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.has_edge("msg1", "msg2")
    assert graph.graph.has_edge("msg1", "msg3")
    
    # Find root cause
    root = CausalAnalyzer.find_root_cause(graph, "msg3")
    assert root == "msg1"
    
    # Check metrics
    metrics = graph.get_metrics()
    assert metrics["total_messages"] == 3
    assert metrics["causal_edges"] >= 2
    
    print("✅ Causal graph passed!")


def test_encryption():
    """Test encryption functionality"""
    print("\n🧪 Testing Encryption...")
    
    config = SecurityConfig(enable_encryption=True)
    encryption = EncryptionManager(config)
    
    # Test encryption/decryption
    data = b"Secret message"
    key_id, encrypted = encryption.encrypt(data)
    
    assert key_id != "none"
    assert encrypted != data
    
    decrypted = encryption.decrypt(key_id, encrypted)
    assert decrypted == data
    
    print("✅ Encryption passed!")


def test_authentication():
    """Test authentication system"""
    print("\n🧪 Testing Authentication...")
    
    config = SecurityConfig(enable_auth=True)
    auth = AuthenticationManager(config)
    
    # Register agent
    creds = auth.register_agent("agent1", "tenant1", {"pub", "sub"})
    assert creds.agent_id == "agent1"
    assert creds.tenant_id == "tenant1"
    assert "pub" in creds.permissions
    
    # Test authentication
    valid = auth.authenticate("agent1", creds.nkey_seed)
    assert valid
    
    invalid = auth.authenticate("agent1", "wrong_credential")
    assert not invalid
    
    print("✅ Authentication passed!")


def test_subject_authorization():
    """Test subject-based authorization"""
    print("\n🧪 Testing Subject Authorization...")
    
    config = SecurityConfig(enable_subject_acl=True)
    authz = SubjectAuthorization(config)
    
    # Add custom rule
    authz.add_rule("tenant1", "aura.tenant1.custom.>", {"pub", "sub"})
    
    # Test permissions
    allowed = authz.check_permission(
        "tenant1", "agent1", "aura.tenant1.custom.test", "pub"
    )
    assert allowed
    
    # Test tenant isolation
    not_allowed = authz.check_permission(
        "tenant2", "agent1", "aura.tenant1.custom.test", "pub"
    )
    assert not not_allowed
    
    print("✅ Subject authorization passed!")


def test_data_protection():
    """Test data masking and protection"""
    print("\n🧪 Testing Data Protection...")
    
    config = SecurityConfig(enable_masking=True)
    protection = DataProtection(config)
    
    # Test masking
    data = {
        "username": "john_doe",
        "password": "super_secret_password",
        "token": "abc123xyz789",
        "safe_data": "This is safe",
        "nested": {
            "secret": "hidden_value"
        }
    }
    
    masked = protection.mask_sensitive_data(data)
    
    assert masked["password"] != data["password"]
    assert masked["password"].endswith("word")  # Last 4 chars
    assert masked["token"].endswith("z789")
    assert masked["safe_data"] == data["safe_data"]
    assert masked["nested"]["secret"] == "***REDACTED***"
    
    # Test audit redaction
    text = "My SSN is 123-45-6789 and email is john@example.com"
    redacted = protection.redact_for_audit(text)
    assert "***-**-****" in redacted
    assert "jo***@example.com" in redacted
    
    print("✅ Data protection passed!")


def test_message_chunking():
    """Test message chunking for large payloads"""
    print("\n🧪 Testing Message Chunking...")
    
    config = SecurityConfig(
        max_message_size=100,
        enable_chunking=True,
        chunk_size=50
    )
    chunker = MessageChunker(config)
    
    # Create large message
    large_data = b"x" * 150
    
    # Chunk it
    chunks = chunker.chunk_message("msg123", large_data)
    assert len(chunks) == 3
    assert chunks[0].total_chunks == 3
    assert chunks[0].chunk_index == 0
    
    # Reassemble
    reassembled = chunker.reassemble_message(chunks)
    assert reassembled == large_data
    
    print("✅ Message chunking passed!")


def test_subject_builders():
    """Test subject builder utilities"""
    print("\n🧪 Testing Subject Builders...")
    
    # Test various subject patterns
    subject = SubjectBuilder.agent_direct("tenant1", "high", "agent123")
    assert subject == "aura.tenant1.a2a.high.agent123"
    
    subject = SubjectBuilder.broadcast("tenant1", "alerts")
    assert subject == "aura.tenant1.broadcast.alerts"
    
    subject = SubjectBuilder.swarm("tenant1", "swarm1", "sync")
    assert subject == "aura.tenant1.swarm.swarm1.sync"
    
    subject = SubjectBuilder.kv_bucket("tenant1", "state")
    assert subject == "AURA_STATE_TENANT1"
    
    print("✅ Subject builders passed!")


def test_ontology():
    """Test AURA ontology definitions"""
    print("\n🧪 Testing Ontology...")
    
    # Test ontology validation
    valid_task = {
        "id": "task123",
        "type": "analysis",
        "priority": 1,
        "deadline": "2025-01-01T00:00:00",
        "requirements": {"memory": "2GB"}
    }
    
    valid = AURA_CORE_ONTOLOGY.validate_content(valid_task, "task")
    assert valid
    
    # Test invalid content
    invalid_task = {
        "id": "task123"
        # Missing required fields
    }
    
    valid = AURA_CORE_ONTOLOGY.validate_content(invalid_task, "task")
    assert not valid
    
    print("✅ Ontology validation passed!")


async def main():
    """Run all tests"""
    print("🚀 AURA Communication System Test (No External Deps)")
    print("=" * 50)
    
    try:
        # Synchronous tests
        test_performatives()
        test_protocol_validation()
        test_causal_graph()
        test_encryption()
        test_authentication()
        test_subject_authorization()
        test_data_protection()
        test_message_chunking()
        test_subject_builders()
        test_ontology()
        
        # Async tests
        await test_conversation_management()
        
        print("\n✅ ALL TESTS PASSED! 🎉")
        print("\nCore communication features are working correctly!")
        print("\n📝 Next steps:")
        print("1. Fix remaining pass statements in neural_mesh.py")
        print("2. Add NATS mock or make it optional")
        print("3. Integrate with other AURA components")
        print("4. Add production deployment configs")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())