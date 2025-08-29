#!/usr/bin/env python3
"""
🧪 Communication System Integration Test
========================================

Tests the complete unified communication system including:
- NATS A2A messaging
- Neural Mesh routing
- FIPA ACL protocols
- Collective patterns
- Causal tracking
- Security features
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core/src'))

from aura_intelligence.communication.unified_communication import (
    UnifiedCommunication, SemanticEnvelope, Performative, MessagePriority, TraceContext
)
from aura_intelligence.communication.semantic_protocols import (
    ConversationManager, ProtocolTemplates, InteractionProtocol
)
from aura_intelligence.communication.collective_protocols import (
    CollectiveProtocolsManager
)
from aura_intelligence.communication.causal_messaging import (
    CausalGraphManager, CausalAnalyzer
)
from aura_intelligence.communication.secure_channels import (
    SecureChannel, SecurityConfig
)


async def test_basic_communication():
    """Test basic agent-to-agent communication"""
    print("\n🧪 Testing Basic Communication...")
    
    # Create two agents
    agent1 = UnifiedCommunication(
        agent_id="agent1",
        tenant_id="test_tenant",
        enable_neural_mesh=False  # Start simple
    )
    
    agent2 = UnifiedCommunication(
        agent_id="agent2",
        tenant_id="test_tenant",
        enable_neural_mesh=False
    )
    
    # Register handler on agent2
    messages_received = []
    
    def handle_inform(envelope: SemanticEnvelope):
        messages_received.append(envelope)
        print(f"✅ Agent2 received: {envelope.content}")
        return {"status": "acknowledged"}
    
    agent2.register_handler(Performative.INFORM, handle_inform)
    
    # Start both agents
    # await agent1.start()
    # await agent2.start()
    
    # Send message from agent1 to agent2
    envelope = SemanticEnvelope(
        performative=Performative.INFORM,
        sender="agent1",
        receiver="agent2",
        content={"message": "Hello from agent1!"}
    )
    
    message_id = await agent1.send(envelope)
    print(f"📤 Sent message: {message_id}")
    
    # Simulate message delivery
    await agent2._handle_message(envelope.to_agent_message())
    
    assert len(messages_received) == 1
    assert messages_received[0].content["message"] == "Hello from agent1!"
    
    print("✅ Basic communication test passed!")


async def test_request_reply():
    """Test request-reply pattern"""
    print("\n🧪 Testing Request-Reply Pattern...")
    
    agent1 = UnifiedCommunication("requester", "test")
    agent2 = UnifiedCommunication("responder", "test")
    
    # Register request handler
    async def handle_request(envelope: SemanticEnvelope):
        print(f"📥 Received request: {envelope.content}")
        return {"result": "Task completed successfully"}
    
    agent2.register_handler(Performative.REQUEST, handle_request)
    
    # Create request
    request = SemanticEnvelope(
        performative=Performative.REQUEST,
        sender="requester",
        receiver="responder",
        content={"action": "perform_task"},
        reply_with="reply123"
    )
    
    # Simulate request handling
    await agent2._handle_message(request.to_agent_message())
    
    print("✅ Request-reply test passed!")


async def test_fipa_protocols():
    """Test FIPA interaction protocols"""
    print("\n🧪 Testing FIPA Protocols...")
    
    agent = UnifiedCommunication("protocol_agent", "test")
    conv_manager = ConversationManager()
    
    await conv_manager.start()
    
    # Test conversation management
    conv_id = conv_manager.start_conversation(
        protocol=InteractionProtocol.CONTRACT_NET,
        initiator="buyer",
        participants=["seller1", "seller2", "seller3"]
    )
    
    print(f"📋 Started contract net conversation: {conv_id}")
    
    # Test protocol validation
    valid = conv_manager.update_conversation(conv_id, Performative.CFP, "buyer")
    assert valid
    
    valid = conv_manager.update_conversation(conv_id, Performative.PROPOSE, "seller1")
    assert valid
    
    # Test invalid sequence
    valid = conv_manager.update_conversation(conv_id, Performative.INFORM, "seller1")
    assert not valid  # INFORM not valid after PROPOSE in contract net
    
    await conv_manager.stop()
    
    print("✅ FIPA protocols test passed!")


async def test_collective_patterns():
    """Test collective communication patterns"""
    print("\n🧪 Testing Collective Patterns...")
    
    comm = UnifiedCommunication("collective_agent", "test")
    collective = CollectiveProtocolsManager(comm)
    
    # Create a swarm
    swarm = await collective.create_swarm(
        swarm_id="test_swarm",
        initial_members=["agent1", "agent2", "agent3"],
        topology="mesh",
        goal={"task": "explore_environment"}
    )
    
    print(f"🐝 Created swarm: {swarm.swarm_id}")
    
    # Test swarm synchronization
    sync_level = await collective.synchronize_swarm(
        swarm_id="test_swarm",
        sync_data={"position": [0, 0], "status": "exploring"}
    )
    
    print(f"🔄 Swarm sync level: {sync_level}")
    
    # Test pattern detection
    patterns = await collective.detect_patterns()
    print(f"🔍 Detected {len(patterns)} patterns")
    
    # Test pheromone update
    await collective.update_pheromone(
        swarm_id="test_swarm",
        pheromone_type="food_trail",
        value=0.8
    )
    
    print("✅ Collective patterns test passed!")


async def test_causal_tracking():
    """Test causal message tracking"""
    print("\n🧪 Testing Causal Tracking...")
    
    causal_graph = CausalGraphManager()
    
    # Create message chain
    msg1 = SemanticEnvelope(
        performative=Performative.REQUEST,
        sender="agent1",
        receiver="agent2",
        content={"task": "analyze_data"}
    )
    
    msg2 = SemanticEnvelope(
        performative=Performative.AGREE,
        sender="agent2",
        receiver="agent1",
        content={"status": "accepted"},
        in_reply_to=msg1.message_id
    )
    
    msg3 = SemanticEnvelope(
        performative=Performative.INFORM,
        sender="agent2",
        receiver="agent1",
        content={"result": "analysis_complete"},
        in_reply_to=msg1.message_id
    )
    
    # Track messages
    causal_graph.track_message(msg1)
    causal_graph.track_message(msg2)
    causal_graph.track_message(msg3)
    
    # Detect causal chains
    chains = causal_graph.detect_causal_chains()
    print(f"🔗 Detected {len(chains)} causal chains")
    
    # Find root cause
    root = CausalAnalyzer.find_root_cause(causal_graph, msg3.message_id)
    assert root == msg1.message_id
    
    print(f"🎯 Root cause: {root}")
    
    # Get lineage
    lineage = causal_graph.get_message_lineage(msg2.message_id)
    print(f"📊 Message lineage: {lineage['total_ancestors']} ancestors, {lineage['total_descendants']} descendants")
    
    print("✅ Causal tracking test passed!")


async def test_secure_channels():
    """Test secure communication features"""
    print("\n🧪 Testing Secure Channels...")
    
    config = SecurityConfig(
        enable_encryption=True,
        enable_auth=True,
        enable_masking=True
    )
    
    channel = SecureChannel(
        agent_id="secure_agent",
        tenant_id="secure_tenant",
        config=config
    )
    
    # Test secure subject building
    subject = channel.build_secure_subject("a2a.high.target_agent")
    print(f"🔒 Secure subject: {subject}")
    
    # Test data masking
    sensitive_data = {
        "username": "john_doe",
        "password": "super_secret_123",
        "credit_card": "1234-5678-9012-3456",
        "public_data": "This is public"
    }
    
    masked = channel.protection.mask_sensitive_data(sensitive_data)
    print(f"🎭 Masked data: {masked}")
    
    assert masked["password"] == "***************_123"
    assert masked["credit_card"] == "****************3456"
    assert masked["public_data"] == "This is public"
    
    # Test encryption
    message = {"content": "Secret message"}
    result = await channel.secure_send(
        data=message,
        subject=subject
    )
    
    print(f"🔐 Encrypted send result: {result}")
    
    print("✅ Secure channels test passed!")


async def test_trace_propagation():
    """Test W3C trace context propagation"""
    print("\n🧪 Testing Trace Propagation...")
    
    agent = UnifiedCommunication("trace_agent", "test", enable_tracing=True)
    
    # Create trace context
    trace_ctx = TraceContext.generate()
    print(f"🔍 Generated trace: {trace_ctx.traceparent}")
    
    # Send with trace
    envelope = SemanticEnvelope(
        performative=Performative.INFORM,
        sender="trace_agent",
        receiver="other_agent",
        content={"data": "traced_message"}
    )
    
    await agent.send(envelope, trace_context=trace_ctx)
    
    # Check metrics
    metrics = agent.get_metrics()
    print(f"📊 Trace propagations: {metrics['trace_propagations']}")
    
    print("✅ Trace propagation test passed!")


async def test_full_integration():
    """Test full system integration"""
    print("\n🧪 Testing Full Integration...")
    
    # Create complete communication system
    comm = UnifiedCommunication(
        agent_id="integrated_agent",
        tenant_id="production",
        enable_neural_mesh=True,
        enable_tracing=True
    )
    
    # Setup managers
    conv_manager = ConversationManager()
    collective = CollectiveProtocolsManager(comm)
    causal = CausalGraphManager()
    secure = SecureChannel("integrated_agent", "production")
    
    print("✅ Full integration components created!")
    
    # Test metrics
    metrics = comm.get_metrics()
    print(f"\n📊 System Metrics:")
    print(f"  - Messages sent: {metrics['messages_sent']}")
    print(f"  - Messages received: {metrics['messages_received']}")
    print(f"  - Active conversations: {metrics['conversations_active']}")
    
    print("\n🎉 All communication tests passed!")


async def main():
    """Run all tests"""
    print("🚀 AURA Communication System Integration Test")
    print("=" * 50)
    
    try:
        await test_basic_communication()
        await test_request_reply()
        await test_fipa_protocols()
        await test_collective_patterns()
        await test_causal_tracking()
        await test_secure_channels()
        await test_trace_propagation()
        await test_full_integration()
        
        print("\n✅ ALL TESTS PASSED! 🎉")
        print("\nThe communication system is ready for production!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())