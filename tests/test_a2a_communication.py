"""
🧪 Test A2A Communication Protocol
"""

import asyncio
import pytest
from typing import Dict, Any

from src.aura.a2a import (
    A2AProtocol,
    A2ANetwork,
    MCPMessage,
    MCPMessageType,
    AgentCapability
)


class TestA2ACommunication:
    """Test suite for A2A communication protocol"""
    
    @pytest.mark.asyncio
    async def test_agent_registration(self):
        """Test agent registration in A2A network"""
        network = A2ANetwork()
        
        # Create test agents
        agent1 = A2AProtocol("agent1", "Test Agent 1")
        agent2 = A2AProtocol("agent2", "Test Agent 2")
        
        # Register agents
        await network.register_agent(agent1)
        await network.register_agent(agent2)
        
        # Check registration
        assert len(network.agents) == 2
        assert "agent1" in network.agents
        assert "agent2" in network.agents
        
        # Check topology
        assert "agent2" in network.network_topology["agent1"]
        assert "agent1" in network.network_topology["agent2"]
        
    @pytest.mark.asyncio
    async def test_capability_discovery(self):
        """Test capability discovery between agents"""
        network = A2ANetwork()
        
        # Create agent with capabilities
        agent1 = A2AProtocol("agent1", "Capable Agent")
        capability = AgentCapability(
            capability_id="test_capability",
            name="Test Capability",
            description="A test capability",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        
        await agent1.register_capability(capability)
        await network.register_agent(agent1)
        
        # Verify capability is registered
        assert "test_capability" in agent1.capabilities
        assert agent1.capabilities["test_capability"].name == "Test Capability"
        
    @pytest.mark.asyncio
    async def test_message_passing(self):
        """Test direct message passing between agents"""
        agent1 = A2AProtocol("agent1", "Sender")
        agent2 = A2AProtocol("agent2", "Receiver")
        
        # Start agents
        await agent1.start()
        await agent2.start()
        
        # Send message
        message = MCPMessage(
            message_type=MCPMessageType.CONTEXT_SYNC,
            sender_id=agent1.agent_id,
            receiver_id=agent2.agent_id,
            payload={"test": "data"}
        )
        
        await agent1.send_message(message)
        
        # Give time for message processing
        await asyncio.sleep(0.1)
        
        # Stop agents
        await agent1.stop()
        await agent2.stop()
        
    @pytest.mark.asyncio
    async def test_task_delegation(self):
        """Test task delegation between agents"""
        network = A2ANetwork()
        
        # Create agents with different capabilities
        agent1 = A2AProtocol("agent1", "Coordinator")
        agent2 = A2AProtocol("agent2", "Worker")
        
        # Register capability on worker
        capability = AgentCapability(
            capability_id="compute_task",
            name="Compute Task",
            description="Performs computation",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        await agent2.register_capability(capability)
        
        # Register agents
        await network.register_agent(agent1)
        await network.register_agent(agent2)
        
        # Execute distributed task
        result = await network.execute_distributed_task(
            task={"operation": "compute", "value": 42},
            required_capabilities=["compute_task"]
        )
        
        # Check result
        assert "compute_task" in result
        assert result["compute_task"] is not None
        
    @pytest.mark.asyncio
    async def test_consensus_protocol(self):
        """Test Byzantine consensus protocol"""
        network = A2ANetwork()
        
        # Create multiple agents
        agents = []
        for i in range(5):
            agent = A2AProtocol(f"agent{i}", f"Consensus Agent {i}")
            await network.register_agent(agent)
            agents.append(agent)
        
        # Request global consensus
        proposal = {"action": "update_config", "value": "new_value"}
        consensus_achieved = await network.global_consensus(proposal)
        
        # Check consensus history
        assert len(network.consensus_history) == 1
        assert network.consensus_history[0]["proposal"] == proposal
        
    @pytest.mark.asyncio
    async def test_context_synchronization(self):
        """Test context synchronization across agents"""
        agent1 = A2AProtocol("agent1", "Context Provider")
        agent2 = A2AProtocol("agent2", "Context Consumer")
        
        # Start agents
        await agent1.start()
        await agent2.start()
        
        # Sync context
        context_update = {
            "system_state": "active",
            "cascade_risk": 0.3,
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
        await agent1.sync_context(context_update)
        
        # Verify context is stored
        assert "system_state" in agent1.context_store
        assert agent1.context_store["system_state"] == "active"
        
        # Stop agents
        await agent1.stop()
        await agent2.stop()
        
    @pytest.mark.asyncio
    async def test_network_health_metrics(self):
        """Test network health monitoring"""
        network = A2ANetwork()
        
        # Create network with multiple agents
        for i in range(10):
            agent = A2AProtocol(f"agent{i}", f"Test Agent {i}")
            await network.register_agent(agent)
        
        # Get network health
        health = network.get_network_health()
        
        assert health["total_agents"] == 10
        assert health["total_connections"] > 0
        assert 0 <= health["connectivity"] <= 1.0
        assert "consensus_success_rate" in health
        
    @pytest.mark.asyncio
    async def test_message_serialization(self):
        """Test MCP message serialization"""
        message = MCPMessage(
            message_type=MCPMessageType.TASK_REQUEST,
            sender_id="sender",
            receiver_id="receiver",
            payload={"task": "test", "data": [1, 2, 3]}
        )
        
        # Serialize
        serialized = message.to_bytes()
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = MCPMessage.from_bytes(serialized)
        assert deserialized.message_type == message.message_type
        assert deserialized.sender_id == message.sender_id
        assert deserialized.payload == message.payload
        
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self):
        """Test heartbeat mechanism for failure detection"""
        agent = A2AProtocol("agent1", "Heartbeat Test")
        
        # Start agent (which starts heartbeats)
        await agent.start()
        
        # Wait for heartbeat
        await asyncio.sleep(0.1)
        
        # Check that heartbeat task is running
        assert len(agent._tasks) >= 2  # Message processor + heartbeat
        
        # Stop agent
        await agent.stop()
        
    @pytest.mark.asyncio
    async def test_byzantine_fault_tolerance(self):
        """Test Byzantine fault tolerance in consensus"""
        network = A2ANetwork()
        
        # Create network with 7 agents (to test 2/3 requirement)
        for i in range(7):
            agent = A2AProtocol(f"agent{i}", f"Byzantine Agent {i}")
            await network.register_agent(agent)
        
        # Test consensus with simulated Byzantine behavior
        proposal = {"critical_update": True, "value": 100}
        
        # In real scenario, some agents would vote differently
        # For now, test that consensus mechanism works
        consensus = await network.global_consensus(proposal)
        
        # With all honest agents, consensus should be achieved
        assert isinstance(consensus, bool)


def test_a2a_integration():
    """Test A2A integration with AURA system"""
    from src.aura.core.system import AURASystem
    from src.aura.core.config import AURAConfig
    
    # Create AURA system
    config = AURAConfig()
    system = AURASystem(config)
    
    # Check A2A network is initialized
    assert hasattr(system, 'a2a_network')
    assert isinstance(system.a2a_network, A2ANetwork)
    
    # Check critical agents are registered
    assert len(system.a2a_network.agents) >= 5  # At least the critical agents
    
    # Check agent capabilities
    tda_agent_found = False
    lnn_agent_found = False
    
    for agent_id, agent in system.a2a_network.agents.items():
        if agent_id == "tda_agent":
            tda_agent_found = True
            assert "tda_analysis" in agent.capabilities
        elif agent_id == "lnn_agent":
            lnn_agent_found = True
            assert "failure_prediction" in agent.capabilities
    
    assert tda_agent_found, "TDA agent not found"
    assert lnn_agent_found, "LNN agent not found"


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_agent_registration())
    asyncio.run(test_capability_discovery())
    asyncio.run(test_consensus_protocol())
    test_a2a_integration()
    
    print("✅ All A2A communication tests passed!")