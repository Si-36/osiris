"""
🤝 AURA A2A (Agent-to-Agent) Communication Module

Production-grade agent communication protocol with MCP integration.
"""

from .agent_protocol import (
    A2AProtocol,
    A2AMessage,
    MCPContext,
    MessageType,
    AgentRole,
    create_a2a_network,
    demonstrate_cascade_prevention
)

__all__ = [
    'A2AProtocol',
    'A2AMessage',
    'MCPContext',
    'MessageType',
    'AgentRole',
    'create_a2a_network',
    'demonstrate_cascade_prevention'
]

__version__ = "1.0.0"