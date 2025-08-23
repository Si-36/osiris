"""
🤝 AURA A2A (Agent-to-Agent) Communication Module
"""

from .protocol import (
    A2AProtocol,
    A2ANetwork,
    MCPMessage,
    MCPMessageType,
    AgentCapability,
    AgentPeer
)

__all__ = [
    "A2AProtocol",
    "A2ANetwork", 
    "MCPMessage",
    "MCPMessageType",
    "AgentCapability",
    "AgentPeer"
]

__version__ = "1.0.0"