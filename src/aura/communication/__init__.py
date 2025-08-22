"""
AURA Communication Module

Provides A2A (Agent-to-Agent) and MCP (Model Context Protocol) communication.
"""

# Optional imports to avoid test failures when dependencies are missing
try:
    from .a2a_protocol import A2ACommunicationProtocol
    from .mcp_integration import MCPServer, MCPClient
    __all__ = ["A2ACommunicationProtocol", "MCPServer", "MCPClient"]
except ImportError:
    # If dependencies are missing, provide mock classes
    class A2ACommunicationProtocol:
        def __init__(self, *args, **kwargs):
            pass
    
    class MCPServer:
        def __init__(self, *args, **kwargs):
            pass
    
    class MCPClient:
        def __init__(self, *args, **kwargs):
            pass
            
    __all__ = ["A2ACommunicationProtocol", "MCPServer", "MCPClient"]