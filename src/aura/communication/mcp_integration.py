"""
🧠 AURA Model Context Protocol (MCP) Integration

Implements the MCP standard for sharing context, tools, and resources between agents.
Based on Anthropic's MCP specification for interoperability.

Features:
- Context sharing across agent boundaries
- Tool registration and discovery
- Resource management
- Capability negotiation
- Streaming responses
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
import logging

# Configure logging
logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types based on the standard"""
    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"
    
    # Discovery
    LIST_TOOLS = "tools/list"
    LIST_RESOURCES = "resources/list"
    LIST_PROMPTS = "prompts/list"
    
    # Execution
    CALL_TOOL = "tools/call"
    READ_RESOURCE = "resources/read"
    
    # Context
    GET_CONTEXT = "context/get"
    SET_CONTEXT = "context/set"
    SUBSCRIBE_CONTEXT = "context/subscribe"
    
    # Sampling
    CREATE_MESSAGE = "sampling/createMessage"
    
    # Notifications
    PROGRESS = "notifications/progress"
    RESOURCE_UPDATED = "notifications/resources/updated"
    TOOL_UPDATED = "notifications/tools/updated"


@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compliant format"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.input_schema,
                "required": list(self.input_schema.keys())
            }
        }


@dataclass
class MCPResource:
    """MCP resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compliant format"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


class MCPRequest(BaseModel):
    """Standard MCP request format"""
    jsonrpc: str = "2.0"
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = {}


class MCPResponse(BaseModel):
    """Standard MCP response format"""
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPNotification(BaseModel):
    """Standard MCP notification format"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = {}


class MCPCapabilities(BaseModel):
    """MCP server capabilities"""
    tools: bool = True
    resources: bool = True
    prompts: bool = True
    logging: bool = True
    experimental: Dict[str, bool] = {
        "aura_topology": True,
        "failure_prediction": True,
        "consensus": True
    }


class MCPContext:
    """Shared context for MCP operations"""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Any:
        """Get context value"""
        async with self._lock:
            return self.data.get(key)
    
    async def set(self, key: str, value: Any):
        """Set context value and notify subscribers"""
        async with self._lock:
            old_value = self.data.get(key)
            self.data[key] = value
            
        # Notify subscribers
        if key in self.subscriptions:
            for callback in self.subscriptions[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, value, old_value)
                    else:
                        callback(key, value, old_value)
                except Exception as e:
                    logger.error(f"Error in context subscriber: {e}")
    
    async def subscribe(self, key: str, callback: Callable):
        """Subscribe to context changes"""
        async with self._lock:
            if key not in self.subscriptions:
                self.subscriptions[key] = []
            self.subscriptions[key].append(callback)


class MCPServer:
    """
    MCP Server implementation for AURA agents.
    
    Provides tools, resources, and context to other agents/systems.
    """
    
    def __init__(self, server_name: str = "aura-mcp-server"):
        self.server_name = server_name
        self.server_info = {
            "name": server_name,
            "version": "1.0.0",
            "protocolVersion": "0.1.0"
        }
        
        # Registry
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.context = MCPContext()
        
        # Request handlers
        self.handlers: Dict[str, Callable] = {
            MCPMessageType.INITIALIZE.value: self._handle_initialize,
            MCPMessageType.LIST_TOOLS.value: self._handle_list_tools,
            MCPMessageType.LIST_RESOURCES.value: self._handle_list_resources,
            MCPMessageType.CALL_TOOL.value: self._handle_call_tool,
            MCPMessageType.READ_RESOURCE.value: self._handle_read_resource,
            MCPMessageType.GET_CONTEXT.value: self._handle_get_context,
            MCPMessageType.SET_CONTEXT.value: self._handle_set_context,
        }
        
        # Metrics
        self.request_count = 0
        self.tool_calls = 0
        
        # Register default AURA tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default AURA-specific tools"""
        
        # Topology analysis tool
        self.register_tool(MCPTool(
            name="analyze_topology",
            description="Analyze agent network topology",
            input_schema={
                "agent_ids": {"type": "array", "items": {"type": "string"}},
                "depth": {"type": "integer", "default": 2}
            },
            handler=self._analyze_topology_handler
        ))
        
        # Failure prediction tool
        self.register_tool(MCPTool(
            name="predict_failure",
            description="Predict agent failure probability",
            input_schema={
                "agent_id": {"type": "string"},
                "time_window": {"type": "integer", "default": 60}
            },
            handler=self._predict_failure_handler
        ))
        
        # Intervention tool
        self.register_tool(MCPTool(
            name="request_intervention",
            description="Request AURA intervention for failure prevention",
            input_schema={
                "target_agents": {"type": "array", "items": {"type": "string"}},
                "intervention_type": {"type": "string", "enum": ["isolate", "restart", "migrate", "monitor"]},
                "reason": {"type": "string"}
            },
            handler=self._intervention_handler,
            requires_confirmation=True
        ))
    
    def register_tool(self, tool: MCPTool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource):
        """Register a resource"""
        self.resources[resource.uri] = resource
        logger.info(f"Registered MCP resource: {resource.uri}")
    
    async def handle_request(self, request: Union[Dict, MCPRequest]) -> MCPResponse:
        """Handle incoming MCP request"""
        self.request_count += 1
        
        # Parse request
        if isinstance(request, dict):
            request = MCPRequest(**request)
        
        # Route to handler
        handler = self.handlers.get(request.method)
        if not handler:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )
        
        try:
            result = await handler(request.params or {})
            return MCPResponse(id=request.id, result=result)
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e)
                }
            )
    
    # Request handlers
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        client_info = params.get("clientInfo", {})
        logger.info(f"MCP client connected: {client_info.get('name', 'unknown')}")
        
        return {
            "serverInfo": self.server_info,
            "capabilities": MCPCapabilities().dict()
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list tools request"""
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()]
        }
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list resources request"""
        return {
            "resources": [resource.to_dict() for resource in self.resources.values()]
        }
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Any:
        """Handle tool call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        self.tool_calls += 1
        
        # Execute tool
        if asyncio.iscoroutinefunction(tool.handler):
            result = await tool.handler(**arguments)
        else:
            result = tool.handler(**arguments)
        
        return {"result": result}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
        
        # In a real implementation, this would read actual resource data
        return {
            "contents": {
                "uri": uri,
                "mimeType": self.resources[uri].mime_type,
                "text": f"Resource content for {uri}"
            }
        }
    
    async def _handle_get_context(self, params: Dict[str, Any]) -> Any:
        """Handle context get request"""
        key = params.get("key")
        if key:
            return {"value": await self.context.get(key)}
        else:
            return {"context": self.context.data}
    
    async def _handle_set_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context set request"""
        key = params.get("key")
        value = params.get("value")
        
        if key:
            await self.context.set(key, value)
            return {"success": True}
        else:
            raise ValueError("Key is required for context set")
    
    # Tool handlers
    async def _analyze_topology_handler(self, agent_ids: List[str], depth: int = 2) -> Dict[str, Any]:
        """Analyze topology handler"""
        # Simulated topology analysis
        return {
            "agent_count": len(agent_ids),
            "clusters": 3,
            "avg_connectivity": 0.42,
            "critical_nodes": agent_ids[:2] if len(agent_ids) >= 2 else agent_ids,
            "vulnerability_score": 0.15,
            "analysis_depth": depth
        }
    
    async def _predict_failure_handler(self, agent_id: str, time_window: int = 60) -> Dict[str, Any]:
        """Predict failure handler"""
        # Simulated failure prediction
        return {
            "agent_id": agent_id,
            "failure_probability": 0.08,
            "cascade_risk": 0.12,
            "time_window": time_window,
            "contributing_factors": [
                "high_memory_usage",
                "network_latency"
            ],
            "recommended_action": "monitor"
        }
    
    async def _intervention_handler(self, target_agents: List[str], 
                                  intervention_type: str, reason: str) -> Dict[str, Any]:
        """Intervention handler"""
        # Simulated intervention
        return {
            "intervention_id": str(uuid.uuid4()),
            "target_agents": target_agents,
            "type": intervention_type,
            "reason": reason,
            "status": "initiated",
            "estimated_completion": 15  # seconds
        }


class MCPClient:
    """
    MCP Client for connecting to MCP servers.
    
    Allows AURA agents to use tools and resources from other systems.
    """
    
    def __init__(self, client_name: str = "aura-mcp-client"):
        self.client_name = client_name
        self.client_info = {
            "name": client_name,
            "version": "1.0.0"
        }
        
        self.server_info: Optional[Dict[str, Any]] = None
        self.capabilities: Optional[MCPCapabilities] = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.available_resources: Dict[str, Dict[str, Any]] = {}
        
        # Connection info
        self.connected = False
        self.session_id = str(uuid.uuid4())
    
    async def connect(self, server: MCPServer) -> bool:
        """Connect to MCP server"""
        try:
            # Send initialize request
            response = await server.handle_request(MCPRequest(
                method=MCPMessageType.INITIALIZE.value,
                params={"clientInfo": self.client_info}
            ))
            
            if response.error:
                logger.error(f"Failed to initialize: {response.error}")
                return False
            
            self.server_info = response.result.get("serverInfo")
            self.capabilities = MCPCapabilities(**response.result.get("capabilities", {}))
            
            # Discover tools
            await self._discover_tools(server)
            
            # Discover resources
            await self._discover_resources(server)
            
            self.connected = True
            logger.info(f"Connected to MCP server: {self.server_info['name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def _discover_tools(self, server: MCPServer):
        """Discover available tools"""
        response = await server.handle_request(MCPRequest(
            method=MCPMessageType.LIST_TOOLS.value
        ))
        
        if response.result:
            tools = response.result.get("tools", [])
            self.available_tools = {tool["name"]: tool for tool in tools}
            logger.info(f"Discovered {len(self.available_tools)} MCP tools")
    
    async def _discover_resources(self, server: MCPServer):
        """Discover available resources"""
        response = await server.handle_request(MCPRequest(
            method=MCPMessageType.LIST_RESOURCES.value
        ))
        
        if response.result:
            resources = response.result.get("resources", [])
            self.available_resources = {res["uri"]: res for res in resources}
            logger.info(f"Discovered {len(self.available_resources)} MCP resources")
    
    async def call_tool(self, server: MCPServer, tool_name: str, 
                       arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server"""
        if not self.connected:
            raise RuntimeError("Not connected to MCP server")
        
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        response = await server.handle_request(MCPRequest(
            method=MCPMessageType.CALL_TOOL.value,
            params={
                "name": tool_name,
                "arguments": arguments
            }
        ))
        
        if response.error:
            raise RuntimeError(f"Tool call failed: {response.error}")
        
        return response.result.get("result")
    
    async def read_resource(self, server: MCPServer, uri: str) -> Any:
        """Read a resource from the server"""
        if not self.connected:
            raise RuntimeError("Not connected to MCP server")
        
        if uri not in self.available_resources:
            raise ValueError(f"Resource not found: {uri}")
        
        response = await server.handle_request(MCPRequest(
            method=MCPMessageType.READ_RESOURCE.value,
            params={"uri": uri}
        ))
        
        if response.error:
            raise RuntimeError(f"Resource read failed: {response.error}")
        
        return response.result.get("contents")


# Example: AURA-specific MCP extensions
class AURAMCPExtensions:
    """AURA-specific MCP protocol extensions"""
    
    @staticmethod
    def create_topology_tool() -> MCPTool:
        """Create advanced topology analysis tool"""
        async def analyze_advanced_topology(
            agent_ids: List[str],
            analysis_type: str = "comprehensive",
            include_predictions: bool = True
        ) -> Dict[str, Any]:
            
            # Simulated advanced analysis
            result = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "topology": {
                    "nodes": len(agent_ids),
                    "edges": len(agent_ids) * 2,  # Simulated
                    "density": 0.42,
                    "diameter": 4,
                    "clustering_coefficient": 0.65
                },
                "communities": [
                    {"id": 1, "agents": agent_ids[:len(agent_ids)//3], "cohesion": 0.85},
                    {"id": 2, "agents": agent_ids[len(agent_ids)//3:2*len(agent_ids)//3], "cohesion": 0.78},
                    {"id": 3, "agents": agent_ids[2*len(agent_ids)//3:], "cohesion": 0.82}
                ],
                "critical_paths": [
                    {"from": agent_ids[0], "to": agent_ids[-1], "hops": 3, "reliability": 0.92}
                ]
            }
            
            if include_predictions:
                result["predictions"] = {
                    "failure_hotspots": agent_ids[:2],
                    "cascade_probability": 0.15,
                    "recommended_interventions": [
                        {"type": "add_redundancy", "location": agent_ids[1]},
                        {"type": "increase_monitoring", "agents": agent_ids[:3]}
                    ]
                }
            
            return result
        
        return MCPTool(
            name="analyze_advanced_topology",
            description="Perform advanced topology analysis with ML predictions",
            input_schema={
                "agent_ids": {"type": "array", "items": {"type": "string"}},
                "analysis_type": {"type": "string", "enum": ["basic", "comprehensive", "predictive"]},
                "include_predictions": {"type": "boolean", "default": True}
            },
            handler=analyze_advanced_topology
        )
    
    @staticmethod
    def create_consensus_tool() -> MCPTool:
        """Create Byzantine consensus tool"""
        async def byzantine_consensus(
            proposal: Dict[str, Any],
            participants: List[str],
            timeout: int = 30
        ) -> Dict[str, Any]:
            
            # Simulated consensus
            await asyncio.sleep(2)  # Simulate voting time
            
            votes = {p: hash(p) % 2 == 0 for p in participants}  # Random votes
            agree_count = sum(votes.values())
            
            return {
                "consensus_id": str(uuid.uuid4()),
                "proposal": proposal,
                "participants": len(participants),
                "votes_collected": len(votes),
                "agree": agree_count,
                "disagree": len(votes) - agree_count,
                "consensus_reached": agree_count / len(votes) >= 0.66,
                "timestamp": datetime.now().isoformat()
            }
        
        return MCPTool(
            name="byzantine_consensus",
            description="Execute Byzantine fault-tolerant consensus",
            input_schema={
                "proposal": {"type": "object"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "timeout": {"type": "integer", "default": 30}
            },
            handler=byzantine_consensus,
            requires_confirmation=True
        )


# Example usage
async def example_mcp_usage():
    """Example of using MCP in AURA"""
    
    # Create MCP server for an AURA agent
    server = MCPServer("aura-agent-001-mcp")
    
    # Register custom AURA tools
    server.register_tool(AURAMCPExtensions.create_topology_tool())
    server.register_tool(AURAMCPExtensions.create_consensus_tool())
    
    # Register resources
    server.register_resource(MCPResource(
        uri="aura://topology/current",
        name="Current Topology",
        description="Current agent network topology"
    ))
    
    # Create MCP client for another agent
    client = MCPClient("aura-agent-002-mcp")
    
    # Connect client to server
    connected = await client.connect(server)
    if not connected:
        print("Failed to connect")
        return
    
    print(f"Connected to server: {client.server_info['name']}")
    print(f"Available tools: {list(client.available_tools.keys())}")
    
    # Call topology analysis tool
    result = await client.call_tool(
        server,
        "analyze_advanced_topology",
        {
            "agent_ids": ["agent-001", "agent-002", "agent-003", "agent-004"],
            "analysis_type": "comprehensive",
            "include_predictions": True
        }
    )
    
    print(f"\nTopology Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Call consensus tool
    consensus_result = await client.call_tool(
        server,
        "byzantine_consensus",
        {
            "proposal": {"action": "scale_up", "new_agents": 5},
            "participants": ["agent-001", "agent-002", "agent-003", "agent-004", "agent-005"],
            "timeout": 10
        }
    )
    
    print(f"\nConsensus Result:")
    print(json.dumps(consensus_result, indent=2))
    
    # Read resource
    resource_data = await client.read_resource(server, "aura://topology/current")
    print(f"\nResource Data: {resource_data}")
    
    # Server metrics
    print(f"\nServer Metrics:")
    print(f"  Requests handled: {server.request_count}")
    print(f"  Tool calls: {server.tool_calls}")


if __name__ == "__main__":
    asyncio.run(example_mcp_usage())