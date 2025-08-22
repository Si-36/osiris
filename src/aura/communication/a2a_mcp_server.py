#!/usr/bin/env python3
"""
🔌 AURA A2A/MCP Communication Server

Agent-to-Agent (A2A) communication with Model Context Protocol (MCP) support.
Enables secure, scalable inter-agent communication with context sharing.

Based on latest 2025 AI communication patterns and MCP v1.0 specification.
"""

import asyncio
import json
import uuid
import time
import jwt
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
message_counter = Counter('a2a_messages_total', 'Total A2A messages', ['type', 'status'])
mcp_requests = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'status'])
active_connections = Gauge('a2a_active_connections', 'Active WebSocket connections')
message_latency = Histogram('a2a_message_latency_seconds', 'Message delivery latency')
context_size = Histogram('mcp_context_size_bytes', 'MCP context size')

# Security
security = HTTPBearer()


@dataclass
class AgentIdentity:
    """Agent identity and capabilities"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    permissions: List[str]
    metadata: Dict[str, Any]
    authenticated_at: datetime
    

@dataclass
class A2AMessage:
    """Agent-to-Agent message structure"""
    message_id: str
    from_agent: str
    to_agent: str
    message_type: str  # request, response, broadcast, notification
    payload: Dict[str, Any]
    timestamp: datetime
    ttl: Optional[int] = None  # Time to live in seconds
    requires_ack: bool = False
    context_id: Optional[str] = None
    

@dataclass
class MCPContext:
    """Model Context Protocol context"""
    context_id: str
    agent_ids: List[str]
    shared_state: Dict[str, Any]
    constraints: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    

class MCPRequest(BaseModel):
    """MCP request structure"""
    method: str  # create_context, update_context, get_context, delete_context
    context_id: Optional[str] = None
    agent_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    

class MCPResponse(BaseModel):
    """MCP response structure"""
    success: bool
    context_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    

class A2AMCPServer:
    """
    🔌 A2A/MCP Communication Server
    
    Provides:
    - Agent-to-Agent messaging with WebSocket support
    - Model Context Protocol for shared context
    - Secure authentication and authorization
    - Message routing and broadcasting
    - Context persistence and synchronization
    - Prometheus metrics and monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="AURA A2A/MCP Server")
        self.redis_client: Optional[redis.Redis] = None
        
        # In-memory stores
        self.agents: Dict[str, AgentIdentity] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.contexts: Dict[str, MCPContext] = {}
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Configuration
        self.jwt_secret = config.get('JWT_SECRET', 'aura-secret-2025')
        self.jwt_algorithm = 'HS256'
        self.message_ttl = config.get('MESSAGE_TTL', 300)  # 5 minutes
        self.context_ttl = config.get('CONTEXT_TTL', 3600)  # 1 hour
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "agents": len(self.agents),
                "connections": len(self.connections),
                "contexts": len(self.contexts)
            }
            
        @self.app.get("/metrics")
        async def metrics():
            return generate_latest()
            
        @self.app.post("/auth/agent")
        async def authenticate_agent(
            agent_id: str,
            agent_type: str,
            capabilities: List[str] = [],
            permissions: List[str] = []
        ):
            """Authenticate an agent and return JWT token"""
            # Create agent identity
            agent = AgentIdentity(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                permissions=permissions,
                metadata={},
                authenticated_at=datetime.utcnow()
            )
            
            # Store agent
            self.agents[agent_id] = agent
            
            # Generate JWT
            payload = {
                'agent_id': agent_id,
                'agent_type': agent_type,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            
            return {"token": token, "agent_id": agent_id}
            
        @self.app.websocket("/ws/a2a/{agent_id}")
        async def a2a_websocket(websocket: WebSocket, agent_id: str):
            """WebSocket endpoint for A2A communication"""
            await self._handle_websocket(websocket, agent_id)
            
        @self.app.post("/mcp/request")
        async def mcp_request(
            request: MCPRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Handle MCP requests"""
            # Verify JWT
            agent_id = self._verify_token(credentials.credentials)
            if agent_id != request.agent_id:
                raise HTTPException(status_code=403, detail="Agent ID mismatch")
                
            return await self._handle_mcp_request(request)
            
    async def initialize(self):
        """Initialize the server"""
        # Connect to Redis
        try:
            self.redis_client = await redis.from_url(
                self.config.get('REDIS_URL', 'redis://localhost:6379'),
                password=self.config.get('REDIS_PASSWORD')
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory only.")
            
    async def _handle_websocket(self, websocket: WebSocket, agent_id: str):
        """Handle WebSocket connection for agent"""
        await websocket.accept()
        
        # Verify agent exists
        if agent_id not in self.agents:
            await websocket.close(code=4001, reason="Unauthorized")
            return
            
        # Store connection
        self.connections[agent_id] = websocket
        active_connections.inc()
        
        try:
            # Send welcome message
            await self._send_message(agent_id, {
                "type": "welcome",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Handle messages
            while True:
                data = await websocket.receive_json()
                await self._process_a2a_message(agent_id, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Agent {agent_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error for {agent_id}: {e}")
        finally:
            # Clean up
            if agent_id in self.connections:
                del self.connections[agent_id]
            active_connections.dec()
            
    async def _process_a2a_message(self, from_agent: str, data: Dict[str, Any]):
        """Process incoming A2A message"""
        start_time = time.time()
        
        try:
            # Create message
            message = A2AMessage(
                message_id=str(uuid.uuid4()),
                from_agent=from_agent,
                to_agent=data.get('to_agent', ''),
                message_type=data.get('type', 'request'),
                payload=data.get('payload', {}),
                timestamp=datetime.utcnow(),
                ttl=data.get('ttl', self.message_ttl),
                requires_ack=data.get('requires_ack', False),
                context_id=data.get('context_id')
            )
            
            # Route message
            if message.to_agent:
                # Direct message
                await self._route_message(message)
            else:
                # Broadcast
                await self._broadcast_message(message)
                
            # Record metrics
            message_counter.labels(type=message.message_type, status='success').inc()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            message_counter.labels(type='unknown', status='error').inc()
            
        finally:
            message_latency.observe(time.time() - start_time)
            
    async def _route_message(self, message: A2AMessage):
        """Route message to specific agent"""
        if message.to_agent in self.connections:
            await self._send_message(message.to_agent, asdict(message))
            
            # Store in Redis if available
            if self.redis_client:
                key = f"a2a:message:{message.message_id}"
                await self.redis_client.setex(
                    key, 
                    message.ttl or self.message_ttl,
                    json.dumps(asdict(message), default=str)
                )
        else:
            logger.warning(f"Agent {message.to_agent} not connected")
            
    async def _broadcast_message(self, message: A2AMessage):
        """Broadcast message to all connected agents"""
        tasks = []
        for agent_id, ws in self.connections.items():
            if agent_id != message.from_agent:
                tasks.append(self._send_message(agent_id, asdict(message)))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _send_message(self, agent_id: str, data: Dict[str, Any]):
        """Send message to specific agent"""
        if agent_id in self.connections:
            try:
                await self.connections[agent_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send message to {agent_id}: {e}")
                
    async def _handle_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Handle MCP request"""
        start_time = time.time()
        
        try:
            if request.method == "create_context":
                response = await self._create_context(request)
            elif request.method == "update_context":
                response = await self._update_context(request)
            elif request.method == "get_context":
                response = await self._get_context(request)
            elif request.method == "delete_context":
                response = await self._delete_context(request)
            else:
                response = MCPResponse(
                    success=False,
                    error=f"Unknown method: {request.method}"
                )
                
            # Record metrics
            mcp_requests.labels(
                method=request.method, 
                status='success' if response.success else 'error'
            ).inc()
            
            return response
            
        except Exception as e:
            logger.error(f"MCP request error: {e}")
            mcp_requests.labels(method=request.method, status='error').inc()
            return MCPResponse(success=False, error=str(e))
            
    async def _create_context(self, request: MCPRequest) -> MCPResponse:
        """Create new MCP context"""
        context_id = str(uuid.uuid4())
        
        context = MCPContext(
            context_id=context_id,
            agent_ids=[request.agent_id],
            shared_state=request.data.get('initial_state', {}),
            constraints=request.data.get('constraints', {}),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
        
        # Store context
        self.contexts[context_id] = context
        
        # Store in Redis if available
        if self.redis_client:
            key = f"mcp:context:{context_id}"
            await self.redis_client.setex(
                key,
                self.context_ttl,
                json.dumps(asdict(context), default=str)
            )
            
        # Record metric
        context_size.observe(len(json.dumps(asdict(context))))
        
        return MCPResponse(
            success=True,
            context_id=context_id,
            data={"context": asdict(context)}
        )
        
    async def _update_context(self, request: MCPRequest) -> MCPResponse:
        """Update existing MCP context"""
        if not request.context_id or request.context_id not in self.contexts:
            return MCPResponse(success=False, error="Context not found")
            
        context = self.contexts[request.context_id]
        
        # Update shared state
        if 'state_updates' in request.data:
            context.shared_state.update(request.data['state_updates'])
            
        # Add agent if not present
        if request.agent_id not in context.agent_ids:
            context.agent_ids.append(request.agent_id)
            
        # Update metadata
        context.updated_at = datetime.utcnow()
        context.version += 1
        
        # Notify other agents in context
        await self._notify_context_update(context, request.agent_id)
        
        return MCPResponse(
            success=True,
            context_id=context.context_id,
            data={"context": asdict(context)}
        )
        
    async def _get_context(self, request: MCPRequest) -> MCPResponse:
        """Get MCP context"""
        if not request.context_id:
            return MCPResponse(success=False, error="Context ID required")
            
        # Try memory first
        if request.context_id in self.contexts:
            context = self.contexts[request.context_id]
            return MCPResponse(
                success=True,
                context_id=context.context_id,
                data={"context": asdict(context)}
            )
            
        # Try Redis
        if self.redis_client:
            key = f"mcp:context:{request.context_id}"
            data = await self.redis_client.get(key)
            if data:
                context_data = json.loads(data)
                return MCPResponse(
                    success=True,
                    context_id=request.context_id,
                    data={"context": context_data}
                )
                
        return MCPResponse(success=False, error="Context not found")
        
    async def _delete_context(self, request: MCPRequest) -> MCPResponse:
        """Delete MCP context"""
        if not request.context_id or request.context_id not in self.contexts:
            return MCPResponse(success=False, error="Context not found")
            
        # Remove from memory
        del self.contexts[request.context_id]
        
        # Remove from Redis
        if self.redis_client:
            key = f"mcp:context:{request.context_id}"
            await self.redis_client.delete(key)
            
        return MCPResponse(success=True, context_id=request.context_id)
        
    async def _notify_context_update(self, context: MCPContext, from_agent: str):
        """Notify agents about context update"""
        notification = {
            "type": "context_update",
            "context_id": context.context_id,
            "version": context.version,
            "updated_by": from_agent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Notify all agents in context
        tasks = []
        for agent_id in context.agent_ids:
            if agent_id != from_agent and agent_id in self.connections:
                tasks.append(self._send_message(agent_id, notification))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    def _verify_token(self, token: str) -> str:
        """Verify JWT token and return agent ID"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload['agent_id']
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type].append(handler)
        
    async def run(self, host: str = "0.0.0.0", port: int = 8090):
        """Run the server"""
        await self.initialize()
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            ws_ping_interval=20,
            ws_ping_timeout=60
        )
        server = uvicorn.Server(config)
        await server.serve()


# Example usage and standalone runner
async def main():
    """Run A2A/MCP server"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = {
        'JWT_SECRET': os.getenv('JWT_SECRET', 'aura-jwt-secret-2025'),
        'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379'),
        'REDIS_PASSWORD': os.getenv('REDIS_PASSWORD'),
        'MESSAGE_TTL': int(os.getenv('MESSAGE_TTL', '300')),
        'CONTEXT_TTL': int(os.getenv('CONTEXT_TTL', '3600'))
    }
    
    server = A2AMCPServer(config)
    
    logger.info("🚀 Starting AURA A2A/MCP Communication Server")
    logger.info(f"📡 WebSocket: ws://localhost:8090/ws/a2a/{{agent_id}}")
    logger.info(f"🔌 MCP API: http://localhost:8090/mcp/request")
    
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())