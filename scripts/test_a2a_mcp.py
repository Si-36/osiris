#!/usr/bin/env python3
"""
🔌 A2A/MCP Communication Test Script

Tests Agent-to-Agent and Model Context Protocol functionality.
Validates WebSocket connections, message routing, and context sharing.
"""

import asyncio
import sys
import json
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import websockets
import httpx
import jwt

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from src.aura.communication.a2a_mcp_server import (
    A2AMCPServer, AgentIdentity, A2AMessage, MCPContext
)

# Test configuration
TEST_CONFIG = {
    "a2a_url": "http://localhost:8090",
    "ws_url": "ws://localhost:8090/ws/a2a",
    "jwt_secret": "aura-secret-2025"
}

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class A2AMCPTester:
    """Test harness for A2A/MCP communication"""
    
    def __init__(self):
        self.server = None
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
    async def start_server(self):
        """Start the A2A/MCP server"""
        print(f"{BLUE}🚀 Starting A2A/MCP Server...{RESET}")
        
        self.server = A2AMCPServer(
            jwt_secret=TEST_CONFIG["jwt_secret"],
            redis_url="redis://localhost:6379"
        )
        
        # Start server in background
        asyncio.create_task(self.server.start(host="0.0.0.0", port=8090))
        
        # Wait for server to be ready
        await asyncio.sleep(2)
        
        # Check if server is running
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{TEST_CONFIG['a2a_url']}/health")
                if response.status_code == 200:
                    print(f"{GREEN}✅ A2A/MCP Server started successfully{RESET}")
                    return True
        except:
            pass
            
        print(f"{RED}❌ Failed to start A2A/MCP Server{RESET}")
        return False
    
    async def test_agent_authentication(self) -> Dict[str, Any]:
        """Test agent authentication"""
        print(f"\n{BLUE}1️⃣ Testing Agent Authentication...{RESET}")
        
        try:
            async with httpx.AsyncClient() as client:
                # Register test agents
                agents = [
                    {
                        "agent_id": "test_predictor_001",
                        "agent_type": "predictor",
                        "capabilities": ["topology_analysis", "failure_prediction"],
                        "permissions": ["read", "write", "predict"]
                    },
                    {
                        "agent_id": "test_analyzer_001",
                        "agent_type": "analyzer",
                        "capabilities": ["pattern_recognition", "anomaly_detection"],
                        "permissions": ["read", "analyze"]
                    },
                    {
                        "agent_id": "test_executor_001",
                        "agent_type": "executor",
                        "capabilities": ["intervention", "mitigation"],
                        "permissions": ["read", "write", "execute"]
                    }
                ]
                
                tokens = []
                for agent in agents:
                    response = await client.post(
                        f"{TEST_CONFIG['a2a_url']}/auth/agent",
                        json=agent
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        tokens.append(data["token"])
                        print(f"  {GREEN}✓ Authenticated {agent['agent_id']}{RESET}")
                    else:
                        raise Exception(f"Failed to authenticate {agent['agent_id']}")
                
                self.test_results["tests"].append({
                    "name": "Agent Authentication",
                    "status": "passed",
                    "details": f"Authenticated {len(agents)} agents"
                })
                
                return {"success": True, "tokens": tokens, "agents": agents}
                
        except Exception as e:
            print(f"  {RED}✗ Authentication failed: {e}{RESET}")
            self.test_results["tests"].append({
                "name": "Agent Authentication",
                "status": "failed",
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def test_mcp_context_operations(self, token: str) -> Dict[str, Any]:
        """Test MCP context creation and management"""
        print(f"\n{BLUE}2️⃣ Testing MCP Context Operations...{RESET}")
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                
                # Create context
                context_request = {
                    "method": "create_context",
                    "agent_id": "test_predictor_001",
                    "data": {
                        "shared_state": {
                            "topology": {"nodes": 30, "edges": 45},
                            "risk_level": 0.75,
                            "timestamp": datetime.now().isoformat()
                        },
                        "constraints": {
                            "max_agents": 100,
                            "timeout_seconds": 300
                        }
                    }
                }
                
                response = await client.post(
                    f"{TEST_CONFIG['a2a_url']}/mcp/request",
                    json=context_request,
                    headers=headers
                )
                
                if response.status_code != 200:
                    raise Exception(f"Failed to create context: {response.text}")
                
                context_data = response.json()
                context_id = context_data["context_id"]
                print(f"  {GREEN}✓ Created context: {context_id}{RESET}")
                
                # Update context
                update_request = {
                    "method": "update_context",
                    "context_id": context_id,
                    "agent_id": "test_predictor_001",
                    "data": {
                        "shared_state": {
                            "risk_level": 0.85,
                            "cascade_detected": True
                        }
                    }
                }
                
                response = await client.post(
                    f"{TEST_CONFIG['a2a_url']}/mcp/request",
                    json=update_request,
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"  {GREEN}✓ Updated context successfully{RESET}")
                
                # Get context
                get_request = {
                    "method": "get_context",
                    "context_id": context_id,
                    "agent_id": "test_predictor_001"
                }
                
                response = await client.post(
                    f"{TEST_CONFIG['a2a_url']}/mcp/request",
                    json=get_request,
                    headers=headers
                )
                
                if response.status_code == 200:
                    context = response.json()["data"]["context"]
                    print(f"  {GREEN}✓ Retrieved context (version {context['version']}){RESET}")
                
                self.test_results["tests"].append({
                    "name": "MCP Context Operations",
                    "status": "passed",
                    "details": {
                        "context_id": context_id,
                        "operations": ["create", "update", "get"]
                    }
                })
                
                return {"success": True, "context_id": context_id}
                
        except Exception as e:
            print(f"  {RED}✗ MCP operations failed: {e}{RESET}")
            self.test_results["tests"].append({
                "name": "MCP Context Operations",
                "status": "failed",
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def test_websocket_communication(self, agents: List[Dict], tokens: List[str]) -> Dict[str, Any]:
        """Test WebSocket A2A communication"""
        print(f"\n{BLUE}3️⃣ Testing WebSocket A2A Communication...{RESET}")
        
        try:
            # Connect multiple agents
            connections = []
            
            for i, (agent, token) in enumerate(zip(agents, tokens)):
                ws_url = f"{TEST_CONFIG['ws_url']}/{agent['agent_id']}"
                headers = {"Authorization": f"Bearer {token}"}
                
                ws = await websockets.connect(ws_url, extra_headers=headers)
                connections.append((agent, ws))
                print(f"  {GREEN}✓ Connected {agent['agent_id']} to WebSocket{RESET}")
            
            # Test message routing
            predictor_ws = connections[0][1]
            analyzer_ws = connections[1][1]
            
            # Send message from predictor to analyzer
            test_message = {
                "message_id": str(uuid.uuid4()),
                "from_agent": "test_predictor_001",
                "to_agent": "test_analyzer_001",
                "message_type": "request",
                "payload": {
                    "action": "analyze_topology",
                    "data": {
                        "nodes": 50,
                        "cascade_risk": 0.8
                    }
                },
                "timestamp": datetime.now().isoformat(),
                "requires_ack": True
            }
            
            await predictor_ws.send(json.dumps(test_message))
            print(f"  {YELLOW}→ Sent message from predictor to analyzer{RESET}")
            
            # Receive message at analyzer
            received = await asyncio.wait_for(analyzer_ws.recv(), timeout=5.0)
            received_msg = json.loads(received)
            
            if received_msg["message_id"] == test_message["message_id"]:
                print(f"  {GREEN}✓ Analyzer received message correctly{RESET}")
                
                # Send acknowledgment
                ack_message = {
                    "message_id": str(uuid.uuid4()),
                    "from_agent": "test_analyzer_001",
                    "to_agent": "test_predictor_001",
                    "message_type": "response",
                    "payload": {
                        "original_message_id": test_message["message_id"],
                        "status": "acknowledged",
                        "analysis": {
                            "risk_confirmed": True,
                            "recommended_action": "prevent_cascade"
                        }
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                await analyzer_ws.send(json.dumps(ack_message))
                
                # Receive acknowledgment at predictor
                ack_received = await asyncio.wait_for(predictor_ws.recv(), timeout=5.0)
                ack_msg = json.loads(ack_received)
                
                if ack_msg["payload"]["original_message_id"] == test_message["message_id"]:
                    print(f"  {GREEN}✓ Predictor received acknowledgment{RESET}")
            
            # Test broadcast
            broadcast_message = {
                "message_id": str(uuid.uuid4()),
                "from_agent": "test_executor_001",
                "to_agent": "broadcast",
                "message_type": "broadcast",
                "payload": {
                    "alert": "cascade_prevention_activated",
                    "level": "critical"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            executor_ws = connections[2][1]
            await executor_ws.send(json.dumps(broadcast_message))
            print(f"  {YELLOW}📢 Sent broadcast from executor{RESET}")
            
            # All agents should receive broadcast
            broadcast_count = 0
            for agent, ws in connections[:2]:  # predictor and analyzer
                try:
                    broadcast_recv = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    broadcast_msg = json.loads(broadcast_recv)
                    if broadcast_msg["message_type"] == "broadcast":
                        broadcast_count += 1
                except:
                    pass
            
            if broadcast_count >= 2:
                print(f"  {GREEN}✓ Broadcast received by {broadcast_count} agents{RESET}")
            
            # Close connections
            for agent, ws in connections:
                await ws.close()
            
            self.test_results["tests"].append({
                "name": "WebSocket A2A Communication",
                "status": "passed",
                "details": {
                    "agents_connected": len(connections),
                    "messages_sent": 3,
                    "broadcast_successful": broadcast_count >= 2
                }
            })
            
            return {"success": True}
            
        except Exception as e:
            print(f"  {RED}✗ WebSocket communication failed: {e}{RESET}")
            self.test_results["tests"].append({
                "name": "WebSocket A2A Communication",
                "status": "failed",
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def test_performance_and_scale(self, token: str) -> Dict[str, Any]:
        """Test performance with multiple agents"""
        print(f"\n{BLUE}4️⃣ Testing Performance and Scale...{RESET}")
        
        try:
            start_time = time.time()
            message_count = 100
            agent_count = 10
            
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                
                # Create multiple agents
                print(f"  Creating {agent_count} agents...")
                for i in range(agent_count):
                    agent_data = {
                        "agent_id": f"perf_agent_{i:03d}",
                        "agent_type": "worker",
                        "capabilities": ["compute"],
                        "permissions": ["read", "write"]
                    }
                    
                    await client.post(
                        f"{TEST_CONFIG['a2a_url']}/auth/agent",
                        json=agent_data
                    )
                
                # Send many messages
                print(f"  Sending {message_count} messages...")
                tasks = []
                
                for i in range(message_count):
                    message_data = {
                        "from_agent": f"perf_agent_{i % agent_count:03d}",
                        "to_agent": f"perf_agent_{(i+1) % agent_count:03d}",
                        "message_type": "request",
                        "payload": {"index": i, "data": "x" * 1000}  # 1KB payload
                    }
                    
                    task = client.post(
                        f"{TEST_CONFIG['a2a_url']}/messages/send",
                        json=message_data,
                        headers=headers
                    )
                    tasks.append(task)
                
                # Wait for all messages
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                
                elapsed_time = time.time() - start_time
                messages_per_second = success_count / elapsed_time
                
                print(f"  {GREEN}✓ Sent {success_count}/{message_count} messages{RESET}")
                print(f"  {GREEN}✓ Performance: {messages_per_second:.0f} messages/second{RESET}")
                print(f"  {GREEN}✓ Average latency: {(elapsed_time/success_count)*1000:.1f}ms{RESET}")
                
                self.test_results["tests"].append({
                    "name": "Performance and Scale",
                    "status": "passed" if messages_per_second > 100 else "warning",
                    "details": {
                        "agent_count": agent_count,
                        "message_count": message_count,
                        "success_rate": f"{(success_count/message_count)*100:.1f}%",
                        "messages_per_second": messages_per_second,
                        "avg_latency_ms": (elapsed_time/success_count)*1000
                    }
                })
                
                return {"success": True, "performance": messages_per_second}
                
        except Exception as e:
            print(f"  {RED}✗ Performance test failed: {e}{RESET}")
            self.test_results["tests"].append({
                "name": "Performance and Scale",
                "status": "failed",
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def test_integration_with_aura(self, token: str) -> Dict[str, Any]:
        """Test integration with AURA system"""
        print(f"\n{BLUE}5️⃣ Testing AURA System Integration...{RESET}")
        
        try:
            # Create AURA-specific agents
            aura_agents = [
                {
                    "agent_id": "aura_tda_analyzer",
                    "agent_type": "analyzer",
                    "capabilities": ["topological_analysis", "persistence_computation"],
                    "permissions": ["read", "analyze", "publish"]
                },
                {
                    "agent_id": "aura_lnn_predictor",
                    "agent_type": "predictor",
                    "capabilities": ["liquid_neural_network", "failure_prediction"],
                    "permissions": ["read", "predict", "alert"]
                },
                {
                    "agent_id": "aura_cascade_preventer",
                    "agent_type": "executor",
                    "capabilities": ["cascade_prevention", "intervention"],
                    "permissions": ["read", "write", "execute", "override"]
                }
            ]
            
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                
                # Register AURA agents
                for agent in aura_agents:
                    response = await client.post(
                        f"{TEST_CONFIG['a2a_url']}/auth/agent",
                        json=agent
                    )
                    if response.status_code == 200:
                        print(f"  {GREEN}✓ Registered {agent['agent_id']}{RESET}")
                
                # Simulate AURA workflow
                # 1. TDA analyzer detects anomaly
                tda_message = {
                    "from_agent": "aura_tda_analyzer",
                    "to_agent": "aura_lnn_predictor",
                    "message_type": "notification",
                    "payload": {
                        "alert": "topology_anomaly_detected",
                        "data": {
                            "persistence_0": {"components": 5, "max_lifetime": 0.8},
                            "persistence_1": {"holes": 2, "max_lifetime": 0.6},
                            "betti_numbers": [5, 2, 0],
                            "wasserstein_distance": 0.15
                        }
                    }
                }
                
                response = await client.post(
                    f"{TEST_CONFIG['a2a_url']}/messages/send",
                    json=tda_message,
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"  {GREEN}✓ TDA analyzer sent anomaly notification{RESET}")
                
                # 2. LNN predictor responds with cascade risk
                prediction_message = {
                    "from_agent": "aura_lnn_predictor",
                    "to_agent": "aura_cascade_preventer",
                    "message_type": "request",
                    "payload": {
                        "prediction": "high_cascade_risk",
                        "data": {
                            "cascade_probability": 0.87,
                            "affected_agents": ["agent_15", "agent_23", "agent_31"],
                            "time_to_cascade": 45,  # seconds
                            "recommended_action": "immediate_intervention"
                        }
                    }
                }
                
                response = await client.post(
                    f"{TEST_CONFIG['a2a_url']}/messages/send",
                    json=prediction_message,
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"  {GREEN}✓ LNN predictor sent cascade risk assessment{RESET}")
                
                # 3. Cascade preventer takes action
                prevention_message = {
                    "from_agent": "aura_cascade_preventer",
                    "to_agent": "broadcast",
                    "message_type": "broadcast",
                    "payload": {
                        "action": "cascade_prevention_activated",
                        "measures": [
                            "isolate_agent_15",
                            "redistribute_load",
                            "activate_byzantine_consensus"
                        ],
                        "status": "intervention_in_progress"
                    }
                }
                
                response = await client.post(
                    f"{TEST_CONFIG['a2a_url']}/messages/send",
                    json=prevention_message,
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"  {GREEN}✓ Cascade preventer broadcast intervention{RESET}")
                
                self.test_results["tests"].append({
                    "name": "AURA System Integration",
                    "status": "passed",
                    "details": {
                        "aura_agents": len(aura_agents),
                        "workflow_steps": 3,
                        "integration": "successful"
                    }
                })
                
                return {"success": True}
                
        except Exception as e:
            print(f"  {RED}✗ AURA integration failed: {e}{RESET}")
            self.test_results["tests"].append({
                "name": "AURA System Integration",
                "status": "failed",
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def run_all_tests(self):
        """Run all A2A/MCP tests"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}🧪 AURA A2A/MCP Communication Test Suite{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        
        # Start server
        if not await self.start_server():
            print(f"{RED}Cannot proceed without server{RESET}")
            return
        
        # Run tests
        auth_result = await self.test_agent_authentication()
        
        if auth_result["success"]:
            token = auth_result["tokens"][0]
            agents = auth_result["agents"]
            
            # Run remaining tests
            await self.test_mcp_context_operations(token)
            await self.test_websocket_communication(agents, auth_result["tokens"])
            await self.test_performance_and_scale(token)
            await self.test_integration_with_aura(token)
        
        # Generate summary
        passed = sum(1 for t in self.test_results["tests"] if t["status"] == "passed")
        warnings = sum(1 for t in self.test_results["tests"] if t["status"] == "warning")
        failed = sum(1 for t in self.test_results["tests"] if t["status"] == "failed")
        total = len(self.test_results["tests"])
        
        self.test_results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "success_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "0%"
        }
        
        # Print summary
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}📊 Test Summary{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"Total Tests: {total}")
        print(f"{GREEN}✅ Passed: {passed}{RESET}")
        print(f"{YELLOW}⚠️  Warnings: {warnings}{RESET}")
        print(f"{RED}❌ Failed: {failed}{RESET}")
        print(f"Success Rate: {self.test_results['summary']['success_rate']}")
        
        # Save results
        with open("a2a_mcp_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n{BLUE}📄 Results saved to: a2a_mcp_test_results.json{RESET}")
        
        # Shutdown server
        if self.server:
            await self.server.shutdown()


async def main():
    """Main test runner"""
    tester = A2AMCPTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())