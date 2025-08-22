#!/usr/bin/env python3
"""
AURA Agent Failure Prevention Demo
Demonstrates preventing cascading failures in multi-agent systems through topological intelligence
"""

import asyncio
import time
import random
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import math

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AURA Agent Failure Prevention")

# Agent states
class AgentState:
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"

class Agent:
    """Represents an agent in the multi-agent system"""
    def __init__(self, agent_id: str, x: float, y: float):
        self.id = agent_id
        self.x = x
        self.y = y
        self.state = AgentState.HEALTHY
        self.connections: Set[str] = set()
        self.load = random.uniform(0.3, 0.7)
        self.failure_probability = 0.01
        self.messages_sent = 0
        self.messages_failed = 0
        
    def add_connection(self, other_id: str):
        self.connections.add(other_id)
        
    def update_load(self):
        """Simulate load changes"""
        self.load += random.uniform(-0.1, 0.1)
        self.load = max(0.1, min(0.95, self.load))
        
        # High load increases failure probability
        if self.load > 0.8:
            self.failure_probability = 0.1
        elif self.load > 0.9:
            self.failure_probability = 0.3
        else:
            self.failure_probability = 0.01

class TopologicalSignature:
    """Represents topological features of the agent network"""
    def __init__(self):
        self.connected_components = 0
        self.clustering_coefficient = 0.0
        self.average_degree = 0.0
        self.diameter = 0
        self.bottlenecks: List[str] = []
        self.risk_score = 0.0

class AURAEngine:
    """Core AURA engine for topological analysis and failure prevention"""
    
    def __init__(self):
        self.failure_patterns = self._load_failure_patterns()
        self.interventions_made = 0
        self.failures_prevented = 0
        
    def _load_failure_patterns(self) -> List[Dict]:
        """Load known failure patterns"""
        return [
            {
                "name": "cascade_overload",
                "signature": {"high_load_cluster": True, "bottleneck_count": 2},
                "risk": 0.8
            },
            {
                "name": "island_formation",
                "signature": {"decreasing_connectivity": True, "components": 2},
                "risk": 0.7
            },
            {
                "name": "hub_failure",
                "signature": {"high_degree_failing": True, "degree": 5},
                "risk": 0.9
            }
        ]
    
    async def analyze_topology(self, agents: Dict[str, Agent]) -> TopologicalSignature:
        """Extract topological features from agent network"""
        sig = TopologicalSignature()
        
        # Calculate connected components
        components = self._find_components(agents)
        sig.connected_components = len(components)
        
        # Calculate average degree
        degrees = [len(agent.connections) for agent in agents.values()]
        sig.average_degree = sum(degrees) / len(degrees) if degrees else 0
        
        # Find bottlenecks (agents whose removal would disconnect the graph)
        sig.bottlenecks = self._find_bottlenecks(agents)
        
        # Calculate clustering coefficient
        sig.clustering_coefficient = self._clustering_coefficient(agents)
        
        # Calculate risk score based on patterns
        sig.risk_score = self._calculate_risk(agents, sig)
        
        return sig
    
    def _find_components(self, agents: Dict[str, Agent]) -> List[Set[str]]:
        """Find connected components using DFS"""
        visited = set()
        components = []
        
        for agent_id in agents:
            if agent_id not in visited:
                component = set()
                stack = [agent_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        agent = agents.get(current)
                        if agent:
                            for conn in agent.connections:
                                if conn not in visited and conn in agents:
                                    stack.append(conn)
                
                components.append(component)
        
        return components
    
    def _find_bottlenecks(self, agents: Dict[str, Agent]) -> List[str]:
        """Find critical agents whose failure would partition the network"""
        bottlenecks = []
        original_components = len(self._find_components(agents))
        
        for agent_id, agent in agents.items():
            if agent.state == AgentState.FAILED:
                continue
                
            # Temporarily remove agent
            temp_agents = {k: v for k, v in agents.items() if k != agent_id}
            
            # Remove connections to this agent
            for other in temp_agents.values():
                other.connections.discard(agent_id)
            
            # Check if removal increases components
            new_components = len(self._find_components(temp_agents))
            if new_components > original_components:
                bottlenecks.append(agent_id)
            
            # Restore connections
            for other in temp_agents.values():
                if agent_id in agents[other.id].connections:
                    other.connections.add(agent_id)
        
        return bottlenecks
    
    def _clustering_coefficient(self, agents: Dict[str, Agent]) -> float:
        """Calculate average clustering coefficient"""
        coefficients = []
        
        for agent in agents.values():
            if len(agent.connections) < 2:
                continue
                
            neighbors = list(agent.connections)
            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            actual_connections = 0
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in agents[neighbors[i]].connections:
                        actual_connections += 1
            
            if possible_connections > 0:
                coefficients.append(actual_connections / possible_connections)
        
        return sum(coefficients) / len(coefficients) if coefficients else 0
    
    def _calculate_risk(self, agents: Dict[str, Agent], sig: TopologicalSignature) -> float:
        """Calculate failure risk based on topological patterns"""
        risk = 0.0
        
        # Check for high load clusters
        high_load_agents = [a for a in agents.values() if a.load > 0.8]
        if len(high_load_agents) > 3:
            risk += 0.3
        
        # Check for bottlenecks under stress
        for bottleneck_id in sig.bottlenecks:
            if agents[bottleneck_id].load > 0.7:
                risk += 0.2
        
        # Check for decreasing connectivity
        if sig.average_degree < 3:
            risk += 0.2
        
        # Check for multiple components (network fragmenting)
        if sig.connected_components > 1:
            risk += 0.3
        
        return min(risk, 1.0)
    
    async def predict_failure(self, agents: Dict[str, Agent], sig: TopologicalSignature) -> Optional[Dict]:
        """Predict impending failures based on topology"""
        if sig.risk_score > 0.6:
            # Identify most likely failure point
            risk_agents = []
            
            for agent_id, agent in agents.items():
                if agent.state != AgentState.HEALTHY:
                    continue
                    
                agent_risk = agent.failure_probability
                
                # Bottlenecks are higher risk
                if agent_id in sig.bottlenecks:
                    agent_risk *= 2
                
                # High load agents are higher risk
                if agent.load > 0.8:
                    agent_risk *= 1.5
                    
                risk_agents.append((agent_id, agent_risk))
            
            if risk_agents:
                risk_agents.sort(key=lambda x: x[1], reverse=True)
                return {
                    "at_risk_agents": risk_agents[:5],
                    "failure_type": "cascade_likely",
                    "time_to_failure": random.randint(5, 20),
                    "impact": "high"
                }
        
        return None
    
    async def prevent_failure(self, agents: Dict[str, Agent], prediction: Dict) -> Dict:
        """Intervene to prevent predicted failure"""
        interventions = []
        
        # Redistribute load from at-risk agents
        for agent_id, risk in prediction["at_risk_agents"]:
            agent = agents[agent_id]
            
            # Find healthy neighbors
            healthy_neighbors = [
                n for n in agent.connections 
                if n in agents and agents[n].state == AgentState.HEALTHY 
                and agents[n].load < 0.6
            ]
            
            if healthy_neighbors:
                # Redistribute load
                load_transfer = min(0.2, agent.load - 0.6)
                agent.load -= load_transfer
                
                for neighbor_id in healthy_neighbors[:3]:
                    agents[neighbor_id].load += load_transfer / 3
                
                interventions.append({
                    "type": "load_balance",
                    "from": agent_id,
                    "to": healthy_neighbors[:3],
                    "amount": load_transfer
                })
        
        # Add redundant connections for bottlenecks
        topology = await self.analyze_topology(agents)
        for bottleneck_id in topology.bottlenecks[:2]:
            bottleneck = agents[bottleneck_id]
            
            # Find nearby agents to create redundant paths
            nearby = self._find_nearby_agents(agents, bottleneck_id, radius=0.3)
            for other_id in nearby[:2]:
                if other_id not in bottleneck.connections:
                    bottleneck.add_connection(other_id)
                    agents[other_id].add_connection(bottleneck_id)
                    
                    interventions.append({
                        "type": "add_connection",
                        "from": bottleneck_id,
                        "to": other_id
                    })
        
        self.interventions_made += 1
        
        return {
            "interventions": interventions,
            "risk_reduced_to": max(0, prediction["at_risk_agents"][0][1] - 0.5)
        }
    
    def _find_nearby_agents(self, agents: Dict[str, Agent], agent_id: str, radius: float) -> List[str]:
        """Find agents within a certain distance"""
        target = agents[agent_id]
        nearby = []
        
        for other_id, other in agents.items():
            if other_id != agent_id:
                dist = math.sqrt((target.x - other.x)**2 + (target.y - other.y)**2)
                if dist <= radius:
                    nearby.append(other_id)
        
        return nearby

class MultiAgentSystem:
    """Simulates a multi-agent system"""
    
    def __init__(self, num_agents: int = 20):
        self.agents = self._create_agents(num_agents)
        self.time_step = 0
        self.failures = []
        self.aura_enabled = False
        
    def _create_agents(self, num_agents: int) -> Dict[str, Agent]:
        """Create agents with random positions and connections"""
        agents = {}
        
        # Create agents in a rough grid
        grid_size = int(math.sqrt(num_agents))
        for i in range(num_agents):
            x = (i % grid_size) / grid_size + random.uniform(-0.1, 0.1)
            y = (i // grid_size) / grid_size + random.uniform(-0.1, 0.1)
            
            agent = Agent(f"agent_{i}", x, y)
            agents[agent.id] = agent
        
        # Create connections (nearby agents more likely to connect)
        for agent_id, agent in agents.items():
            num_connections = random.randint(2, 5)
            nearby = sorted(agents.items(), 
                          key=lambda a: math.sqrt((a[1].x-agent.x)**2 + (a[1].y-agent.y)**2))
            
            for other_id, other in nearby[1:num_connections+1]:
                agent.add_connection(other_id)
                other.add_connection(agent_id)
        
        return agents
    
    async def simulate_step(self) -> Dict:
        """Simulate one time step"""
        self.time_step += 1
        events = []
        
        # Update agent loads
        for agent in self.agents.values():
            if agent.state == AgentState.HEALTHY:
                agent.update_load()
        
        # Check for failures
        for agent in self.agents.values():
            if agent.state == AgentState.HEALTHY:
                if random.random() < agent.failure_probability:
                    agent.state = AgentState.FAILING
                    events.append({
                        "type": "agent_failing",
                        "agent_id": agent.id,
                        "time": self.time_step
                    })
        
        # Process failing agents
        for agent in self.agents.values():
            if agent.state == AgentState.FAILING:
                # Propagate stress to connected agents
                for conn_id in agent.connections:
                    if conn_id in self.agents:
                        conn_agent = self.agents[conn_id]
                        if conn_agent.state == AgentState.HEALTHY:
                            conn_agent.load += 0.1
                            conn_agent.failure_probability *= 1.5
                
                # Agent fails completely
                agent.state = AgentState.FAILED
                self.failures.append(agent.id)
                events.append({
                    "type": "agent_failed",
                    "agent_id": agent.id,
                    "time": self.time_step
                })
        
        return {
            "time_step": self.time_step,
            "events": events,
            "healthy_agents": sum(1 for a in self.agents.values() if a.state == AgentState.HEALTHY),
            "failed_agents": len(self.failures)
        }

# Initialize systems
system = MultiAgentSystem(num_agents=30)
aura_engine = AURAEngine()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time system monitoring via WebSocket"""
    await websocket.accept()
    
    # Reset system
    global system, aura_engine
    system = MultiAgentSystem(num_agents=30)
    aura_engine = AURAEngine()
    
    try:
        while True:
            # Get current topology
            topology = await aura_engine.analyze_topology(system.agents)
            
            # AURA prediction and prevention
            prediction = None
            intervention = None
            
            if system.aura_enabled:
                prediction = await aura_engine.predict_failure(system.agents, topology)
                if prediction and topology.risk_score > 0.7:
                    intervention = await aura_engine.prevent_failure(system.agents, prediction)
            
            # Simulate system
            step_result = await system.simulate_step()
            
            # Send update
            await websocket.send_json({
                "agents": [
                    {
                        "id": agent.id,
                        "x": agent.x,
                        "y": agent.y,
                        "state": agent.state,
                        "load": agent.load,
                        "connections": list(agent.connections)
                    }
                    for agent in system.agents.values()
                ],
                "topology": {
                    "risk_score": topology.risk_score,
                    "components": topology.connected_components,
                    "bottlenecks": topology.bottlenecks,
                    "avg_degree": topology.average_degree
                },
                "prediction": prediction,
                "intervention": intervention,
                "metrics": step_result,
                "aura_enabled": system.aura_enabled
            })
            
            await asyncio.sleep(1)
    except:
        pass

@app.post("/toggle_aura")
async def toggle_aura():
    """Toggle AURA protection on/off"""
    system.aura_enabled = not system.aura_enabled
    return {"aura_enabled": system.aura_enabled}

@app.post("/reset")
async def reset_system():
    """Reset the simulation"""
    global system, aura_engine
    system = MultiAgentSystem(num_agents=30)
    aura_engine = AURAEngine()
    return {"status": "reset"}

@app.get("/")
async def home():
    """Agent failure prevention dashboard"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AURA Agent Failure Prevention</title>
        <style>
            body { font-family: Arial; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
            .container { max-width: 1600px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 20px; }
            .main-grid { display: grid; grid-template-columns: 800px 1fr; gap: 20px; }
            .visualization { background: #0a0a0a; border: 2px solid #333; border-radius: 10px; padding: 20px; }
            .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; }
            .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 20px; }
            .metric { background: #1a1a1a; padding: 15px; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
            #network-canvas { border: 1px solid #444; background: #000; }
            button { background: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; font-size: 16px; }
            button:hover { background: #45a049; }
            .aura-off button { background: #f44336; }
            .aura-off button:hover { background: #da190b; }
            .warning { color: #ff9800; }
            .danger { color: #f44336; }
            .safe { color: #4CAF50; }
            .prediction { background: #3a3a0a; border: 1px solid #ffeb3b; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .intervention { background: #0a3a0a; border: 1px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 AURA Agent Failure Prevention System</h1>
                <p>Preventing cascading failures through topological intelligence</p>
            </div>
            
            <div class="main-grid">
                <div class="visualization">
                    <h3>Multi-Agent System Topology</h3>
                    <canvas id="network-canvas" width="760" height="600"></canvas>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div>Healthy Agents</div>
                            <div class="metric-value safe" id="healthy-agents">0</div>
                        </div>
                        <div class="metric">
                            <div>Failed Agents</div>
                            <div class="metric-value danger" id="failed-agents">0</div>
                        </div>
                        <div class="metric">
                            <div>Risk Score</div>
                            <div class="metric-value" id="risk-score">0%</div>
                        </div>
                        <div class="metric">
                            <div>Network Components</div>
                            <div class="metric-value" id="components">1</div>
                        </div>
                    </div>
                </div>
                
                <div class="controls">
                    <h3>AURA Control Panel</h3>
                    <div id="aura-status" class="aura-off">
                        <button onclick="toggleAURA()">Enable AURA Protection</button>
                    </div>
                    
                    <button onclick="resetSystem()" style="margin-top: 10px;">Reset Simulation</button>
                    
                    <h4>Topology Analysis</h4>
                    <div id="topology-info">
                        <p>Bottlenecks: <span id="bottlenecks">None</span></p>
                        <p>Avg Connections: <span id="avg-degree">0</span></p>
                    </div>
                    
                    <div id="prediction-panel"></div>
                    <div id="intervention-panel"></div>
                    
                    <h4>How It Works</h4>
                    <p>1. AURA analyzes the topology in real-time</p>
                    <p>2. Detects patterns that lead to cascading failures</p>
                    <p>3. Predicts failures before they happen</p>
                    <p>4. Automatically intervenes to prevent cascade</p>
                    
                    <h4>Legend</h4>
                    <p>🟢 Healthy Agent</p>
                    <p>🟡 High Load Agent</p>
                    <p>🔴 Failed Agent</p>
                    <p>⚠️ Bottleneck Agent</p>
                </div>
            </div>
        </div>
        
        <script>
            const canvas = document.getElementById('network-canvas');
            const ctx = canvas.getContext('2d');
            const ws = new WebSocket('ws://localhost:8080/ws');
            
            let auraEnabled = false;
            let currentData = null;
            
            ws.onmessage = function(event) {
                currentData = JSON.parse(event.data);
                updateVisualization(currentData);
                updateMetrics(currentData);
                updatePanels(currentData);
            };
            
            function updateVisualization(data) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw connections
                ctx.strokeStyle = '#444';
                ctx.lineWidth = 1;
                
                for (const agent of data.agents) {
                    for (const connId of agent.connections) {
                        const other = data.agents.find(a => a.id === connId);
                        if (other) {
                            ctx.beginPath();
                            ctx.moveTo(agent.x * 700 + 30, agent.y * 540 + 30);
                            ctx.lineTo(other.x * 700 + 30, other.y * 540 + 30);
                            ctx.stroke();
                        }
                    }
                }
                
                // Draw agents
                for (const agent of data.agents) {
                    const x = agent.x * 700 + 30;
                    const y = agent.y * 540 + 30;
                    
                    // Agent color based on state
                    if (agent.state === 'failed') {
                        ctx.fillStyle = '#f44336';
                    } else if (agent.load > 0.8) {
                        ctx.fillStyle = '#ff9800';
                    } else {
                        ctx.fillStyle = '#4CAF50';
                    }
                    
                    // Draw agent
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // Mark bottlenecks
                    if (data.topology.bottlenecks.includes(agent.id)) {
                        ctx.strokeStyle = '#ffeb3b';
                        ctx.lineWidth = 3;
                        ctx.beginPath();
                        ctx.arc(x, y, 12, 0, 2 * Math.PI);
                        ctx.stroke();
                    }
                    
                    // Draw load bar
                    ctx.fillStyle = '#666';
                    ctx.fillRect(x - 10, y + 12, 20, 3);
                    ctx.fillStyle = agent.load > 0.8 ? '#ff9800' : '#4CAF50';
                    ctx.fillRect(x - 10, y + 12, 20 * agent.load, 3);
                }
            }
            
            function updateMetrics(data) {
                document.getElementById('healthy-agents').textContent = data.metrics.healthy_agents;
                document.getElementById('failed-agents').textContent = data.metrics.failed_agents;
                
                const riskPercent = (data.topology.risk_score * 100).toFixed(0);
                const riskElement = document.getElementById('risk-score');
                riskElement.textContent = riskPercent + '%';
                
                if (data.topology.risk_score > 0.7) {
                    riskElement.className = 'metric-value danger';
                } else if (data.topology.risk_score > 0.4) {
                    riskElement.className = 'metric-value warning';
                } else {
                    riskElement.className = 'metric-value safe';
                }
                
                document.getElementById('components').textContent = data.topology.components;
                document.getElementById('bottlenecks').textContent = 
                    data.topology.bottlenecks.length > 0 ? data.topology.bottlenecks.join(', ') : 'None';
                document.getElementById('avg-degree').textContent = data.topology.avg_degree.toFixed(1);
            }
            
            function updatePanels(data) {
                // Update prediction panel
                const predPanel = document.getElementById('prediction-panel');
                if (data.prediction) {
                    predPanel.innerHTML = `
                        <div class="prediction">
                            <h4>⚠️ Failure Prediction</h4>
                            <p>Type: ${data.prediction.failure_type}</p>
                            <p>Time to failure: ${data.prediction.time_to_failure}s</p>
                            <p>At risk: ${data.prediction.at_risk_agents.map(a => a[0]).join(', ')}</p>
                        </div>
                    `;
                } else {
                    predPanel.innerHTML = '';
                }
                
                // Update intervention panel
                const intPanel = document.getElementById('intervention-panel');
                if (data.intervention) {
                    intPanel.innerHTML = `
                        <div class="intervention">
                            <h4>✅ AURA Intervention</h4>
                            ${data.intervention.interventions.map(i => 
                                `<p>${i.type}: ${i.from} → ${i.to || ''}</p>`
                            ).join('')}
                        </div>
                    `;
                } else {
                    intPanel.innerHTML = '';
                }
            }
            
            async function toggleAURA() {
                const response = await fetch('/toggle_aura', { method: 'POST' });
                const result = await response.json();
                auraEnabled = result.aura_enabled;
                
                const statusDiv = document.getElementById('aura-status');
                if (auraEnabled) {
                    statusDiv.className = 'aura-on';
                    statusDiv.innerHTML = '<button onclick="toggleAURA()">Disable AURA Protection</button>';
                } else {
                    statusDiv.className = 'aura-off';
                    statusDiv.innerHTML = '<button onclick="toggleAURA()">Enable AURA Protection</button>';
                }
            }
            
            async function resetSystem() {
                await fetch('/reset', { method: 'POST' });
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    print("🚀 Starting AURA Agent Failure Prevention Demo")
    print("🧠 Demonstrates preventing cascading failures through topological intelligence")
    print("📊 Open http://localhost:8080 to see the system in action")
    print("⚡ Toggle AURA on/off to see the difference!")
    uvicorn.run(app, host="0.0.0.0", port=8080)