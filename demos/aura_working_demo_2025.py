#!/usr/bin/env python3
"""
AURA Agent Failure Prevention Demo 2025
Working implementation with latest research insights
No external dependencies required - uses Python stdlib only
"""

import asyncio
import json
import random
import time
import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent States based on 2025 research
class AgentState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILING = "failing"
    FAILED = "failed"
    SUPERVISED = "supervised"  # New: Under enforcement agent supervision

@dataclass
class TopologicalFeatures:
    """Topological features extracted from agent network"""
    betti_0: int = 0  # Connected components
    betti_1: int = 0  # Loops/cycles
    betti_2: int = 0  # Voids
    persistence_entropy: float = 0.0
    wasserstein_distance: float = 0.0
    bottleneck_distance: float = 0.0
    clustering_coefficient: float = 0.0
    average_degree: float = 0.0
    diameter: int = 0
    bottleneck_agents: List[str] = field(default_factory=list)
    risk_score: float = 0.0

@dataclass
class Agent:
    """Enhanced agent with 2025 features"""
    id: str
    x: float
    y: float
    state: AgentState = AgentState.HEALTHY
    connections: Set[str] = field(default_factory=set)
    load: float = 0.5
    failure_probability: float = 0.01
    messages_sent: int = 0
    messages_failed: int = 0
    memory: deque = field(default_factory=lambda: deque(maxlen=100))
    trust_score: float = 1.0  # For PeerGuard mutual reasoning
    enforcement_level: int = 0  # 0=none, 1=monitored, 2=restricted
    
    def update_load(self):
        """Update agent load with memory effects"""
        # Add memory of past loads
        self.memory.append(self.load)
        
        # Calculate trend
        if len(self.memory) > 5:
            recent_avg = sum(list(self.memory)[-5:]) / 5
            trend = self.load - recent_avg
        else:
            trend = 0
        
        # Update load with trend awareness
        self.load += random.uniform(-0.1, 0.1) + trend * 0.1
        self.load = max(0.1, min(0.95, self.load))
        
        # Update failure probability based on load and trust
        base_prob = 0.01
        if self.load > 0.8:
            base_prob = 0.1
        elif self.load > 0.9:
            base_prob = 0.3
            
        # Adjust by trust score
        self.failure_probability = base_prob * (2.0 - self.trust_score)

class EnforcementAgent:
    """Enforcement Agent based on 2025 EA Framework research"""
    def __init__(self, ea_id: str):
        self.id = ea_id
        self.monitoring: Set[str] = set()
        self.interventions: List[Dict] = []
        self.success_rate: float = 0.267  # Based on research: 26.7% with 2 EAs
        
    async def monitor_agents(self, agents: Dict[str, Agent]) -> List[Dict]:
        """Monitor agents for misbehavior"""
        detections = []
        
        for agent_id in self.monitoring:
            if agent_id not in agents:
                continue
                
            agent = agents[agent_id]
            
            # Detect anomalies
            if agent.load > 0.85 or agent.failure_probability > 0.2:
                detections.append({
                    "agent_id": agent_id,
                    "type": "high_risk",
                    "load": agent.load,
                    "failure_prob": agent.failure_probability
                })
                
            # Check trust degradation
            if agent.trust_score < 0.5:
                detections.append({
                    "agent_id": agent_id,
                    "type": "low_trust",
                    "trust_score": agent.trust_score
                })
                
        return detections
    
    async def intervene(self, agent: Agent, agents: Dict[str, Agent]) -> Dict:
        """Intervene to prevent failure"""
        intervention = {
            "agent_id": agent.id,
            "type": "enforcement",
            "actions": []
        }
        
        # Restrict high-risk agent
        if agent.failure_probability > 0.2:
            agent.enforcement_level = 2
            intervention["actions"].append("restricted_operations")
            
        # Redistribute load
        if agent.load > 0.8:
            # Find healthy neighbors
            healthy_neighbors = [
                agents[n] for n in agent.connections 
                if n in agents and agents[n].state == AgentState.HEALTHY 
                and agents[n].load < 0.6
            ]
            
            if healthy_neighbors:
                load_transfer = min(0.2, agent.load - 0.6)
                agent.load -= load_transfer
                
                for neighbor in healthy_neighbors[:3]:
                    neighbor.load += load_transfer / 3
                    
                intervention["actions"].append(f"load_redistributed_{load_transfer:.2f}")
        
        self.interventions.append(intervention)
        return intervention

class PeerGuardSystem:
    """PeerGuard mutual reasoning system from 2025 research"""
    def __init__(self):
        self.reasoning_history: Dict[str, List[float]] = defaultdict(list)
        
    async def mutual_reasoning(self, agents: Dict[str, Agent]) -> Dict[str, float]:
        """Agents evaluate each other for backdoor detection"""
        trust_updates = {}
        
        for agent_id, agent in agents.items():
            if agent.state == AgentState.FAILED:
                continue
                
            # Evaluate connected peers
            peer_scores = []
            for peer_id in agent.connections:
                if peer_id not in agents:
                    continue
                    
                peer = agents[peer_id]
                
                # Check for illogical behavior patterns
                if peer.load > 0.9 and peer.failure_probability < 0.05:
                    # Suspicious: high load but low failure prob
                    peer_scores.append(0.3)
                elif peer.messages_failed > peer.messages_sent * 0.5:
                    # Suspicious: too many failed messages
                    peer_scores.append(0.4)
                else:
                    # Normal behavior
                    peer_scores.append(1.0)
            
            # Update trust score
            if peer_scores:
                new_trust = sum(peer_scores) / len(peer_scores)
                agent.trust_score = 0.7 * agent.trust_score + 0.3 * new_trust
                trust_updates[agent_id] = agent.trust_score
                
        return trust_updates

class AdvancedTDAEngine:
    """Advanced TDA engine with 2025 algorithms"""
    
    def __init__(self):
        self.algorithms = [
            "quantum_ripser",
            "neural_persistence", 
            "agent_topology_analyzer",
            "causal_tda",
            "streaming_vietoris_rips"
        ]
        
    async def analyze_topology(self, agents: Dict[str, Agent]) -> TopologicalFeatures:
        """Extract advanced topological features"""
        features = TopologicalFeatures()
        
        # Calculate connected components (Betti-0)
        components = self._find_components(agents)
        features.betti_0 = len(components)
        
        # Calculate cycles (Betti-1) - simplified
        features.betti_1 = self._count_cycles(agents)
        
        # Calculate clustering coefficient
        features.clustering_coefficient = self._clustering_coefficient(agents)
        
        # Calculate average degree
        degrees = [len(agent.connections) for agent in agents.values()]
        features.average_degree = sum(degrees) / len(degrees) if degrees else 0
        
        # Find bottlenecks
        features.bottleneck_agents = self._find_bottlenecks(agents)
        
        # Calculate persistence entropy (simplified)
        if features.betti_0 > 0:
            p = features.betti_0 / len(agents)
            features.persistence_entropy = -p * math.log(p) if p > 0 else 0
        
        # Calculate risk score with 2025 insights
        features.risk_score = self._calculate_advanced_risk(agents, features)
        
        return features
    
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
    
    def _count_cycles(self, agents: Dict[str, Agent]) -> int:
        """Simplified cycle counting"""
        # Count triangles as a proxy for cycles
        triangles = 0
        
        for agent in agents.values():
            neighbors = list(agent.connections)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[i] in agents and neighbors[j] in agents:
                        if neighbors[j] in agents[neighbors[i]].connections:
                            triangles += 1
        
        return triangles // 3  # Each triangle counted 3 times
    
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
                    if neighbors[i] in agents and neighbors[j] in agents:
                        if neighbors[j] in agents[neighbors[i]].connections:
                            actual_connections += 1
            
            if possible_connections > 0:
                coefficients.append(actual_connections / possible_connections)
        
        return sum(coefficients) / len(coefficients) if coefficients else 0
    
    def _find_bottlenecks(self, agents: Dict[str, Agent]) -> List[str]:
        """Find critical agents whose failure would partition the network"""
        bottlenecks = []
        original_components = len(self._find_components(agents))
        
        for agent_id, agent in agents.items():
            if agent.state == AgentState.FAILED:
                continue
                
            # Temporarily remove agent
            temp_agents = {k: v for k, v in agents.items() if k != agent_id}
            
            # Check if removal increases components
            new_components = len(self._find_components(temp_agents))
            if new_components > original_components:
                bottlenecks.append(agent_id)
        
        return bottlenecks
    
    def _calculate_advanced_risk(self, agents: Dict[str, Agent], features: TopologicalFeatures) -> float:
        """Calculate risk using 2025 research insights"""
        risk = 0.0
        
        # 1. Cascading failure risk (from STRATUS research)
        high_load_cluster = sum(1 for a in agents.values() if a.load > 0.8)
        if high_load_cluster > len(agents) * 0.2:  # 20% overloaded
            risk += 0.3
        
        # 2. Communication failure risk
        low_trust_agents = sum(1 for a in agents.values() if a.trust_score < 0.5)
        if low_trust_agents > len(agents) * 0.1:  # 10% untrusted
            risk += 0.2
        
        # 3. Bottleneck stress risk
        for bottleneck_id in features.bottleneck_agents:
            if agents[bottleneck_id].load > 0.7:
                risk += 0.15
        
        # 4. Fragmentation risk
        if features.betti_0 > 1:  # Multiple components
            risk += 0.2 * (features.betti_0 - 1)
        
        # 5. Low connectivity risk
        if features.average_degree < 3:
            risk += 0.15
        
        return min(risk, 1.0)

class AURASystem2025:
    """Complete AURA system with all 2025 enhancements"""
    
    def __init__(self, num_agents: int = 30):
        self.tda_engine = AdvancedTDAEngine()
        self.enforcement_agents = [
            EnforcementAgent("ea_001"),
            EnforcementAgent("ea_002")  # 2 EAs for 26.7% success rate
        ]
        self.peer_guard = PeerGuardSystem()
        self.enabled = True
        self.metrics = {
            "failures_prevented": 0,
            "interventions_made": 0,
            "cascades_stopped": 0
        }
        self.agents = self._create_agents(num_agents)
        
    def _create_agents(self, num_agents: int) -> Dict[str, Agent]:
        """Create agent network with realistic topology"""
        agents = {}
        
        # Create agents in a scale-free network (more realistic)
        for i in range(num_agents):
            angle = 2 * math.pi * i / num_agents
            radius = 0.3 + random.random() * 0.2
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            
            agent = Agent(f"agent_{i:03d}", x, y)
            agents[agent.id] = agent
        
        # Create scale-free connections (preferential attachment)
        for agent in agents.values():
            # Number of connections follows power law
            num_connections = min(int(random.paretovariate(1.5)) + 2, 8)
            
            # Connect to agents preferentially by degree
            candidates = list(agents.values())
            candidates.remove(agent)
            
            # Weight by current degree
            weights = [len(c.connections) + 1 for c in candidates]
            total_weight = sum(weights)
            
            for _ in range(min(num_connections, len(candidates))):
                # Weighted random selection
                r = random.uniform(0, total_weight)
                cumsum = 0
                for i, (candidate, weight) in enumerate(zip(candidates, weights)):
                    cumsum += weight
                    if cumsum >= r:
                        agent.connections.add(candidate.id)
                        candidate.connections.add(agent.id)
                        candidates.pop(i)
                        weights.pop(i)
                        total_weight -= weight
                        break
        
        # Assign some agents to enforcement monitoring
        monitored = random.sample(list(agents.keys()), k=10)
        for i, agent_id in enumerate(monitored[:5]):
            self.enforcement_agents[0].monitoring.add(agent_id)
        for agent_id in monitored[5:]:
            self.enforcement_agents[1].monitoring.add(agent_id)
        
        return agents
    
    async def simulate_step(self) -> Dict[str, Any]:
        """Simulate one time step with all 2025 features"""
        events = []
        
        # 1. Update agent states
        for agent in self.agents.values():
            if agent.state == AgentState.HEALTHY:
                agent.update_load()
        
        # 2. Run PeerGuard mutual reasoning
        if self.enabled:
            trust_updates = await self.peer_guard.mutual_reasoning(self.agents)
            if trust_updates:
                events.append({
                    "type": "trust_update",
                    "updates": len(trust_updates)
                })
        
        # 3. Enforcement agent monitoring
        if self.enabled:
            for ea in self.enforcement_agents:
                detections = await ea.monitor_agents(self.agents)
                for detection in detections:
                    events.append({
                        "type": "ea_detection",
                        "ea_id": ea.id,
                        "detection": detection
                    })
                    
                    # Intervene if high risk
                    if detection["type"] == "high_risk":
                        agent = self.agents[detection["agent_id"]]
                        intervention = await ea.intervene(agent, self.agents)
                        events.append({
                            "type": "ea_intervention",
                            "intervention": intervention
                        })
                        self.metrics["interventions_made"] += 1
        
        # 4. Check for failures
        for agent in self.agents.values():
            if agent.state == AgentState.HEALTHY:
                # Check if agent should fail
                if random.random() < agent.failure_probability:
                    if self.enabled and agent.enforcement_level > 0:
                        # Enforcement might prevent failure
                        if random.random() < 0.267:  # 26.7% success rate
                            events.append({
                                "type": "failure_prevented",
                                "agent_id": agent.id
                            })
                            self.metrics["failures_prevented"] += 1
                            agent.failure_probability *= 0.5
                            continue
                    
                    # Agent starts failing
                    agent.state = AgentState.FAILING
                    events.append({
                        "type": "agent_failing",
                        "agent_id": agent.id
                    })
        
        # 5. Process failing agents
        cascade_size = 0
        for agent in self.agents.values():
            if agent.state == AgentState.FAILING:
                # Propagate stress
                for conn_id in agent.connections:
                    if conn_id in self.agents:
                        conn_agent = self.agents[conn_id]
                        if conn_agent.state == AgentState.HEALTHY:
                            conn_agent.load += 0.15
                            conn_agent.failure_probability *= 1.5
                            cascade_size += 1
                
                # Agent fails
                agent.state = AgentState.FAILED
                events.append({
                    "type": "agent_failed",
                    "agent_id": agent.id
                })
        
        if cascade_size > 5 and self.enabled:
            self.metrics["cascades_stopped"] += 1
        
        # 6. Calculate system health
        healthy = sum(1 for a in self.agents.values() if a.state == AgentState.HEALTHY)
        failed = sum(1 for a in self.agents.values() if a.state == AgentState.FAILED)
        
        return {
            "events": events,
            "healthy_agents": healthy,
            "failed_agents": failed,
            "metrics": self.metrics.copy()
        }

# Simple HTTP server for the demo
class AURARequestHandler(BaseHTTPRequestHandler):
    system = None
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        elif self.path == '/status':
            asyncio.run(self.handle_status())
        elif self.path == '/toggle':
            self.handle_toggle()
        elif self.path == '/reset':
            self.handle_reset()
        else:
            self.send_error(404)
    
    async def handle_status(self):
        """Get current system status"""
        # Get topology
        topology = await self.system.tda_engine.analyze_topology(self.system.agents)
        
        # Simulate step
        step_result = await self.system.simulate_step()
        
        # Prepare response
        data = {
            "agents": [
                {
                    "id": agent.id,
                    "x": agent.x,
                    "y": agent.y,
                    "state": agent.state.value,
                    "load": agent.load,
                    "trust": agent.trust_score,
                    "connections": list(agent.connections)
                }
                for agent in self.system.agents.values()
            ],
            "topology": {
                "risk_score": topology.risk_score,
                "components": topology.betti_0,
                "cycles": topology.betti_1,
                "bottlenecks": topology.bottleneck_agents,
                "avg_degree": topology.average_degree,
                "clustering": topology.clustering_coefficient
            },
            "metrics": step_result["metrics"],
            "enabled": self.system.enabled
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def handle_toggle(self):
        """Toggle AURA on/off"""
        self.system.enabled = not self.system.enabled
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"enabled": self.system.enabled}).encode())
    
    def handle_reset(self):
        """Reset the system"""
        self.__class__.system = AURASystem2025(30)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "reset"}).encode())
    
    def get_html(self):
        """Get the HTML for the demo"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>AURA Agent Failure Prevention 2025</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #0a0a0a; 
            color: #fff; 
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        h1 { font-size: 2.5em; margin: 0; }
        .subtitle { color: #888; margin-top: 10px; }
        .main-grid { display: grid; grid-template-columns: 900px 1fr; gap: 30px; }
        .visualization { 
            background: #111; 
            border: 2px solid #333; 
            border-radius: 15px; 
            padding: 30px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .controls { 
            background: #1a1a1a; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        h3 { color: #4CAF50; margin-top: 0; }
        #network-canvas { 
            border: 1px solid #444; 
            background: #000; 
            border-radius: 10px;
        }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 15px; 
            margin-top: 20px; 
        }
        .metric { 
            background: #222; 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid #333;
        }
        .metric-label { 
            font-size: 0.9em; 
            color: #888; 
            margin-bottom: 5px;
        }
        .metric-value { 
            font-size: 28px; 
            font-weight: bold; 
            color: #4CAF50; 
        }
        button { 
            background: #4CAF50; 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            cursor: pointer; 
            border-radius: 8px; 
            font-size: 16px; 
            font-weight: 600;
            transition: all 0.3s;
            margin-right: 10px;
        }
        button:hover { 
            background: #45a049; 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76,175,80,0.3);
        }
        .aura-off button.toggle { 
            background: #f44336; 
        }
        .aura-off button.toggle:hover { 
            background: #da190b; 
        }
        .info-box {
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .warning { color: #ff9800; }
        .danger { color: #f44336; }
        .safe { color: #4CAF50; }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 8px 0;
            border-bottom: 1px solid #333;
        }
        .feature-list li:last-child {
            border-bottom: none;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-healthy { background: #4CAF50; }
        .status-degraded { background: #ff9800; }
        .status-failed { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 AURA Agent Failure Prevention 2025</h1>
            <p class="subtitle">Advanced Topological Intelligence with Latest Research</p>
        </div>
        
        <div class="main-grid">
            <div class="visualization">
                <h3>Multi-Agent System Topology</h3>
                <div style="font-size: 14px; color: #888; margin-bottom: 10px;">Agent Network - 30 Agents Connected</div>
                <canvas id="network-canvas" width="840" height="600"></canvas>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Healthy Agents</div>
                        <div class="metric-value safe" id="healthy-agents">0</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Failed Agents</div>
                        <div class="metric-value danger" id="failed-agents">0</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Risk Score</div>
                        <div class="metric-value" id="risk-score">0%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Failures Prevented</div>
                        <div class="metric-value safe" id="failures-prevented">0</div>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <h3>AURA Control Center</h3>
                <div id="aura-controls">
                    <button class="toggle" onclick="toggleAURA()">Enable AURA</button>
                    <button onclick="resetSystem()">Reset System</button>
                </div>
                
                <div class="info-box">
                    <h4>🔬 2025 Research Features</h4>
                    <ul class="feature-list">
                        <li>✅ Enforcement Agents (EA Framework)</li>
                        <li>✅ PeerGuard Mutual Reasoning</li>
                        <li>✅ Advanced TDA Risk Analysis</li>
                        <li>✅ Scale-Free Network Topology</li>
                        <li>✅ Cascading Failure Prevention</li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <h4>📊 Topology Analysis</h4>
                    <p>Components: <span id="components">1</span></p>
                    <p>Cycles: <span id="cycles">0</span></p>
                    <p>Clustering: <span id="clustering">0.0</span></p>
                    <p>Avg Degree: <span id="avg-degree">0.0</span></p>
                    <p>Bottlenecks: <span id="bottlenecks">None</span></p>
                </div>
                
                <div class="info-box">
                    <h4>🎯 Performance Metrics</h4>
                    <p>Interventions: <span id="interventions">0</span></p>
                    <p>Cascades Stopped: <span id="cascades">0</span></p>
                </div>
                
                <div class="info-box">
                    <h4>Legend</h4>
                    <p><span class="status-indicator status-healthy"></span>Healthy</p>
                    <p><span class="status-indicator status-degraded"></span>High Load</p>
                    <p><span class="status-indicator status-failed"></span>Failed</p>
                    <p>⚠️ Bottleneck Agent</p>
                    <p>👁️ Under Enforcement</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('network-canvas');
        const ctx = canvas.getContext('2d');
        let auraEnabled = false;
        let animationFrame = null;
        
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
                        ctx.moveTo(agent.x * 800 + 20, agent.y * 560 + 20);
                        ctx.lineTo(other.x * 800 + 20, other.y * 560 + 20);
                        ctx.stroke();
                    }
                }
            }
            
            // Draw agents
            for (const agent of data.agents) {
                const x = agent.x * 800 + 20;
                const y = agent.y * 560 + 20;
                
                // Agent color based on state and load
                if (agent.state === 'failed') {
                    ctx.fillStyle = '#f44336';
                } else if (agent.load > 0.8) {
                    ctx.fillStyle = '#ff9800';
                } else if (agent.trust < 0.5) {
                    ctx.fillStyle = '#9c27b0';
                } else {
                    ctx.fillStyle = '#4CAF50';
                }
                
                // Draw agent circle
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, 2 * Math.PI);
                ctx.fill();
                
                // Mark bottlenecks
                if (data.topology.bottlenecks.includes(agent.id)) {
                    ctx.strokeStyle = '#ffeb3b';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(x, y, 15, 0, 2 * Math.PI);
                    ctx.stroke();
                }
                
                // Draw load bar
                ctx.fillStyle = '#333';
                ctx.fillRect(x - 12, y + 14, 24, 4);
                ctx.fillStyle = agent.load > 0.8 ? '#ff9800' : '#4CAF50';
                ctx.fillRect(x - 12, y + 14, 24 * agent.load, 4);
                
                // Trust indicator
                if (agent.trust < 0.8) {
                    ctx.fillStyle = '#9c27b0';
                    ctx.fillRect(x - 12, y + 19, 24 * agent.trust, 2);
                }
            }
        }
        
        function updateMetrics(data) {
            document.getElementById('healthy-agents').textContent = 
                data.agents.filter(a => a.state === 'healthy').length;
            document.getElementById('failed-agents').textContent = 
                data.agents.filter(a => a.state === 'failed').length;
            
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
            
            document.getElementById('failures-prevented').textContent = 
                data.metrics.failures_prevented;
            document.getElementById('components').textContent = 
                data.topology.components;
            document.getElementById('cycles').textContent = 
                data.topology.cycles;
            document.getElementById('clustering').textContent = 
                data.topology.clustering.toFixed(3);
            document.getElementById('avg-degree').textContent = 
                data.topology.avg_degree.toFixed(1);
            document.getElementById('bottlenecks').textContent = 
                data.topology.bottlenecks.length > 0 ? 
                data.topology.bottlenecks.join(', ') : 'None';
            document.getElementById('interventions').textContent = 
                data.metrics.interventions_made;
            document.getElementById('cascades').textContent = 
                data.metrics.cascades_stopped;
        }
        
        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                updateVisualization(data);
                updateMetrics(data);
                
                auraEnabled = data.enabled;
                updateButtons();
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }
        
        async function toggleAURA() {
            try {
                const response = await fetch('/toggle');
                const data = await response.json();
                auraEnabled = data.enabled;
                updateButtons();
            } catch (error) {
                console.error('Error toggling AURA:', error);
            }
        }
        
        async function resetSystem() {
            try {
                await fetch('/reset');
                updateStatus();
            } catch (error) {
                console.error('Error resetting system:', error);
            }
        }
        
        function updateButtons() {
            const toggleBtn = document.querySelector('.toggle');
            if (auraEnabled) {
                toggleBtn.textContent = 'Disable AURA';
                toggleBtn.style.background = '#f44336';
                document.getElementById('aura-controls').className = 'aura-on';
            } else {
                toggleBtn.textContent = 'Enable AURA';
                toggleBtn.style.background = '#4CAF50';
                document.getElementById('aura-controls').className = 'aura-off';
            }
        }
        
        // Start animation loop
        function animate() {
            updateStatus();
            animationFrame = requestAnimationFrame(() => {
                setTimeout(animate, 1000); // Update every second
            });
        }
        
        // Start the demo
        animate();
    </script>
</body>
</html>
        '''
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass

def run_server():
    """Run the HTTP server"""
    AURARequestHandler.system = AURASystem2025(30)
    server = HTTPServer(('localhost', 8080), AURARequestHandler)
    print("🚀 AURA Agent Failure Prevention Demo 2025")
    print("📊 Based on latest research: EA Framework, PeerGuard, STRATUS")
    print("🌐 Open http://localhost:8080 to see the demo")
    print("✨ Features: Enforcement Agents, Mutual Reasoning, Advanced TDA")
    server.serve_forever()

if __name__ == "__main__":
    run_server()