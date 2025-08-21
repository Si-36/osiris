"""
Interactive Demo Framework for AURA Intelligence
Production-ready demo system with real-time visualization

2025 Best Practices:
- Interactive playground environments
- Real-time data streaming
- Visual analytics
- Scenario-based demonstrations
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = structlog.get_logger()


@dataclass
class DemoScenario:
    """Definition of a demo scenario"""
    name: str
    description: str
    services: List[str]
    steps: List[Dict[str, Any]]
    expected_outcomes: List[str]
    visualizations: List[str]
    duration_seconds: int = 60
    
    
@dataclass 
class DemoMetrics:
    """Real-time metrics for demo"""
    timestamp: float = field(default_factory=time.time)
    service_latencies: Dict[str, float] = field(default_factory=dict)
    energy_consumption: float = 0.0
    accuracy_scores: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0
    error_rate: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class InteractiveDemoRunner:
    """
    Orchestrates interactive demos with real-time feedback
    """
    
    def __init__(self, service_urls: Dict[str, str]):
        self.service_urls = service_urls
        self.logger = logger.bind(component="demo_runner")
        self.active_demos: Dict[str, Any] = {}
        self.metrics_history: List[DemoMetrics] = []
        self.websocket_clients: List[WebSocket] = []
        
    async def run_scenario(self, scenario: DemoScenario) -> Dict[str, Any]:
        """Run a complete demo scenario"""
        demo_id = f"{scenario.name}_{int(time.time())}"
        
        self.logger.info(f"Starting demo scenario: {scenario.name}")
        
        demo_state = {
            "id": demo_id,
            "scenario": scenario,
            "start_time": time.time(),
            "current_step": 0,
            "results": [],
            "metrics": [],
            "status": "running"
        }
        
        self.active_demos[demo_id] = demo_state
        
        try:
            # Execute each step
            for i, step in enumerate(scenario.steps):
                demo_state["current_step"] = i
                
                # Broadcast progress
                await self._broadcast_update({
                    "type": "step_progress",
                    "demo_id": demo_id,
                    "step": i,
                    "total_steps": len(scenario.steps),
                    "description": step.get("description", f"Step {i+1}")
                })
                
                # Execute step
                result = await self._execute_step(step, demo_state)
                demo_state["results"].append(result)
                
                # Collect metrics
                metrics = await self._collect_metrics(scenario.services)
                demo_state["metrics"].append(metrics)
                self.metrics_history.append(metrics)
                
                # Broadcast metrics
                await self._broadcast_metrics(demo_id, metrics)
                
                # Delay between steps for visibility
                await asyncio.sleep(step.get("delay", 2))
            
            demo_state["status"] = "completed"
            demo_state["end_time"] = time.time()
            
            # Generate summary
            summary = self._generate_summary(demo_state)
            
            await self._broadcast_update({
                "type": "demo_complete",
                "demo_id": demo_id,
                "summary": summary
            })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Demo failed: {str(e)}")
            demo_state["status"] = "failed"
            demo_state["error"] = str(e)
            
            await self._broadcast_update({
                "type": "demo_error",
                "demo_id": demo_id,
                "error": str(e)
            })
            
            raise
    
    async def _execute_step(self, step: Dict[str, Any], demo_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single demo step"""
        step_type = step["type"]
        result = {
            "step_type": step_type,
            "timestamp": time.time(),
            "success": False
        }
        
        try:
            if step_type == "neuromorphic_processing":
                result.update(await self._demo_neuromorphic(step))
                
            elif step_type == "memory_storage":
                result.update(await self._demo_memory(step))
                
            elif step_type == "consensus_decision":
                result.update(await self._demo_consensus(step))
                
            elif step_type == "adaptive_learning":
                result.update(await self._demo_lnn(step))
                
            elif step_type == "intelligent_routing":
                result.update(await self._demo_routing(step))
                
            elif step_type == "custom":
                # Execute custom function
                func = step["function"]
                result.update(await func(self.service_urls, step.get("params", {})))
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Step failed: {step_type}", error=str(e))
        
        return result
    
    async def _demo_neuromorphic(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Demo neuromorphic processing"""
        async with httpx.AsyncClient() as client:
            # Generate spike data
            spike_data = step.get("spike_data", [
                [random.randint(0, 1) for _ in range(125)] 
                for _ in range(10)
            ])
            
            response = await client.post(
                f"{self.service_urls['neuromorphic']}/api/v1/process/spike",
                json={
                    "spike_data": spike_data,
                    "time_steps": step.get("time_steps", 10)
                }
            )
            
            result = response.json()
            
            return {
                "service": "neuromorphic",
                "energy_consumed_pj": result["energy_consumed_pj"],
                "latency_us": result["latency_us"],
                "spike_output": result["spike_output"][:5]  # Sample
            }
    
    async def _demo_memory(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Demo memory storage with shape analysis"""
        async with httpx.AsyncClient() as client:
            data = step.get("data", {
                "demo_data": [random.random() for _ in range(100)],
                "metadata": {"source": "demo", "timestamp": time.time()}
            })
            
            response = await client.post(
                f"{self.service_urls['memory']}/api/v1/store",
                json={
                    "data": data,
                    "enable_shape_analysis": True
                }
            )
            
            result = response.json()
            
            return {
                "service": "memory",
                "memory_id": result["memory_id"],
                "tier": result["tier"],
                "shape_analysis": result.get("shape_analysis", {})
            }
    
    async def _demo_consensus(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Demo Byzantine consensus"""
        async with httpx.AsyncClient() as client:
            # Quick consensus demo
            nodes = ["demo_alpha", "demo_beta", "demo_gamma", "demo_delta"]
            
            # Register nodes
            for node in nodes:
                await client.post(
                    f"{self.service_urls['byzantine']}/api/v1/nodes/register",
                    json={"node_id": node, "address": f"http://{node}:8002"}
                )
            
            # Propose
            propose_resp = await client.post(
                f"{self.service_urls['byzantine']}/api/v1/consensus/propose",
                json={
                    "node_id": nodes[0],
                    "topic": step.get("topic", "demo_decision"),
                    "value": step.get("value", {"decision": "approve"})
                }
            )
            
            proposal_id = propose_resp.json()["proposal_id"]
            
            # Vote
            for node in nodes:
                await client.post(
                    f"{self.service_urls['byzantine']}/api/v1/consensus/vote",
                    json={
                        "node_id": node,
                        "proposal_id": proposal_id,
                        "vote_value": {"decision": "approve"},
                        "confidence": 0.9
                    }
                )
            
            await asyncio.sleep(1)
            
            # Get result
            state_resp = await client.get(
                f"{self.service_urls['byzantine']}/api/v1/consensus/state/{proposal_id}"
            )
            
            state = state_resp.json()
            
            return {
                "service": "byzantine",
                "proposal_id": proposal_id,
                "consensus_reached": state["status"] == "DECIDED",
                "final_decision": state.get("final_decision"),
                "confidence": state.get("final_confidence", 0)
            }
    
    async def _demo_lnn(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Demo adaptive learning"""
        async with httpx.AsyncClient() as client:
            # Inference
            input_data = step.get("input_data", [0.1 * i for i in range(128)])
            
            inference_resp = await client.post(
                f"{self.service_urls['lnn']}/api/v1/inference",
                json={
                    "model_id": "adaptive",
                    "input_data": input_data,
                    "return_dynamics": True
                }
            )
            
            inference_result = inference_resp.json()
            
            # Adaptation
            if step.get("adapt", True):
                adapt_resp = await client.post(
                    f"{self.service_urls['lnn']}/api/v1/adapt",
                    json={
                        "model_id": "adaptive",
                        "feedback_signal": step.get("feedback", 0.8),
                        "adaptation_strength": 0.1
                    }
                )
                
                adapt_result = adapt_resp.json()
            else:
                adapt_result = {}
            
            return {
                "service": "lnn",
                "inference_latency_ms": inference_result["latency_ms"],
                "adaptations": inference_result.get("adaptations", {}),
                "dynamics": inference_result.get("dynamics", {}),
                "adaptation_success": adapt_result.get("success", False)
            }
    
    async def _demo_routing(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Demo intelligent routing"""
        async with httpx.AsyncClient() as client:
            request_data = step.get("request_data", {
                "type": "inference",
                "data": [random.random() for _ in range(100)],
                "priority": 0.8,
                "complexity": 0.6
            })
            
            response = await client.post(
                f"{self.service_urls['moe']}/api/v1/route",
                json={
                    "data": request_data,
                    "routing_strategy": step.get("strategy")
                }
            )
            
            result = response.json()
            
            return {
                "service": "moe",
                "selected_services": result["selected_services"],
                "routing_strategy": result["routing_strategy"],
                "confidence_scores": result["confidence_scores"],
                "reasoning": result["reasoning"],
                "latency_ms": result["latency_ms"]
            }
    
    async def _collect_metrics(self, services: List[str]) -> DemoMetrics:
        """Collect real-time metrics from services"""
        metrics = DemoMetrics()
        
        async with httpx.AsyncClient() as client:
            # Collect latencies
            for service in services:
                if service in self.service_urls:
                    try:
                        start = time.perf_counter()
                        response = await client.get(
                            f"{self.service_urls[service]}/api/v1/health",
                            timeout=2.0
                        )
                        latency = (time.perf_counter() - start) * 1000
                        
                        if response.status_code == 200:
                            metrics.service_latencies[service] = latency
                            
                            # Extract service-specific metrics
                            health_data = response.json()
                            
                            if service == "neuromorphic" and "avg_energy_pj" in health_data:
                                metrics.energy_consumption = health_data["avg_energy_pj"]
                            
                    except Exception:
                        metrics.service_latencies[service] = -1
        
        # Calculate aggregate metrics
        if metrics.service_latencies:
            valid_latencies = [l for l in metrics.service_latencies.values() if l > 0]
            if valid_latencies:
                metrics.throughput = 1000 / np.mean(valid_latencies)  # Rough estimate
        
        return metrics
    
    async def _broadcast_update(self, update: Dict[str, Any]):
        """Broadcast update to all WebSocket clients"""
        message = json.dumps(update)
        
        for client in self.websocket_clients:
            try:
                await client.send_text(message)
            except Exception:
                # Client disconnected
                self.websocket_clients.remove(client)
    
    async def _broadcast_metrics(self, demo_id: str, metrics: DemoMetrics):
        """Broadcast metrics update"""
        await self._broadcast_update({
            "type": "metrics_update",
            "demo_id": demo_id,
            "timestamp": metrics.timestamp,
            "latencies": metrics.service_latencies,
            "energy_pj": metrics.energy_consumption,
            "throughput": metrics.throughput
        })
    
    def _generate_summary(self, demo_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo summary"""
        scenario = demo_state["scenario"]
        results = demo_state["results"]
        metrics = demo_state["metrics"]
        
        # Calculate statistics
        total_energy = sum(
            r.get("energy_consumed_pj", 0) 
            for r in results 
            if "energy_consumed_pj" in r
        )
        
        avg_latency = np.mean([
            m.service_latencies.get(s, 0)
            for m in metrics
            for s in m.service_latencies
            if m.service_latencies.get(s, 0) > 0
        ])
        
        success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
        
        return {
            "demo_id": demo_state["id"],
            "scenario_name": scenario.name,
            "duration_seconds": demo_state.get("end_time", time.time()) - demo_state["start_time"],
            "steps_completed": len(results),
            "success_rate": success_rate,
            "total_energy_pj": total_energy,
            "avg_latency_ms": avg_latency,
            "services_used": list(set(r.get("service", "") for r in results if "service" in r)),
            "key_results": self._extract_key_results(results)
        }
    
    def _extract_key_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract key results for summary"""
        key_results = []
        
        for result in results:
            if result.get("service") == "neuromorphic" and "energy_consumed_pj" in result:
                key_results.append(
                    f"Neuromorphic processing: {result['energy_consumed_pj']:.1f} pJ"
                )
            
            elif result.get("service") == "byzantine" and "consensus_reached" in result:
                if result["consensus_reached"]:
                    key_results.append(
                        f"Consensus reached with {result.get('confidence', 0):.2%} confidence"
                    )
            
            elif result.get("service") == "lnn" and "adaptation_success" in result:
                if result["adaptation_success"]:
                    key_results.append("LNN successfully adapted to feedback")
            
            elif result.get("service") == "moe" and "selected_services" in result:
                key_results.append(
                    f"Routed to {', '.join(result['selected_services'])} using {result.get('routing_strategy', 'unknown')}"
                )
        
        return key_results
    
    def create_visualization(self, demo_id: str, viz_type: str) -> go.Figure:
        """Create interactive visualizations"""
        demo_state = self.active_demos.get(demo_id)
        if not demo_state:
            return None
        
        metrics = demo_state["metrics"]
        
        if viz_type == "latency_timeline":
            return self._create_latency_timeline(metrics)
        elif viz_type == "energy_consumption":
            return self._create_energy_chart(metrics, demo_state["results"])
        elif viz_type == "service_flow":
            return self._create_service_flow(demo_state["results"])
        elif viz_type == "performance_radar":
            return self._create_performance_radar(demo_state)
        
        return None
    
    def _create_latency_timeline(self, metrics: List[DemoMetrics]) -> go.Figure:
        """Create latency timeline visualization"""
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Service Latencies Over Time",)
        )
        
        services = set()
        for m in metrics:
            services.update(m.service_latencies.keys())
        
        for service in services:
            x = [m.timestamp for m in metrics if service in m.service_latencies]
            y = [m.service_latencies[service] for m in metrics if service in m.service_latencies]
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    name=service,
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            hovermode='x unified',
            template="plotly_dark"
        )
        
        return fig
    
    def _create_energy_chart(self, metrics: List[DemoMetrics], results: List[Dict[str, Any]]) -> go.Figure:
        """Create energy consumption chart"""
        # Extract energy data
        energy_data = []
        
        for i, result in enumerate(results):
            if "energy_consumed_pj" in result:
                energy_data.append({
                    "step": i + 1,
                    "service": result.get("service", "unknown"),
                    "energy": result["energy_consumed_pj"]
                })
        
        if not energy_data:
            return None
        
        fig = px.bar(
            energy_data,
            x="step",
            y="energy",
            color="service",
            title="Energy Consumption by Step",
            labels={"energy": "Energy (pJ)", "step": "Step"}
        )
        
        fig.update_layout(template="plotly_dark")
        
        return fig
    
    def _create_service_flow(self, results: List[Dict[str, Any]]) -> go.Figure:
        """Create service flow diagram"""
        # Extract service sequence
        services = [r.get("service", "unknown") for r in results if "service" in r]
        
        # Create Sankey diagram
        source = []
        target = []
        value = []
        
        for i in range(len(services) - 1):
            source.append(i)
            target.append(i + 1)
            value.append(1)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=services,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title_text="Service Flow",
            font_size=10,
            template="plotly_dark"
        )
        
        return fig
    
    def _create_performance_radar(self, demo_state: Dict[str, Any]) -> go.Figure:
        """Create performance radar chart"""
        summary = self._generate_summary(demo_state)
        
        categories = ['Latency', 'Energy', 'Throughput', 'Accuracy', 'Reliability']
        
        # Normalize metrics to 0-100 scale
        values = [
            100 - min(summary['avg_latency_ms'], 100),  # Lower is better
            100 - min(summary['total_energy_pj'] / 10, 100),  # Lower is better
            min(summary.get('throughput', 50), 100),
            summary['success_rate'] * 100,
            95  # Placeholder for reliability
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title="Overall Performance",
            template="plotly_dark"
        )
        
        return fig


# Pre-defined demo scenarios
def create_demo_scenarios() -> List[DemoScenario]:
    """Create pre-defined demo scenarios"""
    
    # Scenario 1: Energy-Efficient Intelligence
    energy_scenario = DemoScenario(
        name="energy_efficient_intelligence",
        description="Demonstrate ultra-low energy AI processing",
        services=["neuromorphic", "memory", "moe"],
        steps=[
            {
                "type": "neuromorphic_processing",
                "description": "Process complex spike patterns with minimal energy",
                "spike_data": [[1,0,1,0,1] * 25 for _ in range(20)],
                "time_steps": 20
            },
            {
                "type": "memory_storage",
                "description": "Store results in energy-optimal memory tier",
                "delay": 1
            },
            {
                "type": "intelligent_routing",
                "description": "Route based on energy efficiency",
                "request_data": {
                    "type": "inference",
                    "priority": 0.3,
                    "constraint": "minimize_energy"
                }
            }
        ],
        expected_outcomes=[
            "Sub-nanojoule processing",
            "Automatic tier selection",
            "Energy-aware routing"
        ],
        visualizations=["energy_consumption", "latency_timeline"]
    )
    
    # Scenario 2: Adaptive Learning
    adaptive_scenario = DemoScenario(
        name="real_time_adaptation",
        description="Show real-time learning and adaptation",
        services=["lnn", "neuromorphic", "memory"],
        steps=[
            {
                "type": "adaptive_learning",
                "description": "Initial inference with base model",
                "adapt": False
            },
            {
                "type": "adaptive_learning",
                "description": "Adapt based on feedback",
                "feedback": 0.9,
                "adapt": True
            },
            {
                "type": "adaptive_learning",
                "description": "Improved inference after adaptation",
                "adapt": False
            }
        ],
        expected_outcomes=[
            "Dynamic parameter updates",
            "Improved performance",
            "No retraining needed"
        ],
        visualizations=["performance_radar", "service_flow"]
    )
    
    # Scenario 3: Fault-Tolerant Decision Making
    consensus_scenario = DemoScenario(
        name="fault_tolerant_decisions",
        description="Byzantine consensus for critical decisions",
        services=["byzantine", "moe", "lnn"],
        steps=[
            {
                "type": "consensus_decision",
                "description": "Propose critical system decision",
                "topic": "system_critical_update",
                "value": {"action": "update_model", "risk": "medium"}
            },
            {
                "type": "intelligent_routing",
                "description": "Route consensus result to affected services",
                "request_data": {
                    "type": "consensus",
                    "priority": 0.95
                }
            }
        ],
        expected_outcomes=[
            "Consensus despite failures",
            "High confidence decisions",
            "Automatic service coordination"
        ],
        visualizations=["service_flow", "latency_timeline"]
    )
    
    return [energy_scenario, adaptive_scenario, consensus_scenario]


# FastAPI app for interactive UI
def create_demo_app(demo_runner: InteractiveDemoRunner) -> FastAPI:
    """Create FastAPI app for demo UI"""
    
    app = FastAPI(title="AURA Intelligence Demo")
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve demo UI"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AURA Intelligence Demo</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px;
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                .container { max-width: 1200px; margin: 0 auto; }
                .scenario-card {
                    background: #2a2a2a;
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                }
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    margin: 20px 0;
                }
                .metric-card {
                    background: #333;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }
                button {
                    background: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                button:hover { background: #45a049; }
                #visualization { margin: 20px 0; }
                .log-entry {
                    background: #222;
                    padding: 5px 10px;
                    margin: 2px 0;
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 AURA Intelligence Interactive Demo</h1>
                
                <div id="scenarios">
                    <h2>Demo Scenarios</h2>
                    <div class="scenario-card">
                        <h3>⚡ Energy-Efficient Intelligence</h3>
                        <p>Demonstrate ultra-low energy AI processing with neuromorphic computing</p>
                        <button onclick="runDemo('energy_efficient_intelligence')">Run Demo</button>
                    </div>
                    
                    <div class="scenario-card">
                        <h3>🔄 Real-Time Adaptation</h3>
                        <p>Show liquid neural networks adapting in real-time without retraining</p>
                        <button onclick="runDemo('real_time_adaptation')">Run Demo</button>
                    </div>
                    
                    <div class="scenario-card">
                        <h3>🏛️ Fault-Tolerant Decisions</h3>
                        <p>Byzantine consensus ensuring reliable decisions despite failures</p>
                        <button onclick="runDemo('fault_tolerant_decisions')">Run Demo</button>
                    </div>
                </div>
                
                <div id="metrics" class="metrics" style="display:none;">
                    <div class="metric-card">
                        <div>Latency</div>
                        <div class="metric-value" id="latency">-- ms</div>
                    </div>
                    <div class="metric-card">
                        <div>Energy</div>
                        <div class="metric-value" id="energy">-- pJ</div>
                    </div>
                    <div class="metric-card">
                        <div>Throughput</div>
                        <div class="metric-value" id="throughput">-- req/s</div>
                    </div>
                    <div class="metric-card">
                        <div>Progress</div>
                        <div class="metric-value" id="progress">0%</div>
                    </div>
                </div>
                
                <div id="visualization"></div>
                
                <div id="logs" style="display:none;">
                    <h3>Live Logs</h3>
                    <div id="log-container" style="max-height: 300px; overflow-y: auto;"></div>
                </div>
            </div>
            
            <script>
                let ws = null;
                let currentDemo = null;
                
                function connectWebSocket() {
                    ws = new WebSocket('ws://localhost:8888/ws');
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleUpdate(data);
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };
                }
                
                function runDemo(scenarioName) {
                    currentDemo = scenarioName;
                    document.getElementById('metrics').style.display = 'block';
                    document.getElementById('logs').style.display = 'block';
                    
                    fetch(`/api/demo/${scenarioName}`, { method: 'POST' })
                        .then(response => response.json())
                        .then(data => console.log('Demo started:', data));
                }
                
                function handleUpdate(data) {
                    if (data.type === 'metrics_update') {
                        updateMetrics(data);
                    } else if (data.type === 'step_progress') {
                        updateProgress(data);
                    } else if (data.type === 'demo_complete') {
                        showSummary(data.summary);
                    }
                    
                    // Add to logs
                    addLog(JSON.stringify(data));
                }
                
                function updateMetrics(data) {
                    // Update latency
                    const latencies = Object.values(data.latencies || {});
                    if (latencies.length > 0) {
                        const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                        document.getElementById('latency').textContent = avgLatency.toFixed(1) + ' ms';
                    }
                    
                    // Update energy
                    if (data.energy_pj) {
                        document.getElementById('energy').textContent = data.energy_pj.toFixed(1) + ' pJ';
                    }
                    
                    // Update throughput
                    if (data.throughput) {
                        document.getElementById('throughput').textContent = data.throughput.toFixed(1) + ' req/s';
                    }
                }
                
                function updateProgress(data) {
                    const progress = (data.step / data.total_steps) * 100;
                    document.getElementById('progress').textContent = progress.toFixed(0) + '%';
                    addLog(`Step ${data.step + 1}/${data.total_steps}: ${data.description}`);
                }
                
                function addLog(message) {
                    const logContainer = document.getElementById('log-container');
                    const entry = document.createElement('div');
                    entry.className = 'log-entry';
                    entry.textContent = new Date().toLocaleTimeString() + ' - ' + message;
                    logContainer.appendChild(entry);
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
                
                function showSummary(summary) {
                    alert(`Demo Complete!
Success Rate: ${(summary.success_rate * 100).toFixed(1)}%
Total Energy: ${summary.total_energy_pj.toFixed(1)} pJ
Average Latency: ${summary.avg_latency_ms.toFixed(1)} ms
Duration: ${summary.duration_seconds.toFixed(1)} seconds`);
                }
                
                // Connect on load
                connectWebSocket();
            </script>
        </body>
        </html>
        """
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates"""
        await websocket.accept()
        demo_runner.websocket_clients.append(websocket)
        
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            demo_runner.websocket_clients.remove(websocket)
    
    @app.post("/api/demo/{scenario_name}")
    async def run_demo(scenario_name: str):
        """Run a demo scenario"""
        scenarios = create_demo_scenarios()
        scenario = next((s for s in scenarios if s.name == scenario_name), None)
        
        if not scenario:
            return {"error": "Scenario not found"}
        
        # Run scenario in background
        asyncio.create_task(demo_runner.run_scenario(scenario))
        
        return {"status": "started", "scenario": scenario_name}
    
    @app.get("/api/visualization/{demo_id}/{viz_type}")
    async def get_visualization(demo_id: str, viz_type: str):
        """Get visualization data"""
        fig = demo_runner.create_visualization(demo_id, viz_type)
        if fig:
            return fig.to_json()
        return {"error": "Visualization not available"}
    
    return app


if __name__ == "__main__":
    import random  # Add import
    import uvicorn
    
    # Service URLs
    service_urls = {
        "neuromorphic": "http://localhost:8000",
        "memory": "http://localhost:8001",
        "byzantine": "http://localhost:8002",
        "lnn": "http://localhost:8003",
        "moe": "http://localhost:8005"
    }
    
    # Create demo runner
    demo_runner = InteractiveDemoRunner(service_urls)
    
    # Create app
    app = create_demo_app(demo_runner)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8888)