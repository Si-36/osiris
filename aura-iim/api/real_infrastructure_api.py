"""
Real AURA Infrastructure Monitor API
====================================
Production-grade API using real TDA and LNN components
"""

import sys
import os
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Add paths for real AURA components
sys.path.insert(0, '/workspace/aura-iim')
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/src')

# Import real components
from collectors.infrastructure_collector import InfrastructureCollector
from pipeline.data_pipeline import DataPipeline
from analysis.real_infrastructure_tda import RealInfrastructureTDA
from prediction.real_failure_predictor import RealFailurePredictor

# Try to import agent system
try:
    from agents.real_infrastructure_agents import InfrastructureCouncil
    HAS_AGENTS = True
except:
    HAS_AGENTS = False


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for infrastructure analysis"""
    include_predictions: bool = Field(True, description="Include failure predictions")
    include_recommendations: bool = Field(True, description="Include actionable recommendations")
    time_range_hours: int = Field(4, description="Prediction time horizon")


class MetricsUpdate(BaseModel):
    """Real-time metrics update"""
    server_id: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_connections: int
    custom_metrics: Optional[Dict[str, float]] = None


# ============================================================================
# Real Infrastructure Monitor API
# ============================================================================

app = FastAPI(
    title="AURA Infrastructure Monitor - Professional Edition",
    description="Real-time infrastructure monitoring with TDA and LNN",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Component Initialization
# ============================================================================

# Real components
collector = InfrastructureCollector()
pipeline = DataPipeline()
tda_analyzer = RealInfrastructureTDA()
predictor = RealFailurePredictor()

# Agent council if available
if HAS_AGENTS:
    council = InfrastructureCouncil()
else:
    council = None

# Global state
analysis_history = []
active_websockets = set()


# ============================================================================
# Background Tasks
# ============================================================================

async def continuous_monitoring():
    """Background task for continuous infrastructure monitoring"""
    
    while True:
        try:
            # Collect metrics
            metrics = await collector.collect_metrics()
            
            # Convert to point cloud
            point_cloud = await pipeline.process_metrics(metrics)
            
            if point_cloud is not None:
                # Run TDA analysis
                tda_features = await tda_analyzer.analyze_infrastructure(point_cloud)
                
                # Check for anomalies
                if tda_features.get('anomaly_detected', False):
                    # Run predictions
                    predictions = await predictor.predict_failures(
                        tda_features,
                        analysis_history[-100:] if analysis_history else [],
                        metrics
                    )
                    
                    # Broadcast alert to all websockets
                    alert = {
                        'type': 'anomaly_alert',
                        'timestamp': datetime.now().isoformat(),
                        'severity': predictions['risk_level'],
                        'tda_features': tda_features,
                        'predictions': predictions
                    }
                    
                    for ws in active_websockets:
                        try:
                            await ws.send_json(alert)
                        except:
                            active_websockets.remove(ws)
                
                # Store in history
                analysis_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics,
                    'tda_features': tda_features
                })
                
                # Keep history size manageable
                if len(analysis_history) > 1000:
                    analysis_history.pop(0)
        
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        await asyncio.sleep(5)  # Run every 5 seconds


# Start background monitoring
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(continuous_monitoring())


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and system info"""
    return {
        "system": "AURA Infrastructure Monitor",
        "version": "2.0.0",
        "status": "operational",
        "components": {
            "tda": "Real Rips/Persistence",
            "lnn": "Real MIT Liquid Networks",
            "agents": "Available" if HAS_AGENTS else "Not loaded",
            "monitoring": "Active"
        },
        "uptime": datetime.now().isoformat()
    }


@app.post("/api/analyze")
async def analyze_infrastructure(request: AnalysisRequest = AnalysisRequest()):
    """
    Main analysis endpoint - runs full TDA + LNN analysis
    """
    
    try:
        # 1. Collect current metrics
        metrics = await collector.collect_metrics()
        
        # 2. Process into point cloud
        point_cloud = await pipeline.process_metrics(metrics)
        
        if point_cloud is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for analysis"
            )
        
        # 3. Real TDA analysis
        tda_features = await tda_analyzer.analyze_infrastructure(point_cloud)
        
        # 4. Prepare response
        response = {
            "timestamp": datetime.now().isoformat(),
            "topology": {
                "betti_numbers": {
                    "b0": tda_features['betti_0'],
                    "b1": tda_features['betti_1'],
                    "b2": tda_features['betti_2']
                },
                "persistence": {
                    "max": tda_features['max_persistence'],
                    "entropy": tda_features['persistence_entropy'],
                    "pairs": tda_features['persistence_pairs']
                },
                "anomaly": {
                    "detected": tda_features['anomaly_detected'],
                    "score": tda_features['anomaly_score']
                },
                "interpretations": tda_features['interpretations']
            },
            "critical_features": tda_features['critical_features']
        }
        
        # 5. Add predictions if requested
        if request.include_predictions:
            predictions = await predictor.predict_failures(
                tda_features,
                analysis_history[-100:] if analysis_history else [],
                metrics
            )
            
            response["predictions"] = {
                "risk_score": predictions['risk_score'],
                "risk_level": predictions['risk_level'],
                "confidence": predictions['confidence'],
                "time_to_failure": predictions['time_to_failure'],
                "scenarios": predictions['failure_scenarios']
            }
            
            if request.include_recommendations:
                response["recommendations"] = predictions['recommendations']
        
        # 6. Agent consensus if available
        if HAS_AGENTS and council:
            agent_decision = await council.make_decision(
                tda_features,
                predictions if request.include_predictions else {}
            )
            response["agent_consensus"] = agent_decision
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/metrics/current")
async def get_current_metrics():
    """Get current infrastructure metrics"""
    
    metrics = await collector.collect_metrics()
    
    return {
        "timestamp": metrics['timestamp'],
        "summary": {
            "cpu_average": np.mean(metrics['cpu']['percent']),
            "memory_percent": metrics['memory']['percent'],
            "disk_percent": metrics['disk']['percent'],
            "network_connections": metrics['network']['connections']
        },
        "detailed": metrics
    }


@app.get("/api/topology/history")
async def get_topology_history(hours: int = 1):
    """Get historical topology analysis"""
    
    # Filter history by time
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    
    filtered_history = [
        {
            'timestamp': h['timestamp'].isoformat(),
            'betti_0': h['tda_features']['betti_0'],
            'betti_1': h['tda_features']['betti_1'],
            'anomaly_score': h['tda_features']['anomaly_score']
        }
        for h in analysis_history
        if h['timestamp'].timestamp() > cutoff_time
    ]
    
    return {
        "hours": hours,
        "data_points": len(filtered_history),
        "history": filtered_history
    }


@app.post("/api/metrics/update")
async def update_metrics(update: MetricsUpdate):
    """Endpoint for external systems to push metrics"""
    
    # Store custom metrics
    # This would integrate with your infrastructure
    
    return {"status": "accepted", "server_id": update.server_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket for streaming analysis
    """
    
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        while True:
            # Continuous streaming
            metrics = await collector.collect_metrics()
            point_cloud = await pipeline.process_metrics(metrics)
            
            if point_cloud is not None:
                # Real-time TDA
                tda_features = await tda_analyzer.analyze_infrastructure(point_cloud)
                
                # Stream update
                update = {
                    "type": "topology_update",
                    "timestamp": datetime.now().isoformat(),
                    "topology": {
                        "betti": [
                            tda_features['betti_0'],
                            tda_features['betti_1'],
                            tda_features['betti_2']
                        ],
                        "anomaly_score": tda_features['anomaly_score'],
                        "critical_count": len(tda_features['critical_features'])
                    },
                    "metrics": {
                        "cpu": np.mean(metrics['cpu']['percent']),
                        "memory": metrics['memory']['percent'],
                        "connections": metrics['network']['connections']
                    }
                }
                
                # Add predictions if anomaly detected
                if tda_features.get('anomaly_detected', False):
                    predictions = await predictor.predict_failures(
                        tda_features,
                        analysis_history[-50:] if analysis_history else [],
                        metrics
                    )
                    
                    update["alert"] = {
                        "risk_level": predictions['risk_level'],
                        "time_to_failure": predictions['time_to_failure']['formatted'],
                        "top_scenario": predictions['failure_scenarios'][0] if predictions['failure_scenarios'] else None
                    }
                
                await websocket.send_json(update)
            
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_websockets.remove(websocket)
        await websocket.close()


@app.get("/demo", response_class=HTMLResponse)
async def demo_dashboard():
    """
    Professional demo dashboard with real-time visualization
    """
    
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>AURA Infrastructure Monitor - Professional</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #0a0e27;
            color: #e0e6ed;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 300;
            letter-spacing: 2px;
        }
        .header .subtitle {
            color: #a0b3d3;
            font-size: 14px;
            margin-top: 5px;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        .card {
            background: #151a35;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border: 1px solid #1e2749;
        }
        .card h2 {
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 400;
            color: #6b8cff;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .metric-label {
            font-size: 12px;
            color: #7a8ca0;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 300;
            color: #fff;
        }
        .metric-value.critical { color: #ff4757; }
        .metric-value.warning { color: #ffa502; }
        .metric-value.normal { color: #26de81; }
        .risk-gauge {
            width: 200px;
            height: 200px;
            margin: 0 auto;
        }
        .alert {
            background: #ff4757;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.8; }
            100% { opacity: 1; }
        }
        #topology-viz {
            height: 300px;
        }
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-indicator.green { background: #26de81; }
        .status-indicator.yellow { background: #ffa502; }
        .status-indicator.red { background: #ff4757; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AURA INFRASTRUCTURE MONITOR</h1>
        <div class="subtitle">Real-time Topology Analysis & Failure Prediction</div>
    </div>
    
    <div class="container">
        <!-- Topology Card -->
        <div class="card">
            <h2>Infrastructure Topology</h2>
            <div id="topology-viz"></div>
            <div class="metric">
                <div class="metric-label">Components (B₀)</div>
                <div class="metric-value" id="betti0">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Loops (B₁)</div>
                <div class="metric-value" id="betti1">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Anomaly Score</div>
                <div class="metric-value" id="anomaly-score">-</div>
            </div>
        </div>
        
        <!-- Risk Assessment Card -->
        <div class="card">
            <h2>Risk Assessment</h2>
            <div id="risk-gauge" class="risk-gauge"></div>
            <div id="risk-details"></div>
        </div>
        
        <!-- Metrics Timeline -->
        <div class="card" style="grid-column: span 2;">
            <h2>System Metrics Timeline</h2>
            <div id="metrics-timeline"></div>
        </div>
        
        <!-- Alerts -->
        <div class="card" style="grid-column: span 2;">
            <h2>Active Alerts</h2>
            <div id="alerts-container"></div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        // Data storage
        const timelineData = {
            timestamps: [],
            cpu: [],
            memory: [],
            anomaly: [],
            betti0: [],
            betti1: []
        };
        
        // Initialize visualizations
        function initVisualizations() {
            // Risk gauge
            const gaugeData = [{
                type: 'indicator',
                mode: 'gauge+number',
                value: 0,
                title: {text: 'Risk Score', font: {size: 14, color: '#6b8cff'}},
                gauge: {
                    axis: {range: [0, 100], tickcolor: '#7a8ca0'},
                    bar: {color: '#26de81'},
                    steps: [
                        {range: [0, 30], color: 'rgba(38, 222, 129, 0.1)'},
                        {range: [30, 70], color: 'rgba(255, 165, 2, 0.1)'},
                        {range: [70, 100], color: 'rgba(255, 71, 87, 0.1)'}
                    ],
                    threshold: {
                        line: {color: 'red', width: 4},
                        thickness: 0.75,
                        value: 90
                    }
                }
            }];
            
            const gaugeLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {color: '#e0e6ed'},
                margin: {t: 0, b: 0, l: 0, r: 0}
            };
            
            Plotly.newPlot('risk-gauge', gaugeData, gaugeLayout, {displayModeBar: false});
            
            // Timeline
            const timelineLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {color: '#e0e6ed'},
                xaxis: {
                    showgrid: false,
                    title: 'Time',
                    color: '#7a8ca0'
                },
                yaxis: {
                    showgrid: true,
                    gridcolor: '#1e2749',
                    title: 'Value',
                    color: '#7a8ca0'
                },
                margin: {t: 20, b: 40, l: 60, r: 20},
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: 1.1
                }
            };
            
            Plotly.newPlot('metrics-timeline', [], timelineLayout, {displayModeBar: false});
        }
        
        // Update visualizations
        function updateVisualizations(data) {
            // Update topology metrics
            document.getElementById('betti0').textContent = data.topology.betti[0];
            document.getElementById('betti1').textContent = data.topology.betti[1];
            
            const anomalyScore = data.topology.anomaly_score.toFixed(2);
            const anomalyEl = document.getElementById('anomaly-score');
            anomalyEl.textContent = anomalyScore;
            
            // Color code anomaly score
            if (anomalyScore > 2) {
                anomalyEl.className = 'metric-value critical';
            } else if (anomalyScore > 1) {
                anomalyEl.className = 'metric-value warning';
            } else {
                anomalyEl.className = 'metric-value normal';
            }
            
            // Update timeline data
            const now = new Date();
            timelineData.timestamps.push(now);
            timelineData.cpu.push(data.metrics.cpu);
            timelineData.memory.push(data.metrics.memory);
            timelineData.anomaly.push(data.topology.anomaly_score);
            timelineData.betti0.push(data.topology.betti[0]);
            timelineData.betti1.push(data.topology.betti[1]);
            
            // Keep only last 50 points
            if (timelineData.timestamps.length > 50) {
                Object.keys(timelineData).forEach(key => {
                    timelineData[key].shift();
                });
            }
            
            // Update timeline plot
            const traces = [
                {
                    x: timelineData.timestamps,
                    y: timelineData.cpu,
                    name: 'CPU %',
                    type: 'scatter',
                    line: {color: '#6b8cff', width: 2}
                },
                {
                    x: timelineData.timestamps,
                    y: timelineData.memory,
                    name: 'Memory %',
                    type: 'scatter',
                    line: {color: '#ffa502', width: 2}
                },
                {
                    x: timelineData.timestamps,
                    y: timelineData.anomaly.map(a => a * 20),
                    name: 'Anomaly Score (×20)',
                    type: 'scatter',
                    line: {color: '#ff4757', width: 2}
                }
            ];
            
            Plotly.react('metrics-timeline', traces);
            
            // Update risk gauge
            if (data.alert) {
                const riskPercent = data.alert.risk_level === 'CRITICAL' ? 90 :
                                   data.alert.risk_level === 'HIGH' ? 70 :
                                   data.alert.risk_level === 'MEDIUM' ? 50 : 20;
                
                Plotly.update('risk-gauge', {
                    value: riskPercent,
                    'gauge.bar.color': riskPercent > 70 ? '#ff4757' : 
                                      riskPercent > 50 ? '#ffa502' : '#26de81'
                });
                
                // Update risk details
                document.getElementById('risk-details').innerHTML = `
                    <div class="alert">
                        <strong>${data.alert.risk_level} RISK</strong><br>
                        Time to failure: ${data.alert.time_to_failure}<br>
                        ${data.alert.top_scenario ? data.alert.top_scenario.description : ''}
                    </div>
                `;
            }
        }
        
        // WebSocket handlers
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'topology_update') {
                updateVisualizations(data);
            } else if (data.type === 'anomaly_alert') {
                // Add to alerts
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert';
                alertDiv.innerHTML = `
                    <strong>${data.severity} ANOMALY DETECTED</strong><br>
                    ${new Date(data.timestamp).toLocaleTimeString()}<br>
                    Risk Score: ${(data.predictions.risk_score * 100).toFixed(1)}%
                `;
                
                const container = document.getElementById('alerts-container');
                container.insertBefore(alertDiv, container.firstChild);
                
                // Keep only last 5 alerts
                while (container.children.length > 5) {
                    container.removeChild(container.lastChild);
                }
            }
        };
        
        ws.onopen = function() {
            console.log('Connected to AURA Infrastructure Monitor');
            initVisualizations();
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
        // Initialize on load
        window.onload = function() {
            if (ws.readyState === WebSocket.OPEN) {
                initVisualizations();
            }
        };
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)