"""
AURA Infrastructure Monitor API
Simple demo API to get started
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.infrastructure_collector import InfrastructureCollector

app = FastAPI(title="AURA Infrastructure Monitor")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global collector
collector = InfrastructureCollector()

# Background task to collect metrics
async def collect_metrics_background():
    while True:
        await collector.collect_metrics()
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(collect_metrics_background())

@app.get("/")
async def root():
    return {"message": "AURA Infrastructure Monitor API", "status": "operational"}

@app.get("/metrics/current")
async def get_current_metrics():
    """Get current infrastructure metrics"""
    metrics = await collector.collect_metrics()
    return metrics

@app.get("/metrics/history")
async def get_metrics_history(limit: int = 100):
    """Get historical metrics"""
    history = collector.metrics_history[-limit:]
    return {"count": len(history), "metrics": history}

@app.get("/analysis/topology")
async def get_topology_analysis():
    """Get topology analysis (mock for now)"""
    point_cloud = collector.get_metrics_for_tda()
    
    if point_cloud is None:
        return {"error": "Not enough data for analysis"}
    
    # Mock TDA results
    return {
        "point_cloud_shape": point_cloud.shape,
        "betti_0": 1,  # Connected components
        "betti_1": 0,  # Loops
        "anomaly_score": 0.2,
        "risk_level": "low"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time metrics stream"""
    await websocket.accept()
    
    try:
        while True:
            # Get current metrics
            metrics = await collector.collect_metrics()
            
            # Mock topology analysis
            point_cloud = collector.get_metrics_for_tda()
            
            message = {
                "type": "metrics_update",
                "metrics": metrics,
                "analysis": {
                    "risk_score": 0.1 + (metrics['cpu']['load_average'][0] / 10),
                    "anomalies": []
                }
            }
            
            await websocket.send_json(message)
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Simple HTML demo page
@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AURA Infrastructure Monitor Demo</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; 
                  background: #f0f0f0; border-radius: 5px; }
        #chart { width: 100%; height: 400px; }
    </style>
</head>
<body>
    <h1>🚀 AURA Infrastructure Monitor</h1>
    <div id="metrics"></div>
    <div id="chart"></div>
    
    <script>
        const ws = new WebSocket('ws://localhost:8000/ws');
        const cpuHistory = [];
        const memoryHistory = [];
        const timestamps = [];
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const metrics = data.metrics;
            
            // Update metrics display
            const avgCpu = metrics.cpu.percent.reduce((a,b) => a+b) / metrics.cpu.percent.length;
            document.getElementById('metrics').innerHTML = `
                <div class="metric">
                    <h3>CPU</h3>
                    <p>${avgCpu.toFixed(1)}%</p>
                </div>
                <div class="metric">
                    <h3>Memory</h3>
                    <p>${metrics.memory.percent.toFixed(1)}%</p>
                </div>
                <div class="metric">
                    <h3>Risk Score</h3>
                    <p>${(data.analysis.risk_score * 100).toFixed(1)}%</p>
                </div>
            `;
            
            // Update chart
            cpuHistory.push(avgCpu);
            memoryHistory.push(metrics.memory.percent);
            timestamps.push(new Date(metrics.timestamp));
            
            if (cpuHistory.length > 50) {
                cpuHistory.shift();
                memoryHistory.shift();
                timestamps.shift();
            }
            
            Plotly.newPlot('chart', [
                {x: timestamps, y: cpuHistory, name: 'CPU %', type: 'scatter'},
                {x: timestamps, y: memoryHistory, name: 'Memory %', type: 'scatter'}
            ], {
                title: 'Infrastructure Metrics',
                xaxis: {title: 'Time'},
                yaxis: {title: 'Usage %'}
            });
        };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
