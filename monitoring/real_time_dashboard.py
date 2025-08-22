#!/usr/bin/env python3
"""
AURA Real-Time Performance Dashboard - Phase 3
Production-grade monitoring dashboard with live metrics
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    fastapi_available = True
except ImportError:
    fastapi_available = False

# Performance Dashboard Manager
class RealTimeDashboard:
    """Production-grade real-time performance dashboard"""
    
    def __init__(self):
        self.app = None
        self.active_connections: List[WebSocket] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Performance tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        
        # Component registry
        self.components = {}
        self.system_health = "initializing"
        
    def create_app(self) -> FastAPI:
        """Create FastAPI application with dashboard endpoints"""
        if not fastapi_available:
            raise ImportError("FastAPI required for dashboard. Install with: pip install fastapi uvicorn")
        
        self.app = FastAPI(
            title="AURA Intelligence Dashboard",
            description="Real-time performance monitoring for AURA AI system",
            version="2025.1.0"
        )
        
        # WebSocket endpoint for real-time updates
        @self.app.websocket("/ws/metrics")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Send current metrics
                    metrics = await self.get_current_metrics()
                    await websocket.send_json(metrics)
                    await asyncio.sleep(1)  # Update every second
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
        
        # Dashboard HTML endpoint
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self.get_dashboard_html()
        
        # API endpoints
        @self.app.get("/api/metrics")
        async def get_metrics():
            metrics = await self.get_current_metrics()
            return JSONResponse(metrics)
        
        @self.app.get("/api/health")
        async def get_health():
            health_data = await self.get_system_health()
            return JSONResponse(health_data)
        
        @self.app.get("/api/components")
        async def get_components():
            component_data = await self.get_component_status()
            return JSONResponse(component_data)
        
        @self.app.post("/api/test-load")
        async def test_load(request: Request):
            """Trigger load test for performance analysis"""
            body = await request.json()
            load_results = await self.run_load_test(body.get("num_requests", 10))
            return JSONResponse(load_results)
        
        return self.app
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for real-time display"""
        try:
            from aura_intelligence.components.real_components import (
                redis_pool,
                batch_processor,
                gpu_manager
            )
            
            # Get Redis pool stats
            redis_stats = redis_pool.get_pool_stats()
            
            # Get batch processor stats
            batch_stats = batch_processor.get_performance_stats()
            
            # Get GPU stats
            gpu_stats = gpu_manager.get_memory_info()
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = {
                "timestamp": time.time(),
                "system": {
                    "uptime_seconds": uptime_seconds,
                    "total_requests": self.total_requests,
                    "error_count": self.error_count,
                    "error_rate": (self.error_count / max(1, self.total_requests)) * 100,
                    "status": self.system_health
                },
                "redis": redis_stats,
                "batch_processing": batch_stats,
                "gpu": gpu_stats,
                "memory_usage": self.get_memory_usage(),
                "performance": {
                    "requests_per_second": self.calculate_rps(),
                    "avg_response_time_ms": batch_stats.get("avg_processing_time_ms", 0)
                }
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            return {
                "timestamp": time.time(),
                "error": str(e),
                "system": {"status": "error"}
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Import components for health check
            from aura_intelligence.components.real_components import (
                RealAttentionComponent,
                RealLNNComponent,
                redis_pool
            )
            
            # Test key components
            bert_health = await RealAttentionComponent("health_check_bert").health_check()
            lnn_health = await RealLNNComponent("health_check_lnn").health_check()
            
            # Overall health calculation
            component_healthy = (
                bert_health.get("status") == "healthy" and
                lnn_health.get("status") == "healthy"
            )
            
            redis_healthy = redis_pool.get_pool_stats().get("status") == "active"
            
            overall_health = "healthy" if (component_healthy and redis_healthy) else "unhealthy"
            self.system_health = overall_health
            
            return {
                "overall_status": overall_health,
                "components": {
                    "bert_attention": bert_health,
                    "lnn_processing": lnn_health
                },
                "infrastructure": {
                    "redis_pool": redis_pool.get_pool_stats(),
                    "gpu_available": bert_health.get("gpu_info", {}).get("gpu_available", False)
                },
                "last_check": time.time()
            }
            
        except Exception as e:
            self.system_health = "error"
            return {
                "overall_status": "error",
                "error": str(e),
                "last_check": time.time()
            }
    
    async def get_component_status(self) -> Dict[str, Any]:
        """Get detailed status of all registered components"""
        component_status = {}
        
        try:
            from aura_intelligence.components.real_components import (
                RealAttentionComponent,
                RealLNNComponent,
                RealSwitchMoEComponent,
                RealVAEComponent
            )
            
            # Test multiple component types
            components_to_test = [
                ("bert_attention", RealAttentionComponent),
                ("lnn_neural", RealLNNComponent),
                ("switch_moe", RealSwitchMoEComponent),
                ("vae_generative", RealVAEComponent)
            ]
            
            for comp_name, comp_class in components_to_test:
                try:
                    component = comp_class(f"dashboard_{comp_name}")
                    health = await component.health_check()
                    component_status[comp_name] = health
                except Exception as e:
                    component_status[comp_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "components": component_status,
                "total_components": len(component_status),
                "healthy_components": len([c for c in component_status.values() if c.get("status") == "healthy"]),
                "last_update": time.time()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "last_update": time.time()
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get system memory usage statistics"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percentage": memory.percent
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def calculate_rps(self) -> float:
        """Calculate requests per second over the last minute"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Get metrics from last minute
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history 
            if current_time - m["timestamp"] <= 60
        ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate RPS from batch processor data
        time_span = recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"]
        if time_span <= 0:
            return 0.0
        
        latest_items = recent_metrics[-1]["batch_processing"]["items_processed"]
        oldest_items = recent_metrics[0]["batch_processing"]["items_processed"]
        
        return (latest_items - oldest_items) / time_span
    
    async def run_load_test(self, num_requests: int = 10) -> Dict[str, Any]:
        """Run load test and return performance metrics"""
        try:
            from aura_intelligence.components.real_components import RealLNNComponent
            import numpy as np
            
            component = RealLNNComponent("load_test")
            
            # Generate test requests
            requests = [
                {"values": np.random.randn(10).tolist()}
                for _ in range(num_requests)
            ]
            
            # Run load test
            start_time = time.perf_counter()
            results = await component.process_batch(requests)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            successful_requests = len([r for r in results if not r.get("error")])
            
            return {
                "num_requests": num_requests,
                "successful_requests": successful_requests,
                "duration_ms": duration_ms,
                "requests_per_second": num_requests / (duration_ms / 1000),
                "avg_latency_ms": duration_ms / num_requests,
                "success_rate": (successful_requests / num_requests) * 100
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_dashboard_html(self) -> str:
        """Generate HTML dashboard with real-time metrics"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>AURA Intelligence Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .metric-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #64b5f6;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.8;
            margin: 5px 0;
        }
        .status-healthy { color: #4caf50; }
        .status-unhealthy { color: #f44336; }
        .status-warning { color: #ff9800; }
        .loading { text-align: center; padding: 50px; }
        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .control-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        button {
            background: #64b5f6;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover { background: #42a5f5; }
        .log-area {
            background: #1a1a1a;
            color: #00ff00;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            max-height: 200px;
            overflow-y: auto;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AURA Intelligence Dashboard</h1>
        <p>Real-time Performance Monitoring - Phase 3 Production Dashboard</p>
    </div>
    
    <div class="control-panel">
        <h3>🎛️ Control Panel</h3>
        <button onclick="runLoadTest(10)">Load Test (10 requests)</button>
        <button onclick="runLoadTest(50)">Load Test (50 requests)</button>
        <button onclick="refreshData()">🔄 Refresh</button>
        <button onclick="toggleAutoRefresh()">⏸️ Toggle Auto-refresh</button>
        <div class="log-area" id="logs">Connecting to AURA system...</div>
    </div>

    <div class="metrics-grid" id="metrics">
        <div class="loading">Loading AURA metrics...</div>
    </div>

    <div class="chart-container">
        <h3>📊 Performance Trends</h3>
        <canvas id="performanceChart" width="800" height="200"></canvas>
    </div>

    <script>
        let ws = null;
        let autoRefresh = true;
        let metricsHistory = [];
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/metrics`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
                metricsHistory.push(data);
                if (metricsHistory.length > 100) {
                    metricsHistory.shift();
                }
                updateChart();
            };
            
            ws.onclose = function() {
                logMessage("Connection lost, reconnecting...");
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onopen = function() {
                logMessage("Connected to AURA dashboard");
            };
        }
        
        function updateDashboard(data) {
            const metricsDiv = document.getElementById('metrics');
            
            if (data.error) {
                metricsDiv.innerHTML = `<div class="metric-card"><div class="status-unhealthy">Error: ${data.error}</div></div>`;
                return;
            }
            
            const html = `
                <div class="metric-card">
                    <div class="metric-title">🏥 System Health</div>
                    <div class="metric-value status-${data.system.status === 'healthy' ? 'healthy' : 'unhealthy'}">
                        ${data.system.status.toUpperCase()}
                    </div>
                    <div class="metric-label">Uptime: ${formatUptime(data.system.uptime_seconds)}</div>
                    <div class="metric-label">Error Rate: ${data.system.error_rate.toFixed(2)}%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">🚀 Performance</div>
                    <div class="metric-value">${data.performance.requests_per_second.toFixed(1)} RPS</div>
                    <div class="metric-label">Avg Response: ${data.performance.avg_response_time_ms.toFixed(2)}ms</div>
                    <div class="metric-label">Total Requests: ${data.system.total_requests}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">📦 Batch Processing</div>
                    <div class="metric-value">${data.batch_processing.batches_processed}</div>
                    <div class="metric-label">Items: ${data.batch_processing.items_processed}</div>
                    <div class="metric-label">Avg Batch Size: ${data.batch_processing.avg_batch_size.toFixed(1)}</div>
                    <div class="metric-label">Queue: ${data.batch_processing.current_queue_size}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">🔗 Redis Pool</div>
                    <div class="metric-value status-${data.redis.status === 'active' ? 'healthy' : 'unhealthy'}">
                        ${data.redis.status.toUpperCase()}
                    </div>
                    <div class="metric-label">Max Connections: ${data.redis.max_connections || 'N/A'}</div>
                    <div class="metric-label">In Use: ${data.redis.in_use_connections || 0}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">🎮 GPU Status</div>
                    <div class="metric-value status-${data.gpu.gpu_available ? 'healthy' : 'warning'}">
                        ${data.gpu.gpu_available ? 'ENABLED' : 'CPU ONLY'}
                    </div>
                    <div class="metric-label">Device: ${data.gpu.current_device || 'cpu'}</div>
                    <div class="metric-label">Memory: ${formatBytes(data.gpu.memory_allocated || 0)}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">💾 Memory Usage</div>
                    <div class="metric-value">${data.memory_usage.percentage || 0}%</div>
                    <div class="metric-label">Used: ${data.memory_usage.used_gb || 0} GB</div>
                    <div class="metric-label">Available: ${data.memory_usage.available_gb || 0} GB</div>
                </div>
            `;
            
            metricsDiv.innerHTML = html;
        }
        
        function updateChart() {
            const canvas = document.getElementById('performanceChart');
            const ctx = canvas.getContext('2d');
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (metricsHistory.length < 2) return;
            
            // Draw RPS trend
            ctx.strokeStyle = '#64b5f6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const maxRPS = Math.max(...metricsHistory.map(m => m.performance?.requests_per_second || 0));
            const scale = maxRPS > 0 ? (canvas.height - 40) / maxRPS : 1;
            
            metricsHistory.forEach((metric, index) => {
                const x = (index / (metricsHistory.length - 1)) * canvas.width;
                const y = canvas.height - 20 - (metric.performance?.requests_per_second || 0) * scale;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Add labels
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.fillText(`Max RPS: ${maxRPS.toFixed(1)}`, 10, 20);
            ctx.fillText(`Current: ${(metricsHistory[metricsHistory.length - 1]?.performance?.requests_per_second || 0).toFixed(1)}`, 10, 35);
        }
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hours}h ${minutes}m ${secs}s`;
        }
        
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        function logMessage(message) {
            const logs = document.getElementById('logs');
            const timestamp = new Date().toLocaleTimeString();
            logs.innerHTML += `\\n[${timestamp}] ${message}`;
            logs.scrollTop = logs.scrollHeight;
        }
        
        async function runLoadTest(numRequests) {
            logMessage(`Starting load test with ${numRequests} requests...`);
            try {
                const response = await fetch('/api/test-load', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({num_requests: numRequests})
                });
                const result = await response.json();
                logMessage(`Load test completed: ${result.requests_per_second?.toFixed(1)} RPS, ${result.success_rate?.toFixed(1)}% success`);
            } catch (error) {
                logMessage(`Load test failed: ${error.message}`);
            }
        }
        
        function refreshData() {
            logMessage('Refreshing dashboard data...');
            if (ws && ws.readyState === WebSocket.OPEN) {
                // Data refreshes automatically via WebSocket
                logMessage('Dashboard data updated');
            }
        }
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const button = event.target;
            button.textContent = autoRefresh ? '⏸️ Toggle Auto-refresh' : '▶️ Toggle Auto-refresh';
            logMessage(`Auto-refresh ${autoRefresh ? 'enabled' : 'disabled'}`);
        }
        
        // Initialize dashboard
        connectWebSocket();
    </script>
</body>
</html>
        """

# Dashboard runner
async def run_dashboard(host: str = "127.0.0.1", port: int = 8081):
    """Run the real-time dashboard server"""
    
    if not fastapi_available:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
        return False
    
    print("🚀 Starting AURA Real-Time Dashboard...")
    print(f"📊 Dashboard URL: http://{host}:{port}")
    print("🔗 WebSocket metrics: ws://localhost:8081/ws/metrics")
    print("🛠️ API endpoints: /api/metrics, /api/health, /api/components")
    
    dashboard = RealTimeDashboard()
    app = dashboard.create_app()
    
    # Initialize Redis pool for the dashboard
    try:
        from aura_intelligence.components.real_components import redis_pool
        await redis_pool.initialize()
        print("✅ Redis pool initialized for dashboard")
    except Exception as e:
        print(f"⚠️ Redis pool initialization warning: {e}")
    
    # Run the server
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
        return True
    except KeyboardInterrupt:
        print("\n🛑 Dashboard shutdown requested")
        return True
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA Real-Time Performance Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    args = parser.parse_args()
    
    try:
        success = asyncio.run(run_dashboard(args.host, args.port))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Dashboard interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Dashboard failed: {e}")
        sys.exit(1)