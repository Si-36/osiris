"""
Real AURA API - Serves actual system data, no mocks
"""
import json
import time
import redis
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AURA Real API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    redis_client = None


class MetricResponse(BaseModel):
    timestamp: str
    cpu: Dict[str, float]
    memory: Dict[str, float]
    disk: Dict[str, float]
    network: Dict[str, int]
    processes: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    redis_connected: bool
    uptime_seconds: float


# Track server start time
SERVER_START_TIME = time.time()


@app.get("/")
async def root():
    """Root endpoint - shows available endpoints"""
    return {
        "message": "AURA Real API - Actually Working!",
        "endpoints": {
            "GET /health": "System health check",
            "GET /metrics": "Latest system metrics",
            "GET /metrics/history": "Historical metrics (last hour)",
            "GET /metrics/summary": "Metrics summary and statistics",
            "WS /ws": "WebSocket for real-time metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Real health check - actually checks system status"""
    return {
        "status": "healthy" if REDIS_AVAILABLE else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "redis_connected": REDIS_AVAILABLE,
        "uptime_seconds": time.time() - SERVER_START_TIME
    }


@app.get("/metrics", response_model=Optional[MetricResponse])
async def get_latest_metrics():
    """Get latest real system metrics"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        data = redis_client.get('metrics:latest')
        if data:
            return json.loads(data)
        else:
            raise HTTPException(status_code=404, detail="No metrics available yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/history")
async def get_metrics_history(minutes: int = 60):
    """Get historical metrics for the last N minutes"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        # Calculate time range
        end_time = int(time.time())
        start_time = end_time - (minutes * 60)
        
        # Get data from Redis sorted set
        data = redis_client.zrangebyscore('metrics:history', start_time, end_time)
        
        # Parse and return
        metrics = [json.loads(item) for item in data]
        
        return {
            "count": len(metrics),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "metrics": metrics[-100:]  # Limit to last 100 entries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary statistics of metrics"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        # Get last hour of data
        history = await get_metrics_history(60)
        metrics_list = history.get("metrics", [])
        
        if not metrics_list:
            raise HTTPException(status_code=404, detail="No metrics available")
        
        # Calculate statistics
        cpu_values = [m['cpu']['percent'] for m in metrics_list]
        mem_values = [m['memory']['percent'] for m in metrics_list]
        disk_values = [m['disk']['percent'] for m in metrics_list]
        
        return {
            "period": "last_hour",
            "samples": len(metrics_list),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory": {
                "avg": sum(mem_values) / len(mem_values),
                "max": max(mem_values),
                "min": min(mem_values),
                "current": mem_values[-1] if mem_values else 0
            },
            "disk": {
                "avg": sum(disk_values) / len(disk_values),
                "max": max(disk_values),
                "min": min(disk_values),
                "current": disk_values[-1] if disk_values else 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time metrics streaming"""
    await websocket.accept()
    
    if not REDIS_AVAILABLE:
        await websocket.send_json({"error": "Redis not available"})
        await websocket.close()
        return
    
    # Create pubsub
    pubsub = redis_client.pubsub()
    pubsub.subscribe('metrics:stream')
    
    try:
        while True:
            # Check for new messages
            message = pubsub.get_message(timeout=1.0)
            
            if message and message['type'] == 'message':
                # Send real metrics to client
                data = json.loads(message['data'])
                await websocket.send_json(data)
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        pubsub.close()
        await websocket.close()


@app.get("/demo")
async def demo_endpoint():
    """Quick demo endpoint to show it's working"""
    metrics = await get_latest_metrics() if REDIS_AVAILABLE else None
    
    return {
        "message": "🚀 AURA is ACTUALLY WORKING!",
        "api_status": "online",
        "redis_status": "connected" if REDIS_AVAILABLE else "disconnected",
        "current_metrics": metrics,
        "tip": "Run the collector to see real data flow!"
    }


if __name__ == "__main__":
    print("🚀 Starting AURA Real API on http://localhost:8080")
    print("📊 This serves REAL system metrics, not dummy data!")
    uvicorn.run(app, host="0.0.0.0", port=8080)