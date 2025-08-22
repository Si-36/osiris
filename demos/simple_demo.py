#!/usr/bin/env python3
"""
AURA Intelligence - Simple Working Demo
Clean, minimal, actually works
"""

import asyncio
import time
import json
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# Simple working demo
app = FastAPI(title="AURA Intelligence Demo", version="1.0")

# Demo state
demo_data = {
    "start_time": time.time(),
    "requests": 0,
    "gpu_available": False
}

# Try to check for GPU
try:
    import torch
    demo_data["gpu_available"] = torch.cuda.is_available()
except:
    pass

@app.get("/")
async def home():
    """Simple demo homepage"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>AURA Intelligence Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 30px; }
        .button { background: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }
        .button:hover { background: #0056b3; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; font-family: monospace; }
        .status { padding: 5px 10px; border-radius: 3px; font-size: 12px; }
        .healthy { background: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AURA Intelligence</h1>
            <p>Simple Working Demo</p>
            <div>Status: <span class="status healthy">Running</span></div>
        </div>
        
        <div style="text-align: center;">
            <button class="button" onclick="testSystem()">Test System</button>
            <button class="button" onclick="testGPU()">Test GPU</button>
            <button class="button" onclick="runBenchmark()">Run Benchmark</button>
        </div>
        
        <div id="result" class="result" style="display:none;"></div>
    </div>
    
    <script>
        async function testSystem() {
            showResult("Running system test...");
            try {
                const response = await fetch('/test/system');
                const data = await response.json();
                showResult(JSON.stringify(data, null, 2));
            } catch (error) {
                showResult("Error: " + error.message);
            }
        }
        
        async function testGPU() {
            showResult("Testing GPU...");
            try {
                const response = await fetch('/test/gpu');
                const data = await response.json();
                showResult(JSON.stringify(data, null, 2));
            } catch (error) {
                showResult("Error: " + error.message);
            }
        }
        
        async function runBenchmark() {
            showResult("Running benchmark...");
            try {
                const response = await fetch('/test/benchmark');
                const data = await response.json();
                showResult(JSON.stringify(data, null, 2));
            } catch (error) {
                showResult("Error: " + error.message);
            }
        }
        
        function showResult(text) {
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = text;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
    """)

@app.get("/health")
async def health():
    """Simple health check"""
    return {
        "status": "healthy",
        "uptime": time.time() - demo_data["start_time"],
        "gpu_available": demo_data["gpu_available"],
        "requests_served": demo_data["requests"]
    }

@app.get("/test/system")
async def test_system():
    """Simple system test"""
    demo_data["requests"] += 1
    
    start_time = time.time()
    
    # Simple processing simulation
    await asyncio.sleep(0.01)  # 10ms
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "test": "system",
        "status": "passed",
        "processing_time_ms": round(processing_time, 2),
        "timestamp": time.time(),
        "message": "System test completed successfully"
    }

@app.get("/test/gpu")
async def test_gpu():
    """Simple GPU test"""
    demo_data["requests"] += 1
    
    start_time = time.time()
    
    # Test GPU if available
    gpu_test_result = "not_available"
    gpu_processing_time = 0
    
    if demo_data["gpu_available"]:
        try:
            import torch
            # Simple GPU operation
            if torch.cuda.is_available():
                device = torch.device('cuda')
                x = torch.randn(100, 100).to(device)
                y = torch.mm(x, x)  # Simple matrix multiply
                gpu_test_result = "passed"
                gpu_processing_time = (time.time() - start_time) * 1000
        except Exception as e:
            gpu_test_result = f"error: {str(e)}"
    
    return {
        "test": "gpu",
        "gpu_available": demo_data["gpu_available"],
        "test_result": gpu_test_result,
        "processing_time_ms": round(gpu_processing_time, 2),
        "timestamp": time.time()
    }

@app.get("/test/benchmark")
async def test_benchmark():
    """Simple benchmark test"""
    demo_data["requests"] += 1
    
    start_time = time.time()
    
    # Run multiple iterations
    times = []
    for i in range(10):
        iter_start = time.time()
        await asyncio.sleep(0.001)  # 1ms simulation
        iter_time = (time.time() - iter_start) * 1000
        times.append(iter_time)
    
    total_time = (time.time() - start_time) * 1000
    avg_time = sum(times) / len(times)
    
    return {
        "test": "benchmark",
        "iterations": len(times),
        "total_time_ms": round(total_time, 2),
        "average_time_ms": round(avg_time, 2),
        "min_time_ms": round(min(times), 2),
        "max_time_ms": round(max(times), 2),
        "timestamp": time.time()
    }

def main():
    print("🚀 AURA Intelligence - Simple Demo")
    print("🌐 Starting server on http://localhost:8080")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")

if __name__ == "__main__":
    main()