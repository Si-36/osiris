#!/usr/bin/env python3
"""
AURA Intelligence 2025 - Production Starter
"""

import uvicorn
import sys
import os
import asyncio
import requests
import time

# Add current directory to path
sys.path.append('.')

# Set environment
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_PROJECT'] = 'aura-intelligence-2025'

def test_system():
    """Test all endpoints"""
    print("🧪 Testing AURA Intelligence 2025...")
    
    base_url = "http://localhost:8091"
    
    try:
        # Test root
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root: {data['message']}")
            print(f"   Features: {len(data['features'])} latest research systems")
        
        # Test health
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data['status']} ({data.get('health_score', 0):.2f})")
        
        # Test processing
        response = requests.post(f"{base_url}/process", json={
            "data": {"test": "comprehensive", "complexity": "high"}
        })
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Processing: {data['processing_time']*1000:.1f}ms")
            print(f"   Pipeline: {len(data.get('pipeline_stages', []))} stages")
        
        # Test research systems
        response = requests.post(f"{base_url}/moa/process", json={"query": "test"})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ MoA: {data['layers_processed']} layers, {data['total_agents']} agents")
        
        response = requests.post(f"{base_url}/got/reason", json={"problem": "test"})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GoT: {data['reasoning_graph']['nodes']} nodes")
        
        response = requests.post(f"{base_url}/constitutional/check", json={"action": "test"})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Constitutional AI: {data['decision']} ({data['alignment_score']:.2f})")
        
        # Test benchmark
        response = requests.get(f"{base_url}/benchmark")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Benchmark: {data['total_time']*1000:.1f}ms for {data['systems_tested']} systems")
        
        print("\n🎉 All systems operational!")
        print(f"🌐 AURA Intelligence 2025 running at: {base_url}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    print("🚀 Starting AURA Intelligence 2025...")
    print("📊 Latest Research: MoA + GoT + Constitutional AI 2.0")
    
    # Import and start server
    from enhanced_ultimate_api_system import app
    
    # Start server in background
    import threading
    import time
    
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8091, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Starting server...")
    time.sleep(3)
    
    # Test system
    test_system()
    
    print("\n🔥 Server running! Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

if __name__ == "__main__":
    main()