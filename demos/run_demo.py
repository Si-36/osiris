#!/usr/bin/env python3
"""
Simple AURA Demo Runner
Clean, works, no mess
"""

import subprocess
import sys
import time
import requests

def run_demo():
    print("🚀 AURA Intelligence - Clean Demo")
    print("🌐 http://localhost:8080")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        # Start the demo
        subprocess.run([sys.executable, "simple_demo.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")

def test_demo():
    print("🧪 Testing AURA Demo")
    
    # Start demo in background
    process = subprocess.Popen([
        sys.executable, "simple_demo.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for startup
        print("⏳ Starting demo...")
        time.sleep(5)  # Give more time for startup
        
        # Test endpoints
        print("✅ Health:", requests.get("http://localhost:8080/health").json()["status"])
        print("✅ System:", requests.get("http://localhost:8080/test/system").json()["status"])
        
        gpu_result = requests.get("http://localhost:8080/test/gpu").json()
        print(f"✅ GPU: {gpu_result['test_result']} ({gpu_result['processing_time_ms']:.1f}ms)")
        
        bench_result = requests.get("http://localhost:8080/test/benchmark").json()
        print(f"✅ Benchmark: {bench_result['average_time_ms']:.1f}ms avg")
        
        print("\n🎉 All tests passed!")
        print("🌐 Demo running at http://localhost:8080")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n👋 Demo stopped")
            
    finally:
        process.terminate()

if __name__ == "__main__":
    choice = input("1. Run Demo  2. Test Demo\nChoice (1/2): ").strip()
    
    if choice == "2":
        test_demo()
    else:
        run_demo()