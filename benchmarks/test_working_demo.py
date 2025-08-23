#!/usr/bin/env python3
"""
🧪 Test the working demo - Simple validation
"""

import time
import requests
import sys

def test_working_demo():
    """Test the working demo endpoints"""
    base_url = "http://localhost:8080"
    
    print("🧪 Testing AURA Working Demo")
    print("=" * 40)
    
    # Wait for system
    print("⏳ Waiting for system...")
    for i in range(20):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ System ready after {i+1} seconds")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ System failed to start")
        return False
    
    try:
        # Test health
        print("\n🏥 Testing health...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   Status: {health['status']}")
            print(f"   Uptime: {health['uptime_seconds']:.1f}s")
            print(f"   Errors: {len(health['errors'])}")
        
        # Test scenarios
        print("\n🎯 Testing scenarios...")
        scenarios = ["simple_test", "gpu_test", "performance_test"]
        
        for scenario in scenarios:
            response = requests.post(
                f"{base_url}/demo",
                json={"scenario": scenario, "data": {}, "config": {}},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                status = "✅" if result['success'] else "❌"
                print(f"   {status} {scenario}: {result['duration_ms']:.1f}ms")
            else:
                print(f"   ❌ {scenario}: HTTP {response.status_code}")
        
        # Test web interface
        print("\n🌐 Testing web interface...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ Web interface loads")
        else:
            print("   ❌ Web interface failed")
        
        print("\n🎉 All tests completed!")
        print(f"🌐 Demo available at: {base_url}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_working_demo()
    sys.exit(0 if success else 1)