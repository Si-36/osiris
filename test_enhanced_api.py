#!/usr/bin/env python3
"""
Test Enhanced AURA API with real integrations
"""
import requests
import json
import time

def test_api():
    base_url = "http://localhost:8080"
    
    print("🧪 Testing Enhanced AURA API...")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health: {health['working_components']}/5 components, {health['integrations']}/3 integrations")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test neural processing (existing)
    try:
        response = requests.post(f"{base_url}/neural/process", json={
            "data": [1.0, 2.0, 3.0, 4.0, 5.0],
            "task_type": "neural"
        })
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Neural processing: {result['success']}")
        else:
            print(f"❌ Neural processing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Neural processing error: {e}")
    
    # Test event publishing (new)
    try:
        response = requests.post(f"{base_url}/events/publish", json={
            "type": "component_health",
            "source": "test_client",
            "data": {"status": "testing", "timestamp": time.time()}
        })
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Event publishing: {result['success']}")
        else:
            print(f"❌ Event publishing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Event publishing error: {e}")
    
    # Test decision storage (new)
    try:
        response = requests.post(f"{base_url}/graph/store_decision", json={
            "decision_id": f"test_{int(time.time())}",
            "vote": "APPROVE",
            "confidence": 0.85,
            "reasoning": "Test decision from API"
        })
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Decision storage: {result['success']}")
        else:
            print(f"❌ Decision storage failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Decision storage error: {e}")
    
    # Test system status
    try:
        response = requests.get(f"{base_url}/system/status")
        if response.status_code == 200:
            status = response.json()
            components = len(status.get('components', {}))
            integrations = len(status.get('integrations', {}))
            print(f"✅ System status: {components} components, {integrations} integrations")
        else:
            print(f"❌ System status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ System status error: {e}")

if __name__ == "__main__":
    test_api()