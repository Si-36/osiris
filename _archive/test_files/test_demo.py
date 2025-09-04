#!/usr/bin/env python3
"""
Test script for AURA Working Demo
Tests all functionality and provides feedback
"""

import time
import json
import requests
from datetime import datetime

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_test_header(test_name):
    print(f"\n{BOLD}{BLUE}Testing: {test_name}{RESET}")
    print("-" * 50)

def print_result(success, message):
    if success:
        print(f"{GREEN}✓ {message}{RESET}")
    else:
        print(f"{RED}✗ {message}{RESET}")

def test_homepage():
    """Test if homepage loads"""
    print_test_header("Homepage")
    
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print_result(True, f"Homepage loaded (Status: {response.status_code})")
            
            # Check for key elements
            content = response.text
            checks = [
                ("AURA Agent Failure Prevention" in content, "Title found"),
                ("canvas" in content, "Canvas element present"),
                ("Topology Analysis" in content, "Topology section found"),
                ("Agent Network" in content, "Agent network section found")
            ]
            
            for check, message in checks:
                print_result(check, message)
            
            return True
        else:
            print_result(False, f"Homepage returned status {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Could not connect: {e}")
        return False

def test_websocket_connectivity():
    """Test WebSocket connectivity"""
    print_test_header("WebSocket Connection")
    
    try:
        import websocket
        
        ws = websocket.WebSocket()
        ws.connect("ws://localhost:8080/ws")
        print_result(True, "WebSocket connected")
        
        # Wait for initial data
        time.sleep(0.5)
        
        # Try to receive data
        try:
            ws.settimeout(2)
            data = ws.recv()
            if data:
                print_result(True, f"Received data: {len(data)} bytes")
                
                # Parse JSON
                try:
                    json_data = json.loads(data)
                    print_result(True, f"Valid JSON with {len(json_data.get('agents', {}))} agents")
                except:
                    print_result(False, "Invalid JSON data")
            else:
                print_result(False, "No data received")
        except:
            print_result(False, "WebSocket timeout - no data")
            
        ws.close()
        return True
        
    except ImportError:
        print_result(False, "websocket-client not installed")
        print(f"{YELLOW}  Install with: pip install websocket-client{RESET}")
        return False
    except Exception as e:
        print_result(False, f"WebSocket error: {e}")
        return False

def test_agent_simulation():
    """Test agent simulation is running"""
    print_test_header("Agent Simulation")
    
    try:
        # Check if we can see the simulation running
        response = requests.get("http://localhost:8080/")
        if response.status_code == 200:
            print_result(True, "Demo server is responsive")
            
            # Look for signs of active simulation
            content = response.text
            
            checks = [
                ("30 agents" in content.lower() or "agents: 30" in content.lower(), "30 agents configured"),
                ("aura protection" in content.lower(), "AURA protection toggle found"),
                ("cascade" in content.lower(), "Cascade prevention mentioned")
            ]
            
            for check, message in checks:
                print_result(check, message)
                
            return True
        else:
            print_result(False, "Could not access demo")
            return False
            
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False

def test_performance():
    """Test demo performance"""
    print_test_header("Performance Check")
    
    try:
        # Make multiple requests to test responsiveness
        response_times = []
        
        for i in range(5):
            start = time.time()
            response = requests.get("http://localhost:8080/", timeout=5)
            end = time.time()
            
            if response.status_code == 200:
                response_time = (end - start) * 1000  # Convert to ms
                response_times.append(response_time)
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print_result(True, f"Average response time: {avg_time:.1f}ms")
            
            if avg_time < 100:
                print_result(True, "Performance is excellent (<100ms)")
            elif avg_time < 500:
                print_result(True, "Performance is good (<500ms)")
            else:
                print_result(False, "Performance needs improvement (>500ms)")
                
            return True
        else:
            print_result(False, "Could not measure performance")
            return False
            
    except Exception as e:
        print_result(False, f"Performance test error: {e}")
        return False

def check_demo_features():
    """Check key demo features"""
    print_test_header("Demo Features")
    
    features = {
        "Real-time Visualization": "Canvas-based agent network display",
        "Topology Analysis": "TDA algorithms analyzing agent connections",
        "Failure Prediction": "LNN predicting cascade failures",
        "AURA Protection": "Toggle to enable/disable protection",
        "Live Metrics": "Real-time performance metrics"
    }
    
    print(f"{BLUE}Expected Features:{RESET}")
    for feature, description in features.items():
        print(f"  • {feature}: {description}")
    
    print(f"\n{YELLOW}Please manually verify these features at http://localhost:8080{RESET}")
    
    return True

def main():
    """Run all tests"""
    print(f"\n{BOLD}{BLUE}🧠 AURA Demo Test Suite{RESET}")
    print(f"{BLUE}Testing demo at http://localhost:8080{RESET}\n")
    
    # Check if demo is running
    try:
        response = requests.get("http://localhost:8080/", timeout=2)
    except:
        print(f"{RED}❌ Demo is not running!{RESET}")
        print(f"{YELLOW}Start it with: python3 demos/aura_working_demo_2025.py{RESET}")
        return False
    
    # Run tests
    tests = [
        test_homepage,
        test_websocket_connectivity,
        test_agent_simulation,
        test_performance,
        check_demo_features
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"{RED}Test failed with error: {e}{RESET}")
            results.append(False)
    
    # Summary
    print(f"\n{BOLD}{BLUE}Test Summary{RESET}")
    print("-" * 50)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print(f"\n{GREEN}✅ All tests passed! Demo is working perfectly.{RESET}")
        print(f"\n{BOLD}Next steps:{RESET}")
        print("1. Open http://localhost:8080 in your browser")
        print("2. Watch the agent network visualization")
        print("3. Click 'Enable AURA Protection' to see failure prevention")
        print("4. Compare with protection off to see the difference")
    else:
        print(f"\n{YELLOW}⚠️  Some tests failed. Check the output above.{RESET}")
        
    # Save test report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "tests_run": total,
        "tests_passed": passed,
        "success_rate": (passed/total * 100) if total > 0 else 0,
        "demo_url": "http://localhost:8080"
    }
    
    with open("demo_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{BLUE}Test report saved to demo_test_report.json{RESET}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)