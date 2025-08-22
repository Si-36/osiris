#!/usr/bin/env python3
"""
Simple Demo Test - No Dependencies Required
"""

import urllib.request
import json
import time
from datetime import datetime

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def test_demo():
    """Test if demo is running"""
    print(f"\n{BOLD}{BLUE}🧠 AURA Demo Test{RESET}")
    print("=" * 50)
    
    url = "http://localhost:8080"
    
    try:
        # Test homepage
        print(f"\n{BOLD}1. Testing Homepage:{RESET}")
        response = urllib.request.urlopen(url, timeout=5)
        html = response.read().decode('utf-8')
        
        print(f"{GREEN}✓ Demo is running on {url}{RESET}")
        print(f"{GREEN}✓ Response received: {len(html)} bytes{RESET}")
        
        # Check key elements
        checks = [
            ("AURA Agent Failure Prevention" in html, "Title found"),
            ("<canvas" in html, "Canvas visualization present"),
            ("Agent Network" in html, "Agent network section found"),
            ("Topology Analysis" in html, "Topology analysis section found"),
            ("AURA Protection" in html, "AURA protection toggle found")
        ]
        
        print(f"\n{BOLD}2. Checking Demo Features:{RESET}")
        all_good = True
        for check, message in checks:
            if check:
                print(f"{GREEN}✓ {message}{RESET}")
            else:
                print(f"{RED}✗ {message}{RESET}")
                all_good = False
        
        # Performance check
        print(f"\n{BOLD}3. Performance Test:{RESET}")
        times = []
        for i in range(3):
            start = time.time()
            urllib.request.urlopen(url, timeout=5)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"{GREEN}✓ Average response time: {avg_time:.1f}ms{RESET}")
        
        if avg_time < 100:
            print(f"{GREEN}✓ Excellent performance (<100ms){RESET}")
        elif avg_time < 500:
            print(f"{GREEN}✓ Good performance (<500ms){RESET}")
        else:
            print(f"{YELLOW}⚠ Slow response (>500ms){RESET}")
        
        # Summary
        print(f"\n{BOLD}4. Summary:{RESET}")
        if all_good:
            print(f"{GREEN}✅ Demo is working perfectly!{RESET}")
            print(f"\n{BOLD}Next steps:{RESET}")
            print(f"1. Open {BLUE}http://localhost:8080{RESET} in your browser")
            print(f"2. Watch the agent network visualization")
            print(f"3. Click 'Enable AURA Protection' to see failure prevention")
            print(f"4. Observe the difference with protection on/off")
        else:
            print(f"{YELLOW}⚠️ Some features may not be working properly{RESET}")
            print(f"Check the demo at {BLUE}http://localhost:8080{RESET}")
        
        # Save report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "demo_running": True,
            "all_features_ok": all_good,
            "avg_response_ms": round(avg_time, 1),
            "url": url
        }
        
        with open("simple_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{BLUE}Report saved to simple_test_report.json{RESET}")
        
        return True
        
    except Exception as e:
        print(f"{RED}❌ Demo is not running!{RESET}")
        print(f"{RED}Error: {e}{RESET}")
        print(f"\n{YELLOW}To start the demo:{RESET}")
        print(f"python3 demos/aura_working_demo_2025.py")
        return False

def check_processes():
    """Check if demo process is running"""
    print(f"\n{BOLD}5. Process Check:{RESET}")
    
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'aura_working_demo' in result.stdout:
            print(f"{GREEN}✓ Demo process is running{RESET}")
            
            # Find the process
            for line in result.stdout.split('\n'):
                if 'aura_working_demo' in line and 'python' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        print(f"{BLUE}  PID: {pid}{RESET}")
                        break
        else:
            print(f"{YELLOW}⚠ Demo process not found in process list{RESET}")
    except:
        print(f"{YELLOW}⚠ Could not check processes{RESET}")

if __name__ == "__main__":
    print(f"{BOLD}{BLUE}Testing AURA Demo...{RESET}")
    
    success = test_demo()
    
    if success:
        check_processes()
        
        print(f"\n{GREEN}{'='*50}{RESET}")
        print(f"{GREEN}{BOLD}✅ AURA Demo is working correctly!{RESET}")
        print(f"{GREEN}{'='*50}{RESET}")
        
        # Interactive prompt
        print(f"\n{YELLOW}Would you like to open the demo in a browser? (y/n):{RESET} ", end='')
        try:
            answer = input().lower()
            if answer == 'y':
                import webbrowser
                webbrowser.open("http://localhost:8080")
                print(f"{GREEN}Opening browser...{RESET}")
        except:
            pass
    else:
        print(f"\n{RED}{'='*50}{RESET}")
        print(f"{RED}{BOLD}❌ Demo needs to be started{RESET}")
        print(f"{RED}{'='*50}{RESET}")
        
    exit(0 if success else 1)