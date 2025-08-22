#!/usr/bin/env python3
"""
AURA Intelligence Startup Helper
Simple script to start AURA with guidance
"""

import os
import sys
import subprocess
import time

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_banner():
    """Print AURA banner"""
    banner = f"""
{BOLD}{BLUE}
    ╔═══════════════════════════════════════════════════════╗
    ║              🧠 AURA INTELLIGENCE SYSTEM              ║
    ║                                                       ║
    ║     "We see the shape of failure before it happens"  ║
    ╚═══════════════════════════════════════════════════════╝
{RESET}
    Project ID: bc-a397ac41-47c3-4620-a5ec-c56fb1f50fd0
    Version: 2025.1.0
    Components: 213
    """
    print(banner)

def check_port(port=8080):
    """Check if port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def main():
    """Main startup function"""
    print_banner()
    
    print(f"\n{BOLD}🔍 System Check:{RESET}")
    
    # Check environment
    if os.path.exists('.env'):
        print(f"{GREEN}✓ Environment configured{RESET}")
    else:
        print(f"{RED}✗ .env file missing{RESET}")
        print(f"{YELLOW}  Create .env from .env.example{RESET}")
        return
    
    # Check demo
    demo_path = "demos/aura_working_demo_2025.py"
    if os.path.exists(demo_path):
        print(f"{GREEN}✓ Demo found{RESET}")
    else:
        print(f"{RED}✗ Demo not found{RESET}")
        return
    
    # Check port
    if check_port(8080):
        print(f"{GREEN}✓ Port 8080 available{RESET}")
    else:
        print(f"{YELLOW}⚠ Port 8080 in use{RESET}")
        print("  Killing existing process...")
        os.system("pkill -f 'aura_working_demo' 2>/dev/null || true")
        time.sleep(2)
    
    # Start options
    print(f"\n{BOLD}🚀 Startup Options:{RESET}")
    print(f"{BLUE}1.{RESET} Run Main Demo (Recommended)")
    print(f"{BLUE}2.{RESET} Run Benchmarks")
    print(f"{BLUE}3.{RESET} Start Infrastructure (Docker)")
    print(f"{BLUE}4.{RESET} View Documentation")
    print(f"{BLUE}5.{RESET} Exit")
    
    try:
        choice = input(f"\n{BOLD}Select option (1-5):{RESET} ")
        
        if choice == "1":
            print(f"\n{GREEN}Starting AURA Demo...{RESET}")
            print(f"{YELLOW}Opening http://localhost:8080 in 3 seconds...{RESET}")
            
            # Start demo
            subprocess.Popen([sys.executable, demo_path])
            
            # Wait and open browser
            time.sleep(3)
            try:
                import webbrowser
                webbrowser.open("http://localhost:8080")
            except:
                print(f"\n{YELLOW}Please open http://localhost:8080 in your browser{RESET}")
            
            print(f"\n{GREEN}✓ AURA is running!{RESET}")
            print(f"{BLUE}Press Ctrl+C to stop{RESET}")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Shutting down...{RESET}")
                os.system("pkill -f 'aura_working_demo' 2>/dev/null || true")
                
        elif choice == "2":
            print(f"\n{GREEN}Running benchmarks...{RESET}")
            subprocess.run([sys.executable, "benchmarks/aura_benchmark_100_agents.py"])
            
        elif choice == "3":
            print(f"\n{GREEN}Starting infrastructure...{RESET}")
            print(f"{YELLOW}This requires Docker. Starting docker-compose...{RESET}")
            os.chdir("infrastructure")
            subprocess.run(["docker-compose", "up", "-d"])
            os.chdir("..")
            print(f"\n{GREEN}Infrastructure started!{RESET}")
            print(f"Neo4j: http://localhost:7474")
            print(f"Grafana: http://localhost:3000")
            
        elif choice == "4":
            print(f"\n{BOLD}📚 Documentation:{RESET}")
            print("• README.md - Main documentation")
            print("• AURA_FINAL_INDEX.md - Complete system index")
            print("• documentation/ - All docs")
            
            # Try to open README
            try:
                with open("README.md", "r") as f:
                    lines = f.readlines()[:20]
                    print(f"\n{BOLD}README.md Preview:{RESET}")
                    for line in lines:
                        print(line.rstrip())
            except:
                pass
                
        elif choice == "5":
            print(f"\n{BLUE}Goodbye!{RESET}")
            return
            
        else:
            print(f"\n{RED}Invalid option{RESET}")
            
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        
    # Offer to run again
    if choice != "5":
        input(f"\n{BLUE}Press Enter to return to menu...{RESET}")
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Exiting...{RESET}")
        os.system("pkill -f 'aura_working_demo' 2>/dev/null || true")
        sys.exit(0)