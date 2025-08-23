#!/usr/bin/env python3
"""
🚀 AURA Intelligence Setup and Run Script
✨ Automatically sets up dependencies and starts all services
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

class AURASetup:
    """Complete AURA setup and startup manager"""
    
    def __init__(self):
        self.workspace = Path(__file__).parent
        self.errors = []
        self.services = []
        
    def check_command(self, cmd):
        """Check if a command exists"""
        try:
            subprocess.run(['which', cmd], check=True, capture_output=True)
            return True
        except:
            return False
            
    def install_python_deps(self):
        """Install essential Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        essential_deps = [
            "fastapi",
            "uvicorn[standard]",
            "websockets",
            "httpx",
            "pydantic",
            "python-dotenv",
            "numpy",
            "scipy",
            "scikit-learn",
            "networkx",
            "aiofiles",
            "prometheus-client",
            "psutil",
            "colorama",
            "rich",
            "aioredis",
            "asyncio",
            "plotext"
        ]
        
        # Try to install using pip with --user flag
        for dep in essential_deps:
            try:
                print(f"  Installing {dep}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", dep],
                    capture_output=True,
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f"  ⚠️ Failed to install {dep}")
                
    def start_infrastructure(self):
        """Start infrastructure services"""
        print("\n🏗️ Starting infrastructure services...")
        
        # Check if Docker is available
        if self.check_command('docker'):
            docker_compose = self.workspace / "infrastructure" / "docker-compose.yml"
            if docker_compose.exists():
                print("  Starting Docker services...")
                try:
                    subprocess.run(
                        ["docker-compose", "-f", str(docker_compose), "up", "-d"],
                        check=True,
                        capture_output=True
                    )
                    print("  ✅ Docker services started")
                except:
                    print("  ⚠️ Docker services failed to start")
            else:
                print("  ⚠️ docker-compose.yml not found")
        else:
            print("  ⚠️ Docker not available - skipping infrastructure")
            
    def start_demo(self):
        """Start the main demo"""
        print("\n🎯 Starting AURA demo...")
        
        demo_file = self.workspace / "demos" / "aura_working_demo_2025.py"
        if demo_file.exists():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(demo_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.services.append(("Demo", process))
                time.sleep(2)  # Wait for startup
                
                # Check if running
                if process.poll() is None:
                    print("  ✅ Demo started at http://localhost:8080")
                else:
                    print("  ❌ Demo failed to start")
            except Exception as e:
                print(f"  ❌ Demo error: {e}")
        else:
            print("  ❌ Demo file not found")
            
    def start_api(self):
        """Start the API server"""
        print("\n🌐 Starting API server...")
        
        api_file = self.workspace / "src" / "aura" / "api" / "unified_api.py"
        if api_file.exists():
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "src.aura.api.unified_api:app", "--reload"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.services.append(("API", process))
                time.sleep(2)  # Wait for startup
                
                # Check if running
                result = subprocess.run(
                    ["curl", "-s", "http://localhost:8000/health"],
                    capture_output=True
                )
                if result.returncode == 0:
                    print("  ✅ API started at http://localhost:8000")
                else:
                    print("  ⚠️ API may not be fully ready")
            except Exception as e:
                print(f"  ❌ API error: {e}")
        else:
            print("  ❌ API file not found")
            
    def start_monitoring(self):
        """Start monitoring dashboard"""
        print("\n📊 Starting monitoring...")
        
        monitor_file = self.workspace / "start_monitoring_v2.py"
        if monitor_file.exists():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(monitor_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.services.append(("Monitoring", process))
                print("  ✅ Monitoring started")
            except Exception as e:
                print(f"  ❌ Monitoring error: {e}")
                
    def check_status(self):
        """Check system status"""
        print("\n📋 System Status:")
        
        # Component counts
        try:
            sys.path.insert(0, str(self.workspace / "src"))
            from aura.core.system import AURASystem, AURAConfig
            
            config = AURAConfig()
            system = AURASystem(config)
            components = system.get_all_components()
            
            print("\n  Component Counts:")
            for category, items in components.items():
                print(f"    {category.upper()}: {len(items)}")
                
        except Exception as e:
            print(f"  ⚠️ Could not load components: {e}")
            
        # Service status
        print("\n  Running Services:")
        for name, process in self.services:
            if process.poll() is None:
                print(f"    ✅ {name}")
            else:
                print(f"    ❌ {name} (stopped)")
                
    def run_tests(self):
        """Run basic tests"""
        print("\n🧪 Running tests...")
        
        # Try simple test first
        test_file = self.workspace / "test_everything_v2.py"
        if test_file.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse results
                if "Pass Rate:" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "Pass Rate:" in line:
                            print(f"  {line.strip()}")
                            break
                else:
                    print("  ⚠️ Test output not recognized")
                    
            except Exception as e:
                print(f"  ❌ Test error: {e}")
                
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("🎉 AURA Intelligence System Status")
        print("="*60)
        
        print("\n📌 Quick Access:")
        print("  - Demo: http://localhost:8080")
        print("  - API: http://localhost:8000")
        print("  - API Docs: http://localhost:8000/docs")
        print("  - Metrics: http://localhost:8000/metrics")
        
        if self.check_command('docker'):
            print("\n🐳 Docker Services:")
            print("  - Neo4j: http://localhost:7474")
            print("  - Prometheus: http://localhost:9090")
            print("  - Grafana: http://localhost:3000")
            
        print("\n📚 Documentation:")
        print("  - README.md - Main documentation")
        print("  - AURA_ULTIMATE_INDEX_2025.md - Complete component index")
        print("  - AURA_COMPLETE_DOCUMENTATION_2025.md - Full system docs")
        
        print("\n🔧 Commands:")
        print("  - Run tests: python3 test_everything_v2.py")
        print("  - Run benchmark: python3 benchmarks/aura_benchmark_100_agents.py")
        print("  - Stop services: kill <PID> or Ctrl+C")
        
        print("\n✨ System is ready for use!")
        
    def run(self):
        """Run complete setup"""
        print("🚀 AURA Intelligence Setup")
        print("="*60)
        
        # Install dependencies
        self.install_python_deps()
        
        # Start services
        self.start_infrastructure()
        self.start_demo()
        self.start_api()
        self.start_monitoring()
        
        # Check status
        self.check_status()
        
        # Run tests
        self.run_tests()
        
        # Summary
        self.print_summary()
        
        # Keep running
        print("\n💡 Press Ctrl+C to stop all services...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping all services...")
            for name, process in self.services:
                if process.poll() is None:
                    process.terminate()
                    print(f"  Stopped {name}")
            print("👋 Goodbye!")

if __name__ == "__main__":
    setup = AURASetup()
    setup.run()