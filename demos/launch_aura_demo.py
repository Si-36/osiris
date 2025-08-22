#!/usr/bin/env python3
"""
🚀 AURA Intelligence - Ultimate Launch Script
ONE COMMAND to see everything working together!
"""

import os
import sys
import time
import subprocess
import asyncio
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AURADemoLauncher:
    """Ultimate AURA Intelligence demo launcher"""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
        # Demo options
        self.options = {
            '1': {'name': '🎯 Quick Demo', 'desc': 'Launch demo app + run sample tests', 'action': 'quick'},
            '2': {'name': '🧪 Full E2E Tests', 'desc': 'Complete test suite with performance analysis', 'action': 'full_test'},
            '3': {'name': '🔥 Live Demo Server', 'desc': 'Interactive web interface with real-time monitoring', 'action': 'live_demo'},
            '4': {'name': '⚡ Performance Benchmark', 'desc': 'GPU performance validation and benchmarking', 'action': 'benchmark'},
            '5': {'name': '🎪 Everything!', 'desc': 'Full demo + tests + monitoring + benchmarks', 'action': 'everything'}
        }
    
    def display_banner(self):
        """Display the AURA banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  🚀 AURA Intelligence - Complete E2E Demo Launcher 2025              ║
║                                                                      ║
║  🎯 Production-Ready AI System with:                                 ║
║     ⚡ GPU Acceleration (3.2ms BERT processing)                      ║
║     🧠 Liquid Neural Networks + Multi-Agent Coordination             ║
║     📊 Real-time Monitoring + Business Intelligence                  ║
║     🐳 Container Orchestration + Kubernetes Deployment              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def display_menu(self):
        """Display the demo menu"""
        print("\n🎮 Choose Your Demo Experience:")
        print("=" * 70)
        
        for key, option in self.options.items():
            print(f"  {key}. {option['name']}")
            print(f"     {option['desc']}")
            print()
        
        print("  0. Exit")
        print("=" * 70)
    
    def get_user_choice(self):
        """Get user's demo choice"""
        while True:
            try:
                choice = input("👉 Enter your choice (1-5, 0 to exit): ").strip()
                
                if choice == '0':
                    return 'exit'
                elif choice in self.options:
                    return self.options[choice]['action']
                else:
                    print("❌ Invalid choice. Please enter 1-5 or 0.")
                    
            except KeyboardInterrupt:
                return 'exit'
    
    async def run_quick_demo(self):
        """Run quick demo with sample tests"""
        logger.info("🎯 Running Quick Demo...")
        
        print("\n🚀 Starting AURA Intelligence Demo...")
        print("   📊 Web interface: http://localhost:8080")
        print("   📈 Real-time dashboard: http://localhost:8766")
        print("   🎮 Press Ctrl+C when ready to continue...")
        
        try:
            # Start demo application
            demo_process = subprocess.Popen([
                sys.executable, "aura_complete_demo.py"
            ])
            self.processes.append(demo_process)
            
            # Wait for user to explore
            await self.wait_for_interrupt()
            
            print("\n🧪 Running sample tests...")
            
            # Run a few quick tests
            test_process = subprocess.run([
                sys.executable, "-c", """
import asyncio
import requests
import time

async def quick_test():
    print('⏳ Waiting for system...')
    for i in range(30):
        try:
            response = requests.get('http://localhost:8080/health', timeout=2)
            if response.status_code == 200:
                print('✅ System ready!')
                break
        except:
            pass
        await asyncio.sleep(1)
    
    print('🧪 Testing AI reasoning scenario...')
    try:
        response = requests.post('http://localhost:8080/demo', json={
            'task': 'ai_reasoning',
            'data': {'text': 'Quick demo test', 'complexity': 'low'},
            'use_gpu': True,
            'use_agents': True
        }, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'success':
                print(f'✅ Demo test passed in {result["performance_metrics"]["total_processing_time_ms"]:.1f}ms')
                print(f'   📊 Components used: {result["performance_metrics"]["component_count"]}')
                print(f'   📊 Efficiency: {result["performance_metrics"]["efficiency_score"]*100:.1f}%')
            else:
                print('❌ Demo test failed')
        else:
            print(f'❌ Demo test failed: HTTP {response.status_code}')
    except Exception as e:
        print(f'❌ Demo test error: {e}')

asyncio.run(quick_test())
                """
            ], capture_output=True, text=True)
            
            print(test_process.stdout)
            if test_process.stderr:
                print("Errors:", test_process.stderr)
            
        finally:
            self.cleanup_processes()
    
    async def run_full_test(self):
        """Run complete E2E test suite"""
        logger.info("🧪 Running Full E2E Test Suite...")
        
        print("\n🧪 Launching Complete E2E Test Suite...")
        print("   This will take 2-3 minutes to complete")
        print("   Testing all components, scenarios, and performance")
        
        try:
            # Run comprehensive tests
            test_process = subprocess.run([
                sys.executable, "test_e2e_complete.py"
            ], capture_output=False)  # Show output directly
            
            # Check results
            if test_process.returncode == 0:
                print("\n🎉 All E2E tests passed! System is production ready!")
            else:
                print("\n⚠️  Some tests failed. Check results above.")
                
        except KeyboardInterrupt:
            print("\n👋 Tests interrupted by user")
    
    async def run_live_demo(self):
        """Run interactive live demo"""
        logger.info("🔥 Running Live Demo Server...")
        
        print("\n🔥 Starting Interactive Live Demo...")
        print("   🌐 Open your browser to: http://localhost:8080")
        print("   🎮 Interactive web interface with real-time metrics")
        print("   📊 WebSocket dashboard at: http://localhost:8766")
        print("   🛑 Press Ctrl+C to stop")
        
        try:
            # Start demo server
            demo_process = subprocess.Popen([
                sys.executable, "aura_complete_demo.py"
            ])
            self.processes.append(demo_process)
            
            print(f"\n✅ Demo server started (PID: {demo_process.pid})")
            print("👆 Click the link above to start exploring!")
            
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        finally:
            self.cleanup_processes()
    
    async def run_benchmark(self):
        """Run performance benchmarks"""
        logger.info("⚡ Running Performance Benchmarks...")
        
        print("\n⚡ Running Performance Benchmarks...")
        print("   🔬 GPU acceleration validation")
        print("   📊 Comprehensive performance analysis")
        
        try:
            # First run the optimization validation
            print("📋 Step 1: Running optimization validation...")
            opt_process = subprocess.run([
                sys.executable, "test_production_optimization_complete.py"
            ], capture_output=False)
            
            # Then run GPU benchmarks if available
            if os.path.exists("scripts/gpu-benchmark.sh"):
                print("\n📋 Step 2: Running GPU benchmarks...")
                print("   (This requires the demo server to be running)")
                
                # Start demo in background
                demo_process = subprocess.Popen([
                    sys.executable, "aura_complete_demo.py"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.processes.append(demo_process)
                
                # Wait a bit for startup
                await asyncio.sleep(10)
                
                # Run GPU benchmark
                benchmark_process = subprocess.run([
                    "bash", "scripts/gpu-benchmark.sh", "bert"
                ], capture_output=False)
            
            print("\n🎯 Benchmark complete! Check results above.")
            
        finally:
            self.cleanup_processes()
    
    async def run_everything(self):
        """Run everything - the ultimate demo"""
        logger.info("🎪 Running EVERYTHING!")
        
        print("\n🎪 ULTIMATE AURA INTELLIGENCE DEMONSTRATION")
        print("=" * 70)
        print("This will run ALL demos in sequence:")
        print("1. 🏥 System health validation")
        print("2. ⚡ Performance optimization tests")
        print("3. 🧪 Complete E2E test suite")  
        print("4. 🔥 Live interactive demo")
        print()
        
        try:
            # Step 1: Health validation
            print("📋 STEP 1: System Health Validation")
            print("-" * 40)
            health_process = subprocess.run([
                sys.executable, "test_production_optimization_complete.py"
            ], capture_output=False)
            
            input("\n👉 Press Enter to continue to E2E tests...")
            
            # Step 2: E2E tests
            print("\n📋 STEP 2: Complete E2E Test Suite")
            print("-" * 40)
            test_process = subprocess.run([
                sys.executable, "test_e2e_complete.py"
            ], capture_output=False)
            
            input("\n👉 Press Enter to continue to live demo...")
            
            # Step 3: Live demo
            print("\n📋 STEP 3: Live Interactive Demo")
            print("-" * 40)
            print("🔥 Starting live demo server...")
            print("🌐 Open: http://localhost:8080")
            print("📊 Dashboard: http://localhost:8766")
            print("🛑 Press Ctrl+C when done exploring")
            
            demo_process = subprocess.Popen([
                sys.executable, "aura_complete_demo.py"
            ])
            self.processes.append(demo_process)
            
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            print("\n👋 Ultimate demo stopped by user")
        finally:
            self.cleanup_processes()
            
        print("\n🎉 ULTIMATE DEMO COMPLETE!")
        print("📊 All results have been saved to respective JSON files")
    
    async def wait_for_interrupt(self):
        """Wait for user interrupt"""
        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            pass
    
    def cleanup_processes(self):
        """Clean up all spawned processes"""
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        self.processes.clear()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            self.running = False
            self.cleanup_processes()
            print("\n👋 AURA Demo launcher shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main launcher execution"""
        self.setup_signal_handlers()
        
        while True:
            try:
                self.display_banner()
                self.display_menu()
                
                choice = self.get_user_choice()
                
                if choice == 'exit':
                    print("\n👋 Thank you for exploring AURA Intelligence!")
                    break
                elif choice == 'quick':
                    await self.run_quick_demo()
                elif choice == 'full_test':
                    await self.run_full_test()
                elif choice == 'live_demo':
                    await self.run_live_demo()
                elif choice == 'benchmark':
                    await self.run_benchmark()
                elif choice == 'everything':
                    await self.run_everything()
                
                if choice != 'live_demo':  # live_demo handles its own continuation
                    input("\n👉 Press Enter to return to main menu...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\n👉 Press Enter to continue...")
        
        self.cleanup_processes()


def main():
    """Main execution"""
    # Check if we're in the right directory
    if not os.path.exists("aura_complete_demo.py"):
        print("❌ Please run this script from the osiris-2 project directory")
        print("   cd /home/sina/projects/osiris-2")
        print("   python3 launch_aura_demo.py")
        return 1
    
    # Check dependencies
    try:
        import uvicorn
        import fastapi
        import websockets
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install fastapi uvicorn websockets")
        return 1
    
    # Run the launcher
    launcher = AURADemoLauncher()
    try:
        asyncio.run(launcher.run())
        return 0
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())