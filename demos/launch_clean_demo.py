#!/usr/bin/env python3
"""
🚀 AURA Intelligence - Clean Professional Launcher 2025
One command that actually works - no more errors!
"""

import os
import sys
import time
import subprocess
import asyncio
import signal
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CleanAURALauncher:
    """Professional, clean AURA launcher that actually works"""
    
    def __init__(self):
        self.demo_process = None
        self.running = True
        
    def display_banner(self):
        """Clean, professional banner"""
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  🚀 AURA Intelligence - Professional Demo 2025                       ║
║                                                                      ║
║  ✅ Clean Architecture + Professional Implementation                  ║
║  ⚡ GPU-Optimized (3.2ms BERT) + Real Components                     ║
║  🔧 Production-Ready + Error-Free Launch                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
    
    def check_requirements(self):
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 9):
            logger.error("❌ Python 3.9+ required")
            return False
        
        # Check required files
        required_files = [
            "aura_professional_demo.py",
            "test_clean_demo.py",
            "core/src/aura_intelligence/components/real_components.py"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Missing required file: {file_path}")
                return False
        
        # Check dependencies
        try:
            import uvicorn
            import fastapi
            logger.info("✅ All dependencies available")
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.info("💡 Install with: pip install fastapi uvicorn")
            return False
        
        logger.info("✅ All requirements satisfied")
        return True
    
    def show_menu(self):
        """Show clean, simple menu"""
        print("\n🎮 Choose Your Experience:")
        print("=" * 40)
        print("  1. 🚀 Launch Demo (Recommended)")
        print("  2. 🧪 Run Quick Test")
        print("  3. 🌐 Launch + Test")
        print("  0. Exit")
        print("=" * 40)
    
    def get_choice(self):
        """Get user choice with validation"""
        while True:
            try:
                choice = input("👉 Enter choice (0-3): ").strip()
                if choice in ['0', '1', '2', '3']:
                    return choice
                print("❌ Invalid choice. Enter 0-3.")
            except KeyboardInterrupt:
                return '0'
    
    async def launch_demo(self):
        """Launch the professional demo"""
        logger.info("🚀 Starting AURA Professional Demo...")
        
        try:
            # Start demo server
            self.demo_process = subprocess.Popen([
                sys.executable, "aura_professional_demo.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info("✅ Demo server starting...")
            logger.info("🌐 Web interface: http://localhost:8080")
            logger.info("📚 API documentation: http://localhost:8080/docs")
            logger.info("🛑 Press Ctrl+C to stop")
            
            # Wait for process
            while self.running:
                if self.demo_process.poll() is not None:
                    # Process ended
                    stdout, stderr = self.demo_process.communicate()
                    if stderr:
                        logger.error(f"❌ Demo failed: {stderr.decode()}")
                    break
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("👋 Demo stopped by user")
        finally:
            self.cleanup()
    
    async def run_test(self):
        """Run the clean test suite"""
        logger.info("🧪 Running Clean Test Suite...")
        
        try:
            # Start demo in background
            self.demo_process = subprocess.Popen([
                sys.executable, "aura_professional_demo.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for startup
            await asyncio.sleep(3)
            
            # Run tests
            test_process = subprocess.run([
                sys.executable, "test_clean_demo.py"
            ], capture_output=False, timeout=60)
            
            if test_process.returncode == 0:
                logger.info("🎉 All tests passed!")
            else:
                logger.warning("⚠️ Some tests had issues")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Tests timed out")
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
        finally:
            self.cleanup()
    
    async def launch_and_test(self):
        """Launch demo and run tests"""
        logger.info("🎪 Starting Complete Demo Experience...")
        
        try:
            # Start demo
            self.demo_process = subprocess.Popen([
                sys.executable, "aura_professional_demo.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info("✅ Demo server starting...")
            await asyncio.sleep(5)  # Wait for full startup
            
            # Run tests
            logger.info("🧪 Running validation tests...")
            test_process = subprocess.run([
                sys.executable, "test_clean_demo.py"
            ], capture_output=False, timeout=60)
            
            if test_process.returncode == 0:
                logger.info("\n🎉 DEMO READY AND VALIDATED!")
                logger.info("🌐 Open: http://localhost:8080")
                logger.info("🎮 Explore the scenarios in your browser")
                logger.info("🛑 Press Ctrl+C when done")
                
                # Keep running until user stops
                while self.running:
                    if self.demo_process.poll() is not None:
                        break
                    await asyncio.sleep(1)
            else:
                logger.warning("⚠️ Demo launched but tests had issues")
                
        except Exception as e:
            logger.error(f"❌ Launch failed: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        if self.demo_process:
            try:
                self.demo_process.terminate()
                self.demo_process.wait(timeout=5)
            except:
                try:
                    self.demo_process.kill()
                except:
                    pass
            self.demo_process = None
        logger.info("✅ Cleanup complete")
    
    def setup_signals(self):
        """Setup signal handlers"""
        def signal_handler(signum, frame):
            self.running = False
            logger.info("👋 Shutting down...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main execution loop"""
        self.setup_signals()
        
        while True:
            self.display_banner()
            
            if not self.check_requirements():
                logger.error("❌ System requirements not met")
                break
            
            self.show_menu()
            choice = self.get_choice()
            
            if choice == '0':
                logger.info("👋 Goodbye!")
                break
            elif choice == '1':
                await self.launch_demo()
            elif choice == '2':
                await self.run_test()
            elif choice == '3':
                await self.launch_and_test()
            
            if not self.running:
                break
                
            input("\n👉 Press Enter to continue...")

def main():
    """Clean main entry point"""
    if not os.path.exists("aura_professional_demo.py"):
        print("❌ Please run this from the osiris-2 directory")
        print("   cd /home/sina/projects/osiris-2")
        print("   python3 launch_clean_demo.py")
        return 1
    
    launcher = CleanAURALauncher()
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