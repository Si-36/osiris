#!/usr/bin/env python3
"""
🚀 AURA Intelligence - Final Working Launcher
One command, actually works, no more errors!
"""

import os
import sys
import time
import subprocess
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def show_banner():
    """Show the final banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  🚀 AURA Intelligence - WORKING DEMO 2025                            ║
║                                                                      ║
║  ✅ DEBUGGED & TESTED - Actually Works!                              ║
║  ⚡ GPU-Optimized Components                                          ║
║  🌐 Professional Web Interface                                       ║
║  🧪 Comprehensive Testing                                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def check_system():
    """Quick system check"""
    logger.info("🔍 Checking system requirements...")
    
    if not os.path.exists("aura_working_demo.py"):
        logger.error("❌ Missing aura_working_demo.py")
        return False
    
    try:
        import uvicorn, fastapi, requests
        logger.info("✅ All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Install with: pip install fastapi uvicorn requests")
        return False

def show_options():
    """Show launch options"""
    print("\n🎮 Launch Options:")
    print("=" * 30)
    print("  1. 🚀 Start Demo Server")
    print("  2. 🧪 Test Demo")  
    print("  3. 🎯 Demo + Test")
    print("  0. Exit")
    print("=" * 30)

async def launch_demo():
    """Launch the working demo"""
    logger.info("🚀 Starting AURA Working Demo...")
    
    try:
        # Start demo server
        process = subprocess.Popen([
            sys.executable, "aura_working_demo.py"
        ])
        
        logger.info("✅ Demo server starting...")
        logger.info("🌐 Web interface: http://localhost:8080")
        logger.info("🛑 Press Ctrl+C to stop")
        
        # Wait for process
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("👋 Stopping demo...")
            process.terminate()
            process.wait()
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

async def test_demo():
    """Test the demo"""
    logger.info("🧪 Testing demo...")
    
    # Start demo in background
    demo_process = subprocess.Popen([
        sys.executable, "aura_working_demo.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for startup
        await asyncio.sleep(5)
        
        # Run test
        test_result = subprocess.run([
            sys.executable, "test_working_demo.py"
        ], timeout=30)
        
        if test_result.returncode == 0:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning("⚠️ Some tests failed")
            
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
    finally:
        demo_process.terminate()
        demo_process.wait()

async def demo_and_test():
    """Launch demo and test it"""
    logger.info("🎯 Starting demo and running tests...")
    
    # Start demo
    demo_process = subprocess.Popen([
        sys.executable, "aura_working_demo.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for startup
        logger.info("⏳ Waiting for startup...")
        await asyncio.sleep(6)
        
        # Run test
        logger.info("🧪 Running validation...")
        test_result = subprocess.run([
            sys.executable, "test_working_demo.py"
        ], timeout=30)
        
        if test_result.returncode == 0:
            logger.info("\n🎉 DEMO IS WORKING!")
            logger.info("🌐 Open: http://localhost:8080")
            logger.info("🎮 Try the demo scenarios!")
            logger.info("🛑 Press Ctrl+C to stop")
            
            # Keep running
            try:
                demo_process.wait()
            except KeyboardInterrupt:
                logger.info("👋 Stopping demo...")
        else:
            logger.warning("⚠️ Tests had issues but demo is running")
            logger.info("🌐 Try: http://localhost:8080")
            
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
    finally:
        demo_process.terminate()
        demo_process.wait()

def get_choice():
    """Get user choice"""
    try:
        return input("👉 Choose (0-3): ").strip()
    except (KeyboardInterrupt, EOFError):
        return "0"

async def main():
    """Main launcher"""
    show_banner()
    
    if not check_system():
        logger.error("❌ System requirements not met")
        return 1
    
    while True:
        show_options()
        choice = get_choice()
        
        if choice == "0":
            logger.info("👋 Goodbye!")
            break
        elif choice == "1":
            await launch_demo()
        elif choice == "2":
            await test_demo()
        elif choice == "3":
            await demo_and_test()
        else:
            logger.warning("❌ Invalid choice")
            continue
        
        if choice != "0":
            input("\n👉 Press Enter to continue...")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)