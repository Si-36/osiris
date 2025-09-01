#!/usr/bin/env python3
"""
Complete Test Runner for Local Environment
=========================================
Run this in your local environment where dependencies are installed
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✅ Success!")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
        else:
            print(f"❌ Failed with exit code: {result.returncode}")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Run all tests in sequence"""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║          AURA PERSISTENCE - COMPLETE TEST SUITE               ║
║                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check environment
    print("\n📍 Environment Check:")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Test sequence
    tests = [
        # 1. Docker setup (optional)
        ("docker compose -f docker-compose.persistence.yml ps", 
         "Check Docker Services (optional)"),
        
        # 2. Persistence tests
        ("python3 test_persistence_integration_complete.py",
         "Test Persistence Integration"),
        
        # 3. Debug test if needed
        ("python3 test_persistence_complete_debug.py",
         "Run Debug Test Suite"),
        
        # 4. Full agent integration
        ("python3 test_all_agents_integrated.py",
         "Test All Agents Integration"),
    ]
    
    results = {}
    
    for cmd, desc in tests:
        # Skip docker if not available
        if "docker" in cmd:
            if subprocess.run("which docker", shell=True, capture_output=True).returncode != 0:
                print(f"\n⚠️  Skipping Docker test - Docker not available")
                continue
        
        success = run_command(cmd, desc)
        results[desc] = success
        
        # Stop on critical failures
        if not success and "Debug" not in desc:
            print(f"\n⚠️  Stopping due to failure in: {desc}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The persistence system is ready!")
        print("\nNext steps:")
        print("1. Start using enhanced agents with causal memory")
        print("2. Run benchmarks to see performance improvements")
        print("3. Deploy to production with confidence!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements-persistence.txt")
        print("2. Check Docker is running if using docker-compose")
        print("3. Verify PostgreSQL is accessible if not using legacy mode")

if __name__ == "__main__":
    main()