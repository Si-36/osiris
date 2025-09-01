#!/usr/bin/env python3
"""
Verify Local Environment
=======================
Run this to confirm your environment is ready
"""

import subprocess
import sys

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package)
        return True, "Installed"
    except ImportError:
        return False, "Not installed"

def main():
    print("🔍 Verifying Local Environment")
    print("=" * 50)
    
    # Required packages
    packages = [
        "asyncpg",
        "duckdb", 
        "zstandard",
        "aiokafka",
        "torch",
        "numpy",
        "structlog"
    ]
    
    all_good = True
    
    for package in packages:
        installed, status = check_package(package)
        symbol = "✅" if installed else "❌"
        print(f"{symbol} {package}: {status}")
        if not installed:
            all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("✅ All dependencies installed!")
        print("\nYou can now run:")
        print("  python3 TEST_FULL_PERSISTENCE_INTEGRATION.py")
    else:
        print("❌ Some dependencies missing!")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements-persistence.txt")

if __name__ == "__main__":
    main()