#!/usr/bin/env python3
"""
Install AURA dependencies locally without virtual environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip with user flag"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def main():
    print("Installing AURA dependencies locally...")
    
    # Essential packages for AURA
    essential_packages = [
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "websockets==12.0",
        "httpx==0.26.0",
        "pydantic==2.5.3",
        "python-dotenv==1.0.0",
        "numpy==1.26.3",
        "scipy==1.11.4",
        "scikit-learn==1.4.0",
        "networkx==3.2.1",
        "aiofiles==23.2.1",
        "prometheus-client==0.19.0",
        "psutil==5.9.7",
        "colorama==0.4.6",
        "rich==13.7.0"
    ]
    
    # First, upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "pip"])
    
    # Install packages
    success_count = 0
    for package in essential_packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n✓ Installed {success_count}/{len(essential_packages)} packages")
    
    # Add user site-packages to Python path
    user_site = subprocess.check_output([sys.executable, "-m", "site", "--user-site"]).decode().strip()
    print(f"\nPackages installed to: {user_site}")
    print("\nTo use these packages, add to your Python scripts:")
    print(f"import sys; sys.path.insert(0, '{user_site}')")

if __name__ == "__main__":
    main()