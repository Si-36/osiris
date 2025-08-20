#!/usr/bin/env python3
"""
Deploy REAL AURA system with fixed integrations
Uses your existing working_aura_api.py with real infrastructure
"""
import subprocess
import sys
import time
import requests
from pathlib import Path

def check_dependency(package_name: str) -> bool:
    """Check if a package is installed"""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def install_real_dependencies():
    """Install real dependencies"""
    print("📦 Installing real dependencies...")
    
    dependencies = [
        "kafka-python==2.0.2",
        "neo4j==5.15.0", 
        "ray[serve]==2.8.0",
        "redis==5.0.1"
    ]
    
    for dep in dependencies:
        package_name = dep.split('==')[0]
        if not check_dependency(package_name):
            print(f"Installing {package_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         capture_output=True)

def start_infrastructure():
    """Start real infrastructure with Docker"""
    print("🐳 Starting real infrastructure...")
    
    compose_file = Path(__file__).parent / "infrastructure" / "docker-compose.real.yml"
    
    if compose_file.exists():
        try:
            subprocess.run([
                "docker-compose", 
                "-f", str(compose_file), 
                "up", "-d"
            ], check=True)
            print("✅ Infrastructure started")
            
            # Wait for services to be ready
            print("⏳ Waiting for services to be ready...")
            time.sleep(10)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start infrastructure: {e}")
            return False
    else:
        print("⚠️  Docker compose file not found, skipping infrastructure")
        return False

def test_api():
    """Test the working API"""
    print("🧪 Testing AURA API...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API Health: {health_data['working_components']}/5 components working")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API not responding: {e}")
        return False

def start_api():
    """Start the working AURA API"""
    print("🚀 Starting AURA API...")
    
    api_file = Path(__file__).parent / "working_aura_api.py"
    
    if api_file.exists():
        try:
            # Start API in background
            process = subprocess.Popen([
                sys.executable, str(api_file)
            ])
            
            # Wait for API to start
            time.sleep(5)
            
            # Test if API is responding
            if test_api():
                print("✅ AURA API started successfully")
                return process
            else:
                print("❌ API failed to start properly")
                process.terminate()
                return None
                
        except Exception as e:
            print(f"❌ Failed to start API: {e}")
            return None
    else:
        print("❌ working_aura_api.py not found")
        return None

def main():
    """Deploy the real system"""
    print("🎯 Deploying REAL AURA Intelligence System")
    print("=" * 50)
    
    # Step 1: Install dependencies
    install_real_dependencies()
    
    # Step 2: Start infrastructure
    infrastructure_started = start_infrastructure()
    
    # Step 3: Start API
    api_process = start_api()
    
    if api_process:
        print("=" * 50)
        print("🎉 REAL AURA System deployed successfully!")
        print()
        print("📍 API Endpoints:")
        print("   • Health: http://localhost:8080/health")
        print("   • Docs: http://localhost:8080/docs")
        print("   • Neural: http://localhost:8080/neural/process")
        print("   • Memory: http://localhost:8080/memory/store")
        print()
        
        if infrastructure_started:
            print("🔧 Infrastructure:")
            print("   • Redis: localhost:6379")
            print("   • Kafka: localhost:9092") 
            print("   • Neo4j: http://localhost:7474")
            print("   • Ray: http://localhost:8265")
            print()
        
        print("Press Ctrl+C to stop the system")
        
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping AURA system...")
            api_process.terminate()
            
            if infrastructure_started:
                compose_file = Path(__file__).parent / "infrastructure" / "docker-compose.real.yml"
                subprocess.run([
                    "docker-compose", 
                    "-f", str(compose_file), 
                    "down"
                ])
            
            print("✅ System stopped")
    else:
        print("❌ Failed to deploy system")
        sys.exit(1)

if __name__ == "__main__":
    main()