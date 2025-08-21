"""
Fix Common Integration Issues for AURA Microservices
Handles missing dependencies, service startup, and configuration
"""

import subprocess
import sys
import os
import time
import psutil
import socket
from pathlib import Path


class IntegrationFixer:
    """Diagnose and fix integration issues"""
    
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
        
    def check_port_availability(self, port: int, service_name: str) -> bool:
        """Check if a port is available or in use"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is active ({service_name})")
            return True
        else:
            print(f"❌ Port {port} is not accessible ({service_name})")
            self.issues.append(f"{service_name} not running on port {port}")
            return False
    
    def check_services(self):
        """Check if all required services are running"""
        print("🔍 Checking service availability...\n")
        
        services = [
            (8000, "Neuromorphic Service"),
            (8001, "Memory Service"),
            (6379, "Redis"),
            (7687, "Neo4j")
        ]
        
        all_running = True
        for port, name in services:
            if not self.check_port_availability(port, name):
                all_running = False
                
        return all_running
    
    def check_python_dependencies(self):
        """Check if all required Python packages are installed"""
        print("\n🔍 Checking Python dependencies...\n")
        
        required_packages = {
            'neuromorphic': [
                'fastapi', 'uvicorn', 'spikingjelly', 'torch', 
                'opentelemetry-api', 'structlog'
            ],
            'memory': [
                'fastapi', 'uvicorn', 'redis', 'neo4j', 'faiss-cpu',
                'numpy', 'opentelemetry-api', 'structlog'
            ]
        }
        
        missing = {}
        
        for service, packages in required_packages.items():
            service_missing = []
            for package in packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    service_missing.append(package)
                    
            if service_missing:
                missing[service] = service_missing
                self.issues.append(f"{service} missing packages: {', '.join(service_missing)}")
        
        if missing:
            print("❌ Missing packages found")
            for service, packages in missing.items():
                print(f"   {service}: {', '.join(packages)}")
        else:
            print("✅ All Python dependencies installed")
            
        return len(missing) == 0
    
    def fix_missing_packages(self):
        """Install missing packages"""
        print("\n🔧 Installing missing packages...\n")
        
        # Check which service directories exist
        neuromorphic_dir = Path("/workspace/aura-microservices/neuromorphic")
        memory_dir = Path("/workspace/aura-microservices/memory")
        
        if neuromorphic_dir.exists() and (neuromorphic_dir / "requirements.txt").exists():
            print("Installing Neuromorphic dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r",
                str(neuromorphic_dir / "requirements.txt")
            ], capture_output=True)
            self.fixes_applied.append("Installed Neuromorphic dependencies")
            
        if memory_dir.exists() and (memory_dir / "requirements.txt").exists():
            print("Installing Memory dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(memory_dir / "requirements.txt")
            ], capture_output=True)
            self.fixes_applied.append("Installed Memory dependencies")
    
    def start_docker_services(self):
        """Start required Docker services"""
        print("\n🐳 Starting Docker services...\n")
        
        docker_commands = [
            ("Redis", ["docker", "run", "-d", "--name", "redis-aura", "-p", "6379:6379", "redis:latest"]),
            ("Neo4j", ["docker", "run", "-d", "--name", "neo4j-aura", "-p", "7474:7474", "-p", "7687:7687", 
                      "-e", "NEO4J_AUTH=neo4j/password", "neo4j:latest"])
        ]
        
        for service_name, cmd in docker_commands:
            try:
                # Check if container already exists
                check = subprocess.run(["docker", "ps", "-a", "-q", "-f", f"name={cmd[3]}"], 
                                     capture_output=True, text=True)
                
                if check.stdout.strip():
                    # Container exists, try to start it
                    subprocess.run(["docker", "start", cmd[3]], capture_output=True)
                    print(f"✅ Started existing {service_name} container")
                else:
                    # Create new container
                    subprocess.run(cmd, capture_output=True)
                    print(f"✅ Created and started {service_name} container")
                    
                self.fixes_applied.append(f"Started {service_name}")
                
            except Exception as e:
                print(f"❌ Failed to start {service_name}: {e}")
                self.issues.append(f"Could not start {service_name} Docker container")
    
    def create_startup_scripts(self):
        """Create convenience startup scripts"""
        print("\n📝 Creating startup scripts...\n")
        
        # Create start_services.sh
        start_script = """#!/bin/bash
# Start AURA Microservices

echo "🚀 Starting AURA Microservices..."

# Start Docker services
echo "Starting Docker services..."
docker start redis-aura neo4j-aura 2>/dev/null || {
    docker run -d --name redis-aura -p 6379:6379 redis:latest
    docker run -d --name neo4j-aura -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
}

# Wait for services
echo "Waiting for services to be ready..."
sleep 5

# Start Neuromorphic service
echo "Starting Neuromorphic service..."
cd /workspace/aura-microservices/neuromorphic
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
NEURO_PID=$!

# Start Memory service
echo "Starting Memory service..."
cd /workspace/aura-microservices/memory
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload &
MEMORY_PID=$!

echo ""
echo "✅ Services started!"
echo "   Neuromorphic: http://localhost:8000"
echo "   Memory: http://localhost:8001"
echo "   Redis: localhost:6379"
echo "   Neo4j: http://localhost:7474"
echo ""
echo "PIDs: Neuromorphic=$NEURO_PID, Memory=$MEMORY_PID"
echo "To stop: kill $NEURO_PID $MEMORY_PID"
"""
        
        script_path = Path("/workspace/aura-microservices/start_services.sh")
        script_path.write_text(start_script)
        script_path.chmod(0o755)
        print("✅ Created start_services.sh")
        
        # Create stop_services.sh
        stop_script = """#!/bin/bash
# Stop AURA Microservices

echo "🛑 Stopping AURA Microservices..."

# Kill Python services
pkill -f "uvicorn src.api.main:app"

# Stop Docker services
docker stop redis-aura neo4j-aura

echo "✅ Services stopped"
"""
        
        stop_path = Path("/workspace/aura-microservices/stop_services.sh")
        stop_path.write_text(stop_script)
        stop_path.chmod(0o755)
        print("✅ Created stop_services.sh")
        
        self.fixes_applied.append("Created startup scripts")
    
    def create_missing_init_files(self):
        """Create missing __init__.py files"""
        print("\n📁 Checking Python module structure...\n")
        
        base_dirs = [
            "/workspace/aura-microservices/neuromorphic/src",
            "/workspace/aura-microservices/memory/src"
        ]
        
        for base_dir in base_dirs:
            base_path = Path(base_dir)
            if base_path.exists():
                for dir_path in base_path.rglob("*"):
                    if dir_path.is_dir() and not dir_path.name.startswith('.'):
                        init_file = dir_path / "__init__.py"
                        if not init_file.exists():
                            init_file.touch()
                            print(f"✅ Created {init_file}")
                            self.fixes_applied.append(f"Created {init_file}")
    
    def run_diagnostics_and_fix(self):
        """Run full diagnostics and apply fixes"""
        print("=" * 60)
        print("🔧 AURA Microservices Integration Fixer")
        print("=" * 60)
        
        # Check services
        services_ok = self.check_services()
        
        # Check dependencies
        deps_ok = self.check_python_dependencies()
        
        # Apply fixes if needed
        if not deps_ok:
            self.fix_missing_packages()
            
        if not services_ok:
            self.start_docker_services()
            
        # Always create helpful scripts
        self.create_startup_scripts()
        self.create_missing_init_files()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Summary")
        print("=" * 60)
        
        if self.issues:
            print("\n⚠️  Issues found:")
            for issue in self.issues:
                print(f"   - {issue}")
                
        if self.fixes_applied:
            print("\n✅ Fixes applied:")
            for fix in self.fixes_applied:
                print(f"   - {fix}")
                
        if not self.issues or self.fixes_applied:
            print("\n🎉 Integration should now work! Try running:")
            print("   cd /workspace/aura-microservices")
            print("   ./start_services.sh")
            print("   python test_integration.py")
        
        return len(self.issues) == 0


def main():
    fixer = IntegrationFixer()
    success = fixer.run_diagnostics_and_fix()
    
    if not success:
        print("\n💡 If issues persist, try:")
        print("   1. Install Docker if not available")
        print("   2. Create Python virtual environment")
        print("   3. Check firewall/network settings")
        print("   4. Review service logs for errors")


if __name__ == "__main__":
    main()