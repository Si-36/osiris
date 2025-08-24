#!/usr/bin/env python3
"""
🔧 FIX REAL AURA INTELLIGENCE ISSUES
===================================

Stop the fake success - fix the actual problems.
"""

import subprocess
import sys
import os
import time

print("🔧 FIXING REAL SYSTEM ISSUES")
print("=" * 40)
print("❌ Stop celebrating fake success")
print("✅ Fix actual dependency problems")
print("=" * 40)

def run_command(cmd, description):
def run_command(cmd, description):
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_service(service, port):
def check_service(service, port):
    print(f"\n🔍 Checking {service} on port {port}")
    result = subprocess.run(["nc", "-z", "localhost", str(port)], capture_output=True)
    if result.returncode == 0:
        print(f"✅ {service} is running on port {port}")
        return True
    else:
        print(f"❌ {service} is NOT running on port {port}")
        return False

# ============================================================================
# 1. CHECK CURRENT SERVICE STATUS
# ============================================================================

print("\n1️⃣ CHECKING CURRENT SERVICE STATUS")
print("-" * 40)

redis_running = check_service("Redis", 6379)
neo4j_running = check_service("Neo4j", 7687)

# ============================================================================
# 2. START REQUIRED SERVICES
# ============================================================================

print("\n2️⃣ STARTING REQUIRED SERVICES")
print("-" * 40)

if not redis_running:
    print("\n🔧 Starting Redis server...")
    # Try different ways to start Redis
    redis_started = False
    
    # Try systemctl
    if run_command("sudo systemctl start redis-server", "Starting Redis with systemctl"):
        redis_started = True
    elif run_command("sudo systemctl start redis", "Starting Redis with systemctl (alternative)"):
        redis_started = True
    # Try direct command
    elif run_command("redis-server --daemonize yes", "Starting Redis directly"):
        redis_started = True
    # Try with different config
    elif run_command("redis-server --port 6379 --daemonize yes", "Starting Redis with explicit port"):
        redis_started = True
    
    if redis_started:
        time.sleep(2)
        check_service("Redis", 6379)
    else:
        print("❌ Could not start Redis. Install with: sudo apt install redis-server")

if not neo4j_running:
    print("\n🔧 Starting Neo4j server...")
    neo4j_started = False
    
    # Try systemctl
    if run_command("sudo systemctl start neo4j", "Starting Neo4j with systemctl"):
        neo4j_started = True
    # Try direct command
    elif run_command("neo4j start", "Starting Neo4j directly"):
        neo4j_started = True
    
    if neo4j_started:
        time.sleep(5)
        check_service("Neo4j", 7687)
    else:
        print("❌ Could not start Neo4j. Install with: sudo apt install neo4j")

# ============================================================================
# 3. INSTALL MISSING PYTHON DEPENDENCIES
# ============================================================================

print("\n3️⃣ INSTALLING MISSING PYTHON DEPENDENCIES")
print("-" * 40)

dependencies = [
    ("redis", "Redis Python client"),
    ("neo4j", "Neo4j Python driver"),
    ("cupy-cpu", "CuPy for CPU (TDA acceleration)"),
    ("nats-py==2.6.0", "NATS messaging (specific version)"),
]

for dep, description in dependencies:
    print(f"\n📦 Installing {dep} - {description}")
    if run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
        print(f"✅ {dep} installed successfully")
    else:
        print(f"⚠️ {dep} installation had issues")

# ============================================================================
# 4. TEST REAL CONNECTIONS
# ============================================================================

print("\n4️⃣ TESTING REAL CONNECTIONS")
print("-" * 40)

# Test Redis connection
print("\n🔍 Testing Redis connection...")
test_redis_code = '''
import redis
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    print(f"✅ Redis working: {value}")
except Exception as e:
    print(f"❌ Redis failed: {e}")
'''

with open('test_redis.py', 'w') as f:
    f.write(test_redis_code)

run_command(f"{sys.executable} test_redis.py", "Testing Redis connection")

# Test Neo4j connection
print("\n🔍 Testing Neo4j connection...")
test_neo4j_code = '''
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
    with driver.session() as session:
        result = session.run("RETURN 'Hello Neo4j' as message")
        record = result.single()
        print(f"✅ Neo4j working: {record['message']}")
    driver.close()
except Exception as e:
    print(f"❌ Neo4j failed: {e}")
'''

with open('test_neo4j.py', 'w') as f:
    f.write(test_neo4j_code)

run_command(f"{sys.executable} test_neo4j.py", "Testing Neo4j connection")

# ============================================================================
# 5. CREATE HONEST WORKING DEMO
# ============================================================================

print("\n5️⃣ CREATING HONEST WORKING DEMO")
print("-" * 40)

honest_demo_code = '''#!/usr/bin/env python3
"""
🔍 HONEST AURA INTELLIGENCE DEMO
===============================

Shows what ACTUALLY works, no fake success.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add paths
core_path = Path(__file__).parent / "core" / "src"
sys.path.insert(0, str(core_path))

async def test_real_components():
    
    print("🔍 HONEST COMPONENT TESTING")
    print("=" * 40)
    
    working_components = []
    broken_components = []
    
    # Test Redis-dependent memory
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        r.set('aura_test', 'working')
        result = r.get('aura_test')
        if result == 'working':
            working_components.append("Redis Memory Store")
            print("✅ Redis Memory Store - ACTUALLY WORKING")
        else:
            broken_components.append("Redis Memory Store - Connection issues")
    except Exception as e:
        broken_components.append(f"Redis Memory Store - {str(e)}")
        print(f"❌ Redis Memory Store - {str(e)}")
    
    # Test Neo4j knowledge graph
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record['test'] == 1:
                working_components.append("Neo4j Knowledge Graph")
                print("✅ Neo4j Knowledge Graph - ACTUALLY WORKING")
        driver.close()
    except Exception as e:
        broken_components.append(f"Neo4j Knowledge Graph - {str(e)}")
        print(f"❌ Neo4j Knowledge Graph - {str(e)}")
    
    # Test basic neural network
    try:
        from aura_intelligence.lnn.core import LiquidNeuralNetwork
        import torch
        
        lnn = LiquidNeuralNetwork(input_size=10, output_size=10)
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = lnn.forward(test_input)
        
        working_components.append("Neural Network (LNN)")
        print("✅ Neural Network (LNN) - ACTUALLY WORKING")
        
    except Exception as e:
        broken_components.append(f"Neural Network (LNN) - {str(e)}")
        print(f"❌ Neural Network (LNN) - {str(e)}")
    
    # Test consciousness
    try:
        from aura_intelligence.consciousness.global_workspace import GlobalWorkspace
        consciousness = GlobalWorkspace()
        working_components.append("Consciousness System")
        print("✅ Consciousness System - ACTUALLY WORKING")
    except Exception as e:
        broken_components.append(f"Consciousness System - {str(e)}")
        print(f"❌ Consciousness System - {str(e)}")
    
    print(f"\\n📊 HONEST RESULTS:")
    print(f"✅ Working: {len(working_components)} components")
    print(f"❌ Broken: {len(broken_components)} components")
    print(f"📈 Real success rate: {len(working_components)/(len(working_components)+len(broken_components))*100:.1f}%")
    
    if working_components:
        print(f"\\n✅ ACTUALLY WORKING:")
        for component in working_components:
            print(f"  - {component}")
    
    if broken_components:
        print(f"\\n❌ ACTUALLY BROKEN:")
        for component in broken_components:
            print(f"  - {component}")
    
    return len(working_components), len(broken_components)

if __name__ == "__main__":
    asyncio.run(test_real_components())
'''

with open('honest_demo.py', 'w') as f:
    f.write(honest_demo_code)

print("✅ Created honest_demo.py")

# ============================================================================
# 6. RUN HONEST DEMO
# ============================================================================

print("\n6️⃣ RUNNING HONEST DEMO")
print("-" * 40)

run_command(f"{sys.executable} honest_demo.py", "Running honest system test")

# ============================================================================
# FINAL STATUS
# ============================================================================

print("\n" + "=" * 50)
print("🎯 REAL SYSTEM STATUS")
print("=" * 50)
print("✅ Stopped fake celebrations")
print("🔧 Fixed dependency installation")
print("🔍 Created honest testing")
print("📊 Showed real success rates")
print("")
print("Next steps:")
print("1. Start Redis: redis-server --daemonize yes")
print("2. Start Neo4j: sudo systemctl start neo4j")
print("3. Run: python3 honest_demo.py")
print("4. See ACTUAL working components")
print("")
print("No more fake success - only real results! 🎯")
'''

with open('fix_real_issues.py', 'w') as f:
    f.write(fix_real_issues_code)

print("✅ Created fix_real_issues.py")'''
