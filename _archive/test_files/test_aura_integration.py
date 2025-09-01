#!/usr/bin/env python3
"""
Test AURA Integration - Verify all our components work together
"""

import asyncio
import sys
sys.path.append('core/src')

print("🧪 Testing AURA Integration...")

# Test 1: Can we import everything?
try:
    from aura_intelligence import (
        AURA,
        AURAModelRouter,
        AURAMemorySystem,
        AgentTopologyAnalyzer,
        UnifiedOrchestrationEngine,
        SwarmCoordinator,
        AURAMainSystem,
        UnifiedEventMesh,
        UnifiedCommunication,
        AURAAgent,
        __version__
    )
    print(f"✅ All imports successful! AURA v{__version__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Can we create instances?
print("\n🔧 Creating component instances...")

try:
    # Neural
    router = AURAModelRouter()
    print("✅ Neural Router created")
    
    # Memory
    memory = AURAMemorySystem()
    print("✅ Memory System created")
    
    # TDA
    tda = AgentTopologyAnalyzer()
    print("✅ TDA Analyzer created")
    
    # Orchestration
    orchestration = UnifiedOrchestrationEngine()
    print("✅ Orchestration Engine created")
    
    # Swarm
    swarm = SwarmCoordinator(
        algorithm="pso",
        n_particles=10
    )
    print("✅ Swarm Coordinator created")
    
except Exception as e:
    print(f"❌ Component creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Can we create main system?
print("\n🏗️ Creating main AURA system...")

try:
    aura = AURA()
    print(f"✅ AURA system created: {aura}")
except Exception as e:
    print(f"❌ AURA creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Quick functionality test
async def test_basic_functionality():
    print("\n⚡ Testing basic functionality...")
    
    try:
        # Test memory store/retrieve
        memory = AURAMemorySystem()
        await memory.initialize()
        
        # Store something
        entry_id = await memory.store({
            "content": "Test memory",
            "metadata": {"test": True}
        })
        print(f"✅ Memory stored: {entry_id}")
        
        # Test neural routing
        router = AURAModelRouter()
        result = await router.route({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "fast"
        })
        print("✅ Neural routing works")
        
        # Test TDA analysis
        tda = AgentTopologyAnalyzer()
        features = tda.analyze_workflow([
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"}
        ])
        print("✅ TDA analysis works")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()

# Run tests
if __name__ == "__main__":
    asyncio.run(test_basic_functionality())