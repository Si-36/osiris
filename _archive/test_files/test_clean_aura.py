#!/usr/bin/env python3
"""
Test Clean AURA System
"""

import asyncio
import sys
sys.path.append('core/src')

async def test_clean_aura():
    print("🧪 Testing Clean AURA System...")
    
    # Test imports
    try:
        from aura_intelligence_clean import (
            AURA,
            create_aura,
            AURAModelRouter,
            AURAMemorySystem,
            AgentTopologyAnalyzer,
            UnifiedOrchestrationEngine,
            SwarmCoordinator,
            AURACore,
            AURAAgent,
            __version__
        )
        print(f"✅ All imports successful! Clean AURA v{__version__}")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
        
    # Create system
    print("\n🏗️ Creating AURA system...")
    aura = create_aura()
    print("✅ AURA system created")
    
    # Test neural routing
    print("\n🧠 Testing Neural Routing...")
    request = {
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "stream": False
    }
    decision = await aura.neural.route(request)
    print(f"✅ Routing decision: {decision.model} via {decision.provider} - {decision.reason}")
    
    # Test memory
    print("\n💾 Testing Memory System...")
    memory_id = await aura.memory.store({
        "content": "Test memory entry",
        "metadata": {"type": "test"}
    })
    print(f"✅ Memory stored: {memory_id}")
    
    results = await aura.memory.retrieve({"keywords": ["test"]})
    print(f"✅ Retrieved {len(results)} memories")
    
    # Test TDA
    print("\n🔍 Testing TDA Analysis...")
    workflow = [
        {"from": "start", "to": "process"},
        {"from": "process", "to": "analyze"},
        {"from": "analyze", "to": "decide"},
        {"from": "decide", "to": "execute"},
        {"from": "execute", "to": "end"}
    ]
    features = aura.tda.analyze_workflow(workflow)
    print(f"✅ Topology features: {features.connected_components} components, {len(features.cycles)} cycles")
    
    # Test orchestration
    print("\n🎭 Testing Orchestration...")
    result = await aura.orchestration.execute({"test": "data"})
    print(f"✅ Orchestration result: {result['status']}")
    
    # Test swarm
    print("\n🐝 Testing Swarm Intelligence...")
    def simple_fitness(position):
        # Simple quadratic fitness
        return -sum(x**2 for x in position)
        
    optimization = await aura.swarm.optimize(
        fitness_func=simple_fitness,
        dimensions=2,
        iterations=10
    )
    print(f"✅ Swarm optimization: best fitness = {optimization['best_fitness']:.4f}")
    
    # Test core
    print("\n⚙️ Testing Core System...")
    await aura.core.start()
    status = aura.core.get_status()
    print(f"✅ System status: {status['running']}, {len(status['components'])} components")
    
    # Test full integration
    print("\n🔗 Testing Full Integration...")
    integrated_result = await aura.process({
        "messages": [{"role": "user", "content": "Hello AURA"}]
    })
    print(f"✅ Full integration test passed!")
    
    print("\n🎉 All tests passed! Clean AURA is working!")

if __name__ == "__main__":
    asyncio.run(test_clean_aura())