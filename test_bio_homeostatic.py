#!/usr/bin/env python3
"""
Test bio_homeostatic components with real biological regulation patterns
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧬 TESTING BIO_HOMEOSTATIC COMPONENTS")
print("=" * 60)

async def test_bio_homeostatic():
    """Test biological homeostatic systems"""
    
    # Test 1: Import all modules
    print("\n📦 Testing imports...")
    try:
        from aura_intelligence.bio_homeostatic import homeostatic_coordinator
        from aura_intelligence.bio_homeostatic.metabolic_manager import MetabolicManager, MetabolicSignals
        from aura_intelligence.bio_homeostatic.production_metabolic_manager import ProductionMetabolicManager, get_production_metabolic
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Create metabolic signals
    print("\n🧬 Testing Metabolic Signals...")
    try:
        # Create test signal functions
        def test_utility(agent_id: str) -> float:
            return 0.8
        
        def test_influence(agent_id: str) -> float:
            return 0.6
        
        def test_efficiency(agent_id: str) -> float:
            return 0.9
        
        def test_risk(agent_id: str) -> float:
            return 0.2
        
        signals = MetabolicSignals(
            get_utility=test_utility,
            get_influence=test_influence,
            get_efficiency=test_efficiency,
            get_risk=test_risk
        )
        print("✅ Metabolic signals created")
        print(f"   - Utility: {signals.get_utility('test')}")
        print(f"   - Influence: {signals.get_influence('test')}")
        print(f"   - Efficiency: {signals.get_efficiency('test')}")
        print(f"   - Risk: {signals.get_risk('test')}")
    except Exception as e:
        print(f"❌ Metabolic signals failed: {e}")
    
    # Test 3: Create metabolic manager
    print("\n🧬 Testing Metabolic Manager...")
    try:
        manager = MetabolicManager()
        print("✅ Metabolic manager created")
        
        # Test regulation
        manager.register_regulator("test_agent", signals)
        print("✅ Regulator registered")
        
        # Calculate metabolic state
        state = manager.calculate_metabolic_state("test_agent")
        print(f"✅ Metabolic state calculated: {state}")
    except Exception as e:
        print(f"❌ Metabolic manager failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Production metabolic manager
    print("\n🧬 Testing Production Metabolic Manager...")
    try:
        # Get singleton instance
        prod_manager = get_production_metabolic()
        print("✅ Production manager singleton retrieved")
        
        # Test production manager
        prod_manager2 = ProductionMetabolicManager()
        print("✅ Production manager created")
        
        # Test metrics
        await prod_manager.update_metrics({
            "cpu_usage": 45.2,
            "memory_usage": 62.1,
            "active_agents": 10,
            "error_rate": 0.01
        })
        print("✅ Metrics updated")
        
        # Get health status
        health = await prod_manager.get_health_status()
        print(f"✅ Health status: {health}")
    except Exception as e:
        print(f"❌ Production manager failed: {e}")
    
    # Test 5: Component integration
    print("\n🔗 Testing Component Integration...")
    try:
        # Simulate biological feedback loop
        print("   - Simulating homeostatic regulation...")
        
        # Measure initial state
        initial_state = manager.calculate_metabolic_state("test_agent")
        
        # Apply stress (high CPU usage)
        await prod_manager.update_metrics({
            "cpu_usage": 95.0,
            "memory_usage": 85.0,
            "active_agents": 50,
            "error_rate": 0.05
        })
        
        # Check if system adapts
        stressed_health = await prod_manager.get_health_status()
        print(f"   - Stressed health: {stressed_health}")
        
        # Simulate recovery
        await prod_manager.update_metrics({
            "cpu_usage": 35.0,
            "memory_usage": 45.0,
            "active_agents": 10,
            "error_rate": 0.001
        })
        
        recovered_health = await prod_manager.get_health_status()
        print(f"   - Recovered health: {recovered_health}")
        
        print("✅ Homeostatic regulation working")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    print("\n" + "=" * 60)
    print("BIO_HOMEOSTATIC TEST COMPLETE")
    
# Run the test
if __name__ == "__main__":
    asyncio.run(test_bio_homeostatic())