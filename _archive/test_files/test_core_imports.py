#!/usr/bin/env python3
"""
Test what AURA components actually import successfully
"""
import sys
sys.path.insert(0, '/workspace/core/src')

def test_imports():
    results = {}
    
    # Test basic imports
    imports_to_test = [
        ("supervisor", "from aura_intelligence.orchestration.workflows.nodes.supervisor import SupervisorNode, UnifiedAuraSupervisor"),
        ("tda_integration", "from aura_intelligence.orchestration.workflows.nodes.tda_supervisor_integration import ProductionTopologicalAnalyzer"),
        ("lnn_integration", "from aura_intelligence.orchestration.workflows.nodes.lnn_supervisor_integration import ProductionLiquidNeuralDecisionEngine"),
        ("state", "from aura_intelligence.orchestration.workflows.state import CollectiveState, NodeResult"),
    ]
    
    for name, import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            results[name] = "SUCCESS"
            print(f"✅ {name}: SUCCESS")
        except Exception as e:
            results[name] = f"FAILED: {str(e)[:100]}"
            print(f"❌ {name}: {str(e)[:100]}")
    
    return results

def test_unified_supervisor():
    """Test creating UnifiedAuraSupervisor"""
    try:
        from aura_intelligence.orchestration.workflows.nodes.supervisor import UnifiedAuraSupervisor
        
        # Try to create supervisor
        supervisor = UnifiedAuraSupervisor()
        print("\n✅ UnifiedAuraSupervisor created successfully!")
        print(f"   - Name: {supervisor.name}")
        print(f"   - TDA Available: {supervisor.tda_available}")
        print(f"   - LNN Available: {supervisor.lnn_available}")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed to create UnifiedAuraSupervisor: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing AURA Core Imports...")
    print("=" * 50)
    
    results = test_imports()
    
    print("\n🧪 Testing UnifiedAuraSupervisor...")
    print("=" * 50)
    
    success = test_unified_supervisor()
    
    print("\n📊 Summary:")
    print("=" * 50)
    for name, status in results.items():
        print(f"{name}: {status}")
    print(f"\nUnifiedAuraSupervisor: {'SUCCESS' if success else 'FAILED'}")