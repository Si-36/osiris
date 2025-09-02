#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE AURA SYSTEM TEST SUITE
========================================

This test suite validates all restored and fixed components.
Run this AFTER installing all dependencies.

Required packages:
pip install langgraph temporalio nats-py neo4j prometheus_client scikit-learn \
            aiokafka confluent-kafka cachetools authlib fastavro semver
"""

import sys
import os
import traceback
from typing import Dict, Any, List
from datetime import datetime

# Add AURA to path
sys.path.insert(0, 'core/src')

# Test results
results = {
    "timestamp": datetime.now().isoformat(),
    "tests": {},
    "summary": {"passed": 0, "failed": 0, "total": 0}
}

def test_section(name: str):
    """Decorator for test sections"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"🧪 Testing: {name}")
            print(f"{'='*60}")
            try:
                result = func()
                results["tests"][name] = {"status": "✅ PASSED", "details": result}
                results["summary"]["passed"] += 1
                print(f"✅ {name}: PASSED")
                return result
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                results["tests"][name] = {"status": "❌ FAILED", "error": error_msg}
                results["summary"]["failed"] += 1
                print(f"❌ {name}: FAILED")
                print(f"   Error: {error_msg}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
                return None
            finally:
                results["summary"]["total"] += 1
        return wrapper
    return decorator

# ==================== IMPORT TESTS ====================

@test_section("Core Imports")
def test_core_imports():
    """Test all core module imports"""
    from aura_intelligence import (
        AURAMemorySystem,
        UnifiedOrchestrationEngine,
        ExecutiveController,
        ConsciousnessState,
    )
    return "All core imports successful"

@test_section("Memory Components")
def test_memory():
    """Test memory system components"""
    from aura_intelligence.memory import (
        HybridMemoryManager,
        ShapeAwareMemory,
        CollectiveMemory,
        FastRPEmbeddings,
    )
    from aura_intelligence.memory.shape_memory_v2 import ShapeAwareMemoryV2
    
    # Test instantiation
    memory = ShapeAwareMemoryV2()
    return f"Memory components loaded: {type(memory).__name__}"

@test_section("DPO System (Advanced)")
def test_dpo():
    """Test restored DPO with advanced features"""
    from aura_intelligence.dpo.dpo_2025_advanced import (
        AURAAdvancedDPO,
        PreferenceType,
        DPOConfig,
    )
    
    # Check for advanced features
    config = DPOConfig()
    dpo = AURAAdvancedDPO(config)
    
    # Verify advanced methods exist
    assert hasattr(dpo, 'gpo_loss'), "GPO loss not found"
    assert hasattr(dpo, 'dmpo_loss'), "DMPO loss not found"
    assert hasattr(dpo, 'icai_loss'), "ICAI loss not found"
    assert hasattr(dpo, 'compute_saom'), "SAOM not found"
    
    return "DPO with GPO/DMPO/ICAI/SAOM verified"

@test_section("Orchestration System")
def test_orchestration():
    """Test unified orchestration with hierarchical layers"""
    from aura_intelligence.orchestration import UnifiedOrchestrationEngine
    from aura_intelligence.orchestration.hierarchical_orchestrator import (
        HierarchicalOrchestrator,
        OrchestrationLayer,
    )
    
    # Test hierarchical layers
    assert OrchestrationLayer.STRATEGIC
    assert OrchestrationLayer.TACTICAL
    assert OrchestrationLayer.OPERATIONAL
    
    return "Orchestration with 3-layer hierarchy verified"

@test_section("Agent System")
def test_agents():
    """Test agent components"""
    from aura_intelligence.agents.agent_core import AURAAgentCore
    from aura_intelligence.agents import SimpleAgent
    
    # Check LangGraph integration
    try:
        from aura_intelligence.agents.production_langgraph_agent import (
            ProductionLangGraphAgent
        )
        langgraph_status = "LangGraph agent available"
    except:
        langgraph_status = "LangGraph optional (not installed)"
    
    return f"Agent system loaded. {langgraph_status}"

@test_section("TDA Algorithms")
def test_tda():
    """Test Topological Data Analysis components"""
    from aura_intelligence.tda import (
        TDAProcessor,
        PersistenceDiagram,
        TopologicalSignature,
    )
    from aura_intelligence.tda.legacy.persistence_simple import TDAProcessor as LegacyTDA
    
    # Count available algorithms
    processor = LegacyTDA()
    methods = [m for m in dir(processor) if not m.startswith('_')]
    
    return f"TDA with {len(methods)} methods available"

@test_section("Collective Intelligence")
def test_collective():
    """Test restored collective intelligence"""
    from aura_intelligence.collective.collective_memory_restored import (
        CollectiveMemoryManager
    )
    
    # Check for advanced features
    manager = CollectiveMemoryManager()
    
    # Verify key methods
    assert hasattr(manager, 'reach_consensus'), "Consensus not found"
    assert hasattr(manager, 'semantic_clustering'), "Clustering not found"
    assert hasattr(manager, 'build_causal_chains'), "Causal chains not found"
    
    methods = [m for m in dir(manager) if not m.startswith('_')]
    return f"Collective with {len(methods)} methods restored"

@test_section("MoE Routing")
def test_moe():
    """Test advanced MoE routing strategies"""
    from aura_intelligence.moe.advanced_moe_restored import (
        AdvancedMoESystem,
        TokenChoiceRouter,
        ExpertChoiceRouter,
        SoftMoERouter,
    )
    
    # Test routing strategies
    system = AdvancedMoESystem(num_experts=8)
    
    assert hasattr(system, 'token_choice_routing'), "TokenChoice not found"
    assert hasattr(system, 'expert_choice_routing'), "ExpertChoice not found"
    assert hasattr(system, 'soft_routing'), "SoftMoE not found"
    
    return "MoE with TokenChoice/ExpertChoice/SoftMoE verified"

@test_section("CoRaL System")
def test_coral():
    """Test best CoRaL implementation"""
    from aura_intelligence.coral.best_coral import (
        BestCoRaLSystem,
        Mamba2Block,
        MinimalTransformer,
        GraphAttention,
    )
    
    # Verify components
    coral = BestCoRaLSystem(
        input_dim=768,
        hidden_dim=1024,
        num_heads=8
    )
    
    assert hasattr(coral, 'mamba_blocks'), "Mamba-2 not found"
    assert hasattr(coral, 'transformer'), "Transformer not found"
    assert hasattr(coral, 'graph_attention'), "GraphAttention not found"
    
    return "CoRaL with Mamba-2/Transformer/GraphAttention verified"

@test_section("Infrastructure")
def test_infrastructure():
    """Test infrastructure components"""
    from aura_intelligence.infrastructure import (
        UnifiedEventMesh,
        EnhancedEnterpriseGuardrails,
        MultiProviderClient,
    )
    
    # Test guardrails config
    from aura_intelligence.infrastructure.enhanced_guardrails import GuardrailsConfig
    config = GuardrailsConfig()
    
    return "Infrastructure with EventMesh/Guardrails/MultiProvider ready"

@test_section("Neural Components")
def test_neural():
    """Test neural network components"""
    from aura_intelligence.neural import (
        LiquidNeuralNetwork,
        MambaModel,
        AURAModelRouter,
    )
    
    # Check for LNN features
    from aura_intelligence.neural.liquid_v3 import LiquidTimeConstantCell
    
    return "Neural with LNN/Mamba/Router verified"

# ==================== INTEGRATION TESTS ====================

@test_section("Full System Integration")
def test_full_integration():
    """Test complete system integration"""
    import aura_intelligence
    
    # Check version and components
    components = dir(aura_intelligence)
    core_components = [c for c in components if not c.startswith('_')]
    
    return f"Full system with {len(core_components)} components integrated"

# ==================== MAIN TEST RUNNER ====================

def main():
    """Run all tests and generate report"""
    print("\n" + "="*60)
    print("🚀 AURA COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Timestamp: {results['timestamp']}")
    
    # Run all tests
    test_core_imports()
    test_memory()
    test_dpo()
    test_orchestration()
    test_agents()
    test_tda()
    test_collective()
    test_moe()
    test_coral()
    test_infrastructure()
    test_neural()
    test_full_integration()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {results['summary']['passed']}/{results['summary']['total']}")
    print(f"❌ Failed: {results['summary']['failed']}/{results['summary']['total']}")
    
    # Save results
    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Results saved to test_results.json")
    
    # Return exit code
    return 0 if results['summary']['failed'] == 0 else 1

if __name__ == "__main__":
    exit(main())