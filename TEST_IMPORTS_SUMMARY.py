#!/usr/bin/env python3
"""
Summary of AURA Import Testing Progress
======================================

This script provides a quick overview of what's working and what needs fixing.
"""

import sys
import os
import importlib
import traceback

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core/src'))

# Component categories to test
COMPONENT_CATEGORIES = {
    "Memory": [
        ("aura_intelligence.memory", ["HybridMemoryManager", "HierarchicalMemorySystem"]),
        ("aura_intelligence.memory.shape_memory_v2_prod", ["ShapeMemoryV2"]),
        ("aura_intelligence.memory.shape_memory_v2_gpu_wrapper", ["ShapeAwareMemoryV2GPUWrapper"]),
    ],
    "Persistence": [
        ("aura_intelligence.persistence.causal_state_manager", ["CausalPersistenceManager", "CausalContext"]),
        ("aura_intelligence.persistence.state_manager", ["StatePersistenceManager"]),
    ],
    "Neural": [
        ("aura_intelligence.neural.lnn_core", ["LiquidNeuralNetwork"]),
        ("aura_intelligence.neural.moe_switch", ["SwitchTransformerMoE"]),
        ("aura_intelligence.neural.mamba_v2", ["MambaV2Model"]),
    ],
    "TDA": [
        ("aura_intelligence.tda.real_tda", ["RealTDA"]),
        ("aura_intelligence.tda.agent_topology", ["AgentTopologyAnalyzer"]),
        ("aura_intelligence.tda.algorithms", ["compute_persistence_diagram"]),
    ],
    "Consensus": [
        ("aura_intelligence.consensus.simple", ["SimpleConsensus"]),
        ("aura_intelligence.consensus.raft", ["RaftConsensus"]),
        ("aura_intelligence.consensus.byzantine", ["ByzantineConsensus"]),
    ],
    "Events": [
        ("aura_intelligence.events.event_producer", ["EventProducer"]),
        ("aura_intelligence.events.event_processor", ["EventProcessor"]),
    ],
    "Agents": [
        ("aura_intelligence.agents.base", ["AURAAgent"]),
        ("aura_intelligence.agents.simple_agent", ["SimpleAgent"]),
        ("aura_intelligence.agents.test_agents", ["create_code_agent"]),
    ],
    "Resilience": [
        ("aura_intelligence.resilience.circuit_breaker", ["CircuitBreaker"]),
        ("aura_intelligence.resilience.bulkhead", ["Bulkhead"]),
        ("aura_intelligence.resilience.retry", ["RetryPolicy"]),
    ],
    "GPU Adapters": [
        ("aura_intelligence.adapters.memory_adapter_gpu", ["GPUMemoryAdapter"]),
        ("aura_intelligence.adapters.tda_adapter_gpu", ["TDAGPUAdapter"]),
        ("aura_intelligence.adapters.orchestration_adapter_gpu", ["GPUOrchestrationAdapter"]),
    ],
    "Orchestration": [
        ("aura_intelligence.orchestration.unified_orchestration_engine", ["UnifiedOrchestrationEngine"]),
        ("aura_intelligence.orchestration.workflows.nodes.supervisor", ["UnifiedAuraSupervisor"]),
    ],
}

def test_module_import(module_path, components=None):
    """Test if a module can be imported and optionally check for specific components"""
    try:
        module = importlib.import_module(module_path)
        if components:
            missing = []
            for comp in components:
                if not hasattr(module, comp):
                    missing.append(comp)
            if missing:
                return False, f"Missing components: {', '.join(missing)}"
        return True, "Success"
    except ImportError as e:
        # Extract the actual missing module
        error_msg = str(e)
        if "No module named" in error_msg:
            missing_module = error_msg.split("'")[1]
            return False, f"Missing dependency: {missing_module}"
        return False, f"Import error: {error_msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("🔍 AURA COMPONENT IMPORT STATUS")
    print("=" * 60)
    
    total_categories = len(COMPONENT_CATEGORIES)
    working_categories = 0
    all_missing_deps = set()
    
    for category, modules in COMPONENT_CATEGORIES.items():
        print(f"\n📦 {category}")
        print("-" * 30)
        
        category_working = True
        for module_path, components in modules:
            success, message = test_module_import(module_path, components)
            
            if success:
                print(f"  ✅ {module_path}")
            else:
                print(f"  ❌ {module_path}: {message}")
                category_working = False
                
                # Extract missing dependency
                if "Missing dependency:" in message:
                    dep = message.split(": ")[1]
                    all_missing_deps.add(dep)
        
        if category_working:
            working_categories += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print(f"  Categories tested: {total_categories}")
    print(f"  Categories working: {working_categories}")
    print(f"  Success rate: {working_categories/total_categories*100:.1f}%")
    
    if all_missing_deps:
        print(f"\n📦 Missing Dependencies:")
        for dep in sorted(all_missing_deps):
            print(f"  - {dep}")
    
    print("\n💡 Next Steps:")
    print("  1. Install missing dependencies")
    print("  2. Fix import errors in failing modules")
    print("  3. Continue with integration testing")

if __name__ == "__main__":
    main()