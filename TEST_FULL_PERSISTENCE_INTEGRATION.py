#!/usr/bin/env python3
"""
Full Persistence Integration Test
================================
Tests the complete persistence system with all AURA components
Run this in your local environment where all dependencies are installed
"""

import asyncio
import sys
import os
from datetime import datetime
import traceback

# Add to Python path
sys.path.insert(0, os.path.abspath('./core/src'))

async def test_complete_integration():
    """Test the full persistence integration"""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║     AURA PERSISTENCE - COMPLETE INTEGRATION TEST              ║
║                   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Test 1: Basic Persistence System
    print("\n" + "="*60)
    print("TEST 1: Basic Persistence System")
    print("="*60)
    
    try:
        from aura_intelligence.persistence.causal_state_manager import (
            CausalPersistenceManager,
            StateType,
            CausalContext,
            get_causal_manager
        )
        
        manager = await get_causal_manager()
        print("✅ Persistence manager initialized")
        
        # Test save/load
        test_state = {
            "test": "integration",
            "timestamp": datetime.now().isoformat()
        }
        
        state_id = await manager.save_state(
            StateType.COMPONENT_STATE,
            "test_component",
            test_state,
            causal_context=CausalContext(
                causes=["test_execution"],
                effects=["validation"],
                confidence=0.95
            )
        )
        print(f"✅ Saved state: {state_id}")
        
        loaded = await manager.load_state(
            StateType.COMPONENT_STATE,
            "test_component"
        )
        print(f"✅ Loaded state successfully")
        
        results["Basic Persistence"] = True
        
    except Exception as e:
        print(f"❌ Basic persistence failed: {e}")
        traceback.print_exc()
        results["Basic Persistence"] = False
    
    # Test 2: Enhanced Agent
    print("\n" + "="*60)
    print("TEST 2: Enhanced Agent with Persistence")
    print("="*60)
    
    try:
        from aura_intelligence.agents.enhanced_code_agent import create_enhanced_code_agent
        
        agent = create_enhanced_code_agent("test_agent_001")
        print("✅ Created enhanced code agent")
        
        # Test decision saving
        decision_id = await agent.save_decision(
            decision="analyze_code",
            context={
                "file": "test.py",
                "reason": "integration_test",
                "confidence": 0.9
            },
            confidence=0.9
        )
        print(f"✅ Saved agent decision: {decision_id}")
        
        # Test memory loading
        memory = await agent.load_memory()
        print(f"✅ Loaded agent memory")
        
        results["Enhanced Agent"] = True
        
    except Exception as e:
        print(f"❌ Enhanced agent failed: {e}")
        traceback.print_exc()
        results["Enhanced Agent"] = False
    
    # Test 3: Neural Persistence
    print("\n" + "="*60)
    print("TEST 3: Neural Network Persistence")
    print("="*60)
    
    try:
        from aura_intelligence.neural.persistence_integration import (
            PersistentLNN,
            PersistentMoE,
            PersistentMamba
        )
        
        # Test LNN persistence
        lnn = PersistentLNN("test_lnn_001")
        
        checkpoint_id = await lnn.save_checkpoint(
            epoch=1,
            metrics={
                "loss": 0.5,
                "accuracy": 0.85,
                "validation_accuracy": 0.83
            }
        )
        print(f"✅ Saved LNN checkpoint: {checkpoint_id}")
        
        # Test experimental branch
        branch_id = await lnn.create_experiment_branch("high_lr_experiment")
        print(f"✅ Created experiment branch: {branch_id}")
        
        results["Neural Persistence"] = True
        
    except Exception as e:
        print(f"❌ Neural persistence failed: {e}")
        traceback.print_exc()
        results["Neural Persistence"] = False
    
    # Test 4: Memory System
    print("\n" + "="*60)
    print("TEST 4: Memory System Persistence")
    print("="*60)
    
    try:
        from aura_intelligence.adapters.memory_persistence_integration import (
            PersistentMemoryInterface
        )
        
        memory = PersistentMemoryInterface("test_memory")
        
        # Store memory
        memory_id = await memory.store_memory(
            key="test_fact",
            value="AURA uses causal persistence",
            memory_type="semantic",
            metadata={"importance": 0.9}
        )
        print(f"✅ Stored memory: {memory_id}")
        
        # Retrieve memory
        retrieved = await memory.retrieve_memory(key="test_fact")
        print(f"✅ Retrieved memory: {len(retrieved)} items")
        
        # Test memory evolution
        def enhance_memory(mem):
            if mem:
                mem["enhanced"] = True
                mem["enhancement_time"] = datetime.now().isoformat()
            return mem
        
        evolution_id = await memory.evolve_memory(
            "test_fact",
            enhance_memory
        )
        print(f"✅ Evolved memory: {evolution_id}")
        
        results["Memory System"] = True
        
    except Exception as e:
        print(f"❌ Memory system failed: {e}")
        traceback.print_exc()
        results["Memory System"] = False
    
    # Test 5: TDA Persistence
    print("\n" + "="*60)
    print("TEST 5: TDA Persistence")
    print("="*60)
    
    try:
        from aura_intelligence.tda.persistence_integration import PersistentTDA
        import numpy as np
        
        tda = PersistentTDA("test_tda")
        
        # Create sample persistence diagram
        diagram = np.array([
            [0.1, 0.5],
            [0.2, 0.8],
            [0.3, 0.9]
        ])
        
        diagram_id = await tda.save_persistence_diagram(
            diagram=diagram,
            data_source="test_data",
            computation_params={
                "max_dimension": 1,
                "max_edge_length": 1.0
            },
            metadata={"domain": "test"}
        )
        print(f"✅ Saved persistence diagram: {diagram_id}")
        
        # Load and verify
        loaded_diagram = await tda.load_persistence_diagram("test_data")
        print(f"✅ Loaded persistence diagram")
        
        results["TDA Persistence"] = True
        
    except Exception as e:
        print(f"❌ TDA persistence failed: {e}")
        traceback.print_exc()
        results["TDA Persistence"] = False
    
    # Test 6: Full System Integration
    print("\n" + "="*60)
    print("TEST 6: Full System Integration")
    print("="*60)
    
    try:
        # Test cross-component causality
        from aura_intelligence.persistence.causal_state_manager import get_causal_manager
        
        manager = await get_causal_manager()
        
        # Create a decision that affects multiple components
        causal_context = CausalContext(
            causes=["user_request", "high_complexity_detected"],
            effects=["agent_decision", "memory_update", "topology_computed"],
            confidence=0.85
        )
        
        # Save interconnected state
        integration_id = await manager.save_state(
            StateType.COMPONENT_STATE,
            "integration_test",
            {
                "agent": "code_analyzer",
                "memory": "semantic_updated",
                "tda": "topology_extracted",
                "neural": "model_adapted",
                "timestamp": datetime.now().isoformat()
            },
            causal_context=causal_context
        )
        
        print(f"✅ Saved integrated state: {integration_id}")
        
        # Test causal chain
        if not manager.legacy_mode:
            chain = await manager.get_causal_chain(integration_id)
            print(f"✅ Retrieved causal chain: {len(chain) if chain else 0} links")
        
        results["Full Integration"] = True
        
    except Exception as e:
        print(f"❌ Full integration failed: {e}")
        traceback.print_exc()
        results["Full Integration"] = False
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("""
🎉 FULL INTEGRATION SUCCESS! 🎉

The AURA persistence system is fully integrated with:
- ✅ Causal state management
- ✅ Enhanced agents with decision tracking
- ✅ Neural network checkpointing with branches
- ✅ Memory system with evolution
- ✅ TDA with topology persistence
- ✅ Cross-component causality

Next steps:
1. Run performance benchmarks
2. Test with real workloads
3. Deploy to production
        """)
    else:
        print("""
⚠️  Some integration tests failed.

Check the errors above and ensure:
1. All dependencies are installed
2. PostgreSQL is running (or using legacy mode)
3. No syntax errors remain
        """)
    
    return passed == total

async def main():
    """Main entry point"""
    success = await test_complete_integration()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)