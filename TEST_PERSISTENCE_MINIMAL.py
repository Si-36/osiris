#!/usr/bin/env python3
"""Minimal test for persistence system - bypass import errors"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))

async def test_minimal():
    print("🚀 Testing AURA Persistence System (Minimal)...")
    print("=" * 60)
    
    try:
        # Test 1: Direct import of causal_state_manager
        print("\n1️⃣ Testing direct causal persistence import...")
        from aura_intelligence.persistence.causal_state_manager import (
            CausalPersistenceManager,
            StateType,
            CausalContext
        )
        print("✅ Causal persistence imports successful!")
        
        # Test 2: Create manager instance
        print("\n2️⃣ Creating persistence manager...")
        manager = CausalPersistenceManager()
        await manager.initialize()
        print("✅ Manager initialized!")
        
        # Test 3: Basic save operation
        print("\n3️⃣ Testing basic save...")
        state_id = await manager.save_state(
            state_type=StateType.AGENT_MEMORY,
            component_id="test_agent",
            state_data={"test": "data", "version": 1},
            metadata={"test_run": True}
        )
        print(f"✅ State saved with ID: {state_id}")
        
        # Test 4: Load operation
        print("\n4️⃣ Testing load...")
        loaded_state = await manager.load_state(
            state_type=StateType.AGENT_MEMORY,
            component_id="test_agent"
        )
        print(f"✅ State loaded: {loaded_state['data']}")
        
        # Test 5: Causal tracking
        print("\n5️⃣ Testing causal tracking...")
        causal_context = CausalContext(
            causes=["init_state"],
            effects=["decision_made"],
            confidence=0.95
        )
        causal_id = await manager.save_state(
            state_type=StateType.AGENT_MEMORY,
            component_id="test_agent_causal",
            state_data={"decision": "explore", "reason": "high_confidence"},
            causal_context=causal_context
        )
        print(f"✅ Causal state saved: {causal_id}")
        
        # Test 6: Memory-native operations
        print("\n6️⃣ Testing memory-native architecture...")
        from aura_intelligence.persistence.memory_native import MemoryNativeAI
        memory_ai = MemoryNativeAI()
        thought = {"query": "test", "context": "minimal"}
        result = await memory_ai.think_with_memory(thought)
        print(f"✅ Memory-native processing: {result}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal())
    sys.exit(0 if success else 1)