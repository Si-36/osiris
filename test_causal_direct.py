#!/usr/bin/env python3
"""
Direct test of causal persistence
"""
import asyncio
import sys
import os

# Set up path
sys.path.insert(0, os.path.abspath('./core/src'))

async def main():
    print("Testing causal persistence directly...")
    
    # Import just the causal manager
    from aura_intelligence.persistence.causal_state_manager import (
        CausalPersistenceManager,
        StateType,
        CausalContext
    )
    
    # Create manager
    manager = CausalPersistenceManager(
        postgres_url="postgresql://localhost/aura",
        legacy_path="./test_pickle"
    )
    
    await manager.initialize()
    
    print(f"Manager initialized. Legacy mode: {manager.legacy_mode}")
    
    # Test save
    test_data = {"test": "data", "value": 42}
    context = CausalContext(
        causes=["test_execution"],
        effects=["test_complete"],
        confidence=0.9
    )
    
    state_id = await manager.save_state(
        StateType.COMPONENT_STATE,
        "test_component",
        test_data,
        causal_context=context
    )
    
    print(f"Saved state with ID: {state_id}")
    
    # Test load
    loaded = await manager.load_state(
        StateType.COMPONENT_STATE,
        "test_component"
    )
    
    print(f"Loaded state: {loaded}")
    
    # Test compute on retrieval
    def enhance(data):
        data["enhanced"] = True
        return data
    
    enhanced = await manager.load_state(
        StateType.COMPONENT_STATE,
        "test_component",
        compute_on_retrieval=enhance
    )
    
    print(f"Enhanced state: {enhanced}")
    
    await manager.close()
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(main())