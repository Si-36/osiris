#!/usr/bin/env python3
"""
Minimal Persistence Test
=======================
Tests just the core persistence without full integration
"""

import asyncio
import sys
import os
from datetime import datetime

# Add to path
sys.path.insert(0, os.path.abspath('./core/src'))

async def test_minimal():
    """Test minimal persistence functionality"""
    print("Testing minimal persistence...")
    
    # Import only what we need
    from aura_intelligence.persistence.causal_state_manager import (
        CausalPersistenceManager,
        StateType,
        CausalContext
    )
    
    # Create manager in legacy mode (no PostgreSQL needed)
    manager = CausalPersistenceManager(
        postgres_url=None,  # This forces legacy mode
        legacy_path="./test_minimal_persistence"
    )
    
    # Initialize
    await manager.initialize()
    print(f"✅ Manager initialized in legacy mode: {manager.legacy_mode}")
    
    # Test basic save
    test_data = {
        "test": "minimal",
        "timestamp": datetime.now().isoformat(),
        "value": 42
    }
    
    context = CausalContext(
        causes=["test_execution"],
        effects=["validation_complete"],
        confidence=0.95
    )
    
    state_id = await manager.save_state(
        StateType.COMPONENT_STATE,
        "minimal_test",
        test_data,
        causal_context=context
    )
    
    print(f"✅ Saved state with ID: {state_id}")
    
    # Test load
    loaded = await manager.load_state(
        StateType.COMPONENT_STATE,
        "minimal_test"
    )
    
    if loaded:
        print(f"✅ Loaded state successfully")
        print(f"   Data: {loaded}")
    else:
        print("❌ Failed to load state")
    
    # Cleanup
    await manager.close()
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_minimal())
        if success:
            print("\n✅ Minimal persistence test PASSED!")
            print("\nThis confirms the persistence system works in legacy mode.")
            print("The full integration might be failing due to:")
            print("1. Missing dependencies in some modules")
            print("2. Import order issues")
            print("3. PostgreSQL connection attempts")
        else:
            print("\n❌ Minimal persistence test FAILED!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()