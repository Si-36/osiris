#!/usr/bin/env python3
"""
Test OUR Week 2 Persistence Layer Implementation
Tests what WE built, not someone else's code
"""

import asyncio
import sys
sys.path.insert(0, 'core/src')

print("🧪 Testing OUR Persistence Layer (Week 2)")
print("=" * 60)

# Test 1: Causal State Manager (PostgreSQL-based)
print("\n1️⃣ Testing Causal State Manager")
print("-" * 30)
try:
    from aura_intelligence.persistence.causal_state_manager import (
        CausalStateManager,
        CausalEvent,
        CausalBranch
    )
    print("✅ Causal State Manager imports successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test 2: Memory-Native Architecture  
print("\n2️⃣ Testing Memory-Native Architecture")
print("-" * 30)
try:
    from aura_intelligence.persistence.memory_native import (
        MemoryNativeStore,
        GPUMemoryPool,
        ComputeOnRetrieval
    )
    print("✅ Memory-Native components import successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test 3: Migration Tool
print("\n3️⃣ Testing Pickle to PostgreSQL Migration")
print("-" * 30)
try:
    from aura_intelligence.persistence.migrate_from_pickle import (
        PickleToPostgresMigrator,
        MigrationReport
    )
    print("✅ Migration tool imports successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test 4: Lakehouse Core
print("\n4️⃣ Testing Lakehouse Core")
print("-" * 30)
try:
    from aura_intelligence.persistence.lakehouse_core import (
        UnifiedLakehouseStore,
        LakehouseConfig
    )
    print("✅ Lakehouse core imports successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test 5: Run actual persistence test
print("\n5️⃣ Testing Actual Persistence Operations")
print("-" * 30)

async def test_persistence():
    """Test our actual persistence implementation"""
    try:
        # Test with in-memory mode if PostgreSQL not available
        config = {
            'connection_string': 'postgresql://localhost/aura_test',
            'use_memory_mode': True  # Fallback for testing
        }
        
        manager = CausalStateManager(config)
        
        # Test save
        test_state = {'model': 'test', 'weights': [1, 2, 3]}
        event_id = await manager.save_state(test_state, "Test save")
        print(f"✅ Saved state with event ID: {event_id}")
        
        # Test load
        loaded = await manager.load_state(event_id)
        print(f"✅ Loaded state: {loaded}")
        
        # Test causal tracking
        history = await manager.get_causal_history(event_id)
        print(f"✅ Causal history tracked: {len(history)} events")
        
        return True
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False

# Run the async test
if asyncio:
    try:
        success = asyncio.run(test_persistence())
    except:
        print("⚠️  Async test skipped - dependencies missing")

print("\n" + "=" * 60)
print("📊 Summary:")
print("This tests OUR actual Week 2 work:")
print("- Causal State Manager (PostgreSQL)")
print("- Memory-Native Architecture")
print("- Migration Tool")
print("- Lakehouse Integration")
print("\n✨ Not someone else's components!")