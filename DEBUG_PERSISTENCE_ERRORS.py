#!/usr/bin/env python3
"""
Debug Persistence Errors
=======================
Shows exactly what's failing in the integration
"""

import sys
import os
import traceback

# Add to path
sys.path.insert(0, os.path.abspath('./core/src'))

def test_imports():
    """Test each import individually"""
    print("=" * 60)
    print("TESTING IMPORTS ONE BY ONE")
    print("=" * 60)
    
    imports = [
        ("asyncio", "import asyncio"),
        ("asyncpg", "import asyncpg"),
        ("duckdb", "import duckdb"),
        ("zstandard", "import zstandard"),
        ("aiokafka", "import aiokafka"),
        ("persistence.causal_state_manager", "from aura_intelligence.persistence.causal_state_manager import CausalPersistenceManager"),
        ("agents.enhanced_code_agent", "from aura_intelligence.agents.enhanced_code_agent import create_enhanced_code_agent"),
        ("neural.persistence_integration", "from aura_intelligence.neural.persistence_integration import PersistentLNN"),
        ("adapters.memory_persistence", "from aura_intelligence.adapters.memory_persistence_integration import PersistentMemoryInterface"),
        ("tda.persistence_integration", "from aura_intelligence.tda.persistence_integration import PersistentTDA"),
    ]
    
    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            if "aura_intelligence" in name:
                # Show the full traceback for our modules
                traceback.print_exc()
                print()

def test_basic_persistence():
    """Test basic persistence functionality"""
    print("\n" + "=" * 60)
    print("TESTING BASIC PERSISTENCE")
    print("=" * 60)
    
    try:
        # Try direct import first
        import asyncio
        from aura_intelligence.persistence.causal_state_manager import (
            CausalPersistenceManager,
            StateType,
            CausalContext
        )
        
        async def test():
            # Test with legacy mode (no PostgreSQL needed)
            manager = CausalPersistenceManager(
                postgres_url=None,  # Force legacy mode
                legacy_path="./test_persistence_debug"
            )
            
            await manager.initialize()
            print(f"✅ Manager initialized (Legacy mode: {manager.legacy_mode})")
            
            # Simple save/load test
            test_data = {"test": "debug", "value": 42}
            
            state_id = await manager.save_state(
                StateType.COMPONENT_STATE,
                "debug_component",
                test_data,
                causal_context=CausalContext(
                    causes=["debug_test"],
                    effects=["test_complete"],
                    confidence=1.0
                )
            )
            print(f"✅ Saved state: {state_id}")
            
            loaded = await manager.load_state(
                StateType.COMPONENT_STATE,
                "debug_component"
            )
            print(f"✅ Loaded state: {loaded}")
            
            await manager.close()
            return True
        
        success = asyncio.run(test())
        return success
        
    except Exception as e:
        print(f"❌ Basic persistence failed: {e}")
        traceback.print_exc()
        return False

def check_file_structure():
    """Check if all required files exist"""
    print("\n" + "=" * 60)
    print("CHECKING FILE STRUCTURE")
    print("=" * 60)
    
    files_to_check = [
        "core/src/aura_intelligence/persistence/causal_state_manager.py",
        "core/src/aura_intelligence/persistence/memory_native.py",
        "core/src/aura_intelligence/agents/persistence_mixin.py",
        "core/src/aura_intelligence/agents/enhanced_code_agent.py",
        "core/src/aura_intelligence/neural/persistence_integration.py",
        "core/src/aura_intelligence/adapters/memory_persistence_integration.py",
        "core/src/aura_intelligence/tda/persistence_integration.py",
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")

def main():
    """Run all debug tests"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          PERSISTENCE ERROR DEBUGGING                      ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check files first
    check_file_structure()
    
    # Test imports
    test_imports()
    
    # Test basic functionality
    test_basic_persistence()
    
    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETE")
    print("=" * 60)
    print("\nCheck the errors above to identify the root cause.")

if __name__ == "__main__":
    main()