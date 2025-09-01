#!/usr/bin/env python3
"""
Safe Persistence Test - Handles Missing Dependencies Gracefully
===============================================================
This test will show you exactly what works and what needs fixing
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "core" / "src"))

def test_imports():
    """Test importing persistence components with detailed error reporting"""
    print("\n🔍 Testing Core Imports...")
    print("=" * 60)
    
    # Test basic imports first
    basic_imports = [
        ("asyncio", "asyncio"),
        ("dataclasses", "dataclasses"),
        ("typing", "typing"),
        ("pathlib", "pathlib"),
        ("json", "json"),
        ("structlog", "structlog"),
    ]
    
    for name, module in basic_imports:
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError as e:
            print(f"❌ {name}: {e}")
    
    print("\n🔍 Testing AURA Components...")
    print("=" * 60)
    
    # Test AURA imports with detailed path
    aura_imports = [
        ("Persistence Manager", "aura_intelligence.persistence.causal_state_manager"),
        ("Memory Native", "aura_intelligence.persistence.memory_native"),
        ("Base Agent", "aura_intelligence.agents.base"),
        ("Enhanced Code Agent", "aura_intelligence.agents.enhanced_code_agent"),
        ("Neural Integration", "aura_intelligence.neural.persistence_integration"),
        ("Memory Adapter", "aura_intelligence.adapters.memory_persistence_integration"),
        ("TDA Integration", "aura_intelligence.tda.persistence_integration"),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for name, module_path in aura_imports:
        try:
            module = __import__(module_path, fromlist=[''])
            successful_imports.append((name, module_path))
            print(f"✅ {name}: Loaded successfully")
        except ImportError as e:
            failed_imports.append((name, module_path, str(e)))
            print(f"❌ {name}: Import failed - {e}")
        except Exception as e:
            failed_imports.append((name, module_path, f"{type(e).__name__}: {e}"))
            print(f"❌ {name}: Error - {type(e).__name__}: {e}")
    
    return successful_imports, failed_imports

async def test_basic_persistence():
    """Test basic persistence functionality if imports work"""
    print("\n🧪 Testing Basic Persistence...")
    print("=" * 60)
    
    try:
        from aura_intelligence.persistence.causal_state_manager import (
            CausalPersistenceManager,
            StateType,
            CausalContext,
        )
        
        # Create manager
        manager = CausalPersistenceManager()
        print("✅ Created CausalPersistenceManager")
        
        # Test basic save
        await manager.initialize()
        print("✅ Manager initialized")
        
        # Create test state
        test_state = {
            "test": True,
            "timestamp": "2025-08-31",
            "data": {"key": "value"}
        }
        
        # Save state
        state_id = await manager.save_state(
            StateType.AGENT_MEMORY,
            "test_component",
            test_state
        )
        print(f"✅ Saved state with ID: {state_id}")
        
        # Load state
        loaded = await manager.load_state(StateType.AGENT_MEMORY, "test_component")
        print(f"✅ Loaded state: {loaded is not None}")
        
        await manager.cleanup()
        print("✅ Cleanup completed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot test persistence - missing imports: {e}")
        return False
    except Exception as e:
        print(f"❌ Persistence test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_structure():
    """Check if all persistence files exist"""
    print("\n📁 Checking File Structure...")
    print("=" * 60)
    
    files_to_check = [
        "core/src/aura_intelligence/persistence/causal_state_manager.py",
        "core/src/aura_intelligence/persistence/memory_native.py",
        "core/src/aura_intelligence/persistence/migrate_from_pickle.py",
        "core/src/aura_intelligence/agents/persistence_mixin.py",
        "core/src/aura_intelligence/agents/enhanced_code_agent.py",
        "docker-compose.persistence.yml",
        "init_scripts/postgres/01_init_aura.sql",
        "requirements-persistence.txt",
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")

def check_syntax_errors():
    """Quick syntax check on key files"""
    print("\n🔍 Checking Syntax in Key Files...")
    print("=" * 60)
    
    key_files = [
        "core/src/aura_intelligence/agents/resilience/bulkhead.py",
        "core/src/aura_intelligence/events/streams.py",
        "core/src/aura_intelligence/events/connectors.py",
        "core/src/aura_intelligence/persistence/causal_state_manager.py",
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
                compile(code, file_path, 'exec')
                print(f"✅ {file_path} - Syntax OK")
            except SyntaxError as e:
                print(f"❌ {file_path} - Syntax Error: {e}")
            except Exception as e:
                print(f"⚠️  {file_path} - {type(e).__name__}")

async def main():
    """Run all tests"""
    print("🚀 AURA Persistence System - Safe Test")
    print("=" * 60)
    print("This test handles missing dependencies gracefully")
    
    # Check file structure
    check_file_structure()
    
    # Check syntax
    check_syntax_errors()
    
    # Test imports
    successful, failed = test_imports()
    
    # Summary
    print("\n📊 Import Summary:")
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")
    
    if failed:
        print("\n❌ Failed Imports Details:")
        for name, path, error in failed:
            print(f"   - {name}: {error}")
    
    # Test persistence if possible
    if successful:
        await test_basic_persistence()
    
    print("\n✅ Test Complete!")
    print("\n💡 Next Steps:")
    print("1. Install missing dependencies:")
    print("   pip install aiokafka fastavro confluent-kafka structlog")
    print("2. Make sure PostgreSQL is running (or use SQLite mode)")
    print("3. Run the full test: python3 TEST_PERSISTENCE_NOW.py")

if __name__ == "__main__":
    asyncio.run(main())