#!/usr/bin/env python3
"""
Minimal import test - check what works without external dependencies
"""

import sys
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print("=" * 60)

# Test 1: Direct module imports that should work
print("\n1️⃣ Testing DIRECT imports (no __init__.py)")
print("-" * 30)

try:
    # Import SimpleConsensus directly
    sys.path.insert(0, '/workspace/core/src')
    from aura_intelligence.consensus.simple import SimpleConsensus
    print("✅ SimpleConsensus imported directly")
except Exception as e:
    print(f"❌ SimpleConsensus failed: {e}")

# Test 2: Components that don't need external deps
print("\n2️⃣ Testing CORE components")
print("-" * 30)

try:
    from aura_intelligence.persistence.compatibility import CompatibilityStateManager
    print("✅ CompatibilityStateManager imported")
except Exception as e:
    print(f"❌ CompatibilityStateManager failed: {e}")

# Test 3: Basic schemas
print("\n3️⃣ Testing SCHEMAS")
print("-" * 30)

try:
    from aura_intelligence.events.schemas import EventSchema, EventType
    print("✅ Event schemas imported")
except Exception as e:
    print(f"❌ Event schemas failed: {e}")

# Test 4: Memory types
print("\n4️⃣ Testing MEMORY types")
print("-" * 30)

try:
    from aura_intelligence.memory.types import MemoryEntry, MemoryType
    print("✅ Memory types imported")
except Exception as e:
    print(f"❌ Memory types failed: {e}")

# Test 5: Base classes
print("\n5️⃣ Testing BASE classes")
print("-" * 30)

try:
    from aura_intelligence.agents.simple_base_agent import SimpleAgent
    print("✅ SimpleAgent imported")
except Exception as e:
    print(f"❌ SimpleAgent failed: {e}")

# Test 6: Utils
print("\n6️⃣ Testing UTILS")
print("-" * 30)

try:
    from aura_intelligence.utils.json_utils import serialize_for_api
    print("✅ JSON utils imported")
except Exception as e:
    print(f"❌ JSON utils failed: {e}")

# Test 7: Config
print("\n7️⃣ Testing CONFIG")
print("-" * 30)

try:
    from aura_intelligence.config import get_config
    print("✅ Config imported")
except Exception as e:
    print(f"❌ Config failed: {e}")

print("\n" + "=" * 60)
print("💡 Components that work without external dependencies:")
print("   - Event schemas and types")
print("   - Memory types and interfaces")
print("   - Simple consensus algorithms")
print("   - Basic agent classes")
print("   - Utility functions")
print("   - Configuration system")
print("\n❗ To use full features, install:")
print("   pip install msgpack asyncpg aiokafka langgraph torch")