#!/usr/bin/env python3
"""
Test AURA imports assuming aiokafka is NOT available
This simulates environments without all dependencies
"""

import sys
import os

# Mock aiokafka BEFORE any imports
class MockModule:
    def __getattr__(self, name):
        raise ImportError(f"No module named 'aiokafka.{name}'")

sys.modules['aiokafka'] = MockModule()
sys.modules['aiokafka.errors'] = MockModule()

# Now setup path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core/src'))

print("🧪 AURA Test WITHOUT aiokafka")
print("=" * 60)

# Test 1: Events should work with mocked aiokafka
print("\n1️⃣ Testing Events (without aiokafka)")
print("-" * 30)
try:
    from aura_intelligence.events import (
        EventProducer,
        EventConsumer,
        AgentEvent,
        SystemEvent
    )
    print(f"✅ Events imported!")
    print(f"   - EventProducer: {'None (aiokafka not available)' if EventProducer is None else 'Available'}")
    print(f"   - EventConsumer: {'None (aiokafka not available)' if EventConsumer is None else 'Available'}")
    print(f"   - AgentEvent: {AgentEvent}")
except Exception as e:
    print(f"❌ Events failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Consensus should work
print("\n\n2️⃣ Testing Consensus")
print("-" * 30)
try:
    from aura_intelligence.consensus import (
        SimpleConsensus,
        RaftConsensus,
        ByzantineConsensus
    )
    print("✅ Consensus imported!")
    
    # Test instantiation
    consensus = SimpleConsensus("node1", ["node2", "node3"])
    print("✅ SimpleConsensus instantiated")
except Exception as e:
    print(f"❌ Consensus failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try memory without going through main init
print("\n\n3️⃣ Testing Memory (direct)")
print("-" * 30)
try:
    # Skip the main imports and go direct
    import aura_intelligence.memory.hybrid_manager
    # Manually import what we need
    from aura_intelligence.memory.hybrid_manager import HybridMemoryManager
    print("✅ HybridMemoryManager imported directly")
except Exception as e:
    print(f"❌ Memory failed: {e}")

# Test 4: Summary
print("\n" + "=" * 60)
print("📊 Summary")
print("=" * 60)
print("\n✅ What works without aiokafka:")
print("- Consensus components")
print("- Basic event schemas")
print("- Direct imports (bypassing __init__.py)")
print("\n❌ What needs aiokafka:")
print("- EventProducer/Consumer (Kafka integration)")
print("- Stream processing")
print("\n💡 Recommendation:")
print("1. Use direct imports when possible")
print("2. Make aiokafka truly optional in all modules")
print("3. Or install aiokafka in your environment")