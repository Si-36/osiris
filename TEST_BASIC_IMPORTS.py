#!/usr/bin/env python3
"""
Test basic imports step by step
================================
"""
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("Testing imports step by step...\n")

# Test 1: Basic persistence
try:
    print("1. Testing causal_state_manager...")
    from aura_intelligence.persistence.causal_state_manager import CausalPersistenceManager
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Memory native
try:
    print("\n2. Testing memory_native...")
    from aura_intelligence.persistence.memory_native import MemoryNativeAI
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Agents
try:
    print("\n3. Testing agents...")
    from aura_intelligence.agents.base import BaseAgent
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Resilience
try:
    print("\n4. Testing resilience...")
    from aura_intelligence.agents.resilience.bulkhead import Bulkhead
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 5: Observer
try:
    print("\n5. Testing observer...")
    from aura_intelligence.agents.v2.observer import ObserverAgentV2
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n✅ Basic import test complete!")