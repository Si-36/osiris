#!/usr/bin/env python3
"""Test core imports work without external dependencies"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core/src'))

print("🧪 Testing Core Import Structure...")
print("=" * 60)

# Test 1: Memory imports
print("\n1️⃣ Testing Memory imports...")
try:
    from aura_intelligence.memory import (
        HybridMemoryManager,
        MemoryManager,
        HierarchicalMemorySystem,
        HierarchicalMemoryManager,
        UnifiedMemoryInterface
    )
    print("✅ Memory imports successful!")
    print(f"   - HybridMemoryManager: {HybridMemoryManager}")
    print(f"   - MemoryManager is HybridMemoryManager: {MemoryManager is HybridMemoryManager}")
    print(f"   - HierarchicalMemorySystem: {HierarchicalMemorySystem}")
    print(f"   - HierarchicalMemoryManager is HierarchicalMemorySystem: {HierarchicalMemoryManager is HierarchicalMemorySystem}")
except Exception as e:
    print(f"❌ Memory import error: {e}")

# Test 2: Resilience imports
print("\n2️⃣ Testing Resilience imports...")
try:
    from aura_intelligence.resilience import (
        AdaptiveCircuitBreaker,
        CircuitBreaker,
        RetryPolicy,
        AdaptiveTimeout
    )
    print("✅ Resilience imports successful!")
    print(f"   - AdaptiveCircuitBreaker: {AdaptiveCircuitBreaker}")
    print(f"   - CircuitBreaker is AdaptiveCircuitBreaker: {CircuitBreaker is AdaptiveCircuitBreaker}")
except Exception as e:
    print(f"❌ Resilience import error: {e}")

# Test 3: Persistence imports
print("\n3️⃣ Testing Persistence imports...")
try:
    from aura_intelligence.persistence import (
        BackupManager,
        BackupSchedule,
        RestoreEngine,
        PointInTimeRecovery,
        ReplicationManager,
        CrossRegionSync
    )
    print("✅ Persistence backup imports successful!")
    
    from aura_intelligence.persistence.causal_state_manager import (
        CausalStateManager,
        StateType,
        CausalContext,
        get_causal_manager
    )
    print("✅ Causal persistence imports successful!")
except Exception as e:
    print(f"❌ Persistence import error: {e}")

# Test 4: Consensus imports
print("\n4️⃣ Testing Consensus imports...")
try:
    from aura_intelligence.consensus import (
        SimpleConsensus,
        RaftConsensus,
        ByzantineConsensus,
        ConsensusRequest,
        ConsensusResult
    )
    print("✅ Consensus imports successful!")
except Exception as e:
    print(f"❌ Consensus import error: {e}")

# Test 5: Neural imports (without external deps)
print("\n5️⃣ Testing Neural imports...")
try:
    # These should work without torch
    from aura_intelligence.neural import (
        ProviderAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        GeminiAdapter
    )
    print("✅ Neural provider imports successful!")
except Exception as e:
    print(f"❌ Neural import error: {e}")

# Test 6: Check if we can import without langgraph
print("\n6️⃣ Testing Agent base imports...")
try:
    from aura_intelligence.agents.base import AgentBase, AgentConfig
    print("✅ Agent base imports successful!")
except Exception as e:
    print(f"❌ Agent base import error: {e}")

print("\n" + "=" * 60)
print("🎯 Core import test complete!")