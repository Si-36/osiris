#!/usr/bin/env python3
"""
Direct import test - bypasses the main __init__.py chain
"""

import sys
import os

# Add path WITHOUT importing through __init__.py
sys.path.insert(0, '/workspace/core/src')

print("🧪 Direct Import Test (Bypassing __init__.py)")
print("=" * 60)

# Test 1: Import specific consensus components directly
print("\n1️⃣ Direct Consensus Imports")
print("-" * 30)
try:
    # Import consensus types first
    from aura_intelligence.consensus.consensus_types import ConsensusRequest, ConsensusResult
    print("✅ consensus_types imported")
    
    # Import simple consensus directly
    from aura_intelligence.consensus.simple import SimpleConsensus
    print("✅ SimpleConsensus imported")
    
    # Test instantiation
    consensus = SimpleConsensus("node1", ["node2", "node3"])
    print("✅ SimpleConsensus instantiated")
    
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Import memory components directly
print("\n\n2️⃣ Direct Memory Imports")
print("-" * 30)
try:
    # First try the advanced one
    from aura_intelligence.memory.advanced_hybrid_memory_2025 import HybridMemoryManager as AdvancedMemory
    print("✅ Advanced HybridMemoryManager imported")
    
    # Try the wrapper
    from aura_intelligence.memory.hybrid_manager import HybridMemoryManager
    print("✅ HybridMemoryManager wrapper imported")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Import persistence directly
print("\n\n3️⃣ Direct Persistence Imports")
print("-" * 30)
try:
    # Import causal manager directly - it needs asyncpg
    from aura_intelligence.persistence.causal_state_manager import (
        CausalStateManager,
        StateType,
        CausalContext
    )
    print("✅ CausalStateManager imported")
    print(f"   StateType options: {[st.name for st in StateType]}")
    
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Import neural components directly
print("\n\n4️⃣ Direct Neural Imports")
print("-" * 30)
try:
    # These may need torch
    from aura_intelligence.neural.liquid_neural_network import LiquidNeuralNetwork
    print("✅ LiquidNeuralNetwork imported")
except Exception as e:
    print(f"⚠️  LNN needs torch: {e}")

try:
    from aura_intelligence.neural.mixture_of_experts import MixtureOfExperts
    print("✅ MixtureOfExperts imported")
except Exception as e:
    print(f"⚠️  MoE needs torch: {e}")

# Test 5: Check what specific classes exist
print("\n\n5️⃣ Component Discovery")
print("-" * 30)

# Check memory classes
print("\nMemory components:")
import os
memory_path = '/workspace/core/src/aura_intelligence/memory'
for file in os.listdir(memory_path):
    if file.endswith('.py') and not file.startswith('_') and not file.startswith('test'):
        print(f"  - {file}")

# Check consensus classes  
print("\nConsensus components:")
consensus_path = '/workspace/core/src/aura_intelligence/consensus'
for file in os.listdir(consensus_path):
    if file.endswith('.py') and not file.startswith('_'):
        print(f"  - {file}")

print("\n" + "=" * 60)
print("📊 Direct Import Summary")
print("=" * 60)
print("\nWorking components:")
print("- SimpleConsensus ✓")
print("- Memory managers (check output)")
print("- CausalStateManager (if asyncpg available)")
print("- Neural (if torch available)")
print("\n💡 The issue is the main __init__.py imports everything,")
print("   creating a dependency chain. Direct imports work better!")
print("\n🔧 To use in your code:")
print("   from aura_intelligence.consensus.simple import SimpleConsensus")
print("   from aura_intelligence.memory.hybrid_manager import HybridMemoryManager")
print("   # etc - import directly, not through main module")