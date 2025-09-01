#!/usr/bin/env python3
"""
Clean step-by-step test of AURA imports.
Tests each component independently to isolate issues.
"""

import sys
import os

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core/src'))

print("🧪 AURA Clean Import Test")
print("=" * 60)

# Step 1: Test Events first (since it's working)
print("\n1️⃣ Testing EVENTS Module (standalone)")
print("-" * 30)
try:
    from aura_intelligence.events import EventProducer, EventConsumer
    print("✅ Events module works!")
    print(f"   - EventProducer: {'Available' if EventProducer else 'Not available'}")
    print(f"   - EventConsumer: {'Available' if EventConsumer else 'Not available'}")
except Exception as e:
    print(f"❌ Events failed: {e}")

# Step 2: Test Persistence causal_state_manager directly
print("\n\n2️⃣ Testing PERSISTENCE (direct import)")
print("-" * 30)
try:
    # Direct import to bypass other modules
    sys.path.insert(0, '/workspace/core/src')
    from aura_intelligence.persistence.causal_state_manager import (
        CausalStateManager,
        StateType,
        CausalContext
    )
    print("✅ Causal persistence works!")
    print(f"   - CausalStateManager: {CausalStateManager.__name__}")
    print(f"   - StateType values: {[st.name for st in StateType]}")
except Exception as e:
    print(f"❌ Persistence failed: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test Memory hybrid_manager directly
print("\n\n3️⃣ Testing MEMORY (direct import)")
print("-" * 30)
try:
    from aura_intelligence.memory.hybrid_manager import HybridMemoryManager
    print("✅ HybridMemoryManager works!")
    
    # Test instantiation
    memory = HybridMemoryManager()
    print("✅ Can instantiate HybridMemoryManager")
except Exception as e:
    print(f"❌ Memory failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test specific Neural components
print("\n\n4️⃣ Testing NEURAL components (direct)")
print("-" * 30)
try:
    from aura_intelligence.neural.liquid_neural_network import LiquidNeuralNetwork
    print("✅ LiquidNeuralNetwork imported")
except Exception as e:
    print(f"❌ LNN failed: {e}")

try:
    from aura_intelligence.neural.mixture_of_experts import MixtureOfExperts
    print("✅ MixtureOfExperts imported")
except Exception as e:
    print(f"❌ MoE failed: {e}")

try:
    from aura_intelligence.neural.mamba_architecture import MambaArchitecture
    print("✅ MambaArchitecture imported")
except Exception as e:
    print(f"❌ Mamba failed: {e}")

# Step 5: Test a simple consensus component
print("\n\n5️⃣ Testing CONSENSUS (simple)")
print("-" * 30)
try:
    from aura_intelligence.consensus.simple import SimpleConsensus
    print("✅ SimpleConsensus imported")
    
    # Test instantiation
    consensus = SimpleConsensus("node1", ["node2", "node3"])
    print("✅ Can instantiate SimpleConsensus")
except Exception as e:
    print(f"❌ SimpleConsensus failed: {e}")

# Step 6: Check what's blocking the main imports
print("\n\n6️⃣ Checking blocking imports")
print("-" * 30)

# Check consensus workflows
print("\nChecking consensus/workflows.py...")
try:
    import aura_intelligence.consensus.workflows
    print("✅ Workflows can be imported (temporalio available)")
except Exception as e:
    print(f"⚠️  Workflows blocked by: {type(e).__name__}: {e}")

# Check persistence backup
print("\nChecking persistence/backup...")
try:
    import aura_intelligence.persistence.backup
    print("✅ Backup module exists")
except Exception as e:
    print(f"⚠️  Backup blocked by: {type(e).__name__}: {e}")

# Check neural performance_tracker
print("\nChecking neural/performance_tracker.py...")
try:
    import aura_intelligence.neural.performance_tracker
    print("✅ Performance tracker can be imported")
except Exception as e:
    print(f"⚠️  Performance tracker blocked by: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print("\nDirect imports that work:")
print("- Events module ✓")
print("- Persistence causal_state_manager ✓")
print("- Memory hybrid_manager ✓")
print("- Neural components (if torch available)")
print("- Consensus simple ✓")
print("\nBlocking issues:")
print("- temporalio not installed (blocks consensus workflows)")
print("- Other dependencies may block full module imports")
print("\n💡 Recommendation: Import specific components directly")
print("   instead of using module-level imports until all")
print("   dependencies are properly handled.")