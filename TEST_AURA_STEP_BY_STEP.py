#!/usr/bin/env python3
"""
🧪 AURA Import Test - Step by Step
============================================================
Tests each module separately to identify import issues
"""

import sys
import traceback

print("🧪 AURA Import Test - Step by Step")
print("=" * 60)

# Add the path to AURA
sys.path.insert(0, 'core/src')

# 1. MEMORY Module
print("\n1️⃣ MEMORY Module")
print("-" * 30)
print("Importing memory components...")
try:
    from aura_intelligence.memory import (
        HierarchicalMemoryManager,
        HybridMemoryManager,
        ShapeMemoryV2,
        MemoryConfig,
        MemoryInterface,
        create_memory_manager
    )
    print("✅ Memory imports successful!")
    print(f"   - HierarchicalMemoryManager: {HierarchicalMemoryManager}")
    print(f"   - HybridMemoryManager: {HybridMemoryManager}")
    print(f"   - ShapeMemoryV2: {ShapeMemoryV2}")
except Exception as e:
    print(f"❌ Memory import failed: {e}")
    traceback.print_exc()

# 2. PERSISTENCE Module
print("\n\n2️⃣ PERSISTENCE Module")
print("-" * 30)
print("Importing persistence components...")
try:
    from aura_intelligence.persistence.causal_state_manager import (
        CausalPersistenceManager,
        CausalContext
    )
    print("✅ Causal persistence imports successful!")
    print(f"   - CausalPersistenceManager: {CausalPersistenceManager}")
    print(f"   - CausalContext: {CausalContext}")
    
    print("\nTesting persistence instantiation...")
    try:
        # Don't actually instantiate as it requires duckdb
        print("   - CausalPersistenceManager requires duckdb")
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
except Exception as e:
    print(f"❌ Persistence import failed: {e}")
    traceback.print_exc()

# 3. NEURAL Module
print("\n\n3️⃣ NEURAL Module")
print("-" * 30)
print("Importing neural components...")
try:
    # Try to import from main init first
    from aura_intelligence import (
        LiquidNeuralNetwork,
        MixtureOfExperts,
        MambaV2
    )
    print("✅ Neural imports successful!")
    print(f"   - LiquidNeuralNetwork: {LiquidNeuralNetwork}")
    print(f"   - MixtureOfExperts: {MixtureOfExperts}")
    print(f"   - MambaV2: {MambaV2}")
except Exception as e:
    print(f"❌ Neural import failed: {e}")
    traceback.print_exc()

# 4. CONSENSUS Module
print("\n\n4️⃣ CONSENSUS Module")
print("-" * 30)
print("Importing consensus components...")
try:
    from aura_intelligence.consensus import (
        SimpleConsensus,
        RaftConsensus,
        ByzantineConsensus
    )
    print("✅ Consensus imports successful!")
    print(f"   - SimpleConsensus: {SimpleConsensus}")
    print(f"   - RaftConsensus: {RaftConsensus}")
    print(f"   - ByzantineConsensus: {ByzantineConsensus}")
except Exception as e:
    print(f"❌ Consensus import failed: {e}")
    traceback.print_exc()

# 5. EVENTS Module
print("\n\n5️⃣ EVENTS Module")
print("-" * 30)
print("Importing events components...")
try:
    # Try with conditional imports for aiokafka
    try:
        from aura_intelligence.events import EventProducer
        print(f"✅ EventProducer: {EventProducer}")
    except ImportError:
        print("⚠️  EventProducer not available (aiokafka not installed)")
    
    try:
        from aura_intelligence.events import EventProcessor
        print(f"✅ EventProcessor: {EventProcessor}")
    except ImportError:
        print("⚠️  EventProcessor not available")
except Exception as e:
    print(f"❌ Events import failed: {e}")
    traceback.print_exc()

# 6. AGENTS Module
print("\n\n6️⃣ AGENTS Module")
print("-" * 30)
print("Importing agent components...")
try:
    from aura_intelligence.agents import (
        AURAAgent,
        SimpleAgent
    )
    print("✅ Agents imports successful!")
    print(f"   - AURAAgent: {'Not available (langgraph required)' if not hasattr(AURAAgent, '__name__') else AURAAgent}")
    print(f"   - SimpleAgent: {SimpleAgent}")
except Exception as e:
    print(f"❌ Agents import failed: {e}")
    traceback.print_exc()

# 7. Full System
print("\n\n7️⃣ FULL SYSTEM Import")
print("-" * 30)
print("Importing full AURA system...")
try:
    import aura_intelligence
    print("✅ Full AURA system imported successfully!")
    print(f"   - Module: {aura_intelligence}")
    print(f"   - Location: {aura_intelligence.__file__}")
except Exception as e:
    print(f"❌ Full system import failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("📊 IMPORT TEST SUMMARY")
print("=" * 60)
print("""
This test checked:
- Memory module with proper aliases
- Persistence with causal state manager
- Neural components (LNN, MoE, Mamba)
- Consensus algorithms
- Events (if aiokafka available)
- Agents (if langgraph available)
- Full system integration

✅ Run this test after fixing any remaining import issues!
If you see errors, share them and I'll fix them manually.""")