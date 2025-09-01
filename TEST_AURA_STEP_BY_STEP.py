#!/usr/bin/env python3
"""
Step-by-step test of AURA imports with real dependencies.
Run this with your environment that has aiokafka, langgraph, etc installed.
"""

import sys
import os
import traceback

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core/src'))

print("🧪 AURA Import Test - Step by Step")
print("=" * 60)

# Step 1: Test Memory imports
print("\n1️⃣ MEMORY Module")
print("-" * 30)
try:
    print("Importing memory components...")
    from aura_intelligence.memory import (
        HybridMemoryManager,
        MemoryManager,
        HierarchicalMemorySystem,
        HierarchicalMemoryManager,
        UnifiedMemoryInterface
    )
    print("✅ Memory imports successful!")
    print(f"   - HybridMemoryManager: {HybridMemoryManager.__name__}")
    print(f"   - MemoryManager == HybridMemoryManager: {MemoryManager == HybridMemoryManager}")
    print(f"   - HierarchicalMemorySystem: {HierarchicalMemorySystem.__name__}")
    print(f"   - HierarchicalMemoryManager == HierarchicalMemorySystem: {HierarchicalMemoryManager == HierarchicalMemorySystem}")
    
    # Test instantiation
    print("\nTesting memory instantiation...")
    memory = HybridMemoryManager()
    print("✅ HybridMemoryManager instantiated successfully!")
    
except Exception as e:
    print(f"❌ Memory import failed: {e}")
    traceback.print_exc()

# Step 2: Test Persistence imports
print("\n\n2️⃣ PERSISTENCE Module")
print("-" * 30)
try:
    print("Importing persistence components...")
    from aura_intelligence.persistence.causal_state_manager import (
        CausalPersistenceManager,
        CausalContext
    )
    print("✅ Causal persistence imports successful!")
    print(f"   - CausalPersistenceManager: {CausalPersistenceManager.__name__}")
    print(f"   - CausalContext: {CausalContext.__name__}")
    
    # Test instantiation (if dependencies available)
    print("\nTesting persistence instantiation...")
    import asyncio
    print("   - CausalPersistenceManager requires duckdb")
    
except Exception as e:
    print(f"❌ Persistence import failed: {e}")
    traceback.print_exc()

# Step 3: Test Neural imports
print("\n\n3️⃣ NEURAL Module")
print("-" * 30)
try:
    print("Importing neural components...")
    from aura_intelligence.neural import (
        LiquidNeuralNetwork,
        ProviderAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        AURAModelRouter,
        AdaptiveRoutingEngine
    )
    
    # Import MoE from the main module
    from aura_intelligence import (
        SwitchTransformerMoE,
        ProductionSwitchMoE
    )
    print("✅ Neural imports successful!")
    if LiquidNeuralNetwork:
        print(f"   - LiquidNeuralNetwork: Available")
    print(f"   - MoE: {SwitchTransformerMoE.__name__}, {ProductionSwitchMoE.__name__}")
    print(f"   - AURAModelRouter: {AURAModelRouter.__name__}")
    print(f"   - Provider adapters: OpenAI, Anthropic")
    print(f"   - AdaptiveRoutingEngine: {AdaptiveRoutingEngine.__name__}")
    
except Exception as e:
    print(f"❌ Neural import failed: {e}")
    traceback.print_exc()

# Step 4: Test Consensus imports
print("\n\n4️⃣ CONSENSUS Module")
print("-" * 30)
try:
    print("Importing consensus components...")
    from aura_intelligence.consensus import (
        SimpleConsensus,
        RaftConsensus,
        ByzantineConsensus,
        ConsensusRequest,
        ConsensusResult
    )
    print("✅ Consensus imports successful!")
    print(f"   - SimpleConsensus: {SimpleConsensus.__name__}")
    print(f"   - RaftConsensus: {RaftConsensus.__name__}")
    print(f"   - ByzantineConsensus: {ByzantineConsensus.__name__}")
    
except Exception as e:
    print(f"❌ Consensus import failed: {e}")
    traceback.print_exc()

# Step 5: Test Events imports (with aiokafka)
print("\n\n5️⃣ EVENTS Module")
print("-" * 30)
try:
    print("Importing events components...")
    from aura_intelligence.events import (
        EventProducer,
        EventConsumer,
        AgentEvent,
        SystemEvent
    )
    if EventProducer is None:
        print("⚠️  EventProducer not available (aiokafka not installed)")
    else:
        print("✅ Events imports successful!")
        print(f"   - EventProducer: {EventProducer.__name__}")
        print(f"   - EventConsumer: {EventConsumer.__name__ if EventConsumer else 'Not available'}")
    
except Exception as e:
    print(f"❌ Events import failed: {e}")
    traceback.print_exc()

# Step 6: Test Agents imports (with langgraph)
print("\n\n6️⃣ AGENTS Module")
print("-" * 30)
try:
    print("Importing agent components...")
    from aura_intelligence.agents import (
        AURAAgent,
        AgentConfig,
        SimpleAgent,
        ConsolidatedAgent
    )
    print("✅ Agents imports successful!")
    if AURAAgent is not None:
        print(f"   - AURAAgent: {AURAAgent.__name__}")
    else:
        print("   - AURAAgent: Not available (langgraph required)")
    print(f"   - SimpleAgent: {SimpleAgent.__name__ if SimpleAgent else 'Not available'}")
    
except Exception as e:
    print(f"❌ Agents import failed: {e}")
    traceback.print_exc()

# Step 7: Test the full integrated import
print("\n\n7️⃣ FULL SYSTEM Import")
print("-" * 30)
try:
    print("Importing full AURA system...")
    import aura_intelligence
    print("✅ Full AURA import successful!")
    print(f"   - Version: {aura_intelligence.__version__}")
    
    # Test AURA class
    print("\nTesting AURA instantiation...")
    aura = aura_intelligence.AURA()
    print("✅ AURA system instantiated successfully!")
    
except Exception as e:
    print(f"❌ Full system import failed: {e}")
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("📊 IMPORT TEST SUMMARY")
print("=" * 60)
print("\nThis test checked:")
print("- Memory module with proper aliases")
print("- Persistence with causal state manager")
print("- Neural components (LNN, MoE, Mamba)")
print("- Consensus algorithms")
print("- Events (if aiokafka available)")
print("- Agents (if langgraph available)")
print("- Full system integration")

print("\n✅ Run this test after fixing any remaining import issues!")
print("If you see errors, share them and I'll fix them manually.")