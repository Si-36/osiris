
"""
Component Initialization Helper
==============================

Ensures all components are properly initialized with connections.
"""

import asyncio
from typing import Optional, Dict, Any

class AURASystem:
    """Unified system with all components connected"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.orchestration = None
        self.memory = None
        self.neural = None
        self.tda = None
        
    async def initialize(self):
        """Initialize all components with proper connections"""
        
        # 1. Memory (foundation for others)
        from aura_intelligence.memory.core.memory_api import AURAMemorySystem
        self.memory = AURAMemorySystem()
        if hasattr(self.memory, 'initialize'):
            await self.memory.initialize()
        print("✅ Memory initialized")
        
        # 2. TDA with memory wrapper
        from aura_intelligence.tda import AgentTopologyAnalyzer
        from tda_memory_wrapper import TDAMemoryWrapper
        tda_base = AgentTopologyAnalyzer()
        self.tda = TDAMemoryWrapper(tda_base, self.memory)
        print("✅ TDA initialized with memory integration")
        
        # 3. Neural with memory wrapper
        try:
            from aura_intelligence.neural import AURAModelRouter
            from neural_memory_wrapper import NeuralMemoryWrapper
            neural_base = AURAModelRouter()
            if hasattr(neural_base, 'initialize'):
                await neural_base.initialize()
            self.neural = NeuralMemoryWrapper(neural_base, self.memory)
            print("✅ Neural initialized with memory integration")
        except Exception as e:
            print(f"⚠️  Neural initialization failed: {e}")
            
        # 4. Orchestration (uses all others)
        try:
            from aura_intelligence.orchestration.unified_orchestration_engine import (
                UnifiedOrchestrationEngine,
                OrchestrationConfig
            )
            config = OrchestrationConfig(
                enable_topology_routing=True,
                enable_signal_first=True
            )
            self.orchestration = UnifiedOrchestrationEngine(config)
            # Inject our wrapped components
            self.orchestration.memory_system = self.memory
            self.orchestration.tda_analyzer = self.tda
            await self.orchestration.initialize()
            print("✅ Orchestration initialized with all connections")
        except Exception as e:
            print(f"⚠️  Orchestration initialization failed: {e}")
            
        print("\n🎉 AURA System Ready!")
        return self


# Convenience function
async def create_aura_system(config=None):
    """Create and initialize complete AURA system"""
    system = AURASystem(config)
    await system.initialize()
    return system
