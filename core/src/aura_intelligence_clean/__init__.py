"""
🌟 AURA Intelligence System - Clean Rebuild
==========================================

Version 3.0 - Built from scratch with our best components
"""

__version__ = "3.0.0"

# Import our clean components
from .neural import AURAModelRouter
from .memory import AURAMemorySystem  
from .tda import AgentTopologyAnalyzer
from .orchestration import UnifiedOrchestrationEngine
from .swarm import SwarmCoordinator
from .core import AURACore, SelfHealingEngine
from .agents import AURAAgent, create_observer_agent, create_analyst_agent

class AURA:
    """Main AURA System - Clean and Working"""
    
    def __init__(self):
        # Initialize our components
        self.neural = AURAModelRouter()
        self.memory = AURAMemorySystem()
        self.tda = AgentTopologyAnalyzer()
        self.orchestration = UnifiedOrchestrationEngine()
        self.swarm = SwarmCoordinator()
        self.core = AURACore()
        
        # Connect them
        self._connect_components()
        
    def _connect_components(self):
        """Connect components together"""
        # Memory uses TDA for topology
        self.memory.set_topology_analyzer(self.tda)
        
        # Orchestration uses memory
        self.orchestration.set_memory(self.memory)
        
        # Core manages everything
        self.core.register_component("neural", self.neural)
        self.core.register_component("memory", self.memory)
        self.core.register_component("tda", self.tda)
        self.core.register_component("orchestration", self.orchestration)
        self.core.register_component("swarm", self.swarm)
        
    async def process(self, request: dict) -> dict:
        """Main processing interface"""
        # Route through neural
        routing_decision = await self.neural.route(request)
        
        # Store in memory
        await self.memory.store(request)
        
        # Process through orchestration
        result = await self.orchestration.execute(routing_decision)
        
        return result

# Simple factory
def create_aura() -> AURA:
    """Create a new AURA instance"""
    return AURA()

__all__ = [
    'AURA',
    'create_aura',
    'AURAModelRouter',
    'AURAMemorySystem',
    'AgentTopologyAnalyzer',
    'UnifiedOrchestrationEngine',
    'SwarmCoordinator',
    'AURACore',
    'AURAAgent',
    '__version__',
]