"""
🌟 AURA Intelligence System
==========================

The World's Most Advanced Agent Infrastructure Platform
Built with Topological Intelligence, Consciousness-Aware Routing, and Self-Healing

Version: 3.0.0 - Complete Rebuild
"""

__version__ = "3.0.0"

# ===================== OUR CORE COMPONENTS =====================

# 1. Neural System - Intelligent Model Routing
from .neural import (
    AURAModelRouter,
    AdaptiveRoutingEngine,
    ProviderAdapter,
    ModelPerformanceTracker,
    CacheManager,
)

# 1.5. MoE System - Mixture of Experts
from .moe import (
    SwitchTransformerMoE,
    ProductionSwitchMoE,
    SwitchMoEWithLNN,
    create_production_switch_moe,
)

# 2. Memory System - Topological Memory with Hardware Tiers
from .memory.core.memory_api import (
    AURAMemorySystem,
    MemoryType,
    MemoryQuery,
    MemoryRecord,
    RetrievalResult,
)

# 3. TDA System - Agent Topology Analysis
from .tda import (
    AgentTopologyAnalyzer,
    RealtimeTopologyMonitor,
    WorkflowFeatures,
    compute_persistence,
    diagram_entropy,
    diagram_distance,
)

# 4. Orchestration - LangGraph-based Orchestration
from .orchestration import (
    UnifiedOrchestrationEngine,
    OrchestrationConfig,
    WorkflowDefinition,
    WorkflowCheckpointManager,
)

# 5. Swarm Intelligence - Collective Coordination
from .swarm_intelligence import (
    SwarmCoordinator,
    SwarmAlgorithm,
    PheromoneType,
)

# 6. Core System - Main System & Self-Healing
from .core.aura_main_system import (
    AURAMainSystem,
    SystemConfig,
    SystemMetrics,
)
from .core.self_healing_engine import (
    SelfHealingEngine,
    HealingStrategy,
    FailureType,
)
from .core.executive_controller import (
    ExecutiveController,
    ConsciousnessLevel,
    ConsciousnessState,
)

# 7. Infrastructure - Event Mesh & Guardrails
from .infrastructure import (
    UnifiedEventMesh,
    EnhancedGuardrails,
    MultiProviderClient,
)

# 8. Communication - NATS + Neural Mesh
from .communication import (
    UnifiedCommunication,
    EnhancedNeuralMesh,
    Performative,
    SemanticEnvelope,
)

# 9. Agents - Core Agent System
try:
    from .agents import (
        AURAAgent,
        AgentConfig,
        LNNCouncilOrchestrator,
        agent_templates,
    )
    AGENTS_AVAILABLE = True
except ImportError:
    # Agents require external dependencies (langgraph)
    AGENTS_AVAILABLE = False
    AURAAgent = None
    AgentConfig = None
    LNNCouncilOrchestrator = None
    agent_templates = None

# ===================== MAIN SYSTEM =====================

class AURA:
    """
    The main AURA system interface.
    
    This is the primary entry point for using AURA Intelligence.
    """
    
    def __init__(self, config: SystemConfig = None):
        """Initialize AURA with optional configuration."""
        self.system = AURAMainSystem(config or SystemConfig())
    
    async def start(self):
        """Start the AURA system."""
        await self.system.start()
    
    async def stop(self):
        """Stop the AURA system."""
        await self.system.stop()
    
    def __repr__(self):
        return f"<AURA Intelligence System v{__version__}>"


# ===================== EXPORTS =====================

__all__ = [
    # Main Interface
    "AURA",
    "__version__",
    
    # Neural System
    "AURAModelRouter",
    "AdaptiveRoutingEngine",
    "ProviderAdapter",
    "PerformanceTracker",
    "CacheManager",
    
    # Memory System
    "AURAMemorySystem",
    "MemoryConfig",
    "MemoryEntry",
    "QueryResult",
    
    # TDA System
    "AgentTopologyAnalyzer",
    "RealtimeTopologyMonitor",
    "compute_persistence",
    "diagram_entropy", 
    "diagram_distance",
    "WorkflowFeatures",
    
    # Orchestration
    "UnifiedOrchestrationEngine",
    "OrchestrationConfig",
    "WorkflowDefinition",
    "WorkflowCheckpointManager",
    
    # Swarm Intelligence
    "SwarmCoordinator",
    "SwarmAlgorithm",
    "PheromoneType",
    
    # Core System
    "AURAMainSystem",
    "SystemConfig",
    "SystemMetrics",
    "SelfHealingEngine",
    "HealingStrategy",
    "FailureType",
    "ExecutiveController",
    "ConsciousnessLevel",
    "ConsciousnessState",
    
    # Infrastructure
    "UnifiedEventMesh",
    "EnhancedGuardrails",
    "MultiProviderClient",
    
    # Communication
    "UnifiedCommunication",
    "EnhancedNeuralMesh",
    "Performative",
    "SemanticEnvelope",
    
]

# Add agent exports only if available
if AGENTS_AVAILABLE:
    __all__.extend([
        "AURAAgent",
        "AgentConfig",
        "LNNCouncilOrchestrator",
        "agent_templates",
    ])

# ===================== QUICK START =====================

def create_aura(config: dict = None) -> AURA:
    """
    Create and configure an AURA instance.
    
    Example:
        >>> aura = create_aura({'enable_swarm': True})
        >>> await aura.start()
    """
    if config:
        system_config = SystemConfig(**config)
    else:
        system_config = SystemConfig()
    
    return AURA(system_config)


# Make it easy to use
if __name__ == "__main__":
    print(f"AURA Intelligence System v{__version__}")
    print("Use: aura = create_aura() to get started")