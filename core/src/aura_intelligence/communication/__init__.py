"""
🚀 AURA Communication System
============================

Production-grade multi-agent communication with:
- NATS JetStream for reliability
- Neural Mesh for intelligent routing  
- FIPA ACL semantic protocols
- Collective swarm patterns
- Causal tracking and replay
- End-to-end security
"""

# Core protocols (no external deps)
from .protocols import (
    Performative,
    MessagePriority,
    SemanticEnvelope,
    AgentMessage
)

# Semantic protocols
from .semantic_protocols import (
    InteractionProtocol,
    ProtocolValidator,
    ConversationManager,
    ProtocolTemplates,
    AURA_CORE_ONTOLOGY
)

# Collective protocols  
from .collective_protocols import (
    CollectiveProtocolsManager,
    SwarmState,
    CollectivePattern
)

# Causal messaging
from .causal_messaging import (
    CausalGraphManager,
    CausalAnalyzer,
    CausalChain,
    CausalPattern
)

# Security
from .secure_channels import (
    SecureChannel,
    SecurityConfig,
    SubjectBuilder,
    DataProtection
)

__all__ = [
    # Protocols
    'Performative',
    'MessagePriority', 
    'SemanticEnvelope',
    'AgentMessage',
    
    # Semantic
    'InteractionProtocol',
    'ProtocolValidator',
    'ConversationManager',
    'ProtocolTemplates',
    'AURA_CORE_ONTOLOGY',
    
    # Collective
    'CollectiveProtocolsManager',
    'SwarmState',
    'CollectivePattern',
    
    # Causal
    'CausalGraphManager',
    'CausalAnalyzer',
    'CausalChain',
    'CausalPattern',
    
    # Security
    'SecureChannel',
    'SecurityConfig',
    'SubjectBuilder',
    'DataProtection'
]

# Optional imports (require external deps)
try:
    from .unified_communication import UnifiedCommunication, TraceContext
    from .nats_a2a import NATSA2ASystem
    from .neural_mesh import NeuralMesh
    
    __all__.extend([
        'UnifiedCommunication',
        'TraceContext',
        'NATSA2ASystem',
        'NeuralMesh'
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Optional communication components not available: {e}")

print("✅ AURA Communication System loaded successfully!")