"""
🏗️ AURA Infrastructure Layer
============================

Production infrastructure for event-driven architecture, safety guardrails,
and multi-provider AI integration.

Components:
- Unified Event Mesh (NATS + Kafka)
- Enterprise Guardrails (Safety & Cost)
- Multi-Provider Client (OpenAI, Anthropic, Gemini)
"""

# Event Mesh
from .unified_event_mesh import (
    UnifiedEventMesh,
    CloudEvent,
    EventChannel,
    EventConfig,
    SchemaRegistry,
    create_event_mesh,
    create_event
)

# Guardrails
from .enhanced_guardrails import (
    EnhancedEnterpriseGuardrails,
    GuardrailsConfig,
    GuardrailsMetrics,
    TenantContext,
    TenantManager,
    get_guardrails,
    secure_ainvoke
)

# Multi-Provider AI
from .multi_provider_client import (
    MultiProviderClient,
    MultiProviderConfig,
    ProviderConfig,
    ProviderResponse
)

__all__ = [
    # Event Mesh
    "UnifiedEventMesh",
    "CloudEvent",
    "EventChannel",
    "EventConfig",
    "SchemaRegistry",
    "create_event_mesh",
    "create_event",
    
    # Guardrails
    "EnhancedEnterpriseGuardrails",
    "GuardrailsConfig",
    "GuardrailsMetrics",
    "TenantContext",
    "TenantManager",
    "get_guardrails",
    "secure_ainvoke",
    
    # Multi-Provider
    "MultiProviderClient",
    "MultiProviderConfig",
    "ProviderConfig",
    "ProviderResponse"
]


# Version
__version__ = "1.0.0"