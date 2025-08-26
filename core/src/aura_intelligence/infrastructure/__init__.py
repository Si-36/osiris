#!/usr/bin/env python3
"""
🏗️ AURA Intelligence Infrastructure Layer
Enterprise-grade infrastructure components for production AI systems

Components:
- Gemini API Client: Real AI integration with LangChain compatibility
- Enterprise Guardrails: Security, cost management, and compliance
- Kafka Event Mesh: High-throughput event streaming
"""

from .gemini_client import (
    GeminiClient,
    GeminiClientManager,
    ChatGemini,
    GeminiConfig,
    GeminiResponse,
    create_gemini_client,
    test_gemini_connection
)

from .guardrails import (
    EnterpriseGuardrails,
    GuardrailsConfig,
    RateLimiter,
    CostTracker,
    SecurityValidator,
    CircuitBreaker,
    get_guardrails,
    secure_ainvoke
)

try:
    from .kafka_event_mesh import (
        KafkaEventMesh,
        KafkaConfig,
        Event,
        EventMesh
    )
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    # Create fallback classes
    class KafkaEventMesh:
        def __init__(self, *args, **kwargs):
            raise ImportError("Kafka dependencies not available")
    
    class KafkaConfig:
        pass
    
    class Event:
        pass

    __all__ = [
            # Gemini AI Client
            'GeminiClient',
            'GeminiClientManager',
            'ChatGemini',
            'GeminiConfig',
            'GeminiResponse',
            'create_gemini_client',
            'test_gemini_connection',
    
            # Enterprise Guardrails
            'EnterpriseGuardrails',
            'GuardrailsConfig',
            'RateLimiter',
            'CostTracker',
            'SecurityValidator',
            'CircuitBreaker',
            'get_guardrails',
            'secure_ainvoke',
    
            # Event Mesh (if available)
            'KafkaEventMesh',
            'KafkaConfig',
            'Event',
            'EventMesh',
            'KAFKA_AVAILABLE'
    ]

    # Module metadata
    __version__ = "2.0.0"
    __author__ = "AURA Intelligence Team"
    __description__ = "Enterprise infrastructure for AI systems"