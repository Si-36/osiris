"""
AURA Component Adapters
======================

Advanced adapters for registry integration.
"""

from .base_adapter import (
    BaseAdapter,
    ComponentMetadata,
    HealthStatus,
    HealthMetrics,
    CircuitBreaker,
    CircuitBreakerState
)

try:
    from .neural_adapter import NeuralRouterAdapter
except ImportError:
    # Allow testing without full neural dependencies
    NeuralRouterAdapter = None

__all__ = [
    'BaseAdapter',
    'ComponentMetadata', 
    'HealthStatus',
    'HealthMetrics',
    'CircuitBreaker',
    'CircuitBreakerState',
    'NeuralRouterAdapter',
]