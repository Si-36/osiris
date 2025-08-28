"""
Neural Routing Package for AURA Intelligence - Production 2025

This package implements intelligent multi-provider model routing:
- Provider adapters (OpenAI, Anthropic, Together, Ollama)
- Adaptive routing based on LNN-inspired learning
- Smart context management for different models
- Performance tracking and feedback loops
- Cost optimization and failover chains

Based on RouterBench research: routing beats single "best" models
"""

# Production routing components
try:
    from .provider_adapters import (
        ProviderType,
        ModelCapability,
        ModelConfig,
        ProviderRequest,
        ProviderResponse,
        ProviderAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        TogetherAdapter,
        OllamaAdapter,
        ProviderFactory
    )
except ImportError:
    # Use simplified version without external dependencies
    from .provider_adapters_simple import (
        ProviderType,
        ModelCapability,
        ModelConfig,
        ProviderRequest,
        ProviderResponse,
        ProviderAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        TogetherAdapter,
        OllamaAdapter,
        ProviderFactory
    )

from .model_router import (
    RoutingReason,
    RoutingPolicy,
    RoutingContext,
    RoutingDecision,
    AdaptiveRoutingEngine,
    AURAModelRouter
)

from .adaptive_routing_engine import (
    RoutingState,
    LiquidRoutingCell,
    AdaptiveLNNRouter,
    RouterBenchEvaluator
)

from .context_manager import (
    ContextWindow,
    ContextStrategy,
    StandardContextStrategy,
    LongContextStrategy,
    ChunkedContextStrategy,
    ContextManager
)

from .performance_tracker import (
    PerformanceMetric,
    PerformanceEvent,
    ProviderProfile,
    ModelPerformanceTracker
)

# Legacy LNN imports for compatibility (if needed)
try:
    from ..lnn import (
        LiquidNeuralNetwork,
        LiquidCell,
        LiquidLayer,
        LNNConfig
    )
    _lnn_available = True
except ImportError:
    _lnn_available = False
    LiquidNeuralNetwork = None
    LiquidCell = None
    LiquidLayer = None
    LNNConfig = None

__all__ = [
    # Provider system
    'ProviderType',
    'ModelCapability',
    'ModelConfig',
    'ProviderRequest',
    'ProviderResponse',
    'ProviderAdapter',
    'OpenAIAdapter',
    'AnthropicAdapter',
    'TogetherAdapter',
    'OllamaAdapter',
    'ProviderFactory',
    
    # Routing system
    'RoutingReason',
    'RoutingPolicy',
    'RoutingContext',
    'RoutingDecision',
    'AdaptiveRoutingEngine',
    'AURAModelRouter',
    
    # Adaptive learning
    'RoutingState',
    'LiquidRoutingCell',
    'AdaptiveLNNRouter',
    'RouterBenchEvaluator',
    
    # Context management
    'ContextWindow',
    'ContextStrategy',
    'StandardContextStrategy',
    'LongContextStrategy',
    'ChunkedContextStrategy',
    'ContextManager',
    
    # Performance tracking
    'PerformanceMetric',
    'PerformanceEvent',
    'ProviderProfile',
    'ModelPerformanceTracker',
    
    # Legacy support
    'LiquidNeuralNetwork',
    '_lnn_available'
]