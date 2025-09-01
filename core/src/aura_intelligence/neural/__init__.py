"""
Neural Routing Package for AURA Intelligence - Production 2025

This package implements COMPLETE intelligent multi-provider model routing:
- Provider adapters with Responses API, background mode, long-context
- Adaptive routing based on LNN-inspired learning  
- Two-layer semantic caching (exact + vector similarity)
- Zero-downtime fallback chains with circuit breakers
- Cost optimization with multi-objective scoring
- Advanced load balancing with priority queues
- Performance tracking and RouterBench evaluation
- Smart context management across providers
- Per-tenant policies and budget enforcement
- Proactive health monitoring and auto-scaling

Based on 2025 research:
- RouterBench: routing beats single "best" models by 30-40%
- Semantic caching provides 20%+ hit rate on paraphrases
- Active-active patterns ensure 99.9%+ uptime
- Multi-objective optimization balances quality/cost/latency
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

# Cache management
from .cache_manager import (
    CacheStrategy,
    CacheConfig,
    CacheEntry,
    CacheStats,
    ExactCache,
    SemanticCache,
    CacheManager
)

# Fallback and reliability
from .fallback_chain import (
    HealthStatus,
    LoadBalancingStrategy,
    ProviderHealth,
    FallbackConfig,
    CircuitBreakerManager,
    HealthMonitor,
    LoadBalancer,
    FallbackChain
)

# Cost optimization
from .cost_optimizer import (
    CostTier,
    OptimizationObjective,
    TenantPolicy,
    CostEstimate,
    TenantUsage,
    ModelCostDatabase,
    CostOptimizer
)

# Advanced load balancing
from .load_balancer import (
    QueuePriority,
    ProviderState,
    QueuedRequest,
    ProviderPool,
    LoadBalancerConfig,
    PriorityQueueManager,
    ElasticPoolManager,
    AdvancedLoadBalancer
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
    
    # Cache management
    'CacheStrategy',
    'CacheConfig',
    'CacheEntry',
    'CacheStats',
    'ExactCache',
    'SemanticCache',
    'CacheManager',
    
    # Fallback and reliability
    'HealthStatus',
    'LoadBalancingStrategy',
    'ProviderHealth',
    'FallbackConfig',
    'CircuitBreakerManager',
    'HealthMonitor',
    'LoadBalancer',
    'FallbackChain',
    
    # Cost optimization
    'CostTier',
    'OptimizationObjective',
    'TenantPolicy',
    'CostEstimate',
    'TenantUsage',
    'ModelCostDatabase',
    'CostOptimizer',
    
    # Advanced load balancing
    'QueuePriority',
    'ProviderState',
    'QueuedRequest',
    'ProviderPool',
    'LoadBalancerConfig',
    'PriorityQueueManager',
    'ElasticPoolManager',
    'AdvancedLoadBalancer',
    
    # Legacy support
    'LiquidNeuralNetwork',
    '_lnn_available'
]