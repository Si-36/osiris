"""
Operational Orchestration Layer
===============================
Real-time event routing, circuit breakers, and task scheduling
Based on 2025 research: NATS JetStream, GABFT gossip, cognitive load management
"""

from .event_router import (
    EventRouter,
    CognitivePriorityEngine,
    Event,
    EventPriority,
    ProcessingPool
)
from .circuit_breaker import (
    CognitiveCircuitBreaker,
    CircuitState,
    BreakerId,
    AdaptiveTrustScorer
)
from .task_scheduler import (
    AdaptiveTaskScheduler,
    Task,
    TaskPriority,
    CognitiveLoadManager
)
from .gossip_router import (
    GossipRouter,
    GossipMessage,
    ConsensusState
)
from .latency_scheduler import (
    LatencyAwareScheduler,
    LatencyEstimator,
    SLAConfig
)

__all__ = [
    # Event routing
    'EventRouter',
    'CognitivePriorityEngine',
    'Event',
    'EventPriority',
    'ProcessingPool',
    
    # Circuit breakers
    'CognitiveCircuitBreaker',
    'CircuitState',
    'BreakerId',
    'AdaptiveTrustScorer',
    
    # Task scheduling
    'AdaptiveTaskScheduler',
    'Task',
    'TaskPriority',
    'CognitiveLoadManager',
    
    # Gossip routing
    'GossipRouter',
    'GossipMessage',
    'ConsensusState',
    
    # Latency scheduling
    'LatencyAwareScheduler',
    'LatencyEstimator',
    'SLAConfig'
]