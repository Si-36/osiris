"""
AURA Model Router - Intelligent Multi-Provider Routing System
Based on RouterBench research: routing beats single "best" models

Key Features:
- Policy-driven routing based on request characteristics
- Adaptive learning from performance outcomes
- Cost/latency/quality optimization
- Zero-downtime failover with circuit breakers
- Integration with AURA's TDA for risk assessment
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import structlog
from pathlib import Path

from .provider_adapters import (
    ProviderType, ModelCapability, ModelConfig,
    ProviderRequest, ProviderResponse, ProviderFactory
)
try:
    from ..observability import create_tracer, create_meter
    tracer = create_tracer("model_router")
    meter = create_meter("model_router")
except ImportError:
    # Mock for testing
    class MockTracer:
        def start_as_current_span(self, name):
            from contextlib import contextmanager
            @contextmanager
            def mock_span():
                class MockSpan:
                    def set_attribute(self, key, value): pass
                yield MockSpan()
            return mock_span()
    tracer = MockTracer()
    
    class MockMeter:
        def create_counter(self, **kwargs):
            class MockCounter:
                def add(self, *args, **kwargs): pass
            return MockCounter()
        def create_histogram(self, **kwargs):
            class MockHistogram:
                def record(self, *args, **kwargs): pass
            return MockHistogram()
    meter = MockMeter()
# Optional integrations
try:
    from ..tda import TopologicalAnalyzer
except ImportError:
    TopologicalAnalyzer = None
    
try:
    from ..memory import MemoryManager
except ImportError:
    MemoryManager = None

logger = structlog.get_logger(__name__)

# Metrics
routing_decisions = meter.create_counter(
    name="aura.router.decisions",
    description="Routing decisions by provider and reason"
)

routing_latency = meter.create_histogram(
    name="aura.router.latency",
    description="Time to make routing decision in ms",
    unit="ms"
)

routing_quality = meter.create_histogram(
    name="aura.router.quality_score",
    description="Estimated quality score of routing decision"
)


class RoutingReason(str, Enum):
    """Reasons for routing decisions"""
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LONG_CONTEXT = "long_context"
    PRIVACY_REQUIRED = "privacy_required"
    TOOLS_REQUIRED = "tools_required"
    BACKGROUND_REQUIRED = "background_required"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CACHE_HIT = "cache_hit"
    FALLBACK = "fallback"


@dataclass
class RoutingPolicy:
    """Policy configuration for routing decisions"""
    # Optimization weights (sum to 1.0)
    quality_weight: float = 0.4
    cost_weight: float = 0.3
    latency_weight: float = 0.3
    
    # Hard constraints
    max_cost_per_request: Optional[float] = None
    max_latency_ms: Optional[int] = None
    min_quality_score: Optional[float] = None
    
    # Feature requirements
    require_privacy: bool = False
    require_tools: bool = False
    require_background: bool = False
    allowed_providers: Optional[List[ProviderType]] = None
    blocked_providers: Optional[List[ProviderType]] = None
    
    # Context limits
    max_context_length: int = 128000
    
    # Advanced settings
    enable_semantic_cache: bool = True
    enable_fallback: bool = True
    fallback_timeout_ms: int = 30000
    

@dataclass
class RoutingContext:
    """Context for routing decision"""
    request: ProviderRequest
    policy: RoutingPolicy
    topology_score: float = 0.0  # From TDA analysis
    urgency_score: float = 0.5  # 0=can wait, 1=urgent
    complexity_score: float = 0.5  # 0=simple, 1=complex
    privacy_score: float = 0.0  # 0=public, 1=highly sensitive
    
    # Computed features
    estimated_tokens: int = 0
    requires_reasoning: bool = False
    requires_coding: bool = False
    requires_vision: bool = False
    

@dataclass
class RoutingDecision:
    """Routing decision with metadata"""
    provider: ProviderType
    model: str
    reason: RoutingReason
    confidence: float
    estimated_cost: float
    estimated_latency_ms: float
    estimated_quality: float
    fallback_chain: List[Tuple[ProviderType, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRoutingEngine:
    """Learns optimal routing patterns from outcomes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Simple neural network for routing (inspired by LNN)
        # Input: [context_len, complexity, urgency, has_tools, is_private, ...]
        # Output: [provider_scores]
        self.feature_dim = 10
        self.num_providers = len(ProviderType)
        
        # Initialize with small random weights
        self.weights = np.random.randn(self.feature_dim, self.num_providers) * 0.1
        self.bias = np.zeros(self.num_providers)
        
        # Learning rate
        self.learning_rate = config.get("learning_rate", 0.01)
        
        # Performance history for learning
        self.history: List[Dict[str, Any]] = []
        self.max_history = 10000
        
    def extract_features(self, context: RoutingContext) -> np.ndarray:
        """Extract feature vector from routing context"""
        features = np.zeros(self.feature_dim)
        
        # Normalize features to [0, 1]
        features[0] = min(context.estimated_tokens / 100000, 1.0)  # Context length
        features[1] = context.complexity_score
        features[2] = context.urgency_score
        features[3] = float(context.request.tools is not None)
        features[4] = context.privacy_score
        features[5] = float(context.request.background)
        features[6] = context.topology_score
        features[7] = float(context.requires_reasoning)
        features[8] = float(context.requires_coding)
        features[9] = float(context.requires_vision)
        
        return features
        
    def predict_scores(self, context: RoutingContext) -> Dict[ProviderType, float]:
        """Predict provider scores based on context"""
        features = self.extract_features(context)
        
        # Simple linear model: scores = features @ weights + bias
        raw_scores = features @ self.weights + self.bias
        
        # Apply softmax for probabilities
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        # Map to provider types
        provider_types = list(ProviderType)
        scores = {
            provider_types[i]: float(probabilities[i])
            for i in range(len(provider_types))
        }
        
        return scores
        
    def update(self, context: RoutingContext, provider: ProviderType, outcome: Dict[str, Any]):
        """Update routing model based on outcome"""
        # Calculate reward based on outcome
        reward = self._calculate_reward(outcome, context.policy)
        
        # Store in history
        self.history.append({
            "context": context,
            "provider": provider,
            "outcome": outcome,
            "reward": reward,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Trim history if too large
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        # Update weights using simple gradient
        features = self.extract_features(context)
        provider_idx = list(ProviderType).index(provider)
        
        # Gradient update (simplified reinforcement learning)
        prediction = self.predict_scores(context)
        target = np.zeros(self.num_providers)
        target[provider_idx] = reward
        
        # Update weights
        for i, p_type in enumerate(ProviderType):
            error = target[i] - prediction[p_type]
            self.weights[:, i] += self.learning_rate * error * features
            self.bias[i] += self.learning_rate * error
            
    def _calculate_reward(self, outcome: Dict[str, Any], policy: RoutingPolicy) -> float:
        """Calculate reward based on outcome and policy"""
        # Normalize metrics
        cost_score = 1.0 - min(outcome.get("cost_usd", 0) / 10.0, 1.0)  # Lower is better
        latency_score = 1.0 - min(outcome.get("latency_ms", 0) / 10000.0, 1.0)  # Lower is better
        quality_score = outcome.get("quality_score", 0.5)  # Higher is better
        success = float(not outcome.get("error", False))
        
        # Weighted combination based on policy
        reward = (
            policy.quality_weight * quality_score * success +
            policy.cost_weight * cost_score +
            policy.latency_weight * latency_score
        )
        
        return reward


class AURAModelRouter:
    """Main router that orchestrates provider selection and execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize providers
        self.providers: Dict[ProviderType, Any] = {}
        self._init_providers()
        
        # Adaptive routing engine
        self.routing_engine = AdaptiveRoutingEngine(config.get("routing", {}))
        
        # TDA for topology analysis (optional)
        self.tda_analyzer = TopologicalAnalyzer() if TopologicalAnalyzer else None
        
        # Provider health tracking
        self.provider_health: Dict[ProviderType, float] = {
            p: 1.0 for p in ProviderType
        }
        
        # Cache for semantic matching (will be connected to cache_manager.py)
        self.cache_enabled = config.get("enable_cache", True)
        
    def _init_providers(self):
        """Initialize provider adapters"""
        provider_configs = self.config.get("providers", {})
        
        for provider_type in ProviderType:
            if provider_type.value in provider_configs:
                config = provider_configs[provider_type.value]
                if config.get("enabled", True):
                    api_key = config.get("api_key", "")
                    self.providers[provider_type] = ProviderFactory.create(
                        provider_type, api_key, config
                    )
                    
    async def route_request(self, request: ProviderRequest, policy: Optional[RoutingPolicy] = None) -> RoutingDecision:
        """Make routing decision for request"""
        with tracer.start_as_current_span("route_request") as span:
            start_time = time.time()
            
            # Use default policy if not provided
            if policy is None:
                policy = RoutingPolicy()
                
            # Build routing context
            context = await self._build_context(request, policy)
            span.set_attribute("context.complexity", context.complexity_score)
            span.set_attribute("context.estimated_tokens", context.estimated_tokens)
            
            # Get provider scores from adaptive engine
            provider_scores = self.routing_engine.predict_scores(context)
            
            # Apply constraints and health checks
            valid_providers = await self._filter_providers(context, provider_scores)
            
            if not valid_providers:
                raise ValueError("No valid providers available for request")
                
            # Select best provider
            best_provider, best_model, reason = self._select_best_provider(
                valid_providers, context, provider_scores
            )
            
            # Build fallback chain
            fallback_chain = self._build_fallback_chain(
                valid_providers, best_provider, context
            )
            
            # Estimate metrics
            estimated_cost = self._estimate_cost(best_provider, best_model, context)
            estimated_latency = self._estimate_latency(best_provider, best_model, context)
            estimated_quality = self._estimate_quality(best_provider, best_model, context)
            
            # Record metrics
            routing_latency.record(
                (time.time() - start_time) * 1000,
                {"provider": best_provider.value, "reason": reason.value}
            )
            routing_decisions.add(
                1,
                {"provider": best_provider.value, "reason": reason.value}
            )
            routing_quality.record(estimated_quality)
            
            decision = RoutingDecision(
                provider=best_provider,
                model=best_model,
                reason=reason,
                confidence=provider_scores[best_provider],
                estimated_cost=estimated_cost,
                estimated_latency_ms=estimated_latency,
                estimated_quality=estimated_quality,
                fallback_chain=fallback_chain,
                metadata={
                    "context": context,
                    "scores": provider_scores
                }
            )
            
            span.set_attribute("decision.provider", decision.provider.value)
            span.set_attribute("decision.model", decision.model)
            span.set_attribute("decision.reason", decision.reason.value)
            
            return decision
            
    async def execute_request(self, request: ProviderRequest, decision: RoutingDecision) -> ProviderResponse:
        """Execute request with routing decision and fallback"""
        with tracer.start_as_current_span("execute_request") as span:
            span.set_attribute("provider", decision.provider.value)
            span.set_attribute("model", decision.model)
            
            # Update request with selected model
            request.model = decision.model
            
            # Try primary provider
            try:
                provider = self.providers[decision.provider]
                response = await provider.complete(request)
                
                # Update health on success
                self.provider_health[decision.provider] = min(
                    self.provider_health[decision.provider] * 1.1, 1.0
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Primary provider {decision.provider} failed", error=str(e))
                
                # Update health on failure
                self.provider_health[decision.provider] *= 0.9
                
                # Try fallback chain
                if decision.fallback_chain:
                    for fallback_provider, fallback_model in decision.fallback_chain:
                        try:
                            logger.info(f"Trying fallback {fallback_provider}")
                            request.model = fallback_model
                            provider = self.providers[fallback_provider]
                            response = await provider.complete(request)
                            
                            # Update health
                            self.provider_health[fallback_provider] = min(
                                self.provider_health[fallback_provider] * 1.1, 1.0
                            )
                            
                            # Mark as fallback
                            response.metadata["fallback"] = True
                            response.metadata["original_provider"] = decision.provider.value
                            
                            return response
                            
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback provider {fallback_provider} failed",
                                error=str(fallback_error)
                            )
                            self.provider_health[fallback_provider] *= 0.9
                            continue
                            
                # All providers failed
                raise Exception("All providers failed")
                
    async def track_outcome(self, request: ProviderRequest, decision: RoutingDecision, 
                          response: ProviderResponse, quality_score: Optional[float] = None):
        """Track routing outcome for learning"""
        outcome = {
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
            "quality_score": quality_score or self._estimate_quality(
                decision.provider, decision.model, decision.metadata["context"]
            ),
            "error": False,
            "cache_hit": response.cache_hit
        }
        
        # Update routing engine
        self.routing_engine.update(
            decision.metadata["context"],
            decision.provider,
            outcome
        )
        
    async def _build_context(self, request: ProviderRequest, policy: RoutingPolicy) -> RoutingContext:
        """Build routing context from request"""
        # Estimate tokens
        estimated_tokens = len(request.prompt) // 4  # Rough estimate
        
        # Analyze with TDA
        topology_score = 0.0
        if self.tda_analyzer:
            try:
                topology_analysis = await self.tda_analyzer.analyze_text(request.prompt)
                topology_score = topology_analysis.get("complexity", 0.0)
            except Exception:
                pass
                
        # Detect requirements
        prompt_lower = request.prompt.lower()
        requires_reasoning = any(word in prompt_lower for word in ["explain", "why", "reason", "analyze"])
        requires_coding = any(word in prompt_lower for word in ["code", "function", "implement", "debug"])
        requires_vision = False  # Would check for image inputs
        
        # Estimate complexity
        complexity_score = min(
            (estimated_tokens / 10000) + 
            (0.3 if requires_reasoning else 0) +
            (0.2 if requires_coding else 0),
            1.0
        )
        
        # Privacy score (would integrate with data classification)
        privacy_score = 1.0 if policy.require_privacy else 0.0
        
        context = RoutingContext(
            request=request,
            policy=policy,
            topology_score=topology_score,
            urgency_score=0.5,  # Default, could be set by caller
            complexity_score=complexity_score,
            privacy_score=privacy_score,
            estimated_tokens=estimated_tokens,
            requires_reasoning=requires_reasoning,
            requires_coding=requires_coding,
            requires_vision=requires_vision
        )
        
        return context
        
    async def _filter_providers(self, context: RoutingContext, 
                              scores: Dict[ProviderType, float]) -> Dict[ProviderType, List[str]]:
        """Filter providers based on constraints and availability"""
        valid_providers = {}
        
        for provider_type, provider in self.providers.items():
            # Check if provider is allowed
            if context.policy.allowed_providers:
                if provider_type not in context.policy.allowed_providers:
                    continue
                    
            if context.policy.blocked_providers:
                if provider_type in context.policy.blocked_providers:
                    continue
                    
            # Check health
            if self.provider_health[provider_type] < 0.1:
                continue
                
            # Check specific requirements
            if context.policy.require_privacy and provider_type != ProviderType.OLLAMA:
                continue
                
            # Get available models
            available_models = []
            
            if provider_type == ProviderType.OPENAI:
                models = ["gpt-5", "gpt-4o"]
                if context.request.tools or context.request.background:
                    models = ["gpt-5"]  # Only GPT-5 supports full Responses API
                    
            elif provider_type == ProviderType.ANTHROPIC:
                models = ["claude-opus-4.1", "claude-sonnet-4.1"]
                if context.estimated_tokens > 100000:
                    models = ["claude-opus-4.1"]  # Better for long context
                    
            elif provider_type == ProviderType.TOGETHER:
                if context.estimated_tokens > 128000:
                    models = ["mamba-2-2.8b"]  # Ultra long context
                elif context.urgency_score > 0.8:
                    models = ["turbo-mixtral"]  # Fast inference
                else:
                    models = ["llama-3.1-70b", "mamba-2-2.8b"]
                    
            elif provider_type == ProviderType.OLLAMA:
                models = ["llama3-70b", "mixtral-8x7b"]
                
            if models:
                valid_providers[provider_type] = models
                
        return valid_providers
        
    def _select_best_provider(self, valid_providers: Dict[ProviderType, List[str]], 
                            context: RoutingContext,
                            scores: Dict[ProviderType, float]) -> Tuple[ProviderType, str, RoutingReason]:
        """Select best provider and model based on scores and context"""
        
        # Special case routing
        if context.request.tools or context.request.background:
            if ProviderType.OPENAI in valid_providers:
                return ProviderType.OPENAI, "gpt-5", RoutingReason.TOOLS_REQUIRED
                
        if context.estimated_tokens > 128000:
            if ProviderType.TOGETHER in valid_providers:
                return ProviderType.TOGETHER, "mamba-2-2.8b", RoutingReason.LONG_CONTEXT
                
        if context.policy.require_privacy:
            if ProviderType.OLLAMA in valid_providers:
                models = valid_providers[ProviderType.OLLAMA]
                return ProviderType.OLLAMA, models[0], RoutingReason.PRIVACY_REQUIRED
                
        # Score-based selection
        best_score = -1
        best_provider = None
        best_model = None
        reason = RoutingReason.QUALITY_OPTIMIZED
        
        for provider_type, models in valid_providers.items():
            provider_score = scores[provider_type] * self.provider_health[provider_type]
            
            # Adjust score based on context
            if context.urgency_score > 0.8:
                # Prefer fast providers
                if provider_type in [ProviderType.TOGETHER, ProviderType.OPENAI]:
                    provider_score *= 1.2
                    reason = RoutingReason.LATENCY_OPTIMIZED
                    
            elif context.policy.cost_weight > 0.5:
                # Prefer cheap providers
                if provider_type == ProviderType.TOGETHER:
                    provider_score *= 1.3
                    reason = RoutingReason.COST_OPTIMIZED
                    
            if provider_score > best_score:
                best_score = provider_score
                best_provider = provider_type
                best_model = models[0]  # Pick first available model
                
        return best_provider, best_model, reason
        
    def _build_fallback_chain(self, valid_providers: Dict[ProviderType, List[str]], 
                            primary: ProviderType, context: RoutingContext) -> List[Tuple[ProviderType, str]]:
        """Build fallback chain for reliability"""
        fallback_chain = []
        
        # Sort providers by reliability and capability match
        provider_priority = {
            ProviderType.OPENAI: 4,
            ProviderType.ANTHROPIC: 3,
            ProviderType.TOGETHER: 2,
            ProviderType.OLLAMA: 1
        }
        
        sorted_providers = sorted(
            valid_providers.items(),
            key=lambda x: provider_priority.get(x[0], 0),
            reverse=True
        )
        
        for provider_type, models in sorted_providers:
            if provider_type != primary:
                fallback_chain.append((provider_type, models[0]))
                
        return fallback_chain[:3]  # Keep top 3 fallbacks
        
    def _estimate_cost(self, provider: ProviderType, model: str, context: RoutingContext) -> float:
        """Estimate cost for request"""
        # This would use actual pricing data
        base_costs = {
            ProviderType.OPENAI: 0.01,
            ProviderType.ANTHROPIC: 0.015,
            ProviderType.TOGETHER: 0.001,
            ProviderType.OLLAMA: 0.0
        }
        
        base_cost = base_costs.get(provider, 0.01)
        token_multiplier = context.estimated_tokens / 1000
        
        return base_cost * token_multiplier
        
    def _estimate_latency(self, provider: ProviderType, model: str, context: RoutingContext) -> float:
        """Estimate latency for request"""
        # Base latencies
        base_latencies = {
            ProviderType.OPENAI: 2000,
            ProviderType.ANTHROPIC: 3000,
            ProviderType.TOGETHER: 1000,
            ProviderType.OLLAMA: 5000
        }
        
        base_latency = base_latencies.get(provider, 2000)
        
        # Adjust for context length
        if context.estimated_tokens > 50000:
            base_latency *= 2
            
        return base_latency
        
    def _estimate_quality(self, provider: ProviderType, model: str, context: RoutingContext) -> float:
        """Estimate quality score for provider/model on this request"""
        # Base quality scores
        quality_scores = {
            ProviderType.OPENAI: 0.9,
            ProviderType.ANTHROPIC: 0.85,
            ProviderType.TOGETHER: 0.7,
            ProviderType.OLLAMA: 0.75
        }
        
        score = quality_scores.get(provider, 0.7)
        
        # Adjust based on task match
        if context.requires_reasoning and provider == ProviderType.ANTHROPIC:
            score += 0.05
        elif context.requires_coding and provider == ProviderType.ANTHROPIC:
            score += 0.1
        elif context.estimated_tokens > 100000 and provider == ProviderType.TOGETHER:
            score += 0.1
            
        return min(score, 1.0)


# Export main classes
__all__ = [
    "RoutingReason",
    "RoutingPolicy",
    "RoutingContext",
    "RoutingDecision",
    "AdaptiveRoutingEngine",
    "AURAModelRouter"
]