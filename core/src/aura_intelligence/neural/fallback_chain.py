"""
Fallback Chain and Load Balancer - Zero-Downtime Multi-Provider Orchestration
Based on 2025 research: Active-active routing with proactive health checks

Key Features:
- Circuit breakers with adaptive thresholds
- Proactive health monitoring
- Warm cache maintenance across providers
- Intelligent fallback ordering
- Rate limit aware distribution
- Cost-aware load balancing
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Deque, Set
from enum import Enum
import numpy as np
import structlog

from .provider_adapters import (
    ProviderType, ProviderRequest, ProviderResponse,
    ProviderAdapter, ModelConfig
)
from .model_router import RoutingDecision, RoutingContext
from ..observability import create_tracer, create_meter
from ..resilience import CircuitBreaker, HealthChecker

logger = structlog.get_logger(__name__)
tracer = create_tracer("fallback_chain")
meter = create_meter("fallback_chain")

# Metrics
fallback_executions = meter.create_counter(
    name="aura.fallback.executions",
    description="Fallback executions by provider"
)

circuit_trips = meter.create_counter(
    name="aura.fallback.circuit_trips",
    description="Circuit breaker trips by provider"
)

provider_health_score = meter.create_gauge(
    name="aura.fallback.health_score",
    description="Provider health score (0-1)"
)

load_distribution = meter.create_histogram(
    name="aura.fallback.load_distribution",
    description="Load distribution across providers"
)


class HealthStatus(str, Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_CONNECTIONS = "least_connections"
    LATENCY_BASED = "latency_based"
    COST_OPTIMIZED = "cost_optimized"
    ADAPTIVE = "adaptive"


@dataclass
class ProviderHealth:
    """Health information for a provider"""
    provider: ProviderType
    status: HealthStatus
    health_score: float  # 0-1
    
    # Metrics
    success_rate: float
    avg_latency_ms: float
    error_rate: float
    
    # Recent performance
    recent_successes: Deque[bool] = field(default_factory=lambda: deque(maxlen=100))
    recent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_errors: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    
    # Timestamps
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Rate limiting
    current_rpm: int = 0
    max_rpm: int = 60
    rate_limit_resets_at: Optional[datetime] = None
    
    def update(self, success: bool, latency_ms: float = 0, error: Optional[str] = None):
        """Update health metrics"""
        self.recent_successes.append(success)
        if success:
            self.recent_latencies.append(latency_ms)
            self.last_success = datetime.now(timezone.utc)
        else:
            self.last_failure = datetime.now(timezone.utc)
            if error:
                self.recent_errors.append(error)
                
        # Recalculate metrics
        if len(self.recent_successes) > 0:
            self.success_rate = sum(self.recent_successes) / len(self.recent_successes)
            self.error_rate = 1.0 - self.success_rate
            
        if len(self.recent_latencies) > 0:
            self.avg_latency_ms = np.mean(self.recent_latencies)
            
        # Update health score
        self._update_health_score()
        
    def _update_health_score(self):
        """Calculate overall health score"""
        # Base score from success rate
        score = self.success_rate * 0.5
        
        # Latency factor (normalized)
        if self.avg_latency_ms < 1000:
            latency_score = 1.0
        elif self.avg_latency_ms < 5000:
            latency_score = 1.0 - (self.avg_latency_ms - 1000) / 4000
        else:
            latency_score = 0.0
        score += latency_score * 0.3
        
        # Recency factor
        if self.last_success:
            time_since_success = (datetime.now(timezone.utc) - self.last_success).total_seconds()
            if time_since_success < 60:
                recency_score = 1.0
            elif time_since_success < 300:
                recency_score = 1.0 - (time_since_success - 60) / 240
            else:
                recency_score = 0.0
            score += recency_score * 0.2
            
        self.health_score = min(max(score, 0.0), 1.0)
        
        # Update status
        if self.health_score > 0.8:
            self.status = HealthStatus.HEALTHY
        elif self.health_score > 0.5:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.UNHEALTHY


@dataclass
class FallbackConfig:
    """Configuration for fallback chain"""
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    # Health check settings
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5
    min_healthy_providers: int = 2
    
    # Load balancing
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    prefer_primary: bool = True
    primary_weight: float = 2.0
    
    # Warm cache settings
    warm_cache_size: int = 10
    warm_cache_refresh_interval: int = 300
    
    # Rate limiting
    global_rate_limit: int = 1000  # RPM across all providers
    rate_limit_buffer: float = 0.9  # Use 90% of limit
    
    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000
    exponential_backoff: bool = True


class CircuitBreakerManager:
    """Manages circuit breakers for all providers"""
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.breakers: Dict[ProviderType, CircuitBreaker] = {}
        
    def get_breaker(self, provider: ProviderType) -> CircuitBreaker:
        """Get or create circuit breaker for provider"""
        if provider not in self.breakers:
            self.breakers[provider] = CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                expected_exception=Exception,
                half_open_max_calls=self.config.half_open_max_calls
            )
        return self.breakers[provider]
        
    def can_call(self, provider: ProviderType) -> bool:
        """Check if provider can be called"""
        breaker = self.get_breaker(provider)
        return breaker.state != "open"
        
    def record_success(self, provider: ProviderType):
        """Record successful call"""
        breaker = self.get_breaker(provider)
        breaker.record_success()
        
    def record_failure(self, provider: ProviderType):
        """Record failed call"""
        breaker = self.get_breaker(provider)
        breaker.record_failure()
        
        if breaker.state == "open":
            circuit_trips.add(1, {"provider": provider.value})
            logger.warning(f"Circuit breaker opened for {provider.value}")


class HealthMonitor:
    """Monitors health of all providers"""
    
    def __init__(self, providers: Dict[ProviderType, ProviderAdapter], config: FallbackConfig):
        self.providers = providers
        self.config = config
        self.health_data: Dict[ProviderType, ProviderHealth] = {}
        self._monitoring_task = None
        
        # Initialize health data
        for provider_type in providers:
            self.health_data[provider_type] = ProviderHealth(
                provider=provider_type,
                status=HealthStatus.UNKNOWN,
                health_score=0.5,
                success_rate=1.0,
                avg_latency_ms=0,
                error_rate=0
            )
            
    async def start(self):
        """Start health monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
        
    async def stop(self):
        """Stop health monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Health monitoring stopped")
        
    async def _monitor_loop(self):
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_all_providers()
                self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def _check_all_providers(self):
        """Run health checks on all providers"""
        tasks = []
        for provider_type, adapter in self.providers.items():
            tasks.append(self._check_provider(provider_type, adapter))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update health data
        for i, (provider_type, _) in enumerate(self.providers.items()):
            if isinstance(results[i], Exception):
                self.record_health_check(provider_type, False, error=str(results[i]))
            else:
                success, latency = results[i]
                self.record_health_check(provider_type, success, latency)
                
    async def _check_provider(self, provider_type: ProviderType, adapter: ProviderAdapter) -> Tuple[bool, float]:
        """Check single provider health"""
        start_time = time.time()
        
        try:
            # Use adapter's health check method
            success = await asyncio.wait_for(
                adapter.health_check(),
                timeout=self.config.health_check_timeout
            )
            latency = (time.time() - start_time) * 1000
            return success, latency
            
        except asyncio.TimeoutError:
            return False, self.config.health_check_timeout * 1000
        except Exception:
            return False, 0
            
    def record_health_check(self, provider: ProviderType, success: bool, 
                           latency_ms: float = 0, error: Optional[str] = None):
        """Record health check result"""
        if provider in self.health_data:
            self.health_data[provider].update(success, latency_ms, error)
            self.health_data[provider].last_check = datetime.now(timezone.utc)
            
    def record_request_outcome(self, provider: ProviderType, success: bool,
                             latency_ms: float = 0, error: Optional[str] = None):
        """Record actual request outcome"""
        if provider in self.health_data:
            self.health_data[provider].update(success, latency_ms, error)
            
    def get_healthy_providers(self) -> List[ProviderType]:
        """Get list of healthy providers"""
        healthy = []
        for provider, health in self.health_data.items():
            if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                healthy.append(provider)
        return healthy
        
    def get_provider_health(self, provider: ProviderType) -> Optional[ProviderHealth]:
        """Get health data for specific provider"""
        return self.health_data.get(provider)
        
    def _update_metrics(self):
        """Update health metrics"""
        for provider, health in self.health_data.items():
            provider_health_score.set(health.health_score, {"provider": provider.value})


class LoadBalancer:
    """Intelligent load balancing across providers"""
    
    def __init__(self, config: FallbackConfig, health_monitor: HealthMonitor):
        self.config = config
        self.health_monitor = health_monitor
        
        # Connection tracking
        self.active_connections: Dict[ProviderType, int] = defaultdict(int)
        
        # Round-robin state
        self.rr_index = 0
        
        # Request history for adaptive strategy
        self.request_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        
    def select_provider(self, available_providers: List[ProviderType],
                       context: Optional[RoutingContext] = None,
                       preferred: Optional[ProviderType] = None) -> Optional[ProviderType]:
        """Select provider based on load balancing strategy"""
        
        if not available_providers:
            return None
            
        # Filter by health
        healthy_providers = [
            p for p in available_providers
            if self.health_monitor.get_provider_health(p).status != HealthStatus.UNHEALTHY
        ]
        
        if not healthy_providers:
            # Fall back to any available
            healthy_providers = available_providers
            
        # Apply strategy
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_providers)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random(healthy_providers)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_providers)
        elif self.config.strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._latency_based(healthy_providers)
        elif self.config.strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            return self._cost_optimized(healthy_providers, context)
        elif self.config.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive(healthy_providers, context, preferred)
        else:
            return healthy_providers[0]
            
    def _round_robin(self, providers: List[ProviderType]) -> ProviderType:
        """Simple round-robin selection"""
        selected = providers[self.rr_index % len(providers)]
        self.rr_index += 1
        return selected
        
    def _weighted_random(self, providers: List[ProviderType]) -> ProviderType:
        """Weighted random based on health scores"""
        weights = []
        for provider in providers:
            health = self.health_monitor.get_provider_health(provider)
            weights.append(health.health_score if health else 0.5)
            
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(providers)] * len(providers)
            
        return np.random.choice(providers, p=weights)
        
    def _least_connections(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider with least active connections"""
        min_connections = float('inf')
        selected = providers[0]
        
        for provider in providers:
            connections = self.active_connections[provider]
            if connections < min_connections:
                min_connections = connections
                selected = provider
                
        return selected
        
    def _latency_based(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider with lowest latency"""
        min_latency = float('inf')
        selected = providers[0]
        
        for provider in providers:
            health = self.health_monitor.get_provider_health(provider)
            if health and health.avg_latency_ms < min_latency:
                min_latency = health.avg_latency_ms
                selected = provider
                
        return selected
        
    def _cost_optimized(self, providers: List[ProviderType], 
                       context: Optional[RoutingContext]) -> ProviderType:
        """Select cheapest provider that meets requirements"""
        # This would integrate with model configs for cost data
        # For now, prefer Together > OpenAI > Anthropic
        cost_order = [
            ProviderType.TOGETHER,
            ProviderType.OLLAMA,
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC
        ]
        
        for provider in cost_order:
            if provider in providers:
                return provider
                
        return providers[0]
        
    def _adaptive(self, providers: List[ProviderType],
                 context: Optional[RoutingContext],
                 preferred: Optional[ProviderType]) -> ProviderType:
        """Adaptive selection based on multiple factors"""
        
        # Prefer primary if healthy and configured
        if self.config.prefer_primary and preferred and preferred in providers:
            health = self.health_monitor.get_provider_health(preferred)
            if health and health.health_score > 0.7:
                return preferred
                
        # Score each provider
        scores = {}
        for provider in providers:
            health = self.health_monitor.get_provider_health(provider)
            if not health:
                scores[provider] = 0.5
                continue
                
            # Base score from health
            score = health.health_score * 0.4
            
            # Latency factor
            if health.avg_latency_ms < 1000:
                score += 0.3
            elif health.avg_latency_ms < 3000:
                score += 0.2
            elif health.avg_latency_ms < 5000:
                score += 0.1
                
            # Connection load factor
            connections = self.active_connections[provider]
            if connections < 10:
                score += 0.2
            elif connections < 50:
                score += 0.1
                
            # Rate limit factor
            if health.current_rpm < health.max_rpm * 0.8:
                score += 0.1
                
            scores[provider] = score
            
        # Select highest scoring provider
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def record_connection_start(self, provider: ProviderType):
        """Record connection start"""
        self.active_connections[provider] += 1
        
    def record_connection_end(self, provider: ProviderType):
        """Record connection end"""
        self.active_connections[provider] = max(0, self.active_connections[provider] - 1)
        
    def record_request(self, provider: ProviderType, success: bool, 
                      latency_ms: float, context: Optional[RoutingContext] = None):
        """Record request for adaptive learning"""
        self.request_history.append({
            "provider": provider,
            "success": success,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc),
            "context": context
        })


class FallbackChain:
    """Main fallback chain orchestrator"""
    
    def __init__(self, providers: Dict[ProviderType, ProviderAdapter],
                 config: Optional[FallbackConfig] = None):
        self.providers = providers
        self.config = config or FallbackConfig()
        
        # Initialize components
        self.circuit_manager = CircuitBreakerManager(self.config)
        self.health_monitor = HealthMonitor(providers, self.config)
        self.load_balancer = LoadBalancer(self.config, self.health_monitor)
        
        # Warm cache for faster fallbacks
        self.warm_cache: Dict[str, Any] = {}
        self._warm_cache_task = None
        
    async def start(self):
        """Start fallback chain services"""
        await self.health_monitor.start()
        self._warm_cache_task = asyncio.create_task(self._maintain_warm_cache())
        logger.info("Fallback chain started")
        
    async def stop(self):
        """Stop fallback chain services"""
        await self.health_monitor.stop()
        if self._warm_cache_task:
            self._warm_cache_task.cancel()
        logger.info("Fallback chain stopped")
        
    async def execute(self, request: ProviderRequest,
                     primary_provider: ProviderType,
                     primary_model: str,
                     context: Optional[RoutingContext] = None) -> ProviderResponse:
        """Execute request with automatic fallback"""
        
        with tracer.start_as_current_span("fallback_execute") as span:
            span.set_attribute("primary_provider", primary_provider.value)
            
            # Build provider order
            provider_order = self._build_fallback_order(primary_provider)
            
            # Track attempts
            attempts = 0
            last_error = None
            
            for provider in provider_order:
                attempts += 1
                span.set_attribute(f"attempt_{attempts}_provider", provider.value)
                
                # Check circuit breaker
                if not self.circuit_manager.can_call(provider):
                    logger.debug(f"Circuit breaker open for {provider.value}")
                    continue
                    
                # Check health
                health = self.health_monitor.get_provider_health(provider)
                if health and health.status == HealthStatus.UNHEALTHY:
                    logger.debug(f"Provider {provider.value} is unhealthy")
                    continue
                    
                # Try the provider
                try:
                    # Track connection
                    self.load_balancer.record_connection_start(provider)
                    
                    # Get appropriate model for this provider
                    model = self._get_model_for_provider(provider, primary_model)
                    request.model = model
                    
                    # Execute with timeout
                    start_time = time.time()
                    adapter = self.providers[provider]
                    
                    response = await asyncio.wait_for(
                        adapter.complete(request),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Success!
                    self.circuit_manager.record_success(provider)
                    self.health_monitor.record_request_outcome(provider, True, latency_ms)
                    self.load_balancer.record_request(provider, True, latency_ms, context)
                    
                    # Mark if fallback was used
                    if provider != primary_provider:
                        response.metadata["fallback"] = True
                        response.metadata["original_provider"] = primary_provider.value
                        response.metadata["attempts"] = attempts
                        fallback_executions.add(1, {"provider": provider.value})
                        
                    span.set_attribute("final_provider", provider.value)
                    span.set_attribute("attempts", attempts)
                    
                    return response
                    
                except Exception as e:
                    # Failure
                    last_error = e
                    self.circuit_manager.record_failure(provider)
                    self.health_monitor.record_request_outcome(provider, False, error=str(e))
                    self.load_balancer.record_request(provider, False, 0, context)
                    
                    logger.warning(
                        f"Provider {provider.value} failed",
                        error=str(e),
                        attempt=attempts
                    )
                    
                    # Apply retry delay if not last attempt
                    if attempts < len(provider_order) and self.config.retry_delay_ms > 0:
                        delay = self.config.retry_delay_ms / 1000
                        if self.config.exponential_backoff:
                            delay *= (2 ** (attempts - 1))
                        await asyncio.sleep(delay)
                        
                finally:
                    # Always track connection end
                    self.load_balancer.record_connection_end(provider)
                    
            # All providers failed
            span.set_attribute("all_failed", True)
            raise Exception(f"All providers failed. Last error: {last_error}")
            
    def _build_fallback_order(self, primary: ProviderType) -> List[ProviderType]:
        """Build ordered list of providers for fallback"""
        # Start with primary
        order = [primary]
        
        # Get healthy providers
        healthy = self.health_monitor.get_healthy_providers()
        
        # Add other healthy providers
        for provider in healthy:
            if provider != primary and provider not in order:
                order.append(provider)
                
        # Add remaining providers (even unhealthy as last resort)
        for provider in self.providers:
            if provider not in order:
                order.append(provider)
                
        # Use load balancer to optimize order (except primary)
        if len(order) > 1:
            # Reorder non-primary providers based on load balancing
            remaining = order[1:]
            optimized = []
            
            while remaining:
                selected = self.load_balancer.select_provider(remaining)
                if selected:
                    optimized.append(selected)
                    remaining.remove(selected)
                else:
                    break
                    
            order = [primary] + optimized + remaining
            
        return order
        
    def _get_model_for_provider(self, provider: ProviderType, primary_model: str) -> str:
        """Map model to appropriate model for provider"""
        # Model mapping logic
        model_map = {
            ProviderType.OPENAI: {
                "claude-opus-4.1": "gpt-5",
                "mamba-2-2.8b": "gpt-4o",
                "llama3-70b": "gpt-5"
            },
            ProviderType.ANTHROPIC: {
                "gpt-5": "claude-opus-4.1",
                "gpt-4o": "claude-sonnet-4.1",
                "mamba-2-2.8b": "claude-opus-4.1"
            },
            ProviderType.TOGETHER: {
                "gpt-5": "llama-3.1-70b",
                "claude-opus-4.1": "llama-3.1-70b",
                "gpt-4o": "turbo-mixtral"
            },
            ProviderType.OLLAMA: {
                "gpt-5": "llama3-70b",
                "claude-opus-4.1": "llama3-70b",
                "mamba-2-2.8b": "mixtral-8x7b"
            }
        }
        
        if provider in model_map and primary_model in model_map[provider]:
            return model_map[provider][primary_model]
            
        # Default models per provider
        defaults = {
            ProviderType.OPENAI: "gpt-5",
            ProviderType.ANTHROPIC: "claude-opus-4.1",
            ProviderType.TOGETHER: "llama-3.1-70b",
            ProviderType.OLLAMA: "llama3-70b"
        }
        
        return defaults.get(provider, primary_model)
        
    async def _maintain_warm_cache(self):
        """Maintain warm connections to providers"""
        while True:
            try:
                await asyncio.sleep(self.config.warm_cache_refresh_interval)
                
                # Send lightweight requests to keep connections warm
                for provider_type, adapter in self.providers.items():
                    try:
                        await adapter.health_check()
                    except Exception:
                        pass
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Warm cache maintenance error: {e}")


# Export main classes
__all__ = [
    "HealthStatus",
    "LoadBalancingStrategy",
    "ProviderHealth",
    "FallbackConfig",
    "CircuitBreakerManager",
    "HealthMonitor",
    "LoadBalancer",
    "FallbackChain"
]