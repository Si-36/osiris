"""
Performance Tracker - Track Model Performance and Feed Back to Routing
Transforms memory_hooks.py into production performance tracking

Key Features:
- Track model performance per request type
- Build performance profiles for each provider/model
- Feed performance data back to adaptive routing
- Integration with AURA's persistence layer
- Cost and latency tracking
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Deque
from enum import Enum
import numpy as np
import structlog

from .provider_adapters import ProviderType, ProviderRequest, ProviderResponse
from .model_router import RoutingDecision, RoutingContext
# Use the causal persistence manager
try:
    from ..persistence.causal_state_manager import CausalPersistenceManager as PersistenceManager
except ImportError:
    # Fallback if persistence not available
    PersistenceManager = None
from ..observability import create_tracer, create_meter

logger = structlog.get_logger(__name__)
tracer = create_tracer("performance_tracker")
meter = create_meter("performance_tracker")

# Metrics
performance_events = meter.create_counter(
    name="aura.performance.events",
    description="Performance tracking events by type"
)

performance_score = meter.create_histogram(
    name="aura.performance.score",
    description="Performance scores by provider"
)

cost_savings = meter.create_counter(
    name="aura.performance.cost_savings_usd",
    description="Cumulative cost savings in USD"
)


class PerformanceMetric(str, Enum):
    """Types of performance metrics tracked"""
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    ERROR_RATE = "error_rate"
    TOKEN_EFFICIENCY = "token_efficiency"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceEvent:
    """Single performance event for tracking"""
    timestamp: datetime
    request_id: str
    provider: ProviderType
    model: str
    
    # Request characteristics
    request_type: str  # reasoning, coding, summarization, etc.
    context_length: int
    has_tools: bool
    is_streaming: bool
    
    # Performance metrics
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    
    # Outcome
    success: bool
    error_type: Optional[str] = None
    quality_score: Optional[float] = None  # 0-1, if available
    cache_hit: bool = False
    fallback_used: bool = False
    
    # Context
    routing_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ProviderProfile:
    """Performance profile for a provider/model combination"""
    provider: ProviderType
    model: str
    
    # Aggregate metrics (sliding window)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_cost_per_1k_tokens: float = 0.0
    error_rate: float = 0.0
    avg_quality_score: float = 0.5
    
    # Per request type metrics
    request_type_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recent performance (for trend detection)
    recent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_errors: Deque[bool] = field(default_factory=lambda: deque(maxlen=100))
    recent_quality_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    
    # Capabilities observed
    max_successful_context: int = 0
    supports_tools_observed: bool = False
    supports_streaming_observed: bool = False
    
    # Update tracking
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_requests: int = 0
    
    def update(self, event: PerformanceEvent):
        """Update profile with new performance event"""
        self.total_requests += 1
        self.last_updated = event.timestamp
        
        # Update latency
        self.recent_latencies.append(event.latency_ms)
        self.avg_latency_ms = np.mean(self.recent_latencies)
        if len(self.recent_latencies) >= 20:
            self.p95_latency_ms = np.percentile(self.recent_latencies, 95)
            
        # Update error rate
        self.recent_errors.append(not event.success)
        self.error_rate = sum(self.recent_errors) / len(self.recent_errors)
        
        # Update quality
        if event.quality_score is not None:
            self.recent_quality_scores.append(event.quality_score)
            self.avg_quality_score = np.mean(self.recent_quality_scores)
            
        # Update cost
        total_tokens = event.input_tokens + event.output_tokens
        if total_tokens > 0:
            cost_per_token = event.cost_usd / total_tokens
            self.avg_cost_per_1k_tokens = cost_per_token * 1000
            
        # Update capabilities
        if event.success and event.context_length > self.max_successful_context:
            self.max_successful_context = event.context_length
        if event.has_tools and event.success:
            self.supports_tools_observed = True
        if event.is_streaming and event.success:
            self.supports_streaming_observed = True
            
        # Update per-request-type metrics
        if event.request_type not in self.request_type_performance:
            self.request_type_performance[event.request_type] = {
                "avg_latency": 0.0,
                "success_rate": 0.0,
                "avg_quality": 0.5,
                "count": 0
            }
            
        rt_perf = self.request_type_performance[event.request_type]
        rt_perf["count"] += 1
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        rt_perf["avg_latency"] = (1 - alpha) * rt_perf["avg_latency"] + alpha * event.latency_ms
        rt_perf["success_rate"] = (1 - alpha) * rt_perf["success_rate"] + alpha * float(event.success)
        if event.quality_score is not None:
            rt_perf["avg_quality"] = (1 - alpha) * rt_perf["avg_quality"] + alpha * event.quality_score


class ModelPerformanceTracker:
    """Main performance tracking system"""
    
    def __init__(self, persistence_manager: Optional[PersistenceManager] = None):
        self.persistence = persistence_manager
        
        # Provider profiles
        self.profiles: Dict[Tuple[ProviderType, str], ProviderProfile] = {}
        
        # Recent events for quick access
        self.recent_events: Deque[PerformanceEvent] = deque(maxlen=1000)
        
        # Cost tracking
        self.baseline_costs: Dict[str, float] = {}  # request_type -> avg_cost
        self.total_savings = 0.0
        
        # Request type detection patterns
        self.request_patterns = {
            "reasoning": ["explain", "why", "reason", "analyze", "understand"],
            "coding": ["code", "function", "implement", "debug", "program"],
            "summarization": ["summarize", "summary", "brief", "overview"],
            "translation": ["translate", "convert", "language"],
            "creative": ["story", "poem", "creative", "imagine"],
            "factual": ["what is", "define", "fact", "information"],
            "conversation": ["chat", "talk", "discuss", "conversation"]
        }
        
    async def track_request(self,
                          request: ProviderRequest,
                          decision: RoutingDecision,
                          response: ProviderResponse,
                          context: Optional[RoutingContext] = None,
                          quality_score: Optional[float] = None) -> PerformanceEvent:
        """Track performance of a model request"""
        
        with tracer.start_as_current_span("track_performance") as span:
            # Detect request type
            request_type = self._detect_request_type(request.prompt)
            
            # Create performance event
            event = PerformanceEvent(
                timestamp=datetime.now(timezone.utc),
                request_id=request.metadata.get("request_id", ""),
                provider=response.provider,
                model=response.model,
                request_type=request_type,
                context_length=len(request.prompt),
                has_tools=request.tools is not None,
                is_streaming=request.stream,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
                input_tokens=response.usage.get("input_tokens", 0),
                output_tokens=response.usage.get("output_tokens", 0),
                success=True,  # If we got a response
                quality_score=quality_score,
                cache_hit=response.cache_hit,
                fallback_used=response.metadata.get("fallback", False),
                routing_reason=decision.reason.value if decision else None,
                metadata={
                    "confidence": decision.confidence if decision else None,
                    "estimated_cost": decision.estimated_cost if decision else None,
                    "estimated_latency": decision.estimated_latency_ms if decision else None
                }
            )
            
            # Update provider profile
            profile_key = (response.provider, response.model)
            if profile_key not in self.profiles:
                self.profiles[profile_key] = ProviderProfile(
                    provider=response.provider,
                    model=response.model
                )
            
            self.profiles[profile_key].update(event)
            
            # Track cost savings
            if not response.cache_hit:
                await self._track_cost_savings(event)
            
            # Store event
            self.recent_events.append(event)
            
            # Persist if configured
            if self.persistence:
                await self._persist_event(event)
                
            # Update metrics
            performance_events.add(1, {
                "type": "success" if event.success else "error",
                "provider": event.provider.value,
                "request_type": event.request_type
            })
            
            if event.quality_score is not None:
                performance_score.record(event.quality_score, {
                    "provider": event.provider.value,
                    "model": event.model
                })
                
            span.set_attribute("request_type", request_type)
            span.set_attribute("quality_score", quality_score or -1)
            
            return event
            
    async def track_error(self,
                        request: ProviderRequest,
                        provider: ProviderType,
                        model: str,
                        error: Exception,
                        decision: Optional[RoutingDecision] = None) -> PerformanceEvent:
        """Track failed request"""
        
        request_type = self._detect_request_type(request.prompt)
        
        event = PerformanceEvent(
            timestamp=datetime.now(timezone.utc),
            request_id=request.metadata.get("request_id", ""),
            provider=provider,
            model=model,
            request_type=request_type,
            context_length=len(request.prompt),
            has_tools=request.tools is not None,
            is_streaming=request.stream,
            latency_ms=0,  # Failed before completion
            cost_usd=0,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error_type=type(error).__name__,
            routing_reason=decision.reason.value if decision else None
        )
        
        # Update profile
        profile_key = (provider, model)
        if profile_key not in self.profiles:
            self.profiles[profile_key] = ProviderProfile(provider=provider, model=model)
        
        self.profiles[profile_key].update(event)
        
        # Store and persist
        self.recent_events.append(event)
        if self.persistence:
            await self._persist_event(event)
            
        performance_events.add(1, {
            "type": "error",
            "provider": provider.value,
            "error_type": event.error_type
        })
        
        return event
        
    def get_provider_scores(self, request_type: Optional[str] = None) -> Dict[Tuple[ProviderType, str], float]:
        """Get performance scores for all providers"""
        scores = {}
        
        for profile_key, profile in self.profiles.items():
            if request_type and request_type in profile.request_type_performance:
                # Request-type specific score
                rt_perf = profile.request_type_performance[request_type]
                score = (
                    0.4 * rt_perf["avg_quality"] +
                    0.3 * rt_perf["success_rate"] +
                    0.3 * (1.0 - min(rt_perf["avg_latency"] / 10000, 1.0))  # Normalize latency
                )
            else:
                # Overall score
                score = (
                    0.4 * profile.avg_quality_score +
                    0.3 * (1.0 - profile.error_rate) +
                    0.2 * (1.0 - min(profile.avg_latency_ms / 10000, 1.0)) +
                    0.1 * (1.0 - min(profile.avg_cost_per_1k_tokens / 0.1, 1.0))
                )
                
            scores[profile_key] = score
            
        return scores
        
    def get_best_provider_for_request(self, request_type: str, 
                                     constraints: Optional[Dict[str, Any]] = None) -> Optional[Tuple[ProviderType, str]]:
        """Get best performing provider for specific request type"""
        scores = self.get_provider_scores(request_type)
        
        if not scores:
            return None
            
        # Apply constraints
        if constraints:
            max_latency = constraints.get("max_latency_ms")
            max_cost = constraints.get("max_cost_per_1k_tokens")
            require_tools = constraints.get("require_tools", False)
            
            filtered_scores = {}
            for profile_key, score in scores.items():
                profile = self.profiles[profile_key]
                
                if max_latency and profile.avg_latency_ms > max_latency:
                    continue
                if max_cost and profile.avg_cost_per_1k_tokens > max_cost:
                    continue
                if require_tools and not profile.supports_tools_observed:
                    continue
                    
                filtered_scores[profile_key] = score
                
            scores = filtered_scores
            
        if not scores:
            return None
            
        # Return best scoring provider
        best_key = max(scores.items(), key=lambda x: x[1])[0]
        return best_key
        
    def get_provider_health(self) -> Dict[ProviderType, float]:
        """Get health scores for all providers (0-1)"""
        health_scores = defaultdict(list)
        
        for profile in self.profiles.values():
            # Health based on error rate and recent performance
            health = 1.0 - profile.error_rate
            
            # Penalize if no recent requests
            age = datetime.now(timezone.utc) - profile.last_updated
            if age > timedelta(hours=1):
                health *= 0.9
            elif age > timedelta(hours=24):
                health *= 0.5
                
            health_scores[profile.provider].append(health)
            
        # Average health per provider
        return {
            provider: np.mean(scores) if scores else 0.5
            for provider, scores in health_scores.items()
        }
        
    def _detect_request_type(self, prompt: str) -> str:
        """Detect type of request from prompt"""
        prompt_lower = prompt.lower()
        
        for request_type, patterns in self.request_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                return request_type
                
        return "general"
        
    async def _track_cost_savings(self, event: PerformanceEvent):
        """Track cost savings compared to baseline"""
        # Update baseline for request type
        if event.request_type not in self.baseline_costs:
            self.baseline_costs[event.request_type] = event.cost_usd
        else:
            # Exponential moving average
            alpha = 0.05
            self.baseline_costs[event.request_type] = (
                (1 - alpha) * self.baseline_costs[event.request_type] + 
                alpha * event.cost_usd
            )
            
        # Calculate savings if we did better than baseline
        baseline = self.baseline_costs[event.request_type]
        if event.cost_usd < baseline:
            savings = baseline - event.cost_usd
            self.total_savings += savings
            cost_savings.add(savings)
            
    async def _persist_event(self, event: PerformanceEvent):
        """Persist event to storage"""
        try:
            await self.persistence.store_performance_event(
                "neural_performance",
                asdict(event)
            )
        except Exception as e:
            logger.error(f"Failed to persist performance event: {e}")
            
    async def generate_performance_report(self, 
                                        time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate performance report"""
        if time_window:
            cutoff = datetime.now(timezone.utc) - time_window
            events = [e for e in self.recent_events if e.timestamp > cutoff]
        else:
            events = list(self.recent_events)
            
        if not events:
            return {"error": "No events in time window"}
            
        report = {
            "time_window": str(time_window) if time_window else "all_time",
            "total_requests": len(events),
            "total_cost": sum(e.cost_usd for e in events),
            "total_savings": self.total_savings,
            "providers": {}
        }
        
        # Group by provider
        by_provider = defaultdict(list)
        for event in events:
            by_provider[(event.provider, event.model)].append(event)
            
        # Analyze each provider
        for (provider, model), provider_events in by_provider.items():
            success_events = [e for e in provider_events if e.success]
            
            provider_report = {
                "total_requests": len(provider_events),
                "success_rate": len(success_events) / len(provider_events),
                "avg_latency_ms": np.mean([e.latency_ms for e in success_events]) if success_events else 0,
                "p95_latency_ms": np.percentile([e.latency_ms for e in success_events], 95) if success_events else 0,
                "avg_cost_usd": np.mean([e.cost_usd for e in success_events]) if success_events else 0,
                "total_cost_usd": sum(e.cost_usd for e in success_events),
                "avg_quality": np.mean([e.quality_score for e in success_events if e.quality_score]) if success_events else 0.5,
                "request_types": {}
            }
            
            # By request type
            by_type = defaultdict(list)
            for event in provider_events:
                by_type[event.request_type].append(event)
                
            for req_type, type_events in by_type.items():
                type_success = [e for e in type_events if e.success]
                provider_report["request_types"][req_type] = {
                    "count": len(type_events),
                    "success_rate": len(type_success) / len(type_events),
                    "avg_latency_ms": np.mean([e.latency_ms for e in type_success]) if type_success else 0
                }
                
            report["providers"][f"{provider.value}/{model}"] = provider_report
            
        return report


# Export main classes
__all__ = [
    "PerformanceMetric",
    "PerformanceEvent",
    "ProviderProfile",
    "ModelPerformanceTracker"
]