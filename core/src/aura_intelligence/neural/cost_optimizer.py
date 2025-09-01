"""
Cost Optimizer - Multi-Objective Optimization for Model Routing
Based on 2025 research: Balance quality, cost, and latency with tenant policies

Key Features:
- Multi-objective scoring with configurable weights
- Per-tenant budget tracking and enforcement
- Real-time cost estimation and monitoring
- Historical cost analysis for optimization
- Policy-based routing constraints
- ROI tracking for routing decisions
"""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
import structlog
from pathlib import Path

from .provider_adapters import ProviderType, ModelConfig, ProviderRequest
from .model_router import RoutingContext, RoutingPolicy
from ..persistence.state_manager import StatePersistenceManager
from ..observability import create_tracer, create_meter

logger = structlog.get_logger(__name__)
tracer = create_tracer("cost_optimizer")
meter = create_meter("cost_optimizer")

# Metrics
cost_estimations = meter.create_counter(
    name="aura.cost.estimations",
    description="Cost estimations performed"
)

budget_violations = meter.create_counter(
    name="aura.cost.budget_violations",
    description="Budget constraint violations"
)

cost_per_tenant = meter.create_histogram(
    name="aura.cost.per_tenant_usd",
    description="Cost per tenant in USD"
)

optimization_score = meter.create_histogram(
    name="aura.cost.optimization_score",
    description="Multi-objective optimization scores"
)


class CostTier(str, Enum):
    """Cost tiers for models"""
    PREMIUM = "premium"      # $0.01+ per 1K tokens
    STANDARD = "standard"    # $0.001-0.01 per 1K tokens
    ECONOMY = "economy"      # $0.0001-0.001 per 1K tokens
    FREE = "free"           # Local models


class OptimizationObjective(str, Enum):
    """Optimization objectives"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class TenantPolicy:
    """Per-tenant routing policy"""
    tenant_id: str
    
    # Budget constraints
    max_cost_per_request: Optional[float] = None
    max_cost_per_hour: Optional[float] = None
    max_cost_per_day: Optional[float] = None
    max_cost_per_month: Optional[float] = None
    
    # Quality constraints
    min_quality_score: float = 0.7
    
    # Latency constraints
    max_latency_p95_ms: Optional[int] = None
    max_latency_p99_ms: Optional[int] = None
    
    # Provider constraints
    allowed_providers: Optional[List[ProviderType]] = None
    blocked_providers: Optional[List[ProviderType]] = None
    allowed_models: Optional[List[str]] = None
    blocked_models: Optional[List[str]] = None
    
    # Cost tiers
    allowed_cost_tiers: List[CostTier] = field(
        default_factory=lambda: [CostTier.PREMIUM, CostTier.STANDARD, CostTier.ECONOMY]
    )
    
    # Optimization settings
    objective: OptimizationObjective = OptimizationObjective.BALANCED
    custom_weights: Optional[Dict[str, float]] = None  # quality, cost, latency weights
    
    # Privacy settings
    require_local_models: bool = False
    require_data_residency: Optional[str] = None  # Region constraint
    
    # Feature flags
    enable_caching: bool = True
    enable_fallback: bool = True
    enable_semantic_routing: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True


@dataclass
class CostEstimate:
    """Cost estimate for a routing decision"""
    provider: ProviderType
    model: str
    
    # Token estimates
    estimated_input_tokens: int
    estimated_output_tokens: int
    
    # Cost breakdown
    input_cost: float
    output_cost: float
    total_cost: float
    
    # Quality and latency estimates
    estimated_quality: float
    estimated_latency_ms: float
    
    # Optimization score
    optimization_score: float
    
    # Metadata
    meets_budget: bool
    meets_quality: bool
    meets_latency: bool
    cost_tier: CostTier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TenantUsage:
    """Usage tracking for a tenant"""
    tenant_id: str
    
    # Current period usage
    hour_start: datetime
    hour_cost: float = 0.0
    hour_requests: int = 0
    
    day_start: datetime
    day_cost: float = 0.0
    day_requests: int = 0
    
    month_start: datetime
    month_cost: float = 0.0
    month_requests: int = 0
    
    # Historical data
    hourly_costs: deque = field(default_factory=lambda: deque(maxlen=24))
    daily_costs: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Provider breakdown
    provider_costs: Dict[str, float] = field(default_factory=dict)
    model_usage: Dict[str, int] = field(default_factory=dict)
    
    def update(self, cost: float, provider: str, model: str):
        """Update usage with new request"""
        now = datetime.now(timezone.utc)
        
        # Check period boundaries
        if now.hour != self.hour_start.hour:
            self.hourly_costs.append(self.hour_cost)
            self.hour_start = now.replace(minute=0, second=0, microsecond=0)
            self.hour_cost = 0.0
            self.hour_requests = 0
            
        if now.date() != self.day_start.date():
            self.daily_costs.append(self.day_cost)
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self.day_cost = 0.0
            self.day_requests = 0
            
        if now.month != self.month_start.month:
            self.month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            self.month_cost = 0.0
            self.month_requests = 0
            
        # Update counters
        self.hour_cost += cost
        self.hour_requests += 1
        self.day_cost += cost
        self.day_requests += 1
        self.month_cost += cost
        self.month_requests += 1
        
        # Update breakdowns
        self.provider_costs[provider] = self.provider_costs.get(provider, 0.0) + cost
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
        
    def get_remaining_budget(self, policy: TenantPolicy) -> Dict[str, float]:
        """Get remaining budget for each period"""
        remaining = {}
        
        if policy.max_cost_per_hour:
            remaining["hour"] = max(0, policy.max_cost_per_hour - self.hour_cost)
            
        if policy.max_cost_per_day:
            remaining["day"] = max(0, policy.max_cost_per_day - self.day_cost)
            
        if policy.max_cost_per_month:
            remaining["month"] = max(0, policy.max_cost_per_month - self.month_cost)
            
        return remaining


class ModelCostDatabase:
    """Database of model costs and performance characteristics"""
    
    def __init__(self):
        # Base cost data (USD per 1K tokens)
        self.cost_data = {
            # OpenAI
            "gpt-5": {
                "input": 0.01,
                "output": 0.03,
                "quality": 0.95,
                "latency_ms": 2000,
                "tier": CostTier.PREMIUM
            },
            "gpt-4o": {
                "input": 0.005,
                "output": 0.015,
                "quality": 0.90,
                "latency_ms": 1000,
                "tier": CostTier.STANDARD
            },
            
            # Anthropic
            "claude-opus-4.1": {
                "input": 0.015,
                "output": 0.075,
                "quality": 0.93,
                "latency_ms": 3000,
                "tier": CostTier.PREMIUM
            },
            "claude-sonnet-4.1": {
                "input": 0.003,
                "output": 0.015,
                "quality": 0.88,
                "latency_ms": 1500,
                "tier": CostTier.STANDARD
            },
            
            # Together
            "mamba-2-2.8b": {
                "input": 0.0002,
                "output": 0.0002,
                "quality": 0.75,
                "latency_ms": 500,
                "tier": CostTier.ECONOMY
            },
            "llama-3.1-70b": {
                "input": 0.0009,
                "output": 0.0009,
                "quality": 0.85,
                "latency_ms": 1000,
                "tier": CostTier.ECONOMY
            },
            "turbo-mixtral": {
                "input": 0.0002,
                "output": 0.0002,
                "quality": 0.80,
                "latency_ms": 200,
                "tier": CostTier.ECONOMY
            },
            
            # Ollama (local)
            "llama3-70b": {
                "input": 0.0,
                "output": 0.0,
                "quality": 0.82,
                "latency_ms": 5000,
                "tier": CostTier.FREE
            },
            "mixtral-8x7b": {
                "input": 0.0,
                "output": 0.0,
                "quality": 0.78,
                "latency_ms": 3000,
                "tier": CostTier.FREE
            }
        }
        
        # Quality adjustments based on task type
        self.quality_adjustments = {
            "reasoning": {
                "gpt-5": 0.0,
                "claude-opus-4.1": 0.02,
                "llama-3.1-70b": -0.05
            },
            "coding": {
                "claude-opus-4.1": 0.05,
                "gpt-5": 0.02,
                "llama-3.1-70b": -0.03
            },
            "creative": {
                "gpt-5": 0.03,
                "claude-opus-4.1": 0.02,
                "llama-3.1-70b": 0.0
            }
        }
        
    def get_model_cost(self, model: str) -> Optional[Dict[str, Any]]:
        """Get cost data for model"""
        return self.cost_data.get(model)
        
    def get_quality_score(self, model: str, task_type: str) -> float:
        """Get quality score adjusted for task type"""
        base_quality = self.cost_data.get(model, {}).get("quality", 0.5)
        
        adjustment = 0.0
        if task_type in self.quality_adjustments:
            adjustment = self.quality_adjustments[task_type].get(model, 0.0)
            
        return min(max(base_quality + adjustment, 0.0), 1.0)


class CostOptimizer:
    """Main cost optimization engine"""
    
    def __init__(self, persistence: Optional[StatePersistenceManager] = None):
        self.persistence = persistence
        self.cost_db = ModelCostDatabase()
        
        # Tenant tracking
        self.tenant_policies: Dict[str, TenantPolicy] = {}
        self.tenant_usage: Dict[str, TenantUsage] = {}
        
        # Global tracking
        self.global_usage = TenantUsage(tenant_id="__global__")
        
        # Historical data for optimization
        self.optimization_history: deque = deque(maxlen=10000)
        
        # Load persisted policies
        self._load_policies()
        
    def _load_policies(self):
        """Load tenant policies from persistence"""
        # TODO: Implement loading from persistence
        pass
        
    async def add_tenant_policy(self, policy: TenantPolicy):
        """Add or update tenant policy"""
        self.tenant_policies[policy.tenant_id] = policy
        
        # Initialize usage tracking
        if policy.tenant_id not in self.tenant_usage:
            self.tenant_usage[policy.tenant_id] = TenantUsage(
                tenant_id=policy.tenant_id,
                hour_start=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                day_start=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
                month_start=datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            )
            
        # Persist
        if self.persistence:
            await self.persistence.store_tenant_policy(policy)
            
        logger.info(f"Added tenant policy for {policy.tenant_id}")
        
    def get_tenant_policy(self, tenant_id: str) -> TenantPolicy:
        """Get tenant policy with defaults"""
        if tenant_id in self.tenant_policies:
            return self.tenant_policies[tenant_id]
            
        # Return default policy
        return TenantPolicy(tenant_id=tenant_id)
        
    async def estimate_costs(self, providers: List[Tuple[ProviderType, str]],
                           context: RoutingContext,
                           tenant_id: str) -> List[CostEstimate]:
        """Estimate costs for each provider/model combination"""
        
        with tracer.start_as_current_span("estimate_costs") as span:
            span.set_attribute("num_providers", len(providers))
            span.set_attribute("tenant_id", tenant_id)
            
            policy = self.get_tenant_policy(tenant_id)
            estimates = []
            
            for provider, model in providers:
                estimate = self._estimate_single(provider, model, context, policy)
                if estimate:
                    estimates.append(estimate)
                    
            # Sort by optimization score
            estimates.sort(key=lambda e: e.optimization_score, reverse=True)
            
            cost_estimations.add(1, {"tenant": tenant_id})
            
            return estimates
            
    def _estimate_single(self, provider: ProviderType, model: str,
                        context: RoutingContext, policy: TenantPolicy) -> Optional[CostEstimate]:
        """Estimate cost for single provider/model"""
        
        # Get cost data
        cost_data = self.cost_db.get_model_cost(model)
        if not cost_data:
            return None
            
        # Estimate tokens
        # Simple estimation - can be made more sophisticated
        prompt_tokens = context.estimated_tokens
        output_tokens = min(prompt_tokens // 2, 2000)  # Rough estimate
        
        # Calculate costs
        input_cost = (prompt_tokens / 1000) * cost_data["input"]
        output_cost = (output_tokens / 1000) * cost_data["output"]
        total_cost = input_cost + output_cost
        
        # Get quality score
        request_type = self._detect_request_type(context)
        quality = self.cost_db.get_quality_score(model, request_type)
        
        # Get latency
        latency = cost_data["latency_ms"]
        
        # Check constraints
        meets_budget = self._check_budget(total_cost, policy, context.request.metadata.get("tenant_id", ""))
        meets_quality = quality >= policy.min_quality_score
        meets_latency = True
        if policy.max_latency_p95_ms:
            meets_latency = latency <= policy.max_latency_p95_ms
            
        # Calculate optimization score
        opt_score = self._calculate_optimization_score(
            cost=total_cost,
            quality=quality,
            latency=latency,
            policy=policy,
            meets_constraints=meets_budget and meets_quality and meets_latency
        )
        
        estimate = CostEstimate(
            provider=provider,
            model=model,
            estimated_input_tokens=prompt_tokens,
            estimated_output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            estimated_quality=quality,
            estimated_latency_ms=latency,
            optimization_score=opt_score,
            meets_budget=meets_budget,
            meets_quality=meets_quality,
            meets_latency=meets_latency,
            cost_tier=cost_data["tier"]
        )
        
        return estimate
        
    def _detect_request_type(self, context: RoutingContext) -> str:
        """Detect request type for quality adjustments"""
        if context.requires_reasoning:
            return "reasoning"
        elif context.requires_coding:
            return "coding"
        elif "story" in context.request.prompt.lower() or "poem" in context.request.prompt.lower():
            return "creative"
        return "general"
        
    def _check_budget(self, cost: float, policy: TenantPolicy, tenant_id: str) -> bool:
        """Check if cost fits within budget constraints"""
        
        # Per-request limit
        if policy.max_cost_per_request and cost > policy.max_cost_per_request:
            return False
            
        # Get usage
        usage = self.tenant_usage.get(tenant_id)
        if not usage:
            return True
            
        # Check period limits
        remaining = usage.get_remaining_budget(policy)
        
        if "hour" in remaining and cost > remaining["hour"]:
            return False
        if "day" in remaining and cost > remaining["day"]:
            return False
        if "month" in remaining and cost > remaining["month"]:
            return False
            
        return True
        
    def _calculate_optimization_score(self, cost: float, quality: float, latency: float,
                                    policy: TenantPolicy, meets_constraints: bool) -> float:
        """Calculate multi-objective optimization score"""
        
        if not meets_constraints:
            return 0.0
            
        # Get weights based on objective
        if policy.objective == OptimizationObjective.MINIMIZE_COST:
            weights = {"quality": 0.2, "cost": 0.7, "latency": 0.1}
        elif policy.objective == OptimizationObjective.MAXIMIZE_QUALITY:
            weights = {"quality": 0.7, "cost": 0.2, "latency": 0.1}
        elif policy.objective == OptimizationObjective.MINIMIZE_LATENCY:
            weights = {"quality": 0.2, "cost": 0.1, "latency": 0.7}
        elif policy.objective == OptimizationObjective.CUSTOM and policy.custom_weights:
            weights = policy.custom_weights
        else:  # BALANCED
            weights = {"quality": 0.4, "cost": 0.3, "latency": 0.3}
            
        # Normalize scores
        cost_score = 1.0 - min(cost / 0.1, 1.0)  # Normalize to $0.10 max
        latency_score = 1.0 - min(latency / 10000, 1.0)  # Normalize to 10s max
        
        # Calculate weighted score
        score = (
            weights.get("quality", 0.4) * quality +
            weights.get("cost", 0.3) * cost_score +
            weights.get("latency", 0.3) * latency_score
        )
        
        return score
        
    async def record_usage(self, tenant_id: str, provider: ProviderType,
                         model: str, cost: float, success: bool):
        """Record actual usage for tracking"""
        
        # Update tenant usage
        if tenant_id not in self.tenant_usage:
            self.tenant_usage[tenant_id] = TenantUsage(
                tenant_id=tenant_id,
                hour_start=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                day_start=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
                month_start=datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            )
            
        usage = self.tenant_usage[tenant_id]
        usage.update(cost, provider.value, model)
        
        # Update global usage
        self.global_usage.update(cost, provider.value, model)
        
        # Check for budget violations
        policy = self.get_tenant_policy(tenant_id)
        remaining = usage.get_remaining_budget(policy)
        
        for period, budget in remaining.items():
            if budget <= 0:
                budget_violations.add(1, {"tenant": tenant_id, "period": period})
                logger.warning(
                    f"Budget violation for tenant {tenant_id}",
                    period=period,
                    remaining=budget
                )
                
        # Record metrics
        cost_per_tenant.record(cost, {"tenant": tenant_id})
        
        # Store in history for analysis
        self.optimization_history.append({
            "timestamp": datetime.now(timezone.utc),
            "tenant_id": tenant_id,
            "provider": provider.value,
            "model": model,
            "cost": cost,
            "success": success
        })
        
        # Persist if configured
        if self.persistence:
            await self.persistence.store_usage_record({
                "tenant_id": tenant_id,
                "provider": provider.value,
                "model": model,
                "cost": cost,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
    async def get_cost_report(self, tenant_id: str, 
                            time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate cost report for tenant"""
        
        usage = self.tenant_usage.get(tenant_id)
        if not usage:
            return {"error": "No usage data for tenant"}
            
        report = {
            "tenant_id": tenant_id,
            "current_hour_cost": usage.hour_cost,
            "current_day_cost": usage.day_cost,
            "current_month_cost": usage.month_cost,
            "provider_breakdown": usage.provider_costs,
            "model_usage": usage.model_usage,
            "hourly_trend": list(usage.hourly_costs),
            "daily_trend": list(usage.daily_costs)
        }
        
        # Add policy info
        policy = self.get_tenant_policy(tenant_id)
        report["policy"] = {
            "objective": policy.objective.value,
            "budgets": {
                "per_request": policy.max_cost_per_request,
                "per_hour": policy.max_cost_per_hour,
                "per_day": policy.max_cost_per_day,
                "per_month": policy.max_cost_per_month
            },
            "remaining_budget": usage.get_remaining_budget(policy)
        }
        
        # Calculate savings if using optimization
        if time_window:
            savings = await self._calculate_savings(tenant_id, time_window)
            report["estimated_savings"] = savings
            
        return report
        
    async def _calculate_savings(self, tenant_id: str, time_window: timedelta) -> Dict[str, float]:
        """Calculate estimated savings from optimization"""
        
        # Compare actual costs to what would have been spent with most expensive option
        cutoff = datetime.now(timezone.utc) - time_window
        
        actual_cost = 0.0
        baseline_cost = 0.0
        
        for record in self.optimization_history:
            if record["tenant_id"] == tenant_id and record["timestamp"] > cutoff:
                actual_cost += record["cost"]
                
                # Estimate baseline (e.g., always using GPT-5)
                baseline = self.cost_db.get_model_cost("gpt-5")
                if baseline:
                    # Rough estimate based on actual usage
                    baseline_cost += record["cost"] * (baseline["input"] / 0.001)  # Normalize
                    
        savings = max(0, baseline_cost - actual_cost)
        savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return {
            "actual_cost": actual_cost,
            "baseline_cost": baseline_cost,
            "savings_usd": savings,
            "savings_percent": savings_percent
        }
        
    def select_by_budget(self, estimates: List[CostEstimate], 
                        remaining_budget: float) -> Optional[CostEstimate]:
        """Select best option within remaining budget"""
        
        # Filter by budget
        within_budget = [e for e in estimates if e.total_cost <= remaining_budget]
        
        if not within_budget:
            return None
            
        # Return highest scoring option
        return max(within_budget, key=lambda e: e.optimization_score)


# Export main classes
__all__ = [
    "CostTier",
    "OptimizationObjective",
    "TenantPolicy",
    "CostEstimate",
    "TenantUsage",
    "ModelCostDatabase",
    "CostOptimizer"
]