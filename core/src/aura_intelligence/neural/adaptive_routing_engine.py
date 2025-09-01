"""
Adaptive Routing Engine - LNN-Inspired Learning for Model Selection
Transforms Liquid Neural Network concepts into routing intelligence

Key Features:
- Online learning from routing outcomes
- Continuous adaptation to provider performance
- Multi-objective optimization (cost/latency/quality)
- Integration with RouterBench methodology
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import json
from pathlib import Path
import structlog

from .provider_adapters import ProviderType, ModelCapability
try:
    from ..observability import create_tracer, create_meter
    tracer = create_tracer("adaptive_routing")
    meter = create_meter("adaptive_routing")
except ImportError:
    # Mock for testing
    class MockTracer:
        def start_as_current_span(self, name):
            from contextlib import contextmanager
            @contextmanager
            def mock_span():
                yield None
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

logger = structlog.get_logger(__name__)

# Metrics
adaptation_updates = meter.create_counter(
    name="aura.routing.adaptations",
    description="Number of routing model updates"
)

routing_reward = meter.create_histogram(
    name="aura.routing.reward",
    description="Reward signal for routing decisions"
)


@dataclass
class RoutingState:
    """State representation for routing decisions (inspired by LNN state-space)"""
    # Request features
    context_length: float  # Normalized 0-1
    complexity: float
    urgency: float
    has_tools: float
    requires_privacy: float
    requires_background: float
    
    # Provider states (health/performance)
    provider_health: Dict[str, float]
    provider_latencies: Dict[str, float]  # Recent avg
    provider_costs: Dict[str, float]  # Recent avg
    provider_qualities: Dict[str, float]  # Recent avg
    
    # System state
    current_load: float
    time_of_day: float  # Normalized 0-1
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural processing"""
        features = [
            self.context_length,
            self.complexity,
            self.urgency,
            self.has_tools,
            self.requires_privacy,
            self.requires_background,
            self.current_load,
            self.time_of_day
        ]
        
        # Add provider-specific features
        for provider in ProviderType:
            features.extend([
                self.provider_health.get(provider.value, 0.5),
                self.provider_latencies.get(provider.value, 0.5),
                self.provider_costs.get(provider.value, 0.5),
                self.provider_qualities.get(provider.value, 0.5)
            ])
            
        return torch.tensor(features, dtype=torch.float32)


class LiquidRoutingCell(nn.Module if TORCH_AVAILABLE else object):
    """
    Liquid-inspired routing cell with continuous dynamics
    Adapts the LNN ODE concept to routing decisions
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_providers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_providers = num_providers
        
        # Liquid parameters (simplified from full LNN)
        self.input_weights = nn.Linear(input_dim, hidden_dim)
        self.recurrent_weights = nn.Linear(hidden_dim, hidden_dim)
        self.output_weights = nn.Linear(hidden_dim, num_providers)
        
        # Time constants (tau) for dynamics
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        
        # Adaptive parameters
        self.adaptation_rate = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with liquid dynamics"""
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim)
            
        # Input contribution
        input_current = self.input_weights(x)
        
        # Recurrent contribution with liquid dynamics
        recurrent_current = self.recurrent_weights(h)
        
        # Update hidden state with ODE-inspired dynamics
        # dh/dt = (-h + input_current + recurrent_current) / tau
        h_new = h + ((-h + input_current + recurrent_current) / self.tau)
        h_new = torch.tanh(h_new)  # Bounded activation
        
        # Output routing scores
        scores = self.output_weights(h_new)
        
        return scores, h_new


class AdaptiveLNNRouter(nn.Module if TORCH_AVAILABLE else object):
    """
    Main adaptive routing network inspired by Liquid Neural Networks
    Learns optimal routing policies from experience
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.num_providers = len(ProviderType)
        self.feature_dim = 8 + (self.num_providers * 4)  # Base + provider features
        self.hidden_dim = config.get("hidden_dim", 64)
        
        # Liquid routing cells
        self.routing_cell = LiquidRoutingCell(
            self.feature_dim,
            self.hidden_dim,
            self.num_providers
        )
        
        # Value estimation head (for reinforcement learning)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=config.get("buffer_size", 10000))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.get("learning_rate", 0.001))
        
        # Hidden state memory
        self.hidden_states: Dict[str, torch.Tensor] = {}
        
    def forward(self, state: RoutingState, session_id: Optional[str] = None) -> Dict[str, float]:
        """Forward pass returning provider scores"""
        with tracer.start_as_current_span("lnn_routing_forward"):
            # Convert state to tensor
            x = state.to_tensor().unsqueeze(0)  # Add batch dimension
            
            # Get or create hidden state for session
            h = None
            if session_id and session_id in self.hidden_states:
                h = self.hidden_states[session_id]
                
            # Forward through liquid cell
            scores, h_new = self.routing_cell(x, h)
            
            # Store hidden state
            if session_id:
                self.hidden_states[session_id] = h_new.detach()
                
            # Apply softmax for probabilities
            probs = F.softmax(scores, dim=-1).squeeze(0)
            
            # Map to provider scores
            provider_scores = {}
            for i, provider in enumerate(ProviderType):
                provider_scores[provider] = float(probs[i])
                
            return provider_scores
            
    def compute_value(self, state: RoutingState) -> float:
        """Estimate value of current state"""
        x = state.to_tensor().unsqueeze(0)
        _, h = self.routing_cell(x)
        value = self.value_head(h)
        return float(value.squeeze())
        
    def update(self, experience: Dict[str, Any]):
        """Update model from routing experience"""
        with tracer.start_as_current_span("lnn_routing_update"):
            # Add to experience buffer
            self.experience_buffer.append(experience)
            
            # Batch update if enough experiences
            if len(self.experience_buffer) >= self.config.get("batch_size", 32):
                self._batch_update()
                
            adaptation_updates.add(1)
            
    def _batch_update(self):
        """Perform batch update from experience buffer"""
        batch_size = min(self.config.get("batch_size", 32), len(self.experience_buffer))
        
        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Prepare batch tensors
        states = torch.stack([exp["state"].to_tensor() for exp in batch])
        providers = torch.tensor([list(ProviderType).index(exp["provider"]) for exp in batch])
        rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32)
        
        # Forward pass
        scores, _ = self.routing_cell(states)
        probs = F.softmax(scores, dim=-1)
        
        # Policy gradient loss
        selected_probs = probs.gather(1, providers.unsqueeze(1)).squeeze()
        policy_loss = -(torch.log(selected_probs + 1e-8) * rewards).mean()
        
        # Value loss (TD learning)
        values = self.value_head(self.routing_cell(states)[1]).squeeze()
        value_targets = rewards  # Simplified, could use TD(λ)
        value_loss = F.mse_loss(values, value_targets)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Log metrics
        routing_reward.record(float(rewards.mean()))
        
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        checkpoint = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "experience_buffer": list(self.experience_buffer)[-1000:],  # Keep recent
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved routing checkpoint to {path}")
        
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Restore experience buffer
        for exp in checkpoint.get("experience_buffer", []):
            self.experience_buffer.append(exp)
            
        logger.info(f"Loaded routing checkpoint from {path}")


class RouterBenchEvaluator:
    """
    Evaluates routing decisions against RouterBench methodology
    Compares against baselines and tracks improvement
    """
    
    def __init__(self, router: AdaptiveLNNRouter):
        self.router = router
        self.baseline_results: Dict[str, float] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        
    async def evaluate_on_benchmark(self, benchmark_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run RouterBench-style evaluation"""
        results = {
            "total_requests": len(benchmark_data),
            "cost_vs_best_single": [],
            "quality_vs_best_single": [],
            "latency_vs_best_single": [],
            "composite_scores": []
        }
        
        for request in benchmark_data:
            # Get routing decision
            state = self._request_to_state(request)
            provider_scores = self.router.forward(state)
            selected_provider = max(provider_scores.items(), key=lambda x: x[1])[0]
            
            # Compare to baselines
            best_single_cost = request.get("best_single_cost", 1.0)
            best_single_quality = request.get("best_single_quality", 0.5)
            best_single_latency = request.get("best_single_latency", 1000)
            
            # Estimate our metrics
            our_cost = self._estimate_cost(selected_provider, request)
            our_quality = self._estimate_quality(selected_provider, request)
            our_latency = self._estimate_latency(selected_provider, request)
            
            # Calculate improvements
            cost_improvement = (best_single_cost - our_cost) / best_single_cost
            quality_improvement = (our_quality - best_single_quality) / best_single_quality
            latency_improvement = (best_single_latency - our_latency) / best_single_latency
            
            results["cost_vs_best_single"].append(cost_improvement)
            results["quality_vs_best_single"].append(quality_improvement)
            results["latency_vs_best_single"].append(latency_improvement)
            
            # Composite score (weighted average)
            composite = (
                0.4 * quality_improvement +
                0.3 * cost_improvement +
                0.3 * latency_improvement
            )
            results["composite_scores"].append(composite)
            
        # Aggregate results
        results["avg_cost_improvement"] = np.mean(results["cost_vs_best_single"])
        results["avg_quality_improvement"] = np.mean(results["quality_vs_best_single"])
        results["avg_latency_improvement"] = np.mean(results["latency_vs_best_single"])
        results["avg_composite_score"] = np.mean(results["composite_scores"])
        
        # Store for history
        self.evaluation_history.append({
            "timestamp": datetime.now(timezone.utc),
            "results": results
        })
        
        return results
        
    def _request_to_state(self, request: Dict[str, Any]) -> RoutingState:
        """Convert benchmark request to routing state"""
        return RoutingState(
            context_length=min(request.get("tokens", 0) / 100000, 1.0),
            complexity=request.get("complexity", 0.5),
            urgency=request.get("urgency", 0.5),
            has_tools=float(request.get("has_tools", False)),
            requires_privacy=float(request.get("requires_privacy", False)),
            requires_background=float(request.get("requires_background", False)),
            provider_health={p.value: 1.0 for p in ProviderType},
            provider_latencies={p.value: 0.5 for p in ProviderType},
            provider_costs={p.value: 0.5 for p in ProviderType},
            provider_qualities={p.value: 0.5 for p in ProviderType},
            current_load=0.5,
            time_of_day=0.5
        )
        
    def _estimate_cost(self, provider: ProviderType, request: Dict[str, Any]) -> float:
        """Estimate cost for provider on request"""
        base_costs = {
            ProviderType.OPENAI: 0.01,
            ProviderType.ANTHROPIC: 0.015,
            ProviderType.TOGETHER: 0.001,
            ProviderType.OLLAMA: 0.0
        }
        tokens = request.get("tokens", 1000)
        return base_costs.get(provider, 0.01) * (tokens / 1000)
        
    def _estimate_quality(self, provider: ProviderType, request: Dict[str, Any]) -> float:
        """Estimate quality for provider on request"""
        base_quality = {
            ProviderType.OPENAI: 0.9,
            ProviderType.ANTHROPIC: 0.85,
            ProviderType.TOGETHER: 0.7,
            ProviderType.OLLAMA: 0.75
        }
        return base_quality.get(provider, 0.7)
        
    def _estimate_latency(self, provider: ProviderType, request: Dict[str, Any]) -> float:
        """Estimate latency for provider on request"""
        base_latency = {
            ProviderType.OPENAI: 2000,
            ProviderType.ANTHROPIC: 3000,
            ProviderType.TOGETHER: 1000,
            ProviderType.OLLAMA: 5000
        }
        return base_latency.get(provider, 2000)


# Export main classes
__all__ = [
    "RoutingState",
    "LiquidRoutingCell",
    "AdaptiveLNNRouter",
    "RouterBenchEvaluator"
]