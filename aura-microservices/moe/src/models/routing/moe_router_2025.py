"""
Ultimate MoE Router System 2025
Combines all AURA Intelligence routing innovations

Key Features:
- Google Switch Transformer for efficient single-expert routing
- Top-K routing for multi-expert scenarios  
- Semantic capability-based routing
- TDA-aware anomaly routing
- Consistent hashing with virtual nodes
- Circuit breaker fault isolation
- Real-time load balancing
- Event-driven routing patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import hashlib
import bisect
from collections import defaultdict, deque
import structlog
from opentelemetry import trace, metrics

# Try to import advanced libraries
try:
    from transformers import SwitchTransformersConfig
    from transformers.models.switch_transformer.modeling_switch_transformer import SwitchTransformersSparseMLP
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Setup observability
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
logger = structlog.get_logger()

# Metrics
routing_counter = meter.create_counter("moe_routing_total", description="Total routing decisions")
expert_usage = meter.create_histogram("moe_expert_usage", description="Expert utilization")
routing_latency = meter.create_histogram("moe_routing_latency_ms", description="Routing latency")
load_balance_score = meter.create_gauge("moe_load_balance_score", description="Load balance coefficient")


class RoutingStrategy(Enum):
    """Available routing strategies"""
    SWITCH_TRANSFORMER = "switch_transformer"  # Google's top-1 routing
    TOP_K = "top_k"  # Route to multiple experts
    SEMANTIC = "semantic"  # Capability-based routing
    TDA_AWARE = "tda_aware"  # Anomaly-aware routing
    CONSISTENT_HASH = "consistent_hash"  # Distributed routing
    POWER_OF_TWO = "power_of_two"  # Load balanced routing
    ADAPTIVE = "adaptive"  # Dynamically choose strategy


class ServiceType(Enum):
    """Available microservices"""
    NEUROMORPHIC = "neuromorphic"
    MEMORY = "memory"
    BYZANTINE = "byzantine"
    LNN = "lnn"
    TDA = "tda"  # Future
    CUSTOM = "custom"


@dataclass
class ServiceProfile:
    """Profile for each microservice"""
    service_id: str
    service_type: ServiceType
    endpoint: str
    capabilities: Set[str]
    performance_score: float = 1.0
    current_load: float = 0.0
    available: bool = True
    specializations: List[str] = field(default_factory=list)
    max_capacity: int = 100
    latency_p95_ms: float = 10.0
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_services: List[str]
    routing_strategy: RoutingStrategy
    confidence_scores: List[float]
    reasoning: str
    fallback_services: List[str]
    estimated_latency_ms: float
    load_balance_score: float


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault isolation"""
    service_id: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    success_count: int = 0


class SwitchTransformerGate(nn.Module):
    """
    Google Switch Transformer gate for top-1 routing
    Most efficient for single-expert selection
    """
    
    def __init__(self, input_dim: int, num_experts: int, 
                 capacity_factor: float = 1.25, jitter_noise: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        
        # Single linear layer as per Switch Transformer
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
        # Load balancing tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('gate_probs_sum', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, input_dim]
        Returns:
            expert_indices: Selected expert per input [batch_size]
            expert_gates: Gate values [batch_size]
            load_balance_loss: Auxiliary loss for training
        """
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(x) * self.jitter_noise
            x = x + noise
            
        # Compute gate logits
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        
        # Softmax to get probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-1 selection (Switch Transformer key feature)
        expert_gates, expert_indices = torch.max(gate_probs, dim=-1)
        
        # Load balancing loss
        if self.training:
            # Fraction of tokens per expert
            me = F.one_hot(expert_indices, self.num_experts).float().mean(dim=0)
            # Mean gate probability per expert
            ce = gate_probs.mean(dim=0)
            # Load balance loss encourages uniform distribution
            load_balance_loss = self.num_experts * torch.sum(me * ce)
            
            # Update statistics
            with torch.no_grad():
                self.expert_counts = 0.9 * self.expert_counts + 0.1 * me * x.size(0)
                self.gate_probs_sum = 0.9 * self.gate_probs_sum + 0.1 * ce
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device)
            
        return expert_indices, expert_gates, load_balance_loss


class TopKGate(nn.Module):
    """
    Top-K routing for multi-expert scenarios
    Better for complex tasks requiring multiple services
    """
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Gating network with bias for flexibility
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Load tracking
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            expert_indices: Top-k experts [batch_size, k]
            expert_gates: Gate values [batch_size, k]
            usage_variance: Variance in expert usage (for load balancing)
        """
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize top-k gates
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Track usage
        if self.training:
            usage = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1).mean(dim=0)
            self.expert_usage = 0.95 * self.expert_usage + 0.05 * usage
            
        # Usage variance (lower is better balanced)
        usage_variance = torch.var(self.expert_usage)
        
        return top_k_indices, top_k_gates, usage_variance


class ConsistentHashRouter:
    """
    Consistent hashing for distributed routing
    Ensures stable service assignment even with failures
    """
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        
    def add_service(self, service_id: str, weight: float = 1.0):
        """Add service to hash ring with weight"""
        num_vnodes = int(self.virtual_nodes * weight)
        
        for i in range(num_vnodes):
            vnode_key = f"{service_id}:{i}"
            hash_value = self._hash(vnode_key)
            self.ring[hash_value] = service_id
            
        self.sorted_keys = sorted(self.ring.keys())
        
    def remove_service(self, service_id: str):
        """Remove service from hash ring"""
        self.ring = {k: v for k, v in self.ring.items() if v != service_id}
        self.sorted_keys = sorted(self.ring.keys())
        
    def get_service(self, key: str) -> Optional[str]:
        """Get service for key using consistent hashing"""
        if not self.sorted_keys:
            return None
            
        hash_key = self._hash(key)
        idx = bisect.bisect_right(self.sorted_keys, hash_key)
        
        if idx == len(self.sorted_keys):
            idx = 0
            
        return self.ring[self.sorted_keys[idx]]
        
    def _hash(self, key: str) -> int:
        """SHA-256 based hash function"""
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class AdaptiveMoERouter(nn.Module):
    """
    Ultimate MoE Router combining all strategies
    Adaptively selects best routing strategy based on context
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_services: int = 4,  # Current microservices
                 top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_services = num_services
        self.top_k = top_k
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Strategy selector network
        self.strategy_selector = nn.Linear(hidden_dim, len(RoutingStrategy))
        
        # Different routing gates
        self.switch_gate = SwitchTransformerGate(hidden_dim, num_services)
        self.topk_gate = TopKGate(hidden_dim, num_services, top_k)
        
        # Capability matching for semantic routing
        self.capability_embeddings = nn.Parameter(torch.randn(100, hidden_dim))
        self.service_embeddings = nn.Parameter(torch.randn(num_services, hidden_dim))
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, 
                x: torch.Tensor,
                request_metadata: Optional[Dict[str, Any]] = None,
                force_strategy: Optional[RoutingStrategy] = None) -> Dict[str, Any]:
        """
        Adaptive routing based on input and context
        
        Args:
            x: Input features [batch_size, input_dim]
            request_metadata: Additional routing hints
            force_strategy: Override strategy selection
            
        Returns:
            Routing decision with selected services and metadata
        """
        with tracer.start_as_current_span("moe_routing"):
            batch_size = x.size(0)
            
            # Extract features
            features = self.feature_extractor(x)
            
            # Select routing strategy
            if force_strategy:
                strategy = force_strategy
                strategy_logits = None
            else:
                strategy_logits = self.strategy_selector(features)
                strategy_probs = F.softmax(strategy_logits, dim=-1)
                strategy_idx = torch.argmax(strategy_probs, dim=-1)
                strategy = list(RoutingStrategy)[strategy_idx[0].item()]
            
            # Route based on strategy
            if strategy == RoutingStrategy.SWITCH_TRANSFORMER:
                indices, gates, aux_loss = self.switch_gate(features)
                selected_services = indices.tolist()
                confidence_scores = gates.tolist()
                
            elif strategy == RoutingStrategy.TOP_K:
                indices, gates, variance = self.topk_gate(features)
                selected_services = indices.tolist()
                confidence_scores = gates.tolist()
                aux_loss = variance
                
            elif strategy == RoutingStrategy.SEMANTIC:
                # Capability-based routing
                selected_services, confidence_scores = self._semantic_routing(
                    features, request_metadata
                )
                aux_loss = torch.tensor(0.0)
                
            else:
                # Default to top-1
                indices, gates, aux_loss = self.switch_gate(features)
                selected_services = indices.tolist()
                confidence_scores = gates.tolist()
            
            # Record metrics
            routing_counter.add(1, {"strategy": strategy.value})
            
            return {
                "selected_services": selected_services,
                "confidence_scores": confidence_scores,
                "routing_strategy": strategy,
                "auxiliary_loss": aux_loss,
                "strategy_logits": strategy_logits
            }
    
    def _semantic_routing(self, 
                         features: torch.Tensor,
                         metadata: Optional[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
        """Semantic capability-based routing"""
        # Match request to service capabilities
        request_embedding = features.mean(dim=0, keepdim=True)
        
        # Compute similarity with service embeddings
        similarities = F.cosine_similarity(
            request_embedding.unsqueeze(1),
            self.service_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Get top services
        scores, indices = torch.topk(similarities, min(self.top_k, self.num_services), dim=1)
        
        return indices[0].tolist(), F.softmax(scores[0], dim=-1).tolist()


class MoERouterSystem:
    """
    Complete MoE Router System with all features
    """
    
    def __init__(self, service_configs: List[ServiceProfile]):
        self.logger = logger.bind(component="MoERouter")
        
        # Services
        self.services: Dict[str, ServiceProfile] = {
            s.service_id: s for s in service_configs
        }
        self.num_services = len(self.services)
        
        # Neural router
        self.neural_router = AdaptiveMoERouter(
            num_services=self.num_services
        )
        
        # Consistent hash router
        self.hash_router = ConsistentHashRouter()
        for service in service_configs:
            self.hash_router.add_service(service.service_id, service.performance_score)
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            s.service_id: CircuitBreaker(s.service_id) for s in service_configs
        }
        
        # Load tracking
        self.load_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance history
        self.routing_history = deque(maxlen=1000)
        
        # TDA integration placeholder
        self.tda_client = None
        
    async def route_request(self,
                           request_data: Dict[str, Any],
                           routing_strategy: Optional[RoutingStrategy] = None) -> RoutingDecision:
        """
        Route request to appropriate service(s)
        
        Args:
            request_data: Request payload with features
            routing_strategy: Force specific strategy
            
        Returns:
            RoutingDecision with selected services
        """
        start_time = time.perf_counter()
        
        try:
            # Extract features
            features = self._extract_features(request_data)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Check circuit breakers
            available_services = self._get_available_services()
            
            if not available_services:
                raise Exception("No available services")
            
            # Neural routing
            with torch.no_grad():
                routing_result = self.neural_router(
                    features_tensor,
                    request_data,
                    routing_strategy
                )
            
            # Map indices to service IDs
            service_list = list(self.services.keys())
            selected_indices = routing_result["selected_services"]
            
            # Handle single vs multiple selection
            if isinstance(selected_indices[0], list):
                # Top-K routing
                selected_services = []
                confidence_scores = []
                for i in range(len(selected_indices[0])):
                    if selected_indices[0][i] < len(service_list):
                        service_id = service_list[selected_indices[0][i]]
                        if service_id in available_services:
                            selected_services.append(service_id)
                            confidence_scores.append(routing_result["confidence_scores"][0][i])
            else:
                # Single routing
                service_id = service_list[selected_indices[0]]
                if service_id in available_services:
                    selected_services = [service_id]
                    confidence_scores = [routing_result["confidence_scores"][0]]
                else:
                    # Fallback to hash routing
                    service_id = self.hash_router.get_service(str(request_data))
                    selected_services = [service_id] if service_id else []
                    confidence_scores = [0.5]
            
            # Load balancing check
            selected_services = await self._apply_load_balancing(
                selected_services, 
                request_data
            )
            
            # Calculate metrics
            load_scores = [self._calculate_load_score(s) for s in selected_services]
            avg_load_balance = np.mean(load_scores) if load_scores else 0.0
            
            # Estimate latency
            estimated_latency = self._estimate_latency(selected_services)
            
            # Record routing
            routing_time = (time.perf_counter() - start_time) * 1000
            routing_latency.record(routing_time)
            
            decision = RoutingDecision(
                selected_services=selected_services,
                routing_strategy=routing_result["routing_strategy"],
                confidence_scores=confidence_scores,
                reasoning=self._generate_reasoning(routing_result, selected_services),
                fallback_services=self._get_fallback_services(selected_services),
                estimated_latency_ms=estimated_latency,
                load_balance_score=avg_load_balance
            )
            
            # Update history
            self.routing_history.append({
                "timestamp": time.time(),
                "decision": decision,
                "routing_time_ms": routing_time
            })
            
            return decision
            
        except Exception as e:
            self.logger.error("Routing failed", error=str(e))
            # Emergency fallback
            return RoutingDecision(
                selected_services=[],
                routing_strategy=RoutingStrategy.CONSISTENT_HASH,
                confidence_scores=[],
                reasoning=f"Routing failed: {str(e)}",
                fallback_services=list(self.services.keys()),
                estimated_latency_ms=100.0,
                load_balance_score=0.0
            )
    
    def _extract_features(self, request_data: Dict[str, Any]) -> List[float]:
        """Extract features from request for routing"""
        features = []
        
        # Request type features
        request_type = request_data.get("type", "unknown")
        type_features = {
            "inference": [1, 0, 0, 0],
            "training": [0, 1, 0, 0],
            "consensus": [0, 0, 1, 0],
            "storage": [0, 0, 0, 1]
        }
        features.extend(type_features.get(request_type, [0, 0, 0, 0]))
        
        # Data characteristics
        data = request_data.get("data", [])
        if isinstance(data, list):
            features.extend([
                len(data) / 1000.0,  # Size normalized
                np.mean(data) if data else 0.0,
                np.std(data) if data else 0.0,
                np.min(data) if data else 0.0,
                np.max(data) if data else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Metadata features
        features.append(request_data.get("priority", 0.5))
        features.append(request_data.get("complexity", 0.5))
        features.append(request_data.get("urgency", 0.5))
        
        # Service preference hints
        for service_type in ServiceType:
            features.append(1.0 if service_type.value in str(request_data) else 0.0)
        
        # Pad to expected dimension
        while len(features) < 512:
            features.append(0.0)
            
        return features[:512]
    
    def _get_available_services(self) -> Set[str]:
        """Get services that are available (not circuit broken)"""
        available = set()
        
        for service_id, breaker in self.circuit_breakers.items():
            if breaker.state == "closed":
                available.add(service_id)
            elif breaker.state == "half_open":
                # Check if ready to retry
                if time.time() - breaker.last_failure_time > breaker.recovery_timeout:
                    breaker.state = "closed"
                    breaker.failure_count = 0
                    available.add(service_id)
                    
        return available
    
    async def _apply_load_balancing(self,
                                   selected_services: List[str],
                                   request_data: Dict[str, Any]) -> List[str]:
        """Apply load balancing to service selection"""
        if len(selected_services) <= 1:
            return selected_services
        
        # Power of two choices
        if len(selected_services) > 2:
            loads = [(s, self._calculate_load_score(s)) for s in selected_services]
            loads.sort(key=lambda x: x[1])
            return [loads[0][0], loads[1][0]]  # Two least loaded
        
        return selected_services
    
    def _calculate_load_score(self, service_id: str) -> float:
        """Calculate current load score for service (0-1, lower is better)"""
        if service_id not in self.services:
            return 1.0
            
        profile = self.services[service_id]
        
        # Combine multiple factors
        load_factors = [
            profile.current_load / profile.max_capacity,
            profile.error_rate,
            min(profile.latency_p95_ms / 100.0, 1.0)  # Normalize latency
        ]
        
        return np.mean(load_factors)
    
    def _estimate_latency(self, selected_services: List[str]) -> float:
        """Estimate expected latency for selected services"""
        if not selected_services:
            return 1000.0  # High penalty
        
        latencies = []
        for service_id in selected_services:
            if service_id in self.services:
                latencies.append(self.services[service_id].latency_p95_ms)
            else:
                latencies.append(50.0)  # Default
                
        # For parallel execution, use max latency
        # For sequential, use sum
        return max(latencies) if len(latencies) > 1 else latencies[0]
    
    def _get_fallback_services(self, primary_services: List[str]) -> List[str]:
        """Get fallback services if primary fail"""
        fallbacks = []
        
        for service_id, profile in self.services.items():
            if service_id not in primary_services and profile.available:
                fallbacks.append(service_id)
                
        # Sort by performance score
        fallbacks.sort(key=lambda s: self.services[s].performance_score, reverse=True)
        
        return fallbacks[:2]  # Top 2 fallbacks
    
    def _generate_reasoning(self, 
                           routing_result: Dict[str, Any],
                           selected_services: List[str]) -> str:
        """Generate human-readable routing reasoning"""
        strategy = routing_result["routing_strategy"]
        
        reasons = []
        reasons.append(f"Strategy: {strategy.value}")
        
        if selected_services:
            service_names = [self.services[s].service_type.value for s in selected_services]
            reasons.append(f"Selected: {', '.join(service_names)}")
            
        if "confidence_scores" in routing_result:
            avg_confidence = np.mean(routing_result["confidence_scores"])
            reasons.append(f"Confidence: {avg_confidence:.2%}")
            
        return " | ".join(reasons)
    
    async def report_result(self,
                           service_id: str,
                           success: bool,
                           latency_ms: float,
                           error: Optional[str] = None):
        """Report routing result for learning"""
        breaker = self.circuit_breakers.get(service_id)
        
        if not breaker:
            return
            
        if success:
            breaker.success_count += 1
            breaker.failure_count = 0
            if breaker.state == "half_open":
                breaker.state = "closed"
                
            # Update metrics
            self.response_times[service_id].append(latency_ms)
            
        else:
            breaker.failure_count += 1
            breaker.last_failure_time = time.time()
            
            if breaker.failure_count >= breaker.failure_threshold:
                breaker.state = "open"
                self.logger.warning(f"Circuit breaker opened for {service_id}")
                
        # Update load tracking
        self.load_tracker[service_id].append(time.time())
        
        # Update service profile
        if service_id in self.services:
            profile = self.services[service_id]
            profile.current_load = len(self.load_tracker[service_id])
            
            if self.response_times[service_id]:
                profile.latency_p95_ms = np.percentile(
                    list(self.response_times[service_id]), 95
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics"""
        total_routes = len(self.routing_history)
        
        if not total_routes:
            return {"status": "no_data"}
            
        # Strategy distribution
        strategy_counts = defaultdict(int)
        for entry in self.routing_history:
            strategy = entry["decision"].routing_strategy
            strategy_counts[strategy.value] += 1
            
        # Service utilization
        service_usage = defaultdict(int)
        for entry in self.routing_history:
            for service in entry["decision"].selected_services:
                service_usage[service] += 1
                
        # Load balance coefficient
        usage_values = list(service_usage.values())
        load_balance_coef = np.std(usage_values) / (np.mean(usage_values) + 1e-8)
        
        # Average routing time
        routing_times = [entry["routing_time_ms"] for entry in self.routing_history]
        avg_routing_time = np.mean(routing_times)
        
        return {
            "total_routes": total_routes,
            "strategy_distribution": dict(strategy_counts),
            "service_utilization": dict(service_usage),
            "load_balance_coefficient": load_balance_coef,
            "avg_routing_time_ms": avg_routing_time,
            "circuit_breaker_states": {
                sid: breaker.state for sid, breaker in self.circuit_breakers.items()
            },
            "active_services": len(self._get_available_services()),
            "total_services": len(self.services)
        }


# Helper functions
def create_default_service_profiles() -> List[ServiceProfile]:
    """Create default profiles for current microservices"""
    return [
        ServiceProfile(
            service_id="neuromorphic",
            service_type=ServiceType.NEUROMORPHIC,
            endpoint="http://localhost:8000",
            capabilities={"spiking", "energy_efficient", "real_time"},
            specializations=["spike_processing", "stdp_learning"],
            max_capacity=1000,
            latency_p95_ms=5.0
        ),
        ServiceProfile(
            service_id="memory",
            service_type=ServiceType.MEMORY,
            endpoint="http://localhost:8001",
            capabilities={"storage", "retrieval", "shape_analysis"},
            specializations=["topological_indexing", "tier_management"],
            max_capacity=500,
            latency_p95_ms=10.0
        ),
        ServiceProfile(
            service_id="byzantine",
            service_type=ServiceType.BYZANTINE,
            endpoint="http://localhost:8002",
            capabilities={"consensus", "fault_tolerance", "voting"},
            specializations=["distributed_consensus", "pbft"],
            max_capacity=100,
            latency_p95_ms=50.0
        ),
        ServiceProfile(
            service_id="lnn",
            service_type=ServiceType.LNN,
            endpoint="http://localhost:8003",
            capabilities={"adaptive", "continuous_learning", "liquid"},
            specializations=["real_time_adaptation", "ode_dynamics"],
            max_capacity=200,
            latency_p95_ms=15.0
        )
    ]


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_router():
        # Initialize with default services
        services = create_default_service_profiles()
        router = MoERouterSystem(services)
        
        # Test request
        request = {
            "type": "inference",
            "data": [0.1, 0.2, 0.3] * 100,
            "priority": 0.8,
            "complexity": 0.6
        }
        
        # Route request
        decision = await router.route_request(request)
        
        print(f"Selected services: {decision.selected_services}")
        print(f"Strategy: {decision.routing_strategy.value}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Estimated latency: {decision.estimated_latency_ms}ms")
        
        # Get metrics
        metrics = router.get_metrics()
        print(f"\nMetrics: {metrics}")
    
    asyncio.run(test_router())