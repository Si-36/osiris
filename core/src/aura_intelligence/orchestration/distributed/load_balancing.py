"""
⚖️ Intelligent Load Balancing - TDA-Aware Distribution

Advanced load balancing using 2025 patterns:
- Consistent hashing with virtual nodes
- TDA-aware routing based on anomaly patterns
- Adaptive load balancing with ML predictions
- Circuit breaker patterns for fault isolation
- Real-time capacity planning

Research Sources:
- Consistent Hashing and Random Trees - Karger et al.
- The Power of Two Choices in Randomized Load Balancing
- Adaptive Load Balancing in Distributed Systems
- Circuit Breaker Pattern - Fowler
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import hashlib
import bisect
import random
from collections import defaultdict, deque

from .coordination_core import NodeId, NodeInfo, NodeState, LoadBalancer

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    POWER_OF_TWO = "power_of_two"
    TDA_AWARE = "tda_aware"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class LoadMetrics:
    """Node load metrics"""
    cpu_usage: float
    memory_usage: float
    active_connections: int
    request_rate: float
    response_time_p95: float
    error_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class CircuitBreaker:
    """Circuit breaker for fault isolation"""
    node_id: NodeId
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0

@dataclass
class ConsistentHashRing:
    """Consistent hash ring for distributed load balancing"""
    virtual_nodes: int = 150
    ring: Dict[int, NodeId] = field(default_factory=dict)
    sorted_keys: List[int] = field(default_factory=list)
    
    def add_node(self, node_id: NodeId, weight: float = 1.0):
        """Add node to hash ring"""
        num_virtual = int(self.virtual_nodes * weight)
        
        for i in range(num_virtual):
            virtual_key = self._hash(f"{node_id}:{i}")
            self.ring[virtual_key] = node_id
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: NodeId):
        """Remove node from hash ring"""
        keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
        for key in keys_to_remove:
            del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[NodeId]:
        """Get node for key using consistent hashing"""
        if not self.sorted_keys:
            return None
        
        hash_key = self._hash(key)
        idx = bisect.bisect_right(self.sorted_keys, hash_key)
        
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class TDALoadBalancer:
    """
    TDA-aware intelligent load balancer with adaptive strategies
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.TDA_AWARE,
        tda_integration: Optional[Any] = None
    ):
        self.strategy = strategy
        self.tda_integration = tda_integration
        
        # Node management
        self.nodes: Dict[NodeId, NodeInfo] = {}
        self.load_metrics: Dict[NodeId, LoadMetrics] = {}
        self.circuit_breakers: Dict[NodeId, CircuitBreaker] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.consistent_hash_ring = ConsistentHashRing()
        
        # TDA-aware routing
        self.anomaly_weights: Dict[NodeId, float] = {}
        self.capacity_predictions: Dict[NodeId, float] = {}
        
        # Performance tracking
        self.request_history: deque = deque(maxlen=1000)
        self.response_times: Dict[NodeId, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.05
        self.rebalance_threshold = 0.2
    
    async def add_node(self, node_info: NodeInfo):
        """Add node to load balancer"""
        self.nodes[node_info.node_id] = node_info
        self.load_metrics[node_info.node_id] = LoadMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            active_connections=0,
            request_rate=0.0,
            response_time_p95=0.0,
            error_rate=0.0
        )
        self.circuit_breakers[node_info.node_id] = CircuitBreaker(node_info.node_id)
        
        # Add to consistent hash ring
        weight = self._calculate_node_weight(node_info)
        self.consistent_hash_ring.add_node(node_info.node_id, weight)
        
        # Initialize TDA-aware weights
        self.anomaly_weights[node_info.node_id] = 1.0
        self.capacity_predictions[node_info.node_id] = 1.0
    
    async def remove_node(self, node_id: NodeId):
        """Remove node from load balancer"""
        self.nodes.pop(node_id, None)
        self.load_metrics.pop(node_id, None)
        self.circuit_breakers.pop(node_id, None)
        self.anomaly_weights.pop(node_id, None)
        self.capacity_predictions.pop(node_id, None)
        
        # Remove from consistent hash ring
        self.consistent_hash_ring.remove_node(node_id)
    
    async def select_node(self, request: Dict[str, Any]) -> Optional[NodeId]:
        """Select optimal node for request using configured strategy"""
        available_nodes = await self._get_available_nodes()
        
        if not available_nodes:
            return None
        
        # Apply strategy-specific selection
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return await self._consistent_hash_select(request, available_nodes)
        elif self.strategy == LoadBalancingStrategy.POWER_OF_TWO:
            return await self._power_of_two_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.TDA_AWARE:
            return await self._tda_aware_select(request, available_nodes)
        else:
            return random.choice(available_nodes)
    
    async def update_node_load(self, node_id: NodeId, load_metrics: LoadMetrics):
        """Update node load metrics"""
        if node_id not in self.nodes:
            return
        
        self.load_metrics[node_id] = load_metrics
        
        # Update circuit breaker
        circuit_breaker = self.circuit_breakers[node_id]
        
        if load_metrics.error_rate > 0.1:  # 10% error rate threshold
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = datetime.now(timezone.utc)
            
            if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
                circuit_breaker.state = CircuitState.OPEN
        else:
            # Reset failure count on success
            circuit_breaker.failure_count = 0
            
            if circuit_breaker.state == CircuitState.HALF_OPEN:
                circuit_breaker.half_open_calls += 1
                if circuit_breaker.half_open_calls >= circuit_breaker.half_open_max_calls:
                    circuit_breaker.state = CircuitState.CLOSED
                    circuit_breaker.half_open_calls = 0
        
        # Update TDA-aware weights
        await self._update_tda_weights(node_id, load_metrics)
        
        # Trigger rebalancing if needed
        await self._check_rebalancing_needed()
    
    async def get_cluster_load(self) -> Dict[NodeId, float]:
        """Get cluster-wide load distribution"""
        cluster_load = {}
        
        for node_id, metrics in self.load_metrics.items():
            # Calculate composite load score
            load_score = (
                metrics.cpu_usage * 0.3 +
                metrics.memory_usage * 0.2 +
                (metrics.active_connections / 100) * 0.2 +
                (metrics.response_time_p95 / 1000) * 0.2 +
                metrics.error_rate * 0.1
            )
            
            cluster_load[node_id] = min(load_score, 1.0)
        
        return cluster_load
    
    async def _get_available_nodes(self) -> List[NodeId]:
        """Get list of available nodes (not in circuit breaker open state)"""
        available = []
        current_time = datetime.now(timezone.utc)
        
        for node_id, node_info in self.nodes.items():
            if node_info.state != NodeState.ACTIVE:
                continue
            
            circuit_breaker = self.circuit_breakers[node_id]
            
            if circuit_breaker.state == CircuitState.CLOSED:
                available.append(node_id)
            elif circuit_breaker.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (circuit_breaker.last_failure_time and
                    current_time - circuit_breaker.last_failure_time > 
                    timedelta(seconds=circuit_breaker.recovery_timeout)):
                    
                    circuit_breaker.state = CircuitState.HALF_OPEN
                    circuit_breaker.half_open_calls = 0
                    available.append(node_id)
            elif circuit_breaker.state == CircuitState.HALF_OPEN:
                available.append(node_id)
        
        return available
    
    async def _round_robin_select(self, available_nodes: List[NodeId]) -> NodeId:
        """Round-robin node selection"""
        if not available_nodes:
            return None
        
        node = available_nodes[self.round_robin_index % len(available_nodes)]
        self.round_robin_index += 1
        return node
    
    async def _least_connections_select(self, available_nodes: List[NodeId]) -> NodeId:
        """Select node with least active connections"""
        if not available_nodes:
            return None
        
        min_connections = float('inf')
        selected_node = None
        
        for node_id in available_nodes:
            metrics = self.load_metrics.get(node_id)
            if metrics and metrics.active_connections < min_connections:
                min_connections = metrics.active_connections
                selected_node = node_id
        
        return selected_node or available_nodes[0]
    
    async def _weighted_round_robin_select(self, available_nodes: List[NodeId]) -> NodeId:
        """Weighted round-robin based on node capacity"""
        if not available_nodes:
            return None
        
        # Calculate weights based on inverse load
        weights = []
        for node_id in available_nodes:
            metrics = self.load_metrics.get(node_id)
            if metrics:
                # Higher weight for lower load
                weight = max(0.1, 1.0 - (metrics.cpu_usage + metrics.memory_usage) / 2)
            else:
                weight = 1.0
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return available_nodes[0]
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return available_nodes[i]
        
        return available_nodes[-1]
    
    async def _consistent_hash_select(
        self, 
        request: Dict[str, Any], 
        available_nodes: List[NodeId]
    ) -> NodeId:
        """Consistent hash-based selection"""
        # Use request key for consistent hashing
        request_key = request.get('key', request.get('workflow_id', str(random.random())))
        
        selected_node = self.consistent_hash_ring.get_node(request_key)
        
        # Fallback if selected node is not available
        if selected_node not in available_nodes:
            return available_nodes[0] if available_nodes else None
        
        return selected_node
    
    async def _power_of_two_select(self, available_nodes: List[NodeId]) -> NodeId:
        """Power of two choices load balancing"""
        if len(available_nodes) < 2:
            return available_nodes[0] if available_nodes else None
        
        # Randomly select two nodes
        candidates = random.sample(available_nodes, min(2, len(available_nodes)))
        
        # Choose the one with lower load
        best_node = candidates[0]
        best_load = float('inf')
        
        for node_id in candidates:
            metrics = self.load_metrics.get(node_id)
            if metrics:
                load = metrics.cpu_usage + metrics.memory_usage + metrics.active_connections / 100
                if load < best_load:
                    best_load = load
                    best_node = node_id
        
        return best_node
    
    async def _tda_aware_select(
        self, 
        request: Dict[str, Any], 
        available_nodes: List[NodeId]
    ) -> NodeId:
        """TDA-aware intelligent node selection"""
        if not available_nodes:
            return None
        
        # Get TDA context for request
        tda_correlation_id = request.get('tda_correlation_id')
        tda_context = None
        
        if self.tda_integration and tda_correlation_id:
            try:
                tda_context = await self.tda_integration.get_context(tda_correlation_id)
            except Exception:
                pass
        
        # Calculate node scores based on multiple factors
        node_scores = {}
        
        for node_id in available_nodes:
            score = await self._calculate_tda_aware_score(
                node_id, request, tda_context
            )
            node_scores[node_id] = score
        
        # Select node with highest score (with some exploration)
        if random.random() < self.exploration_rate:
            # Exploration: random selection
            return random.choice(available_nodes)
        else:
            # Exploitation: best score
            return max(node_scores.items(), key=lambda x: x[1])[0]
    
    async def _calculate_tda_aware_score(
        self, 
        node_id: NodeId, 
        request: Dict[str, Any], 
        tda_context: Optional[Any]
    ) -> float:
        """Calculate TDA-aware node selection score"""
        base_score = 1.0
        
        # Factor 1: Current load (inverse relationship)
        metrics = self.load_metrics.get(node_id)
        if metrics:
            load_factor = 1.0 - (metrics.cpu_usage + metrics.memory_usage) / 2
            base_score *= load_factor
        
        # Factor 2: TDA anomaly correlation
        anomaly_weight = self.anomaly_weights.get(node_id, 1.0)
        base_score *= anomaly_weight
        
        # Factor 3: Predicted capacity
        capacity_prediction = self.capacity_predictions.get(node_id, 1.0)
        base_score *= capacity_prediction
        
        # Factor 4: Historical performance
        response_times = self.response_times.get(node_id, deque())
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            # Lower response time = higher score
            response_factor = max(0.1, 1.0 - avg_response_time / 1000)
            base_score *= response_factor
        
        # Factor 5: TDA context-specific adjustments
        if tda_context:
            # If high anomaly severity, prefer nodes with lower anomaly correlation
            if hasattr(tda_context, 'anomaly_severity') and tda_context.anomaly_severity > 0.7:
                base_score *= (2.0 - anomaly_weight)  # Inverse preference
        
        return max(0.01, base_score)  # Ensure minimum score
    
    async def _update_tda_weights(self, node_id: NodeId, metrics: LoadMetrics):
        """Update TDA-aware weights based on performance"""
        if not self.tda_integration:
            return
        
        try:
            # Get current TDA patterns
            patterns = await self.tda_integration.get_current_patterns("15m")
            
            # Correlate node performance with TDA anomalies
            if patterns and "anomalies" in patterns:
                anomaly_severity = patterns.get("anomalies", {}).get("severity", 0.0)
                
                # Update anomaly weight based on correlation
                if metrics.error_rate > 0.05 and anomaly_severity > 0.5:
                    # High error rate during anomalies - reduce weight
                    self.anomaly_weights[node_id] *= (1.0 - self.learning_rate)
                elif metrics.error_rate < 0.01 and anomaly_severity > 0.5:
                    # Low error rate during anomalies - increase weight
                    self.anomaly_weights[node_id] *= (1.0 + self.learning_rate)
                
                # Clamp weights
                self.anomaly_weights[node_id] = max(0.1, min(2.0, self.anomaly_weights[node_id]))
        
        except Exception:
            pass
    
    async def _check_rebalancing_needed(self):
        """Check if cluster rebalancing is needed"""
        cluster_load = await self.get_cluster_load()
        
        if not cluster_load:
            return
        
        loads = list(cluster_load.values())
        if not loads:
            return
        
        # Calculate load variance
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        
        # Trigger rebalancing if variance is too high
        if variance > self.rebalance_threshold:
            await self._trigger_rebalancing(cluster_load)
    
    async def _trigger_rebalancing(self, cluster_load: Dict[NodeId, float]):
        """Trigger cluster rebalancing"""
        # Find overloaded and underloaded nodes
        mean_load = sum(cluster_load.values()) / len(cluster_load)
        
        overloaded = [node_id for node_id, load in cluster_load.items() if load > mean_load * 1.5]
        underloaded = [node_id for node_id, load in cluster_load.items() if load < mean_load * 0.5]
        
        if overloaded and underloaded:
            # Notify TDA about rebalancing decision
            if self.tda_integration:
                await self.tda_integration.send_orchestration_result(
                    {
                        "event": "load_rebalancing_triggered",
                        "overloaded_nodes": overloaded,
                        "underloaded_nodes": underloaded,
                        "mean_load": mean_load,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    "load_balancer_rebalancing"
                )
    
    def _calculate_node_weight(self, node_info: NodeInfo) -> float:
        """Calculate node weight for consistent hashing"""
        # Base weight on node capabilities and resources
        weight = 1.0
        
        # Adjust based on capabilities
        if "high_memory" in node_info.capabilities:
            weight *= 1.5
        if "gpu_enabled" in node_info.capabilities:
            weight *= 2.0
        if "ssd_storage" in node_info.capabilities:
            weight *= 1.2
        
        # Adjust based on load factor
        weight *= (2.0 - node_info.load_factor)  # Lower load = higher weight
        
        return max(0.1, weight)
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        return {
            "strategy": self.strategy.value,
            "total_nodes": len(self.nodes),
            "available_nodes": len([n for n in self.nodes.values() if n.state == NodeState.ACTIVE]),
            "circuit_breakers_open": len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN]),
            "average_response_time": self._calculate_average_response_time(),
            "request_count": len(self.request_history),
            "anomaly_weights": dict(self.anomaly_weights),
            "capacity_predictions": dict(self.capacity_predictions)
        }
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all nodes"""
        all_times = []
        for times in self.response_times.values():
            all_times.extend(times)
        
        return sum(all_times) / len(all_times) if all_times else 0.0