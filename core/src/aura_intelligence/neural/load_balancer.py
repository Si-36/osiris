"""
Load Balancer - Advanced Multi-Provider Load Distribution
Based on 2025 patterns: Latency-aware, quota-conscious, priority-based routing

Key Features:
- Multiple load balancing strategies
- Real-time latency tracking
- Rate limit awareness
- Priority queue with backpressure
- Elastic provider pools
- Spot instance integration
"""

import asyncio
import time
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from enum import Enum
import numpy as np
import structlog

from .provider_adapters import ProviderType, ProviderRequest, ModelConfig
from .fallback_chain import ProviderHealth, HealthStatus
from ..observability import create_tracer, create_meter

logger = structlog.get_logger(__name__)
tracer = create_tracer("load_balancer")
meter = create_meter("load_balancer")

# Metrics
request_distribution = meter.create_histogram(
    name="aura.load_balancer.request_distribution",
    description="Distribution of requests across providers"
)

queue_depth = meter.create_gauge(
    name="aura.load_balancer.queue_depth",
    description="Current queue depth per provider"
)

backpressure_events = meter.create_counter(
    name="aura.load_balancer.backpressure_events",
    description="Backpressure events triggered"
)

provider_utilization = meter.create_gauge(
    name="aura.load_balancer.utilization",
    description="Provider utilization percentage"
)


class QueuePriority(int, Enum):
    """Request priority levels"""
    CRITICAL = 0     # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4   # Lowest priority


class ProviderState(str, Enum):
    """Provider operational state"""
    ACTIVE = "active"
    DRAINING = "draining"     # Finishing existing, not accepting new
    SUSPENDED = "suspended"   # Temporarily unavailable
    SCALING = "scaling"       # Scaling up/down


@dataclass
class QueuedRequest:
    """Request in the priority queue"""
    request: ProviderRequest
    priority: QueuePriority
    timestamp: datetime
    attempt: int = 0
    callback: Optional[Callable] = None
    deadline: Optional[datetime] = None
    
    def __lt__(self, other):
        """For heap queue ordering"""
        # Lower priority value = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # For same priority, older requests first
        return self.timestamp < other.timestamp


@dataclass
class ProviderPool:
    """Pool of provider instances"""
    provider: ProviderType
    
    # Instance management
    min_instances: int = 1
    max_instances: int = 10
    current_instances: int = 1
    target_instances: int = 1
    
    # State
    state: ProviderState = ProviderState.ACTIVE
    
    # Capacity tracking
    max_concurrent_per_instance: int = 10
    current_load: int = 0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    
    # Cost tracking (for spot instances)
    spot_price: Optional[float] = None
    on_demand_price: Optional[float] = None
    spot_percentage: float = 0.0  # % of instances that are spot
    
    def get_capacity(self) -> int:
        """Get total capacity"""
        return self.current_instances * self.max_concurrent_per_instance
        
    def get_utilization(self) -> float:
        """Get utilization percentage"""
        capacity = self.get_capacity()
        if capacity == 0:
            return 0.0
        return self.current_load / capacity
        
    def can_accept_request(self) -> bool:
        """Check if pool can accept new request"""
        return (
            self.state == ProviderState.ACTIVE and
            self.current_load < self.get_capacity()
        )


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    # Queue settings
    max_queue_size: int = 1000
    queue_timeout_seconds: int = 300
    
    # Scaling settings
    scale_up_threshold: float = 0.8   # Utilization %
    scale_down_threshold: float = 0.3
    scale_cooldown_seconds: int = 60
    
    # Priority settings
    enable_priority_queue: bool = True
    priority_boost_after_seconds: int = 30  # Boost priority of waiting requests
    
    # Backpressure settings
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.9  # Queue utilization
    
    # Load balancing
    algorithm: str = "weighted_least_connections"
    sticky_sessions: bool = False
    session_timeout_seconds: int = 3600
    
    # Health checks
    health_check_interval_seconds: int = 10
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    
    # Rate limiting
    global_rate_limit_rpm: int = 10000
    per_provider_rate_limit_rpm: Dict[str, int] = field(default_factory=dict)


class PriorityQueueManager:
    """Manages priority queues for each provider"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.queues: Dict[ProviderType, List[QueuedRequest]] = defaultdict(list)
        self.queue_locks: Dict[ProviderType, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Background task for queue management
        self._queue_task = None
        
    async def start(self):
        """Start queue management"""
        self._queue_task = asyncio.create_task(self._queue_management_loop())
        
    async def stop(self):
        """Stop queue management"""
        if self._queue_task:
            self._queue_task.cancel()
            
    async def enqueue(self, provider: ProviderType, request: ProviderRequest,
                     priority: QueuePriority = QueuePriority.NORMAL,
                     callback: Optional[Callable] = None,
                     deadline: Optional[datetime] = None) -> bool:
        """Add request to queue"""
        
        async with self.queue_locks[provider]:
            queue = self.queues[provider]
            
            # Check queue size
            if len(queue) >= self.config.max_queue_size:
                if self.config.enable_backpressure:
                    backpressure_events.add(1, {"provider": provider.value})
                    return False
                    
            # Create queued request
            queued = QueuedRequest(
                request=request,
                priority=priority,
                timestamp=datetime.now(timezone.utc),
                callback=callback,
                deadline=deadline
            )
            
            # Add to heap queue
            heapq.heappush(queue, queued)
            
            # Update metrics
            queue_depth.set(len(queue), {"provider": provider.value})
            
            return True
            
    async def dequeue(self, provider: ProviderType) -> Optional[QueuedRequest]:
        """Get next request from queue"""
        
        async with self.queue_locks[provider]:
            queue = self.queues[provider]
            
            if not queue:
                return None
                
            # Get highest priority request
            queued = heapq.heappop(queue)
            
            # Update metrics
            queue_depth.set(len(queue), {"provider": provider.value})
            
            # Check if expired
            if queued.deadline and datetime.now(timezone.utc) > queued.deadline:
                # Request expired, try next
                return await self.dequeue(provider)
                
            return queued
            
    async def get_queue_stats(self, provider: ProviderType) -> Dict[str, Any]:
        """Get queue statistics"""
        
        async with self.queue_locks[provider]:
            queue = self.queues[provider]
            
            if not queue:
                return {
                    "size": 0,
                    "oldest_age_seconds": 0,
                    "priority_breakdown": {}
                }
                
            now = datetime.now(timezone.utc)
            oldest = min(queue, key=lambda q: q.timestamp)
            
            # Priority breakdown
            priority_counts = defaultdict(int)
            for req in queue:
                priority_counts[req.priority.name] += 1
                
            return {
                "size": len(queue),
                "oldest_age_seconds": (now - oldest.timestamp).total_seconds(),
                "priority_breakdown": dict(priority_counts)
            }
            
    async def _queue_management_loop(self):
        """Background task for queue management"""
        
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Boost priority of old requests
                await self._boost_old_requests()
                
                # Clean expired requests
                await self._clean_expired_requests()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue management error: {e}")
                
    async def _boost_old_requests(self):
        """Boost priority of requests waiting too long"""
        
        if not self.config.priority_boost_after_seconds:
            return
            
        now = datetime.now(timezone.utc)
        boost_cutoff = now - timedelta(seconds=self.config.priority_boost_after_seconds)
        
        for provider in list(self.queues.keys()):
            async with self.queue_locks[provider]:
                queue = self.queues[provider]
                
                # Find requests needing boost
                boosted = []
                remaining = []
                
                while queue:
                    req = heapq.heappop(queue)
                    
                    if req.timestamp < boost_cutoff and req.priority > QueuePriority.HIGH:
                        # Boost priority
                        req.priority = QueuePriority(max(0, req.priority - 1))
                        boosted.append(req)
                    else:
                        remaining.append(req)
                        
                # Rebuild queue
                self.queues[provider] = remaining
                for req in boosted + remaining:
                    heapq.heappush(self.queues[provider], req)
                    
                if boosted:
                    logger.info(f"Boosted {len(boosted)} requests for {provider.value}")
                    
    async def _clean_expired_requests(self):
        """Remove expired requests from queues"""
        
        now = datetime.now(timezone.utc)
        
        for provider in list(self.queues.keys()):
            async with self.queue_locks[provider]:
                queue = self.queues[provider]
                
                # Filter out expired
                valid = []
                expired = 0
                
                while queue:
                    req = heapq.heappop(queue)
                    
                    if req.deadline and now > req.deadline:
                        expired += 1
                        if req.callback:
                            # Notify timeout
                            asyncio.create_task(req.callback(None, "timeout"))
                    else:
                        valid.append(req)
                        
                # Rebuild queue
                self.queues[provider] = []
                for req in valid:
                    heapq.heappush(self.queues[provider], req)
                    
                if expired:
                    logger.warning(f"Expired {expired} requests for {provider.value}")


class ElasticPoolManager:
    """Manages elastic scaling of provider pools"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.pools: Dict[ProviderType, ProviderPool] = {}
        self.last_scale_time: Dict[ProviderType, datetime] = {}
        
        # Scaling task
        self._scaling_task = None
        
    async def start(self):
        """Start pool management"""
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        
    async def stop(self):
        """Stop pool management"""
        if self._scaling_task:
            self._scaling_task.cancel()
            
    def add_pool(self, provider: ProviderType, pool: ProviderPool):
        """Add provider pool"""
        self.pools[provider] = pool
        self.last_scale_time[provider] = datetime.now(timezone.utc)
        
    def get_pool(self, provider: ProviderType) -> Optional[ProviderPool]:
        """Get provider pool"""
        return self.pools.get(provider)
        
    async def _scaling_loop(self):
        """Background scaling decisions"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for provider, pool in self.pools.items():
                    await self._check_scaling(provider, pool)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling error: {e}")
                
    async def _check_scaling(self, provider: ProviderType, pool: ProviderPool):
        """Check if scaling is needed"""
        
        # Check cooldown
        now = datetime.now(timezone.utc)
        last_scale = self.last_scale_time.get(provider)
        if last_scale:
            cooldown = timedelta(seconds=self.config.scale_cooldown_seconds)
            if now - last_scale < cooldown:
                return
                
        utilization = pool.get_utilization()
        
        # Scale up?
        if utilization > self.config.scale_up_threshold:
            if pool.current_instances < pool.max_instances:
                pool.target_instances = min(
                    pool.current_instances + 1,
                    pool.max_instances
                )
                pool.state = ProviderState.SCALING
                self.last_scale_time[provider] = now
                logger.info(
                    f"Scaling up {provider.value}",
                    current=pool.current_instances,
                    target=pool.target_instances
                )
                
        # Scale down?
        elif utilization < self.config.scale_down_threshold:
            if pool.current_instances > pool.min_instances:
                pool.target_instances = max(
                    pool.current_instances - 1,
                    pool.min_instances
                )
                pool.state = ProviderState.SCALING
                self.last_scale_time[provider] = now
                logger.info(
                    f"Scaling down {provider.value}",
                    current=pool.current_instances,
                    target=pool.target_instances
                )
                
    def update_pool_state(self, provider: ProviderType, instances: int):
        """Update pool after scaling"""
        pool = self.pools.get(provider)
        if pool:
            pool.current_instances = instances
            if pool.current_instances == pool.target_instances:
                pool.state = ProviderState.ACTIVE


class AdvancedLoadBalancer:
    """Advanced load balancer with all features"""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        
        # Components
        self.queue_manager = PriorityQueueManager(self.config)
        self.pool_manager = ElasticPoolManager(self.config)
        
        # Session affinity
        self.session_map: Dict[str, ProviderType] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        
        # Rate limiting
        self.rate_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        
        # Health tracking
        self.health_data: Dict[ProviderType, ProviderHealth] = {}
        
    async def start(self):
        """Start load balancer"""
        await self.queue_manager.start()
        await self.pool_manager.start()
        logger.info("Advanced load balancer started")
        
    async def stop(self):
        """Stop load balancer"""
        await self.queue_manager.stop()
        await self.pool_manager.stop()
        logger.info("Advanced load balancer stopped")
        
    async def route_request(self, request: ProviderRequest,
                          available_providers: List[ProviderType],
                          session_id: Optional[str] = None,
                          priority: QueuePriority = QueuePriority.NORMAL) -> Optional[ProviderType]:
        """Route request to best provider"""
        
        with tracer.start_as_current_span("advanced_route") as span:
            span.set_attribute("num_providers", len(available_providers))
            span.set_attribute("priority", priority.name)
            
            # Check session affinity
            if self.config.sticky_sessions and session_id:
                provider = self._check_session_affinity(session_id, available_providers)
                if provider:
                    span.set_attribute("session_affinity", True)
                    return provider
                    
            # Filter by health and capacity
            viable_providers = []
            for provider in available_providers:
                pool = self.pool_manager.get_pool(provider)
                if pool and pool.can_accept_request():
                    if self._check_rate_limit(provider):
                        viable_providers.append(provider)
                        
            if not viable_providers:
                # All providers at capacity or rate limited
                span.set_attribute("all_at_capacity", True)
                
                # Try queueing if enabled
                if self.config.enable_priority_queue:
                    # Pick provider with shortest queue
                    best_provider = await self._find_best_queue(available_providers)
                    if best_provider:
                        success = await self.queue_manager.enqueue(
                            best_provider,
                            request,
                            priority
                        )
                        if success:
                            span.set_attribute("queued", True)
                            return None  # Indicate queued
                            
                return None
                
            # Apply load balancing algorithm
            selected = self._apply_algorithm(viable_providers)
            
            # Update session affinity
            if self.config.sticky_sessions and session_id and selected:
                self.session_map[session_id] = selected
                self.session_timestamps[session_id] = datetime.now(timezone.utc)
                
            # Update metrics
            if selected:
                pool = self.pool_manager.get_pool(selected)
                if pool:
                    pool.current_load += 1
                    provider_utilization.set(
                        pool.get_utilization() * 100,
                        {"provider": selected.value}
                    )
                    
            span.set_attribute("selected_provider", selected.value if selected else "none")
            return selected
            
    def release_connection(self, provider: ProviderType):
        """Release connection after request completes"""
        pool = self.pool_manager.get_pool(provider)
        if pool:
            pool.current_load = max(0, pool.current_load - 1)
            provider_utilization.set(
                pool.get_utilization() * 100,
                {"provider": provider.value}
            )
            
    def _check_session_affinity(self, session_id: str, 
                               available: List[ProviderType]) -> Optional[ProviderType]:
        """Check if session has affinity"""
        
        if session_id not in self.session_map:
            return None
            
        provider = self.session_map[session_id]
        timestamp = self.session_timestamps.get(session_id)
        
        # Check timeout
        if timestamp:
            age = datetime.now(timezone.utc) - timestamp
            if age.total_seconds() > self.config.session_timeout_seconds:
                # Session expired
                del self.session_map[session_id]
                del self.session_timestamps[session_id]
                return None
                
        # Check if provider still available
        if provider in available:
            return provider
            
        return None
        
    def _check_rate_limit(self, provider: ProviderType) -> bool:
        """Check rate limits"""
        
        now = datetime.now(timezone.utc)
        
        # Global rate limit
        global_window = self.rate_windows["__global__"]
        global_window.append(now)
        
        # Remove old entries (older than 1 minute)
        cutoff = now - timedelta(seconds=60)
        while global_window and global_window[0] < cutoff:
            global_window.popleft()
            
        global_rpm = len(global_window)
        if global_rpm >= self.config.global_rate_limit_rpm:
            return False
            
        # Per-provider rate limit
        if provider.value in self.config.per_provider_rate_limit_rpm:
            provider_window = self.rate_windows[provider.value]
            provider_window.append(now)
            
            while provider_window and provider_window[0] < cutoff:
                provider_window.popleft()
                
            provider_rpm = len(provider_window)
            limit = self.config.per_provider_rate_limit_rpm[provider.value]
            
            if provider_rpm >= limit:
                return False
                
        return True
        
    def _apply_algorithm(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        """Apply load balancing algorithm"""
        
        if not providers:
            return None
            
        if self.config.algorithm == "weighted_least_connections":
            return self._weighted_least_connections(providers)
        elif self.config.algorithm == "latency_optimized":
            return self._latency_optimized(providers)
        elif self.config.algorithm == "cost_optimized":
            return self._cost_optimized(providers)
        elif self.config.algorithm == "random":
            return np.random.choice(providers)
        else:
            # Default to round-robin
            return providers[0]
            
    def _weighted_least_connections(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        """Select provider with least weighted connections"""
        
        best_score = float('inf')
        best_provider = providers[0]
        
        for provider in providers:
            pool = self.pool_manager.get_pool(provider)
            if not pool:
                continue
                
            # Weight by capacity and performance
            connections = pool.current_load
            weight = 1.0
            
            # Adjust weight by performance
            if pool.avg_latency_ms > 0:
                weight *= (pool.avg_latency_ms / 1000)  # Penalize slow providers
                
            weight *= (2.0 - pool.success_rate)  # Penalize unreliable providers
            
            weighted_connections = connections * weight
            
            if weighted_connections < best_score:
                best_score = weighted_connections
                best_provider = provider
                
        return best_provider
        
    def _latency_optimized(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        """Select provider with lowest latency"""
        
        best_latency = float('inf')
        best_provider = providers[0]
        
        for provider in providers:
            pool = self.pool_manager.get_pool(provider)
            if pool and pool.avg_latency_ms < best_latency:
                best_latency = pool.avg_latency_ms
                best_provider = provider
                
        return best_provider
        
    def _cost_optimized(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        """Select cheapest provider (considering spot instances)"""
        
        # Simplified - would integrate with cost data
        cost_priority = [
            ProviderType.OLLAMA,     # Free (local)
            ProviderType.TOGETHER,   # Cheapest
            ProviderType.OPENAI,     # Medium
            ProviderType.ANTHROPIC   # Most expensive
        ]
        
        for provider in cost_priority:
            if provider in providers:
                return provider
                
        return providers[0]
        
    async def _find_best_queue(self, providers: List[ProviderType]) -> Optional[ProviderType]:
        """Find provider with best queue characteristics"""
        
        best_score = float('inf')
        best_provider = None
        
        for provider in providers:
            stats = await self.queue_manager.get_queue_stats(provider)
            
            # Score based on queue size and age
            score = stats["size"] + (stats["oldest_age_seconds"] / 60)
            
            if score < best_score:
                best_score = score
                best_provider = provider
                
        return best_provider
        
    async def process_queued_requests(self, provider: ProviderType, 
                                    capacity: int) -> List[QueuedRequest]:
        """Get queued requests up to capacity"""
        
        requests = []
        
        for _ in range(capacity):
            queued = await self.queue_manager.dequeue(provider)
            if not queued:
                break
            requests.append(queued)
            
        return requests


# Export main classes
__all__ = [
    "QueuePriority",
    "ProviderState",
    "QueuedRequest",
    "ProviderPool",
    "LoadBalancerConfig",
    "PriorityQueueManager",
    "ElasticPoolManager",
    "AdvancedLoadBalancer"
]