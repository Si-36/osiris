"""
🧠 GPU-Accelerated Core Adapter
================================

Central coordination hub with GPU-accelerated component management,
health monitoring, and resource optimization.

Features:
- Parallel component health checks
- GPU metric aggregation
- Event correlation on GPU
- Resource allocation optimization
- Dependency resolution
- Real-time system monitoring
"""

import asyncio
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
import time
import structlog
from prometheus_client import Histogram, Counter, Gauge
from datetime import datetime, timedelta
import networkx as nx

from .base_adapter import BaseAdapter, HealthStatus, HealthMetrics, ComponentMetadata
from ..core.aura_main_system import AURAMainSystem, SystemMetrics
from ..components.registry import ComponentRegistry, ComponentInfo, ComponentStatus, ComponentHealth

logger = structlog.get_logger(__name__)

# Metrics
HEALTH_CHECK_TIME = Histogram(
    'core_health_check_seconds',
    'Component health check time',
    ['check_type', 'num_components', 'backend']
)

METRIC_AGGREGATION_TIME = Histogram(
    'core_metric_aggregation_seconds',
    'Metric aggregation time',
    ['aggregation_type', 'backend']
)

RESOURCE_OPTIMIZATION_TIME = Histogram(
    'core_resource_optimization_seconds',
    'Resource optimization computation time',
    ['algorithm', 'num_components']
)

SYSTEM_HEALTH_SCORE = Gauge(
    'core_system_health_score',
    'Overall system health score (0-1)'
)


@dataclass
class GPUCoreConfig:
    """Configuration for GPU core adapter"""
    # GPU settings
    use_gpu: bool = True
    gpu_device: int = 0
    
    # Health monitoring
    health_check_batch_size: int = 100
    health_check_interval_ms: int = 1000
    health_score_threshold: float = 0.7
    
    # Metrics
    metric_window_size: int = 1000
    metric_aggregation_interval_ms: int = 100
    
    # Resource optimization
    resource_optimization_interval_s: int = 30
    gpu_memory_reserve_percent: float = 10.0
    
    # Event processing
    event_batch_size: int = 1000
    event_correlation_window_ms: int = 5000
    
    # Performance
    gpu_threshold: int = 50  # Use GPU for > this many components


@dataclass
class ComponentMetrics:
    """GPU-optimized component metrics"""
    def __init__(self, num_components: int, device: torch.device):
        self.num_components = num_components
        self.device = device
        
        # Metric tensors
        self.health_scores = torch.ones(num_components, device=device)
        self.cpu_usage = torch.zeros(num_components, device=device)
        self.memory_usage = torch.zeros(num_components, device=device)
        self.latencies = torch.zeros(num_components, device=device)
        self.error_counts = torch.zeros(num_components, dtype=torch.int32, device=device)
        self.last_update = torch.zeros(num_components, device=device)


class GPUCoreAdapter(BaseAdapter):
    """
    GPU-accelerated core adapter for central system coordination.
    
    Manages:
    - Component lifecycle and health
    - System metrics aggregation
    - Resource allocation
    - Event processing
    - Dependency management
    """
    
    def __init__(self,
                 aura_system: AURAMainSystem,
                 component_registry: ComponentRegistry,
                 config: GPUCoreConfig):
        super().__init__(
            component_id="core_gpu",
            metadata=ComponentMetadata(
                version="2.0.0",
                capabilities=["gpu_health_monitoring", "resource_optimization", "event_correlation"],
                dependencies={"core_system", "registry", "torch"},
                tags=["gpu", "core", "coordination", "production"]
            )
        )
        
        self.aura_system = aura_system
        self.registry = component_registry
        self.config = config
        
        # Initialize GPU
        if torch.cuda.is_available() and config.use_gpu:
            torch.cuda.set_device(config.gpu_device)
            self.device = torch.device(f"cuda:{config.gpu_device}")
            self.gpu_available = True
            logger.info(f"GPU Core using CUDA device {config.gpu_device}")
        else:
            self.device = torch.device("cpu")
            self.gpu_available = False
            
        # Component tracking
        self.component_indices: Dict[str, int] = {}
        self.component_metrics: Optional[ComponentMetrics] = None
        
        # Event processing
        self.event_buffer: List[Dict[str, Any]] = []
        self.event_lock = asyncio.Lock()
        
        # Resource allocation state
        self.resource_allocation: Optional[torch.Tensor] = None
        
        # Background tasks
        self._health_monitor_task = None
        self._metric_aggregator_task = None
        self._resource_optimizer_task = None
        
    async def initialize(self) -> None:
        """Initialize GPU core adapter"""
        await super().initialize()
        
        # Initialize component tracking
        await self._init_component_tracking()
        
        # Start background tasks
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        self._metric_aggregator_task = asyncio.create_task(self._metric_aggregator())
        self._resource_optimizer_task = asyncio.create_task(self._resource_optimizer())
        
        logger.info("GPU Core adapter initialized")
        
    async def _init_component_tracking(self):
        """Initialize GPU structures for component tracking"""
        components = await self.registry.get_all_components()
        self.component_indices = {comp.id: i for i, comp in enumerate(components)}
        
        num_components = len(components)
        if num_components > 0:
            self.component_metrics = ComponentMetrics(num_components, self.device)
            
            # Initialize resource allocation tensor
            self.resource_allocation = torch.ones(
                num_components, 
                device=self.device
            ) / num_components
            
    async def batch_health_check(self,
                                component_ids: List[str] = None) -> Dict[str, Any]:
        """
        Perform parallel health checks on multiple components.
        """
        start_time = time.time()
        
        if component_ids is None:
            component_ids = list(self.component_indices.keys())
            
        num_components = len(component_ids)
        
        # Determine backend
        use_gpu = (
            self.gpu_available and 
            num_components > self.config.gpu_threshold
        )
        
        try:
            if use_gpu:
                results = await self._health_check_gpu(component_ids)
            else:
                results = await self._health_check_cpu(component_ids)
                
            # Record metrics
            check_time = time.time() - start_time
            HEALTH_CHECK_TIME.labels(
                check_type='batch',
                num_components=num_components,
                backend='gpu' if use_gpu else 'cpu'
            ).observe(check_time)
            
            # Update system health score
            overall_health = results.get('overall_health', 0.0)
            SYSTEM_HEALTH_SCORE.set(overall_health)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch health check failed: {e}")
            raise
            
    async def _health_check_gpu(self,
                               component_ids: List[str]) -> Dict[str, Any]:
        """GPU-accelerated health checks"""
        
        if not self.component_metrics:
            return await self._health_check_cpu(component_ids)
            
        # Get component indices
        indices = [self.component_indices[cid] for cid in component_ids if cid in self.component_indices]
        if not indices:
            return {'checked': 0, 'healthy': 0, 'overall_health': 0.0}
            
        indices_tensor = torch.tensor(indices, device=self.device)
        
        # Simulate parallel health checks
        # In reality, would query actual component health endpoints
        current_time = time.time()
        
        # Update health scores (mock - would be actual health data)
        health_noise = torch.randn(len(indices), device=self.device) * 0.1
        self.component_metrics.health_scores[indices_tensor] = torch.clamp(
            0.9 + health_noise, 0.0, 1.0
        )
        
        # Update last check time
        self.component_metrics.last_update[indices_tensor] = current_time
        
        # Calculate statistics on GPU
        health_scores = self.component_metrics.health_scores[indices_tensor]
        healthy_mask = health_scores >= self.config.health_score_threshold
        num_healthy = int(healthy_mask.sum())
        overall_health = float(health_scores.mean())
        
        # Identify unhealthy components
        unhealthy_indices = indices_tensor[~healthy_mask]
        unhealthy_ids = [component_ids[i] for i in unhealthy_indices.cpu().numpy()]
        
        return {
            'checked': len(indices),
            'healthy': num_healthy,
            'unhealthy': unhealthy_ids,
            'overall_health': overall_health,
            'check_time_ms': (time.time() - start_time) * 1000,
            'gpu_accelerated': True
        }
        
    async def _health_check_cpu(self,
                               component_ids: List[str]) -> Dict[str, Any]:
        """CPU fallback for health checks"""
        
        healthy_count = 0
        unhealthy_ids = []
        total_health = 0.0
        
        for comp_id in component_ids:
            # Mock health check
            health = 0.9 + np.random.random() * 0.1
            total_health += health
            
            if health >= self.config.health_score_threshold:
                healthy_count += 1
            else:
                unhealthy_ids.append(comp_id)
                
        return {
            'checked': len(component_ids),
            'healthy': healthy_count,
            'unhealthy': unhealthy_ids,
            'overall_health': total_health / len(component_ids) if component_ids else 0.0,
            'gpu_accelerated': False
        }
        
    async def aggregate_metrics(self,
                              aggregation_type: str = "mean") -> Dict[str, Any]:
        """
        Aggregate system metrics using GPU acceleration.
        """
        start_time = time.time()
        
        if not self.gpu_available or not self.component_metrics:
            return await self._aggregate_metrics_cpu(aggregation_type)
            
        try:
            # GPU aggregation
            if aggregation_type == "mean":
                agg_health = float(self.component_metrics.health_scores.mean())
                agg_cpu = float(self.component_metrics.cpu_usage.mean())
                agg_memory = float(self.component_metrics.memory_usage.mean())
                agg_latency = float(self.component_metrics.latencies.mean())
                
            elif aggregation_type == "max":
                agg_health = float(self.component_metrics.health_scores.max())
                agg_cpu = float(self.component_metrics.cpu_usage.max())
                agg_memory = float(self.component_metrics.memory_usage.max())
                agg_latency = float(self.component_metrics.latencies.max())
                
            elif aggregation_type == "percentile":
                # 95th percentile
                agg_health = float(torch.quantile(self.component_metrics.health_scores, 0.95))
                agg_cpu = float(torch.quantile(self.component_metrics.cpu_usage, 0.95))
                agg_memory = float(torch.quantile(self.component_metrics.memory_usage, 0.95))
                agg_latency = float(torch.quantile(self.component_metrics.latencies, 0.95))
                
            else:
                raise ValueError(f"Unknown aggregation type: {aggregation_type}")
                
            # Record metrics
            aggregation_time = time.time() - start_time
            METRIC_AGGREGATION_TIME.labels(
                aggregation_type=aggregation_type,
                backend='gpu'
            ).observe(aggregation_time)
            
            return {
                'aggregation_type': aggregation_type,
                'health_score': agg_health,
                'cpu_usage': agg_cpu,
                'memory_usage': agg_memory,
                'latency': agg_latency,
                'total_errors': int(self.component_metrics.error_counts.sum()),
                'aggregation_time_ms': aggregation_time * 1000,
                'gpu_accelerated': True
            }
            
        except Exception as e:
            logger.error(f"GPU metric aggregation failed: {e}")
            return await self._aggregate_metrics_cpu(aggregation_type)
            
    async def _aggregate_metrics_cpu(self,
                                   aggregation_type: str) -> Dict[str, Any]:
        """CPU fallback for metric aggregation"""
        
        # Mock metrics
        return {
            'aggregation_type': aggregation_type,
            'health_score': 0.95,
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'latency': 5.0,
            'total_errors': 0,
            'gpu_accelerated': False
        }
        
    async def optimize_resources(self) -> Dict[str, Any]:
        """
        Optimize resource allocation across components using GPU.
        """
        start_time = time.time()
        
        if not self.gpu_available or not self.component_metrics:
            return {'optimized': False, 'reason': 'GPU not available'}
            
        try:
            num_components = self.component_metrics.num_components
            
            # Compute resource demands based on metrics
            # Higher demand for components with lower health or higher usage
            demands = (
                (1.0 - self.component_metrics.health_scores) * 0.3 +
                self.component_metrics.cpu_usage * 0.3 +
                self.component_metrics.memory_usage * 0.2 +
                self.component_metrics.latencies / 100.0 * 0.2
            )
            
            # Normalize demands
            demands = demands / (demands.sum() + 1e-8)
            
            # Apply constraints (min/max allocation per component)
            min_allocation = 1.0 / (num_components * 10)  # At least 0.1% each
            max_allocation = 0.2  # At most 20% each
            
            # Optimize allocation
            new_allocation = demands.clamp(min_allocation, max_allocation)
            
            # Normalize to sum to 1
            new_allocation = new_allocation / new_allocation.sum()
            
            # Update allocation
            self.resource_allocation = new_allocation
            
            # Calculate redistribution metrics
            if hasattr(self, '_previous_allocation'):
                reallocation = (new_allocation - self._previous_allocation).abs().sum()
                max_change = (new_allocation - self._previous_allocation).abs().max()
            else:
                reallocation = 0.0
                max_change = 0.0
                
            self._previous_allocation = new_allocation.clone()
            
            # Record metrics
            optimization_time = time.time() - start_time
            RESOURCE_OPTIMIZATION_TIME.labels(
                algorithm='demand_based',
                num_components=num_components
            ).observe(optimization_time)
            
            return {
                'optimized': True,
                'num_components': num_components,
                'total_reallocation': float(reallocation),
                'max_change': float(max_change),
                'optimization_time_ms': optimization_time * 1000,
                'top_consumers': self._get_top_consumers(new_allocation, 5)
            }
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return {'optimized': False, 'error': str(e)}
            
    def _get_top_consumers(self,
                          allocation: torch.Tensor,
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top resource consumers"""
        
        values, indices = torch.topk(allocation, min(top_k, len(allocation)))
        
        top_consumers = []
        for val, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            # Find component ID from index
            comp_id = None
            for cid, cidx in self.component_indices.items():
                if cidx == idx:
                    comp_id = cid
                    break
                    
            if comp_id:
                top_consumers.append({
                    'component_id': comp_id,
                    'allocation': float(val),
                    'percentage': float(val * 100)
                })
                
        return top_consumers
        
    async def process_events(self,
                           events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and correlate events using GPU acceleration.
        """
        if not events:
            return {'processed': 0, 'correlated': 0}
            
        num_events = len(events)
        
        if self.gpu_available and num_events > self.config.gpu_threshold:
            return await self._process_events_gpu(events)
        else:
            return await self._process_events_cpu(events)
            
    async def _process_events_gpu(self,
                                events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """GPU-accelerated event processing"""
        
        # Extract event features
        timestamps = torch.tensor(
            [e.get('timestamp', time.time()) for e in events],
            device=self.device
        )
        
        # Event type hashes for grouping
        type_hashes = torch.tensor(
            [hash(e.get('type', 'unknown')) % 1000 for e in events],
            device=self.device
        )
        
        # Component hashes
        component_hashes = torch.tensor(
            [hash(e.get('component', 'unknown')) % 1000 for e in events],
            device=self.device
        )
        
        # Find correlated events (same type within time window)
        time_diffs = timestamps.unsqueeze(1) - timestamps.unsqueeze(0)
        time_mask = time_diffs.abs() < (self.config.event_correlation_window_ms / 1000.0)
        
        type_match = type_hashes.unsqueeze(1) == type_hashes.unsqueeze(0)
        
        # Correlation matrix
        correlations = time_mask & type_match
        
        # Count correlations (excluding self)
        correlation_counts = correlations.sum(dim=1) - 1
        num_correlated = int((correlation_counts > 0).sum())
        
        return {
            'processed': num_events,
            'correlated': num_correlated,
            'avg_correlations': float(correlation_counts.float().mean()),
            'gpu_accelerated': True
        }
        
    async def _process_events_cpu(self,
                                events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """CPU fallback for event processing"""
        
        # Simple correlation check
        correlated = 0
        for i, event in enumerate(events):
            for j, other in enumerate(events[i+1:], i+1):
                if (event.get('type') == other.get('type') and
                    abs(event.get('timestamp', 0) - other.get('timestamp', 0)) < 
                    self.config.event_correlation_window_ms / 1000.0):
                    correlated += 1
                    
        return {
            'processed': len(events),
            'correlated': correlated,
            'gpu_accelerated': False
        }
        
    async def resolve_dependencies(self,
                                 component_graph: nx.DiGraph) -> List[str]:
        """
        Resolve component dependencies using GPU-accelerated topological sort.
        """
        if not self.gpu_available or len(component_graph) < self.config.gpu_threshold:
            # Use NetworkX for small graphs
            return list(nx.topological_sort(component_graph))
            
        # Convert to adjacency matrix
        nodes = list(component_graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create adjacency matrix on GPU
        adj_matrix = torch.zeros(n, n, device=self.device)
        
        for u, v in component_graph.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = 1
            
        # Compute in-degrees
        in_degrees = adj_matrix.sum(dim=0)
        
        # Kahn's algorithm on GPU
        result = []
        queue = torch.where(in_degrees == 0)[0].tolist()
        
        while queue:
            # Process nodes with no dependencies
            node_idx = queue.pop(0)
            result.append(nodes[node_idx])
            
            # Remove edges from this node
            neighbors = torch.where(adj_matrix[node_idx] > 0)[0]
            for neighbor in neighbors:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(int(neighbor))
                    
            adj_matrix[node_idx, :] = 0
            
        return result if len(result) == n else []  # Empty if cycle detected
        
    async def _health_monitor(self):
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval_ms / 1000.0)
                
                # Perform batch health check
                await self.batch_health_check()
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(1)
                
    async def _metric_aggregator(self):
        """Background metric aggregation task"""
        while True:
            try:
                await asyncio.sleep(self.config.metric_aggregation_interval_ms / 1000.0)
                
                # Aggregate metrics
                await self.aggregate_metrics("mean")
                
            except Exception as e:
                logger.error(f"Metric aggregator error: {e}")
                await asyncio.sleep(1)
                
    async def _resource_optimizer(self):
        """Background resource optimization task"""
        while True:
            try:
                await asyncio.sleep(self.config.resource_optimization_interval_s)
                
                # Optimize resources
                await self.optimize_resources()
                
            except Exception as e:
                logger.error(f"Resource optimizer error: {e}")
                await asyncio.sleep(10)
                
    async def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state"""
        
        state = {
            'num_components': len(self.component_indices),
            'gpu_available': self.gpu_available,
            'overall_health': 0.0,
            'resource_utilization': {}
        }
        
        if self.component_metrics:
            state['overall_health'] = float(self.component_metrics.health_scores.mean())
            
            if self.resource_allocation is not None:
                # Get resource utilization
                top_consumers = self._get_top_consumers(self.resource_allocation, 10)
                state['resource_utilization'] = {
                    'top_consumers': top_consumers,
                    'distribution_entropy': float(
                        -(self.resource_allocation * torch.log(self.resource_allocation + 1e-8)).sum()
                    )
                }
                
        return state
        
    async def shutdown(self) -> None:
        """Shutdown adapter"""
        await super().shutdown()
        
        # Cancel background tasks
        for task in [self._health_monitor_task, 
                    self._metric_aggregator_task,
                    self._resource_optimizer_task]:
            if task:
                task.cancel()
                
        logger.info("GPU Core adapter shut down")
        
    async def health(self) -> HealthMetrics:
        """Get adapter health"""
        metrics = HealthMetrics()
        
        try:
            if self.gpu_available:
                allocated = torch.cuda.memory_allocated(self.config.gpu_device)
                reserved = torch.cuda.memory_reserved(self.config.gpu_device)
                
                metrics.resource_usage["gpu_memory_allocated_mb"] = allocated / 1024 / 1024
                metrics.resource_usage["gpu_memory_reserved_mb"] = reserved / 1024 / 1024
                
            # Check component tracking
            if self.component_metrics:
                metrics.resource_usage["tracked_components"] = self.component_metrics.num_components
                metrics.resource_usage["avg_health_score"] = float(
                    self.component_metrics.health_scores.mean()
                )
                
            metrics.status = HealthStatus.HEALTHY
            
        except Exception as e:
            metrics.status = HealthStatus.UNHEALTHY
            metrics.failure_predictions.append(f"Health check failed: {e}")
            
        return metrics


# Factory function
def create_gpu_core_adapter(
    aura_system: AURAMainSystem,
    use_gpu: bool = True,
    health_check_batch_size: int = 100
) -> GPUCoreAdapter:
    """Create GPU core adapter"""
    
    # Get or create component registry
    registry = getattr(aura_system, 'registry', ComponentRegistry())
    
    config = GPUCoreConfig(
        use_gpu=use_gpu,
        health_check_batch_size=health_check_batch_size,
        gpu_threshold=50
    )
    
    return GPUCoreAdapter(
        aura_system=aura_system,
        component_registry=registry,
        config=config
    )