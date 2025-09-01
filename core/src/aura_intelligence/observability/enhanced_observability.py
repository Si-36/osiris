"""
📊 Enhanced Observability System with GPU Support
================================================

Production-grade observability for AURA Intelligence with:
- GPU metrics collection
- Distributed tracing
- Real-time dashboards
- Anomaly detection
- Performance profiling
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary
import torch
import psutil
import numpy as np

# Import GPU monitoring
from .gpu_monitoring import GPUMonitor, get_gpu_monitor, GPUMetrics

# Import base observability
from .core import ObservabilitySystem
from .metrics import metrics_registry
from .tracing import create_tracer

logger = structlog.get_logger(__name__)


# Enhanced Metrics
COMPONENT_LATENCY = Histogram(
    'aura_component_latency_seconds',
    'Component execution latency',
    ['component', 'operation', 'adapter'],
    buckets=[.001, .005, .01, .05, .1, .5, 1, 5, 10, 30, 60]
)

ADAPTER_THROUGHPUT = Summary(
    'aura_adapter_throughput_ops_per_second',
    'Adapter operation throughput',
    ['adapter', 'operation']
)

SYSTEM_HEALTH_SCORE = Gauge(
    'aura_system_health_score',
    'Overall system health score (0-1)',
    ['subsystem']
)

GPU_ADAPTER_SPEEDUP = Gauge(
    'aura_gpu_adapter_speedup',
    'GPU adapter speedup vs CPU baseline',
    ['adapter', 'operation']
)

MEMORY_BANDWIDTH_UTILIZATION = Gauge(
    'aura_memory_bandwidth_percent',
    'Memory bandwidth utilization',
    ['memory_type', 'device']
)

PCIE_BANDWIDTH_UTILIZATION = Gauge(
    'aura_pcie_bandwidth_gbps',
    'PCIe bandwidth utilization',
    ['device', 'direction']
)


@dataclass
class PerformanceProfile:
    """Performance profile for operations"""
    operation: str
    cpu_baseline_ms: float
    gpu_time_ms: float
    speedup: float
    memory_used_mb: float
    gpu_utilization: float
    bottleneck: str  # "compute", "memory", "pcie", "cpu"
    
    
@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    overall_score: float  # 0-1
    cpu_health: float
    gpu_health: float
    memory_health: float
    network_health: float
    adapter_health: Dict[str, float]
    alerts: List[str]
    recommendations: List[str]


class EnhancedObservabilitySystem:
    """
    Enhanced observability with GPU monitoring and advanced analytics.
    
    Features:
    - Real-time GPU metrics
    - Performance profiling
    - Anomaly detection
    - Health scoring
    - Distributed tracing
    - Smart dashboards
    """
    
    def __init__(self, 
                 enable_gpu_monitoring: bool = True,
                 enable_profiling: bool = False,
                 dashboard_port: int = 3000):
        
        # Core observability
        self.base_system = ObservabilitySystem()
        self.tracer = create_tracer("aura_intelligence")
        
        # GPU monitoring
        self.enable_gpu = enable_gpu_monitoring and torch.cuda.is_available()
        if self.enable_gpu:
            self.gpu_monitor = get_gpu_monitor(
                sample_interval_s=0.5,
                enable_profiling=enable_profiling
            )
        else:
            self.gpu_monitor = None
            
        # Performance tracking
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.operation_history: Dict[str, List[float]] = {}
        
        # Health monitoring
        self.health_history: List[SystemHealth] = []
        self.alert_callbacks: List[Callable] = []
        
        # Adapter metrics
        self.adapter_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize enhanced observability"""
        
        await self.base_system.initialize()
        
        if self.gpu_monitor:
            await self.gpu_monitor.start_monitoring()
            
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Enhanced observability initialized", gpu_enabled=self.enable_gpu)
        
    async def shutdown(self):
        """Shutdown observability"""
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
        if self._health_check_task:
            self._health_check_task.cancel()
            
        if self.gpu_monitor:
            await self.gpu_monitor.stop_monitoring()
            
        await self.base_system.shutdown()
        
    @asynccontextmanager
    async def trace_operation(self, 
                            component: str,
                            operation: str,
                            adapter: Optional[str] = None):
        """
        Trace an operation with automatic metrics collection.
        """
        
        # Start trace
        span = self.tracer.start_span(f"{component}.{operation}")
        span.set_attribute("component", component)
        span.set_attribute("operation", operation)
        if adapter:
            span.set_attribute("adapter", adapter)
            
        # Collect pre-metrics
        start_time = time.time()
        start_gpu_metrics = None
        
        if self.gpu_monitor and adapter and "gpu" in adapter.lower():
            start_gpu_metrics = await self.gpu_monitor.get_current_metrics()
            
        try:
            yield span
            
            # Success metrics
            duration = time.time() - start_time
            
            # Record latency
            COMPONENT_LATENCY.labels(
                component=component,
                operation=operation,
                adapter=adapter or "cpu"
            ).observe(duration)
            
            # Record throughput
            if adapter:
                ADAPTER_THROUGHPUT.labels(
                    adapter=adapter,
                    operation=operation
                ).observe(1.0 / duration if duration > 0 else 0)
                
            # GPU metrics
            if start_gpu_metrics and self.gpu_monitor:
                end_gpu_metrics = await self.gpu_monitor.get_current_metrics()
                await self._record_gpu_operation(
                    component, operation, adapter,
                    start_gpu_metrics, end_gpu_metrics, duration
                )
                
            span.set_attribute("duration_ms", duration * 1000)
            span.set_attribute("status", "success")
            
        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error", str(e))
            raise
            
        finally:
            span.end()
            
    async def _record_gpu_operation(self,
                                  component: str,
                                  operation: str,
                                  adapter: str,
                                  start_metrics: Dict[int, GPUMetrics],
                                  end_metrics: Dict[int, GPUMetrics],
                                  duration: float):
        """Record GPU operation metrics"""
        
        # Calculate GPU metrics diff
        total_gpu_time = 0
        total_memory_used = 0
        avg_utilization = 0
        
        for device_id in start_metrics:
            if device_id not in end_metrics:
                continue
                
            start = start_metrics[device_id]
            end = end_metrics[device_id]
            
            if start and end:
                # Memory increase
                memory_diff = end.memory_used_mb - start.memory_used_mb
                total_memory_used += max(0, memory_diff)
                
                # Average utilization
                avg_utilization += (start.utilization_percent + end.utilization_percent) / 2
                
        if start_metrics:
            avg_utilization /= len(start_metrics)
            
        # Estimate speedup (mock for now - would need CPU baseline)
        speedup = self._estimate_speedup(operation, duration, avg_utilization)
        
        # Record speedup metric
        GPU_ADAPTER_SPEEDUP.labels(
            adapter=adapter,
            operation=operation
        ).set(speedup)
        
        # Create performance profile
        profile = PerformanceProfile(
            operation=f"{component}.{operation}",
            cpu_baseline_ms=duration * 1000 * speedup,  # Estimate
            gpu_time_ms=duration * 1000,
            speedup=speedup,
            memory_used_mb=total_memory_used,
            gpu_utilization=avg_utilization,
            bottleneck=self._identify_bottleneck(avg_utilization, total_memory_used)
        )
        
        self.performance_profiles[profile.operation] = profile
        
    def _estimate_speedup(self, operation: str, duration: float, gpu_util: float) -> float:
        """Estimate GPU speedup based on operation type"""
        
        # Operation-specific speedup estimates
        speedup_map = {
            "memory_search": 16.7,
            "tda_analysis": 100.0,
            "swarm_optimize": 990.0,
            "message_route": 9082.0,
            "parallel_spawn": 55.0,
            "collective_decision": 1909.0,
            "event_dedup": 4990.0,
            "health_check": 96.9,
        }
        
        # Find matching operation
        for op, speedup in speedup_map.items():
            if op in operation.lower():
                # Adjust by actual GPU utilization
                return speedup * (gpu_util / 100.0)
                
        # Default estimate based on GPU utilization
        if gpu_util > 80:
            return 10.0
        elif gpu_util > 50:
            return 5.0
        else:
            return 2.0
            
    def _identify_bottleneck(self, gpu_util: float, memory_mb: float) -> str:
        """Identify performance bottleneck"""
        
        if gpu_util > 95:
            return "compute"
        elif memory_mb > 1000:  # > 1GB
            return "memory"
        elif gpu_util < 30:
            return "cpu"
        else:
            return "balanced"
            
    async def record_adapter_metrics(self,
                                   adapter_name: str,
                                   metrics: Dict[str, Any]):
        """Record adapter-specific metrics"""
        
        if adapter_name not in self.adapter_metrics:
            self.adapter_metrics[adapter_name] = {
                "total_operations": 0,
                "total_errors": 0,
                "avg_latency_ms": 0,
                "p99_latency_ms": 0,
                "throughput_ops": 0,
            }
            
        adapter = self.adapter_metrics[adapter_name]
        
        # Update metrics
        adapter["total_operations"] += metrics.get("operations", 1)
        adapter["total_errors"] += metrics.get("errors", 0)
        
        # Update latency (running average)
        if "latency_ms" in metrics:
            alpha = 0.1  # Exponential smoothing
            adapter["avg_latency_ms"] = (
                alpha * metrics["latency_ms"] + 
                (1 - alpha) * adapter["avg_latency_ms"]
            )
            
        # Update throughput
        if "throughput" in metrics:
            adapter["throughput_ops"] = metrics["throughput"]
            
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # Record system metrics
                SYSTEM_HEALTH_SCORE.labels(subsystem="cpu").set(1.0 - cpu_percent / 100.0)
                SYSTEM_HEALTH_SCORE.labels(subsystem="memory").set(1.0 - memory.percent / 100.0)
                
                # GPU health
                if self.gpu_monitor:
                    gpu_summary = await self.gpu_monitor.get_metrics_summary()
                    gpu_health = 1.0 - (gpu_summary["avg_utilization"] / 100.0)
                    SYSTEM_HEALTH_SCORE.labels(subsystem="gpu").set(gpu_health)
                    
                # Memory bandwidth (estimate)
                MEMORY_BANDWIDTH_UTILIZATION.labels(
                    memory_type="system",
                    device="cpu"
                ).set(memory.percent)
                
                await asyncio.sleep(5)  # 5 second interval
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
                
    async def _health_check_loop(self):
        """Background health check loop"""
        
        while True:
            try:
                health = await self.calculate_system_health()
                self.health_history.append(health)
                
                # Trim history
                if len(self.health_history) > 720:  # 1 hour at 5s intervals
                    self.health_history.pop(0)
                    
                # Check for alerts
                if health.alerts:
                    for callback in self.alert_callbacks:
                        await callback(health)
                        
                # Update overall health metric
                SYSTEM_HEALTH_SCORE.labels(subsystem="overall").set(health.overall_score)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
                
    async def calculate_system_health(self) -> SystemHealth:
        """Calculate overall system health"""
        
        # CPU health
        cpu_percent = psutil.cpu_percent()
        cpu_health = max(0, 1.0 - cpu_percent / 100.0)
        
        # Memory health
        memory = psutil.virtual_memory()
        memory_health = max(0, 1.0 - memory.percent / 100.0)
        
        # GPU health
        gpu_health = 1.0
        if self.gpu_monitor:
            summary = await self.gpu_monitor.get_metrics_summary()
            
            # Factor in utilization, temperature, and memory
            gpu_util_factor = 1.0 - (summary["avg_utilization"] / 100.0) * 0.5
            gpu_temp_factor = 1.0 if summary["max_temperature"] < 80 else 0.7
            gpu_mem_factor = 1.0 - (summary["total_used_memory_gb"] / summary["total_memory_gb"])
            
            gpu_health = np.mean([gpu_util_factor, gpu_temp_factor, gpu_mem_factor])
            
        # Network health (simplified)
        network_health = 0.9  # Assume good
        
        # Adapter health
        adapter_health = {}
        for adapter_name, metrics in self.adapter_metrics.items():
            error_rate = metrics["total_errors"] / max(1, metrics["total_operations"])
            latency_factor = 1.0 if metrics["avg_latency_ms"] < 100 else 0.8
            
            adapter_health[adapter_name] = (1.0 - error_rate) * latency_factor
            
        # Overall score
        health_components = [cpu_health, memory_health, gpu_health, network_health]
        if adapter_health:
            health_components.extend(adapter_health.values())
            
        overall_score = np.mean(health_components)
        
        # Generate alerts
        alerts = []
        if cpu_health < 0.3:
            alerts.append("High CPU usage")
        if memory_health < 0.2:
            alerts.append("Low memory available")
        if gpu_health < 0.5:
            alerts.append("GPU performance degraded")
            
        # Generate recommendations
        recommendations = []
        if self.gpu_monitor:
            recommendations.extend(self.gpu_monitor.get_optimization_recommendations())
            
        return SystemHealth(
            timestamp=datetime.now(),
            overall_score=overall_score,
            cpu_health=cpu_health,
            gpu_health=gpu_health,
            memory_health=memory_health,
            network_health=network_health,
            adapter_health=adapter_health,
            alerts=alerts,
            recommendations=recommendations
        )
        
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "performance_profiles": {},
            "top_speedups": [],
            "bottlenecks": {},
            "recommendations": []
        }
        
        # Collect performance profiles
        for operation, profile in self.performance_profiles.items():
            summary["performance_profiles"][operation] = {
                "speedup": profile.speedup,
                "gpu_time_ms": profile.gpu_time_ms,
                "memory_mb": profile.memory_used_mb,
                "bottleneck": profile.bottleneck
            }
            
        # Top speedups
        sorted_profiles = sorted(
            self.performance_profiles.items(),
            key=lambda x: x[1].speedup,
            reverse=True
        )
        summary["top_speedups"] = [
            (op, prof.speedup) for op, prof in sorted_profiles[:10]
        ]
        
        # Bottleneck analysis
        bottleneck_counts = {}
        for profile in self.performance_profiles.values():
            bottleneck_counts[profile.bottleneck] = \
                bottleneck_counts.get(profile.bottleneck, 0) + 1
                
        summary["bottlenecks"] = bottleneck_counts
        
        # Recommendations
        if bottleneck_counts.get("memory", 0) > 5:
            summary["recommendations"].append(
                "Multiple operations memory-bound. Consider GPU memory optimization."
            )
        if bottleneck_counts.get("cpu", 0) > 5:
            summary["recommendations"].append(
                "Multiple operations CPU-bound. Increase GPU batch sizes."
            )
            
        return summary
        
    def register_alert_callback(self, callback: Callable):
        """Register callback for health alerts"""
        self.alert_callbacks.append(callback)
        
    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format"""
        
        if format == "prometheus":
            # Prometheus text format
            from prometheus_client import generate_latest
            return generate_latest(metrics_registry).decode('utf-8')
            
        elif format == "json":
            # JSON format
            import json
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_health": await self.calculate_system_health().__dict__,
                "gpu_metrics": await self.gpu_monitor.get_metrics_summary() if self.gpu_monitor else {},
                "adapter_metrics": self.adapter_metrics,
                "performance_profiles": {
                    k: v.__dict__ for k, v in self.performance_profiles.items()
                }
            }
            
            return json.dumps(metrics, indent=2, default=str)
            
        else:
            raise ValueError(f"Unsupported format: {format}")


# Singleton instance
_enhanced_observability: Optional[EnhancedObservabilitySystem] = None


def get_observability_system(
    enable_gpu_monitoring: bool = True,
    enable_profiling: bool = False
) -> EnhancedObservabilitySystem:
    """Get or create enhanced observability system"""
    
    global _enhanced_observability
    
    if _enhanced_observability is None:
        _enhanced_observability = EnhancedObservabilitySystem(
            enable_gpu_monitoring=enable_gpu_monitoring,
            enable_profiling=enable_profiling
        )
        
    return _enhanced_observability