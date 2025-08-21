"""
📊 REAL-TIME PERFORMANCE DASHBOARD
Live monitoring system for AURA Intelligence with GPU metrics, batch processing, and component health
"""

import asyncio
import time
import json
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from collections import deque, defaultdict

import torch
import structlog

logger = structlog.get_logger()

@dataclass
class PerformanceSnapshot:
    """Real-time performance snapshot"""
    timestamp: float
    
    # System metrics
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    
    # GPU metrics
    gpu_available: bool
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_temperature: float = 0.0
    
    # Component metrics
    components_healthy: int = 0
    components_total: int = 0
    avg_component_response_time_ms: float = 0.0
    
    # Batch processing metrics
    batch_operations_per_second: float = 0.0
    batch_efficiency_percent: float = 0.0
    avg_batch_size: float = 0.0
    pending_batch_operations: int = 0
    
    # Redis metrics
    redis_healthy: bool = False
    redis_operations_per_second: float = 0.0
    redis_cache_hit_rate: float = 0.0
    
    # Overall system health score (0-100)
    health_score: float = 0.0

class RealTimeDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, history_size: int = 300):  # 5 minutes at 1s intervals
        self.history_size = history_size
        self.performance_history: deque = deque(maxlen=history_size)
        self.component_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.update_interval = 1.0  # 1 second updates
        
        # External service references
        self.batch_processor = None
        self.redis_adapter = None
        self.component_registry = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'gpu_memory_warning': 90.0,
            'gpu_memory_critical': 98.0,
            'response_time_warning': 100.0,  # ms
            'response_time_critical': 500.0,  # ms
            'health_score_warning': 70.0,
            'health_score_critical': 50.0
        }
        
        # Alert history
        self.alerts: deque = deque(maxlen=50)
        
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
            
        logger.info("Starting real-time performance dashboard")
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Try to get external services
        await self._initialize_services()
        
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.monitoring_active:
            return
            
        logger.info("Stopping real-time performance dashboard")
        self.monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
    async def _initialize_services(self):
        """Initialize connections to external services"""
        try:
            # Get batch processor
            from ..components.async_batch_processor import get_global_batch_processor
            self.batch_processor = await get_global_batch_processor()
            logger.debug("Connected to batch processor")
        except Exception as e:
            logger.warning(f"Could not connect to batch processor: {e}")
        
        try:
            # Get Redis adapter
            from ..adapters.redis_high_performance import get_ultra_high_performance_adapter
            self.redis_adapter = await get_ultra_high_performance_adapter()
            logger.debug("Connected to Redis adapter")
        except Exception as e:
            logger.warning(f"Could not connect to Redis adapter: {e}")
        
        try:
            # Get component registry
            from ..components.real_registry import get_real_registry
            self.component_registry = get_real_registry()
            logger.debug("Connected to component registry")
        except Exception as e:
            logger.warning(f"Could not connect to component registry: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect performance snapshot
                snapshot = await self._collect_snapshot()
                
                # Add to history
                self.performance_history.append(snapshot)
                
                # Check for alerts
                await self._check_alerts(snapshot)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot"""
        timestamp = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # GPU metrics
        gpu_available = torch.cuda.is_available()
        gpu_memory_used_mb = 0.0
        gpu_memory_total_mb = 0.0
        gpu_utilization_percent = 0.0
        gpu_temperature = 0.0
        
        if gpu_available:
            try:
                gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024**2)
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                
                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization_percent = util.gpu
                    
                    # Get temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature = temp
                except ImportError:
                    pass  # nvidia-ml-py not available
                except Exception:
                    pass  # Other NVML errors
            except Exception as e:
                logger.debug(f"Error collecting GPU metrics: {e}")
        
        # Component metrics
        components_healthy = 0
        components_total = 0
        avg_response_time = 0.0
        
        if self.component_registry:
            try:
                components = self.component_registry.components
                components_total = len(components)
                
                # Simple health check - count components with reasonable response times
                response_times = []
                for component in components.values():
                    if hasattr(component, 'processing_time') and component.processing_time is not None:
                        response_times.append(component.processing_time)
                        if component.processing_time < self.thresholds['response_time_warning']:
                            components_healthy += 1
                    else:
                        components_healthy += 1  # Assume healthy if no timing data
                
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    
            except Exception as e:
                logger.debug(f"Error collecting component metrics: {e}")
        
        # Batch processing metrics
        batch_ops_per_sec = 0.0
        batch_efficiency = 0.0
        avg_batch_size = 0.0
        pending_batch_ops = 0
        
        if self.batch_processor:
            try:
                metrics = await self.batch_processor.get_metrics()
                batch_ops_per_sec = metrics.get('operations_per_second', 0.0)
                batch_efficiency = metrics.get('batch_efficiency', 0.0) * 100
                avg_batch_size = metrics.get('avg_batch_size', 0.0)
                pending_batch_ops = sum(metrics.get('queue_sizes', {}).values())
            except Exception as e:
                logger.debug(f"Error collecting batch metrics: {e}")
        
        # Redis metrics
        redis_healthy = False
        redis_ops_per_sec = 0.0
        redis_cache_hit_rate = 0.0
        
        if self.redis_adapter:
            try:
                health = await self.redis_adapter.health_check()
                redis_healthy = health.get('status') == 'healthy'
                
                perf_metrics = await self.redis_adapter.get_performance_metrics()
                redis_ops_per_sec = perf_metrics.get('operations_per_second', 0.0)
                redis_cache_hit_rate = perf_metrics.get('cache_hit_rate', 0.0) * 100
            except Exception as e:
                logger.debug(f"Error collecting Redis metrics: {e}")
        
        # Calculate overall health score
        health_score = self._calculate_health_score(
            cpu_percent, memory_percent, gpu_memory_used_mb, gpu_memory_total_mb,
            components_healthy, components_total, avg_response_time,
            redis_healthy, batch_efficiency
        )
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_available=gpu_available,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            gpu_temperature=gpu_temperature,
            components_healthy=components_healthy,
            components_total=components_total,
            avg_component_response_time_ms=avg_response_time,
            batch_operations_per_second=batch_ops_per_sec,
            batch_efficiency_percent=batch_efficiency,
            avg_batch_size=avg_batch_size,
            pending_batch_operations=pending_batch_ops,
            redis_healthy=redis_healthy,
            redis_operations_per_second=redis_ops_per_sec,
            redis_cache_hit_rate=redis_cache_hit_rate,
            health_score=health_score
        )
    
    def _calculate_health_score(
        self, cpu_percent: float, memory_percent: float, 
        gpu_memory_used: float, gpu_memory_total: float,
        components_healthy: int, components_total: int, 
        avg_response_time: float, redis_healthy: bool, 
        batch_efficiency: float
    ) -> float:
        """Calculate overall system health score (0-100)"""
        
        score = 100.0
        
        # CPU penalty
        if cpu_percent > self.thresholds['cpu_critical']:
            score -= 30
        elif cpu_percent > self.thresholds['cpu_warning']:
            score -= 15
        
        # Memory penalty
        if memory_percent > self.thresholds['memory_critical']:
            score -= 25
        elif memory_percent > self.thresholds['memory_warning']:
            score -= 10
        
        # GPU memory penalty
        if gpu_memory_total > 0:
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            if gpu_memory_percent > self.thresholds['gpu_memory_critical']:
                score -= 20
            elif gpu_memory_percent > self.thresholds['gpu_memory_warning']:
                score -= 10
        
        # Component health penalty
        if components_total > 0:
            component_health_rate = components_healthy / components_total
            if component_health_rate < 0.8:
                score -= 25
            elif component_health_rate < 0.9:
                score -= 10
        
        # Response time penalty
        if avg_response_time > self.thresholds['response_time_critical']:
            score -= 20
        elif avg_response_time > self.thresholds['response_time_warning']:
            score -= 10
        
        # Redis penalty
        if not redis_healthy:
            score -= 5
        
        # Batch efficiency bonus/penalty
        if batch_efficiency > 80:
            score += 5
        elif batch_efficiency < 50:
            score -= 10
        
        return max(0.0, min(100.0, score))
    
    async def _check_alerts(self, snapshot: PerformanceSnapshot):\n        \"\"\"Check for alert conditions\"\"\"\n        alerts = []\n        \n        # CPU alerts\n        if snapshot.cpu_percent > self.thresholds['cpu_critical']:\n            alerts.append({\n                'level': 'CRITICAL',\n                'type': 'CPU',\n                'message': f'CPU usage critical: {snapshot.cpu_percent:.1f}%',\n                'timestamp': snapshot.timestamp\n            })\n        elif snapshot.cpu_percent > self.thresholds['cpu_warning']:\n            alerts.append({\n                'level': 'WARNING',\n                'type': 'CPU',\n                'message': f'CPU usage high: {snapshot.cpu_percent:.1f}%',\n                'timestamp': snapshot.timestamp\n            })\n        \n        # Memory alerts\n        if snapshot.memory_percent > self.thresholds['memory_critical']:\n            alerts.append({\n                'level': 'CRITICAL',\n                'type': 'MEMORY',\n                'message': f'Memory usage critical: {snapshot.memory_percent:.1f}% ({snapshot.memory_used_gb:.1f}GB)',\n                'timestamp': snapshot.timestamp\n            })\n        elif snapshot.memory_percent > self.thresholds['memory_warning']:\n            alerts.append({\n                'level': 'WARNING',\n                'type': 'MEMORY',\n                'message': f'Memory usage high: {snapshot.memory_percent:.1f}% ({snapshot.memory_used_gb:.1f}GB)',\n                'timestamp': snapshot.timestamp\n            })\n        \n        # GPU memory alerts\n        if snapshot.gpu_available and snapshot.gpu_memory_total_mb > 0:\n            gpu_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100\n            if gpu_percent > self.thresholds['gpu_memory_critical']:\n                alerts.append({\n                    'level': 'CRITICAL',\n                    'type': 'GPU_MEMORY',\n                    'message': f'GPU memory critical: {gpu_percent:.1f}% ({snapshot.gpu_memory_used_mb:.0f}MB)',\n                    'timestamp': snapshot.timestamp\n                })\n            elif gpu_percent > self.thresholds['gpu_memory_warning']:\n                alerts.append({\n                    'level': 'WARNING',\n                    'type': 'GPU_MEMORY',\n                    'message': f'GPU memory high: {gpu_percent:.1f}% ({snapshot.gpu_memory_used_mb:.0f}MB)',\n                    'timestamp': snapshot.timestamp\n                })\n        \n        # Health score alerts\n        if snapshot.health_score < self.thresholds['health_score_critical']:\n            alerts.append({\n                'level': 'CRITICAL',\n                'type': 'HEALTH',\n                'message': f'System health critical: {snapshot.health_score:.1f}/100',\n                'timestamp': snapshot.timestamp\n            })\n        elif snapshot.health_score < self.thresholds['health_score_warning']:\n            alerts.append({\n                'level': 'WARNING',\n                'type': 'HEALTH',\n                'message': f'System health low: {snapshot.health_score:.1f}/100',\n                'timestamp': snapshot.timestamp\n            })\n        \n        # Component health alerts\n        if snapshot.components_total > 0:\n            unhealthy_count = snapshot.components_total - snapshot.components_healthy\n            if unhealthy_count > 0:\n                alerts.append({\n                    'level': 'WARNING',\n                    'type': 'COMPONENTS',\n                    'message': f'{unhealthy_count} components unhealthy out of {snapshot.components_total}',\n                    'timestamp': snapshot.timestamp\n                })\n        \n        # Add alerts to history\n        for alert in alerts:\n            self.alerts.append(alert)\n            logger.warning(f\"ALERT [{alert['level']}] {alert['type']}: {alert['message']}\")\n    \n    def get_current_status(self) -> Dict[str, Any]:\n        \"\"\"Get current system status\"\"\"\n        if not self.performance_history:\n            return {'status': 'No data available'}\n        \n        latest = self.performance_history[-1]\n        \n        return {\n            'timestamp': latest.timestamp,\n            'health_score': latest.health_score,\n            'system': {\n                'cpu_percent': latest.cpu_percent,\n                'memory_percent': latest.memory_percent,\n                'memory_used_gb': latest.memory_used_gb,\n                'memory_total_gb': latest.memory_total_gb\n            },\n            'gpu': {\n                'available': latest.gpu_available,\n                'memory_used_mb': latest.gpu_memory_used_mb,\n                'memory_total_mb': latest.gpu_memory_total_mb,\n                'utilization_percent': latest.gpu_utilization_percent,\n                'temperature': latest.gpu_temperature\n            },\n            'components': {\n                'healthy': latest.components_healthy,\n                'total': latest.components_total,\n                'avg_response_time_ms': latest.avg_component_response_time_ms\n            },\n            'batch_processing': {\n                'operations_per_second': latest.batch_operations_per_second,\n                'efficiency_percent': latest.batch_efficiency_percent,\n                'avg_batch_size': latest.avg_batch_size,\n                'pending_operations': latest.pending_batch_operations\n            },\n            'redis': {\n                'healthy': latest.redis_healthy,\n                'operations_per_second': latest.redis_operations_per_second,\n                'cache_hit_rate': latest.redis_cache_hit_rate\n            },\n            'recent_alerts': list(self.alerts)[-10:] if self.alerts else []\n        }\n    \n    def get_performance_history(self, minutes: int = 5) -> List[Dict[str, Any]]:\n        \"\"\"Get performance history for specified minutes\"\"\"\n        cutoff_time = time.time() - (minutes * 60)\n        \n        return [\n            asdict(snapshot) for snapshot in self.performance_history\n            if snapshot.timestamp >= cutoff_time\n        ]\n    \n    def get_performance_trends(self) -> Dict[str, Any]:\n        \"\"\"Get performance trends and analysis\"\"\"\n        if len(self.performance_history) < 2:\n            return {'status': 'Insufficient data for trends'}\n        \n        recent = list(self.performance_history)[-60:]  # Last 1 minute\n        \n        # Calculate trends\n        cpu_trend = self._calculate_trend([s.cpu_percent for s in recent])\n        memory_trend = self._calculate_trend([s.memory_percent for s in recent])\n        health_trend = self._calculate_trend([s.health_score for s in recent])\n        response_time_trend = self._calculate_trend([s.avg_component_response_time_ms for s in recent])\n        \n        return {\n            'analysis_period_seconds': len(recent),\n            'trends': {\n                'cpu_percent': cpu_trend,\n                'memory_percent': memory_trend,\n                'health_score': health_trend,\n                'response_time_ms': response_time_trend\n            },\n            'averages': {\n                'cpu_percent': sum(s.cpu_percent for s in recent) / len(recent),\n                'memory_percent': sum(s.memory_percent for s in recent) / len(recent),\n                'health_score': sum(s.health_score for s in recent) / len(recent),\n                'batch_ops_per_second': sum(s.batch_operations_per_second for s in recent) / len(recent)\n            },\n            'alert_count_last_hour': len([a for a in self.alerts if a['timestamp'] > time.time() - 3600])\n        }\n    \n    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:\n        \"\"\"Calculate trend for a series of values\"\"\"\n        if len(values) < 2:\n            return {'direction': 'stable', 'change_percent': 0.0}\n        \n        first_half = values[:len(values)//2]\n        second_half = values[len(values)//2:]\n        \n        first_avg = sum(first_half) / len(first_half)\n        second_avg = sum(second_half) / len(second_half)\n        \n        if first_avg == 0:\n            change_percent = 0.0\n        else:\n            change_percent = ((second_avg - first_avg) / first_avg) * 100\n        \n        if abs(change_percent) < 5:\n            direction = 'stable'\n        elif change_percent > 0:\n            direction = 'increasing'\n        else:\n            direction = 'decreasing'\n        \n        return {\n            'direction': direction,\n            'change_percent': change_percent,\n            'first_half_avg': first_avg,\n            'second_half_avg': second_avg\n        }\n    \n    def export_metrics(self, filepath: str):\n        \"\"\"Export metrics to JSON file\"\"\"\n        data = {\n            'exported_at': time.time(),\n            'performance_history': [asdict(s) for s in self.performance_history],\n            'alerts': list(self.alerts),\n            'current_status': self.get_current_status(),\n            'trends': self.get_performance_trends()\n        }\n        \n        with open(filepath, 'w') as f:\n            json.dump(data, f, indent=2, default=str)\n        \n        logger.info(f\"Performance metrics exported to {filepath}\")\n\n# Global dashboard instance\n_global_dashboard = None\n\nasync def get_global_dashboard() -> RealTimeDashboard:\n    \"\"\"Get global dashboard instance\"\"\"\n    global _global_dashboard\n    if _global_dashboard is None:\n        _global_dashboard = RealTimeDashboard()\n        await _global_dashboard.start_monitoring()\n    return _global_dashboard\n\nasync def start_dashboard_monitoring():\n    \"\"\"Start global dashboard monitoring\"\"\"\n    dashboard = await get_global_dashboard()\n    return dashboard\n\nasync def stop_dashboard_monitoring():\n    \"\"\"Stop global dashboard monitoring\"\"\"\n    global _global_dashboard\n    if _global_dashboard:\n        await _global_dashboard.stop_monitoring()\n        _global_dashboard = None