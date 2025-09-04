#!/usr/bin/env python3
"""
Production Monitoring System for AURA Intelligence
Comprehensive monitoring, alerting, and reliability features
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
import requests
from collections import defaultdict, deque
from contextlib import contextmanager
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass 
class Alert:
    """System alert"""
    id: str
    severity: str  # critical, warning, info
    message: str
    timestamp: float
    component: str
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Component health check result"""
    component: str
    status: str  # healthy, degraded, unhealthy
    timestamp: float
    response_time: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker for reliability"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def record(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self.metrics[metric_name].append(point)
    
    def get_recent(self, metric_name: str, minutes: int = 5) -> List[MetricPoint]:
        """Get recent metric points"""
        cutoff_time = time.time() - (minutes * 60)
        with self.lock:
            return [
                point for point in self.metrics[metric_name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_average(self, metric_name: str, minutes: int = 5) -> float:
        """Get average value over time period"""
        points = self.get_recent(metric_name, minutes)
        if not points:
            return 0.0
        return sum(p.value for p in points) / len(points)
    
    def get_percentile(self, metric_name: str, percentile: float, minutes: int = 5) -> float:
        """Get percentile value"""
        points = self.get_recent(metric_name, minutes)
        if not points:
            return 0.0
        
        values = sorted([p.value for p in points])
        index = int(percentile / 100 * len(values))
        return values[min(index, len(values) - 1)]

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.lock = threading.Lock()
    
    def add_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def create_alert(self, severity: str, message: str, component: str, **metadata):
        """Create new alert"""
        alert = Alert(
            id=f"alert_{int(time.time() * 1000)}",
            severity=severity,
            message=message,
            component=component,
            timestamp=time.time(),
            metadata=metadata
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"ALERT [{severity.upper()}] {component}: {message}")
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get unresolved alerts"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]

class HealthMonitor:
    """Monitors component health"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def add_circuit_breaker(self, component: str, **kwargs):
        """Add circuit breaker for component"""
        self.circuit_breakers[component] = CircuitBreaker(**kwargs)
    
    async def check_component_health(self, component: str, check_func: Callable) -> HealthCheck:
        """Check component health"""
        start_time = time.time()
        
        try:
            # Use circuit breaker if available
            if component in self.circuit_breakers:
                result = self.circuit_breakers[component].call(check_func)
            else:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
            
            response_time = time.time() - start_time
            
            health_check = HealthCheck(
                component=component,
                status="healthy",
                timestamp=time.time(),
                response_time=response_time,
                details=result if isinstance(result, dict) else {"status": "ok"}
            )
            
            # Record metrics
            self.metrics.record(f"{component}.health", 1.0)
            self.metrics.record(f"{component}.response_time", response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            health_check = HealthCheck(
                component=component,
                status="unhealthy",
                timestamp=time.time(),
                response_time=response_time,
                error=error_msg,
                details={"error": error_msg, "traceback": traceback.format_exc()}
            )
            
            # Record metrics
            self.metrics.record(f"{component}.health", 0.0)
            self.metrics.record(f"{component}.response_time", response_time)
            
            # Create alert
            self.alerts.create_alert(
                severity="critical",
                message=f"Component {component} health check failed: {error_msg}",
                component=component,
                response_time=response_time
            )
        
        self.health_checks[component] = health_check
        return health_check
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score"""
        if not self.health_checks:
            return 0.0
        
        healthy_count = sum(1 for hc in self.health_checks.values() if hc.status == "healthy")
        return healthy_count / len(self.health_checks)

class PerformanceMonitor:
    """Monitors system performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    @contextmanager
    def measure_time(self, operation: str, labels: Dict[str, str] = None):
        """Context manager to measure operation time"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.record(f"{operation}.duration", duration, labels)
    
    def record_request(self, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        labels = {
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        
        self.metrics.record("api.request.duration", duration, labels)
        self.metrics.record("api.request.count", 1.0, labels)
        
        if status_code >= 400:
            self.metrics.record("api.error.count", 1.0, labels)
    
    def get_performance_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "avg_response_time": self.metrics.get_average("api.request.duration", minutes),
            "p95_response_time": self.metrics.get_percentile("api.request.duration", 95, minutes),
            "p99_response_time": self.metrics.get_percentile("api.request.duration", 99, minutes),
            "request_rate": len(self.metrics.get_recent("api.request.count", 1)) / 60,  # per second
            "error_rate": len(self.metrics.get_recent("api.error.count", minutes)) / max(1, len(self.metrics.get_recent("api.request.count", minutes)))
        }

class ProductionMonitoringSystem:
    """Complete production monitoring system"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.health_monitor = HealthMonitor(self.metrics, self.alerts)
        self.performance = PerformanceMonitor(self.metrics)
        
        # Monitoring configuration
        self.config = {
            "health_check_interval": 30,  # seconds
            "metrics_retention": 3600,    # seconds
            "alert_cooldown": 300,        # seconds
            "performance_threshold": {
                "response_time_p95": 1.0,  # seconds
                "error_rate": 0.05,        # 5%
                "health_score": 0.9        # 90%
            }
        }
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Setup default alert callbacks
        self.alerts.add_callback(self._log_alert)
    
    def _log_alert(self, alert: Alert):
        """Default alert logging callback"""
        logger.warning(f"🚨 ALERT: {alert.severity.upper()} in {alert.component}: {alert.message}")
    
    async def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("🔍 Production monitoring system started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("🔍 Production monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_checks()
                await self._check_performance_thresholds()
                await asyncio.sleep(self.config["health_check_interval"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """Perform health checks on all components"""
        # Check API health
        async def check_api():
            try:
                response = requests.get("http://localhost:8002/health", timeout=5)
                return {"status_code": response.status_code, "response_time": response.elapsed.total_seconds()}
            except Exception as e:
                raise Exception(f"API health check failed: {e}")
        
        await self.health_monitor.check_component_health("api", check_api)
        
        # Add more component checks as needed
    
    async def _check_performance_thresholds(self):
        """Check if performance thresholds are exceeded"""
        perf_summary = self.performance.get_performance_summary()
        thresholds = self.config["performance_threshold"]
        
        # Check response time
        if perf_summary["p95_response_time"] > thresholds["response_time_p95"]:
            self.alerts.create_alert(
                severity="warning",
                message=f"P95 response time is {perf_summary['p95_response_time']:.3f}s (threshold: {thresholds['response_time_p95']}s)",
                component="performance",
                metric="response_time_p95",
                value=perf_summary["p95_response_time"]
            )
        
        # Check error rate
        if perf_summary["error_rate"] > thresholds["error_rate"]:
            self.alerts.create_alert(
                severity="critical",
                message=f"Error rate is {perf_summary['error_rate']:.2%} (threshold: {thresholds['error_rate']:.2%})",
                component="performance",
                metric="error_rate", 
                value=perf_summary["error_rate"]
            )
        
        # Check health score
        health_score = self.health_monitor.get_system_health_score()
        if health_score < thresholds["health_score"]:
            self.alerts.create_alert(
                severity="critical",
                message=f"System health score is {health_score:.2%} (threshold: {thresholds['health_score']:.2%})",
                component="system",
                metric="health_score",
                value=health_score
            )
    
    def add_webhook_alert(self, webhook_url: str):
        """Add webhook for alerts"""
        def webhook_callback(alert: Alert):
            try:
                payload = {
                    "alert": asdict(alert),
                    "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                    "system": "AURA Intelligence"
                }
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as e:
                logger.error(f"Webhook alert failed: {e}")
        
        self.alerts.add_callback(webhook_callback)
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for monitoring UI"""
        return {
            "system_health": {
                "score": self.health_monitor.get_system_health_score(),
                "components": {
                    name: asdict(hc) for name, hc in self.health_monitor.health_checks.items()
                }
            },
            "performance": self.performance.get_performance_summary(),
            "alerts": {
                "active": [asdict(alert) for alert in self.alerts.get_active_alerts()],
                "total_count": len(self.alerts.alerts),
                "active_count": len(self.alerts.get_active_alerts())
            },
            "metrics": {
                "available_metrics": list(self.metrics.metrics.keys()),
                "total_points": sum(len(points) for points in self.metrics.metrics.values())
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for metric_name, points in self.metrics.metrics.items():
            if not points:
                continue
            
            latest_point = points[-1]
            
            # Convert metric name to Prometheus format
            prom_name = metric_name.replace(".", "_")
            
            # Add metric help
            lines.append(f"# HELP aura_{prom_name} AURA Intelligence metric")
            lines.append(f"# TYPE aura_{prom_name} gauge")
            
            # Add metric value
            labels_str = ""
            if latest_point.labels:
                label_pairs = [f'{k}="{v}"' for k, v in latest_point.labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"
            
            lines.append(f"aura_{prom_name}{labels_str} {latest_point.value}")
        
        return "\n".join(lines)

# Global monitoring system instance
monitoring_system = ProductionMonitoringSystem()

async def start_monitoring_system():
    """Start the monitoring system"""
    await monitoring_system.start_monitoring()

async def stop_monitoring_system():
    """Stop the monitoring system"""
    await monitoring_system.stop_monitoring()

if __name__ == "__main__":
    import asyncio
    
    async def demo_monitoring():
        """Demonstrate monitoring system"""
        print("🔍 Starting Production Monitoring System Demo...")
        
        await monitoring_system.start_monitoring()
        
        # Simulate some metrics
        for i in range(10):
            monitoring_system.metrics.record("demo.counter", float(i))
            monitoring_system.performance.record_request("/test", 200, 0.1)
            await asyncio.sleep(1)
        
        # Get dashboard data
        dashboard_data = monitoring_system.get_monitoring_dashboard_data()
        print(f"📊 Dashboard Data: {json.dumps(dashboard_data, indent=2)}")
        
        await monitoring_system.stop_monitoring()
        print("✅ Monitoring demo complete")
    
    asyncio.run(demo_monitoring())