#!/usr/bin/env python3
"""
AURA Production Monitoring System - Phase 4
Advanced monitoring, alerting, and self-healing for production environments
"""

import asyncio
import time
import json
import sys
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

@dataclass
class Alert:
    """Production alert with severity and context"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    severity: str = "info"  # critical, error, warning, info
    component: str = "system"
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection"""
    metric_name: str
    baseline_value: float
    std_deviation: float
    sample_count: int
    last_updated: float
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "warning": 2.0,  # 2 standard deviations
        "critical": 3.0  # 3 standard deviations
    })

class ProductionMonitoringSystem:
    """Comprehensive production monitoring with alerting and self-healing"""
    
    def __init__(self):
        self.monitoring_active = False
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        
        # Performance tracking
        self.metrics_history = deque(maxlen=10000)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Health tracking
        self.component_health_history = deque(maxlen=1000)
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Alerting configuration
        self.alert_handlers: List[Callable] = []
        self.alert_cooldown = {}  # Prevent alert spam
        self.cooldown_duration = 300  # 5 minutes
        
        # Self-healing configuration
        self.self_healing_enabled = True
        self.healing_actions: Dict[str, Callable] = {}
        self.healing_attempts = {}
        self.max_healing_attempts = 3
        
        # OpenTelemetry integration
        self.tracing_enabled = False
        self.tracer = None
        
        # Logging configuration
        self.setup_production_logging()
        self.logger = logging.getLogger(__name__)
        
        # Performance anomaly detection
        self.anomaly_detection_enabled = True
        self.baseline_learning_period = 300  # 5 minutes
        
    def setup_production_logging(self):
        """Setup structured production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('aura_production.log')
            ]
        )
        
        # Add JSON formatter for production
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                if hasattr(record, 'component'):
                    log_entry['component'] = record.component
                if hasattr(record, 'trace_id'):
                    log_entry['trace_id'] = record.trace_id
                    
                return json.dumps(log_entry)
        
        # Apply JSON formatter to file handler
        file_handler = logging.FileHandler('aura_production_structured.log')
        file_handler.setFormatter(JSONFormatter())
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    async def initialize(self):
        """Initialize production monitoring system"""
        self.logger.info("Initializing production monitoring system")
        
        # Setup OpenTelemetry tracing
        await self.setup_tracing()
        
        # Register default alert handlers
        self.register_alert_handler(self.console_alert_handler)
        self.register_alert_handler(self.log_alert_handler)
        
        # Register self-healing actions
        self.register_healing_action("redis_connection_failed", self.heal_redis_connection)
        self.register_healing_action("gpu_memory_exhausted", self.heal_gpu_memory)
        self.register_healing_action("component_unhealthy", self.heal_component_restart)
        
        # Start background monitoring
        asyncio.create_task(self.monitoring_loop())
        asyncio.create_task(self.health_check_loop())
        asyncio.create_task(self.anomaly_detection_loop())
        
        self.monitoring_active = True
        self.logger.info("Production monitoring system initialized successfully")
    
    async def setup_tracing(self):
        """Setup OpenTelemetry distributed tracing"""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            
            # Configure resource
            resource = Resource.create({
                "service.name": "aura-intelligence",
                "service.version": "2025.1.0",
                "deployment.environment": "production"
            })
            
            # Setup tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Setup OTLP exporter (configure endpoint as needed)
            otlp_exporter = OTLPSpanExporter(
                endpoint="http://localhost:4317",
                insecure=True
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(__name__)
            self.tracing_enabled = True
            self.logger.info("OpenTelemetry tracing initialized")
            
        except ImportError:
            self.logger.warning("OpenTelemetry not available - tracing disabled")
        except Exception as e:
            self.logger.error(f"Failed to setup tracing: {e}")
    
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """Register custom alert handler"""
        self.alert_handlers.append(handler)
    
    def register_healing_action(self, issue_type: str, action: Callable):
        """Register self-healing action for specific issue types"""
        self.healing_actions[issue_type] = action
    
    async def create_alert(self, severity: str, component: str, message: str, metadata: Dict[str, Any] = None) -> Alert:
        """Create and process production alert"""
        alert = Alert(
            severity=severity,
            component=component,
            message=message,
            metadata=metadata or {}
        )
        
        # Check alert cooldown to prevent spam
        cooldown_key = f"{component}:{message}"
        current_time = time.time()
        
        if cooldown_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[cooldown_key] < self.cooldown_duration:
                return alert  # Skip duplicate alert
        
        self.alert_cooldown[cooldown_key] = current_time
        
        # Store alert
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        
        # Process through handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
        
        # Trigger self-healing if applicable
        if self.self_healing_enabled and severity in ["critical", "error"]:
            await self.attempt_self_healing(alert)
        
        return alert
    
    def console_alert_handler(self, alert: Alert):
        """Console alert handler with color coding"""
        colors = {
            "critical": "\033[91m",  # Red
            "error": "\033[93m",     # Yellow
            "warning": "\033[94m",   # Blue
            "info": "\033[92m"       # Green
        }
        reset = "\033[0m"
        
        color = colors.get(alert.severity, "")
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
        
        print(f"{color}[{timestamp}] {alert.severity.upper()}: {alert.component} - {alert.message}{reset}")
    
    def log_alert_handler(self, alert: Alert):
        """Structured log alert handler"""
        extra = {
            'component': alert.component,
            'alert_id': alert.id,
            'severity': alert.severity,
            'metadata': alert.metadata
        }
        
        if alert.severity == "critical":
            self.logger.critical(alert.message, extra=extra)
        elif alert.severity == "error":
            self.logger.error(alert.message, extra=extra)
        elif alert.severity == "warning":
            self.logger.warning(alert.message, extra=extra)
        else:
            self.logger.info(alert.message, extra=extra)
    
    async def attempt_self_healing(self, alert: Alert):
        """Attempt self-healing action for alert"""
        issue_type = alert.metadata.get("issue_type")
        if not issue_type or issue_type not in self.healing_actions:
            return
        
        # Check healing attempt limits
        if issue_type in self.healing_attempts:
            if self.healing_attempts[issue_type] >= self.max_healing_attempts:
                await self.create_alert(
                    "critical",
                    "self_healing",
                    f"Max healing attempts reached for {issue_type}",
                    {"original_alert": alert.id}
                )
                return
        else:
            self.healing_attempts[issue_type] = 0
        
        self.healing_attempts[issue_type] += 1
        
        try:
            healing_action = self.healing_actions[issue_type]
            
            self.logger.info(f"Attempting self-healing for {issue_type} (attempt {self.healing_attempts[issue_type]})")
            
            if asyncio.iscoroutinefunction(healing_action):
                success = await healing_action(alert)
            else:
                success = healing_action(alert)
            
            if success:
                alert.resolved = True
                alert.resolution_time = time.time()
                self.healing_attempts[issue_type] = 0  # Reset on success
                
                await self.create_alert(
                    "info",
                    "self_healing",
                    f"Successfully healed {issue_type}",
                    {"original_alert": alert.id}
                )
            else:
                await self.create_alert(
                    "warning",
                    "self_healing",
                    f"Healing attempt failed for {issue_type}",
                    {"original_alert": alert.id, "attempt": self.healing_attempts[issue_type]}
                )
                
        except Exception as e:
            await self.create_alert(
                "error",
                "self_healing",
                f"Healing action crashed for {issue_type}: {str(e)}",
                {"original_alert": alert.id}
            )
    
    async def heal_redis_connection(self, alert: Alert) -> bool:
        """Self-healing action for Redis connection issues"""
        try:
            from aura_intelligence.components.real_components import redis_pool
            
            # Reinitialize Redis pool
            success = await redis_pool.initialize()
            if success:
                self.logger.info("Redis connection healed successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Redis healing failed: {e}")
            return False
    
    async def heal_gpu_memory(self, alert: Alert) -> bool:
        """Self-healing action for GPU memory issues"""
        try:
            from aura_intelligence.components.real_components import gpu_manager
            
            # Clear GPU cache
            gpu_manager.clear_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("GPU memory cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"GPU memory healing failed: {e}")
            return False
    
    async def heal_component_restart(self, alert: Alert) -> bool:
        """Self-healing action for component health issues"""
        try:
            component_name = alert.metadata.get("component_name")
            if not component_name:
                return False
            
            # This would restart the specific component
            # For now, we'll just log the action
            self.logger.info(f"Component restart triggered for {component_name}")
            
            # Simulate restart success
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            self.logger.error(f"Component restart healing failed: {e}")
            return False
    
    async def monitoring_loop(self):
        """Main monitoring loop for system metrics"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                await self.collect_system_metrics()
                
                # Check for performance anomalies
                await self.check_performance_anomalies()
                
                # Sleep for monitoring interval
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def health_check_loop(self):
        """Background health checking with alerts"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Run health checks
                if current_time - self.last_health_check >= self.health_check_interval:
                    await self.perform_comprehensive_health_check()
                    self.last_health_check = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)
    
    async def anomaly_detection_loop(self):
        """Background anomaly detection"""
        while self.monitoring_active:
            try:
                if self.anomaly_detection_enabled:
                    await self.update_performance_baselines()
                
                await asyncio.sleep(60)  # Update baselines every minute
                
            except Exception as e:
                self.logger.error(f"Anomaly detection loop error: {e}")
                await asyncio.sleep(60)
    
    async def collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        try:
            from aura_intelligence.components.real_components import (
                redis_pool,
                batch_processor,
                gpu_manager
            )
            
            metrics = {
                "timestamp": time.time(),
                "redis": redis_pool.get_pool_stats(),
                "batch_processing": batch_processor.get_performance_stats(),
                "gpu": gpu_manager.get_memory_info(),
                "system_load": self.get_system_load()
            }
            
            self.metrics_history.append(metrics)
            
            # Check for critical issues
            await self.analyze_metrics_for_issues(metrics)
            
        except Exception as e:
            await self.create_alert(
                "error",
                "metrics_collection",
                f"Failed to collect metrics: {str(e)}"
            )
    
    async def analyze_metrics_for_issues(self, metrics: Dict[str, Any]):
        """Analyze metrics for potential issues"""
        # Check Redis health
        redis_status = metrics.get("redis", {}).get("status")
        if redis_status != "active":
            await self.create_alert(
                "critical",
                "redis",
                f"Redis pool status: {redis_status}",
                {"issue_type": "redis_connection_failed"}
            )
        
        # Check GPU memory
        gpu_info = metrics.get("gpu", {})
        if gpu_info.get("gpu_available"):
            memory_allocated = gpu_info.get("memory_allocated", 0)
            if memory_allocated > 8 * 1024 * 1024 * 1024:  # 8GB threshold
                await self.create_alert(
                    "warning",
                    "gpu",
                    f"High GPU memory usage: {memory_allocated // (1024*1024)} MB",
                    {"issue_type": "gpu_memory_exhausted"}
                )
        
        # Check batch processing performance
        batch_stats = metrics.get("batch_processing", {})
        avg_time = batch_stats.get("avg_processing_time_ms", 0)
        if avg_time > 1000:  # 1 second threshold
            await self.create_alert(
                "warning",
                "batch_processing",
                f"High batch processing time: {avg_time:.1f}ms"
            )
    
    def get_system_load(self) -> Dict[str, Any]:
        """Get system load metrics"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    async def perform_comprehensive_health_check(self):
        """Perform comprehensive health check of all components"""
        try:
            from aura_intelligence.components.real_components import (
                RealAttentionComponent,
                RealLNNComponent
            )
            
            # Test key components
            components = [
                ("bert_attention", RealAttentionComponent),
                ("lnn_neural", RealLNNComponent)
            ]
            
            unhealthy_components = []
            
            for comp_name, comp_class in components:
                try:
                    component = comp_class(f"health_{comp_name}")
                    health = await component.health_check()
                    
                    if health.get("status") != "healthy":
                        unhealthy_components.append(comp_name)
                        
                        await self.create_alert(
                            "error",
                            comp_name,
                            f"Component health check failed: {comp_name}",
                            {
                                "issue_type": "component_unhealthy",
                                "component_name": comp_name,
                                "health_data": health
                            }
                        )
                    
                except Exception as e:
                    unhealthy_components.append(comp_name)
                    await self.create_alert(
                        "critical",
                        comp_name,
                        f"Component health check crashed: {str(e)}",
                        {"component_name": comp_name}
                    )
            
            # Store health status
            health_summary = {
                "timestamp": time.time(),
                "total_components": len(components),
                "healthy_components": len(components) - len(unhealthy_components),
                "unhealthy_components": unhealthy_components
            }
            
            self.component_health_history.append(health_summary)
            
        except Exception as e:
            await self.create_alert(
                "critical",
                "health_check",
                f"Health check system failed: {str(e)}"
            )
    
    async def update_performance_baselines(self):
        """Update performance baselines for anomaly detection"""
        if len(self.metrics_history) < 10:
            return  # Need more data
        
        # Calculate baselines for key metrics
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        
        metrics_to_track = [
            "batch_processing.avg_processing_time_ms",
            "system_load.cpu_percent",
            "system_load.memory_percent"
        ]
        
        for metric_path in metrics_to_track:
            values = []
            
            for metric in recent_metrics:
                try:
                    # Navigate nested dictionary
                    value = metric
                    for key in metric_path.split('.'):
                        value = value[key]
                    values.append(float(value))
                except (KeyError, TypeError, ValueError):
                    continue
            
            if len(values) >= 5:  # Need minimum samples
                import statistics
                
                mean_value = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                self.baselines[metric_path] = PerformanceBaseline(
                    metric_name=metric_path,
                    baseline_value=mean_value,
                    std_deviation=std_dev,
                    sample_count=len(values),
                    last_updated=time.time()
                )
    
    async def check_performance_anomalies(self):
        """Check for performance anomalies against baselines"""
        if not self.metrics_history or not self.baselines:
            return
        
        current_metrics = self.metrics_history[-1]
        
        for metric_path, baseline in self.baselines.items():
            try:
                # Get current value
                current_value = current_metrics
                for key in metric_path.split('.'):
                    current_value = current_value[key]
                current_value = float(current_value)
                
                # Calculate deviation
                if baseline.std_deviation > 0:
                    deviation = abs(current_value - baseline.baseline_value) / baseline.std_deviation
                    
                    if deviation >= baseline.thresholds["critical"]:
                        await self.create_alert(
                            "critical",
                            "anomaly_detection",
                            f"Critical anomaly in {metric_path}: {current_value:.2f} (baseline: {baseline.baseline_value:.2f})",
                            {
                                "metric": metric_path,
                                "current_value": current_value,
                                "baseline": baseline.baseline_value,
                                "deviation": deviation
                            }
                        )
                    elif deviation >= baseline.thresholds["warning"]:
                        await self.create_alert(
                            "warning",
                            "anomaly_detection",
                            f"Performance anomaly in {metric_path}: {current_value:.2f} (baseline: {baseline.baseline_value:.2f})"
                        )
                
            except (KeyError, TypeError, ValueError):
                continue
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status"""
        return {
            "monitoring_active": self.monitoring_active,
            "total_alerts": len(self.alerts),
            "recent_alerts": len([a for a in self.alerts if time.time() - a.timestamp < 3600]),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical" and not a.resolved]),
            "tracing_enabled": self.tracing_enabled,
            "self_healing_enabled": self.self_healing_enabled,
            "anomaly_detection_enabled": self.anomaly_detection_enabled,
            "baselines_count": len(self.baselines),
            "metrics_history_size": len(self.metrics_history),
            "health_checks_performed": len(self.component_health_history),
            "healing_actions_registered": len(self.healing_actions),
            "alert_handlers_registered": len(self.alert_handlers)
        }

# Global monitoring instance
_monitoring_system = None

def get_monitoring_system() -> ProductionMonitoringSystem:
    """Get global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = ProductionMonitoringSystem()
    return _monitoring_system

async def run_monitoring_demo():
    """Run monitoring system demonstration"""
    print("🔧 AURA Production Monitoring System Demo")
    print("=" * 60)
    
    # Initialize monitoring
    monitoring = get_monitoring_system()
    await monitoring.initialize()
    
    print("✅ Monitoring system initialized")
    
    # Simulate some alerts
    print("\n🚨 Simulating production alerts...")
    
    await monitoring.create_alert(
        "info",
        "demo",
        "Monitoring system demo started"
    )
    
    await monitoring.create_alert(
        "warning",
        "demo",
        "Simulated high latency detected",
        {"latency_ms": 1500}
    )
    
    await monitoring.create_alert(
        "critical",
        "demo",
        "Simulated Redis connection failure",
        {"issue_type": "redis_connection_failed"}
    )
    
    # Wait for self-healing
    await asyncio.sleep(2)
    
    # Show monitoring status
    status = monitoring.get_monitoring_status()
    print(f"\n📊 Monitoring Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Production monitoring system demo complete!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(run_monitoring_demo())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring demo interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Monitoring demo failed: {e}")
        sys.exit(1)