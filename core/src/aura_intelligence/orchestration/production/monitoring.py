"""
Production Monitoring System

Real-time monitoring with SLA tracking and TDA integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)

@dataclass
class SLAThreshold:
    """SLA threshold definition"""
    metric_name: str
    threshold_value: float
    comparison: str  # "lt", "gt", "eq"
    severity: str    # "critical", "warning", "info"
    description: str

@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Optional[str] = None

class ProductionMonitor:
    """
    Production monitoring system with SLA tracking.
    
    Monitors orchestration health and performance in real-time.
    """
    
    def __init__(self, tda_integration: Optional[Any] = None):
        self.tda_integration = tda_integration
        self.sla_thresholds: Dict[str, SLAThreshold] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_callbacks: List[Callable] = []
        
        # Performance metrics
        self.uptime_start = datetime.utcnow()
        self.total_requests = 0
        self.failed_requests = 0
        self.response_times: deque = deque(maxlen=1000)
        
        # SLA tracking
        self.sla_violations: List[Dict[str, Any]] = []
        self.sla_compliance: Dict[str, float] = {}
        
        self._setup_default_slas()
        
        logger.info("Production Monitor initialized")
    
    def _setup_default_slas(self) -> None:
        """Setup default SLA thresholds"""
        pass
        default_slas = [
            SLAThreshold(
                metric_name="response_time_p99",
                threshold_value=100.0,  # 100ms
                comparison="lt",
                severity="warning",
                description="99th percentile response time should be < 100ms"
            ),
            SLAThreshold(
                metric_name="error_rate",
                threshold_value=1.0,  # 1%
                comparison="lt", 
                severity="critical",
                description="Error rate should be < 1%"
            ),
            SLAThreshold(
                metric_name="availability",
                threshold_value=99.9,  # 99.9%
                comparison="gt",
                severity="critical",
                description="System availability should be > 99.9%"
            ),
            SLAThreshold(
                metric_name="workflow_success_rate",
                threshold_value=95.0,  # 95%
                comparison="gt",
                severity="warning",
                description="Workflow success rate should be > 95%"
            )
        ]
        
        for sla in default_slas:
            self.sla_thresholds[sla.metric_name] = sla
    
        async def start_monitoring(self) -> None:
        """Start production monitoring"""
        pass
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Production monitoring started")
    
        async def stop_monitoring(self) -> None:
        """Stop production monitoring"""
        pass
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
        pass
        
        logger.info("Production monitoring stopped")
    
    def record_metric(self, metric_name: str, value: float, 
        timestamp: Optional[datetime] = None) -> None:
        """Record metric value"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.metrics_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Check SLA compliance
        self._check_sla_compliance(metric_name, value)
    
    def record_request(self, success: bool, response_time_ms: float) -> None:
        """Record request metrics"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        self.response_times.append(response_time_ms)
        
        # Record derived metrics
        self.record_metric("response_time", response_time_ms)
        self.record_metric("error_rate", self._calculate_error_rate())
        self.record_metric("availability", self._calculate_availability())
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        pass
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    def _calculate_availability(self) -> float:
        """Calculate system availability"""
        pass
        uptime = (datetime.utcnow() - self.uptime_start).total_seconds()
        # Simple availability calculation (can be enhanced)
        error_rate = self._calculate_error_rate()
        return max(0.0, 100.0 - error_rate)
    
    def _check_sla_compliance(self, metric_name: str, value: float) -> None:
        """Check if metric violates SLA"""
        if metric_name not in self.sla_thresholds:
            return
        
        sla = self.sla_thresholds[metric_name]
        violation = False
        
        if sla.comparison == "lt" and value >= sla.threshold_value:
            violation = True
        elif sla.comparison == "gt" and value <= sla.threshold_value:
            violation = True
        elif sla.comparison == "eq" and value != sla.threshold_value:
            violation = True
        
        if violation:
            violation_record = {
                'metric_name': metric_name,
                'threshold': sla.threshold_value,
                'actual_value': value,
                'severity': sla.severity,
                'description': sla.description,
                'timestamp': datetime.utcnow()
            }
            
            self.sla_violations.append(violation_record)
            
            # Trigger alerts
            asyncio.create_task(self._trigger_alert(violation_record))
            
            logger.warning(f"SLA violation: {metric_name} = {value} "
                          f"(threshold: {sla.comparison} {sla.threshold_value})")
    
        async def _trigger_alert(self, violation: Dict[str, Any]) -> None:
        """Trigger alert for SLA violation"""
        for callback in self.alert_callbacks:
        try:
            await callback(violation)
        except Exception as e:
        logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
        async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        pass
        while self.is_monitoring:
        try:
            # Perform health checks
        await self._perform_health_checks()
                
        # Update SLA compliance metrics
        self._update_sla_compliance()
                
        # TDA integration
        if self.tda_integration:
            await self._update_tda_metrics()
                
        # Wait before next check
        await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
        logger.error(f"Monitoring loop error: {e}")
        await asyncio.sleep(60)  # Back off on error
    
        async def _perform_health_checks(self) -> None:
        """Perform system health checks"""
        pass
        components = [
            "orchestration_engine",
            "event_processing", 
            "pattern_matching",
            "consensus_system",
            "tda_integration"
        ]
        
        for component in components:
            health = await self._check_component_health(component)
            self.health_checks[component] = health
    
        async def _check_component_health(self, component: str) -> HealthCheck:
        """Check health of specific component"""
        # Mock health check implementation
        # In production, this would check actual component status
        
        if component == "orchestration_engine":
            metrics = {
        'active_workflows': len(self.metrics_history.get('active_workflows', [])),
        'avg_response_time': self._get_recent_average('response_time'),
        'error_rate': self._calculate_error_rate()
        }
            
        if metrics['error_rate'] > 5.0:
            status = "unhealthy"
        elif metrics['error_rate'] > 1.0:
        status = "degraded"
        else:
        status = "healthy"
        
        elif component == "tda_integration":
        if self.tda_integration:
            metrics = {'integration_active': 1.0}
        status = "healthy"
        else:
        metrics = {'integration_active': 0.0}
        status = "degraded"
        
        else:
        # Default health check
        metrics = {'status': 1.0}
        status = "healthy"
        
        return HealthCheck(
        component=component,
        status=status,
        metrics=metrics
        )
    
    def _get_recent_average(self, metric_name: str, minutes: int = 5) -> float:
        """Get recent average for metric"""
        history = self.metrics_history.get(metric_name, deque())
        if not history:
            return 0.0
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = [
            entry['value'] for entry in history
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_values:
            return 0.0
        
        return sum(recent_values) / len(recent_values)
    
    def _update_sla_compliance(self) -> None:
        """Update SLA compliance percentages"""
        pass
        for metric_name, sla in self.sla_thresholds.items():
        history = self.metrics_history.get(metric_name, deque())
        if not history:
            continue
            
        # Calculate compliance over last hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_values = [
        entry['value'] for entry in history
        if entry['timestamp'] > cutoff_time
        ]
            
        if not recent_values:
            continue
            
        compliant_count = 0
        for value in recent_values:
        if sla.comparison == "lt" and value < sla.threshold_value:
            compliant_count += 1
        elif sla.comparison == "gt" and value > sla.threshold_value:
        compliant_count += 1
        elif sla.comparison == "eq" and value == sla.threshold_value:
        compliant_count += 1
            
        compliance_rate = (compliant_count / len(recent_values)) * 100
        self.sla_compliance[metric_name] = compliance_rate
    
        async def _update_tda_metrics(self) -> None:
        """Update TDA-related metrics"""
        pass
        # Mock TDA metrics update
        # In production, this would query actual TDA system
        tda_metrics = {
            'tda_anomaly_score': 0.3,
            'tda_correlation_active': 1.0,
            'tda_response_time': 15.0
        }
        
        for metric_name, value in tda_metrics.items():
            self.record_metric(metric_name, value)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        pass
        uptime_seconds = (datetime.utcnow() - self.uptime_start).total_seconds()
        
        # Calculate response time percentiles
        if self.response_times:
            sorted_times = sorted(self.response_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
        p50 = p95 = p99 = 0.0
        
        return {
        'monitoring_active': self.is_monitoring,
        'uptime_seconds': uptime_seconds,
        'total_requests': self.total_requests,
        'error_rate': f"{self._calculate_error_rate():.2f}%",
        'availability': f"{self._calculate_availability():.2f}%",
        'response_times': {
        'p50': f"{p50:.2f}ms",
        'p95': f"{p95:.2f}ms",
        'p99': f"{p99:.2f}ms"
        },
        'sla_violations': len(self.sla_violations),
        'sla_compliance': {
        name: f"{rate:.1f}%"
        for name, rate in self.sla_compliance.items()
        },
        'health_checks': {
        component: health.status
        for component, health in self.health_checks.items()
        },
        'tda_integration': self.tda_integration is not None
        }

    # Factory function
    def create_production_monitor(tda_integration: Optional[Any] = None) -> ProductionMonitor:
        """Create production monitor with optional TDA integration"""
        return ProductionMonitor(tda_integration=tda_integration)
