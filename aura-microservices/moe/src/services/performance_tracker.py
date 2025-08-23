"""
Performance Tracker Service
Track and analyze routing performance metrics
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics for a time window"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    strategy_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    service_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timestamp: float = field(default_factory=time.time)


class PerformanceTracker:
    """
    Track routing performance over time
    """
    
    def __init__(self, window_size: int = 3600):  # 1 hour default
        self.window_size = window_size
        self.logger = logger.bind(service="performance_tracker")
        
        # Performance windows
        self.current_window = PerformanceMetrics()
        self.historical_windows: deque = deque(maxlen=24)  # 24 hours
        
        # Real-time tracking
        self.recent_latencies: deque = deque(maxlen=1000)
        self.strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.service_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Anomaly detection
        self.baseline_latency_p95 = 50.0  # ms
        self.anomaly_threshold = 2.0  # 2x baseline
        self.anomalies: deque = deque(maxlen=100)
        
        # Start time
        self.start_time = time.time()
        
    def track_request(self,
                     routing_decision: Any,
                     success: bool,
                     latency_ms: float,
                     error: Optional[str] = None):
        """
        Track a routing request
        """
        # Update current window
        self.current_window.total_requests += 1
        
        if success:
            self.current_window.successful_requests += 1
        else:
            self.current_window.failed_requests += 1
            if error:
                self.current_window.error_types[error] += 1
        
        # Track latency
        self.current_window.total_latency_ms += latency_ms
        self.current_window.latencies.append(latency_ms)
        self.recent_latencies.append(latency_ms)
        
        # Track strategy
        strategy = routing_decision.routing_strategy.value
        self.current_window.strategy_counts[strategy] += 1
        self.strategy_performance[strategy].append(latency_ms)
        
        # Track services
        for service in routing_decision.selected_services:
            self.current_window.service_counts[service] += 1
            self.service_performance[service].append(latency_ms)
        
        # Check for anomalies
        if latency_ms > self.baseline_latency_p95 * self.anomaly_threshold:
            self.anomalies.append({
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "strategy": strategy,
                "services": routing_decision.selected_services
            })
        
        # Roll window if needed
        if time.time() - self.current_window.timestamp > self.window_size:
            self._roll_window()
    
    def _roll_window(self):
        """
        Roll to new performance window
        """
        # Save current window
        self.historical_windows.append(self.current_window)
        
        # Update baseline
        if self.current_window.latencies:
            self.baseline_latency_p95 = np.percentile(self.current_window.latencies, 95)
        
        # Create new window
        self.current_window = PerformanceMetrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        """
        # Calculate current metrics
        if self.current_window.total_requests > 0:
            success_rate = (
                self.current_window.successful_requests / 
                self.current_window.total_requests
            )
            avg_latency = (
                self.current_window.total_latency_ms / 
                self.current_window.total_requests
            )
        else:
            success_rate = 0.0
            avg_latency = 0.0
        
        # Calculate percentiles
        if self.recent_latencies:
            latencies = list(self.recent_latencies)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
        else:
            p50 = p95 = p99 = 0.0
        
        # Strategy performance
        strategy_latencies = {}
        for strategy, latencies in self.strategy_performance.items():
            if latencies:
                strategy_latencies[strategy] = {
                    "avg": np.mean(latencies),
                    "p95": np.percentile(latencies, 95)
                }
        
        # Service performance
        service_latencies = {}
        for service, latencies in self.service_performance.items():
            if latencies:
                service_latencies[service] = {
                    "avg": np.mean(latencies),
                    "p95": np.percentile(latencies, 95)
                }
        
        return {
            "current_window": {
                "total_requests": self.current_window.total_requests,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "strategy_distribution": dict(self.current_window.strategy_counts),
                "service_distribution": dict(self.current_window.service_counts),
                "error_distribution": dict(self.current_window.error_types)
            },
            "latency_percentiles": {
                "p50": p50,
                "p95": p95,
                "p99": p99
            },
            "strategy_performance": strategy_latencies,
            "service_performance": service_latencies,
            "anomalies": {
                "count": len(self.anomalies),
                "recent": list(self.anomalies)[-10:]  # Last 10
            },
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance trends
        """
        if len(self.historical_windows) < 2:
            return {"status": "insufficient_data"}
        
        # Extract time series
        success_rates = []
        avg_latencies = []
        request_volumes = []
        
        for window in self.historical_windows:
            if window.total_requests > 0:
                success_rates.append(
                    window.successful_requests / window.total_requests
                )
                avg_latencies.append(
                    window.total_latency_ms / window.total_requests
                )
                request_volumes.append(window.total_requests)
        
        # Calculate trends
        def calculate_trend(values: List[float]) -> str:
            if len(values) < 2:
                return "stable"
            
            # Simple linear regression
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if abs(slope) < 0.01:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
        
        return {
            "success_rate_trend": calculate_trend(success_rates),
            "latency_trend": calculate_trend(avg_latencies),
            "volume_trend": calculate_trend(request_volumes),
            "current_vs_baseline": {
                "latency_ratio": (
                    avg_latencies[-1] / self.baseline_latency_p95 
                    if avg_latencies else 1.0
                ),
                "success_rate_delta": (
                    success_rates[-1] - np.mean(success_rates[:-1])
                    if len(success_rates) > 1 else 0.0
                )
            }
        }
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get performance-based recommendations
        """
        recommendations = []
        metrics = self.get_metrics()
        
        # Check success rate
        success_rate = metrics["current_window"]["success_rate"]
        if success_rate < 0.95:
            recommendations.append({
                "type": "warning",
                "metric": "success_rate",
                "value": success_rate,
                "recommendation": "Consider adjusting circuit breaker thresholds"
            })
        
        # Check latency
        p95_latency = metrics["latency_percentiles"]["p95"]
        if p95_latency > 100:  # 100ms threshold
            recommendations.append({
                "type": "warning",
                "metric": "p95_latency",
                "value": p95_latency,
                "recommendation": "High latency detected - consider load rebalancing"
            })
        
        # Check strategy performance
        strategy_perf = metrics["strategy_performance"]
        if strategy_perf:
            best_strategy = min(
                strategy_perf.items(),
                key=lambda x: x[1]["avg"]
            )
            worst_strategy = max(
                strategy_perf.items(),
                key=lambda x: x[1]["avg"]
            )
            
            if worst_strategy[1]["avg"] > best_strategy[1]["avg"] * 2:
                recommendations.append({
                    "type": "optimization",
                    "metric": "strategy_performance",
                    "recommendation": f"Consider using {best_strategy[0]} more often"
                })
        
        # Check for anomalies
        if len(self.anomalies) > 10:
            recommendations.append({
                "type": "alert",
                "metric": "anomalies",
                "value": len(self.anomalies),
                "recommendation": "High anomaly rate - investigate service health"
            })
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.current_window = PerformanceMetrics()
        self.historical_windows.clear()
        self.recent_latencies.clear()
        self.strategy_performance.clear()
        self.service_performance.clear()
        self.anomalies.clear()
        self.start_time = time.time()