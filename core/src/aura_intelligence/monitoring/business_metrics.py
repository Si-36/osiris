"""
AURA Intelligence Business Metrics & Analytics
Real-time business intelligence for AI system performance
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import logging

from ..adapters.redis_adapter import RedisAdapter


@dataclass
class BusinessMetric:
    """Business metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class PerformanceKPI:
    """Key Performance Indicator for business tracking"""
    kpi_name: str
    current_value: float
    target_value: float
    trend: str  # 'improving', 'declining', 'stable'
    impact_score: float  # 0-100
    recommendations: List[str]


class BusinessMetricsCollector:
    """Advanced business metrics collection and analysis"""
    
    def __init__(self, redis_adapter: RedisAdapter):
        self.redis_adapter = redis_adapter
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer = deque(maxlen=10000)
        self.kpi_cache = {}
        
        # Business metric configurations
        self.metric_configs = {
            'processing_efficiency': {
                'target': 0.95,  # 95% efficiency target
                'weight': 0.3,
                'description': 'AI processing efficiency score'
            },
            'customer_satisfaction': {
                'target': 0.9,   # 90% satisfaction target
                'weight': 0.25,
                'description': 'Derived from response quality and speed'
            },
            'cost_per_request': {
                'target': 0.001,  # $0.001 per request target
                'weight': 0.2,
                'description': 'Infrastructure cost per API request'
            },
            'revenue_per_hour': {
                'target': 100.0,  # $100/hour target
                'weight': 0.15,
                'description': 'Revenue generation rate'
            },
            'system_reliability': {
                'target': 0.999,  # 99.9% uptime target
                'weight': 0.1,
                'description': 'System availability and reliability'
            }
        }
    
        async def collect_request_metrics(self, request_data: Dict[str, Any]) -> BusinessMetric:
        """Collect metrics from individual requests"""
        processing_time = request_data.get('processing_time', 0)
        gpu_utilized = request_data.get('gpu_utilized', False)
        response_quality = request_data.get('response_quality', 0.8)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(
            processing_time, gpu_utilized, response_quality
        )
        
        metric = BusinessMetric(
            name='request_efficiency',
            value=efficiency_score,
            unit='score',
            timestamp=time.time(),
            tags={
                'gpu_used': str(gpu_utilized),
                'response_type': request_data.get('response_type', 'unknown')
            },
            metadata={
                'processing_time_ms': processing_time,
                'quality_score': response_quality,
                'cost_estimate': self._estimate_request_cost(processing_time, gpu_utilized)
            }
        )
        
        await self._store_metric(metric)
        return metric
    
    def _calculate_efficiency_score(self, processing_time: float, gpu_utilized: bool, quality: float) -> float:
        """Calculate efficiency score based on multiple factors"""
        # Time efficiency (faster = better, with diminishing returns)
        time_score = min(1.0, 50.0 / max(processing_time, 1.0))  # 50ms target
        
        # GPU utilization bonus
        gpu_bonus = 0.1 if gpu_utilized else 0.0
        
        # Quality weight
        quality_weight = 0.7
        
        # Combined score
        efficiency = (time_score * 0.5 + quality * quality_weight + gpu_bonus)
        return min(1.0, efficiency)
    
    def _estimate_request_cost(self, processing_time: float, gpu_utilized: bool) -> float:
        """Estimate infrastructure cost per request"""
        # Base compute cost
        base_cost = 0.0001  # $0.0001 base
        
        # Time-based cost
        time_cost = processing_time * 0.00001  # $0.00001 per ms
        
        # GPU premium
        gpu_cost = 0.0005 if gpu_utilized else 0.0
        
        return base_cost + time_cost + gpu_cost
    
        async def calculate_kpis(self) -> List[PerformanceKPI]:
        """Calculate business KPIs from collected metrics"""
        pass
        kpis = []
        
        # Get recent metrics (last hour)
        recent_metrics = await self._get_recent_metrics(hours=1)
        
        # Processing Efficiency KPI
        efficiency_kpi = await self._calculate_efficiency_kpi(recent_metrics)
        kpis.append(efficiency_kpi)
        
        # Customer Satisfaction KPI
        satisfaction_kpi = await self._calculate_satisfaction_kpi(recent_metrics)
        kpis.append(satisfaction_kpi)
        
        # Cost Efficiency KPI
        cost_kpi = await self._calculate_cost_kpi(recent_metrics)
        kpis.append(cost_kpi)
        
        # Revenue KPI
        revenue_kpi = await self._calculate_revenue_kpi(recent_metrics)
        kpis.append(revenue_kpi)
        
        # System Reliability KPI
        reliability_kpi = await self._calculate_reliability_kpi()
        kpis.append(reliability_kpi)
        
        # Cache KPIs
        self.kpi_cache = {kpi.kpi_name: kpi for kpi in kpis}
        
        return kpis
    
        async def _calculate_efficiency_kpi(self, metrics: List[BusinessMetric]) -> PerformanceKPI:
        """Calculate processing efficiency KPI"""
        efficiency_metrics = [m for m in metrics if m.name == 'request_efficiency']
        
        if not efficiency_metrics:
            current_value = 0.5  # Default if no data
        else:
            current_value = np.mean([m.value for m in efficiency_metrics])
        
        target_value = self.metric_configs['processing_efficiency']['target']
        trend = self._calculate_trend(efficiency_metrics, 'value')
        impact_score = (current_value / target_value) * 100
        
        recommendations = []
        if current_value < target_value:
            if current_value < 0.8:
                recommendations.append("Optimize GPU utilization for faster processing")
                recommendations.append("Implement model compression for efficiency")
            recommendations.append("Review and optimize slow processing pipelines")
        
        return PerformanceKPI(
            kpi_name='processing_efficiency',
            current_value=current_value,
            target_value=target_value,
            trend=trend,
            impact_score=impact_score,
            recommendations=recommendations
        )
    
        async def _calculate_satisfaction_kpi(self, metrics: List[BusinessMetric]) -> PerformanceKPI:
        """Calculate customer satisfaction KPI based on response quality and speed"""
        quality_scores = []
        response_times = []
        
        for metric in metrics:
            if 'quality_score' in metric.metadata:
                quality_scores.append(metric.metadata['quality_score'])
            if 'processing_time_ms' in metric.metadata:
                response_times.append(metric.metadata['processing_time_ms'])
        
        # Satisfaction based on quality and speed
        if quality_scores and response_times:
            avg_quality = np.mean(quality_scores)
            avg_time = np.mean(response_times)
            
            # Speed satisfaction (sub-50ms = excellent, 50-100ms = good, >100ms = poor)
            speed_satisfaction = max(0.3, min(1.0, 100.0 / max(avg_time, 10.0)))
            
            # Combined satisfaction
            current_value = (avg_quality * 0.7 + speed_satisfaction * 0.3)
        else:
            current_value = 0.8  # Default
        
        target_value = self.metric_configs['customer_satisfaction']['target']
        trend = 'stable'  # Simplified for now
        impact_score = (current_value / target_value) * 100
        
        recommendations = []
        if current_value < target_value:
            if np.mean(response_times) > 50:
                recommendations.append("Reduce response times to improve satisfaction")
            if np.mean(quality_scores) < 0.85:
                recommendations.append("Improve AI model accuracy and response quality")
        
        return PerformanceKPI(
            kpi_name='customer_satisfaction',
            current_value=current_value,
            target_value=target_value,
            trend=trend,
            impact_score=impact_score,
            recommendations=recommendations
        )
    
        async def _calculate_cost_kpi(self, metrics: List[BusinessMetric]) -> PerformanceKPI:
        """Calculate cost per request KPI"""
        costs = []
        
        for metric in metrics:
            if 'cost_estimate' in metric.metadata:
                costs.append(metric.metadata['cost_estimate'])
        
        current_value = np.mean(costs) if costs else 0.001
        target_value = self.metric_configs['cost_per_request']['target']
        trend = self._calculate_trend(metrics, 'cost_estimate', metadata_key=True)
        impact_score = max(0, min(200, (target_value / current_value) * 100))
        
        recommendations = []
        if current_value > target_value:
            recommendations.append("Optimize GPU usage to reduce compute costs")
            recommendations.append("Implement request batching for efficiency")
            recommendations.append("Consider model quantization to reduce resource usage")
        
        return PerformanceKPI(
            kpi_name='cost_per_request',
            current_value=current_value,
            target_value=target_value,
            trend=trend,
            impact_score=impact_score,
            recommendations=recommendations
        )
    
        async def _calculate_revenue_kpi(self, metrics: List[BusinessMetric]) -> PerformanceKPI:
        """Calculate revenue per hour KPI"""
        # Simplified revenue calculation based on request volume and efficiency
        request_count = len(metrics)
        avg_efficiency = np.mean([m.value for m in metrics if m.name == 'request_efficiency'])
        
        # Estimate revenue (this would be real billing data in production)
        estimated_revenue_per_request = 0.01  # $0.01 per request
        hourly_revenue = request_count * estimated_revenue_per_request * avg_efficiency
        
        current_value = hourly_revenue
        target_value = self.metric_configs['revenue_per_hour']['target']
        trend = 'stable'  # Simplified
        impact_score = (current_value / target_value) * 100
        
        recommendations = []
        if current_value < target_value:
            recommendations.append("Increase API usage through better marketing")
            recommendations.append("Improve service quality to justify premium pricing")
            recommendations.append("Add value-added features for revenue growth")
        
        return PerformanceKPI(
            kpi_name='revenue_per_hour',
            current_value=current_value,
            target_value=target_value,
            trend=trend,
            impact_score=impact_score,
            recommendations=recommendations
        )
    
        async def _calculate_reliability_kpi(self) -> PerformanceKPI:
        """Calculate system reliability KPI"""
        pass
        # Get system health data
        try:
            health_data = await self.redis_adapter.get_data("system_health_history")
            if health_data:
                health_scores = [h.get('health_score', 0.8) for h in health_data[-100:]]
                current_value = np.mean(health_scores)
            else:
                current_value = 0.95  # Default high reliability
        except Exception:
            current_value = 0.95
        
        target_value = self.metric_configs['system_reliability']['target']
        trend = 'stable'
        impact_score = (current_value / target_value) * 100
        
        recommendations = []
        if current_value < target_value:
            recommendations.append("Investigate and fix system reliability issues")
            recommendations.append("Implement better error handling and recovery")
            recommendations.append("Add redundancy for critical components")
        
        return PerformanceKPI(
            kpi_name='system_reliability',
            current_value=current_value,
            target_value=target_value,
            trend=trend,
            impact_score=impact_score,
            recommendations=recommendations
        )
    
    def _calculate_trend(self, metrics: List[BusinessMetric], field: str, metadata_key: bool = False) -> str:
        """Calculate trend direction for metrics"""
        if len(metrics) < 10:
            return 'stable'
        
        # Get values from last 20 metrics
        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-20:]
        
        values = []
        for metric in recent_metrics:
            if metadata_key and field in metric.metadata:
                values.append(metric.metadata[field])
            elif not metadata_key and hasattr(metric, field):
                values.append(getattr(metric, field))
        
        if len(values) < 5:
            return 'stable'
        
        # Calculate trend using linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
        async def _get_recent_metrics(self, hours: int = 1) -> List[BusinessMetric]:
        """Get metrics from the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        # Get from buffer first (most recent)
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
        # If not enough data, get from Redis
        if len(recent_metrics) < 10:
            try:
                stored_metrics = await self.redis_adapter.get_data(f"business_metrics_{int(cutoff_time)}")
                if stored_metrics:
                    for metric_data in stored_metrics:
                        metric = BusinessMetric(**metric_data)
                        if metric.timestamp >= cutoff_time:
                            recent_metrics.append(metric)
            except Exception as e:
                self.logger.warning(f"Could not retrieve stored metrics: {e}")
        
        return recent_metrics
    
        async def _store_metric(self, metric: BusinessMetric):
        """Store metric in buffer and Redis"""
        # Add to buffer
        self.metrics_buffer.append(metric)
        
        # Store in Redis (batch every 100 metrics for efficiency)
        if len(self.metrics_buffer) % 100 == 0:
            await self._flush_metrics_to_redis()
    
        async def _flush_metrics_to_redis(self):
        """Flush metrics buffer to Redis"""
        pass
        if not self.metrics_buffer:
            return
        
        try:
            # Group metrics by hour for efficient storage
            metrics_by_hour = defaultdict(list)
            
            for metric in self.metrics_buffer:
                hour_key = int(metric.timestamp // 3600) * 3600  # Round to hour
                metrics_by_hour[hour_key].append(asdict(metric))
            
            # Store each hour's metrics
            for hour_key, metrics in metrics_by_hour.items():
                redis_key = f"business_metrics_{hour_key}"
                await self.redis_adapter.store_data(redis_key, metrics)
                
                # Set expiration (keep data for 30 days)
                await self.redis_adapter.redis_client.expire(redis_key, 30 * 24 * 3600)
            
            self.logger.info(f"Flushed {len(self.metrics_buffer)} metrics to Redis")
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics to Redis: {e}")
    
        async def get_business_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive business dashboard data"""
        pass
        kpis = await self.calculate_kpis()
        recent_metrics = await self._get_recent_metrics(hours=24)
        
        # Calculate aggregated statistics
        hourly_stats = self._calculate_hourly_stats(recent_metrics)
        
        return {
            'kpis': [asdict(kpi) for kpi in kpis],
            'overall_score': self._calculate_overall_business_score(kpis),
            'hourly_stats': hourly_stats,
            'recommendations': self._get_priority_recommendations(kpis),
            'alerts': self._generate_business_alerts(kpis),
            'timestamp': time.time()
        }
    
    def _calculate_overall_business_score(self, kpis: List[PerformanceKPI]) -> float:
        """Calculate weighted overall business performance score"""
        total_score = 0.0
        total_weight = 0.0
        
        for kpi in kpis:
            config = self.metric_configs.get(kpi.kpi_name, {})
            weight = config.get('weight', 0.1)
            
            # Normalize impact score to 0-1 range
            normalized_score = min(1.0, kpi.impact_score / 100.0)
            
            total_score += normalized_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_hourly_stats(self, metrics: List[BusinessMetric]) -> Dict[str, Any]:
        """Calculate hourly statistics for trending"""
        hourly_data = defaultdict(lambda: {
            'requests': 0,
            'avg_efficiency': 0.0,
            'total_cost': 0.0,
            'avg_quality': 0.0
        })
        
        for metric in metrics:
            hour = int(metric.timestamp // 3600) * 3600
            hourly_data[hour]['requests'] += 1
            
            if metric.name == 'request_efficiency':
                hourly_data[hour]['avg_efficiency'] += metric.value
            
            if 'cost_estimate' in metric.metadata:
                hourly_data[hour]['total_cost'] += metric.metadata['cost_estimate']
            
            if 'quality_score' in metric.metadata:
                hourly_data[hour]['avg_quality'] += metric.metadata['quality_score']
        
        # Average the accumulated values
        for hour_data in hourly_data.values():
            if hour_data['requests'] > 0:
                hour_data['avg_efficiency'] /= hour_data['requests']
                hour_data['avg_quality'] /= hour_data['requests']
        
        return dict(hourly_data)
    
    def _get_priority_recommendations(self, kpis: List[PerformanceKPI]) -> List[Dict[str, Any]]:
        """Get prioritized recommendations across all KPIs"""
        all_recommendations = []
        
        for kpi in kpis:
            for rec in kpi.recommendations:
                all_recommendations.append({
                    'recommendation': rec,
                    'kpi': kpi.kpi_name,
                    'impact_score': kpi.impact_score,
                    'priority': self._calculate_priority(kpi)
                })
        
        # Sort by priority (lower score = higher priority)
        all_recommendations.sort(key=lambda x: x['priority'])
        
        return all_recommendations[:5]  # Top 5 recommendations
    
    def _calculate_priority(self, kpi: PerformanceKPI) -> float:
        """Calculate recommendation priority (lower = higher priority)"""
        # Priority based on how far below target and KPI weight
        target_gap = max(0, kpi.target_value - kpi.current_value)
        weight = self.metric_configs.get(kpi.kpi_name, {}).get('weight', 0.1)
        
        return target_gap / (weight + 0.01)  # Lower score = higher priority
    
    def _generate_business_alerts(self, kpis: List[PerformanceKPI]) -> List[Dict[str, Any]]:
        """Generate business-level alerts"""
        alerts = []
        
        for kpi in kpis:
            # Critical alert if significantly below target
            if kpi.impact_score < 70:
                alerts.append({
                    'level': 'critical',
                    'kpi': kpi.kpi_name,
                    'message': f"{kpi.kpi_name} is {kpi.impact_score:.1f}% of target",
                    'current_value': kpi.current_value,
                    'target_value': kpi.target_value
                })
            # Warning if moderately below target
            elif kpi.impact_score < 85:
                alerts.append({
                    'level': 'warning',
                    'kpi': kpi.kpi_name,
                    'message': f"{kpi.kpi_name} is below target ({kpi.impact_score:.1f}%)",
                    'current_value': kpi.current_value,
                    'target_value': kpi.target_value
                })
        
        return alerts


# Export main class
__all__ = ['BusinessMetricsCollector', 'BusinessMetric', 'PerformanceKPI']