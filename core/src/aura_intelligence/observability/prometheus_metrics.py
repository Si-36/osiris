"""Prometheus Metrics for all 209 AURA components"""
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from typing import Dict, Any
from ..components.real_registry import get_real_registry

class PrometheusMetricsManager:
    """Alias for compatibility"""
    pass

class AURAPrometheusMetrics:
    """Complete Prometheus metrics for AURA Intelligence"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.component_registry = get_real_registry()
        
        # Component metrics
        self.component_processing_time = Histogram(
            'aura_component_processing_seconds',
            'Component processing time',
            ['component_id', 'component_type'],
            registry=self.registry
        )
        
        self.component_data_processed = Counter(
            'aura_component_data_processed_total',
            'Total data processed by component',
            ['component_id', 'component_type'],
            registry=self.registry
        )
        
        self.component_health = Gauge(
            'aura_component_health_score',
            'Component health score 0-1',
            ['component_id', 'component_type'],
            registry=self.registry
        )
        
        # Bio-homeostatic metrics
        self.metabolic_budget = Gauge(
            'aura_metabolic_budget',
            'Component energy budget',
            ['component_id'],
            registry=self.registry
        )
        
        self.metabolic_consumption = Gauge(
            'aura_metabolic_consumption',
            'Component energy consumption',
            ['component_id'],
            registry=self.registry
        )
        
        self.metabolic_throttles = Counter(
            'aura_metabolic_throttles_total',
            'Component throttle events',
            ['component_id', 'reason'],
            registry=self.registry
        )
        
        # CoRaL metrics
        self.coral_influence = Gauge(
            'aura_coral_causal_influence',
            'CoRaL causal influence score',
            registry=self.registry
        )
        
        self.coral_throughput = Gauge(
            'aura_coral_throughput_items_per_sec',
            'CoRaL processing throughput',
            registry=self.registry
        )
        
        # TDA metrics
        self.tda_topology_score = Gauge(
            'aura_tda_topology_score',
            'TDA topology health score',
            ['system_id'],
            registry=self.registry
        )
        
        self.tda_bottlenecks = Gauge(
            'aura_tda_bottlenecks_count',
            'Number of detected bottlenecks',
            ['system_id'],
            registry=self.registry
        )
        
        # DPO metrics
        self.dpo_preference_pairs = Gauge(
            'aura_dpo_preference_pairs_total',
            'Total DPO preference pairs',
            registry=self.registry
        )
        
        self.dpo_training_loss = Gauge(
            'aura_dpo_training_loss',
            'DPO training loss',
            registry=self.registry
        )
        
        # Swarm metrics
        self.swarm_errors_detected = Counter(
            'aura_swarm_errors_detected_total',
            'Swarm intelligence errors detected',
            registry=self.registry
        )
        
        self.swarm_health = Gauge(
            'aura_swarm_avg_health',
            'Average swarm component health',
            registry=self.registry
        )
    
    def update_component_metrics(self):
        """Update metrics for all 209 components"""
        pass
        for comp_id, component in self.component_registry.components.items():
            comp_type = component.type.value
            
            # Update component health (based on processing efficiency)
            health_score = 1.0 if component.status == 'active' else 0.0
            if component.processing_time > 0:
                health_score *= min(1.0, 1.0 / (component.processing_time * 10))
            
            self.component_health.labels(
                component_id=comp_id,
                component_type=comp_type
            ).set(health_score)
            
            # Update processing time
            if component.processing_time > 0:
                self.component_processing_time.labels(
                    component_id=comp_id,
                    component_type=comp_type
                ).observe(component.processing_time)
            
            # Update data processed counter
            self.component_data_processed.labels(
                component_id=comp_id,
                component_type=comp_type
            )._value._value = component.data_processed
    
    def update_bio_metrics(self, metabolic_data: Dict[str, Any]):
        """Update bio-homeostatic metrics"""
        for comp_id, data in metabolic_data.items():
            if isinstance(data, dict):
                budget = data.get('budget', 0)
                consumption = data.get('consumption', 0)
                
                self.metabolic_budget.labels(component_id=comp_id).set(budget)
                self.metabolic_consumption.labels(component_id=comp_id).set(consumption)
    
    def update_coral_metrics(self, coral_stats: Dict[str, Any]):
        """Update CoRaL system metrics"""
        self.coral_influence.set(coral_stats.get('avg_influence', 0))
        self.coral_throughput.set(coral_stats.get('throughput', 0))
    
    def update_tda_metrics(self, tda_data: Dict[str, Any]):
        """Update TDA system metrics"""
        for system_id, health in tda_data.items():
            if hasattr(health, 'topology_score'):
                self.tda_topology_score.labels(system_id=system_id).set(health.topology_score)
                self.tda_bottlenecks.labels(system_id=system_id).set(len(health.bottlenecks))
    
    def update_dpo_metrics(self, dpo_stats: Dict[str, Any]):
        """Update DPO system metrics"""
        training = dpo_stats.get('training_progress', {})
        self.dpo_preference_pairs.set(training.get('total_preference_pairs', 0))
        self.dpo_training_loss.set(training.get('average_loss', 0))
    
    def update_swarm_metrics(self, swarm_stats: Dict[str, Any]):
        """Update swarm intelligence metrics"""
        self.swarm_health.set(swarm_stats.get('avg_component_health', 1.0))
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        pass
        return generate_latest(self.registry).decode('utf-8')

# Global metrics instance
_aura_metrics = None

def get_aura_metrics():
        global _aura_metrics
        if _aura_metrics is None:
        _aura_metrics = AURAPrometheusMetrics()
        return _aura_metrics
