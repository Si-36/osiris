"""
Production Prometheus Metrics for Enhanced AURA Systems
Real metrics that matter for production monitoring
"""
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
import time
import asyncio
from typing import Dict, Any, Optional
from functools import wraps

# Create custom registry for AURA metrics
aura_registry = CollectorRegistry()

# Enhanced System Metrics
liquid_adaptations = Counter(
'aura_liquid_adaptations_total',
'Total liquid neural network adaptations',
['component_id'],
registry=aura_registry
)

liquid_complexity = Gauge(
'aura_liquid_complexity',
'Current liquid network complexity',
['component_id'],
registry=aura_registry
)

liquid_neurons = Gauge(
'aura_liquid_neurons_active',
'Number of active liquid neurons',
['component_id', 'layer'],
registry=aura_registry
)

# Mamba-2 Context Metrics
mamba_context_length = Gauge(
'aura_mamba_context_length',
'Current Mamba-2 context buffer length',
registry=aura_registry
)

mamba_processing_time = Histogram(
'aura_mamba_processing_seconds',
'Mamba-2 processing time by context size',
buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
registry=aura_registry
)

mamba_throughput = Gauge(
'aura_mamba_throughput_contexts_per_second',
'Mamba-2 processing throughput',
registry=aura_registry
)

# Constitutional AI 3.0 Metrics
constitutional_evaluations = Counter(
'aura_constitutional_evaluations_total',
'Total constitutional evaluations',
['rule_id', 'outcome'],
registry=aura_registry
)

constitutional_corrections = Counter(
'aura_constitutional_corrections_total',
'Total auto-corrections made',
['correction_type'],
registry=aura_registry
)

constitutional_compliance = Histogram(
'aura_constitutional_compliance_score',
'Constitutional compliance scores',
buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
registry=aura_registry
)

safety_score = Gauge(
'aura_safety_score',
'Current cross-modal safety score',
registry=aura_registry
)

# System Integration Metrics
system_requests = Counter(
'aura_system_requests_total',
'Total system requests',
['system_type', 'status'],
registry=aura_registry
)

system_latency = Histogram(
'aura_system_latency_seconds',
'System processing latency',
['system_type'],
buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
registry=aura_registry
)

enhancement_status = Gauge(
'aura_enhancement_active',
'Enhancement active status (1=active, 0=inactive)',
['enhancement_name'],
registry=aura_registry
)

# Component Health Metrics
component_health = Gauge(
'aura_component_health_score',
'Component health score 0-1',
['component_id', 'component_type'],
registry=aura_registry
)

component_processing_time = Histogram(
'aura_component_processing_seconds',
'Component processing time',
['component_id', 'component_type'],
registry=aura_registry
)

# Memory System Metrics
memory_operations = Counter(
'aura_memory_operations_total',
'Total memory operations',
['operation_type', 'status'],
registry=aura_registry
)

memory_retrieval_accuracy = Histogram(
'aura_memory_retrieval_accuracy',
'Memory retrieval accuracy scores',
buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
registry=aura_registry
)

class AURAMetricsCollector:
    """Collects and updates AURA system metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.last_update = time.time()

    def record_liquid_adaptation(self, component_id: str, complexity: float, active_neurons: int, layer: str = "main"):
        """Record liquid neural network adaptation"""
        liquid_adaptations.labels(component_id=component_id).inc()
        liquid_complexity.labels(component_id=component_id).set(complexity)
        liquid_neurons.labels(component_id=component_id, layer=layer).set(active_neurons)

    def record_mamba_processing(self, context_length: int, processing_time: float, throughput: float):
        """Record Mamba-2 processing metrics"""
        mamba_context_length.set(context_length)
        mamba_processing_time.observe(processing_time)
        mamba_throughput.set(throughput)

    def record_constitutional_evaluation(self, rule_id: str, outcome: str, compliance_score: float,
                                        safety_score_val: float, corrections: Dict[str, int] = None):
        """Record Constitutional AI 3.0 evaluation"""
        constitutional_evaluations.labels(rule_id=rule_id, outcome=outcome).inc()
        constitutional_compliance.observe(compliance_score)
        safety_score.set(safety_score_val)

        if corrections:
            for correction_type, count in corrections.items():
                constitutional_corrections.labels(correction_type=correction_type).inc(count)

    def record_system_request(self, system_type: str, status: str, latency: float):
        """Record system request metrics"""
        system_requests.labels(system_type=system_type, status=status).inc()
        system_latency.labels(system_type=system_type).observe(latency)

    def update_enhancement_status(self, enhancements: Dict[str, bool]):
        """Update enhancement status metrics"""
        for enhancement_name, is_active in enhancements.items():
            enhancement_status.labels(enhancement_name=enhancement_name).set(1 if is_active else 0)

    def record_component_health(self, component_id: str, component_type: str, 
                               health_score: float, processing_time: float):
        """Record component health metrics"""
        component_health.labels(component_id=component_id, component_type=component_type).set(health_score)
        component_processing_time.labels(component_id=component_id, component_type=component_type).observe(processing_time)

    def record_memory_operation(self, operation_type: str, status: str, accuracy: Optional[float] = None):
        """Record memory operation metrics"""
        memory_operations.labels(operation_type=operation_type, status=status).inc()
        if accuracy is not None:
            memory_retrieval_accuracy.observe(accuracy)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(aura_registry).decode('utf-8')

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'metrics_collected': len(aura_registry._collector_to_names),
            'last_update': self.last_update,
            'registry_name': 'aura_enhanced_systems'
        }

# Decorator for automatic metrics collection
def track_system_metrics(system_type: str):
    """Decorator to automatically track system metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                metrics_collector.record_system_request(system_type, status, latency)

        return wrapper
    return decorator

# Global metrics collector
metrics_collector = AURAMetricsCollector()

# Convenience functions for enhanced systems
def record_liquid_metrics(component_id: str, liquid_info: Dict[str, Any]):
    """Record liquid neural network metrics"""
    metrics_collector.record_liquid_adaptation(
        component_id=component_id,
        complexity=liquid_info.get('complexity', 0.0),
        active_neurons=liquid_info.get('active_neurons', 0),
        layer=liquid_info.get('layer', 'main')
    )

def record_mamba_metrics(mamba_result: Dict[str, Any]):
    """Record Mamba-2 metrics"""
    metrics_collector.record_mamba_processing(
        context_length=mamba_result.get('context_buffer_size', 0),
        processing_time=mamba_result.get('processing_time_ms', 0) / 1000.0,
        throughput=mamba_result.get('throughput', 0)
    )

def record_constitutional_metrics(constitutional_result: Dict[str, Any]):
    """Record Constitutional AI 3.0 metrics"""
    eval_result = constitutional_result.get('constitutional_evaluation', {})

    # Count corrections by type
    corrections = {}
    if eval_result.get('auto_corrected', False):
        corrections['auto_correction'] = 1

    metrics_collector.record_constitutional_evaluation(
        rule_id='overall',
        outcome='approved' if eval_result.get('approved', False) else 'rejected',
        compliance_score=eval_result.get('constitutional_compliance', 0.0),
        safety_score_val=eval_result.get('safety_score', 0.0),
        corrections=corrections
    )

async def update_system_metrics(enhanced_aura):
    """Periodic update of system metrics"""
    while True:
        try:
            # Update enhancement status
            status = enhanced_aura.get_enhancement_status()
            enhancements = {
                name: info.get('status') == 'active'
                for name, info in status.items()
                if isinstance(info, dict) and 'status' in info
            }
            metrics_collector.update_enhancement_status(enhancements)

            # Update component health from registry
            component_stats = enhanced_aura.registry.get_component_stats()
            for comp_id, component in enhanced_aura.registry.components.items():
                health_score = 1.0 if component.status == 'active' else 0.0
                processing_time = component.processing_time

                metrics_collector.record_component_health(
                    component_id=comp_id,
                    component_type=component.type.value,
                    health_score=health_score,
                    processing_time=processing_time
                )

            metrics_collector.last_update = time.time()

        except Exception as e:
            print(f"Metrics update error: {e}")

        await asyncio.sleep(30)  # Update every 30 seconds
