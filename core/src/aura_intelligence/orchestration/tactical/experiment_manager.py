"""
Experiment Manager - Shadow Mode Testing
========================================
Runs experimental workflows in shadow mode for safe testing
Based on DZone 2025 LLM orchestration patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
import time
from enum import Enum
import json
import uuid
from collections import defaultdict
import numpy as np

# AURA imports  
from .pipeline_registry import Pipeline, PipelineRegistry
from ...observability.metrics import MetricsCollector

import logging
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle states"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment"""
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Latency metrics (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    # Comparison with control
    latency_delta_percent: float = 0.0
    success_rate_delta: float = 0.0
    
    # Resource usage
    avg_cpu_usage: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Experiment:
    """Shadow mode experiment definition"""
    experiment_id: str
    name: str
    description: str
    
    # Experiment configuration
    treatment_pipeline: Pipeline
    control_pipeline_id: str
    
    # Sampling configuration
    sample_rate: float = 1.0  # Percentage of traffic to shadow
    max_executions: Optional[int] = None
    duration_hours: Optional[float] = None
    
    # Status tracking
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    treatment_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    control_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    
    # Guardrails
    max_latency_increase_percent: float = 20.0
    min_success_rate: float = 0.95
    auto_stop_on_failure: bool = True
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    

class ExperimentManager:
    """
    Manages shadow mode experiments for safe testing
    Runs experimental pipelines alongside production
    """
    
    def __init__(self, 
                 pipeline_registry: PipelineRegistry,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.pipeline_registry = pipeline_registry
        
        # Experiment storage
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: Set[str] = set()
        
        # Metrics tracking
        self.metrics_collector = MetricsCollector()
        self.execution_latencies: Dict[str, List[float]] = defaultdict(list)
        
        # Background tasks
        self.monitor_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Experiment Manager initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_concurrent_experiments': 5,
            'default_sample_rate': 0.1,  # 10%
            'metrics_buffer_size': 1000,
            'health_check_interval': 60,  # seconds
            'result_persistence_enabled': True
        }
        
    async def create_experiment(self,
                              name: str,
                              treatment_pipeline: Pipeline,
                              control_pipeline_id: str,
                              **kwargs) -> Experiment:
        """Create a new shadow experiment"""
        # Validate
        if control_pipeline_id not in self.pipeline_registry.active_pipelines:
            raise ValueError(f"Control pipeline {control_pipeline_id} not active")
            
        # Create experiment
        experiment = Experiment(
            experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
            name=name,
            description=kwargs.get('description', ''),
            treatment_pipeline=treatment_pipeline,
            control_pipeline_id=control_pipeline_id,
            sample_rate=kwargs.get('sample_rate', self.config['default_sample_rate']),
            max_executions=kwargs.get('max_executions'),
            duration_hours=kwargs.get('duration_hours'),
            tags=kwargs.get('tags', []),
            owner=kwargs.get('owner', '')
        )
        
        # Store
        self.experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Created experiment {experiment.experiment_id}: {name}")
        
        return experiment
        
    async def start_experiment(self, experiment_id: str):
        """Start running an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        
        # Check concurrent limit
        if len(self.active_experiments) >= self.config['max_concurrent_experiments']:
            raise RuntimeError("Too many concurrent experiments")
            
        # Update status
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()
        self.active_experiments.add(experiment_id)
        
        # Start monitoring
        self.monitor_tasks[experiment_id] = asyncio.create_task(
            self._monitor_experiment(experiment_id)
        )
        
        logger.info(f"Started experiment {experiment_id}")
        
    async def _monitor_experiment(self, experiment_id: str):
        """Monitor experiment health and stop if needed"""
        experiment = self.experiments[experiment_id]
        
        while experiment.status == ExperimentStatus.RUNNING:
            try:
                # Check completion criteria
                if await self._should_complete_experiment(experiment):
                    await self.complete_experiment(experiment_id)
                    break
                    
                # Check health
                if await self._check_experiment_health(experiment):
                    if experiment.auto_stop_on_failure:
                        await self.stop_experiment(experiment_id, "Health check failed")
                        break
                        
                await asyncio.sleep(self.config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error monitoring experiment {experiment_id}: {e}")
                experiment.status = ExperimentStatus.FAILED
                break
                
    async def _should_complete_experiment(self, experiment: Experiment) -> bool:
        """Check if experiment should complete"""
        # Max executions reached
        if (experiment.max_executions and 
            experiment.treatment_metrics.execution_count >= experiment.max_executions):
            return True
            
        # Duration exceeded
        if experiment.duration_hours and experiment.started_at:
            elapsed_hours = (time.time() - experiment.started_at) / 3600
            if elapsed_hours >= experiment.duration_hours:
                return True
                
        return False
        
    async def _check_experiment_health(self, experiment: Experiment) -> bool:
        """Check if experiment is healthy"""
        treatment = experiment.treatment_metrics
        control = experiment.control_metrics
        
        # Need minimum data
        if treatment.execution_count < 10:
            return True
            
        # Check success rate
        treatment_success_rate = treatment.success_count / treatment.execution_count
        if treatment_success_rate < experiment.min_success_rate:
            logger.warning(f"Experiment {experiment.experiment_id}: "
                         f"Low success rate {treatment_success_rate:.2%}")
            return False
            
        # Check latency increase
        if control.latency_p95 > 0:
            latency_increase = ((treatment.latency_p95 - control.latency_p95) / 
                              control.latency_p95 * 100)
            if latency_increase > experiment.max_latency_increase_percent:
                logger.warning(f"Experiment {experiment.experiment_id}: "
                             f"High latency increase {latency_increase:.1f}%")
                return False
                
        return True
        
    async def execute_shadow(self,
                           pipeline_id: str,
                           context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute shadow experiments for a pipeline
        Returns None (shadow results not used)
        """
        # Find applicable experiments
        applicable_experiments = [
            exp for exp in self.experiments.values()
            if (exp.status == ExperimentStatus.RUNNING and
                exp.control_pipeline_id == pipeline_id)
        ]
        
        for experiment in applicable_experiments:
            # Sample decision
            if np.random.random() > experiment.sample_rate:
                continue
                
            # Execute in background
            asyncio.create_task(
                self._execute_experiment_async(experiment, context)
            )
            
        return None
        
    async def _execute_experiment_async(self,
                                      experiment: Experiment,
                                      context: Dict[str, Any]):
        """Execute experiment asynchronously"""
        try:
            # Execute treatment pipeline
            start_time = time.perf_counter()
            
            treatment_result = await self._execute_treatment(
                experiment.treatment_pipeline,
                context
            )
            
            treatment_latency = (time.perf_counter() - start_time) * 1000
            
            # Execute control for comparison
            control_start = time.perf_counter()
            
            control_result = await self.pipeline_registry.execute_pipeline(
                experiment.control_pipeline_id,
                context
            )
            
            control_latency = (time.perf_counter() - control_start) * 1000
            
            # Update metrics
            await self._update_experiment_metrics(
                experiment,
                treatment_result,
                treatment_latency,
                control_result,
                control_latency
            )
            
        except Exception as e:
            logger.error(f"Experiment {experiment.experiment_id} execution failed: {e}")
            experiment.treatment_metrics.failure_count += 1
            
    async def _execute_treatment(self,
                               pipeline: Pipeline,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute treatment pipeline"""
        # Shadow execution - isolated from production
        shadow_context = context.copy()
        shadow_context['is_shadow'] = True
        
        # Would execute actual pipeline phases
        # For now, mock execution
        result = {
            'success': np.random.random() > 0.05,  # 95% success
            'phases': {},
            'latency_ms': np.random.gamma(50, 10)  # Gamma distribution
        }
        
        return result
        
    async def _update_experiment_metrics(self,
                                       experiment: Experiment,
                                       treatment_result: Dict[str, Any],
                                       treatment_latency: float,
                                       control_result: Dict[str, Any],
                                       control_latency: float):
        """Update experiment metrics"""
        # Update treatment metrics
        treatment = experiment.treatment_metrics
        treatment.execution_count += 1
        
        if treatment_result.get('success'):
            treatment.success_count += 1
        else:
            treatment.failure_count += 1
            
        # Track latencies
        exp_key = f"{experiment.experiment_id}_treatment"
        self.execution_latencies[exp_key].append(treatment_latency)
        
        # Limit buffer size
        if len(self.execution_latencies[exp_key]) > self.config['metrics_buffer_size']:
            self.execution_latencies[exp_key] = (
                self.execution_latencies[exp_key][-self.config['metrics_buffer_size']:]
            )
            
        # Calculate percentiles
        latencies = self.execution_latencies[exp_key]
        if len(latencies) >= 10:
            treatment.latency_p50 = np.percentile(latencies, 50)
            treatment.latency_p95 = np.percentile(latencies, 95) 
            treatment.latency_p99 = np.percentile(latencies, 99)
            
        # Update control metrics
        control = experiment.control_metrics
        control.execution_count += 1
        
        if control_result.get('success'):
            control.success_count += 1
            
        # Calculate deltas
        if control.execution_count > 0:
            control_success_rate = control.success_count / control.execution_count
            treatment_success_rate = treatment.success_count / treatment.execution_count
            treatment.success_rate_delta = treatment_success_rate - control_success_rate
            
        if control.latency_p95 > 0:
            treatment.latency_delta_percent = (
                (treatment.latency_p95 - control.latency_p95) / 
                control.latency_p95 * 100
            )
            
        # Emit metrics
        self.metrics_collector.histogram(
            'experiment_latency_ms',
            treatment_latency,
            labels={
                'experiment_id': experiment.experiment_id,
                'variant': 'treatment'
            }
        )
        
    async def pause_experiment(self, experiment_id: str):
        """Pause a running experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.PAUSED
        self.active_experiments.discard(experiment_id)
        
        # Cancel monitoring
        if experiment_id in self.monitor_tasks:
            self.monitor_tasks[experiment_id].cancel()
            
        logger.info(f"Paused experiment {experiment_id}")
        
    async def resume_experiment(self, experiment_id: str):
        """Resume a paused experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Experiment {experiment_id} is not paused")
            
        await self.start_experiment(experiment_id)
        
    async def complete_experiment(self, experiment_id: str):
        """Complete an experiment and analyze results"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = time.time()
        self.active_experiments.discard(experiment_id)
        
        # Cancel monitoring
        if experiment_id in self.monitor_tasks:
            self.monitor_tasks[experiment_id].cancel()
            
        # Analyze results
        analysis = self._analyze_experiment_results(experiment)
        
        logger.info(f"Completed experiment {experiment_id}: {analysis['recommendation']}")
        
        return analysis
        
    def _analyze_experiment_results(self, experiment: Experiment) -> Dict[str, Any]:
        """Analyze experiment results and make recommendation"""
        treatment = experiment.treatment_metrics
        control = experiment.control_metrics
        
        # Calculate statistics
        treatment_success_rate = (treatment.success_count / treatment.execution_count 
                                if treatment.execution_count > 0 else 0)
        control_success_rate = (control.success_count / control.execution_count
                              if control.execution_count > 0 else 0)
        
        # Determine recommendation
        recommendation = "neutral"
        confidence = 0.0
        
        if treatment.execution_count >= 100:  # Minimum sample size
            # Success rate improvement
            if treatment_success_rate > control_success_rate + 0.02:
                recommendation = "adopt"
                confidence = 0.8
            # Performance regression
            elif (treatment.latency_delta_percent > 10 or
                  treatment_success_rate < control_success_rate - 0.02):
                recommendation = "reject"
                confidence = 0.9
            # No significant difference
            else:
                recommendation = "neutral"
                confidence = 0.6
                
        return {
            'experiment_id': experiment.experiment_id,
            'recommendation': recommendation,
            'confidence': confidence,
            'treatment_success_rate': treatment_success_rate,
            'control_success_rate': control_success_rate,
            'latency_delta_percent': treatment.latency_delta_percent,
            'sample_size': treatment.execution_count,
            'duration_hours': ((experiment.completed_at - experiment.started_at) / 3600
                             if experiment.completed_at and experiment.started_at else 0)
        }
        
    async def stop_experiment(self, experiment_id: str, reason: str = ""):
        """Stop an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.CANCELLED
        experiment.completed_at = time.time()
        self.active_experiments.discard(experiment_id)
        
        # Cancel monitoring
        if experiment_id in self.monitor_tasks:
            self.monitor_tasks[experiment_id].cancel()
            
        logger.info(f"Stopped experiment {experiment_id}: {reason}")
        
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment status"""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
            
        experiment = self.experiments[experiment_id]
        
        return {
            'experiment_id': experiment.experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'sample_rate': experiment.sample_rate,
            'treatment_pipeline': str(experiment.treatment_pipeline.version),
            'control_pipeline': experiment.control_pipeline_id,
            'metrics': {
                'treatment': {
                    'executions': experiment.treatment_metrics.execution_count,
                    'success_rate': (experiment.treatment_metrics.success_count / 
                                   max(experiment.treatment_metrics.execution_count, 1)),
                    'latency_p95': experiment.treatment_metrics.latency_p95
                },
                'control': {
                    'executions': experiment.control_metrics.execution_count,
                    'success_rate': (experiment.control_metrics.success_count /
                                   max(experiment.control_metrics.execution_count, 1))
                },
                'deltas': {
                    'success_rate': experiment.treatment_metrics.success_rate_delta,
                    'latency_percent': experiment.treatment_metrics.latency_delta_percent
                }
            },
            'runtime_hours': ((time.time() - experiment.started_at) / 3600
                            if experiment.started_at else 0)
        }
        
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments"""
        return [
            self.get_experiment_status(exp_id)
            for exp_id in self.active_experiments
        ]