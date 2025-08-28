"""
Pipeline Registry - Semantic Versioning & A/B Testing
====================================================
Manages cognitive pipelines with versioning and gradual rollout
Based on PracData 2025 patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
from enum import Enum
import json
import hashlib
from collections import defaultdict
import semver

# AURA imports
from ...components.registry import get_registry, ComponentRole

import logging
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline deployment stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class PipelineVersion:
    """Semantic version for pipelines"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self):
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
        
    def __lt__(self, other):
        return semver.compare(str(self), str(other)) < 0
        
    @classmethod
    def parse(cls, version_str: str):
        """Parse version string"""
        v = semver.VersionInfo.parse(version_str)
        return cls(
            major=v.major,
            minor=v.minor,
            patch=v.patch,
            prerelease=v.prerelease,
            build=v.build
        )


@dataclass
class Pipeline:
    """Cognitive pipeline definition"""
    pipeline_id: str
    name: str
    version: PipelineVersion
    stage: PipelineStage = PipelineStage.DEVELOPMENT
    
    # Pipeline components
    phases: List[str] = field(default_factory=list)  # Ordered phase names
    phase_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Routing configuration
    conditional_routes: Dict[str, Callable] = field(default_factory=dict)
    fallback_routes: Dict[str, str] = field(default_factory=dict)
    
    # Performance targets
    sla_targets: Dict[str, float] = field(default_factory=dict)
    
    # Deployment info
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deployed_at: Optional[float] = None
    
    # A/B testing
    traffic_percentage: float = 100.0
    ab_test_id: Optional[str] = None
    
    # Metrics
    execution_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    
    def get_checksum(self) -> str:
        """Calculate pipeline checksum for versioning"""
        content = json.dumps({
            'phases': self.phases,
            'phase_configs': self.phase_configs,
            'sla_targets': self.sla_targets
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class PipelineRegistry:
    """
    Central registry for cognitive pipelines
    Manages versioning, A/B testing, and gradual rollout
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Pipeline storage
        self.pipelines: Dict[str, Dict[str, Pipeline]] = defaultdict(dict)  # id -> version -> pipeline
        self.active_pipelines: Dict[str, str] = {}  # id -> active version
        
        # A/B test tracking
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Execution history
        self.execution_history = deque(maxlen=10000)
        
        # Component registry
        self.component_registry = get_registry()
        
        logger.info("Pipeline Registry initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_versions_per_pipeline': 10,
            'default_canary_percentage': 10.0,
            'ab_test_duration_hours': 24,
            'min_executions_for_promotion': 100,
            'success_rate_threshold': 0.95
        }
        
    async def register_pipeline(self, pipeline: Pipeline) -> str:
        """
        Register a new pipeline version
        Returns registration ID
        """
        # Validate pipeline
        if not self._validate_pipeline(pipeline):
            raise ValueError(f"Invalid pipeline: {pipeline.pipeline_id}")
            
        # Store pipeline
        version_str = str(pipeline.version)
        self.pipelines[pipeline.pipeline_id][version_str] = pipeline
        
        # Clean old versions
        await self._cleanup_old_versions(pipeline.pipeline_id)
        
        # Set as active if first version
        if pipeline.pipeline_id not in self.active_pipelines:
            self.active_pipelines[pipeline.pipeline_id] = version_str
            pipeline.stage = PipelineStage.PRODUCTION
            
        logger.info(f"Registered pipeline {pipeline.pipeline_id} version {version_str}")
        
        return f"{pipeline.pipeline_id}:{version_str}"
        
    def _validate_pipeline(self, pipeline: Pipeline) -> bool:
        """Validate pipeline configuration"""
        # Check required fields
        if not pipeline.pipeline_id or not pipeline.phases:
            return False
            
        # Validate phases exist in registry
        for phase in pipeline.phases:
            if phase not in ['perception', 'inference', 'consensus', 'action']:
                # Check if custom phase is registered
                try:
                    self.component_registry.get_component(phase)
                except:
                    logger.warning(f"Phase {phase} not found in registry")
                    return False
                    
        # Validate SLA targets
        for metric, target in pipeline.sla_targets.items():
            if target <= 0:
                return False
                
        return True
        
    async def _cleanup_old_versions(self, pipeline_id: str):
        """Remove old versions beyond retention limit"""
        versions = self.pipelines[pipeline_id]
        
        if len(versions) > self.config['max_versions_per_pipeline']:
            # Sort by version
            sorted_versions = sorted(versions.keys(), 
                                   key=lambda v: PipelineVersion.parse(v))
            
            # Keep only recent versions
            to_remove = sorted_versions[:-self.config['max_versions_per_pipeline']]
            
            for version in to_remove:
                # Don't remove active version
                if self.active_pipelines.get(pipeline_id) != version:
                    del versions[version]
                    logger.debug(f"Removed old version {version} of {pipeline_id}")
                    
    async def promote_pipeline(self, 
                             pipeline_id: str,
                             version: str,
                             strategy: str = "canary") -> Dict[str, Any]:
        """
        Promote pipeline version to production
        Strategies: canary, blue_green, gradual
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        if version not in self.pipelines[pipeline_id]:
            raise ValueError(f"Version {version} not found for {pipeline_id}")
            
        pipeline = self.pipelines[pipeline_id][version]
        
        if strategy == "canary":
            return await self._promote_canary(pipeline)
        elif strategy == "blue_green":
            return await self._promote_blue_green(pipeline)
        elif strategy == "gradual":
            return await self._promote_gradual(pipeline)
        else:
            raise ValueError(f"Unknown promotion strategy: {strategy}")
            
    async def _promote_canary(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Canary deployment with gradual traffic shift"""
        pipeline_id = pipeline.pipeline_id
        version = str(pipeline.version)
        
        # Create A/B test
        ab_test_id = f"ab_{pipeline_id}_{int(time.time())}"
        
        self.ab_tests[ab_test_id] = {
            'pipeline_id': pipeline_id,
            'control_version': self.active_pipelines[pipeline_id],
            'treatment_version': version,
            'start_time': time.time(),
            'traffic_split': self.config['default_canary_percentage'],
            'metrics': defaultdict(list)
        }
        
        # Update pipeline
        pipeline.stage = PipelineStage.CANARY
        pipeline.traffic_percentage = self.config['default_canary_percentage']
        pipeline.ab_test_id = ab_test_id
        pipeline.deployed_at = time.time()
        
        logger.info(f"Started canary deployment for {pipeline_id} v{version}")
        
        # Schedule gradual rollout
        asyncio.create_task(self._manage_canary_rollout(ab_test_id))
        
        return {
            'deployment_type': 'canary',
            'ab_test_id': ab_test_id,
            'initial_traffic': pipeline.traffic_percentage,
            'estimated_duration_hours': self.config['ab_test_duration_hours']
        }
        
    async def _manage_canary_rollout(self, ab_test_id: str):
        """Manage gradual canary rollout"""
        ab_test = self.ab_tests[ab_test_id]
        pipeline_id = ab_test['pipeline_id']
        treatment_version = ab_test['treatment_version']
        pipeline = self.pipelines[pipeline_id][treatment_version]
        
        # Gradual rollout schedule
        rollout_schedule = [
            (6, 25),   # 6 hours: 25%
            (12, 50),  # 12 hours: 50%
            (18, 75),  # 18 hours: 75%
            (24, 100)  # 24 hours: 100%
        ]
        
        start_time = ab_test['start_time']
        
        for hours, percentage in rollout_schedule:
            # Wait until scheduled time
            wait_time = start_time + hours * 3600 - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
            # Check if should continue
            if not await self._check_canary_health(ab_test_id):
                await self._rollback_canary(ab_test_id)
                return
                
            # Update traffic split
            ab_test['traffic_split'] = percentage
            pipeline.traffic_percentage = percentage
            
            logger.info(f"Canary {ab_test_id}: Traffic increased to {percentage}%")
            
        # Finalize deployment
        await self._finalize_canary(ab_test_id)
        
    async def _check_canary_health(self, ab_test_id: str) -> bool:
        """Check if canary is healthy"""
        ab_test = self.ab_tests[ab_test_id]
        metrics = ab_test['metrics']
        
        if not metrics['success_rate'] or not metrics['latency']:
            return True  # Not enough data
            
        # Compare with control
        control_success = np.mean(metrics['control_success_rate'][-100:])
        treatment_success = np.mean(metrics['treatment_success_rate'][-100:])
        
        control_latency = np.mean(metrics['control_latency'][-100:])
        treatment_latency = np.mean(metrics['treatment_latency'][-100:])
        
        # Check degradation
        if treatment_success < control_success * 0.95:
            logger.warning(f"Canary {ab_test_id}: Success rate degraded")
            return False
            
        if treatment_latency > control_latency * 1.2:
            logger.warning(f"Canary {ab_test_id}: Latency increased")
            return False
            
        return True
        
    async def _rollback_canary(self, ab_test_id: str):
        """Rollback failed canary"""
        ab_test = self.ab_tests[ab_test_id]
        pipeline_id = ab_test['pipeline_id']
        treatment_version = ab_test['treatment_version']
        
        pipeline = self.pipelines[pipeline_id][treatment_version]
        pipeline.stage = PipelineStage.DEPRECATED
        pipeline.traffic_percentage = 0.0
        
        ab_test['traffic_split'] = 0.0
        ab_test['rollback_time'] = time.time()
        
        logger.warning(f"Rolled back canary {ab_test_id}")
        
    async def _finalize_canary(self, ab_test_id: str):
        """Finalize successful canary"""
        ab_test = self.ab_tests[ab_test_id]
        pipeline_id = ab_test['pipeline_id']
        treatment_version = ab_test['treatment_version']
        control_version = ab_test['control_version']
        
        # Update active version
        self.active_pipelines[pipeline_id] = treatment_version
        
        # Update stages
        treatment_pipeline = self.pipelines[pipeline_id][treatment_version]
        treatment_pipeline.stage = PipelineStage.PRODUCTION
        treatment_pipeline.traffic_percentage = 100.0
        
        if control_version in self.pipelines[pipeline_id]:
            control_pipeline = self.pipelines[pipeline_id][control_version]
            control_pipeline.stage = PipelineStage.DEPRECATED
            control_pipeline.traffic_percentage = 0.0
            
        ab_test['completed_time'] = time.time()
        
        logger.info(f"Finalized canary {ab_test_id}: {treatment_version} now in production")
        
    async def _promote_blue_green(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Instant blue-green deployment"""
        pipeline_id = pipeline.pipeline_id
        version = str(pipeline.version)
        old_version = self.active_pipelines.get(pipeline_id)
        
        # Switch active version
        self.active_pipelines[pipeline_id] = version
        
        # Update stages
        pipeline.stage = PipelineStage.PRODUCTION
        pipeline.traffic_percentage = 100.0
        pipeline.deployed_at = time.time()
        
        if old_version and old_version in self.pipelines[pipeline_id]:
            old_pipeline = self.pipelines[pipeline_id][old_version]
            old_pipeline.stage = PipelineStage.DEPRECATED
            old_pipeline.traffic_percentage = 0.0
            
        logger.info(f"Blue-green deployment: {pipeline_id} {old_version} -> {version}")
        
        return {
            'deployment_type': 'blue_green',
            'old_version': old_version,
            'new_version': version,
            'cutover_time': time.time()
        }
        
    async def execute_pipeline(self,
                             pipeline_id: str,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pipeline with A/B testing support
        """
        # Select version based on traffic split
        version = await self._select_pipeline_version(pipeline_id)
        
        if not version:
            raise ValueError(f"No active version for pipeline {pipeline_id}")
            
        pipeline = self.pipelines[pipeline_id][version]
        
        # Execute pipeline
        start_time = time.perf_counter()
        result = await self._execute_pipeline_phases(pipeline, context)
        duration = (time.perf_counter() - start_time) * 1000
        
        # Update metrics
        await self._update_pipeline_metrics(pipeline, result, duration)
        
        # Track execution
        self.execution_history.append({
            'pipeline_id': pipeline_id,
            'version': version,
            'timestamp': time.time(),
            'duration_ms': duration,
            'success': result.get('success', False)
        })
        
        return result
        
    async def _select_pipeline_version(self, pipeline_id: str) -> Optional[str]:
        """Select pipeline version based on A/B test traffic split"""
        active_version = self.active_pipelines.get(pipeline_id)
        
        if not active_version:
            return None
            
        # Check for active A/B tests
        for ab_test in self.ab_tests.values():
            if (ab_test['pipeline_id'] == pipeline_id and 
                'completed_time' not in ab_test):
                
                # Traffic split decision
                if np.random.rand() * 100 < ab_test['traffic_split']:
                    return ab_test['treatment_version']
                else:
                    return ab_test['control_version']
                    
        return active_version
        
    async def _execute_pipeline_phases(self,
                                     pipeline: Pipeline,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline phases in order"""
        result = {
            'pipeline_id': pipeline.pipeline_id,
            'version': str(pipeline.version),
            'phases': {}
        }
        
        current_context = context.copy()
        
        for phase in pipeline.phases:
            try:
                # Get phase configuration
                phase_config = pipeline.phase_configs.get(phase, {})
                
                # Execute phase
                phase_result = await self._execute_phase(
                    phase,
                    current_context,
                    phase_config
                )
                
                result['phases'][phase] = phase_result
                
                # Update context for next phase
                current_context.update(phase_result.get('output', {}))
                
                # Check conditional routing
                if phase in pipeline.conditional_routes:
                    next_phase = pipeline.conditional_routes[phase](phase_result)
                    if next_phase == 'end':
                        break
                        
            except Exception as e:
                logger.error(f"Pipeline phase {phase} failed: {e}")
                result['phases'][phase] = {'error': str(e)}
                
                # Use fallback route
                if phase in pipeline.fallback_routes:
                    fallback = pipeline.fallback_routes[phase]
                    logger.info(f"Using fallback route: {phase} -> {fallback}")
                else:
                    result['success'] = False
                    return result
                    
        result['success'] = True
        return result
        
    async def _execute_phase(self,
                           phase: str,
                           context: Dict[str, Any],
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline phase"""
        # This would integrate with actual phase implementations
        # For now, return mock result
        return {
            'phase': phase,
            'status': 'success',
            'output': {f'{phase}_result': True},
            'metrics': {
                'duration_ms': np.random.randint(10, 100),
                'confidence': np.random.rand()
            }
        }
        
    async def _update_pipeline_metrics(self,
                                     pipeline: Pipeline,
                                     result: Dict[str, Any],
                                     duration: float):
        """Update pipeline execution metrics"""
        pipeline.execution_count += 1
        
        if result.get('success'):
            pipeline.success_count += 1
            
        # Update average latency
        alpha = 0.1  # Exponential moving average
        pipeline.avg_latency_ms = (
            alpha * duration + (1 - alpha) * pipeline.avg_latency_ms
        )
        
        # Update A/B test metrics if active
        if pipeline.ab_test_id and pipeline.ab_test_id in self.ab_tests:
            ab_test = self.ab_tests[pipeline.ab_test_id]
            
            version_type = ('treatment' if str(pipeline.version) == ab_test['treatment_version'] 
                           else 'control')
            
            ab_test['metrics'][f'{version_type}_success_rate'].append(
                1.0 if result.get('success') else 0.0
            )
            ab_test['metrics'][f'{version_type}_latency'].append(duration)
            
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get current status of a pipeline"""
        if pipeline_id not in self.pipelines:
            return {'error': 'Pipeline not found'}
            
        active_version = self.active_pipelines.get(pipeline_id)
        versions = self.pipelines[pipeline_id]
        
        # Find active A/B test
        active_ab_test = None
        for ab_id, ab_test in self.ab_tests.items():
            if ab_test['pipeline_id'] == pipeline_id and 'completed_time' not in ab_test:
                active_ab_test = ab_test
                break
                
        return {
            'pipeline_id': pipeline_id,
            'active_version': active_version,
            'versions': {
                v: {
                    'stage': p.stage.value,
                    'traffic_percentage': p.traffic_percentage,
                    'execution_count': p.execution_count,
                    'success_rate': p.success_count / max(p.execution_count, 1),
                    'avg_latency_ms': p.avg_latency_ms
                }
                for v, p in versions.items()
            },
            'active_ab_test': {
                'id': ab_id,
                'traffic_split': active_ab_test['traffic_split'],
                'duration_hours': (time.time() - active_ab_test['start_time']) / 3600
            } if active_ab_test else None
        }


from collections import deque
import numpy as np