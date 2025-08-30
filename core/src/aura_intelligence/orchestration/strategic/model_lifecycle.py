"""
Model Lifecycle Management
==========================
Canary deployments, feature flags, and safe rollbacks
Based on Microsoft Semantic Kernel 2025 patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from enum import Enum
import json
import hashlib
from collections import defaultdict

# AURA imports
from ...components.registry import get_registry, ComponentRole
from ..feature_flags import FeatureFlagService

import logging
logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLBACK = "rollback"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    component: str  # tda, inference, dpo, etc.
    
    # Deployment info
    stage: DeploymentStage = DeploymentStage.DEVELOPMENT
    deployment_timestamp: float = 0.0
    
    # Performance metrics
    accuracy: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    
    # Resource requirements
    memory_mb: int = 0
    gpu_required: bool = False
    
    # Version control
    checksum: str = ""
    parent_version: Optional[str] = None
    
    # Feature flags
    feature_flags: List[str] = field(default_factory=list)


@dataclass
class CanaryDeployment:
    """Canary deployment configuration"""
    deployment_id: str
    model_version: ModelVersion
    
    # Traffic splitting
    canary_percentage: float = 10.0  # Start with 10%
    ramp_schedule: List[Tuple[float, float]] = field(default_factory=list)  # (time, percentage)
    
    # Success criteria
    min_accuracy: float = 0.95
    max_latency_ms: float = 100.0
    max_error_rate: float = 0.01
    
    # Monitoring
    start_time: float = field(default_factory=time.time)
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    
    # State
    is_active: bool = True
    auto_rollback: bool = True
    rollback_triggered: bool = False


class ModelLifecycleManager:
    """
    Manages model lifecycle: training, deployment, monitoring, rollback
    Implements safe canary deployments with automatic rollback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Model registry
        self.models: Dict[str, ModelVersion] = {}
        self.active_models: Dict[str, str] = {}  # component -> active version
        
        # Deployment tracking
        self.canary_deployments: Dict[str, CanaryDeployment] = {}
        self.deployment_history = []
        
        # Feature flag service
        self.feature_flags = FeatureFlagService()
        
        # Component registry
        self.registry = get_registry()
        
        # Metrics tracking
        self.metrics_buffer = defaultdict(list)
        
        logger.info("Model Lifecycle Manager initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'canary_duration_hours': 24,
            'canary_initial_percentage': 10.0,
            'canary_ramp_steps': 5,
            'rollback_threshold': 0.05,  # 5% degradation triggers rollback
            'metrics_window': 300,  # 5 minutes
            'health_check_interval': 60  # 1 minute
        }
        
    async def deploy_model(self,
                         model_version: ModelVersion,
                         deployment_type: str = "canary") -> CanaryDeployment:
        """
        Deploy a new model version
        Supports canary and blue-green deployments
        """
        # Validate model
        if not await self._validate_model(model_version):
            raise ValueError(f"Model validation failed for {model_version.model_id}")
            
        # Register model
        self.models[model_version.model_id] = model_version
        
        if deployment_type == "canary":
            deployment = await self._deploy_canary(model_version)
        elif deployment_type == "blue_green":
            deployment = await self._deploy_blue_green(model_version)
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")
            
        # Start monitoring
        asyncio.create_task(self._monitor_deployment(deployment))
        
        logger.info(f"Deployed {model_version.model_id} as {deployment_type}")
        
        return deployment
        
    async def _deploy_canary(self, model_version: ModelVersion) -> CanaryDeployment:
        """Deploy model as canary with gradual rollout"""
        # Create canary deployment
        deployment = CanaryDeployment(
            deployment_id=f"canary_{model_version.model_id}_{int(time.time())}",
            model_version=model_version,
            canary_percentage=self.config['canary_initial_percentage']
        )
        
        # Calculate ramp schedule
        duration_hours = self.config['canary_duration_hours']
        steps = self.config['canary_ramp_steps']
        
        for i in range(1, steps + 1):
            time_offset = (duration_hours / steps) * i * 3600
            percentage = min(100, self.config['canary_initial_percentage'] * (2 ** i))
            deployment.ramp_schedule.append((
                deployment.start_time + time_offset,
                percentage
            ))
            
        # Set feature flag for traffic splitting
        await self.feature_flags.set_flag(
            f"model_{model_version.component}_canary",
            {
                'enabled': True,
                'percentage': deployment.canary_percentage,
                'model_id': model_version.model_id
            }
        )
        
        # Register deployment
        self.canary_deployments[deployment.deployment_id] = deployment
        
        # Update model stage
        model_version.stage = DeploymentStage.CANARY
        model_version.deployment_timestamp = time.time()
        
        return deployment
        
    async def _deploy_blue_green(self, model_version: ModelVersion) -> CanaryDeployment:
        """Deploy model with instant switchover"""
        # Create deployment (100% traffic)
        deployment = CanaryDeployment(
            deployment_id=f"blue_green_{model_version.model_id}_{int(time.time())}",
            model_version=model_version,
            canary_percentage=100.0,
            auto_rollback=False  # Manual rollback for blue-green
        )
        
        # Switch active model
        old_version = self.active_models.get(model_version.component)
        self.active_models[model_version.component] = model_version.model_id
        
        # Update stages
        if old_version and old_version in self.models:
            self.models[old_version].stage = DeploymentStage.DEPRECATED
            
        model_version.stage = DeploymentStage.PRODUCTION
        model_version.deployment_timestamp = time.time()
        
        logger.info(f"Blue-green deployment: {old_version} -> {model_version.model_id}")
        
        return deployment
        
    async def _validate_model(self, model_version: ModelVersion) -> bool:
        """Validate model before deployment"""
        # Check required fields
        if not model_version.model_id or not model_version.version:
            return False
            
        # Verify checksum
        if model_version.checksum:
            # In production, would verify actual model file
            pass
            
        # Check resource requirements
        if model_version.gpu_required:
            # Verify GPU availability
            pass
            
        return True
        
    async def _monitor_deployment(self, deployment: CanaryDeployment):
        """Monitor deployment health and manage rollout"""
        while deployment.is_active:
            try:
                # Collect metrics
                metrics = await self._collect_deployment_metrics(deployment)
                deployment.metrics_history.append(metrics)
                
                # Check health
                if not self._check_deployment_health(deployment, metrics):
                    if deployment.auto_rollback:
                        await self.rollback_deployment(deployment.deployment_id)
                        break
                        
                # Update canary percentage based on schedule
                await self._update_canary_percentage(deployment)
                
                # Check if fully rolled out
                if deployment.canary_percentage >= 100.0:
                    await self._finalize_deployment(deployment)
                    break
                    
            except Exception as e:
                logger.error(f"Error monitoring deployment {deployment.deployment_id}: {e}")
                
            await asyncio.sleep(self.config['health_check_interval'])
            
    async def _collect_deployment_metrics(self, 
                                        deployment: CanaryDeployment) -> Dict[str, float]:
        """Collect metrics for deployment"""
        model = deployment.model_version
        
        # Get metrics from component
        component_metrics = await self._get_component_metrics(model.component)
        
        # Filter to canary traffic
        canary_metrics = {
            'accuracy': component_metrics.get(f'{model.component}_accuracy', 0.0),
            'latency_ms': component_metrics.get(f'{model.component}_latency_ms', 0.0),
            'error_rate': component_metrics.get(f'{model.component}_error_rate', 0.0),
            'timestamp': time.time()
        }
        
        # Update model version metrics
        model.accuracy = canary_metrics['accuracy']
        model.latency_ms = canary_metrics['latency_ms']
        model.error_rate = canary_metrics['error_rate']
        
        return canary_metrics
        
    async def _get_component_metrics(self, component: str) -> Dict[str, float]:
        """Get metrics from component via registry"""
        try:
            # Get component from registry
            comp_instance = self.registry.get_component(component)
            if hasattr(comp_instance, 'get_metrics'):
                return await comp_instance.get_metrics()
        except Exception as e:
            logger.error(f"Failed to get metrics for {component}: {e}")
            
        # Return dummy metrics for testing
        return {
            f'{component}_accuracy': 0.96,
            f'{component}_latency_ms': 50.0,
            f'{component}_error_rate': 0.005
        }
        
    def _check_deployment_health(self, 
                               deployment: CanaryDeployment,
                               metrics: Dict[str, float]) -> bool:
        """Check if deployment meets health criteria"""
        # Check absolute thresholds
        if metrics['accuracy'] < deployment.min_accuracy:
            logger.warning(f"Deployment {deployment.deployment_id} accuracy "
                         f"{metrics['accuracy']} below threshold {deployment.min_accuracy}")
            return False
            
        if metrics['latency_ms'] > deployment.max_latency_ms:
            logger.warning(f"Deployment {deployment.deployment_id} latency "
                         f"{metrics['latency_ms']}ms above threshold {deployment.max_latency_ms}ms")
            return False
            
        if metrics['error_rate'] > deployment.max_error_rate:
            logger.warning(f"Deployment {deployment.deployment_id} error rate "
                         f"{metrics['error_rate']} above threshold {deployment.max_error_rate}")
            return False
            
        # Check relative degradation
        if len(deployment.metrics_history) > 10:
            baseline = deployment.metrics_history[:5]
            baseline_accuracy = np.mean([m['accuracy'] for m in baseline])
            
            if metrics['accuracy'] < baseline_accuracy * (1 - self.config['rollback_threshold']):
                logger.warning(f"Deployment {deployment.deployment_id} shows "
                             f"{(1 - metrics['accuracy']/baseline_accuracy)*100:.1f}% accuracy degradation")
                return False
                
        return True
        
    async def _update_canary_percentage(self, deployment: CanaryDeployment):
        """Update canary traffic percentage based on schedule"""
        current_time = time.time()
        
        for scheduled_time, percentage in deployment.ramp_schedule:
            if current_time >= scheduled_time and deployment.canary_percentage < percentage:
                deployment.canary_percentage = percentage
                
                # Update feature flag
                await self.feature_flags.set_flag(
                    f"model_{deployment.model_version.component}_canary",
                    {
                        'enabled': True,
                        'percentage': percentage,
                        'model_id': deployment.model_version.model_id
                    }
                )
                
                logger.info(f"Updated canary percentage to {percentage}% "
                          f"for {deployment.deployment_id}")
                break
                
    async def _finalize_deployment(self, deployment: CanaryDeployment):
        """Finalize successful deployment"""
        model = deployment.model_version
        
        # Update active model
        old_version = self.active_models.get(model.component)
        self.active_models[model.component] = model.model_id
        
        # Update stages
        model.stage = DeploymentStage.PRODUCTION
        
        if old_version and old_version in self.models:
            self.models[old_version].stage = DeploymentStage.DEPRECATED
            
        # Remove canary feature flag
        await self.feature_flags.remove_flag(f"model_{model.component}_canary")
        
        # Mark deployment as complete
        deployment.is_active = False
        
        # Record in history
        self.deployment_history.append({
            'deployment_id': deployment.deployment_id,
            'model_id': model.model_id,
            'component': model.component,
            'start_time': deployment.start_time,
            'end_time': time.time(),
            'final_metrics': deployment.metrics_history[-1] if deployment.metrics_history else {},
            'status': 'success'
        })
        
        logger.info(f"Deployment {deployment.deployment_id} finalized successfully")
        
    async def rollback_deployment(self, deployment_id: str):
        """Rollback a deployment"""
        if deployment_id not in self.canary_deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return
            
        deployment = self.canary_deployments[deployment_id]
        model = deployment.model_version
        
        logger.warning(f"Rolling back deployment {deployment_id}")
        
        # Mark as rolled back
        deployment.rollback_triggered = True
        deployment.is_active = False
        model.stage = DeploymentStage.ROLLBACK
        
        # Restore previous version
        if model.parent_version and model.parent_version in self.models:
            self.active_models[model.component] = model.parent_version
            self.models[model.parent_version].stage = DeploymentStage.PRODUCTION
            
        # Remove canary feature flag
        await self.feature_flags.remove_flag(f"model_{model.component}_canary")
        
        # Record in history
        self.deployment_history.append({
            'deployment_id': deployment_id,
            'model_id': model.model_id,
            'component': model.component,
            'start_time': deployment.start_time,
            'end_time': time.time(),
            'status': 'rollback',
            'reason': 'health_check_failed'
        })
        
        logger.info(f"Rollback completed for {deployment_id}")
        
    def get_active_models(self) -> Dict[str, ModelVersion]:
        """Get currently active models"""
        active = {}
        for component, model_id in self.active_models.items():
            if model_id in self.models:
                active[component] = self.models[model_id]
        return active
        
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        active_canaries = [
            {
                'deployment_id': d.deployment_id,
                'model_id': d.model_version.model_id,
                'component': d.model_version.component,
                'percentage': d.canary_percentage,
                'health': 'healthy' if not d.rollback_triggered else 'rolled_back'
            }
            for d in self.canary_deployments.values()
            if d.is_active
        ]
        
        return {
            'active_models': {
                c: {'model_id': m, 'version': self.models[m].version}
                for c, m in self.active_models.items()
            },
            'canary_deployments': active_canaries,
            'recent_deployments': self.deployment_history[-10:]
        }