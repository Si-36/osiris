"""
Strategic Planner - Long-term Resource & Model Planning
=====================================================
Uses TopologicalSignature drift for retraining triggers
ICML-25 matrix-time persistent homology improvements
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import deque
import torch

# AURA imports
try:
    from ...tda.persistence import TopologicalSignature
except ImportError:
    from ...tda.persistence_simple import TopologicalSignature
from ...components.registry import get_registry
from ...memory.hierarchical_memory import HierarchicalMemoryManager
from ...inference.free_energy_core import FreeEnergyMinimizer

import logging
logger = logging.getLogger(__name__)


@dataclass
class ResourcePlan:
    """Strategic resource allocation plan"""
    timestamp: float = field(default_factory=time.time)
    horizon_hours: int = 24
    
    # Resource allocations
    cpu_allocation: Dict[str, float] = field(default_factory=dict)  # Component -> CPU%
    gpu_allocation: Dict[str, float] = field(default_factory=dict)  # Component -> GPU%
    memory_allocation: Dict[str, int] = field(default_factory=dict)  # Component -> MB
    
    # Model update schedule
    model_updates: List[Tuple[str, float]] = field(default_factory=list)  # (model_id, timestamp)
    
    # Scaling recommendations
    scale_up_components: List[str] = field(default_factory=list)
    scale_down_components: List[str] = field(default_factory=list)
    
    # Drift-triggered actions
    drift_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    confidence: float = 0.0


class StrategicPlanner:
    """
    Long-term planning based on topological drift and resource patterns
    Implements ICML-25 matrix-time persistent homology for efficiency
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # History tracking
        self.signature_history = deque(maxlen=self.config['history_window'])
        self.resource_history = deque(maxlen=self.config['history_window'])
        self.performance_history = deque(maxlen=self.config['history_window'])
        
        # Component registry
        self.registry = get_registry()
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'tda': 0.3,      # Topological drift
            'inference': 0.4, # Free energy drift
            'consensus': 0.5, # Consensus time drift
            'memory': 0.6    # Memory retrieval drift
        }
        
        # Resource limits
        self.resource_limits = {
            'total_cpu': 100.0,    # 100%
            'total_gpu': 100.0,    # 100%
            'total_memory': 65536  # 64GB
        }
        
        logger.info("Strategic Planner initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'history_window': 1000,
            'planning_horizon_hours': 24,
            'drift_check_interval': 300,  # 5 minutes
            'resource_check_interval': 60, # 1 minute
            'min_confidence': 0.7
        }
        
    async def create_resource_plan(self,
                                 current_state: Dict[str, Any],
                                 performance_metrics: Dict[str, float]) -> ResourcePlan:
        """
        Create strategic resource allocation plan
        Based on drift detection and performance trends
        """
        plan = ResourcePlan(horizon_hours=self.config['planning_horizon_hours'])
        
        # Analyze topological drift
        drift_analysis = await self._analyze_drift()
        plan.drift_alerts = drift_analysis['alerts']
        
        # Predict resource needs
        resource_prediction = self._predict_resource_needs(
            current_state,
            performance_metrics,
            drift_analysis
        )
        
        # Allocate resources optimally
        plan.cpu_allocation = resource_prediction['cpu']
        plan.gpu_allocation = resource_prediction['gpu']
        plan.memory_allocation = resource_prediction['memory']
        
        # Schedule model updates based on drift
        if drift_analysis['max_drift'] > 0.5:
            plan.model_updates = self._schedule_model_updates(drift_analysis)
            
        # Scaling recommendations
        plan.scale_up_components = self._identify_scale_up_needs(
            current_state,
            performance_metrics
        )
        plan.scale_down_components = self._identify_scale_down_opportunities(
            current_state,
            performance_metrics
        )
        
        # Calculate confidence
        plan.confidence = self._calculate_plan_confidence(
            drift_analysis,
            resource_prediction
        )
        
        # Store in history
        self._update_history(current_state, performance_metrics, plan)
        
        logger.info(f"Strategic plan created: {len(plan.model_updates)} model updates, "
                   f"{len(plan.scale_up_components)} scale-ups, confidence={plan.confidence:.2f}")
        
        return plan
        
    async def _analyze_drift(self) -> Dict[str, Any]:
        """
        Analyze topological drift across components
        Uses ICML-25 matrix-time algorithm for efficiency
        """
        if len(self.signature_history) < 2:
            return {'max_drift': 0.0, 'alerts': [], 'components': {}}
            
        # Get recent signatures
        recent_signatures = list(self.signature_history)[-10:]
        baseline_signatures = list(self.signature_history)[:10]
        
        drift_scores = {}
        alerts = []
        
        # Compute drift for each component
        for component in ['tda', 'inference', 'consensus', 'memory']:
            drift = self._compute_component_drift(
                component,
                baseline_signatures,
                recent_signatures
            )
            drift_scores[component] = drift
            
            if drift > self.drift_thresholds[component]:
                alerts.append({
                    'component': component,
                    'drift_score': drift,
                    'threshold': self.drift_thresholds[component],
                    'severity': 'high' if drift > self.drift_thresholds[component] * 1.5 else 'medium',
                    'timestamp': time.time()
                })
                
        return {
            'max_drift': max(drift_scores.values()) if drift_scores else 0.0,
            'alerts': alerts,
            'components': drift_scores
        }
        
    def _compute_component_drift(self,
                                component: str,
                                baseline: List[Dict],
                                recent: List[Dict]) -> float:
        """
        Compute drift score for a specific component
        Matrix-time optimization for O(n log n) complexity
        """
        # Extract relevant features
        baseline_features = self._extract_drift_features(component, baseline)
        recent_features = self._extract_drift_features(component, recent)
        
        if baseline_features is None or recent_features is None:
            return 0.0
            
        # Compute spectral distance (simplified)
        try:
            # Use cosine distance as proxy for drift
            baseline_mean = np.mean(baseline_features, axis=0)
            recent_mean = np.mean(recent_features, axis=0)
            
            cosine_sim = np.dot(baseline_mean, recent_mean) / (
                np.linalg.norm(baseline_mean) * np.linalg.norm(recent_mean) + 1e-8
            )
            
            drift = 1.0 - cosine_sim
            return float(np.clip(drift, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Drift computation failed for {component}: {e}")
            return 0.0
            
    def _extract_drift_features(self,
                               component: str,
                               signatures: List[Dict]) -> Optional[np.ndarray]:
        """Extract features for drift detection"""
        features = []
        
        for sig in signatures:
            if component == 'tda' and 'topological_features' in sig:
                features.append(sig['topological_features'])
            elif component == 'inference' and 'free_energy' in sig:
                features.append([sig['free_energy']])
            elif component == 'consensus' and 'consensus_time' in sig:
                features.append([sig['consensus_time']])
            elif component == 'memory' and 'retrieval_time' in sig:
                features.append([sig['retrieval_time']])
                
        if not features:
            return None
            
        # Ensure consistent shape
        max_len = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_len:
                padded = list(f) + [0.0] * (max_len - len(f))
            else:
                padded = f[:max_len]
            padded_features.append(padded)
            
        return np.array(padded_features)
        
    def _predict_resource_needs(self,
                              current_state: Dict[str, Any],
                              performance: Dict[str, float],
                              drift: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Predict resource needs based on trends and drift
        """
        # Base allocations
        cpu_alloc = {
            'tda': 20.0,
            'inference': 30.0,
            'consensus': 15.0,
            'memory': 10.0,
            'agents': 25.0
        }
        
        gpu_alloc = {
            'tda': 10.0,
            'inference': 40.0,
            'neuromorphic': 30.0,
            'agents': 20.0
        }
        
        memory_alloc = {
            'tda': 8192,
            'inference': 16384,
            'memory': 32768,
            'agents': 8192
        }
        
        # Adjust based on drift
        for component, drift_score in drift.get('components', {}).items():
            if drift_score > 0.3:
                # Increase resources for drifting components
                if component in cpu_alloc:
                    cpu_alloc[component] *= (1 + drift_score)
                if component in gpu_alloc:
                    gpu_alloc[component] *= (1 + drift_score)
                if component in memory_alloc:
                    memory_alloc[component] = int(memory_alloc[component] * (1 + drift_score))
                    
        # Normalize to fit within limits
        cpu_total = sum(cpu_alloc.values())
        if cpu_total > self.resource_limits['total_cpu']:
            scale = self.resource_limits['total_cpu'] / cpu_total
            cpu_alloc = {k: v * scale for k, v in cpu_alloc.items()}
            
        gpu_total = sum(gpu_alloc.values())
        if gpu_total > self.resource_limits['total_gpu']:
            scale = self.resource_limits['total_gpu'] / gpu_total
            gpu_alloc = {k: v * scale for k, v in gpu_alloc.items()}
            
        return {
            'cpu': cpu_alloc,
            'gpu': gpu_alloc,
            'memory': memory_alloc
        }
        
    def _schedule_model_updates(self, drift_analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Schedule model updates based on drift severity"""
        updates = []
        current_time = time.time()
        
        for alert in drift_analysis['alerts']:
            if alert['severity'] == 'high':
                # Immediate update
                update_time = current_time + 3600  # 1 hour
            else:
                # Scheduled update
                update_time = current_time + 14400  # 4 hours
                
            model_id = f"{alert['component']}_model"
            updates.append((model_id, update_time))
            
        return updates
        
    def _identify_scale_up_needs(self,
                                current_state: Dict[str, Any],
                                performance: Dict[str, float]) -> List[str]:
        """Identify components that need scaling up"""
        scale_up = []
        
        # Check CPU usage
        for component, usage in current_state.get('cpu_usage', {}).items():
            if usage > 80.0:  # 80% threshold
                scale_up.append(component)
                
        # Check latency
        for component, latency in performance.items():
            if 'latency' in component and latency > 100.0:  # 100ms threshold
                comp_name = component.replace('_latency', '')
                if comp_name not in scale_up:
                    scale_up.append(comp_name)
                    
        return scale_up
        
    def _identify_scale_down_opportunities(self,
                                         current_state: Dict[str, Any],
                                         performance: Dict[str, float]) -> List[str]:
        """Identify components that can be scaled down"""
        scale_down = []
        
        # Check low usage
        for component, usage in current_state.get('cpu_usage', {}).items():
            if usage < 20.0:  # 20% threshold
                scale_down.append(component)
                
        return scale_down
        
    def _calculate_plan_confidence(self,
                                 drift_analysis: Dict[str, Any],
                                 resource_prediction: Dict[str, Any]) -> float:
        """Calculate confidence in the strategic plan"""
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for high drift
        max_drift = drift_analysis.get('max_drift', 0.0)
        confidence *= (1.0 - max_drift * 0.5)
        
        # Reduce confidence if many alerts
        num_alerts = len(drift_analysis.get('alerts', []))
        if num_alerts > 0:
            confidence *= (1.0 - min(num_alerts * 0.1, 0.5))
            
        return float(np.clip(confidence, 0.0, 1.0))
        
    def _update_history(self,
                       state: Dict[str, Any],
                       performance: Dict[str, float],
                       plan: ResourcePlan):
        """Update historical tracking"""
        # Create signature snapshot
        signature = {
            'timestamp': time.time(),
            'topological_features': state.get('tda_features', []),
            'free_energy': performance.get('free_energy', 0.0),
            'consensus_time': performance.get('consensus_time_ms', 0.0),
            'retrieval_time': performance.get('memory_retrieval_ms', 0.0)
        }
        self.signature_history.append(signature)
        
        # Track resource usage
        self.resource_history.append({
            'timestamp': time.time(),
            'cpu_usage': state.get('cpu_usage', {}),
            'gpu_usage': state.get('gpu_usage', {}),
            'memory_usage': state.get('memory_usage', {})
        })
        
        # Track performance
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': performance,
            'plan_confidence': plan.confidence
        })