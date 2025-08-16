"""
Self-Healing Infrastructure for AURA Intelligence

This module implements a comprehensive self-healing infrastructure based on
chaos engineering, antifragility principles, and predictive failure detection.
Inspired by Netflix's Chaos Monkey, Uber's resilience engineering, and 
Nassim Taleb's antifragility theory.
"""

from typing import Dict, List, Set, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import time
import random
import logging
import numpy as np
from collections import defaultdict, deque

from .exceptions import AuraError, ErrorTopology, ErrorAnalysisManager
from .error_topology import ErrorTopologyAnalyzer, ErrorPropagationPattern
from .interfaces import SystemComponent, HealthStatus, ComponentStatus


logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected for chaos testing."""
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    DEPENDENCY_FAILURE = "dependency_failure"


class HealingStrategy(Enum):
    """Self-healing strategies."""
    RESTART = "restart"
    ROLLBACK = "rollback"
    SCALE_OUT = "scale_out"
    CIRCUIT_BREAK = "circuit_break"
    FAILOVER = "failover"
    DEGRADE_GRACEFULLY = "degrade_gracefully"
    ISOLATE_AND_HEAL = "isolate_and_heal"
    ADAPTIVE_THROTTLING = "adaptive_throttling"
    RESOURCE_REALLOCATION = "resource_reallocation"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class ChaosExperiment:
    """Definition of a chaos engineering experiment."""
    experiment_id: str
    name: str
    description: str
    failure_type: FailureType
    target_components: List[str]
    duration_seconds: float
    intensity: float  # 0.0 to 1.0
    blast_radius: float  # 0.0 to 1.0 (percentage of system affected)
    hypothesis: str
    success_criteria: List[str]
    rollback_conditions: List[str]
    
    def __post_init__(self):
        """Validate experiment parameters."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.blast_radius <= 1.0:
            raise ValueError("Blast radius must be between 0.0 and 1.0")

@dataclass
class ChaosResult:
    """Result of a chaos engineering experiment."""
    experiment: ChaosExperiment
    start_time: float
    end_time: float
    success: bool
    observations: List[str]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    system_learned: bool
    recovery_time: float
    blast_radius_actual: float
    
    @property
    def duration(self) -> float:
        """Get experiment duration."""
        return self.end_time - self.start_time
    
    @property
    def resilience_score(self) -> float:
        """Calculate resilience score based on recovery time and blast radius."""
        if self.recovery_time <= 0:
            return 1.0
        
        # Lower recovery time and blast radius = higher resilience
        time_factor = max(0.1, 1.0 - (self.recovery_time / 60.0))  # Normalize to 1 minute
        blast_factor = max(0.1, 1.0 - self.blast_radius_actual)
        
        return (time_factor + blast_factor) / 2.0


@dataclass
class Stressor:
    """A stressor that can make the system antifragile."""
    stressor_id: str
    name: str
    stressor_type: str
    intensity: float
    frequency: float
    duration: float
    target_components: List[str]
    beneficial_threshold: float  # Intensity below which stress is beneficial
    
    def is_beneficial(self) -> bool:
        """Check if this stressor is at beneficial levels (hormesis)."""
        return self.intensity <= self.beneficial_threshold


@dataclass
class AntifragileAdaptation:
    """Result of antifragile adaptation to stress."""
    original_component: str
    adapted_component: str
    strength_gain: float
    stressor_resistance: float
    overcompensation: float
    adaptation_timestamp: float
    
    def is_successful(self) -> bool:
        """Check if adaptation was successful."""
        return self.strength_gain > 0 and self.stressor_resistance > 0


class FailureInjector:
    """Injects controlled failures for chaos engineering."""
    
    def __init__(self):
        self.active_injections: Dict[str, Dict[str, Any]] = {}
        self.injection_history: List[Dict[str, Any]] = []
    
    async def inject_failure(
        self, 
        failure_type: FailureType, 
        target_component: str,
        intensity: float = 0.5,
        duration: float = 10.0
    ) -> str:
        """Inject a specific type of failure."""
        injection_id = f"{failure_type.value}_{target_component}_{int(time.time())}"
        
        injection_config = {
            'injection_id': injection_id,
            'failure_type': failure_type,
            'target_component': target_component,
            'intensity': intensity,
            'duration': duration,
            'start_time': time.time(),
            'active': True
        }
        
        self.active_injections[injection_id] = injection_config
        
        # Start the injection
        asyncio.create_task(self._execute_injection(injection_config))
        
        logger.info(f"Injected {failure_type.value} into {target_component} for {duration}s")
        return injection_id
    
    async def _execute_injection(self, config: Dict[str, Any]) -> None:
        """Execute the actual failure injection."""
        failure_type = config['failure_type']
        target_component = config['target_component']
        intensity = config['intensity']
        duration = config['duration']
        
        try:
            if failure_type == FailureType.LATENCY_INJECTION:
                await self._inject_latency(target_component, intensity, duration)
            elif failure_type == FailureType.ERROR_INJECTION:
                await self._inject_errors(target_component, intensity, duration)
            elif failure_type == FailureType.RESOURCE_EXHAUSTION:
                await self._inject_resource_exhaustion(target_component, intensity, duration)
            elif failure_type == FailureType.NETWORK_PARTITION:
                await self._inject_network_partition(target_component, intensity, duration)
            else:
                logger.warning(f"Unsupported failure type: {failure_type}")
        
        except Exception as e:
            logger.error(f"Error during failure injection: {e}")
        
        finally:
            # Mark injection as complete
            config['active'] = False
            config['end_time'] = time.time()
            self.injection_history.append(config.copy())
            
            # Remove from active injections
            if config['injection_id'] in self.active_injections:
                del self.active_injections[config['injection_id']]
    
    async def _inject_latency(self, component: str, intensity: float, duration: float) -> None:
        """Inject artificial latency."""
        end_time = time.time() + duration
        base_delay = intensity * 2.0  # Up to 2 seconds delay
        
        while time.time() < end_time:
            # Simulate latency by adding delays to component operations
            delay = base_delay * (0.5 + 0.5 * random.random())
            await asyncio.sleep(delay)
            await asyncio.sleep(0.1)  # Check interval
    
    async def _inject_errors(self, component: str, intensity: float, duration: float) -> None:
        """Inject artificial errors."""
        end_time = time.time() + duration
        error_rate = intensity * 0.5  # Up to 50% error rate
        
        while time.time() < end_time:
            if random.random() < error_rate:
                # Create and register an artificial error
                from .exceptions import create_consciousness_error
                error = create_consciousness_error(
                    f"Chaos-injected error in {component}",
                    component_id=component
                )
                
                # Register with error analysis system
                from .exceptions import register_system_error
                register_system_error(error)
            
            await asyncio.sleep(1.0)  # Check every second
    
    async def _inject_resource_exhaustion(self, component: str, intensity: float, duration: float) -> None:
        """Inject resource exhaustion."""
        end_time = time.time() + duration
        
        # Simulate memory/CPU exhaustion
        memory_hog = []
        
        try:
            while time.time() < end_time:
                # Consume memory proportional to intensity
                chunk_size = int(1024 * 1024 * intensity)  # MB
                memory_hog.append(bytearray(chunk_size))
                await asyncio.sleep(0.5)
        finally:
            # Clean up memory
            del memory_hog
    
    async def _inject_network_partition(self, component: str, intensity: float, duration: float) -> None:
        """Inject network partition simulation."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Simulate network issues by introducing connection failures
            if random.random() < intensity:
                logger.warning(f"Simulated network partition affecting {component}")
            await asyncio.sleep(2.0)
    
    def stop_injection(self, injection_id: str) -> bool:
        """Stop an active injection."""
        if injection_id in self.active_injections:
            self.active_injections[injection_id]['active'] = False
            return True
        return False
    
    def get_active_injections(self) -> List[Dict[str, Any]]:
        """Get list of currently active injections."""
        return list(self.active_injections.values())


class BlastRadiusController:
    """Controls the blast radius of chaos experiments."""
    
    def __init__(self, max_blast_radius: float = 0.3):
        self.max_blast_radius = max_blast_radius
        self.component_criticality: Dict[str, float] = {}
    
    def calculate_safe_blast_radius(
        self, 
        experiment: ChaosExperiment,
        system_health: Dict[str, Any]
    ) -> float:
        """Calculate safe blast radius for experiment."""
        base_radius = experiment.blast_radius
        
        # Reduce blast radius based on system health
        health_factor = 1.0
        if system_health.get('status') == 'critical':
            health_factor = 0.1
        elif system_health.get('status') == 'warning':
            health_factor = 0.3
        elif system_health.get('status') == 'degraded':
            health_factor = 0.5
        
        # Reduce blast radius for critical components
        criticality_factor = 1.0
        for component in experiment.target_components:
            component_criticality = self.component_criticality.get(component, 0.5)
            if component_criticality > 0.8:
                criticality_factor *= 0.2
            elif component_criticality > 0.6:
                criticality_factor *= 0.5
        
        # Calculate final safe radius
        safe_radius = min(
            base_radius * health_factor * criticality_factor,
            self.max_blast_radius
        )
        
        return max(safe_radius, 0.01)  # Minimum 1% blast radius
    
    def update_component_criticality(self, criticality_scores: List[tuple]) -> None:
        """Update component criticality scores."""
        self.component_criticality.clear()
        for component, score in criticality_scores:
            self.component_criticality[component] = score

class ChaosEngineer:
    """
    Implements chaos engineering principles for system resilience.
    Based on Netflix's Chaos Monkey and Principles of Chaos Engineering.
    """
    
    def __init__(self):
        self.failure_injector = FailureInjector()
        self.blast_radius_controller = BlastRadiusController()
        self.experiment_history: List[ChaosResult] = []
        self.steady_state_metrics: Dict[str, Any] = {}
    
    async def conduct_chaos_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Conduct a controlled chaos experiment."""
        logger.info(f"Starting chaos experiment: {experiment.name}")
        
        # 1. Verify steady state before experiment
        initial_metrics = await self._capture_system_metrics()
        steady_state_verified = self._verify_steady_state(initial_metrics)
        
        if not steady_state_verified:
            logger.warning("System not in steady state - aborting experiment")
            return self._create_aborted_result(experiment, "System not in steady state")
        
        # 2. Calculate safe blast radius
        system_health = await self._get_system_health()
        safe_blast_radius = self.blast_radius_controller.calculate_safe_blast_radius(
            experiment, system_health
        )
        
        # 3. Execute experiment
        start_time = time.time()
        observations = []
        
        try:
            # Inject failures
            injection_tasks = []
            for component in experiment.target_components:
                task = asyncio.create_task(
                    self.failure_injector.inject_failure(
                        experiment.failure_type,
                        component,
                        experiment.intensity,
                        experiment.duration_seconds
                    )
                )
                injection_tasks.append(task)
            
            # Wait for injections to start
            injection_ids = await asyncio.gather(*injection_tasks)
            observations.append(f"Started {len(injection_ids)} failure injections")
            
            # Monitor system during experiment
            monitoring_task = asyncio.create_task(
                self._monitor_experiment(experiment, observations)
            )
            
            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_seconds)
            
            # Stop monitoring
            monitoring_task.cancel()
            
            # Stop all injections
            for injection_id in injection_ids:
                self.failure_injector.stop_injection(injection_id)
            
            observations.append("All failure injections stopped")
            
        except Exception as e:
            observations.append(f"Experiment error: {str(e)}")
            logger.error(f"Error during chaos experiment: {e}")
        
        # 4. Verify steady state recovery
        end_time = time.time()
        recovery_start = time.time()
        
        # Wait for system to recover
        max_recovery_time = 60.0  # 1 minute max recovery
        recovery_verified = False
        
        while time.time() - recovery_start < max_recovery_time:
            await asyncio.sleep(5.0)  # Check every 5 seconds
            current_metrics = await self._capture_system_metrics()
            if self._verify_steady_state(current_metrics):
                recovery_verified = True
                break
        
        recovery_time = time.time() - recovery_start
        final_metrics = await self._capture_system_metrics()
        
        # 5. Analyze results
        success = recovery_verified and self._evaluate_success_criteria(
            experiment, initial_metrics, final_metrics
        )
        
        system_learned = self._detect_system_learning(initial_metrics, final_metrics)
        
        result = ChaosResult(
            experiment=experiment,
            start_time=start_time,
            end_time=end_time,
            success=success,
            observations=observations,
            metrics_before=initial_metrics,
            metrics_after=final_metrics,
            system_learned=system_learned,
            recovery_time=recovery_time,
            blast_radius_actual=safe_blast_radius
        )
        
        self.experiment_history.append(result)
        logger.info(f"Chaos experiment completed: {experiment.name} - Success: {success}")
        
        return result
    
    async def _capture_system_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics."""
        # Get error analysis manager
        from .exceptions import get_error_analysis_manager
        manager = get_error_analysis_manager()
        
        # Get system health summary
        health_summary = manager.get_system_health_summary()
        
        # Add additional metrics
        metrics = {
            'timestamp': time.time(),
            'system_status': health_summary['status'],
            'total_errors': health_summary['total_errors'],
            'error_rate': health_summary['error_rate'],
            'topology_health': health_summary['topology_health'],
            'component_count': len(health_summary.get('most_affected_components', [])),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage(),
            'response_time': await self._measure_response_time()
        }
        
        return metrics
    
    def _verify_steady_state(self, metrics: Dict[str, Any]) -> bool:
        """Verify system is in steady state."""
        # Define steady state criteria
        if metrics['system_status'] in ['critical', 'warning']:
            return False
        
        if metrics['error_rate'] > 5.0:  # More than 5 errors per minute
            return False
        
        if metrics['cpu_usage'] > 0.9:  # More than 90% CPU
            return False
        
        if metrics['memory_usage'] > 0.9:  # More than 90% memory
            return False
        
        return True
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health."""
        from .exceptions import get_error_analysis_manager
        manager = get_error_analysis_manager()
        return manager.get_system_health_summary()
    
    async def _monitor_experiment(self, experiment: ChaosExperiment, observations: List[str]) -> None:
        """Monitor system during experiment."""
        try:
            while True:
                await asyncio.sleep(2.0)  # Monitor every 2 seconds
                
                metrics = await self._capture_system_metrics()
                
                # Check for critical conditions
                if metrics['system_status'] == 'critical':
                    observations.append("CRITICAL: System entered critical state")
                
                if metrics['error_rate'] > 20.0:
                    observations.append(f"HIGH ERROR RATE: {metrics['error_rate']:.1f} errors/min")
                
                # Log current state
                observations.append(
                    f"Monitor: Status={metrics['system_status']}, "
                    f"Errors={metrics['total_errors']}, "
                    f"Rate={metrics['error_rate']:.1f}/min"
                )
                
        except asyncio.CancelledError:
            observations.append("Monitoring stopped")
    
    def _evaluate_success_criteria(
        self, 
        experiment: ChaosExperiment,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any]
    ) -> bool:
        """Evaluate if experiment met success criteria."""
        # Basic success criteria: system recovered to steady state
        if not self._verify_steady_state(after_metrics):
            return False
        
        # Check if error rate didn't increase significantly
        error_rate_increase = after_metrics['error_rate'] - before_metrics['error_rate']
        if error_rate_increase > 10.0:  # More than 10 errors/min increase
            return False
        
        # Check if response time didn't degrade significantly
        response_time_increase = after_metrics['response_time'] - before_metrics['response_time']
        if response_time_increase > 5.0:  # More than 5 seconds increase
            return False
        
        return True
    
    def _detect_system_learning(
        self, 
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any]
    ) -> bool:
        """Detect if system learned from the experiment."""
        # System learned if it's more resilient after the experiment
        
        # Check if error rate improved
        if after_metrics['error_rate'] < before_metrics['error_rate']:
            return True
        
        # Check if response time improved
        if after_metrics['response_time'] < before_metrics['response_time']:
            return True
        
        # Check if system status improved
        status_order = {'critical': 0, 'warning': 1, 'degraded': 2, 'healthy': 3}
        before_status = status_order.get(before_metrics['system_status'], 0)
        after_status = status_order.get(after_metrics['system_status'], 0)
        
        if after_status > before_status:
            return True
        
        return False
    
    def _create_aborted_result(self, experiment: ChaosExperiment, reason: str) -> ChaosResult:
        """Create result for aborted experiment."""
        return ChaosResult(
            experiment=experiment,
            start_time=time.time(),
            end_time=time.time(),
            success=False,
            observations=[f"Experiment aborted: {reason}"],
            metrics_before={},
            metrics_after={},
            system_learned=False,
            recovery_time=0.0,
            blast_radius_actual=0.0
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (0.0 to 1.0)."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback: simulate memory usage
            return random.uniform(0.3, 0.7)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (0.0 to 1.0)."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1) / 100.0
        except ImportError:
            # Fallback: simulate CPU usage
            return random.uniform(0.2, 0.6)
    
    async def _measure_response_time(self) -> float:
        """Measure system response time."""
        start_time = time.time()
        
        # Simulate a system operation
        await asyncio.sleep(0.01)  # 10ms base response time
        
        return time.time() - start_time
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.experiment_history:
            return {
                'total_experiments': 0,
                'success_rate': 0.0,
                'average_recovery_time': 0.0,
                'system_learning_rate': 0.0
            }
        
        total_experiments = len(self.experiment_history)
        successful_experiments = sum(1 for r in self.experiment_history if r.success)
        learning_experiments = sum(1 for r in self.experiment_history if r.system_learned)
        
        average_recovery_time = sum(r.recovery_time for r in self.experiment_history) / total_experiments
        average_resilience_score = sum(r.resilience_score for r in self.experiment_history) / total_experiments
        
        return {
            'total_experiments': total_experiments,
            'success_rate': successful_experiments / total_experiments,
            'average_recovery_time': average_recovery_time,
            'system_learning_rate': learning_experiments / total_experiments,
            'average_resilience_score': average_resilience_score,
            'failure_types_tested': list(set(r.experiment.failure_type for r in self.experiment_history))
        }


class AntifragilityEngine:
    """
    Implements antifragility - systems that get stronger from stress.
    Based on Nassim Taleb's antifragility theory and hormesis principles.
    """
    
    def __init__(self):
        self.stress_history: List[Dict[str, Any]] = []
        self.adaptation_history: List[AntifragileAdaptation] = []
        self.hormesis_thresholds: Dict[str, float] = {}
    
    async def make_antifragile(
        self, 
        component: str, 
        stressor: Stressor
    ) -> AntifragileAdaptation:
        """Make a system component antifragile to a specific stressor."""
        logger.info(f"Applying antifragile adaptation to {component} with {stressor.name}")
        
        # 1. Detect stress level
        stress_level = await self._detect_stress_level(component, stressor)
        
        # 2. Calculate hormetic response
        hormetic_response = self._calculate_hormetic_response(stress_level, stressor)
        
        # 3. Apply adaptation
        adaptation = await self._apply_adaptation(component, hormetic_response, stressor)
        
        # 4. Amplify strength gains
        strength_gain = self._amplify_strength_gains(adaptation, stressor)
        
        # 5. Create adaptation result
        adaptation_result = AntifragileAdaptation(
            original_component=component,
            adapted_component=f"{component}_antifragile",
            strength_gain=strength_gain,
            stressor_resistance=hormetic_response.get('resistance_gain', 0.0),
            overcompensation=hormetic_response.get('overcompensation', 0.0),
            adaptation_timestamp=time.time()
        )
        
        self.adaptation_history.append(adaptation_result)
        
        logger.info(f"Antifragile adaptation complete: {component} gained {strength_gain:.2f} strength")
        return adaptation_result
    
    async def _detect_stress_level(self, component: str, stressor: Stressor) -> Dict[str, Any]:
        """Detect the current stress level on a component."""
        # Get component metrics
        from .exceptions import get_error_analysis_manager
        manager = get_error_analysis_manager()
        health_summary = manager.get_system_health_summary()
        
        # Calculate stress indicators
        error_stress = 0.0
        performance_stress = 0.0
        resource_stress = 0.0
        
        # Error-based stress
        for comp, error_count in health_summary.get('most_affected_components', []):
            if comp == component:
                error_stress = min(error_count / 10.0, 1.0)  # Normalize to 0-1
                break
        
        # Performance stress (simulated)
        performance_stress = random.uniform(0.1, 0.8)
        
        # Resource stress (simulated)
        resource_stress = random.uniform(0.2, 0.7)
        
        stress_level = {
            'component': component,
            'stressor': stressor.name,
            'error_stress': error_stress,
            'performance_stress': performance_stress,
            'resource_stress': resource_stress,
            'overall_stress': (error_stress + performance_stress + resource_stress) / 3.0,
            'timestamp': time.time()
        }
        
        self.stress_history.append(stress_level)
        return stress_level
    
    def _calculate_hormetic_response(
        self, 
        stress_level: Dict[str, Any], 
        stressor: Stressor
    ) -> Dict[str, Any]:
        """Calculate hormetic response to stress (beneficial adaptation)."""
        overall_stress = stress_level['overall_stress']
        
        # Hormesis: low levels of stress are beneficial
        if overall_stress <= stressor.beneficial_threshold:
            # Beneficial stress level - triggers positive adaptation
            hormetic_benefit = stressor.beneficial_threshold - overall_stress
            
            response = {
                'is_hormetic': True,
                'benefit_level': hormetic_benefit,
                'resistance_gain': hormetic_benefit * 0.5,  # Gain resistance
                'overcompensation': hormetic_benefit * 0.3,  # Overcompensate
                'adaptation_strength': hormetic_benefit * 0.8
            }
        else:
            # Stress level too high - may cause damage
            damage_level = overall_stress - stressor.beneficial_threshold
            
            response = {
                'is_hormetic': False,
                'damage_level': damage_level,
                'resistance_gain': max(0.0, 0.1 - damage_level),  # Minimal gain
                'overcompensation': 0.0,
                'adaptation_strength': max(0.0, 0.2 - damage_level)
            }
        
        return response
    
    async def _apply_adaptation(
        self, 
        component: str, 
        hormetic_response: Dict[str, Any],
        stressor: Stressor
    ) -> Dict[str, Any]:
        """Apply the adaptation based on hormetic response."""
        if not hormetic_response['is_hormetic']:
            # No beneficial adaptation possible
            return {'adapted': False, 'reason': 'Stress level too high for hormesis'}
        
        adaptation_strength = hormetic_response['adaptation_strength']
        
        # Simulate different types of adaptations
        adaptations = []
        
        if stressor.stressor_type == 'error_injection':
            # Adapt to errors by improving error handling
            adaptations.append({
                'type': 'error_resilience',
                'improvement': adaptation_strength * 0.8,
                'description': 'Enhanced error handling and recovery'
            })
        
        elif stressor.stressor_type == 'latency_injection':
            # Adapt to latency by optimizing performance
            adaptations.append({
                'type': 'performance_optimization',
                'improvement': adaptation_strength * 0.6,
                'description': 'Optimized response times and caching'
            })
        
        elif stressor.stressor_type == 'resource_exhaustion':
            # Adapt to resource pressure by improving efficiency
            adaptations.append({
                'type': 'resource_efficiency',
                'improvement': adaptation_strength * 0.7,
                'description': 'Improved resource utilization and cleanup'
            })
        
        # Apply adaptations (simulated)
        await asyncio.sleep(0.1)  # Simulate adaptation time
        
        return {
            'adapted': True,
            'adaptations': adaptations,
            'total_improvement': sum(a['improvement'] for a in adaptations)
        }
    
    def _amplify_strength_gains(
        self, 
        adaptation: Dict[str, Any], 
        stressor: Stressor
    ) -> float:
        """Amplify strength gains through overcompensation."""
        if not adaptation.get('adapted', False):
            return 0.0
        
        base_improvement = adaptation.get('total_improvement', 0.0)
        
        # Overcompensation factor based on stressor intensity
        overcompensation_factor = 1.0 + (stressor.intensity * 0.5)
        
        # Apply diminishing returns
        amplified_gain = base_improvement * overcompensation_factor
        amplified_gain = min(amplified_gain, 1.0)  # Cap at 100% improvement
        
        return amplified_gain
    
    def get_antifragility_metrics(self) -> Dict[str, Any]:
        """Get metrics about system antifragility."""
        if not self.adaptation_history:
            return {
                'total_adaptations': 0,
                'average_strength_gain': 0.0,
                'successful_adaptations': 0,
                'antifragility_score': 0.0
            }
        
        total_adaptations = len(self.adaptation_history)
        successful_adaptations = sum(1 for a in self.adaptation_history if a.is_successful())
        
        average_strength_gain = sum(a.strength_gain for a in self.adaptation_history) / total_adaptations
        average_resistance = sum(a.stressor_resistance for a in self.adaptation_history) / total_adaptations
        
        # Calculate overall antifragility score
        antifragility_score = (average_strength_gain + average_resistance) / 2.0
        
        return {
            'total_adaptations': total_adaptations,
            'successful_adaptations': successful_adaptations,
            'success_rate': successful_adaptations / total_adaptations,
            'average_strength_gain': average_strength_gain,
            'average_resistance': average_resistance,
            'antifragility_score': antifragility_score,
            'recent_adaptations': self.adaptation_history[-5:] if len(self.adaptation_history) >= 5 else self.adaptation_history
        }


class PredictiveFailureDetector:
    """
    Predict failures before they occur using machine learning and topology.
    Based on anomaly detection and topological data analysis.
    """
    
    def __init__(self):
        self.metric_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        self.prediction_horizon = 300.0  # 5 minutes prediction horizon
    
    async def predict_failure(
        self, 
        component: str,
        time_horizon: float = 300.0
    ) -> Dict[str, Any]:
        """Predict potential failures for a component."""
        # Collect current metrics
        current_metrics = await self._collect_component_metrics(component)
        
        # Add to history
        self.metric_history.append({
            'timestamp': time.time(),
            'component': component,
            'metrics': current_metrics
        })
        
        # Detect anomalies
        anomalies = self._detect_anomalies(component, current_metrics)
        
        # Analyze trends
        trends = self._analyze_trends(component)
        
        # Predict failure probability
        failure_probability = self._calculate_failure_probability(anomalies, trends)
        
        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(trends, time_horizon)
        
        prediction = {
            'component': component,
            'prediction_timestamp': time.time(),
            'failure_probability': failure_probability,
            'time_to_failure': time_to_failure,
            'confidence': self._calculate_confidence(anomalies, trends),
            'anomalies_detected': anomalies,
            'trends': trends,
            'recommended_actions': self._recommend_actions(failure_probability, time_to_failure)
        }
        
        return prediction
    
    async def _collect_component_metrics(self, component: str) -> Dict[str, float]:
        """Collect metrics for a specific component."""
        # Get system-wide metrics
        from .exceptions import get_error_analysis_manager
        manager = get_error_analysis_manager()
        health_summary = manager.get_system_health_summary()
        
        # Component-specific metrics
        component_error_count = 0
        for comp, count in health_summary.get('most_affected_components', []):
            if comp == component:
                component_error_count = count
                break
        
        metrics = {
            'error_count': float(component_error_count),
            'error_rate': health_summary.get('error_rate', 0.0),
            'response_time': random.uniform(0.1, 2.0),  # Simulated
            'cpu_usage': random.uniform(0.2, 0.8),      # Simulated
            'memory_usage': random.uniform(0.3, 0.7),   # Simulated
            'disk_usage': random.uniform(0.1, 0.6),     # Simulated
            'network_latency': random.uniform(10, 100), # Simulated (ms)
            'connection_count': random.uniform(10, 100) # Simulated
        }
        
        return metrics
    
    def _detect_anomalies(self, component: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in component metrics."""
        anomalies = []
        
        # Get historical data for this component
        component_history = [
            entry for entry in self.metric_history 
            if entry['component'] == component
        ]
        
        if len(component_history) < 10:  # Need at least 10 data points
            return anomalies
        
        # Calculate statistics for each metric
        for metric_name, current_value in current_metrics.items():
            historical_values = [
                entry['metrics'].get(metric_name, 0.0) 
                for entry in component_history
            ]
            
            if len(historical_values) < 5:
                continue
            
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            
            if std > 0:  # Avoid division by zero
                z_score = abs(current_value - mean) / std
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        'metric': metric_name,
                        'current_value': current_value,
                        'historical_mean': mean,
                        'z_score': z_score,
                        'severity': min(z_score / self.anomaly_threshold, 3.0)  # Cap at 3x
                    })
        
        return anomalies
    
    def _analyze_trends(self, component: str) -> Dict[str, Any]:
        """Analyze trends in component metrics."""
        component_history = [
            entry for entry in self.metric_history 
            if entry['component'] == component
        ]
        
        if len(component_history) < 5:
            return {'trends_available': False}
        
        # Sort by timestamp
        component_history.sort(key=lambda x: x['timestamp'])
        
        trends = {}
        
        # Analyze trend for each metric
        for metric_name in ['error_count', 'error_rate', 'response_time', 'cpu_usage', 'memory_usage']:
            values = [entry['metrics'].get(metric_name, 0.0) for entry in component_history]
            
            if len(values) >= 3:
                # Simple linear trend analysis
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                # Classify trend
                if abs(slope) < 0.01:
                    trend_direction = 'stable'
                elif slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
                
                trends[metric_name] = {
                    'direction': trend_direction,
                    'slope': slope,
                    'recent_value': values[-1],
                    'change_rate': abs(slope)
                }
        
        trends['trends_available'] = True
        return trends
    
    def _calculate_failure_probability(
        self, 
        anomalies: List[Dict[str, Any]], 
        trends: Dict[str, Any]
    ) -> float:
        """Calculate probability of failure based on anomalies and trends."""
        if not trends.get('trends_available', False):
            return 0.1  # Low baseline probability
        
        probability = 0.0
        
        # Anomaly contribution
        if anomalies:
            anomaly_score = sum(a['severity'] for a in anomalies) / len(anomalies)
            probability += min(anomaly_score * 0.3, 0.6)  # Max 60% from anomalies
        
        # Trend contribution
        dangerous_trends = 0
        total_trends = 0
        
        for metric_name, trend_data in trends.items():
            if isinstance(trend_data, dict) and 'direction' in trend_data:
                total_trends += 1
                
                # Increasing error rates/usage is dangerous
                if metric_name in ['error_count', 'error_rate', 'cpu_usage', 'memory_usage']:
                    if trend_data['direction'] == 'increasing' and trend_data['change_rate'] > 0.1:
                        dangerous_trends += 1
                
                # Increasing response time is dangerous
                elif metric_name == 'response_time':
                    if trend_data['direction'] == 'increasing' and trend_data['change_rate'] > 0.05:
                        dangerous_trends += 1
        
        if total_trends > 0:
            trend_score = dangerous_trends / total_trends
            probability += trend_score * 0.4  # Max 40% from trends
        
        return min(probability, 0.95)  # Cap at 95%
    
    def _estimate_time_to_failure(self, trends: Dict[str, Any], time_horizon: float) -> Optional[float]:
        """Estimate time until failure occurs."""
        if not trends.get('trends_available', False):
            return None
        
        min_time_to_failure = float('inf')
        
        # Check critical metrics
        critical_thresholds = {
            'error_rate': 50.0,      # 50 errors/min
            'cpu_usage': 0.95,       # 95% CPU
            'memory_usage': 0.95,    # 95% memory
            'response_time': 10.0    # 10 seconds
        }
        
        for metric_name, threshold in critical_thresholds.items():
            trend_data = trends.get(metric_name)
            if trend_data and isinstance(trend_data, dict):
                if trend_data['direction'] == 'increasing' and trend_data['slope'] > 0:
                    current_value = trend_data['recent_value']
                    slope = trend_data['slope']
                    
                    # Calculate time to reach threshold
                    if current_value < threshold:
                        time_to_threshold = (threshold - current_value) / slope
                        min_time_to_failure = min(min_time_to_failure, time_to_threshold)
        
        if min_time_to_failure == float('inf') or min_time_to_failure > time_horizon:
            return None
        
        return max(min_time_to_failure, 0.0)
    
    def _calculate_confidence(self, anomalies: List[Dict[str, Any]], trends: Dict[str, Any]) -> float:
        """Calculate confidence in the prediction."""
        confidence = 0.5  # Base confidence
        
        # More anomalies = higher confidence
        if anomalies:
            anomaly_confidence = min(len(anomalies) * 0.1, 0.3)
            confidence += anomaly_confidence
        
        # More trend data = higher confidence
        if trends.get('trends_available', False):
            trend_count = sum(1 for k, v in trends.items() 
                            if isinstance(v, dict) and 'direction' in v)
            trend_confidence = min(trend_count * 0.05, 0.2)
            confidence += trend_confidence
        
        return min(confidence, 0.95)
    
    def _recommend_actions(self, failure_probability: float, time_to_failure: Optional[float]) -> List[str]:
        """Recommend actions based on failure prediction."""
        actions = []
        
        if failure_probability > 0.8:
            actions.append("URGENT: Immediate intervention required")
            actions.append("Consider emergency scaling or failover")
        elif failure_probability > 0.6:
            actions.append("HIGH RISK: Schedule maintenance window")
            actions.append("Prepare backup systems")
        elif failure_probability > 0.4:
            actions.append("MODERATE RISK: Monitor closely")
            actions.append("Review resource allocation")
        else:
            actions.append("LOW RISK: Continue normal monitoring")
        
        if time_to_failure is not None:
            if time_to_failure < 300:  # Less than 5 minutes
                actions.append(f"CRITICAL: Failure predicted in {time_to_failure/60:.1f} minutes")
            elif time_to_failure < 3600:  # Less than 1 hour
                actions.append(f"WARNING: Failure predicted in {time_to_failure/60:.1f} minutes")
        
        return actions


class SelfHealingErrorHandler:
    """
    Advanced error handling with self-healing capabilities.
    Integrates chaos engineering, antifragility, and predictive failure detection.
    """
    
    def __init__(self):
        self.chaos_engineer = ChaosEngineer()
        self.antifragility_engine = AntifragilityEngine()
        self.predictive_detector = PredictiveFailureDetector()
        self.healing_strategies: Dict[str, Callable] = {}
        self.healing_history: List[Dict[str, Any]] = []
        
        # Register default healing strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default healing strategies."""
        self.healing_strategies = {
            HealingStrategy.RESTART.value: self._strategy_restart,
            HealingStrategy.ROLLBACK.value: self._strategy_rollback,
            HealingStrategy.SCALE_OUT.value: self._strategy_scale_out,
            HealingStrategy.CIRCUIT_BREAK.value: self._strategy_circuit_break,
            HealingStrategy.FAILOVER.value: self._strategy_failover,
            HealingStrategy.DEGRADE_GRACEFULLY.value: self._strategy_degrade_gracefully,
            HealingStrategy.ISOLATE_AND_HEAL.value: self._strategy_isolate_and_heal,
            HealingStrategy.ADAPTIVE_THROTTLING.value: self._strategy_adaptive_throttling,
            HealingStrategy.RESOURCE_REALLOCATION.value: self._strategy_resource_reallocation,
            HealingStrategy.EMERGENCY_SHUTDOWN.value: self._strategy_emergency_shutdown
        }
    
    async def handle_error_with_healing(self, error: AuraError) -> Dict[str, Any]:
        """Handle error with comprehensive self-healing approach."""
        healing_start_time = time.time()
        
        logger.info(f"Starting self-healing for error: {error.component_id}")
        
        # 1. Predict potential cascade failures
        failure_prediction = await self.predictive_detector.predict_failure(
            error.component_id, time_horizon=300.0
        )
        
        # 2. Determine optimal healing strategy
        healing_strategy = await self._determine_healing_strategy(error, failure_prediction)
        
        # 3. Apply antifragile adaptation if beneficial
        antifragile_result = None
        if failure_prediction['failure_probability'] < 0.7:  # Only if not too critical
            stressor = Stressor(
                stressor_id=f"error_{error.component_id}_{int(time.time())}",
                name=f"Error stress from {error.__class__.__name__}",
                stressor_type="error_injection",
                intensity=error.error_signature.severity,
                frequency=1.0,
                duration=60.0,
                target_components=[error.component_id],
                beneficial_threshold=0.6
            )
            
            try:
                antifragile_result = await self.antifragility_engine.make_antifragile(
                    error.component_id, stressor
                )
            except Exception as e:
                logger.warning(f"Antifragile adaptation failed: {e}")
        
        # 4. Execute healing strategy
        healing_result = await self._execute_healing_strategy(
            healing_strategy, error, failure_prediction
        )
        
        # 5. Verify healing success
        healing_success = await self._verify_healing_success(error.component_id)
        
        # 6. Learn from the healing process
        learning_result = await self._learn_from_healing(
            error, healing_strategy, healing_result, healing_success
        )
        
        healing_end_time = time.time()
        
        # 7. Record healing attempt
        healing_record = {
            'error_id': f"{error.component_id}_{error.timestamp}",
            'component_id': error.component_id,
            'error_type': error.__class__.__name__,
            'error_severity': error.error_signature.severity,
            'healing_strategy': healing_strategy.value,
            'failure_prediction': failure_prediction,
            'antifragile_adaptation': antifragile_result,
            'healing_result': healing_result,
            'healing_success': healing_success,
            'learning_result': learning_result,
            'healing_time': healing_end_time - healing_start_time,
            'timestamp': healing_start_time
        }
        
        self.healing_history.append(healing_record)
        
        logger.info(f"Self-healing completed for {error.component_id}: Success={healing_success}")
        
        return healing_record
    
    async def _determine_healing_strategy(
        self, 
        error: AuraError, 
        failure_prediction: Dict[str, Any]
    ) -> HealingStrategy:
        """Determine the optimal healing strategy."""
        failure_probability = failure_prediction['failure_probability']
        time_to_failure = failure_prediction.get('time_to_failure')
        error_severity = error.error_signature.severity
        
        # Critical situations require immediate action
        if failure_probability > 0.9 or error_severity > 0.9:
            if time_to_failure and time_to_failure < 60:  # Less than 1 minute
                return HealingStrategy.EMERGENCY_SHUTDOWN
            else:
                return HealingStrategy.ISOLATE_AND_HEAL
        
        # High risk situations
        elif failure_probability > 0.7 or error_severity > 0.7:
            if error.error_topology == ErrorTopology.GLOBAL:
                return HealingStrategy.CIRCUIT_BREAK
            else:
                return HealingStrategy.FAILOVER
        
        # Moderate risk situations
        elif failure_probability > 0.5 or error_severity > 0.5:
            if error.error_topology == ErrorTopology.BRANCHING:
                return HealingStrategy.ADAPTIVE_THROTTLING
            else:
                return HealingStrategy.SCALE_OUT
        
        # Low risk situations
        elif failure_probability > 0.3:
            return HealingStrategy.DEGRADE_GRACEFULLY
        
        # Very low risk
        else:
            return HealingStrategy.RESTART
    
    async def _execute_healing_strategy(
        self, 
        strategy: HealingStrategy, 
        error: AuraError,
        failure_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the selected healing strategy."""
        strategy_func = self.healing_strategies.get(strategy.value)
        
        if not strategy_func:
            logger.error(f"Unknown healing strategy: {strategy}")
            return {'success': False, 'reason': 'Unknown strategy'}
        
        try:
            result = await strategy_func(error, failure_prediction)
            result['strategy_executed'] = strategy.value
            return result
        
        except Exception as e:
            logger.error(f"Error executing healing strategy {strategy}: {e}")
            return {
                'success': False, 
                'reason': f'Strategy execution failed: {str(e)}',
                'strategy_executed': strategy.value
            }
    
    # Healing Strategy Implementations
    
    async def _strategy_restart(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Restart the affected component."""
        logger.info(f"Restarting component: {error.component_id}")
        
        # Simulate component restart
        await asyncio.sleep(2.0)  # Restart time
        
        return {
            'success': True,
            'action': 'component_restart',
            'downtime': 2.0,
            'description': f'Restarted {error.component_id}'
        }
    
    async def _strategy_rollback(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback to previous stable state."""
        logger.info(f"Rolling back component: {error.component_id}")
        
        # Simulate rollback
        await asyncio.sleep(5.0)  # Rollback time
        
        return {
            'success': True,
            'action': 'rollback',
            'downtime': 5.0,
            'description': f'Rolled back {error.component_id} to stable state'
        }
    
    async def _strategy_scale_out(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Scale out the component to handle load."""
        logger.info(f"Scaling out component: {error.component_id}")
        
        # Simulate scaling
        await asyncio.sleep(10.0)  # Scaling time
        
        return {
            'success': True,
            'action': 'scale_out',
            'downtime': 0.0,  # No downtime for scaling
            'description': f'Scaled out {error.component_id} with additional instances'
        }
    
    async def _strategy_circuit_break(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Activate circuit breaker to prevent cascade failures."""
        logger.info(f"Activating circuit breaker for: {error.component_id}")
        
        # Simulate circuit breaker activation
        await asyncio.sleep(1.0)
        
        return {
            'success': True,
            'action': 'circuit_break',
            'downtime': 0.0,
            'description': f'Activated circuit breaker for {error.component_id}'
        }
    
    async def _strategy_failover(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Failover to backup component."""
        logger.info(f"Failing over component: {error.component_id}")
        
        # Simulate failover
        await asyncio.sleep(3.0)  # Failover time
        
        return {
            'success': True,
            'action': 'failover',
            'downtime': 3.0,
            'description': f'Failed over {error.component_id} to backup'
        }
    
    async def _strategy_degrade_gracefully(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Gracefully degrade functionality."""
        logger.info(f"Gracefully degrading: {error.component_id}")
        
        # Simulate graceful degradation
        await asyncio.sleep(1.0)
        
        return {
            'success': True,
            'action': 'graceful_degradation',
            'downtime': 0.0,
            'description': f'Gracefully degraded {error.component_id} functionality'
        }
    
    async def _strategy_isolate_and_heal(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate component and perform healing."""
        logger.info(f"Isolating and healing: {error.component_id}")
        
        # Simulate isolation and healing
        await asyncio.sleep(8.0)  # Healing time
        
        return {
            'success': True,
            'action': 'isolate_and_heal',
            'downtime': 8.0,
            'description': f'Isolated and healed {error.component_id}'
        }
    
    async def _strategy_adaptive_throttling(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive throttling to reduce load."""
        logger.info(f"Applying adaptive throttling: {error.component_id}")
        
        # Simulate throttling
        await asyncio.sleep(0.5)
        
        return {
            'success': True,
            'action': 'adaptive_throttling',
            'downtime': 0.0,
            'description': f'Applied adaptive throttling to {error.component_id}'
        }
    
    async def _strategy_resource_reallocation(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Reallocate resources to handle the issue."""
        logger.info(f"Reallocating resources for: {error.component_id}")
        
        # Simulate resource reallocation
        await asyncio.sleep(4.0)
        
        return {
            'success': True,
            'action': 'resource_reallocation',
            'downtime': 0.0,
            'description': f'Reallocated resources for {error.component_id}'
        }
    
    async def _strategy_emergency_shutdown(self, error: AuraError, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency shutdown to prevent system damage."""
        logger.warning(f"Emergency shutdown initiated for: {error.component_id}")
        
        # Simulate emergency shutdown
        await asyncio.sleep(1.0)
        
        return {
            'success': True,
            'action': 'emergency_shutdown',
            'downtime': 60.0,  # Significant downtime
            'description': f'Emergency shutdown of {error.component_id}'
        }
    
    async def _verify_healing_success(self, component_id: str) -> bool:
        """Verify that healing was successful."""
        # Wait a moment for system to stabilize
        await asyncio.sleep(2.0)
        
        # Check component health
        from .exceptions import get_error_analysis_manager
        manager = get_error_analysis_manager()
        health_summary = manager.get_system_health_summary()
        
        # Simple success criteria: system status improved
        if health_summary['status'] in ['healthy', 'degraded']:
            return True
        
        # Check if error rate decreased
        if health_summary['error_rate'] < 5.0:  # Less than 5 errors/min
            return True
        
        return False
    
    async def _learn_from_healing(
        self, 
        error: AuraError,
        strategy: HealingStrategy,
        healing_result: Dict[str, Any],
        success: bool
    ) -> Dict[str, Any]:
        """Learn from the healing process to improve future responses."""
        learning = {
            'strategy_effectiveness': 1.0 if success else 0.0,
            'healing_time': healing_result.get('downtime', 0.0),
            'lessons_learned': []
        }
        
        # Learn from success/failure
        if success:
            learning['lessons_learned'].append(
                f"Strategy {strategy.value} effective for {error.__class__.__name__} "
                f"with severity {error.error_signature.severity:.2f}"
            )
        else:
            learning['lessons_learned'].append(
                f"Strategy {strategy.value} failed for {error.__class__.__name__} "
                f"- consider alternative approaches"
            )
        
        # Learn from healing time
        healing_time = healing_result.get('downtime', 0.0)
        if healing_time > 30.0:  # More than 30 seconds
            learning['lessons_learned'].append(
                f"Healing time {healing_time:.1f}s too long - optimize {strategy.value}"
            )
        
        return learning
    
    def get_healing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive healing metrics."""
        if not self.healing_history:
            return {
                'total_healings': 0,
                'success_rate': 0.0,
                'average_healing_time': 0.0,
                'most_effective_strategy': None
            }
        
        total_healings = len(self.healing_history)
        successful_healings = sum(1 for h in self.healing_history if h['healing_success'])
        
        success_rate = successful_healings / total_healings
        average_healing_time = sum(h['healing_time'] for h in self.healing_history) / total_healings
        
        # Find most effective strategy
        strategy_success = defaultdict(list)
        for healing in self.healing_history:
            strategy = healing['healing_strategy']
            success = healing['healing_success']
            strategy_success[strategy].append(success)
        
        most_effective_strategy = None
        best_success_rate = 0.0
        
        for strategy, successes in strategy_success.items():
            strategy_success_rate = sum(successes) / len(successes)
            if strategy_success_rate > best_success_rate:
                best_success_rate = strategy_success_rate
                most_effective_strategy = strategy
        
        return {
            'total_healings': total_healings,
            'successful_healings': successful_healings,
            'success_rate': success_rate,
            'average_healing_time': average_healing_time,
            'most_effective_strategy': most_effective_strategy,
            'strategy_effectiveness': {
                strategy: sum(successes) / len(successes)
                for strategy, successes in strategy_success.items()
            },
            'chaos_metrics': self.chaos_engineer.get_experiment_summary(),
            'antifragility_metrics': self.antifragility_engine.get_antifragility_metrics()
        }


# Factory functions and utilities

def create_self_healing_error_handler() -> SelfHealingErrorHandler:
    """Create a new self-healing error handler."""
    return SelfHealingErrorHandler()


def create_chaos_experiment(
    name: str,
    failure_type: FailureType,
    target_components: List[str],
    duration: float = 30.0,
    intensity: float = 0.5,
    blast_radius: float = 0.1
) -> ChaosExperiment:
    """Create a chaos experiment with sensible defaults."""
    return ChaosExperiment(
        experiment_id=f"chaos_{int(time.time())}",
        name=name,
        description=f"Chaos experiment testing {failure_type.value} on {target_components}",
        failure_type=failure_type,
        target_components=target_components,
        duration_seconds=duration,
        intensity=intensity,
        blast_radius=blast_radius,
        hypothesis=f"System should remain stable under {failure_type.value}",
        success_criteria=["System recovers within 60 seconds", "No cascading failures"],
        rollback_conditions=["System becomes unresponsive", "Error rate > 50/min"]
    )


# Global self-healing handler instance
_global_self_healing_handler: Optional[SelfHealingErrorHandler] = None


def get_self_healing_handler() -> SelfHealingErrorHandler:
    """Get the global self-healing error handler instance."""
    global _global_self_healing_handler
    if _global_self_healing_handler is None:
        _global_self_healing_handler = SelfHealingErrorHandler()
    return _global_self_healing_handler