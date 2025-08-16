"""
Exception Hierarchy for AURA Intelligence

This module defines the comprehensive exception hierarchy with topological
error classification for systematic error handling and recovery.
"""

from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import traceback
import time
import asyncio

# Avoid circular imports
if TYPE_CHECKING:
    from .error_topology import ErrorTopologyAnalyzer


class ErrorTopology(Enum):
    """Topological classification of errors."""
    UNKNOWN = "unknown"
    ISOLATED = "isolated"          # Error affects single component
    CONNECTED = "connected"        # Error spreads to connected components  
    CLUSTERED = "clustered"        # Error affects component cluster
    GLOBAL = "global"             # Error affects entire system
    CYCLIC = "cyclic"             # Error creates dependency cycles
    BRANCHING = "branching"       # Error branches to multiple paths


@dataclass
class ErrorSignature:
    """Topological signature of an error."""
    topology: ErrorTopology
    component_id: str
    error_type: str
    propagation_path: List[str]
    affected_components: List[str]
    severity: float  # 0.0 to 1.0
    timestamp: float
    
    @classmethod
    def from_topology(cls, topology: ErrorTopology) -> 'ErrorSignature':
        """Create error signature from topology."""
        return cls(
            topology=topology,
            component_id="unknown",
            error_type="unknown",
            propagation_path=[],
            affected_components=[],
            severity=0.5,
            timestamp=time.time()
        )


@dataclass
class RecoveryStrategy:
    """Strategy for error recovery."""
    strategy_type: str
    priority: int
    estimated_recovery_time: float
    success_probability: float
    side_effects: List[str]
    
    def __post_init__(self):
        """Validate recovery strategy."""
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("Success probability must be between 0.0 and 1.0")


class AuraError(Exception):
    """
    Base exception for all AURA Intelligence errors.
    
    Includes topological error classification and recovery strategy
    determination for systematic error handling.
    """
    
    def __init__(
        self, 
        message: str, 
        error_topology: ErrorTopology = ErrorTopology.UNKNOWN,
        component_id: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_topology = error_topology
        self.component_id = component_id
        self.context = context or {}
        self.timestamp = time.time()
        self.traceback_info = traceback.format_exc()
        
        # Compute error signature
        self.error_signature = self._compute_error_signature()
        
        # Determine recovery strategy
        self.recovery_strategy = self._determine_recovery_strategy()
    
    def _compute_error_signature(self) -> ErrorSignature:
        """Compute topological signature of the error."""
        return ErrorSignature(
            topology=self.error_topology,
            component_id=self.component_id,
            error_type=self.__class__.__name__,
            propagation_path=self.context.get('propagation_path', []),
            affected_components=self.context.get('affected_components', []),
            severity=self._calculate_severity(),
            timestamp=self.timestamp
        )
    
    def _calculate_severity(self) -> float:
        """Calculate error severity based on topology and context."""
        base_severity = {
            ErrorTopology.ISOLATED: 0.2,
            ErrorTopology.CONNECTED: 0.4,
            ErrorTopology.CLUSTERED: 0.6,
            ErrorTopology.GLOBAL: 0.9,
            ErrorTopology.CYCLIC: 0.8,
            ErrorTopology.BRANCHING: 0.7,
            ErrorTopology.UNKNOWN: 0.5
        }
        
        severity = base_severity.get(self.error_topology, 0.5)
        
        # Adjust based on context
        if self.context.get('critical_component', False):
            severity += 0.2
        if self.context.get('cascade_potential', False):
            severity += 0.1
        
        return min(severity, 1.0)
    
    def _determine_recovery_strategy(self) -> RecoveryStrategy:
        """Determine optimal recovery strategy based on error characteristics."""
        if self.error_topology == ErrorTopology.ISOLATED:
            return RecoveryStrategy(
                strategy_type="component_restart",
                priority=1,
                estimated_recovery_time=5.0,
                success_probability=0.9,
                side_effects=["temporary_unavailability"]
            )
        elif self.error_topology == ErrorTopology.GLOBAL:
            return RecoveryStrategy(
                strategy_type="system_rollback",
                priority=10,
                estimated_recovery_time=60.0,
                success_probability=0.7,
                side_effects=["data_loss", "service_interruption"]
            )
        else:
            return RecoveryStrategy(
                strategy_type="adaptive_recovery",
                priority=5,
                estimated_recovery_time=15.0,
                success_probability=0.8,
                side_effects=["performance_degradation"]
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'topology': self.error_topology.value,
            'component_id': self.component_id,
            'context': self.context,
            'timestamp': self.timestamp,
            'signature': {
                'topology': self.error_signature.topology.value,
                'severity': self.error_signature.severity,
                'affected_components': self.error_signature.affected_components
            },
            'recovery_strategy': {
                'type': self.recovery_strategy.strategy_type,
                'priority': self.recovery_strategy.priority,
                'success_probability': self.recovery_strategy.success_probability
            }
        }


class ConsciousnessError(AuraError):
    """Errors in consciousness layer components."""
    
    def __init__(self, message: str, consciousness_level: float = 0.0, **kwargs):
        super().__init__(message, **kwargs)
        self.consciousness_level = consciousness_level
        
        # Consciousness errors often have global impact
        if self.error_topology == ErrorTopology.UNKNOWN:
            self.error_topology = ErrorTopology.GLOBAL


class TopologicalComputationError(AuraError):
    """Errors in topological computation components."""
    
    def __init__(self, message: str, computation_type: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.computation_type = computation_type
        
        # Topological errors often create cycles
        if self.error_topology == ErrorTopology.UNKNOWN:
            self.error_topology = ErrorTopology.CYCLIC


class SwarmCoordinationError(AuraError):
    """Errors in swarm coordination and agent management."""
    
    def __init__(self, message: str, swarm_size: int = 0, **kwargs):
        super().__init__(message, **kwargs)
        self.swarm_size = swarm_size
        
        # Swarm errors often branch to multiple agents
        if self.error_topology == ErrorTopology.UNKNOWN:
            self.error_topology = ErrorTopology.BRANCHING


class QuantumComputationError(AuraError):
    """Errors in quantum computation components."""
    
    def __init__(self, message: str, quantum_fidelity: float = 0.0, **kwargs):
        super().__init__(message, **kwargs)
        self.quantum_fidelity = quantum_fidelity
        
        # Quantum errors are often isolated due to decoherence
        if self.error_topology == ErrorTopology.UNKNOWN:
            self.error_topology = ErrorTopology.ISOLATED


class ConfigurationError(AuraError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key


class DatabaseError(AuraError):
    """Database-related errors."""
    
    def __init__(self, message: str, database_type: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.database_type = database_type


class NetworkError(AuraError):
    """Network-related errors."""
    
    def __init__(self, message: str, network_partition: bool = False, **kwargs):
        super().__init__(message, **kwargs)
        self.network_partition = network_partition
        
        # Network partitions create clustered errors
        if network_partition and self.error_topology == ErrorTopology.UNKNOWN:
            self.error_topology = ErrorTopology.CLUSTERED


class ValidationError(AuraError):
    """Data validation errors."""
    
    def __init__(self, message: str, validation_type: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type


class SecurityError(AuraError):
    """Security-related errors."""
    
    def __init__(self, message: str, security_level: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.security_level = security_level
        
        # Security errors often have global impact
        if self.error_topology == ErrorTopology.UNKNOWN:
            self.error_topology = ErrorTopology.GLOBAL


# Convenience functions for error creation
def create_consciousness_error(
    message: str, 
    component_id: str = "consciousness",
    consciousness_level: float = 0.0
) -> ConsciousnessError:
    """Create a consciousness error with appropriate context."""
    return ConsciousnessError(
        message=message,
        component_id=component_id,
        consciousness_level=consciousness_level,
        context={'critical_component': True}
    )


def create_topological_error(
    message: str,
    component_id: str = "topology",
    computation_type: str = "persistent_homology"
) -> TopologicalComputationError:
    """Create a topological computation error with appropriate context."""
    return TopologicalComputationError(
        message=message,
        component_id=component_id,
        computation_type=computation_type,
        context={'cascade_potential': True}
    )


def create_swarm_error(
    message: str,
    component_id: str = "swarm",
    swarm_size: int = 0,
    affected_agents: List[str] = None
) -> SwarmCoordinationError:
    """Create a swarm coordination error with appropriate context."""
    return SwarmCoordinationError(
        message=message,
        component_id=component_id,
        swarm_size=swarm_size,
        context={
            'affected_components': affected_agents or [],
            'cascade_potential': swarm_size > 10
        }
    )


def create_quantum_error(
    message: str,
    component_id: str = "quantum",
    quantum_fidelity: float = 0.0
) -> QuantumComputationError:
    """Create a quantum computation error with appropriate context."""
    return QuantumComputationError(
        message=message,
        component_id=component_id,
        quantum_fidelity=quantum_fidelity,
        context={'critical_component': quantum_fidelity < 0.5}
    )


class ErrorAnalysisManager:
    """
    Manager for comprehensive error analysis and topology tracking.
    
    Integrates with ErrorTopologyAnalyzer to provide system-wide
    error analysis and recovery optimization.
    """
    
    def __init__(self):
        self.topology_analyzer: Optional['ErrorTopologyAnalyzer'] = None
        self.error_history: List[AuraError] = []
        self.analysis_cache: Dict[str, Any] = {}
        self._initialize_topology_analyzer()
    
    def _initialize_topology_analyzer(self):
        """Initialize the topology analyzer (lazy loading to avoid circular imports)."""
        try:
            from .error_topology import ErrorTopologyAnalyzer
            self.topology_analyzer = ErrorTopologyAnalyzer()
        except ImportError:
            # Graceful degradation if topology analyzer is not available
            self.topology_analyzer = None
    
    def register_error(self, error: AuraError) -> Dict[str, Any]:
        """
        Register an error and perform comprehensive analysis.
        
        Returns analysis results including topology metrics and recovery recommendations.
        """
        # Add to history
        self.error_history.append(error)
        
        # Perform topology analysis if available
        analysis_result = {
            'error_registered': True,
            'timestamp': time.time(),
            'error_signature': error.error_signature.topology.value,
            'severity': error.error_signature.severity
        }
        
        if self.topology_analyzer:
            try:
                # Add error to topology analyzer
                error_node = self.topology_analyzer.add_error(error)
                
                # Perform topology analysis
                topology_metrics = self.topology_analyzer.analyze_topology()
                
                # Detect propagation pattern
                propagation_pattern = self.topology_analyzer.detect_propagation_pattern(error.component_id)
                
                # Predict cascade
                cascade_prediction = self.topology_analyzer.predict_error_cascade(error)
                
                # Get critical components
                critical_components = self.topology_analyzer.compute_critical_components()
                
                # Optimize recovery strategy
                available_resources = {'cpu': 1.0, 'memory': 1.0}  # Default resources
                optimized_strategy = self.topology_analyzer.optimize_recovery_strategy(
                    error, available_resources
                )
                
                analysis_result.update({
                    'topology_analysis': {
                        'clustering_coefficient': topology_metrics.clustering_coefficient,
                        'average_path_length': topology_metrics.average_path_length,
                        'density': topology_metrics.density,
                        'is_small_world': topology_metrics.is_small_world(),
                        'is_scale_free': topology_metrics.is_scale_free()
                    },
                    'propagation_pattern': propagation_pattern.value,
                    'cascade_prediction': cascade_prediction[:5],  # Top 5 predictions
                    'critical_components': critical_components[:3],  # Top 3 critical
                    'optimized_recovery': {
                        'strategy_type': optimized_strategy.strategy_type,
                        'priority': optimized_strategy.priority,
                        'estimated_time': optimized_strategy.estimated_recovery_time,
                        'success_probability': optimized_strategy.success_probability
                    }
                })
                
            except Exception as e:
                analysis_result['topology_analysis_error'] = str(e)
        
        return analysis_result
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary based on error analysis."""
        if not self.error_history:
            return {
                'status': 'healthy',
                'total_errors': 0,
                'error_rate': 0.0,
                'topology_health': 'unknown'
            }
        
        # Basic statistics
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 300]  # Last 5 minutes
        error_rate = len(recent_errors) / 5.0  # Errors per minute
        
        # Severity analysis
        severity_distribution = {}
        for error in self.error_history:
            severity_level = self._categorize_severity(error.error_signature.severity)
            severity_distribution[severity_level] = severity_distribution.get(severity_level, 0) + 1
        
        # Topology health
        topology_health = 'unknown'
        if self.topology_analyzer:
            try:
                topology_summary = self.topology_analyzer.get_topology_summary()
                
                # Determine topology health based on metrics
                if topology_summary['density'] > 0.7:
                    topology_health = 'critical'  # High interconnectedness
                elif topology_summary['clustering_coefficient'] > 0.5:
                    topology_health = 'warning'  # High clustering
                else:
                    topology_health = 'stable'
                    
            except Exception:
                topology_health = 'analysis_failed'
        
        # Overall status
        if error_rate > 10:
            status = 'critical'
        elif error_rate > 5:
            status = 'warning'
        elif error_rate > 1:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'error_rate': error_rate,
            'severity_distribution': severity_distribution,
            'topology_health': topology_health,
            'most_affected_components': self._get_most_affected_components(),
            'recommendations': self._generate_recommendations(status, error_rate, topology_health)
        }
    
    def _categorize_severity(self, severity: float) -> str:
        """Categorize severity level."""
        if severity >= 0.8:
            return 'critical'
        elif severity >= 0.6:
            return 'high'
        elif severity >= 0.4:
            return 'medium'
        elif severity >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _get_most_affected_components(self) -> List[Tuple[str, int]]:
        """Get components with most errors."""
        component_counts = {}
        for error in self.error_history:
            component_counts[error.component_id] = component_counts.get(error.component_id, 0) + 1
        
        # Sort by count (descending)
        sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_components[:5]  # Top 5
    
    def _generate_recommendations(self, status: str, error_rate: float, topology_health: str) -> List[str]:
        """Generate recommendations based on system health."""
        recommendations = []
        
        if status == 'critical':
            recommendations.append("Immediate intervention required - system stability at risk")
            recommendations.append("Consider activating emergency recovery procedures")
        
        if error_rate > 5:
            recommendations.append("High error rate detected - investigate root causes")
            recommendations.append("Consider scaling resources or load balancing")
        
        if topology_health == 'critical':
            recommendations.append("Error topology shows high interconnectedness - risk of cascading failures")
            recommendations.append("Implement circuit breakers and isolation mechanisms")
        
        if topology_health == 'warning':
            recommendations.append("Error clustering detected - monitor for cascade patterns")
        
        if not recommendations:
            recommendations.append("System health appears stable - continue monitoring")
        
        return recommendations
    
    async def analyze_error_patterns(self) -> Dict[str, Any]:
        """Perform advanced pattern analysis on error history."""
        if not self.error_history:
            return {'patterns': [], 'insights': []}
        
        patterns = []
        insights = []
        
        # Temporal pattern analysis
        temporal_patterns = self._analyze_temporal_patterns()
        patterns.extend(temporal_patterns)
        
        # Component pattern analysis
        component_patterns = self._analyze_component_patterns()
        patterns.extend(component_patterns)
        
        # Severity pattern analysis
        severity_patterns = self._analyze_severity_patterns()
        patterns.extend(severity_patterns)
        
        # Generate insights
        if temporal_patterns:
            insights.append("Temporal clustering of errors detected - possible systemic issues")
        
        if len(set(e.component_id for e in self.error_history)) < len(self.error_history) * 0.3:
            insights.append("Errors concentrated in few components - targeted intervention needed")
        
        if any(e.error_signature.severity > 0.8 for e in self.error_history[-10:]):
            insights.append("Recent high-severity errors detected - system stability at risk")
        
        return {
            'patterns': patterns,
            'insights': insights,
            'analysis_timestamp': time.time()
        }
    
    def _analyze_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in error occurrence."""
        if len(self.error_history) < 3:
            return []
        
        patterns = []
        
        # Sort errors by timestamp
        sorted_errors = sorted(self.error_history, key=lambda e: e.timestamp)
        
        # Look for clusters in time
        time_gaps = []
        for i in range(1, len(sorted_errors)):
            gap = sorted_errors[i].timestamp - sorted_errors[i-1].timestamp
            time_gaps.append(gap)
        
        # Detect clusters (gaps < 60 seconds)
        cluster_count = sum(1 for gap in time_gaps if gap < 60)
        
        if cluster_count > len(time_gaps) * 0.5:
            patterns.append({
                'type': 'temporal_clustering',
                'description': 'Errors occurring in temporal clusters',
                'confidence': cluster_count / len(time_gaps),
                'recommendation': 'Investigate systemic causes of error bursts'
            })
        
        return patterns
    
    def _analyze_component_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in component error distribution."""
        patterns = []
        
        component_counts = {}
        for error in self.error_history:
            component_counts[error.component_id] = component_counts.get(error.component_id, 0) + 1
        
        if not component_counts:
            return patterns
        
        # Check for concentration
        total_errors = len(self.error_history)
        max_component_errors = max(component_counts.values())
        
        if max_component_errors > total_errors * 0.5:
            patterns.append({
                'type': 'component_concentration',
                'description': 'Errors concentrated in single component',
                'confidence': max_component_errors / total_errors,
                'recommendation': 'Focus recovery efforts on most affected component'
            })
        
        return patterns
    
    def _analyze_severity_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in error severity."""
        patterns = []
        
        if len(self.error_history) < 5:
            return patterns
        
        # Check for severity escalation
        recent_errors = self.error_history[-5:]
        severity_trend = [e.error_signature.severity for e in recent_errors]
        
        # Simple trend detection
        increasing_trend = all(severity_trend[i] <= severity_trend[i+1] 
                              for i in range(len(severity_trend)-1))
        
        if increasing_trend and severity_trend[-1] > 0.7:
            patterns.append({
                'type': 'severity_escalation',
                'description': 'Error severity increasing over time',
                'confidence': 0.8,
                'recommendation': 'Immediate attention required - escalating severity detected'
            })
        
        return patterns


# Global error analysis manager instance
_global_error_manager: Optional[ErrorAnalysisManager] = None


def get_error_analysis_manager() -> ErrorAnalysisManager:
    """Get the global error analysis manager instance."""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ErrorAnalysisManager()
    return _global_error_manager


def register_system_error(error: AuraError) -> Dict[str, Any]:
    """Register an error with the global error analysis system."""
    manager = get_error_analysis_manager()
    return manager.register_error(error)