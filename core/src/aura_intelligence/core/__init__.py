"""
AURA Intelligence Core Module

This module provides the foundational abstractions and interfaces for the
cognitive-topological architecture, including homotopy type theory foundations.
"""

from .interfaces import (
    SystemComponent,
    CognitiveComponent,
    TopologicalComponent,
    QuantumComponent
)
from .types import (
    AuraType,
    TypeUniverse,
    PathSpace,
    HigherGroupoid
)
from .exceptions import (
    AuraError,
    ConsciousnessError,
    TopologicalComputationError,
    SwarmCoordinationError,
    QuantumComputationError,
    ErrorAnalysisManager,
    get_error_analysis_manager,
    register_system_error
)
from .error_topology import (
    ErrorTopologyAnalyzer,
    ErrorPropagationPattern,
    TopologicalMetrics,
    create_error_topology_analyzer,
    analyze_error_topology
)
from .self_healing import (
    SelfHealingErrorHandler,
    ChaosEngineer,
    AntifragilityEngine,
    PredictiveFailureDetector,
    FailureType,
    HealingStrategy,
    ChaosExperiment,
    create_self_healing_error_handler,
    create_chaos_experiment,
    get_self_healing_handler
)
from .config import (
    AuraConfig,
    ConfigurationManager,
    ConfigLoader,
    ConfigValidator,
    DatabaseConfig,
    RedisConfig,
    Neo4jConfig,
    QuantumConfig,
    ConsciousnessConfig,
    SwarmConfig,
    TDAConfig,
    ObservabilityConfig,
    Environment,
    ConfigSource,
    ConfigMetadata,
    get_config_manager,
    get_config,
    reload_config,
    create_development_config,
    create_production_config,
    create_testing_config
)
from .unified_system import UnifiedSystem, SystemMetrics
from .unified_config import UnifiedConfig
from .unified_interfaces import (
    UnifiedComponent,
    ComponentStatus,
    ComponentMetrics,
    SystemEvent,
    Priority,
    ComponentRegistry,
    get_component_registry
)

# Testing imports commented out due to missing dependencies
# from .testing import (
#     # Advanced Testing Framework
#     AdvancedTestingFramework,
#     PropertyBasedTester,
#     ModelChecker,
#     TheoremProver,
#     ConsciousnessTester,
#     QuantumTester,
#     ChaosTester,
#     
#     # Testing Types and Results
#     TestingLevel,
#     VerificationResult,
#     TestResult,
#     PropertySpecification,
#     
#     # Factory Functions
#     create_advanced_testing_framework,
#     get_testing_framework
# )

__all__ = [
    # Interfaces
    'SystemComponent',
    'CognitiveComponent', 
    'TopologicalComponent',
    'QuantumComponent',
    # Types
    'AuraType',
    'TypeUniverse',
    'PathSpace',
    'HigherGroupoid',
    # Exceptions
    'AuraError',
    'ConsciousnessError',
    'TopologicalComputationError',
    'SwarmCoordinationError',
    'QuantumComputationError',
    'ErrorAnalysisManager',
    'get_error_analysis_manager',
    'register_system_error',
    # Error Topology
    'ErrorTopologyAnalyzer',
    'ErrorPropagationPattern',
    'TopologicalMetrics',
    'create_error_topology_analyzer',
    'analyze_error_topology',
    # Self-Healing
    'SelfHealingErrorHandler',
    'ChaosEngineer',
    'AntifragilityEngine',
    'PredictiveFailureDetector',
    'FailureType',
    'HealingStrategy',
    'ChaosExperiment',
    'create_self_healing_error_handler',
    'create_chaos_experiment',
    'get_self_healing_handler',
    # Configuration Management
    'AuraConfig',
    'ConfigurationManager',
    'ConfigLoader',
    'ConfigValidator',
    'DatabaseConfig',
    'RedisConfig',
    'Neo4jConfig',
    'QuantumConfig',
    'ConsciousnessConfig',
    'SwarmConfig',
    'TDAConfig',
    'ObservabilityConfig',
    'Environment',
    'ConfigSource',
    'ConfigMetadata',
    'get_config_manager',
    'get_config',
    'reload_config',
    'create_development_config',
    'create_production_config',
    'create_testing_config',
    # Unified System
    'UnifiedSystem',
    'SystemMetrics',
    'UnifiedConfig',
    'UnifiedComponent',
    'ComponentStatus',
    'ComponentMetrics',
    'SystemEvent',
    'Priority',
    'ComponentRegistry',
    'get_component_registry',
    # Advanced Testing Framework
    'AdvancedTestingFramework',
    'PropertyBasedTester',
    'ModelChecker',
    'TheoremProver',
    'ConsciousnessTester',
    'QuantumTester',
    'ChaosTester',
    'TestingLevel',
    'VerificationResult',
    'TestResult',
    'PropertySpecification',
    'create_advanced_testing_framework',
    'get_testing_framework'
]