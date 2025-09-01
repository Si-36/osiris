"""
🔧 AURA Intelligence Configuration Module

Modular, type-safe configuration system using Pydantic v2.

IMPORTANT: This module maintains backward compatibility with legacy naming conventions.
The *Settings classes are the modern implementation, while *Config aliases are provided
for backward compatibility with existing code and tests.
"""

from .base_compat import BaseSettings, EnvironmentType, EnhancementLevel
from .base import get_config, AURAConfig
from .agent import AgentSettings
from .memory import MemorySettings
from .api import APISettings
from .observability import ObservabilitySettings
from .integration import IntegrationSettings
from .security import SecuritySettings
from .deployment import DeploymentSettings
from .aura import AURASettings

# Backward compatibility aliases
# These map the modern *Settings classes to the legacy *Config names

UltimateAURAConfig = AURASettings
AuraConfig = AURASettings  # Main config alias
AgentConfig = AgentSettings
MemoryConfig = MemorySettings
ObservabilityConfig = ObservabilitySettings
KnowledgeConfig = AURASettings  # Knowledge is part of main AURA settings
TopologyConfig = AURASettings  # Topology is part of main AURA settings

# Additional legacy aliases that core module expects
DatabaseConfig = AURASettings  # Database is part of main settings
RedisConfig = MemorySettings   # Redis is part of memory settings
Neo4jConfig = AURASettings     # Neo4j is part of main settings
QuantumConfig = AURASettings   # Quantum is part of main settings
ConsciousnessConfig = AURASettings  # Consciousness is part of main settings
SwarmConfig = AgentSettings    # Swarm is part of agent settings
TDAConfig = AURASettings       # TDA is part of main settings
ConfigurationManager = AURASettings  # Manager is the main settings class
ConfigLoader = AURASettings    # Loader is the main settings class
ConfigValidator = AURASettings # Validator is the main settings class

# Additional missing imports that core expects
Environment = EnvironmentType  # Environment alias
ConfigSource = AURASettings    # Config source is main settings
ConfigMetadata = AURASettings  # Metadata is main settings

# Factory function aliases (will be defined after the factory functions)
def get_config_manager():
    return get_config()

def reload_config():
    return get_config()

# Factory functions for configuration
def get_ultimate_config() -> AURASettings:
    """
    Get the ultimate AURA configuration with default settings.
    
    Returns:
        AURASettings: Default ultimate configuration
    """
    config = AURASettings()
    # Set ultimate defaults
    config.agent.enhancement_level = EnhancementLevel.ULTIMATE
    config.agent.agent_count = 10
    config.memory.enable_consciousness = True
    return config

def get_production_config() -> AURASettings:
    """
    Get production-ready AURA configuration.
    
    Returns:
        AURASettings: Production configuration
    """
    config = AURASettings(environment=EnvironmentType.PRODUCTION)
    # Production defaults
    config.agent.enhancement_level = EnhancementLevel.ADVANCED
    config.agent.agent_count = 5
    config.security.enable_auth = True
    config.observability.enable_metrics = True
    config.observability.enable_tracing = True
    config.deployment.deployment_mode = "production"
    return config

def get_enterprise_config() -> AURASettings:
    """
    Get enterprise AURA configuration with all features enabled.
    
    Returns:
        AURASettings: Enterprise configuration
    """
    config = AURASettings(environment=EnvironmentType.PRODUCTION)
    # Enterprise features
    config.agent.enhancement_level = EnhancementLevel.ULTIMATE
    config.agent.agent_count = 20
    config.memory.enable_consciousness = True
    config.memory.enable_mem0 = True
    config.security.enable_auth = True
    config.security.enable_encryption = True
    config.observability.enable_metrics = True
    config.observability.enable_tracing = True
    config.observability.enable_profiling = True
    config.deployment.deployment_mode = "multi_region"
    config.deployment.canary_enabled = True
    return config

def get_development_config() -> AURASettings:
    """
    Get development AURA configuration for local testing.
    
    Returns:
        AURASettings: Development configuration
    """
    config = AURASettings(environment=EnvironmentType.DEVELOPMENT)
    # Development defaults
    config.agent.enhancement_level = EnhancementLevel.BASIC
    config.agent.agent_count = 3
    config.security.enable_auth = False
    config.observability.enable_metrics = True
    config.observability.enable_tracing = False
    return config

def create_testing_config() -> AURASettings:
    """
    Get testing AURA configuration.
    
    Returns:
        AURASettings: Testing configuration
    """
    return get_development_config()

# Additional factory function aliases
create_development_config = get_development_config
create_production_config = get_production_config

__all__ = [
    # Base classes
    "BaseSettings",
    "EnvironmentType",
    "EnhancementLevel",
    
    # Modern Settings classes
    "AgentSettings",
    "MemorySettings",
    "APISettings",
    "ObservabilitySettings",
    "IntegrationSettings",
    "SecuritySettings",
    "DeploymentSettings",
    "AURASettings",
    
    # Backward compatibility aliases
    "UltimateAURAConfig",
    "AuraConfig",
    "AgentConfig",
    "MemoryConfig",
    "ObservabilityConfig",
    "KnowledgeConfig",
    "TopologyConfig",
    "DatabaseConfig",
    "RedisConfig",
    "Neo4jConfig",
    "QuantumConfig",
    "ConsciousnessConfig",
    "SwarmConfig",
    "TDAConfig",
    "ConfigurationManager",
    "ConfigLoader",
    "ConfigValidator",
    "Environment",
    "ConfigSource",
    "ConfigMetadata",
    
    # Factory functions
    "get_ultimate_config",
    "get_production_config",
    "get_enterprise_config",
    "get_development_config",
    "get_config",
    "get_config_manager",
    "reload_config",
    "create_development_config",
    "create_production_config",
    "create_testing_config",
]