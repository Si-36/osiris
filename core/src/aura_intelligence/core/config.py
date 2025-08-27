"""
Advanced Configuration Management System

This module provides a comprehensive configuration management system following
12-factor app principles with environment-based configuration, validation,
and type safety.

Key Features:
- Environment-based configuration with multiple sources
- Schema validation and type checking
- Configuration hot-reloading and change detection
- Secure secret management with encryption
- Configuration versioning and rollback
- Multi-environment support (dev, staging, prod)
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib
import time
from abc import ABC, abstractmethod

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigSource(Enum):
    """Configuration source types."""
    ENVIRONMENT = "environment"
    FILE = "file"
    REMOTE = "remote"
    VAULT = "vault"
    DATABASE = "database"


@dataclass
class ConfigMetadata:
    """Metadata for configuration values."""
    source: ConfigSource
    last_updated: float
    version: str
    checksum: str
    encrypted: bool = False
    sensitive: bool = False


class ConfigValidator:
    """Advanced configuration validation with schema checking."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.schema_cache: Dict[str, Dict] = {}
    
    def add_validation_rule(self, key: str, validator_func: Callable) -> None:
        """Add a validation rule for a configuration key."""
        if key not in self.validation_rules:
            self.validation_rules[key] = []
        self.validation_rules[key].append(validator_func)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate configuration against all rules."""
        errors = {}
        
        for key, value in config.items():
            pass
        if key in self.validation_rules:
            key_errors = []
        for validator_func in self.validation_rules[key]:
            pass
        try:
            if not validator_func(value):
                key_errors.append(f"Validation failed for {key}")
        except Exception as e:
            pass
        key_errors.append(f"Validation error for {key}: {str(e)}")
                
        if key_errors:
            errors[key] = key_errors
        
        return errors
    
    def validate_type(self, value: Any, expected_type: Type) -> bool:
        """Validate that a value matches the expected type."""
        try:
            if expected_type == bool and isinstance(value, str):
                return value.lower() in ('true', 'false', '1', '0', 'yes', 'no')
            return isinstance(value, expected_type)
        except Exception:
            return False
    
    def validate_range(self, value: Union[int, float], min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None) -> bool:
        """Validate that a numeric value is within a specified range."""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True


class ConfigLoader:
    """Multi-source configuration loader with priority handling."""
    
    def __init__(self, validator: Optional[ConfigValidator] = None):
        self.validator = validator or ConfigValidator()
        self.sources: List[ConfigSource] = []
        self.config_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, ConfigMetadata] = {}
        self.watchers: List[Callable] = []
    
    def add_source(self, source: ConfigSource, priority: int = 0) -> None:
        """Add a configuration source with priority."""
        self.sources.append((source, priority))
        self.sources.sort(key=lambda x: x[1], reverse=True)  # Higher priority first
    
    def load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        for key, value in os.environ.items():
            pass
        if key.startswith('AURA_'):
            config_key = key[5:].lower()  # Remove AURA_ prefix
        config[config_key] = self._parse_env_value(value)
                
        # Create metadata
        self.metadata_cache[config_key] = ConfigMetadata(
        source=ConfigSource.ENVIRONMENT,
        last_updated=time.time(),
        version="1.0",
        checksum=hashlib.sha256(str(value).encode()).hexdigest(),
        sensitive=self._is_sensitive_key(config_key)
        )
        
        return config
    
    def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a file (JSON, YAML, or .env)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    config = yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    config = json.load(f)
                elif file_path.suffix.lower() == '.env':
                    config = self._parse_env_file(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Create metadata for all keys
            file_content = file_path.read_text()
            checksum = hashlib.sha256(file_content.encode()).hexdigest()
            
            for key in config.keys():
                self.metadata_cache[key] = ConfigMetadata(
                    source=ConfigSource.FILE,
                    last_updated=file_path.stat().st_mtime,
                    version="1.0",
                    checksum=checksum,
                    sensitive=self._is_sensitive_key(key)
                )
            
            return config
            
        except Exception as e:
            logging.error(f"Failed to load config from {file_path}: {e}")
            return {}
    
    def load_from_remote(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Load configuration from a remote source."""
        # Placeholder for remote configuration loading
        # In a real implementation, this would use HTTP requests
        logging.info(f"Remote config loading from {url} not implemented yet")
        return {}
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries with priority."""
        merged = {}
        
        for config in configs:
            for key, value in config.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        
        return merged
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            pass
        return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
        else:
        return int(value)
        except ValueError:
        pass
        
        # JSON values
        if value.startswith(('{', '[', '"')):
            try:
                return json.loads(value)
        except json.JSONDecodeError:
        pass
        
        # Comma-separated lists
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        return value
    
    def _parse_env_file(self, file_handle) -> Dict[str, Any]:
        """Parse .env file format."""
        config = {}
        for line in file_handle:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = self._parse_env_value(value.strip())
        return config
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a configuration key contains sensitive information."""
        sensitive_patterns = [
        'password', 'secret', 'key', 'token', 'credential',
        'auth', 'private', 'cert', 'ssl', 'tls'
        ]
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)


    @dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "aura_intelligence"
    username: str = "aura"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis configuration settings."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)
    connection_pool_max_connections: int = 50


@dataclass
class Neo4jConfig:
    """Neo4j configuration settings."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 100
    connection_acquisition_timeout: int = 60
    trust: str = "TRUST_ALL_CERTIFICATES"
    encrypted: bool = False


@dataclass
class QuantumConfig:
    """Quantum computing configuration settings."""
    backend: str = "qiskit_aer"
    shots: int = 1024
    max_qubits: int = 32
    optimization_level: int = 1
    seed_simulator: Optional[int] = None
    memory: bool = False
    parameter_binds: Optional[Dict[str, Any]] = None
    
    # IBM Quantum specific
    ibm_token: Optional[str] = None
    ibm_backend: str = "ibmq_qasm_simulator"
    
    # AWS Braket specific
    aws_region: str = "us-east-1"
    braket_device: str = "LocalSimulator"


@dataclass
class ConsciousnessConfig:
    """Consciousness system configuration settings."""
    global_workspace_size: int = 1000
    attention_window_size: int = 100
    consciousness_threshold: float = 0.7
    integration_phi_threshold: float = 0.5
    working_memory_capacity: int = 7
    executive_function_enabled: bool = True
    metacognitive_monitoring: bool = True
    consciousness_stream_buffer_size: int = 10000


@dataclass
class SwarmConfig:
    """Swarm intelligence configuration settings."""
    max_agents: int = 1000
    emergence_threshold: float = 0.8
    collective_decision_threshold: float = 0.6
    phase_transition_sensitivity: float = 0.1
    self_organization_rate: float = 0.05
    byzantine_fault_tolerance: bool = True
    consensus_algorithm: str = "raft"
    swarm_topology_update_interval: int = 30


@dataclass
class TDAConfig:
    """Topological Data Analysis configuration settings."""
    max_dimension: int = 3
    max_edge_length: float = 1.0
    resolution: float = 0.1
    gpu_acceleration: bool = False
    distributed_computing: bool = False
    quantum_enhancement: bool = False
    streaming_mode: bool = False
    persistence_threshold: float = 0.01
    
    # Algorithm-specific settings
    ripser_threshold: float = 1.0
    gudhi_expansion_limit: int = 1000
    giotto_n_jobs: int = -1


@dataclass
class ObservabilityConfig:
    """Observability and monitoring configuration settings."""
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    logging_level: str = "INFO"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    prometheus_port: int = 8000
    grafana_dashboard_enabled: bool = True
    
    # OpenTelemetry settings
    otel_service_name: str = "aura-intelligence"
    otel_service_version: str = "1.0.0"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"


class AuraConfig(BaseSettings):
    """
    Main configuration class for AURA Intelligence system.
    
    This class uses Pydantic BaseSettings for automatic environment variable
    loading and validation following 12-factor app principles.
    """
    
    # Environment and deployment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    
    # Application settings
    app_name: str = "AURA Intelligence"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Security settings
    secret_key: str = Field(..., min_length=32)
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    consciousness: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # Advanced settings
    max_workers: int = 4
    request_timeout: int = 30
    rate_limit_per_minute: int = 1000
    
    model_config = SettingsConfigDict(
        env_prefix="AURA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
        @field_validator('secret_key', 'jwt_secret_key')
        @classmethod
    def validate_secret_keys(cls, v):
        if len(v) < 32:
            raise ValueError('Secret keys must be at least 32 characters long')
        return v
    
    @field_validator('api_port')
    @classmethod
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class ConfigurationManager:
    """
    Advanced configuration manager with hot-reloading, validation,
    and multi-source support.
    """
    
    def __init__(self, config_class: Type[BaseSettings] = AuraConfig):
        self.config_class = config_class
        self.loader = ConfigLoader()
        self.validator = ConfigValidator()
        self._config: Optional[BaseSettings] = None
        self._config_hash: Optional[str] = None
        self._watchers: List[Callable[[BaseSettings], None]] = []
        
        # Set up validation rules
        self._setup_validation_rules()
    
    def load_config(self, config_files: Optional[List[Union[str, Path]]] = None) -> BaseSettings:
        """Load configuration from multiple sources with validation."""
        configs = []
        
        # Load from environment variables
        env_config = self.loader.load_from_environment()
        if env_config:
            configs.append(env_config)
        
        # Load from files
        if config_files:
            for config_file in config_files:
                file_config = self.loader.load_from_file(config_file)
                if file_config:
                    configs.append(file_config)
        
        # Merge all configurations
        merged_config = self.loader.merge_configs(*configs) if configs else {}
        
        # Validate configuration
        validation_errors = self.validator.validate_config(merged_config)
        if validation_errors:
            error_msg = "Configuration validation failed:\n"
            for key, errors in validation_errors.items():
                error_msg += f"  {key}: {', '.join(errors)}\n"
            raise ValueError(error_msg)
        
        # Create configuration instance
        try:
            self._config = self.config_class(**merged_config)
        except Exception as e:
            # Fallback to default configuration
            logging.warning(f"Failed to create config with merged values: {e}")
            self._config = self.config_class()
        
        # Calculate configuration hash for change detection
        config_str = json.dumps(self._config.dict(), sort_keys=True, default=str)
        new_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Notify watchers if configuration changed
        if self._config_hash and self._config_hash != new_hash:
            for watcher in self._watchers:
                try:
                    watcher(self._config)
                except Exception as e:
                    logging.error(f"Configuration watcher failed: {e}")
        
        self._config_hash = new_hash
        return self._config
    
    def get_config(self) -> BaseSettings:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config
    
    def reload_config(self, config_files: Optional[List[Union[str, Path]]] = None) -> BaseSettings:
        """Reload configuration from sources."""
        return self.load_config(config_files)
    
    def add_config_watcher(self, watcher: Callable[[BaseSettings], None]) -> None:
        """Add a watcher that gets called when configuration changes."""
        self._watchers.append(watcher)
    
    def remove_config_watcher(self, watcher: Callable[[BaseSettings], None]) -> None:
        """Remove a configuration watcher."""
        if watcher in self._watchers:
            self._watchers.remove(watcher)
    
    def validate_current_config(self) -> Dict[str, List[str]]:
        """Validate the current configuration."""
        if self._config is None:
            return {"general": ["No configuration loaded"]}
        
        return self.validator.validate_config(self._config.dict())
    
    def get_config_metadata(self, key: str) -> Optional[ConfigMetadata]:
        """Get metadata for a configuration key."""
        return self.loader.metadata_cache.get(key)
    
    def _setup_validation_rules(self) -> None:
        """Set up validation rules for configuration values."""
        pass
        # Database validation
        self.validator.add_validation_rule(
        'database.port',
        lambda x: self.validator.validate_range(x, 1, 65535)
        )
        
        # Redis validation
        self.validator.add_validation_rule(
        'redis.port',
        lambda x: self.validator.validate_range(x, 1, 65535)
        )
        
        # Quantum validation
        self.validator.add_validation_rule(
        'quantum.shots',
        lambda x: self.validator.validate_range(x, 1, 100000)
        )
        
        # Consciousness validation
        self.validator.add_validation_rule(
        'consciousness.consciousness_threshold',
        lambda x: self.validator.validate_range(x, 0.0, 1.0)
        )
        
        # TDA validation
        self.validator.add_validation_rule(
        'tda.max_dimension',
        lambda x: self.validator.validate_range(x, 0, 10)
        )


    # Global configuration manager instance
        _config_manager: Optional[ConfigurationManager] = None


    def get_config_manager() -> ConfigurationManager:
        """Get the global configuration manager instance."""
        global _config_manager
        if _config_manager is None:
        _config_manager = ConfigurationManager()
        return _config_manager


    def get_config() -> AuraConfig:
        """Get the current AURA configuration."""
        return get_config_manager().get_config()


    def reload_config(config_files: Optional[List[Union[str, Path]]] = None) -> AuraConfig:
        """Reload the AURA configuration."""
        return get_config_manager().reload_config(config_files)


    # Configuration factory functions
    def create_development_config() -> AuraConfig:
        """Create a development configuration."""
        return AuraConfig(
        environment=Environment.DEVELOPMENT,
        debug=True,
        secret_key="dev-secret-key-32-characters-long",
        jwt_secret_key="dev-jwt-secret-key-32-characters-long"
        )


    def create_testing_config() -> AuraConfig:
        """Create a testing configuration."""
        return AuraConfig(
        environment=Environment.TESTING,
        testing=True,
        debug=True,
        secret_key="test-secret-key-32-characters-long",
        jwt_secret_key="test-jwt-secret-key-32-characters-long"
        )


    def create_production_config() -> AuraConfig:
        """Create a production configuration."""
        return AuraConfig(
        environment=Environment.PRODUCTION,
        debug=False,
    # Production secrets should come from environment variables
        secret_key=os.getenv("AURA_SECRET_KEY", "change-me-in-production-32-chars"),
        jwt_secret_key=os.getenv("AURA_JWT_SECRET_KEY", "change-me-in-production-32-chars")
        )
