#!/usr/bin/env python3
"""
Unified Configuration System
Consolidates all scattered configuration files into a single, type-safe system
"""

import os
import json
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid

# ============================================================================
# CONFIGURATION ENUMS
# ============================================================================

class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class StorageBackend(Enum):
    """Storage backend types."""
    REDIS = "redis"
    MEMORY = "memory"
    FAISS = "faiss"
    WEAVIATE = "weaviate"
    NEO4J = "neo4j"
    DUCKDB = "duckdb"

class ActivationType(Enum):
    """Neural network activation types."""
    LIQUID = "liquid"
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"

class SolverType(Enum):
    """ODE solver types."""
    EULER = "euler"
    RK4 = "rk4"
    ADAPTIVE = "adaptive"

# ============================================================================
# CORE CONFIGURATION CLASSES
# ============================================================================

@dataclass
class SystemConfig:
    """Core system configuration."""
    environment: Environment = Environment.DEVELOPMENT
    service_name: str = "aura-intelligence"
    service_version: str = "1.0.0"
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    max_workers: int = 4
    timeout_seconds: int = 30
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create from environment variables."""
        pass
        return cls(
            environment=Environment(os.getenv("ENVIRONMENT", "development")),
            service_name=os.getenv("SERVICE_NAME", "aura-intelligence"),
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30"))
        )

@dataclass
class DatabaseConfig:
    """Database configuration."""
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # DuckDB
    duckdb_path: str = ":memory:"
    duckdb_threads: int = 4
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create from environment variables."""
        pass
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
            redis_socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            duckdb_path=os.getenv("DUCKDB_PATH", ":memory:"),
            duckdb_threads=int(os.getenv("DUCKDB_THREADS", "4"))
        )

@dataclass
class AgentConfig:
    """Unified agent configuration."""
    # General agent settings
    max_agents: int = 7
    cycle_interval: float = 1.0
    max_cycles: int = 100
    enable_consciousness: bool = True
    
    # Council agent settings
    council_input_size: int = 128
    council_output_size: int = 32
    council_hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    council_confidence_threshold: float = 0.7
    council_max_inference_time: float = 2.0
    council_enable_fallback: bool = True
    
    # Bio agent settings
    bio_enable_metabolism: bool = True
    bio_enable_communication: bool = True
    bio_enable_swarm: bool = True
    bio_population_size: int = 10
    
    # Neural settings
    activation_type: ActivationType = ActivationType.LIQUID
    solver_type: SolverType = SolverType.RK4
    dt: float = 0.01
    use_gpu: bool = False
    batch_size: int = 16
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create from environment variables."""
        pass
        return cls(
            max_agents=int(os.getenv("MAX_AGENTS", "7")),
            cycle_interval=float(os.getenv("CYCLE_INTERVAL", "1.0")),
            max_cycles=int(os.getenv("MAX_CYCLES", "100")),
            enable_consciousness=os.getenv("ENABLE_CONSCIOUSNESS", "true").lower() == "true",
            council_input_size=int(os.getenv("COUNCIL_INPUT_SIZE", "128")),
            council_output_size=int(os.getenv("COUNCIL_OUTPUT_SIZE", "32")),
            council_confidence_threshold=float(os.getenv("COUNCIL_CONFIDENCE_THRESHOLD", "0.7")),
            council_max_inference_time=float(os.getenv("COUNCIL_MAX_INFERENCE_TIME", "2.0")),
            council_enable_fallback=os.getenv("COUNCIL_ENABLE_FALLBACK", "true").lower() == "true",
            bio_enable_metabolism=os.getenv("BIO_ENABLE_METABOLISM", "true").lower() == "true",
            bio_enable_communication=os.getenv("BIO_ENABLE_COMMUNICATION", "true").lower() == "true",
            bio_enable_swarm=os.getenv("BIO_ENABLE_SWARM", "true").lower() == "true",
            bio_population_size=int(os.getenv("BIO_POPULATION_SIZE", "10")),
            use_gpu=os.getenv("USE_GPU", "false").lower() == "true",
            batch_size=int(os.getenv("BATCH_SIZE", "16"))
        )

@dataclass
class MemoryConfig:
    """Unified memory configuration."""
    # General memory settings
    storage_backend: StorageBackend = StorageBackend.REDIS
    embedding_dim: int = 128
    max_memory_size_mb: int = 1024
    cleanup_interval_hours: int = 24
    
    # Shape memory settings
    enable_fusion_scoring: bool = True
    fusion_alpha: float = 0.7
    fusion_beta: float = 0.3
    fusion_tau_hours: float = 168.0
    fastrp_iterations: int = 3
    
    # Performance settings
    max_concurrent_operations: int = 100
    operation_timeout: float = 5.0
    metrics_update_interval: int = 30
    
    # Circuit breaker settings
    circuit_breaker_fail_max: int = 5
    circuit_breaker_reset_timeout: int = 30
    
    # Retry settings
    retry_max_attempts: int = 3
    retry_wait_min: float = 0.1
    retry_wait_max: float = 2.0
    
    # ETL settings
    etl_batch_size: int = 1000
    etl_similarity_threshold: float = 0.8
    etl_max_similar_edges: int = 10
    
    @classmethod
    def from_env(cls) -> 'MemoryConfig':
        """Create from environment variables."""
        pass
        return cls(
            storage_backend=StorageBackend(os.getenv("MEMORY_STORAGE_BACKEND", "redis")),
            embedding_dim=int(os.getenv("MEMORY_EMBEDDING_DIM", "128")),
            max_memory_size_mb=int(os.getenv("MEMORY_MAX_SIZE_MB", "1024")),
            cleanup_interval_hours=int(os.getenv("MEMORY_CLEANUP_INTERVAL_HOURS", "24")),
            enable_fusion_scoring=os.getenv("MEMORY_ENABLE_FUSION_SCORING", "true").lower() == "true",
            fusion_alpha=float(os.getenv("MEMORY_FUSION_ALPHA", "0.7")),
            fusion_beta=float(os.getenv("MEMORY_FUSION_BETA", "0.3")),
            fusion_tau_hours=float(os.getenv("MEMORY_FUSION_TAU_HOURS", "168.0")),
            fastrp_iterations=int(os.getenv("MEMORY_FASTRP_ITERATIONS", "3")),
            max_concurrent_operations=int(os.getenv("MEMORY_MAX_CONCURRENT_OPS", "100")),
            operation_timeout=float(os.getenv("MEMORY_OPERATION_TIMEOUT", "5.0")),
            circuit_breaker_fail_max=int(os.getenv("MEMORY_CIRCUIT_BREAKER_FAIL_MAX", "5")),
            retry_max_attempts=int(os.getenv("MEMORY_RETRY_MAX_ATTEMPTS", "3"))
        )

@dataclass
class ObservabilityConfig:
    """Unified observability configuration."""
    # Core observability
    organism_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    organism_generation: int = 1
    deployment_environment: str = "production"
    service_version: str = "2025.8.12"
    
    # LangSmith configuration
    langsmith_api_key: str = ""
    langsmith_project: str = "aura-intelligence"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_enable_streaming: bool = True
    langsmith_enable_evaluation: bool = True
    langsmith_batch_size: int = 10
    
    # OpenTelemetry configuration
    otel_service_name: str = "aura-intelligence"
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_api_key: str = ""
    otel_enable_auto_instrumentation: bool = True
    otel_batch_size: int = 512
    otel_export_timeout: int = 30000
    
    # Prometheus configuration
    prometheus_port: int = 8000
    prometheus_enable_multiprocess: bool = True
    prometheus_registry_path: str = "/tmp/prometheus_multiproc"
    
    # Logging configuration
    log_format: str = "json"
    log_enable_correlation: bool = True
    log_enable_crypto_signatures: bool = True
    
    # Advanced features
    enable_cost_tracking: bool = True
    enable_anomaly_detection: bool = True
    enable_performance_profiling: bool = True
    enable_cryptographic_audit: bool = True
    enable_real_time_streaming: bool = False
    enable_auto_recovery: bool = True
    
    # Health monitoring
    health_check_interval: int = 30
    health_score_threshold: float = 0.7
    
    @classmethod
    def from_env(cls) -> 'ObservabilityConfig':
        """Create from environment variables."""
        pass
        return cls(
            organism_id=os.getenv("ORGANISM_ID", str(uuid.uuid4())),
            organism_generation=int(os.getenv("ORGANISM_GENERATION", "1")),
            deployment_environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
            service_version=os.getenv("SERVICE_VERSION", "2025.8.12"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "aura-intelligence"),
            langsmith_enable_streaming=os.getenv("LANGSMITH_STREAMING", "true").lower() == "true",
            otel_service_name=os.getenv("OTEL_SERVICE_NAME", "aura-intelligence"),
            otel_exporter_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            otel_api_key=os.getenv("OTEL_API_KEY", ""),
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
            enable_cost_tracking=os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true",
            enable_anomaly_detection=os.getenv("ENABLE_ANOMALY_DETECTION", "true").lower() == "true",
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        )

@dataclass
class NeuralConfig:
    """Unified neural network configuration."""
    # General neural settings
    enable_lnn: bool = True
    enable_bio_neural: bool = True
    enable_topological: bool = True
    
    # LNN settings
    lnn_time_constants_tau_min: float = 0.1
    lnn_time_constants_tau_max: float = 10.0
    lnn_wiring_sparsity: float = 0.8
    lnn_solver_dt: float = 0.01
    
    # Bio neural settings
    bio_neural_enable_metabolism: bool = True
    bio_neural_enable_adaptation: bool = True
    bio_neural_mutation_rate: float = 0.01
    
    # Topological settings
    tda_enable_persistent_homology: bool = True
    tda_enable_sheaf_theory: bool = True
    tda_max_dimension: int = 3
    
    # Performance settings
    use_cuda: bool = False
    mixed_precision: bool = False
    gradient_clipping: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'NeuralConfig':
        """Create from environment variables."""
        pass
        return cls(
            enable_lnn=os.getenv("NEURAL_ENABLE_LNN", "true").lower() == "true",
            enable_bio_neural=os.getenv("NEURAL_ENABLE_BIO", "true").lower() == "true",
            enable_topological=os.getenv("NEURAL_ENABLE_TOPOLOGICAL", "true").lower() == "true",
            lnn_time_constants_tau_min=float(os.getenv("LNN_TAU_MIN", "0.1")),
            lnn_time_constants_tau_max=float(os.getenv("LNN_TAU_MAX", "10.0")),
            lnn_wiring_sparsity=float(os.getenv("LNN_WIRING_SPARSITY", "0.8")),
            bio_neural_mutation_rate=float(os.getenv("BIO_NEURAL_MUTATION_RATE", "0.01")),
            tda_max_dimension=int(os.getenv("TDA_MAX_DIMENSION", "3")),
            use_cuda=os.getenv("NEURAL_USE_CUDA", "false").lower() == "true",
            mixed_precision=os.getenv("NEURAL_MIXED_PRECISION", "false").lower() == "true",
            gradient_clipping=float(os.getenv("NEURAL_GRADIENT_CLIPPING", "1.0"))
        )

@dataclass
class OrchestrationConfig:
    """Unified orchestration configuration."""
    # General orchestration
    enable_semantic_orchestration: bool = True
    enable_event_driven: bool = True
    enable_workflows: bool = True
    
    # Workflow settings
    max_concurrent_workflows: int = 10
    workflow_timeout_minutes: int = 60
    enable_workflow_persistence: bool = True
    
    # Event settings
    event_buffer_size: int = 1000
    event_batch_size: int = 100
    event_processing_interval: float = 1.0
    
    # Semantic settings
    semantic_similarity_threshold: float = 0.8
    semantic_max_context_length: int = 4096
    
    @classmethod
    def from_env(cls) -> 'OrchestrationConfig':
        """Create from environment variables."""
        pass
        return cls(
            enable_semantic_orchestration=os.getenv("ORCHESTRATION_ENABLE_SEMANTIC", "true").lower() == "true",
            enable_event_driven=os.getenv("ORCHESTRATION_ENABLE_EVENT_DRIVEN", "true").lower() == "true",
            enable_workflows=os.getenv("ORCHESTRATION_ENABLE_WORKFLOWS", "true").lower() == "true",
            max_concurrent_workflows=int(os.getenv("ORCHESTRATION_MAX_CONCURRENT_WORKFLOWS", "10")),
            workflow_timeout_minutes=int(os.getenv("ORCHESTRATION_WORKFLOW_TIMEOUT_MINUTES", "60")),
            event_buffer_size=int(os.getenv("ORCHESTRATION_EVENT_BUFFER_SIZE", "1000")),
            semantic_similarity_threshold=float(os.getenv("ORCHESTRATION_SEMANTIC_SIMILARITY_THRESHOLD", "0.8"))
        )

# ============================================================================
# UNIFIED CONFIGURATION CLASS
# ============================================================================

@dataclass
class UnifiedConfig:
    """
    Unified configuration for the entire AURA Intelligence system.
    
    This replaces all scattered configuration files with a single,
    type-safe, environment-aware configuration system.
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    
    @classmethod
    def from_env(cls) -> 'UnifiedConfig':
        """Create unified configuration from environment variables."""
        pass
        return cls(
            system=SystemConfig.from_env(),
            database=DatabaseConfig.from_env(),
            agents=AgentConfig.from_env(),
            memory=MemoryConfig.from_env(),
            observability=ObservabilityConfig.from_env(),
            neural=NeuralConfig.from_env(),
            orchestration=OrchestrationConfig.from_env()
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'UnifiedConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedConfig':
        """Create configuration from dictionary."""
        return cls(
            system=SystemConfig(**config_dict.get('system', {})),
            database=DatabaseConfig(**config_dict.get('database', {})),
            agents=AgentConfig(**config_dict.get('agents', {})),
            memory=MemoryConfig(**config_dict.get('memory', {})),
            observability=ObservabilityConfig(**config_dict.get('observability', {})),
            neural=NeuralConfig(**config_dict.get('neural', {})),
            orchestration=OrchestrationConfig(**config_dict.get('orchestration', {}))
        )
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        
        if not include_secrets:
            # Remove sensitive information
            sensitive_fields = [
                'redis_password', 'neo4j_password', 'langsmith_api_key', 
                'otel_api_key', 'organism_id'
            ]
            
    def remove_sensitive(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {
                        k: remove_sensitive(v) for k, v in obj.items()
                        if k not in sensitive_fields
                    }
                elif isinstance(obj, list):
                    return [remove_sensitive(item) for item in obj]
                else:
                    return obj
            
            config_dict = remove_sensitive(config_dict)
        
        return config_dict
    
    def save_to_file(self, config_path: Union[str, Path], include_secrets: bool = False) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict(include_secrets=include_secrets)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        pass
        issues = []
        
        # Validate system configuration
        if self.system.max_workers <= 0:
            issues.append("System max_workers must be positive")
        
        if self.system.timeout_seconds <= 0:
            issues.append("System timeout_seconds must be positive")
        
        # Validate agent configuration
        if self.agents.max_agents <= 0:
            issues.append("Agents max_agents must be positive")
        
        if not (0.0 <= self.agents.council_confidence_threshold <= 1.0):
            issues.append("Council confidence_threshold must be between 0.0 and 1.0")
        
        # Validate memory configuration
        if self.memory.embedding_dim <= 0:
            issues.append("Memory embedding_dim must be positive")
        
        if not (0.0 <= self.memory.fusion_alpha <= 1.0):
            issues.append("Memory fusion_alpha must be between 0.0 and 1.0")
        
        if not (0.0 <= self.memory.fusion_beta <= 1.0):
            issues.append("Memory fusion_beta must be between 0.0 and 1.0")
        
        # Check fusion weights sum to 1.0
        if abs(self.memory.fusion_alpha + self.memory.fusion_beta - 1.0) > 0.01:
            issues.append("Memory fusion_alpha + fusion_beta must equal 1.0")
        
        # Validate observability configuration
        if self.observability.prometheus_port <= 0 or self.observability.prometheus_port > 65535:
            issues.append("Observability prometheus_port must be between 1 and 65535")
        
        if not (0.0 <= self.observability.health_score_threshold <= 1.0):
            issues.append("Observability health_score_threshold must be between 0.0 and 1.0")
        
        # Validate neural configuration
        if self.neural.lnn_time_constants_tau_min <= 0:
            issues.append("Neural LNN tau_min must be positive")
        
        if self.neural.lnn_time_constants_tau_max <= self.neural.lnn_time_constants_tau_min:
            issues.append("Neural LNN tau_max must be greater than tau_min")
        
        # Validate orchestration configuration
        if self.orchestration.max_concurrent_workflows <= 0:
            issues.append("Orchestration max_concurrent_workflows must be positive")
        
        return issues
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        pass
        return self.system.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        pass
        return self.system.environment == Environment.DEVELOPMENT

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigurationManager:
    """
    Centralized configuration management with hot reloading,
    validation, and environment-specific overrides.
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self._config = config or UnifiedConfig.from_env()
        self._config_file_path: Optional[Path] = None
        self._watchers: List[Callable[[UnifiedConfig], None]] = []
        self._validation_enabled = True
    
    @property
    def config(self) -> UnifiedConfig:
        """Get current configuration."""
        pass
        return self._config
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file."""
        self._config_file_path = Path(config_path)
        self._config = UnifiedConfig.from_file(self._config_file_path)
        self._notify_watchers()
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        pass
        self._config = UnifiedConfig.from_env()
        self._notify_watchers()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        # Convert current config to dict
        config_dict = self._config.to_dict(include_secrets=True)
        
        # Apply updates
    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, updates)
        
        # Create new config from updated dict
        new_config = UnifiedConfig.from_dict(config_dict)
        
        # Validate if enabled
        if self._validation_enabled:
            issues = new_config.validate()
            if issues:
                raise ValueError(f"Configuration validation failed: {'; '.join(issues)}")
        
        self._config = new_config
        self._notify_watchers()
    
    def validate_config(self) -> List[str]:
        """Validate current configuration."""
        pass
        return self._config.validate()
    
    def add_config_watcher(self, callback: Callable[[UnifiedConfig], None]) -> None:
        """Add configuration change watcher."""
        self._watchers.append(callback)
    
    def remove_config_watcher(self, callback: Callable[[UnifiedConfig], None]) -> None:
        """Remove configuration change watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def enable_validation(self, enabled: bool = True) -> None:
        """Enable or disable configuration validation."""
        self._validation_enabled = enabled
    
    def save_current_config(self, config_path: Union[str, Path], include_secrets: bool = False) -> None:
        """Save current configuration to file."""
        self._config.save_to_file(config_path, include_secrets=include_secrets)
    
    def _notify_watchers(self) -> None:
        """Notify all configuration watchers."""
        pass
        for watcher in self._watchers:
            try:
                watcher(self._config)
            except Exception as e:
                print(f"Configuration watcher error: {e}")

# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Global configuration manager
_global_config_manager: Optional[ConfigurationManager] = None

    def get_config_manager() -> ConfigurationManager:
        """Get the global configuration manager."""
        global _global_config_manager
        if _global_config_manager is None:
        _global_config_manager = ConfigurationManager()
        return _global_config_manager

    def get_config() -> UnifiedConfig:
        """Get the current unified configuration."""
        return get_config_manager().config

    def update_config(updates: Dict[str, Any]) -> None:
        """Update the global configuration."""
        get_config_manager().update_config(updates)

    def load_config_from_file(config_path: Union[str, Path]) -> None:
        """Load configuration from file."""
        get_config_manager().load_from_file(config_path)
