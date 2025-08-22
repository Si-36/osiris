"""
AURA Configuration Management
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AURAConfig:
    """Configuration for AURA Intelligence System"""
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("AURA_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("AURA_DEBUG", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("AURA_LOG_LEVEL", "INFO"))
    
    # API Keys
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    # Database Configuration
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "aura_db"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "aura_user"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    
    # Performance Settings
    max_agents: int = field(default_factory=lambda: int(os.getenv("MAX_AGENTS", "200")))
    max_connections_per_agent: int = field(default_factory=lambda: int(os.getenv("MAX_CONNECTIONS_PER_AGENT", "50")))
    topology_update_interval: int = field(default_factory=lambda: int(os.getenv("TOPOLOGY_UPDATE_INTERVAL", "100")))
    memory_cache_size: int = field(default_factory=lambda: int(os.getenv("MEMORY_CACHE_SIZE", "1000")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    
    # Feature Flags
    enable_quantum_tda: bool = field(default_factory=lambda: os.getenv("ENABLE_QUANTUM_TDA", "false").lower() == "true")
    enable_neuromorphic: bool = field(default_factory=lambda: os.getenv("ENABLE_NEUROMORPHIC", "true").lower() == "true")
    enable_byzantine_consensus: bool = field(default_factory=lambda: os.getenv("ENABLE_BYZANTINE_CONSENSUS", "true").lower() == "true")
    enable_shape_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_SHAPE_MEMORY", "true").lower() == "true")
    enable_liquid_neural_networks: bool = field(default_factory=lambda: os.getenv("ENABLE_LIQUID_NEURAL_NETWORKS", "true").lower() == "true")
    
    # GPU Configuration
    cuda_visible_devices: str = field(default_factory=lambda: os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    enable_gpu_acceleration: bool = field(default_factory=lambda: os.getenv("ENABLE_GPU_ACCELERATION", "true").lower() == "true")
    gpu_memory_fraction: float = field(default_factory=lambda: float(os.getenv("GPU_MEMORY_FRACTION", "0.8")))
    
    # Paths
    model_cache_dir: str = field(default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "./models"))
    lnn_model_path: str = field(default_factory=lambda: os.getenv("LNN_MODEL_PATH", "./models/lnn"))
    tda_cache_dir: str = field(default_factory=lambda: os.getenv("TDA_CACHE_DIR", "./cache/tda"))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))
    logs_dir: str = field(default_factory=lambda: os.getenv("LOGS_DIR", "./logs"))
    
    # API Configuration
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    enable_cors: bool = field(default_factory=lambda: os.getenv("ENABLE_CORS", "true").lower() == "true")
    api_workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "4")))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "max_agents": self.max_agents,
            "features": {
                "quantum_tda": self.enable_quantum_tda,
                "neuromorphic": self.enable_neuromorphic,
                "byzantine_consensus": self.enable_byzantine_consensus,
                "shape_memory": self.enable_shape_memory,
                "liquid_neural_networks": self.enable_liquid_neural_networks,
            },
            "gpu": {
                "enabled": self.enable_gpu_acceleration,
                "devices": self.cuda_visible_devices,
                "memory_fraction": self.gpu_memory_fraction,
            }
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        # Check required API keys
        if not self.langsmith_api_key:
            print("Warning: LANGSMITH_API_KEY not set")
        
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY not set")
        
        # Check paths exist
        for path in [self.model_cache_dir, self.tda_cache_dir, self.data_dir, self.logs_dir]:
            os.makedirs(path, exist_ok=True)
        
        return True