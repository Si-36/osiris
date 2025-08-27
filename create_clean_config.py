#!/usr/bin/env python3
"""
Create a clean, working unified_config.py
"""

config_content = '''"""
Unified Configuration for AURA Intelligence System
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SystemConfig:
    """System-wide configuration."""
    environment: Environment = Environment.DEVELOPMENT
    service_name: str = "aura-intelligence"
    service_version: str = "1.0.0"
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    max_workers: int = 4
    timeout_seconds: int = 30
    

@dataclass
class DatabaseConfig:
    """Database configuration."""
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


@dataclass
class UnifiedConfig:
    """Unified configuration for AURA Intelligence."""
    system: SystemConfig = field(default_factory=SystemConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    @classmethod
    def from_env(cls) -> 'UnifiedConfig':
        """Create configuration from environment variables."""
        return cls(
            system=SystemConfig(
                environment=Environment(os.getenv("ENVIRONMENT", "development")),
                service_name=os.getenv("SERVICE_NAME", "aura-intelligence"),
                service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
                log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
                debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
                max_workers=int(os.getenv("MAX_WORKERS", "4")),
                timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30"))
            ),
            database=DatabaseConfig(
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                redis_password=os.getenv("REDIS_PASSWORD"),
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.getenv("NEO4J_PASSWORD", "password")
            )
        )


# Global instance
_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = UnifiedConfig.from_env()
    return _config


def set_config(config: UnifiedConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
'''

# Create backup
import shutil
shutil.copy(
    'core/src/aura_intelligence/core/unified_config.py',
    'core/src/aura_intelligence/core/unified_config.py.backup'
)

# Write new clean config
with open('core/src/aura_intelligence/core/unified_config.py', 'w') as f:
    f.write(config_content)

print("✅ Created clean unified_config.py")
print("✅ Backup saved as unified_config.py.backup")