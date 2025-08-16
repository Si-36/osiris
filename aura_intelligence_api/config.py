"""
AURA Intelligence Configuration
==============================

Configuration management for the AURA Intelligence system.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json

class AURAConfig:
    """Configuration manager for AURA Intelligence"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file"""
        possible_paths = [
            "config/aura_config.yaml",
            "config/config.yaml", 
            "aura_config.yaml",
            "config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # Return default path if none found
        return "config/aura_config.yaml"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        # Default configuration
        default_config = {
            "system": {
                "name": "AURA Intelligence",
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "timeout": 300,
                "max_request_size": 10 * 1024 * 1024  # 10MB
            },
            "neural": {
                "lnn_params": 5514,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "auto"  # auto, cpu, cuda
            },
            "memory": {
                "mem0_enabled": True,
                "vector_search_enabled": True,
                "redis_enabled": False,
                "neo4j_enabled": False,
                "max_memory_size": 1000000
            },
            "ai": {
                "gemini_enabled": True,
                "gemini_model": "gemini-pro",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "agents": {
                "langgraph_enabled": True,
                "council_enabled": True,
                "max_agents": 10,
                "timeout": 120
            },
            "consciousness": {
                "global_workspace_enabled": True,
                "attention_enabled": True,
                "decision_threshold": 0.7
            },
            "advanced": {
                "tda_enabled": True,
                "cuda_enabled": False,
                "streaming_enabled": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "aura_intelligence.log"
            },
            "security": {
                "api_key_required": False,
                "rate_limiting": True,
                "max_requests_per_minute": 100
            }
        }
        
        # Try to load from file
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                # Merge with defaults
                config = self._merge_configs(default_config, file_config)
                return config
                
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
        
        return default_config
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            else:
                json.dump(self.config, f, indent=2)
    
    def create_default_config(self):
        """Create default configuration file"""
        self.save()
        print(f"Created default configuration at: {self.config_path}")

# Global configuration instance
config = AURAConfig()

# Environment-specific overrides
if config.get('system.environment') == 'production':
    config.set('logging.level', 'WARNING')
    config.set('api.workers', 4)
    config.set('security.api_key_required', True)
elif config.get('system.environment') == 'development':
    config.set('logging.level', 'DEBUG')
    config.set('api.workers', 1)

# GPU detection
try:
    import torch
    if torch.cuda.is_available() and config.get('neural.device') == 'auto':
        config.set('neural.device', 'cuda')
        config.set('advanced.cuda_enabled', True)
    elif config.get('neural.device') == 'auto':
        config.set('neural.device', 'cpu')
except ImportError:
    config.set('neural.device', 'cpu')

def get_config() -> AURAConfig:
    """Get the global configuration instance"""
    return config