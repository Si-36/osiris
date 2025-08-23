"""
Model Registry Service for LNN
Manages multiple LNN models and their configurations
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import time
import pickle
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()


@dataclass
class ModelMetadata:
    """Metadata for registered models"""
    model_id: str
    implementation: str
    config: Any
    creation_time: float = field(default_factory=time.time)
    last_inference_time: float = 0.0
    total_inferences: int = 0
    total_adaptations: int = 0
    checkpoint_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ModelRegistryService:
    """
    Service for managing LNN models
    
    Features:
    - Model registration and discovery
    - Configuration management
    - Model persistence
    - Performance tracking
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.models: Dict[str, nn.Module] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.storage_path = storage_path or Path("/tmp/lnn_models")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(service="model_registry")
        
    def register_model(self,
                      model_id: str,
                      model: nn.Module,
                      config: Any,
                      tags: Optional[List[str]] = None) -> bool:
        """Register a new model"""
        try:
            if model_id in self.models:
                self.logger.warning("Model already exists", model_id=model_id)
                return False
            
            # Store model
            self.models[model_id] = model
            
            # Create metadata
            info = model.get_info() if hasattr(model, 'get_info') else {}
            self.metadata[model_id] = ModelMetadata(
                model_id=model_id,
                implementation=info.get("implementation", "unknown"),
                config=config,
                tags=tags or []
            )
            
            self.logger.info("Model registered", model_id=model_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to register model", error=str(e))
            return False
    
    def get_model(self, model_id: str) -> Optional[nn.Module]:
        """Get model by ID"""
        model = self.models.get(model_id)
        
        if model:
            # Update access time
            if model_id in self.metadata:
                self.metadata[model_id].last_inference_time = time.time()
                self.metadata[model_id].total_inferences += 1
        
        return model
    
    def get_config(self, model_id: str) -> Optional[Any]:
        """Get model configuration"""
        metadata = self.metadata.get(model_id)
        return metadata.config if metadata else None
    
    def get_all_models(self) -> Dict[str, nn.Module]:
        """Get all registered models"""
        return self.models.copy()
    
    def list_models(self, tags: Optional[List[str]] = None) -> List[str]:
        """List model IDs, optionally filtered by tags"""
        if tags:
            return [
                model_id for model_id, meta in self.metadata.items()
                if any(tag in meta.tags for tag in tags)
            ]
        return list(self.models.keys())
    
    def save_model(self, model_id: str, checkpoint_name: Optional[str] = None) -> Optional[str]:
        """Save model checkpoint"""
        try:
            model = self.models.get(model_id)
            metadata = self.metadata.get(model_id)
            
            if not model or not metadata:
                return None
            
            # Generate checkpoint name
            if not checkpoint_name:
                checkpoint_name = f"{model_id}_{int(time.time())}.pt"
            
            checkpoint_path = self.storage_path / checkpoint_name
            
            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            # Add optimizer state if available
            if hasattr(model, 'optimizer') and model.optimizer:
                checkpoint["optimizer_state_dict"] = model.optimizer.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            
            # Update metadata
            metadata.checkpoint_path = str(checkpoint_path)
            
            self.logger.info("Model saved", model_id=model_id, path=str(checkpoint_path))
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error("Failed to save model", error=str(e))
            return None
    
    def load_model(self, 
                  model_id: str,
                  checkpoint_path: str,
                  model_class: Optional[type] = None) -> bool:
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model instance if class provided
            if model_class:
                config = checkpoint["metadata"].config
                model = model_class(config)
                model.load_state_dict(checkpoint["model_state_dict"])
                
                # Register loaded model
                self.register_model(
                    model_id,
                    model,
                    config,
                    checkpoint["metadata"].tags
                )
                
                self.logger.info("Model loaded", model_id=model_id)
                return True
            else:
                self.logger.error("Model class not provided")
                return False
                
        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """Remove model from registry"""
        try:
            if model_id in self.models:
                del self.models[model_id]
                del self.metadata[model_id]
                self.logger.info("Model removed", model_id=model_id)
                return True
            return False
            
        except Exception as e:
            self.logger.error("Failed to remove model", error=str(e))
            return False
    
    def update_metadata(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update model metadata"""
        if model_id in self.metadata:
            metadata = self.metadata[model_id]
            
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            return True
        return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_params = 0
        total_inferences = 0
        total_adaptations = 0
        
        for model_id, model in self.models.items():
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            total_params += params
            
            # Get metadata stats
            if model_id in self.metadata:
                meta = self.metadata[model_id]
                total_inferences += meta.total_inferences
                
                # Get adaptation count from model
                if hasattr(model, 'get_info'):
                    info = model.get_info()
                    total_adaptations += info.get('total_adaptations', 0)
        
        return {
            "total_models": len(self.models),
            "total_parameters": total_params,
            "total_inferences": total_inferences,
            "total_adaptations": total_adaptations,
            "models_by_implementation": self._count_by_implementation(),
            "storage_used_mb": self._calculate_storage_usage()
        }
    
    def _count_by_implementation(self) -> Dict[str, int]:
        """Count models by implementation type"""
        counts = {}
        for meta in self.metadata.values():
            impl = meta.implementation
            counts[impl] = counts.get(impl, 0) + 1
        return counts
    
    def _calculate_storage_usage(self) -> float:
        """Calculate storage usage in MB"""
        total_size = 0
        
        for checkpoint_file in self.storage_path.glob("*.pt"):
            total_size += checkpoint_file.stat().st_size
        
        return total_size / (1024 * 1024)
    
    def export_registry(self, export_path: str) -> bool:
        """Export registry metadata"""
        try:
            export_data = {
                "models": {
                    model_id: {
                        "implementation": meta.implementation,
                        "creation_time": meta.creation_time,
                        "total_inferences": meta.total_inferences,
                        "tags": meta.tags
                    }
                    for model_id, meta in self.metadata.items()
                },
                "stats": self.get_model_stats(),
                "export_time": time.time()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to export registry", error=str(e))
            return False