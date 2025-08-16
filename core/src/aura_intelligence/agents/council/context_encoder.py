"""
Context Encoder for LNN Council Agent (2025 Architecture)

Advanced feature engineering with domain-specific encodings.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import structlog

from .config import LNNCouncilConfig
from .models import GPUAllocationRequest

logger = structlog.get_logger()


class ContextEncoder:
    """
    Context encoder for GPU allocation requests.
    
    2025 Features:
    - Domain-specific feature engineering
    - Hierarchical encoding
    - Temporal features
    - Categorical embeddings
    """
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        
        # GPU type embeddings (learned)
        self.gpu_embeddings = {
            "A100": torch.tensor([1.0, 0.9, 0.95, 0.8]),
            "H100": torch.tensor([1.0, 1.0, 1.0, 0.9]),
            "V100": torch.tensor([0.8, 0.7, 0.8, 0.7]),
            "RTX4090": torch.tensor([0.7, 0.6, 0.7, 0.9]),
            "RTX3090": torch.tensor([0.6, 0.5, 0.6, 0.8])
        }
        
        logger.info("Context encoder initialized")
    
    def encode_request(self, request: GPUAllocationRequest) -> torch.Tensor:
        """
        Encode GPU allocation request into feature vector.
        
        Args:
            request: GPU allocation request
            
        Returns:
            Feature tensor of shape [1, input_size]
        """
        
        features = []
        
        # 1. GPU specifications (normalized)
        gpu_embedding = self.gpu_embeddings.get(
            request.gpu_type, 
            torch.tensor([0.5, 0.5, 0.5, 0.5])
        )
        features.extend(gpu_embedding.tolist())
        
        # 2. Resource requirements (normalized)
        features.extend([
            request.gpu_count / 8.0,  # Max 8 GPUs
            request.memory_gb / 80.0,  # Max 80GB per GPU
            request.compute_hours / 168.0,  # Max 1 week
            request.priority / 10.0  # Priority 1-10
        ])
        
        # 3. Special requirements (binary encoding)
        special_reqs = request.special_requirements or []
        req_features = [
            float("high_memory" in special_reqs),
            float("low_latency" in special_reqs),
            float("multi_gpu" in special_reqs),
            float("distributed" in special_reqs),
            float("inference_only" in special_reqs)
        ]
        features.extend(req_features)
        
        # 4. Hardware requirements (binary)
        features.extend([
            float(request.requires_infiniband),
            float(request.requires_nvlink)
        ])
        
        # 5. Temporal features (time-based)
        temporal_features = self._encode_temporal_features(request)
        features.extend(temporal_features)
        
        # 6. Derived features (domain knowledge)
        derived_features = self._encode_derived_features(request)
        features.extend(derived_features)
        
        # 7. Pad or truncate to input_size
        features = self._pad_features(features)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _encode_temporal_features(self, request: GPUAllocationRequest) -> list:
        """Encode temporal aspects of the request."""
        
        # Time of day (cyclical encoding)
        hour = request.created_at.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (cyclical encoding)
        day = request.created_at.weekday()
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        # Urgency (based on deadline if available)
        urgency = 0.5  # Default
        if hasattr(request, 'deadline') and request.deadline:
            time_to_deadline = (request.deadline - request.created_at).total_seconds()
            urgency = max(0.0, min(1.0, 1.0 - time_to_deadline / (7 * 24 * 3600)))
        
        return [hour_sin, hour_cos, day_sin, day_cos, urgency]
    
    def _encode_derived_features(self, request: GPUAllocationRequest) -> list:
        """Encode derived features based on domain knowledge."""
        
        # Resource intensity
        total_gpu_hours = request.gpu_count * request.compute_hours
        memory_intensity = request.memory_gb * request.gpu_count
        
        # Complexity score
        complexity_factors = [
            request.gpu_count / 8.0,
            len(request.special_requirements or []) / 5.0,
            float(request.requires_infiniband or request.requires_nvlink)
        ]
        complexity = np.mean(complexity_factors)
        
        # Cost estimate (normalized)
        estimated_cost = total_gpu_hours * 2.5  # $2.5/GPU-hour
        cost_normalized = min(estimated_cost / 10000.0, 1.0)  # Cap at $10k
        
        # Priority-adjusted urgency
        priority_weight = request.priority / 10.0
        adjusted_urgency = priority_weight * 0.5  # Base urgency
        
        return [
            total_gpu_hours / 1000.0,  # Normalize
            memory_intensity / 640.0,  # Max 8 * 80GB
            complexity,
            cost_normalized,
            adjusted_urgency
        ]
    
    def _pad_features(self, features: list) -> list:
        """Pad or truncate features to match input_size."""
        
        target_size = self.config.input_size
        
        if len(features) < target_size:
            # Pad with zeros
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            # Truncate
            features = features[:target_size]
        
        return features
    
    def get_feature_names(self) -> list:
        """Get names of encoded features for interpretability."""
        
        names = [
            # GPU embeddings
            "gpu_compute", "gpu_memory", "gpu_efficiency", "gpu_availability",
            
            # Resource requirements
            "gpu_count_norm", "memory_gb_norm", "compute_hours_norm", "priority_norm",
            
            # Special requirements
            "req_high_memory", "req_low_latency", "req_multi_gpu", 
            "req_distributed", "req_inference_only",
            
            # Hardware requirements
            "requires_infiniband", "requires_nvlink",
            
            # Temporal features
            "hour_sin", "hour_cos", "day_sin", "day_cos", "urgency",
            
            # Derived features
            "total_gpu_hours_norm", "memory_intensity_norm", 
            "complexity", "cost_normalized", "priority_urgency"
        ]
        
        # Add padding feature names if needed
        while len(names) < self.config.input_size:
            names.append(f"padding_{len(names)}")
        
        return names[:self.config.input_size]