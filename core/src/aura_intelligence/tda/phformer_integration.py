"""
PHFormer 2.0 Integration - Advanced Topology Transformers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import time

try:
    from phformer2 import PHFormer2
    PHFORMER_AVAILABLE = True
except ImportError:
    PHFORMER_AVAILABLE = False

class PHFormerProcessor:
    def __init__(self, model_size='base', device='cpu'):
        self.device = device
        
        if PHFORMER_AVAILABLE:
            self.model = PHFormer2.from_pretrained(f"phformer2-{model_size}")
            self.model_type = "PHFormer2"
        else:
            # Fallback transformer
            self.model = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(256, 8, batch_first=True), 
                    num_layers=4
                ),
                nn.AdaptiveAvgPool1d(128)
            )
            self.model_type = "Fallback"
        
        self.model.to(device).eval()
    
    def persistence_to_image(self, persistence_diagram: List[List[float]], resolution=32):
        if not persistence_diagram:
            return torch.zeros(1, resolution, resolution)
        
        image = np.zeros((resolution, resolution))
        for birth, death in persistence_diagram:
            if death == float('inf'):
                death = birth + 1.0
            
            x = int(birth * resolution / 2.0)
            y = int((death - birth) * resolution / 2.0)
            x, y = max(0, min(x, resolution-1)), max(0, min(y, resolution-1))
            
            sigma = resolution / 16.0
            for i in range(resolution):
                for j in range(resolution):
                    weight = np.exp(-((i-x)**2 + (j-y)**2) / (2*sigma**2))
                    image[i, j] += weight
        
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    
    def process_topology(self, betti_numbers: List[int], persistence_diagram: List[List[float]]):
        start_time = time.perf_counter()
        
        persistence_image = self.persistence_to_image(persistence_diagram).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(persistence_image.unsqueeze(0))
            else:
                # Fallback processing
                flat_image = persistence_image.flatten().unsqueeze(0)
                if flat_image.shape[1] != 1024:
                    flat_image = torch.nn.functional.adaptive_avg_pool1d(flat_image.unsqueeze(1), 1024).squeeze(1)
                features = self.model(flat_image.unsqueeze(1)).squeeze(1)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'topology_embeddings': features.cpu().numpy(),
            'persistence_image': persistence_image.cpu().numpy(),
            'processing_time_ms': processing_time,
            'model_type': self.model_type
        }

def get_phformer_processor(model_size='base', device='cpu'):
    return PHFormerProcessor(model_size, device)