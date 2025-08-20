"""
Shape Analysis Service
Extracts topological features for shape-aware indexing
"""

import numpy as np
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger()


class ShapeAnalysisService:
    """
    Analyzes data to extract topological features
    Used for shape-aware memory indexing
    """
    
    def __init__(self):
        self.logger = logger
        
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Extract topological features from data
        
        Args:
            data: Input data of any type
            
        Returns:
            Dictionary of topological features
        """
        try:
            # Convert to numpy array if possible
            if isinstance(data, (list, tuple)):
                array_data = np.array(data).flatten()
            elif isinstance(data, dict):
                # Extract numeric values from dict
                values = self._extract_numeric_values(data)
                array_data = np.array(values)
            elif isinstance(data, np.ndarray):
                array_data = data.flatten()
            else:
                # For non-numeric data, use hash-based features
                return self._hash_based_features(data)
            
            # Compute topological features
            features = {
                'betti_numbers': self._compute_betti_numbers(array_data),
                'persistence_diagram': self._compute_persistence(array_data),
                'wasserstein_distance': self._compute_wasserstein(array_data),
                'topological_signature': self._create_signature(array_data),
                'data_shape': array_data.shape,
                'data_type': type(data).__name__
            }
            
            return features
            
        except Exception as e:
            self.logger.warning("Shape analysis failed", error=str(e))
            return self._default_features()
    
    def _extract_numeric_values(self, data: Dict) -> List[float]:
        """Extract numeric values from dictionary"""
        values = []
        for v in data.values():
            if isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, (list, tuple)):
                values.extend(self._extract_numeric_values({'k': v})['k'])
        return values
    
    def _compute_betti_numbers(self, data: np.ndarray) -> List[int]:
        """
        Compute Betti numbers (simplified)
        b0: connected components
        b1: loops/cycles  
        b2: voids/cavities
        """
        # Simplified computation - real implementation would use persistent homology
        b0 = 1  # Assume connected
        
        # Detect cycles by looking for repeated patterns
        if len(data) > 3:
            diffs = np.diff(data)
            b1 = int(np.sum(np.abs(diffs[:-1] - diffs[1:]) < 0.1))
        else:
            b1 = 0
            
        # Higher dimensional features (simplified)
        b2 = 0
        if len(data) > 10:
            # Look for nested structures
            b2 = int(np.std(data) > np.mean(np.abs(data)))
            
        return [b0, b1, b2]
    
    def _compute_persistence(self, data: np.ndarray) -> List[List[float]]:
        """Compute persistence diagram (simplified)"""
        if len(data) < 2:
            return [[0, 1]]
            
        # Find local extrema as critical points
        persistence_pairs = []
        
        for i in range(1, len(data)-1):
            if data[i] > data[i-1] and data[i] > data[i+1]:  # Local max
                birth = float(i) / len(data)
                death = 1.0
                persistence_pairs.append([birth, death])
            elif data[i] < data[i-1] and data[i] < data[i+1]:  # Local min
                birth = 0.0
                death = float(i) / len(data)
                persistence_pairs.append([birth, death])
                
        if not persistence_pairs:
            persistence_pairs = [[0, 1]]
            
        return persistence_pairs[:10]  # Limit to 10 pairs
    
    def _compute_wasserstein(self, data: np.ndarray) -> float:
        """Compute Wasserstein distance to reference (simplified)"""
        # Distance from uniform distribution
        if len(data) == 0:
            return 0.0
            
        uniform = np.ones_like(data) * np.mean(data)
        return float(np.mean(np.abs(data - uniform)))
    
    def _create_signature(self, data: np.ndarray) -> np.ndarray:
        """Create topological signature vector"""
        signature = np.zeros(128, dtype=np.float32)
        
        # Basic statistics
        if len(data) > 0:
            signature[0] = np.mean(data)
            signature[1] = np.std(data)
            signature[2] = np.min(data)
            signature[3] = np.max(data)
            
            # Percentiles
            percentiles = np.percentile(data, [10, 25, 50, 75, 90])
            signature[4:9] = percentiles
            
            # Fourier components (if applicable)
            if len(data) > 10:
                fft = np.fft.fft(data)[:10]
                signature[10:20] = np.abs(fft)
                
        return signature
    
    def _hash_based_features(self, data: Any) -> Dict[str, Any]:
        """Create features for non-numeric data using hashing"""
        data_str = str(data)
        data_hash = hash(data_str)
        
        # Create pseudo-topological features from hash
        signature = np.zeros(128, dtype=np.float32)
        for i in range(128):
            signature[i] = ((data_hash >> i) & 1) * 0.5 + 0.25
            
        return {
            'betti_numbers': [1, 0, 0],
            'persistence_diagram': [[0, 1]],
            'wasserstein_distance': 0.5,
            'topological_signature': signature,
            'data_shape': (1,),
            'data_type': type(data).__name__
        }
    
    def _default_features(self) -> Dict[str, Any]:
        """Default features when analysis fails"""
        return {
            'betti_numbers': [1, 0, 0],
            'persistence_diagram': [[0, 1]],
            'wasserstein_distance': 0.0,
            'topological_signature': np.zeros(128, dtype=np.float32),
            'data_shape': (0,),
            'data_type': 'unknown'
        }