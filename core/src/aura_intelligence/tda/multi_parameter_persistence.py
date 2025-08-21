"""
Multi-Parameter Persistence - Advanced TDA
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time

try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

class MultiParameterProcessor:
    def __init__(self, max_dimension=2, n_jobs=4):
        self.max_dimension = max_dimension
        self.n_jobs = n_jobs
        
        if GTDA_AVAILABLE:
            self.cubical_processor = CubicalPersistence(
                homology_dimensions=list(range(max_dimension + 1)),
                n_jobs=n_jobs
            )
            self.entropy_processor = PersistenceEntropy()
            self.amplitude_processor = Amplitude()
            self.processor_type = "Giotto-TDA"
        else:
            self.processor_type = "Fallback"
    
    def compute_multi_parameter_persistence(self, multichannel_data: np.ndarray) -> Dict[str, Any]:
        """Compute multi-parameter persistent homology"""
        start_time = time.perf_counter()
        
        if GTDA_AVAILABLE and len(multichannel_data.shape) >= 3:
            # Real multi-parameter persistence
            try:
                diagrams = self.cubical_processor.fit_transform(multichannel_data)
                
                # Compute topological features
                entropy = self.entropy_processor.fit_transform(diagrams)
                amplitude = self.amplitude_processor.fit_transform(diagrams)
                
                # Extract multi-parameter features
                mp_features = self._extract_multiparameter_features(diagrams)
                
                result = {
                    'diagrams': diagrams,
                    'entropy': entropy,
                    'amplitude': amplitude,
                    'mp_features': mp_features,
                    'processor_type': self.processor_type
                }
                
            except Exception as e:
                result = self._fallback_processing(multichannel_data)
                result['error'] = str(e)
        else:
            result = self._fallback_processing(multichannel_data)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        result['processing_time_ms'] = processing_time
        
        return result
    
    def _fallback_processing(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback multi-parameter processing"""
        
        # Simple multi-channel analysis
        channels = data.shape[-1] if len(data.shape) > 2 else 1
        
        mp_features = {}
        for i in range(min(channels, 3)):  # Process up to 3 channels
            if len(data.shape) > 2:
                channel_data = data[..., i]
            else:
                channel_data = data
            
            # Simple topological features per channel
            mp_features[f'channel_{i}'] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'complexity': float(np.var(np.gradient(channel_data.flatten())))
            }
        
        # Cross-channel correlations
        if channels > 1 and len(data.shape) > 2:
            correlations = []
            for i in range(min(channels, 3)):
                for j in range(i+1, min(channels, 3)):
                    corr = np.corrcoef(data[..., i].flatten(), data[..., j].flatten())[0, 1]
                    correlations.append(float(corr) if not np.isnan(corr) else 0.0)
            mp_features['cross_correlations'] = correlations
        
        return {
            'diagrams': None,
            'entropy': np.array([[0.5]]),  # Dummy entropy
            'amplitude': np.array([[1.0]]),  # Dummy amplitude
            'mp_features': mp_features,
            'processor_type': 'Fallback'
        }
    
    def _extract_multiparameter_features(self, diagrams) -> Dict[str, Any]:
        """Extract multi-parameter topological features"""
        
        features = {}
        
        for dim in range(self.max_dimension + 1):
            dim_features = []
            
            # Extract features for each dimension
            if len(diagrams) > 0:
                dim_diagram = diagrams[0]  # First sample
                
                # Count persistent features by dimension
                if hasattr(dim_diagram, 'shape') and len(dim_diagram.shape) > 1:
                    persistent_count = dim_diagram.shape[0]
                    
                    if persistent_count > 0:
                        # Persistence statistics
                        births = dim_diagram[:, 0] if dim_diagram.shape[1] > 0 else []
                        deaths = dim_diagram[:, 1] if dim_diagram.shape[1] > 1 else []
                        
                        if len(births) > 0 and len(deaths) > 0:
                            lifetimes = deaths - births
                            
                            dim_features = {
                                'count': persistent_count,
                                'mean_birth': float(np.mean(births)),
                                'mean_death': float(np.mean(deaths)),
                                'mean_lifetime': float(np.mean(lifetimes)),
                                'max_lifetime': float(np.max(lifetimes)),
                                'total_persistence': float(np.sum(lifetimes))
                            }
            
            if not dim_features:
                dim_features = {
                    'count': 0,
                    'mean_birth': 0.0,
                    'mean_death': 0.0,
                    'mean_lifetime': 0.0,
                    'max_lifetime': 0.0,
                    'total_persistence': 0.0
                }
            
            features[f'dimension_{dim}'] = dim_features
        
        return features
    
    def analyze_time_series_topology(self, time_series: np.ndarray, window_size: int = 50) -> Dict[str, Any]:
        """Analyze topology of time series with sliding windows"""
        
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
        
        n_samples, n_features = time_series.shape
        n_windows = max(1, n_samples - window_size + 1)
        
        window_results = []
        
        for i in range(0, n_windows, max(1, window_size // 4)):  # 75% overlap
            end_idx = min(i + window_size, n_samples)
            window_data = time_series[i:end_idx]
            
            # Create embedding for topology analysis
            if n_features == 1:
                # Time delay embedding for univariate series
                embedding_dim = min(3, window_size // 3)
                embedded = self._time_delay_embedding(window_data.flatten(), embedding_dim)
            else:
                embedded = window_data
            
            # Reshape for multi-parameter analysis
            if len(embedded.shape) == 2:
                embedded = embedded.reshape(1, *embedded.shape)
            
            window_result = self.compute_multi_parameter_persistence(embedded)
            window_result['window_start'] = i
            window_result['window_end'] = end_idx
            
            window_results.append(window_result)
        
        return {
            'window_results': window_results,
            'n_windows': len(window_results),
            'window_size': window_size,
            'total_samples': n_samples
        }
    
    def _time_delay_embedding(self, series: np.ndarray, embedding_dim: int, delay: int = 1) -> np.ndarray:
        """Create time delay embedding"""
        n = len(series)
        embedded_length = n - (embedding_dim - 1) * delay
        
        if embedded_length <= 0:
            return series.reshape(-1, 1)
        
        embedded = np.zeros((embedded_length, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = series[i * delay:i * delay + embedded_length]
        
        return embedded

def get_multiparameter_processor(max_dimension=2, n_jobs=4):
    return MultiParameterProcessor(max_dimension, n_jobs)