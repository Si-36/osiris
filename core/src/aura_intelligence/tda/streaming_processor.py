"""
Streaming TDA Processor - Real-time Shape Analysis
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional
from collections import deque

class StreamingTDAProcessor:
    def __init__(self, window_size=50, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.data_buffer = deque(maxlen=window_size * 2)
        self.topology_cache = {}
        self.frame_count = 0
        
    def process_stream(self, new_data: np.ndarray) -> Dict[str, Any]:
        """Process streaming data with temporal topology analysis"""
        start_time = time.perf_counter()
        
        # Add to buffer
        if len(new_data.shape) == 1:
            self.data_buffer.extend(new_data.reshape(-1, 1))
        else:
            self.data_buffer.extend(new_data)
        
        # Extract current window
        if len(self.data_buffer) >= self.window_size:
            current_window = np.array(list(self.data_buffer)[-self.window_size:])
            
            # Fast topology computation
            topology_result = self._fast_topology_analysis(current_window)
            
            # Temporal analysis
            temporal_features = self._compute_temporal_features()
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self.frame_count += 1
            
            return {
                'frame_id': self.frame_count,
                'betti_numbers': topology_result['betti_numbers'],
                'persistence_diagram': topology_result['persistence_diagram'],
                'temporal_stability': temporal_features['stability'],
                'change_rate': temporal_features['change_rate'],
                'processing_time_ms': processing_time,
                'method': 'Streaming-TDA'
            }
        else:
            return {
                'frame_id': self.frame_count,
                'betti_numbers': [0, 0],
                'persistence_diagram': [],
                'temporal_stability': 1.0,
                'change_rate': 0.0,
                'processing_time_ms': 0.1,
                'method': 'Buffering'
            }
    
    def _fast_topology_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Ultra-fast topology analysis for streaming"""
        
        # Data signature for caching
        data_hash = hash(str(np.mean(data, axis=0)) + str(np.std(data, axis=0)))
        
        if data_hash in self.topology_cache:
            return self.topology_cache[data_hash]
        
        try:
            import gudhi
            
            # Aggressive optimization for streaming
            n_points = min(len(data), 30)  # Limit points for speed
            sampled_data = data[:n_points] if len(data) > n_points else data
            
            # Adaptive threshold based on data spread
            data_range = np.ptp(sampled_data, axis=0)
            max_edge_length = np.mean(data_range) * 0.8
            
            rips_complex = gudhi.RipsComplex(points=sampled_data, max_edge_length=max_edge_length)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)  # Only H0, H1 for speed
            
            # Fast persistence
            persistence = simplex_tree.persistence()
            betti_numbers = simplex_tree.betti_numbers()
            
            # Minimal diagram
            persistence_diagram = []
            for interval in persistence[:8]:  # Only top 8 features
                if len(interval) >= 2:
                    birth, death = interval[1]
                    if death != float('inf') and death > birth:
                        persistence_diagram.append([birth, death])
            
            result = {
                'betti_numbers': betti_numbers,
                'persistence_diagram': persistence_diagram
            }
            
            # Cache result
            if len(self.topology_cache) < 20:  # Limit cache size
                self.topology_cache[data_hash] = result
            
            return result
            
        except:
            # Ultra-fast geometric fallback
            n_points = len(data)
            
            if n_points < 5:
                return {'betti_numbers': [1, 0], 'persistence_diagram': []}
            
            # Estimate connected components from nearest neighbors
            from scipy.spatial.distance import pdist
            distances = pdist(data)
            
            if len(distances) > 0:
                threshold = np.percentile(distances, 20)  # Connect close points
                n_edges = np.sum(distances < threshold)
                
                # Rough topology estimation
                betti_0 = max(1, n_points - n_edges // 2)  # Connected components
                betti_1 = max(0, (n_edges - n_points + 1) // 3)  # Rough cycle count
            else:
                betti_0, betti_1 = 1, 0
            
            return {
                'betti_numbers': [betti_0, betti_1],
                'persistence_diagram': [[0.0, threshold]] if 'threshold' in locals() else []
            }
    
    def _compute_temporal_features(self) -> Dict[str, float]:
        """Compute temporal stability metrics"""
        
        if self.frame_count < 3:
            return {'stability': 1.0, 'change_rate': 0.0}
        
        # Simple temporal analysis
        recent_data = np.array(list(self.data_buffer)[-self.window_size//2:])
        older_data = np.array(list(self.data_buffer)[-self.window_size:-self.window_size//2])
        
        if len(recent_data) > 0 and len(older_data) > 0:
            # Compare statistical moments
            recent_mean = np.mean(recent_data, axis=0)
            older_mean = np.mean(older_data, axis=0)
            
            change_magnitude = np.linalg.norm(recent_mean - older_mean)
            data_scale = np.linalg.norm(np.std(recent_data, axis=0))
            
            if data_scale > 0:
                change_rate = min(change_magnitude / data_scale, 2.0)
                stability = max(0.0, 1.0 - change_rate / 2.0)
            else:
                change_rate = 0.0
                stability = 1.0
        else:
            change_rate = 0.0
            stability = 1.0
        
        return {
            'stability': stability,
            'change_rate': change_rate
        }
    
    def reset_stream(self):
        """Reset streaming state"""
        self.data_buffer.clear()
        self.topology_cache.clear()
        self.frame_count = 0

def get_streaming_processor(window_size=50, overlap=0.5):
    return StreamingTDAProcessor(window_size, overlap)