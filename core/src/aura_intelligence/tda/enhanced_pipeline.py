"""
Enhanced TDA Pipeline - PHFormer + Multi-Parameter + Streaming
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional

class EnhancedTDAPipeline:
    def __init__(self, device='auto'):
        self.device = device
        self.frame_count = 0
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components"""
        pass
        try:
            from .phformer_integration import get_phformer_processor
            from .multi_parameter_persistence import get_multiparameter_processor
            from .streaming_processor import get_streaming_processor
            from .gpu_acceleration import get_gpu_accelerator
            
            self.phformer = get_phformer_processor('base', self.device)
            self.mp_processor = get_multiparameter_processor(max_dimension=2)
            self.streaming = get_streaming_processor(window_size=40)
            self.gpu_accel = get_gpu_accelerator(self.device)
            
            self.components_loaded = True
            
        except ImportError as e:
            self.components_loaded = False
            self.fallback_reason = str(e)
    
    def process_enhanced(self, data: np.ndarray, pattern_name: str = "Unknown") -> Dict[str, Any]:
        """Enhanced processing with all research components"""
        start_time = time.perf_counter()
        
        if not self.components_loaded:
            return self._fallback_process(data, pattern_name, start_time)
        
        try:
            # Step 1: Streaming TDA Analysis
            stream_result = self.streaming.process_stream(data)
            
            # Step 2: Multi-Parameter Analysis (if enough data)
            if len(data) > 10 and len(data.shape) >= 2:
                # Reshape for multi-parameter
                if len(data.shape) == 2:
                    mp_data = data.reshape(1, *data.shape)
                else:
                    mp_data = data
                
                mp_result = self.mp_processor.compute_multi_parameter_persistence(mp_data)
                mp_features = self._extract_mp_features(mp_result)
            else:
                mp_features = {'entropy': 0.5, 'amplitude': 1.0, 'complexity': 0.3}
            
            # Step 3: PHFormer Enhancement
            betti_numbers = stream_result['betti_numbers']
            persistence_diagram = stream_result['persistence_diagram']
            
            phformer_result = self.phformer.process_topology(betti_numbers, persistence_diagram)
            
            # Step 4: Feature Fusion
            enhanced_features = self._fuse_features(
                stream_result, mp_features, phformer_result
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self.frame_count += 1
            
            return {
                'frame_id': self.frame_count,
                'pattern': pattern_name,
                'betti_numbers': betti_numbers,
                'persistence_diagram': persistence_diagram,
                'enhanced_features': enhanced_features,
                'temporal_stability': stream_result.get('temporal_stability', 1.0),
                'change_rate': stream_result.get('change_rate', 0.0),
                'processing_time_ms': processing_time,
                'method': 'Enhanced-Research-Pipeline',
                'components': {
                    'streaming': True,
                    'multi_parameter': len(data) > 10,
                    'phformer': True
                }
            }
            
        except Exception as e:
            return self._fallback_process(data, pattern_name, start_time, error=str(e))
    
    def _extract_mp_features(self, mp_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract key features from multi-parameter result"""
        
        features = {}
        
        # Entropy features
        if 'entropy' in mp_result and mp_result['entropy'] is not None:
            entropy_val = np.mean(mp_result['entropy'])
            features['entropy'] = float(entropy_val) if not np.isnan(entropy_val) else 0.5
        else:
            features['entropy'] = 0.5
        
        # Amplitude features
        if 'amplitude' in mp_result and mp_result['amplitude'] is not None:
            amp_val = np.mean(mp_result['amplitude'])
            features['amplitude'] = float(amp_val) if not np.isnan(amp_val) else 1.0
        else:
            features['amplitude'] = 1.0
        
        # Multi-parameter complexity
        if 'mp_features' in mp_result and mp_result['mp_features']:
            mp_feats = mp_result['mp_features']
            
            # Calculate complexity from channel features
            complexity = 0.0
            channel_count = 0
            
            for key, value in mp_feats.items():
                if isinstance(value, dict) and 'complexity' in value:
                    complexity += value['complexity']
                    channel_count += 1
                elif key == 'cross_correlations' and isinstance(value, list):
                    complexity += np.mean(np.abs(value)) if value else 0.0
                    channel_count += 1
            
            features['complexity'] = complexity / max(channel_count, 1)
        else:
            features['complexity'] = 0.3
        
        return features
    
    def _fuse_features(self, stream_result: Dict, mp_features: Dict, phformer_result: Dict) -> Dict[str, Any]:
        """Fuse features from all components"""
        
        # Base topological features
        fused = {
            'betti_0': stream_result['betti_numbers'][0] if stream_result['betti_numbers'] else 0,
            'betti_1': stream_result['betti_numbers'][1] if len(stream_result['betti_numbers']) > 1 else 0,
            'persistence_count': len(stream_result['persistence_diagram'])
        }
        
        # Temporal features
        fused.update({
            'temporal_stability': stream_result.get('temporal_stability', 1.0),
            'change_rate': stream_result.get('change_rate', 0.0)
        })
        
        # Multi-parameter features
        fused.update({
            'mp_entropy': mp_features.get('entropy', 0.5),
            'mp_amplitude': mp_features.get('amplitude', 1.0),
            'mp_complexity': mp_features.get('complexity', 0.3)
        })
        
        # PHFormer features
        if 'topology_embeddings' in phformer_result:
            embeddings = phformer_result['topology_embeddings']
            if hasattr(embeddings, 'shape') and len(embeddings.shape) > 0:
                fused.update({
                    'phformer_mean': float(np.mean(embeddings)),
                    'phformer_std': float(np.std(embeddings)),
                    'phformer_max': float(np.max(embeddings))
                })
            else:
                fused.update({
                    'phformer_mean': 0.0,
                    'phformer_std': 1.0,
                    'phformer_max': 1.0
                })
        
        # Composite features
        fused['topology_richness'] = (
            fused['betti_0'] + fused['betti_1'] * 2 + 
            fused['persistence_count'] * 0.1
        )
        
        fused['stability_score'] = (
            fused['temporal_stability'] * 0.6 + 
            (1.0 - fused['change_rate']) * 0.4
        )
        
        return fused
    
    def _fallback_process(self, data: np.ndarray, pattern_name: str, start_time: float, error: str = None) -> Dict[str, Any]:
        """Fallback processing when components fail"""
        
        # Simple geometric analysis
        n_points = len(data)
        
        if n_points > 5:
            # Basic connectivity analysis
            from scipy.spatial.distance import pdist
            try:
                distances = pdist(data)
                threshold = np.percentile(distances, 25)
                
                betti_0 = max(1, n_points // 10)
                betti_1 = max(0, int(np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0)
            except:
                betti_0, betti_1 = 1, 0
        else:
            betti_0, betti_1 = 1, 0
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self.frame_count += 1
        
        return {
            'frame_id': self.frame_count,
            'pattern': pattern_name,
            'betti_numbers': [betti_0, betti_1],
            'persistence_diagram': [[0.0, 0.5]],
            'enhanced_features': {
                'betti_0': betti_0,
                'betti_1': betti_1,
                'topology_richness': betti_0 + betti_1,
                'stability_score': 0.8
            },
            'temporal_stability': 1.0,
            'change_rate': 0.0,
            'processing_time_ms': processing_time,
            'method': 'Fallback-Pipeline',
            'error': error,
            'components': {
                'streaming': False,
                'multi_parameter': False,
                'phformer': False
            }
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        pass
        return {
            'components_loaded': self.components_loaded,
            'device': self.device,
            'frames_processed': self.frame_count,
            'fallback_reason': getattr(self, 'fallback_reason', None)
        }

    def get_enhanced_pipeline(device='auto'):
        return EnhancedTDAPipeline(device)
