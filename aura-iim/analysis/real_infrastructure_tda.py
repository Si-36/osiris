"""
Real Infrastructure TDA Analysis
================================
Uses the actual AURA TDA engine for infrastructure monitoring
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio

# Add paths for real AURA components
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/src')

# Import REAL TDA components
from aura.tda.algorithms import (
    RipsComplex,
    PersistentHomology,
    wasserstein_distance,
    compute_persistence_landscape
)

# Import from core if available
try:
    from core.src.aura_intelligence.tda.unified_engine_2025 import UnifiedTDAEngine2025
    HAS_UNIFIED_ENGINE = True
except:
    HAS_UNIFIED_ENGINE = False


class RealInfrastructureTDA:
    """Production-grade TDA analysis for infrastructure monitoring"""
    
    def __init__(self):
        # Real TDA components
        self.rips = RipsComplex()
        self.ph = PersistentHomology()
        
        # State tracking
        self.baseline_persistence = None
        self.baseline_landscape = None
        self.historical_distances = []
        
        # Thresholds
        self.anomaly_threshold = 2.0  # Standard deviations
        self.critical_threshold = 3.0
        
        # Unified engine if available
        if HAS_UNIFIED_ENGINE:
            self.unified_engine = UnifiedTDAEngine2025()
        else:
            self.unified_engine = None
    
    async def analyze_infrastructure(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        """
        Analyze infrastructure topology using real TDA
        
        Args:
            point_cloud: Infrastructure metrics as point cloud (n_samples, n_features)
            
        Returns:
            Dictionary with topological features and anomaly detection
        """
        
        # 1. Compute Rips complex and Betti numbers
        rips_result = self.rips.compute(point_cloud, max_edge_length=3.0)
        
        # 2. Compute persistence homology
        persistence_pairs = self.ph.compute_persistence(point_cloud)
        
        # 3. Compute persistence landscape
        landscape = compute_persistence_landscape(persistence_pairs)
        
        # 4. Extract topological features
        features = {
            'timestamp': datetime.now().isoformat(),
            'betti_0': rips_result['betti_0'],  # Connected components
            'betti_1': rips_result['betti_1'],  # Loops/cycles
            'betti_2': 0,  # 2D voids (compute if needed)
            'num_edges': rips_result['num_edges'],
            'num_triangles': rips_result['num_triangles'],
            'persistence_pairs': len(persistence_pairs),
            'max_persistence': float(min(max((d - b for b, d in persistence_pairs), default=0), 1000)),  # Cap infinity
            'persistence_entropy': self._compute_persistence_entropy(persistence_pairs),
            'landscape_l1_norm': np.sum(np.abs(landscape)),
            'anomaly_detected': False,
            'anomaly_score': 0.0,
            'critical_features': []
        }
        
        # 5. Baseline comparison
        if self.baseline_persistence is not None:
            # Compute Wasserstein distance to baseline
            w_distance = wasserstein_distance(
                self.baseline_persistence,
                persistence_pairs
            )
            features['wasserstein_distance'] = w_distance
            
            # Track historical distances
            self.historical_distances.append(w_distance)
            if len(self.historical_distances) > 100:
                self.historical_distances.pop(0)
            
            # Anomaly detection
            if len(self.historical_distances) > 10:
                mean_dist = np.mean(self.historical_distances[:-1])
                std_dist = np.std(self.historical_distances[:-1])
                z_score = (w_distance - mean_dist) / (std_dist + 1e-6)
                
                features['anomaly_score'] = z_score
                features['anomaly_detected'] = z_score > self.anomaly_threshold
                features['critical_alert'] = z_score > self.critical_threshold
        else:
            # Set baseline on first run
            self.baseline_persistence = persistence_pairs
            self.baseline_landscape = landscape
        
        # 6. Identify critical topological features
        critical_features = self._identify_critical_features(
            persistence_pairs,
            rips_result
        )
        features['critical_features'] = critical_features
        
        # 7. Infrastructure-specific interpretations
        features['interpretations'] = self._interpret_topology(features)
        
        # 8. If unified engine available, use advanced features
        if self.unified_engine and HAS_UNIFIED_ENGINE:
            advanced = await self._compute_advanced_features(point_cloud)
            features['advanced'] = advanced
        
        return features
    
    def _compute_persistence_entropy(self, persistence_pairs: List[Tuple[float, float]]) -> float:
        """Compute entropy of persistence diagram"""
        if not persistence_pairs:
            return 0.0
        
        # Compute lifetimes
        lifetimes = [death - birth for birth, death in persistence_pairs]
        total_lifetime = sum(lifetimes)
        
        if total_lifetime == 0:
            return 0.0
        
        # Compute probabilities
        probs = [l / total_lifetime for l in lifetimes]
        
        # Compute entropy
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        return entropy
    
    def _identify_critical_features(
        self,
        persistence_pairs: List[Tuple[float, float]],
        rips_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify critical infrastructure features from topology"""
        
        critical = []
        
        # Long-lived features (high persistence)
        for i, (birth, death) in enumerate(persistence_pairs):
            persistence = death - birth
            if persistence > 1.0:  # Threshold for significance
                critical.append({
                    'type': 'persistent_feature',
                    'dimension': 1,  # Assuming 1D features
                    'birth': birth,
                    'death': death,
                    'persistence': persistence,
                    'interpretation': self._interpret_persistent_feature(persistence, birth)
                })
        
        # Network partitioning (multiple components)
        if rips_result['betti_0'] > 1:
            critical.append({
                'type': 'network_partition',
                'components': rips_result['betti_0'],
                'severity': 'critical' if rips_result['betti_0'] > 3 else 'warning',
                'interpretation': f"Infrastructure split into {rips_result['betti_0']} isolated components"
            })
        
        # Circular dependencies (many loops)
        if rips_result['betti_1'] > 10:
            critical.append({
                'type': 'circular_dependencies',
                'loops': rips_result['betti_1'],
                'severity': 'warning',
                'interpretation': f"Detected {rips_result['betti_1']} dependency loops"
            })
        
        # Sort by importance
        critical.sort(key=lambda x: x.get('persistence', 0) + 
                     (10 if x.get('severity') == 'critical' else 0), 
                     reverse=True)
        
        return critical[:5]  # Top 5 critical features
    
    def _interpret_persistent_feature(self, persistence: float, birth: float) -> str:
        """Map persistence features to infrastructure meaning"""
        
        if persistence > 2.0:
            if birth < 0.5:
                return "Critical system bottleneck detected - immediate attention required"
            else:
                return "Major infrastructure anomaly - investigate resource allocation"
        elif persistence > 1.0:
            if birth < 0.5:
                return "Resource contention pattern detected"
            else:
                return "Unusual service interaction pattern"
        else:
            return "Minor topology variation"
    
    def _interpret_topology(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Infrastructure-specific topology interpretations"""
        
        interpretations = {}
        
        # Connected components
        if features['betti_0'] == 1:
            interpretations['connectivity'] = "Infrastructure fully connected ✓"
        elif features['betti_0'] <= 3:
            interpretations['connectivity'] = f"⚠️ Infrastructure has {features['betti_0']} isolated segments"
        else:
            interpretations['connectivity'] = f"🚨 CRITICAL: {features['betti_0']} network partitions detected"
        
        # Loops/cycles
        if features['betti_1'] == 0:
            interpretations['dependencies'] = "No circular dependencies ✓"
        elif features['betti_1'] <= 5:
            interpretations['dependencies'] = f"⚠️ {features['betti_1']} circular dependencies found"
        else:
            interpretations['dependencies'] = f"🚨 High complexity: {features['betti_1']} dependency loops"
        
        # Persistence entropy
        if features['persistence_entropy'] < 1.0:
            interpretations['stability'] = "Topology stable ✓"
        elif features['persistence_entropy'] < 2.0:
            interpretations['stability'] = "⚠️ Moderate topology changes detected"
        else:
            interpretations['stability'] = "🚨 High topology volatility - system unstable"
        
        # Anomaly detection
        if features.get('critical_alert', False):
            interpretations['alert'] = "🚨 CRITICAL ANOMALY - Immediate action required"
        elif features['anomaly_detected']:
            interpretations['alert'] = "⚠️ Anomaly detected - Monitor closely"
        else:
            interpretations['alert'] = "System operating normally ✓"
        
        return interpretations
    
    async def _compute_advanced_features(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        """Compute advanced features using unified engine"""
        
        try:
            # Use unified engine for advanced analysis
            result = await self.unified_engine.analyze(
                point_cloud,
                compute_gpu=True,
                compute_spectral=True,
                compute_distributed=False  # Not needed for single instance
            )
            
            return {
                'gpu_accelerated': True,
                'spectral_gap': result.get('spectral_gap', 0),
                'gpu_speedup': result.get('gpu_speedup', 1.0),
                'advanced_metrics': result.get('advanced_metrics', {})
            }
        except Exception as e:
            return {
                'gpu_accelerated': False,
                'error': str(e)
            }
    
    def predict_failure_location(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict which infrastructure components are likely to fail"""
        
        predictions = []
        
        # Based on topological features
        for critical in features.get('critical_features', []):
            if critical['type'] == 'persistent_feature':
                predictions.append({
                    'component': f"Service cluster at distance {critical['birth']:.2f}",
                    'failure_probability': min(critical['persistence'] / 3.0, 0.95),
                    'time_to_failure': f"{2.0 / critical['persistence']:.1f} hours",
                    'reason': critical['interpretation']
                })
            elif critical['type'] == 'network_partition':
                predictions.append({
                    'component': f"Network segment {critical['components']}",
                    'failure_probability': 0.8,
                    'time_to_failure': "< 1 hour",
                    'reason': "Network isolation detected"
                })
        
        return predictions


async def demo_real_tda():
    """Demo the real TDA infrastructure analysis"""
    
    print("🚀 Real Infrastructure TDA Analysis Demo")
    print("=" * 60)
    
    analyzer = RealInfrastructureTDA()
    
    # Simulate infrastructure metrics over time
    for t in range(5):
        print(f"\n⏰ Time step {t+1}")
        
        # Generate infrastructure point cloud
        # Features: CPU, Memory, Network, Disk, Connections, Latency
        n_servers = 50
        
        if t < 3:
            # Normal operation
            point_cloud = np.random.randn(n_servers, 6) * 0.3 + np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        else:
            # Introduce anomaly
            point_cloud = np.random.randn(n_servers, 6) * 0.5
            # Create network partition
            point_cloud[25:35, 4] += 2.0  # Spike in connections
            point_cloud[35:45, 5] += 1.5  # Increased latency
        
        # Analyze
        features = await analyzer.analyze_infrastructure(point_cloud)
        
        print(f"\n📊 Topology Analysis:")
        print(f"   Betti₀ (components): {features['betti_0']}")
        print(f"   Betti₁ (loops): {features['betti_1']}")
        print(f"   Max persistence: {features['max_persistence']:.3f}")
        print(f"   Anomaly score: {features['anomaly_score']:.2f}")
        
        print(f"\n💡 Interpretations:")
        for key, interp in features['interpretations'].items():
            print(f"   {key}: {interp}")
        
        if features['critical_features']:
            print(f"\n⚠️ Critical Features:")
            for cf in features['critical_features'][:3]:
                print(f"   - {cf.get('interpretation', cf['type'])}")
        
        # Predict failures
        predictions = analyzer.predict_failure_location(features)
        if predictions:
            print(f"\n🔮 Failure Predictions:")
            for pred in predictions:
                print(f"   - {pred['component']}: {pred['failure_probability']:.1%} in {pred['time_to_failure']}")
        
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(demo_real_tda())