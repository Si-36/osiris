"""
Data Pipeline for Infrastructure Metrics
========================================
Processes raw metrics into point clouds for TDA analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio


class DataPipeline:
    """Production-grade data pipeline for infrastructure monitoring"""
    
    def __init__(self):
        # Configuration
        self.feature_dimensions = 8  # Number of features per server
        self.normalization_ranges = {
            'cpu': (0, 100),
            'memory': (0, 100),
            'disk': (0, 100),
            'network': (0, 1000),  # Connections
            'latency': (0, 1000),  # ms
            'throughput': (0, 1000),  # MB/s
            'errors': (0, 100),  # per minute
            'queue_depth': (0, 1000)
        }
        
        # State tracking
        self.feature_buffer = []
        self.min_samples = 10  # Minimum samples for analysis
        
    async def process_metrics(self, metrics: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert raw metrics to point cloud for TDA
        
        Args:
            metrics: Raw infrastructure metrics
            
        Returns:
            Point cloud as numpy array or None if insufficient data
        """
        
        try:
            # Extract features
            features = self._extract_features(metrics)
            
            if features is None:
                return None
            
            # Normalize features
            normalized = self._normalize_features(features)
            
            # Create point cloud
            point_cloud = self._create_point_cloud(normalized)
            
            return point_cloud
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            return None
    
    def _extract_features(self, metrics: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract relevant features from raw metrics"""
        
        features = []
        
        # CPU features
        cpu_data = metrics.get('cpu', {})
        if 'percent' in cpu_data:
            cpu_avg = np.mean(cpu_data['percent'])
            cpu_max = np.max(cpu_data['percent'])
            cpu_std = np.std(cpu_data['percent'])
        else:
            cpu_avg = cpu_max = cpu_std = 0
        
        # Memory features
        memory = metrics.get('memory', {})
        mem_percent = memory.get('percent', 0)
        mem_available = memory.get('available', 0) / (1024**3)  # GB
        
        # Disk features
        disk = metrics.get('disk', {})
        disk_percent = disk.get('percent', 0)
        disk_io_read = disk.get('read_rate', 0) / (1024**2)  # MB/s
        disk_io_write = disk.get('write_rate', 0) / (1024**2)  # MB/s
        
        # Network features
        network = metrics.get('network', {})
        connections = network.get('connections', 0)
        net_sent = network.get('bytes_sent_rate', 0) / (1024**2)  # MB/s
        net_recv = network.get('bytes_recv_rate', 0) / (1024**2)  # MB/s
        
        # Process features
        processes = metrics.get('processes', {})
        active_processes = processes.get('active', 0)
        zombie_processes = processes.get('zombie', 0)
        
        # Create feature vector
        feature_vector = np.array([
            cpu_avg,
            cpu_max,
            cpu_std,
            mem_percent,
            disk_percent,
            connections / 100.0,  # Scale down
            (net_sent + net_recv) / 100.0,  # Combined throughput
            min(disk_io_read + disk_io_write, 100) / 100.0  # I/O pressure
        ])
        
        return feature_vector
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        
        # Simple min-max normalization
        # In production, you'd use more sophisticated methods
        normalized = np.clip(features, 0, 1)
        
        return normalized
    
    def _create_point_cloud(self, features: np.ndarray) -> np.ndarray:
        """
        Create point cloud for TDA analysis
        
        For single server, we create a sliding window of features
        For multiple servers, each server is a point
        """
        
        # Add to buffer
        self.feature_buffer.append(features)
        
        # Keep buffer size manageable
        if len(self.feature_buffer) > 100:
            self.feature_buffer.pop(0)
        
        # Need minimum samples
        if len(self.feature_buffer) < self.min_samples:
            return None
        
        # Stack features into point cloud
        point_cloud = np.array(self.feature_buffer)
        
        # Add synthetic points for better topology
        synthetic = self._generate_synthetic_points(point_cloud)
        
        # Combine real and synthetic
        combined = np.vstack([point_cloud, synthetic])
        
        return combined
    
    def _generate_synthetic_points(self, real_points: np.ndarray) -> np.ndarray:
        """
        Generate synthetic points to enhance topological features
        
        This helps TDA detect patterns in sparse data
        """
        
        n_real = real_points.shape[0]
        n_synthetic = max(20, n_real // 2)
        
        synthetic = []
        
        # Interpolated points (detect paths)
        for _ in range(n_synthetic // 2):
            idx1, idx2 = np.random.choice(n_real, 2, replace=False)
            alpha = np.random.rand()
            interp = alpha * real_points[idx1] + (1 - alpha) * real_points[idx2]
            synthetic.append(interp)
        
        # Noisy points (detect clusters)
        mean = np.mean(real_points, axis=0)
        std = np.std(real_points, axis=0) + 1e-6
        
        for _ in range(n_synthetic // 2):
            noise = np.random.randn(real_points.shape[1]) * std * 0.1
            noisy = mean + noise
            synthetic.append(np.clip(noisy, 0, 1))
        
        return np.array(synthetic)
    
    async def process_batch(self, metrics_list: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Process batch of metrics from multiple servers"""
        
        all_features = []
        
        for metrics in metrics_list:
            features = self._extract_features(metrics)
            if features is not None:
                normalized = self._normalize_features(features)
                all_features.append(normalized)
        
        if len(all_features) < self.min_samples:
            return None
        
        return np.array(all_features)
    
    def add_topology_enhancing_features(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Add features that enhance topological signal
        """
        
        n_points, n_features = point_cloud.shape
        
        # Add distance-based features
        center = np.mean(point_cloud, axis=0)
        distances = np.linalg.norm(point_cloud - center, axis=1)
        
        # Add connectivity features
        connectivity = np.zeros(n_points)
        for i in range(n_points):
            # Count nearby points
            dists = np.linalg.norm(point_cloud - point_cloud[i], axis=1)
            connectivity[i] = np.sum(dists < 0.3)  # Within threshold
        
        # Combine
        enhanced = np.column_stack([
            point_cloud,
            distances.reshape(-1, 1),
            connectivity.reshape(-1, 1)
        ])
        
        return enhanced


# Demo function
async def demo_pipeline():
    """Demo the data pipeline"""
    
    print("🚀 Data Pipeline Demo")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    # Simulate metrics over time
    for t in range(15):
        # Generate mock metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': np.random.rand(4) * 50 + 25 + t * 2  # Increasing
            },
            'memory': {
                'percent': 40 + t * 3 + np.random.rand() * 10,
                'available': 8 * 1024**3  # 8GB
            },
            'disk': {
                'percent': 60 + np.random.rand() * 20,
                'read_rate': np.random.rand() * 100 * 1024**2,
                'write_rate': np.random.rand() * 50 * 1024**2
            },
            'network': {
                'connections': int(100 + t * 10 + np.random.rand() * 50),
                'bytes_sent_rate': np.random.rand() * 10 * 1024**2,
                'bytes_recv_rate': np.random.rand() * 20 * 1024**2
            },
            'processes': {
                'active': int(200 + np.random.rand() * 100),
                'zombie': int(np.random.rand() * 5)
            }
        }
        
        # Process metrics
        point_cloud = await pipeline.process_metrics(metrics)
        
        if point_cloud is not None:
            print(f"\n⏰ Time {t+1}:")
            print(f"   Point cloud shape: {point_cloud.shape}")
            print(f"   Feature ranges: [{point_cloud.min():.3f}, {point_cloud.max():.3f}]")
            print(f"   Cloud density: {len(point_cloud)} points")
        else:
            print(f"\n⏰ Time {t+1}: Buffering data...")
        
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(demo_pipeline())