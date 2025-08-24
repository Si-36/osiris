"""
Infrastructure Metrics Collector
Collects real-time CPU, memory, network, and disk metrics
"""

import asyncio
import psutil
import json
from datetime import datetime
from typing import Dict, Any

class InfrastructureCollector:
    """Collects infrastructure metrics for TDA analysis"""
    
    def __init__(self):
        self.metrics_history = []
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        load_avg = psutil.getloadavg()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Network metrics
        net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'load_average': list(load_avg),
                'cores': psutil.cpu_count()
            },
            'memory': {
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3),
                'swap_percent': swap.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'connections': connections
            },
            'disk': {
                'percent': disk_usage.percent,
                'free_gb': disk_usage.free / (1024**3),
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            }
        }
        
        self.metrics_history.append(metrics)
        return metrics

    def get_metrics_for_tda(self, window_size: int = 100):
        """Convert metrics to point cloud for TDA analysis"""
        import numpy as np
        
        # Get recent metrics
        recent = self.metrics_history[-window_size:]
        
        if len(recent) < 10:
            return None
            
        # Create feature vectors
        points = []
        for m in recent:
            # Extract key features
            features = [
                sum(m['cpu']['percent']) / len(m['cpu']['percent']),  # Avg CPU
                m['cpu']['load_average'][0],                          # 1-min load
                m['memory']['percent'],                                # Memory %
                m['network']['connections'] / 1000,                    # Connections (scaled)
                m['disk']['percent'],                                  # Disk %
                (m['network']['bytes_sent'] + m['network']['bytes_recv']) / (1024**3)  # Network GB
            ]
            points.append(features)
        
        return np.array(points)

async def demo_collector():
    """Demo the collector"""
    collector = InfrastructureCollector()
    
    print("🔍 Starting infrastructure monitoring...")
    print("="*50)
    
    for i in range(5):
        metrics = await collector.collect_metrics()
        
        print(f"\n📊 Metrics at {metrics['timestamp']}:")
        print(f"   CPU: {sum(metrics['cpu']['percent'])/len(metrics['cpu']['percent']):.1f}%")
        print(f"   Memory: {metrics['memory']['percent']:.1f}%")
        print(f"   Network Connections: {metrics['network']['connections']}")
        print(f"   Disk: {metrics['disk']['percent']:.1f}%")
        
        await asyncio.sleep(2)
    
    # Get point cloud for TDA
    point_cloud = collector.get_metrics_for_tda()
    if point_cloud is not None:
        print(f"\n✅ Generated point cloud for TDA: shape {point_cloud.shape}")

if __name__ == "__main__":
    asyncio.run(demo_collector())
