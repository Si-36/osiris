"""
Real Metric Collector - Actually collects system metrics
No dummy data, no mocks, just real system information
"""
import json
import time
import psutil
import redis
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealMetricCollector:
    """Collects REAL system metrics - CPU, Memory, Disk, Network"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis.ping()
            logger.info("✅ Connected to Redis")
        except redis.ConnectionError:
            logger.warning("⚠️  Redis not available, running in standalone mode")
            self.redis = None
            
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect real system metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent,
                'available': psutil.virtual_memory().available
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'percent': psutil.disk_usage('/').percent
            },
            'network': self._get_network_stats(),
            'processes': len(psutil.pids())
        }
        return metrics
    
    def _get_network_stats(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in Redis"""
        if self.redis:
            try:
                # Store latest metrics
                self.redis.set('metrics:latest', json.dumps(metrics))
                
                # Store in time series
                timestamp = int(time.time())
                self.redis.zadd('metrics:history', {json.dumps(metrics): timestamp})
                
                # Keep only last hour of data
                one_hour_ago = timestamp - 3600
                self.redis.zremrangebyscore('metrics:history', 0, one_hour_ago)
                
                # Publish to channel for real-time subscribers
                self.redis.publish('metrics:stream', json.dumps(metrics))
                
            except Exception as e:
                logger.error(f"Failed to store metrics: {e}")
    
    def run(self, interval: int = 5):
        """Run collector loop"""
        logger.info(f"🚀 Starting metric collection every {interval}s")
        
        while True:
            try:
                # Collect real metrics
                metrics = self.collect_metrics()
                
                # Log to console (so we can see it's working)
                logger.info(f"📊 CPU: {metrics['cpu']['percent']}% | "
                          f"MEM: {metrics['memory']['percent']}% | "
                          f"DISK: {metrics['disk']['percent']}%")
                
                # Store in Redis
                self.store_metrics(metrics)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("👋 Shutting down collector")
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(interval)


if __name__ == "__main__":
    collector = RealMetricCollector()
    collector.run()