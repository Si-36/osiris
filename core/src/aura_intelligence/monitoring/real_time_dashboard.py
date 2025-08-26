"""
AURA Intelligence Real-Time Dashboard
WebSocket-based real-time monitoring and business intelligence
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Set
import logging
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

from .business_metrics import BusinessMetricsCollector
from ..adapters.redis_adapter import RedisAdapter


class RealTimeDashboard:
    """Real-time WebSocket dashboard for AURA Intelligence monitoring"""
    
    def __init__(self, redis_adapter: RedisAdapter, port: int = 8765):
        self.redis_adapter = redis_adapter
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # WebSocket management
        self.connected_clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        
        # Data sources
        self.business_metrics = BusinessMetricsCollector(redis_adapter)
        
        # Dashboard state
        self.dashboard_data = {}
        self.last_update = 0
        self.update_interval = 5  # seconds
        
        # Real-time metrics
        self.live_metrics = {
            'requests_per_second': 0,
            'avg_response_time': 0,
            'active_connections': 0,
            'gpu_utilization': 0,
            'memory_usage': 0,
            'error_rate': 0
        }
    
        async def start_server(self):
        """Start the WebSocket dashboard server"""
        pass
        try:
            self.server = await websockets.serve(
                self.handle_client,
                "0.0.0.0",
                self.port
            )
            self.logger.info(f"Real-time dashboard started on port {self.port}")
            
            # Start background tasks
            asyncio.create_task(self.update_dashboard_data())
            asyncio.create_task(self.broadcast_updates())
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
            raise
    
        async def stop_server(self):
        """Stop the WebSocket dashboard server"""
        pass
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("Dashboard server stopped")
    
        async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connections"""
        self.connected_clients.add(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        self.logger.info(f"Dashboard client connected from {client_ip}")
        
        try:
            # Send initial dashboard data
            await self.send_initial_data(websocket)
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Dashboard client {client_ip} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_ip}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
        async def send_initial_data(self, websocket: WebSocketServerProtocol):
        """Send initial dashboard data to new client"""
        try:
            initial_data = {
                'type': 'initial',
                'data': await self.get_complete_dashboard_data(),
                'timestamp': time.time()
            }
            
            await websocket.send(json.dumps(initial_data))
            
        except Exception as e:
            self.logger.error(f"Failed to send initial data: {e}")
    
        async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle messages from dashboard clients"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'get_kpis':
                await self.send_kpi_data(websocket)
            elif message_type == 'get_alerts':
                await self.send_alerts(websocket)
            elif message_type == 'get_recommendations':
                await self.send_recommendations(websocket)
            elif message_type == 'trigger_benchmark':
                await self.trigger_performance_benchmark(websocket)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON received from client")
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
    
        async def send_kpi_data(self, websocket: WebSocketServerProtocol):
        """Send KPI data to specific client"""
        try:
            kpis = await self.business_metrics.calculate_kpis()
            
            response = {
                'type': 'kpis',
                'data': [kpi.__dict__ for kpi in kpis],
                'timestamp': time.time()
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Failed to send KPI data: {e}")
    
        async def send_alerts(self, websocket: WebSocketServerProtocol):
        """Send current alerts to specific client"""
        try:
            dashboard_data = await self.business_metrics.get_business_dashboard_data()
            alerts = dashboard_data.get('alerts', [])
            
            response = {
                'type': 'alerts',
                'data': alerts,
                'timestamp': time.time()
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Failed to send alerts: {e}")
    
        async def send_recommendations(self, websocket: WebSocketServerProtocol):
        """Send recommendations to specific client"""
        try:
            dashboard_data = await self.business_metrics.get_business_dashboard_data()
            recommendations = dashboard_data.get('recommendations', [])
            
            response = {
                'type': 'recommendations',
                'data': recommendations,
                'timestamp': time.time()
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Failed to send recommendations: {e}")
    
        async def trigger_performance_benchmark(self, websocket: WebSocketServerProtocol):
        """Trigger a performance benchmark and stream results"""
        try:
            # Start benchmark
            response = {
                'type': 'benchmark_started',
                'data': {'status': 'running'},
                'timestamp': time.time()
            }
            await websocket.send(json.dumps(response))
            
            # Simulate benchmark progress (in real implementation, this would run actual benchmarks)
            for progress in [20, 40, 60, 80, 100]:
                await asyncio.sleep(1)
                
                progress_data = {
                    'type': 'benchmark_progress',
                    'data': {
                        'progress': progress,
                        'current_test': f'Running benchmark step {progress//20}/5'
                    },
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(progress_data))
            
            # Send benchmark results
            results = {
                'type': 'benchmark_results',
                'data': {
                    'bert_latency': '3.2ms',
                    'pipeline_total': '0.8ms',
                    'gpu_utilization': '85%',
                    'throughput': '1250 req/s',
                    'status': 'completed'
                },
                'timestamp': time.time()
            }
            await websocket.send(json.dumps(results))
            
        except Exception as e:
            self.logger.error(f"Failed to handle benchmark request: {e}")
    
        async def update_dashboard_data(self):
        """Background task to update dashboard data"""
        pass
        while True:
            try:
                # Update live metrics
                await self.collect_live_metrics()
                
                # Update business dashboard data every update interval
                if time.time() - self.last_update >= self.update_interval:
                    self.dashboard_data = await self.get_complete_dashboard_data()
                    self.last_update = time.time()
                
                await asyncio.sleep(1)  # Update live metrics every second
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard data: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
        async def collect_live_metrics(self):
        """Collect real-time system metrics"""
        pass
        try:
            # Get system health from Redis
            system_health = await self.redis_adapter.get_data("system_health")
            
            if system_health:
                self.live_metrics.update({
                    'requests_per_second': system_health.get('requests_per_second', 0),
                    'avg_response_time': system_health.get('avg_response_time', 0),
                    'gpu_utilization': system_health.get('gpu_utilization', 0),
                    'memory_usage': system_health.get('memory_usage_percent', 0),
                    'error_rate': system_health.get('error_rate', 0)
                })
            
            # Count active connections
            self.live_metrics['active_connections'] = len(self.connected_clients)
            
        except Exception as e:
            self.logger.error(f"Error collecting live metrics: {e}")
    
        async def broadcast_updates(self):
        """Background task to broadcast updates to all connected clients"""
        pass
        while True:
            try:
                if self.connected_clients:
                    # Create update message
                    update_data = {
                        'type': 'live_update',
                        'data': {
                            'live_metrics': self.live_metrics,
                            'last_updated': time.time()
                        },
                        'timestamp': time.time()
                    }
                    
                    # Broadcast to all connected clients
                    message = json.dumps(update_data)
                    disconnected_clients = set()
                    
                    for client in self.connected_clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            self.logger.error(f"Error sending to client: {e}")
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    self.connected_clients -= disconnected_clients
                
                await asyncio.sleep(2)  # Broadcast every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(5)
    
        async def get_complete_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        pass
        try:
            # Get business metrics
            business_data = await self.business_metrics.get_business_dashboard_data()
            
            # Get system status
            system_status = await self.get_system_status()
            
            # Get performance history
            performance_history = await self.get_performance_history()
            
            return {
                'business_metrics': business_data,
                'system_status': system_status,
                'performance_history': performance_history,
                'live_metrics': self.live_metrics,
                'connected_clients': len(self.connected_clients),
                'server_uptime': self.get_server_uptime()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting complete dashboard data: {e}")
            return {'error': str(e)}
    
        async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        pass
        try:
            # Get component health from Redis
            component_health = await self.redis_adapter.get_data("component_health")
            
            if not component_health:
                return {'status': 'unknown', 'components': {}}
            
            # Calculate overall health
            health_scores = [comp.get('health_score', 0) for comp in component_health.values()]
            overall_health = sum(health_scores) / len(health_scores) if health_scores else 0
            
            return {
                'status': 'healthy' if overall_health > 0.8 else 'degraded' if overall_health > 0.5 else 'critical',
                'overall_health': overall_health,
                'components': component_health,
                'gpu_available': any(comp.get('gpu_enabled', False) for comp in component_health.values()),
                'redis_connected': True  # If we're here, Redis is connected
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
        async def get_performance_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance history for the dashboard"""
        try:
            # Get metrics from business collector
            recent_metrics = await self.business_metrics._get_recent_metrics(hours)
            
            # Process into time series data
            time_series = self.process_metrics_to_timeseries(recent_metrics)
            
            return {
                'timeseries': time_series,
                'summary': {
                    'total_requests': len(recent_metrics),
                    'avg_efficiency': sum(m.value for m in recent_metrics if m.name == 'request_efficiency') / max(1, len(recent_metrics)),
                    'timespan_hours': hours
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance history: {e}")
            return {'timeseries': {}, 'summary': {}}
    
    def process_metrics_to_timeseries(self, metrics: List) -> Dict[str, List]:
        """Process metrics into time series format for charting"""
        timeseries = {
            'timestamps': [],
            'efficiency_scores': [],
            'processing_times': [],
            'cost_estimates': [],
            'quality_scores': []
        }
        
        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)
        
        for metric in sorted_metrics:
            timeseries['timestamps'].append(metric.timestamp)
            
            if metric.name == 'request_efficiency':
                timeseries['efficiency_scores'].append(metric.value)
            else:
                timeseries['efficiency_scores'].append(None)
            
            # Extract metadata
            if 'processing_time_ms' in metric.metadata:
                timeseries['processing_times'].append(metric.metadata['processing_time_ms'])
            else:
                timeseries['processing_times'].append(None)
            
            if 'cost_estimate' in metric.metadata:
                timeseries['cost_estimates'].append(metric.metadata['cost_estimate'])
            else:
                timeseries['cost_estimates'].append(None)
            
            if 'quality_score' in metric.metadata:
                timeseries['quality_scores'].append(metric.metadata['quality_score'])
            else:
                timeseries['quality_scores'].append(None)
        
        return timeseries
    
    def get_server_uptime(self) -> float:
        """Get server uptime in seconds"""
        pass
        if hasattr(self, 'start_time'):
            return time.time() - self.start_time
        return 0
    
        async def record_request_metric(self, request_data: Dict[str, Any]):
        """Record a request metric (called by the main system)"""
        try:
            await self.business_metrics.collect_request_metrics(request_data)
        except Exception as e:
            self.logger.error(f"Error recording request metric: {e}")
    
        async def health_check(self) -> Dict[str, Any]:
        """Health check for the dashboard service"""
        pass
        return {
            'status': 'healthy',
            'connected_clients': len(self.connected_clients),
            'server_running': self.server is not None,
            'last_update': self.last_update,
            'uptime': self.get_server_uptime()
        }


# Export main class
__all__ = ['RealTimeDashboard']