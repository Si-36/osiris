"""
Load Monitor Service
Real-time monitoring and load balancing for services
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from collections import deque, defaultdict
import numpy as np
import structlog
import httpx

logger = structlog.get_logger()


class LoadMonitorService:
    """
    Monitor service loads and perform rebalancing
    """
    
    def __init__(self, router_system):
        self.router = router_system
        self.logger = logger.bind(service="load_monitor")
        
        # Monitoring data
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Thresholds
        self.overload_threshold = 0.8
        self.underload_threshold = 0.2
        self.imbalance_threshold = 0.3
        
        # Monitoring state
        self.monitoring_active = True
        self.check_interval = 10.0  # seconds
        self.last_rebalance_time = 0
        self.rebalance_cooldown = 60.0  # seconds
        
    async def monitor_services(self):
        """
        Continuous monitoring loop
        """
        self.logger.info("Starting load monitoring")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Check for imbalance
                if self._detect_imbalance():
                    self.logger.warning("Load imbalance detected")
                    
                    # Auto-rebalance if cooldown passed
                    if time.time() - self.last_rebalance_time > self.rebalance_cooldown:
                        await self.rebalance_load()
                
                # Check for overloaded services
                overloaded = self._get_overloaded_services()
                if overloaded:
                    self.logger.warning(
                        "Overloaded services detected",
                        services=list(overloaded)
                    )
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self):
        """
        Collect current metrics from all services
        """
        for service_id, profile in self.router.services.items():
            # Record current load
            self.load_history[service_id].append({
                "timestamp": time.time(),
                "load": profile.current_load,
                "capacity": profile.max_capacity,
                "utilization": profile.current_load / profile.max_capacity
            })
            
            # Record latency
            if service_id in self.router.response_times:
                recent_latencies = list(self.router.response_times[service_id])
                if recent_latencies:
                    self.latency_history[service_id].append({
                        "timestamp": time.time(),
                        "p50": np.percentile(recent_latencies, 50),
                        "p95": np.percentile(recent_latencies, 95),
                        "p99": np.percentile(recent_latencies, 99)
                    })
    
    def _detect_imbalance(self) -> bool:
        """
        Detect if services are imbalanced
        """
        if len(self.router.services) < 2:
            return False
        
        utilizations = []
        for service_id, profile in self.router.services.items():
            if service_id in self.router._get_available_services():
                utilization = profile.current_load / profile.max_capacity
                utilizations.append(utilization)
        
        if not utilizations:
            return False
        
        # Calculate coefficient of variation
        mean_util = np.mean(utilizations)
        std_util = np.std(utilizations)
        
        if mean_util > 0:
            cv = std_util / mean_util
            return cv > self.imbalance_threshold
        
        return False
    
    def _get_overloaded_services(self) -> Set[str]:
        """
        Get services that are overloaded
        """
        overloaded = set()
        
        for service_id, profile in self.router.services.items():
            utilization = profile.current_load / profile.max_capacity
            if utilization > self.overload_threshold:
                overloaded.add(service_id)
        
        return overloaded
    
    def _get_underloaded_services(self) -> Set[str]:
        """
        Get services that are underloaded
        """
        underloaded = set()
        
        for service_id, profile in self.router.services.items():
            if service_id in self.router._get_available_services():
                utilization = profile.current_load / profile.max_capacity
                if utilization < self.underload_threshold:
                    underloaded.add(service_id)
        
        return underloaded
    
    async def rebalance_load(self, 
                           target_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform load rebalancing
        """
        self.logger.info("Starting load rebalancing")
        start_time = time.time()
        
        # Get current state
        overloaded = self._get_overloaded_services()
        underloaded = self._get_underloaded_services()
        
        if not overloaded or not underloaded:
            return {
                "status": "no_rebalancing_needed",
                "migrations": 0
            }
        
        # Calculate target distribution if not provided
        if not target_distribution:
            target_distribution = self._calculate_target_distribution()
        
        # Simulate migrations (in production, would migrate actual requests)
        migrations = 0
        
        for source in overloaded:
            source_profile = self.router.services[source]
            source_excess = source_profile.current_load - (
                source_profile.max_capacity * self.overload_threshold
            )
            
            for target in underloaded:
                if source_excess <= 0:
                    break
                
                target_profile = self.router.services[target]
                target_available = (
                    target_profile.max_capacity * self.overload_threshold - 
                    target_profile.current_load
                )
                
                if target_available > 0:
                    migration_amount = min(source_excess, target_available)
                    
                    # Update loads
                    source_profile.current_load -= migration_amount
                    target_profile.current_load += migration_amount
                    
                    source_excess -= migration_amount
                    migrations += int(migration_amount)
        
        self.last_rebalance_time = time.time()
        duration = time.time() - start_time
        
        self.logger.info(
            "Load rebalancing completed",
            migrations=migrations,
            duration_ms=duration * 1000
        )
        
        return {
            "status": "rebalanced",
            "migrations": migrations,
            "duration_ms": duration * 1000,
            "overloaded_services": list(overloaded),
            "underloaded_services": list(underloaded)
        }
    
    def _calculate_target_distribution(self) -> Dict[str, float]:
        """
        Calculate optimal load distribution
        """
        available_services = self.router._get_available_services()
        total_capacity = sum(
            self.router.services[sid].max_capacity 
            for sid in available_services
        )
        
        distribution = {}
        for service_id in available_services:
            capacity = self.router.services[service_id].max_capacity
            distribution[service_id] = capacity / total_capacity
        
        return distribution
    
    async def health_check_service(self, service_id: str) -> Dict[str, Any]:
        """
        Perform health check on specific service
        """
        if service_id not in self.router.services:
            return {"status": "unknown", "error": "Service not found"}
        
        profile = self.router.services[service_id]
        
        try:
            # Simple HTTP health check
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{profile.endpoint}/api/v1/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "latency_ms": response.elapsed.total_seconds() * 1000,
                        "details": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            self.error_counts[service_id] += 1
            return {
                "status": "error",
                "error": str(e),
                "error_count": self.error_counts[service_id]
            }
    
    def get_load_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive load metrics
        """
        metrics = {
            "services": {},
            "overall": {
                "total_load": 0,
                "total_capacity": 0,
                "average_utilization": 0.0,
                "load_balance_score": 0.0
            }
        }
        
        utilizations = []
        
        for service_id, profile in self.router.services.items():
            utilization = profile.current_load / profile.max_capacity
            utilizations.append(utilization)
            
            metrics["services"][service_id] = {
                "current_load": profile.current_load,
                "max_capacity": profile.max_capacity,
                "utilization": utilization,
                "is_overloaded": utilization > self.overload_threshold,
                "is_underloaded": utilization < self.underload_threshold,
                "error_count": self.error_counts[service_id]
            }
            
            metrics["overall"]["total_load"] += profile.current_load
            metrics["overall"]["total_capacity"] += profile.max_capacity
        
        if utilizations:
            metrics["overall"]["average_utilization"] = np.mean(utilizations)
            
            # Load balance score (inverse of coefficient of variation)
            if metrics["overall"]["average_utilization"] > 0:
                cv = np.std(utilizations) / metrics["overall"]["average_utilization"]
                metrics["overall"]["load_balance_score"] = 1.0 / (1.0 + cv)
        
        return metrics
    
    def stop_monitoring(self):
        """Stop monitoring loop"""
        self.monitoring_active = False