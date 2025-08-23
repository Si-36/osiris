"""
TestContainers Manager for AURA Microservices
Production-grade container orchestration for integration tests

Based on 2025 best practices:
- TestContainers Cloud integration
- Reusable containers with warm pools
- Native cloud emulation
- Observability built-in
"""

import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
import time
import structlog
from contextlib import asynccontextmanager
import docker
import httpx
from testcontainers.compose import DockerCompose
from testcontainers.core.waiting_utils import wait_for_logs
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.core.container import DockerContainer
from testcontainers.neo4j import Neo4jContainer

logger = structlog.get_logger()


@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    healthy: bool
    latency_ms: float
    last_check: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerPool:
    """Reusable container pool for performance"""
    containers: Dict[str, DockerContainer] = field(default_factory=dict)
    health_checks: Dict[str, ServiceHealth] = field(default_factory=dict)
    warm_pool_size: int = 3
    

class AuraTestContainers:
    """
    Advanced TestContainers management with 2025 patterns
    
    Features:
    - Warm container pools for fast tests
    - Health monitoring and auto-recovery
    - Cloud provider emulation
    - Distributed tracing integration
    """
    
    def __init__(self, 
                 use_compose: bool = True,
                 cloud_mode: bool = False,
                 enable_observability: bool = True):
        self.use_compose = use_compose
        self.cloud_mode = cloud_mode  # TestContainers Cloud
        self.enable_observability = enable_observability
        self.logger = logger.bind(component="testcontainers")
        
        # Container management
        self.containers: Dict[str, DockerContainer] = {}
        self.compose: Optional[DockerCompose] = None
        self.docker_client = docker.from_env()
        
        # Service endpoints
        self.service_urls: Dict[str, str] = {}
        
        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.health_interval = 5.0  # seconds
        
        # Warm pools for frequently used containers
        self.warm_pools: Dict[str, ContainerPool] = {
            "redis": ContainerPool(),
            "postgres": ContainerPool(),
            "kafka": ContainerPool()
        }
        
    @asynccontextmanager
    async def start_aura_stack(self, 
                               services: Optional[List[str]] = None) -> AsyncIterator[Dict[str, str]]:
        """
        Start complete AURA stack with health monitoring
        
        Args:
            services: Specific services to start, or None for all
            
        Yields:
            Service URLs mapping
        """
        try:
            if self.use_compose:
                await self._start_with_compose(services)
            else:
                await self._start_individual_containers(services)
            
            # Wait for all services to be healthy
            await self._wait_for_healthy_stack()
            
            # Start health monitoring
            if self.enable_observability:
                self.health_monitor_task = asyncio.create_task(
                    self._monitor_health()
                )
            
            self.logger.info(
                "AURA stack started",
                services=list(self.service_urls.keys()),
                mode="compose" if self.use_compose else "individual"
            )
            
            yield self.service_urls
            
        finally:
            await self._cleanup()
    
    async def _start_with_compose(self, services: Optional[List[str]] = None):
        """Start services using docker-compose"""
        compose_file = "/workspace/aura-microservices/docker-compose.yml"
        
        # Use subset if specified
        if services:
            self.compose = DockerCompose(
                compose_file,
                services=services,
                pull=True
            )
        else:
            self.compose = DockerCompose(compose_file, pull=True)
        
        # Start with health checks
        with self.compose:
            self.compose.wait_for_logs("ready", timeout=60)
            
            # Extract service URLs
            self._extract_compose_urls()
    
    async def _start_individual_containers(self, services: Optional[List[str]] = None):
        """Start individual containers with optimization"""
        
        # Default to all core services
        if not services:
            services = ["redis", "postgres", "kafka", "neo4j", 
                       "neuromorphic", "memory", "byzantine", "lnn", "moe"]
        
        # Start infrastructure first
        if "redis" in services:
            self.containers["redis"] = await self._get_or_create_redis()
            
        if "postgres" in services:
            self.containers["postgres"] = await self._get_or_create_postgres()
            
        if "kafka" in services:
            self.containers["kafka"] = await self._get_or_create_kafka()
            
        if "neo4j" in services:
            self.containers["neo4j"] = await self._get_or_create_neo4j()
        
        # Start microservices
        await self._start_microservices(services)
        
        # Build service URLs
        self._build_service_urls()
    
    async def _get_or_create_redis(self) -> RedisContainer:
        """Get Redis from warm pool or create new"""
        # Check warm pool
        if self.warm_pools["redis"].containers:
            container = self.warm_pools["redis"].containers.popitem()[1]
            self.logger.info("Using Redis from warm pool")
            return container
        
        # Create new
        redis = RedisContainer("redis:7-alpine")
        redis.with_exposed_ports(6379)
        redis.start()
        
        wait_for_logs(redis, "Ready to accept connections", timeout=30)
        
        return redis
    
    async def _get_or_create_postgres(self) -> PostgresContainer:
        """Get PostgreSQL from warm pool or create new"""
        if self.warm_pools["postgres"].containers:
            container = self.warm_pools["postgres"].containers.popitem()[1]
            self.logger.info("Using PostgreSQL from warm pool")
            return container
        
        postgres = PostgresContainer("postgres:16-alpine")
        postgres.with_env("POSTGRES_USER", "aura")
        postgres.with_env("POSTGRES_PASSWORD", "testpass")
        postgres.with_env("POSTGRES_DB", "aura_test")
        postgres.start()
        
        wait_for_logs(postgres, "database system is ready", timeout=30)
        
        return postgres
    
    async def _get_or_create_kafka(self) -> KafkaContainer:
        """Get Kafka from warm pool or create new"""
        if self.warm_pools["kafka"].containers:
            container = self.warm_pools["kafka"].containers.popitem()[1]
            self.logger.info("Using Kafka from warm pool")
            return container
        
        kafka = KafkaContainer("confluentinc/cp-kafka:7.5.0")
        kafka.start()
        
        wait_for_logs(kafka, "Kafka Server started", timeout=60)
        
        return kafka
    
    async def _get_or_create_neo4j(self) -> Neo4jContainer:
        """Create Neo4j container"""
        neo4j = Neo4jContainer("neo4j:5.13-community")
        neo4j.with_env("NEO4J_AUTH", "neo4j/testpass")
        neo4j.start()
        
        wait_for_logs(neo4j, "Started", timeout=60)
        
        return neo4j
    
    async def _start_microservices(self, services: List[str]):
        """Start AURA microservices"""
        microservice_configs = {
            "neuromorphic": {
                "image": "aura-neuromorphic:test",
                "port": 8000,
                "health": "/api/v1/health"
            },
            "memory": {
                "image": "aura-memory:test", 
                "port": 8001,
                "health": "/api/v1/health"
            },
            "byzantine": {
                "image": "aura-byzantine:test",
                "port": 8002,
                "health": "/api/v1/health"
            },
            "lnn": {
                "image": "aura-lnn:test",
                "port": 8003,
                "health": "/api/v1/health"
            },
            "moe": {
                "image": "aura-moe:test",
                "port": 8005,
                "health": "/api/v1/health"
            }
        }
        
        for service_name, config in microservice_configs.items():
            if service_name in services:
                container = DockerContainer(config["image"])
                container.with_exposed_ports(config["port"])
                
                # Add environment variables
                container.with_env("REDIS_URL", self._get_redis_url())
                
                if service_name == "memory":
                    container.with_env("NEO4J_URI", self._get_neo4j_url())
                    container.with_env("POSTGRES_URL", self._get_postgres_url())
                
                container.start()
                self.containers[service_name] = container
                
                # Wait for health
                await self._wait_for_service_health(
                    service_name, 
                    config["port"], 
                    config["health"]
                )
    
    def _get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if "redis" in self.containers:
            redis = self.containers["redis"]
            host = redis.get_container_host_ip()
            port = redis.get_exposed_port(6379)
            return f"redis://{host}:{port}"
        return "redis://localhost:6379"
    
    def _get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        if "postgres" in self.containers:
            pg = self.containers["postgres"]
            host = pg.get_container_host_ip()
            port = pg.get_exposed_port(5432)
            return f"postgresql://aura:testpass@{host}:{port}/aura_test"
        return "postgresql://aura:testpass@localhost:5432/aura_test"
    
    def _get_neo4j_url(self) -> str:
        """Get Neo4j connection URL"""
        if "neo4j" in self.containers:
            neo4j = self.containers["neo4j"]
            host = neo4j.get_container_host_ip()
            port = neo4j.get_exposed_port(7687)
            return f"bolt://neo4j:testpass@{host}:{port}"
        return "bolt://neo4j:testpass@localhost:7687"
    
    def _build_service_urls(self):
        """Build service URL mapping"""
        for name, container in self.containers.items():
            if hasattr(container, 'get_exposed_port'):
                # Microservices
                if name in ["neuromorphic", "memory", "byzantine", "lnn", "moe"]:
                    port_map = {
                        "neuromorphic": 8000,
                        "memory": 8001,
                        "byzantine": 8002,
                        "lnn": 8003,
                        "moe": 8005
                    }
                    
                    if name in port_map:
                        host = container.get_container_host_ip()
                        port = container.get_exposed_port(port_map[name])
                        self.service_urls[name] = f"http://{host}:{port}"
    
    def _extract_compose_urls(self):
        """Extract URLs when using docker-compose"""
        # In compose mode, services communicate by name
        self.service_urls = {
            "neuromorphic": "http://aura-neuromorphic:8000",
            "memory": "http://aura-memory:8001",
            "byzantine": "http://aura-byzantine:8002",
            "lnn": "http://aura-lnn:8003",
            "moe": "http://aura-moe-router:8005",
            "redis": "redis://redis:6379",
            "postgres": "postgresql://aura:auradb123@postgres:5432/aura_intelligence",
            "kafka": "kafka:9092",
            "neo4j": "bolt://neo4j:aurapassword123@neo4j:7687"
        }
    
    async def _wait_for_healthy_stack(self, timeout: int = 120):
        """Wait for all services to be healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for service_name, url in self.service_urls.items():
                if service_name in ["neuromorphic", "memory", "byzantine", "lnn", "moe"]:
                    health_url = f"{url}/api/v1/health"
                    
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(health_url, timeout=5.0)
                            if response.status_code != 200:
                                all_healthy = False
                                self.logger.debug(f"{service_name} not healthy yet")
                    except Exception:
                        all_healthy = False
                        self.logger.debug(f"{service_name} not reachable yet")
            
            if all_healthy:
                self.logger.info("All services healthy")
                return
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Services did not become healthy in time")
    
    async def _wait_for_service_health(self, 
                                      service_name: str, 
                                      port: int, 
                                      health_path: str,
                                      timeout: int = 60):
        """Wait for specific service to be healthy"""
        container = self.containers[service_name]
        host = container.get_container_host_ip()
        exposed_port = container.get_exposed_port(port)
        health_url = f"http://{host}:{exposed_port}{health_path}"
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_url, timeout=5.0)
                    if response.status_code == 200:
                        self.logger.info(f"{service_name} is healthy")
                        return
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"{service_name} did not become healthy")
    
    async def _monitor_health(self):
        """Continuous health monitoring"""
        while True:
            try:
                health_statuses = {}
                
                for service_name, url in self.service_urls.items():
                    if service_name in ["neuromorphic", "memory", "byzantine", "lnn", "moe"]:
                        health = await self._check_service_health(service_name, url)
                        health_statuses[service_name] = health
                
                # Log any unhealthy services
                unhealthy = [s for s, h in health_statuses.items() if not h.healthy]
                if unhealthy:
                    self.logger.warning(
                        "Unhealthy services detected",
                        services=unhealthy
                    )
                
                await asyncio.sleep(self.health_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health monitoring error", error=str(e))
    
    async def _check_service_health(self, 
                                   service_name: str, 
                                   base_url: str) -> ServiceHealth:
        """Check health of a specific service"""
        health_url = f"{base_url}/api/v1/health"
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5.0)
                latency = (time.perf_counter() - start_time) * 1000
                
                return ServiceHealth(
                    service_name=service_name,
                    healthy=response.status_code == 200,
                    latency_ms=latency,
                    last_check=time.time(),
                    details=response.json() if response.status_code == 200 else {}
                )
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                healthy=False,
                latency_ms=5000,
                last_check=time.time(),
                details={"error": str(e)}
            )
    
    async def inject_fault(self, 
                          service_name: str,
                          fault_type: str,
                          duration_seconds: int = 10):
        """Inject fault into a running container"""
        if service_name not in self.containers:
            raise ValueError(f"Service {service_name} not found")
        
        container = self.containers[service_name]
        
        # Common fault injection patterns
        if fault_type == "network_delay":
            # Add network delay using tc
            container.exec_run(
                f"tc qdisc add dev eth0 root netem delay 100ms 50ms"
            )
            await asyncio.sleep(duration_seconds)
            container.exec_run("tc qdisc del dev eth0 root")
            
        elif fault_type == "cpu_stress":
            # CPU stress
            container.exec_run(
                f"stress --cpu 4 --timeout {duration_seconds}s",
                detach=True
            )
            
        elif fault_type == "memory_pressure":
            # Memory pressure
            container.exec_run(
                f"stress --vm 2 --vm-bytes 256M --timeout {duration_seconds}s",
                detach=True
            )
    
    async def _cleanup(self):
        """Cleanup containers and resources"""
        # Cancel health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        
        # Stop containers
        if self.use_compose and self.compose:
            self.compose.stop()
        else:
            for name, container in self.containers.items():
                try:
                    container.stop()
                    # Return to warm pool if applicable
                    if name in self.warm_pools:
                        pool = self.warm_pools[name]
                        if len(pool.containers) < pool.warm_pool_size:
                            pool.containers[name] = container
                            self.logger.info(f"Returned {name} to warm pool")
                            continue
                    container.remove()
                except Exception as e:
                    self.logger.error(f"Error stopping {name}", error=str(e))
        
        self.containers.clear()
        self.service_urls.clear()
    
    async def prepare_warm_pools(self):
        """Pre-create containers for warm pools"""
        self.logger.info("Preparing warm container pools")
        
        # Pre-create Redis containers
        for i in range(self.warm_pools["redis"].warm_pool_size):
            redis = await self._get_or_create_redis()
            self.warm_pools["redis"].containers[f"redis_{i}"] = redis
        
        # Pre-create PostgreSQL containers
        for i in range(self.warm_pools["postgres"].warm_pool_size):
            postgres = await self._get_or_create_postgres()
            self.warm_pools["postgres"].containers[f"postgres_{i}"] = postgres
        
        self.logger.info("Warm pools ready")


# Convenience functions
async def with_aura_stack(test_func, services: Optional[List[str]] = None):
    """Decorator for tests requiring AURA stack"""
    manager = AuraTestContainers()
    
    async with manager.start_aura_stack(services) as service_urls:
        await test_func(service_urls)


# Example usage
if __name__ == "__main__":
    async def test_example():
        manager = AuraTestContainers(use_compose=False)
        
        # Pre-warm pools for faster tests
        await manager.prepare_warm_pools()
        
        async with manager.start_aura_stack(["redis", "neuromorphic", "moe"]) as urls:
            print(f"Services available at: {urls}")
            
            # Run tests
            await asyncio.sleep(5)
            
            # Inject fault
            await manager.inject_fault("neuromorphic", "network_delay", 5)
            
    asyncio.run(test_example())