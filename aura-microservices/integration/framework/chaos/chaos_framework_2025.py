"""
Advanced Chaos Engineering Framework 2025
Production-grade fault injection with observability

Latest patterns:
- Continuous chaos in production
- Game day automation
- Hypothesis-driven experiments
- Observability-first approach
- AI-driven fault prediction
"""

import asyncio
import random
import time
from typing import Dict, Any, List, Optional, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
import httpx
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = structlog.get_logger()

# Chaos metrics
EXPERIMENTS_RUN = Counter('chaos_experiments_total', 'Total chaos experiments', ['type', 'target'])
BLAST_RADIUS = Histogram('chaos_blast_radius', 'Impact radius of chaos', ['experiment'])
RECOVERY_TIME = Histogram('chaos_recovery_seconds', 'Recovery time', ['experiment', 'service'])
HYPOTHESIS_VALIDATED = Counter('chaos_hypothesis_validated', 'Hypothesis validation', ['result'])


class FaultType(Enum):
    """2025 Fault taxonomy"""
    # Network faults
    NETWORK_PARTITION = "network_partition"
    NETWORK_DELAY = "network_delay"
    PACKET_LOSS = "packet_loss"
    BANDWIDTH_THROTTLE = "bandwidth_throttle"
    DNS_FAILURE = "dns_failure"
    
    # Service faults
    SERVICE_CRASH = "service_crash"
    SERVICE_HANG = "service_hang"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    
    # Data faults
    DATA_CORRUPTION = "data_corruption"
    SCHEMA_DRIFT = "schema_drift"
    INVALID_ENCODING = "invalid_encoding"
    
    # Time faults
    CLOCK_SKEW = "clock_skew"
    TIME_TRAVEL = "time_travel"
    
    # Cloud faults
    ZONE_FAILURE = "zone_failure"
    REGION_FAILURE = "region_failure"
    PROVIDER_OUTAGE = "provider_outage"
    
    # AI/ML specific
    MODEL_DRIFT = "model_drift"
    ADVERSARIAL_INPUT = "adversarial_input"
    POISONED_TRAINING_DATA = "poisoned_training"


@dataclass
class ChaosHypothesis:
    """Hypothesis for chaos experiment"""
    description: str
    steady_state_metrics: Dict[str, Any]
    expected_impact: str
    recovery_time_slo: timedelta
    blast_radius: List[str]  # Services expected to be affected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "steady_state": self.steady_state_metrics,
            "expected_impact": self.expected_impact,
            "recovery_slo_seconds": self.recovery_time_slo.total_seconds(),
            "blast_radius": self.blast_radius
        }


@dataclass
class ChaosExperiment:
    """Chaos experiment definition"""
    name: str
    hypothesis: ChaosHypothesis
    target_services: List[str]
    fault_specs: List[Dict[str, Any]]
    duration: timedelta
    rollback_on_failure: bool = True
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of chaos experiment"""
    experiment_name: str
    start_time: datetime
    end_time: datetime
    hypothesis_validated: bool
    steady_state_before: Dict[str, Any]
    steady_state_after: Dict[str, Any]
    actual_impact: List[str]
    recovery_time: timedelta
    errors: List[str] = field(default_factory=list)
    rollback_performed: bool = False
    observations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class SteadyStateProbe(Protocol):
    """Protocol for steady state probes"""
    
    async def measure(self, services: List[str]) -> Dict[str, Any]:
        """Measure steady state metrics"""
        ...
    
    async def validate(self, metrics: Dict[str, Any], threshold: Dict[str, Any]) -> bool:
        """Validate if steady state is maintained"""
        ...


class DefaultSteadyStateProbe:
    """Default implementation of steady state probe"""
    
    def __init__(self, service_urls: Dict[str, str]):
        self.service_urls = service_urls
        self.logger = logger.bind(probe="steady_state")
    
    async def measure(self, services: List[str]) -> Dict[str, Any]:
        """Measure service health and performance"""
        metrics = {
            "timestamp": time.time(),
            "services": {}
        }
        
        async with httpx.AsyncClient() as client:
            for service in services:
                if service in self.service_urls:
                    try:
                        start = time.perf_counter()
                        response = await client.get(
                            f"{self.service_urls[service]}/api/v1/health",
                            timeout=5.0
                        )
                        latency = (time.perf_counter() - start) * 1000
                        
                        metrics["services"][service] = {
                            "healthy": response.status_code == 200,
                            "latency_ms": latency,
                            "status_code": response.status_code
                        }
                        
                        if response.status_code == 200:
                            health_data = response.json()
                            metrics["services"][service].update(health_data)
                            
                    except Exception as e:
                        metrics["services"][service] = {
                            "healthy": False,
                            "error": str(e)
                        }
        
        # Aggregate metrics
        healthy_count = sum(1 for s in metrics["services"].values() if s.get("healthy", False))
        metrics["overall"] = {
            "healthy_services": healthy_count,
            "total_services": len(services),
            "health_percentage": healthy_count / max(len(services), 1) * 100,
            "avg_latency_ms": np.mean([
                s.get("latency_ms", 0) 
                for s in metrics["services"].values() 
                if "latency_ms" in s
            ])
        }
        
        return metrics
    
    async def validate(self, metrics: Dict[str, Any], threshold: Dict[str, Any]) -> bool:
        """Check if metrics meet threshold"""
        overall = metrics.get("overall", {})
        
        # Check health percentage
        if "min_health_percentage" in threshold:
            if overall.get("health_percentage", 0) < threshold["min_health_percentage"]:
                return False
        
        # Check latency
        if "max_latency_ms" in threshold:
            if overall.get("avg_latency_ms", float('inf')) > threshold["max_latency_ms"]:
                return False
        
        # Check specific service requirements
        if "required_services" in threshold:
            for service in threshold["required_services"]:
                if not metrics.get("services", {}).get(service, {}).get("healthy", False):
                    return False
        
        return True


class ChaosInjector:
    """
    Advanced chaos injection with multiple strategies
    """
    
    def __init__(self, container_manager=None):
        self.container_manager = container_manager
        self.active_faults: Dict[str, Any] = {}
        self.logger = logger.bind(component="chaos_injector")
    
    async def inject_fault(self,
                          target: str,
                          fault_type: FaultType,
                          duration: timedelta,
                          intensity: float = 0.5,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Inject a specific fault
        
        Returns fault_id for tracking
        """
        fault_id = f"{target}_{fault_type.value}_{int(time.time())}"
        
        self.logger.info(
            "Injecting fault",
            fault_id=fault_id,
            target=target,
            fault_type=fault_type.value,
            duration_seconds=duration.total_seconds()
        )
        
        EXPERIMENTS_RUN.labels(type=fault_type.value, target=target).inc()
        
        # Route to appropriate injection method
        if fault_type in [FaultType.NETWORK_DELAY, FaultType.NETWORK_PARTITION, FaultType.PACKET_LOSS]:
            await self._inject_network_fault(target, fault_type, duration, intensity)
        elif fault_type in [FaultType.CPU_SPIKE, FaultType.MEMORY_LEAK]:
            await self._inject_resource_fault(target, fault_type, duration, intensity)
        elif fault_type in [FaultType.SERVICE_CRASH, FaultType.SERVICE_HANG]:
            await self._inject_service_fault(target, fault_type, duration)
        elif fault_type == FaultType.CLOCK_SKEW:
            await self._inject_time_fault(target, duration, intensity)
        else:
            self.logger.warning(f"Unsupported fault type: {fault_type}")
        
        # Track active fault
        self.active_faults[fault_id] = {
            "target": target,
            "type": fault_type,
            "start_time": time.time(),
            "duration": duration.total_seconds(),
            "metadata": metadata or {}
        }
        
        # Schedule cleanup
        asyncio.create_task(self._cleanup_fault(fault_id, duration))
        
        return fault_id
    
    async def _inject_network_fault(self,
                                   target: str,
                                   fault_type: FaultType,
                                   duration: timedelta,
                                   intensity: float):
        """Inject network-related faults"""
        if not self.container_manager:
            self.logger.warning("No container manager, simulating fault")
            return
        
        if fault_type == FaultType.NETWORK_DELAY:
            # Add latency using tc (traffic control)
            delay_ms = int(100 * intensity)  # Up to 100ms
            jitter_ms = int(delay_ms * 0.3)
            
            if target in self.container_manager.containers:
                container = self.container_manager.containers[target]
                container.exec_run(
                    f"tc qdisc add dev eth0 root netem delay {delay_ms}ms {jitter_ms}ms"
                )
        
        elif fault_type == FaultType.PACKET_LOSS:
            # Add packet loss
            loss_percent = int(20 * intensity)  # Up to 20% loss
            
            if target in self.container_manager.containers:
                container = self.container_manager.containers[target]
                container.exec_run(
                    f"tc qdisc add dev eth0 root netem loss {loss_percent}%"
                )
        
        elif fault_type == FaultType.NETWORK_PARTITION:
            # Block traffic using iptables
            if target in self.container_manager.containers:
                container = self.container_manager.containers[target]
                # Block all incoming traffic
                container.exec_run("iptables -A INPUT -j DROP")
    
    async def _inject_resource_fault(self,
                                    target: str,
                                    fault_type: FaultType,
                                    duration: timedelta,
                                    intensity: float):
        """Inject resource-related faults"""
        if not self.container_manager:
            return
        
        if target not in self.container_manager.containers:
            return
        
        container = self.container_manager.containers[target]
        
        if fault_type == FaultType.CPU_SPIKE:
            # CPU stress
            cpu_count = int(4 * intensity)  # Up to 4 CPUs
            container.exec_run(
                f"stress --cpu {cpu_count} --timeout {int(duration.total_seconds())}s",
                detach=True
            )
        
        elif fault_type == FaultType.MEMORY_LEAK:
            # Memory stress
            mem_size = int(512 * intensity)  # Up to 512MB
            container.exec_run(
                f"stress --vm 1 --vm-bytes {mem_size}M --timeout {int(duration.total_seconds())}s",
                detach=True
            )
    
    async def _inject_service_fault(self,
                                   target: str,
                                   fault_type: FaultType,
                                   duration: timedelta):
        """Inject service-level faults"""
        if not self.container_manager:
            return
        
        if target not in self.container_manager.containers:
            return
        
        container = self.container_manager.containers[target]
        
        if fault_type == FaultType.SERVICE_CRASH:
            # Kill main process
            container.exec_run("pkill -9 python", detach=True)
            
            # Restart after duration
            async def restart():
                await asyncio.sleep(duration.total_seconds())
                container.restart()
            
            asyncio.create_task(restart())
        
        elif fault_type == FaultType.SERVICE_HANG:
            # Send SIGSTOP to pause process
            container.exec_run("pkill -STOP python")
            
            # Resume after duration
            async def resume():
                await asyncio.sleep(duration.total_seconds())
                container.exec_run("pkill -CONT python")
            
            asyncio.create_task(resume())
    
    async def _inject_time_fault(self,
                                target: str,
                                duration: timedelta,
                                intensity: float):
        """Inject time-related faults"""
        if not self.container_manager:
            return
        
        if target not in self.container_manager.containers:
            return
        
        container = self.container_manager.containers[target]
        
        # Change system time
        skew_seconds = int(3600 * intensity)  # Up to 1 hour
        container.exec_run(f"date -s '+{skew_seconds} seconds'")
        
        # Reset after duration
        async def reset_time():
            await asyncio.sleep(duration.total_seconds())
            container.exec_run(f"date -s '-{skew_seconds} seconds'")
        
        asyncio.create_task(reset_time())
    
    async def _cleanup_fault(self, fault_id: str, duration: timedelta):
        """Cleanup fault after duration"""
        await asyncio.sleep(duration.total_seconds())
        
        if fault_id in self.active_faults:
            fault_info = self.active_faults[fault_id]
            target = fault_info["target"]
            
            # Cleanup network rules
            if self.container_manager and target in self.container_manager.containers:
                container = self.container_manager.containers[target]
                
                # Remove tc rules
                container.exec_run("tc qdisc del dev eth0 root", detach=True)
                
                # Remove iptables rules
                container.exec_run("iptables -F", detach=True)
            
            del self.active_faults[fault_id]
            self.logger.info(f"Cleaned up fault {fault_id}")


class ChaosOrchestrator:
    """
    Orchestrate chaos experiments with hypothesis validation
    """
    
    def __init__(self,
                 service_urls: Dict[str, str],
                 injector: ChaosInjector,
                 steady_state_probe: Optional[SteadyStateProbe] = None):
        self.service_urls = service_urls
        self.injector = injector
        self.steady_state_probe = steady_state_probe or DefaultSteadyStateProbe(service_urls)
        self.logger = logger.bind(component="chaos_orchestrator")
        self.experiment_history: List[ExperimentResult] = []
    
    async def run_experiment(self, experiment: ChaosExperiment) -> ExperimentResult:
        """
        Run a complete chaos experiment
        """
        self.logger.info(
            f"Starting chaos experiment",
            name=experiment.name,
            dry_run=experiment.dry_run
        )
        
        start_time = datetime.now()
        result = ExperimentResult(
            experiment_name=experiment.name,
            start_time=start_time,
            end_time=start_time,  # Will update
            hypothesis_validated=False,
            steady_state_before={},
            steady_state_after={},
            actual_impact=[],
            recovery_time=timedelta(0)
        )
        
        try:
            # 1. Measure steady state before
            self.logger.info("Measuring steady state before experiment")
            result.steady_state_before = await self.steady_state_probe.measure(
                experiment.target_services
            )
            
            # Validate initial steady state
            if not await self.steady_state_probe.validate(
                result.steady_state_before,
                experiment.hypothesis.steady_state_metrics
            ):
                result.errors.append("System not in steady state before experiment")
                result.observations.append("Experiment aborted - initial state invalid")
                return result
            
            # 2. Inject faults
            if not experiment.dry_run:
                fault_ids = []
                for fault_spec in experiment.fault_specs:
                    fault_id = await self.injector.inject_fault(
                        target=fault_spec["target"],
                        fault_type=FaultType(fault_spec["type"]),
                        duration=timedelta(seconds=fault_spec.get("duration", 60)),
                        intensity=fault_spec.get("intensity", 0.5),
                        metadata=fault_spec.get("metadata", {})
                    )
                    fault_ids.append(fault_id)
                
                # 3. Monitor during experiment
                await self._monitor_experiment(experiment, result)
                
                # 4. Wait for experiment duration
                await asyncio.sleep(experiment.duration.total_seconds())
            
            # 5. Measure recovery
            recovery_start = time.time()
            recovered = False
            
            while time.time() - recovery_start < experiment.hypothesis.recovery_time_slo.total_seconds():
                current_state = await self.steady_state_probe.measure(experiment.target_services)
                
                if await self.steady_state_probe.validate(
                    current_state,
                    experiment.hypothesis.steady_state_metrics
                ):
                    recovered = True
                    result.recovery_time = timedelta(seconds=time.time() - recovery_start)
                    break
                
                await asyncio.sleep(5)  # Check every 5 seconds
            
            # 6. Final state measurement
            result.steady_state_after = await self.steady_state_probe.measure(
                experiment.target_services
            )
            
            # 7. Validate hypothesis
            result.hypothesis_validated = (
                recovered and
                result.recovery_time <= experiment.hypothesis.recovery_time_slo and
                set(result.actual_impact).issubset(set(experiment.hypothesis.blast_radius))
            )
            
            HYPOTHESIS_VALIDATED.labels(
                result="validated" if result.hypothesis_validated else "invalidated"
            ).inc()
            
            # Record observations
            if result.hypothesis_validated:
                result.observations.append("Hypothesis validated - system recovered within SLO")
            else:
                if not recovered:
                    result.observations.append("System did not recover to steady state")
                if result.recovery_time > experiment.hypothesis.recovery_time_slo:
                    result.observations.append(
                        f"Recovery time ({result.recovery_time.total_seconds():.1f}s) "
                        f"exceeded SLO ({experiment.hypothesis.recovery_time_slo.total_seconds():.1f}s)"
                    )
                if not set(result.actual_impact).issubset(set(experiment.hypothesis.blast_radius)):
                    unexpected = set(result.actual_impact) - set(experiment.hypothesis.blast_radius)
                    result.observations.append(
                        f"Unexpected services impacted: {unexpected}"
                    )
            
        except Exception as e:
            self.logger.error(f"Experiment failed", error=str(e))
            result.errors.append(str(e))
            
            # Rollback if enabled
            if experiment.rollback_on_failure and not experiment.dry_run:
                await self._rollback_experiment(experiment)
                result.rollback_performed = True
        
        finally:
            result.end_time = datetime.now()
            self.experiment_history.append(result)
            
            # Record metrics
            RECOVERY_TIME.labels(
                experiment=experiment.name,
                service="overall"
            ).observe(result.recovery_time.total_seconds())
            
            BLAST_RADIUS.labels(
                experiment=experiment.name
            ).observe(len(result.actual_impact))
        
        return result
    
    async def _monitor_experiment(self, 
                                 experiment: ChaosExperiment,
                                 result: ExperimentResult):
        """Monitor system during experiment"""
        monitor_task = asyncio.create_task(
            self._continuous_monitoring(experiment, result)
        )
        
        # Let monitoring run for experiment duration
        await asyncio.sleep(experiment.duration.total_seconds())
        
        # Cancel monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    async def _continuous_monitoring(self,
                                   experiment: ChaosExperiment,
                                   result: ExperimentResult):
        """Continuously monitor services during experiment"""
        impacted_services = set()
        
        while True:
            try:
                # Measure current state
                current_state = await self.steady_state_probe.measure(
                    experiment.target_services + experiment.hypothesis.blast_radius
                )
                
                # Check for impacted services
                for service, metrics in current_state.get("services", {}).items():
                    if not metrics.get("healthy", True):
                        impacted_services.add(service)
                    elif "latency_ms" in metrics:
                        # Check for degraded performance
                        baseline = result.steady_state_before.get("services", {}).get(service, {})
                        baseline_latency = baseline.get("latency_ms", 0)
                        
                        if baseline_latency > 0:
                            degradation = metrics["latency_ms"] / baseline_latency
                            if degradation > 2.0:  # 2x latency increase
                                impacted_services.add(service)
                
                result.actual_impact = list(impacted_services)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Monitoring error", error=str(e))
    
    async def _rollback_experiment(self, experiment: ChaosExperiment):
        """Rollback experiment by restoring services"""
        self.logger.info("Rolling back experiment")
        
        # Restart affected services
        if hasattr(self.injector, 'container_manager'):
            for service in experiment.target_services:
                if service in self.injector.container_manager.containers:
                    container = self.injector.container_manager.containers[service]
                    container.restart()
    
    async def run_game_day(self, experiments: List[ChaosExperiment]) -> Dict[str, Any]:
        """
        Run a game day with multiple experiments
        """
        self.logger.info(f"Starting game day with {len(experiments)} experiments")
        
        game_day_results = {
            "start_time": datetime.now(),
            "experiments": [],
            "summary": {
                "total": len(experiments),
                "validated": 0,
                "failed": 0,
                "observations": []
            }
        }
        
        for experiment in experiments:
            # Run experiment
            result = await self.run_experiment(experiment)
            
            game_day_results["experiments"].append({
                "name": experiment.name,
                "hypothesis": experiment.hypothesis.to_dict(),
                "result": {
                    "validated": result.hypothesis_validated,
                    "recovery_time_seconds": result.recovery_time.total_seconds(),
                    "actual_impact": result.actual_impact,
                    "observations": result.observations,
                    "errors": result.errors
                }
            })
            
            # Update summary
            if result.hypothesis_validated:
                game_day_results["summary"]["validated"] += 1
            if result.errors:
                game_day_results["summary"]["failed"] += 1
            
            # Wait between experiments
            await asyncio.sleep(60)  # 1 minute cooldown
        
        game_day_results["end_time"] = datetime.now()
        game_day_results["summary"]["success_rate"] = (
            game_day_results["summary"]["validated"] / 
            game_day_results["summary"]["total"]
        )
        
        # Generate insights
        self._generate_game_day_insights(game_day_results)
        
        return game_day_results
    
    def _generate_game_day_insights(self, results: Dict[str, Any]):
        """Generate insights from game day results"""
        insights = []
        
        # Check for common failure patterns
        failed_services = []
        slow_recovery = []
        
        for exp in results["experiments"]:
            if not exp["result"]["validated"]:
                if exp["result"]["recovery_time_seconds"] > exp["hypothesis"]["recovery_slo_seconds"]:
                    slow_recovery.append(exp["name"])
                
                for service in exp["result"]["actual_impact"]:
                    if service not in exp["hypothesis"]["blast_radius"]:
                        failed_services.append(service)
        
        if failed_services:
            most_common = max(set(failed_services), key=failed_services.count)
            insights.append(
                f"Service '{most_common}' was unexpectedly impacted {failed_services.count(most_common)} times"
            )
        
        if slow_recovery:
            insights.append(
                f"{len(slow_recovery)} experiments had slower than expected recovery"
            )
        
        results["summary"]["observations"] = insights


# Example experiments
def create_example_experiments() -> List[ChaosExperiment]:
    """Create example chaos experiments"""
    
    # Network partition experiment
    network_experiment = ChaosExperiment(
        name="network_partition_neuromorphic",
        hypothesis=ChaosHypothesis(
            description="Neuromorphic service can handle network partition from dependencies",
            steady_state_metrics={
                "min_health_percentage": 80,
                "max_latency_ms": 100,
                "required_services": ["moe", "memory"]
            },
            expected_impact="Increased latency but no failures",
            recovery_time_slo=timedelta(seconds=30),
            blast_radius=["neuromorphic", "moe"]
        ),
        target_services=["neuromorphic"],
        fault_specs=[
            {
                "target": "neuromorphic",
                "type": "network_partition",
                "duration": 60,
                "intensity": 0.8
            }
        ],
        duration=timedelta(seconds=60)
    )
    
    # Cascading failure experiment
    cascade_experiment = ChaosExperiment(
        name="cascade_failure_byzantine",
        hypothesis=ChaosHypothesis(
            description="Byzantine consensus prevents cascade failures",
            steady_state_metrics={
                "min_health_percentage": 75,
                "required_services": ["byzantine", "moe"]
            },
            expected_impact="Degraded consensus but maintained quorum",
            recovery_time_slo=timedelta(seconds=120),
            blast_radius=["byzantine", "lnn", "moe"]
        ),
        target_services=["byzantine", "lnn"],
        fault_specs=[
            {
                "target": "byzantine",
                "type": "cpu_spike",
                "duration": 30,
                "intensity": 0.9
            },
            {
                "target": "lnn",
                "type": "service_hang",
                "duration": 45
            }
        ],
        duration=timedelta(seconds=90)
    )
    
    return [network_experiment, cascade_experiment]


if __name__ == "__main__":
    async def example_usage():
        # Service URLs
        service_urls = {
            "neuromorphic": "http://localhost:8000",
            "memory": "http://localhost:8001",
            "byzantine": "http://localhost:8002",
            "lnn": "http://localhost:8003",
            "moe": "http://localhost:8005"
        }
        
        # Create chaos components
        injector = ChaosInjector()
        orchestrator = ChaosOrchestrator(service_urls, injector)
        
        # Create experiments
        experiments = create_example_experiments()
        
        # Run game day
        results = await orchestrator.run_game_day(experiments)
        
        print(f"Game Day Results:")
        print(f"Success Rate: {results['summary']['success_rate']:.2%}")
        print(f"Insights: {results['summary']['observations']}")
    
    asyncio.run(example_usage())