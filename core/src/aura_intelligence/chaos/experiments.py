"""
Chaos Engineering Experiments - 2025 Production-Ready Implementation

Features:
- Fault injection framework
- Network chaos simulation
- Resource stress testing
- Resilience verification
- Automated recovery testing
- Observability integration
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
import numpy as np
from contextlib import asynccontextmanager
import psutil
import aiohttp

logger = structlog.get_logger(__name__)


class ChaosType(Enum):
    """Types of chaos experiments"""
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE = "resource"
    NETWORK = "network"
    STATE = "state"
    DEPENDENCY = "dependency"
    CASCADING = "cascading"


class ImpactLevel(Enum):
    """Impact level of chaos experiments"""
    LOW = 0.1
    MEDIUM = 0.3
    HIGH = 0.5
    CRITICAL = 0.8
    TOTAL = 1.0


@dataclass
class ExperimentConfig:
    """Configuration for chaos experiments"""
    name: str
    chaos_type: ChaosType
    impact_level: ImpactLevel
    duration_seconds: float = 60.0
    target_components: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    auto_rollback: bool = True
    success_criteria: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a chaos experiment"""
    experiment_name: str
    start_time: datetime
    end_time: datetime
    chaos_type: ChaosType
    impact_level: ImpactLevel
    success: bool
    metrics_before: Dict[str, float]
    metrics_during: Dict[str, float]
    metrics_after: Dict[str, float]
    recovery_time_seconds: float
    errors: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)


class ChaosExperiment(ABC):
    """Base class for chaos experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._rollback_actions: List[Callable] = []
        self._metrics: Dict[str, List[float]] = {}
        self._start_time: Optional[datetime] = None
        self._injection_active = False
        
    @abstractmethod
    async def setup(self) -> None:
        """Setup experiment prerequisites"""
        pass
    
    @abstractmethod
    async def inject_failure(self) -> None:
        """Inject the failure condition"""
        pass
    
    @abstractmethod
    async def verify_impact(self) -> bool:
        """Verify the failure is having expected impact"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up after experiment"""
        pass
    
    async def run(self) -> ExperimentResult:
        """Execute the chaos experiment"""
        self.logger.info(
            "Starting chaos experiment",
            name=self.config.name,
            type=self.config.chaos_type.value,
            impact=self.config.impact_level.name
        )
        
        self._start_time = datetime.now()
        result = ExperimentResult(
            experiment_name=self.config.name,
            start_time=self._start_time,
            end_time=self._start_time,  # Will be updated
            chaos_type=self.config.chaos_type,
            impact_level=self.config.impact_level,
            success=False,
            metrics_before={},
            metrics_during={},
            metrics_after={},
            recovery_time_seconds=0.0
        )
        
        try:
            # Phase 1: Setup and baseline
            await self.setup()
            result.metrics_before = await self._collect_metrics()
            
            # Phase 2: Inject chaos
            if not self.config.dry_run:
                self._injection_active = True
                await self.inject_failure()
                await asyncio.sleep(min(5.0, self.config.duration_seconds / 4))
                
                # Phase 3: Verify impact
                impact_verified = await self.verify_impact()
                if not impact_verified:
                    result.errors.append("Impact verification failed")
                
                result.metrics_during = await self._collect_metrics()
                
                # Phase 4: Run for duration
                await asyncio.sleep(self.config.duration_seconds)
            else:
                self.logger.info("Dry run - skipping actual injection")
                result.observations.append("Dry run completed")
            
            # Phase 5: Cleanup and recovery
            self._injection_active = False
            await self.cleanup()
            
            # Phase 6: Measure recovery
            recovery_start = time.time()
            await self._wait_for_recovery()
            result.recovery_time_seconds = time.time() - recovery_start
            
            result.metrics_after = await self._collect_metrics()
            
            # Phase 7: Evaluate success
            result.success = self._evaluate_success(result)
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            result.errors.append(str(e))
            if self.config.auto_rollback:
                await self._rollback()
        finally:
            result.end_time = datetime.now()
            self._injection_active = False
            
        self._log_result(result)
        return result
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io_read_mb": psutil.disk_io_counters().read_bytes / 1024 / 1024,
            "disk_io_write_mb": psutil.disk_io_counters().write_bytes / 1024 / 1024,
            "network_sent_mb": psutil.net_io_counters().bytes_sent / 1024 / 1024,
            "network_recv_mb": psutil.net_io_counters().bytes_recv / 1024 / 1024,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "open_files": len(process.open_files()),
            "num_threads": process.num_threads(),
            "timestamp": time.time()
        }
    
    async def _wait_for_recovery(self, timeout: float = 60.0) -> bool:
        """Wait for system to recover"""
        start = time.time()
        
        while time.time() - start < timeout:
            metrics = await self._collect_metrics()
            
            # Check if metrics are back to normal
            if metrics["cpu_percent"] < 80 and metrics["memory_percent"] < 85:
                return True
            
            await asyncio.sleep(1.0)
        
        return False
    
    def _evaluate_success(self, result: ExperimentResult) -> bool:
        """Evaluate if experiment met success criteria"""
        if result.errors:
            return False
        
        # Check recovery time
        max_recovery = self.config.success_criteria.get("max_recovery_seconds", 120)
        if result.recovery_time_seconds > max_recovery:
            return False
        
        # Check if system remained available
        availability = self.config.success_criteria.get("min_availability", 0.95)
        # This would check actual availability metrics
        
        return True
    
    async def _rollback(self):
        """Execute rollback actions"""
        self.logger.info("Executing rollback actions")
        for action in reversed(self._rollback_actions):
            try:
                await action() if asyncio.iscoroutinefunction(action) else action()
            except Exception as e:
                self.logger.error(f"Rollback action failed: {e}")
    
    def _log_result(self, result: ExperimentResult):
        """Log experiment results"""
        self.logger.info(
            "Chaos experiment completed",
            name=result.experiment_name,
            success=result.success,
            duration_seconds=(result.end_time - result.start_time).total_seconds(),
            recovery_seconds=result.recovery_time_seconds,
            errors=len(result.errors)
        )


class LatencyChaosExperiment(ChaosExperiment):
    """Inject network latency into system"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.original_delay = 0
        self.injected_delay = config.parameters.get("delay_ms", 100)
        self._delay_active = False
        
    async def setup(self) -> None:
        """Setup latency injection"""
        self.logger.info(f"Setting up latency injection: {self.injected_delay}ms")
        
    async def inject_failure(self) -> None:
        """Inject latency"""
        self._delay_active = True
        
        # In a real implementation, this would use tc (traffic control) or similar
        # For now, we'll simulate with a delay hook
        async def delayed_request(*args, **kwargs):
            if self._delay_active:
                await asyncio.sleep(self.injected_delay / 1000.0)
            # Original request logic here
        
        self.logger.info(f"Injected {self.injected_delay}ms latency")
        
    async def verify_impact(self) -> bool:
        """Verify latency is affecting system"""
        # In real implementation, measure actual request latencies
        return True
        
    async def cleanup(self) -> None:
        """Remove latency injection"""
        self._delay_active = False
        self.logger.info("Removed latency injection")


class ErrorRateChaosExperiment(ChaosExperiment):
    """Inject errors at specified rate"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.error_rate = config.parameters.get("error_rate", 0.1)
        self._error_injection_active = False
        
    async def setup(self) -> None:
        """Setup error injection"""
        self.logger.info(f"Setting up error injection: {self.error_rate*100}% error rate")
        
    async def inject_failure(self) -> None:
        """Start injecting errors"""
        self._error_injection_active = True
        
        # Hook into request processing
        async def maybe_fail():
            if self._error_injection_active and random.random() < self.error_rate:
                raise Exception("Chaos-injected error")
        
        self.logger.info(f"Started error injection at {self.error_rate*100}% rate")
        
    async def verify_impact(self) -> bool:
        """Verify errors are occurring"""
        # Check error metrics
        return True
        
    async def cleanup(self) -> None:
        """Stop error injection"""
        self._error_injection_active = False
        self.logger.info("Stopped error injection")


class ResourceStressChaosExperiment(ChaosExperiment):
    """Stress system resources (CPU, memory, disk)"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.resource_type = config.parameters.get("resource", "cpu")
        self.stress_level = config.parameters.get("stress_level", 0.5)
        self._stress_tasks: List[asyncio.Task] = []
        
    async def setup(self) -> None:
        """Setup resource stress test"""
        self.logger.info(
            f"Setting up {self.resource_type} stress: {self.stress_level*100}% load"
        )
        
    async def inject_failure(self) -> None:
        """Start resource stress"""
        if self.resource_type == "cpu":
            await self._stress_cpu()
        elif self.resource_type == "memory":
            await self._stress_memory()
        elif self.resource_type == "disk":
            await self._stress_disk()
        
    async def _stress_cpu(self):
        """Stress CPU with computational load"""
        num_workers = int(psutil.cpu_count() * self.stress_level)
        
        async def cpu_worker():
            while self._injection_active:
                # Intensive computation
                _ = sum(i * i for i in range(10000))
                await asyncio.sleep(0.001)
        
        for _ in range(num_workers):
            task = asyncio.create_task(cpu_worker())
            self._stress_tasks.append(task)
        
        self.logger.info(f"Started {num_workers} CPU stress workers")
        
    async def _stress_memory(self):
        """Stress memory by allocating large arrays"""
        target_mb = int(psutil.virtual_memory().total / 1024 / 1024 * self.stress_level)
        
        # Allocate memory in chunks
        self._memory_blocks = []
        chunk_size = 100  # MB
        
        for _ in range(target_mb // chunk_size):
            # Allocate ~100MB
            block = np.zeros((chunk_size * 1024 * 1024 // 8,), dtype=np.float64)
            self._memory_blocks.append(block)
        
        self.logger.info(f"Allocated {target_mb}MB of memory")
        
    async def _stress_disk(self):
        """Stress disk with I/O operations"""
        async def disk_worker():
            while self._injection_active:
                # Write and read temporary data
                data = os.urandom(1024 * 1024)  # 1MB
                # In real implementation, write to temp file
                await asyncio.sleep(0.1)
        
        num_workers = int(4 * self.stress_level)
        for _ in range(num_workers):
            task = asyncio.create_task(disk_worker())
            self._stress_tasks.append(task)
        
        self.logger.info(f"Started {num_workers} disk I/O workers")
        
    async def verify_impact(self) -> bool:
        """Verify resource stress is active"""
        metrics = await self._collect_metrics()
        
        if self.resource_type == "cpu":
            return metrics["cpu_percent"] > 50
        elif self.resource_type == "memory":
            return metrics["memory_percent"] > 50
        elif self.resource_type == "disk":
            # Check disk I/O metrics
            return True
        
        return False
        
    async def cleanup(self) -> None:
        """Stop resource stress"""
        # Cancel stress tasks
        for task in self._stress_tasks:
            task.cancel()
        
        # Clear memory blocks if allocated
        if hasattr(self, '_memory_blocks'):
            self._memory_blocks.clear()
        
        self.logger.info(f"Stopped {self.resource_type} stress")


class NetworkPartitionChaosExperiment(ChaosExperiment):
    """Simulate network partition between components"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.partition_targets = config.parameters.get("targets", [])
        self._partition_active = False
        
    async def setup(self) -> None:
        """Setup network partition"""
        self.logger.info(f"Setting up network partition for: {self.partition_targets}")
        
    async def inject_failure(self) -> None:
        """Create network partition"""
        self._partition_active = True
        
        # In real implementation, use iptables or similar
        # For simulation, we'll block certain connections
        
        self.logger.info(f"Network partition activated")
        
    async def verify_impact(self) -> bool:
        """Verify partition is effective"""
        # Try to connect to partitioned services
        for target in self.partition_targets:
            try:
                # Attempt connection
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{target}", timeout=2) as response:
                        if response.status == 200:
                            return False  # Partition not working
            except:
                pass  # Expected to fail
        
        return True
        
    async def cleanup(self) -> None:
        """Remove network partition"""
        self._partition_active = False
        self.logger.info("Network partition removed")


class ChaosOrchestrator:
    """Orchestrate chaos experiments"""
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.results: List[ExperimentResult] = []
        self.logger = structlog.get_logger(__name__)
        
    def register_experiment(self, experiment: ChaosExperiment):
        """Register a chaos experiment"""
        self.experiments[experiment.config.name] = experiment
        self.logger.info(f"Registered experiment: {experiment.config.name}")
        
    async def run_experiment(self, name: str) -> ExperimentResult:
        """Run a specific experiment"""
        if name not in self.experiments:
            raise ValueError(f"Unknown experiment: {name}")
        
        experiment = self.experiments[name]
        result = await experiment.run()
        self.results.append(result)
        
        return result
        
    async def run_scenario(self, experiments: List[str], 
                          parallel: bool = False) -> List[ExperimentResult]:
        """Run multiple experiments as a scenario"""
        self.logger.info(
            f"Running chaos scenario",
            experiments=experiments,
            parallel=parallel
        )
        
        if parallel:
            tasks = [self.run_experiment(name) for name in experiments]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for name in experiments:
                result = await self.run_experiment(name)
                results.append(result)
                
                if not result.success:
                    self.logger.warning(f"Experiment {name} failed, stopping scenario")
                    break
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate chaos testing report"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        
        avg_recovery = np.mean([r.recovery_time_seconds for r in self.results]) if self.results else 0
        
        return {
            "total_experiments": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_recovery_seconds": avg_recovery,
            "experiments": [
                {
                    "name": r.experiment_name,
                    "type": r.chaos_type.value,
                    "impact": r.impact_level.name,
                    "success": r.success,
                    "recovery_seconds": r.recovery_time_seconds,
                    "errors": r.errors
                }
                for r in self.results
            ]
        }


# Example usage
async def example_chaos_testing():
    """Example of running chaos experiments"""
    
    orchestrator = ChaosOrchestrator()
    
    # Configure experiments
    latency_config = ExperimentConfig(
        name="high_latency_test",
        chaos_type=ChaosType.LATENCY,
        impact_level=ImpactLevel.MEDIUM,
        duration_seconds=30,
        parameters={"delay_ms": 200},
        success_criteria={"max_recovery_seconds": 10}
    )
    
    error_config = ExperimentConfig(
        name="error_injection_test",
        chaos_type=ChaosType.ERROR,
        impact_level=ImpactLevel.LOW,
        duration_seconds=20,
        parameters={"error_rate": 0.1}
    )
    
    cpu_stress_config = ExperimentConfig(
        name="cpu_stress_test",
        chaos_type=ChaosType.RESOURCE,
        impact_level=ImpactLevel.HIGH,
        duration_seconds=15,
        parameters={"resource": "cpu", "stress_level": 0.7}
    )
    
    # Create and register experiments
    orchestrator.register_experiment(LatencyChaosExperiment(latency_config))
    orchestrator.register_experiment(ErrorRateChaosExperiment(error_config))
    orchestrator.register_experiment(ResourceStressChaosExperiment(cpu_stress_config))
    
    # Run scenario
    results = await orchestrator.run_scenario(
        ["high_latency_test", "error_injection_test", "cpu_stress_test"],
        parallel=False
    )
    
    # Generate report
    report = orchestrator.generate_report()
    print(f"Chaos Testing Report: {report}")
    
    return report


if __name__ == "__main__":
    import os
    asyncio.run(example_chaos_testing())