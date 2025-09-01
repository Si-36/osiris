"""
🚀 GPU-Accelerated Orchestration Adapter
========================================

Supercharges orchestration with Ray GPU scheduling for
distributed AI workloads.

Features:
- GPU-aware task placement
- Ray placement groups for GPU affinity
- Dynamic GPU allocation
- Multi-GPU task distribution
- GPU memory monitoring
- Automatic CPU fallback
"""

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import asyncio
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import structlog
from prometheus_client import Histogram, Counter, Gauge

from .base_adapter import BaseAdapter, HealthStatus, HealthMetrics, ComponentMetadata
from ..orchestration.unified_orchestration_engine import UnifiedOrchestrationEngine
from ..orchestration.enhancements.distributed_orchestration import (
    DistributedOrchestrationManager,
    DistributedTaskResult,
    OrchestrationWorker
)

logger = structlog.get_logger(__name__)

# Metrics
ORCHESTRATION_GPU_TIME = Histogram(
    'orchestration_gpu_seconds',
    'GPU task execution time',
    ['task_type', 'num_gpus', 'status']
)

GPU_UTILIZATION = Gauge(
    'orchestration_gpu_utilization',
    'Current GPU utilization percentage',
    ['gpu_id']
)

GPU_MEMORY_USED = Gauge(
    'orchestration_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)


@dataclass
class GPUOrchestrationConfig:
    """Configuration for GPU orchestration"""
    # GPU settings
    enable_gpu: bool = True
    default_gpus_per_task: float = 0.5  # Fractional GPUs supported
    max_gpus_per_task: int = 4
    
    # Placement strategy
    use_placement_groups: bool = True
    placement_group_strategy: str = "STRICT_PACK"  # or "PACK", "SPREAD"
    
    # Resource limits
    gpu_memory_fraction: float = 0.8  # Reserve 20% for system
    
    # Task routing
    gpu_task_types: List[str] = field(default_factory=lambda: [
        "neural_inference",
        "matrix_computation", 
        "embedding_generation",
        "model_training",
        "batch_processing"
    ])
    
    # Monitoring
    monitor_interval_seconds: int = 30


@ray.remote(num_gpus=1)
class GPUOrchestrationWorker(OrchestrationWorker):
    """
    GPU-enabled orchestration worker.
    Extends base worker with GPU-accelerated operations.
    """
    
    def __init__(self, worker_id: str, gpu_id: int = 0):
        super().__init__(worker_id, capabilities=["gpu", "cuda", "tensor_ops"])
        self.gpu_id = gpu_id
        
        # Initialize CUDA
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"GPU worker {worker_id} using CUDA device {gpu_id}")
        else:
            self.device = torch.device("cpu")
            logger.warning(f"GPU worker {worker_id} falling back to CPU")
            
        # Add GPU-specific handlers
        self.task_handlers.update({
            "neural_inference": self._handle_neural_inference,
            "matrix_computation": self._handle_matrix_computation,
            "embedding_generation": self._handle_embedding_generation,
        })
        
    async def _handle_neural_inference(self, task_data: Dict[str, Any]) -> Any:
        """GPU-accelerated neural network inference"""
        model_name = task_data.get("model", "default")
        input_data = task_data.get("input")
        
        # Convert to tensor and move to GPU
        if isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).to(self.device)
        else:
            tensor = torch.tensor(input_data).to(self.device)
            
        # Simulate neural inference
        with torch.no_grad():
            # Mock inference - in real system would load actual model
            output = torch.nn.functional.relu(tensor)
            output = torch.nn.functional.softmax(output, dim=-1)
            
        return output.cpu().numpy().tolist()
        
    async def _handle_matrix_computation(self, task_data: Dict[str, Any]) -> Any:
        """GPU-accelerated matrix operations"""
        operation = task_data.get("operation", "matmul")
        matrices = task_data.get("matrices", [])
        
        if len(matrices) < 2:
            raise ValueError("Need at least 2 matrices")
            
        # Move to GPU
        gpu_matrices = [
            torch.from_numpy(m).float().to(self.device) if isinstance(m, np.ndarray)
            else torch.tensor(m).float().to(self.device)
            for m in matrices
        ]
        
        # Perform operation
        if operation == "matmul":
            result = torch.matmul(gpu_matrices[0], gpu_matrices[1])
        elif operation == "add":
            result = gpu_matrices[0] + gpu_matrices[1]
        elif operation == "hadamard":
            result = gpu_matrices[0] * gpu_matrices[1]
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        return result.cpu().numpy().tolist()
        
    async def _handle_embedding_generation(self, task_data: Dict[str, Any]) -> Any:
        """GPU-accelerated embedding generation"""
        texts = task_data.get("texts", [])
        embedding_dim = task_data.get("dim", 768)
        
        # Simulate embedding generation
        # In real system, would use actual embedding model
        embeddings = []
        
        for text in texts:
            # Mock embedding based on text length
            length = len(text)
            embedding = torch.randn(embedding_dim).to(self.device)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            embeddings.append(embedding.cpu().numpy().tolist())
            
        return embeddings
        
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        if torch.cuda.is_available():
            return {
                "gpu_id": self.gpu_id,
                "allocated_mb": torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved(self.gpu_id) / 1024 / 1024,
                "utilization": torch.cuda.utilization(self.gpu_id) if hasattr(torch.cuda, 'utilization') else 0
            }
        return {"gpu_id": self.gpu_id, "available": False}


class GPUOrchestrationAdapter(BaseAdapter):
    """
    GPU-accelerated orchestration adapter using Ray placement groups.
    """
    
    def __init__(self,
                 orchestration_engine: UnifiedOrchestrationEngine,
                 config: GPUOrchestrationConfig):
        super().__init__(
            component_id="orchestration_gpu",
            metadata=ComponentMetadata(
                version="2.0.0",
                capabilities=["gpu_scheduling", "placement_groups", "multi_gpu"],
                dependencies={"ray", "torch", "orchestration_core"},
                tags=["gpu", "distributed", "production"]
            )
        )
        
        self.engine = orchestration_engine
        self.config = config
        self.placement_groups: Dict[str, Any] = {}
        self.gpu_workers: List[Any] = []
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize GPU orchestration"""
        if self._initialized:
            return
            
        await super().initialize()
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()
            
        # Get available GPUs
        resources = ray.available_resources()
        self.available_gpus = int(resources.get("GPU", 0))
        
        logger.info(f"GPU Orchestration: {self.available_gpus} GPUs available")
        
        # Create GPU workers
        if self.available_gpus > 0 and self.config.enable_gpu:
            await self._create_gpu_workers()
            
        # Start monitoring
        if self.config.monitor_interval_seconds > 0:
            asyncio.create_task(self._monitor_gpus())
            
        self._initialized = True
        
    async def _create_gpu_workers(self):
        """Create GPU-enabled workers"""
        # Create one worker per GPU
        for gpu_id in range(self.available_gpus):
            worker = GPUOrchestrationWorker.remote(
                worker_id=f"gpu_worker_{gpu_id}",
                gpu_id=gpu_id
            )
            self.gpu_workers.append(worker)
            
        logger.info(f"Created {len(self.gpu_workers)} GPU workers")
        
    async def execute_with_gpu(self,
                              task_type: str,
                              task_data: Dict[str, Any],
                              num_gpus: float = None) -> DistributedTaskResult:
        """
        Execute task with GPU acceleration.
        
        Args:
            task_type: Type of task to execute
            task_data: Task parameters
            num_gpus: Number of GPUs required (can be fractional)
        """
        if not self._initialized:
            await self.initialize()
            
        start_time = time.time()
        
        # Determine GPU requirement
        if num_gpus is None:
            num_gpus = self.config.default_gpus_per_task
            
        # Check if this task type should use GPU
        use_gpu = (
            self.config.enable_gpu and
            self.available_gpus > 0 and
            task_type in self.config.gpu_task_types and
            num_gpus > 0
        )
        
        try:
            if use_gpu:
                result = await self._execute_gpu_task(task_type, task_data, num_gpus)
            else:
                # Fallback to CPU execution
                result = await self.engine.execute_distributed(task_type, task_data)
                
            # Record metrics
            execution_time = time.time() - start_time
            ORCHESTRATION_GPU_TIME.labels(
                task_type=task_type,
                num_gpus=num_gpus if use_gpu else 0,
                status='success'
            ).observe(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"GPU task execution failed: {e}")
            ORCHESTRATION_GPU_TIME.labels(
                task_type=task_type,
                num_gpus=num_gpus if use_gpu else 0,
                status='error'
            ).observe(time.time() - start_time)
            raise
            
    async def _execute_gpu_task(self,
                               task_type: str,
                               task_data: Dict[str, Any],
                               num_gpus: float) -> DistributedTaskResult:
        """Execute task on GPU worker"""
        
        # For fractional GPUs or placement group strategy
        if self.config.use_placement_groups and num_gpus >= 1:
            return await self._execute_with_placement_group(task_type, task_data, num_gpus)
            
        # Simple execution on single GPU worker
        if self.gpu_workers:
            # Round-robin selection (could be improved with load balancing)
            worker_idx = hash(task_data.get("task_id", "")) % len(self.gpu_workers)
            worker = self.gpu_workers[worker_idx]
            
            result = await worker.execute_task.remote(task_type, task_data)
            return result
        else:
            raise RuntimeError("No GPU workers available")
            
    async def _execute_with_placement_group(self,
                                          task_type: str,
                                          task_data: Dict[str, Any],
                                          num_gpus: float) -> DistributedTaskResult:
        """Execute task using Ray placement group for multi-GPU"""
        
        # Create placement group for this task
        bundles = [{"GPU": 1} for _ in range(int(num_gpus))]
        if num_gpus % 1 > 0:
            # Handle fractional GPU
            bundles.append({"GPU": num_gpus % 1})
            
        pg = placement_group(bundles, strategy=self.config.placement_group_strategy)
        ray.get(pg.ready())
        
        try:
            # Create temporary worker with placement group
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0
            )
            
            @ray.remote(num_gpus=num_gpus, scheduling_strategy=strategy)
            def execute_multi_gpu_task(task_type, task_data):
                # This would run the actual multi-GPU computation
                # For now, simulate with sleep
                import time
                time.sleep(0.1)
                return DistributedTaskResult(
                    task_id=task_data.get("task_id", "multi_gpu"),
                    result={"status": "completed", "gpus_used": num_gpus},
                    execution_time=0.1,
                    worker_id="multi_gpu_worker",
                    metadata={"placement_group": True}
                )
                
            result = await execute_multi_gpu_task.remote(task_type, task_data)
            return result
            
        finally:
            # Clean up placement group
            remove_placement_group(pg)
            
    async def batch_execute_gpu(self,
                               tasks: List[Tuple[str, Dict[str, Any]]],
                               max_concurrent: int = None) -> List[DistributedTaskResult]:
        """
        Execute multiple tasks in parallel on GPUs.
        
        Optimizes GPU utilization by batching.
        """
        if max_concurrent is None:
            max_concurrent = self.available_gpus * 2  # Allow some oversubscription
            
        # Group tasks by type for better batching
        grouped_tasks = {}
        for task_type, task_data in tasks:
            if task_type not in grouped_tasks:
                grouped_tasks[task_type] = []
            grouped_tasks[task_type].append(task_data)
            
        results = []
        
        # Process each group
        for task_type, task_group in grouped_tasks.items():
            # Execute in batches
            for i in range(0, len(task_group), max_concurrent):
                batch = task_group[i:i + max_concurrent]
                
                # Execute batch in parallel
                batch_futures = [
                    self.execute_with_gpu(task_type, task_data)
                    for task_data in batch
                ]
                
                batch_results = await asyncio.gather(*batch_futures)
                results.extend(batch_results)
                
        return results
        
    async def _monitor_gpus(self):
        """Monitor GPU utilization and memory"""
        while self._initialized:
            try:
                if torch.cuda.is_available():
                    for gpu_id in range(self.available_gpus):
                        # Memory metrics
                        allocated = torch.cuda.memory_allocated(gpu_id)
                        GPU_MEMORY_USED.labels(gpu_id=str(gpu_id)).set(allocated)
                        
                        # Utilization (if available)
                        if hasattr(torch.cuda, 'utilization'):
                            util = torch.cuda.utilization(gpu_id)
                            GPU_UTILIZATION.labels(gpu_id=str(gpu_id)).set(util)
                            
                # Get stats from workers
                if self.gpu_workers:
                    stats_futures = [
                        worker.get_gpu_stats.remote() 
                        for worker in self.gpu_workers
                    ]
                    stats = await asyncio.gather(*stats_futures)
                    
                    for stat in stats:
                        logger.debug(f"GPU {stat['gpu_id']} stats", **stat)
                        
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                
            await asyncio.sleep(self.config.monitor_interval_seconds)
            
    async def scale_gpu_workers(self, target_workers: int):
        """Scale the number of GPU workers"""
        current = len(self.gpu_workers)
        
        if target_workers > current:
            # Add workers
            for i in range(current, min(target_workers, self.available_gpus)):
                worker = GPUOrchestrationWorker.remote(
                    worker_id=f"gpu_worker_{i}",
                    gpu_id=i % self.available_gpus  # Wrap around if needed
                )
                self.gpu_workers.append(worker)
                
        elif target_workers < current:
            # Remove workers
            workers_to_remove = self.gpu_workers[target_workers:]
            self.gpu_workers = self.gpu_workers[:target_workers]
            
            # Terminate removed workers
            for worker in workers_to_remove:
                ray.kill(worker)
                
        logger.info(f"Scaled GPU workers from {current} to {len(self.gpu_workers)}")
        
    async def get_gpu_recommendations(self, 
                                    workload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get GPU allocation recommendations for a workload.
        
        Analyzes workload characteristics to recommend optimal GPU usage.
        """
        recommendations = {
            "use_gpu": False,
            "num_gpus": 0,
            "placement_strategy": "PACK",
            "reasoning": []
        }
        
        # Check workload size
        data_size = workload.get("data_size", 0)
        task_type = workload.get("task_type", "")
        parallelism = workload.get("parallelism", 1)
        
        # GPU is beneficial for large data or specific task types
        if task_type in self.config.gpu_task_types:
            recommendations["use_gpu"] = True
            recommendations["reasoning"].append(f"Task type '{task_type}' benefits from GPU")
            
        if data_size > 10000:  # Large dataset
            recommendations["use_gpu"] = True
            recommendations["reasoning"].append(f"Large dataset ({data_size} items)")
            
        # Determine number of GPUs
        if recommendations["use_gpu"]:
            if data_size > 1000000 or parallelism > 4:
                recommendations["num_gpus"] = min(4, self.available_gpus)
                recommendations["placement_strategy"] = "SPREAD"
                recommendations["reasoning"].append("Very large workload - use multiple GPUs")
            elif data_size > 100000 or parallelism > 1:
                recommendations["num_gpus"] = min(2, self.available_gpus)
                recommendations["reasoning"].append("Medium workload - use 2 GPUs")
            else:
                recommendations["num_gpus"] = 1
                recommendations["reasoning"].append("Small-medium workload - single GPU sufficient")
                
        return recommendations
        
    async def health(self) -> HealthMetrics:
        """Get adapter health status"""
        metrics = HealthMetrics()
        
        try:
            # Check Ray status
            if ray.is_initialized():
                resources = ray.available_resources()
                metrics.resource_usage["ray_cpus"] = resources.get("CPU", 0)
                metrics.resource_usage["ray_gpus"] = resources.get("GPU", 0)
                metrics.resource_usage["ray_memory"] = resources.get("memory", 0) / 1e9  # GB
                
                # Check GPU workers
                alive_workers = 0
                for worker in self.gpu_workers:
                    try:
                        # Ping worker
                        ray.get(worker.get_gpu_stats.remote(), timeout=1)
                        alive_workers += 1
                    except:
                        pass
                        
                metrics.resource_usage["gpu_workers_alive"] = alive_workers
                metrics.resource_usage["gpu_workers_total"] = len(self.gpu_workers)
                
                if alive_workers < len(self.gpu_workers):
                    metrics.status = HealthStatus.DEGRADED
                    metrics.failure_predictions.append(
                        f"{len(self.gpu_workers) - alive_workers} GPU workers unresponsive"
                    )
                else:
                    metrics.status = HealthStatus.HEALTHY
                    
            else:
                metrics.status = HealthStatus.UNHEALTHY
                metrics.failure_predictions.append("Ray not initialized")
                
        except Exception as e:
            metrics.status = HealthStatus.UNHEALTHY
            metrics.failure_predictions.append(f"Health check failed: {e}")
            
        return metrics


# Factory function
def create_gpu_orchestration_adapter(
    orchestration_engine: UnifiedOrchestrationEngine,
    enable_gpu: bool = True,
    use_placement_groups: bool = True
) -> GPUOrchestrationAdapter:
    """Create GPU orchestration adapter with default config"""
    
    config = GPUOrchestrationConfig(
        enable_gpu=enable_gpu,
        use_placement_groups=use_placement_groups,
        default_gpus_per_task=1.0,
        gpu_task_types=[
            "neural_inference",
            "matrix_computation",
            "embedding_generation",
            "model_training",
            "batch_processing",
            "tda_analysis",  # For TDA GPU tasks
            "memory_indexing",  # For memory GPU tasks
        ]
    )
    
    return GPUOrchestrationAdapter(
        orchestration_engine=orchestration_engine,
        config=config
    )