"""
🔍 GPU Monitoring and Observability
===================================

Production-grade GPU monitoring for AURA Intelligence System.
Tracks GPU utilization, memory, temperature, and performance metrics.

Features:
- Real-time GPU metrics collection
- Multi-GPU support
- CUDA kernel profiling
- Tensor Core utilization
- Thermal monitoring
- Power consumption tracking
- Prometheus/Grafana integration
"""

import asyncio
import torch
import psutil
import time

# Try to import pynvml (optional)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from prometheus_client import Histogram, Counter, Gauge, Info
import numpy as np

# Try to initialize NVIDIA Management Library
NVML_AVAILABLE = False
if PYNVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except:
        NVML_AVAILABLE = False
    
logger = structlog.get_logger(__name__)


# Prometheus Metrics for GPU
GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id', 'device_name']
)

GPU_MEMORY_USED = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['device_id', 'device_name']
)

GPU_MEMORY_TOTAL = Gauge(
    'gpu_memory_total_bytes',
    'Total GPU memory in bytes',
    ['device_id', 'device_name']
)

GPU_TEMPERATURE = Gauge(
    'gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['device_id', 'device_name']
)

GPU_POWER_DRAW = Gauge(
    'gpu_power_draw_watts',
    'GPU power draw in watts',
    ['device_id', 'device_name']
)

GPU_CLOCK_SPEED = Gauge(
    'gpu_clock_speed_mhz',
    'GPU clock speed in MHz',
    ['device_id', 'device_name', 'clock_type']
)

CUDA_KERNEL_LAUNCHES = Counter(
    'cuda_kernel_launches_total',
    'Total CUDA kernel launches',
    ['kernel_name', 'device_id']
)

CUDA_KERNEL_DURATION = Histogram(
    'cuda_kernel_duration_seconds',
    'CUDA kernel execution duration',
    ['kernel_name', 'device_id'],
    buckets=[.0001, .0005, .001, .005, .01, .05, .1, .5, 1.0]
)

TENSOR_CORE_UTILIZATION = Gauge(
    'tensor_core_utilization_percent',
    'Tensor Core utilization percentage',
    ['device_id', 'device_name']
)


@dataclass
class GPUMetrics:
    """Container for GPU metrics"""
    device_id: int
    device_name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    gpu_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    pcie_throughput_mb: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    compute_mode: Optional[str] = None
    
    
@dataclass
class CUDAMetrics:
    """CUDA-specific metrics"""
    active_blocks: int
    active_warps: int
    kernel_launches: int
    memory_transfers: int
    tensor_core_usage: float
    fp16_gflops: float
    fp32_gflops: float
    int8_tops: float


class GPUMonitor:
    """
    Comprehensive GPU monitoring system.
    
    Features:
    - Real-time metrics collection
    - Multi-GPU support
    - CUDA profiling
    - Anomaly detection
    - Historical tracking
    """
    
    def __init__(self, 
                 sample_interval_s: float = 1.0,
                 history_size: int = 3600,  # 1 hour at 1s intervals
                 enable_profiling: bool = False):
        
        self.sample_interval = sample_interval_s
        self.history_size = history_size
        self.enable_profiling = enable_profiling
        
        # GPU device info
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_info: Dict[int, Dict[str, Any]] = {}
        
        # Metrics history
        self.metrics_history: Dict[int, List[GPUMetrics]] = {}
        self.cuda_metrics: Dict[int, CUDAMetrics] = {}
        
        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Initialize GPU info
        self._initialize_gpu_info()
        
        # CUDA profiler
        if self.enable_profiling and torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=self._handle_profiler_trace,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        else:
            self.profiler = None
            
    def _initialize_gpu_info(self):
        """Initialize GPU device information"""
        
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPUs available")
            return
            
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            
            self.gpu_info[i] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": props.total_memory / 1024 / 1024,
                "multiprocessors": props.multi_processor_count,
                "cuda_cores": self._estimate_cuda_cores(props),
                "tensor_cores": self._estimate_tensor_cores(props),
                "memory_bandwidth_gb": self._estimate_bandwidth(props),
            }
            
            self.metrics_history[i] = []
            
            logger.info(f"GPU {i}: {props.name}", **self.gpu_info[i])
            
    def _estimate_cuda_cores(self, props) -> int:
        """Estimate CUDA cores based on compute capability"""
        # Simplified estimation based on architecture
        cores_per_mp = {
            (3, 0): 192,  # Kepler
            (3, 5): 192,  # Kepler
            (5, 0): 128,  # Maxwell
            (5, 2): 128,  # Maxwell
            (6, 0): 64,   # Pascal
            (6, 1): 128,  # Pascal
            (7, 0): 64,   # Volta
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,  # Ampere
            (9, 0): 128,  # Hopper
        }
        
        cc = (props.major, props.minor)
        cores = cores_per_mp.get(cc, 64)
        return props.multi_processor_count * cores
        
    def _estimate_tensor_cores(self, props) -> int:
        """Estimate Tensor Cores based on compute capability"""
        # Tensor cores introduced in Volta (7.0)
        if props.major < 7:
            return 0
            
        tensor_cores_per_mp = {
            (7, 0): 8,    # Volta
            (7, 5): 8,    # Turing
            (8, 0): 4,    # Ampere (but more powerful)
            (8, 6): 4,    # Ampere
            (9, 0): 4,    # Hopper (but much more powerful)
        }
        
        cc = (props.major, props.minor)
        cores = tensor_cores_per_mp.get(cc, 0)
        return props.multi_processor_count * cores
        
    def _estimate_bandwidth(self, props) -> float:
        """Estimate memory bandwidth in GB/s"""
        # Rough estimation based on architecture
        bandwidth_map = {
            "V100": 900,
            "A100": 1555,
            "A6000": 768,
            "RTX 3090": 936,
            "RTX 4090": 1008,
            "H100": 3350,
        }
        
        for gpu_name, bandwidth in bandwidth_map.items():
            if gpu_name in props.name:
                return bandwidth
                
        return 500  # Default estimate
        
    async def start_monitoring(self):
        """Start GPU monitoring"""
        
        if self._is_monitoring:
            logger.warning("GPU monitoring already started")
            return
            
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.profiler:
            self.profiler.start()
            
        logger.info("GPU monitoring started")
        
    async def stop_monitoring(self):
        """Stop GPU monitoring"""
        
        if not self._is_monitoring:
            return
            
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
        if self.profiler:
            self.profiler.stop()
            
        logger.info("GPU monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self._is_monitoring:
            try:
                # Collect metrics for all GPUs
                for device_id in range(self.num_gpus):
                    metrics = await self._collect_gpu_metrics(device_id)
                    
                    # Update history
                    history = self.metrics_history[device_id]
                    history.append(metrics)
                    
                    # Trim history
                    if len(history) > self.history_size:
                        history.pop(0)
                        
                    # Update Prometheus metrics
                    self._update_prometheus_metrics(metrics)
                    
                    # Check for anomalies
                    anomalies = self._detect_anomalies(device_id)
                    if anomalies:
                        logger.warning(f"GPU {device_id} anomalies detected", **anomalies)
                        
                await asyncio.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(self.sample_interval)
                
    async def _collect_gpu_metrics(self, device_id: int) -> GPUMetrics:
        """Collect metrics for a specific GPU"""
        
        torch.cuda.set_device(device_id)
        device_name = self.gpu_info[device_id]["name"]
        
        # Basic PyTorch metrics
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024  # MB
        memory_reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024    # MB
        memory_total = self.gpu_info[device_id]["total_memory_mb"]
        
        # Initialize metrics
        metrics = GPUMetrics(
            device_id=device_id,
            device_name=device_name,
            utilization_percent=0.0,
            memory_used_mb=memory_allocated,
            memory_total_mb=memory_total,
            memory_percent=(memory_allocated / memory_total) * 100
        )
        
        # NVML metrics if available
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.utilization_percent = util.gpu
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics.temperature_c = temp
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                metrics.power_draw_w = power
                
                # Clocks
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                metrics.gpu_clock_mhz = gpu_clock
                metrics.memory_clock_mhz = mem_clock
                
                # Fan speed
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    metrics.fan_speed_percent = fan_speed
                except:
                    pass
                    
            except Exception as e:
                logger.debug(f"NVML error for GPU {device_id}: {e}")
                
        return metrics
        
    def _update_prometheus_metrics(self, metrics: GPUMetrics):
        """Update Prometheus metrics"""
        
        labels = {
            'device_id': str(metrics.device_id),
            'device_name': metrics.device_name
        }
        
        GPU_UTILIZATION.labels(**labels).set(metrics.utilization_percent)
        GPU_MEMORY_USED.labels(**labels).set(metrics.memory_used_mb * 1024 * 1024)
        GPU_MEMORY_TOTAL.labels(**labels).set(metrics.memory_total_mb * 1024 * 1024)
        
        if metrics.temperature_c is not None:
            GPU_TEMPERATURE.labels(**labels).set(metrics.temperature_c)
            
        if metrics.power_draw_w is not None:
            GPU_POWER_DRAW.labels(**labels).set(metrics.power_draw_w)
            
        if metrics.gpu_clock_mhz is not None:
            GPU_CLOCK_SPEED.labels(**labels, clock_type='gpu').set(metrics.gpu_clock_mhz)
            
        if metrics.memory_clock_mhz is not None:
            GPU_CLOCK_SPEED.labels(**labels, clock_type='memory').set(metrics.memory_clock_mhz)
            
    def _detect_anomalies(self, device_id: int) -> Dict[str, Any]:
        """Detect anomalies in GPU metrics"""
        
        history = self.metrics_history[device_id]
        if len(history) < 10:
            return {}
            
        anomalies = {}
        current = history[-1]
        
        # High temperature
        if current.temperature_c and current.temperature_c > 80:
            anomalies["high_temperature"] = current.temperature_c
            
        # High memory usage
        if current.memory_percent > 95:
            anomalies["high_memory_usage"] = current.memory_percent
            
        # Power throttling
        if current.power_draw_w:
            recent_power = [m.power_draw_w for m in history[-10:] if m.power_draw_w]
            if recent_power:
                avg_power = np.mean(recent_power)
                if current.power_draw_w < avg_power * 0.8:
                    anomalies["power_throttling"] = current.power_draw_w
                    
        # Utilization drops
        recent_util = [m.utilization_percent for m in history[-10:]]
        avg_util = np.mean(recent_util)
        if avg_util > 50 and current.utilization_percent < 10:
            anomalies["utilization_drop"] = current.utilization_percent
            
        return anomalies
        
    def _handle_profiler_trace(self, prof):
        """Handle profiler trace data"""
        
        # Export trace for analysis
        prof.export_chrome_trace(f"gpu_trace_{datetime.now().isoformat()}.json")
        
        # Extract key metrics
        cuda_time = sum([item.cuda_time for item in prof.key_averages()])
        kernel_count = len([item for item in prof.key_averages() if item.is_async])
        
        logger.info(f"Profiler trace: {cuda_time}us CUDA time, {kernel_count} kernels")
        
    async def get_current_metrics(self, device_id: Optional[int] = None) -> Union[GPUMetrics, Dict[int, GPUMetrics]]:
        """Get current GPU metrics"""
        
        if device_id is not None:
            history = self.metrics_history.get(device_id, [])
            return history[-1] if history else None
        else:
            return {
                i: history[-1] if history else None
                for i, history in self.metrics_history.items()
            }
            
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all GPU metrics"""
        
        summary = {
            "num_gpus": self.num_gpus,
            "total_memory_gb": 0,
            "total_used_memory_gb": 0,
            "avg_utilization": 0,
            "max_temperature": 0,
            "total_power_draw": 0,
            "gpus": []
        }
        
        utilizations = []
        
        for device_id in range(self.num_gpus):
            history = self.metrics_history.get(device_id, [])
            if not history:
                continue
                
            current = history[-1]
            
            gpu_summary = {
                "device_id": device_id,
                "name": current.device_name,
                "utilization": current.utilization_percent,
                "memory_used_gb": current.memory_used_mb / 1024,
                "memory_total_gb": current.memory_total_mb / 1024,
                "temperature": current.temperature_c,
                "power_draw": current.power_draw_w
            }
            
            summary["gpus"].append(gpu_summary)
            summary["total_memory_gb"] += gpu_summary["memory_total_gb"]
            summary["total_used_memory_gb"] += gpu_summary["memory_used_gb"]
            
            utilizations.append(current.utilization_percent)
            
            if current.temperature_c:
                summary["max_temperature"] = max(summary["max_temperature"], current.temperature_c)
                
            if current.power_draw_w:
                summary["total_power_draw"] += current.power_draw_w
                
        if utilizations:
            summary["avg_utilization"] = np.mean(utilizations)
            
        return summary
        
    def get_optimization_recommendations(self) -> List[str]:
        """Get GPU optimization recommendations"""
        
        recommendations = []
        
        for device_id in range(self.num_gpus):
            history = self.metrics_history.get(device_id, [])
            if len(history) < 100:
                continue
                
            recent = history[-100:]
            
            # Memory optimization
            avg_memory = np.mean([m.memory_percent for m in recent])
            if avg_memory > 90:
                recommendations.append(
                    f"GPU {device_id}: High memory usage ({avg_memory:.1f}%). "
                    "Consider gradient checkpointing or model parallelism."
                )
            elif avg_memory < 30:
                recommendations.append(
                    f"GPU {device_id}: Low memory usage ({avg_memory:.1f}%). "
                    "Consider increasing batch size."
                )
                
            # Utilization optimization
            avg_util = np.mean([m.utilization_percent for m in recent])
            if avg_util < 50:
                recommendations.append(
                    f"GPU {device_id}: Low utilization ({avg_util:.1f}%). "
                    "Check for CPU bottlenecks or I/O wait."
                )
                
            # Temperature warnings
            temps = [m.temperature_c for m in recent if m.temperature_c]
            if temps:
                max_temp = max(temps)
                if max_temp > 80:
                    recommendations.append(
                        f"GPU {device_id}: High temperature ({max_temp}°C). "
                        "Check cooling or reduce power limit."
                    )
                    
        return recommendations


# Singleton instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(
    sample_interval_s: float = 1.0,
    enable_profiling: bool = False
) -> GPUMonitor:
    """Get or create GPU monitor instance"""
    
    global _gpu_monitor
    
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(
            sample_interval_s=sample_interval_s,
            enable_profiling=enable_profiling
        )
        
    return _gpu_monitor


# Utility functions for quick metrics
async def get_gpu_utilization() -> Dict[int, float]:
    """Get current GPU utilization for all devices"""
    
    monitor = get_gpu_monitor()
    metrics = await monitor.get_current_metrics()
    
    return {
        device_id: m.utilization_percent if m else 0.0
        for device_id, m in metrics.items()
    }


async def get_gpu_memory_usage() -> Dict[int, Tuple[float, float]]:
    """Get current GPU memory usage (used, total) in GB"""
    
    monitor = get_gpu_monitor()
    metrics = await monitor.get_current_metrics()
    
    return {
        device_id: (
            m.memory_used_mb / 1024 if m else 0.0,
            m.memory_total_mb / 1024 if m else 0.0
        )
        for device_id, m in metrics.items()
    }