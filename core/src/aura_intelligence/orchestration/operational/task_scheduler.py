"""
Adaptive Task Scheduler with Cognitive Load Management
=====================================================
Dynamically adjusts concurrency based on system cognitive load
Based on Jeeva AI 2025 patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
import time
from enum import IntEnum
from collections import deque
import heapq
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

# AURA imports
from ...components.registry import get_registry
try:
    from ...observability.prometheus_integration import MetricsCollector
except ImportError:
    # Fallback if MetricsCollector is not available
    MetricsCollector = None

import logging
logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Task:
    """Task definition"""
    task_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Scheduling
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    # Resource requirements
    estimated_cpu_ms: float = 100.0
    estimated_memory_mb: float = 10.0
    requires_gpu: bool = False
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue"""
        # Consider both priority and deadline
        if self.deadline and other.deadline:
            # Earlier deadline = higher priority
            return (self.priority, self.deadline) < (other.priority, other.deadline)
        elif self.deadline:
            return True  # Deadline tasks first
        elif other.deadline:
            return False
        else:
            return self.priority < other.priority


class CognitiveLoadManager:
    """
    Monitors and manages system cognitive load
    Adjusts concurrency limits dynamically
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Load metrics
        self.cpu_history = deque(maxlen=60)  # 1 minute
        self.memory_history = deque(maxlen=60)
        self.latency_history = deque(maxlen=100)
        
        # Cognitive metrics
        self.free_energy_history = deque(maxlen=30)
        self.consensus_time_history = deque(maxlen=30)
        
        # Load thresholds
        self.load_levels = {
            'low': {'cpu': 40, 'memory': 50, 'latency_ms': 50},
            'normal': {'cpu': 70, 'memory': 70, 'latency_ms': 100},
            'high': {'cpu': 85, 'memory': 85, 'latency_ms': 200},
            'critical': {'cpu': 95, 'memory': 95, 'latency_ms': 500}
        }
        
        # Start monitoring
        self._monitoring_task = None
        
        logger.info("Cognitive Load Manager initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'monitoring_interval': 1.0,  # seconds
            'smoothing_factor': 0.8,
            'prediction_window': 5,  # seconds ahead
            'enable_predictive': True
        }
        
    async def start_monitoring(self):
        """Start load monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop load monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            await asyncio.gather(self._monitoring_task, return_exceptions=True)
            
    async def _monitor_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)
                
                # Get cognitive metrics from registry
                cognitive_metrics = await self._get_cognitive_metrics()
                
                if 'free_energy' in cognitive_metrics:
                    self.free_energy_history.append(cognitive_metrics['free_energy'])
                    
                if 'consensus_time_ms' in cognitive_metrics:
                    self.consensus_time_history.append(cognitive_metrics['consensus_time_ms'])
                    
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _get_cognitive_metrics(self) -> Dict[str, float]:
        """Get cognitive metrics from system components"""
        # Would integrate with actual components
        return {
            'free_energy': np.random.rand() * 2.0,
            'consensus_time_ms': np.random.rand() * 100
        }
        
    def get_current_load(self) -> Dict[str, float]:
        """Get current system load"""
        return {
            'cpu': np.mean(self.cpu_history) if self.cpu_history else 0.0,
            'memory': np.mean(self.memory_history) if self.memory_history else 0.0,
            'latency_ms': np.mean(self.latency_history) if self.latency_history else 0.0,
            'free_energy': np.mean(self.free_energy_history) if self.free_energy_history else 0.0
        }
        
    def get_load_level(self) -> str:
        """Determine current load level"""
        current = self.get_current_load()
        
        # Check thresholds
        for level in ['critical', 'high', 'normal', 'low']:
            thresholds = self.load_levels[level]
            
            if (current['cpu'] >= thresholds['cpu'] or
                current['memory'] >= thresholds['memory'] or
                current['latency_ms'] >= thresholds['latency_ms']):
                return level
                
        return 'low'
        
    def predict_load(self, seconds_ahead: float = 5.0) -> Dict[str, float]:
        """Predict future load using simple linear regression"""
        if not self.config['enable_predictive'] or len(self.cpu_history) < 10:
            return self.get_current_load()
            
        # Simple trend prediction
        x = np.arange(len(self.cpu_history))
        
        # CPU trend
        cpu_values = list(self.cpu_history)
        cpu_trend = np.polyfit(x, cpu_values, 1)[0]
        predicted_cpu = cpu_values[-1] + cpu_trend * seconds_ahead
        
        # Memory trend
        memory_values = list(self.memory_history)
        memory_trend = np.polyfit(x, memory_values, 1)[0]
        predicted_memory = memory_values[-1] + memory_trend * seconds_ahead
        
        return {
            'cpu': np.clip(predicted_cpu, 0, 100),
            'memory': np.clip(predicted_memory, 0, 100),
            'latency_ms': self.get_current_load()['latency_ms'],
            'trend': 'increasing' if cpu_trend > 0 else 'decreasing'
        }
        
    def get_concurrency_recommendation(self) -> Dict[str, int]:
        """Recommend concurrency limits based on load"""
        load_level = self.get_load_level()
        
        recommendations = {
            'low': {'max_workers': 20, 'max_queue': 1000},
            'normal': {'max_workers': 10, 'max_queue': 500},
            'high': {'max_workers': 5, 'max_queue': 200},
            'critical': {'max_workers': 2, 'max_queue': 50}
        }
        
        return recommendations[load_level]
        
    def should_throttle(self) -> bool:
        """Check if should throttle new tasks"""
        load_level = self.get_load_level()
        return load_level in ['high', 'critical']
        
    def get_backpressure_signal(self) -> float:
        """Get backpressure signal (0-1, higher = more pressure)"""
        current = self.get_current_load()
        
        # Weighted combination
        cpu_pressure = current['cpu'] / 100.0
        memory_pressure = current['memory'] / 100.0
        latency_pressure = min(1.0, current['latency_ms'] / 500.0)
        
        # Add cognitive pressure
        cognitive_pressure = 0.0
        if current['free_energy'] > 1.0:
            cognitive_pressure = min(1.0, (current['free_energy'] - 1.0) / 2.0)
            
        # Weighted average
        backpressure = (
            0.3 * cpu_pressure +
            0.3 * memory_pressure +
            0.2 * latency_pressure +
            0.2 * cognitive_pressure
        )
        
        return np.clip(backpressure, 0.0, 1.0)


class AdaptiveTaskScheduler:
    """
    Task scheduler with cognitive load awareness
    Dynamically adjusts concurrency and scheduling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Task management
        self.pending_tasks: List[Task] = []  # Priority queue
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.task_dependencies: Dict[str, Set[str]] = {}
        
        # Load management
        self.load_manager = CognitiveLoadManager()
        
        # Worker pool
        self.max_workers = self.config['initial_max_workers']
        self.workers: List[asyncio.Task] = []
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        self.total_scheduled = 0
        self.total_completed = 0
        self.total_failed = 0
        
        # State
        self.running = False
        
        logger.info("Adaptive Task Scheduler initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'initial_max_workers': 10,
            'min_workers': 2,
            'max_workers': 50,
            'task_timeout': 300,  # 5 minutes
            'adaptation_interval': 10,  # seconds
            'enable_deadline_scheduling': True,
            'enable_dependency_resolution': True
        }
        
    async def start(self):
        """Start the scheduler"""
        self.running = True
        
        # Start load monitoring
        await self.load_manager.start_monitoring()
        
        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
            
        # Start adaptation loop
        asyncio.create_task(self._adaptation_loop())
        
        logger.info(f"Task Scheduler started with {self.max_workers} workers")
        
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        
        # Stop monitoring
        await self.load_manager.stop_monitoring()
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
            
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Task Scheduler stopped")
        
    async def schedule_task(self, task: Task) -> str:
        """Schedule a task for execution"""
        self.total_scheduled += 1
        
        # Check dependencies
        if task.depends_on and self.config['enable_dependency_resolution']:
            self.task_dependencies[task.task_id] = set(task.depends_on)
            
        # Check if should throttle
        if self.load_manager.should_throttle():
            # Apply backpressure
            backpressure = self.load_manager.get_backpressure_signal()
            
            if backpressure > 0.8 and task.priority > TaskPriority.HIGH:
                # Delay non-critical tasks
                task.scheduled_time = time.time() + backpressure * 10
                
        # Add to queue
        heapq.heappush(self.pending_tasks, task)
        
        # Emit metrics
        self.metrics_collector.increment(
            'scheduler_tasks_queued',
            labels={'priority': task.priority.name}
        )
        
        logger.debug(f"Scheduled task {task.task_id} with priority {task.priority.name}")
        
        return task.task_id
        
    async def _worker(self, worker_id: int):
        """Worker coroutine"""
        while self.running:
            try:
                # Get next task
                task = await self._get_next_task()
                
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Execute task
                await self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
                
    async def _get_next_task(self) -> Optional[Task]:
        """Get next task to execute"""
        if not self.pending_tasks:
            return None
            
        # Check deadline scheduling
        if self.config['enable_deadline_scheduling']:
            # Find urgent deadline tasks
            current_time = time.time()
            
            for i, task in enumerate(self.pending_tasks):
                if task.deadline and task.deadline - current_time < 60:  # 1 minute
                    # Remove and return urgent task
                    urgent_task = self.pending_tasks.pop(i)
                    heapq.heapify(self.pending_tasks)
                    return urgent_task
                    
        # Check dependencies
        for i, task in enumerate(self.pending_tasks):
            if self._can_run_task(task):
                # Remove and return
                runnable_task = self.pending_tasks.pop(i)
                heapq.heapify(self.pending_tasks)
                return runnable_task
                
        return None
        
    def _can_run_task(self, task: Task) -> bool:
        """Check if task can run (dependencies satisfied)"""
        if not self.config['enable_dependency_resolution']:
            return True
            
        if task.task_id not in self.task_dependencies:
            return True
            
        # Check if all dependencies completed
        deps = self.task_dependencies[task.task_id]
        
        for dep_id in deps:
            # Check if dependency is completed
            if not any(t.task_id == dep_id for t in self.completed_tasks):
                return False
                
        return True
        
    async def _execute_task(self, task: Task):
        """Execute a single task"""
        task.start_time = time.time()
        
        # Add to running
        task_handle = asyncio.create_task(self._run_task_with_timeout(task))
        self.running_tasks[task.task_id] = task_handle
        
        try:
            # Wait for completion
            await task_handle
            
            # Record success
            task.end_time = time.time()
            self.total_completed += 1
            self.completed_tasks.append(task)
            
            # Update load metrics
            latency_ms = (task.end_time - task.start_time) * 1000
            self.load_manager.latency_history.append(latency_ms)
            
            # Emit metrics
            self.metrics_collector.histogram(
                'scheduler_task_duration_ms',
                latency_ms,
                labels={'priority': task.priority.name, 'status': 'success'}
            )
            
            logger.debug(f"Task {task.task_id} completed in {latency_ms:.1f}ms")
            
        except Exception as e:
            # Record failure
            task.error = e
            task.end_time = time.time()
            self.total_failed += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.scheduled_time = time.time() + 2 ** task.retry_count  # Exponential backoff
                await self.schedule_task(task)
                
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            # Clean up
            self.running_tasks.pop(task.task_id, None)
            
            # Clean up dependencies
            self.task_dependencies.pop(task.task_id, None)
            
    async def _run_task_with_timeout(self, task: Task):
        """Run task with timeout"""
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                task.func(*task.args, **task.kwargs),
                timeout=self.config['task_timeout']
            )
            
            task.result = result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.task_id} timed out after {self.config['task_timeout']}s")
            
    async def _adaptation_loop(self):
        """Adapt concurrency based on load"""
        while self.running:
            try:
                # Get recommendation
                recommendation = self.load_manager.get_concurrency_recommendation()
                new_max_workers = recommendation['max_workers']
                
                # Adjust workers
                if new_max_workers > self.max_workers:
                    # Add workers
                    for i in range(self.max_workers, new_max_workers):
                        worker = asyncio.create_task(self._worker(i))
                        self.workers.append(worker)
                        
                    logger.info(f"Increased workers to {new_max_workers}")
                    
                elif new_max_workers < self.max_workers:
                    # Remove workers (gracefully)
                    to_remove = self.max_workers - new_max_workers
                    
                    for _ in range(to_remove):
                        if self.workers:
                            worker = self.workers.pop()
                            worker.cancel()
                            
                    logger.info(f"Decreased workers to {new_max_workers}")
                    
                self.max_workers = new_max_workers
                
                # Check predicted load
                predicted = self.load_manager.predict_load()
                
                if predicted['trend'] == 'increasing' and predicted['cpu'] > 80:
                    logger.warning(f"Load trending up: predicted CPU {predicted['cpu']:.1f}%")
                    
                await asyncio.sleep(self.config['adaptation_interval'])
                
            except Exception as e:
                logger.error(f"Adaptation error: {e}")
                await asyncio.sleep(30)
                
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        load = self.load_manager.get_current_load()
        load_level = self.load_manager.get_load_level()
        backpressure = self.load_manager.get_backpressure_signal()
        
        return {
            'workers': {
                'current': self.max_workers,
                'active': len(self.running_tasks),
                'config_max': self.config['max_workers']
            },
            'tasks': {
                'pending': len(self.pending_tasks),
                'running': len(self.running_tasks),
                'completed': self.total_completed,
                'failed': self.total_failed,
                'total_scheduled': self.total_scheduled
            },
            'load': {
                'level': load_level,
                'metrics': load,
                'backpressure': backpressure
            },
            'performance': {
                'success_rate': self.total_completed / max(self.total_scheduled, 1),
                'avg_latency_ms': np.mean(self.load_manager.latency_history) if self.load_manager.latency_history else 0
            }
        }
        
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for specific task to complete"""
        start_time = time.time()
        
        while True:
            # Check if completed
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    if task.error:
                        raise task.error
                    return task.result
                    
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
                
            await asyncio.sleep(0.1)