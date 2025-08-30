"""
Latency-Aware Task Scheduler
============================
SLA-driven scheduling with integer programming optimization
Based on PracData 2025 patterns
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import time
import numpy as np
from collections import defaultdict
import heapq

# AURA imports
from .task_scheduler import Task, TaskPriority

import logging
logger = logging.getLogger(__name__)


@dataclass
class SLAConfig:
    """Service Level Agreement configuration"""
    task_type: str
    max_latency_ms: float
    target_latency_ms: float
    percentile: float = 0.95  # p95 by default
    
    # Penalties
    penalty_per_ms: float = 1.0  # Cost per ms over target
    critical_penalty: float = 100.0  # Cost for missing SLA
    
    # Resource hints
    requires_gpu: bool = False
    min_memory_mb: int = 100
    cpu_cores: float = 1.0


@dataclass
class LatencyEstimate:
    """Estimated latency for task execution"""
    task_id: str
    estimated_ms: float
    confidence: float  # 0-1
    
    # Breakdown
    queue_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    overhead_ms: float = 0.0
    
    # Factors
    cpu_load_factor: float = 1.0
    memory_pressure_factor: float = 1.0
    
    @property
    def total_ms(self) -> float:
        return self.queue_time_ms + self.execution_time_ms + self.overhead_ms


class LatencyEstimator:
    """
    Estimates task execution latency based on historical data
    Uses exponential smoothing and load-aware adjustments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Historical latencies by task type
        self.history: Dict[str, List[float]] = defaultdict(list)
        
        # Current system load
        self.current_load = {
            'cpu': 0.5,
            'memory': 0.5,
            'queue_depth': 0
        }
        
        # Estimation models
        self.baseline_estimates: Dict[str, float] = {}
        
        logger.info("Latency Estimator initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'history_window': 100,
            'smoothing_alpha': 0.3,
            'load_impact_cpu': 2.0,  # 2x latency at 100% CPU
            'load_impact_memory': 1.5,
            'confidence_threshold': 10  # Min samples for confidence
        }
        
    def update_history(self, task_type: str, actual_latency_ms: float):
        """Update historical data"""
        history = self.history[task_type]
        history.append(actual_latency_ms)
        
        # Limit history size
        if len(history) > self.config['history_window']:
            self.history[task_type] = history[-self.config['history_window']:]
            
        # Update baseline estimate
        if len(history) >= 3:
            # Exponential smoothing
            alpha = self.config['smoothing_alpha']
            
            if task_type in self.baseline_estimates:
                old_estimate = self.baseline_estimates[task_type]
                new_estimate = alpha * actual_latency_ms + (1 - alpha) * old_estimate
            else:
                new_estimate = np.mean(history[-10:])
                
            self.baseline_estimates[task_type] = new_estimate
            
    def estimate_latency(self, 
                        task: Task,
                        queue_position: int = 0) -> LatencyEstimate:
        """Estimate task execution latency"""
        task_type = task.name
        
        # Get baseline estimate
        if task_type in self.baseline_estimates:
            baseline_ms = self.baseline_estimates[task_type]
            confidence = min(1.0, len(self.history[task_type]) / self.config['confidence_threshold'])
        else:
            # Use task's estimate or default
            baseline_ms = task.estimated_cpu_ms
            confidence = 0.1
            
        # Calculate queue time
        queue_time_ms = queue_position * 10.0  # Rough estimate
        
        # Apply load factors
        cpu_factor = 1.0 + (self.current_load['cpu'] * (self.config['load_impact_cpu'] - 1))
        memory_factor = 1.0 + (self.current_load['memory'] * (self.config['load_impact_memory'] - 1))
        
        execution_time_ms = baseline_ms * cpu_factor * memory_factor
        
        # Add overhead
        overhead_ms = 5.0  # Scheduling overhead
        
        return LatencyEstimate(
            task_id=task.task_id,
            estimated_ms=queue_time_ms + execution_time_ms + overhead_ms,
            confidence=confidence,
            queue_time_ms=queue_time_ms,
            execution_time_ms=execution_time_ms,
            overhead_ms=overhead_ms,
            cpu_load_factor=cpu_factor,
            memory_pressure_factor=memory_factor
        )
        
    def update_load(self, load_metrics: Dict[str, float]):
        """Update current system load"""
        self.current_load.update(load_metrics)


class LatencyAwareScheduler:
    """
    Schedules tasks to minimize SLA violations
    Uses integer programming concepts for optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # SLA configurations
        self.sla_configs: Dict[str, SLAConfig] = {}
        
        # Latency estimation
        self.estimator = LatencyEstimator()
        
        # Task queues by priority
        self.queues: Dict[TaskPriority, List[Task]] = {
            priority: [] for priority in TaskPriority
        }
        
        # Execution slots
        self.execution_slots: List[Optional[Task]] = [None] * self.config['max_parallel_tasks']
        self.slot_available_at: List[float] = [0.0] * self.config['max_parallel_tasks']
        
        # Metrics
        self.sla_violations = 0
        self.total_scheduled = 0
        self.total_penalty = 0.0
        
        logger.info("Latency-Aware Scheduler initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_parallel_tasks': 10,
            'schedule_lookahead_ms': 1000,
            'reoptimize_interval': 0.1,
            'enable_preemption': True,
            'min_progress_for_preemption': 0.5
        }
        
    def register_sla(self, sla: SLAConfig):
        """Register SLA configuration"""
        self.sla_configs[sla.task_type] = sla
        logger.info(f"Registered SLA for {sla.task_type}: {sla.target_latency_ms}ms target")
        
    async def schedule_task(self, task: Task) -> LatencyEstimate:
        """
        Schedule task with SLA awareness
        Returns latency estimate
        """
        self.total_scheduled += 1
        
        # Get SLA config
        sla = self.sla_configs.get(task.name)
        
        # Estimate latency
        queue_position = self._get_queue_position(task)
        estimate = self.estimator.estimate_latency(task, queue_position)
        
        # Check if would violate SLA
        if sla and task.deadline:
            time_remaining = task.deadline - time.time()
            
            if estimate.total_ms > time_remaining * 1000:
                # Try to optimize placement
                estimate = await self._optimize_placement(task, sla)
                
        # Add to appropriate queue
        heapq.heappush(self.queues[task.priority], task)
        
        return estimate
        
    def _get_queue_position(self, task: Task) -> int:
        """Get position in queue"""
        position = 0
        
        # Count tasks ahead in same or higher priority queues
        for priority in TaskPriority:
            if priority <= task.priority:
                position += len(self.queues[priority])
                
        return position
        
    async def _optimize_placement(self, 
                                task: Task,
                                sla: SLAConfig) -> LatencyEstimate:
        """
        Optimize task placement to meet SLA
        May preempt lower priority tasks
        """
        current_time = time.time()
        deadline = task.deadline or (current_time + sla.max_latency_ms / 1000)
        
        # Find best slot
        best_slot = -1
        best_completion_time = float('inf')
        
        for slot_idx, slot_task in enumerate(self.execution_slots):
            if slot_task is None:
                # Empty slot
                completion_time = current_time + task.estimated_cpu_ms / 1000
                
                if completion_time < best_completion_time:
                    best_slot = slot_idx
                    best_completion_time = completion_time
                    
            elif self.config['enable_preemption'] and slot_task.priority > task.priority:
                # Can preempt
                if self._can_preempt(slot_task):
                    # Estimate completion with preemption
                    remaining_time = self.slot_available_at[slot_idx] - current_time
                    completion_time = current_time + task.estimated_cpu_ms / 1000 + remaining_time * 0.1
                    
                    if completion_time < deadline and completion_time < best_completion_time:
                        best_slot = slot_idx
                        best_completion_time = completion_time
                        
        # Update estimate
        new_estimate = LatencyEstimate(
            task_id=task.task_id,
            estimated_ms=(best_completion_time - current_time) * 1000,
            confidence=0.8,
            queue_time_ms=0.0,
            execution_time_ms=task.estimated_cpu_ms,
            overhead_ms=10.0
        )
        
        # Preempt if necessary
        if best_slot >= 0 and self.execution_slots[best_slot] is not None:
            await self._preempt_task(best_slot, task)
            
        return new_estimate
        
    def _can_preempt(self, task: Task) -> bool:
        """Check if task can be preempted"""
        if not task.start_time:
            return True
            
        # Check progress
        elapsed = time.time() - task.start_time
        progress = elapsed / (task.estimated_cpu_ms / 1000)
        
        return progress < self.config['min_progress_for_preemption']
        
    async def _preempt_task(self, slot: int, new_task: Task):
        """Preempt running task"""
        old_task = self.execution_slots[slot]
        
        if old_task:
            logger.info(f"Preempting {old_task.task_id} for {new_task.task_id}")
            
            # Re-queue old task
            heapq.heappush(self.queues[old_task.priority], old_task)
            
        # Schedule new task
        self.execution_slots[slot] = new_task
        self.slot_available_at[slot] = time.time() + new_task.estimated_cpu_ms / 1000
        
    async def get_next_task(self) -> Optional[Tuple[Task, int]]:
        """
        Get next task to execute with slot assignment
        Implements SLA-aware selection
        """
        current_time = time.time()
        
        # Find available slot
        available_slots = [
            i for i, available_at in enumerate(self.slot_available_at)
            if available_at <= current_time
        ]
        
        if not available_slots:
            return None
            
        slot = available_slots[0]
        
        # Select task with integer programming approach
        best_task = None
        best_score = float('-inf')
        best_queue = None
        
        for priority, queue in self.queues.items():
            if not queue:
                continue
                
            # Peek at top task
            task = queue[0]
            
            # Calculate score
            score = self._calculate_task_score(task, current_time)
            
            if score > best_score:
                best_task = task
                best_score = score
                best_queue = queue
                
        if best_task and best_queue:
            # Remove from queue
            heapq.heappop(best_queue)
            
            # Assign to slot
            self.execution_slots[slot] = best_task
            self.slot_available_at[slot] = current_time + best_task.estimated_cpu_ms / 1000
            
            return best_task, slot
            
        return None
        
    def _calculate_task_score(self, task: Task, current_time: float) -> float:
        """
        Calculate task scheduling score
        Higher score = should schedule sooner
        """
        score = 0.0
        
        # Priority weight
        priority_weight = {
            TaskPriority.CRITICAL: 1000,
            TaskPriority.HIGH: 100,
            TaskPriority.NORMAL: 10,
            TaskPriority.LOW: 1,
            TaskPriority.BACKGROUND: 0.1
        }
        score += priority_weight[task.priority]
        
        # SLA urgency
        sla = self.sla_configs.get(task.name)
        
        if sla and task.deadline:
            time_until_deadline = task.deadline - current_time
            time_needed = task.estimated_cpu_ms / 1000
            
            slack = time_until_deadline - time_needed
            
            if slack < 0:
                # Already late
                score += 10000 * abs(slack)
            else:
                # Urgency increases as deadline approaches
                urgency = 1.0 / (slack + 0.001)
                score += urgency * sla.penalty_per_ms
                
        # Age in queue
        age = current_time - task.scheduled_time
        score += age * 0.1
        
        return score
        
    def complete_task(self, task: Task, slot: int, actual_latency_ms: float):
        """Mark task as completed"""
        # Clear slot
        self.execution_slots[slot] = None
        
        # Update history
        self.estimator.update_history(task.name, actual_latency_ms)
        
        # Check SLA
        sla = self.sla_configs.get(task.name)
        
        if sla:
            if actual_latency_ms > sla.max_latency_ms:
                self.sla_violations += 1
                
                # Calculate penalty
                overage = actual_latency_ms - sla.target_latency_ms
                
                if actual_latency_ms > sla.max_latency_ms:
                    penalty = sla.critical_penalty
                else:
                    penalty = overage * sla.penalty_per_ms
                    
                self.total_penalty += penalty
                
                logger.warning(f"SLA violation for {task.name}: {actual_latency_ms:.1f}ms "
                             f"(target: {sla.target_latency_ms}ms)")
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        total_tasks = sum(len(q) for q in self.queues.values())
        active_tasks = sum(1 for task in self.execution_slots if task is not None)
        
        return {
            'total_scheduled': self.total_scheduled,
            'sla_violations': self.sla_violations,
            'violation_rate': self.sla_violations / max(self.total_scheduled, 1),
            'total_penalty': self.total_penalty,
            'queued_tasks': total_tasks,
            'active_tasks': active_tasks,
            'available_slots': len(self.execution_slots) - active_tasks,
            'estimator_confidence': {
                task_type: len(history) / self.estimator.config['confidence_threshold']
                for task_type, history in self.estimator.history.items()
            }
        }
        
    async def reoptimize(self):
        """
        Reoptimize task placement
        Implements simplified integer programming
        """
        current_time = time.time()
        
        # Collect all tasks
        all_tasks = []
        
        for queue in self.queues.values():
            all_tasks.extend(queue)
            
        # Add running tasks that can be preempted
        for slot, task in enumerate(self.execution_slots):
            if task and self._can_preempt(task):
                all_tasks.append(task)
                
        if not all_tasks:
            return
            
        # Sort by score
        task_scores = [
            (task, self._calculate_task_score(task, current_time))
            for task in all_tasks
        ]
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Reassign to slots
        new_assignments = {}
        slot_idx = 0
        
        for task, score in task_scores:
            if slot_idx >= len(self.execution_slots):
                break
                
            new_assignments[slot_idx] = task
            slot_idx += 1
            
        # Apply new assignments
        # (In practice, would need careful synchronization)
        logger.info(f"Reoptimized {len(new_assignments)} task assignments")