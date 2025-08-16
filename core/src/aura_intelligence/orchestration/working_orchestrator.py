#!/usr/bin/env python3
"""
Working Orchestrator
Clean orchestrator for agent coordination
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Task definition."""
    task_id: str
    task_type: str
    context: Dict[str, Any]
    priority: str = "medium"
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class OrchestratorMetrics:
    """Orchestrator metrics."""
    tasks_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0

class WorkingOrchestrator:
    """Working orchestrator implementation."""
    
    def __init__(self, orchestrator_id: str):
        self.orchestrator_id = orchestrator_id
        self.agents: Dict[str, Any] = {}
        self.tasks: List[Task] = []
        self.metrics = OrchestratorMetrics()
        self.running = False
        
        print(f"🎼 Orchestrator initialized: {orchestrator_id}")
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register an agent with the orchestrator."""
        self.agents[agent_id] = agent
        print(f"📝 Agent registered: {agent_id}")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for processing."""
        self.tasks.append(task)
        print(f"📋 Task submitted: {task.task_id} ({task.task_type})")
        
        # Process task immediately if orchestrator is running
        if self.running:
            await self._process_task(task)
        
        return task.task_id
    
    async def start(self):
        """Start the orchestrator."""
        self.running = True
        print(f"🚀 Orchestrator started: {self.orchestrator_id}")
        
        # Process any pending tasks
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        for task in pending_tasks:
            await self._process_task(task)
    
    async def stop(self):
        """Stop the orchestrator."""
        self.running = False
        print(f"🛑 Orchestrator stopped: {self.orchestrator_id}")
    
    async def _process_task(self, task: Task):
        """Process a single task."""
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        
        try:
            # Select appropriate agent
            agent = self._select_agent_for_task(task)
            
            if not agent:
                raise Exception("No suitable agent available")
            
            # Execute task
            if hasattr(agent, 'make_decision'):
                result = await agent.make_decision(task.context)
            else:
                result = await agent.process_task(task.context)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(task, processing_time)
            
            print(f"✅ Task completed: {task.task_id} ({processing_time:.1f}ms)")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            self.metrics.tasks_failed += 1
            print(f"❌ Task failed: {task.task_id} - {str(e)}")
    
    def _select_agent_for_task(self, task: Task) -> Optional[Any]:
        """Select the best agent for a task."""
        # Simple selection logic - can be enhanced
        if task.task_type in ["analysis", "optimization", "monitoring", "scaling"]:
            # Look for council agents first
            for agent_id, agent in self.agents.items():
                if "council" in agent_id.lower():
                    return agent
        
        # Return first available agent
        if self.agents:
            return next(iter(self.agents.values()))
        
        return None
    
    def _update_metrics(self, task: Task, processing_time: float):
        """Update orchestrator metrics."""
        self.metrics.tasks_processed += 1
        
        if task.status == TaskStatus.COMPLETED:
            self.metrics.tasks_completed += 1
        
        # Update average processing time
        if self.metrics.average_processing_time == 0:
            self.metrics.average_processing_time = processing_time
        else:
            self.metrics.average_processing_time = (
                self.metrics.average_processing_time * 0.8 + processing_time * 0.2
            )
        
        # Update success rate
        if self.metrics.tasks_processed > 0:
            self.metrics.success_rate = self.metrics.tasks_completed / self.metrics.tasks_processed
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "orchestrator_id": self.orchestrator_id,
            "running": self.running,
            "registered_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "metrics": {
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": self.metrics.success_rate,
                "average_processing_time_ms": self.metrics.average_processing_time
            },
            "recent_tasks": [
                {
                    "task_id": t.task_id,
                    "task_type": t.task_type,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat()
                }
                for t in self.tasks[-10:]  # Last 10 tasks
            ]
        }

# Factory function
def create_orchestrator(orchestrator_id: str) -> WorkingOrchestrator:
    """Create a new orchestrator."""
    return WorkingOrchestrator(orchestrator_id)
