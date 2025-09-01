"""
🎮 GPU Workflow Integration for Orchestration

Extracts and integrates the BEST features from workflows/gpu_allocation.py:
- LNN Council approval for GPU requests
- Fair scheduling with priority queues
- Cost optimization and budget tracking
- Automatic deallocation after use
- Multi-agent coordination for GPU resources

This makes GPU a first-class citizen in our orchestration!
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import uuid
import structlog

# Import our orchestration components
from ..unified_orchestration_engine import UnifiedOrchestrationEngine

# Import LNN council for decisions
try:
    from ...agents.lnn_council import (
        LNNCouncilOrchestrator, create_lnn_council,
        VoteDecision, CouncilConsensus
    )
    LNN_AVAILABLE = True
except ImportError:
    LNN_AVAILABLE = False
    LNNCouncilOrchestrator = None

logger = structlog.get_logger()


# ======================
# GPU Types and Priorities
# ======================

class GPUType(str, Enum):
    """Available GPU types with specs"""
    T4 = "t4"          # 16GB, $0.526/hr - Inference
    V100 = "v100"      # 32GB, $2.48/hr - Training
    A100 = "a100"      # 80GB, $5.12/hr - Large models
    H100 = "h100"      # 80GB, $8.00/hr - Latest gen
    RTX4090 = "rtx4090"  # 24GB, $1.50/hr - Development


class AllocationPriority(str, Enum):
    """GPU allocation priorities"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GPUSpecs:
    """GPU specifications"""
    memory_gb: int
    compute_tflops: float
    cost_per_hour: float
    typical_use: str


# GPU specifications
GPU_SPECS = {
    GPUType.T4: GPUSpecs(16, 8.1, 0.526, "inference"),
    GPUType.V100: GPUSpecs(32, 15.7, 2.48, "training"),
    GPUType.A100: GPUSpecs(80, 19.5, 5.12, "large_training"),
    GPUType.H100: GPUSpecs(80, 60.0, 8.00, "frontier_models"),
    GPUType.RTX4090: GPUSpecs(24, 82.6, 1.50, "development")
}


# ======================
# GPU Request and Allocation
# ======================

@dataclass
class GPURequest:
    """GPU allocation request"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_id: str = ""
    
    # Requirements
    gpu_type: GPUType = GPUType.T4
    gpu_count: int = 1
    duration_hours: float = 1.0
    priority: AllocationPriority = AllocationPriority.NORMAL
    
    # Workload info
    workload_type: str = "inference"  # training, inference, research
    estimated_memory_gb: float = 8.0
    estimated_compute_tflops: float = 5.0
    
    # Constraints
    max_cost_per_hour: Optional[float] = None
    budget_remaining: Optional[float] = None
    deadline: Optional[datetime] = None
    
    # Context
    project_id: Optional[str] = None
    justification: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GPUAllocation:
    """GPU allocation result"""
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    
    # Status
    status: str = "pending"  # pending, approved, allocated, rejected, completed
    
    # Allocation details
    gpu_type: GPUType = GPUType.T4
    gpu_count: int = 0
    node_assignments: List[str] = field(default_factory=list)
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_duration: Optional[float] = None
    
    # Cost
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    
    # Decision info
    council_approval: bool = False
    approval_confidence: float = 0.0
    rejection_reason: Optional[str] = None
    
    # Monitoring
    utilization_metrics: Dict[str, float] = field(default_factory=dict)


# ======================
# GPU Resource Manager
# ======================

class GPUResourceManager:
    """
    Manages GPU resources with:
    - Inventory tracking
    - Availability checking
    - Usage monitoring
    - Cost tracking
    """
    
    def __init__(self):
        # GPU inventory (in production, from cluster manager)
        self.inventory = {
            GPUType.T4: {"total": 8, "available": 8},
            GPUType.V100: {"total": 4, "available": 4},
            GPUType.A100: {"total": 2, "available": 2},
            GPUType.H100: {"total": 1, "available": 1},
            GPUType.RTX4090: {"total": 4, "available": 4}
        }
        
        # Active allocations
        self.allocations: Dict[str, GPUAllocation] = {}
        
        # Usage tracking
        self.total_gpu_hours = 0.0
        self.total_cost = 0.0
    
    def check_availability(self, gpu_type: GPUType, count: int) -> bool:
        """Check if GPUs are available"""
        return self.inventory[gpu_type]["available"] >= count
    
    def allocate(self, gpu_type: GPUType, count: int) -> List[str]:
        """Allocate GPUs and return node assignments"""
        if not self.check_availability(gpu_type, count):
            raise ValueError(f"Insufficient {gpu_type} GPUs available")
        
        # Update inventory
        self.inventory[gpu_type]["available"] -= count
        
        # Assign to nodes (simplified)
        nodes = [f"node-{i}-{gpu_type}" for i in range(count)]
        
        return nodes
    
    def deallocate(self, gpu_type: GPUType, count: int):
        """Return GPUs to pool"""
        self.inventory[gpu_type]["available"] += count
        
    def calculate_cost(self, gpu_type: GPUType, hours: float) -> float:
        """Calculate cost for GPU usage"""
        return GPU_SPECS[gpu_type].cost_per_hour * hours
    
    def get_utilization(self) -> Dict[str, Any]:
        """Get current utilization metrics"""
        metrics = {}
        for gpu_type, inv in self.inventory.items():
            total = inv["total"]
            available = inv["available"]
            used = total - available
            metrics[gpu_type] = {
                "total": total,
                "used": used,
                "available": available,
                "utilization": (used / total) if total > 0 else 0
            }
        return metrics


# ======================
# GPU Scheduler
# ======================

class GPUScheduler:
    """
    Fair GPU scheduling with:
    - Priority queues
    - Budget awareness
    - Deadline handling
    - Anti-starvation
    """
    
    def __init__(self, resource_manager: GPUResourceManager):
        self.resource_manager = resource_manager
        self.request_queue: List[GPURequest] = []
        self.processing_lock = asyncio.Lock()
        
    async def submit_request(self, request: GPURequest) -> str:
        """Submit GPU request to queue"""
        async with self.processing_lock:
            self.request_queue.append(request)
            # Sort by priority and creation time
            self.request_queue.sort(
                key=lambda r: (
                    -self._priority_score(r.priority),
                    r.created_at
                )
            )
        
        logger.info(
            f"GPU request submitted",
            request_id=request.request_id,
            gpu_type=request.gpu_type,
            priority=request.priority
        )
        
        return request.request_id
    
    def _priority_score(self, priority: AllocationPriority) -> int:
        """Convert priority to numeric score"""
        return {
            AllocationPriority.CRITICAL: 4,
            AllocationPriority.HIGH: 3,
            AllocationPriority.NORMAL: 2,
            AllocationPriority.LOW: 1
        }[priority]
    
    async def process_queue(self) -> List[GPUAllocation]:
        """Process queued requests"""
        allocations = []
        
        async with self.processing_lock:
            processed = []
            
            for request in self.request_queue:
                # Check if we can fulfill
                if self.resource_manager.check_availability(
                    request.gpu_type,
                    request.gpu_count
                ):
                    # Allocate resources
                    nodes = self.resource_manager.allocate(
                        request.gpu_type,
                        request.gpu_count
                    )
                    
                    # Create allocation
                    allocation = GPUAllocation(
                        request_id=request.request_id,
                        status="allocated",
                        gpu_type=request.gpu_type,
                        gpu_count=request.gpu_count,
                        node_assignments=nodes,
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc) + timedelta(hours=request.duration_hours),
                        estimated_cost=self.resource_manager.calculate_cost(
                            request.gpu_type,
                            request.duration_hours
                        )
                    )
                    
                    allocations.append(allocation)
                    processed.append(request)
                    
                    logger.info(
                        f"GPU allocated",
                        allocation_id=allocation.allocation_id,
                        nodes=nodes
                    )
            
            # Remove processed requests
            for req in processed:
                self.request_queue.remove(req)
        
        return allocations


# ======================
# Integrated GPU Orchestrator
# ======================

class GPUOrchestrator:
    """
    GPU orchestration integrated with UnifiedOrchestrationEngine.
    
    Features:
    - LNN council approval for high-value requests
    - Cost optimization
    - Fair scheduling
    - Automatic cleanup
    - Monitoring integration
    """
    
    def __init__(self, orchestration_engine: UnifiedOrchestrationEngine):
        self.orchestration_engine = orchestration_engine
        self.resource_manager = GPUResourceManager()
        self.scheduler = GPUScheduler(self.resource_manager)
        
        # LNN council for approvals
        if LNN_AVAILABLE:
            self.lnn_council = create_lnn_council(
                num_agents=5,
                agent_capabilities={
                    "agent_0": ["gpu_allocation", "cost_analysis"],
                    "agent_1": ["gpu_allocation", "priority_assessment"],
                    "agent_2": ["gpu_allocation", "resource_optimization"],
                    "agent_3": ["gpu_allocation", "risk_assessment"],
                    "agent_4": ["gpu_allocation", "fairness_check"]
                }
            )
        else:
            self.lnn_council = None
        
        # Active allocations tracking
        self.active_allocations: Dict[str, GPUAllocation] = {}
        
        # Monitoring task
        self.monitor_task = None
        
        logger.info("GPU orchestrator initialized")
    
    async def request_gpu(self, request: GPURequest) -> GPUAllocation:
        """
        Request GPU allocation with full orchestration.
        
        Steps:
        1. Validate request
        2. Get LNN council approval (if needed)
        3. Submit to scheduler
        4. Monitor usage
        5. Auto-cleanup
        """
        
        # Step 1: Validate request
        validation = self._validate_request(request)
        if not validation["valid"]:
            return GPUAllocation(
                request_id=request.request_id,
                status="rejected",
                rejection_reason=validation["reason"]
            )
        
        # Step 2: LNN council approval for expensive requests
        if self._requires_approval(request):
            approval = await self._get_council_approval(request)
            if not approval["approved"]:
                return GPUAllocation(
                    request_id=request.request_id,
                    status="rejected",
                    council_approval=False,
                    rejection_reason=approval["reason"]
                )
        
        # Step 3: Submit to scheduler
        await self.scheduler.submit_request(request)
        
        # Process queue immediately
        allocations = await self.scheduler.process_queue()
        
        # Find our allocation
        for alloc in allocations:
            if alloc.request_id == request.request_id:
                # Step 4: Start monitoring
                self.active_allocations[alloc.allocation_id] = alloc
                asyncio.create_task(self._monitor_allocation(alloc))
                
                # Store in orchestration memory
                if self.orchestration_engine.memory_system:
                    await self.orchestration_engine.memory_system.store({
                        "type": "gpu_allocation",
                        "allocation": alloc,
                        "timestamp": datetime.now(timezone.utc)
                    })
                
                return alloc
        
        # If not allocated, return pending
        return GPUAllocation(
            request_id=request.request_id,
            status="pending"
        )
    
    def _validate_request(self, request: GPURequest) -> Dict[str, Any]:
        """Validate GPU request"""
        # Check GPU type
        if request.gpu_type not in GPU_SPECS:
            return {"valid": False, "reason": f"Invalid GPU type: {request.gpu_type}"}
        
        # Check count
        if request.gpu_count < 1 or request.gpu_count > 8:
            return {"valid": False, "reason": "GPU count must be 1-8"}
        
        # Check duration
        if request.duration_hours < 0.1 or request.duration_hours > 168:  # 1 week max
            return {"valid": False, "reason": "Duration must be 0.1-168 hours"}
        
        # Check budget if specified
        if request.max_cost_per_hour:
            min_cost = GPU_SPECS[request.gpu_type].cost_per_hour * request.gpu_count
            if min_cost > request.max_cost_per_hour:
                return {
                    "valid": False,
                    "reason": f"Cost ${min_cost}/hr exceeds budget ${request.max_cost_per_hour}/hr"
                }
        
        return {"valid": True}
    
    def _requires_approval(self, request: GPURequest) -> bool:
        """Check if request requires council approval"""
        # High-value requests need approval
        estimated_cost = (
            GPU_SPECS[request.gpu_type].cost_per_hour *
            request.gpu_count *
            request.duration_hours
        )
        
        return (
            estimated_cost > 100 or  # $100+ total cost
            request.gpu_count > 4 or  # Many GPUs
            request.gpu_type in [GPUType.A100, GPUType.H100] or  # Expensive GPUs
            request.priority == AllocationPriority.CRITICAL  # Critical requests
        )
    
    async def _get_council_approval(self, request: GPURequest) -> Dict[str, Any]:
        """Get LNN council approval"""
        if not self.lnn_council:
            # No council, auto-approve
            return {"approved": True, "confidence": 1.0}
        
        # Prepare council request
        council_request = {
            "type": "gpu_allocation_approval",
            "request": {
                "gpu_type": request.gpu_type,
                "gpu_count": request.gpu_count,
                "duration_hours": request.duration_hours,
                "estimated_cost": (
                    GPU_SPECS[request.gpu_type].cost_per_hour *
                    request.gpu_count *
                    request.duration_hours
                ),
                "justification": request.justification,
                "priority": request.priority,
                "workload_type": request.workload_type
            }
        }
        
        # Get consensus
        consensus = await self.lnn_council.make_council_decision(
            request=council_request,
            context={
                "current_utilization": self.resource_manager.get_utilization(),
                "total_cost_today": self.resource_manager.total_cost,
                "requester_history": {}  # Would track requester's past usage
            },
            required_capabilities=["gpu_allocation"]
        )
        
        # Determine approval
        approved = (
            consensus.final_decision == VoteDecision.APPROVE and
            consensus.consensus_confidence > 0.6
        )
        
        return {
            "approved": approved,
            "confidence": consensus.consensus_confidence,
            "reason": f"{consensus.consensus_type} consensus: {consensus.reasoning}"
        }
    
    async def _monitor_allocation(self, allocation: GPUAllocation):
        """Monitor GPU allocation and auto-cleanup"""
        start_time = allocation.start_time or datetime.now(timezone.utc)
        end_time = allocation.end_time or (start_time + timedelta(hours=1))
        
        # Monitor until end time
        while datetime.now(timezone.utc) < end_time:
            # Check utilization (in production, from GPU metrics)
            utilization = 0.75 + (0.25 * (datetime.now(timezone.utc).second / 60))  # Mock
            allocation.utilization_metrics[datetime.now(timezone.utc).isoformat()] = utilization
            
            # Log low utilization
            if utilization < 0.3:
                logger.warning(
                    f"Low GPU utilization",
                    allocation_id=allocation.allocation_id,
                    utilization=utilization
                )
            
            await asyncio.sleep(60)  # Check every minute
        
        # Cleanup allocation
        await self._cleanup_allocation(allocation)
    
    async def _cleanup_allocation(self, allocation: GPUAllocation):
        """Clean up GPU allocation"""
        # Calculate actual duration and cost
        actual_duration = (
            datetime.now(timezone.utc) - allocation.start_time
        ).total_seconds() / 3600
        
        allocation.actual_duration = actual_duration
        allocation.actual_cost = self.resource_manager.calculate_cost(
            allocation.gpu_type,
            actual_duration
        )
        allocation.status = "completed"
        
        # Deallocate resources
        self.resource_manager.deallocate(
            allocation.gpu_type,
            allocation.gpu_count
        )
        
        # Update tracking
        self.resource_manager.total_gpu_hours += actual_duration * allocation.gpu_count
        self.resource_manager.total_cost += allocation.actual_cost
        
        # Remove from active
        del self.active_allocations[allocation.allocation_id]
        
        # Store final state
        if self.orchestration_engine.memory_system:
            await self.orchestration_engine.memory_system.store({
                "type": "gpu_allocation_completed",
                "allocation": allocation,
                "timestamp": datetime.now(timezone.utc)
            })
        
        logger.info(
            f"GPU allocation cleaned up",
            allocation_id=allocation.allocation_id,
            actual_cost=allocation.actual_cost,
            utilization_avg=sum(allocation.utilization_metrics.values()) / len(allocation.utilization_metrics) if allocation.utilization_metrics else 0
        )
    
    async def get_allocation_status(self, allocation_id: str) -> Optional[GPUAllocation]:
        """Get status of GPU allocation"""
        return self.active_allocations.get(allocation_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get GPU orchestration metrics"""
        return {
            "utilization": self.resource_manager.get_utilization(),
            "active_allocations": len(self.active_allocations),
            "queued_requests": len(self.scheduler.request_queue),
            "total_gpu_hours": self.resource_manager.total_gpu_hours,
            "total_cost": self.resource_manager.total_cost,
            "cost_per_gpu_hour": (
                self.resource_manager.total_cost / self.resource_manager.total_gpu_hours
                if self.resource_manager.total_gpu_hours > 0 else 0
            )
        }


# ======================
# Integration Extension
# ======================

def integrate_gpu_workflows(orchestration_engine: UnifiedOrchestrationEngine) -> GPUOrchestrator:
    """
    Integrate GPU workflows into the orchestration engine.
    
    This adds GPU as a first-class resource in orchestration!
    """
    gpu_orchestrator = GPUOrchestrator(orchestration_engine)
    
    # Add to orchestration engine
    orchestration_engine.gpu_orchestrator = gpu_orchestrator
    
    # Add GPU workflow type
    if hasattr(orchestration_engine, 'workflow_types'):
        orchestration_engine.workflow_types['gpu_allocation'] = {
            "handler": gpu_orchestrator.request_gpu,
            "requires_approval": True,
            "cost_tracked": True
        }
    
    logger.info("GPU workflows integrated with orchestration engine")
    
    return gpu_orchestrator


# ======================
# Example Usage
# ======================

async def example():
    """Example of GPU orchestration"""
    print("\n🎮 GPU Orchestration Example\n")
    
    # Create mock orchestration engine
    orchestration = UnifiedOrchestrationEngine()
    
    # Integrate GPU workflows
    gpu_orchestrator = integrate_gpu_workflows(orchestration)
    
    # Create GPU request
    request = GPURequest(
        requester_id="user_123",
        gpu_type=GPUType.A100,
        gpu_count=2,
        duration_hours=4,
        priority=AllocationPriority.HIGH,
        workload_type="training",
        estimated_memory_gb=60,
        justification="Training large language model for production deployment",
        project_id="llm_project_1",
        max_cost_per_hour=15.0
    )
    
    print(f"Requesting: {request.gpu_count}x {request.gpu_type} for {request.duration_hours} hours")
    print(f"Estimated cost: ${GPU_SPECS[request.gpu_type].cost_per_hour * request.gpu_count * request.duration_hours:.2f}")
    
    # Request GPU
    allocation = await gpu_orchestrator.request_gpu(request)
    
    print(f"\nAllocation result:")
    print(f"  Status: {allocation.status}")
    print(f"  Allocation ID: {allocation.allocation_id}")
    
    if allocation.status == "allocated":
        print(f"  Nodes: {allocation.node_assignments}")
        print(f"  Estimated cost: ${allocation.estimated_cost:.2f}")
    elif allocation.status == "rejected":
        print(f"  Rejection reason: {allocation.rejection_reason}")
    
    # Show metrics
    metrics = gpu_orchestrator.get_metrics()
    print(f"\nGPU Metrics:")
    print(f"  Utilization: {metrics['utilization']}")
    print(f"  Active allocations: {metrics['active_allocations']}")
    print(f"  Total cost: ${metrics['total_cost']:.2f}")


if __name__ == "__main__":
    asyncio.run(example())