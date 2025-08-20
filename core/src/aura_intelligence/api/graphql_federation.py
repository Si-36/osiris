"""
GraphQL Federation 2.0 - Unified API for all AURA systems
2025 production-grade API gateway
"""
import asyncio
from typing import Dict, Any, List, Optional
import strawberry
from strawberry.federation import Schema
from strawberry.types import Info
import structlog

# Import your existing systems
from ..production_integration_2025 import get_production_system
from ..memory.cxl_memory_pool import get_cxl_memory_pool

logger = structlog.get_logger()

@strawberry.type
class MemoryStats:
    """Memory system statistics"""
    pool_size_gb: float
    utilization: float
    total_segments: int
    memory_components: int

@strawberry.type
class ComponentHealth:
    """Component health status"""
    component_id: str
    status: str
    latency_ms: float
    success_rate: float

@strawberry.type
class TDAAnalysis:
    """TDA analysis result"""
    system_id: str
    topology_score: float
    risk_level: str
    bottlenecks: List[str]
    recommendations: List[str]

@strawberry.type
class CoRaLResult:
    """CoRaL communication result"""
    messages_generated: int
    decisions_made: int
    causal_influence: float
    throughput: float

@strawberry.type
class SystemHealth:
    """Overall system health"""
    overall_health: float
    components: List[ComponentHealth]
    memory_stats: MemoryStats
    ray_cluster_resources: str

@strawberry.input
class MemoryOperationInput:
    """Input for memory operations"""
    operation_type: str  # "store" or "retrieve"
    content: Optional[str] = None
    context_type: Optional[str] = "general"
    k: Optional[int] = 10

@strawberry.input
class TDAAnalysisInput:
    """Input for TDA analysis"""
    system_id: str
    agents: List[str]
    communications: Optional[List[str]] = None

@strawberry.input
class CoRaLInput:
    """Input for CoRaL communication"""
    contexts: List[str]

@strawberry.type
class Query:
    """GraphQL Query root"""
    
    @strawberry.field
    async def system_health(self, info: Info) -> SystemHealth:
        """Get comprehensive system health"""
        production_system = await get_production_system()
        health_data = await production_system.get_system_health()
        
        # Convert to GraphQL types
        components = [
            ComponentHealth(
                component_id=comp_id,
                status=details["status"],
                latency_ms=details.get("details", {}).get("latency", 0) * 1000,
                success_rate=1.0 if details["status"] == "healthy" else 0.5
            )
            for comp_id, details in health_data["components"].items()
        ]
        
        # Get memory stats
        cxl_pool = get_cxl_memory_pool()
        memory_data = cxl_pool.get_pool_stats()
        
        memory_stats = MemoryStats(
            pool_size_gb=memory_data["pool_size_gb"],
            utilization=memory_data["utilization"],
            total_segments=memory_data["total_segments"],
            memory_components=memory_data["memory_components"]
        )
        
        return SystemHealth(
            overall_health=health_data["overall_health"],
            components=components,
            memory_stats=memory_stats,
            ray_cluster_resources=str(health_data["ray_cluster"])
        )
    
    @strawberry.field
    async def component_status(self, info: Info, component_id: str) -> ComponentHealth:
        """Get specific component status"""
        production_system = await get_production_system()
        health_data = await production_system.get_system_health()
        
        if component_id in health_data["components"]:
            details = health_data["components"][component_id]
            return ComponentHealth(
                component_id=component_id,
                status=details["status"],
                latency_ms=details.get("details", {}).get("latency", 0) * 1000,
                success_rate=1.0 if details["status"] == "healthy" else 0.5
            )
        else:
            return ComponentHealth(
                component_id=component_id,
                status="not_found",
                latency_ms=0,
                success_rate=0
            )

@strawberry.type
class Mutation:
    """GraphQL Mutation root"""
    
    @strawberry.mutation
    async def memory_operation(self, info: Info, input: MemoryOperationInput) -> str:
        """Perform memory operation"""
        production_system = await get_production_system()
        
        request = {
            "type": "memory_operation",
            "data": {
                "operation_type": input.operation_type,
                "content": input.content,
                "context_type": input.context_type,
                "k": input.k
            }
        }
        
        result = await production_system.process_unified_request(request)
        
        if result["success"]:
            return f"Memory operation completed: {input.operation_type}"
        else:
            return f"Memory operation failed: {result.get('error', 'Unknown error')}"
    
    @strawberry.mutation
    async def analyze_system_tda(self, info: Info, input: TDAAnalysisInput) -> TDAAnalysis:
        """Perform TDA analysis"""
        production_system = await get_production_system()
        
        request = {
            "type": "system_analysis",
            "data": {
                "system_id": input.system_id,
                "agents": [{"id": agent_id} for agent_id in input.agents],
                "communications": input.communications or []
            }
        }
        
        result = await production_system.process_unified_request(request)
        
        if result["success"]:
            analysis = result["result"]
            return TDAAnalysis(
                system_id=analysis.get("system_id", input.system_id),
                topology_score=analysis.get("topology_score", 0.0),
                risk_level=analysis.get("risk_level", "unknown"),
                bottlenecks=analysis.get("bottlenecks", []),
                recommendations=analysis.get("recommendations", [])
            )
        else:
            return TDAAnalysis(
                system_id=input.system_id,
                topology_score=0.0,
                risk_level="error",
                bottlenecks=[f"Analysis failed: {result.get('error', 'Unknown error')}"],
                recommendations=["Check system logs for details"]
            )
    
    @strawberry.mutation
    async def coral_communicate(self, info: Info, input: CoRaLInput) -> CoRaLResult:
        """Perform CoRaL communication"""
        production_system = await get_production_system()
        
        request = {
            "type": "general_request",
            "contexts": [{"data": context} for context in input.contexts]
        }
        
        result = await production_system.process_unified_request(request)
        
        if result["success"]:
            coral_result = result["unified_results"].get("coral", {}).get("result", {})
            return CoRaLResult(
                messages_generated=coral_result.get("messages_generated", 0),
                decisions_made=coral_result.get("decisions_made", 0),
                causal_influence=coral_result.get("causal_influence", 0.0),
                throughput=coral_result.get("throughput", 0.0)
            )
        else:
            return CoRaLResult(
                messages_generated=0,
                decisions_made=0,
                causal_influence=0.0,
                throughput=0.0
            )

@strawberry.type
class Subscription:
    """GraphQL Subscription root for real-time updates"""
    
    @strawberry.subscription
    async def system_health_stream(self, info: Info) -> SystemHealth:
        """Stream real-time system health updates"""
        production_system = await get_production_system()
        
        while True:
            try:
                health_data = await production_system.get_system_health()
                
                components = [
                    ComponentHealth(
                        component_id=comp_id,
                        status=details["status"],
                        latency_ms=details.get("details", {}).get("latency", 0) * 1000,
                        success_rate=1.0 if details["status"] == "healthy" else 0.5
                    )
                    for comp_id, details in health_data["components"].items()
                ]
                
                cxl_pool = get_cxl_memory_pool()
                memory_data = cxl_pool.get_pool_stats()
                
                memory_stats = MemoryStats(
                    pool_size_gb=memory_data["pool_size_gb"],
                    utilization=memory_data["utilization"],
                    total_segments=memory_data["total_segments"],
                    memory_components=memory_data["memory_components"]
                )
                
                yield SystemHealth(
                    overall_health=health_data["overall_health"],
                    components=components,
                    memory_stats=memory_stats,
                    ray_cluster_resources=str(health_data["ray_cluster"])
                )
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error("Health stream error", error=str(e))
                await asyncio.sleep(10)  # Wait longer on error

# Create federated schema
schema = Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    enable_federation_2=True
)