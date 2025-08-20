"""
CrewAI Orchestrator - Main Interface

Clean, focused orchestrator that composes geometric components.
"""

from typing import Dict, List, Any, Optional
import logging

from .geometric_space import HyperbolicSpace, GeometricRouter
from .flow_engine import FlowEngine
from ...observability.hybrid import HybridObservability

logger = logging.getLogger(__name__)

class CrewAIOrchestrator:
    """Main CrewAI orchestrator with geometric intelligence (30 lines)"""
    
    def __init__(self, observability: Optional[HybridObservability] = None):
        self.observability = observability
        self.space = HyperbolicSpace()
        self.router = GeometricRouter(self.space)
        self.engine = FlowEngine(self.router)
        
        logger.info("CrewAI Orchestrator initialized with geometric intelligence")
    
    def register_agent(self, agent_id: str, capabilities: List[str], 
                      embedding: List[float]) -> None:
        """Register agent with capabilities"""
        import numpy as np
        self.router.register(agent_id, capabilities, np.array(embedding))
    
    async def create_flow(self, config: Dict[str, Any]) -> str:
        """Create geometric flow"""
        flow_id = self.engine.create_flow(config)
        
        if self.observability:
            from ...observability.core import pure_metric
            await self.observability.emit_metric(
                pure_metric("crewai_flow_created", 1.0)
                .with_tag("flow_id", flow_id)
            )
        
        return flow_id
    
    async def execute_flow(self, flow_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute flow with geometric routing"""
        try:
            result = await self.engine.execute_flow(flow_id, config)
            
            if self.observability:
                from ...observability.core import pure_metric
                await self.observability.emit_metric(
                    pure_metric("crewai_flow_completed", 1.0)
                    .with_tag("flow_id", flow_id)
                    .with_tag("status", "success")
                )
            
            return result
        except Exception as e:
            logger.error(f"Flow {flow_id} failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """System health check"""
        return {
            'status': 'healthy',
            'active_flows': len(self.engine.active_flows),
            'registered_agents': len(self.router.agents),
            'geometric_intelligence': True
        }