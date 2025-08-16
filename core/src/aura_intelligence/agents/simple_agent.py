#!/usr/bin/env python3
"""
Simple Agent Implementation
Clean, minimal agent that works with the unified interfaces
"""

import asyncio
import time
from typing import Dict, Any, Optional

from ..core.unified_interfaces import AgentComponent, ComponentStatus, ComponentMetrics

class SimpleAgent(AgentComponent):
    """
    Simple agent implementation that actually works.
    No complex dependencies, just the core AgentComponent interface.
    """
    
    def __init__(self, agent_id: str, agent_type: str = "simple", config: Dict[str, Any] = None):
        super().__init__(agent_id, config or {})
        self.agent_type = agent_type
        self.decision_count = 0
        self.success_count = 0
        
        print(f"🤖 Simple Agent: {agent_id} ({agent_type})")
    
    # ========================================================================
    # LIFECYCLE METHODS (Required by UnifiedComponent)
    # ========================================================================
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            self.status = ComponentStatus.ACTIVE
            print(f"✅ {self.component_id} initialized")
            return True
        except Exception as e:
            print(f"❌ {self.component_id} initialization failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the agent."""
        if self.status != ComponentStatus.ACTIVE:
            return await self.initialize()
        return True
    
    async def stop(self) -> bool:
        """Stop the agent."""
        self.status = ComponentStatus.INACTIVE
        print(f"🛑 {self.component_id} stopped")
        return True
    
    async def health_check(self) -> ComponentMetrics:
        """Perform health check."""
        success_rate = self.success_count / max(1, self.decision_count)
        self.metrics.health_score = success_rate
        self.metrics.status = self.status
        return self.metrics
    
    # ========================================================================
    # CONFIGURATION METHODS (Required by UnifiedComponent)
    # ========================================================================
    
    async def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update configuration."""
        try:
            self.config.update(config_updates)
            print(f"🔧 {self.component_id} config updated")
            return True
        except Exception as e:
            print(f"❌ {self.component_id} config update failed: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        return isinstance(config, dict)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string"}
            }
        }
    
    # ========================================================================
    # PROCESSING METHODS (Required by UnifiedComponent)
    # ========================================================================
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process input data."""
        start_time = time.time()
        
        try:
            # Simple processing
            result = {
                "processed_by": self.component_id,
                "agent_type": self.agent_type,
                "input_type": type(input_data).__name__,
                "input_size": len(str(input_data)),
                "status": "processed"
            }
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self._update_operation_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_operation_metrics(False, response_time)
            return {"error": str(e), "agent_id": self.component_id}
    
    # ========================================================================
    # AGENT METHODS (Required by AgentComponent)
    # ========================================================================
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on context."""
        start_time = time.time()
        self.decision_count += 1
        
        try:
            # Simple decision logic based on agent type
            if self.agent_type == "council":
                action = "analyze_and_recommend"
                confidence = 0.8
                reasoning = ["Neural processing", "Pattern analysis", "Recommendation generated"]
            elif self.agent_type == "bio":
                action = "adapt_and_evolve"
                confidence = 0.7
                reasoning = ["Biological assessment", "Energy evaluation", "Adaptation strategy"]
            else:
                action = "process_and_respond"
                confidence = 0.6
                reasoning = ["Input analysis", "Standard processing", "Response generation"]
            
            self.success_count += 1
            
            return {
                "decision_id": f"{self.component_id}_{self.decision_count}",
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "agent_type": self.agent_type,
                "response_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            return {
                "decision_id": f"{self.component_id}_error_{self.decision_count}",
                "action": "error_recovery",
                "confidence": 0.1,
                "reasoning": [f"Error: {str(e)}"],
                "agent_type": self.agent_type,
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Learn from feedback."""
        try:
            feedback_score = feedback.get("score", 0.5)
            
            # Simple learning: adjust success rate based on feedback
            if feedback_score > 0.7:
                self.success_count += 1
            
            print(f"📚 {self.component_id} learned from feedback: {feedback_score}")
            return True
            
        except Exception as e:
            print(f"❌ {self.component_id} learning failed: {e}")
            return False
    
    def get_agent_type(self) -> str:
        """Get the agent type."""
        return self.agent_type
    
    # ========================================================================
    # STATUS AND METRICS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.component_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "decision_count": self.decision_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(1, self.decision_count)
        }

# ============================================================================
# SIMPLE FACTORY
# ============================================================================

def create_simple_agent(agent_id: str, agent_type: str = "simple", config: Dict[str, Any] = None) -> SimpleAgent:
    """Create a simple agent."""
    return SimpleAgent(agent_id, agent_type, config)

# ============================================================================
# SIMPLE REGISTRY
# ============================================================================

class SimpleAgentRegistry:
    """Simple registry for agents."""
    
    def __init__(self):
        self.agents: Dict[str, SimpleAgent] = {}
    
    def register(self, agent: SimpleAgent) -> None:
        """Register an agent."""
        self.agents[agent.component_id] = agent
        print(f"📝 Registered: {agent.component_id}")
    
    def get_agent(self, agent_id: str) -> Optional[SimpleAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> Dict[str, str]:
        """List all agents."""
        return {agent_id: agent.agent_type for agent_id, agent in self.agents.items()}
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status == ComponentStatus.ACTIVE])
        }

# Global registry
_simple_registry = SimpleAgentRegistry()

def get_simple_registry() -> SimpleAgentRegistry:
    """Get the global simple registry."""
    return _simple_registry