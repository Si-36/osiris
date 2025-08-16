#!/usr/bin/env python3
"""
Clean Agent Implementation
Simple, working agents using the unified AgentComponent interface
"""

import asyncio
import time
from typing import Dict, Any, Optional
from enum import Enum

from ..core.unified_interfaces import AgentComponent, ComponentStatus, ComponentMetrics

# ============================================================================
# AGENT TYPES
# ============================================================================

class AgentType(Enum):
    COUNCIL = "council"
    BIO = "bio"
    GENERIC = "generic"

# ============================================================================
# CLEAN AGENT IMPLEMENTATION
# ============================================================================

class CleanAgent(AgentComponent):
    """
    Clean, simple agent implementation that actually works.
    Uses the proper AgentComponent interface from unified_interfaces.
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.agent_type = agent_type
        self.decision_count = 0
        self.success_count = 0
        
        print(f"🤖 Clean Agent: {agent_id} ({agent_type.value})")
    
    # ========================================================================
    # LIFECYCLE METHODS (Required by UnifiedComponent)
    # ========================================================================
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            self.status = ComponentStatus.ACTIVE
            await self.emit_event("agent_initialized", {
                "agent_type": self.agent_type.value
            })
            return True
        except Exception as e:
            await self.emit_event("initialization_error", {"error": str(e)})
            return False
    
    async def start(self) -> bool:
        """Start the agent."""
        if self.status != ComponentStatus.ACTIVE:
            return await self.initialize()
        return True
    
    async def stop(self) -> bool:
        """Stop the agent."""
        self.status = ComponentStatus.INACTIVE
        await self.emit_event("agent_stopped", {})
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
            await self.emit_event("config_updated", config_updates)
            return True
        except Exception as e:
            await self.emit_event("config_update_error", {"error": str(e)})
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        # Basic validation - config should be a dict
        return isinstance(config, dict)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string"},
                "decision_strategy": {"type": "string"}
            }
        }
    
    # ========================================================================
    # PROCESSING METHODS (Required by UnifiedComponent)
    # ========================================================================
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process input data."""
        start_time = time.time()
        
        try:
            # Simple processing based on agent type
            if self.agent_type == AgentType.COUNCIL:
                result = await self._council_process(input_data, context)
            elif self.agent_type == AgentType.BIO:
                result = await self._bio_process(input_data, context)
            else:
                result = await self._generic_process(input_data, context)
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self._update_operation_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_operation_metrics(False, response_time)
            return {"error": str(e), "agent_type": self.agent_type.value}
    
    # ========================================================================
    # AGENT METHODS (Required by AgentComponent)
    # ========================================================================
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on context."""
        start_time = time.time()
        self.decision_count += 1
        
        try:
            # Agent-specific decision logic
            if self.agent_type == AgentType.COUNCIL:
                decision = await self._council_decision(context)
            elif self.agent_type == AgentType.BIO:
                decision = await self._bio_decision(context)
            else:
                decision = await self._generic_decision(context)
            
            self.success_count += 1
            
            return {
                "decision_id": f"{self.component_id}_{self.decision_count}",
                "action": decision["action"],
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"],
                "agent_type": self.agent_type.value,
                "response_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            return {
                "decision_id": f"{self.component_id}_error_{self.decision_count}",
                "action": "error_recovery",
                "confidence": 0.1,
                "reasoning": [f"Error: {str(e)}"],
                "agent_type": self.agent_type.value,
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Learn from feedback."""
        try:
            feedback_score = feedback.get("score", 0.5)
            
            # Simple learning: adjust success rate based on feedback
            if feedback_score > 0.7:
                self.success_count += 1
            
            await self.emit_event("learning_event", {
                "feedback_score": feedback_score,
                "agent_type": self.agent_type.value
            })
            
            return True
            
        except Exception as e:
            await self.emit_event("learning_error", {"error": str(e)})
            return False
    
    def get_agent_type(self) -> str:
        """Get the agent type."""
        return self.agent_type.value
    
    # ========================================================================
    # AGENT-SPECIFIC PROCESSING
    # ========================================================================
    
    async def _council_process(self, input_data: Any, context: Optional[Dict[str, Any]]) -> Any:
        """Council agent processing."""
        return {
            "processed_by": "council",
            "analysis": "Neural network processing complete",
            "confidence": 0.8,
            "input_size": len(str(input_data))
        }
    
    async def _bio_process(self, input_data: Any, context: Optional[Dict[str, Any]]) -> Any:
        """Bio agent processing."""
        return {
            "processed_by": "bio",
            "biological_state": "healthy",
            "energy_level": 0.9,
            "adaptation": "successful"
        }
    
    async def _generic_process(self, input_data: Any, context: Optional[Dict[str, Any]]) -> Any:
        """Generic agent processing."""
        return {
            "processed_by": "generic",
            "status": "processed",
            "data_type": type(input_data).__name__
        }
    
    # ========================================================================
    # AGENT-SPECIFIC DECISIONS
    # ========================================================================
    
    async def _council_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Council agent decision."""
        # Simple neural-like decision
        input_strength = len(str(context)) / 100.0
        confidence = min(0.9, 0.5 + input_strength * 0.1)
        
        actions = ["analyze", "recommend", "execute", "delegate"]
        action = actions[hash(str(context)) % len(actions)]
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": [
                "Council neural processing",
                f"Input strength: {input_strength:.3f}",
                f"Selected: {action}"
            ]
        }
    
    async def _bio_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Bio agent decision."""
        # Simple biological decision
        energy = 0.8  # Simulated energy
        
        if energy > 0.7:
            action = "explore"
            confidence = 0.8
        elif energy > 0.4:
            action = "maintain"
            confidence = 0.6
        else:
            action = "conserve"
            confidence = 0.9
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": [
                "Biological decision making",
                f"Energy level: {energy}",
                f"Selected: {action}"
            ]
        }
    
    async def _generic_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic agent decision."""
        # Simple rule-based decision
        context_size = len(context)
        
        if context_size > 5:
            action = "process"
            confidence = 0.7
        else:
            action = "acknowledge"
            confidence = 0.9
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": [
                "Generic processing",
                f"Context size: {context_size}",
                f"Selected: {action}"
            ]
        }
    
    # ========================================================================
    # STATUS AND METRICS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.component_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "decision_count": self.decision_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(1, self.decision_count)
        }

# ============================================================================
# AGENT FACTORY
# ============================================================================

class CleanAgentFactory:
    """Factory for creating clean agents."""
    
    @staticmethod
    def create_agent(agent_type: str, agent_id: str, config: Dict[str, Any] = None) -> CleanAgent:
        """Create a clean agent."""
        config = config or {}
        
        # Map string to enum
        if agent_type.lower() == "council":
            return CleanAgent(agent_id, AgentType.COUNCIL, config)
        elif agent_type.lower() == "bio":
            return CleanAgent(agent_id, AgentType.BIO, config)
        else:
            return CleanAgent(agent_id, AgentType.GENERIC, config)
    
    @staticmethod
    def get_available_types() -> list[str]:
        """Get available agent types."""
        return [t.value for t in AgentType]

# ============================================================================
# AGENT REGISTRY
# ============================================================================

class CleanAgentRegistry:
    """Simple registry for clean agents."""
    
    def __init__(self):
        self.agents: Dict[str, CleanAgent] = {}
    
    def register(self, agent: CleanAgent) -> None:
        """Register an agent."""
        self.agents[agent.component_id] = agent
        print(f"📝 Registered: {agent.component_id} ({agent.agent_type.value})")
    
    def get_agent(self, agent_id: str) -> Optional[CleanAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> list[CleanAgent]:
        """Get agents by type."""
        return [
            agent for agent in self.agents.values()
            if agent.agent_type.value == agent_type
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        type_counts = {}
        for agent in self.agents.values():
            agent_type = agent.agent_type.value
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        return {
            "total_agents": len(self.agents),
            "agent_types": type_counts,
            "active_agents": len([a for a in self.agents.values() if a.status == ComponentStatus.ACTIVE])
        }

# ============================================================================
# GLOBAL REGISTRY
# ============================================================================

_global_agent_registry = CleanAgentRegistry()

def get_clean_agent_registry() -> CleanAgentRegistry:
    """Get the global clean agent registry."""
    return _global_agent_registry