"""
Persistence Mixin for Agents
===========================
Adds causal persistence capabilities to any agent
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from ..persistence.causal_state_manager import (
    get_causal_manager,
    StateType,
    CausalContext
)

logger = structlog.get_logger()

class PersistenceMixin:
    """Mixin to add persistence capabilities to agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistence_manager = None
        self._decision_history = []
        self._causal_graph = {}
        
    async def _ensure_persistence(self):
        """Ensure persistence manager is initialized"""
        if self._persistence_manager is None:
            self._persistence_manager = await get_causal_manager()
            
    async def save_decision(self, 
                           decision: str,
                           context: Dict[str, Any],
                           confidence: float = 1.0) -> str:
        """Save a decision with full causal context"""
        await self._ensure_persistence()
        
        # Extract causes from context
        causes = self._extract_causes(context)
        
        # Predict effects
        effects = self._predict_effects(decision, context)
        
        # Create counterfactuals
        counterfactuals = self._generate_counterfactuals(decision, context)
        
        # Create causal context
        causal_context = CausalContext(
            causes=causes,
            effects=effects,
            counterfactuals=counterfactuals,
            confidence=confidence,
            energy_cost=context.get("energy_cost", 0.1),
            decision_path=self._decision_history[-10:]  # Last 10 decisions
        )
        
        # Save state
        state_data = {
            "decision": decision,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "agent_id": getattr(self, "agent_id", "unknown"),
            "agent_type": self.__class__.__name__,
            "confidence": confidence
        }
        
        state_id = await self._persistence_manager.save_state(
            StateType.AGENT_MEMORY,
            getattr(self, "agent_id", "unknown"),
            state_data,
            causal_context=causal_context
        )
        
        # Update history
        self._decision_history.append({
            "decision": decision,
            "state_id": state_id,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Saved decision with causal context", 
                   state_id=state_id,
                   decision=decision,
                   causes=causes[:2])  # Log first 2 causes
        
        return state_id
    
    async def load_memory(self, 
                         compute_fn: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        """Load agent memory with optional computation"""
        await self._ensure_persistence()
        
        return await self._persistence_manager.load_state(
            StateType.AGENT_MEMORY,
            getattr(self, "agent_id", "unknown"),
            compute_on_retrieval=compute_fn
        )
    
    async def create_experiment_branch(self, experiment_name: str) -> str:
        """Create a branch for experimentation"""
        await self._ensure_persistence()
        
        branch_id = await self._persistence_manager.create_branch(
            getattr(self, "agent_id", "unknown"),
            experiment_name
        )
        
        logger.info(f"Created experiment branch", 
                   branch_id=branch_id,
                   experiment=experiment_name)
        
        return branch_id
    
    async def get_decision_chain(self, decision_id: str) -> List[Dict[str, Any]]:
        """Get the causal chain for a decision"""
        await self._ensure_persistence()
        
        return await self._persistence_manager.get_causal_chain(decision_id)
    
    def _extract_causes(self, context: Dict[str, Any]) -> List[str]:
        """Extract causal factors from context"""
        causes = []
        
        # User input is always a cause
        if "user_input" in context:
            causes.append("user_input")
        
        # Previous decisions influence current
        if self._decision_history:
            causes.append(f"previous_decision:{self._decision_history[-1]['decision']}")
        
        # Context-specific causes
        if context.get("priority") == "high":
            causes.append("high_priority_request")
        
        if context.get("uncertainty", 0) > 0.5:
            causes.append("high_uncertainty")
            
        # Tool usage
        if "tools_used" in context:
            causes.extend([f"tool:{tool}" for tool in context["tools_used"]])
        
        return causes
    
    def _predict_effects(self, decision: str, context: Dict[str, Any]) -> List[str]:
        """Predict effects of a decision"""
        effects = []
        
        # Decision-specific effects
        if decision == "explore":
            effects.append("new_information_gathered")
        elif decision == "execute":
            effects.append("action_performed")
        elif decision == "wait":
            effects.append("delayed_action")
        
        # Context-based effects
        if context.get("risk_level", "low") == "high":
            effects.append("high_risk_outcome_possible")
        
        # Resource effects
        if context.get("energy_cost", 0) > 0.5:
            effects.append("significant_resource_consumption")
        
        return effects
    
    def _generate_counterfactuals(self, 
                                 decision: str, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate counterfactual scenarios"""
        counterfactuals = {}
        
        # Alternative decisions
        alternatives = ["explore", "exploit", "wait", "delegate"]
        alternatives.remove(decision) if decision in alternatives else None
        
        for alt in alternatives[:2]:  # Top 2 alternatives
            counterfactuals[f"could_have_{alt}"] = {
                "probability": 1.0 / (len(alternatives) + 1),
                "expected_outcome": self._predict_outcome(alt, context)
            }
        
        return counterfactuals
    
    def _predict_outcome(self, decision: str, context: Dict[str, Any]) -> str:
        """Predict outcome for a decision"""
        if decision == "explore":
            return "information_gain"
        elif decision == "exploit":
            return "immediate_reward"
        elif decision == "wait":
            return "delayed_decision"
        else:
            return "unknown_outcome"

class PersistentAgent(PersistenceMixin):
    """Base class for agents with persistence"""
    
    def __init__(self, agent_id: str, *args, **kwargs):
        self.agent_id = agent_id
        super().__init__(*args, **kwargs)
    
    async def think_with_memory(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Think using persistent memory"""
        # Load relevant memories
        def enhance_with_thought(memory):
            if memory:
                memory["current_thought"] = thought
                memory["thought_relevance"] = self._calculate_relevance(memory, thought)
            return memory
        
        enhanced_memory = await self.load_memory(compute_fn=enhance_with_thought)
        
        # Make decision based on memory + thought
        decision = await self._make_decision(thought, enhanced_memory)
        
        # Save decision with causality
        await self.save_decision(
            decision["action"],
            {
                "thought": thought,
                "memory_influence": enhanced_memory is not None,
                "confidence": decision["confidence"]
            },
            confidence=decision["confidence"]
        )
        
        return decision
    
    def _calculate_relevance(self, memory: Dict[str, Any], thought: Dict[str, Any]) -> float:
        """Calculate relevance between memory and thought"""
        # Simple relevance calculation
        relevance = 0.0
        
        # Check for common keys
        common_keys = set(memory.keys()) & set(thought.keys())
        relevance += len(common_keys) * 0.1
        
        # Check for similar values
        for key in common_keys:
            if memory.get(key) == thought.get(key):
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    async def _make_decision(self, 
                            thought: Dict[str, Any], 
                            memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision based on thought and memory"""
        # Override in subclasses
        return {
            "action": "explore",
            "confidence": 0.5,
            "reasoning": "Default decision"
        }