"""
Workflow Step Implementations (2025 Architecture)

Each step is a focused class with single responsibility.
"""

import asyncio
from abc import ABC, abstractmethod
import numpy as np
import structlog

from aura_intelligence.config import LNNCouncilConfig
from .models import LNNCouncilState

logger = structlog.get_logger()


class WorkflowStep(ABC):
    """Base class for workflow steps."""
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
    
    @abstractmethod
    async def execute(self, state: LNNCouncilState) -> LNNCouncilState:
        """Execute the step."""
        pass


class AnalyzeRequestStep(WorkflowStep):
    """Analyze GPU allocation request complexity."""
    
    async def execute(self, state: LNNCouncilState) -> LNNCouncilState:
        """Analyze request and determine complexity."""
        request = state.current_request
        if not request:
            raise ValueError("No request to analyze")
        
        # Calculate complexity score
        complexity_factors = [
            request.gpu_count / 8.0,
            request.memory_gb / 80.0,
            request.compute_hours / 168.0,
            (10 - request.priority) / 9.0
        ]
        
        complexity_score = np.mean(complexity_factors)
        state.context["request_complexity"] = complexity_score
        
        # Determine next step
        state.next_step = "gather_context" if complexity_score > 0.8 else "neural_inference"
        
        state.add_message("system", f"Request analyzed: complexity={complexity_score:.3f}")
        return state


class GatherContextStep(WorkflowStep):
    """Gather context from knowledge graphs and memory."""
    
    def __init__(self, config: LNNCouncilConfig):
        super().__init__(config)
        # Initialize context providers (lazy loading)
        self._memory_provider = None
        self._knowledge_provider = None
    
    async def execute(self, state: LNNCouncilState) -> LNNCouncilState:
        """Gather context for decision making using real providers."""
        from .memory_context import MemoryContextProvider
        from .knowledge_context import KnowledgeGraphContextProvider
        
        # Initialize providers if needed
        if self._memory_provider is None:
            self._memory_provider = MemoryContextProvider(self.config)
        
        if self._knowledge_provider is None:
            self._knowledge_provider = KnowledgeGraphContextProvider(self.config)
        
        # Gather context from multiple sources in parallel
        context_tasks = []
        
        # Memory context
        context_tasks.append(self._gather_memory_context(state))
        
        # Knowledge graph context  
        context_tasks.append(self._gather_knowledge_context(state))
        
        # System context
        context_tasks.append(self._gather_system_context(state))
        
        # Execute in parallel
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
        
        # Process results
        context_sources = 0
        for i, result in enumerate(context_results):
            if isinstance(result, Exception):
                logger.warning(f"Context source {i} failed: {result}")
                continue
            
            if i == 0 and result is not None:  # Memory
                state.context_cache["memory_context"] = result
                context_sources += 1
            elif i == 1 and result is not None:  # Knowledge
                state.context_cache["knowledge_context"] = result
                context_sources += 1
            elif i == 2:  # System
                state.context_cache.update(result or {})
                context_sources += 1
        
        state.context["context_sources"] = context_sources
        state.context["context_gathering_completed"] = True
        
        state.next_step = "neural_inference"
        return state
    
    async def _gather_memory_context(self, state: LNNCouncilState):
        """Gather memory context."""
        try:
            return await self._memory_provider.get_memory_context(state)
        except Exception as e:
            logger.warning(f"Memory context failed: {e}")
            return None
    
    async def _gather_knowledge_context(self, state: LNNCouncilState):
        """Gather knowledge graph context."""
        try:
            return await self._knowledge_provider.get_knowledge_context(state)
        except Exception as e:
            logger.warning(f"Knowledge context failed: {e}")
            return None
    
    async def _gather_system_context(self, state: LNNCouncilState):
        """Gather system context."""
        return {
            "current_utilization": {"gpu_usage": 0.75, "queue_length": 12},
            "user_history": {"successful_allocations": 15, "avg_usage": 0.85},
            "system_constraints": {"maintenance_window": None, "capacity_limit": 0.9},
            "available_resources": {"A100": 8, "H100": 4, "V100": 12}
        }


class NeuralInferenceStep(WorkflowStep):
    """Perform neural network inference."""
    
    async def execute(self, state: LNNCouncilState) -> LNNCouncilState:
        """Run neural inference for decision making."""
        from .neural_engine import NeuralDecisionEngine
        
        # Initialize neural engine if needed
        neural_engine = NeuralDecisionEngine(self.config)
        
        # Run inference
        inference_start = asyncio.get_event_loop().time()
        decision_result = await neural_engine.make_decision(state)
        state.neural_inference_time = asyncio.get_event_loop().time() - inference_start
        
        # Update state with results
        state.context.update(decision_result)
        state.confidence_score = decision_result.get("confidence_score", 0.0)
        
        # Check confidence threshold
        if state.confidence_score >= self.config.confidence_threshold:
            state.next_step = "validate_decision"
        else:
            state.fallback_triggered = True
            state.next_step = "validate_decision"
        
        return state


class ValidateDecisionStep(WorkflowStep):
    """Validate decision against constraints."""
    
    async def execute(self, state: LNNCouncilState) -> LNNCouncilState:
        """Validate the decision."""
        decision = state.context.get("neural_decision", "deny")
        request = state.current_request
        
        if not request:
            raise ValueError("No request for validation")
        
        # Simple validation logic
        validation_passed = True
        
        if decision == "approve":
            # Check resource availability (simulated)
            estimated_cost = request.gpu_count * request.compute_hours * 2.5
            budget_limit = 10000.0  # Simulated budget
            
            if estimated_cost > budget_limit:
                validation_passed = False
                state.context["neural_decision"] = "deny"
                state.context["override_reason"] = "budget_exceeded"
        
        state.context["validation_passed"] = validation_passed
        state.next_step = "finalize_output"
        
        return state


class FinalizeOutputStep(WorkflowStep):
    """Finalize the decision output."""
    
    async def execute(self, state: LNNCouncilState) -> LNNCouncilState:
        """Finalize the decision."""
        state.completed = True
        state.next_step = None
        
        logger.info(
            "Decision finalized",
            decision=state.context.get("neural_decision"),
            confidence=state.confidence_score,
            fallback_used=state.fallback_triggered
        )
        
        return state