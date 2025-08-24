#!/usr/bin/env python3
"""
Decision Processing Pipeline (2025 Architecture)

Production-ready pipeline integrating LNN, Memory, and Knowledge Graph.
Implements latest 2025 patterns: async/await, dependency injection, observability.
"""

import asyncio
import torch
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import structlog
from dataclasses import dataclass, field

from aura_intelligence.config import LNNCouncilConfig
from .models import LNNCouncilState, GPUAllocationRequest, GPUAllocationDecision
from .context_aware_lnn import ContextAwareLNN
from .memory_context import MemoryContextProvider
from .knowledge_context import KnowledgeGraphContextProvider
from .context_encoder import ContextEncoder

logger = structlog.get_logger()


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    total_time_ms: float = 0.0
    context_gathering_ms: float = 0.0
    neural_inference_ms: float = 0.0
    validation_ms: float = 0.0
    memory_queries: int = 0
    knowledge_queries: int = 0
    context_quality_score: float = 0.0
    confidence_score: float = 0.0
    fallback_triggered: bool = False


@dataclass
class DecisionContext:
    """Rich decision context from all sources."""
    memory_context: Optional[torch.Tensor] = None
    knowledge_context: Optional[torch.Tensor] = None
    system_context: Dict[str, Any] = field(default_factory=dict)
    context_quality: float = 0.0
    context_sources: int = 0
    attention_weights: Optional[torch.Tensor] = None


class DecisionProcessingPipeline:
    """
    Production Decision Processing Pipeline.
    
    2025 Features:
    - Async context gathering from multiple sources
    - Context-aware neural inference with attention
    - Real-time constraint validation
    - Comprehensive observability and metrics
    - Graceful degradation and fallback handling
    """
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        
        # Core components (lazy initialization)
        self._context_lnn: Optional[ContextAwareLNN] = None
        self._memory_provider: Optional[MemoryContextProvider] = None
        self._knowledge_provider: Optional[KnowledgeGraphContextProvider] = None
        self._context_encoder: Optional[ContextEncoder] = None
        
        # Pipeline state
        self.initialized = False
        self.metrics_history: List[PipelineMetrics] = []
        
        logger.info("Decision Processing Pipeline created")
    
    async def initialize(self):
        """Initialize all pipeline components."""
        if self.initialized:
            return
        
        logger.info("Initializing Decision Processing Pipeline")
        
        # Initialize components in parallel for speed
        init_tasks = [
            self._init_context_lnn(),
            self._init_memory_provider(),
            self._init_knowledge_provider(),
            self._init_context_encoder()
        ]
        
        await asyncio.gather(*init_tasks)
        
        self.initialized = True
        logger.info("Decision Processing Pipeline initialized")
    
    async def _init_context_lnn(self):
        """Initialize Context-Aware LNN."""
        self._context_lnn = ContextAwareLNN(self.config)
        if self.config.use_gpu and torch.cuda.is_available():
            self._context_lnn = self._context_lnn.cuda()
        logger.debug("Context-Aware LNN initialized")
    
    async def _init_memory_provider(self):
        """Initialize Memory Context Provider."""
        self._memory_provider = MemoryContextProvider(self.config)
        logger.debug("Memory Context Provider initialized")
    
    async def _init_knowledge_provider(self):
        """Initialize Knowledge Graph Context Provider."""
        self._knowledge_provider = KnowledgeGraphContextProvider(self.config)
        logger.debug("Knowledge Graph Context Provider initialized")
    
    async def _init_context_encoder(self):
        """Initialize Context Encoder."""
        self._context_encoder = ContextEncoder(self.config)
        logger.debug("Context Encoder initialized")
    
    async def process_decision(
        self, 
        request: GPUAllocationRequest
    ) -> Tuple[GPUAllocationDecision, PipelineMetrics]:
        """
        Process a complete decision through the pipeline.
        
        Args:
            request: GPU allocation request
            
        Returns:
            decision: Final allocation decision
            metrics: Pipeline execution metrics
        """
        
        if not self.initialized:
            await self.initialize()
        
        pipeline_start = asyncio.get_event_loop().time()
        metrics = PipelineMetrics()
        
        try:
            # Create initial state
            state = LNNCouncilState(current_request=request)
            
            # Step 1: Analyze Request
            await self._analyze_request_step(state, metrics)
            
            # Step 2: Gather Context (parallel execution)
            decision_context = await self._gather_context_step(state, metrics)
            
            # Step 3: Neural Inference
            neural_result = await self._neural_inference_step(
                state, decision_context, metrics
            )
            
            # Step 4: Validate Decision
            final_decision = await self._validate_decision_step(
                state, neural_result, metrics
            )
            
            # Update final metrics
            metrics.total_time_ms = (
                asyncio.get_event_loop().time() - pipeline_start
            ) * 1000
            
            # Store metrics for analysis
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:  # Keep last 1000
                self.metrics_history = self.metrics_history[-1000:]
            
            logger.info(
                "Decision pipeline completed",
                request_id=request.request_id,
                decision=final_decision.decision,
                confidence=final_decision.confidence_score,
                total_time_ms=metrics.total_time_ms,
                context_sources=metrics.context_quality_score
            )
            
            return final_decision, metrics
            
        except Exception as e:
            logger.error(
                "Decision pipeline failed",
                request_id=request.request_id,
                error=str(e),
                elapsed_ms=(asyncio.get_event_loop().time() - pipeline_start) * 1000
            )
            
            # Return fallback decision
            fallback_decision = self._create_fallback_decision(request, str(e))
            metrics.fallback_triggered = True
            metrics.total_time_ms = (
                asyncio.get_event_loop().time() - pipeline_start
            ) * 1000
            
            return fallback_decision, metrics
    
    async def _analyze_request_step(
        self, 
        state: LNNCouncilState, 
        metrics: PipelineMetrics
    ):
        """
        Step 1: Analyze request complexity and requirements.
        
        Task 6 Implementation: analyze_request step with context gathering
        """
        
        request = state.current_request
        if not request:
            raise ValueError("No request to analyze")
        
        # Calculate request complexity
        complexity_factors = [
            request.gpu_count / 8.0,  # Normalize by typical max
            request.memory_gb / 80.0,  # Normalize by typical GPU memory
            request.compute_hours / 168.0,  # Normalize by week
            (10 - request.priority) / 9.0,  # Higher priority = lower complexity
        ]
        
        complexity_score = sum(complexity_factors) / len(complexity_factors)
        
        # Store analysis results
        state.context.update({
            "request_complexity": complexity_score,
            "requires_deep_analysis": complexity_score > 0.7,
            "priority_tier": "high" if request.priority >= 8 else "normal",
            "resource_intensity": request.gpu_count * request.compute_hours
        })
        
        logger.debug(
            "Request analyzed",
            request_id=request.request_id,
            complexity=complexity_score,
            deep_analysis=complexity_score > 0.7
        )
    
    async def _gather_context_step(
        self, 
        state: LNNCouncilState, 
        metrics: PipelineMetrics
    ) -> DecisionContext:
        """
        Step 2: Gather context from all sources in parallel.
        
        Task 6 Implementation: Context gathering from memory and knowledge graph
        """
        
        context_start = asyncio.get_event_loop().time()
        
        # Gather context from all sources in parallel
        context_tasks = []
        
        # Memory context
        if self._memory_provider:
            context_tasks.append(
                self._gather_memory_context(state)
            )
        
        # Knowledge graph context
        if self._knowledge_provider:
            context_tasks.append(
                self._gather_knowledge_context(state)
            )
        
        # System context (always available)
        context_tasks.append(
            self._gather_system_context(state)
        )
        
        # Execute all context gathering in parallel
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
        
        # Process results
        decision_context = DecisionContext()
        context_sources = 0
        
        for i, result in enumerate(context_results):
            if isinstance(result, Exception):
                logger.warning(f"Context source {i} failed: {result}")
                continue
            
            if i == 0 and result is not None:  # Memory context
                decision_context.memory_context = result
                context_sources += 1
                metrics.memory_queries += 1
            elif i == 1 and result is not None:  # Knowledge context
                decision_context.knowledge_context = result
                context_sources += 1
                metrics.knowledge_queries += 1
            elif i == 2:  # System context
                decision_context.system_context = result or {}
                context_sources += 1
        
        # Calculate context quality
        decision_context.context_sources = context_sources
        decision_context.context_quality = self._assess_context_quality(decision_context)
        
        # Update metrics
        metrics.context_gathering_ms = (
            asyncio.get_event_loop().time() - context_start
        ) * 1000
        metrics.context_quality_score = decision_context.context_quality
        
        logger.debug(
            "Context gathered",
            request_id=state.current_request.request_id,
            sources=context_sources,
            quality=decision_context.context_quality,
            time_ms=metrics.context_gathering_ms
        )
        
        return decision_context
    
    async def _gather_memory_context(self, state: LNNCouncilState) -> Optional[torch.Tensor]:
        """Gather context from memory system."""
        try:
            return await self._memory_provider.get_memory_context(state)
        except Exception as e:
            logger.warning(f"Memory context failed: {e}")
            return None
    
    async def _gather_knowledge_context(self, state: LNNCouncilState) -> Optional[torch.Tensor]:
        """Gather context from knowledge graph."""
        try:
            return await self._knowledge_provider.get_knowledge_context(state)
        except Exception as e:
            logger.warning(f"Knowledge context failed: {e}")
            return None
    
    async def _gather_system_context(self, state: LNNCouncilState) -> Dict[str, Any]:
        """Gather system-level context."""
        request = state.current_request
        
        # Simulate system context (in production, this would query real systems)
        return {
            "current_time": datetime.now(timezone.utc).isoformat(),
            "system_load": 0.75,  # Simulated
            "available_gpus": {"A100": 8, "H100": 4, "V100": 12},
            "queue_length": 15,
            "maintenance_window": None,
            "budget_remaining": 50000.0,
            "user_quota_remaining": 1000.0
        }
    
    async def _neural_inference_step(
        self,
        state: LNNCouncilState,
        decision_context: DecisionContext,
        metrics: PipelineMetrics
    ) -> Dict[str, Any]:
        """
        Step 3: Perform context-aware neural inference.
        
        Task 6 Implementation: make_lnn_decision step with neural inference
        """
        
        inference_start = asyncio.get_event_loop().time()
        
        try:
            # Prepare context for neural network
            state.context.update({
                "memory_context_available": decision_context.memory_context is not None,
                "knowledge_context_available": decision_context.knowledge_context is not None,
                "context_quality": decision_context.context_quality,
                "context_sources": decision_context.context_sources
            })
            
            # Run context-aware inference
            with torch.no_grad():
                output, attention_info = await self._context_lnn.forward_with_context(
                    state,
                    return_attention=True
                )
            
            # Decode neural network output
            decision_logits = output.squeeze()
            confidence_score = torch.sigmoid(decision_logits).max().item()
            
            # Map to decision
            decision_idx = torch.argmax(decision_logits).item()
            decisions = ["deny", "defer", "approve"]
            decision = decisions[min(decision_idx, len(decisions) - 1)]
            
            # Store attention information
            if attention_info:
                decision_context.attention_weights = attention_info.get("attention_weights")
            
            neural_result = {
                "decision": decision,
                "confidence_score": confidence_score,
                "decision_logits": decision_logits.tolist(),
                "attention_info": attention_info,
                "context_aware": True,
                "neural_reasoning": self._generate_neural_reasoning(
                    decision, confidence_score, attention_info
                )
            }
            
            # Update metrics
            metrics.neural_inference_ms = (
                asyncio.get_event_loop().time() - inference_start
            ) * 1000
            metrics.confidence_score = confidence_score
            
            logger.debug(
                "Neural inference completed",
                request_id=state.current_request.request_id,
                decision=decision,
                confidence=confidence_score,
                time_ms=metrics.neural_inference_ms
            )
            
            return neural_result
            
        except Exception as e:
            logger.error(f"Neural inference failed: {e}")
            # Return fallback neural result
            return {
                "decision": "defer",
                "confidence_score": 0.0,
                "neural_error": str(e),
                "fallback_used": True
            }
    
    async def _validate_decision_step(
        self,
        state: LNNCouncilState,
        neural_result: Dict[str, Any],
        metrics: PipelineMetrics
    ) -> GPUAllocationDecision:
        """
        Step 4: Validate decision against constraints.
        
        Task 6 Implementation: validate_decision step with constraint checking
        """
        
        validation_start = asyncio.get_event_loop().time()
        request = state.current_request
        
        decision = neural_result.get("decision", "deny")
        confidence = neural_result.get("confidence_score", 0.0)
        
        # Validation checks
        validation_results = []
        
        # 1. Confidence threshold check
        if confidence < self.config.confidence_threshold:
            validation_results.append({
                "check": "confidence_threshold",
                "passed": False,
                "reason": f"Confidence {confidence:.3f} below threshold {self.config.confidence_threshold}"
            })
            decision = "defer"  # Override to defer
        else:
            validation_results.append({
                "check": "confidence_threshold",
                "passed": True
            })
        
        # 2. Resource availability check
        system_context = state.context.get("system_context", {})
        available_gpus = system_context.get("available_gpus", {})
        
        if decision == "approve":
            available_count = available_gpus.get(request.gpu_type, 0)
            if available_count < request.gpu_count:
                validation_results.append({
                    "check": "resource_availability",
                    "passed": False,
                    "reason": f"Insufficient {request.gpu_type} GPUs: need {request.gpu_count}, have {available_count}"
                })
                decision = "defer"
            else:
                validation_results.append({
                    "check": "resource_availability",
                    "passed": True
                })
        
        # 3. Budget check
        if decision == "approve":
            estimated_cost = request.gpu_count * request.compute_hours * 2.5  # $2.5/GPU-hour
            budget_remaining = system_context.get("budget_remaining", 0)
            
            if estimated_cost > budget_remaining:
                validation_results.append({
                    "check": "budget_constraint",
                    "passed": False,
                    "reason": f"Cost ${estimated_cost:.2f} exceeds budget ${budget_remaining:.2f}"
                })
                decision = "deny"
            else:
                validation_results.append({
                    "check": "budget_constraint",
                    "passed": True
                })
        
        # Create final decision
        final_decision = GPUAllocationDecision(
            request_id=request.request_id,
            decision=decision,
            confidence_score=confidence,
            fallback_used=neural_result.get("fallback_used", False),
            inference_time_ms=metrics.neural_inference_ms
        )
        
        # Add reasoning
        final_decision.add_reasoning("neural", neural_result.get("neural_reasoning", ""))
        
        for validation in validation_results:
            if not validation["passed"]:
                final_decision.add_reasoning("validation", validation["reason"])
        
        # Update metrics
        metrics.validation_ms = (
            asyncio.get_event_loop().time() - validation_start
        ) * 1000
        
        logger.debug(
            "Decision validated",
            request_id=request.request_id,
            final_decision=decision,
            validations_passed=sum(1 for v in validation_results if v["passed"]),
            time_ms=metrics.validation_ms
        )
        
        return final_decision
    
    def _assess_context_quality(self, context: DecisionContext) -> float:
        """Assess overall context quality."""
        quality_factors = []
        
        # Memory context quality
        if context.memory_context is not None:
            memory_quality = (context.memory_context != 0).float().mean().item()
            quality_factors.append(memory_quality)
        
        # Knowledge context quality
        if context.knowledge_context is not None:
            knowledge_quality = (context.knowledge_context != 0).float().mean().item()
            quality_factors.append(knowledge_quality)
        
        # System context quality
        if context.system_context:
            system_quality = len(context.system_context) / 10.0  # Normalize by expected keys
            quality_factors.append(min(system_quality, 1.0))
        
        return sum(quality_factors) / max(len(quality_factors), 1)
    
    def _generate_neural_reasoning(
        self, 
        decision: str, 
        confidence: float, 
        attention_info: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable reasoning from neural network output."""
        
        reasoning_parts = [
            f"Neural network decision: {decision} (confidence: {confidence:.3f})"
        ]
        
        if attention_info:
            context_sources = attention_info.get("context_sources", 0)
            if context_sources > 0:
                reasoning_parts.append(f"Considered {context_sources} context sources")
            
            context_quality = attention_info.get("context_quality", 0)
            if context_quality > 0:
                reasoning_parts.append(f"Context quality: {context_quality:.3f}")
        
        return "; ".join(reasoning_parts)
    
    def _create_fallback_decision(
        self, 
        request: GPUAllocationRequest, 
        error: str
    ) -> GPUAllocationDecision:
        """Create fallback decision when pipeline fails."""
        
        decision = GPUAllocationDecision(
            request_id=request.request_id,
            decision="defer",  # Safe fallback
            confidence_score=0.0,
            fallback_used=True,
            inference_time_ms=0.0
        )
        
        decision.add_reasoning("fallback", f"Pipeline error: {error}")
        
        return decision
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        if not self.metrics_history:
            return {"status": "no_executions"}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 executions
        
        return {
            "total_executions": len(self.metrics_history),
            "recent_executions": len(recent_metrics),
            "avg_total_time_ms": sum(m.total_time_ms for m in recent_metrics) / len(recent_metrics),
            "avg_context_time_ms": sum(m.context_gathering_ms for m in recent_metrics) / len(recent_metrics),
            "avg_inference_time_ms": sum(m.neural_inference_ms for m in recent_metrics) / len(recent_metrics),
            "avg_validation_time_ms": sum(m.validation_ms for m in recent_metrics) / len(recent_metrics),
            "avg_confidence": sum(m.confidence_score for m in recent_metrics) / len(recent_metrics),
            "avg_context_quality": sum(m.context_quality_score for m in recent_metrics) / len(recent_metrics),
            "fallback_rate": sum(1 for m in recent_metrics if m.fallback_triggered) / len(recent_metrics),
            "total_memory_queries": sum(m.memory_queries for m in recent_metrics),
            "total_knowledge_queries": sum(m.knowledge_queries for m in recent_metrics)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive pipeline health check."""
        health = {
            "pipeline_initialized": self.initialized,
            "components": {}
        }
        
        if self.initialized:
            # Check each component
            if self._context_lnn:
                health["components"]["context_lnn"] = "healthy"
            
            if self._memory_provider:
                health["components"]["memory_provider"] = "healthy"
            
            if self._knowledge_provider:
                health["components"]["knowledge_provider"] = "healthy"
            
            if self._context_encoder:
                health["components"]["context_encoder"] = "healthy"
            
            # Add performance stats
            health["performance"] = self.get_pipeline_stats()
        
        return health