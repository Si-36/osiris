"""
Memory Integration Layer (2025 Architecture)

Advanced memory integration with Mem0 for learning and context.
Implements latest 2025 research in memory-augmented neural networks.
"""

import torch
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
import structlog

from aura_intelligence.config import LNNCouncilConfig
from .models import LNNCouncilState, GPUAllocationRequest, GPUAllocationDecision

logger = structlog.get_logger()


class LNNMemoryIntegration:
    """
    Advanced Memory Integration Layer for LNN Council Agent.
    
    2025 Features:
    - Episodic memory for decision patterns
    - Semantic similarity search
    - Meta-learning from outcomes
    - Memory-guided neural attention
    - Adaptive forgetting mechanisms
    """
    
    def __init__(self, config: LNNCouncilConfig):
        self.config = config
        self.mem0_adapter = None  # Will be injected
        
        # Memory caches (2025 pattern: multi-level caching)
        self.episodic_cache = {}  # Recent decisions
        self.semantic_cache = {}  # Similar patterns
        self.meta_cache = {}      # Learning patterns
        
        # Learning statistics
        self.decision_count = 0
        self.learning_updates = 0
        self.memory_quality_score = 0.0
        
        logger.info("LNN Memory Integration initialized")
    
    def set_mem0_adapter(self, adapter):
        """Inject Mem0 adapter (dependency injection pattern)."""
        pass
        self.mem0_adapter = adapter
        logger.info("Mem0 adapter connected to LNN Memory Integration")
    
        async def get_memory_context(self, state: LNNCouncilState) -> Optional[torch.Tensor]:
        """
        Get comprehensive memory context for decision making.
        
        2025 Features:
        - Multi-level memory retrieval
        - Semantic similarity matching
        - Temporal pattern analysis
        - Context quality assessment
        
        Args:
            state: Current agent state
            
        Returns:
            Memory context tensor with episodic and semantic features
        """
        
        request = state.current_request
        if not request:
            return None
        
        try:
            # 1. Episodic memory: Recent similar decisions
            episodic_context = await self._get_episodic_context(request)
            
            # 2. Semantic memory: Pattern-based similarities
            semantic_context = await self._get_semantic_context(request)
            
            # 3. Meta-learning: Decision outcome patterns
            meta_context = await self._get_meta_learning_context(request)
            
            # 4. Temporal patterns: Time-based decision trends
            temporal_context = await self._get_temporal_patterns(request)
            
            # 5. Combine all memory sources
            combined_context = self._combine_memory_contexts([
                episodic_context,
                semantic_context,
                meta_context,
                temporal_context
            ])
            
            if combined_context is not None:
                quality_score = self._assess_memory_quality(combined_context)
                self.memory_quality_score = quality_score
                
                logger.info(
                    "Memory context retrieved",
                    user_id=request.user_id,
                    project_id=request.project_id,
                    memory_quality=quality_score,
                    context_sources=4
                )
            
            return combined_context
            
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
            return None
    
        async def _get_episodic_context(self, request: GPUAllocationRequest) -> Optional[Dict[str, Any]]:
        """Get episodic memory context (recent similar decisions)."""
        
        if not self.mem0_adapter:
            return self._get_mock_episodic_context(request)
        
        try:
            # Search for recent decisions by this user
            from aura_intelligence.adapters.mem0_adapter import SearchQuery
            
            query = SearchQuery(
                agent_id="lnn_council_agent",
                query_text=f"user:{request.user_id} GPU allocation decision",
                limit=5,
                time_range=(
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
            )
            
            memories = await self.mem0_adapter.search_memories(query)
            
            if not memories:
                return None
            
            # Analyze recent decision patterns
            recent_decisions = []
            for memory in memories:
                metadata = memory.metadata
                recent_decisions.append({
                    "decision": metadata.get("decision", "deny"),
                    "confidence": metadata.get("confidence", 0.5),
                    "gpu_count": metadata.get("gpu_count", 1),
                    "outcome_success": metadata.get("outcome_success", True)
                })
            
            # Calculate episodic features
            approval_rate = sum(1 for d in recent_decisions if d["decision"] == "approve") / len(recent_decisions)
            avg_confidence = sum(d["confidence"] for d in recent_decisions) / len(recent_decisions)
            success_rate = sum(1 for d in recent_decisions if d["outcome_success"]) / len(recent_decisions)
            
            return {
                "episodic_approval_rate": approval_rate,
                "episodic_confidence": avg_confidence,
                "episodic_success_rate": success_rate,
                "episodic_count": len(recent_decisions),
                "episodic_recency": 1.0  # All recent
            }
            
        except Exception as e:
            logger.warning(f"Failed to get episodic context: {e}")
            return None
    
        async def _get_semantic_context(self, request: GPUAllocationRequest) -> Optional[Dict[str, Any]]:
        """Get semantic memory context (pattern-based similarities)."""
        
        if not self.mem0_adapter:
            return self._get_mock_semantic_context(request)
        
        try:
            # Semantic search for similar GPU requests
            semantic_query = f"GPU:{request.gpu_type} memory:{request.memory_gb}GB hours:{request.compute_hours}"
            
            from aura_intelligence.adapters.mem0_adapter import SearchQuery
            
            query = SearchQuery(
                agent_id="lnn_council_agent",
                query_text=semantic_query,
                limit=10,
                metadata_filters={"decision_type": "gpu_allocation"}
            )
            
            memories = await self.mem0_adapter.search_memories(query)
            
            if not memories:
                return None
            
            # Analyze semantic patterns
            similar_decisions = []
            for memory in memories:
                metadata = memory.metadata
                similarity_score = memory.similarity_score if hasattr(memory, 'similarity_score') else 0.8
                
                similar_decisions.append({
                    "decision": metadata.get("decision", "deny"),
                    "similarity": similarity_score,
                    "outcome": metadata.get("outcome_success", True)
                })
            
            # Calculate semantic features
            weighted_approval = sum(
                (1 if d["decision"] == "approve" else 0) * d["similarity"] 
                for d in similar_decisions
            ) / sum(d["similarity"] for d in similar_decisions)
            
            pattern_strength = len(similar_decisions) / 10.0  # Normalize
            avg_similarity = sum(d["similarity"] for d in similar_decisions) / len(similar_decisions)
            
            return {
                "semantic_approval_rate": weighted_approval,
                "semantic_pattern_strength": pattern_strength,
                "semantic_similarity": avg_similarity,
                "semantic_count": len(similar_decisions)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            return None
    
        async def _get_meta_learning_context(self, request: GPUAllocationRequest) -> Optional[Dict[str, Any]]:
        """Get meta-learning context (learning from decision outcomes)."""
        
        if not self.mem0_adapter:
            return self._get_mock_meta_context(request)
        
        try:
            # Search for decision outcomes and learning patterns
            from aura_intelligence.adapters.mem0_adapter import SearchQuery
            
            query = SearchQuery(
                agent_id="lnn_council_agent",
                query_text="decision outcome learning pattern",
                limit=20,
                metadata_filters={"has_outcome": True}
            )
            
            memories = await self.mem0_adapter.search_memories(query)
            
            if not memories:
                return None
            
            # Analyze learning patterns
            learning_data = []
            for memory in memories:
                metadata = memory.metadata
                learning_data.append({
                    "predicted_confidence": metadata.get("confidence", 0.5),
                    "actual_outcome": metadata.get("outcome_success", True),
                    "decision": metadata.get("decision", "deny"),
                    "learning_weight": metadata.get("learning_weight", 1.0)
                })
            
            # Calculate meta-learning features
            confidence_calibration = self._calculate_confidence_calibration(learning_data)
            decision_accuracy = sum(1 for d in learning_data if d["actual_outcome"]) / len(learning_data)
            learning_trend = self._calculate_learning_trend(learning_data)
            
            return {
                "meta_confidence_calibration": confidence_calibration,
                "meta_decision_accuracy": decision_accuracy,
                "meta_learning_trend": learning_trend,
                "meta_sample_size": len(learning_data)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get meta-learning context: {e}")
            return None
    
        async def _get_temporal_patterns(self, request: GPUAllocationRequest) -> Optional[Dict[str, Any]]:
        """Get temporal decision patterns."""
        
        if not self.mem0_adapter:
            return self._get_mock_temporal_context(request)
        
        try:
            # Get decisions from different time periods
            now = datetime.now()
            time_periods = [
                (now - timedelta(days=7), now, "recent"),
                (now - timedelta(days=30), now - timedelta(days=7), "medium"),
                (now - timedelta(days=90), now - timedelta(days=30), "distant")
            ]
            
            temporal_data = {}
            
            for start_time, end_time, period_name in time_periods:
                from aura_intelligence.adapters.mem0_adapter import SearchQuery
                
                query = SearchQuery(
                    agent_id="lnn_council_agent",
                    query_text=f"GPU allocation {request.gpu_type}",
                    limit=10,
                    time_range=(start_time, end_time)
                )
                
                memories = await self.mem0_adapter.search_memories(query)
                
                if memories:
                    approval_rate = sum(
                        1 for m in memories 
                        if m.metadata.get("decision") == "approve"
                    ) / len(memories)
                    
                    temporal_data[f"{period_name}_approval_rate"] = approval_rate
                    temporal_data[f"{period_name}_count"] = len(memories)
                else:
                    temporal_data[f"{period_name}_approval_rate"] = 0.5
                    temporal_data[f"{period_name}_count"] = 0
            
            # Calculate temporal trends
            trend_score = self._calculate_temporal_trend(temporal_data)
            temporal_data["temporal_trend"] = trend_score
            
            return temporal_data
            
        except Exception as e:
            logger.warning(f"Failed to get temporal patterns: {e}")
            return None

        async def _get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get historical context for a user."""
        
        if not self.mem0_adapter:
            return self._get_mock_user_context(user_id)
        
        try:
            # Query Mem0 for user's historical decisions
            user_memories = await self.mem0_adapter.search(
                query=f"user:{user_id} GPU allocation decisions",
                limit=10
            )
            
            if not user_memories:
                return None
            
            # Aggregate user patterns
            total_requests = len(user_memories)
            approved_requests = sum(1 for m in user_memories if m.get("decision") == "approve")
            avg_gpu_count = sum(m.get("gpu_count", 0) for m in user_memories) / total_requests
            
            return {
                "total_requests": total_requests,
                "approval_rate": approved_requests / total_requests,
                "avg_gpu_count": avg_gpu_count,
                "reliability_score": min(total_requests / 20.0, 1.0)  # Max at 20 requests
            }
            
        except Exception as e:
            logger.warning(f"Failed to get user context: {e}")
            return None
    
        async def _get_project_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get historical context for a project."""
        
        if not self.mem0_adapter:
            return self._get_mock_project_context(project_id)
        
        try:
            # Query Mem0 for project's resource usage patterns
            project_memories = await self.mem0_adapter.search(
                query=f"project:{project_id} resource utilization",
                limit=15
            )
            
            if not project_memories:
                return None
            
            # Aggregate project patterns
            total_gpu_hours = sum(m.get("gpu_hours_used", 0) for m in project_memories)
            avg_utilization = sum(m.get("utilization", 0) for m in project_memories) / len(project_memories)
            budget_efficiency = sum(m.get("cost_efficiency", 0.5) for m in project_memories) / len(project_memories)
            
            return {
                "total_gpu_hours": total_gpu_hours,
                "avg_utilization": avg_utilization,
                "budget_efficiency": budget_efficiency,
                "project_maturity": min(len(project_memories) / 10.0, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get project context: {e}")
            return None
    
        async def _get_similar_requests(self, request) -> Optional[Dict[str, Any]]:
        """Get context from similar historical requests."""
        pass
        
        if not self.mem0_adapter:
            return self._get_mock_similar_context(request)
        
        try:
            # Create semantic query for similar requests
            query = f"GPU:{request.gpu_type} count:{request.gpu_count} memory:{request.memory_gb}GB"
            
            similar_memories = await self.mem0_adapter.search(
                query=query,
                limit=5
            )
            
            if not similar_memories:
                return None
            
            # Analyze similar request outcomes
            similar_decisions = [m.get("decision", "deny") for m in similar_memories]
            approval_rate = similar_decisions.count("approve") / len(similar_decisions)
            
            avg_confidence = sum(m.get("confidence", 0.5) for m in similar_memories) / len(similar_memories)
            
            return {
                "similar_count": len(similar_memories),
                "similar_approval_rate": approval_rate,
                "avg_confidence": avg_confidence,
                "pattern_strength": min(len(similar_memories) / 5.0, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get similar requests context: {e}")
            return None
    
    def _combine_contexts(self, contexts: List[Optional[Dict[str, Any]]]) -> Optional[torch.Tensor]:
        """Combine multiple context sources into a single tensor."""
        
        # Filter out None contexts
        valid_contexts = [ctx for ctx in contexts if ctx is not None]
        
        if not valid_contexts:
            return None
        
        # Extract features from each context
        features = []
        
        for ctx in valid_contexts:
            if "approval_rate" in ctx:  # User context
                features.extend([
                    ctx.get("approval_rate", 0.5),
                    ctx.get("avg_gpu_count", 0) / 8.0,
                    ctx.get("reliability_score", 0.5)
                ])
            
            if "avg_utilization" in ctx:  # Project context
                features.extend([
                    ctx.get("avg_utilization", 0.5),
                    ctx.get("budget_efficiency", 0.5),
                    ctx.get("project_maturity", 0.5)
                ])
            
            if "similar_approval_rate" in ctx:  # Similar requests
                features.extend([
                    ctx.get("similar_approval_rate", 0.5),
                    ctx.get("avg_confidence", 0.5),
                    ctx.get("pattern_strength", 0.5)
                ])
        
        # Pad to input size
        while len(features) < self.config.input_size:
            features.append(0.0)
        features = features[:self.config.input_size]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _assess_context_quality(self, context_tensor: torch.Tensor) -> float:
        """Assess the quality of memory context."""
        
        # Simple quality metric based on non-zero features
        non_zero_features = (context_tensor != 0).sum().item()
        total_features = context_tensor.numel()
        
        return non_zero_features / total_features
    
    # Mock methods for testing without Mem0
    def _get_mock_user_context(self, user_id: str) -> Dict[str, Any]:
        """Mock user context for testing."""
        return {
            "total_requests": 15,
            "approval_rate": 0.8,
            "avg_gpu_count": 2.5,
            "reliability_score": 0.75
        }
    
    def _get_mock_project_context(self, project_id: str) -> Dict[str, Any]:
        """Mock project context for testing."""
        return {
            "total_gpu_hours": 240.0,
            "avg_utilization": 0.85,
            "budget_efficiency": 0.9,
            "project_maturity": 0.6
        }
    
    def _get_mock_similar_context(self, request) -> Dict[str, Any]:
        """Mock similar requests context for testing."""
        pass
        return {
            "similar_count": 8,
            "similar_approval_rate": 0.75,
            "avg_confidence": 0.82,
            "pattern_strength": 1.0
        }
    
        async def store_decision_outcome(
        self, 
        request: GPUAllocationRequest, 
        decision: GPUAllocationDecision,
        outcome: Optional[Dict[str, Any]] = None
        ):
        """
        Store decision outcome for future learning.
        
        2025 Features:
        - Structured memory storage
        - Learning weight calculation
        - Outcome tracking
        - Meta-learning updates
        """
        
        if not self.mem0_adapter:
            logger.info("Mock: Storing decision outcome", 
                       decision=decision.decision, 
                       confidence=decision.confidence_score)
            return
        
        try:
            # Create comprehensive memory record
            memory_content = f"GPU allocation decision: {decision.decision} for {request.user_id}"
            
            memory_metadata = {
                # Request details
                "user_id": request.user_id,
                "project_id": request.project_id,
                "gpu_type": request.gpu_type,
                "gpu_count": request.gpu_count,
                "memory_gb": request.memory_gb,
                "compute_hours": request.compute_hours,
                "priority": request.priority,
                
                # Decision details
                "decision": decision.decision,
                "confidence": decision.confidence_score,
                "fallback_used": decision.fallback_used,
                "inference_time_ms": decision.inference_time_ms,
                
                # Learning metadata
                "decision_type": "gpu_allocation",
                "has_outcome": outcome is not None,
                "learning_weight": self._calculate_learning_weight(decision, outcome),
                "timestamp": request.created_at.isoformat()
            }
            
            # Add outcome data if available
            if outcome:
                memory_metadata.update({
                    "outcome_success": outcome.get("success", True),
                    "actual_utilization": outcome.get("utilization", 0.8),
                    "cost_efficiency": outcome.get("cost_efficiency", 0.7),
                    "user_satisfaction": outcome.get("satisfaction", 0.8)
                })
            
            # Store in Mem0
            from aura_intelligence.adapters.mem0_adapter import Memory, MemoryType
            
            memory = Memory(
                agent_id="lnn_council_agent",
                memory_type=MemoryType.EPISODIC,
                content=memory_content,
                metadata=memory_metadata,
                timestamp=datetime.now()
            )
            
            memory_id = await self.mem0_adapter.add_memory(memory)
            
            # Update learning statistics
            self.decision_count += 1
            if outcome:
                self.learning_updates += 1
            
            logger.info(
                "Decision outcome stored in memory", 
                memory_id=memory_id,
                decision=decision.decision,
                has_outcome=outcome is not None
            )
            
        except Exception as e:
            logger.warning(f"Failed to store decision outcome: {e}")
    
        async def learn_from_outcome(
        self,
        request: GPUAllocationRequest,
        decision: GPUAllocationDecision,
        actual_outcome: Dict[str, Any]
        ):
        """
        Learn from decision outcomes to improve future decisions.
        
        2025 Meta-Learning:
        - Confidence calibration
        - Pattern recognition
        - Adaptive thresholds
        """
        
        try:
            # Calculate learning signals
            prediction_error = self._calculate_prediction_error(decision, actual_outcome)
            confidence_error = self._calculate_confidence_error(decision, actual_outcome)
            
            # Update learning statistics
            learning_signal = {
                "prediction_error": prediction_error,
                "confidence_error": confidence_error,
                "decision_quality": actual_outcome.get("success", True),
                "learning_rate": self._adaptive_learning_rate()
            }
            
            # Store learning signal
            if self.mem0_adapter:
                learning_content = f"Learning signal for decision {decision.request_id}"
                learning_metadata = {
                    "learning_type": "outcome_feedback",
                    "request_id": decision.request_id,
                    "user_id": request.user_id,
                    **learning_signal
                }
                
                from aura_intelligence.adapters.mem0_adapter import Memory, MemoryType
                
                learning_memory = Memory(
                    agent_id="lnn_council_agent",
                    memory_type=MemoryType.SEMANTIC,
                    content=learning_content,
                    metadata=learning_metadata,
                    timestamp=datetime.now()
                )
                
                await self.mem0_adapter.add_memory(learning_memory)
            
            logger.info(
                "Learning from outcome completed",
                prediction_error=prediction_error,
                confidence_error=confidence_error
            )
            
        except Exception as e:
            logger.warning(f"Failed to learn from outcome: {e}")
    
    def _calculate_learning_weight(self, decision: GPUAllocationDecision, outcome: Optional[Dict[str, Any]]) -> float:
        """Calculate learning weight for this decision."""
        base_weight = 1.0
        
        # Higher weight for decisions with outcomes
        if outcome:
            base_weight *= 1.5
        
        # Higher weight for high-confidence decisions
        confidence_weight = decision.confidence_score
        
        # Higher weight for non-fallback decisions
        if not decision.fallback_used:
            base_weight *= 1.2
        
        return min(base_weight * confidence_weight, 2.0)
    
    def _calculate_prediction_error(self, decision: GPUAllocationDecision, outcome: Dict[str, Any]) -> float:
        """Calculate prediction error for learning."""
        predicted_success = 1.0 if decision.decision == "approve" else 0.0
        actual_success = 1.0 if outcome.get("success", True) else 0.0
        
        return abs(predicted_success - actual_success)
    
    def _calculate_confidence_error(self, decision: GPUAllocationDecision, outcome: Dict[str, Any]) -> float:
        """Calculate confidence calibration error."""
        predicted_confidence = decision.confidence_score
        actual_success = 1.0 if outcome.get("success", True) else 0.0
        
        return abs(predicted_confidence - actual_success)
    
    def _adaptive_learning_rate(self) -> float:
        """Calculate adaptive learning rate based on experience."""
        pass
        base_rate = 0.1
        experience_factor = min(self.decision_count / 1000.0, 1.0)
        
        # Lower learning rate as we gain experience
        return base_rate * (1.0 - 0.5 * experience_factor)
    
    def _calculate_confidence_calibration(self, learning_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence calibration score."""
        if not learning_data:
            return 0.5
        
        # Simple calibration: how well confidence matches outcomes
        calibration_errors = []
        for data in learning_data:
            predicted_conf = data["predicted_confidence"]
            actual_outcome = 1.0 if data["actual_outcome"] else 0.0
            calibration_errors.append(abs(predicted_conf - actual_outcome))
        
        avg_error = sum(calibration_errors) / len(calibration_errors)
        return max(0.0, 1.0 - avg_error)  # Convert error to calibration score
    
    def _calculate_learning_trend(self, learning_data: List[Dict[str, Any]]) -> float:
        """Calculate learning trend (improvement over time)."""
        if len(learning_data) < 2:
            return 0.5
        
        # Simple trend: are recent decisions better?
        mid_point = len(learning_data) // 2
        early_accuracy = sum(1 for d in learning_data[:mid_point] if d["actual_outcome"]) / mid_point
        recent_accuracy = sum(1 for d in learning_data[mid_point:] if d["actual_outcome"]) / (len(learning_data) - mid_point)
        
        # Normalize to 0-1 range
        trend = (recent_accuracy - early_accuracy + 1.0) / 2.0
        return max(0.0, min(1.0, trend))
    
    def _calculate_temporal_trend(self, temporal_data: Dict[str, Any]) -> float:
        """Calculate temporal trend in decisions."""
        recent_rate = temporal_data.get("recent_approval_rate", 0.5)
        medium_rate = temporal_data.get("medium_approval_rate", 0.5)
        distant_rate = temporal_data.get("distant_approval_rate", 0.5)
        
        # Calculate trend: positive if approval rates increasing
        trend = (recent_rate - distant_rate + 1.0) / 2.0
        return max(0.0, min(1.0, trend))
    
    def _combine_memory_contexts(self, contexts: List[Optional[Dict[str, Any]]]) -> Optional[torch.Tensor]:
        """Combine multiple memory contexts into a single tensor."""
        
        # Filter out None contexts
        valid_contexts = [ctx for ctx in contexts if ctx is not None]
        
        if not valid_contexts:
            return None
        
        # Extract features from each context type
        features = []
        
        for ctx in valid_contexts:
            # Episodic features
            if "episodic_approval_rate" in ctx:
                features.extend([
                    ctx.get("episodic_approval_rate", 0.5),
                    ctx.get("episodic_confidence", 0.5),
                    ctx.get("episodic_success_rate", 0.5),
                    ctx.get("episodic_count", 0) / 10.0,  # Normalize
                    ctx.get("episodic_recency", 0.5)
                ])
            
            # Semantic features
            if "semantic_approval_rate" in ctx:
                features.extend([
                    ctx.get("semantic_approval_rate", 0.5),
                    ctx.get("semantic_pattern_strength", 0.5),
                    ctx.get("semantic_similarity", 0.5),
                    ctx.get("semantic_count", 0) / 10.0  # Normalize
                ])
            
            # Meta-learning features
            if "meta_confidence_calibration" in ctx:
                features.extend([
                    ctx.get("meta_confidence_calibration", 0.5),
                    ctx.get("meta_decision_accuracy", 0.5),
                    ctx.get("meta_learning_trend", 0.5),
                    ctx.get("meta_sample_size", 0) / 20.0  # Normalize
                ])
            
            # Temporal features
            if "recent_approval_rate" in ctx:
                features.extend([
                    ctx.get("recent_approval_rate", 0.5),
                    ctx.get("medium_approval_rate", 0.5),
                    ctx.get("distant_approval_rate", 0.5),
                    ctx.get("temporal_trend", 0.5)
                ])
        
        # Pad to input size
        while len(features) < self.config.input_size:
            features.append(0.0)
        features = features[:self.config.input_size]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _assess_memory_quality(self, memory_tensor: torch.Tensor) -> float:
        """Assess the quality of memory context."""
        
        # Quality based on information density
        non_zero_features = (memory_tensor != 0).sum().item()
        total_features = memory_tensor.numel()
        
        density_score = non_zero_features / total_features
        
        # Quality based on feature variance (more diverse = better)
        variance_score = torch.var(memory_tensor).item()
        normalized_variance = min(variance_score * 10, 1.0)
        
        # Combined quality score
        return (density_score + normalized_variance) / 2.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory integration statistics."""
        pass
        return {
            "decision_count": self.decision_count,
            "learning_updates": self.learning_updates,
            "memory_quality_score": self.memory_quality_score,
            "learning_rate": self._adaptive_learning_rate(),
            "cache_sizes": {
                "episodic": len(self.episodic_cache),
                "semantic": len(self.semantic_cache),
                "meta": len(self.meta_cache)
            }
        }
    
    # Mock methods for testing without Mem0
    def _get_mock_episodic_context(self, request) -> Dict[str, Any]:
        """Mock episodic context for testing."""
        pass
        return {
            "episodic_approval_rate": 0.8,
            "episodic_confidence": 0.75,
            "episodic_success_rate": 0.85,
            "episodic_count": 5,
            "episodic_recency": 1.0
        }
    
    def _get_mock_semantic_context(self, request) -> Dict[str, Any]:
        """Mock semantic context for testing."""
        pass
        return {
            "semantic_approval_rate": 0.7,
            "semantic_pattern_strength": 0.8,
            "semantic_similarity": 0.85,
            "semantic_count": 8
        }
    
    def _get_mock_meta_context(self, request) -> Dict[str, Any]:
        """Mock meta-learning context for testing."""
        pass
        return {
            "meta_confidence_calibration": 0.8,
            "meta_decision_accuracy": 0.85,
            "meta_learning_trend": 0.6,
            "meta_sample_size": 15
        }
    
    def _get_mock_temporal_context(self, request) -> Dict[str, Any]:
        """Mock temporal context for testing."""
        pass
        return {
            "recent_approval_rate": 0.8,
            "medium_approval_rate": 0.7,
            "distant_approval_rate": 0.6,
            "temporal_trend": 0.7
        }