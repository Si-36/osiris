"""
Causal Pattern Tracker - Learns Cause-Effect Relationships in Agent Systems
=========================================================================

Tracks how topological patterns evolve and lead to specific outcomes.
This enables PREDICTIVE memory - anticipating failures before they happen!

Key Features:
- Pattern → Outcome tracking with confidence scores
- Temporal causality chains
- Failure sequence learning
- Predictive analytics
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timezone
import structlog
import json

# Import from TDA module
from ...tda import WorkflowFeatures

# Import our topology adapter
from .topology_adapter import MemoryTopologySignature

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

@dataclass
class CausalPattern:
    """A pattern that leads to specific outcomes"""
    pattern_id: str
    pattern_type: str  # "topology", "sequence", "communication"
    
    # Pattern characteristics
    topology_signature: Optional[MemoryTopologySignature] = None
    feature_vector: Optional[np.ndarray] = None
    
    # Outcome tracking
    outcomes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_occurrences: int = 0
    
    # Temporal aspects
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    avg_time_to_outcome: float = 0.0
    
    # Confidence metrics
    confidence_score: float = 0.0
    stability_score: float = 0.0
    
    @property
    def failure_probability(self) -> float:
        """Probability this pattern leads to failure"""
        if self.total_occurrences == 0:
            return 0.0
        failures = self.outcomes.get("failure", 0)
        return failures / self.total_occurrences
    
    @property
    def success_probability(self) -> float:
        """Probability this pattern leads to success"""
        if self.total_occurrences == 0:
            return 0.0
        successes = self.outcomes.get("success", 0)
        return successes / self.total_occurrences


@dataclass
class CausalChain:
    """Sequence of patterns that lead to an outcome"""
    chain_id: str
    patterns: List[CausalPattern]
    
    # Chain properties
    start_pattern: str
    end_outcome: str
    chain_length: int
    
    # Temporal properties
    avg_duration: float
    min_duration: float
    max_duration: float
    
    # Confidence
    occurrences: int = 0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chain_id": self.chain_id,
            "pattern_ids": [p.pattern_id for p in self.patterns],
            "start_pattern": self.start_pattern,
            "end_outcome": self.end_outcome,
            "chain_length": self.chain_length,
            "avg_duration": self.avg_duration,
            "confidence": self.confidence
        }


@dataclass
class CausalAnalysis:
    """Results of causal analysis"""
    primary_causes: List[CausalPattern]
    causal_chains: List[CausalChain]
    
    # Predictions
    likely_outcome: str
    outcome_probability: float
    time_to_outcome: float
    
    # Recommendations
    preventive_actions: List[Dict[str, Any]]
    risk_factors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "primary_causes": [p.pattern_id for p in self.primary_causes],
            "causal_chains": [c.to_dict() for c in self.causal_chains],
            "likely_outcome": self.likely_outcome,
            "outcome_probability": self.outcome_probability,
            "time_to_outcome": self.time_to_outcome,
            "preventive_actions": self.preventive_actions,
            "risk_factors": self.risk_factors
        }


# ==================== Main Causal Tracker ====================

class CausalPatternTracker:
    """
    Tracks cause-effect relationships between topological patterns
    
    This is what enables PREDICTIVE MEMORY - we learn which patterns
    lead to failures and can warn before they happen!
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Pattern storage
        self.patterns: Dict[str, CausalPattern] = {}
        self.chains: Dict[str, CausalChain] = {}
        
        # Temporal tracking
        self.active_sequences: Dict[str, List[Tuple[str, float]]] = {}
        self.sequence_timeout = self.config.get("sequence_timeout", 3600)  # 1 hour
        
        # Learning parameters
        self.min_occurrences = self.config.get("min_occurrences", 5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.decay_factor = self.config.get("decay_factor", 0.95)
        
        # Neo4j connection (placeholder - would connect to real Neo4j)
        self.graph_store = None
        
        # Statistics
        self.stats = {
            "patterns_tracked": 0,
            "chains_discovered": 0,
            "predictions_made": 0,
            "correct_predictions": 0
        }
        
        logger.info(
            "Causal pattern tracker initialized",
            min_occurrences=self.min_occurrences,
            confidence_threshold=self.confidence_threshold
        )
    
    # ==================== Pattern Tracking ====================
    
    async def track_pattern(self,
                          workflow_id: str,
                          pattern: Optional[MemoryTopologySignature],
                          outcome: str,
                          timestamp: Optional[float] = None) -> str:
        """
        Track a pattern and its outcome
        
        This is how we LEARN - every pattern-outcome pair makes us smarter!
        """
        if not pattern:
            return None
            
        timestamp = timestamp or time.time()
        
        # Generate pattern ID from topology
        pattern_id = self._generate_pattern_id(pattern)
        
        # Get or create pattern record
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = CausalPattern(
                pattern_id=pattern_id,
                pattern_type="topology",
                topology_signature=pattern,
                feature_vector=pattern.fastrp_embedding
            )
        
        causal_pattern = self.patterns[pattern_id]
        
        # Update outcome tracking
        causal_pattern.outcomes[outcome] += 1
        causal_pattern.total_occurrences += 1
        causal_pattern.last_seen = timestamp
        
        # Update confidence based on consistency
        causal_pattern.confidence_score = self._calculate_confidence(causal_pattern)
        
        # Track in active sequences for chain discovery
        if workflow_id not in self.active_sequences:
            self.active_sequences[workflow_id] = []
        
        self.active_sequences[workflow_id].append((pattern_id, timestamp))
        
        # Check for completed chains
        if outcome in ["success", "failure", "completed"]:
            await self._process_completed_sequence(workflow_id, outcome, timestamp)
        
        # Update statistics
        self.stats["patterns_tracked"] += 1
        
        logger.info(
            "Pattern tracked",
            pattern_id=pattern_id,
            outcome=outcome,
            occurrences=causal_pattern.total_occurrences,
            failure_prob=causal_pattern.failure_probability
        )
        
        return pattern_id
    
    # ==================== Causal Analysis ====================
    
    async def analyze_patterns(self,
                             current_patterns: List[MemoryTopologySignature]) -> Dict[str, Any]:
        """
        Analyze current patterns for causal relationships
        
        Returns predictions and recommendations
        """
        if not current_patterns:
            return {"chains": [], "failure_probability": 0.0}
        
        # Find matching historical patterns
        matching_patterns = []
        for current in current_patterns:
            pattern_id = self._generate_pattern_id(current)
            if pattern_id in self.patterns:
                matching_patterns.append(self.patterns[pattern_id])
        
        if not matching_patterns:
            return {"chains": [], "failure_probability": 0.0}
        
        # Calculate aggregate failure probability
        total_weight = sum(p.confidence_score * p.total_occurrences for p in matching_patterns)
        if total_weight > 0:
            weighted_failure_prob = sum(
                p.failure_probability * p.confidence_score * p.total_occurrences 
                for p in matching_patterns
            ) / total_weight
        else:
            weighted_failure_prob = 0.0
        
        # Find relevant causal chains
        relevant_chains = self._find_relevant_chains(matching_patterns)
        
        # Generate analysis
        analysis = CausalAnalysis(
            primary_causes=matching_patterns[:3],  # Top 3
            causal_chains=relevant_chains[:5],     # Top 5 chains
            likely_outcome="failure" if weighted_failure_prob > 0.5 else "success",
            outcome_probability=weighted_failure_prob,
            time_to_outcome=self._estimate_time_to_outcome(matching_patterns),
            preventive_actions=self._generate_preventive_actions(matching_patterns, relevant_chains),
            risk_factors=self._identify_risk_factors(matching_patterns)
        )
        
        self.stats["predictions_made"] += 1
        
        return analysis.to_dict()
    
    async def predict_outcome(self,
                            topology: MemoryTopologySignature,
                            time_horizon: float = 300) -> Dict[str, Any]:
        """
        Predict the likely outcome of a topology pattern
        
        This is the MAGIC - predicting failures before they happen!
        """
        pattern_id = self._generate_pattern_id(topology)
        
        # Direct pattern match
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            
            if pattern.total_occurrences >= self.min_occurrences:
                return {
                    "pattern_found": True,
                    "pattern_id": pattern_id,
                    "failure_probability": pattern.failure_probability,
                    "success_probability": pattern.success_probability,
                    "confidence": pattern.confidence_score,
                    "based_on_occurrences": pattern.total_occurrences,
                    "avg_time_to_outcome": pattern.avg_time_to_outcome
                }
        
        # Find similar patterns using embedding
        similar_patterns = self._find_similar_patterns(topology, k=5)
        
        if similar_patterns:
            # Weighted prediction based on similarity
            total_weight = sum(sim for _, sim in similar_patterns)
            
            if total_weight > 0:
                weighted_failure = sum(
                    self.patterns[pid].failure_probability * sim 
                    for pid, sim in similar_patterns
                ) / total_weight
                
                avg_confidence = np.mean([
                    self.patterns[pid].confidence_score 
                    for pid, _ in similar_patterns
                ])
                
                return {
                    "pattern_found": False,
                    "similar_patterns": len(similar_patterns),
                    "failure_probability": weighted_failure,
                    "success_probability": 1 - weighted_failure,
                    "confidence": avg_confidence * 0.8,  # Reduce confidence for similar
                    "based_on_similar": True
                }
        
        return {
            "pattern_found": False,
            "failure_probability": 0.5,  # Unknown
            "success_probability": 0.5,
            "confidence": 0.0,
            "reason": "No matching patterns found"
        }
    
    # ==================== Chain Discovery ====================
    
    async def _process_completed_sequence(self,
                                        workflow_id: str,
                                        outcome: str,
                                        end_time: float):
        """Process completed workflow sequence to discover chains"""
        if workflow_id not in self.active_sequences:
            return
        
        sequence = self.active_sequences[workflow_id]
        if len(sequence) < 2:
            return
        
        # Create chain ID
        pattern_ids = [pid for pid, _ in sequence]
        chain_id = self._generate_chain_id(pattern_ids, outcome)
        
        # Calculate temporal properties
        start_time = sequence[0][1]
        duration = end_time - start_time
        
        # Get or create chain
        if chain_id not in self.chains:
            self.chains[chain_id] = CausalChain(
                chain_id=chain_id,
                patterns=[self.patterns[pid] for pid in pattern_ids if pid in self.patterns],
                start_pattern=pattern_ids[0],
                end_outcome=outcome,
                chain_length=len(sequence),
                avg_duration=duration,
                min_duration=duration,
                max_duration=duration
            )
        else:
            # Update existing chain
            chain = self.chains[chain_id]
            chain.occurrences += 1
            chain.avg_duration = (
                chain.avg_duration * (chain.occurrences - 1) + duration
            ) / chain.occurrences
            chain.min_duration = min(chain.min_duration, duration)
            chain.max_duration = max(chain.max_duration, duration)
        
        # Update confidence
        chain = self.chains[chain_id]
        chain.confidence = min(chain.occurrences / self.min_occurrences, 1.0)
        
        # Clean up
        del self.active_sequences[workflow_id]
        
        self.stats["chains_discovered"] += 1
        
        logger.info(
            "Causal chain discovered/updated",
            chain_id=chain_id,
            length=len(sequence),
            outcome=outcome,
            occurrences=chain.occurrences
        )
    
    # ==================== Helper Methods ====================
    
    def _generate_pattern_id(self, topology: MemoryTopologySignature) -> str:
        """Generate stable ID for pattern"""
        if topology.pattern_id:
            return topology.pattern_id
        
        # Use key features to create ID
        features = topology.workflow_features
        key = f"{features.num_agents}:{features.num_edges}:{features.has_cycles}:{len(features.bottleneck_agents)}"
        
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def _generate_chain_id(self, pattern_ids: List[str], outcome: str) -> str:
        """Generate ID for causal chain"""
        chain_str = "->".join(pattern_ids) + f"->{outcome}"
        import hashlib
        return hashlib.md5(chain_str.encode()).hexdigest()[:12]
    
    def _calculate_confidence(self, pattern: CausalPattern) -> float:
        """Calculate confidence score for pattern"""
        if pattern.total_occurrences < self.min_occurrences:
            # Low confidence for insufficient data
            base_confidence = pattern.total_occurrences / self.min_occurrences * 0.5
        else:
            # High confidence for sufficient data
            base_confidence = 0.5 + 0.5 * min(pattern.total_occurrences / (self.min_occurrences * 10), 1.0)
        
        # Adjust for outcome consistency
        if pattern.total_occurrences > 0:
            max_outcome_ratio = max(
                count / pattern.total_occurrences 
                for count in pattern.outcomes.values()
            )
            consistency_factor = max_outcome_ratio
        else:
            consistency_factor = 0.0
        
        # Apply time decay
        time_since_last = time.time() - pattern.last_seen
        decay = self.decay_factor ** (time_since_last / 86400)  # Daily decay
        
        return float(base_confidence * consistency_factor * decay)
    
    def _find_similar_patterns(self,
                             topology: MemoryTopologySignature,
                             k: int = 5) -> List[Tuple[str, float]]:
        """Find similar patterns using embeddings"""
        if not topology.fastrp_embedding:
            return []
        
        similarities = []
        
        for pid, pattern in self.patterns.items():
            if pattern.feature_vector is not None:
                # Cosine similarity
                similarity = np.dot(topology.fastrp_embedding, pattern.feature_vector)
                if similarity > 0.7:  # Threshold
                    similarities.append((pid, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def _find_relevant_chains(self, patterns: List[CausalPattern]) -> List[CausalChain]:
        """Find chains that involve the given patterns"""
        pattern_ids = {p.pattern_id for p in patterns}
        
        relevant_chains = []
        for chain in self.chains.values():
            # Check if chain contains any of our patterns
            chain_pattern_ids = {p.pattern_id for p in chain.patterns}
            if pattern_ids & chain_pattern_ids:  # Intersection
                relevant_chains.append(chain)
        
        # Sort by confidence and recency
        relevant_chains.sort(
            key=lambda c: c.confidence * c.occurrences,
            reverse=True
        )
        
        return relevant_chains
    
    def _estimate_time_to_outcome(self, patterns: List[CausalPattern]) -> float:
        """Estimate time until outcome based on patterns"""
        if not patterns:
            return 0.0
        
        # Weight by confidence
        total_weight = sum(p.confidence_score for p in patterns)
        if total_weight > 0:
            weighted_time = sum(
                p.avg_time_to_outcome * p.confidence_score 
                for p in patterns
            ) / total_weight
            return float(weighted_time)
        
        return 300.0  # Default 5 minutes
    
    def _generate_preventive_actions(self,
                                   patterns: List[CausalPattern],
                                   chains: List[CausalChain]) -> List[Dict[str, Any]]:
        """Generate preventive action recommendations"""
        actions = []
        
        # Analyze patterns for common issues
        bottleneck_patterns = [
            p for p in patterns 
            if p.topology_signature and p.topology_signature.bottleneck_severity > 0.7
        ]
        
        if bottleneck_patterns:
            actions.append({
                "action": "scale_bottlenecks",
                "description": "Scale out bottleneck agents to prevent overload",
                "priority": "high",
                "agents": list(set(
                    agent 
                    for p in bottleneck_patterns 
                    for agent in p.topology_signature.workflow_features.bottleneck_agents[:3]
                ))
            })
        
        # Check for cycle patterns
        cycle_patterns = [
            p for p in patterns
            if p.topology_signature and p.topology_signature.workflow_features.has_cycles
        ]
        
        if cycle_patterns:
            actions.append({
                "action": "break_cycles",
                "description": "Remove circular dependencies to prevent deadlocks",
                "priority": "medium",
                "confidence": 0.8
            })
        
        return actions
    
    def _identify_risk_factors(self, patterns: List[CausalPattern]) -> List[str]:
        """Identify key risk factors from patterns"""
        risk_factors = []
        
        # High failure probability patterns
        high_risk = [p for p in patterns if p.failure_probability > 0.7]
        if high_risk:
            risk_factors.append(f"{len(high_risk)} high-risk patterns detected")
        
        # Unstable patterns
        unstable = [
            p for p in patterns 
            if p.topology_signature and p.topology_signature.stability_score < 0.5
        ]
        if unstable:
            risk_factors.append(f"{len(unstable)} unstable topology patterns")
        
        # Recent failures
        recent_failures = [
            p for p in patterns 
            if time.time() - p.last_seen < 3600 and p.failure_probability > 0.5
        ]
        if recent_failures:
            risk_factors.append(f"{len(recent_failures)} patterns with recent failures")
        
        return risk_factors
    
    # ==================== Persistence ====================
    
    async def save_to_neo4j(self):
        """Save causal patterns to Neo4j (placeholder)"""
        # In production, would save to Neo4j for persistence
        logger.info(
            "Would save to Neo4j",
            patterns=len(self.patterns),
            chains=len(self.chains)
        )
    
    async def load_from_neo4j(self):
        """Load causal patterns from Neo4j (placeholder)"""
        # In production, would load from Neo4j
        logger.info("Would load from Neo4j")
    
    # ==================== Lifecycle ====================
    
    async def initialize(self):
        """Initialize causal tracker"""
        # Load historical patterns
        await self.load_from_neo4j()
        
        # Start background tasks
        asyncio.create_task(self._cleanup_old_sequences())
        
        logger.info("Causal tracker initialized")
    
    async def _cleanup_old_sequences(self):
        """Clean up old incomplete sequences"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            current_time = time.time()
            to_remove = []
            
            for workflow_id, sequence in self.active_sequences.items():
                if sequence:
                    last_time = sequence[-1][1]
                    if current_time - last_time > self.sequence_timeout:
                        to_remove.append(workflow_id)
            
            for workflow_id in to_remove:
                del self.active_sequences[workflow_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} stale sequences")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return {
            **self.stats,
            "active_patterns": len(self.patterns),
            "discovered_chains": len(self.chains),
            "active_sequences": len(self.active_sequences),
            "high_risk_patterns": sum(
                1 for p in self.patterns.values() 
                if p.failure_probability > 0.7
            )
        }


# ==================== Public API ====================

__all__ = [
    "CausalPatternTracker",
    "CausalPattern",
    "CausalChain",
    "CausalAnalysis"
]