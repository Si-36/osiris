"""Reputation service for Byzantine consensus"""

from typing import Dict, List, Optional
import numpy as np
import structlog

logger = structlog.get_logger()


class ReputationService:
    """Manages reputation across the Byzantine consensus network"""
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self.global_reputation: Dict[str, float] = {}
        self.reputation_history: Dict[str, List[float]] = {}
        self.logger = logger
        
    def initialize_reputation(self, node_id: str, initial_score: float = 1.0):
        """Initialize reputation for a new node"""
        self.global_reputation[node_id] = initial_score
        self.reputation_history[node_id] = [initial_score]
        
    def update_global_reputation(self, node_id: str, behavior_scores: List[float]):
        """Update global reputation based on multiple behavior observations"""
        if node_id not in self.global_reputation:
            self.initialize_reputation(node_id)
            
        # Weighted average with decay
        current = self.global_reputation[node_id]
        new_score = current * self.decay_factor
        
        if behavior_scores:
            avg_behavior = np.mean(behavior_scores)
            new_score += (1 - self.decay_factor) * avg_behavior
            
        self.global_reputation[node_id] = np.clip(new_score, 0.0, 1.0)
        self.reputation_history[node_id].append(new_score)
        
        # Keep history bounded
        if len(self.reputation_history[node_id]) > 1000:
            self.reputation_history[node_id] = self.reputation_history[node_id][-1000:]
            
    def get_reputation_report(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive reputation report for a node"""
        if node_id not in self.global_reputation:
            return {"error": "Node not found"}
            
        history = self.reputation_history[node_id]
        
        return {
            "node_id": node_id,
            "current_reputation": self.global_reputation[node_id],
            "reputation_trend": self._calculate_trend(history),
            "volatility": np.std(history[-10:]) if len(history) > 10 else 0,
            "history_length": len(history)
        }
        
    def _calculate_trend(self, history: List[float]) -> str:
        """Calculate reputation trend"""
        if len(history) < 2:
            return "stable"
            
        recent = history[-10:]
        if len(recent) < 2:
            return "stable"
            
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"