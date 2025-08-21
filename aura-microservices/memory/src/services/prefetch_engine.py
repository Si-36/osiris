"""Predictive Prefetch Engine for memory optimization"""

from typing import List, Dict, Any, Optional
import time
import numpy as np
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger()


class PredictivePrefetchEngine:
    """Predicts and prefetches data based on access patterns"""
    
    def __init__(self, window_size: int = 100, confidence_threshold: float = 0.8):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.access_history = deque(maxlen=window_size)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.logger = logger
        
    def record_access(self, key: str):
        """Record an access event"""
        current_time = time.time()
        
        if self.access_history:
            last_key = self.access_history[-1][0]
            # Update transition matrix
            self.transition_matrix[last_key][key] += 1
            
        self.access_history.append((key, current_time))
        
    def predict_next_access(self, current_key: str, n: int = 5) -> List[tuple]:
        """Predict next likely accesses"""
        predictions = []
        
        if current_key in self.transition_matrix:
            transitions = self.transition_matrix[current_key]
            total_transitions = sum(transitions.values())
            
            if total_transitions > 0:
                # Calculate probabilities
                for next_key, count in transitions.items():
                    probability = count / total_transitions
                    if probability >= self.confidence_threshold / n:
                        predictions.append((next_key, probability))
                        
                # Sort by probability
                predictions.sort(key=lambda x: x[1], reverse=True)
                predictions = predictions[:n]
                
        return predictions
        
    def get_access_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about access patterns"""
        if not self.access_history:
            return {}
            
        # Calculate access frequency
        access_counts = defaultdict(int)
        for key, _ in self.access_history:
            access_counts[key] += 1
            
        # Find most common sequences
        sequences = []
        for i in range(len(self.access_history) - 1):
            seq = (self.access_history[i][0], self.access_history[i+1][0])
            sequences.append(seq)
            
        sequence_counts = defaultdict(int)
        for seq in sequences:
            sequence_counts[seq] += 1
            
        most_common_sequences = sorted(
            sequence_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "total_accesses": len(self.access_history),
            "unique_keys": len(access_counts),
            "most_accessed": sorted(
                access_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "common_sequences": most_common_sequences
        }