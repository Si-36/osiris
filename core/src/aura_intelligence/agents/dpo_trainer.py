"""
Direct Preference Optimization (DPO) for AURA Agents
Uses existing action records for preference learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import numpy as np

from ..agents.schemas.action import ActionRecord


class DPOTrainer:
    """
    DPO trainer using AURA's action recording system
    Learns from agent preferences without reward modeling
    """
    
    def __init__(self, model_dim: int = 256, learning_rate: float = 1e-4):
        self.model_dim = model_dim
        self.learning_rate = learning_rate
        
        # Simple policy network
        self.policy_net = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Action value
        )
        
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate
        )
        
    def create_preference_pairs(
        self, 
        action_records: List[ActionRecord]
        ) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Create preference pairs from action records"""
        
        pairs = []
        
        # Sort by confidence (proxy for preference)
        sorted_records = sorted(
            action_records, 
            key=lambda x: x.confidence, 
            reverse=True
        )
        
        # Create pairs: high confidence vs low confidence
        for i in range(len(sorted_records) - 1):
            high_conf = sorted_records[i]
            low_conf = sorted_records[i + 1]
            
            # Convert to tensors
            high_tensor = self._action_to_tensor(high_conf)
            low_tensor = self._action_to_tensor(low_conf)
            
            # Preference strength
            preference = high_conf.confidence - low_conf.confidence
            
            pairs.append((high_tensor, low_tensor, preference))
            
        return pairs
    
    def _action_to_tensor(self, action: ActionRecord) -> torch.Tensor:
        """Convert action record to tensor"""
        features = [
            action.confidence,
            action.structured_intent.get_risk_score(),
            float(action.duration_ms or 0) / 1000.0,  # Normalize
            len(action.side_effects),
            len(action.affected_systems),
            float(action.rollback_available),
        ]
        
        # Pad to model_dim
        while len(features) < self.model_dim:
            features.append(0.0)
            
        return torch.tensor(features[:self.model_dim], dtype=torch.float32)
    
    def dpo_loss(
        self, 
        preferred: torch.Tensor, 
        rejected: torch.Tensor,
        preference_strength: float,
        beta: float = 0.1
        ) -> torch.Tensor:
        """DPO loss function"""
        
        # Get policy logits
        preferred_logits = self.policy_net(preferred)
        rejected_logits = self.policy_net(rejected)
        
        # DPO loss
        logits_diff = preferred_logits - rejected_logits
        loss = -F.logsigmoid(beta * logits_diff * preference_strength)
        
        return loss.mean()
    
    def train_step(self, preference_pairs: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        """Single training step"""
        
        total_loss = 0.0
        
        for preferred, rejected, strength in preference_pairs:
            self.optimizer.zero_grad()
            
            loss = self.dpo_loss(preferred, rejected, strength)
            loss.backward()
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(preference_pairs)
    
    def evaluate_action(self, action: ActionRecord) -> float:
        """Evaluate action quality using trained model"""
        with torch.no_grad():
            action_tensor = self._action_to_tensor(action)
            score = self.policy_net(action_tensor)
            return torch.sigmoid(score).item()