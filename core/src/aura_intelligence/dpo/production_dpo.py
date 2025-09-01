"""
Production DPO System - 2025 Constitutional AI
Real preference learning with offline mining and constitutional safety
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import json
from dataclasses import dataclass
from collections import deque
import asyncio

@dataclass
class PreferenceRecord:
    """Real preference record from system interactions"""
    chosen_action: Dict[str, Any]
    rejected_action: Dict[str, Any] 
    context: Dict[str, Any]
    preference_strength: float
    timestamp: float
    source: str  # 'human', 'ai_feedback', 'constitutional'

class ConstitutionalSafetyChecker:
    """Constitutional AI 3.0 with cross-modal safety"""
    
    def __init__(self):
        self.safety_rules = [
            {'id': 'harm_prevention', 'weight': 1.0, 'threshold': 0.9},
            {'id': 'truthfulness', 'weight': 0.9, 'threshold': 0.8},
            {'id': 'fairness', 'weight': 0.8, 'threshold': 0.7},
            {'id': 'privacy', 'weight': 0.9, 'threshold': 0.8},
            {'id': 'autonomy', 'weight': 0.7, 'threshold': 0.6}
        ]
        
        # Safety classifier (simplified)
        self.safety_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.violation_history = deque(maxlen=1000)
    
    def evaluate_safety(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action safety using constitutional principles"""
        
        # Extract features for safety evaluation
        features = self._extract_safety_features(action, context)
        
        # Neural safety score
        with torch.no_grad():
            safety_tensor = torch.tensor(features, dtype=torch.float32)
            neural_safety_score = float(self.safety_net(safety_tensor))
        
        # Rule-based safety checks
        rule_scores = {}
        violations = []
        
        for rule in self.safety_rules:
            score = self._evaluate_rule(action, context, rule['id'])
            rule_scores[rule['id']] = score
            
            if score < rule['threshold']:
                violations.append({
                    'rule': rule['id'],
                    'score': score,
                    'threshold': rule['threshold'],
                    'severity': 'high' if score < rule['threshold'] - 0.2 else 'medium'
                })
        
        # Combined safety score
        weighted_score = sum(
            rule_scores[rule['id']] * rule['weight'] 
            for rule in self.safety_rules
        ) / sum(rule['weight'] for rule in self.safety_rules)
        
        final_safety_score = 0.6 * neural_safety_score + 0.4 * weighted_score
        
        # Record violations
        if violations:
            self.violation_history.append({
                'timestamp': time.time(),
                'violations': violations,
                'action': action.get('type', 'unknown')
            })
        
        return {
            'safety_score': final_safety_score,
            'neural_safety': neural_safety_score,
            'rule_scores': rule_scores,
            'violations': violations,
            'safe': final_safety_score >= 0.8 and len(violations) == 0,
            'constitutional_version': '3.0'
        }
    
    def _extract_safety_features(self, action: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
        """Extract features for safety evaluation"""
        features = []
        
        # Action features
        features.append(action.get('confidence', 0.5))
        features.append(action.get('risk_level', 0.5) if isinstance(action.get('risk_level'), (int, float)) else 0.5)
        features.append(len(str(action)) / 1000.0)  # Action complexity
        
        # Context features
        features.append(context.get('user_safety_level', 0.8))
        features.append(context.get('system_load', 0.5))
        features.append(context.get('previous_violations', 0) / 10.0)
        
        # Text analysis features (simplified)
        action_text = str(action).lower()
        harmful_keywords = ['harm', 'damage', 'attack', 'exploit', 'manipulate']
        harmful_score = sum(1 for word in harmful_keywords if word in action_text) / len(harmful_keywords)
        features.append(1.0 - harmful_score)  # Invert so higher = safer
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.5)
        
        return features[:128]
    
    def _evaluate_rule(self, action: Dict[str, Any], context: Dict[str, Any], rule_id: str) -> float:
        """Evaluate specific constitutional rule"""
        
        if rule_id == 'harm_prevention':
            # Check for potential harm indicators
            harm_indicators = ['delete', 'remove', 'stop', 'disable', 'break']
            action_str = str(action).lower()
            harm_score = sum(1 for indicator in harm_indicators if indicator in action_str)
            return max(0.0, 1.0 - harm_score * 0.3)
        
        elif rule_id == 'truthfulness':
            # Check for truthfulness indicators
            confidence = action.get('confidence', 0.5)
            has_evidence = 'evidence' in action or 'source' in action
            return min(1.0, confidence + (0.2 if has_evidence else 0))
        
        elif rule_id == 'fairness':
            # Check for bias indicators
            bias_words = ['discriminate', 'bias', 'unfair', 'prejudice']
            action_str = str(action).lower()
            bias_score = sum(1 for word in bias_words if word in action_str)
            return max(0.0, 1.0 - bias_score * 0.4)
        
        elif rule_id == 'privacy':
            # Check for privacy concerns
            private_data = ['password', 'email', 'phone', 'address', 'ssn']
            action_str = str(action).lower()
            privacy_violations = sum(1 for data in private_data if data in action_str)
            return max(0.0, 1.0 - privacy_violations * 0.5)
        
        elif rule_id == 'autonomy':
            # Check for autonomy respect
            coercive_words = ['force', 'must', 'require', 'demand', 'compel']
            action_str = str(action).lower()
            coercion_score = sum(1 for word in coercive_words if word in action_str)
            return max(0.0, 1.0 - coercion_score * 0.3)
        
        return 0.8  # Default score

class ProductionDPOTrainer:
    """Production-grade DPO trainer with real preference mining"""
    
    def __init__(self, beta: float = 0.1):
        self.beta = beta
        
        # DPO policy network
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Reference policy (frozen)
        self.reference_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Copy initial weights
        self.reference_net.load_state_dict(self.policy_net.state_dict())
        
        # Freeze reference network
        for param in self.reference_net.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Constitutional safety
        self.safety_checker = ConstitutionalSafetyChecker()
        
        # Training data
        self.preference_buffer = deque(maxlen=10000)
        self.training_metrics = {
            'total_updates': 0,
            'avg_loss': 0.0,
            'preference_accuracy': 0.0,
            'safety_violations': 0
        }
    
    def mine_preferences_offline(self, action_history: List[Dict[str, Any]]) -> List[PreferenceRecord]:
        """Mine preferences from historical action data"""
        preferences = []
        
        # Group actions by context similarity
        for i in range(len(action_history) - 1):
            action_a = action_history[i]
            action_b = action_history[i + 1]
            
            # Check if actions are comparable (similar context)
            if self._actions_comparable(action_a, action_b):
                pass
                
                # Determine preference based on outcomes
                preference_strength = self._calculate_preference_strength(action_a, action_b)
                
                if abs(preference_strength) > 0.1:  # Significant preference
                    chosen = action_a if preference_strength > 0 else action_b
                    rejected = action_b if preference_strength > 0 else action_a
                    
                    preference = PreferenceRecord(
                        chosen_action=chosen,
                        rejected_action=rejected,
                        context=action_a.get('context', {}),
                        preference_strength=abs(preference_strength),
                        timestamp=time.time(),
                        source='offline_mining'
                    )
                    
                    preferences.append(preference)
        
        return preferences
    
    def _actions_comparable(self, action_a: Dict[str, Any], action_b: Dict[str, Any]) -> bool:
        """Check if two actions are comparable for preference learning"""
        # Simple similarity check based on action type and context
        type_a = action_a.get('type', 'unknown')
        type_b = action_b.get('type', 'unknown')
        
        # Same action type
        if type_a != type_b:
            return False
        
        # Similar context (simplified)
        context_a = action_a.get('context', {})
        context_b = action_b.get('context', {})
        
        common_keys = set(context_a.keys()) & set(context_b.keys())
        if len(common_keys) < 2:
            return False
        
        return True
    
    def _calculate_preference_strength(self, action_a: Dict[str, Any], action_b: Dict[str, Any]) -> float:
        """Calculate preference strength between two actions"""
        
        # Factors for preference calculation
        factors = []
        
        # Success rate
        success_a = action_a.get('success', 0.5)
        success_b = action_b.get('success', 0.5)
        factors.append(success_a - success_b)
        
        # Confidence
        conf_a = action_a.get('confidence', 0.5)
        conf_b = action_b.get('confidence', 0.5)
        factors.append(conf_a - conf_b)
        
        # Safety scores
        safety_a = self.safety_checker.evaluate_safety(action_a, action_a.get('context', {}))
        safety_b = self.safety_checker.evaluate_safety(action_b, action_b.get('context', {}))
        factors.append(safety_a['safety_score'] - safety_b['safety_score'])
        
        # Processing time (prefer faster)
        time_a = action_a.get('processing_time', 1.0)
        time_b = action_b.get('processing_time', 1.0)
        factors.append((time_b - time_a) / max(time_a, time_b))  # Normalize
        
        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]
        preference_strength = sum(f * w for f, w in zip(factors, weights))
        
        return np.clip(preference_strength, -1.0, 1.0)
    
    def _encode_action_context(self, action: Dict[str, Any], context: Dict[str, Any]) -> torch.Tensor:
        """Encode action and context for DPO network"""
        features = []
        
        # Action features
        features.append(action.get('confidence', 0.5))
        features.append(action.get('success', 0.5))
        features.append(hash(action.get('type', 'default')) % 1000 / 1000.0)
        
        # Context features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(hash(value) % 1000 / 1000.0)
        
        # Safety features
        safety_eval = self.safety_checker.evaluate_safety(action, context)
        features.append(safety_eval['safety_score'])
        features.extend([safety_eval['rule_scores'].get(rule['id'], 0.5) for rule in self.safety_checker.safety_rules])
        
        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32)
    
    def compute_dpo_loss(self, batch: List[PreferenceRecord]) -> torch.Tensor:
        """Compute DPO loss with constitutional weighting"""
        losses = []
        
        for record in batch:
            # Encode chosen and rejected actions
            chosen_encoding = self._encode_action_context(record.chosen_action, record.context)
            rejected_encoding = self._encode_action_context(record.rejected_action, record.context)
            
            # Policy scores
            chosen_score = self.policy_net(chosen_encoding)
            rejected_score = self.policy_net(rejected_encoding)
            
            # Reference scores (frozen)
            with torch.no_grad():
                chosen_ref = self.reference_net(chosen_encoding)
                rejected_ref = self.reference_net(rejected_encoding)
            
            # DPO loss computation
            log_ratio_chosen = chosen_score - chosen_ref
            log_ratio_rejected = rejected_score - rejected_ref
            
            logits_diff = self.beta * (log_ratio_chosen - log_ratio_rejected)
            
            # Weight by preference strength and constitutional compliance
            safety_chosen = self.safety_checker.evaluate_safety(record.chosen_action, record.context)
            constitutional_weight = safety_chosen['safety_score']
            
            weighted_strength = record.preference_strength * constitutional_weight
            loss = -F.logsigmoid(logits_diff * weighted_strength)
            
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
        async def train_batch(self, batch_size: int = 32) -> Dict[str, Any]:
            pass
        """Train on batch of preferences"""
        if len(self.preference_buffer) < batch_size:
            return {
                'status': 'insufficient_data',
                'available_preferences': len(self.preference_buffer)
            }
        
        # Sample batch
        batch_indices = np.random.choice(len(self.preference_buffer), batch_size, replace=False)
        batch = [self.preference_buffer[i] for i in batch_indices]
        
        # Compute loss
        loss = self.compute_dpo_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update metrics
        self.training_metrics['total_updates'] += 1
        self.training_metrics['avg_loss'] = 0.9 * self.training_metrics['avg_loss'] + 0.1 * float(loss)
        
        # Calculate preference accuracy
        with torch.no_grad():
            correct_preferences = 0
            for record in batch:
                chosen_encoding = self._encode_action_context(record.chosen_action, record.context)
                rejected_encoding = self._encode_action_context(record.rejected_action, record.context)
                
                chosen_score = self.policy_net(chosen_encoding)
                rejected_score = self.policy_net(rejected_encoding)
                
                if chosen_score > rejected_score:
                    correct_preferences += 1
            
            accuracy = correct_preferences / len(batch)
            self.training_metrics['preference_accuracy'] = 0.9 * self.training_metrics['preference_accuracy'] + 0.1 * accuracy
        
        return {
            'status': 'training_complete',
            'loss': float(loss),
            'batch_size': batch_size,
            'preference_accuracy': accuracy,
            'total_updates': self.training_metrics['total_updates'],
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def add_preference(self, preference: PreferenceRecord):
        """Add preference to training buffer"""
        self.preference_buffer.append(preference)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        pass
        return {
            'training_metrics': self.training_metrics,
            'preference_buffer_size': len(self.preference_buffer),
            'constitutional_violations': len(self.safety_checker.violation_history),
            'model_parameters': sum(p.numel() for p in self.policy_net.parameters()),
            'dpo_beta': self.beta,
            'constitutional_version': '3.0'
        }

# Global instance
_dpo_trainer = None

    def get_production_dpo() -> ProductionDPOTrainer:
        """Get global DPO trainer instance"""
        global _dpo_trainer
        if _dpo_trainer is None:
            pass
        _dpo_trainer = ProductionDPOTrainer()
        return _dpo_trainer
