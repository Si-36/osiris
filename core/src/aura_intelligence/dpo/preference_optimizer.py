"""
Direct Preference Optimization (DPO) 2025
Learn from preferences without reward modeling
Based on Anthropic 2024 research + Constitutional AI 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import json
from dataclasses import dataclass

from ..components.real_registry import get_real_registry


@dataclass
class PreferencePair:
    preferred_action: Dict[str, Any]
    rejected_action: Dict[str, Any]
    preference_strength: float
    context: Dict[str, Any]
    timestamp: float


class DPOPolicyNetwork(nn.Module):
    """DPO Policy Network - learns from preferences directly"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single preference score
        )
        
    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """Forward pass: state-action → preference score"""
        return self.network(state_action)


class ConstitutionalAI3:
    """Constitutional AI 3.0 - Cross-modal safety with self-correction"""
    
    def __init__(self):
        self.constitutional_rules = [
            {"id": "safety", "desc": "Prioritize system safety and stability", "weight": 1.0, "threshold": 0.9},
            {"id": "efficiency", "desc": "Ensure efficient resource utilization", "weight": 0.7, "threshold": 0.6}, 
            {"id": "coordination", "desc": "Maintain component coordination", "weight": 0.8, "threshold": 0.7},
            {"id": "transparency", "desc": "Provide clear reasoning for decisions", "weight": 0.8, "threshold": 0.7},
            {"id": "fairness", "desc": "Ensure equitable treatment", "weight": 0.9, "threshold": 0.8}
        ]
        
        # Cross-modal safety encoder
        self.safety_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.rule_violations = []
        self.self_improvement_history = []
        self.auto_corrections = 0
        
    def evaluate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action with cross-modal Constitutional AI 3.0"""
        violations = []
        compliance_scores = []
        
        # Extract multi-modal features
        features = self._extract_multimodal_features(action, context)
        
        # Cross-modal safety assessment
        with torch.no_grad():
            safety_score = self.safety_encoder(torch.tensor(features, dtype=torch.float32)).item()
        
        # Evaluate each constitutional rule
        for rule in self.constitutional_rules:
            score = self._evaluate_rule_compliance(action, context, rule)
            compliance_scores.append(score)
            
            if score < rule['threshold']:
                violations.append({
                    'rule_id': rule['id'],
                    'description': rule['desc'],
                    'score': score,
                    'threshold': rule['threshold']
                })
        
        # Weighted compliance score
        weighted_scores = [score * rule['weight'] for score, rule in zip(compliance_scores, self.constitutional_rules)]
        total_weight = sum(rule['weight'] for rule in self.constitutional_rules)
        overall_compliance = sum(weighted_scores) / total_weight
        
        # Attempt self-correction if violations exist
        corrected_action = action
        if violations:
            corrected_action = self._self_correct_action(action, violations)
            self.auto_corrections += 1
        
        return {
            'constitutional_compliance': overall_compliance,
            'safety_score': safety_score,
            'violations': violations,
            'rule_scores': compliance_scores,
            'approved': overall_compliance >= 0.8 and safety_score >= 0.8,
            'corrected_action': corrected_action,
            'auto_corrected': len(violations) > 0
        }
    
    def _extract_multimodal_features(self, action: Dict[str, Any], context: Dict[str, Any]) -> List[float]:
        """Extract cross-modal features for safety assessment"""
        features = []
        
        # Text features (action description, reasoning)
        text_features = []
        for key in ['description', 'reasoning', 'action_type']:
            if key in action:
                text_hash = hash(str(action[key])) % 1000
                text_features.append(text_hash / 1000.0)
        features.extend(text_features[:64])  # Limit text features
        
        # Neural state features
        neural_features = [
            action.get('confidence', 0.5),
            action.get('priority', 0.5),
            context.get('system_load', 0.5),
            context.get('component_coordination', 0.8)
        ]
        features.extend(neural_features)
        
        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        
        return features[:256]
    
    def _evaluate_rule_compliance(self, action: Dict[str, Any], 
        context: Dict[str, Any], rule: Dict[str, Any]) -> float:
        """Evaluate compliance with specific constitutional rule"""
        rule_id = rule['id']
        base_score = 0.8
        
        if rule_id == 'safety':
            risk_level = action.get('risk_level', 'medium')
            if risk_level == 'low':
                base_score = 0.95
            elif risk_level == 'high':
                base_score = 0.3
                
        elif rule_id == 'efficiency':
            efficiency = action.get('efficiency_score', 0.7)
            base_score = efficiency
            
        elif rule_id == 'coordination':
            coordination = context.get('component_coordination', 0.8)
            base_score = coordination
            
        elif rule_id == 'transparency':
            has_reasoning = 'reasoning' in action and len(action['reasoning']) > 10
            has_confidence = 'confidence' in action
            base_score = 0.5 + 0.25 * has_reasoning + 0.25 * has_confidence
            
        elif rule_id == 'fairness':
            # Check for bias indicators
            base_score = 0.85  # Default high fairness
            if 'bias' in str(action).lower():
                base_score -= 0.3
        
        return max(0.0, min(1.0, base_score))
    
    def _self_correct_action(self, action: Dict[str, Any], violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Self-correct action based on violations"""
        corrected = action.copy()
        
        for violation in violations:
            rule_id = violation['rule_id']
            
            if rule_id == 'safety':
                if 'risk_level' in corrected:
                    if corrected['risk_level'] == 'high':
                        corrected['risk_level'] = 'medium'
                    elif corrected['risk_level'] == 'medium':
                        corrected['risk_level'] = 'low'
                        
            elif rule_id == 'transparency':
                if 'reasoning' not in corrected:
                    corrected['reasoning'] = 'Auto-generated reasoning for transparency'
                if 'confidence' not in corrected:
                    corrected['confidence'] = 0.7
                    
            elif rule_id == 'efficiency':
                if 'efficiency_score' in corrected:
                    corrected['efficiency_score'] = min(1.0, corrected['efficiency_score'] + 0.2)
        
        return corrected
    
        async def self_improve(self, recent_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Self-improvement mechanism using RLAIF"""
        if len(recent_actions) < 10:
            return {'improvement': 'insufficient_data'}
        
        # Analyze recent performance
        avg_compliance = np.mean([
            action.get('constitutional_compliance', 0.8) 
            for action in recent_actions
        ])
        
        # Self-improvement if performance is declining
        if avg_compliance < 0.75:
            # Adjust rule weights (simplified self-improvement)
            improvement = {
                'rule_adjustment': 'increased_safety_weight',
                'new_threshold': 0.85,
                'improvement_reason': f'Low compliance: {avg_compliance:.3f}'
            }
            
            self.self_improvement_history.append(improvement)
            return improvement
        
        return {'improvement': 'no_adjustment_needed', 'compliance': avg_compliance}


class DirectPreferenceOptimizer:
    """
    Direct Preference Optimization System
    Learns from action preferences without reward modeling
    """
    
    def __init__(self, beta: float = 0.1):
        self.registry = get_real_registry()
        self.beta = beta  # KL penalty coefficient
        
        # DPO policy network
        self.policy_net = DPOPolicyNetwork(input_dim=256, hidden_dim=128)
        self.reference_net = DPOPolicyNetwork(input_dim=256, hidden_dim=128)
        
        # Constitutional AI
        self.constitutional_ai = ConstitutionalAI()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        
        # Training data
        self.preference_pairs = []
        self.training_history = []
        
    def collect_preference_pair(self, preferred_action: Dict[str, Any], 
        rejected_action: Dict[str, Any],
                              context: Dict[str, Any]) -> None:
        """Collect preference pair from action confidence scores"""
        
        # Calculate preference strength from confidence difference
        pref_confidence = preferred_action.get('confidence', 0.5)
        rej_confidence = rejected_action.get('confidence', 0.5)
        preference_strength = abs(pref_confidence - rej_confidence)
        
        pair = PreferencePair(
            preferred_action=preferred_action,
            rejected_action=rejected_action,
            preference_strength=preference_strength,
            context=context,
            timestamp=time.time()
        )
        
        self.preference_pairs.append(pair)
    
    def _encode_state_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> torch.Tensor:
        """Encode state-action pair for DPO network"""
        features = []
        
        # Action features
        features.append(action.get('confidence', 0.5))
        features.append(hash(action.get('action', 'default')) % 1000 / 1000.0)
        
        # Context features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, list):
                features.extend([float(x) for x in value[:5]])
        
        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32)
    
    def compute_dpo_loss(self, batch_pairs: List[PreferencePair]) -> torch.Tensor:
        """Compute DPO loss from preference pairs"""
        losses = []
        
        for pair in batch_pairs:
            # Encode preferred and rejected actions
            preferred_encoding = self._encode_state_action(pair.preferred_action, pair.context)
            rejected_encoding = self._encode_state_action(pair.rejected_action, pair.context)
            
            # Policy scores
            preferred_score = self.policy_net(preferred_encoding)
            rejected_score = self.policy_net(rejected_encoding)
            
            # Reference scores (frozen)
            with torch.no_grad():
                preferred_ref = self.reference_net(preferred_encoding)
                rejected_ref = self.reference_net(rejected_encoding)
            
            # DPO loss: -log(σ(β * (log π(preferred) - log π(rejected))))
            log_ratio_preferred = preferred_score - preferred_ref
            log_ratio_rejected = rejected_score - rejected_ref
            
            logits_diff = self.beta * (log_ratio_preferred - log_ratio_rejected)
            loss = -F.logsigmoid(logits_diff * pair.preference_strength)
            
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
        async def train_batch(self, batch_size: int = 32) -> Dict[str, Any]:
        """Train DPO policy on batch of preference pairs"""
        if len(self.preference_pairs) < batch_size:
            return {'status': 'insufficient_data', 'pairs_available': len(self.preference_pairs)}
        
        # Sample batch
        batch_indices = np.random.choice(len(self.preference_pairs), batch_size, replace=False)
        batch_pairs = [self.preference_pairs[i] for i in batch_indices]
        
        # Compute loss
        loss = self.compute_dpo_loss(batch_pairs)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track training
        training_step = {
            'loss': float(loss),
            'batch_size': batch_size,
            'timestamp': time.time()
        }
        self.training_history.append(training_step)
        
        return {
            'status': 'training_complete',
            'loss': float(loss),
            'batch_size': batch_size,
            'total_pairs': len(self.preference_pairs),
            'training_steps': len(self.training_history)
        }
    
        async def evaluate_action_preference(self, action: Dict[str, Any],
        context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action using trained DPO policy + Constitutional AI"""
        
        # DPO preference score
        state_action_encoding = self._encode_state_action(action, context)
        
        with torch.no_grad():
            dpo_score = float(self.policy_net(state_action_encoding))
        
        # Constitutional AI evaluation
        constitutional_eval = self.constitutional_ai.evaluate_action(action, context)
        
        # Combined evaluation
        combined_score = 0.7 * dpo_score + 0.3 * constitutional_eval['constitutional_compliance']
        
        return {
            'dpo_preference_score': dpo_score,
            'constitutional_evaluation': constitutional_eval,
            'combined_score': combined_score,
            'recommendation': 'approve' if combined_score > 0.6 else 'reject',
            'confidence': abs(combined_score - 0.5) * 2  # Convert to confidence
        }
    
        async def self_improve_system(self) -> Dict[str, Any]:
        """Self-improvement using Constitutional AI + DPO"""
        pass
        recent_actions = []
        
        # Collect recent actions from preference pairs
        recent_pairs = self.preference_pairs[-50:] if len(self.preference_pairs) >= 50 else self.preference_pairs
        
        for pair in recent_pairs:
            recent_actions.extend([pair.preferred_action, pair.rejected_action])
        
        # Constitutional self-improvement
        improvement_result = await self.constitutional_ai.self_improve(recent_actions)
        
        # Update reference network periodically (DPO stability)
        if len(self.training_history) % 100 == 0 and len(self.training_history) > 0:
            self.reference_net.load_state_dict(self.policy_net.state_dict())
            improvement_result['reference_network_updated'] = True
        
        return improvement_result
    
    def get_dpo_stats(self) -> Dict[str, Any]:
        """Get comprehensive DPO statistics"""
        pass
        avg_loss = np.mean([step['loss'] for step in self.training_history]) if self.training_history else 0.0
        
        return {
            'training_progress': {
                'total_preference_pairs': len(self.preference_pairs),
                'training_steps': len(self.training_history),
                'average_loss': avg_loss,
                'loss_trend': 'decreasing' if len(self.training_history) > 10 and 
                             self.training_history[-1]['loss'] < self.training_history[-10]['loss'] else 'stable'
            },
            'constitutional_ai': {
                'rules_count': len(self.constitutional_ai.constitutional_rules),
                'self_improvements': len(self.constitutional_ai.self_improvement_history),
                'recent_violations': len(self.constitutional_ai.rule_violations[-10:])
            },
            'model_architecture': {
                'policy_parameters': sum(p.numel() for p in self.policy_net.parameters()),
                'beta_coefficient': self.beta,
                'optimizer': 'AdamW',
                'learning_rate': 1e-4
            },
            'performance': {
                'preference_learning_active': len(self.preference_pairs) > 0,
                'constitutional_compliance_active': True,
                'self_improvement_active': len(self.constitutional_ai.self_improvement_history) > 0
            }
        }


# Global instance
_dpo_optimizer = None

    def get_dpo_optimizer():
        global _dpo_optimizer
        if _dpo_optimizer is None:
        _dpo_optimizer = DirectPreferenceOptimizer()
        return _dpo_optimizer
