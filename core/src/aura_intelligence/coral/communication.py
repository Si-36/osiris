"""
CoRaL Communication System for AURA Intelligence
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time


@dataclass
class CoRaLMessage:
    sender_id: str
    content: np.ndarray
    timestamp: float
    confidence: float


class InformationAgent(nn.Module):
    def __init__(self, input_dim: int = 128, message_dim: int = 32):
        super().__init__()
        self.world_model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.message_encoder = nn.Sequential(
            nn.Linear(64, message_dim),
            nn.Tanh()
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        world_state = self.world_model(context)
        message = self.message_encoder(world_state)
        return message
    
    def generate_message(self, context: Dict[str, Any]) -> CoRaLMessage:
        context_tensor = self._context_to_tensor(context)
        with torch.no_grad():
            message_vec = self.forward(context_tensor)
            
        return CoRaLMessage(
            sender_id="information_agent",
            content=message_vec.numpy(),
            timestamp=time.time(),
            confidence=0.8
        )
    
    def _context_to_tensor(self, context: Dict[str, Any]) -> torch.Tensor:
        features = [
            context.get('system_health', 0.5),
            context.get('component_count', 50) / 100.0,
            context.get('memory_usage', 0.3),
            context.get('topology_score', 0.7)
        ]
        while len(features) < 128:
            features.append(0.0)
        return torch.tensor(features[:128], dtype=torch.float32).unsqueeze(0)


class ControlAgent(nn.Module):
    def __init__(self, obs_dim: int = 64, message_dim: int = 32, action_dim: int = 8):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim + message_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, observation: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat([observation, message], dim=-1)
        action_logits = self.policy_net(combined_input)
        return F.softmax(action_logits, dim=-1)
    
    def make_decision(self, observation: Dict[str, Any], message: CoRaLMessage) -> Dict[str, Any]:
        obs_tensor = self._obs_to_tensor(observation)
        msg_tensor = torch.tensor(message.content, dtype=torch.float32)
        
        # Ensure tensors have same batch dimension
        if msg_tensor.dim() == 1:
            msg_tensor = msg_tensor.unsqueeze(0)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            action_probs = self.forward(obs_tensor, msg_tensor)
            action = torch.argmax(action_probs, dim=-1).item()
            
        return {
            'action': action,
            'action_probs': action_probs.numpy(),
            'confidence': float(torch.max(action_probs))
        }
    
    def _obs_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        features = [
            observation.get('active_components', 50) / 100.0,
            observation.get('response_time', 0.1),
            observation.get('throughput', 0.8),
            observation.get('error_rate', 0.05)
        ]
        while len(features) < 64:
            features.append(0.0)
        return torch.tensor(features[:64], dtype=torch.float32).unsqueeze(0)


class CoRaLCommunicationSystem:
    def __init__(self):
        self.information_agents = {}
        self.control_agents = {}
        
        # Create 50 IA and 50 CA for demonstration
        for i in range(50):
            self.information_agents[f"ia_{i}"] = InformationAgent()
            self.control_agents[f"ca_{i}"] = ControlAgent()
            
        self.message_history = []
        self.influence_scores = []
        
    async def communication_round(self, global_context: Dict[str, Any]) -> Dict[str, Any]:
        # Phase 1: Information agents generate messages
        ia_messages = {}
        for agent_id, ia in self.information_agents.items():
            message = ia.generate_message(global_context)
            ia_messages[agent_id] = message
            self.message_history.append(message)
        
        # Phase 2: Control agents make decisions
        ca_decisions = {}
        total_influence = 0.0
        
        for agent_id, ca in self.control_agents.items():
            if ia_messages:
                message = list(ia_messages.values())[0]
                decision = ca.make_decision(global_context, message)
                
                # Simple causal influence calculation
                influence = np.random.uniform(0.1, 0.9)  # Placeholder
                
                ca_decisions[agent_id] = {
                    'decision': decision,
                    'causal_influence': influence
                }
                
                total_influence += influence
                self.influence_scores.append(influence)
        
        avg_influence = total_influence / len(ca_decisions) if ca_decisions else 0.0
        
        return {
            'ia_messages': len(ia_messages),
            'ca_decisions': len(ca_decisions),
            'average_causal_influence': avg_influence,
            'communication_efficiency': min(1.0, avg_influence * 2)
        }
    
    def get_communication_stats(self) -> Dict[str, Any]:
        return {
            'total_information_agents': len(self.information_agents),
            'total_control_agents': len(self.control_agents),
            'messages_sent': len(self.message_history),
            'average_influence': np.mean(self.influence_scores) if self.influence_scores else 0.0
        }


_global_coral_system: Optional[CoRaLCommunicationSystem] = None

def get_coral_system() -> CoRaLCommunicationSystem:
    global _global_coral_system
    if _global_coral_system is None:
        _global_coral_system = CoRaLCommunicationSystem()
    return _global_coral_system