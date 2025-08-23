"""
LNN Council System - 2025 Production
Liquid Neural Networks with Byzantine consensus
"""

import asyncio
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Import our components
from ..lnn.real_mit_lnn import get_real_mit_lnn
from .mcp_communication_hub import get_mcp_communication_hub, AgentMessage, MessageType
from .mem0_neo4j_bridge import get_mem0_neo4j_bridge

@dataclass
class CouncilDecision:
    decision: str
    confidence: float
    reasoning: List[str]
    agent_votes: Dict[str, Any]
    topological_context: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class LNNCouncilAgent:
    """Individual council agent with Liquid Neural Network"""
    
    def __init__(self, agent_id: str, specialization: str = "general"):
        self.agent_id = agent_id
        self.specialization = specialization
        
        # Real MIT LNN
        self.lnn = get_real_mit_lnn(input_size=256, hidden_size=128, output_size=64)
        
        # Decision history
        self.decision_history = []
        self.performance_metrics = {
            'decisions_made': 0,
            'consensus_rate': 0.0,
            'avg_confidence': 0.0
        }
    
    async def process_decision_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process decision request using LNN"""
        
        # Step 1: Encode context for LNN
        context_tensor = self._encode_context(context)
        
        # Step 2: LNN inference
        with torch.no_grad():
            lnn_output = self.lnn(context_tensor)
        
        # Step 3: Interpret output as decision
        decision_data = self._interpret_lnn_output(lnn_output, context)
        
        # Step 4: Update metrics
        self._update_metrics(decision_data)
        
        return decision_data
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode context dictionary as tensor for LNN"""
        features = []
        
        # Extract numeric features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple hash encoding
                features.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    features.extend([float(x) for x in value[:10]])
        
        # Pad or truncate to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        features = features[:256]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _interpret_lnn_output(self, output: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret LNN output as decision"""
        # Handle tuple output from CfC
        if isinstance(output, tuple):
            output = output[0]
        
        output_np = output.squeeze().numpy()
        
        # Decision based on output pattern
        decision_score = float(output_np.mean())
        
        if decision_score > 0.6:
            decision = "approve"
            confidence = min(decision_score, 1.0)
        elif decision_score < 0.4:
            decision = "reject"
            confidence = min(1.0 - decision_score, 1.0)
        else:
            decision = "abstain"
            confidence = 0.5
        
        # Generate reasoning
        reasoning = [
            f"LNN analysis of {self.specialization} context",
            f"Decision score: {decision_score:.3f}",
            f"Based on {len(context)} context features"
        ]
        
        return {
            'agent_id': self.agent_id,
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'lnn_output': output_np.tolist()[:10],  # First 10 values
            'specialization': self.specialization
        }
    
    def _update_metrics(self, decision_data: Dict[str, Any]):
        """Update agent performance metrics"""
        self.performance_metrics['decisions_made'] += 1
        
        # Update average confidence
        old_avg = self.performance_metrics['avg_confidence']
        new_confidence = decision_data['confidence']
        count = self.performance_metrics['decisions_made']
        
        self.performance_metrics['avg_confidence'] = (
            (old_avg * (count - 1) + new_confidence) / count
        )

class LNNCouncilSystem:
    """Complete LNN Council System with Byzantine consensus"""
    
    def __init__(self):
        self.council_agents = {}
        self.mcp_hub = get_mcp_communication_hub()
        self.memory_bridge = get_mem0_neo4j_bridge()
        
        # Byzantine fault tolerance settings
        self.min_agents = 3
        self.consensus_threshold = 0.67  # 2/3 majority
        
    async def initialize(self):
        """Initialize council system"""
        await self.mcp_hub.initialize()
        await self.memory_bridge.initialize()
        
        # Create specialized council agents
        specializations = [
            "security_analysis",
            "performance_optimization", 
            "resource_allocation",
            "risk_assessment",
            "quality_assurance"
        ]
        
        for i, spec in enumerate(specializations):
            agent_id = f"council_agent_{i}_{spec}"
            agent = LNNCouncilAgent(agent_id, spec)
            self.council_agents[agent_id] = agent
            
            # Register with MCP hub
            self.mcp_hub.register_agent(agent_id, self._handle_agent_message)
    
    async def _handle_agent_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle messages for council agents"""
        if message.message_type == MessageType.DECISION_REQUEST:
            agent = self.council_agents.get(message.receiver_id)
            if agent:
                return await agent.process_decision_request(message.payload)
        
        return {'status': 'message_not_handled'}
    
    async def make_council_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make decision using full council with Byzantine consensus"""
        
        # Step 1: Get topological context from memory
        topological_context = await self._get_topological_context(context)
        
        # Step 2: Enhance context with topological features
        enhanced_context = {**context, 'topological_features': topological_context}
        
        # Step 3: Get decisions from all council agents
        agent_decisions = await self._collect_agent_decisions(enhanced_context)
        
        # Step 4: Apply Byzantine consensus
        final_decision = self._byzantine_consensus(agent_decisions)
        
        # Step 5: Store decision in memory
        await self._store_decision_memory(final_decision, enhanced_context)
        
        return final_decision
    
    async def _get_topological_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get topological context from Neo4j"""
        
        # Extract context data for topological analysis
        context_data = []
        for key, value in context.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    context_data.append(value)
        
        if not context_data:
            return {'betti_numbers': [1, 0], 'complexity_score': 0.0}
        
        # Get topological signature
        try:
            import numpy as np
            data_array = np.array(context_data)
            signature = await self.memory_bridge.tda_bridge._compute_topology(data_array)
            
            return {
                'betti_numbers': signature.betti_numbers,
                'complexity_score': signature.complexity_score,
                'shape_hash': signature.shape_hash
            }
        except:
            return {'betti_numbers': [1, 0], 'complexity_score': 0.0}
    
    async def _collect_agent_decisions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect decisions from all council agents"""
        
        decisions = []
        tasks = []
        
        for agent_id, agent in self.council_agents.items():
            task = agent.process_decision_request(context)
            tasks.append(task)
        
        # Wait for all agents to respond
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                decisions.append(result)
            else:
                # Handle agent failure
                agent_id = list(self.council_agents.keys())[i]
                decisions.append({
                    'agent_id': agent_id,
                    'decision': 'abstain',
                    'confidence': 0.0,
                    'reasoning': [f'Agent failed: {str(result)}'],
                    'failed': True
                })
        
        return decisions
    
    def _byzantine_consensus(self, agent_decisions: List[Dict[str, Any]]) -> CouncilDecision:
        """Apply Byzantine fault tolerant consensus"""
        
        # Filter out failed agents
        valid_decisions = [d for d in agent_decisions if not d.get('failed', False)]
        
        if len(valid_decisions) < self.min_agents:
            return CouncilDecision(
                decision="insufficient_agents",
                confidence=0.0,
                reasoning=["Not enough valid agent responses for consensus"],
                agent_votes={}
            )
        
        # Count votes
        vote_counts = {}
        confidence_by_vote = {}
        
        for decision_data in valid_decisions:
            vote = decision_data['decision']
            confidence = decision_data['confidence']
            
            if vote not in vote_counts:
                vote_counts[vote] = 0
                confidence_by_vote[vote] = []
            
            vote_counts[vote] += 1
            confidence_by_vote[vote].append(confidence)
        
        # Find majority decision
        total_votes = len(valid_decisions)
        majority_vote = None
        majority_count = 0
        
        for vote, count in vote_counts.items():
            if count > majority_count:
                majority_count = count
                majority_vote = vote
        
        # Check if consensus threshold is met
        consensus_ratio = majority_count / total_votes
        
        if consensus_ratio >= self.consensus_threshold:
            # Strong consensus
            avg_confidence = sum(confidence_by_vote[majority_vote]) / len(confidence_by_vote[majority_vote])
            
            reasoning = [
                f"Byzantine consensus achieved: {majority_count}/{total_votes} agents",
                f"Consensus strength: {consensus_ratio:.2%}",
                f"Average confidence: {avg_confidence:.3f}"
            ]
            
            return CouncilDecision(
                decision=majority_vote,
                confidence=avg_confidence * consensus_ratio,  # Weight by consensus strength
                reasoning=reasoning,
                agent_votes={d['agent_id']: d['decision'] for d in valid_decisions}
            )
        else:
            # No strong consensus
            return CouncilDecision(
                decision="no_consensus",
                confidence=0.0,
                reasoning=[
                    f"No consensus reached: {majority_count}/{total_votes} agents",
                    f"Consensus strength: {consensus_ratio:.2%} < {self.consensus_threshold:.2%}"
                ],
                agent_votes={d['agent_id']: d['decision'] for d in valid_decisions}
            )
    
    async def _store_decision_memory(self, decision: CouncilDecision, context: Dict[str, Any]):
        """Store council decision in memory for future reference"""
        
        memory_content = {
            'decision_type': 'council_decision',
            'decision': decision.decision,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'agent_votes': decision.agent_votes,
            'context_summary': {
                'context_keys': list(context.keys()),
                'topological_features': context.get('topological_features', {})
            }
        }
        
        # Store in hybrid memory
        await self.memory_bridge.store_hybrid_memory(
            agent_id="council_system",
            content=memory_content
        )
    
    async def get_council_stats(self) -> Dict[str, Any]:
        """Get council system statistics"""
        
        agent_stats = {}
        for agent_id, agent in self.council_agents.items():
            agent_stats[agent_id] = {
                'specialization': agent.specialization,
                'decisions_made': agent.performance_metrics['decisions_made'],
                'avg_confidence': agent.performance_metrics['avg_confidence']
            }
        
        return {
            'total_agents': len(self.council_agents),
            'min_agents_required': self.min_agents,
            'consensus_threshold': self.consensus_threshold,
            'agent_statistics': agent_stats,
            'system_status': 'operational'
        }

# Global instance
_lnn_council_system = None

def get_lnn_council_system():
    global _lnn_council_system
    if _lnn_council_system is None:
        _lnn_council_system = LNNCouncilSystem()
    return _lnn_council_system