#!/usr/bin/env python3
"""
Working Council Agent
Clean, functional council agent implementation
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class DecisionType(Enum):
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    SCALING = "scaling"

@dataclass
class CouncilDecision:
    """Council decision with confidence and reasoning."""
    action: str
    decision_type: DecisionType
    confidence: float
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    decisions_made: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)

class CouncilAgent:
    """Working Council Agent implementation."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.status = "active"
        self.metrics = AgentMetrics()
        self.decisions: List[CouncilDecision] = []
        self.knowledge_base: Dict[str, Any] = {}
        
        print(f"🏛️ Council Agent initialized: {agent_id}")
    
        async def make_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make an intelligent council decision."""
        start_time = time.time()
        
        # Analyze context
        task = context.get('task', 'general_analysis')
        priority = context.get('priority', 'medium')
        data = context.get('data', {})
        
        # Determine decision type
        decision_type = self._determine_decision_type(task)
        
        # Generate decision based on council logic
        if decision_type == DecisionType.ANALYSIS:
            decision = await self._make_analysis_decision(context)
        elif decision_type == DecisionType.OPTIMIZATION:
            decision = await self._make_optimization_decision(context)
        elif decision_type == DecisionType.MONITORING:
            decision = await self._make_monitoring_decision(context)
        elif decision_type == DecisionType.SCALING:
            decision = await self._make_scaling_decision(context)
        else:
            decision = await self._make_general_decision(context)
        
        # Update metrics
        self._update_metrics(decision)
        
        # Store decision
        self.decisions.append(decision)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"🧠 Council decision: {decision.action} (confidence: {decision.confidence:.1%}, {processing_time:.1f}ms)")
        
        return decision
    
    def _determine_decision_type(self, task: str) -> DecisionType:
        """Determine the type of decision needed."""
        if any(word in task.lower() for word in ['analyze', 'analysis', 'examine', 'study']):
            return DecisionType.ANALYSIS
        elif any(word in task.lower() for word in ['optimize', 'improve', 'enhance', 'tune']):
            return DecisionType.OPTIMIZATION
        elif any(word in task.lower() for word in ['monitor', 'watch', 'observe', 'track']):
            return DecisionType.MONITORING
        elif any(word in task.lower() for word in ['scale', 'expand', 'grow', 'resize']):
            return DecisionType.SCALING
        else:
            return DecisionType.ANALYSIS
    
        async def _make_analysis_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make an analysis decision."""
        await asyncio.sleep(0.1)  # Simulate processing
        
        priority = context.get('priority', 'medium')
        data = context.get('data', {})
        
        if priority == 'high':
            action = "immediate_deep_analysis"
            confidence = 0.9
            reasoning = [
                "High priority task detected",
                "Deploying advanced analysis algorithms",
                "Real-time processing enabled"
            ]
        elif data:
            action = "data_driven_analysis"
            confidence = 0.8
            reasoning = [
                "Data context available",
                "Applying statistical analysis",
                "Pattern recognition active"
            ]
        else:
            action = "standard_analysis"
            confidence = 0.7
            reasoning = [
                "Standard analysis protocol",
                "Baseline assessment approach"
            ]
        
        return CouncilDecision(
            action=action,
            decision_type=DecisionType.ANALYSIS,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"priority": priority, "data_available": bool(data)}
        )
    
        async def _make_optimization_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make an optimization decision."""
        await asyncio.sleep(0.05)
        
        data = context.get('data', {})
        
        if data.get('cpu_usage', 0) > 80:
            action = "cpu_optimization"
            confidence = 0.85
            reasoning = [
                "High CPU usage detected",
                "Implementing CPU optimization strategies",
                "Load balancing recommended"
            ]
        elif data.get('memory_usage', 0) > 80:
            action = "memory_optimization"
            confidence = 0.85
            reasoning = [
                "High memory usage detected",
                "Memory cleanup protocols activated",
                "Garbage collection optimization"
            ]
        else:
            action = "general_optimization"
            confidence = 0.75
            reasoning = [
                "General optimization assessment",
                "Performance tuning recommendations"
            ]
        
        return CouncilDecision(
            action=action,
            decision_type=DecisionType.OPTIMIZATION,
            confidence=confidence,
            reasoning=reasoning,
            metadata=data
        )
    
        async def _make_monitoring_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make a monitoring decision."""
        priority = context.get('priority', 'medium')
        
        if priority == 'high':
            action = "enhanced_monitoring"
            confidence = 0.9
            reasoning = [
                "High priority monitoring required",
                "Real-time alerting enabled",
                "Comprehensive metrics collection"
            ]
        else:
            action = "standard_monitoring"
            confidence = 0.8
            reasoning = [
                "Standard monitoring protocol",
                "Regular health checks active"
            ]
        
        return CouncilDecision(
            action=action,
            decision_type=DecisionType.MONITORING,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"priority": priority}
        )
    
        async def _make_scaling_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make a scaling decision."""
        data = context.get('data', {})
        load = data.get('load', 50)
        
        if load > 80:
            action = "scale_up"
            confidence = 0.9
            reasoning = [
                "High load detected",
                "Scaling up resources",
                "Performance optimization required"
            ]
        elif load < 20:
            action = "scale_down"
            confidence = 0.8
            reasoning = [
                "Low load detected",
                "Resource optimization opportunity",
                "Cost reduction possible"
            ]
        else:
            action = "maintain_scale"
            confidence = 0.7
            reasoning = [
                "Load within normal range",
                "Current scaling appropriate"
            ]
        
        return CouncilDecision(
            action=action,
            decision_type=DecisionType.SCALING,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"load": load}
        )
    
        async def _make_general_decision(self, context: Dict[str, Any]) -> CouncilDecision:
        """Make a general decision."""
        action = "analyze_and_recommend"
        confidence = 0.75
        reasoning = [
            "General council analysis",
            "Comprehensive assessment approach",
            "Balanced decision making"
        ]
        
        return CouncilDecision(
            action=action,
            decision_type=DecisionType.ANALYSIS,
            confidence=confidence,
            reasoning=reasoning,
            metadata=context
        )
    
    def _update_metrics(self, decision: CouncilDecision):
        """Update agent metrics."""
        self.metrics.decisions_made += 1
        self.metrics.last_activity = datetime.now()
        
        # Update average confidence
        if self.metrics.average_confidence == 0:
            self.metrics.average_confidence = decision.confidence
        else:
            self.metrics.average_confidence = (
                self.metrics.average_confidence * 0.8 + decision.confidence * 0.2
            )
        
        # Update success rate (decisions with >70% confidence are considered successful)
        successful_decisions = sum(1 for d in self.decisions if d.confidence > 0.7)
        if len(self.decisions) > 0:
            self.metrics.success_rate = successful_decisions / len(self.decisions)
        else:
            self.metrics.success_rate = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        pass
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "decisions_made": self.metrics.decisions_made,
            "success_rate": self.metrics.success_rate,
            "average_confidence": self.metrics.average_confidence,
            "last_activity": self.metrics.last_activity.isoformat(),
            "recent_decisions": [
                {
                    "action": d.action,
                    "type": d.decision_type.value,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in self.decisions[-5:]  # Last 5 decisions
            ]
        }
    
    def add_knowledge(self, key: str, value: Any):
        """Add knowledge to the agent's knowledge base."""
        self.knowledge_base[key] = value
        print(f"📚 Knowledge added: {key}")
    
    def get_knowledge(self, key: str) -> Optional[Any]:
        """Get knowledge from the agent's knowledge base."""
        return self.knowledge_base.get(key)

# Factory function
    def create_council_agent(agent_id: str) -> CouncilAgent:
        """Create a new council agent."""
        return CouncilAgent(agent_id)
