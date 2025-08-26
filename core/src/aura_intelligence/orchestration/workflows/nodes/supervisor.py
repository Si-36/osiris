"""
🎯 Supervisor Node
Decision-making and coordination for workflow orchestration.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import time
from enum import Enum

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

import structlog
from aura_intelligence.resilience import resilient, ResilienceLevel

# Simple replacements for missing aura_common functions
def get_logger(name):
    return structlog.get_logger(name)

def with_correlation_id():
    def decorator(func):
        return func
    return decorator

def is_feature_enabled(feature):
    return True

def resilient_operation(**kwargs):
    def decorator(func):
        return func
    return decorator

from ..state import CollectiveState, NodeResult

logger = get_logger(__name__)


class DecisionType(str, Enum):
    """Types of supervisor decisions."""
    CONTINUE = "continue"
    ESCALATE = "escalate"
    RETRY = "retry"
    COMPLETE = "complete"
    ABORT = "abort"


class SupervisorNode:
    """
    Supervisor node for workflow coordination.
    
    Responsibilities:
    - Evaluate evidence and analysis
    - Make routing decisions
    - Assess risks
    - Coordinate agent consensus
    """
    
    def __init__(self, llm=None, risk_threshold: float = 0.7):
        """
        Initialize supervisor node.
        
        Args:
        llm: Optional LLM for decision making
        risk_threshold: Threshold for risk escalation
        """
        self.llm = llm
        self.risk_threshold = risk_threshold
        self.name = "supervisor"
    
    @with_correlation_id()
    @resilient_operation(
        max_retries=3,
        delay=1.0,
        backoff_factor=2.0
    )
    async def __call__(
        self,
        state: CollectiveState,
        config: Optional[RunnableConfig] = None
        ) -> Dict[str, Any]:
        """
        Execute supervisor decision logic.
        
        Args:
            state: Current workflow state
            config: Optional runtime configuration
            
        Returns:
            Updated state with decision
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Supervisor node starting",
                workflow_id=state["workflow_id"],
                thread_id=state["thread_id"],
                evidence_count=len(state.get("evidence_log", []))
            )
            
            # Analyze current state
            analysis = self._analyze_state(state)
            
            # Assess risk
            risk_score = self._assess_risk(state, analysis)
            
            # Make decision
            decision = await self._make_decision(state, analysis, risk_score)
            
            # Build decision record
            decision_record = self._build_decision_record(
                decision, analysis, risk_score
            )
            
            # Create result
            result = NodeResult(
                success=True,
                node_name=self.name,
                output=decision_record,
                duration_ms=(time.time() - start_time) * 1000,
                next_node=self._determine_next_node(decision)
            )
            
            # Update state
            updates = {
                "supervisor_decisions": [decision_record],
                "current_step": f"supervisor_decided_{decision.value}",
                "risk_assessment": {
                    "score": risk_score,
                    "threshold": self.risk_threshold,
                    "high_risk": risk_score > self.risk_threshold
                }
            }
            
            # Add message
            message = AIMessage(
                content=f"Supervisor decision: {decision.value} (risk: {risk_score:.2f})",
                additional_kwargs={"node": self.name, "decision": decision.value}
            )
            updates["messages"] = [message]
            
            logger.info(
                "Supervisor decision made",
                workflow_id=state["workflow_id"],
                decision=decision.value,
                risk_score=risk_score,
                duration_ms=result.duration_ms
            )
            
            return updates
            
        except Exception as e:
            logger.error(
                "Supervisor node failed",
                workflow_id=state["workflow_id"],
                error=str(e),
                exc_info=e
            )
            
            return {
                "error_log": [{
                    "node": self.name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }],
                "last_error": {
                    "node": self.name,
                    "message": str(e)
                },
                "current_step": "supervisor_error"
            }
    
    def _analyze_state(self, state: CollectiveState) -> Dict[str, Any]:
        """Analyze current workflow state."""
        analysis = {
        "evidence_count": len(state.get("evidence_log", [])),
        "error_count": len(state.get("error_log", [])),
        "decision_count": len(state.get("supervisor_decisions", [])),
        "has_risk_indicators": False,
        "completion_indicators": []
        }
        
        # Check for risk indicators in evidence
        for evidence in state.get("evidence_log", []):
            if evidence.get("risk_indicators"):
                analysis["has_risk_indicators"] = True
                analysis["risk_indicators"] = evidence["risk_indicators"]
                break
        
        # Check for completion indicators
        if state.get("execution_results"):
            analysis["completion_indicators"].append("execution_complete")
        
        if state.get("validation_results", {}).get("valid"):
            analysis["completion_indicators"].append("validation_passed")
        
        return analysis
    
    def _assess_risk(
        self,
        state: CollectiveState,
        analysis: Dict[str, Any]
        ) -> float:
        """Assess risk score based on state and analysis."""
        risk_score = 0.0
        
        # Error-based risk
        error_count = analysis["error_count"]
        if error_count > 0:
            risk_score += min(0.3, error_count * 0.1)
        
        # Risk indicators
        if analysis["has_risk_indicators"]:
            risk_indicators = analysis.get("risk_indicators", [])
            if "high_error_rate" in risk_indicators:
                risk_score += 0.3
            if "high_cpu_usage" in risk_indicators:
                risk_score += 0.2
            if "high_memory_usage" in risk_indicators:
                risk_score += 0.2
        
        # Recovery attempts
        recovery_attempts = state.get("error_recovery_attempts", 0)
        if recovery_attempts > 2:
            risk_score += 0.2
        
        # Cap at 1.0
        return min(1.0, risk_score)
    
    async def _make_decision(
        self,
        state: CollectiveState,
        analysis: Dict[str, Any],
        risk_score: float
    ) -> DecisionType:
        """Make supervisor decision."""
        # High risk - escalate
        if risk_score > self.risk_threshold:
            return DecisionType.ESCALATE
        
        # Errors with low recovery attempts - retry
        if (analysis["error_count"] > 0 and 
            state.get("error_recovery_attempts", 0) < 3):
            return DecisionType.RETRY
        
        # Completion indicators - complete
        if len(analysis["completion_indicators"]) >= 2:
            return DecisionType.COMPLETE
        
        # Too many errors - abort
        if analysis["error_count"] > 5:
            return DecisionType.ABORT
        
        # Default - continue
        return DecisionType.CONTINUE
    
    def _build_decision_record(
        self,
        decision: DecisionType,
        analysis: Dict[str, Any],
        risk_score: float
        ) -> Dict[str, Any]:
        """Build decision record for audit trail."""
        return {
            "node": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision.value,
            "risk_score": risk_score,
            "analysis": analysis,
            "reasoning": self._get_decision_reasoning(decision, analysis, risk_score)
        }
    
    def _get_decision_reasoning(
        self,
        decision: DecisionType,
        analysis: Dict[str, Any],
        risk_score: float
        ) -> str:
        """Get human-readable reasoning for decision."""
        if decision == DecisionType.ESCALATE:
            return f"Risk score {risk_score:.2f} exceeds threshold {self.risk_threshold}"
        elif decision == DecisionType.RETRY:
            return f"Errors detected ({analysis['error_count']}), attempting recovery"
        elif decision == DecisionType.COMPLETE:
            return f"Completion criteria met: {', '.join(analysis['completion_indicators'])}"
        elif decision == DecisionType.ABORT:
            return f"Too many errors ({analysis['error_count']}), aborting workflow"
        else:
            return "Continuing normal workflow execution"
    
    def _determine_next_node(self, decision: DecisionType) -> Optional[str]:
        """Determine next node based on decision."""
        if decision == DecisionType.CONTINUE:
            return "analyst"
        elif decision == DecisionType.RETRY:
            return "observer"
        elif decision == DecisionType.COMPLETE:
            return "end"
        elif decision == DecisionType.ESCALATE:
            return "human_review"
        elif decision == DecisionType.ABORT:
            return "error_handler"
        return None


class UnifiedAuraSupervisor(SupervisorNode):
    """
    🧠 Unified AURA Supervisor - TDA + LNN Integration
    
    Combines topology analysis with adaptive neural decision making for
    next-generation workflow supervision based on cutting-edge research.
    
    Features:
    - Real-time topological analysis of workflow structures
    - Liquid Neural Network adaptive decision making
    - Persistent homology for anomaly detection
    - Multi-head decision outputs (routing, risk, actions)
    - Online adaptation without retraining
    """
    
    def __init__(self, llm=None, risk_threshold: float = 0.7, tda_config=None, lnn_config=None):
        super().__init__(llm, risk_threshold)
        self.name = "unified_aura_supervisor"
        
        # Initialize TDA analyzer
        try:
            from .tda_supervisor_integration import ProductionTopologicalAnalyzer, ProductionTDAConfig
            self.tda_config = tda_config or ProductionTDAConfig()
            self.tda_analyzer = ProductionTopologicalAnalyzer(self.tda_config)
            self.tda_available = True
            logger.info("✅ TDA analyzer initialized")
        except Exception as e:
            logger.warning(f"⚠️ TDA analyzer initialization failed: {e}")
            self.tda_available = False
        
        # Initialize LNN decision engine
        try:
            from .lnn_supervisor_integration import ProductionLiquidNeuralDecisionEngine, ProductionLNNConfig
            import torch
            self.lnn_config = lnn_config or ProductionLNNConfig()
            self.lnn_engine = ProductionLiquidNeuralDecisionEngine(self.lnn_config)
            self.lnn_available = True
            logger.info("✅ LNN decision engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ LNN decision engine initialization failed: {e}")
            self.lnn_available = False
        
        # Performance tracking
        self.decision_history = []
        self.topology_cache = {}
        
        logger.info(f"🧠 UnifiedAuraSupervisor initialized (TDA: {self.tda_available}, LNN: {self.lnn_available})")
    
    async def __call__(self, state: CollectiveState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Enhanced supervisor with TDA + LNN integration"""
        start_time = time.time()
        
        try:
            workflow_id = state["workflow_id"]
            logger.info(f"🧠 Unified supervisor processing workflow: {workflow_id}")
            
            # Phase 1: Topological Analysis
            topology_result = None
            if self.tda_available:
                try:
                    topology_result = await self._analyze_workflow_topology(state)
                    logger.info(f"📊 Topology analysis: complexity={topology_result.get('complexity_score', 0):.3f}")
                except Exception as e:
                    logger.warning(f"TDA analysis failed: {e}")
            
            # Phase 2: Traditional Analysis (fallback)
            traditional_analysis = self._analyze_state(state)
            
            # Phase 3: LNN Adaptive Decision Making
            enhanced_decision = None
            if self.lnn_available and topology_result:
                try:
                    enhanced_decision = await self._make_lnn_decision(state, traditional_analysis, topology_result)
                    logger.info(f"🧠 LNN decision: confidence={enhanced_decision.get('confidence', 0):.3f}")
                except Exception as e:
                    logger.warning(f"LNN decision failed: {e}")
            
            # Phase 4: Unified Risk Assessment
            unified_risk = self._assess_unified_risk(state, traditional_analysis, topology_result, enhanced_decision)
            
            # Phase 5: Final Decision Integration
            final_decision = await self._integrate_decisions(
                state, traditional_analysis, topology_result, enhanced_decision, unified_risk
            )
            
            # Phase 6: Online Learning Update
            if self.lnn_available:
                await self._update_lnn_memory(state, final_decision, unified_risk)
            
            # Build comprehensive decision record
            decision_record = self._build_unified_decision_record(
                final_decision, traditional_analysis, topology_result, enhanced_decision, unified_risk
            )
            
            # Create enhanced result
            result = NodeResult(
                success=True,
                node_name=self.name,
                output=decision_record,
                duration_ms=(time.time() - start_time) * 1000,
                next_node=self._determine_next_node(final_decision)
            )
            
            # Update state with comprehensive information
            updates = {
                "supervisor_decisions": [decision_record],
                "current_step": f"unified_supervisor_decided_{final_decision.value}",
                "risk_assessment": {
                    "unified_risk_score": unified_risk,
                    "topology_complexity": topology_result.get('complexity_score') if topology_result else 0.0,
                    "lnn_confidence": enhanced_decision.get('confidence') if enhanced_decision else 0.0,
                    "threshold": self.risk_threshold,
                    "high_risk": unified_risk > self.risk_threshold
                },
                "topology_analysis": topology_result,
                "lnn_decision": enhanced_decision
            }
            
            # Add enhanced message
            message = AIMessage(
                content=f"🧠 Unified decision: {final_decision.value} (risk: {unified_risk:.2f}, topology: {topology_result.get('complexity_score', 0):.2f})",
                additional_kwargs={
                    "node": self.name, 
                    "decision": final_decision.value,
                    "topology_available": topology_result is not None,
                    "lnn_available": enhanced_decision is not None
                }
            )
            updates["messages"] = [message]
            
            logger.info(
                f"🧠 Unified supervisor decision complete",
                workflow_id=workflow_id,
                decision=final_decision.value,
                unified_risk=unified_risk,
                topology_complexity=topology_result.get('complexity_score') if topology_result else 0.0,
                duration_ms=result.duration_ms
            )
            
            return updates
            
        except Exception as e:
            logger.error(f"❌ Unified supervisor failed: {e}", exc_info=e)
            return await super().__call__(state, config)  # Fallback to traditional supervisor
    
    async def _analyze_workflow_topology(self, state: CollectiveState) -> Optional[Dict[str, Any]]:
        """Analyze workflow topology using TDA"""
        if not self.tda_available:
            return None
        
        try:
            # Convert state to workflow structure for TDA analysis
            workflow_structure = self._extract_workflow_structure(state)
            
            # Check cache first
            structure_hash = hash(str(sorted(workflow_structure.items())))
            if structure_hash in self.topology_cache:
                return self.topology_cache[structure_hash]
            
            # Perform topology analysis
            result = await self.tda_analyzer.analyze_workflow_topology(workflow_structure)
            
            # Cache result
            self.topology_cache[structure_hash] = result
            if len(self.topology_cache) > 100:  # Limit cache size
                oldest_key = next(iter(self.topology_cache))
                del self.topology_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Topology analysis failed: {e}")
            return None
    
    def _extract_workflow_structure(self, state: CollectiveState) -> Dict[str, Any]:
        """Extract workflow structure for topology analysis"""
        return {
            "agents": [
                {"id": f"agent_{i}", "type": "workflow_agent", "active": True}
                for i in range(len(state.get("messages", [])))
            ],
            "tasks": [
                {"id": f"task_{i}", "type": "workflow_task", "status": "active"}
                for i, evidence in enumerate(state.get("evidence_log", []))
            ],
            "messages": [
                {"sender": f"agent_{i%3}", "receiver": f"agent_{(i+1)%3}", "type": "coordination"}
                for i in range(len(state.get("messages", [])))
            ],
            "dependencies": [
                {"from": f"task_{i}", "to": f"task_{i+1}"}
                for i in range(max(0, len(state.get("evidence_log", [])) - 1))
            ],
            "workflow_id": state.get("workflow_id", "unknown"),
            "current_step": state.get("current_step", "unknown")
        }
    
    async def _make_lnn_decision(self, state: CollectiveState, analysis: Dict[str, Any], topology: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make decision using Liquid Neural Network"""
        if not self.lnn_available:
            return None
        
        try:
            import torch
            
            # Prepare input features
            features = self._prepare_lnn_features(state, analysis, topology)
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass through LNN
            with torch.no_grad():
                decision_outputs = self.lnn_engine(input_tensor)
            
            # Extract multi-head outputs
            routing_logits = decision_outputs['routing_head'].squeeze()
            risk_score = torch.sigmoid(decision_outputs['risk_head']).item()
            action_logits = decision_outputs['action_head'].squeeze()
            
            # Convert to probabilities
            routing_probs = torch.softmax(routing_logits, dim=0)
            action_probs = torch.softmax(action_logits, dim=0)
            
            # Map to decisions
            routing_decisions = ['continue', 'escalate', 'retry', 'complete', 'abort']
            action_decisions = ['observe', 'analyze', 'execute', 'review', 'terminate']
            
            routing_decision = routing_decisions[torch.argmax(routing_probs).item()]
            recommended_action = action_decisions[torch.argmax(action_probs).item()]
            confidence = torch.max(routing_probs).item()
            
            return {
                'routing_decision': routing_decision,
                'recommended_action': recommended_action,
                'risk_score': risk_score,
                'confidence': confidence,
                'routing_probabilities': routing_probs.tolist(),
                'action_probabilities': action_probs.tolist()
            }
            
        except Exception as e:
            logger.error(f"LNN decision failed: {e}")
            return None
    
    def _prepare_lnn_features(self, state: CollectiveState, analysis: Dict[str, Any], topology: Dict[str, Any]) -> List[float]:
        """Prepare feature vector for LNN input"""
        features = [
            # Traditional analysis features
            analysis.get('evidence_count', 0) / 10.0,  # Normalize
            analysis.get('error_count', 0) / 5.0,
            analysis.get('decision_count', 0) / 5.0,
            1.0 if analysis.get('has_risk_indicators', False) else 0.0,
            len(analysis.get('completion_indicators', [])) / 3.0,
            
            # Topology features
            topology.get('complexity_score', 0.0),
            topology.get('graph_properties', {}).get('density', 0.0),
            topology.get('analysis', {}).get('anomaly_score', 0.0),
            1.0 if topology.get('graph_properties', {}).get('is_connected', False) else 0.0,
            topology.get('graph_properties', {}).get('clustering', 0.0),
            
            # Temporal features
            len(state.get("messages", [])) / 20.0,
            state.get("error_recovery_attempts", 0) / 3.0,
            
            # Current context
            1.0 if "error" in state.get("current_step", "") else 0.0,
            1.0 if "complete" in state.get("current_step", "") else 0.0,
            1.0 if state.get("execution_results") else 0.0,
            1.0 if state.get("validation_results", {}).get("valid", False) else 0.0
        ]
        
        # Pad to expected input size (32 features)
        while len(features) < 32:
            features.append(0.0)
        
        return features[:32]  # Ensure exactly 32 features
    
    def _assess_unified_risk(self, state: CollectiveState, analysis: Dict[str, Any], 
                           topology: Optional[Dict[str, Any]], lnn_decision: Optional[Dict[str, Any]]) -> float:
        """Assess unified risk combining traditional, topology, and LNN signals"""
        # Start with traditional risk assessment
        traditional_risk = self._assess_risk(state, analysis)
        
        # Add topology-based risk
        topology_risk = 0.0
        if topology:
            complexity = topology.get('complexity_score', 0.0)
            anomaly_score = topology.get('analysis', {}).get('anomaly_score', 0.0)
            topology_risk = min(0.4, complexity * 0.2 + anomaly_score * 0.2)
        
        # Add LNN-based risk
        lnn_risk = 0.0
        if lnn_decision:
            lnn_risk_score = lnn_decision.get('risk_score', 0.0)
            confidence = lnn_decision.get('confidence', 0.5)
            # Weight LNN risk by confidence
            lnn_risk = lnn_risk_score * confidence * 0.3
        
        # Combine risks with weights
        unified_risk = (
            traditional_risk * 0.4 +  # Traditional analysis
            topology_risk * 0.3 +     # Topology analysis
            lnn_risk * 0.3             # LNN analysis
        )
        
        return min(1.0, unified_risk)
    
    async def _integrate_decisions(self, state: CollectiveState, analysis: Dict[str, Any],
                                 topology: Optional[Dict[str, Any]], lnn_decision: Optional[Dict[str, Any]],
                                 unified_risk: float) -> DecisionType:
        """Integrate all decision signals into final decision"""
        # Get traditional decision
        traditional_decision = await self._make_decision(state, analysis, unified_risk)
        
        # If no advanced analysis available, use traditional
        if not lnn_decision:
            return traditional_decision
        
        # Map LNN routing decision to DecisionType
        lnn_routing = lnn_decision.get('routing_decision', 'continue')
        lnn_confidence = lnn_decision.get('confidence', 0.0)
        
        decision_map = {
            'continue': DecisionType.CONTINUE,
            'escalate': DecisionType.ESCALATE,
            'retry': DecisionType.RETRY,
            'complete': DecisionType.COMPLETE,
            'abort': DecisionType.ABORT
        }
        
        lnn_decision_type = decision_map.get(lnn_routing, DecisionType.CONTINUE)
        
        # High confidence LNN decisions take precedence
        if lnn_confidence > 0.8:
            logger.info(f"🧠 High-confidence LNN decision: {lnn_decision_type.value}")
            return lnn_decision_type
        
        # Medium confidence - weighted combination
        elif lnn_confidence > 0.6:
            # If LNN and traditional agree, use that
            if lnn_decision_type == traditional_decision:
                return traditional_decision
            
            # If they disagree, bias toward safety (escalate/abort over continue)
            safety_order = [DecisionType.ABORT, DecisionType.ESCALATE, DecisionType.RETRY, DecisionType.COMPLETE, DecisionType.CONTINUE]
            decisions = [traditional_decision, lnn_decision_type]
            for safe_decision in safety_order:
                if safe_decision in decisions:
                    logger.info(f"🧠 Safety-biased decision: {safe_decision.value}")
                    return safe_decision
        
        # Low confidence - use traditional
        logger.info(f"🧠 Using traditional decision due to low LNN confidence: {traditional_decision.value}")
        return traditional_decision
    
    async def _update_lnn_memory(self, state: CollectiveState, decision: DecisionType, risk: float):
        """Update LNN with outcome for online learning"""
        if not self.lnn_available:
            return
        
        try:
            # Store decision in history for future learning
            decision_record = {
                'workflow_id': state.get('workflow_id'),
                'timestamp': datetime.now(timezone.utc),
                'decision': decision.value,
                'risk_score': risk,
                'state_features': self._extract_state_features(state)
            }
            
            self.decision_history.append(decision_record)
            
            # Keep only recent decisions
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
            
            logger.info(f"🧠 Updated LNN memory with decision: {decision.value}")
            
        except Exception as e:
            logger.error(f"LNN memory update failed: {e}")
    
    def _extract_state_features(self, state: CollectiveState) -> Dict[str, Any]:
        """Extract features from state for learning"""
        return {
            'evidence_count': len(state.get('evidence_log', [])),
            'error_count': len(state.get('error_log', [])),
            'message_count': len(state.get('messages', [])),
            'has_execution_results': bool(state.get('execution_results')),
            'has_validation_results': bool(state.get('validation_results')),
            'recovery_attempts': state.get('error_recovery_attempts', 0),
            'current_step': state.get('current_step', '')
        }
    
    def _build_unified_decision_record(self, decision: DecisionType, analysis: Dict[str, Any],
                                     topology: Optional[Dict[str, Any]], lnn_decision: Optional[Dict[str, Any]],
                                     unified_risk: float) -> Dict[str, Any]:
        """Build comprehensive decision record"""
        record = self._build_decision_record(decision, analysis, unified_risk)
        
        # Add unified supervisor specific information
        record.update({
            "supervisor_type": "unified_aura",
            "topology_analysis": topology,
            "lnn_decision": lnn_decision,
            "unified_risk_score": unified_risk,
            "analysis_capabilities": {
                "tda_available": self.tda_available,
                "lnn_available": self.lnn_available,
                "topology_cached": len(self.topology_cache),
                "decision_history": len(self.decision_history)
            },
            "reasoning": self._get_unified_reasoning(decision, analysis, topology, lnn_decision, unified_risk)
        })
        
        return record
    
    def _get_unified_reasoning(self, decision: DecisionType, analysis: Dict[str, Any],
                             topology: Optional[Dict[str, Any]], lnn_decision: Optional[Dict[str, Any]],
                             unified_risk: float) -> str:
        """Get comprehensive reasoning for unified decision"""
        base_reasoning = self._get_decision_reasoning(decision, analysis, unified_risk)
        
        enhancements = []
        
        if topology:
            complexity = topology.get('complexity_score', 0.0)
            if complexity > 0.7:
                enhancements.append(f"High topology complexity ({complexity:.2f})")
            elif complexity < 0.3:
                enhancements.append(f"Low topology complexity ({complexity:.2f})")
        
        if lnn_decision:
            confidence = lnn_decision.get('confidence', 0.0)
            lnn_routing = lnn_decision.get('routing_decision', 'unknown')
            if confidence > 0.8:
                enhancements.append(f"High-confidence LNN recommendation: {lnn_routing}")
            elif confidence > 0.6:
                enhancements.append(f"Medium-confidence LNN recommendation: {lnn_routing}")
        
        if enhancements:
            return f"{base_reasoning}. Enhanced analysis: {'; '.join(enhancements)}"
        
        return base_reasoning


# Factory function
def create_supervisor_node(
    llm=None,
    risk_threshold: float = 0.7
) -> SupervisorNode:
    """
    Create a supervisor node instance.
    
    Args:
        llm: Optional LLM for decision making
        risk_threshold: Risk threshold for escalation
        
    Returns:
        Configured supervisor node
    """
    return SupervisorNode(llm=llm, risk_threshold=risk_threshold)


def create_unified_aura_supervisor(
    llm=None,
    risk_threshold: float = 0.7,
    tda_config=None,
    lnn_config=None
) -> UnifiedAuraSupervisor:
    """
    Create a unified AURA supervisor instance.
    
    Args:
        llm: Optional LLM for decision making
        risk_threshold: Risk threshold for escalation
        tda_config: TDA configuration
        lnn_config: LNN configuration
        
    Returns:
        Configured unified AURA supervisor
    """
    return UnifiedAuraSupervisor(
        llm=llm, 
        risk_threshold=risk_threshold,
        tda_config=tda_config,
        lnn_config=lnn_config
    )