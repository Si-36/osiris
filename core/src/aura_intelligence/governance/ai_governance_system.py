"""
AI Governance System - 2025 Implementation

Based on latest research:
- EU AI Act compliance (2025)
- Constitutional AI principles
- Multi-stakeholder governance
- Explainable AI requirements
- Risk-based approach
- Continuous monitoring and auditing

Key features:
- Policy enforcement engine
- Risk assessment framework
- Compliance monitoring
- Ethical decision making
- Audit trail generation
- Stakeholder management
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import structlog
from collections import defaultdict, deque
import numpy as np

logger = structlog.get_logger(__name__)


class RiskLevel(str, Enum):
    """AI system risk levels per EU AI Act"""
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ComplianceStatus(str, Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"


class GovernanceAction(str, Enum):
    """Governance actions"""
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    MONITOR = "monitor"
    AUDIT = "audit"


@dataclass
class Policy:
    """Governance policy definition"""
    policy_id: str
    name: str
    description: str
    category: str  # safety, privacy, fairness, transparency
    
    # Rules
    rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Enforcement
    enforcement_level: str = "strict"  # strict, moderate, advisory
    auto_enforce: bool = True
    
    # Metadata
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    
    # Compliance mapping
    regulatory_mapping: Dict[str, str] = field(default_factory=dict)  # EU AI Act, ISO, etc.


@dataclass
class GovernanceDecision:
    """Governance decision record"""
    # Required fields
    decision_id: str
    action: GovernanceAction
    subject: str  # What was evaluated
    risk_level: RiskLevel
    compliance_status: ComplianceStatus
    
    # Optional fields with defaults
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Policies applied
    applied_policies: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Reasoning
    reasoning: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Approval chain
    decided_by: str = "system"
    approved_by: Optional[str] = None
    escalated_to: Optional[str] = None


@dataclass
class StakeholderFeedback:
    """Stakeholder feedback on governance decisions"""
    feedback_id: str
    decision_id: str
    stakeholder_id: str
    stakeholder_type: str  # user, regulator, ethics_board, developer
    
    # Feedback
    agrees_with_decision: bool
    confidence_in_decision: float  # 0-1
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    weight: float = 1.0  # Stakeholder influence weight


class PolicyEngine:
    """
    Policy enforcement engine
    Evaluates actions against governance policies
    """
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.policy_cache: Dict[str, Any] = {}
        
        # Initialize default policies
        self._load_default_policies()
        
        logger.info("Policy engine initialized")
    
    def _load_default_policies(self):
        """Load default governance policies"""
        # Safety policy
        self.add_policy(Policy(
            policy_id="safety_001",
            name="Human Safety",
            description="Prevent actions that could harm humans",
            category="safety",
            rules=[
                {
                    "type": "prohibition",
                    "condition": "action.potential_harm > 0",
                    "message": "Action may cause harm to humans"
                },
                {
                    "type": "requirement",
                    "condition": "action.affects_humans",
                    "requirement": "safety_assessment.completed"
                }
            ],
            enforcement_level="strict",
            regulatory_mapping={"EU_AI_Act": "Article 5"}
        ))
        
        # Privacy policy
        self.add_policy(Policy(
            policy_id="privacy_001",
            name="Data Privacy",
            description="Protect personal data and privacy",
            category="privacy",
            rules=[
                {
                    "type": "requirement",
                    "condition": "data.contains_pii",
                    "requirement": "consent.obtained"
                },
                {
                    "type": "prohibition",
                    "condition": "data.sensitive AND NOT encryption.enabled",
                    "message": "Sensitive data must be encrypted"
                }
            ],
            enforcement_level="strict",
            regulatory_mapping={"GDPR": "Article 6", "EU_AI_Act": "Article 10"}
        ))
        
        # Fairness policy
        self.add_policy(Policy(
            policy_id="fairness_001",
            name="Algorithmic Fairness",
            description="Ensure fair and unbiased decisions",
            category="fairness",
            rules=[
                {
                    "type": "requirement",
                    "condition": "decision.affects_individuals",
                    "requirement": "bias_assessment.passed"
                },
                {
                    "type": "monitoring",
                    "metric": "demographic_parity",
                    "threshold": 0.8
                }
            ],
            enforcement_level="moderate",
            regulatory_mapping={"EU_AI_Act": "Article 15"}
        ))
        
        # Transparency policy
        self.add_policy(Policy(
            policy_id="transparency_001",
            name="AI Transparency",
            description="Ensure AI decisions are explainable",
            category="transparency",
            rules=[
                {
                    "type": "requirement",
                    "condition": "decision.high_stakes",
                    "requirement": "explanation.provided"
                },
                {
                    "type": "documentation",
                    "required_fields": ["decision_factors", "confidence", "alternatives"]
                }
            ],
            enforcement_level="moderate",
            regulatory_mapping={"EU_AI_Act": "Article 13"}
        ))
    
    def add_policy(self, policy: Policy):
        """Add or update policy"""
        self.policies[policy.policy_id] = policy
        self.policy_cache.clear()  # Invalidate cache
        logger.info(f"Policy added: {policy.name}")
    
    async def evaluate(self, 
                      action: str,
                      context: Dict[str, Any],
                      policies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate action against policies"""
        # Select policies to apply
        if policies:
            applicable_policies = [p for pid, p in self.policies.items() if pid in policies]
        else:
            applicable_policies = self._select_applicable_policies(action, context)
        
        # Evaluate each policy
        violations = []
        requirements = []
        
        for policy in applicable_policies:
            policy_result = await self._evaluate_policy(policy, action, context)
            
            if policy_result["violations"]:
                violations.extend(policy_result["violations"])
            if policy_result["requirements"]:
                requirements.extend(policy_result["requirements"])
        
        # Determine overall compliance
        if violations and any(v["enforcement"] == "strict" for v in violations):
            compliance = ComplianceStatus.NON_COMPLIANT
        elif violations or requirements:
            compliance = ComplianceStatus.PARTIAL
        else:
            compliance = ComplianceStatus.COMPLIANT
        
        return {
            "compliance": compliance,
            "violations": violations,
            "requirements": requirements,
            "applied_policies": [p.policy_id for p in applicable_policies]
        }
    
    def _select_applicable_policies(self, 
                                   action: str,
                                   context: Dict[str, Any]) -> List[Policy]:
        """Select policies applicable to the action"""
        applicable = []
        
        for policy in self.policies.values():
            # Check if policy category matches action type
            if self._policy_applies(policy, action, context):
                applicable.append(policy)
        
        return applicable
    
    def _policy_applies(self, policy: Policy, action: str, context: Dict[str, Any]) -> bool:
        """Check if policy applies to action"""
        # Simple keyword matching - in production would use more sophisticated matching
        action_lower = action.lower()
        
        if policy.category == "safety" and any(kw in action_lower for kw in ["harm", "danger", "risk"]):
            return True
        elif policy.category == "privacy" and any(kw in action_lower for kw in ["data", "personal", "pii"]):
            return True
        elif policy.category == "fairness" and any(kw in action_lower for kw in ["decision", "classify", "rank"]):
            return True
        elif policy.category == "transparency" and context.get("requires_explanation", False):
            return True
        
        return False
    
    async def _evaluate_policy(self,
                             policy: Policy,
                             action: str,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate single policy"""
        violations = []
        requirements = []
        
        for rule in policy.rules:
            if rule["type"] == "prohibition":
                if self._evaluate_condition(rule["condition"], action, context):
                    violations.append({
                        "policy_id": policy.policy_id,
                        "rule": rule,
                        "message": rule.get("message", "Policy violation"),
                        "enforcement": policy.enforcement_level
                    })
            
            elif rule["type"] == "requirement":
                if self._evaluate_condition(rule["condition"], action, context):
                    if not self._check_requirement(rule["requirement"], context):
                        requirements.append({
                            "policy_id": policy.policy_id,
                            "rule": rule,
                            "requirement": rule["requirement"],
                            "enforcement": policy.enforcement_level
                        })
        
        return {
            "violations": violations,
            "requirements": requirements
        }
    
    def _evaluate_condition(self, condition: str, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition"""
        # Simple evaluation - in production would use proper expression parser
        try:
            # Create evaluation context
            eval_context = {
                "action": {"type": action, **context.get("action_details", {})},
                "data": context.get("data", {}),
                "decision": context.get("decision", {})
            }
            
            # WARNING: eval is dangerous - use proper parser in production
            return eval(condition, {"__builtins__": {}}, eval_context)
        except:
            return False
    
    def _check_requirement(self, requirement: str, context: Dict[str, Any]) -> bool:
        """Check if requirement is met"""
        # Check if requirement exists in context
        parts = requirement.split(".")
        current = context
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        return bool(current)


class RiskAssessment:
    """
    Risk assessment framework
    Evaluates AI system risks per EU AI Act
    """
    
    def __init__(self):
        self.risk_factors = {
            "human_impact": 0.4,
            "scale": 0.2,
            "autonomy": 0.2,
            "transparency": 0.1,
            "reversibility": 0.1
        }
        
        logger.info("Risk assessment initialized")
    
    async def assess(self, 
                    system: str,
                    capabilities: Dict[str, Any],
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess AI system risk level"""
        # Calculate risk scores
        scores = {}
        
        # Human impact
        scores["human_impact"] = self._assess_human_impact(capabilities, context)
        
        # Scale of deployment
        scores["scale"] = self._assess_scale(context)
        
        # Level of autonomy
        scores["autonomy"] = self._assess_autonomy(capabilities)
        
        # Transparency
        scores["transparency"] = self._assess_transparency(capabilities)
        
        # Reversibility
        scores["reversibility"] = self._assess_reversibility(capabilities)
        
        # Calculate weighted risk score
        total_risk = sum(
            scores[factor] * weight 
            for factor, weight in self.risk_factors.items()
        )
        
        # Determine risk level
        if total_risk >= 0.8:
            risk_level = RiskLevel.UNACCEPTABLE
        elif total_risk >= 0.6:
            risk_level = RiskLevel.HIGH
        elif total_risk >= 0.3:
            risk_level = RiskLevel.LIMITED
        else:
            risk_level = RiskLevel.MINIMAL
        
        return {
            "risk_level": risk_level,
            "total_score": total_risk,
            "factor_scores": scores,
            "recommendations": self._generate_recommendations(risk_level, scores),
            "regulatory_requirements": self._get_regulatory_requirements(risk_level)
        }
    
    def _assess_human_impact(self, capabilities: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess potential impact on humans"""
        impact_score = 0.0
        
        # Check for safety-critical applications
        if context.get("domain") in ["healthcare", "transportation", "criminal_justice"]:
            impact_score += 0.5
        
        # Check for individual decision-making
        if capabilities.get("makes_decisions_about_individuals", False):
            impact_score += 0.3
        
        # Check for physical world interaction
        if capabilities.get("controls_physical_systems", False):
            impact_score += 0.2
        
        return min(impact_score, 1.0)
    
    def _assess_scale(self, context: Dict[str, Any]) -> float:
        """Assess scale of deployment"""
        users = context.get("estimated_users", 0)
        
        if users > 1000000:
            return 1.0
        elif users > 10000:
            return 0.7
        elif users > 100:
            return 0.4
        else:
            return 0.1
    
    def _assess_autonomy(self, capabilities: Dict[str, Any]) -> float:
        """Assess level of system autonomy"""
        autonomy_score = 0.0
        
        if capabilities.get("fully_autonomous", False):
            autonomy_score = 1.0
        elif capabilities.get("semi_autonomous", False):
            autonomy_score = 0.6
        elif capabilities.get("human_in_loop", True):
            autonomy_score = 0.2
        
        return autonomy_score
    
    def _assess_transparency(self, capabilities: Dict[str, Any]) -> float:
        """Assess system transparency"""
        transparency_score = 1.0  # Start with worst case
        
        if capabilities.get("explainable", False):
            transparency_score -= 0.4
        
        if capabilities.get("interpretable", False):
            transparency_score -= 0.3
        
        if capabilities.get("auditable", False):
            transparency_score -= 0.3
        
        return max(transparency_score, 0.0)
    
    def _assess_reversibility(self, capabilities: Dict[str, Any]) -> float:
        """Assess decision reversibility"""
        if capabilities.get("irreversible_actions", False):
            return 1.0
        elif capabilities.get("difficult_to_reverse", False):
            return 0.6
        else:
            return 0.2
    
    def _generate_recommendations(self, risk_level: RiskLevel, scores: Dict[str, float]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]:
            recommendations.append("Implement human oversight mechanisms")
            recommendations.append("Enhance transparency and explainability")
            recommendations.append("Conduct regular audits")
        
        if scores.get("transparency", 0) > 0.5:
            recommendations.append("Improve model interpretability")
            recommendations.append("Provide clear decision explanations")
        
        if scores.get("autonomy", 0) > 0.7:
            recommendations.append("Add human approval for critical decisions")
            recommendations.append("Implement override mechanisms")
        
        return recommendations
    
    def _get_regulatory_requirements(self, risk_level: RiskLevel) -> Dict[str, List[str]]:
        """Get regulatory requirements based on risk level"""
        requirements = defaultdict(list)
        
        if risk_level == RiskLevel.UNACCEPTABLE:
            requirements["EU_AI_Act"].append("Prohibited - Article 5")
        
        elif risk_level == RiskLevel.HIGH:
            requirements["EU_AI_Act"].extend([
                "Conformity assessment required - Article 43",
                "Risk management system - Article 9",
                "Data governance - Article 10",
                "Technical documentation - Article 11",
                "Record-keeping - Article 12",
                "Transparency - Article 13",
                "Human oversight - Article 14",
                "Accuracy and robustness - Article 15"
            ])
        
        elif risk_level == RiskLevel.LIMITED:
            requirements["EU_AI_Act"].append("Transparency obligations - Article 52")
        
        return dict(requirements)


class ComplianceMonitor:
    """
    Continuous compliance monitoring
    Tracks and reports on governance compliance
    """
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Dict[str, Any]] = []
        self.compliance_history: deque = deque(maxlen=10000)
        
        logger.info("Compliance monitor initialized")
    
    async def monitor(self,
                     decision: GovernanceDecision,
                     outcome: Optional[Dict[str, Any]] = None):
        """Monitor governance decision and outcome"""
        # Record decision
        self.compliance_history.append({
            "decision": decision,
            "outcome": outcome,
            "timestamp": datetime.now()
        })
        
        # Update metrics
        self._update_metrics(decision, outcome)
        
        # Check for compliance issues
        await self._check_compliance_thresholds()
    
    def _update_metrics(self, decision: GovernanceDecision, outcome: Optional[Dict[str, Any]]):
        """Update compliance metrics"""
        # Decision metrics
        self.metrics["decisions_total"].append(1)
        self.metrics[f"decisions_{decision.action.value}"].append(1)
        self.metrics[f"risk_{decision.risk_level.value}"].append(1)
        
        # Compliance rate
        if decision.compliance_status == ComplianceStatus.COMPLIANT:
            self.metrics["compliance_rate"].append(1)
        else:
            self.metrics["compliance_rate"].append(0)
        
        # Violation tracking
        self.metrics["violations_total"].append(len(decision.violations))
        
        # Outcome tracking
        if outcome:
            if outcome.get("successful", False):
                self.metrics["successful_outcomes"].append(1)
            else:
                self.metrics["failed_outcomes"].append(1)
    
    async def _check_compliance_thresholds(self):
        """Check if compliance metrics exceed thresholds"""
        # Calculate recent compliance rate
        recent_compliance = list(self.metrics["compliance_rate"])[-100:]
        if recent_compliance:
            compliance_rate = sum(recent_compliance) / len(recent_compliance)
            
            if compliance_rate < 0.8:
                self.alerts.append({
                    "type": "low_compliance_rate",
                    "value": compliance_rate,
                    "threshold": 0.8,
                    "timestamp": datetime.now(),
                    "severity": "high"
                })
        
        # Check violation rate
        recent_violations = list(self.metrics["violations_total"])[-100:]
        if recent_violations:
            avg_violations = sum(recent_violations) / len(recent_violations)
            
            if avg_violations > 2:
                self.alerts.append({
                    "type": "high_violation_rate",
                    "value": avg_violations,
                    "threshold": 2,
                    "timestamp": datetime.now(),
                    "severity": "medium"
                })
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        # Calculate metrics
        total_decisions = len(self.compliance_history)
        
        if total_decisions == 0:
            return {"status": "no_data"}
        
        # Compliance rate
        compliant = sum(
            1 for record in self.compliance_history
            if record["decision"].compliance_status == ComplianceStatus.COMPLIANT
        )
        compliance_rate = compliant / total_decisions
        
        # Risk distribution
        risk_dist = defaultdict(int)
        for record in self.compliance_history:
            risk_dist[record["decision"].risk_level.value] += 1
        
        # Action distribution
        action_dist = defaultdict(int)
        for record in self.compliance_history:
            action_dist[record["decision"].action.value] += 1
        
        return {
            "total_decisions": total_decisions,
            "compliance_rate": compliance_rate,
            "risk_distribution": dict(risk_dist),
            "action_distribution": dict(action_dist),
            "recent_alerts": self.alerts[-10:],
            "timestamp": datetime.now()
        }


class GovernanceOrchestrator:
    """
    Main governance orchestrator
    Coordinates all governance components
    """
    
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.risk_assessment = RiskAssessment()
        self.compliance_monitor = ComplianceMonitor()
        
        # Audit trail
        self.audit_trail: List[GovernanceDecision] = []
        
        # Stakeholder management
        self.stakeholders: Dict[str, Dict[str, Any]] = {}
        self.feedback_queue: deque = deque(maxlen=1000)
        
        logger.info("Governance orchestrator initialized")
    
    async def evaluate_action(self,
                            action: str,
                            context: Dict[str, Any],
                            system_info: Optional[Dict[str, Any]] = None) -> GovernanceDecision:
        """Evaluate proposed action through governance framework"""
        decision_id = hashlib.sha256(
            f"{action}{json.dumps(context)}{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        # Policy evaluation
        policy_result = await self.policy_engine.evaluate(action, context)
        
        # Risk assessment
        risk_result = await self.risk_assessment.assess(
            system=context.get("system", "unknown"),
            capabilities=system_info or {},
            context=context
        )
        
        # Make governance decision
        if policy_result["compliance"] == ComplianceStatus.NON_COMPLIANT:
            governance_action = GovernanceAction.REJECT
        elif risk_result["risk_level"] == RiskLevel.UNACCEPTABLE:
            governance_action = GovernanceAction.REJECT
        elif risk_result["risk_level"] == RiskLevel.HIGH:
            governance_action = GovernanceAction.ESCALATE
        elif policy_result["compliance"] == ComplianceStatus.PARTIAL:
            governance_action = GovernanceAction.MONITOR
        else:
            governance_action = GovernanceAction.APPROVE
        
        # Create decision record
        decision = GovernanceDecision(
            decision_id=decision_id,
            action=governance_action,
            subject=action,
            context=context,
            risk_level=risk_result["risk_level"],
            compliance_status=policy_result["compliance"],
            confidence=self._calculate_confidence(policy_result, risk_result),
            applied_policies=policy_result["applied_policies"],
            violations=policy_result["violations"],
            reasoning=self._generate_reasoning(governance_action, policy_result, risk_result),
            evidence=[
                {"type": "policy_evaluation", "result": policy_result},
                {"type": "risk_assessment", "result": risk_result}
            ]
        )
        
        # Record decision
        self.audit_trail.append(decision)
        await self.compliance_monitor.monitor(decision)
        
        # Request stakeholder feedback if needed
        if governance_action in [GovernanceAction.ESCALATE, GovernanceAction.MONITOR]:
            await self._request_stakeholder_feedback(decision)
        
        return decision
    
    def _calculate_confidence(self, 
                            policy_result: Dict[str, Any],
                            risk_result: Dict[str, Any]) -> float:
        """Calculate confidence in governance decision"""
        # Base confidence
        confidence = 0.5
        
        # Adjust based on policy clarity
        if policy_result["compliance"] == ComplianceStatus.COMPLIANT:
            confidence += 0.3
        elif policy_result["compliance"] == ComplianceStatus.NON_COMPLIANT:
            confidence += 0.2
        
        # Adjust based on risk clarity
        if risk_result["total_score"] < 0.3 or risk_result["total_score"] > 0.7:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self,
                          action: GovernanceAction,
                          policy_result: Dict[str, Any],
                          risk_result: Dict[str, Any]) -> str:
        """Generate human-readable reasoning"""
        reasoning_parts = []
        
        # Action reasoning
        if action == GovernanceAction.REJECT:
            reasoning_parts.append(f"Action rejected due to ")
            if policy_result["violations"]:
                reasoning_parts.append(f"{len(policy_result['violations'])} policy violations")
            if risk_result["risk_level"] in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]:
                reasoning_parts.append(f"unacceptable risk level ({risk_result['risk_level'].value})")
        
        elif action == GovernanceAction.APPROVE:
            reasoning_parts.append("Action approved - complies with all policies and acceptable risk")
        
        elif action == GovernanceAction.ESCALATE:
            reasoning_parts.append(f"Action escalated due to high risk ({risk_result['risk_level'].value})")
        
        # Add recommendations
        if risk_result.get("recommendations"):
            reasoning_parts.append(". Recommendations: " + ", ".join(risk_result["recommendations"]))
        
        return " ".join(reasoning_parts)
    
    async def _request_stakeholder_feedback(self, decision: GovernanceDecision):
        """Request feedback from relevant stakeholders"""
        # Identify relevant stakeholders based on decision type
        relevant_stakeholders = self._identify_stakeholders(decision)
        
        for stakeholder_id in relevant_stakeholders:
            # In production, this would send actual notifications
            logger.info(f"Requesting feedback from stakeholder: {stakeholder_id}")
    
    def _identify_stakeholders(self, decision: GovernanceDecision) -> List[str]:
        """Identify relevant stakeholders for decision"""
        stakeholders = []
        
        # Always include ethics board for high-risk decisions
        if decision.risk_level in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]:
            stakeholders.append("ethics_board")
        
        # Include privacy officer for privacy-related decisions
        if any("privacy" in str(v) for v in decision.violations):
            stakeholders.append("privacy_officer")
        
        # Include legal for compliance issues
        if decision.compliance_status == ComplianceStatus.NON_COMPLIANT:
            stakeholders.append("legal_team")
        
        return stakeholders
    
    async def process_feedback(self, feedback: StakeholderFeedback):
        """Process stakeholder feedback"""
        self.feedback_queue.append(feedback)
        
        # Find original decision
        decision = next(
            (d for d in self.audit_trail if d.decision_id == feedback.decision_id),
            None
        )
        
        if decision and not feedback.agrees_with_decision:
            # Re-evaluate if significant disagreement
            if feedback.stakeholder_type in ["ethics_board", "legal_team"]:
                logger.warning(f"Stakeholder disagreement on decision {decision.decision_id}")
                # In production, would trigger re-evaluation
    
    def register_stakeholder(self, 
                           stakeholder_id: str,
                           stakeholder_type: str,
                           contact_info: Dict[str, str],
                           influence_weight: float = 1.0):
        """Register stakeholder for governance participation"""
        self.stakeholders[stakeholder_id] = {
            "type": stakeholder_type,
            "contact": contact_info,
            "weight": influence_weight,
            "registered_at": datetime.now()
        }
    
    def get_governance_metrics(self) -> Dict[str, Any]:
        """Get governance system metrics"""
        compliance_report = self.compliance_monitor.get_compliance_report()
        
        # Decision statistics
        total_decisions = len(self.audit_trail)
        if total_decisions > 0:
            action_stats = defaultdict(int)
            risk_stats = defaultdict(int)
            
            for decision in self.audit_trail:
                action_stats[decision.action.value] += 1
                risk_stats[decision.risk_level.value] += 1
            
            decision_stats = {
                "total": total_decisions,
                "by_action": dict(action_stats),
                "by_risk": dict(risk_stats)
            }
        else:
            decision_stats = {"total": 0}
        
        # Stakeholder engagement
        feedback_stats = {
            "total_feedback": len(self.feedback_queue),
            "agreement_rate": sum(1 for f in self.feedback_queue if f.agrees_with_decision) / max(len(self.feedback_queue), 1)
        }
        
        return {
            "compliance": compliance_report,
            "decisions": decision_stats,
            "stakeholder_feedback": feedback_stats,
            "active_stakeholders": len(self.stakeholders),
            "timestamp": datetime.now()
        }


# Example usage
async def demonstrate_governance():
    """Demonstrate AI governance system"""
    print("🏛️ AI Governance System Demonstration")
    print("=" * 60)
    
    # Initialize governance
    governance = GovernanceOrchestrator()
    
    # Register stakeholders
    governance.register_stakeholder(
        "ethics_board",
        "ethics_board",
        {"email": "ethics@company.com"},
        influence_weight=2.0
    )
    
    # Test Case 1: Safe action
    print("\n1️⃣ Testing safe action...")
    
    decision1 = await governance.evaluate_action(
        action="generate_report",
        context={
            "data": {"contains_pii": False},
            "purpose": "analytics",
            "system": "reporting_system"
        },
        system_info={
            "makes_decisions_about_individuals": False,
            "explainable": True,
            "human_in_loop": True
        }
    )
    
    print(f"Decision: {decision1.action.value}")
    print(f"Risk Level: {decision1.risk_level.value}")
    print(f"Compliance: {decision1.compliance_status.value}")
    print(f"Reasoning: {decision1.reasoning}")
    
    # Test Case 2: Privacy violation
    print("\n2️⃣ Testing privacy violation...")
    
    decision2 = await governance.evaluate_action(
        action="share_user_data",
        context={
            "data": {"contains_pii": True, "sensitive": True},
            "purpose": "marketing",
            "system": "data_processor"
        },
        system_info={
            "makes_decisions_about_individuals": True,
            "estimated_users": 50000
        }
    )
    
    print(f"Decision: {decision2.action.value}")
    print(f"Violations: {len(decision2.violations)}")
    if decision2.violations:
        print(f"First violation: {decision2.violations[0]['message']}")
    
    # Test Case 3: High-risk AI system
    print("\n3️⃣ Testing high-risk AI system...")
    
    decision3 = await governance.evaluate_action(
        action="automated_hiring_decision",
        context={
            "domain": "employment",
            "decision": {"affects_individuals": True},
            "system": "hr_ai"
        },
        system_info={
            "makes_decisions_about_individuals": True,
            "fully_autonomous": True,
            "explainable": False,
            "estimated_users": 100000
        }
    )
    
    print(f"Decision: {decision3.action.value}")
    print(f"Risk Level: {decision3.risk_level.value}")
    print(f"Recommendations: {decision3.evidence[1]['result']['recommendations']}")
    
    # Simulate stakeholder feedback
    print("\n4️⃣ Processing stakeholder feedback...")
    
    feedback = StakeholderFeedback(
        feedback_id="feedback_001",
        decision_id=decision3.decision_id,
        stakeholder_id="ethics_board",
        stakeholder_type="ethics_board",
        agrees_with_decision=False,
        confidence_in_decision=0.3,
        concerns=["Lack of human oversight", "No explainability"],
        suggestions=["Add human review", "Implement XAI methods"]
    )
    
    await governance.process_feedback(feedback)
    
    # Get governance metrics
    print("\n5️⃣ Governance Metrics")
    print("-" * 40)
    
    metrics = governance.get_governance_metrics()
    print(f"Total decisions: {metrics['decisions']['total']}")
    print(f"Compliance rate: {metrics['compliance'].get('compliance_rate', 0):.1%}")
    print(f"Stakeholder agreement: {metrics['stakeholder_feedback']['agreement_rate']:.1%}")
    
    print("\n✅ Governance demonstration complete")


if __name__ == "__main__":
    asyncio.run(demonstrate_governance())