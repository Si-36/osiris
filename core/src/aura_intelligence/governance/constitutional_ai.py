"""
Constitutional AI for AURA - Enterprise Safety Layer
Prevents harmful decisions through built-in ethical constraints
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ViolationType(Enum):
    HARM_TO_HUMANS = "harm_to_humans"
    PRIVACY_VIOLATION = "privacy_violation"
    BIAS_DISCRIMINATION = "bias_discrimination"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_MISUSE = "data_misuse"

@dataclass
class ConstitutionalRule:
    rule_id: str
    description: str
    violation_type: ViolationType
    severity: int  # 1-10
    auto_block: bool = True

class AURAConstitution:
    """AURA's built-in ethical framework"""
    
    def __init__(self):
        self.rules = [
            ConstitutionalRule(
                "no_harm", 
                "Never recommend actions that could harm humans",
                ViolationType.HARM_TO_HUMANS, 
                10, 
                True
            ),
            ConstitutionalRule(
                "privacy_protection",
                "Never expose personal data without consent", 
                ViolationType.PRIVACY_VIOLATION,
                9,
                True
            ),
            ConstitutionalRule(
                "no_bias",
                "Decisions must be fair across all demographics",
                ViolationType.BIAS_DISCRIMINATION,
                8,
                True
            ),
            ConstitutionalRule(
                "authorized_access_only",
                "Only access data user has permission for",
                ViolationType.UNAUTHORIZED_ACCESS,
                9,
                True
            )
        ]

class ConstitutionalAI:
    """Enterprise safety layer for AURA decisions"""
    
    def __init__(self):
        self.constitution = AURAConstitution()
        self.violation_history = []
    
    async def validate_decision(self, 
                              topology_context: Dict[str, Any],
                              proposed_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate decision against constitutional rules"""
        
        violations = []
        
        # Check each constitutional rule
        for rule in self.constitution.rules:
            violation = await self._check_rule(rule, topology_context, proposed_decision)
            if violation:
                violations.append(violation)
        
        # Determine if decision should be blocked
        should_block = any(v['auto_block'] for v in violations)
        max_severity = max([v['severity'] for v in violations], default=0)
        
        result = {
            'approved': not should_block,
            'violations': violations,
            'max_severity': max_severity,
            'explanation': self._generate_explanation(violations)
        }
        
        # Log for audit
        self.violation_history.append({
            'decision': proposed_decision,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    async def _check_rule(self, rule: ConstitutionalRule, 
                         context: Dict[str, Any], 
                         decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check specific constitutional rule"""
        
        if rule.violation_type == ViolationType.HARM_TO_HUMANS:
            return await self._check_harm(rule, context, decision)
        elif rule.violation_type == ViolationType.PRIVACY_VIOLATION:
            return await self._check_privacy(rule, context, decision)
        elif rule.violation_type == ViolationType.BIAS_DISCRIMINATION:
            return await self._check_bias(rule, context, decision)
        elif rule.violation_type == ViolationType.UNAUTHORIZED_ACCESS:
            return await self._check_access(rule, context, decision)
        
        return None
    
    async def _check_harm(self, rule: ConstitutionalRule, 
                         context: Dict[str, Any], 
                         decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for potential harm to humans"""
        
        # Look for harmful keywords in decision
        harmful_indicators = [
            'delete', 'remove', 'terminate', 'shutdown', 'disable',
            'block', 'deny', 'reject', 'cancel', 'stop'
        ]
        
        decision_text = str(decision).lower()
        
        # Check if decision affects critical systems
        if any(indicator in decision_text for indicator in harmful_indicators):
            if 'critical' in decision_text or 'emergency' in decision_text:
                return {
                    'rule_id': rule.rule_id,
                    'violation_type': rule.violation_type.value,
                    'severity': rule.severity,
                    'auto_block': rule.auto_block,
                    'reason': 'Decision may affect critical systems'
                }
        
        return None
    
    async def _check_privacy(self, rule: ConstitutionalRule,
                           context: Dict[str, Any],
                           decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for privacy violations"""
        
        # Look for personal data exposure
        privacy_indicators = ['email', 'phone', 'address', 'ssn', 'id', 'personal']
        decision_text = str(decision).lower()
        
        if any(indicator in decision_text for indicator in privacy_indicators):
            # Check if user consented
            user_consent = context.get('user_consent', False)
            if not user_consent:
                return {
                    'rule_id': rule.rule_id,
                    'violation_type': rule.violation_type.value,
                    'severity': rule.severity,
                    'auto_block': rule.auto_block,
                    'reason': 'Personal data access without consent'
                }
        
        return None
    
    async def _check_bias(self, rule: ConstitutionalRule,
                         context: Dict[str, Any],
                         decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for discriminatory bias"""
        
        # Simple bias detection - in production use ML model
        bias_indicators = ['race', 'gender', 'age', 'religion', 'nationality']
        decision_text = str(decision).lower()
        
        if any(indicator in decision_text for indicator in bias_indicators):
            return {
                'rule_id': rule.rule_id,
                'violation_type': rule.violation_type.value,
                'severity': rule.severity,
                'auto_block': False,  # Flag for review, don't auto-block
                'reason': 'Decision may contain demographic bias'
            }
        
        return None
    
    async def _check_access(self, rule: ConstitutionalRule,
                          context: Dict[str, Any],
                          decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for unauthorized access"""
        
        # Check if decision requires permissions user doesn't have
        required_permissions = decision.get('required_permissions', [])
        user_permissions = context.get('user_permissions', [])
        
        missing_permissions = set(required_permissions) - set(user_permissions)
        
        if missing_permissions:
            return {
                'rule_id': rule.rule_id,
                'violation_type': rule.violation_type.value,
                'severity': rule.severity,
                'auto_block': rule.auto_block,
                'reason': f'Missing permissions: {list(missing_permissions)}'
            }
        
        return None
    
    def _generate_explanation(self, violations: List[Dict[str, Any]]) -> str:
        """Generate human-readable explanation"""
        
        if not violations:
            return "Decision approved - no constitutional violations detected"
        
        explanations = []
        for violation in violations:
            explanations.append(f"Violation: {violation['reason']} (Severity: {violation['severity']})")
        
        return "; ".join(explanations)
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get violation statistics for monitoring"""
        
        if not self.violation_history:
            return {'total_decisions': 0, 'violations': 0, 'block_rate': 0.0}
        
        total = len(self.violation_history)
        blocked = sum(1 for h in self.violation_history if not h['result']['approved'])
        
        return {
            'total_decisions': total,
            'violations': blocked,
            'block_rate': blocked / total,
            'avg_severity': sum(h['result']['max_severity'] for h in self.violation_history) / total
        }

# Integration with existing council system
class ConstitutionalCouncilAgent:
    """Council agent with constitutional AI integration"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.constitutional_ai = ConstitutionalAI()
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision with constitutional validation"""
        
        # Original decision logic (your existing code)
        raw_decision = await self._raw_decision_logic(context)
        
        # Constitutional validation
        validation = await self.constitutional_ai.validate_decision(context, raw_decision)
        
        if validation['approved']:
            return {
                'decision': raw_decision,
                'constitutional_status': 'approved',
                'explanation': validation['explanation']
            }
        else:
            return {
                'decision': 'blocked',
                'constitutional_status': 'blocked',
                'violations': validation['violations'],
                'explanation': validation['explanation']
            }
    
    async def _raw_decision_logic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Your existing decision logic"""
        return {'action': 'approve', 'confidence': 0.8}

# Global instance
_constitutional_ai = None

def get_constitutional_ai():
    global _constitutional_ai
    if _constitutional_ai is None:
        _constitutional_ai = ConstitutionalAI()
    return _constitutional_ai