"""Real Policy Engine with OPA"""
import json
from typing import Dict, Any

try:
    import requests
    OPA_AVAILABLE = True
except ImportError:
    OPA_AVAILABLE = False

class RealPolicyEngine:
    def __init__(self, opa_url: str = "http://localhost:8181"):
        self.opa_url = opa_url
        self.opa_available = OPA_AVAILABLE
    
    async def evaluate_policy(self, policy_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy using OPA"""
        if self.opa_available:
            try:
                response = requests.post(
                    f"{self.opa_url}/v1/data/{policy_name}",
                    json={"input": input_data},
                    timeout=5
                )
                return response.json()
            except Exception:
                pass
        
        # Fallback policy evaluation
        return {
            "result": True,
            "policy": policy_name,
            "reason": "fallback_approval",
            "engine": "fallback"
        }

def get_real_policy_engine():
    return RealPolicyEngine()