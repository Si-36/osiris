#!/usr/bin/env python3
"""
Create a clean, working supervisor.py based on AURA's requirements
Using the Memory-Aware Supervisor pattern from the original
"""

supervisor_code = '''"""
🧠 Memory-Aware Supervisor Agent - AURA Intelligence System
Transforms reactive decision-making into reflective, learning-based choices.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

# Try imports with fallbacks
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available - using mock implementation")

try:
    from ..observability.knowledge_graph import KnowledgeGraphManager
    from ..observability.config import ObservabilityConfig
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    # Mock classes for testing
    class KnowledgeGraphManager:
        async def get_historical_context(self, evidence: List[Any]) -> List[Dict]:
            """Mock implementation returning empty context"""
            return []
    
    class ObservabilityConfig:
        """Mock config"""
        pass


# Enhanced system prompt with historical context integration
SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor of a collective of AI agents. Your role is to analyze the current system state and decide the next best action.

## Core Directives
1. Analyze the full Evidence Log.
2. Review the Historical Context of similar past situations.
3. Based on all available information, decide the next agent to call or conclude the workflow.

## Evidence Log
{evidence_log}

## Historical Context (Memory of Past Successes)
Here are summaries of similar past workflows that completed successfully. Use this to inform your decision.
{historical_context}

## Your Task
Based on the evidence and historical context, what is the next single action to take? Choose from the available tools: {tool_names}.
If the goal is complete, respond with "FINISH".

Provide your reasoning in this format:
REASONING: [Brief explanation of how evidence and memory inform your decision]
ACTION: [chosen tool or FINISH]
"""


class CollectiveState:
    """Simple state container for workflow data."""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data or {}
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key: str):
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any):
        self.data[key] = value


class Supervisor:
    """
    Memory-aware supervisor implementing the learning loop 'read' phase.
    Makes decisions based on current evidence AND historical context.
    
    This is the core orchestrator for AURA's multi-agent system.
    """
    
    def __init__(self, llm: Optional[Any] = None, tools: List[str] = None):
        """
        Initialize memory-aware supervisor.
        
        Args:
            llm: Language model for decision making (optional)
            tools: List of available tool names
        """
        self.llm = llm
        self.tool_names = tools or []
        self.system_prompt = SUPERVISOR_SYSTEM_PROMPT
        self.decision_history = []
        
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data and return results - wrapper for compatibility.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Analysis results with next action
        """
        # Create a simple state for analysis
        state = CollectiveState({"evidence_log": [data]})
        
        # Use a mock KG manager if not available
        kg_manager = KnowledgeGraphManager()
        
        result = await self.invoke(state, kg_manager)
        
        return {
            "analysis": "completed",
            "next_action": result.get("next", "FINISH"),
            "reasoning": result.get("reasoning", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def invoke(self, state: CollectiveState, kg_manager: KnowledgeGraphManager) -> Dict[str, Any]:
        """
        The new, memory-aware invocation logic for the Supervisor.
        
        Args:
            state: Current workflow state with evidence log
            kg_manager: Knowledge graph manager for memory retrieval
            
        Returns:
            Dict containing the supervisor's decision and reasoning
        """
        try:
            # 1. Retrieve historical context from the knowledge graph
            current_evidence = state.get('evidence_log', [])
            historical_context = await kg_manager.get_historical_context(current_evidence)
            
            # 2. Format the context for the prompt
            formatted_history = self._format_historical_context(historical_context)
            
            # 3. Format the enhanced prompt
            evidence_json = json.dumps(self._convert_numpy(current_evidence), indent=2)
            
            prompt = self.system_prompt.format(
                evidence_log=evidence_json,
                historical_context=formatted_history,
                tool_names=", ".join(self.tool_names)
            )
            
            # 4. Make decision
            if self.llm and LANGCHAIN_AVAILABLE:
                # Use real LLM if available
                messages = [
                    SystemMessage(content=prompt),
                    HumanMessage(content="What is the next action?")
                ]
                response = await self.llm.ainvoke(messages)
                decision = self._parse_response(response.content)
            else:
                # Mock decision for testing
                decision = self._make_mock_decision(current_evidence)
            
            # 5. Store decision in history
            self.decision_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "evidence": current_evidence,
                "decision": decision,
                "historical_context": historical_context
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            return {"next": "FINISH", "reasoning": f"Error occurred: {str(e)}"}
    
    def _convert_numpy(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(v) for v in obj]
        return obj
    
    def _format_historical_context(self, context: List[Dict]) -> str:
        """Format historical context for the prompt."""
        if not context:
            return "No historical context available yet."
        
        formatted = []
        for i, ctx in enumerate(context[:5]):  # Limit to 5 most relevant
            formatted.append(f"Context {i+1}:")
            formatted.append(f"  - Situation: {ctx.get('situation', 'Unknown')}")
            formatted.append(f"  - Action Taken: {ctx.get('action', 'Unknown')}")
            formatted.append(f"  - Outcome: {ctx.get('outcome', 'Unknown')}")
            formatted.append("")
        
        return "\\n".join(formatted)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract reasoning and action."""
        lines = response.strip().split('\\n')
        reasoning = ""
        action = "FINISH"
        
        for line in lines:
            if line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()
        
        return {
            "next": action,
            "reasoning": reasoning
        }
    
    def _make_mock_decision(self, evidence: List[Any]) -> Dict[str, Any]:
        """Make a mock decision for testing without LLM."""
        # Simple logic: if we have evidence, pick a tool; otherwise finish
        if evidence and self.tool_names:
            return {
                "next": self.tool_names[0],
                "reasoning": "Mock decision: Processing evidence with first available tool"
            }
        return {
            "next": "FINISH",
            "reasoning": "Mock decision: No evidence to process or no tools available"
        }


# Example usage and testing
async def test_supervisor():
    """Test the supervisor with mock data."""
    print("🧪 Testing Memory-Aware Supervisor...")
    
    # Create supervisor with mock tools
    supervisor = Supervisor(
        llm=None,  # Will use mock
        tools=["analyzer", "validator", "executor"]
    )
    
    # Test with sample data
    result = await supervisor.analyze({
        "task": "Analyze system performance",
        "metrics": {"cpu": 75, "memory": 60},
        "timestamp": datetime.utcnow().isoformat()
    })
    
    print(f"✅ Analysis Result: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    # Run test
    asyncio.run(test_supervisor())
'''

# Write the clean supervisor
with open('supervisor_clean.py', 'w') as f:
    f.write(supervisor_code)

print("✅ Created supervisor_clean.py")
print("\nFeatures:")
print("- Memory-aware decision making")
print("- Historical context integration")
print("- Mock fallbacks for testing")
print("- Proper async/await patterns")
print("- Error handling")
print("- Compatible with AURA's architecture")