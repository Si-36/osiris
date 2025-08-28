"""
🎼 Orchestration Module - LangGraph Workflow Management

Professional LangGraph orchestration for collective intelligence.
Built on your proven schema foundation.
"""

# from .workflows import CollectiveWorkflow  
from .checkpoints import WorkflowCheckpointManager

try:
    from .langgraph_workflows import AURACollectiveIntelligence, AgentState
    LANGGRAPH_WORKFLOWS_AVAILABLE = True
except ImportError:
    LANGGRAPH_WORKFLOWS_AVAILABLE = False
    AURACollectiveIntelligence = None
    AgentState = None

__all__ = [
    # "CollectiveWorkflow",  
    "WorkflowCheckpointManager",
    "AURACollectiveIntelligence",
    "AgentState"
]
