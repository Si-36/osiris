"""
AURA TDA Module - Agent Topology Analysis
========================================

Production-grade topological data analysis for multi-agent systems.
Focused on practical workflow analysis, bottleneck detection, and
failure prediction.

Key Components:
- agent_topology: Workflow and communication analysis
- algorithms: Core TDA computations  
- realtime_monitor: Streaming topology monitoring
"""

import warnings
from typing import Any, Dict, Optional

# Core imports - new structure
from .agent_topology import (
    AgentTopologyAnalyzer,
    WorkflowFeatures,
    CommunicationFeatures,
    TopologicalAnomaly,
    HealthStatus
)

from .algorithms import (
    PersistenceDiagram,
    compute_persistence,
    diagram_entropy,
    diagram_distance,
    vectorize_diagram,
    validate_point_cloud
)

from .realtime_monitor import (
    RealtimeTopologyMonitor,
    SystemEvent,
    EventType,
    EventAdapter,
    create_monitor
)

# Legacy compatibility with deprecation warnings
def __getattr__(name: str) -> Any:
    """Provide legacy compatibility with deprecation warnings."""
    
    # Map old names to new ones
    legacy_map = {
        # From various engines
        "UnifiedTDAEngine2025": AgentTopologyAnalyzer,
        "UnifiedTDAEngine": AgentTopologyAnalyzer,
        "RealTDAEngine": AgentTopologyAnalyzer,
        "AdvancedTDAEngine": AgentTopologyAnalyzer,
        "ProductionTDAEngine": AgentTopologyAnalyzer,
        
        # From algorithms
        "RipsComplex": "compute_persistence",
        "PersistentHomology": "compute_persistence",
        
        # From models
        "TDARequest": "WorkflowFeatures",
        "TDAResponse": "WorkflowFeatures",
        "TDAMetrics": "WorkflowFeatures",
        
        # Functions
        "get_unified_tda_engine": "AgentTopologyAnalyzer",
        "get_real_tda": "AgentTopologyAnalyzer"
    }
    
    if name in legacy_map:
        warnings.warn(
            f"'{name}' is deprecated and will be removed in v2.0. "
            f"Use '{legacy_map[name]}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Return the new implementation
        new_name = legacy_map[name]
        if new_name == "AgentTopologyAnalyzer":
            return AgentTopologyAnalyzer
        elif new_name == "compute_persistence":
            return compute_persistence
        elif new_name == "WorkflowFeatures":
            return WorkflowFeatures
        else:
            return globals().get(new_name)
            
    # Check if it's a request for old engine instance
    if name in ["unified_tda_engine", "real_tda_engine"]:
        warnings.warn(
            f"Global '{name}' instance is deprecated. "
            f"Create your own AgentTopologyAnalyzer instance.",
            DeprecationWarning,
            stacklevel=2
        )
        # Return a default instance
        return AgentTopologyAnalyzer()
        
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Convenience functions for common use cases
async def analyze_agent_workflow(workflow_id: str, 
                               workflow_data: Dict[str, Any]) -> WorkflowFeatures:
    """
    Analyze a workflow for bottlenecks and risks.
    
    Args:
        workflow_id: Unique workflow identifier
        workflow_data: Dict with 'agents' and 'dependencies'
        
    Returns:
        WorkflowFeatures with analysis results
    """
    analyzer = AgentTopologyAnalyzer()
    return await analyzer.analyze_workflow(workflow_id, workflow_data)


async def analyze_agent_communications(communication_data: Dict[str, Any]) -> CommunicationFeatures:
    """
    Analyze agent communication patterns.
    
    Args:
        communication_data: Dict with 'agents' and 'messages'
        
    Returns:
        CommunicationFeatures with network analysis
    """
    analyzer = AgentTopologyAnalyzer()
    return await analyzer.analyze_communications(communication_data)


def create_topology_analyzer(config: Optional[Dict[str, Any]] = None) -> AgentTopologyAnalyzer:
    """
    Create a configured topology analyzer.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Configured AgentTopologyAnalyzer instance
    """
    return AgentTopologyAnalyzer(config)


# Main public API
__all__ = [
    # Core analyzer
    "AgentTopologyAnalyzer",
    
    # Data structures
    "WorkflowFeatures",
    "CommunicationFeatures", 
    "TopologicalAnomaly",
    "HealthStatus",
    "PersistenceDiagram",
    
    # Algorithms
    "compute_persistence",
    "diagram_entropy",
    "diagram_distance",
    "vectorize_diagram",
    "validate_point_cloud",
    
    # Real-time monitoring
    "RealtimeTopologyMonitor",
    "SystemEvent",
    "EventType",
    "EventAdapter",
    "create_monitor",
    
    # Convenience functions
    "analyze_agent_workflow",
    "analyze_agent_communications",
    "create_topology_analyzer"
]

# Version info
__version__ = "2.0.0"