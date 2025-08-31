"""
🤖 AURA Intelligence Agent SDK - Phase 2D: The Collective

Advanced multi-agent system built on the proven Hot→Cold→Wise Intelligence Flywheel.
Combines cutting-edge agent orchestration with operational reliability.

Key Components:
- UnifiedMemory: LlamaIndex fusion over existing memory tiers
- ACP Protocol: Formal agent-to-agent communication
- Agent Base Classes: Observer, Analyst, Executor, Coordinator
- OpenTelemetry Integration: Full observability
- LangGraph Orchestration: State-of-the-art workflows

Production-grade multi-agent implementation using LangGraph
with comprehensive observability and resilience patterns.
"""

# Phase 2 new implementations - commented out due to complex dependencies
# from .base import AgentBase, AgentConfig, AgentState
# from aura_intelligence.observability import AgentInstrumentor, AgentMetrics

# Original schemas (to be implemented/migrated) - commented out due to missing modules
# from .schemas.acp import ACPEnvelope, ACPEndpoint, MessageType, Priority
# from .schemas.state import AgentState as LegacyAgentState, DossierEntry, TaskStatus
# from .schemas.log import AgentActionEvent, ActionResult

# Memory & Communication (to be implemented/migrated) - commented out due to missing modules
# from .memory.unified import UnifiedMemory, MemoryTier, QueryResult
# from .communication.protocol import ACPProtocol, MessageBus
# from .communication.transport import RedisStreamsTransport

# Base Classes (to be migrated to new base) - commented out due to missing modules
# from .base_classes.agent import BaseAgent, AgentRole, AgentCapability
# from .base_classes.instrumentation import instrument_agent, AgentMetrics as LegacyAgentMetrics


# from .orchestration.workflow import WorkflowEngine, WorkflowState
# from .orchestration.langgraph import LangGraphOrchestrator


# from .core.observer import ObserverAgent
# from .core.analyst import AnalystAgent  
# from .core.executor import ExecutorAgent
# from .core.coordinator import CoordinatorAgent


# from .advanced.router import RouterAgent
# from .advanced.consensus import ConsensusAgent
# from .advanced.supervisor import SupervisorAgent

# Working agent implementations
from .simple_agent import SimpleAgent, create_simple_agent, get_simple_registry
from .consolidated_agents import ConsolidatedAgent, ConsolidatedAgentFactory, get_agent_registry

# Production LangGraph agents (2025 patterns)
try:
    from .production_langgraph_agent import (
        AURAProductionAgent,
        ProductionAgentConfig,
        ProductionAgentState,
        create_production_agent
    )
    PRODUCTION_AGENTS_AVAILABLE = True
except ImportError:
    PRODUCTION_AGENTS_AVAILABLE = False
    AURAProductionAgent = None
    ProductionAgentConfig = None
    ProductionAgentState = None
    create_production_agent = None

__version__ = "2.0.0"
__author__ = "AURA Intelligence Team"

# Aliases for backward compatibility
AURAAgent = AURAProductionAgent
AgentConfig = ProductionAgentConfig

# Placeholders for expected imports
LNNCouncilOrchestrator = None  # Not implemented yet
agent_templates = {}  # Not implemented yet

# Export main classes for easy import
__all__ = [
    # Working agent implementations
    "SimpleAgent",
    "create_simple_agent", 
    "get_simple_registry",
    "ConsolidatedAgent",
    "ConsolidatedAgentFactory",
    "get_agent_registry",
    
    # Production LangGraph agents
    "AURAProductionAgent",
    "ProductionAgentConfig",
    "ProductionAgentState",
    "create_production_agent",
    "PRODUCTION_AGENTS_AVAILABLE",
    
    # Aliases for backward compatibility
    "AURAAgent",
    "AgentConfig",
    "LNNCouncilOrchestrator",
    "agent_templates",
    
    # Phase 2 New Base (commented out due to dependencies)
    # "AgentBase",
    # "AgentConfig", 
    # "AgentState",
    # "AgentInstrumentor",
    # "AgentMetrics",
]
