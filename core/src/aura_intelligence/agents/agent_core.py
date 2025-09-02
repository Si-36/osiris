"""
🎯 AURA Agent Core - Unified Base for All Agents

This is the foundation that automatically integrates our 4 core components:
- Memory System (topological + hardware-aware)
- TDA Analysis (workflow topology)
- Neural Router (intelligent model selection)
- Orchestration Engine (workflow management)

Every AURA agent inherits from this to get automatic:
- State persistence with PostgreSQL
- Memory storage and retrieval
- Model routing optimization
- Workflow orchestration
- Monitoring and metrics
"""

import asyncio
from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import uuid
import structlog

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.postgres import PostgresSaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import BaseTool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    BaseMessage = dict  # Fallback type
    HumanMessage = dict
    AIMessage = dict
    SystemMessage = dict
    BaseTool = object
    add_messages = lambda x: x
    ToolNode = None
    PostgresSaver = None
    print("Warning: LangGraph not available. Using mock implementation.")

# AURA component imports
try:
    from ..memory.core.memory_api import AURAMemorySystem
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    AURAMemorySystem = None

try:
    from ..tda.agent_topology import AgentTopologyAnalyzer
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    AgentTopologyAnalyzer = None

try:
    from ..neural.model_router import AURAModelRouter
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    AURAModelRouter = None

try:
    from ..orchestration.unified_orchestration_engine import UnifiedOrchestrationEngine
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    UnifiedOrchestrationEngine = None

logger = structlog.get_logger()


# ======================
# Agent State Definition
# ======================

@dataclass
class AURAAgentState:
    """
    Unified agent state that all AURA agents use.
    Includes integration points for all 4 core components.
    """
    # Core state
    agent_id: str
    agent_type: str
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Messages (for LangGraph)
    messages: List[BaseMessage] = field(default_factory=list)
    
    # Component states
    memory_context: Dict[str, Any] = field(default_factory=dict)
    topology_context: Dict[str, Any] = field(default_factory=dict)
    routing_context: Dict[str, Any] = field(default_factory=dict)
    workflow_context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    current_task: Optional[str] = None
    task_status: str = "idle"  # idle, running, completed, failed
    last_action: Optional[str] = None
    
    # Metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actions_taken: int = 0
    tokens_used: int = 0
    cost_incurred: float = 0.0
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ======================
# Base Agent Class
# ======================

class AURAAgentCore(ABC):
    """
    Base class for all AURA agents.
    
    Automatically integrates:
    - Memory system for context and learning
    - TDA for workflow analysis
    - Neural router for model selection
    - Orchestration for workflow management
    
    Subclasses need to implement:
    - agent_type: Type of agent (observer, analyst, executor, coordinator)
    - build_tools(): Tools specific to this agent
    - process_task(): Core task processing logic
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize AURA agent with all components"""
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or {}
        
        # Initialize core components
        self._init_components()
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        self.workflow = self.graph.compile(checkpointer=self._get_checkpointer())
        
        # Agent state
        self.state = AURAAgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )
        
        # Metrics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        
        logger.info(
            f"Initialized {self.agent_type} agent",
            agent_id=self.agent_id,
            components={
                "memory": MEMORY_AVAILABLE,
                "tda": TDA_AVAILABLE,
                "neural": NEURAL_AVAILABLE,
                "orchestration": ORCHESTRATION_AVAILABLE
            }
        )
    
    def _init_components(self):
        """Initialize AURA components with graceful fallbacks"""
        
        # Memory System
        if MEMORY_AVAILABLE:
            self.memory = AURAMemorySystem()
            logger.info("✓ Memory system initialized")
        else:
            self.memory = None
            logger.warning("✗ Memory system not available")
            
        # TDA Analyzer
        if TDA_AVAILABLE:
            self.tda = AgentTopologyAnalyzer()
            logger.info("✓ TDA analyzer initialized")
        else:
            self.tda = None
            logger.warning("✗ TDA analyzer not available")
            
        # Neural Router
        if NEURAL_AVAILABLE:
            router_config = self.config.get("neural_router", {})
            self.router = AURAModelRouter(router_config)
            logger.info("✓ Neural router initialized")
        else:
            self.router = None
            logger.warning("✗ Neural router not available")
            
        # Orchestration Engine
        if ORCHESTRATION_AVAILABLE:
            self.orchestrator = UnifiedOrchestrationEngine()
            logger.info("✓ Orchestration engine initialized")
        else:
            self.orchestrator = None
            logger.warning("✗ Orchestration engine not available")
    
    def _get_checkpointer(self):
        """Get checkpointer for state persistence"""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        # Use PostgreSQL for production persistence
        checkpoint_config = self.config.get("checkpoint", {})
        
        if checkpoint_config.get("type") == "postgres":
            try:
                return PostgresSaver.from_conn_string(
                    checkpoint_config.get("connection_string", "")
                )
            except Exception as e:
                logger.warning(f"Failed to create PostgresSaver: {e}")
                
        # Fallback to memory
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    
    def _build_graph(self) -> Optional[StateGraph]:
        """Build the LangGraph workflow"""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        # Create state graph
        workflow = StateGraph(AURAAgentState)
        
        # Add nodes
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("decide", self._decide_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("reflect", self._reflect_node)
        
        # Add tools node if agent has tools
        tools = self.build_tools()
        if tools:
            workflow.add_node("tools", ToolNode(tools))
            
        # Add edges (standard flow)
        workflow.set_entry_point("observe")
        workflow.add_edge("observe", "analyze")
        workflow.add_edge("analyze", "decide")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "decide",
            self._route_decision,
            {
                "execute": "execute",
                "tools": "tools",
                "reflect": "reflect",
                "end": END
            }
        )
        
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("tools", "reflect")
        workflow.add_edge("reflect", END)
        
        return workflow
    
    # ======================
    # Workflow Nodes
    # ======================
    
    async def _observe_node(self, state: AURAAgentState) -> Dict[str, Any]:
        """Observe and gather context"""
        logger.debug(f"Agent {self.agent_id} observing")
        
        updates = {}
        
        # Get memory context
        if self.memory:
            try:
                relevant_memories = await self.memory.retrieve(
                    query=state.current_task or "general context",
                    limit=5
                )
                updates["memory_context"] = {
                    "memories": relevant_memories,
                    "memory_count": len(relevant_memories)
                }
            except Exception as e:
                logger.error(f"Memory retrieval failed: {e}")
                
        # Analyze topology
        if self.tda and state.messages:
            try:
                topology = await self.tda.analyze_conversation_flow(
                    messages=[m.content for m in state.messages[-10:]]
                )
                updates["topology_context"] = {
                    "complexity": topology.get("complexity", 0),
                    "patterns": topology.get("patterns", [])
                }
            except Exception as e:
                logger.error(f"TDA analysis failed: {e}")
                
        return updates
    
    async def _analyze_node(self, state: AURAAgentState) -> Dict[str, Any]:
        """Analyze the situation"""
        logger.debug(f"Agent {self.agent_id} analyzing")
        
        # Delegate to subclass implementation
        analysis = await self.analyze_task(state)
        
        return {"results": {"analysis": analysis}}
    
    async def _decide_node(self, state: AURAAgentState) -> Dict[str, Any]:
        """Make decision on action"""
        logger.debug(f"Agent {self.agent_id} deciding")
        
        # Use neural router for model selection if needed
        if self.router and state.current_task:
            try:
                routing_decision = await self.router.route_request({
                    "prompt": state.current_task,
                    "context": state.memory_context
                })
                state.routing_context = {
                    "selected_model": routing_decision.model_config.model_id,
                    "provider": routing_decision.provider.value,
                    "confidence": routing_decision.confidence
                }
            except Exception as e:
                logger.error(f"Routing failed: {e}")
                
        # Delegate to subclass
        decision = await self.make_decision(state)
        
        return {
            "last_action": decision.get("action", "none"),
            "results": {"decision": decision}
        }
    
    async def _execute_node(self, state: AURAAgentState) -> Dict[str, Any]:
        """Execute the decided action"""
        logger.debug(f"Agent {self.agent_id} executing")
        
        # Track execution
        state.actions_taken += 1
        
        # Delegate to subclass
        result = await self.execute_action(state)
        
        return {"results": {"execution": result}}
    
    async def _reflect_node(self, state: AURAAgentState) -> Dict[str, Any]:
        """Reflect on results and update memory"""
        logger.debug(f"Agent {self.agent_id} reflecting")
        
        # Store in memory if available
        if self.memory and state.results:
            try:
                await self.memory.store({
                    "agent_id": self.agent_id,
                    "task": state.current_task,
                    "results": state.results,
                    "timestamp": datetime.now(timezone.utc),
                    "topology": state.topology_context
                })
            except Exception as e:
                logger.error(f"Memory storage failed: {e}")
                
        # Update metrics
        if not state.errors:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
            
        return {"task_status": "completed"}
    
    def _route_decision(self, state: AURAAgentState) -> str:
        """Route based on decision"""
        action = state.last_action
        
        if action == "use_tools":
            return "tools"
        elif action == "execute":
            return "execute"
        elif action == "complete":
            return "end"
        else:
            return "reflect"
    
    # ======================
    # Public Interface
    # ======================
    
    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run agent on a task"""
        self.total_tasks += 1
        
        # Initialize state
        self.state = AURAAgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            thread_id=thread_id or str(uuid.uuid4()),
            current_task=task,
            task_status="running",
            start_time=datetime.now(timezone.utc),
            messages=[HumanMessage(content=task)]
        )
        
        # Add context
        if context:
            self.state.workflow_context = context
            
        try:
            # Run workflow
            if self.workflow:
                final_state = await self.workflow.ainvoke(
                    self.state,
                    {"thread_id": self.state.thread_id}
                )
            else:
                # Fallback for no LangGraph
                await self._observe_node(self.state)
                await self._analyze_node(self.state)
                await self._decide_node(self.state)
                await self._execute_node(self.state)
                await self._reflect_node(self.state)
                final_state = self.state
                
            # Extract results
            return {
                "success": len(final_state.errors) == 0,
                "results": final_state.results,
                "errors": final_state.errors,
                "metrics": {
                    "actions_taken": final_state.actions_taken,
                    "tokens_used": final_state.tokens_used,
                    "cost_incurred": final_state.cost_incurred,
                    "duration": (
                        datetime.now(timezone.utc) - final_state.start_time
                    ).total_seconds() if final_state.start_time else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            self.failed_tasks += 1
            return {
                "success": False,
                "error": str(e),
                "results": {},
                "metrics": {}
            }
    
    # ======================
    # Abstract Methods
    # ======================
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type of agent (observer, analyst, executor, coordinator)"""
        pass
    
    @abstractmethod
    def build_tools(self) -> List[BaseTool]:
        """Build tools specific to this agent type"""
        pass
    
    @abstractmethod
    async def analyze_task(self, state: AURAAgentState) -> Dict[str, Any]:
        """Analyze the current task"""
        pass
    
    @abstractmethod
    async def make_decision(self, state: AURAAgentState) -> Dict[str, Any]:
        """Make a decision based on analysis"""
        pass
    
    @abstractmethod
    async def execute_action(self, state: AURAAgentState) -> Dict[str, Any]:
        """Execute the decided action"""
        pass
    
    # ======================
    # Utility Methods
    # ======================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (
                self.successful_tasks / max(1, self.total_tasks)
            ),
            "components_available": {
                "memory": self.memory is not None,
                "tda": self.tda is not None,
                "neural": self.router is not None,
                "orchestration": self.orchestrator is not None
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down agent {self.agent_id}")
        
        # Save final state to memory
        if self.memory:
            try:
                await self.memory.store({
                    "agent_id": self.agent_id,
                    "event": "shutdown",
                    "metrics": self.get_metrics(),
                    "timestamp": datetime.now(timezone.utc)
                })
            except Exception as e:
                logger.error(f"Failed to save shutdown state: {e}")


# ======================
# Example Usage
# ======================

if __name__ == "__main__":
    # This is an abstract class, see agent_templates.py for concrete implementations
    print("AURAAgentCore is an abstract base class.")
    print("See agent_templates.py for concrete agent implementations:")
    print("- ObserverAgent")
    print("- AnalystAgent") 
    print("- ExecutorAgent")
    print("- CoordinatorAgent")