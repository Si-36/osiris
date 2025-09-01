"""
Ultimate LangGraph Integration for AURA

This is the central hub that connects all LangGraph components:
- Core system integration
- Council agents
- Deep agents
- Orchestration layer
- Collective intelligence
- Memory systems
- Tool registry
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
import uuid
from pathlib import Path
import json

import structlog

# Try to import LangGraph with fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    SqliteSaver = None
    PostgresSaver = None
    ToolExecutor = None

# Try to import LangChain
try:
    from langchain_core.tools import BaseTool
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = dict
    BaseMessage = dict
    HumanMessage = dict
    AIMessage = dict

# Import all existing LangGraph components with fallback
try:
    from ..orchestration.semantic.langgraph_orchestrator import (
        LangGraphSemanticOrchestrator, 
        SemanticWorkflowConfig,
        OrchestrationStrategy
    )
except ImportError:
    LangGraphSemanticOrchestrator = None
    SemanticWorkflowConfig = None
    OrchestrationStrategy = None

try:
    from ..agents.council.lnn_council_agent import LNNCouncilAgent
except ImportError:
    LNNCouncilAgent = None

try:
    from ..agents.council.production_lnn_council import ProductionLNNCouncilAgent
except ImportError:
    ProductionLNNCouncilAgent = None

try:
    from ..agents.aura_deep_agent import AURADeepAgent
except ImportError:
    AURADeepAgent = None

try:
    from ..agents.langgraph_agent_system import AURAAgent
except ImportError:
    AURAAgent = None

try:
    from ..collective.graph_builder import GraphBuilder, GraphType
except ImportError:
    GraphBuilder = None
    GraphType = None

try:
    from ..communication.collective.supervisor import CollectiveSupervisor
except ImportError:
    CollectiveSupervisor = None

logger = structlog.get_logger(__name__)


class AgentRegistry:
    """Central registry for all LangGraph agents"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.agent_types: Dict[str, Type] = {}
        self.capabilities: Dict[str, List[str]] = {}
        
    def register_agent(self, agent_id: str, agent_instance: Any, 
                      capabilities: List[str] = None):
        """Register an agent with its capabilities"""
        self.agents[agent_id] = agent_instance
        self.agent_types[agent_id] = type(agent_instance)
        self.capabilities[agent_id] = capabilities or []
        logger.info(f"Registered agent", agent_id=agent_id, 
                   type=type(agent_instance).__name__)
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agents that have a specific capability"""
        return [
            agent_id for agent_id, caps in self.capabilities.items()
            if capability in caps
        ]
    
    def list_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents with their info"""
        return {
            agent_id: {
                "type": self.agent_types[agent_id].__name__,
                "capabilities": self.capabilities[agent_id],
                "instance": self.agents[agent_id]
            }
            for agent_id in self.agents
        }


class ToolRegistry:
    """Central registry for all tools available to agents"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        
    def register_tool(self, tool: BaseTool, category: str = "general"):
        """Register a tool"""
        self.tools[tool.name] = tool
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(tool.name)
        logger.info(f"Registered tool", name=tool.name, category=category)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category"""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self.tools.values())


class UltimateLangGraphIntegration:
    """
    The ultimate integration layer for all LangGraph components in AURA
    """
    
    def __init__(self, config: Any, consciousness: Any):
        self.config = config
        self.consciousness = consciousness
        self.initialized = False
        
        # Check if LangGraph is available
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available - running in fallback mode")
            self.fallback_mode = True
        else:
            self.fallback_mode = False
        
        # Registries
        self.agent_registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
        
        # Core components
        self.orchestrator: Optional[LangGraphSemanticOrchestrator] = None
        self.supervisor: Optional[CollectiveSupervisor] = None
        self.graph_builder: Optional[GraphBuilder] = None
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir if hasattr(config, 'checkpoint_dir') 
                                  else "./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Workflow storage
        self.active_workflows: Dict[str, Any] = {}
        
        logger.info("UltimateLangGraphIntegration created", 
                   checkpoint_dir=str(self.checkpoint_dir))
    
    async def initialize(self):
        """Initialize all LangGraph components"""
        logger.info("Initializing LangGraph integration...")
        
        try:
            # 1. Initialize orchestrator
            await self._initialize_orchestrator()
            
            # 2. Initialize supervisor
            await self._initialize_supervisor()
            
            # 3. Initialize graph builder
            await self._initialize_graph_builder()
            
            # 4. Register existing agents
            await self._register_existing_agents()
            
            # 5. Register tools
            await self._register_tools()
            
            # 6. Setup inter-component connections
            await self._setup_connections()
            
            self.initialized = True
            logger.info("✅ LangGraph integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph integration", error=str(e))
            raise
    
    async def _initialize_orchestrator(self):
        """Initialize the semantic orchestrator"""
        # Check for PostgreSQL config
        postgres_url = getattr(self.config, 'postgres_url', None)
        redis_url = getattr(self.config, 'redis_url', None)
        
        self.orchestrator = LangGraphSemanticOrchestrator(
            tda_integration=self.consciousness,  # Use consciousness as TDA context
            postgres_url=postgres_url,
            redis_url=redis_url
        )
        
        logger.info("Orchestrator initialized")
    
    async def _initialize_supervisor(self):
        """Initialize the collective supervisor"""
        supervisor_config = {
            "supervisor_id": "aura-supervisor-001",
            "agent_registry": self.agent_registry,
            "tool_registry": self.tool_registry
        }
        
        self.supervisor = CollectiveSupervisor(
            config=supervisor_config,
            orchestrator=self.orchestrator
        )
        
        logger.info("Supervisor initialized")
    
    async def _initialize_graph_builder(self):
        """Initialize the graph builder"""
        self.graph_builder = GraphBuilder(
            graph_type=GraphType.WORKFLOW,
            enable_persistence=True
        )
        
        logger.info("Graph builder initialized")
    
    async def _register_existing_agents(self):
        """Register all existing LangGraph agents"""
        
        # 1. Register Production Agents (Latest 2025 Patterns)
        try:
            from ..agents.production_langgraph_agent import create_production_agent
            
            # General purpose agent
            general_agent = create_production_agent(
                name="aura-general-001",
                role="general",
                memory_manager=self.memory,
                knowledge_graph=self.knowledge_graph,
                event_producer=self.events
            )
            self.agent_registry.register_agent(
                "aura-general-001",
                general_agent,
                ["general_tasks", "reasoning", "analysis"]
            )
            
            # Analyst agent
            analyst_agent = create_production_agent(
                name="aura-analyst-001",
                role="analyst",
                enable_reasoning=True,
                memory_manager=self.memory,
                knowledge_graph=self.knowledge_graph,
                event_producer=self.events
            )
            self.agent_registry.register_agent(
                "aura-analyst-001",
                analyst_agent,
                ["data_analysis", "pattern_recognition", "insights"]
            )
            
            # Executor agent
            executor_agent = create_production_agent(
                name="aura-executor-001", 
                role="executor",
                tool_choice="required",
                memory_manager=self.memory,
                knowledge_graph=self.knowledge_graph,
                event_producer=self.events
            )
            self.agent_registry.register_agent(
                "aura-executor-001",
                executor_agent,
                ["task_execution", "tool_usage", "implementation"]
            )
            
            logger.info("✅ Registered production LangGraph agents")
            
        except Exception as e:
            logger.warning(f"Could not register production agents", error=str(e))
        
        # 2. Register Council Agents
        try:
            # LNN Council Agent for GPU allocation
            lnn_council = LNNCouncilAgent(
                agent_id="lnn-council-001",
                role="gpu_allocator"
            )
            self.agent_registry.register_agent(
                "lnn-council-001",
                lnn_council,
                ["gpu_allocation", "resource_management", "neural_inference"]
            )
        except Exception as e:
            logger.warning(f"Could not register LNN Council agent", error=str(e))
        
        try:
            # Production LNN Council Agent
            prod_council = ProductionLNNCouncilAgent(
                agent_id="prod-council-001"
            )
            self.agent_registry.register_agent(
                "prod-council-001",
                prod_council,
                ["production_decisions", "consensus", "validation"]
            )
        except Exception as e:
            logger.warning(f"Could not register Production Council agent", error=str(e))
        
        # 2. Register our new Deep Agents
        try:
            # Get tools for deep agent
            tools = self.tool_registry.get_all_tools()
            
            # AURA Deep Agent (DeepAgents architecture)
            deep_agent = AURADeepAgent(
                agent_id="aura-deep-001",
                tools=tools,
                checkpoint_dir=str(self.checkpoint_dir),
                enable_streaming=True
            )
            self.agent_registry.register_agent(
                "aura-deep-001",
                deep_agent,
                ["planning", "sub_agents", "complex_tasks", "reflection"]
            )
            
            # Basic LangGraph Agent
            basic_agent = AURAAgent(
                agent_id="aura-basic-001",
                tools=tools,
                checkpoint_config={"checkpoint_dir": str(self.checkpoint_dir)}
            )
            self.agent_registry.register_agent(
                "aura-basic-001",
                basic_agent,
                ["reasoning", "tool_use", "memory"]
            )
        except Exception as e:
            logger.warning(f"Could not register Deep agents", error=str(e))
        
        logger.info(f"Registered {len(self.agent_registry.agents)} agents")
    
    async def _register_tools(self):
        """Register all available tools"""
        
        # Import and register tools from our new agents
        try:
            from ..agents.aura_deep_agent import calculator, web_search, code_analyzer
            
            self.tool_registry.register_tool(calculator, "math")
            self.tool_registry.register_tool(web_search, "research")
            self.tool_registry.register_tool(code_analyzer, "code")
        except Exception as e:
            logger.warning(f"Could not register deep agent tools", error=str(e))
        
        # Register any other system tools
        # TODO: Import tools from other parts of the system
        
        logger.info(f"Registered {len(self.tool_registry.tools)} tools")
    
    async def _setup_connections(self):
        """Setup connections between components"""
        
        # 1. Connect agents to orchestrator
        if self.orchestrator:
            for agent_id, agent_info in self.agent_registry.list_all_agents().items():
                # Register agent with orchestrator
                await self.orchestrator.register_agent(agent_id, agent_info["capabilities"])
        
        # 2. Connect supervisor to orchestrator
        if self.supervisor and self.orchestrator:
            self.supervisor.set_orchestrator(self.orchestrator)
        
        # 3. Setup shared memory access
        # All agents should have access to AURA's memory system
        if hasattr(self.consciousness, 'memory'):
            for agent_id, agent in self.agent_registry.agents.items():
                if hasattr(agent, 'memory'):
                    # Link to AURA's memory system
                    agent.memory = self.consciousness.memory
        
        logger.info("Component connections established")
    
    async def execute_advanced_workflows(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute advanced workflows using the LangGraph system
        
        This is called from core/system.py in Phase 7
        """
        logger.info(f"Executing advanced workflow", task=task, fallback_mode=self.fallback_mode)
        
        if not self.initialized:
            await self.initialize()
        
        # Handle fallback mode
        if self.fallback_mode:
            return await self._execute_fallback_workflow(task, context)
        
        # Create workflow ID
        workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
        
        try:
            # 1. Analyze task to determine best approach
            analysis = await self._analyze_task(task, context)
            
            # 2. Select appropriate agents
            selected_agents = await self._select_agents(analysis)
            
            # 3. Create workflow configuration
            workflow_config = SemanticWorkflowConfig(
                workflow_id=workflow_id,
                orchestrator_agent="aura-supervisor-001",
                worker_agents=selected_agents,
                routing_strategy=OrchestrationStrategy.SEMANTIC
            )
            
            # 4. Build workflow graph
            workflow_graph = await self._build_workflow_graph(
                workflow_config, 
                task, 
                analysis
            )
            
            # 5. Execute workflow
            result = await self._execute_workflow(
                workflow_graph,
                workflow_id,
                task,
                context
            )
            
            # 6. Store workflow results
            self.active_workflows[workflow_id] = {
                "task": task,
                "agents": selected_agents,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed", 
                        workflow_id=workflow_id, 
                        error=str(e))
            return {
                "error": str(e),
                "workflow_id": workflow_id,
                "status": "failed"
            }
    
    async def _analyze_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task to determine requirements"""
        
        analysis = {
            "complexity": self._estimate_complexity(task),
            "requires_planning": "plan" in task.lower() or "strategy" in task.lower(),
            "requires_tools": any(word in task.lower() for word in ["calculate", "search", "analyze"]),
            "requires_consensus": "decide" in task.lower() or "choose" in task.lower(),
            "task_type": self._classify_task(task)
        }
        
        logger.info(f"Task analysis complete", analysis=analysis)
        return analysis
    
    def _estimate_complexity(self, task: str) -> float:
        """Estimate task complexity (0-1)"""
        # Simple heuristic based on length and keywords
        complexity_keywords = ["complex", "multi-step", "analyze", "research", "plan"]
        
        base_complexity = min(len(task.split()) / 50, 0.5)
        keyword_bonus = sum(0.1 for keyword in complexity_keywords if keyword in task.lower())
        
        return min(base_complexity + keyword_bonus, 1.0)
    
    def _classify_task(self, task: str) -> str:
        """Classify the task type"""
        task_lower = task.lower()
        
        if "calculate" in task_lower or "compute" in task_lower:
            return "computational"
        elif "research" in task_lower or "find" in task_lower:
            return "research"
        elif "plan" in task_lower or "strategy" in task_lower:
            return "planning"
        elif "decide" in task_lower or "choose" in task_lower:
            return "decision"
        else:
            return "general"
    
    async def _select_agents(self, analysis: Dict[str, Any]) -> List[str]:
        """Select appropriate agents based on task analysis"""
        selected = []
        
        # Always include a coordinator
        if analysis["complexity"] > 0.7:
            selected.append("aura-deep-001")  # Complex tasks need deep agent
        else:
            selected.append("aura-basic-001")  # Simple tasks use basic agent
        
        # Add specialized agents based on requirements
        if analysis["requires_consensus"]:
            # Add council agents
            council_agents = self.agent_registry.get_agents_by_capability("consensus")
            selected.extend(council_agents[:2])  # Add up to 2 council agents
        
        if analysis["task_type"] == "computational":
            # Add agents with math capabilities
            math_agents = self.agent_registry.get_agents_by_capability("math")
            selected.extend(math_agents[:1])
        
        # Ensure we have at least one agent
        if not selected:
            selected.append("aura-basic-001")
        
        # Remove duplicates
        selected = list(dict.fromkeys(selected))
        
        logger.info(f"Selected agents", agents=selected)
        return selected
    
    async def _build_workflow_graph(self, config: SemanticWorkflowConfig, 
                                   task: str, analysis: Dict[str, Any]) -> StateGraph:
        """Build a workflow graph for the task"""
        
        # Use the graph builder to create workflow
        self.graph_builder.add_node("start", lambda x: x)
        
        # Add agent nodes
        for agent_id in config.worker_agents:
            agent = self.agent_registry.get_agent(agent_id)
            if agent:
                self.graph_builder.add_node(
                    agent_id,
                    lambda state, aid=agent_id: self._agent_node_wrapper(state, aid)
                )
        
        # Add edges based on task type
        if analysis["requires_planning"]:
            # Sequential execution for planning tasks
            prev_node = "start"
            for agent_id in config.worker_agents:
                self.graph_builder.add_edge(prev_node, agent_id)
                prev_node = agent_id
            self.graph_builder.add_edge(prev_node, END)
        else:
            # Parallel execution for simple tasks
            for agent_id in config.worker_agents:
                self.graph_builder.add_edge("start", agent_id)
                self.graph_builder.add_edge(agent_id, END)
        
        # Compile the graph
        return self.graph_builder.compile()
    
    async def _agent_node_wrapper(self, state: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Wrapper for agent execution in graph"""
        agent = self.agent_registry.get_agent(agent_id)
        if not agent:
            return state
        
        try:
            # Execute agent
            if hasattr(agent, 'invoke'):
                result = await agent.invoke(state.get("task", ""))
            elif hasattr(agent, 'process'):
                result = await agent.process(state.get("task", ""))
            else:
                result = {"error": f"Agent {agent_id} has no execution method"}
            
            # Update state
            state[f"{agent_id}_result"] = result
            
        except Exception as e:
            logger.error(f"Agent execution failed", agent_id=agent_id, error=str(e))
            state[f"{agent_id}_error"] = str(e)
        
        return state
    
    async def _execute_workflow(self, workflow: StateGraph, workflow_id: str,
                               task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow graph"""
        
        # Initial state
        initial_state = {
            "task": task,
            "context": context,
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Execute workflow
        try:
            if hasattr(workflow, 'ainvoke'):
                result = await workflow.ainvoke(initial_state)
            else:
                # Fallback for synchronous execution
                result = workflow.invoke(initial_state)
            
            return {
                "status": "completed",
                "workflow_id": workflow_id,
                "results": result
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error", 
                        workflow_id=workflow_id,
                        error=str(e))
            return {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up LangGraph integration...")
        
        # Save active workflows
        if self.active_workflows:
            workflow_file = self.checkpoint_dir / "active_workflows.json"
            with open(workflow_file, "w") as f:
                json.dump(self.active_workflows, f, indent=2)
        
        # Cleanup agents
        for agent_id, agent in self.agent_registry.agents.items():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        logger.info("LangGraph integration cleanup complete")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of LangGraph integration"""
        return {
            "initialized": self.initialized,
            "registered_agents": len(self.agent_registry.agents),
            "registered_tools": len(self.tool_registry.tools),
            "active_workflows": len(self.active_workflows),
            "orchestrator_active": self.orchestrator is not None,
            "supervisor_active": self.supervisor is not None
        }
    
    async def execute_agent_directly(self, agent_id: str, task: str) -> Dict[str, Any]:
        """Execute a specific agent directly (for testing)"""
        agent = self.agent_registry.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}
        
        try:
            if hasattr(agent, 'invoke'):
                return await agent.invoke(task)
            elif hasattr(agent, 'process'):
                return await agent.process(task)
            else:
                return {"error": f"Agent {agent_id} has no execution method"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _execute_fallback_workflow(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow without LangGraph (fallback mode)"""
        workflow_id = f"fallback-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Executing fallback workflow", workflow_id=workflow_id)
        
        try:
            # Simple sequential execution
            results = {
                "workflow_id": workflow_id,
                "mode": "fallback",
                "task": task,
                "context": context,
                "agents_available": len(self.agent_registry.agents),
                "tools_available": len(self.tool_registry.tools),
                "status": "completed",
                "message": "LangGraph not available - executed in fallback mode"
            }
            
            # If we have any registered agents, try to use one
            if self.agent_registry.agents:
                agent_id = list(self.agent_registry.agents.keys())[0]
                try:
                    agent_result = await self.execute_agent_directly(agent_id, task)
                    results["agent_results"] = {agent_id: agent_result}
                except Exception as e:
                    results["agent_error"] = str(e)
            
            return results
            
        except Exception as e:
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "mode": "fallback"
            }