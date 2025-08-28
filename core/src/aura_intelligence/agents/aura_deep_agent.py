"""
AURA Deep Agent - Based on LangChain DeepAgents Architecture

Features from latest DeepAgents:
- Planning tool with dynamic task decomposition
- Persistent state management
- Sub-agents for specialized tasks
- Virtual file system for state
- Human-in-the-loop with interrupts
- Streaming support
- Checkpointing and recovery
- Multi-agent coordination
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict, Literal, Union
import operator
from enum import Enum
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import Send

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


# State definitions
class Plan(BaseModel):
    """A plan for completing a task"""
    task_id: str
    main_goal: str
    subtasks: List[Dict[str, Any]]
    status: Literal["pending", "in_progress", "completed", "failed"]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class VirtualFile(BaseModel):
    """Virtual file for persistent state"""
    path: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)


class AgentMemory(BaseModel):
    """Agent's long-term memory"""
    facts: List[str] = Field(default_factory=list)
    learnings: List[Dict[str, Any]] = Field(default_factory=list)
    context_windows: List[Dict[str, Any]] = Field(default_factory=list)


class DeepAgentState(TypedDict):
    """State for AURA Deep Agent"""
    # Core state
    messages: Annotated[List[BaseMessage], add_messages]
    current_task: str
    agent_id: str
    
    # Planning
    current_plan: Optional[Plan]
    plans_history: List[Plan]
    active_subtask: Optional[Dict[str, Any]]
    
    # Memory and context
    memory: AgentMemory
    virtual_files: Dict[str, VirtualFile]
    
    # Sub-agents
    sub_agents: Dict[str, Any]
    sub_agent_results: List[Dict[str, Any]]
    
    # Control flow
    next_action: Literal["plan", "execute", "delegate", "reflect", "ask_human", "complete"]
    should_continue: bool
    requires_human: bool
    confidence: float
    
    # Metadata
    iteration: int
    max_iterations: int
    start_time: datetime
    checkpoints: List[Dict[str, Any]]


class SubAgentType(str, Enum):
    """Types of sub-agents"""
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYZER = "analyzer"
    VALIDATOR = "validator"
    DOCUMENTER = "documenter"


class AURADeepAgent:
    """AURA implementation of DeepAgents architecture"""
    
    def __init__(self,
                 agent_id: str,
                 tools: List[BaseTool],
                 checkpoint_dir: str = "./checkpoints",
                 max_iterations: int = 50,
                 enable_streaming: bool = True):
        
        self.agent_id = agent_id
        self.tools = tools
        self.max_iterations = max_iterations
        self.enable_streaming = enable_streaming
        
        # Virtual file system
        self.vfs_root = Path(f"./vfs/{agent_id}")
        self.vfs_root.mkdir(parents=True, exist_ok=True)
        
        # Checkpointing
        self.checkpointer = SqliteSaver.from_conn_string(
            f"{checkpoint_dir}/{agent_id}.db"
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"AURA Deep Agent initialized",
                   agent_id=agent_id,
                   tools=len(tools))
    
    def _build_graph(self) -> StateGraph:
        """Build the DeepAgents-style graph"""
        
        # Create workflow
        workflow = StateGraph(DeepAgentState)
        
        # Add nodes
        workflow.add_node("planner", self.planning_node)
        workflow.add_node("executor", self.execution_node)
        workflow.add_node("delegator", self.delegation_node)
        workflow.add_node("reflector", self.reflection_node)
        workflow.add_node("human_interface", self.human_interface_node)
        workflow.add_node("completer", self.completion_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {
                "execute": "executor",
                "delegate": "delegator",
                "reflect": "reflector",
                "complete": "completer"
            }
        )
        
        workflow.add_conditional_edges(
            "executor",
            self.route_from_executor,
            {
                "tools": "tools",
                "plan": "planner",
                "delegate": "delegator",
                "human": "human_interface",
                "reflect": "reflector"
            }
        )
        
        workflow.add_conditional_edges(
            "tools",
            self.route_from_tools,
            {
                "executor": "executor",
                "reflector": "reflector"
            }
        )
        
        workflow.add_edge("delegator", "reflector")
        workflow.add_edge("reflector", "planner")
        workflow.add_edge("human_interface", "planner")
        workflow.add_edge("completer", END)
        
        # Compile with interrupt support
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_interface"] if self.enable_streaming else []
        )
    
    # Planning Node
    async def planning_node(self, state: DeepAgentState, config: RunnableConfig) -> DeepAgentState:
        """Create and update plans"""
        
        task = state["current_task"]
        current_plan = state.get("current_plan")
        iteration = state.get("iteration", 0)
        
        # Create new plan if needed
        if not current_plan or current_plan.status == "failed":
            plan = await self._create_plan(task, state)
            state["current_plan"] = plan
            state["plans_history"].append(plan)
            
            logger.info(f"Created plan", 
                       plan_id=plan.task_id,
                       subtasks=len(plan.subtasks))
        else:
            # Update existing plan
            plan = current_plan
            completed_subtasks = sum(1 for st in plan.subtasks if st.get("completed"))
            
            if completed_subtasks == len(plan.subtasks):
                plan.status = "completed"
            else:
                # Find next subtask
                for subtask in plan.subtasks:
                    if not subtask.get("completed"):
                        state["active_subtask"] = subtask
                        break
        
        # Determine next action
        if plan.status == "completed":
            state["next_action"] = "complete"
        elif state.get("active_subtask"):
            subtask = state["active_subtask"]
            if subtask.get("requires_delegation"):
                state["next_action"] = "delegate"
            else:
                state["next_action"] = "execute"
        else:
            state["next_action"] = "reflect"
        
        state["iteration"] = iteration + 1
        
        return state
    
    async def _create_plan(self, task: str, state: DeepAgentState) -> Plan:
        """Create a detailed plan for the task"""
        
        # Analyze task complexity
        complexity = self._analyze_complexity(task)
        
        # Generate subtasks based on complexity
        subtasks = []
        
        if "calculate" in task.lower() or "compute" in task.lower():
            subtasks = [
                {
                    "id": "parse",
                    "description": "Parse the mathematical expression",
                    "tool": "calculator",
                    "completed": False
                },
                {
                    "id": "compute",
                    "description": "Perform the calculation",
                    "tool": "calculator",
                    "completed": False
                },
                {
                    "id": "validate",
                    "description": "Validate the result",
                    "completed": False
                }
            ]
        elif "research" in task.lower() or "find" in task.lower():
            subtasks = [
                {
                    "id": "search",
                    "description": "Search for relevant information",
                    "tool": "web_search",
                    "requires_delegation": True,
                    "sub_agent": SubAgentType.RESEARCHER,
                    "completed": False
                },
                {
                    "id": "analyze",
                    "description": "Analyze findings",
                    "requires_delegation": True,
                    "sub_agent": SubAgentType.ANALYZER,
                    "completed": False
                },
                {
                    "id": "summarize",
                    "description": "Summarize results",
                    "completed": False
                }
            ]
        else:
            # Generic plan
            subtasks = [
                {
                    "id": "understand",
                    "description": f"Understand the requirements: {task}",
                    "completed": False
                },
                {
                    "id": "execute",
                    "description": "Execute the main task",
                    "completed": False
                },
                {
                    "id": "verify",
                    "description": "Verify the results",
                    "completed": False
                }
            ]
        
        plan = Plan(
            task_id=str(uuid.uuid4()),
            main_goal=task,
            subtasks=subtasks,
            status="pending"
        )
        
        # Store plan in virtual file system
        await self._save_to_vfs(f"plans/{plan.task_id}.json", plan.json())
        
        return plan
    
    def _analyze_complexity(self, task: str) -> Dict[str, Any]:
        """Analyze task complexity"""
        
        words = task.lower().split()
        
        return {
            "length": len(words),
            "has_numbers": any(word.replace(".", "").isdigit() for word in words),
            "has_operators": any(op in task for op in ["+", "-", "*", "/", "^"]),
            "requires_search": any(word in words for word in ["find", "search", "research"]),
            "requires_analysis": any(word in words for word in ["analyze", "compare", "evaluate"])
        }
    
    # Execution Node
    async def execution_node(self, state: DeepAgentState, config: RunnableConfig) -> DeepAgentState:
        """Execute the current subtask"""
        
        subtask = state.get("active_subtask")
        if not subtask:
            state["next_action"] = "plan"
            return state
        
        # Check if tools are needed
        if subtask.get("tool"):
            # Prepare for tool use
            tool_message = AIMessage(
                content=f"I need to use the {subtask['tool']} tool to {subtask['description']}",
                additional_kwargs={
                    "function_call": {
                        "name": subtask["tool"],
                        "arguments": json.dumps({"input": state["current_task"]})
                    }
                }
            )
            state["messages"].append(tool_message)
            state["next_action"] = "tools"
        else:
            # Direct execution (simplified for demo)
            subtask["completed"] = True
            
            # Update memory
            state["memory"].facts.append(f"Completed: {subtask['description']}")
            
            # Add completion message
            state["messages"].append(
                AIMessage(content=f"Completed subtask: {subtask['description']}")
            )
            
            state["next_action"] = "reflect"
        
        return state
    
    # Delegation Node
    async def delegation_node(self, state: DeepAgentState, config: RunnableConfig) -> DeepAgentState:
        """Delegate to sub-agents"""
        
        subtask = state.get("active_subtask")
        if not subtask:
            return state
        
        sub_agent_type = subtask.get("sub_agent")
        
        # Create sub-agent (simplified)
        sub_agent_id = f"{self.agent_id}-{sub_agent_type}-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Delegating to sub-agent",
                   sub_agent_id=sub_agent_id,
                   type=sub_agent_type)
        
        # Simulate sub-agent execution
        result = await self._run_sub_agent(sub_agent_type, subtask, state)
        
        # Store result
        state["sub_agent_results"].append({
            "sub_agent_id": sub_agent_id,
            "type": sub_agent_type,
            "subtask": subtask["id"],
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mark subtask complete
        subtask["completed"] = True
        
        # Update messages
        state["messages"].append(
            AIMessage(content=f"Sub-agent {sub_agent_type} completed: {result}")
        )
        
        return state
    
    async def _run_sub_agent(self, agent_type: SubAgentType, 
                           subtask: Dict[str, Any], 
                           state: DeepAgentState) -> str:
        """Run a sub-agent (simplified simulation)"""
        
        if agent_type == SubAgentType.RESEARCHER:
            return f"Research findings for '{subtask['description']}': Found relevant information"
        elif agent_type == SubAgentType.ANALYZER:
            return f"Analysis complete: Identified key patterns and insights"
        elif agent_type == SubAgentType.CODER:
            return f"Code generated: Implementation ready"
        elif agent_type == SubAgentType.VALIDATOR:
            return f"Validation passed: All checks successful"
        else:
            return f"{agent_type} completed task"
    
    # Reflection Node
    async def reflection_node(self, state: DeepAgentState, config: RunnableConfig) -> DeepAgentState:
        """Reflect on progress and adjust strategy"""
        
        plan = state.get("current_plan")
        memory = state["memory"]
        
        # Analyze progress
        if plan:
            completed = sum(1 for st in plan.subtasks if st.get("completed"))
            total = len(plan.subtasks)
            progress = completed / total if total > 0 else 0
            
            # Add learning
            learning = {
                "task": state["current_task"],
                "progress": progress,
                "iterations": state.get("iteration", 0),
                "insights": []
            }
            
            if progress < 0.5 and state.get("iteration", 0) > 10:
                learning["insights"].append("Task is taking longer than expected")
                state["confidence"] = 0.6
            elif progress == 1.0:
                learning["insights"].append("Task completed successfully")
                state["confidence"] = 0.95
            else:
                state["confidence"] = 0.8
            
            memory.learnings.append(learning)
            
            # Check if human input needed
            if state["confidence"] < 0.7:
                state["requires_human"] = True
        
        # Save state checkpoint
        checkpoint = {
            "iteration": state.get("iteration", 0),
            "progress": progress if 'progress' in locals() else 0,
            "confidence": state["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        state["checkpoints"].append(checkpoint)
        
        # Save to VFS
        await self._save_to_vfs(
            f"checkpoints/{state['agent_id']}-{checkpoint['iteration']}.json",
            json.dumps(checkpoint)
        )
        
        return state
    
    # Human Interface Node
    async def human_interface_node(self, state: DeepAgentState, config: RunnableConfig) -> DeepAgentState:
        """Handle human-in-the-loop interactions"""
        
        logger.info("Human input requested", confidence=state["confidence"])
        
        # In real implementation, this would pause and wait for human input
        # For demo, we'll simulate human approval
        state["messages"].append(
            HumanMessage(content="Approved. Please continue with the plan.")
        )
        
        state["requires_human"] = False
        state["confidence"] = 0.9  # Boost confidence after human approval
        
        return state
    
    # Completion Node
    async def completion_node(self, state: DeepAgentState, config: RunnableConfig) -> DeepAgentState:
        """Finalize and return results"""
        
        plan = state.get("current_plan")
        sub_results = state.get("sub_agent_results", [])
        
        # Build final response
        response_parts = [
            f"Task completed: {state['current_task']}",
            f"\nExecution Summary:"
        ]
        
        if plan:
            response_parts.append(f"- Total subtasks: {len(plan.subtasks)}")
            response_parts.append(f"- Completed: {sum(1 for st in plan.subtasks if st.get('completed'))}")
        
        if sub_results:
            response_parts.append(f"\nSub-agent Results:")
            for result in sub_results:
                response_parts.append(f"- {result['type']}: {result['result']}")
        
        # Add learnings
        if state["memory"].learnings:
            latest_learning = state["memory"].learnings[-1]
            if latest_learning.get("insights"):
                response_parts.append(f"\nInsights:")
                for insight in latest_learning["insights"]:
                    response_parts.append(f"- {insight}")
        
        final_response = "\n".join(response_parts)
        
        # Save final state
        await self._save_to_vfs(
            f"completed/{state['agent_id']}-final.json",
            json.dumps({
                "task": state["current_task"],
                "response": final_response,
                "stats": {
                    "iterations": state.get("iteration", 0),
                    "confidence": state["confidence"],
                    "duration": (datetime.now() - state["start_time"]).total_seconds()
                }
            })
        )
        
        state["messages"].append(AIMessage(content=final_response))
        state["should_continue"] = False
        
        return state
    
    # Routing functions
    def route_from_planner(self, state: DeepAgentState) -> str:
        """Route from planner based on next_action"""
        return state.get("next_action", "reflect")
    
    def route_from_executor(self, state: DeepAgentState) -> str:
        """Route from executor"""
        if state.get("requires_human"):
            return "human"
        return state.get("next_action", "reflect")
    
    def route_from_tools(self, state: DeepAgentState) -> str:
        """Route after tool execution"""
        # Check if tool execution was successful
        last_message = state["messages"][-1] if state["messages"] else None
        if isinstance(last_message, ToolMessage) and "error" not in last_message.content:
            return "executor"
        return "reflector"
    
    # VFS operations
    async def _save_to_vfs(self, path: str, content: str) -> None:
        """Save to virtual file system"""
        file_path = self.vfs_root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    async def _load_from_vfs(self, path: str) -> Optional[str]:
        """Load from virtual file system"""
        file_path = self.vfs_root / path
        if file_path.exists():
            return file_path.read_text()
        return None
    
    # Main invocation
    async def invoke(self, task: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Invoke the deep agent"""
        
        # Initialize state
        initial_state = DeepAgentState(
            messages=[HumanMessage(content=task)],
            current_task=task,
            agent_id=self.agent_id,
            current_plan=None,
            plans_history=[],
            active_subtask=None,
            memory=AgentMemory(),
            virtual_files={},
            sub_agents={},
            sub_agent_results=[],
            next_action="plan",
            should_continue=True,
            requires_human=False,
            confidence=0.8,
            iteration=0,
            max_iterations=self.max_iterations,
            start_time=datetime.now(),
            checkpoints=[]
        )
        
        # Configure with thread ID for checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id or f"{self.agent_id}-{uuid.uuid4().hex[:8]}"
            }
        }
        
        # Stream execution if enabled
        if self.enable_streaming:
            result = None
            async for event in self.graph.astream(initial_state, config):
                logger.info(f"Stream event", 
                           node=list(event.keys())[0] if event else None)
                result = event
            return result
        else:
            # Regular invocation
            return await self.graph.ainvoke(initial_state, config)


# Create specialized tools
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations"""
    try:
        # Safe eval with limited operations
        allowed_names = {
            k: v for k, v in {"__builtins__": {}}.items()
        }
        allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "len": len
        })
        
        result = eval(expression, allowed_names, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    # Simulated search
    return f"Search results for '{query}': Found information about {query}"


@tool
def code_analyzer(code: str) -> str:
    """Analyze code quality and suggest improvements"""
    lines = code.split('\n')
    return f"Code analysis: {len(lines)} lines, looks good!"


# Demo
async def main():
    """Demo the AURA Deep Agent"""
    
    print("🧠 AURA Deep Agent Demo (DeepAgents Architecture)")
    print("=" * 60)
    
    # Create tools
    tools = [calculator, web_search, code_analyzer]
    
    # Create agent
    agent = AURADeepAgent(
        agent_id="aura-deep-001",
        tools=tools,
        enable_streaming=True
    )
    
    # Test tasks
    tasks = [
        "Calculate the factorial of 7 and explain the result",
        "Research the latest developments in quantum computing and create a summary"
    ]
    
    for task in tasks:
        print(f"\n📋 Task: {task}")
        print("-" * 60)
        
        result = await agent.invoke(task)
        
        # Get final state
        if isinstance(result, dict):
            for key, value in result.items():
                final_state = value
                break
        else:
            final_state = result
        
        # Print results
        messages = final_state.get("messages", [])
        if messages:
            print(f"\n💬 Final Response:")
            print(messages[-1].content if isinstance(messages[-1], AIMessage) else "No response")
        
        print(f"\n📊 Execution Stats:")
        print(f"  Iterations: {final_state.get('iteration', 0)}")
        print(f"  Confidence: {final_state.get('confidence', 0):.2f}")
        print(f"  Sub-agents used: {len(final_state.get('sub_agent_results', []))}")
        
        # Show plan
        plan = final_state.get("current_plan")
        if plan:
            print(f"\n📝 Plan Status: {plan.status}")
            print(f"  Subtasks: {len(plan.subtasks)}")
            completed = sum(1 for st in plan.subtasks if st.get("completed"))
            print(f"  Completed: {completed}/{len(plan.subtasks)}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())