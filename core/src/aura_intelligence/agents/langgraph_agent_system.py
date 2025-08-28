"""
LangGraph-based Agent System for AURA

Using:
- LangGraph for agent workflows and state management
- LangChain for memory and tools
- Mem0 for long-term memory
- Model Context Protocol (MCP) for tool integration
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict, Sequence
from enum import Enum
import operator
from datetime import datetime
import uuid
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_core.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import structlog

logger = structlog.get_logger(__name__)


class AgentState(TypedDict):
    """State definition for LangGraph agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_task: str
    thoughts: List[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    tools_output: List[Dict[str, Any]]
    final_answer: Optional[str]
    metadata: Dict[str, Any]


class ThoughtType(str, Enum):
    """Types of agent thoughts"""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    PLANNING = "planning"
    REFLECTION = "reflection"
    TOOL_SELECTION = "tool_selection"


class AURAAgent:
    """Main AURA agent using LangGraph"""
    
    def __init__(self, 
                 agent_id: str,
                 tools: List[Tool],
                 memory_config: Optional[Dict[str, Any]] = None,
                 checkpoint_config: Optional[Dict[str, Any]] = None):
        
        self.agent_id = agent_id
        self.tools = tools
        self.tool_executor = ToolExecutor(tools)
        
        # Initialize memory
        self.memory = self._init_memory(memory_config)
        
        # Initialize checkpointer for state persistence
        self.checkpointer = MemorySaver() if checkpoint_config else None
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"AURA Agent initialized", agent_id=agent_id, num_tools=len(tools))
    
    def _init_memory(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize agent memory"""
        config = config or {}
        
        # Try to use Mem0 if available
        try:
            from mem0 import Memory
            return Memory(
                org_id=config.get("org_id", "aura"),
                user_id=self.agent_id
            )
        except ImportError:
            # Fallback to LangChain memory
            return ConversationSummaryBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=config.get("max_tokens", 2000)
            )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("think", self.think_node)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("act", self.act_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("respond", self.respond_node)
        
        # Add edges
        workflow.set_entry_point("think")
        
        # Conditional edges based on thought type
        workflow.add_conditional_edges(
            "think",
            self.route_thought,
            {
                "plan": "plan",
                "act": "act",
                "reflect": "reflect",
                "respond": "respond"
            }
        )
        
        workflow.add_edge("plan", "act")
        workflow.add_edge("act", "reflect")
        workflow.add_edge("reflect", "respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def think_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate thoughts about the current state"""
        
        current_task = state["current_task"]
        messages = state["messages"]
        
        # Analyze the task and context
        thought = {
            "type": self._determine_thought_type(current_task, messages),
            "content": f"Analyzing task: {current_task}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.8
        }
        
        # Add thought to state
        thoughts = state.get("thoughts", [])
        thoughts.append(thought)
        
        logger.info(f"Agent thinking", thought_type=thought["type"])
        
        return {"thoughts": thoughts}
    
    def _determine_thought_type(self, task: str, messages: Sequence[BaseMessage]) -> str:
        """Determine what type of thinking is needed"""
        
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["plan", "strategy", "steps"]):
            return ThoughtType.PLANNING
        elif any(word in task_lower for word in ["do", "execute", "perform", "calculate"]):
            return ThoughtType.TOOL_SELECTION
        elif len(messages) > 5:  # Been working for a while
            return ThoughtType.REFLECTION
        else:
            return ThoughtType.REASONING
    
    def route_thought(self, state: AgentState) -> str:
        """Route based on the latest thought"""
        
        if not state.get("thoughts"):
            return "respond"
        
        latest_thought = state["thoughts"][-1]
        thought_type = latest_thought["type"]
        
        if thought_type == ThoughtType.PLANNING:
            return "plan"
        elif thought_type == ThoughtType.TOOL_SELECTION:
            return "act"
        elif thought_type == ThoughtType.REFLECTION:
            return "reflect"
        else:
            return "respond"
    
    async def plan_node(self, state: AgentState) -> Dict[str, Any]:
        """Create a plan for the task"""
        
        task = state["current_task"]
        
        # Simple planning - in practice would use LLM
        plan = [
            {"step": 1, "action": "analyze", "description": f"Understand the requirements of: {task}"},
            {"step": 2, "action": "execute", "description": "Execute necessary tools"},
            {"step": 3, "action": "verify", "description": "Verify the results"}
        ]
        
        logger.info(f"Plan created", num_steps=len(plan))
        
        return {"plan": plan}
    
    async def act_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on the plan"""
        
        plan = state.get("plan", [])
        tools_output = state.get("tools_output", [])
        
        # Execute next planned action
        if plan and len(tools_output) < len(plan):
            current_step = plan[len(tools_output)]
            
            # Select appropriate tool
            tool_name = self._select_tool(current_step, state["current_task"])
            
            if tool_name:
                # Execute tool
                tool_result = await self._execute_tool(tool_name, state["current_task"])
                tools_output.append({
                    "step": current_step["step"],
                    "tool": tool_name,
                    "result": tool_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"Tool executed", tool=tool_name, step=current_step["step"])
        
        return {"tools_output": tools_output}
    
    def _select_tool(self, step: Dict[str, Any], task: str) -> Optional[str]:
        """Select the appropriate tool for the step"""
        
        # Simple heuristic - in practice would use semantic matching
        action = step.get("action", "")
        
        for tool in self.tools:
            if action in tool.name or any(keyword in tool.description.lower() 
                                        for keyword in action.split()):
                return tool.name
        
        return None
    
    async def _execute_tool(self, tool_name: str, input_str: str) -> Any:
        """Execute a tool"""
        
        tool_invocation = ToolInvocation(
            tool=tool_name,
            tool_input={"input": input_str}
        )
        
        try:
            result = await self.tool_executor.ainvoke(tool_invocation)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed", tool=tool_name, error=str(e))
            return {"error": str(e)}
    
    async def reflect_node(self, state: AgentState) -> Dict[str, Any]:
        """Reflect on the actions taken"""
        
        tools_output = state.get("tools_output", [])
        thoughts = state.get("thoughts", [])
        
        # Analyze tool outputs
        successes = sum(1 for output in tools_output if "error" not in output.get("result", {}))
        total = len(tools_output)
        
        reflection = {
            "type": ThoughtType.REFLECTION,
            "content": f"Executed {total} tools with {successes} successes",
            "insights": self._generate_insights(tools_output),
            "timestamp": datetime.now().isoformat()
        }
        
        thoughts.append(reflection)
        
        logger.info(f"Reflection complete", successes=successes, total=total)
        
        return {"thoughts": thoughts}
    
    def _generate_insights(self, tools_output: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from tool outputs"""
        
        insights = []
        
        if not tools_output:
            insights.append("No tools were executed")
        elif all("error" not in output.get("result", {}) for output in tools_output):
            insights.append("All tools executed successfully")
        else:
            insights.append("Some tools encountered errors")
        
        return insights
    
    async def respond_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate final response"""
        
        task = state["current_task"]
        thoughts = state.get("thoughts", [])
        tools_output = state.get("tools_output", [])
        
        # Synthesize response
        response_parts = [f"Task: {task}"]
        
        if thoughts:
            response_parts.append(f"Thoughts: {len(thoughts)} generated")
        
        if tools_output:
            response_parts.append(f"Tools used: {[o['tool'] for o in tools_output]}")
            
            # Get final result
            if tools_output:
                last_result = tools_output[-1].get("result", {})
                if isinstance(last_result, dict) and "output" in last_result:
                    response_parts.append(f"Result: {last_result['output']}")
        
        final_answer = "\n".join(response_parts)
        
        # Store in memory
        await self._store_in_memory(state)
        
        # Add AI message
        messages = list(state["messages"])
        messages.append(AIMessage(content=final_answer))
        
        logger.info(f"Response generated", length=len(final_answer))
        
        return {
            "final_answer": final_answer,
            "messages": messages
        }
    
    async def _store_in_memory(self, state: AgentState):
        """Store the interaction in memory"""
        
        if hasattr(self.memory, 'add'):  # Mem0
            await self.memory.add(
                data=state["current_task"],
                metadata={
                    "agent_id": self.agent_id,
                    "thoughts": len(state.get("thoughts", [])),
                    "tools_used": len(state.get("tools_output", [])),
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:  # LangChain memory
            self.memory.save_context(
                {"input": state["current_task"]},
                {"output": state.get("final_answer", "")}
            )
    
    async def invoke(self, task: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the agent with a task"""
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_task": task,
            "thoughts": [],
            "plan": [],
            "tools_output": [],
            "final_answer": None,
            "metadata": {"agent_id": self.agent_id, "start_time": datetime.now().isoformat()}
        }
        
        # Run the graph
        config = config or {}
        if self.checkpointer:
            config["configurable"] = {"thread_id": str(uuid.uuid4())}
        
        result = await self.graph.ainvoke(initial_state, config)
        
        return result


def create_example_tools() -> List[Tool]:
    """Create example tools for the agent"""
    
    def calculate(expression: str) -> float:
        """Perform mathematical calculations"""
        try:
            # Safe evaluation
            allowed_names = {
                k: v for k, v in globals()["__builtins__"].items()
                if k in ["abs", "round", "min", "max", "sum", "pow"]
            }
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_data(data: str) -> Dict[str, Any]:
        """Analyze data and return insights"""
        return {
            "length": len(data),
            "words": len(data.split()),
            "summary": data[:100] + "..." if len(data) > 100 else data
        }
    
    return [
        Tool(
            name="calculator",
            description="Perform mathematical calculations",
            func=calculate
        ),
        Tool(
            name="analyzer",
            description="Analyze text data",
            func=analyze_data
        )
    ]


# Example usage
async def main():
    """Demonstrate the LangGraph agent"""
    
    print("🤖 AURA LangGraph Agent Demo")
    print("=" * 60)
    
    # Create tools
    tools = create_example_tools()
    
    # Create agent
    agent = AURAAgent(
        agent_id="aura-001",
        tools=tools,
        memory_config={"max_tokens": 1000},
        checkpoint_config={}
    )
    
    # Test tasks
    tasks = [
        "Calculate 15 * 23 + 45",
        "Analyze this text: The AURA system uses advanced AI techniques",
        "Plan how to deploy a web application"
    ]
    
    for task in tasks:
        print(f"\n📋 Task: {task}")
        print("-" * 40)
        
        result = await agent.invoke(task)
        
        print(f"Final Answer: {result['final_answer']}")
        print(f"Thoughts: {len(result['thoughts'])}")
        print(f"Tools Used: {len(result['tools_output'])}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())