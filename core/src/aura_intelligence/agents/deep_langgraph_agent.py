"""
Deep LangGraph Agent with Advanced Capabilities

Features:
- Multi-step reasoning with LangGraph
- Tool calling with function execution
- Memory integration (Mem0 + LangChain)
- Checkpointing and state persistence
- Parallel tool execution
- Conditional routing based on confidence
- Human-in-the-loop capabilities
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict, Sequence, Union
import operator
from enum import Enum
import asyncio
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import structlog
import numpy as np

logger = structlog.get_logger(__name__)


class AgentDecision(BaseModel):
    """Decision made by the agent"""
    action: str = Field(description="The action to take: think, plan, use_tools, ask_human, or respond")
    reasoning: str = Field(description="Reasoning behind the decision")
    confidence: float = Field(description="Confidence in the decision (0-1)")
    tools_needed: List[str] = Field(default_factory=list, description="Tools needed for the action")
    

class DeepAgentState(TypedDict):
    """Enhanced state for deep reasoning agents"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_task: str
    subtasks: List[Dict[str, Any]]
    thoughts: List[Dict[str, Any]]
    decisions: List[AgentDecision]
    tool_calls: List[ToolInvocation]
    tool_results: List[Dict[str, Any]]
    memory_context: List[Dict[str, Any]]
    confidence_threshold: float
    max_iterations: int
    current_iteration: int
    requires_human: bool
    final_response: Optional[str]
    metadata: Dict[str, Any]


class DeepLangGraphAgent:
    """Advanced LangGraph agent with deep reasoning capabilities"""
    
    def __init__(self,
                 agent_id: str,
                 tools: List[BaseTool],
                 model: Optional[Any] = None,  # LLM model
                 memory: Optional[Any] = None,  # Memory system
                 checkpoint_dir: str = "./checkpoints",
                 confidence_threshold: float = 0.7,
                 max_iterations: int = 10):
        
        self.agent_id = agent_id
        self.tools = {tool.name: tool for tool in tools}
        self.tool_executor = ToolExecutor(tools)
        self.model = model
        self.memory = memory
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        
        # Set up checkpointing
        self.checkpointer = SqliteSaver.from_conn_string(f"{checkpoint_dir}/{agent_id}.db")
        
        # Build the workflow
        self.graph = self._build_workflow()
        
        logger.info(f"Deep LangGraph Agent initialized",
                   agent_id=agent_id,
                   num_tools=len(tools),
                   confidence_threshold=confidence_threshold)
    
    def _build_workflow(self) -> StateGraph:
        """Build the advanced workflow graph"""
        
        workflow = StateGraph(DeepAgentState)
        
        # Add nodes
        workflow.add_node("retrieve_memory", self.retrieve_memory_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("decide", self.decide_node)
        workflow.add_node("decompose", self.decompose_node)
        workflow.add_node("use_tools", self.use_tools_node)
        workflow.add_node("evaluate", self.evaluate_node)
        workflow.add_node("ask_human", self.ask_human_node)
        workflow.add_node("synthesize", self.synthesize_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve_memory")
        
        # Add edges
        workflow.add_edge("retrieve_memory", "analyze")
        workflow.add_edge("analyze", "decide")
        
        # Conditional routing from decide node
        workflow.add_conditional_edges(
            "decide",
            self.route_decision,
            {
                "decompose": "decompose",
                "use_tools": "use_tools",
                "ask_human": "ask_human",
                "synthesize": "synthesize",
                "continue": "analyze"
            }
        )
        
        workflow.add_edge("decompose", "analyze")
        workflow.add_edge("use_tools", "evaluate")
        workflow.add_edge("evaluate", "decide")
        workflow.add_edge("ask_human", "analyze")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def retrieve_memory_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Retrieve relevant memories for the task"""
        
        task = state["current_task"]
        memory_context = []
        
        if self.memory:
            try:
                # Try Mem0 API
                if hasattr(self.memory, 'search'):
                    results = await self.memory.search(task, limit=5)
                    memory_context = [
                        {
                            "content": r.get("data", ""),
                            "metadata": r.get("metadata", {}),
                            "relevance": r.get("score", 0.0)
                        }
                        for r in results
                    ]
                # Fallback to LangChain memory
                elif hasattr(self.memory, 'buffer'):
                    buffer_content = self.memory.buffer
                    if buffer_content:
                        memory_context.append({
                            "content": buffer_content,
                            "metadata": {"source": "conversation_buffer"},
                            "relevance": 0.8
                        })
            except Exception as e:
                logger.error(f"Memory retrieval failed", error=str(e))
        
        logger.info(f"Retrieved memories", count=len(memory_context))
        
        return {"memory_context": memory_context}
    
    async def analyze_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Analyze the current state and generate thoughts"""
        
        task = state["current_task"]
        subtasks = state.get("subtasks", [])
        previous_thoughts = state.get("thoughts", [])
        memory_context = state.get("memory_context", [])
        
        # Generate new thought based on current state
        thought = {
            "id": len(previous_thoughts) + 1,
            "type": "analysis",
            "content": self._generate_thought(task, subtasks, memory_context),
            "timestamp": datetime.now().isoformat(),
            "iteration": state.get("current_iteration", 0)
        }
        
        thoughts = previous_thoughts + [thought]
        
        # Update iteration count
        current_iteration = state.get("current_iteration", 0) + 1
        
        logger.info(f"Analysis complete", 
                   thought_id=thought["id"],
                   iteration=current_iteration)
        
        return {
            "thoughts": thoughts,
            "current_iteration": current_iteration
        }
    
    def _generate_thought(self, task: str, subtasks: List[Dict], 
                         memory: List[Dict]) -> str:
        """Generate a thought about the current state"""
        
        # In practice, would use LLM
        if not subtasks:
            return f"Need to understand and decompose the task: {task}"
        elif all(st.get("completed") for st in subtasks):
            return "All subtasks completed, ready to synthesize response"
        else:
            incomplete = [st for st in subtasks if not st.get("completed")]
            return f"Need to complete {len(incomplete)} remaining subtasks"
    
    async def decide_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Make a decision about what to do next"""
        
        thoughts = state.get("thoughts", [])
        subtasks = state.get("subtasks", [])
        tool_results = state.get("tool_results", [])
        current_iteration = state.get("current_iteration", 0)
        
        # Make decision based on state
        if current_iteration >= self.max_iterations:
            decision = AgentDecision(
                action="synthesize",
                reasoning="Reached maximum iterations",
                confidence=0.6
            )
        elif not subtasks:
            decision = AgentDecision(
                action="decompose",
                reasoning="Need to break down the task",
                confidence=0.9
            )
        elif self._needs_tools(subtasks):
            tools_needed = self._identify_tools(subtasks)
            decision = AgentDecision(
                action="use_tools",
                reasoning="Tools required for next steps",
                confidence=0.85,
                tools_needed=tools_needed
            )
        elif self._all_complete(subtasks):
            decision = AgentDecision(
                action="synthesize",
                reasoning="All subtasks completed",
                confidence=0.95
            )
        else:
            decision = AgentDecision(
                action="continue",
                reasoning="Continue processing subtasks",
                confidence=0.8
            )
        
        # Check if human input needed
        if decision.confidence < self.confidence_threshold:
            decision.action = "ask_human"
            decision.reasoning = f"Low confidence ({decision.confidence:.2f}), need human input"
        
        decisions = state.get("decisions", []) + [decision]
        
        logger.info(f"Decision made",
                   action=decision.action,
                   confidence=decision.confidence)
        
        return {"decisions": decisions}
    
    def route_decision(self, state: DeepAgentState) -> str:
        """Route based on the latest decision"""
        
        decisions = state.get("decisions", [])
        if not decisions:
            return "continue"
        
        latest_decision = decisions[-1]
        return latest_decision.action
    
    async def decompose_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Decompose task into subtasks"""
        
        task = state["current_task"]
        
        # Simple decomposition - in practice would use LLM
        subtasks = [
            {
                "id": 1,
                "description": f"Parse and understand: {task}",
                "completed": False,
                "requires_tool": False
            },
            {
                "id": 2,
                "description": f"Execute main operation for: {task}",
                "completed": False,
                "requires_tool": True
            },
            {
                "id": 3,
                "description": "Validate and format results",
                "completed": False,
                "requires_tool": False
            }
        ]
        
        logger.info(f"Task decomposed", num_subtasks=len(subtasks))
        
        return {"subtasks": subtasks}
    
    async def use_tools_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Execute tools based on current needs"""
        
        decisions = state.get("decisions", [])
        latest_decision = decisions[-1] if decisions else None
        tools_needed = latest_decision.tools_needed if latest_decision else []
        
        tool_calls = []
        tool_results = state.get("tool_results", [])
        
        # Execute tools in parallel
        tasks = []
        for tool_name in tools_needed:
            if tool_name in self.tools:
                tool_call = ToolInvocation(
                    tool=tool_name,
                    tool_input={"query": state["current_task"]}
                )
                tool_calls.append(tool_call)
                tasks.append(self._execute_tool_async(tool_call))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for tool_call, result in zip(tool_calls, results):
                tool_results.append({
                    "tool": tool_call.tool,
                    "input": tool_call.tool_input,
                    "output": result if not isinstance(result, Exception) else str(result),
                    "success": not isinstance(result, Exception),
                    "timestamp": datetime.now().isoformat()
                })
        
        logger.info(f"Tools executed", count=len(tool_results))
        
        return {
            "tool_calls": tool_calls,
            "tool_results": tool_results
        }
    
    async def _execute_tool_async(self, tool_call: ToolInvocation) -> Any:
        """Execute a single tool asynchronously"""
        try:
            result = await self.tool_executor.ainvoke(tool_call)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed", 
                        tool=tool_call.tool,
                        error=str(e))
            return {"error": str(e)}
    
    async def evaluate_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Evaluate tool results and update subtasks"""
        
        tool_results = state.get("tool_results", [])
        subtasks = state.get("subtasks", [])
        
        # Update subtask completion based on tool results
        if tool_results:
            successful_tools = sum(1 for r in tool_results if r.get("success", False))
            
            # Mark tool-requiring subtasks as complete if tools succeeded
            for subtask in subtasks:
                if subtask.get("requires_tool") and not subtask.get("completed"):
                    subtask["completed"] = successful_tools > 0
        
        logger.info(f"Evaluation complete",
                   completed_subtasks=sum(1 for st in subtasks if st.get("completed")))
        
        return {"subtasks": subtasks}
    
    async def ask_human_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Request human input"""
        
        # In practice, would integrate with UI/API
        logger.warning("Human input requested but not available in demo mode")
        
        # For demo, add a message indicating human input needed
        messages = list(state["messages"])
        messages.append(AIMessage(
            content="I need human assistance to proceed with confidence. "
                   "The task complexity exceeds my confidence threshold."
        ))
        
        return {
            "messages": messages,
            "requires_human": True
        }
    
    async def synthesize_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Synthesize final response from all information"""
        
        task = state["current_task"]
        thoughts = state.get("thoughts", [])
        tool_results = state.get("tool_results", [])
        subtasks = state.get("subtasks", [])
        
        # Build response
        response_parts = [f"Task completed: {task}"]
        
        # Add subtask summary
        completed = sum(1 for st in subtasks if st.get("completed"))
        response_parts.append(f"Subtasks: {completed}/{len(subtasks)} completed")
        
        # Add tool results summary
        if tool_results:
            successful = sum(1 for r in tool_results if r.get("success"))
            response_parts.append(f"Tools used: {successful}/{len(tool_results)} successful")
            
            # Include actual results
            for result in tool_results:
                if result.get("success") and "output" in result:
                    response_parts.append(f"Result: {result['output']}")
        
        # Add insights from thoughts
        if thoughts:
            latest_thought = thoughts[-1]
            response_parts.append(f"Analysis: {latest_thought['content']}")
        
        final_response = "\n".join(response_parts)
        
        # Store in memory
        if self.memory:
            await self._store_interaction(state, final_response)
        
        # Add final message
        messages = list(state["messages"])
        messages.append(AIMessage(content=final_response))
        
        logger.info(f"Response synthesized", length=len(final_response))
        
        return {
            "final_response": final_response,
            "messages": messages
        }
    
    async def _store_interaction(self, state: DeepAgentState, response: str):
        """Store the interaction in memory"""
        
        try:
            if hasattr(self.memory, 'add'):  # Mem0
                await self.memory.add(
                    messages=[
                        {"role": "user", "content": state["current_task"]},
                        {"role": "assistant", "content": response}
                    ],
                    metadata={
                        "agent_id": self.agent_id,
                        "thoughts": len(state.get("thoughts", [])),
                        "tools_used": len(state.get("tool_results", [])),
                        "iterations": state.get("current_iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            elif hasattr(self.memory, 'save_context'):  # LangChain
                self.memory.save_context(
                    {"input": state["current_task"]},
                    {"output": response}
                )
        except Exception as e:
            logger.error(f"Memory storage failed", error=str(e))
    
    def _needs_tools(self, subtasks: List[Dict]) -> bool:
        """Check if any subtask needs tools"""
        return any(st.get("requires_tool") and not st.get("completed") 
                  for st in subtasks)
    
    def _identify_tools(self, subtasks: List[Dict]) -> List[str]:
        """Identify which tools are needed"""
        # Simple heuristic - in practice would use semantic matching
        tools_needed = []
        
        for subtask in subtasks:
            if subtask.get("requires_tool") and not subtask.get("completed"):
                desc = subtask.get("description", "").lower()
                
                for tool_name in self.tools:
                    if tool_name in desc or any(word in desc 
                                               for word in tool_name.split("_")):
                        tools_needed.append(tool_name)
        
        return list(set(tools_needed))  # Remove duplicates
    
    def _all_complete(self, subtasks: List[Dict]) -> bool:
        """Check if all subtasks are complete"""
        return all(st.get("completed") for st in subtasks)
    
    async def invoke(self, task: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Invoke the agent with a task"""
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "current_task": task,
            "subtasks": [],
            "thoughts": [],
            "decisions": [],
            "tool_calls": [],
            "tool_results": [],
            "memory_context": [],
            "confidence_threshold": self.confidence_threshold,
            "max_iterations": self.max_iterations,
            "current_iteration": 0,
            "requires_human": False,
            "final_response": None,
            "metadata": {
                "agent_id": self.agent_id,
                "thread_id": thread_id,
                "start_time": datetime.now().isoformat()
            }
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        
        result = await self.graph.ainvoke(initial_state, config)
        
        return result


# Example usage
async def main():
    """Demo the deep LangGraph agent"""
    
    print("🧠 Deep LangGraph Agent Demo")
    print("=" * 60)
    
    # Create simple tools
    from langchain_core.tools import Tool
    
    def calculate(expression: str) -> str:
        """Calculate mathematical expression"""
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def search_knowledge(query: str) -> str:
        """Search knowledge base"""
        # Mock search
        return f"Found information about: {query}"
    
    tools = [
        Tool(name="calculator", description="Calculate math expressions", func=calculate),
        Tool(name="search", description="Search for information", func=search_knowledge)
    ]
    
    # Create agent
    agent = DeepLangGraphAgent(
        agent_id="deep-agent-001",
        tools=tools,
        confidence_threshold=0.7,
        max_iterations=5
    )
    
    # Test tasks
    tasks = [
        "Calculate the compound interest on $1000 at 5% for 10 years",
        "Search for information about quantum computing and summarize it"
    ]
    
    for task in tasks:
        print(f"\n📋 Task: {task}")
        print("-" * 60)
        
        result = await agent.invoke(task, thread_id=f"demo-{task[:10]}")
        
        print(f"\n✅ Final Response:")
        print(result.get("final_response", "No response generated"))
        
        print(f"\n📊 Stats:")
        print(f"  Iterations: {result.get('current_iteration', 0)}")
        print(f"  Thoughts: {len(result.get('thoughts', []))}")
        print(f"  Decisions: {len(result.get('decisions', []))}")
        print(f"  Tools Used: {len(result.get('tool_results', []))}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())