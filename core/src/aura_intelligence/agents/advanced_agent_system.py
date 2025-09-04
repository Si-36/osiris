"""
Advanced Agent System - 2025 Implementation

Based on latest research:
- ReAct (Reasoning and Acting) pattern
- Chain-of-Thought (CoT) reasoning
- Tool use and function calling
- Multi-agent orchestration
- Memory-augmented agents
- Autonomous goal setting and planning
- Self-reflection and improvement

Key innovations:
- Hierarchical agent architectures
- Dynamic tool discovery and usage
- Collaborative multi-agent reasoning
- Long-term memory and context
- Adaptive behavior patterns
- Error recovery and self-healing
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime
import structlog
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
import uuid

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class AgentRole(str, Enum):
    """Agent roles in the system"""
    ORCHESTRATOR = "orchestrator"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    OBSERVER = "observer"
    PLANNER = "planner"
    REASONER = "reasoner"


class ThoughtType(str, Enum):
    """Types of thoughts in chain-of-thought"""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTION = "action"
    REFLECTION = "reflection"
    CRITIQUE = "critique"


@dataclass
class Thought:
    """A single thought in the reasoning chain"""
    type: ThoughtType
    content: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Tool:
    """Tool specification for agents"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    cost: float = 0.1  # Computational cost
    reliability: float = 0.95
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        try:
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed: {self.name}", error=str(e))
            raise


@dataclass
class AgentState:
    """Agent's internal state"""
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    memory_keys: List[str] = field(default_factory=list)


class AgentMemory:
    """Long-term memory for agents"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.episodic_memory: deque = deque(maxlen=capacity)
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
        
        # Memory embeddings for retrieval
        self.embedding_dim = 768
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def store_episode(self, episode: Dict[str, Any]):
        """Store an episodic memory"""
        episode_id = str(uuid.uuid4())
        episode['id'] = episode_id
        episode['timestamp'] = datetime.now()
        
        self.episodic_memory.append(episode)
        
        # Create embedding (simplified - would use real embedder)
        if 'content' in episode:
            self.embeddings[episode_id] = np.random.randn(self.embedding_dim)
    
    def retrieve_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar memories"""
        # Simplified similarity search
        # In practice, would use proper embedding and vector search
        
        query_embedding = np.random.randn(self.embedding_dim)
        
        similarities = []
        for episode in self.episodic_memory:
            if episode['id'] in self.embeddings:
                sim = np.dot(query_embedding, self.embeddings[episode['id']])
                similarities.append((sim, episode))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in similarities[:k]]
    
    def update_semantic(self, key: str, value: Any):
        """Update semantic memory"""
        self.semantic_memory[key] = value
    
    def get_semantic(self, key: str) -> Any:
        """Get from semantic memory"""
        return self.semantic_memory.get(key)


class BaseAgent(ABC):
    """Base agent with core capabilities"""
    
    def __init__(self, name: str, role: AgentRole, tools: List[Tool] = None):
        self.name = name
        self.role = role
        self.state = AgentState()
        self.memory = AgentMemory()
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.id = str(uuid.uuid4())
        
        logger.info(f"Agent initialized", name=name, role=role.value)
    
    async def think(self, observation: str) -> Thought:
        """Generate a thought based on observation"""
        # Chain-of-thought reasoning
        thought_type = self._determine_thought_type(observation)
        
        # Generate thought content
        content = await self._generate_thought_content(observation, thought_type)
        
        thought = Thought(
            type=thought_type,
            content=content,
            confidence=self._calculate_confidence(content)
        )
        
        self.state.thoughts.append(thought)
        return thought
    
    def _determine_thought_type(self, observation: str) -> ThoughtType:
        """Determine what type of thought is needed"""
        # Simple heuristic - would use NLP in practice
        if "error" in observation.lower() or "failed" in observation.lower():
            return ThoughtType.REFLECTION
        elif "plan" in observation.lower() or "goal" in observation.lower():
            return ThoughtType.PLANNING
        elif "?" in observation:
            return ThoughtType.REASONING
        else:
            return ThoughtType.OBSERVATION
    
    async def _generate_thought_content(self, observation: str, 
                                      thought_type: ThoughtType) -> str:
        """Generate thought content based on type"""
        # Simplified - would use LLM in practice
        templates = {
            ThoughtType.OBSERVATION: f"I observe that: {observation}",
            ThoughtType.REASONING: f"Based on '{observation}', I reason that...",
            ThoughtType.PLANNING: f"To address '{observation}', I should plan to...",
            ThoughtType.REFLECTION: f"Reflecting on '{observation}', I realize...",
            ThoughtType.ACTION: f"I will take action regarding: {observation}",
            ThoughtType.CRITIQUE: f"Critiquing '{observation}', I find..."
        }
        
        return templates.get(thought_type, f"Thinking about: {observation}")
    
    def _calculate_confidence(self, content: str) -> float:
        """Calculate confidence in thought"""
        # Simplified - would use uncertainty quantification
        if "uncertain" in content or "maybe" in content:
            return 0.5
        elif "definitely" in content or "certain" in content:
            return 0.95
        else:
            return 0.8
    
    async def act(self, thought: Thought) -> Optional[Dict[str, Any]]:
        """Take action based on thought"""
        if thought.type != ThoughtType.ACTION:
            return None
        
        # Select appropriate tool
        tool_name = self._select_tool(thought.content)
        if not tool_name or tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        
        # Execute tool
        try:
            result = await tool.execute(content=thought.content)
            
            action = {
                'tool': tool_name,
                'input': thought.content,
                'output': result,
                'timestamp': datetime.now()
            }
            
            self.state.actions.append(action)
            return action
            
        except Exception as e:
            logger.error(f"Action failed", tool=tool_name, error=str(e))
            return None
    
    def _select_tool(self, content: str) -> Optional[str]:
        """Select appropriate tool for action"""
        # Simplified - would use semantic matching
        for tool_name, tool in self.tools.items():
            if any(keyword in content.lower() 
                   for keyword in tool.description.lower().split()):
                return tool_name
        return None
    
    async def reflect(self) -> Thought:
        """Reflect on recent actions and observations"""
        recent_thoughts = self.state.thoughts[-5:]
        recent_actions = self.state.actions[-3:]
        
        # Generate reflection
        reflection_content = self._generate_reflection(recent_thoughts, recent_actions)
        
        reflection = Thought(
            type=ThoughtType.REFLECTION,
            content=reflection_content,
            confidence=0.9
        )
        
        self.state.thoughts.append(reflection)
        return reflection
    
    def _generate_reflection(self, thoughts: List[Thought], 
                           actions: List[Dict[str, Any]]) -> str:
        """Generate reflection content"""
        # Simplified reflection
        if not actions:
            return "I haven't taken any actions yet. I should be more proactive."
        
        success_rate = sum(1 for a in actions if a.get('output')) / len(actions)
        
        if success_rate > 0.8:
            return "My recent actions have been successful. I should continue this approach."
        elif success_rate > 0.5:
            return "Some actions succeeded, others failed. I need to analyze why."
        else:
            return "Most actions failed. I need to reconsider my approach."
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input and produce output"""
        pass


class ReActAgent(BaseAgent):
    """Agent using ReAct (Reasoning and Acting) pattern"""
    
    def __init__(self, name: str, tools: List[Tool] = None, max_steps: int = 10):
        super().__init__(name, AgentRole.EXECUTOR, tools)
        self.max_steps = max_steps
    
    async def process(self, task: str) -> Dict[str, Any]:
        """Process task using ReAct loop"""
        logger.info(f"ReAct agent processing task", agent=self.name, task=task)
        
        # Initialize
        observation = f"Task: {task}"
        self.state.goals = [task]
        
        for step in range(self.max_steps):
            # Think
            thought = await self.think(observation)
            
            # Decide if action is needed
            if thought.type == ThoughtType.ACTION:
                # Act
                action = await self.act(thought)
                if action:
                    observation = f"Action result: {action['output']}"
                else:
                    observation = "Action failed or no appropriate tool found"
            
            # Check if task is complete
            if self._is_task_complete(task, self.state):
                break
            
            # Reflect periodically
            if step % 3 == 2:
                reflection = await self.reflect()
                observation = f"Reflection: {reflection.content}"
        
        return {
            'task': task,
            'thoughts': [t.content for t in self.state.thoughts],
            'actions': self.state.actions,
            'completed': self._is_task_complete(task, self.state)
        }
    
    def _is_task_complete(self, task: str, state: AgentState) -> bool:
        """Check if task is complete"""
        # Simplified - would use task-specific completion criteria
        return len(state.actions) > 0 and state.actions[-1].get('output') is not None


class PlannerAgent(BaseAgent):
    """Agent specialized in planning"""
    
    def __init__(self, name: str):
        super().__init__(name, AgentRole.PLANNER)
    
    async def process(self, goal: str) -> Dict[str, Any]:
        """Create plan for achieving goal"""
        logger.info(f"Planner agent creating plan", agent=self.name, goal=goal)
        
        # Decompose goal into sub-goals
        sub_goals = await self._decompose_goal(goal)
        
        # Create step-by-step plan
        plan = []
        for sub_goal in sub_goals:
            steps = await self._plan_steps(sub_goal)
            plan.extend(steps)
        
        # Optimize plan
        optimized_plan = self._optimize_plan(plan)
        
        return {
            'goal': goal,
            'sub_goals': sub_goals,
            'plan': optimized_plan,
            'estimated_steps': len(optimized_plan)
        }
    
    async def _decompose_goal(self, goal: str) -> List[str]:
        """Decompose goal into sub-goals"""
        # Simplified - would use goal decomposition algorithms
        if "and" in goal:
            return goal.split(" and ")
        else:
            return [goal]
    
    async def _plan_steps(self, sub_goal: str) -> List[Dict[str, Any]]:
        """Plan steps for sub-goal"""
        # Simplified planning
        return [
            {'step': 1, 'action': 'analyze', 'target': sub_goal},
            {'step': 2, 'action': 'execute', 'target': sub_goal},
            {'step': 3, 'action': 'verify', 'target': sub_goal}
        ]
    
    def _optimize_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize plan by removing redundancies"""
        # Simplified - would use plan optimization algorithms
        seen = set()
        optimized = []
        
        for step in plan:
            key = (step['action'], step['target'])
            if key not in seen:
                seen.add(key)
                optimized.append(step)
        
        return optimized


class MultiAgentOrchestrator:
    """Orchestrates multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.communication_channels: Dict[str, List[str]] = defaultdict(list)
        self.shared_memory = AgentMemory(capacity=50000)
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.id] = agent
        logger.info(f"Agent registered", agent=agent.name, role=agent.role.value)
    
    def establish_communication(self, agent1_id: str, agent2_id: str):
        """Establish bidirectional communication between agents"""
        self.communication_channels[agent1_id].append(agent2_id)
        self.communication_channels[agent2_id].append(agent1_id)
    
    async def delegate_task(self, task: str, agent_id: str) -> Any:
        """Delegate task to specific agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        result = await agent.process(task)
        
        # Store in shared memory
        self.shared_memory.store_episode({
            'agent': agent.name,
            'task': task,
            'result': result,
            'content': str(result)
        })
        
        return result
    
    async def collaborative_task(self, task: str, 
                               agent_roles: List[AgentRole]) -> Dict[str, Any]:
        """Execute task collaboratively with multiple agents"""
        logger.info(f"Starting collaborative task", task=task, 
                   roles=[r.value for r in agent_roles])
        
        # Find agents with required roles
        selected_agents = []
        for role in agent_roles:
            for agent in self.agents.values():
                if agent.role == role:
                    selected_agents.append(agent)
                    break
        
        if len(selected_agents) != len(agent_roles):
            raise ValueError("Not all required roles have agents")
        
        # Execute task pipeline
        results = {}
        current_input = task
        
        for agent in selected_agents:
            result = await agent.process(current_input)
            results[agent.name] = result
            
            # Pass output to next agent
            if isinstance(result, dict) and 'output' in result:
                current_input = result['output']
            else:
                current_input = str(result)
        
        return {
            'task': task,
            'agents': [a.name for a in selected_agents],
            'results': results,
            'final_output': current_input
        }
    
    async def consensus_decision(self, question: str, 
                               min_agents: int = 3) -> Dict[str, Any]:
        """Get consensus decision from multiple agents"""
        # Select diverse agents
        selected = []
        roles_seen = set()
        
        for agent in self.agents.values():
            if agent.role not in roles_seen:
                selected.append(agent)
                roles_seen.add(agent.role)
                if len(selected) >= min_agents:
                    break
        
        # Get individual decisions
        decisions = {}
        for agent in selected:
            thought = await agent.think(question)
            decisions[agent.name] = {
                'thought': thought.content,
                'confidence': thought.confidence
            }
        
        # Aggregate decisions
        consensus = self._aggregate_decisions(decisions)
        
        return {
            'question': question,
            'decisions': decisions,
            'consensus': consensus
        }
    
    def _aggregate_decisions(self, decisions: Dict[str, Dict[str, Any]]) -> str:
        """Aggregate multiple decisions into consensus"""
        # Simplified - would use voting or more sophisticated aggregation
        
        # Weight by confidence
        total_confidence = sum(d['confidence'] for d in decisions.values())
        
        if total_confidence == 0:
            return "No confident decision could be made"
        
        # Find most confident decision
        best_agent = max(decisions.items(), 
                        key=lambda x: x[1]['confidence'])[0]
        
        return f"Consensus (led by {best_agent}): {decisions[best_agent]['thought']}"


# Example tools
def create_example_tools() -> List[Tool]:
    """Create example tools for agents"""
    
    def calculate(expression: str) -> float:
        """Simple calculator"""
        try:
            return eval(expression)
        except:
            return 0.0
    
    def search(query: str) -> str:
        """Mock search function"""
        return f"Search results for: {query}"
    
    async def analyze(content: str) -> Dict[str, Any]:
        """Mock analysis function"""
        return {
            'sentiment': 'positive' if 'good' in content else 'neutral',
            'entities': ['entity1', 'entity2'],
            'summary': content[:50] + '...'
        }
    
    return [
        Tool(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={'expression': str},
            function=calculate
        ),
        Tool(
            name="search",
            description="Search for information",
            parameters={'query': str},
            function=search
        ),
        Tool(
            name="analyzer",
            description="Analyze text content",
            parameters={'content': str},
            function=analyze
        )
    ]


# Demonstration
async def demonstrate_agent_system():
    """Demonstrate the agent system"""
    print("🤖 Advanced Agent System Demonstration")
    print("=" * 60)
    
    # Create tools
    tools = create_example_tools()
    
    # Create agents
    react_agent = ReActAgent("ReactBot", tools)
    planner = PlannerAgent("PlannerBot")
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    orchestrator.register_agent(react_agent)
    orchestrator.register_agent(planner)
    
    # Test 1: Single agent task
    print("\n1️⃣ Single Agent Task (ReAct)")
    result = await react_agent.process("Calculate 15 * 23 + 45")
    print(f"Result: {result}")
    
    # Test 2: Planning
    print("\n2️⃣ Planning Agent")
    plan = await planner.process("Build a web application and deploy it")
    print(f"Plan: {json.dumps(plan, indent=2)}")
    
    # Test 3: Orchestrated task
    print("\n3️⃣ Orchestrated Multi-Agent Task")
    collab_result = await orchestrator.collaborative_task(
        "Analyze market trends and create investment strategy",
        [AgentRole.PLANNER, AgentRole.EXECUTOR]
    )
    print(f"Collaborative Result: {json.dumps(collab_result, indent=2)}")
    
    print("\n" + "=" * 60)
    print("✅ Agent System Demonstration Complete")


if __name__ == "__main__":
    asyncio.run(demonstrate_agent_system())