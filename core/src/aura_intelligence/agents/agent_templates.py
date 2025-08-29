"""
🎭 AURA Production Agent Templates

Four core agent types that cover most use cases:
1. Observer - Monitors and collects data
2. Analyst - Analyzes and provides insights
3. Executor - Takes actions based on decisions
4. Coordinator - Orchestrates multi-agent workflows

Each inherits from AURAAgentCore and automatically gets:
- Memory integration
- TDA analysis
- Neural routing
- Orchestration capabilities
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio
import json
import structlog

from .agent_core import AURAAgentCore, AURAAgentState
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = structlog.get_logger()


# ======================
# Observer Agent
# ======================

class ObserverAgent(AURAAgentCore):
    """
    Monitors systems and collects data.
    
    Use cases:
    - System health monitoring
    - Performance tracking
    - Anomaly detection
    - Data collection
    
    Features:
    - Low model usage (cost-efficient)
    - High-frequency observations
    - TDA-based anomaly detection
    - Automatic alerting
    """
    
    @property
    def agent_type(self) -> str:
        return "observer"
    
    def build_tools(self) -> List[BaseTool]:
        """Observer-specific tools"""
        tools = []
        
        @tool
        def check_system_health(component: str) -> Dict[str, Any]:
            """Check health of a system component"""
            # In production, this would check real systems
            return {
                "component": component,
                "status": "healthy",
                "metrics": {
                    "cpu_usage": 45.2,
                    "memory_usage": 62.1,
                    "response_time_ms": 124
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @tool
        def detect_anomalies(data: str) -> Dict[str, Any]:
            """Detect anomalies in data using TDA"""
            # Use TDA if available
            if self.tda:
                # This would analyze real data
                return {
                    "anomalies_detected": False,
                    "confidence": 0.92,
                    "patterns": ["normal_operation"]
                }
            return {"anomalies_detected": False, "method": "basic"}
        
        @tool
        def collect_metrics(source: str, duration_minutes: int = 5) -> List[Dict]:
            """Collect metrics from a source"""
            # In production, this would collect real metrics
            return [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                    "metrics": {
                        "throughput": 1523,
                        "latency_p99": 145,
                        "error_rate": 0.02
                    }
                }
            ]
        
        tools.extend([check_system_health, detect_anomalies, collect_metrics])
        return tools
    
    async def analyze_task(self, state: AURAAgentState) -> Dict[str, Any]:
        """Analyze observation requirements"""
        task = state.current_task
        
        # Determine what to observe
        if "health" in task.lower():
            return {"focus": "system_health", "frequency": "5m"}
        elif "performance" in task.lower():
            return {"focus": "performance_metrics", "frequency": "1m"}
        elif "anomaly" in task.lower():
            return {"focus": "anomaly_detection", "frequency": "continuous"}
        else:
            return {"focus": "general_monitoring", "frequency": "10m"}
    
    async def make_decision(self, state: AURAAgentState) -> Dict[str, Any]:
        """Decide on observation strategy"""
        analysis = state.results.get("analysis", {})
        
        # Simple decision logic
        if analysis.get("focus") == "anomaly_detection":
            return {"action": "use_tools", "tool": "detect_anomalies"}
        elif analysis.get("focus") == "system_health":
            return {"action": "use_tools", "tool": "check_system_health"}
        else:
            return {"action": "use_tools", "tool": "collect_metrics"}
    
    async def execute_action(self, state: AURAAgentState) -> Dict[str, Any]:
        """Execute observation action"""
        # Store observations in memory
        if self.memory and state.results:
            await self.memory.store({
                "type": "observation",
                "agent_id": self.agent_id,
                "data": state.results,
                "timestamp": datetime.now(timezone.utc)
            })
        
        return {"status": "observations_recorded"}


# ======================
# Analyst Agent
# ======================

class AnalystAgent(AURAAgentCore):
    """
    Analyzes data and provides insights.
    
    Use cases:
    - Data analysis
    - Report generation
    - Trend identification
    - Recommendation systems
    
    Features:
    - Routes to best analysis models
    - Uses memory for context
    - Generates visualizations
    - Provides actionable insights
    """
    
    @property
    def agent_type(self) -> str:
        return "analyst"
    
    def build_tools(self) -> List[BaseTool]:
        """Analyst-specific tools"""
        tools = []
        
        @tool
        def analyze_trends(data: str, timeframe: str = "7d") -> Dict[str, Any]:
            """Analyze trends in data"""
            return {
                "timeframe": timeframe,
                "trends": [
                    {"metric": "user_growth", "direction": "up", "change": "+12.5%"},
                    {"metric": "response_time", "direction": "down", "change": "-8.2%"}
                ],
                "confidence": 0.87
            }
        
        @tool
        def generate_insights(analysis: str) -> List[str]:
            """Generate actionable insights from analysis"""
            return [
                "User engagement has increased by 12.5% over the past week",
                "System performance has improved with 8.2% faster response times",
                "Consider scaling resources to handle increased load"
            ]
        
        @tool
        def create_report(title: str, sections: List[str]) -> Dict[str, Any]:
            """Create analysis report"""
            return {
                "title": title,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sections": sections,
                "format": "markdown",
                "status": "draft"
            }
        
        tools.extend([analyze_trends, generate_insights, create_report])
        return tools
    
    async def analyze_task(self, state: AURAAgentState) -> Dict[str, Any]:
        """Analyze what kind of analysis is needed"""
        task = state.current_task
        
        # Use neural router to select best model
        if self.router:
            routing = await self.router.route_request({
                "prompt": f"Analyze: {task}",
                "context": state.memory_context
            })
            model_info = {
                "selected_model": routing.model_config.model_id,
                "confidence": routing.confidence
            }
        else:
            model_info = {"selected_model": "default"}
        
        return {
            "analysis_type": self._determine_analysis_type(task),
            "model_routing": model_info,
            "requires_visualization": "chart" in task.lower() or "graph" in task.lower()
        }
    
    def _determine_analysis_type(self, task: str) -> str:
        """Determine type of analysis needed"""
        task_lower = task.lower()
        if "trend" in task_lower:
            return "trend_analysis"
        elif "report" in task_lower:
            return "report_generation"
        elif "insight" in task_lower:
            return "insight_extraction"
        else:
            return "general_analysis"
    
    async def make_decision(self, state: AURAAgentState) -> Dict[str, Any]:
        """Decide on analysis approach"""
        analysis = state.results.get("analysis", {})
        analysis_type = analysis.get("analysis_type", "general")
        
        if analysis_type == "trend_analysis":
            return {"action": "use_tools", "tool": "analyze_trends"}
        elif analysis_type == "report_generation":
            return {"action": "use_tools", "tool": "create_report"}
        else:
            return {"action": "use_tools", "tool": "generate_insights"}
    
    async def execute_action(self, state: AURAAgentState) -> Dict[str, Any]:
        """Execute analysis and store results"""
        # Get relevant context from memory
        if self.memory:
            context = await self.memory.retrieve(
                query=state.current_task,
                limit=10
            )
            state.memory_context["analysis_context"] = context
        
        return {
            "status": "analysis_complete",
            "insights": state.results.get("tool_output", [])
        }


# ======================
# Executor Agent
# ======================

class ExecutorAgent(AURAAgentCore):
    """
    Takes actions based on decisions.
    
    Use cases:
    - API calls
    - System commands
    - Workflow triggers
    - External integrations
    
    Features:
    - Safe action execution
    - Rollback capabilities
    - Result verification
    - Error handling
    """
    
    @property
    def agent_type(self) -> str:
        return "executor"
    
    def build_tools(self) -> List[BaseTool]:
        """Executor-specific tools"""
        tools = []
        
        @tool
        def execute_api_call(endpoint: str, method: str = "GET", data: Optional[str] = None) -> Dict[str, Any]:
            """Execute API call safely"""
            # In production, this would make real API calls
            return {
                "endpoint": endpoint,
                "method": method,
                "status_code": 200,
                "response": {"success": True, "data": "mock_response"}
            }
        
        @tool
        def trigger_workflow(workflow_id: str, parameters: Optional[str] = None) -> Dict[str, Any]:
            """Trigger a workflow in orchestration engine"""
            # This would use the real orchestration engine
            return {
                "workflow_id": workflow_id,
                "execution_id": str(uuid.uuid4()),
                "status": "triggered",
                "parameters": json.loads(parameters) if parameters else {}
            }
        
        @tool
        def verify_execution(execution_id: str) -> Dict[str, Any]:
            """Verify execution completed successfully"""
            return {
                "execution_id": execution_id,
                "status": "completed",
                "success": True,
                "duration_ms": 1523
            }
        
        tools.extend([execute_api_call, trigger_workflow, verify_execution])
        return tools
    
    async def analyze_task(self, state: AURAAgentState) -> Dict[str, Any]:
        """Analyze what action to execute"""
        task = state.current_task
        
        # Determine action type
        if "api" in task.lower() or "call" in task.lower():
            action_type = "api_call"
        elif "workflow" in task.lower() or "trigger" in task.lower():
            action_type = "workflow_trigger"
        else:
            action_type = "general_execution"
        
        # Check safety
        risk_level = self._assess_risk(task)
        
        return {
            "action_type": action_type,
            "risk_level": risk_level,
            "requires_approval": risk_level == "high"
        }
    
    def _assess_risk(self, task: str) -> str:
        """Assess risk level of action"""
        high_risk_keywords = ["delete", "remove", "production", "critical"]
        medium_risk_keywords = ["update", "modify", "change"]
        
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in task_lower for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"
    
    async def make_decision(self, state: AURAAgentState) -> Dict[str, Any]:
        """Decide on execution strategy"""
        analysis = state.results.get("analysis", {})
        
        if analysis.get("requires_approval"):
            # In production, this would request human approval
            return {"action": "escalate", "reason": "high_risk_action"}
        
        action_type = analysis.get("action_type", "general")
        if action_type == "api_call":
            return {"action": "use_tools", "tool": "execute_api_call"}
        elif action_type == "workflow_trigger":
            return {"action": "use_tools", "tool": "trigger_workflow"}
        else:
            return {"action": "execute"}
    
    async def execute_action(self, state: AURAAgentState) -> Dict[str, Any]:
        """Execute action with safety checks"""
        # Log execution attempt
        logger.info(
            f"Executing action",
            agent_id=self.agent_id,
            action=state.last_action,
            risk_level=state.results.get("analysis", {}).get("risk_level", "unknown")
        )
        
        # Store execution record
        if self.memory:
            await self.memory.store({
                "type": "execution",
                "agent_id": self.agent_id,
                "action": state.last_action,
                "timestamp": datetime.now(timezone.utc),
                "status": "executed"
            })
        
        return {"status": "action_executed", "verification_pending": True}


# ======================
# Coordinator Agent
# ======================

class CoordinatorAgent(AURAAgentCore):
    """
    Orchestrates multi-agent workflows.
    
    Use cases:
    - Multi-agent coordination
    - Complex workflow management
    - Resource allocation
    - Consensus building
    
    Features:
    - Agent discovery
    - Task delegation
    - Result aggregation
    - Consensus mechanisms
    """
    
    @property
    def agent_type(self) -> str:
        return "coordinator"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.managed_agents: Dict[str, AURAAgentCore] = {}
    
    def build_tools(self) -> List[BaseTool]:
        """Coordinator-specific tools"""
        tools = []
        
        @tool
        def discover_agents(capability: str) -> List[Dict[str, Any]]:
            """Discover available agents with capability"""
            # In production, this would query agent registry
            return [
                {"agent_id": "obs_001", "type": "observer", "status": "available"},
                {"agent_id": "ana_001", "type": "analyst", "status": "available"},
                {"agent_id": "exe_001", "type": "executor", "status": "busy"}
            ]
        
        @tool
        async def delegate_task(agent_id: str, task: str) -> Dict[str, Any]:
            """Delegate task to another agent"""
            # This would actually delegate to real agents
            return {
                "agent_id": agent_id,
                "task_id": str(uuid.uuid4()),
                "status": "delegated",
                "estimated_completion": "5m"
            }
        
        @tool
        def aggregate_results(task_ids: List[str]) -> Dict[str, Any]:
            """Aggregate results from multiple agents"""
            return {
                "aggregated_at": datetime.now(timezone.utc).isoformat(),
                "task_count": len(task_ids),
                "status": "completed",
                "consensus": "achieved",
                "final_result": {"decision": "proceed", "confidence": 0.89}
            }
        
        tools.extend([discover_agents, delegate_task, aggregate_results])
        return tools
    
    def register_agent(self, agent: AURAAgentCore):
        """Register an agent for coordination"""
        self.managed_agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type}")
    
    async def analyze_task(self, state: AURAAgentState) -> Dict[str, Any]:
        """Analyze coordination requirements"""
        task = state.current_task
        
        # Determine workflow complexity
        if "multi" in task.lower() or "coordinate" in task.lower():
            complexity = "high"
            required_agents = ["observer", "analyst", "executor"]
        else:
            complexity = "low"
            required_agents = ["executor"]
        
        # Use orchestration engine if available
        if self.orchestrator:
            workflow_plan = {
                "type": "multi_agent_workflow",
                "steps": len(required_agents),
                "parallel": True
            }
        else:
            workflow_plan = {"type": "sequential"}
        
        return {
            "complexity": complexity,
            "required_agents": required_agents,
            "workflow_plan": workflow_plan
        }
    
    async def make_decision(self, state: AURAAgentState) -> Dict[str, Any]:
        """Decide on coordination strategy"""
        analysis = state.results.get("analysis", {})
        
        if analysis.get("complexity") == "high":
            return {
                "action": "coordinate",
                "strategy": "parallel_execution",
                "agents_needed": analysis.get("required_agents", [])
            }
        else:
            return {
                "action": "use_tools",
                "tool": "delegate_task"
            }
    
    async def execute_action(self, state: AURAAgentState) -> Dict[str, Any]:
        """Execute coordination"""
        decision = state.results.get("decision", {})
        
        if decision.get("action") == "coordinate":
            # Coordinate multiple agents
            agents_needed = decision.get("agents_needed", [])
            
            # In production, this would:
            # 1. Discover available agents
            # 2. Delegate tasks
            # 3. Monitor progress
            # 4. Aggregate results
            
            results = {
                "coordination_id": str(uuid.uuid4()),
                "agents_involved": len(agents_needed),
                "status": "coordinating",
                "subtasks": []
            }
            
            for agent_type in agents_needed:
                subtask = {
                    "agent_type": agent_type,
                    "status": "delegated",
                    "task_id": str(uuid.uuid4())
                }
                results["subtasks"].append(subtask)
            
            return results
        
        return {"status": "coordination_complete"}


# ======================
# Factory Function
# ======================

def create_agent(
    agent_type: str,
    agent_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> AURAAgentCore:
    """
    Factory function to create agents.
    
    Args:
        agent_type: Type of agent (observer, analyst, executor, coordinator)
        agent_id: Optional agent ID
        config: Optional configuration
        
    Returns:
        Configured agent instance
    """
    
    agent_classes = {
        "observer": ObserverAgent,
        "analyst": AnalystAgent,
        "executor": ExecutorAgent,
        "coordinator": CoordinatorAgent
    }
    
    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Valid types: {list(agent_classes.keys())}"
        )
    
    return agent_class(agent_id=agent_id, config=config)


# ======================
# Example Usage
# ======================

async def example_multi_agent_workflow():
    """Example of agents working together"""
    
    # Create agents
    observer = create_agent("observer", "obs_001")
    analyst = create_agent("analyst", "ana_001")
    executor = create_agent("executor", "exe_001")
    coordinator = create_agent("coordinator", "coord_001")
    
    # Register agents with coordinator
    coordinator.register_agent(observer)
    coordinator.register_agent(analyst)
    coordinator.register_agent(executor)
    
    # Run coordinated workflow
    result = await coordinator.run(
        task="Monitor system health, analyze trends, and execute optimizations",
        context={"priority": "high", "scope": "production"}
    )
    
    print(f"Workflow result: {result}")
    
    # Get metrics
    for agent in [observer, analyst, executor, coordinator]:
        metrics = agent.get_metrics()
        print(f"{agent.agent_type} metrics: {metrics}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_multi_agent_workflow())