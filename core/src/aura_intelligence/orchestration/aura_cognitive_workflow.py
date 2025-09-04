"""
AURA Cognitive Workflow - Real implementation of the 5-step cognitive process
Perception -> Planning -> Consensus -> Execution -> Analysis/Consolidation
September 2025 - State-of-the-art LangGraph implementation
"""

import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List
from datetime import datetime
import structlog

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..schemas.aura_execution import (
    AuraWorkflowState,
    TaskStatus,
    ExecutionPlan,
    ExecutionStep,
    ConsensusDecision,
    ObservationResult,
    MemoryContext,
    TopologicalSignature
)

if TYPE_CHECKING:
    from .execution_engine import UnifiedWorkflowExecutor

logger = structlog.get_logger(__name__)


# ============================================================================
# WORKFLOW NODE IMPLEMENTATIONS
# ============================================================================

async def perception_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for Step 1: PERCEPTION
    Understand the task and gather initial context from memory.
    """
    logger.info("🧠 PERCEPTION: Understanding the task")
    
    # Extract current state
    current_state = AuraWorkflowState.model_validate(state)
    executor: 'UnifiedWorkflowExecutor' = state['executor_instance']
    
    # Update task status
    current_state.task.status = TaskStatus.PERCEIVING
    current_state.add_trace("Starting perception phase")
    
    # Check if we have a perception agent
    if "PerceptionAgent" in executor.agents:
        perception_agent = executor.agents["PerceptionAgent"]
        try:
            perception_output = await perception_agent.perceive(
                current_state.task.objective,
                current_state.task.environment
            )
            current_state.task.perception_output = perception_output
            current_state.add_trace(f"Perception complete: {len(perception_output)} insights")
        except Exception as e:
            logger.warning(f"Perception agent failed: {e}, using fallback")
            current_state.task.perception_output = _fallback_perception(current_state.task.objective)
    else:
        # No perception agent, use basic analysis
        current_state.task.perception_output = _fallback_perception(current_state.task.objective)
    
    # Query memory for relevant context
    if executor.memory:
        try:
            memory_context = await executor.memory.query(current_state.task.objective)
            
            # Convert to our MemoryContext schema
            current_state.task.memory_context = MemoryContext(
                episodic_memories=memory_context.get("episodic", []),
                semantic_concepts=memory_context.get("semantic", []),
                causal_patterns=memory_context.get("patterns", []),
                working_memory_items=memory_context.get("working", []),
                synthesis=memory_context.get("synthesis", "")
            )
            
            current_state.add_trace(f"Retrieved {len(memory_context.get('episodic', []))} relevant memories")
        except Exception as e:
            logger.warning(f"Memory query failed: {e}")
            current_state.task.memory_context = MemoryContext()
    
    # Set metrics
    current_state.set_metric("perception_duration", 
                            (datetime.utcnow() - current_state.task.created_at).total_seconds())
    
    logger.info("✅ Perception complete")
    
    # Return updated state
    return {
        "task": current_state.task.model_dump(),
        "execution_trace": current_state.execution_trace,
        "metrics": current_state.metrics
    }


async def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for Step 2: PLANNING
    Create an execution plan using memory and perception.
    """
    logger.info("🗺️ PLANNING: Creating execution strategy")
    
    current_state = AuraWorkflowState.model_validate(state)
    executor: 'UnifiedWorkflowExecutor' = state['executor_instance']
    
    # Update status
    current_state.task.status = TaskStatus.PLANNING
    current_state.add_trace("Starting planning phase")
    
    # Prepare context for planning
    context = {
        "perception": current_state.task.perception_output,
        "memory": current_state.task.memory_context.model_dump() if current_state.task.memory_context else {},
        "environment": current_state.task.environment
    }
    
    # Check if we have a planner agent
    if "PlannerAgent" in executor.agents:
        planner_agent = executor.agents["PlannerAgent"]
        try:
            plan = await planner_agent.create_plan(
                goal=current_state.task.objective,
                context=context
            )
            current_state.plan = ExecutionPlan.model_validate(plan)
        except Exception as e:
            logger.warning(f"Planner agent failed: {e}, using fallback")
            current_state.plan = _create_fallback_plan(current_state.task.objective, executor)
    else:
        # No planner agent, create basic plan
        current_state.plan = _create_fallback_plan(current_state.task.objective, executor)
    
    current_state.add_trace(f"Plan created with {len(current_state.plan.steps)} steps")
    
    logger.info(f"✅ Plan created: {current_state.plan.plan_id}")
    
    return {
        "plan": current_state.plan.model_dump(),
        "execution_trace": current_state.execution_trace
    }


async def osiris_planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced planning using Osiris Unified Intelligence.
    Leverages executor.get_osiris_brain() and real memory context.
    """
    logger.info("🗺️ OSIRIS PLANNING: Creating execution strategy with Osiris")
    current_state = AuraWorkflowState.model_validate(state)
    executor: 'UnifiedWorkflowExecutor' = state['executor_instance']
    
    # Update status
    current_state.task.status = TaskStatus.PLANNING
    current_state.add_trace("Starting OSIRIS planning phase")
    
    try:
        # Get Osiris brain (lazy-initialized)
        osiris = await executor.get_osiris_brain()
        
        # Retrieve richer context from memory if available
        memory_context = {}
        if executor.memory:
            try:
                memory_context = await executor.memory.query(current_state.task.objective)
            except Exception as e:
                logger.warning(f"Memory query for Osiris planning failed: {e}")
                memory_context = {}
        
        # Call Osiris for intelligent planning
        osiris_result = await osiris.process(
            prompt=current_state.task.objective,
            context={
                "retrieved_memory": memory_context,
                "environment_state": current_state.task.environment,
            }
        )
        
        # Transform Osiris output into our ExecutionPlan schema
        raw_steps = osiris_result.get("recommended_actions", [])
        steps: List[ExecutionStep] = []
        for s in raw_steps:
            try:
                steps.append(ExecutionStep(
                    tool=s.get("tool"),
                    params=s.get("params", {}),
                    dependencies=s.get("dependencies", []),
                    expected_output_type=s.get("expected_output_type")
                ))
            except Exception as e:
                logger.warning(f"Invalid Osiris step skipped: {e}")
        
        # Build execution plan
        current_state.plan = ExecutionPlan(
            objective=current_state.task.objective,
            steps=steps,
            estimated_duration=osiris_result.get("estimated_duration"),
            risk_assessment=osiris_result.get("risk_assessment", {}),
            parallelization_possible=osiris_result.get("parallelization_possible", True)
        )
        
        current_state.add_trace(f"Osiris plan created with {len(current_state.plan.steps)} steps")
        logger.info(f"✅ Osiris plan created: {current_state.plan.plan_id}")
        
        return {
            "plan": current_state.plan.model_dump(),
            "execution_trace": current_state.execution_trace
        }
    except Exception as e:
        logger.error(f"Osiris planning failed: {e}", exc_info=True)
        # Fallback to basic plan to keep workflow alive
        current_state.plan = _create_fallback_plan(current_state.task.objective, executor)
        current_state.add_trace("Osiris planning failed, used fallback plan")
        return {
            "plan": current_state.plan.model_dump(),
            "execution_trace": current_state.execution_trace
        }


async def consensus_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for Step 3: CONSENSUS
    Get agreement from agent collective on the plan.
    """
    logger.info("🤝 CONSENSUS: Seeking agreement on plan")
    
    current_state = AuraWorkflowState.model_validate(state)
    executor: 'UnifiedWorkflowExecutor' = state['executor_instance']
    
    # Update status
    current_state.task.status = TaskStatus.CONSENSUS
    current_state.add_trace("Starting consensus phase")
    
    # Check if we have a consensus system
    if executor.consensus:
        try:
            # Get list of agents for consensus
            agents_for_consensus = []
            for agent_name in ["PlannerAgent", "AnalystAgent", "ExecutorAgent"]:
                if agent_name in executor.agents:
                    agents_for_consensus.append(executor.agents[agent_name])
            
            if agents_for_consensus:
                consensus_result = await executor.consensus.reach_consensus(
                    proposal=current_state.plan.model_dump(),
                    agents=agents_for_consensus
                )
                current_state.consensus = ConsensusDecision.model_validate(consensus_result)
            else:
                # No agents, auto-approve
                current_state.consensus = _auto_approve_consensus(current_state.plan)
        except Exception as e:
            logger.warning(f"Consensus system failed: {e}, auto-approving")
            current_state.consensus = _auto_approve_consensus(current_state.plan)
    else:
        # No consensus system, auto-approve
        current_state.consensus = _auto_approve_consensus(current_state.plan)
    
    # Update plan if modified during consensus
    if current_state.consensus.approved_plan:
        current_state.plan = current_state.consensus.approved_plan
    
    current_state.add_trace(f"Consensus {'reached' if current_state.consensus.approved else 'failed'}")
    
    logger.info(f"✅ Consensus: {'Approved' if current_state.consensus.approved else 'Rejected'}")
    
    return {
        "consensus": current_state.consensus.model_dump(),
        "plan": current_state.plan.model_dump() if current_state.plan else None,
        "execution_trace": current_state.execution_trace
    }


async def execution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for Step 4: EXECUTION
    Execute the approved plan using real tools.
    """
    logger.info("⚡ EXECUTION: Running the plan")
    
    current_state = AuraWorkflowState.model_validate(state)
    executor: 'UnifiedWorkflowExecutor' = state['executor_instance']
    
    # Update status
    current_state.task.status = TaskStatus.EXECUTING
    current_state.add_trace("Starting execution phase")
    
    # Check consensus approval
    if not current_state.consensus or not current_state.consensus.approved:
        logger.warning("Plan not approved, skipping execution")
        current_state.add_trace("Execution skipped: plan not approved")
        return {"execution_trace": current_state.execution_trace}
    
    # Execute plan steps
    observations = []
    execution_start = datetime.utcnow()
    
    for i, step in enumerate(current_state.plan.steps):
        logger.info(f"Executing step {i+1}/{len(current_state.plan.steps)}: {step.tool}")
        current_state.add_trace(f"Executing: {step.tool}")
        
        try:
            # Execute the tool
            if executor.tools:
                result = await executor.tools.execute(
                    tool_name=step.tool,
                    params=step.params
                )
                
                # Convert result to ObservationResult if needed
                if isinstance(result, ObservationResult):
                    observations.append(result)
                elif isinstance(result, dict):
                    observations.append(ObservationResult.model_validate(result))
                else:
                    # Wrap in ObservationResult
                    observations.append(ObservationResult(
                        source=step.tool,
                        data={"result": str(result)},
                        topology=TopologicalSignature(
                            betti_numbers=[1, 0, 0],
                            persistence_entropy=0.0,
                            wasserstein_distance_from_norm=0.0
                        ),
                        anomalies=[]
                    ))
                
                current_state.add_trace(f"Step {step.step_id} completed successfully")
                
                # Store experience in memory
                if executor.memory:
                    await executor.memory.process_new_experience({
                        "type": "tool_execution",
                        "tool": step.tool,
                        "params": step.params,
                        "result": observations[-1].model_dump(),
                        "task_id": current_state.task.task_id
                    })
                    
            else:
                logger.warning(f"No tools available, skipping step {step.step_id}")
                
        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")
            current_state.add_trace(f"Step {step.step_id} failed: {str(e)}")
            
            # Create error observation
            observations.append(ObservationResult(
                source=step.tool,
                data={"error": str(e), "step": step.model_dump()},
                topology=TopologicalSignature(
                    betti_numbers=[0, 0, 0],
                    persistence_entropy=0.0,
                    wasserstein_distance_from_norm=1.0
                ),
                anomalies=[{"type": "execution_error", "description": str(e)}]
            ))
    
    # Update state with observations
    current_state.observations = observations
    
    # Calculate execution metrics
    execution_duration = (datetime.utcnow() - execution_start).total_seconds()
    current_state.set_metric("execution_duration", execution_duration)
    current_state.set_metric("steps_executed", len(observations))
    
    logger.info(f"✅ Execution complete: {len(observations)} observations collected")
    
    return {
        "observations": [obs.model_dump() for obs in current_state.observations],
        "execution_trace": current_state.execution_trace,
        "metrics": current_state.metrics
    }


async def analysis_consolidation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for Step 5: ANALYSIS & CONSOLIDATION
    Analyze results, identify patterns, and consolidate learning.
    """
    logger.info("📊 ANALYSIS & CONSOLIDATION: Processing results")
    
    current_state = AuraWorkflowState.model_validate(state)
    executor: 'UnifiedWorkflowExecutor' = state['executor_instance']
    
    # Update status
    current_state.task.status = TaskStatus.ANALYZING
    current_state.add_trace("Starting analysis and consolidation")
    
    # Analyze observations for patterns
    patterns = []
    
    if "AnalystAgent" in executor.agents:
        analyst_agent = executor.agents["AnalystAgent"]
        try:
            # Convert observations back to proper format
            obs_data = [ObservationResult.model_validate(obs) for obs in current_state.observations]
            patterns = await analyst_agent.analyze_patterns(obs_data)
            current_state.patterns = patterns
        except Exception as e:
            logger.warning(f"Analyst agent failed: {e}, using fallback")
            patterns = _analyze_observations_fallback(current_state.observations)
            current_state.patterns = patterns
    else:
        # Fallback analysis
        patterns = _analyze_observations_fallback(current_state.observations)
        current_state.patterns = patterns
    
    # Identify topological anomalies
    topological_anomalies = _extract_topological_anomalies(current_state.observations)
    if topological_anomalies:
        patterns.append({
            "type": "topological",
            "anomalies": topological_anomalies,
            "severity": "medium"
        })
    
    # Use causal tracker if available
    if executor.memory and hasattr(executor.memory, 'causal_tracker'):
        try:
            causal_patterns = await executor.memory.causal_tracker.analyze_sequence([
                obs for obs in current_state.observations
            ])
            if causal_patterns:
                patterns.extend(causal_patterns)
        except Exception as e:
            logger.warning(f"Causal analysis failed: {e}")
    
    # Prepare final result
    final_result = {
        "summary": _generate_summary(current_state),
        "observations_count": len(current_state.observations),
        "patterns_found": len(patterns),
        "patterns": patterns,
        "anomalies": _extract_all_anomalies(current_state.observations),
        "recommendations": _generate_recommendations(patterns),
        "execution_metrics": current_state.metrics,
        "task_id": current_state.task.task_id
    }
    
    # Update task
    current_state.task.final_result = final_result
    current_state.task.status = TaskStatus.COMPLETED
    current_state.task.completed_at = datetime.utcnow()
    
    # Consolidate learning in memory
    if executor.memory:
        try:
            experience = {
                "task": current_state.task.model_dump(),
                "observations": [obs for obs in current_state.observations],
                "patterns": patterns,
                "success": True,
                "duration": current_state.metrics.get("execution_duration", 0)
            }
            
            await executor.memory.process_new_experience(experience)
            current_state.add_trace("Learning consolidated in memory")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {e}")
    
    current_state.add_trace("Analysis and consolidation complete")
    
    logger.info(f"✅ Task completed: {len(patterns)} patterns identified")
    
    return {
        "task": current_state.task.model_dump(),
        "patterns": current_state.patterns,
        "execution_trace": current_state.execution_trace,
        "metrics": current_state.metrics
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _fallback_perception(objective: str) -> Dict[str, Any]:
    """Create fallback perception output"""
    keywords = []
    important_words = ["analyze", "monitor", "detect", "prevent", "optimize", 
                      "performance", "memory", "leak", "error", "failure"]
    
    objective_lower = objective.lower()
    for word in important_words:
        if word in objective_lower:
            keywords.append(word)
    
    return {
        "understanding": "Basic task understanding through keyword analysis",
        "key_concepts": keywords,
        "complexity": "high" if len(keywords) > 3 else "medium"
    }


def _create_fallback_plan(objective: str, executor: 'UnifiedWorkflowExecutor') -> ExecutionPlan:
    """Create a basic fallback plan"""
    steps = []
    
    # Always start with observation if available
    if executor.tools and "SystemObservationTool" in executor.tools.list_tools():
        steps.append(ExecutionStep(
            tool="SystemObservationTool",
            params={
                "target": "system",
                "params": {"duration": "15m"}
            }
        ))
    
    return ExecutionPlan(
        objective=objective,
        steps=steps,
        risk_assessment={"fallback_plan": 0.5}
    )


def _auto_approve_consensus(plan: ExecutionPlan) -> ConsensusDecision:
    """Create auto-approved consensus"""
    return ConsensusDecision(
        approved=True,
        approved_plan=plan,
        voting_results={"auto": True}
    )


def _analyze_observations_fallback(observations: List[Any]) -> List[Dict[str, Any]]:
    """Fallback pattern analysis"""
    patterns = []
    
    # Count anomalies
    total_anomalies = 0
    for obs in observations:
        if isinstance(obs, dict) and "anomalies" in obs:
            total_anomalies += len(obs["anomalies"])
    
    if total_anomalies > 0:
        patterns.append({
            "type": "anomaly_detection",
            "count": total_anomalies,
            "severity": "high" if total_anomalies > 5 else "medium"
        })
    
    return patterns


def _extract_topological_anomalies(observations: List[Any]) -> List[Dict[str, Any]]:
    """Extract topological anomalies from observations"""
    anomalies = []
    
    for obs in observations:
        if isinstance(obs, dict) and "topology" in obs and obs["topology"]:
            topo = obs["topology"]
            if topo.get("betti_numbers", [0])[0] > 5:
                anomalies.append({
                    "type": "high_connectivity",
                    "observation": obs.get("observation_id"),
                    "betti_0": topo["betti_numbers"][0]
                })
            
            if len(topo.get("betti_numbers", [])) > 1 and topo["betti_numbers"][1] > 3:
                anomalies.append({
                    "type": "excessive_loops",
                    "observation": obs.get("observation_id"),
                    "betti_1": topo["betti_numbers"][1]
                })
    
    return anomalies


def _extract_all_anomalies(observations: List[Any]) -> List[Dict[str, Any]]:
    """Extract all anomalies from observations"""
    all_anomalies = []
    
    for obs in observations:
        if isinstance(obs, dict) and "anomalies" in obs:
            all_anomalies.extend(obs["anomalies"])
    
    return all_anomalies


def _generate_summary(state: AuraWorkflowState) -> str:
    """Generate execution summary"""
    return (
        f"Task '{state.task.objective}' completed. "
        f"Executed {len(state.plan.steps) if state.plan else 0} steps, "
        f"collected {len(state.observations)} observations, "
        f"identified {len(state.patterns)} patterns."
    )


def _generate_recommendations(patterns: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on patterns"""
    recommendations = []
    
    for pattern in patterns:
        if pattern.get("type") == "topological":
            recommendations.append("Review system architecture for circular dependencies")
        elif pattern.get("type") == "anomaly_detection":
            recommendations.append("Investigate detected anomalies immediately")
        elif pattern.get("severity") == "high":
            recommendations.append(f"High severity pattern detected: {pattern.get('type')}")
    
    if not recommendations:
        recommendations.append("System operating within normal parameters")
    
    return recommendations


# ============================================================================
# WORKFLOW FACTORY
# ============================================================================

def create_aura_workflow(executor: 'UnifiedWorkflowExecutor') -> StateGraph:
    """
    Factory function to create and compile the AURA workflow graph.
    Implements the 5-step cognitive process.
    """
    logger.info("Creating AURA cognitive workflow graph")
    
    # Create the workflow
    workflow = StateGraph(dict)  # Using dict for flexibility
    
    # Add all nodes for the 5-step process
    workflow.add_node("PERCEIVE", perception_node)
    # Allow switching to Osiris planning via config toggle
    use_osiris = False
    try:
        use_osiris = bool(getattr(executor, 'config', {}).get('use_osiris_planning', True))
    except Exception:
        use_osiris = True
    
    workflow.add_node("PLAN", osiris_planning_node if use_osiris else planning_node)
    workflow.add_node("CONSENSUS", consensus_node)
    workflow.add_node("EXECUTE", execution_node)
    workflow.add_node("ANALYZE", analysis_consolidation_node)
    
    # Define the execution flow
    workflow.set_entry_point("PERCEIVE")
    workflow.add_edge("PERCEIVE", "PLAN")
    workflow.add_edge("PLAN", "CONSENSUS")
    workflow.add_edge("CONSENSUS", "EXECUTE")
    workflow.add_edge("EXECUTE", "ANALYZE")
    workflow.add_edge("ANALYZE", END)
    
    # Add checkpointing for persistence and recovery
    checkpointer = None
    if executor.memory:
        # Use memory system's checkpointer if available
        checkpointer = MemorySaver()
    
    # Compile the workflow
    compiled = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ AURA cognitive workflow graph created and compiled")
    return compiled