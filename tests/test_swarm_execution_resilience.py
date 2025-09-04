import asyncio
import types
import pytest

from core.src.aura_intelligence.orchestration.aura_cognitive_workflow import swarm_execution_node
from core.src.aura_intelligence.schemas.aura_execution import (
    AuraTask, AuraWorkflowState, TaskStatus, ExecutionPlan, ExecutionStep, ConsensusDecision
)


@pytest.mark.asyncio
async def test_swarm_parallel_exec_with_fallback(monkeypatch):
    # Build a simple plan with 3 steps
    steps = [ExecutionStep(tool="SystemObservationTool", params={"target": f"sys{i}"}) for i in range(3)]
    plan = ExecutionPlan(objective="test", steps=steps)

    executor = types.SimpleNamespace(
        get_swarm_coordinator=lambda: asyncio.sleep(0, result=types.SimpleNamespace(coordinate_agents=lambda **k: asyncio.sleep(0, result={"type": "exploration"}))),
        execute_tool_with_retry=lambda name, params: asyncio.sleep(0, result={"source": name, "data": {"ok": True}, "topology": {"betti_numbers": [1,0,0], "persistence_entropy": 0.0, "wasserstein_distance_from_norm": 0.0}, "anomalies": []}),
        config={"use_self_healing_nodes": True, "use_swarm_execution": True},
    )

    state = AuraWorkflowState(task=AuraTask(objective="test", environment={}), plan=plan, consensus=ConsensusDecision(approved=True))
    state_dict = state.model_dump()
    state_dict["executor_instance"] = executor

    result = await swarm_execution_node(state_dict)
    assert "observations" in result
    assert len(result["observations"]) == 3

