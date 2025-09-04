import asyncio
import types
import pytest

from core.src.aura_intelligence.orchestration.aura_cognitive_workflow import osiris_planning_node
from core.src.aura_intelligence.schemas.aura_execution import AuraTask, AuraWorkflowState, TaskStatus


class DummyOsiris:
    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.calls = 0

    async def process(self, prompt: str, context=None, stream=False):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise RuntimeError("osiris fail once")
        return {"recommended_actions": [{"tool": "SystemObservationTool", "params": {"target": "system"}}], "parallelization_possible": True}


@pytest.mark.asyncio
async def test_osiris_planning_healed_and_retried(monkeypatch):
    # Create a single DummyOsiris instance that will be reused
    dummy_osiris = DummyOsiris(fail_once=True)
    
    async def get_osiris_brain():
        return dummy_osiris

    executor = types.SimpleNamespace(
        get_osiris_brain=get_osiris_brain,
        memory=types.SimpleNamespace(query=lambda *a, **k: asyncio.sleep(0, result={})),
        self_healing=types.SimpleNamespace(heal_component=lambda *a, **k: asyncio.sleep(0, result={"success": True})),
        config={"use_self_healing_nodes": True},
    )

    state = AuraWorkflowState(task=AuraTask(objective="test", environment={}))
    state_dict = state.model_dump()
    state_dict["executor_instance"] = executor

    result = await osiris_planning_node(state_dict)
    assert "plan" in result
    assert result["plan"]["steps"]

