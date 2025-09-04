import asyncio
import types

import pytest

from core.src.aura_intelligence.orchestration.resilience import with_advanced_self_healing


class DummySelfHealing:
    def __init__(self, success: bool = True):
        self.success = success
        self.calls = 0

    async def heal_component(self, component_id, issue):
        self.calls += 1
        return {"success": self.success}


@pytest.mark.asyncio
async def test_successful_heal_then_retry():
    attempts = {"n": 0}

    @with_advanced_self_healing(max_retries=1)
    async def node(state: dict) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("first fail")
        return {"ok": True}

    state = {"executor_instance": types.SimpleNamespace(self_healing=DummySelfHealing(True), config={"use_self_healing_nodes": True})}
    res = await node(state)
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_failed_heal_then_reraise():
    @with_advanced_self_healing(max_retries=1)
    async def node(state: dict) -> dict:
        raise RuntimeError("boom")

    state = {"executor_instance": types.SimpleNamespace(self_healing=DummySelfHealing(False), config={"use_self_healing_nodes": True})}
    with pytest.raises(RuntimeError):
        await node(state)


@pytest.mark.asyncio
async def test_no_self_healing_present():
    @with_advanced_self_healing(max_retries=1)
    async def node(state: dict) -> dict:
        raise RuntimeError("boom")

    state = {"executor_instance": types.SimpleNamespace(config={"use_self_healing_nodes": True})}
    with pytest.raises(RuntimeError):
        await node(state)

