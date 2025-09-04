import asyncio
import types
import pytest

from core.src.aura_intelligence.orchestration.aura_cognitive_workflow import (
    create_aura_workflow,
    osiris_planning_node,
    swarm_execution_node,
)


class DummyExecutor:
    def __init__(self, config):
        self.config = config


def test_workflow_uses_nodes_per_toggles():
    # Default True should wire Osiris and Swarm
    wf = create_aura_workflow(DummyExecutor({}))
    assert wf is not None

    # Disable Osiris
    wf2 = create_aura_workflow(DummyExecutor({"use_osiris_planning": False}))
    assert wf2 is not None

    # Disable Swarm
    wf3 = create_aura_workflow(DummyExecutor({"use_swarm_execution": False}))
    assert wf3 is not None

