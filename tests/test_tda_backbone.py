import pytest
import asyncio
from unittest.mock import AsyncMock
from datetime import datetime

from core.src.aura_intelligence.tda.realtime_monitor import SystemEvent, EventType
from core.src.aura_intelligence.orchestration.execution_engine import UnifiedWorkflowExecutor


@pytest.fixture
def executor(tmp_path):
    config = {
        "use_osiris_planning": False,
        "use_swarm_execution": False,
        "use_tda_realtime_monitor": True,
        "tda_queue_maxsize": 2,
        "tda_backpressure_policy": "drop_oldest",
        "tda_pump_batch_size": 10,
        "tda_pump_interval_ms": 10,
        "tda_monitor_concurrency": 2,
        "tda_metrics_prefix": "aura_tda",
    }
    ex = UnifiedWorkflowExecutor(config=config)
    # Stub out monitor to avoid heavy processing
    ex._tda_monitor = AsyncMock()
    # Ensure start is idempotent
    return ex


@pytest.mark.asyncio
async def test_workflow_start_complete_emission(executor):
    # Run a minimal task
    await executor.execute_task("Smoke", {"env": "test"})
    # Let the pump process the two events
    await asyncio.sleep(0.1)

    emitted = executor.metrics.get("aura_tda_events_emitted_total")
    dropped = executor.metrics.get("aura_tda_events_dropped_total")
    errors = executor.metrics.get("aura_tda_errors_total")

    assert emitted == 2, f"Expected 2, got {emitted}"
    assert dropped == 0
    assert errors == 0


@pytest.mark.asyncio
async def test_backpressure_and_drop_policy(executor):
    # Rapidly emit 5 events into small queue
    for _ in range(5):
        executor.emit_event(SystemEvent(
            event_id="evt",
            event_type=EventType.TASK_ASSIGNED,
            timestamp=datetime.utcnow().timestamp(),
            source_agent="test",
            workflow_id="T1",
            metadata={}
        ))
    # Allow pump to run
    await asyncio.sleep(0.1)

    emitted = executor.metrics.get("aura_tda_events_emitted_total")
    dropped = executor.metrics.get("aura_tda_events_dropped_total")
    # With queue size 2, first two put succeed, rest should drop
    assert emitted == 2
    assert dropped == 3


def test_toggle_off_no_queue_or_metrics(tmp_path):
    config = {"use_tda_realtime_monitor": False}
    ex = UnifiedWorkflowExecutor(config=config)
    # No queue and no prefixed metrics
    assert getattr(ex, "_tda_events_queue", None) is None
    ex.emit_event(SystemEvent(
        event_id="evt",
        event_type=EventType.WORKFLOW_STARTED,
        timestamp=datetime.utcnow().timestamp(),
        source_agent="exec",
        workflow_id="T1",
        metadata={}
    ))
    assert ex.metrics.get("aura_tda_events_emitted_total") is None
    assert ex.metrics.get("aura_tda_events_dropped_total") is None
    assert ex.metrics.get("aura_tda_errors_total") is None

