"""
Workflow Resilience Decorators - Self-Healing Integration
Provides an advanced, configurable decorator to wrap LangGraph nodes
with AURA's SelfHealingEngine while preserving async performance.
"""

import asyncio
import sys
import traceback
from functools import wraps
from typing import Any, Dict


class with_advanced_self_healing:
    """
    Configurable decorator to add self-healing around async workflow nodes.
    - Retries with exponential backoff
    - Invokes SelfHealingEngine.heal_component with rich context
    - Optional delegation to ExecutiveController on final failure
    """

    def __init__(self, max_retries: int = 1, backoff_factor: float = 1.5, delegate_on_failure: bool = True):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.delegate_on_failure = delegate_on_failure

    def __call__(self, node_function):
        @wraps(node_function)
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            executor = state.get("executor_instance")

            # If feature flag disabled or self-healing missing, run original
            use_flag = True
            try:
                use_flag = bool(getattr(executor, 'config', {}).get('use_self_healing_nodes', True))
            except Exception:
                use_flag = True

            if not use_flag or not hasattr(executor, 'self_healing') or not executor.self_healing:
                return await node_function(state)

            attempt = 0
            while True:
                try:
                    return await node_function(state)
                except Exception as e:
                    attempt += 1
                    exc_type, exc_value, tb = sys.exc_info()

                    issue_payload = {
                        "error_message": str(exc_value),
                        "error_type": exc_type.__name__ if exc_type else type(e).__name__,
                        "traceback": "".join(traceback.format_tb(tb)) if tb else "",
                        "workflow_state_summary": {k: v for k, v in state.items() if k != 'executor_instance'},
                    }

                    try:
                        heal_result = await executor.self_healing.heal_component(
                            component_id=f"langgraph_node:{node_function.__name__}",
                            issue=issue_payload,
                        )
                    except Exception:
                        heal_result = {"success": False, "error": "heal_component_raised"}

                    if heal_result and heal_result.get("success") is True:
                        # Retry immediately after successful heal
                        try:
                            return await node_function(state)
                        except Exception as retry_err:
                            # Fall through to retry/backoff logic
                            e = retry_err

                    # Not healed or retry failed
                    if attempt > self.max_retries:
                        # Optional delegation to executive
                        if self.delegate_on_failure and hasattr(executor, 'executive') and executor.executive:
                            try:
                                handler = getattr(executor.executive, 'handle_unrecoverable_error', None)
                                if callable(handler):
                                    await handler({
                                        "node": node_function.__name__,
                                        "issue": issue_payload,
                                        "heal_result": heal_result,
                                    })
                            except Exception:
                                pass
                        # Exhausted retries
                        raise

                    # Exponential backoff before next attempt
                    delay = (self.backoff_factor ** (attempt - 1))
                    await asyncio.sleep(delay)

        return wrapper

