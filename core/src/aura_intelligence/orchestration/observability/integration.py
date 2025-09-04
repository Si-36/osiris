"""
🔗 Workflow Observability Integration

Seamless integration between workflow orchestration and hybrid observability:
- Functional composition with existing workflow systems
- Effect-based integration with saga patterns
- Protocol-based extensibility
- Zero-overhead abstractions
"""

from __future__ import annotations
from typing import Optional, Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio

from .hybrid import HybridObservability, WorkflowEvent, ObservabilityConfig, create_hybrid_stack
from .core import SpanContext

class ObservableWorkflowOrchestrator:
    """Workflow orchestrator with integrated hybrid observability"""
    __slots__ = ('_observability', '_active_spans')
    
    def __init__(self, observability: HybridObservability):
        self._observability = observability
        self._active_spans: Dict[str, SpanContext] = {}
    
    @asynccontextmanager
    async def observable_workflow(
        self,
        workflow_id: str,
        operation_name: str,
        correlation_id: Optional[str] = None
        ) -> AsyncContextManager[SpanContext]:
        """Context manager for observable workflow execution"""
        start_time = datetime.now(timezone.utc)
        
        # Start workflow observation
        span_context = await self._observability.record_workflow_start(
            workflow_id=workflow_id,
            operation_name=operation_name,
            correlation_id=correlation_id
        )
        
        self._active_spans[workflow_id] = span_context
        
        try:
            yield span_context
            
            # Success path
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            await self._observability.record_workflow_end(
                workflow_id=workflow_id,
                span_context=span_context,
                status="success",
                duration_ms=duration_ms
            )
            
        except Exception as e:
            # Failure path
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            await self._observability.record_workflow_end(
                workflow_id=workflow_id,
                span_context=span_context,
                status="failed",
                duration_ms=duration_ms,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
            raise
            
        finally:
            # Cleanup
            self._active_spans.pop(workflow_id, None)
    
        async def record_step_execution(
        self,
        workflow_id: str,
        step_name: str,
        duration_seconds: float,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
        ) -> None:
        """Record individual step execution"""
        event = WorkflowEvent(
            workflow_id=workflow_id,
            event_type="step_execution",
            timestamp=datetime.now(timezone.utc),
            data={
                "step_name": step_name,
                "duration_seconds": duration_seconds,
                "status": status,
                **(metadata or {})
            }
        )
        
        # This would trigger the event handlers in the observability system
        # For now, we'll create a simple metric
        from .core import MetricPoint
        metric = MetricPoint(
            name="step_duration",
            value=duration_seconds,
            timestamp=event.timestamp
        ).with_tag("step", step_name).with_tag("status", status).with_tag("workflow_id", workflow_id)
        
        await self._observability._collector.collect(metric).run()
    
        async def record_llm_interaction(
        self,
        workflow_id: str,
        model: str,
        tokens: int,
        latency_ms: float,
        cost: float = 0.0
        ) -> None:
        """Record LLM interaction (optimized for Arize)"""
        await self._observability.record_llm_call(
            workflow_id=workflow_id,
            model=model,
            tokens=tokens,
            latency_ms=latency_ms,
            cost=cost
        )

# Factory function for creating observable orchestrator
def create_observable_orchestrator(
    config: Optional[ObservabilityConfig] = None
) -> ObservableWorkflowOrchestrator:
    """Create observable workflow orchestrator"""
    if config is None:
        # Default configuration
        config = ObservabilityConfig()
    
    observability = create_hybrid_stack(config)
    return ObservableWorkflowOrchestrator(observability)

# Integration with existing saga patterns
class ObservableSagaOrchestrator:
    """Saga orchestrator with integrated observability"""
    __slots__ = ('_base_orchestrator', '_observable_orchestrator')
    
    def __init__(self, base_orchestrator, observable_orchestrator: ObservableWorkflowOrchestrator):
        self._base_orchestrator = base_orchestrator
        self._observable_orchestrator = observable_orchestrator
    
        async def execute_observable_saga(
        self,
        saga_id: str,
        steps: list,
        tda_correlation_id: Optional[str] = None
        ) -> Dict[str, Any]:
        """Execute saga with full observability"""
        
        async with self._observable_orchestrator.observable_workflow(
            workflow_id=saga_id,
            operation_name="saga_execution",
            correlation_id=tda_correlation_id
        ) as span_context:
            
            # Execute the actual saga
            result = await self._base_orchestrator.execute_saga(
                saga_id=saga_id,
                steps=steps,
                tda_correlation_id=tda_correlation_id
            )
            
            # Record saga-specific metrics
            await self._observable_orchestrator.record_step_execution(
                workflow_id=saga_id,
                step_name="saga_compensation" if result.get("status") == "failed" else "saga_completion",
                duration_seconds=result.get("execution_time", 0.0),
                status=result.get("status", "unknown"),
                metadata={
                    "steps_executed": result.get("steps_executed", 0),
                    "compensated_steps": len(result.get("compensated_steps", []))
                }
            )
            
            return result

# Example usage
async def example_integration():
        """Example of workflow observability integration"""
        from .collectors import ArizeConfig, LangSmithConfig
        from .tracers import JaegerConfig
    
    # Configure observability
        config = ObservabilityConfig(
        arize=ArizeConfig(project_name="aura-production"),
        langsmith=LangSmithConfig(project_name="aura-workflows"),
        jaeger=JaegerConfig(service_name="aura-orchestration")
        )
    
    # Create observable orchestrator
        orchestrator = create_observable_orchestrator(config)
    
    # Use in workflow
        async with orchestrator.observable_workflow(
        workflow_id="integration_example",
        operation_name="multi_agent_processing",
        correlation_id="example-123"
        ) as span_context:
        
        # Record step execution
        await orchestrator.record_step_execution(
            workflow_id="integration_example",
            step_name="data_preprocessing",
            duration_seconds=1.5,
            status="success"
        )
        
        # Record LLM interaction
        await orchestrator.record_llm_interaction(
            workflow_id="integration_example",
            model="gpt-4",
            tokens=150,
            latency_ms=1200,
            cost=0.003
        )
        
        print(f"Workflow executed with span: {span_context.span_id}")