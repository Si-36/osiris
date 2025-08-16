"""
LNN Council Agent - Core Implementation (2025 Architecture)

Minimal, focused agent following single responsibility principle.
Each method < 20 lines, total file < 150 lines.
"""

from typing import Dict, Any, Optional, List, Union


# Fallback imports for missing dependencies
try:
    from opentelemetry.exporter import jaeger
except ImportError:
    print("Warning: OpenTelemetry exporter not available, using fallback")
    class MockJaeger:
        def __init__(self, *args, **kwargs): pass
        def export(self, *args, **kwargs): return True
    jaeger = type('jaeger', (), {'JaegerExporter': MockJaeger})

try:
    from opentelemetry import trace
except ImportError:
    print("Warning: OpenTelemetry trace not available, using fallback")
    class MockTrace:
        def get_tracer(self, *args, **kwargs): 
            return type('tracer', (), {'start_span': lambda *a, **k: type('span', (), {'__enter__': lambda s: s, '__exit__': lambda *a: None})()})()
    trace = MockTrace()

from typing import Dict, Any, Union
import asyncio
import structlog

from ..base import AgentBase
from .models import GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState
from .config import LNNCouncilConfig
from .workflow import WorkflowEngine
from .neural_engine import NeuralDecisionEngine
from .fallback import FallbackEngine
from .observability import ObservabilityEngine

logger = structlog.get_logger()


class LNNCouncilAgent(AgentBase[GPUAllocationRequest, GPUAllocationDecision, LNNCouncilState]):
    """
    LNN Council Agent for GPU allocation decisions.
    
    2025 Architecture:
    - Composition over inheritance
    - Single responsibility 
    - Dependency injection
    - < 150 lines total
    """
    
    def __init__(self, config: Union[Dict[str, Any], LNNCouncilConfig]):
        """Initialize with modular components."""
        # Parse config
        self.lnn_config = self._parse_config(config)
        super().__init__(self.lnn_config.to_agent_config())
        
        # Inject dependencies (2025 pattern)
        self.workflow_engine = WorkflowEngine(self.lnn_config)
        self.neural_engine = NeuralDecisionEngine(self.lnn_config)
        self.fallback_engine = FallbackEngine(self.lnn_config)
        self.observability_engine = ObservabilityEngine(self.lnn_config.to_dict())
        
        logger.info("LNN Council Agent initialized", name=self.name)
    
    def _parse_config(self, config: Union[Dict, LNNCouncilConfig]) -> LNNCouncilConfig:
        """Parse configuration from various formats."""
        if isinstance(config, dict):
            return LNNCouncilConfig(**config)
        elif isinstance(config, LNNCouncilConfig):
            return config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
    
    def build_graph(self):
        """Build workflow graph (delegated to WorkflowEngine)."""
        return self.workflow_engine.build_graph()
    
    def _create_initial_state(self, input_data: GPUAllocationRequest) -> LNNCouncilState:
        """Create initial state with observability tracing."""
        # Start decision trace
        request_id = getattr(input_data, 'request_id', f"req_{int(asyncio.get_event_loop().time() * 1000)}")
        self.observability_engine.start_decision_trace(request_id)
        
        state = LNNCouncilState(
            current_request=input_data,
            current_step="analyze_request",
            inference_start_time=asyncio.get_event_loop().time()
        )
        
        # Add initial trace step
        self.observability_engine.add_trace_step(request_id, "initialize", {
            "request_type": "gpu_allocation",
            "priority": getattr(input_data, 'priority', 'unknown'),
            "gpu_count": getattr(input_data, 'gpu_count', 'unknown')
        })
        
        return state
    
    async def _execute_step(self, state: LNNCouncilState, step_name: str) -> LNNCouncilState:
        """Execute step with comprehensive observability."""
        request_id = getattr(state.current_request, 'request_id', 'unknown')
        
        # Monitor step execution
        with self.observability_engine.monitor_component(f"workflow_step", step_name):
            try:
                # Add trace step
                self.observability_engine.add_trace_step(request_id, step_name, {
                    "step_start": asyncio.get_event_loop().time(),
                    "current_step": state.current_step
                })
                
                result_state = await self.workflow_engine.execute_step(state, step_name)
                
                # Record successful step completion
                self.observability_engine.add_trace_step(request_id, f"{step_name}_complete", {
                    "success": True,
                    "next_step": result_state.next_step
                })
                
                return result_state
                
            except Exception as e:
                # Record error in observability
                self.observability_engine.record_error(f"workflow_step_{step_name}", e, {
                    "request_id": request_id,
                    "current_step": state.current_step,
                    "step_name": step_name
                })
                
                if self.lnn_config.enable_fallback and not state.fallback_triggered:
                    logger.warning(f"Step {step_name} failed, using fallback", error=str(e))
                    
                    # Add fallback trace step
                    self.observability_engine.add_trace_step(request_id, f"{step_name}_fallback", {
                        "error": str(e),
                        "fallback_triggered": True
                    })
                    
                    return await self.fallback_engine.handle_failure(state, step_name, e)
                raise
    
    def _extract_output(self, final_state: LNNCouncilState) -> GPUAllocationDecision:
        """Extract final output with decision trace completion."""
        request_id = getattr(final_state.current_request, 'request_id', 'unknown')
        
        # Extract the decision
        decision = self.workflow_engine.extract_output(final_state)
        
        # Complete the decision trace
        confidence_score = getattr(decision, 'confidence_score', final_state.confidence_score)
        reasoning_path = getattr(decision, 'reasoning_path', [])
        final_decision = getattr(decision, 'decision', 'unknown')
        
        self.observability_engine.complete_decision_trace(
            request_id,
            final_decision,
            confidence_score,
            reasoning_path,
            final_state.fallback_triggered
        )
        
        return decision
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with comprehensive observability."""
        base_health = await super().health_check()
        
        # Add component health
        base_health.update({
            "neural_engine": await self.neural_engine.health_check(),
            "workflow_engine": self.workflow_engine.get_status(),
            "fallback_engine": self.fallback_engine.get_health_status(),
            "observability_engine": self.observability_engine.get_performance_summary()
        })
        
        return base_health
    
    async def attempt_recovery(self, subsystem: str) -> bool:
        """Attempt to recover a failed subsystem."""
        return await self.fallback_engine.attempt_recovery(subsystem)
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get detailed fallback metrics."""
        return self.fallback_engine.get_health_status()["metrics"]
    
    def reset_fallback_state(self):
        """Reset fallback state (useful for testing and recovery)."""
        self.fallback_engine.reset_metrics()
    
    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        return self.observability_engine.get_performance_summary()
    
    def get_decision_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed trace for a specific decision."""
        trace = self.observability_engine.get_decision_trace(request_id)
        if trace:
            return {
                "request_id": trace.request_id,
                "start_time": trace.start_time,
                "end_time": trace.end_time,
                "steps": trace.steps,
                "confidence_score": trace.confidence_score,
                "reasoning_path": trace.reasoning_path,
                "final_decision": trace.final_decision,
                "fallback_triggered": trace.fallback_triggered,
                "total_time": (trace.end_time - trace.start_time) if trace.end_time else None
            }
        return None
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent system alerts."""
        alerts = self.observability_engine.get_recent_alerts(hours)
        return [
            {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "context": alert.context,
                "actionable_info": alert.actionable_info
            }
            for alert in alerts
        ]