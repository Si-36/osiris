"""
🧠 Neural Observability Core - Latest 2025 Architecture
Main orchestrator for the complete sensory system of the digital organism.

Professional modular design with dependency injection and clean separation of concerns.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

try:
    from .opentelemetry_integration import OpenTelemetryManager
    from .langsmith_integration import LangSmithManager
    from .prometheus_metrics import PrometheusMetricsManager
    from .structured_logging import StructuredLoggingManager
    from .knowledge_graph import KnowledgeGraphManager
    from .health_monitor import OrganismHealthMonitor
    from .context_managers import ObservabilityContext
except ImportError:
    # Fallback for absolute imports
    from aura_intelligence.observability.langsmith_integration import LangSmithManager
    from aura_intelligence.observability.prometheus_metrics import PrometheusMetricsManager
    from aura_intelligence.observability.structured_logging import StructuredLoggingManager
    from aura_intelligence.observability.knowledge_graph import KnowledgeGraphManager
    from aura_intelligence.observability.health_monitor import OrganismHealthMonitor
    from aura_intelligence.observability.context_managers import ObservabilityContext
    # OpenTelemetry is optional
    try:
        from aura_intelligence.observability.opentelemetry_integration import OpenTelemetryManager
    except ImportError:
        OpenTelemetryManager = None

# Create config class
class ObservabilityConfig:
    def __init__(self):
        self.enable_metrics = True
        self.enable_tracing = True
        self.prometheus_enabled = True
        self.langsmith_enabled = False
        self.organism_id = "aura_organism"
        self.organism_generation = "2025"
        self.langsmith_api_key = None
        
    def is_feature_enabled(self, feature):
        return True

    # Mock CollectiveState for standalone testing
class CollectiveState:
    """Mock CollectiveState for standalone testing."""
    def __init__(self):
        self.messages = []
        self.workflow_id = "mock_workflow"
        self.evidence_log = []
        self.error_log = []
        self.error_recovery_attempts = 0
        self.system_health = {"health_score": 0.95}


class NeuralObservabilityCore:
    """
    Complete sensory system for the digital organism.
    
    Latest 2025 architecture with:
        pass
    - Dependency injection for all components
    - Clean separation of concerns
    - Professional error handling
    - Async-first design
    - Integration with existing CollectiveState
    """
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """
        Initialize the neural observability core.
        
        Args:
            pass
        config: Observability configuration (uses default if None)
        """
        pass
        
        self.config = config or ObservabilityConfig()
        self.start_time = time.time()
        self.is_initialized = False
        
        # Component managers (initialized lazily)
        self._opentelemetry: Optional[OpenTelemetryManager] = None
        self._langsmith: Optional[LangSmithManager] = None
        self._prometheus: Optional[PrometheusMetricsManager] = None
        self._logging: Optional[StructuredLoggingManager] = None
        self._knowledge_graph: Optional[KnowledgeGraphManager] = None
        self._health_monitor: Optional[OrganismHealthMonitor] = None
        
        # Current context tracking
        self._current_contexts: Dict[str, ObservabilityContext] = {}
    
    async def initialize(self) -> None:
        """
        Initialize all observability components.
        
        Professional initialization with proper error handling and dependency management.
        """
        
        if self.is_initialized:
            return
        
        try:
            # Initialize components in dependency order
            await self._initialize_structured_logging()
            await self._initialize_opentelemetry()
            await self._initialize_langsmith()
            await self._initialize_prometheus()
            await self._initialize_knowledge_graph()
            await self._initialize_health_monitor()
            
            self.is_initialized = True
            
            # Log successful initialization
            if self._logging:
                self._logging.logger.info(
                    "neural_observability_initialized",
                    organism_id=self.config.organism_id,
                    generation=self.config.organism_generation,
                    components_initialized={
                        "opentelemetry": self._opentelemetry is not None,
                        "langsmith": self._langsmith is not None,
                        "prometheus": self._prometheus is not None,
                        "logging": self._logging is not None,
                        "knowledge_graph": self._knowledge_graph is not None,
                        "health_monitor": self._health_monitor is not None,
                    },
                    initialization_time_seconds=time.time() - self.start_time
                )
        
        except Exception as e:
            # Graceful degradation - log error but don't fail completely
            print(f"⚠️ Neural observability initialization warning: {e}")
            self.is_initialized = True  # Continue with partial functionality
    
        async def _initialize_structured_logging(self) -> None:
            pass
        """Initialize structured logging first (needed by other components)."""
        try:
            self._logging = StructuredLoggingManager(self.config)
            await self._logging.initialize()
        except Exception as e:
            print(f"⚠️ Structured logging initialization failed: {e}")
    
        async def _initialize_opentelemetry(self) -> None:
            pass
        """Initialize OpenTelemetry tracing and metrics."""
        
        try:
            self._opentelemetry = OpenTelemetryManager(self.config)
            await self._opentelemetry.initialize()
        except Exception as e:
            if self._logging:
                self._logging.logger.warning("opentelemetry_init_failed", error=str(e))
    
        async def _initialize_langsmith(self) -> None:
            pass
        """Initialize LangSmith 2.0 integration."""
        try:
            if self.config.langsmith_api_key:
                self._langsmith = LangSmithManager(self.config)
                await self._langsmith.initialize()
        except Exception as e:
            if self._logging:
                self._logging.logger.warning("langsmith_init_failed", error=str(e))
    
        async def _initialize_prometheus(self) -> None:
            pass
        """Initialize Prometheus metrics."""
        
        try:
            self._prometheus = PrometheusMetricsManager(self.config)
            await self._prometheus.initialize()
        except Exception as e:
            if self._logging:
                self._logging.logger.warning("prometheus_init_failed", error=str(e))
    
        async def _initialize_knowledge_graph(self) -> None:
            pass
        """Initialize knowledge graph for memory consolidation."""
        try:
            if self.config.is_feature_enabled("memory_consolidation"):
                self._knowledge_graph = KnowledgeGraphManager(self.config)
                await self._knowledge_graph.initialize()
        except Exception as e:
            if self._logging:
                self._logging.logger.warning("knowledge_graph_init_failed", error=str(e))
    
        async def _initialize_health_monitor(self) -> None:
            pass
        """Initialize organism health monitoring."""
        
        try:
            self._health_monitor = OrganismHealthMonitor(
                self.config, 
                self._prometheus,
                self._logging
            )
            await self._health_monitor.initialize()
        except Exception as e:
            if self._logging:
                self._logging.logger.warning("health_monitor_init_failed", error=str(e))
    
    @asynccontextmanager
    async def observe_workflow(
        self, 
        state: CollectiveState, 
        workflow_type: str = "collective_intelligence"
    ):
        """
        Observe a complete workflow execution with full telemetry.
        
        Integrates seamlessly with existing CollectiveState from Phase 1 & 2.
        
        Args:
            state: The CollectiveState from our existing system
            workflow_type: Type of workflow being executed
            
        Yields:
            ObservabilityContext: Context for workflow execution
        """
        
        # Ensure initialization
        if not self.is_initialized:
            await self.initialize()
        
        # Extract workflow metadata from state
        workflow_id = state.get("workflow_id", f"workflow_{int(time.time())}")
        evidence_log = state.get("evidence_log", [])
        error_log = state.get("error_log", [])
        recovery_attempts = state.get("error_recovery_attempts", 0)
        system_health = state.get("system_health", {})
        
        metadata = {
            "evidence_count": len(evidence_log),
            "has_errors": bool(error_log),
            "recovery_attempts": recovery_attempts,
            "system_health_status": system_health.get("current_health_status", "unknown"),
            "agents_involved": self._extract_agents_from_state(state)
        }
        
        # Create observability context
        context = ObservabilityContext(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            metadata=metadata,
            config=self.config
        )
        
        # Store context for correlation
        self._current_contexts[workflow_id] = context
        
        start_time = time.time()
        
        try:
            # Start tracing across all systems
            await self._start_workflow_tracing(context, state)
            
            # Yield context for workflow execution
            yield context
            
            # Success path
            duration = time.time() - start_time
            await self._complete_workflow_tracing(context, state, duration, "success")
            
        except Exception as e:
            # Error path
            duration = time.time() - start_time
            await self._complete_workflow_tracing(context, state, duration, "failed", str(e))
            raise
            
        finally:
            # Cleanup
            self._current_contexts.pop(workflow_id, None)
    
        async def _start_workflow_tracing(self, context: ObservabilityContext, state: CollectiveState) -> None:
            pass
        """Start tracing across all observability systems."""
        
        # OpenTelemetry tracing
        if self._opentelemetry:
            await self._opentelemetry.start_workflow_span(context, state)
        
        # LangSmith tracing
        if self._langsmith:
            await self._langsmith.start_workflow_run(context, state)
        
        # Prometheus metrics
        if self._prometheus:
            self._prometheus.record_workflow_started(context)
        
        # Structured logging
        if self._logging:
            self._logging.log_workflow_started(context, state)
        
        # Health monitoring
        if self._health_monitor:
            await self._health_monitor.on_workflow_started(context, state)
    
        async def _complete_workflow_tracing(
        self, 
        context: ObservabilityContext, 
        state: CollectiveState, 
        duration: float, 
        status: str, 
        error: Optional[str] = None
        ) -> None:
            pass
        """Complete tracing across all observability systems."""
        
        # Update context with completion data
        context.duration = duration
        context.status = status
        context.error = error
        
        # OpenTelemetry completion
        if self._opentelemetry:
            await self._opentelemetry.complete_workflow_span(context, state)
        
        # LangSmith completion
        if self._langsmith:
            await self._langsmith.complete_workflow_run(context, state)
        
        # Prometheus metrics
        if self._prometheus:
            self._prometheus.record_workflow_completed(context)
        
        # Structured logging
        if self._logging:
            self._logging.log_workflow_completed(context, state)
        
        # Knowledge graph recording
        if self._knowledge_graph:
            await self._knowledge_graph.record_workflow_event(context, state)
        
        # Health monitoring
        if self._health_monitor:
            await self._health_monitor.on_workflow_completed(context, state)
    
    def _extract_agents_from_state(self, state: CollectiveState) -> list:
        """Extract agent information from CollectiveState."""
        
        messages = state.get("messages", [])
        agents = set()
        
        for message in messages:
            if hasattr(message, 'additional_kwargs') and 'agent' in message.additional_kwargs:
                agents.add(message.additional_kwargs['agent'])
        
        return list(agents)
    
    @asynccontextmanager
    async def observe_agent_call(
        self, 
        agent_name: str, 
        tool_name: str, 
        inputs: Dict[str, Any] = None
    ):
        """
        Observe individual agent/tool calls with comprehensive telemetry.
        
        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool being called
            inputs: Input parameters for the tool
            
        Yields:
            Dict: Context for agent execution
        """
        
        inputs = inputs or {}
        start_time = time.time()
        
        # Get current workflow context for correlation
        current_workflow = None
        for context in self._current_contexts.values():
            current_workflow = context
            break  # Use the most recent workflow context
        
        agent_context = {
            'agent_name': agent_name,
            'tool_name': tool_name,
            'inputs': inputs,
            'start_time': start_time,
            'workflow_context': current_workflow
        }
        
        try:
            # Start agent tracing
            await self._start_agent_tracing(agent_context)
            
            yield agent_context
            
            # Success path
            duration = time.time() - start_time
            await self._complete_agent_tracing(agent_context, duration, "success")
            
        except Exception as e:
            # Error path
            duration = time.time() - start_time
            await self._complete_agent_tracing(agent_context, duration, "failed", str(e))
            raise
    
        async def _start_agent_tracing(self, agent_context: Dict[str, Any]) -> None:
            pass
        """Start agent tracing across all systems."""
        
        if self._opentelemetry:
            await self._opentelemetry.start_agent_span(agent_context)
        
        if self._prometheus:
            self._prometheus.record_agent_started(agent_context)
        
        if self._logging:
            self._logging.log_agent_started(agent_context)
    
        async def _complete_agent_tracing(
        self, 
        agent_context: Dict[str, Any], 
        duration: float, 
        status: str, 
        error: Optional[str] = None
        ) -> None:
            pass
        """Complete agent tracing across all systems."""
        
        agent_context.update({
            'duration': duration,
            'status': status,
            'error': error
        })
        
        if self._opentelemetry:
            await self._opentelemetry.complete_agent_span(agent_context)
        
        if self._prometheus:
            self._prometheus.record_agent_completed(agent_context)
        
        if self._logging:
            self._logging.log_agent_completed(agent_context)
    
        async def track_llm_usage(
        self, 
        model_name: str, 
        input_tokens: int, 
        output_tokens: int,
        latency_seconds: float, 
        cost_usd: Optional[float] = None
        ) -> None:
            pass
        """Track LLM usage with latest 2025 cost and performance patterns."""
        
        if self._prometheus:
            await self._prometheus.track_llm_usage(
                model_name, input_tokens, output_tokens, latency_seconds, cost_usd
            )
        
        if self._logging:
            self._logging.log_llm_usage(
                model_name, input_tokens, output_tokens, latency_seconds, cost_usd
            )
    
        async def track_error_recovery(
        self, 
        error_type: str, 
        recovery_strategy: str, 
        success: bool
        ) -> None:
            pass
        """Track error recovery attempts (integrates with Phase 1, Step 2)."""
        
        if self._prometheus:
            self._prometheus.record_error_recovery(error_type, recovery_strategy, success)
        
        if self._logging:
            self._logging.log_error_recovery(error_type, recovery_strategy, success)
        
        if self._health_monitor:
            await self._health_monitor.on_error_recovery(error_type, recovery_strategy, success)
    
        async def update_system_health(self, health_score: float) -> None:
            pass
        """Update overall system health score."""
        
        if self._prometheus:
            self._prometheus.update_system_health(health_score)
        
        if self._logging:
            self._logging.log_system_health_update(health_score)
        
        if self._health_monitor:
            await self._health_monitor.update_health_score(health_score)
    
        async def shutdown(self) -> None:
            pass
        """Gracefully shutdown all observability components."""
        
        if self._logging:
            self._logging.logger.info("neural_observability_shutting_down")
        
        # Shutdown components in reverse dependency order
        if self._health_monitor:
            await self._health_monitor.shutdown()
        
        if self._knowledge_graph:
            await self._knowledge_graph.shutdown()
        
        if self._prometheus:
            await self._prometheus.shutdown()
        
        if self._langsmith:
            await self._langsmith.shutdown()
        
        if self._opentelemetry:
            await self._opentelemetry.shutdown()
        
        if self._logging:
            await self._logging.shutdown()
        
        self.is_initialized = False
    
    # Properties for component access
    @property
    def opentelemetry(self) -> Optional[OpenTelemetryManager]:
        """Access to OpenTelemetry manager."""
        return self._opentelemetry
    
    @property
    def langsmith(self) -> Optional[LangSmithManager]:
        """Access to LangSmith manager."""
        return self._langsmith
    
    @property
    def prometheus(self) -> Optional[PrometheusMetricsManager]:
        """Access to Prometheus metrics manager."""
        return self._prometheus
    
    @property
    def logging(self) -> Optional[StructuredLoggingManager]:
        """Access to structured logging manager."""
        return self._logging
    
    @property
    def knowledge_graph(self) -> Optional[KnowledgeGraphManager]:
        """Access to knowledge graph manager."""
        return self._knowledge_graph
    
    @property
    def health_monitor(self) -> Optional[OrganismHealthMonitor]:
        """Access to health monitor."""
        return self._health_monitor
