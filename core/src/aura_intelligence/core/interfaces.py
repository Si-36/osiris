"""
Core Interfaces for AURA Intelligence Architecture

This module defines the fundamental interfaces that all system components
must implement, providing the contract for the cognitive-topological architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import asyncio

# Type variables for generic interfaces
T = TypeVar('T')
R = TypeVar('R')


class ComponentStatus(Enum):
    """Status of a system component."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class HealthStatus:
    """Health status of a component."""
    status: ComponentStatus
    message: str
    metrics: Dict[str, Any]
    last_check: float
    error_count: int = 0


class SystemComponent(ABC):
    """
    Base interface for all system components.
    
    Provides the fundamental contract that all components in the AURA
    Intelligence system must implement, including lifecycle management,
    health monitoring, and configuration.
    """
    
    def __init__(self, component_id: str, config: Dict[str, Any]):
        self.component_id = component_id
        self.config = config
        self.status = ComponentStatus.INITIALIZING
        self._health_status = HealthStatus(
            status=ComponentStatus.INITIALIZING,
            message="Component initializing",
            metrics={},
            last_check=0.0
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the component."""
        pass
    
    @abstractmethod

    
    async def health_check(self) -> HealthStatus:
        """Check the health of the component."""
        pass
    
    @abstractmethod

    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        pass
    
    async def restart(self) -> None:
        """Restart the component."""
        await self.stop()
        await self.start()


class CognitiveComponent(SystemComponent):
    """
    Interface for components that participate in cognitive processing.
    
    Extends SystemComponent with consciousness-aware capabilities including
    attention, working memory, and executive function integration.
    """
    
    @abstractmethod
    async def process_conscious_input(self, input_data: Any) -> Any:
        """Process input through consciousness mechanisms."""
        pass
    
    @abstractmethod
    async def get_attention_weight(self) -> float:
        """Get the current attention weight for this component."""
        pass

    
    @abstractmethod
    async def update_working_memory(self, memory_update: Dict[str, Any]) -> None:
        """Update working memory with new information."""
        pass
    
    @abstractmethod
    async def get_consciousness_contribution(self) -> Dict[str, Any]:
        """Get this component's contribution to global consciousness."""



class TopologicalComponent(SystemComponent):
    """
    Interface for components that perform topological computations.
    
    Extends SystemComponent with topological data analysis capabilities
    including persistent homology and sheaf-theoretic operations.
    """
    
    @abstractmethod
    async def compute_topology(self, data: Any) -> Dict[str, Any]:
        """Compute topological features of the data."""
        pass
    
    @abstractmethod
    async def get_topological_signature(self) -> Dict[str, Any]:
        """Get the topological signature of the component's state."""
        pass

    
    @abstractmethod
    async def verify_topological_consistency(self) -> bool:
        """Verify topological consistency of the component."""



class QuantumComponent(SystemComponent):
    """
    Interface for components that perform quantum computations.
    
    Extends SystemComponent with quantum computing capabilities including
    quantum circuit execution and quantum-classical interfaces.
    """
    
    @abstractmethod
    async def execute_quantum_circuit(self, circuit: Any) -> Any:
        """Execute a quantum circuit."""
        pass
    
    @abstractmethod
    async def get_quantum_state(self) -> Dict[str, Any]:
        """Get the current quantum state."""
        pass

    
    @abstractmethod
    async def measure_quantum_observables(self) -> Dict[str, float]:
        """Measure quantum observables."""
        pass

    
    @abstractmethod
    async def verify_quantum_coherence(self) -> bool:
        """Verify quantum coherence is maintained."""



class Configurable(ABC):
    """Interface for configurable components."""
    
    @abstractmethod
    async def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update component configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""


class Observable(ABC):
    """Interface for observable components."""
    
    @abstractmethod
    async def subscribe_to_events(self, event_types: List[str]) -> asyncio.Queue:
        """Subscribe to component events."""
        pass
    
    @abstractmethod
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Emit an event."""


class Recoverable(ABC):
    """Interface for components that support recovery operations."""
    
    @abstractmethod
    async def create_checkpoint(self) -> Dict[str, Any]:
        """Create a recovery checkpoint."""
        pass

    
    @abstractmethod
    async def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from a checkpoint."""
        pass
    
    @abstractmethod
    async def get_recovery_info(self) -> Dict[str, Any]:
        """Get recovery information."""
        pass
