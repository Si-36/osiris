"""
Minimal working version of real_components.py
Just provides the required classes/functions for real_registry.py to work
"""

from typing import Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field
import time
import asyncio


class ComponentType(Enum):
    NEURAL = "neural"
    MEMORY = "memory"
    AGENT = "agent"
    TDA = "tda"
    ORCHESTRATION = "orchestration"
    OBSERVABILITY = "observability"


@dataclass
class RealComponent:
    """Minimal real component for compatibility"""
    id: str
    type: ComponentType
    status: str = "active"
    processing_time: float = 0.0
    data_processed: int = 0
    last_output: Optional[Any] = None
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """Process data through component"""
        # Basic processing - real logic is in real_registry.py
        await asyncio.sleep(0.001)  # Simulate some processing
        return {
            "component_id": self.id,
            "type": self.type.value,
            "processed": True,
            "data": str(data)[:100]
        }


def create_real_component(component_id: str, component_type: str) -> RealComponent:
    """Factory function to create real components"""
    # Map string to enum
    type_map = {
        "neural": ComponentType.NEURAL,
        "memory": ComponentType.MEMORY,
        "agent": ComponentType.AGENT,
        "tda": ComponentType.TDA,
        "orchestration": ComponentType.ORCHESTRATION,
        "observability": ComponentType.OBSERVABILITY
    }
    
    comp_type = type_map.get(component_type, ComponentType.NEURAL)
    return RealComponent(id=component_id, type=comp_type)


# Export what real_registry needs
__all__ = ['RealComponent', 'create_real_component', 'ComponentType']