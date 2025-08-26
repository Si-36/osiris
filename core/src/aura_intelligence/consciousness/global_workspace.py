"""
Global Workspace Theory Implementation

Clean, professional implementation of Global Workspace Theory for distributed consciousness.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import uuid

from ..core.config import get_config


class ConsciousnessLevel(Enum):
    UNCONSCIOUS = 0
    PRECONSCIOUS = 1
    CONSCIOUS = 2
    METACONSCIOUS = 3


@dataclass
class WorkspaceContent:
    """Content in the global workspace."""
    content_id: str
    source: str
    data: Dict[str, Any]
    priority: int = 1
    attention_weight: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ConsciousDecision:
    """A conscious decision with reasoning."""
    decision_id: str
    options: List[Dict[str, Any]]
    chosen: Dict[str, Any]
    confidence: float
    reasoning: List[str]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ConsciousnessStream:
    """Continuous awareness processing."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.stream = []
        self.current_focus: Optional[str] = None
        self.consciousness_level = ConsciousnessLevel.PRECONSCIOUS
    
    def add_content(self, content: WorkspaceContent) -> None:
        """Add content to consciousness stream."""
        self.stream.append(content)
        if len(self.stream) > self.buffer_size:
            self.stream.pop(0)
        
        # Update consciousness level based on attention
        if content.attention_weight > 0.7:
            self.consciousness_level = ConsciousnessLevel.CONSCIOUS
            self.current_focus = content.content_id
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current consciousness state."""
        pass
        return {
            "level": self.consciousness_level.name,
            "focus": self.current_focus,
            "stream_length": len(self.stream),
            "recent_content": [c.content_id for c in self.stream[-5:]]
        }


class GlobalWorkspace:
    """Global workspace for information broadcasting."""
    
    def __init__(self):
        self.content: Dict[str, WorkspaceContent] = {}
        self.subscribers: Dict[str, Set[str]] = {}  # content_type -> component_ids
        self.broadcast_queue = asyncio.Queue()
    
    def subscribe(self, component_id: str, content_types: List[str]) -> None:
        """Subscribe component to content types."""
        for content_type in content_types:
            if content_type not in self.subscribers:
                self.subscribers[content_type] = set()
            self.subscribers[content_type].add(component_id)
    
        async def broadcast(self, content: WorkspaceContent) -> None:
        """Broadcast content to subscribers."""
        self.content[content.content_id] = content
        await self.broadcast_queue.put(content)
    
    def get_top_content(self, limit: int = 10) -> List[WorkspaceContent]:
        """Get content by attention weight."""
        return sorted(
            self.content.values(),
            key=lambda c: c.attention_weight,
            reverse=True
        )[:limit]


class MetaCognitiveController:
    """Main consciousness controller."""
    
    def __init__(self):
        self.workspace = GlobalWorkspace()
        self.stream = ConsciousnessStream()
        self.decisions: List[ConsciousDecision] = []
        self.active = False
    
        async def start(self) -> None:
        """Start the controller."""
        pass
        self.active = True
    
        async def stop(self) -> None:
        """Stop the controller."""
        pass
        self.active = False
    
        async def process_content(self, content: WorkspaceContent) -> None:
        """Process content through workspace and stream."""
        await self.workspace.broadcast(content)
        self.stream.add_content(content)
    
        async def make_decision(
        self,
        options: List[Dict[str, Any]],
        context: Dict[str, Any] = None
        ) -> ConsciousDecision:
        """Make a conscious decision."""
        # Simple scoring: choose option with highest 'score' or first one
        best_option = max(options, key=lambda x: x.get('score', 0))
        
        # Calculate confidence based on score difference
        scores = [opt.get('score', 0) for opt in options]
        confidence = best_option.get('score', 0.5) if scores else 0.5
        
        decision = ConsciousDecision(
            decision_id=str(uuid.uuid4()),
            options=options,
            chosen=best_option,
            confidence=confidence,
            reasoning=[f"Selected option with score {best_option.get('score', 0)}"]
        )
        
        self.decisions.append(decision)
        return decision
    
    def get_state(self) -> Dict[str, Any]:
        """Get controller state."""
        pass
        return {
            "active": self.active,
            "workspace_content_count": len(self.workspace.content),
            "consciousness_state": self.stream.get_current_state(),
            "decisions_made": len(self.decisions)
        }


# Factory functions
    def create_metacognitive_controller() -> MetaCognitiveController:
        """Create metacognitive controller."""
        return MetaCognitiveController()


_global_controller: Optional[MetaCognitiveController] = None


    def get_global_workspace() -> MetaCognitiveController:
        """Get global controller instance."""
        global _global_controller
        if _global_controller is None:
        _global_controller = create_metacognitive_controller()
        return _global_controller
