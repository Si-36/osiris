"""
Attention Mechanism - Integrated with Global Workspace

Instead of creating a separate attention mechanism, we use the existing
Global Workspace which already handles:
- Information prioritization (attention_weight in WorkspaceContent)
- Resource allocation (through priority and attention_weight)
- Salience detection (through priority levels)
- Focus management (through ConsciousnessStream.current_focus)

This is more efficient and avoids duplication.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .global_workspace import (
    GlobalWorkspace, 
    ConsciousnessStream, 
    WorkspaceContent,
    get_global_workspace
)


@dataclass
class AttentionFocus:
    """Attention focus - just an alias for workspace focus."""
    primary_target: str
    secondary_targets: List[str]
    focus_strength: float
    timestamp: float = 0.0


class AttentionMechanism:
    """
    Attention mechanism that uses the existing Global Workspace.
    No need to reinvent the wheel - workspace already handles attention.
    """
    
    def __init__(self):
        self.workspace_controller = get_global_workspace()
    
    def process_event(self, event_data: Dict[str, Any]) -> float:
        """Process event using existing workspace priority system."""
        # Convert event to workspace content
        content = WorkspaceContent(
            content_id=f"event_{event_data.get('source', 'unknown')}",
            source=event_data.get('source', 'unknown'),
            data=event_data,
            priority=self._get_priority(event_data),
            attention_weight=self._calculate_attention_weight(event_data)
        )
        
        # Use existing workspace processing
        self.workspace_controller.process_content(content)
        
        return content.attention_weight
    
    def set_focus(self, target: str, strength: float = 1.0) -> AttentionFocus:
        """Set focus using existing workspace focus."""
        self.workspace_controller.stream.set_attention_focus(target)
        
        return AttentionFocus(
            primary_target=target,
            secondary_targets=[],
            focus_strength=strength
        )
    
    def get_attention_state(self) -> Dict[str, Any]:
        """Get attention state from workspace."""
        pass
        return self.workspace_controller.get_state()
    
    def _get_priority(self, event_data: Dict[str, Any]) -> int:
        """Convert event data to priority."""
        priority_map = {'critical': 5, 'high': 4, 'medium': 3, 'low': 2}
        return priority_map.get(event_data.get('priority', 'medium'), 3)
    
    def _calculate_attention_weight(self, event_data: Dict[str, Any]) -> float:
        """Calculate attention weight from event data."""
        intensity = event_data.get('intensity', 0.5)
        urgency = event_data.get('urgency', 0.5)
        return min(1.0, (intensity + urgency) / 2.0)


# Factory functions - just return workspace-based attention
    def create_attention_mechanism() -> AttentionMechanism:
        """Create attention mechanism using existing workspace."""
        return AttentionMechanism()


    def get_attention_mechanism() -> AttentionMechanism:
        """Get attention mechanism (workspace-based)."""
        return create_attention_mechanism()


# Aliases for compatibility
SalienceDetector = AttentionMechanism
ResourceAllocator = AttentionMechanism
TransformerAttention = AttentionMechanism