"""
📜 Core Protocol Definitions
===========================

Shared protocol definitions to avoid circular imports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


# ==================== FIPA ACL Performatives ====================

class Performative(Enum):
    """FIPA ACL performatives for agent communication"""
    # Assertives
    INFORM = "inform"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"
    
    # Directives
    REQUEST = "request"
    REQUEST_WHEN = "request-when"
    REQUEST_WHENEVER = "request-whenever"
    QUERY_IF = "query-if"
    QUERY_REF = "query-ref"
    
    # Commissives
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    PROPOSE = "propose"
    
    # Permissives
    AGREE = "agree"
    REFUSE = "refuse"
    
    # Declaratives
    FAILURE = "failure"
    NOT_UNDERSTOOD = "not-understood"
    
    # Others
    SUBSCRIBE = "subscribe"
    CANCEL = "cancel"
    CFP = "cfp"  # Call for proposal


# ==================== Message Priority ====================

class MessagePriority(Enum):
    """Message priority levels for routing"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# ==================== Semantic Envelope ====================

@dataclass
class SemanticEnvelope:
    """
    FIPA ACL-inspired message envelope for semantic agent communication
    """
    # Required fields
    performative: Performative
    sender: str
    receiver: Union[str, List[str]]
    content: Any
    
    # Optional fields
    language: str = "aura-json"
    ontology: str = "aura-core"
    protocol: str = "fipa-request"
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reply_with: Optional[str] = None
    in_reply_to: Optional[str] = None
    reply_by: Optional[datetime] = None
    
    # Metadata
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ==================== Agent Message ====================

@dataclass
class AgentMessage:
    """Agent-to-agent message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = ""
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    retry_count: int = 0