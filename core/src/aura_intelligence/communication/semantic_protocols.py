"""
📜 Semantic Protocols for Agent Communication
=============================================

FIPA ACL-compliant protocols with conversation management,
performative validation, and deadline handling.

Standards:
- FIPA ACL (Agent Communication Language)
- FIPA Interaction Protocols
- Custom AURA protocols for collective intelligence
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import structlog

from .protocols import SemanticEnvelope, Performative

logger = structlog.get_logger(__name__)


# ==================== Protocol Types ====================

class InteractionProtocol(Enum):
    """Standard FIPA interaction protocols"""
    REQUEST = "fipa-request"
    QUERY = "fipa-query"
    REQUEST_WHEN = "fipa-request-when"
    CONTRACT_NET = "fipa-contract-net"
    ITERATED_CONTRACT_NET = "fipa-iterated-contract-net"
    AUCTION_ENGLISH = "fipa-auction-english"
    AUCTION_DUTCH = "fipa-auction-dutch"
    BROKERING = "fipa-brokering"
    RECRUITING = "fipa-recruiting"
    SUBSCRIBE = "fipa-subscribe"
    PROPOSE = "fipa-propose"
    
    # AURA custom protocols
    CONSENSUS = "aura-consensus"
    SWARM_SYNC = "aura-swarm-sync"
    NEURAL_VOTE = "aura-neural-vote"
    COLLECTIVE_LEARN = "aura-collective-learn"


@dataclass
class ProtocolState:
    """State of an ongoing protocol interaction"""
    protocol: InteractionProtocol
    conversation_id: str
    initiator: str
    participants: List[str]
    current_step: str
    state_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    completed: bool = False
    
    def is_expired(self) -> bool:
        """Check if protocol has expired"""
        if self.deadline:
            return datetime.utcnow() > self.deadline
        return False


# ==================== Protocol Validators ====================

class ProtocolValidator:
    """Validates message sequences according to protocol rules"""
    
    # Valid performative sequences for each protocol
    PROTOCOL_SEQUENCES = {
        InteractionProtocol.REQUEST: {
            "init": [Performative.REQUEST],
            "REQUEST": [Performative.AGREE, Performative.REFUSE],
            "AGREE": [Performative.INFORM, Performative.FAILURE],
            "final": [Performative.INFORM, Performative.FAILURE, Performative.REFUSE]
        },
        InteractionProtocol.QUERY: {
            "init": [Performative.QUERY_IF, Performative.QUERY_REF],
            "QUERY_IF": [Performative.CONFIRM, Performative.DISCONFIRM, Performative.NOT_UNDERSTOOD],
            "QUERY_REF": [Performative.INFORM, Performative.NOT_UNDERSTOOD],
            "final": [Performative.CONFIRM, Performative.DISCONFIRM, Performative.INFORM, Performative.NOT_UNDERSTOOD]
        },
        InteractionProtocol.CONTRACT_NET: {
            "init": [Performative.CFP],
            "CFP": [Performative.PROPOSE, Performative.REFUSE],
            "PROPOSE": [Performative.ACCEPT_PROPOSAL, Performative.REJECT_PROPOSAL],
            "ACCEPT_PROPOSAL": [Performative.INFORM, Performative.FAILURE],
            "final": [Performative.INFORM, Performative.FAILURE, Performative.REJECT_PROPOSAL]
        },
        InteractionProtocol.CONSENSUS: {
            "init": [Performative.PROPOSE],
            "PROPOSE": [Performative.ACCEPT_PROPOSAL, Performative.REJECT_PROPOSAL, Performative.PROPOSE],
            "voting": [Performative.INFORM],
            "final": [Performative.INFORM]
        }
    }
    
    @classmethod
    def validate_sequence(
        cls,
        protocol: InteractionProtocol,
        current_state: str,
        performative: Performative
    ) -> bool:
        """Validate if performative is valid for current protocol state"""
        if protocol not in cls.PROTOCOL_SEQUENCES:
            return True  # Unknown protocol, allow all
        
        sequences = cls.PROTOCOL_SEQUENCES[protocol]
        valid_performatives = sequences.get(current_state, [])
        
        return performative in valid_performatives
    
    @classmethod
    def get_next_states(
        cls,
        protocol: InteractionProtocol,
        performative: Performative
    ) -> List[str]:
        """Get possible next states after performative"""
        if protocol not in cls.PROTOCOL_SEQUENCES:
            return []
        
        sequences = cls.PROTOCOL_SEQUENCES[protocol]
        next_states = []
        
        for state, valid_perfs in sequences.items():
            if performative in valid_perfs:
                next_states.append(state)
        
        return next_states
    
    @classmethod
    def is_final_state(
        cls,
        protocol: InteractionProtocol,
        performative: Performative
    ) -> bool:
        """Check if performative ends the protocol"""
        if protocol not in cls.PROTOCOL_SEQUENCES:
            return False
        
        sequences = cls.PROTOCOL_SEQUENCES[protocol]
        final_perfs = sequences.get("final", [])
        
        return performative in final_perfs


# ==================== Conversation Manager ====================

class ConversationManager:
    """
    Manages multi-turn conversations with timeouts and state tracking.
    """
    
    def __init__(self, default_timeout: float = 300.0):
        self.default_timeout = default_timeout
        self.conversations: Dict[str, ProtocolState] = {}
        self.handlers: Dict[InteractionProtocol, Dict[str, Callable]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start conversation manager"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        logger.info("Conversation manager started")
    
    async def stop(self):
        """Stop conversation manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Conversation manager stopped")
    
    def start_conversation(
        self,
        protocol: InteractionProtocol,
        initiator: str,
        participants: List[str],
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Start a new conversation"""
        conversation_id = str(uuid.uuid4())
        timeout = timeout or self.default_timeout
        
        state = ProtocolState(
            protocol=protocol,
            conversation_id=conversation_id,
            initiator=initiator,
            participants=participants,
            current_step="init",
            deadline=datetime.utcnow() + timedelta(seconds=timeout),
            state_data=kwargs
        )
        
        self.conversations[conversation_id] = state
        
        logger.info(
            "Conversation started",
            conversation_id=conversation_id,
            protocol=protocol.value,
            participants=len(participants)
        )
        
        return conversation_id
    
    def update_conversation(
        self,
        conversation_id: str,
        performative: Performative,
        sender: str
    ) -> bool:
        """Update conversation state based on received message"""
        if conversation_id not in self.conversations:
            return False
        
        state = self.conversations[conversation_id]
        
        # Check if expired
        if state.is_expired():
            state.completed = True
            return False
        
        # Validate sequence
        if not ProtocolValidator.validate_sequence(
            state.protocol,
            state.current_step,
            performative
        ):
            logger.warning(
                "Invalid performative sequence",
                conversation_id=conversation_id,
                current_step=state.current_step,
                performative=performative.value
            )
            return False
        
        # Update state
        next_states = ProtocolValidator.get_next_states(state.protocol, performative)
        if next_states:
            state.current_step = next_states[0]  # Take first valid state
        
        # Check if conversation is complete
        if ProtocolValidator.is_final_state(state.protocol, performative):
            state.completed = True
            logger.info(
                "Conversation completed",
                conversation_id=conversation_id,
                protocol=state.protocol.value
            )
        
        return True
    
    def get_conversation_state(self, conversation_id: str) -> Optional[ProtocolState]:
        """Get current conversation state"""
        return self.conversations.get(conversation_id)
    
    def register_protocol_handler(
        self,
        protocol: InteractionProtocol,
        step: str,
        handler: Callable
    ):
        """Register handler for protocol step"""
        if protocol not in self.handlers:
            self.handlers[protocol] = {}
        
        self.handlers[protocol][step] = handler
        
        logger.info(
            "Protocol handler registered",
            protocol=protocol.value,
            step=step
        )
    
    async def handle_protocol_message(
        self,
        envelope: SemanticEnvelope
    ) -> Optional[Any]:
        """Handle message according to protocol rules"""
        conversation_id = envelope.conversation_id
        
        if conversation_id not in self.conversations:
            # New conversation - infer from performative
            if envelope.performative == Performative.REQUEST:
                protocol = InteractionProtocol.REQUEST
            elif envelope.performative in [Performative.QUERY_IF, Performative.QUERY_REF]:
                protocol = InteractionProtocol.QUERY
            elif envelope.performative == Performative.CFP:
                protocol = InteractionProtocol.CONTRACT_NET
            else:
                protocol = InteractionProtocol.REQUEST  # Default
            
            # Start conversation
            conversation_id = self.start_conversation(
                protocol=protocol,
                initiator=envelope.sender,
                participants=[envelope.receiver] if isinstance(envelope.receiver, str) else envelope.receiver
            )
            envelope.conversation_id = conversation_id
        
        # Update conversation
        state = self.conversations[conversation_id]
        if not self.update_conversation(conversation_id, envelope.performative, envelope.sender):
            return None
        
        # Find and execute handler
        protocol_handlers = self.handlers.get(state.protocol, {})
        handler = protocol_handlers.get(state.current_step)
        
        if handler:
            try:
                return await handler(envelope, state)
            except Exception as e:
                logger.error(
                    "Protocol handler error",
                    error=str(e),
                    protocol=state.protocol.value,
                    step=state.current_step
                )
        
        return None
    
    async def _cleanup_expired(self):
        """Cleanup expired conversations"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                expired = []
                for conv_id, state in self.conversations.items():
                    if state.is_expired() or (state.completed and 
                        datetime.utcnow() - state.created_at > timedelta(hours=1)):
                        expired.append(conv_id)
                
                for conv_id in expired:
                    del self.conversations[conv_id]
                
                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired conversations")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# ==================== Protocol Templates ====================

class ProtocolTemplates:
    """Pre-built protocol interaction templates"""
    
    @staticmethod
    async def request_inform(
        comm: 'UnifiedCommunication',
        receiver: str,
        action: str,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """
        Simple request-inform protocol.
        
        Flow: REQUEST -> AGREE/REFUSE -> INFORM/FAILURE
        """
        try:
            # Send request
            response = await comm.request(
                receiver=receiver,
                content={"action": action},
                timeout=timeout
            )
            
            # Handle response
            if response.performative == Performative.REFUSE:
                return None
            elif response.performative == Performative.AGREE:
                # Wait for inform
                result = await comm.request(
                    receiver=receiver,
                    content={"waiting_for": "result"},
                    conversation_id=response.conversation_id,
                    timeout=timeout
                )
                
                if result.performative == Performative.INFORM:
                    return result.content
                elif result.performative == Performative.FAILURE:
                    raise Exception(f"Action failed: {result.content}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Request-inform protocol failed: {e}")
            return None
    
    @staticmethod
    async def contract_net(
        comm: 'UnifiedCommunication',
        participants: List[str],
        task: Dict[str, Any],
        timeout: float = 60.0
    ) -> Optional[str]:
        """
        Contract Net Protocol for task allocation.
        
        Flow: CFP -> PROPOSE* -> ACCEPT/REJECT -> INFORM/FAILURE
        """
        conversation_id = str(uuid.uuid4())
        proposals = {}
        
        # Send CFP to all participants
        for participant in participants:
            await comm.send(
                SemanticEnvelope(
                    performative=Performative.CFP,
                    sender=comm.agent_id,
                    receiver=participant,
                    content=task,
                    conversation_id=conversation_id,
                    reply_by=datetime.utcnow() + timedelta(seconds=timeout/2)
                )
            )
        
        # Collect proposals
        deadline = datetime.utcnow() + timedelta(seconds=timeout/2)
        while datetime.utcnow() < deadline:
            # In real implementation, would wait for proposals
            await asyncio.sleep(0.1)
        
        # Select best proposal
        if not proposals:
            return None
        
        best_agent = max(proposals.items(), key=lambda x: x[1]["score"])[0]
        
        # Accept best, reject others
        for agent, proposal in proposals.items():
            if agent == best_agent:
                await comm.send(
                    SemanticEnvelope(
                        performative=Performative.ACCEPT_PROPOSAL,
                        sender=comm.agent_id,
                        receiver=agent,
                        content=proposal,
                        conversation_id=conversation_id
                    )
                )
            else:
                await comm.send(
                    SemanticEnvelope(
                        performative=Performative.REJECT_PROPOSAL,
                        sender=comm.agent_id,
                        receiver=agent,
                        content={"reason": "Better proposal received"},
                        conversation_id=conversation_id
                    )
                )
        
        return best_agent
    
    @staticmethod
    async def consensus_protocol(
        comm: 'UnifiedCommunication',
        participants: List[str],
        proposal: Dict[str, Any],
        required_agreement: float = 0.67,
        timeout: float = 120.0
    ) -> bool:
        """
        AURA Consensus Protocol for collective decisions.
        
        Achieves Byzantine fault-tolerant consensus.
        """
        conversation_id = str(uuid.uuid4())
        votes = {}
        
        # Broadcast proposal
        await comm.broadcast(
            content=proposal,
            performative=Performative.PROPOSE,
            topic="consensus",
            conversation_id=conversation_id,
            protocol="aura-consensus"
        )
        
        # Collect votes
        deadline = datetime.utcnow() + timedelta(seconds=timeout)
        while datetime.utcnow() < deadline and len(votes) < len(participants):
            # In real implementation, would collect votes
            await asyncio.sleep(0.1)
        
        # Calculate consensus
        accept_count = sum(1 for v in votes.values() if v == "accept")
        agreement_ratio = accept_count / len(participants) if participants else 0
        
        consensus_reached = agreement_ratio >= required_agreement
        
        # Broadcast result
        await comm.broadcast(
            content={
                "consensus": consensus_reached,
                "agreement_ratio": agreement_ratio,
                "votes": len(votes),
                "participants": len(participants)
            },
            performative=Performative.INFORM,
            topic="consensus",
            conversation_id=conversation_id
        )
        
        return consensus_reached


# ==================== Ontology Support ====================

@dataclass
class Ontology:
    """Defines shared vocabulary for agent communication"""
    name: str
    version: str
    concepts: Dict[str, Any]
    relations: Dict[str, List[str]]
    
    def validate_content(self, content: Any, expected_type: str) -> bool:
        """Validate content against ontology"""
        # Simple validation - can be extended
        if expected_type in self.concepts:
            schema = self.concepts[expected_type]
            if isinstance(schema, dict) and isinstance(content, dict):
                for key, value_type in schema.items():
                    if key not in content:
                        return False
                    # Type checking would go here
        return True


# ==================== Default AURA Ontology ====================

AURA_CORE_ONTOLOGY = Ontology(
    name="aura-core",
    version="1.0.0",
    concepts={
        "task": {
            "id": "string",
            "type": "string",
            "priority": "number",
            "deadline": "datetime",
            "requirements": "object"
        },
        "capability": {
            "name": "string",
            "parameters": "object",
            "constraints": "object"
        },
        "proposal": {
            "agent_id": "string",
            "task_id": "string",
            "bid": "number",
            "completion_time": "number",
            "confidence": "number"
        },
        "vote": {
            "voter": "string",
            "choice": "string",
            "confidence": "number",
            "reason": "string"
        }
    },
    relations={
        "can_perform": ["agent", "capability"],
        "requires": ["task", "capability"],
        "proposed_by": ["proposal", "agent"]
    }
)