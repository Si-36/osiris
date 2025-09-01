"""
📡 Osiris Assistants API - Standard Interface for Agent Platforms
==============================================================

Exposes the Unified Intelligence via the Assistants pattern:
- Create versioned assistants
- Manage threads and runs
- Stream responses  
- Full governance and audit trail

Compatible with OpenAI Assistants API for easy integration.
"""

import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import structlog

from ..unified.osiris_unified_intelligence import (
    OsirisUnifiedIntelligence,
    create_osiris_unified_intelligence
)

logger = structlog.get_logger(__name__)


class RunStatus(Enum):
    """Standard run statuses"""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class MessageRole(Enum):
    """Message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Assistant:
    """Assistant configuration"""
    id: str = field(default_factory=lambda: f"asst_{uuid.uuid4().hex}")
    name: str = "Osiris Intelligence Assistant"
    description: str = "Advanced AI assistant powered by LNN+MoE+CoRaL+DPO"
    model: str = "osiris-unified-v1"
    instructions: str = "You are a helpful AI assistant with advanced reasoning capabilities."
    tools: List[Dict[str, Any]] = field(default_factory=list)
    file_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))
    
    # Osiris-specific settings
    complexity_threshold: float = 0.5
    max_experts: int = 64
    enable_constitutional: bool = True
    max_context: int = 100_000


@dataclass
class Thread:
    """Conversation thread"""
    id: str = field(default_factory=lambda: f"thread_{uuid.uuid4().hex}")
    created_at: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List['Message'] = field(default_factory=list)


@dataclass
class Message:
    """Thread message"""
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    thread_id: str = ""
    role: MessageRole = MessageRole.USER
    content: str = ""
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    file_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Run:
    """Assistant run"""
    id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    thread_id: str = ""
    assistant_id: str = ""
    status: RunStatus = RunStatus.QUEUED
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Osiris metrics
    intelligence_metrics: Optional[Dict[str, Any]] = None


class OsirisAssistantsAPI:
    """
    Production Assistants API for Osiris.
    
    This provides the standard interface that agent platforms expect:
    - Assistant management
    - Thread/conversation management
    - Run execution with streaming
    - Full audit trail
    """
    
    def __init__(self, osiris: Optional[OsirisUnifiedIntelligence] = None):
        # Use provided Osiris or create default
        self.osiris = osiris or create_osiris_unified_intelligence()
        
        # Storage (in production use persistent storage)
        self.assistants: Dict[str, Assistant] = {}
        self.threads: Dict[str, Thread] = {}
        self.runs: Dict[str, Run] = {}
        
        # Default assistant
        self._create_default_assistant()
        
        logger.info("Osiris Assistants API initialized")
        
    def _create_default_assistant(self):
        """Create the default Osiris assistant"""
        default = Assistant(
            id="asst_default",
            name="Osiris",
            description="The primary Osiris intelligence with full capabilities",
            instructions=(
                "You are Osiris, an advanced AI assistant powered by:\n"
                "- Liquid Neural Networks for adaptive intelligence\n"
                "- Switch Transformer MoE for efficient expert routing\n"
                "- Mamba-2 CoRaL for unlimited context reasoning\n"
                "- Constitutional DPO for safe, aligned responses\n"
                "\nProvide helpful, accurate, and safe responses."
            ),
            metadata={
                "version": "1.0.0",
                "capabilities": ["reasoning", "analysis", "creativity", "technical"]
            }
        )
        self.assistants[default.id] = default
        
    # === Assistant Management ===
    
    async def create_assistant(self,
                              name: str,
                              instructions: str,
                              model: str = "osiris-unified-v1",
                              tools: Optional[List[Dict[str, Any]]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> Assistant:
        """Create a new assistant configuration"""
        assistant = Assistant(
            name=name,
            instructions=instructions,
            model=model,
            tools=tools or [],
            metadata=metadata or {}
        )
        
        self.assistants[assistant.id] = assistant
        logger.info(f"Created assistant: {assistant.id}")
        
        return assistant
        
    async def get_assistant(self, assistant_id: str) -> Optional[Assistant]:
        """Get assistant by ID"""
        return self.assistants.get(assistant_id)
        
    async def update_assistant(self,
                              assistant_id: str,
                              **updates) -> Optional[Assistant]:
        """Update assistant configuration"""
        assistant = self.assistants.get(assistant_id)
        if not assistant:
            return None
            
        # Update allowed fields
        for key, value in updates.items():
            if hasattr(assistant, key) and key not in ['id', 'created_at']:
                setattr(assistant, key, value)
                
        logger.info(f"Updated assistant: {assistant_id}")
        return assistant
        
    async def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant"""
        if assistant_id in self.assistants:
            del self.assistants[assistant_id]
            logger.info(f"Deleted assistant: {assistant_id}")
            return True
        return False
        
    async def list_assistants(self, limit: int = 20) -> List[Assistant]:
        """List all assistants"""
        return list(self.assistants.values())[:limit]
        
    # === Thread Management ===
    
    async def create_thread(self,
                           messages: Optional[List[Dict[str, Any]]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Thread:
        """Create a new conversation thread"""
        thread = Thread(metadata=metadata or {})
        
        # Add initial messages if provided
        if messages:
            for msg_data in messages:
                message = Message(
                    thread_id=thread.id,
                    role=MessageRole(msg_data.get('role', 'user')),
                    content=msg_data.get('content', ''),
                    metadata=msg_data.get('metadata', {})
                )
                thread.messages.append(message)
                
        self.threads[thread.id] = thread
        logger.info(f"Created thread: {thread.id}")
        
        return thread
        
    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        return self.threads.get(thread_id)
        
    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        if thread_id in self.threads:
            del self.threads[thread_id]
            logger.info(f"Deleted thread: {thread_id}")
            return True
        return False
        
    # === Message Management ===
    
    async def create_message(self,
                            thread_id: str,
                            role: str,
                            content: str,
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[Message]:
        """Add a message to a thread"""
        thread = self.threads.get(thread_id)
        if not thread:
            return None
            
        message = Message(
            thread_id=thread_id,
            role=MessageRole(role),
            content=content,
            metadata=metadata or {}
        )
        
        thread.messages.append(message)
        logger.info(f"Added message to thread {thread_id}")
        
        return message
        
    async def list_messages(self, thread_id: str, limit: int = 20) -> List[Message]:
        """List messages in a thread"""
        thread = self.threads.get(thread_id)
        if not thread:
            return []
            
        return thread.messages[-limit:]
        
    # === Run Management ===
    
    async def create_run(self,
                        thread_id: str,
                        assistant_id: str,
                        stream: bool = False,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[Run]:
        """Create and execute a run"""
        
        # Validate
        thread = self.threads.get(thread_id)
        assistant = self.assistants.get(assistant_id)
        
        if not thread or not assistant:
            return None
            
        # Create run
        run = Run(
            thread_id=thread_id,
            assistant_id=assistant_id,
            status=RunStatus.IN_PROGRESS,
            started_at=int(time.time()),
            metadata=metadata or {}
        )
        
        self.runs[run.id] = run
        
        # Execute run
        if stream:
            # Return run immediately, stream will be handled separately
            return run
        else:
            # Execute synchronously
            await self._execute_run(run)
            return run
            
    async def get_run(self, thread_id: str, run_id: str) -> Optional[Run]:
        """Get run by ID"""
        return self.runs.get(run_id)
        
    async def cancel_run(self, thread_id: str, run_id: str) -> Optional[Run]:
        """Cancel a run"""
        run = self.runs.get(run_id)
        if run and run.status == RunStatus.IN_PROGRESS:
            run.status = RunStatus.CANCELLED
            run.completed_at = int(time.time())
            logger.info(f"Cancelled run: {run_id}")
            
        return run
        
    # === Streaming ===
    
    async def stream_run(self, 
                        thread_id: str,
                        run_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream run events"""
        run = self.runs.get(run_id)
        if not run:
            yield {"event": "error", "data": {"message": "Run not found"}}
            return
            
        thread = self.threads.get(thread_id)
        assistant = self.assistants.get(run.assistant_id)
        
        if not thread or not assistant:
            yield {"event": "error", "data": {"message": "Invalid thread or assistant"}}
            return
            
        # Get latest user message
        user_messages = [m for m in thread.messages if m.role == MessageRole.USER]
        if not user_messages:
            yield {"event": "error", "data": {"message": "No user message found"}}
            return
            
        latest_message = user_messages[-1]
        
        # Stream events
        yield {"event": "thread.run.created", "data": {"run_id": run.id}}
        yield {"event": "thread.run.in_progress", "data": {"run_id": run.id}}
        
        # Process with Osiris
        try:
            # Configure Osiris based on assistant settings
            self._configure_osiris_for_assistant(assistant)
            
            # Process with streaming
            response_chunks = []
            async for chunk in self.osiris.process(
                prompt=latest_message.content,
                context={
                    'thread_id': thread_id,
                    'assistant_id': assistant.id,
                    'instructions': assistant.instructions
                },
                stream=True
            ):
                # Stream delta
                yield {
                    "event": "thread.message.delta",
                    "data": {
                        "delta": {"content": chunk.get('chunk', '')},
                        "run_id": run.id
                    }
                }
                
                if not chunk.get('done', False):
                    response_chunks.append(chunk.get('chunk', ''))
                else:
                    # Final metrics
                    run.intelligence_metrics = chunk.get('metrics', {})
                    
            # Create assistant message
            assistant_message = Message(
                thread_id=thread_id,
                role=MessageRole.ASSISTANT,
                content=''.join(response_chunks),
                assistant_id=assistant.id,
                run_id=run.id
            )
            thread.messages.append(assistant_message)
            
            # Complete run
            run.status = RunStatus.COMPLETED
            run.completed_at = int(time.time())
            
            yield {
                "event": "thread.message.completed",
                "data": {"message_id": assistant_message.id}
            }
            
            yield {
                "event": "thread.run.completed",
                "data": {
                    "run_id": run.id,
                    "intelligence_metrics": run.intelligence_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Run failed: {e}")
            run.status = RunStatus.FAILED
            run.failed_at = int(time.time())
            run.error = {"message": str(e)}
            
            yield {
                "event": "thread.run.failed",
                "data": {"run_id": run.id, "error": run.error}
            }
            
    async def _execute_run(self, run: Run):
        """Execute a run synchronously"""
        # Similar to stream_run but without yielding events
        thread = self.threads.get(run.thread_id)
        assistant = self.assistants.get(run.assistant_id)
        
        if not thread or not assistant:
            run.status = RunStatus.FAILED
            run.error = {"message": "Invalid thread or assistant"}
            return
            
        # Get latest user message
        user_messages = [m for m in thread.messages if m.role == MessageRole.USER]
        if not user_messages:
            run.status = RunStatus.FAILED
            run.error = {"message": "No user message found"}
            return
            
        latest_message = user_messages[-1]
        
        try:
            # Configure Osiris
            self._configure_osiris_for_assistant(assistant)
            
            # Process
            response = await self.osiris.process(
                prompt=latest_message.content,
                context={
                    'thread_id': run.thread_id,
                    'assistant_id': assistant.id,
                    'instructions': assistant.instructions
                }
            )
            
            # Create assistant message
            assistant_message = Message(
                thread_id=run.thread_id,
                role=MessageRole.ASSISTANT,
                content=response['content'],
                assistant_id=assistant.id,
                run_id=run.id
            )
            thread.messages.append(assistant_message)
            
            # Update run
            run.status = RunStatus.COMPLETED
            run.completed_at = int(time.time())
            run.intelligence_metrics = response.get('intelligence_metrics', {})
            
        except Exception as e:
            logger.error(f"Run failed: {e}")
            run.status = RunStatus.FAILED
            run.failed_at = int(time.time())
            run.error = {"message": str(e)}
            
    def _configure_osiris_for_assistant(self, assistant: Assistant):
        """Configure Osiris based on assistant settings"""
        # In production, this would dynamically configure the model
        # For now, we use the assistant metadata to guide behavior
        pass
        
    # === API Status ===
    
    async def get_api_status(self) -> Dict[str, Any]:
        """Get API health and metrics"""
        health = self.osiris.get_system_health()
        
        return {
            'api_version': '1.0.0',
            'model': 'osiris-unified-v1',
            'status': health['status'],
            'metrics': {
                'total_assistants': len(self.assistants),
                'total_threads': len(self.threads),
                'total_runs': len(self.runs),
                'active_runs': sum(1 for r in self.runs.values() if r.status == RunStatus.IN_PROGRESS)
            },
            'intelligence_health': health
        }


# === Example Usage ===

async def example_usage():
    """Example of using the Assistants API"""
    
    # Create API instance
    api = OsirisAssistantsAPI()
    
    # Create a custom assistant
    assistant = await api.create_assistant(
        name="Technical Expert",
        instructions="You are a technical expert. Focus on accurate, detailed explanations.",
        metadata={"specialty": "software_engineering"}
    )
    
    # Create a thread with initial message
    thread = await api.create_thread(
        messages=[
            {
                "role": "user",
                "content": "Explain how neural networks work"
            }
        ]
    )
    
    # Create and execute a run
    run = await api.create_run(
        thread_id=thread.id,
        assistant_id=assistant.id,
        stream=True
    )
    
    # Stream the response
    print(f"Streaming run {run.id}...")
    async for event in api.stream_run(thread.id, run.id):
        if event['event'] == 'thread.message.delta':
            print(event['data']['delta']['content'], end='', flush=True)
        elif event['event'] == 'thread.run.completed':
            print(f"\n\nCompleted! Metrics: {event['data']['intelligence_metrics']}")
            
    # Get the final thread messages
    messages = await api.list_messages(thread.id)
    print(f"\nThread has {len(messages)} messages")
    
    # Check API status
    status = await api.get_api_status()
    print(f"\nAPI Status: {status}")


if __name__ == "__main__":
    asyncio.run(example_usage())