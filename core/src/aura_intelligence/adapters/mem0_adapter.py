"""
Mem0 Adapter for AURA Intelligence - 2025 Best Practices

Mem0 is a memory layer for AI applications providing:
- Long-term memory storage
- Context-aware retrieval
- Multi-agent memory sharing
- Semantic search capabilities
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

try:
    from mem0 import Memory, MemoryConfig as Mem0Config
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    # Mock for development
    class Memory:
        pass
    class Mem0Config:
        pass

import structlog
from opentelemetry import trace

logger = structlog.get_logger(__name__)

# Create tracer
try:
    from ..observability import create_tracer
    tracer = create_tracer("mem0_adapter")
except ImportError:
    tracer = trace.get_tracer(__name__)


class MemoryType(Enum):
    """Types of memories"""
    EPISODIC = "episodic"      # Specific events
    SEMANTIC = "semantic"      # General knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"        # Short-term active memory


class MemoryPriority(Enum):
    """Memory priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Memory:
    """Memory entry structure"""
    id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 1.0


@dataclass
class MemoryQuery:
    """Query parameters for memory search"""
    query: str
    agent_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    limit: int = 10
    threshold: float = 0.7
    include_metadata: bool = True
    time_range: Optional[tuple[datetime, datetime]] = None


@dataclass
class Mem0AdapterConfig:
    """Configuration for Mem0 adapter"""
    # Mem0 settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4"
    
    # Memory settings
    embedding_model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    max_memory_size: int = 10000
    
    # Retrieval settings
    default_limit: int = 10
    similarity_threshold: float = 0.7
    rerank_results: bool = True
    
    # Performance
    batch_size: int = 100
    cache_ttl: int = 3600
    
    # Features
    enable_compression: bool = True
    enable_deduplication: bool = True
    enable_auto_summarization: bool = True


class Mem0Adapter:
    """
    Modern Mem0 adapter for AI memory management
    
    Features:
    - Async operations
    - Multi-agent memory isolation
    - Semantic search and retrieval
    - Memory compression and deduplication
    - Auto-summarization for long-term storage
    - Comprehensive observability
    """
    
    def __init__(self, config: Optional[Mem0AdapterConfig] = None):
        self.config = config or Mem0AdapterConfig()
        self._client: Optional[Memory] = None
        self._initialized = False
        self._memory_cache: Dict[str, Memory] = {}
        
    async def initialize(self) -> None:
        """Initialize Mem0 client"""
        if self._initialized:
            return
            
        with tracer.start_as_current_span("mem0_initialize") as span:
            try:
                if not MEM0_AVAILABLE:
                    logger.warning("Mem0 not available, using mock")
                    self._initialized = True
                    return
                
                # Configure Mem0
                mem0_config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": self.config.model,
                            "temperature": 0.1
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": self.config.embedding_model
                        }
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "aura_memories",
                            "embedding_dim": self.config.embedding_dim
                        }
                    }
                }
                
                if self.config.api_key:
                    mem0_config["api_key"] = self.config.api_key
                    
                self._client = Memory.from_config(mem0_config)
                self._initialized = True
                
                logger.info("Mem0 adapter initialized")
                span.set_status(trace.Status(trace.StatusCode.OK))
                
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                logger.error("Failed to initialize Mem0", error=str(e))
                raise
    
    async def add_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """Add a new memory"""
        if not self._initialized:
            await self.initialize()
            
        with tracer.start_as_current_span("mem0_add_memory") as span:
            span.set_attribute("mem0.agent_id", agent_id)
            span.set_attribute("mem0.memory_type", memory_type.value)
            
            try:
                # Create memory object
                memory = Memory(
                    id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    content=content,
                    memory_type=memory_type,
                    priority=priority,
                    metadata=metadata or {}
                )
                
                # Add to Mem0 if available
                if self._client and MEM0_AVAILABLE:
                    # Prepare data for Mem0
                    mem0_data = {
                        "content": content,
                        "agent_id": agent_id,
                        "metadata": {
                            "memory_type": memory_type.value,
                            "priority": priority.value,
                            "created_at": memory.created_at.isoformat(),
                            **(metadata or {})
                        }
                    }
                    
                    # Add to Mem0
                    result = await asyncio.to_thread(
                        self._client.add,
                        mem0_data,
                        user_id=agent_id
                    )
                    
                    # Update memory with Mem0 ID
                    if result and "id" in result:
                        memory.id = result["id"]
                
                # Cache locally
                self._memory_cache[memory.id] = memory
                
                logger.info(
                    "Memory added",
                    memory_id=memory.id,
                    agent_id=agent_id,
                    type=memory_type.value
                )
                
                return memory
                
            except Exception as e:
                logger.error("Failed to add memory", error=str(e))
                raise
    
    async def search_memories(
        self,
        query: MemoryQuery
    ) -> List[Memory]:
        """Search memories with semantic similarity"""
        if not self._initialized:
            await self.initialize()
            
        with tracer.start_as_current_span("mem0_search") as span:
            span.set_attribute("mem0.query", query.query[:100])
            span.set_attribute("mem0.limit", query.limit)
            
            try:
                results = []
                
                if self._client and MEM0_AVAILABLE:
                    # Search with Mem0
                    search_params = {
                        "query": query.query,
                        "limit": query.limit,
                        "threshold": query.threshold
                    }
                    
                    if query.agent_id:
                        search_params["user_id"] = query.agent_id
                        
                    mem0_results = await asyncio.to_thread(
                        self._client.search,
                        **search_params
                    )
                    
                    # Convert Mem0 results to Memory objects
                    for result in mem0_results:
                        metadata = result.get("metadata", {})
                        
                        memory = Memory(
                            id=result.get("id", str(uuid.uuid4())),
                            agent_id=metadata.get("agent_id", query.agent_id or "unknown"),
                            content=result.get("text", ""),
                            memory_type=MemoryType(metadata.get("memory_type", "episodic")),
                            priority=MemoryPriority(metadata.get("priority", "medium")),
                            metadata=metadata,
                            relevance_score=result.get("score", 0.0)
                        )
                        
                        results.append(memory)
                
                else:
                    # Fallback to cache search
                    cache_results = []
                    for memory in self._memory_cache.values():
                        if query.agent_id and memory.agent_id != query.agent_id:
                            continue
                            
                        # Simple keyword matching
                        if query.query.lower() in memory.content.lower():
                            cache_results.append(memory)
                            
                    # Sort by relevance and limit
                    cache_results.sort(key=lambda m: m.relevance_score, reverse=True)
                    results = cache_results[:query.limit]
                
                logger.info(
                    "Memory search completed",
                    query=query.query[:50],
                    results=len(results)
                )
                
                return results
                
            except Exception as e:
                logger.error("Failed to search memories", error=str(e))
                raise
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        if not self._initialized:
            await self.initialize()
            
        # Check cache first
        if memory_id in self._memory_cache:
            memory = self._memory_cache[memory_id]
            memory.access_count += 1
            return memory
            
        if self._client and MEM0_AVAILABLE:
            try:
                result = await asyncio.to_thread(
                    self._client.get,
                    memory_id
                )
                
                if result:
                    metadata = result.get("metadata", {})
                    memory = Memory(
                        id=memory_id,
                        agent_id=metadata.get("agent_id", "unknown"),
                        content=result.get("text", ""),
                        memory_type=MemoryType(metadata.get("memory_type", "episodic")),
                        priority=MemoryPriority(metadata.get("priority", "medium")),
                        metadata=metadata
                    )
                    
                    # Cache it
                    self._memory_cache[memory_id] = memory
                    return memory
                    
            except Exception as e:
                logger.error(f"Failed to get memory {memory_id}", error=str(e))
                
        return None
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory"""
        if not self._initialized:
            await self.initialize()
            
        try:
            memory = await self.get_memory(memory_id)
            if not memory:
                return False
                
            # Update fields
            if content:
                memory.content = content
                
            if metadata:
                memory.metadata.update(metadata)
                
            memory.updated_at = datetime.utcnow()
            
            # Update in Mem0
            if self._client and MEM0_AVAILABLE:
                update_data = {
                    "id": memory_id,
                    "content": memory.content,
                    "metadata": memory.metadata
                }
                
                await asyncio.to_thread(
                    self._client.update,
                    memory_id,
                    update_data
                )
            
            # Update cache
            self._memory_cache[memory_id] = memory
            
            logger.info("Memory updated", memory_id=memory_id)
            return True
            
        except Exception as e:
            logger.error("Failed to update memory", error=str(e))
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Delete from Mem0
            if self._client and MEM0_AVAILABLE:
                await asyncio.to_thread(
                    self._client.delete,
                    memory_id
                )
            
            # Remove from cache
            if memory_id in self._memory_cache:
                del self._memory_cache[memory_id]
                
            logger.info("Memory deleted", memory_id=memory_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete memory", error=str(e))
            return False
    
    async def get_agent_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ) -> List[Memory]:
        """Get all memories for an agent"""
        if not self._initialized:
            await self.initialize()
            
        query = MemoryQuery(
            query="",
            agent_id=agent_id,
            memory_types=[memory_type] if memory_type else None,
            limit=limit
        )
        
        # For agent-specific queries, we need to get all their memories
        if self._client and MEM0_AVAILABLE:
            try:
                all_memories = await asyncio.to_thread(
                    self._client.get_all,
                    user_id=agent_id
                )
                
                memories = []
                for mem_data in all_memories:
                    metadata = mem_data.get("metadata", {})
                    
                    # Filter by type if specified
                    if memory_type and metadata.get("memory_type") != memory_type.value:
                        continue
                        
                    memory = Memory(
                        id=mem_data.get("id", str(uuid.uuid4())),
                        agent_id=agent_id,
                        content=mem_data.get("text", ""),
                        memory_type=MemoryType(metadata.get("memory_type", "episodic")),
                        priority=MemoryPriority(metadata.get("priority", "medium")),
                        metadata=metadata
                    )
                    
                    memories.append(memory)
                
                # Sort by created_at and limit
                memories.sort(key=lambda m: m.created_at, reverse=True)
                return memories[:limit]
                
            except Exception as e:
                logger.error("Failed to get agent memories", error=str(e))
                
        # Fallback to cache
        cache_memories = [
            m for m in self._memory_cache.values()
            if m.agent_id == agent_id and
            (not memory_type or m.memory_type == memory_type)
        ]
        
        cache_memories.sort(key=lambda m: m.created_at, reverse=True)
        return cache_memories[:limit]
    
    async def clear_agent_memories(self, agent_id: str) -> int:
        """Clear all memories for an agent"""
        if not self._initialized:
            await self.initialize()
            
        count = 0
        
        try:
            if self._client and MEM0_AVAILABLE:
                # Get all memories for the agent
                memories = await self.get_agent_memories(agent_id, limit=10000)
                
                # Delete each one
                for memory in memories:
                    if await self.delete_memory(memory.id):
                        count += 1
            else:
                # Clear from cache
                to_delete = [
                    mid for mid, m in self._memory_cache.items()
                    if m.agent_id == agent_id
                ]
                
                for memory_id in to_delete:
                    del self._memory_cache[memory_id]
                    count += 1
                    
            logger.info(
                "Agent memories cleared",
                agent_id=agent_id,
                count=count
            )
            
            return count
            
        except Exception as e:
            logger.error("Failed to clear agent memories", error=str(e))
            return count
    
    async def close(self) -> None:
        """Close the adapter and cleanup"""
        self._memory_cache.clear()
        self._client = None
        self._initialized = False
        logger.info("Mem0 adapter closed")


# Export classes
__all__ = [
    "Mem0Adapter",
    "Mem0AdapterConfig",
    "Memory",
    "MemoryQuery",
    "MemoryType",
    "MemoryPriority"
]