"""
Context Manager - Smart Context Handling Across Different Model Providers
Transforms context_integration.py into production context management

Key Features:
- Provider-specific context preparation
- Smart chunking for token limits
- Long-context optimization for Claude/Mamba
- Context compression and summarization
- Integration with AURA memory system
"""

import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import structlog

from .provider_adapters import ProviderType, ModelConfig
from ..memory import MemoryManager
from ..observability import create_tracer, create_meter

logger = structlog.get_logger(__name__)
tracer = create_tracer("context_manager")
meter = create_meter("context_manager")

# Metrics
context_operations = meter.create_counter(
    name="aura.context.operations",
    description="Context operations by type"
)

context_size = meter.create_histogram(
    name="aura.context.size_tokens",
    description="Context size in tokens",
    unit="tokens"
)

compression_ratio = meter.create_histogram(
    name="aura.context.compression_ratio",
    description="Context compression ratio"
)


@dataclass
class ContextWindow:
    """Managed context window for a request"""
    original_prompt: str
    system_prompt: Optional[str]
    memory_context: List[Dict[str, Any]]
    tool_definitions: Optional[List[Dict[str, Any]]]
    
    # Processed versions
    prepared_prompt: str
    total_tokens: int
    compression_applied: bool = False
    chunks: List[str] = field(default_factory=list)
    
    # Metadata
    provider: Optional[ProviderType] = None
    model: Optional[str] = None
    strategy: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextStrategy:
    """Base class for context preparation strategies"""
    
    def prepare(self, window: ContextWindow, config: ModelConfig) -> ContextWindow:
        """Prepare context for specific model configuration"""
        raise NotImplementedError


class StandardContextStrategy(ContextStrategy):
    """Standard context preparation for most models"""
    
    def prepare(self, window: ContextWindow, config: ModelConfig) -> ContextWindow:
        """Standard preparation with basic token management"""
        # Calculate available token budget
        max_tokens = config.max_context
        system_tokens = self._estimate_tokens(window.system_prompt or "")
        tool_tokens = self._estimate_tool_tokens(window.tool_definitions or [])
        
        available_for_prompt = max_tokens - system_tokens - tool_tokens - 1000  # Reserve for output
        
        # Prepare memory context
        memory_text = self._format_memory_context(window.memory_context)
        memory_tokens = self._estimate_tokens(memory_text)
        
        # Prepare final prompt
        if memory_tokens + self._estimate_tokens(window.original_prompt) > available_for_prompt:
            # Need compression
            compressed_memory = self._compress_memory(window.memory_context, available_for_prompt // 2)
            memory_text = self._format_memory_context(compressed_memory)
            window.compression_applied = True
            
        # Combine everything
        if memory_text:
            window.prepared_prompt = f"{memory_text}\n\n{window.original_prompt}"
        else:
            window.prepared_prompt = window.original_prompt
            
        window.total_tokens = (
            system_tokens + 
            tool_tokens + 
            self._estimate_tokens(window.prepared_prompt)
        )
        
        window.strategy = "standard"
        return window
        
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4
        
    def _estimate_tool_tokens(self, tools: List[Dict[str, Any]]) -> int:
        """Estimate tokens for tool definitions"""
        if not tools:
            return 0
        tool_str = json.dumps(tools)
        return self._estimate_tokens(tool_str)
        
    def _format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format memory entries into context"""
        if not memories:
            return ""
            
        formatted = "Relevant context from memory:\n"
        for memory in memories:
            formatted += f"- {memory.get('content', '')}\n"
            
        return formatted
        
    def _compress_memory(self, memories: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Compress memory to fit token budget"""
        # Simple strategy: take most relevant memories
        sorted_memories = sorted(memories, key=lambda m: m.get("relevance", 0), reverse=True)
        
        compressed = []
        current_tokens = 0
        
        for memory in sorted_memories:
            memory_tokens = self._estimate_tokens(memory.get("content", ""))
            if current_tokens + memory_tokens > max_tokens:
                break
            compressed.append(memory)
            current_tokens += memory_tokens
            
        return compressed


class LongContextStrategy(ContextStrategy):
    """Strategy for long-context models (Claude, Mamba-2)"""
    
    def prepare(self, window: ContextWindow, config: ModelConfig) -> ContextWindow:
        """Prepare for long-context models with enhanced memory integration"""
        # Much larger token budget
        max_tokens = config.max_context
        reserve_tokens = 4000  # More reserve for long outputs
        
        # Include extensive memory context
        memory_text = self._format_extended_memory(window.memory_context)
        
        # Include conversation history if available
        history_text = ""
        if "conversation_history" in window.metadata:
            history_text = self._format_conversation_history(
                window.metadata["conversation_history"]
            )
            
        # Build comprehensive prompt
        parts = []
        if memory_text:
            parts.append("## Relevant Context\n" + memory_text)
        if history_text:
            parts.append("## Conversation History\n" + history_text)
        parts.append("## Current Request\n" + window.original_prompt)
        
        window.prepared_prompt = "\n\n".join(parts)
        window.total_tokens = self._estimate_tokens(window.prepared_prompt)
        
        # Check if we're still within limits
        if window.total_tokens > max_tokens - reserve_tokens:
            # Even long-context models have limits
            window = self._apply_sliding_window(window, max_tokens - reserve_tokens)
            window.compression_applied = True
            
        window.strategy = "long_context"
        return window
        
    def _format_extended_memory(self, memories: List[Dict[str, Any]]) -> str:
        """Format extended memory with more detail"""
        if not memories:
            return ""
            
        formatted = ""
        
        # Group by type
        by_type = {}
        for memory in memories:
            mem_type = memory.get("type", "general")
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory)
            
        # Format each type
        for mem_type, items in by_type.items():
            formatted += f"### {mem_type.title()} Context\n"
            for item in items:
                formatted += f"- {item.get('content', '')}"
                if "metadata" in item:
                    formatted += f" (relevance: {item['metadata'].get('relevance', 0):.2f})"
                formatted += "\n"
            formatted += "\n"
            
        return formatted
        
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history"""
        formatted = ""
        for turn in history[-10:]:  # Last 10 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted += f"{role.title()}: {content}\n"
        return formatted
        
    def _apply_sliding_window(self, window: ContextWindow, max_tokens: int) -> ContextWindow:
        """Apply sliding window compression"""
        # Keep most recent and most relevant parts
        # This is a simplified version - production would be more sophisticated
        text = window.prepared_prompt
        estimated_tokens = self._estimate_tokens(text)
        
        if estimated_tokens > max_tokens:
            # Truncate from the middle, keeping beginning and end
            chars_to_keep = int(max_tokens * 4 * 0.9)  # Conservative estimate
            if len(text) > chars_to_keep:
                keep_start = chars_to_keep // 2
                keep_end = chars_to_keep // 2
                text = text[:keep_start] + "\n...[context trimmed]...\n" + text[-keep_end:]
                
        window.prepared_prompt = text
        window.total_tokens = self._estimate_tokens(text)
        return window
        
    def _estimate_tokens(self, text: str) -> int:
        """Token estimation for long-context models"""
        # More accurate for larger texts
        return int(len(text) / 3.5)


class ChunkedContextStrategy(ContextStrategy):
    """Strategy for chunking large contexts into multiple calls"""
    
    def prepare(self, window: ContextWindow, config: ModelConfig) -> ContextWindow:
        """Prepare context with chunking for models with smaller windows"""
        max_tokens = config.max_context
        chunk_size = max_tokens - 2000  # Reserve for system prompt and output
        
        # Check if chunking is needed
        full_text = window.original_prompt
        if window.memory_context:
            memory_text = self._format_memory_context(window.memory_context)
            full_text = f"{memory_text}\n\n{full_text}"
            
        total_tokens = self._estimate_tokens(full_text)
        
        if total_tokens <= chunk_size:
            # No chunking needed
            window.prepared_prompt = full_text
            window.total_tokens = total_tokens
            window.strategy = "standard"
            return window
            
        # Apply chunking
        chunks = self._create_chunks(full_text, chunk_size)
        window.chunks = chunks
        window.prepared_prompt = chunks[0]  # First chunk
        window.total_tokens = self._estimate_tokens(chunks[0])
        window.strategy = "chunked"
        window.metadata["total_chunks"] = len(chunks)
        window.metadata["current_chunk"] = 0
        
        return window
        
    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Create overlapping chunks"""
        # Estimate characters per chunk
        chars_per_chunk = chunk_size * 4
        overlap = chars_per_chunk // 10  # 10% overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chars_per_chunk
            
            # Try to break at sentence boundary
            if end < len(text):
                period_idx = text.rfind('.', start + chars_per_chunk - overlap, end)
                if period_idx > start:
                    end = period_idx + 1
                    
            chunk = text[start:end]
            
            # Add context markers
            if start > 0:
                chunk = "...continuing from previous context...\n\n" + chunk
            if end < len(text):
                chunk = chunk + "\n\n...context continues..."
                
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
        
    def _format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format memory for chunked context"""
        if not memories:
            return ""
            
        # More concise format for chunked contexts
        formatted = "Key context:\n"
        for memory in memories[:5]:  # Limit to top 5
            formatted += f"• {memory.get('content', '')}\n"
            
        return formatted
        
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4


class ContextManager:
    """Main context manager that orchestrates different strategies"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager
        
        # Initialize strategies
        self.strategies = {
            "standard": StandardContextStrategy(),
            "long_context": LongContextStrategy(),
            "chunked": ChunkedContextStrategy()
        }
        
        # Cache for prepared contexts
        self.context_cache: Dict[str, ContextWindow] = {}
        
    async def prepare_context(self, 
                            prompt: str,
                            provider: ProviderType,
                            model: str,
                            model_config: ModelConfig,
                            system_prompt: Optional[str] = None,
                            tools: Optional[List[Dict[str, Any]]] = None,
                            session_id: Optional[str] = None) -> ContextWindow:
        """Prepare context for specific provider and model"""
        
        with tracer.start_as_current_span("prepare_context") as span:
            span.set_attribute("provider", provider.value)
            span.set_attribute("model", model)
            
            # Check cache
            cache_key = self._compute_cache_key(prompt, provider, model, session_id)
            if cache_key in self.context_cache:
                context_operations.add(1, {"operation": "cache_hit"})
                return self.context_cache[cache_key]
                
            # Retrieve relevant memories
            memory_context = []
            if self.memory_manager and session_id:
                try:
                    memory_context = await self.memory_manager.retrieve_relevant(
                        prompt, 
                        session_id=session_id,
                        limit=10
                    )
                except Exception as e:
                    logger.warning(f"Failed to retrieve memories: {e}")
                    
            # Create context window
            window = ContextWindow(
                original_prompt=prompt,
                system_prompt=system_prompt,
                memory_context=memory_context,
                tool_definitions=tools,
                prepared_prompt="",
                total_tokens=0,
                provider=provider,
                model=model
            )
            
            # Select strategy based on provider and model
            strategy = self._select_strategy(provider, model, model_config)
            
            # Prepare context
            window = strategy.prepare(window, model_config)
            
            # Cache prepared context
            self.context_cache[cache_key] = window
            
            # Record metrics
            context_operations.add(1, {"operation": "prepare", "strategy": window.strategy})
            context_size.record(window.total_tokens, {"provider": provider.value})
            
            if window.compression_applied:
                original_size = self._estimate_tokens(window.original_prompt)
                ratio = window.total_tokens / original_size if original_size > 0 else 1.0
                compression_ratio.record(ratio)
                
            span.set_attribute("total_tokens", window.total_tokens)
            span.set_attribute("strategy", window.strategy)
            span.set_attribute("compression_applied", window.compression_applied)
            
            return window
            
    def _select_strategy(self, provider: ProviderType, model: str, config: ModelConfig) -> ContextStrategy:
        """Select appropriate context strategy"""
        
        # Long context models
        if provider == ProviderType.ANTHROPIC and "opus" in model:
            return self.strategies["long_context"]
        elif provider == ProviderType.TOGETHER and "mamba" in model:
            return self.strategies["long_context"]
            
        # Models that might need chunking
        elif config.max_context < 32000:
            return self.strategies["chunked"]
            
        # Default strategy
        return self.strategies["standard"]
        
    def _compute_cache_key(self, prompt: str, provider: ProviderType, 
                          model: str, session_id: Optional[str]) -> str:
        """Compute cache key for context"""
        key_parts = [prompt[:100], provider.value, model]  # First 100 chars of prompt
        if session_id:
            key_parts.append(session_id)
            
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4
        
    async def get_next_chunk(self, window: ContextWindow) -> Optional[str]:
        """Get next chunk for chunked processing"""
        if window.strategy != "chunked" or not window.chunks:
            return None
            
        current_chunk = window.metadata.get("current_chunk", 0)
        if current_chunk + 1 >= len(window.chunks):
            return None
            
        window.metadata["current_chunk"] = current_chunk + 1
        window.prepared_prompt = window.chunks[current_chunk + 1]
        window.total_tokens = self._estimate_tokens(window.prepared_prompt)
        
        context_operations.add(1, {"operation": "next_chunk"})
        
        return window.prepared_prompt
        
    def merge_chunked_responses(self, responses: List[str], strategy: str = "concatenate") -> str:
        """Merge responses from chunked processing"""
        if strategy == "concatenate":
            # Simple concatenation with cleanup
            merged = "\n\n".join(responses)
            
            # Remove continuation markers
            merged = merged.replace("...continuing from previous context...", "")
            merged = merged.replace("...context continues...", "")
            
            return merged.strip()
            
        elif strategy == "smart_merge":
            # More sophisticated merging (future enhancement)
            # Could use overlap to merge intelligently
            return self.merge_chunked_responses(responses, "concatenate")
            
        else:
            return self.merge_chunked_responses(responses, "concatenate")


# Export main classes
__all__ = [
    "ContextWindow",
    "ContextStrategy",
    "StandardContextStrategy",
    "LongContextStrategy", 
    "ChunkedContextStrategy",
    "ContextManager"
]