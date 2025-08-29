"""
Simple Provider Adapters without observability dependencies
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import aiohttp
import json
import os

# Import the same classes but without observability
# Define simple versions without observability
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    OLLAMA = "ollama"

class ModelCapability(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    
@dataclass
class ModelConfig:
    provider: ProviderType
    model_id: str
    capabilities: List[ModelCapability]
    context_length: int = 4096
    
@dataclass
class ProviderRequest:
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
@dataclass
class ProviderResponse:
    content: str
    model: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]
    
class ProviderAdapter:
    """Base adapter"""
    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            content="Mock response",
            model="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 10},
            metadata={}
        )

class OpenAIAdapter(ProviderAdapter):
    pass

class AnthropicAdapter(ProviderAdapter):
    pass
    
class TogetherAdapter(ProviderAdapter):
    pass
    
class OllamaAdapter(ProviderAdapter):
    pass

class ProviderFactory:
    @staticmethod
    def create(provider_type: ProviderType, **kwargs):
        if provider_type == ProviderType.OPENAI:
            return OpenAIAdapter()
        elif provider_type == ProviderType.ANTHROPIC:
            return AnthropicAdapter()
        elif provider_type == ProviderType.TOGETHER:
            return TogetherAdapter()
        elif provider_type == ProviderType.OLLAMA:
            return OllamaAdapter()

# Re-export everything
__all__ = [
    'ProviderType',
    'ModelCapability',
    'ModelConfig',
    'ProviderRequest',
    'ProviderResponse',
    'ProviderAdapter',
    'OpenAIAdapter',
    'AnthropicAdapter',
    'TogetherAdapter',
    'OllamaAdapter',
    'ProviderFactory',
]