"""
Gemini Client for AURA Intelligence System
Clean implementation with proper syntax and 2025 best practices
"""

import os
import httpx
import time
import asyncio
import logging
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GeminiConfig:
    """Configuration for Gemini API client."""
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    max_retries: int = 3
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

@dataclass 
class GeminiResponse:
    """Response from Gemini API."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class GeminiClient:
    """Async Gemini API client with retry logic and cost tracking."""
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        
        # Cost tracking (approximate pricing)
        self.cost_per_1k_input_tokens = 0.00015  # $0.15 per 1M tokens
        self.cost_per_1k_output_tokens = 0.0006   # $0.60 per 1M tokens
        
        logger.info(f"🤖 Gemini client initialized: {self.config.model}")
    
    async def ainvoke(self, messages: Union[str, List, Dict], **kwargs) -> GeminiResponse:
        """
        Async invoke method compatible with LangChain interface.
        
        Args:
            messages: Input messages (string, list, or dict)
            **kwargs: Additional parameters
            
        Returns:
            GeminiResponse: Response object with content
        """
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(messages)
            
            # Prepare request payload
            payload = {
                "contents": gemini_messages,
                "generationConfig": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "topP": kwargs.get("top_p", 0.8),
                    "topK": kwargs.get("top_k", 40)
                }
            }
            
            # Make API request with retries
            response_data = await self._make_request_with_retries(payload)
            
            # Extract response content
            content = self._extract_content(response_data)
            
            # Calculate usage and cost
            usage = self._calculate_usage(response_data)
            
            # Create response object
            response = GeminiResponse(
                content=content,
                model=self.config.model,
                usage=usage,
                metadata={
                    "duration": time.time() - start_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"✅ Gemini response received in {response.metadata['duration']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}")
            raise
    
    def _convert_messages(self, messages: Union[str, List, Dict]) -> List[Dict]:
        """Convert various message formats to Gemini format."""
        if isinstance(messages, str):
            # Simple string message
            return [{
                "parts": [{"text": messages}]
            }]
        
        elif isinstance(messages, list):
            # List of messages
            gemini_messages = []
            for msg in messages:
                if isinstance(msg, str):
                    gemini_messages.append({
                        "parts": [{"text": msg}]
                    })
                elif isinstance(msg, dict):
                    # Handle role-based messages
                    content = msg.get("content", "")
                    gemini_messages.append({
                        "parts": [{"text": content}]
                    })
            return gemini_messages
        
        elif isinstance(messages, dict):
            # Single dictionary message
            text = messages.get("content", str(messages))
            return [{
                "parts": [{"text": text}]
            }]
        
        else:
            # Fallback
            return [{
                "parts": [{"text": str(messages)}]
            }]
    
    async def _make_request_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with retry logic."""
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.config.api_key
        }
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    url=url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                
                # Handle specific error codes
                if response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Other errors
                response.raise_for_status()
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
                    continue
        
        # All retries failed
        raise last_exception or Exception("Request failed after all retries")
    
    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract text content from Gemini response."""
        try:
            candidates = response_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return ""
    
    def _calculate_usage(self, response_data: Dict[str, Any]) -> Dict[str, int]:
        """Calculate token usage from response."""
        # Note: Gemini API doesn't always return exact token counts
        # This is an approximation based on content length
        content = self._extract_content(response_data)
        
        # Rough approximation: 1 token ≈ 4 characters
        output_tokens = len(content) // 4
        input_tokens = 100  # Rough estimate
        
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    
    def calculate_cost(self, usage: Dict[str, int]) -> Dict[str, float]:
        """Calculate estimated cost based on usage."""
        input_cost = (usage["prompt_tokens"] / 1000) * self.cost_per_1k_input_tokens
        output_cost = (usage["completion_tokens"] / 1000) * self.cost_per_1k_output_tokens
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class ChatGemini:
    """LangChain-compatible Gemini chat model wrapper."""
    
    def __init__(self, 
                 model: str = "gemini-1.5-pro",
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 api_key: Optional[str] = None):
        config = GeminiConfig(
            api_key=api_key or os.getenv("GEMINI_API_KEY", ""),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.client = GeminiClient(config)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def ainvoke(self, messages: Any, **kwargs) -> GeminiResponse:
        """LangChain-compatible async invoke."""
        return await self.client.ainvoke(messages, **kwargs)
    
    def invoke(self, messages: Any, **kwargs) -> GeminiResponse:
        """LangChain-compatible sync invoke."""
        import asyncio
        return asyncio.run(self.ainvoke(messages, **kwargs))
    
    async def aclose(self):
        """Close the client."""
        await self.client.close()


def create_gemini_client(api_key: Optional[str] = None) -> ChatGemini:
    """Create a Gemini client instance."""
    return ChatGemini(api_key=api_key)


class SafeGeminiClient:
    """Gemini client with built-in error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        self.is_available = False
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        
    async def initialize(self) -> bool:
        """Initialize and validate the client."""
        try:
            self.client = create_gemini_client(self.api_key)
            
            # Test the connection
            response = await self.client.ainvoke("Say 'OK' if you can hear me.")
            
            self.is_available = "OK" in response.content.upper()
            
            if self.is_available:
                logger.info("✅ Gemini API connection validated successfully")
            else:
                logger.warning("⚠️ Gemini API connection failed validation")
                
            return self.is_available
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            self.is_available = False
            return False
    
    async def generate_content(self, prompt: str, **kwargs) -> Optional[GeminiResponse]:
        """Generate content if client is available."""
        if not self.is_available:
            logger.warning("Gemini client not available")
            return None
            
        try:
            return await self.client.ainvoke(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return None
    
    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.is_available = False


async def test_gemini_connection() -> bool:
    """Test Gemini API connection."""
    try:
        client = create_gemini_client()
        
        response = await client.ainvoke("Hello! Please respond with 'Connection successful'")
        
        await client.aclose()
        
        success = "successful" in response.content.lower()
        
        if success:
            logger.info("✅ Gemini API connection test successful")
        else:
            logger.warning(f"⚠️ Gemini API responded but unexpected content: {response.content}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Gemini API connection test failed: {e}")
        return False


# Export main classes and functions
__all__ = [
    "GeminiConfig",
    "GeminiResponse", 
    "GeminiClient",
    "ChatGemini",
    "SafeGeminiClient",
    "create_gemini_client",
    "test_gemini_connection"
]