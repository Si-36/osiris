"""
Minimal Correlation ID management - Fixed for syntax errors
This is a simplified version to unblock imports while we focus on the supervisor.
"""

import uuid
from contextvars import ContextVar
from typing import Optional, Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')

# Context variable for correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid.uuid4())

def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID"""
    return _correlation_id.get()

def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID"""
    _correlation_id.set(correlation_id)

def with_correlation_id(correlation_id: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add correlation ID context to functions.
    
    Args:
        correlation_id: Optional correlation ID to use. If None, generates new one.
        
    Returns:
        Decorated function with correlation ID context
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            cid = correlation_id or generate_correlation_id()
            token = _correlation_id.set(cid)
            try:
                return await func(*args, **kwargs)
            finally:
                _correlation_id.reset(token)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            cid = correlation_id or generate_correlation_id()
            token = _correlation_id.set(cid)
            try:
                return func(*args, **kwargs)
            finally:
                _correlation_id.reset(token)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator

class CorrelationContext:
    """Context manager for correlation IDs"""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or generate_correlation_id()
        self.token = None
    
    def __enter__(self):
        self.token = _correlation_id.set(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _correlation_id.reset(self.token)

def extract_correlation_id_from_headers(headers: dict) -> Optional[str]:
    """Extract correlation ID from HTTP headers"""
    return headers.get('X-Correlation-ID') or headers.get('x-correlation-id')

def inject_correlation_id_to_headers(headers: dict) -> dict:
    """Inject current correlation ID into HTTP headers"""
    cid = get_correlation_id()
    if cid:
        headers['X-Correlation-ID'] = cid
    return headers