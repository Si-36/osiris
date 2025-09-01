"""
Error handling utilities for AURA Intelligence
"""

import functools
import asyncio
from typing import TypeVar, Callable, Any, Optional, Union
import time
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def resilient_operation(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for resilient operations with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            pass
        last_exception = None
        current_delay = delay
            
        for attempt in range(max_retries + 1):
            pass
        try:
            pass
        return func(*args, **kwargs)
        except exceptions as e:
            pass
        last_exception = e
        if attempt < max_retries:
            pass
        logger.warning(
        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}",
        error=str(e),
        retry_in=current_delay
        )
        time.sleep(current_delay)
        current_delay *= backoff_factor
        else:
            pass
        logger.error(
        f"All {max_retries + 1} attempts failed for {func.__name__}",
        error=str(e)
        )
            
        raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            pass
        last_exception = None
        current_delay = delay
            
        for attempt in range(max_retries + 1):
            pass
        try:
            pass
        return await func(*args, **kwargs)
        except exceptions as e:
            pass
        last_exception = e
        if attempt < max_retries:
            pass
        logger.warning(
        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}",
        error=str(e),
        retry_in=current_delay
        )
        await asyncio.sleep(current_delay)
        current_delay *= backoff_factor
        else:
            pass
        logger.error(
        f"All {max_retries + 1} attempts failed for {func.__name__}",
        error=str(e)
        )
            
        raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            pass
        return async_wrapper
        else:
            pass
        return sync_wrapper
    
        return decorator


class AuraError(Exception):
    """Base exception for AURA Intelligence errors."""
    pass


class ConfigurationError(AuraError):
    """Raised when configuration is invalid."""
    pass


class ComponentError(AuraError):
    """Raised when a component fails."""
    pass


class IntegrationError(AuraError):
    """Raised when an external integration fails."""
    pass


class ValidationError(AuraError):
    """Raised when validation fails."""
    pass