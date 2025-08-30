"""
AURA Intelligence utilities module.

Common utilities, decorators, and helper functions.
"""

from .decorators import (
    circuit_breaker,
    retry,
    rate_limit,
    timed,
    cached,
    with_retry,
    with_circuit_breaker,
    CircuitBreaker,
    RateLimiter,
)

# Import logger separately since it's used everywhere
try:
    from .logger import get_logger, setup_logging
except ImportError:
    # Fallback if logger module doesn't exist
    import logging
    get_logger = logging.getLogger
    setup_logging = lambda: None

# Import validation if available
try:
    from .validation import validate_config, validate_environment
except ImportError:
    validate_config = lambda x: x
    validate_environment = lambda: None

__all__ = [
    # Decorators
    "circuit_breaker",
    "retry",
    "rate_limit",
    "timed",
    "cached",
    "with_retry",
    "with_circuit_breaker",
    "CircuitBreaker",
    "RateLimiter",
    # Logging
    "setup_logging",
    "get_logger",
    # Validation
    "validate_config",
    "validate_environment",
]
