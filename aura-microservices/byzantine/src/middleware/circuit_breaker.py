"""Circuit breaker middleware for fault tolerance"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import time
from collections import deque
from threading import Lock


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, failure_threshold: int = 5, timeout: float = 60.0):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = deque(maxlen=failure_threshold)
        self.last_failure_time = None
        self.lock = Lock()
        self.is_open = False
        
    async def dispatch(self, request: Request, call_next):
        # Check if circuit is open
        with self.lock:
            if self.is_open:
                if time.time() - self.last_failure_time > self.timeout:
                    # Try to close circuit
                    self.is_open = False
                    self.failures.clear()
                else:
                    return JSONResponse(
                        status_code=503,
                        content={"error": "Service temporarily unavailable"}
                    )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Reset on success
            if response.status_code < 500:
                with self.lock:
                    self.failures.clear()
                    
            return response
            
        except Exception as e:
            # Record failure
            with self.lock:
                self.failures.append(time.time())
                self.last_failure_time = time.time()
                
                if len(self.failures) >= self.failure_threshold:
                    self.is_open = True
                    
            raise