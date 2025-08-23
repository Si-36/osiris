"""Observability middleware for request tracing and metrics"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time
import uuid
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start span
        with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("correlation_id", correlation_id)
            
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.duration", duration)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Response-Time"] = f"{duration:.3f}"
            
            return response