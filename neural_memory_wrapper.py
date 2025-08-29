
# Neural Memory Integration Wrapper
# =================================

from typing import Dict, Any, Optional
import time

class NeuralMemoryWrapper:
    """Wrapper that tracks routing decisions in Memory"""
    
    def __init__(self, neural_router, memory_system=None):
        self.neural = neural_router
        self.memory = memory_system
        
    async def route_request(self, request):
        """Route and track in memory"""
        start_time = time.time()
        
        # Run original routing
        response = await self.neural.route_request(request)
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        
        # Store in memory if available
        if self.memory:
            try:
                await self.memory.store(
                    content={
                        "request_hash": hash(str(request)),
                        "provider": response.provider,
                        "model": response.model,
                        "latency_ms": latency_ms,
                        "timestamp": time.time()
                    },
                    memory_type="SEMANTIC",
                    metadata={"component": "neural", "action": "route"}
                )
            except Exception as e:
                print(f"Failed to store routing decision: {e}")
                
        return response
