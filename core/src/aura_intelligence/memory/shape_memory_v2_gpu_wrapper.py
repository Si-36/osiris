"""
GPU Wrapper for ShapeMemoryV2 - Feature Flag Controlled
======================================================

Minimal wrapper that redirects to GPU adapter when feature flags are enabled.
"""

import redis.asyncio as redis
from typing import Optional, Dict, Any

from .shape_memory_v2 import ShapeAwareMemoryV2, ShapeMemoryV2Config
from ..adapters.memory_adapter_gpu import GPUMemoryAdapter, create_gpu_memory_adapter, GPUMemoryConfig

class ShapeMemoryV2GPUWrapper:
    """Wrapper that uses GPU adapter based on feature flags"""
    
    def __init__(self, 
                 base_memory: ShapeAwareMemoryV2,
                 redis_client: Optional[redis.Redis] = None):
        self.base_memory = base_memory
        self.redis_client = redis_client or redis.Redis()
        self.gpu_adapter = None
        
    async def initialize(self):
        """Initialize and check feature flags"""
        # Get feature flags from Redis
        flags = await self._get_feature_flags()
        
        if flags.get('SHAPEMEMORYV2_GPU_ENABLED', False):
            # Create GPU adapter
            config = GPUMemoryConfig(
                use_gpu=True,
                shadow_mode=flags.get('SHAPEMEMORYV2_SHADOW', True),
                serve_from_gpu=flags.get('SHAPEMEMORYV2_SERVE', False),
                sample_rate=float(flags.get('SHAPEMEMORYV2_SAMPLERATE', '1.0'))
            )
            
            # Create adapter wrapping the base memory system
            # Note: In production, we'd properly integrate with AURAMemorySystem
            # For now, we use the ShapeMemoryV2 methods
            self.gpu_adapter = GPUMemoryAdapter(
                memory_system=self.base_memory,  # Simplified for demo
                config=config,
                redis_client=self.redis_client
            )
            
    async def _get_feature_flags(self) -> Dict[str, Any]:
        """Get feature flags from Redis"""
        try:
            flags = await self.redis_client.hgetall('feature_flags')
            # Decode and convert
            decoded = {}
            for k, v in flags.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                # Convert string booleans
                if val.lower() in ('true', 'false'):
                    decoded[key] = val.lower() == 'true'
                else:
                    decoded[key] = val
            return decoded
        except Exception:
            return {}
            
    async def store(self, *args, **kwargs):
        """Store with optional GPU acceleration"""
        if self.gpu_adapter:
            # Extract what GPU adapter needs
            content = args[0] if args else kwargs.get('content')
            tda_result = args[1] if len(args) > 1 else kwargs.get('tda_result')
            context_type = args[2] if len(args) > 2 else kwargs.get('context_type', 'general')
            metadata = args[3] if len(args) > 3 else kwargs.get('metadata')
            
            # Use GPU adapter
            return await self.gpu_adapter.store(
                content=content,
                tda_signature={
                    "betti_numbers": {
                        "b0": tda_result.betti_numbers.b0,
                        "b1": tda_result.betti_numbers.b1,
                        "b2": tda_result.betti_numbers.b2
                    },
                    "persistence_diagram": tda_result.persistence_diagram.tolist()
                },
                context_type=context_type,
                metadata=metadata
            )
        else:
            # Use base implementation
            return await self.base_memory.store(*args, **kwargs)
            
    async def retrieve(self, *args, **kwargs):
        """Retrieve with optional GPU acceleration"""
        if self.gpu_adapter:
            # For retrieve, we need to handle the embedding
            # This is simplified - in production we'd properly integrate
            return await self.base_memory.retrieve(*args, **kwargs)
        else:
            return await self.base_memory.retrieve(*args, **kwargs)
            
    async def health_check(self):
        """Check health of both base and GPU systems"""
        base_health = await self.base_memory.health_check()
        
        if self.gpu_adapter:
            gpu_health = await self.gpu_adapter.health()
            # Merge health statuses
            base_health['gpu_adapter'] = {
                'status': gpu_health.status.value,
                'latency_ms': gpu_health.latency_ms,
                'metrics': gpu_health.resource_usage
            }
            
            # Check promotion readiness
            should_promote, reasons = self.gpu_adapter.should_promote()
            base_health['gpu_promotion'] = {
                'ready': should_promote,
                'details': reasons
            }
            
        return base_health


# Helper to set feature flags
async def set_gpu_feature_flags(redis_client: redis.Redis,
                               gpu_enabled: bool = True,
                               shadow: bool = True,
                               serve: bool = False,
                               sample_rate: float = 1.0):
    """Helper to set GPU feature flags in Redis"""
    await redis_client.hset('feature_flags', mapping={
        'SHAPEMEMORYV2_GPU_ENABLED': str(gpu_enabled),
        'SHAPEMEMORYV2_SHADOW': str(shadow),
        'SHAPEMEMORYV2_SERVE': str(serve),
        'SHAPEMEMORYV2_SAMPLERATE': str(sample_rate)
    })