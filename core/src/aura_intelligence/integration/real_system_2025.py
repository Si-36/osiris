"""
Real AURA Intelligence System 2025
Uses actual working components with proper data flow
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import your REAL working components
from ..neural.lnn import LiquidNeuralNetwork, LNNConfig
from ..memory.redis_store import RedisVectorStore, RedisConfig


@dataclass
class RealSystemMetrics:
    lnn_inference_time: float = 0.0
    memory_operations: int = 0
    total_processing_time: float = 0.0
    component_health: Dict[str, str] = None


class RealAURASystem:
    """
    Real AURA system using your actual components
    No mocking - real data flow and processing
    """
    
    def __init__(self):
        # Initialize REAL LNN
        self.lnn_config = LNNConfig(
            input_size=128,
            hidden_size=256, 
            output_size=64,
            num_layers=3,
            time_constant=1.0,
            sparsity=0.7
        )
        self.lnn = LiquidNeuralNetwork(self.lnn_config)
        
        # Initialize REAL Redis (with fallback)
        try:
            redis_config = RedisConfig()
            self.redis_store = RedisVectorStore(redis_config)
        except Exception as e:
            print(f"Redis advanced features unavailable: {e}")
            self.redis_store = None
            
        # Simple Redis for basic operations
        import redis
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        self.metrics = RealSystemMetrics(component_health={})
        
    async def process_real_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through REAL components"""
        start_time = time.time()
        
        # Step 1: Convert input to tensor for LNN
        input_tensor = self._prepare_lnn_input(input_data)
        
        # Step 2: Process through REAL LNN
        lnn_start = time.time()
        lnn_output = self.lnn(input_tensor)
        lnn_time = time.time() - lnn_start
        
        # Step 3: Store results in Redis
        memory_ops = 0
        if self.redis_client:
            try:
                # Store LNN output
                key = f"lnn_output:{int(time.time())}"
                self.redis_client.set(key, str(lnn_output.tolist()))
                self.redis_client.expire(key, 3600)  # 1 hour TTL
                memory_ops += 1
                
                # Store input metadata
                meta_key = f"input_meta:{int(time.time())}"
                self.redis_client.hset(meta_key, mapping={
                    'input_size': len(str(input_data)),
                    'processing_time': lnn_time,
                    'timestamp': time.time()
                })
                self.redis_client.expire(meta_key, 3600)
                memory_ops += 1
                
            except Exception as e:
                print(f"Redis operation failed: {e}")
        
        # Step 4: Generate response
        total_time = time.time() - start_time
        
        # Update metrics
        self.metrics.lnn_inference_time = lnn_time
        self.metrics.memory_operations = memory_ops
        self.metrics.total_processing_time = total_time
        
        return {
            'lnn_output': lnn_output.tolist(),
            'lnn_output_shape': list(lnn_output.shape),
            'processing_metrics': {
                'lnn_inference_ms': lnn_time * 1000,
                'total_processing_ms': total_time * 1000,
                'memory_operations': memory_ops
            },
            'component_status': await self._check_component_health(),
            'real_data_flow': True
        }
    
    def _prepare_lnn_input(self, data: Dict[str, Any]) -> 'torch.Tensor':
        """Convert input data to LNN tensor format"""
        import torch
        
        # Extract features from input data
        features = []
        
        # Basic features
        features.append(len(str(data)))  # Data size
        features.append(time.time() % 1000)  # Temporal feature
        
        # Extract numeric values
        def extract_numbers(obj):
            numbers = []
            if isinstance(obj, (int, float)):
                numbers.append(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    numbers.extend(extract_numbers(v))
            elif isinstance(obj, list):
                for item in obj:
                    numbers.extend(extract_numbers(item))
            return numbers
        
        numbers = extract_numbers(data)
        if numbers:
            features.extend(numbers[:10])  # Limit to 10 numbers
        
        # Pad or truncate to LNN input size
        while len(features) < self.lnn_config.input_size:
            features.append(0.0)
        features = features[:self.lnn_config.input_size]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    async def _check_component_health(self) -> Dict[str, str]:
        """Check health of real components"""
        health = {}
        
        # Check LNN
        try:
            test_input = torch.randn(1, self.lnn_config.input_size)
            _ = self.lnn(test_input)
            health['lnn'] = 'healthy'
        except Exception as e:
            health['lnn'] = f'error: {e}'
        
        # Check Redis
        try:
            self.redis_client.ping()
            health['redis'] = 'healthy'
        except Exception as e:
            health['redis'] = f'error: {e}'
            
        # Check Redis Store (advanced)
        if self.redis_store:
            store_health = self.redis_store.health_check()
            health['redis_store'] = store_health['status']
        else:
            health['redis_store'] = 'unavailable'
            
        return health
    
    def get_lnn_metrics(self) -> Dict[str, Any]:
        """Get real LNN metrics"""
        return self.lnn.get_metrics()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'lnn_config': {
                'input_size': self.lnn_config.input_size,
                'hidden_size': self.lnn_config.hidden_size,
                'output_size': self.lnn_config.output_size,
                'num_layers': self.lnn_config.num_layers,
                'parameters': sum(p.numel() for p in self.lnn.parameters())
            },
            'lnn_metrics': self.get_lnn_metrics(),
            'processing_metrics': {
                'last_lnn_inference_ms': self.metrics.lnn_inference_time * 1000,
                'last_memory_ops': self.metrics.memory_operations,
                'last_total_processing_ms': self.metrics.total_processing_time * 1000
            },
            'component_health': self.metrics.component_health
        }


# Test function for real system
async def test_real_system():
    """Test the real AURA system with actual data flow"""
    print("🧪 Testing Real AURA System...")
    
    system = RealAURASystem()
    
    # Test with real data
    test_inputs = [
        {'query': 'test neural processing', 'priority': 5, 'data': [1, 2, 3, 4, 5]},
        {'task': 'memory storage', 'complexity': 'high', 'values': np.random.randn(10).tolist()},
        {'request': 'system analysis', 'metrics': {'cpu': 0.7, 'memory': 0.5, 'disk': 0.3}}
    ]
    
    results = []
    for i, test_data in enumerate(test_inputs):
        print(f"  Processing test {i+1}/3...")
        result = await system.process_real_data(test_data)
        results.append(result)
        
        # Print key metrics
        metrics = result['processing_metrics']
        print(f"    LNN inference: {metrics['lnn_inference_ms']:.2f}ms")
        print(f"    Total processing: {metrics['total_processing_ms']:.2f}ms")
        print(f"    Memory operations: {metrics['memory_operations']}")
    
    # System statistics
    stats = system.get_system_stats()
    print(f"\n📊 System Statistics:")
    print(f"  LNN Parameters: {stats['lnn_config']['parameters']:,}")
    print(f"  LNN Architecture: {stats['lnn_config']['input_size']} → {stats['lnn_config']['hidden_size']} → {stats['lnn_config']['output_size']}")
    
    # Component health
    health = await system._check_component_health()
    print(f"\n🏥 Component Health:")
    for component, status in health.items():
        print(f"  {component}: {status}")
    
    print(f"\n✅ Real system test complete! Processed {len(results)} requests.")
    return results


if __name__ == "__main__":
    asyncio.run(test_real_system())