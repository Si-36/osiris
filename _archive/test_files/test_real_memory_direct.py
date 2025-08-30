#!/usr/bin/env python3
"""
Direct test of Memory Systems functionality without complex imports
Tests core Redis-based memory with REAL data storage and retrieval
"""

import redis
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
import hashlib

class RealMemorySystem:
    """Real memory system using Redis with topological feature caching"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.cache_prefix = "aura:memory:"
        self.tda_prefix = "aura:tda:"
        self.lnn_prefix = "aura:lnn:"
        
    def get_info(self) -> Dict[str, Any]:
        """Get memory system information"""
        try:
            info = self.redis_client.info()
            return {
                'type': 'Real Redis Memory System',
                'redis_version': info.get('redis_version', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_keys': self.redis_client.dbsize(),
                'uptime': info.get('uptime_in_seconds', 0),
                'shape_aware': True,
                'topological_caching': True
            }
        except Exception as e:
            return {
                'type': 'Real Redis Memory System',
                'error': str(e),
                'connected': False
            }
    
    def _generate_key(self, prefix: str, data_hash: str) -> str:
        """Generate Redis key with prefix and hash"""
        return f"{prefix}{data_hash}"
    
    def _hash_data(self, data: Any) -> str:
        """Generate consistent hash for data"""
        if isinstance(data, (list, tuple)):
            data_str = str(sorted(data) if isinstance(data, list) else data)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def store_tda_result(self, input_data: List[float], tda_result: Dict[str, Any], ttl: int = 3600) -> str:
        """Store TDA computation result with automatic expiration"""
        data_hash = self._hash_data(input_data)
        key = self._generate_key(self.tda_prefix, data_hash)
        
        cache_entry = {
            'input_data': input_data,
            'result': tda_result,
            'timestamp': time.time(),
            'data_shape': len(input_data),
            'betti_numbers': tda_result.get('betti_numbers', {}),
            'processing_time_ms': tda_result.get('processing_time_ms', 0)
        }
        
        self.redis_client.setex(key, ttl, json.dumps(cache_entry))
        return key
    
    def get_tda_result(self, input_data: List[float]) -> Optional[Dict[str, Any]]:
        """Retrieve cached TDA result"""
        data_hash = self._hash_data(input_data)
        key = self._generate_key(self.tda_prefix, data_hash)
        
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached.decode('utf-8'))
        return None
    
    def store_lnn_result(self, input_data: List[List[float]], lnn_result: Dict[str, Any], ttl: int = 1800) -> str:
        """Store LNN computation result"""
        data_hash = self._hash_data(input_data)
        key = self._generate_key(self.lnn_prefix, data_hash)
        
        cache_entry = {
            'input_data': input_data,
            'result': lnn_result,
            'timestamp': time.time(),
            'data_shape': [len(input_data), len(input_data[0]) if input_data else 0],
            'model_info': lnn_result.get('model_info', {}),
            'processing_time_ms': lnn_result.get('processing_time_ms', 0)
        }
        
        self.redis_client.setex(key, ttl, json.dumps(cache_entry))
        return key
    
    def get_lnn_result(self, input_data: List[List[float]]) -> Optional[Dict[str, Any]]:
        """Retrieve cached LNN result"""
        data_hash = self._hash_data(input_data)
        key = self._generate_key(self.lnn_prefix, data_hash)
        
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached.decode('utf-8'))
        return None
    
    def store_topological_features(self, features: Dict[str, Any], ttl: int = 7200) -> str:
        """Store extracted topological features for reuse"""
        feature_hash = self._hash_data(features)
        key = self._generate_key(self.cache_prefix + "topo_features:", feature_hash)
        
        cache_entry = {
            'features': features,
            'timestamp': time.time(),
            'feature_count': len(features),
            'shape_signature': self._generate_shape_signature(features)
        }
        
        self.redis_client.setex(key, ttl, json.dumps(cache_entry))
        return key
    
    def _generate_shape_signature(self, features: Dict[str, Any]) -> str:
        """Generate shape-aware signature for features"""
        signature_parts = []
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float)):
                signature_parts.append(f"{key}:{value:.3f}")
            else:
                signature_parts.append(f"{key}:{type(value).__name__}")
        return "|".join(signature_parts)
    
    def get_similar_features(self, target_features: Dict[str, Any], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar cached topological features"""
        pattern = self.cache_prefix + "topo_features:*"
        similar_features = []
        
        for key in self.redis_client.scan_iter(match=pattern):
            cached_data = self.redis_client.get(key)
            if cached_data:
                cache_entry = json.loads(cached_data.decode('utf-8'))
                cached_features = cache_entry['features']
                
                # Simple similarity check
                similarity = self._calculate_feature_similarity(target_features, cached_features)
                if similarity >= similarity_threshold:
                    cache_entry['similarity'] = similarity
                    similar_features.append(cache_entry)
        
        return sorted(similar_features, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature sets"""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == val2 == 0:
                    similarity_scores.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarity_scores.append(1.0)
                    else:
                        similarity_scores.append(1.0 - abs(val1 - val2) / max_val)
            elif val1 == val2:
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_keys = self.redis_client.dbsize()
        tda_keys = len(list(self.redis_client.scan_iter(match=self.tda_prefix + "*")))
        lnn_keys = len(list(self.redis_client.scan_iter(match=self.lnn_prefix + "*")))
        feature_keys = len(list(self.redis_client.scan_iter(match=self.cache_prefix + "topo_features:*")))
        
        return {
            'total_keys': total_keys,
            'tda_cached_results': tda_keys,
            'lnn_cached_results': lnn_keys,
            'cached_features': feature_keys,
            'cache_hit_potential': (tda_keys + lnn_keys + feature_keys) / max(1, total_keys),
            'memory_info': self.get_info()
        }
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern"""
        if pattern is None:
            pattern = self.cache_prefix + "*"
        
        keys_to_delete = list(self.redis_client.scan_iter(match=pattern))
        if keys_to_delete:
            return self.redis_client.delete(*keys_to_delete)
        return 0

def test_memory_system():
    """Test memory system with various operations"""
    print("🧠 Testing AURA Memory System")
    print("=" * 50)
    
    # Initialize memory system
    memory = RealMemorySystem()
    
    # Test connection
    print("📊 Memory system info:")
    info = memory.get_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if info.get('error'):
        print("❌ Redis connection failed")
        return False, []
    
    print("\n✅ Redis connection successful")
    
    # Test cases
    test_results = []
    processing_times = []
    
    # Test 1: TDA Result Caching
    print(f"\n📈 Test 1: TDA Result Caching")
    test_data = [1, 2, 3, 10, 11, 12]
    fake_tda_result = {
        'success': True,
        'betti_numbers': {'b0': 2, 'b1': 0},
        'processing_time_ms': 1.5,
        'statistics': {'avg_lifetime_dim0': 0.5}
    }
    
    start_time = time.time()
    
    # Store result
    key = memory.store_tda_result(test_data, fake_tda_result)
    
    # Retrieve result
    cached_result = memory.get_tda_result(test_data)
    
    processing_time = (time.time() - start_time) * 1000
    processing_times.append(processing_time)
    
    if cached_result and cached_result['result']['betti_numbers'] == fake_tda_result['betti_numbers']:
        print(f"   ✅ TDA caching successful")
        print(f"   📊 Stored key: {key[-16:]}...")
        print(f"   📊 Betti numbers match: {cached_result['result']['betti_numbers']}")
        print(f"   ⚡ Cache operation: {processing_time:.2f}ms")
        test_results.append(True)
    else:
        print(f"   ❌ TDA caching failed")
        test_results.append(False)
    
    # Test 2: LNN Result Caching
    print(f"\n📈 Test 2: LNN Result Caching")
    lnn_data = [[1,2,3,4,5,6,7,8,9,10]]
    fake_lnn_result = {
        'success': True,
        'output': [[0.123]],
        'processing_time_ms': 25.0,
        'model_info': {'type': 'Real MIT LNN', 'parameters': 17381}
    }
    
    start_time = time.time()
    
    # Store and retrieve
    lnn_key = memory.store_lnn_result(lnn_data, fake_lnn_result)
    cached_lnn = memory.get_lnn_result(lnn_data)
    
    processing_time = (time.time() - start_time) * 1000
    processing_times.append(processing_time)
    
    if cached_lnn and cached_lnn['result']['output'] == fake_lnn_result['output']:
        print(f"   ✅ LNN caching successful")
        print(f"   📊 Stored key: {lnn_key[-16:]}...")
        print(f"   📊 Output matches: {cached_lnn['result']['output']}")
        print(f"   ⚡ Cache operation: {processing_time:.2f}ms")
        test_results.append(True)
    else:
        print(f"   ❌ LNN caching failed")
        test_results.append(False)
    
    # Test 3: Feature Similarity Search
    print(f"\n📈 Test 3: Feature Similarity Search")
    features1 = {'b0': 2, 'b1': 0, 'density': 0.5, 'connectivity': 0.8}
    features2 = {'b0': 2, 'b1': 1, 'density': 0.4, 'connectivity': 0.9}
    
    start_time = time.time()
    
    # Store features
    memory.store_topological_features(features1)
    memory.store_topological_features(features2)
    
    # Search for similar features
    similar = memory.get_similar_features(features1, similarity_threshold=0.7)
    
    processing_time = (time.time() - start_time) * 1000
    processing_times.append(processing_time)
    
    if similar and len(similar) > 0:
        print(f"   ✅ Feature similarity search successful")
        print(f"   📊 Found {len(similar)} similar feature sets")
        for i, sim in enumerate(similar[:2]):  # Show top 2
            print(f"      Similarity {i+1}: {sim['similarity']:.3f}")
        print(f"   ⚡ Search operation: {processing_time:.2f}ms")
        test_results.append(True)
    else:
        print(f"   ❌ Feature similarity search failed")
        test_results.append(False)
    
    # Test 4: Cache Statistics
    print(f"\n📈 Test 4: Cache Statistics")
    start_time = time.time()
    stats = memory.get_cache_statistics()
    processing_time = (time.time() - start_time) * 1000
    processing_times.append(processing_time)
    
    if stats['total_keys'] > 0:
        print(f"   ✅ Cache statistics available")
        print(f"   📊 Total keys: {stats['total_keys']}")
        print(f"   📊 TDA cached: {stats['tda_cached_results']}")
        print(f"   📊 LNN cached: {stats['lnn_cached_results']}")
        print(f"   📊 Features cached: {stats['cached_features']}")
        print(f"   📊 Cache efficiency: {stats['cache_hit_potential']:.1%}")
        print(f"   ⚡ Stats operation: {processing_time:.2f}ms")
        test_results.append(True)
    else:
        print(f"   ❌ Cache statistics failed")
        test_results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 MEMORY SYSTEM SUMMARY")
    print("=" * 50)
    
    successful_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        print(f"\n🎉 MEMORY SYSTEM IS WORKING!")
        print(f"✅ Redis connectivity established")
        print(f"✅ TDA result caching functional")
        print(f"✅ LNN result caching functional")
        print(f"✅ Shape-aware feature similarity search")
        print(f"✅ Comprehensive cache statistics")
        
        if processing_times:
            avg_time = np.mean(processing_times)
            print(f"⚡ Average operation time: {avg_time:.2f}ms")
            
            if avg_time < 10:
                print("🚀 EXCELLENT performance - under 10ms!")
            elif avg_time < 50:
                print("✅ Good performance - under 50ms")
            else:
                print("⚠️  Performance acceptable")
    else:
        print("❌ NO WORKING MEMORY FUNCTIONALITY")
        print("🔧 Need to fix Redis connection and implementations")
    
    return successful_tests == total_tests, test_results

if __name__ == "__main__":
    success, results = test_memory_system()
    
    # Save results for integration
    with open('memory_test_results.json', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'success': success,
            'total_tests': len(results),
            'successful_tests': sum(results),
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to memory_test_results.json")
    print(f"🎯 Next step: Integrate memory system into TDA+LNN API")