"""
Request Processor Service
Handles request preprocessing and feature extraction
"""

import numpy as np
from typing import Dict, Any, List, Optional
import hashlib
import time
import structlog

logger = structlog.get_logger()


class RequestProcessor:
    """
    Process and prepare requests for routing
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.processing_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0
        }
        self.logger = logger.bind(service="request_processor")
        
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request and extract features
        """
        start_time = time.perf_counter()
        
        # Generate cache key
        cache_key = self._generate_cache_key(request_data)
        
        # Check cache
        if cache_key in self.feature_cache:
            self.processing_stats["cache_hits"] += 1
            return self.feature_cache[cache_key]
        
        # Extract features
        processed = {
            "original_data": request_data,
            "features": self._extract_features(request_data),
            "metadata": self._extract_metadata(request_data),
            "routing_hints": self._extract_routing_hints(request_data),
            "timestamp": time.time()
        }
        
        # Cache result
        self.feature_cache[cache_key] = processed
        
        # Limit cache size
        if len(self.feature_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.feature_cache.keys(),
                key=lambda k: self.feature_cache[k]["timestamp"]
            )[:100]
            for key in oldest_keys:
                del self.feature_cache[key]
        
        # Update stats
        self.processing_stats["total_processed"] += 1
        processing_time = (time.perf_counter() - start_time) * 1000
        self.processing_stats["avg_processing_time_ms"] = (
            self.processing_stats["avg_processing_time_ms"] * 0.9 + 
            processing_time * 0.1
        )
        
        return processed
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        # Create deterministic string representation
        key_parts = []
        
        for k, v in sorted(request_data.items()):
            if k in ["data", "type", "priority", "complexity"]:
                if isinstance(v, (list, dict)):
                    key_parts.append(f"{k}:{len(str(v))}")
                else:
                    key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _extract_features(self, request_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from request"""
        features = []
        
        # Data size and complexity
        data = request_data.get("data", [])
        if isinstance(data, list):
            features.extend([
                len(data),
                np.mean(data) if data and all(isinstance(x, (int, float)) for x in data[:10]) else 0,
                np.std(data) if data and all(isinstance(x, (int, float)) for x in data[:10]) else 0
            ])
        else:
            features.extend([1, 0, 0])
        
        # Request characteristics
        features.append(request_data.get("priority", 0.5))
        features.append(request_data.get("complexity", 0.5))
        features.append(len(str(request_data)) / 1000.0)  # Normalized size
        
        return features
    
    def _extract_metadata(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from request"""
        return {
            "request_type": request_data.get("type", "unknown"),
            "has_requirements": "requires" in request_data,
            "is_batch": isinstance(request_data.get("data"), list) and len(request_data.get("data", [])) > 1,
            "urgency_level": self._calculate_urgency(request_data)
        }
    
    def _extract_routing_hints(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract routing hints from request"""
        hints = {}
        
        # Service preferences
        if "prefer_services" in request_data:
            hints["preferred_services"] = request_data["prefer_services"]
        
        # Capability requirements
        if "requires" in request_data:
            hints["required_capabilities"] = request_data["requires"]
        
        # Performance requirements
        if "max_latency_ms" in request_data:
            hints["max_latency_ms"] = request_data["max_latency_ms"]
        
        # Load sensitivity
        request_type = request_data.get("type", "")
        if request_type in ["training", "batch_processing"]:
            hints["load_sensitive"] = True
        
        return hints
    
    def _calculate_urgency(self, request_data: Dict[str, Any]) -> str:
        """Calculate request urgency level"""
        priority = request_data.get("priority", 0.5)
        
        if priority >= 0.9:
            return "critical"
        elif priority >= 0.7:
            return "high"
        elif priority >= 0.3:
            return "normal"
        else:
            return "low"
    
    def batch_process(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple requests in batch"""
        return [self.process_request(req) for req in requests]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        cache_hit_rate = (
            self.processing_stats["cache_hits"] / 
            max(1, self.processing_stats["total_processed"])
        )
        
        return {
            **self.processing_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.feature_cache)
        }