"""
Shape-Aware Memory System
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
import hashlib
import json

class ShapeMemory:
    """Shape-aware memory with topological indexing"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory = {}
        self.access_history = deque(maxlen=capacity)
        self.shape_index = {}  # Topological shape -> memory keys
    
    def _compute_shape_signature(self, topology: Dict[str, Any]) -> str:
        """Compute a signature for topological shape"""
        # Create shape signature from Betti numbers and key features
        shape_features = {
            "betti_0": topology.get("betti_0", 1),
            "betti_1": topology.get("betti_1", 0),
            "betti_2": topology.get("betti_2", 0),
            "num_nodes": topology.get("num_nodes", 0),
            "density": round(topology.get("density", 0), 2),
        }
        
        # Hash the shape
        shape_str = json.dumps(shape_features, sort_keys=True)
        return hashlib.md5(shape_str.encode()).hexdigest()[:16]
    
    def store(self, key: str, data: Any, topology: Dict[str, Any]):
        """Store data with topological indexing"""
        # Compute shape signature
        shape_sig = self._compute_shape_signature(topology)
        
        # Store in memory
        self.memory[key] = {
            "data": data,
            "topology": topology,
            "shape": shape_sig,
            "timestamp": len(self.access_history),
        }
        
        # Update shape index
        if shape_sig not in self.shape_index:
            self.shape_index[shape_sig] = []
        self.shape_index[shape_sig].append(key)
        
        # Update access history
        self.access_history.append(key)
        
        # Evict if over capacity
        if len(self.memory) > self.capacity:
            self._evict_lru()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key"""
        if key in self.memory:
            self.access_history.append(key)
            return self.memory[key]["data"]
        return None
    
    def find_similar_shapes(self, topology: Dict[str, Any], threshold: float = 0.9) -> List[str]:
        """Find memories with similar topological shapes"""
        target_shape = self._compute_shape_signature(topology)
        
        # For now, exact match (can be extended to fuzzy matching)
        if target_shape in self.shape_index:
            return self.shape_index[target_shape]
        
        # Find approximate matches
        similar = []
        target_betti = (topology.get("betti_0", 1), topology.get("betti_1", 0))
        
        for shape_sig, keys in self.shape_index.items():
            if keys:
                sample_key = keys[0]
                mem_topology = self.memory[sample_key]["topology"]
                mem_betti = (mem_topology.get("betti_0", 1), mem_topology.get("betti_1", 0))
                
                # Simple similarity based on Betti numbers
                if abs(target_betti[0] - mem_betti[0]) <= 1 and abs(target_betti[1] - mem_betti[1]) <= 2:
                    similar.extend(keys)
        
        return similar
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.access_history:
            lru_key = self.access_history[0]
            if lru_key in self.memory:
                shape_sig = self.memory[lru_key]["shape"]
                del self.memory[lru_key]
                
                # Update shape index
                if shape_sig in self.shape_index:
                    self.shape_index[shape_sig].remove(lru_key)
                    if not self.shape_index[shape_sig]:
                        del self.shape_index[shape_sig]

class HierarchicalMemory:
    """Multi-tier memory system with CXL support"""
    
    def __init__(self):
        self.tiers = {
            "L1_CACHE": ShapeMemory(capacity=100),
            "L2_CACHE": ShapeMemory(capacity=500),
            "L3_CACHE": ShapeMemory(capacity=1000),
            "RAM": ShapeMemory(capacity=5000),
            "CXL_HOT": ShapeMemory(capacity=10000),
        }
        self.access_count = {}
    
    def store(self, key: str, data: Any, topology: Dict[str, Any]):
        """Store data in appropriate tier"""
        # For now, store in L3 cache
        self.tiers["L3_CACHE"].store(key, data, topology)
        self.access_count[key] = 1
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from any tier"""
        # Search through tiers
        for tier_name, tier in self.tiers.items():
            data = tier.retrieve(key)
            if data is not None:
                # Promote to higher tier if frequently accessed
                if key in self.access_count:
                    self.access_count[key] += 1
                    if self.access_count[key] > 10 and tier_name != "L1_CACHE":
                        # Promote to L1
                        topology = tier.memory[key]["topology"]
                        self.tiers["L1_CACHE"].store(key, data, topology)
                return data
        return None
