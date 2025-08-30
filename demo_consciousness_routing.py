#!/usr/bin/env python3
"""
🧠 Demo: Consciousness-Aware Routing in Action
==============================================

Shows the unique consciousness-based routing without dependencies
"""

import asyncio
import numpy as np
from typing import Dict, List, Set
import json


class SimpleNode:
    def __init__(self, id: str, consciousness: float = 0.5):
        self.id = id
        self.consciousness_level = consciousness
        self.position = np.random.randn(3)
        self.connections: Set[str] = set()
        self.attention_focus = {
            "task_priority": np.random.random(),
            "resource_usage": np.random.random(),
            "collaboration": np.random.random()
        }
    
    def distance_to(self, other: 'SimpleNode') -> float:
        return float(np.linalg.norm(self.position - other.position))


class ConsciousnessRouter:
    """Simplified consciousness-aware routing demo"""
    
    def __init__(self):
        self.nodes: Dict[str, SimpleNode] = {}
    
    def add_node(self, node: SimpleNode):
        self.nodes[node.id] = node
        # Auto-connect based on consciousness alignment
        for other_id, other in self.nodes.items():
            if other_id != node.id:
                alignment = self._calculate_alignment(node, other)
                if alignment > 0.6:  # High alignment
                    node.connections.add(other_id)
                    other.connections.add(node.id)
    
    def _calculate_alignment(self, node1: SimpleNode, node2: SimpleNode) -> float:
        """Calculate consciousness alignment between nodes"""
        # Consciousness level similarity
        consciousness_diff = abs(node1.consciousness_level - node2.consciousness_level)
        consciousness_similarity = 1.0 - consciousness_diff
        
        # Attention alignment (cosine similarity)
        keys = set(node1.attention_focus.keys()) | set(node2.attention_focus.keys())
        vec1 = np.array([node1.attention_focus.get(k, 0) for k in keys])
        vec2 = np.array([node2.attention_focus.get(k, 0) for k in keys])
        
        attention_alignment = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Combined alignment
        return consciousness_similarity * 0.5 + attention_alignment * 0.5
    
    def find_consciousness_path(self, source_id: str, target_id: str) -> List[str]:
        """Find path that maximizes consciousness alignment"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        
        # Simple BFS with consciousness weighting
        queue = [(source_id, [source_id], 0.0)]
        visited = set()
        best_path = []
        best_score = -1
        
        while queue:
            current_id, path, score = queue.pop(0)
            
            if current_id == target_id:
                if score > best_score:
                    best_score = score
                    best_path = path
                continue
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            current = self.nodes[current_id]
            
            for next_id in current.connections:
                if next_id not in visited:
                    next_node = self.nodes[next_id]
                    alignment = self._calculate_alignment(current, next_node)
                    new_score = score + alignment
                    queue.append((next_id, path + [next_id], new_score))
        
        return best_path


async def demo_consciousness_routing():
    """Demonstrate consciousness-aware routing"""
    print("🧠 Consciousness-Aware Routing Demo")
    print("=" * 50)
    
    router = ConsciousnessRouter()
    
    # Create agents with different consciousness levels
    agents = [
        SimpleNode("worker_1", consciousness=0.3),    # Low consciousness
        SimpleNode("worker_2", consciousness=0.4),
        SimpleNode("analyst_1", consciousness=0.6),   # Medium consciousness
        SimpleNode("analyst_2", consciousness=0.7),
        SimpleNode("supervisor", consciousness=0.9),  # High consciousness
        SimpleNode("executive", consciousness=1.0),   # Maximum consciousness
    ]
    
    # Set specific attention patterns
    agents[0].attention_focus = {"task_priority": 0.9, "resource_usage": 0.1, "collaboration": 0.2}
    agents[1].attention_focus = {"task_priority": 0.8, "resource_usage": 0.2, "collaboration": 0.3}
    agents[2].attention_focus = {"task_priority": 0.5, "resource_usage": 0.5, "collaboration": 0.5}
    agents[3].attention_focus = {"task_priority": 0.4, "resource_usage": 0.6, "collaboration": 0.6}
    agents[4].attention_focus = {"task_priority": 0.3, "resource_usage": 0.3, "collaboration": 0.8}
    agents[5].attention_focus = {"task_priority": 0.2, "resource_usage": 0.2, "collaboration": 0.9}
    
    # Add to router
    for agent in agents:
        router.add_node(agent)
    
    # Show network
    print("\n📊 Agent Network:")
    for agent in agents:
        print(f"  {agent.id}:")
        print(f"    - Consciousness: {agent.consciousness_level:.1f}")
        print(f"    - Connections: {list(agent.connections)}")
        print(f"    - Attention: {json.dumps(agent.attention_focus, indent=0).replace(chr(10), ', ')}")
    
    # Test routing
    print("\n🚀 Testing Consciousness-Based Routing:")
    
    test_routes = [
        ("worker_1", "executive"),
        ("worker_2", "supervisor"),
        ("analyst_1", "analyst_2"),
    ]
    
    for source, target in test_routes:
        path = router.find_consciousness_path(source, target)
        
        print(f"\n  Route: {source} → {target}")
        print(f"  Path: {' → '.join(path)}")
        
        if len(path) > 1:
            print("  Consciousness levels along path:")
            for node_id in path:
                node = router.nodes[node_id]
                print(f"    - {node_id}: {node.consciousness_level:.1f}")
            
            # Calculate average consciousness alignment
            total_alignment = 0
            for i in range(len(path) - 1):
                node1 = router.nodes[path[i]]
                node2 = router.nodes[path[i+1]]
                alignment = router._calculate_alignment(node1, node2)
                total_alignment += alignment
            
            avg_alignment = total_alignment / (len(path) - 1) if len(path) > 1 else 0
            print(f"  Average alignment score: {avg_alignment:.3f}")
    
    # Demonstrate consciousness-based decision
    print("\n🤝 Consciousness-Based Consensus:")
    
    # Simulate voting with consciousness weights
    proposal = "Upgrade to quantum routing"
    votes = {
        "worker_1": False,      # Low consciousness votes no
        "worker_2": False,
        "analyst_1": True,      # Medium consciousness mixed
        "analyst_2": False,
        "supervisor": True,     # High consciousness votes yes
        "executive": True,
    }
    
    # Calculate weighted result
    total_weight = 0
    weighted_yes = 0
    
    for agent_id, vote in votes.items():
        weight = router.nodes[agent_id].consciousness_level
        total_weight += weight
        if vote:
            weighted_yes += weight
    
    consensus = weighted_yes / total_weight
    
    print(f"  Proposal: {proposal}")
    print(f"  Simple vote: {sum(votes.values())}/{len(votes)} = {sum(votes.values())/len(votes):.1%}")
    print(f"  Consciousness-weighted: {weighted_yes:.1f}/{total_weight:.1f} = {consensus:.1%}")
    print(f"  Decision: {'APPROVED' if consensus > 0.5 else 'REJECTED'} (consciousness-weighted)")
    
    print("\n✨ This is what makes AURA unique - decisions and routing")
    print("   based on consciousness levels, not just connectivity!")


if __name__ == "__main__":
    asyncio.run(demo_consciousness_routing())