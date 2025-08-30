"""
TDA - Clean Implementation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class TopologyFeatures:
    connected_components: int
    cycles: List[List[str]]
    bottlenecks: List[str]
    centrality_scores: Dict[str, float]
    persistence_score: float

class AgentTopologyAnalyzer:
    """Topological Data Analysis for agent workflows"""
    
    def __init__(self):
        self.workflow_history = []
        
    def analyze_workflow(self, edges: List[Dict[str, str]]) -> TopologyFeatures:
        """Analyze workflow topology"""
        # Build graph
        graph = defaultdict(list)
        nodes = set()
        
        for edge in edges:
            from_node = edge.get("from", edge.get("source"))
            to_node = edge.get("to", edge.get("target"))
            graph[from_node].append(to_node)
            nodes.add(from_node)
            nodes.add(to_node)
            
        # Compute features
        components = self._find_components(graph, nodes)
        cycles = self._find_cycles(graph)
        bottlenecks = self._find_bottlenecks(graph, nodes)
        centrality = self._compute_centrality(graph, nodes)
        persistence = self._compute_persistence(len(cycles), len(bottlenecks))
        
        features = TopologyFeatures(
            connected_components=components,
            cycles=cycles,
            bottlenecks=bottlenecks,
            centrality_scores=centrality,
            persistence_score=persistence
        )
        
        # Track history
        self.workflow_history.append(features)
        
        return features
    
    def _find_components(self, graph: Dict[str, List[str]], nodes: set) -> int:
        """Find connected components using DFS"""
        visited = set()
        components = 0
        
        def dfs(node):
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                    
        for node in nodes:
            if node not in visited:
                dfs(node)
                components += 1
                
        return components
    
    def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find cycles in graph"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
                    
            rec_stack.remove(node)
            return False
            
        for node in graph:
            if node not in visited:
                dfs(node, [])
                
        return cycles
    
    def _find_bottlenecks(self, graph: Dict[str, List[str]], nodes: set) -> List[str]:
        """Find bottleneck nodes (high in-degree, low out-degree)"""
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for node, neighbors in graph.items():
            out_degree[node] = len(neighbors)
            for neighbor in neighbors:
                in_degree[neighbor] += 1
                
        bottlenecks = []
        for node in nodes:
            if in_degree[node] > 2 and out_degree[node] <= 1:
                bottlenecks.append(node)
                
        return bottlenecks
    
    def _compute_centrality(self, graph: Dict[str, List[str]], nodes: set) -> Dict[str, float]:
        """Compute simple centrality scores"""
        centrality = {}
        
        for node in nodes:
            # Degree centrality
            in_deg = sum(1 for n in graph.values() if node in n)
            out_deg = len(graph.get(node, []))
            centrality[node] = (in_deg + out_deg) / (2 * len(nodes))
            
        return centrality
    
    def _compute_persistence(self, num_cycles: int, num_bottlenecks: int) -> float:
        """Compute persistence score"""
        # Simple heuristic: fewer cycles and bottlenecks = higher persistence
        return 1.0 / (1 + num_cycles + num_bottlenecks)
    
    def predict_failure_points(self, workflow: List[Dict[str, str]]) -> List[str]:
        """Predict potential failure points"""
        features = self.analyze_workflow(workflow)
        
        failure_points = []
        
        # Bottlenecks are likely failure points
        failure_points.extend(features.bottlenecks)
        
        # Nodes with high centrality are critical
        for node, score in features.centrality_scores.items():
            if score > 0.7:
                failure_points.append(node)
                
        return list(set(failure_points))