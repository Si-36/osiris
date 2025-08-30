"""
Agent Topology Analyzer - Production TDA for Multi-Agent Systems
===============================================================

Analyzes agent workflows and communication patterns to detect bottlenecks,
predict failures, and optimize system performance.

Key Features:
- Workflow DAG analysis with cycle detection
- Communication graph health assessment  
- Bottleneck scoring and critical path analysis
- Failure prediction from topological signatures
- Real-time anomaly detection
"""

import asyncio
import time
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
import structlog

from .algorithms import (
    compute_persistence,
    diagram_entropy,
    diagram_distance,
    vectorize_diagram
)

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

@dataclass
class WorkflowFeatures:
    """Features extracted from workflow topology."""
    workflow_id: str
    timestamp: float
    
    # Graph metrics
    num_agents: int
    num_edges: int
    has_cycles: bool
    longest_path_length: int
    critical_path_agents: List[str]
    
    # Centrality metrics
    bottleneck_agents: List[str]
    betweenness_scores: Dict[str, float]
    clustering_coefficients: Dict[str, float]
    
    # Persistence features
    persistence_entropy: float
    diagram_distance_from_baseline: float
    stability_index: float  # 0-1, higher is more stable
    
    # Risk assessment
    bottleneck_score: float  # 0-1, higher is worse
    failure_risk: float  # 0-1
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp,
            "num_agents": self.num_agents,
            "num_edges": self.num_edges,
            "has_cycles": self.has_cycles,
            "longest_path_length": self.longest_path_length,
            "critical_path_agents": self.critical_path_agents,
            "bottleneck_agents": self.bottleneck_agents,
            "betweenness_scores": self.betweenness_scores,
            "clustering_coefficients": self.clustering_coefficients,
            "persistence_entropy": self.persistence_entropy,
            "diagram_distance_from_baseline": self.diagram_distance_from_baseline,
            "stability_index": self.stability_index,
            "bottleneck_score": self.bottleneck_score,
            "failure_risk": self.failure_risk,
            "recommendations": self.recommendations
        }


@dataclass
class CommunicationFeatures:
    """Features extracted from agent communication topology."""
    timestamp: float
    
    # Network metrics
    total_agents: int
    active_connections: int
    network_density: float
    
    # Component analysis
    num_components: int
    largest_component_size: int
    isolated_agents: List[str]
    
    # Communication patterns
    hub_agents: List[str]  # High degree centrality
    bridge_agents: List[str]  # Connect components
    overloaded_agents: List[str]
    
    # Health metrics
    communication_imbalance: float  # 0-1, variance in load
    fragmentation_score: float  # 0-1, how fragmented
    overall_health: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "total_agents": self.total_agents,
            "active_connections": self.active_connections,
            "network_density": self.network_density,
            "num_components": self.num_components,
            "largest_component_size": self.largest_component_size,
            "isolated_agents": self.isolated_agents,
            "hub_agents": self.hub_agents,
            "bridge_agents": self.bridge_agents,
            "overloaded_agents": self.overloaded_agents,
            "communication_imbalance": self.communication_imbalance,
            "fragmentation_score": self.fragmentation_score,
            "overall_health": self.overall_health
        }


@dataclass
class TopologicalAnomaly:
    """Detected anomaly in system topology."""
    anomaly_id: str
    detected_at: float
    anomaly_type: str  # "bottleneck_formed", "component_split", "cycle_detected", etc.
    severity: float  # 0-1
    affected_agents: List[str]
    description: str
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "anomaly_id": self.anomaly_id,
            "detected_at": self.detected_at,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "affected_agents": self.affected_agents,
            "description": self.description,
            "recommended_action": self.recommended_action
        }


class HealthStatus(str, Enum):
    """System health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


# ==================== Agent Topology Analyzer ====================

class AgentTopologyAnalyzer:
    """
    Production-grade topology analyzer for multi-agent systems.
    Focuses on practical metrics for agent workflows and communication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Baselines for anomaly detection
        self.workflow_baselines: Dict[str, Dict[str, Any]] = {}
        self.communication_baseline: Optional[Dict[str, Any]] = None
        
        # Historical data for trend analysis
        self.workflow_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.communication_history: deque = deque(maxlen=100)
        
        # Anomaly detection state
        self.anomaly_detector = TopologicalAnomalyDetector(
            sensitivity=self.config.get("anomaly_sensitivity", 0.8)
        )
        
        # Thresholds
        self.bottleneck_threshold = self.config.get("bottleneck_threshold", 0.7)
        self.overload_threshold = self.config.get("overload_threshold", 10)
        self.cycle_penalty = self.config.get("cycle_penalty", 0.2)
        
    # ==================== Workflow Analysis ====================
    
    async def analyze_workflow(self, 
                         workflow_id: str,
                         workflow_data: Dict[str, Any]) -> WorkflowFeatures:
        """
        Analyze a workflow DAG for bottlenecks and performance issues.
        
        Args:
        workflow_id: Unique workflow identifier
        workflow_data: Dict with 'agents' and 'dependencies' keys
        
        Returns:
        WorkflowFeatures with comprehensive analysis
        """
        logger.info("Analyzing workflow topology", workflow_id=workflow_id)
        
        # Build workflow graph
        G = self._build_workflow_graph(workflow_data)
        
        # Extract basic metrics
        num_agents = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Detect cycles
        has_cycles, cycles = self._detect_cycles(G)
        
        # Find critical path
        critical_path, path_length = self._find_critical_path(G)
        
        # Compute centrality metrics
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G.to_undirected())
        
        # Identify bottlenecks
        bottleneck_agents = self._identify_bottlenecks(G, betweenness)
        
        # Compute persistence features
        persistence_features = await self._compute_persistence_features(G, workflow_id)
        
        # Calculate risk scores
        bottleneck_score = self._calculate_bottleneck_score(
            betweenness, bottleneck_agents, has_cycles
        )
        
        failure_risk = self._calculate_failure_risk(
            bottleneck_score,
            persistence_features["diagram_distance"],
            persistence_features["entropy_trend"]
        )
        
        # Generate recommendations
        recommendations = self._generate_workflow_recommendations(
            bottleneck_agents, has_cycles, failure_risk
        )
        
        # Build features
        features = WorkflowFeatures(
            workflow_id=workflow_id,
            timestamp=time.time(),
            num_agents=num_agents,
            num_edges=num_edges,
            has_cycles=has_cycles,
            longest_path_length=path_length,
            critical_path_agents=critical_path,
            bottleneck_agents=bottleneck_agents,
            betweenness_scores=betweenness,
            clustering_coefficients=clustering,
            persistence_entropy=persistence_features["entropy"],
            diagram_distance_from_baseline=persistence_features["diagram_distance"],
            stability_index=persistence_features["stability"],
            bottleneck_score=bottleneck_score,
            failure_risk=failure_risk,
            recommendations=recommendations
        )
        
        # Update history and check for anomalies
        self.workflow_history[workflow_id].append(features)
        anomalies = await self.anomaly_detector.check_workflow(features)
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} anomalies", anomalies=anomalies)
            
        return features
            
    def _build_workflow_graph(self, workflow_data: Dict[str, Any]) -> nx.DiGraph:
        """Build directed graph from workflow data."""
        G = nx.DiGraph()
        
        agents = workflow_data.get("agents", [])
        dependencies = workflow_data.get("dependencies", [])
        
        # Add nodes (agents)
        for agent in agents:
            agent_id = agent.get("id", f"agent_{len(G.nodes)}")
            G.add_node(agent_id, **agent)
            
        # Add edges (dependencies)
        for dep in dependencies:
            source = dep.get("source")
            target = dep.get("target")
            weight = dep.get("weight", 1.0)
            
            if source and target:
                G.add_edge(source, target, weight=weight, **dep)
                
        return G
        
    def _detect_cycles(self, G: nx.DiGraph) -> Tuple[bool, List[List[str]]]:
        """Detect cycles in workflow graph."""
        try:
            cycles = list(nx.simple_cycles(G))
            return len(cycles) > 0, cycles
        except:
            return False, []
            
    def _find_critical_path(self, G: nx.DiGraph) -> Tuple[List[str], int]:
        """Find longest path in DAG (critical path)."""
        if nx.is_directed_acyclic_graph(G):
            try:
                # Use topological sort for DAG
                topo_order = list(nx.topological_sort(G))
                
                # Dynamic programming to find longest path
                dist = {node: 0 for node in G.nodes()}
                parent = {node: None for node in G.nodes()}
                
                for u in topo_order:
                    for v in G.successors(u):
                        weight = G[u][v].get("weight", 1)
                        if dist[u] + weight > dist[v]:
                            dist[v] = dist[u] + weight
                            parent[v] = u
                            
                # Find node with maximum distance
                end_node = max(dist.items(), key=lambda x: x[1])[0]
                
                # Reconstruct path
                path = []
                node = end_node
                while node is not None:
                    path.append(node)
                    node = parent[node]
                    
                path.reverse()
                return path, len(path)
                
            except Exception as e:
                logger.error(f"Error finding critical path: {e}")
                return [], 0
        else:
            # For cyclic graphs, return empty
            return [], 0
            
    def _identify_bottlenecks(self, G: nx.Graph, 
                            betweenness: Dict[str, float]) -> List[str]:
        """Identify bottleneck agents based on centrality."""
        if not betweenness:
            return []
            
        # Calculate threshold
        mean_betweenness = np.mean(list(betweenness.values()))
        std_betweenness = np.std(list(betweenness.values()))
        threshold = mean_betweenness + 2 * std_betweenness
        
        # Find bottlenecks
        bottlenecks = []
        for node, score in betweenness.items():
            if score > threshold:
                bottlenecks.append(node)
                
        return sorted(bottlenecks, key=lambda x: betweenness[x], reverse=True)
        
    async def _compute_persistence_features(self, G: nx.Graph, 
                                          workflow_id: str) -> Dict[str, float]:
        """Compute topological persistence features."""
        # Convert graph to point cloud embedding
        if G.number_of_nodes() == 0:
            return {
                "entropy": 0.0,
                "diagram_distance": 0.0,
                "stability": 1.0,
                "entropy_trend": 0.0
            }
            
        # Use graph layout as embedding
        try:
            pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))
            point_cloud = np.array([pos[node] for node in G.nodes()])
        except:
            # Fallback to random if layout fails
            point_cloud = np.random.rand(G.number_of_nodes(), 2)
            
        # Compute persistence diagram
        diagram = compute_persistence(point_cloud)
        
        # Extract features
        entropy = diagram_entropy(diagram)
        
        # Compare to baseline
        baseline = self.workflow_baselines.get(workflow_id)
        if baseline is not None:
            diagram_dist = diagram_distance(diagram, baseline["diagram"])
        else:
            diagram_dist = 0.0
            # Store as baseline
            self.workflow_baselines[workflow_id] = {
                "diagram": diagram,
                "entropy": entropy,
                "timestamp": time.time()
            }
            
        # Calculate stability from history
        history = self.workflow_history[workflow_id]
        if len(history) >= 2:
            recent_entropies = [h.persistence_entropy for h in list(history)[-10:]]
            entropy_variance = np.var(recent_entropies)
            stability = 1.0 / (1.0 + entropy_variance)
            
            # Entropy trend
            if len(recent_entropies) >= 2:
                entropy_trend = recent_entropies[-1] - recent_entropies[-2]
            else:
                entropy_trend = 0.0
        else:
            stability = 1.0
            entropy_trend = 0.0
            
        return {
            "entropy": float(entropy),
            "diagram_distance": float(diagram_dist),
            "stability": float(stability),
            "entropy_trend": float(entropy_trend)
        }
        
    def _calculate_bottleneck_score(self, betweenness: Dict[str, float],
                                  bottleneck_agents: List[str],
                                  has_cycles: bool) -> float:
        """Calculate overall bottleneck score (0-1)."""
        if not betweenness:
            return 0.0
            
        # Base score from bottleneck concentration
        if bottleneck_agents:
            max_betweenness = max(betweenness.values())
            mean_betweenness = np.mean(list(betweenness.values()))
            concentration = max_betweenness / (mean_betweenness + 1e-6)
            base_score = min(concentration / 10.0, 1.0)  # Normalize
        else:
            base_score = 0.0
            
        # Penalty for cycles
        if has_cycles:
            base_score = min(base_score + self.cycle_penalty, 1.0)
            
        # Consider number of bottlenecks
        bottleneck_ratio = len(bottleneck_agents) / len(betweenness)
        if bottleneck_ratio > 0.3:  # Too many bottlenecks
            base_score = min(base_score + 0.1, 1.0)
            
        return float(base_score)
        
    def _calculate_failure_risk(self, bottleneck_score: float,
                              diagram_distance: float,
                              entropy_trend: float) -> float:
        """Calculate failure risk based on multiple factors."""
        # Weighted combination
        risk = (
            0.5 * bottleneck_score +
            0.3 * min(diagram_distance / 2.0, 1.0) +  # Normalize distance
            0.2 * max(entropy_trend, 0.0)  # Only positive trends increase risk
        )
        
        return float(min(risk, 1.0))
        
    def _generate_workflow_recommendations(self, bottleneck_agents: List[str],
                                         has_cycles: bool,
                                         failure_risk: float) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if bottleneck_agents:
            recommendations.append(
                f"Distribute load from bottleneck agents: {', '.join(bottleneck_agents[:3])}"
            )
            
        if has_cycles:
            recommendations.append(
                "Remove circular dependencies to improve workflow stability"
            )
            
        if failure_risk > 0.7:
            recommendations.append(
                "High failure risk detected - consider adding redundancy"
            )
        elif failure_risk > 0.5:
            recommendations.append(
                "Monitor system closely - moderate failure risk"
            )
            
        if not recommendations:
            recommendations.append("System topology appears healthy")
            
        return recommendations
        
    # ==================== Communication Analysis ====================
    
    async def analyze_communications(self, 
                                   communication_data: Dict[str, Any]) -> CommunicationFeatures:
        """
        Analyze agent communication patterns for health and efficiency.
        
        Args:
            communication_data: Dict with 'agents' and 'messages' keys
            
        Returns:
            CommunicationFeatures with network analysis
        """
        logger.info("Analyzing communication topology")
        
        # Build communication graph
        G = self._build_communication_graph(communication_data)
        
        # Basic metrics
        total_agents = G.number_of_nodes()
        active_connections = G.number_of_edges()
        
        # Network density
        if total_agents > 1:
            max_edges = total_agents * (total_agents - 1) / 2
            network_density = active_connections / max_edges
        else:
            network_density = 0.0
            
        # Component analysis
        components = list(nx.connected_components(G.to_undirected()))
        num_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0
        
        # Find isolated agents
        isolated_agents = [node for node in G.nodes() if G.degree(node) == 0]
        
        # Identify special agents
        hub_agents = self._identify_hubs(G)
        bridge_agents = self._identify_bridges(G)
        overloaded_agents = self._identify_overloaded(G, communication_data)
        
        # Calculate health metrics
        imbalance = self._calculate_communication_imbalance(G)
        fragmentation = self._calculate_fragmentation_score(num_components, total_agents)
        overall_health = self._calculate_communication_health(
            network_density, imbalance, fragmentation, len(overloaded_agents)
        )
        
        # Build features
        features = CommunicationFeatures(
            timestamp=time.time(),
            total_agents=total_agents,
            active_connections=active_connections,
            network_density=float(network_density),
            num_components=num_components,
            largest_component_size=largest_component_size,
            isolated_agents=isolated_agents,
            hub_agents=hub_agents,
            bridge_agents=bridge_agents,
            overloaded_agents=overloaded_agents,
            communication_imbalance=float(imbalance),
            fragmentation_score=float(fragmentation),
            overall_health=float(overall_health)
        )
        
        # Update history
        self.communication_history.append(features)
        
        # Check for anomalies
        anomalies = await self.anomaly_detector.check_communications(features)
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} communication anomalies")
            
        return features
        
    def _build_communication_graph(self, communication_data: Dict[str, Any]) -> nx.Graph:
        """Build undirected graph from communication data."""
        G = nx.Graph()
        
        agents = communication_data.get("agents", [])
        messages = communication_data.get("messages", [])
        
        # Add nodes
        for agent in agents:
            agent_id = agent.get("id", f"agent_{len(G.nodes)}")
            G.add_node(agent_id, **agent)
            
        # Add edges weighted by message frequency
        edge_weights = defaultdict(int)
        for msg in messages:
            source = msg.get("source")
            target = msg.get("target")
            if source and target:
                edge_key = tuple(sorted([source, target]))
                edge_weights[edge_key] += 1
                
        # Create edges
        for (source, target), weight in edge_weights.items():
            G.add_edge(source, target, weight=weight)
            
        return G
        
    def _identify_hubs(self, G: nx.Graph) -> List[str]:
        """Identify hub agents with high degree centrality."""
        if G.number_of_nodes() == 0:
            return []
            
        degree_centrality = nx.degree_centrality(G)
        
        # Find nodes with significantly high degree
        mean_degree = np.mean(list(degree_centrality.values()))
        std_degree = np.std(list(degree_centrality.values()))
        threshold = mean_degree + 2 * std_degree
        
        hubs = [node for node, deg in degree_centrality.items() if deg > threshold]
        return sorted(hubs, key=lambda x: degree_centrality[x], reverse=True)
        
    def _identify_bridges(self, G: nx.Graph) -> List[str]:
        """Identify agents that bridge different components."""
        if G.number_of_nodes() < 3:
            return []
            
        # Find articulation points (removal disconnects graph)
        bridges = list(nx.articulation_points(G))
        
        # Sort by betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        bridges.sort(key=lambda x: betweenness.get(x, 0), reverse=True)
        
        return bridges
        
    def _identify_overloaded(self, G: nx.Graph, 
                           communication_data: Dict[str, Any]) -> List[str]:
        """Identify overloaded agents based on message volume."""
        message_counts = defaultdict(int)
        
        for msg in communication_data.get("messages", []):
            source = msg.get("source")
            target = msg.get("target")
            if source:
                message_counts[source] += 1
            if target:
                message_counts[target] += 1
                
        # Find overloaded
        overloaded = []
        for agent, count in message_counts.items():
            if count > self.overload_threshold * G.degree(agent):
                overloaded.append(agent)
                
        return sorted(overloaded, key=lambda x: message_counts[x], reverse=True)
        
    def _calculate_communication_imbalance(self, G: nx.Graph) -> float:
        """Calculate load imbalance across agents."""
        if G.number_of_nodes() < 2:
            return 0.0
            
        degrees = [G.degree(node) for node in G.nodes()]
        if not degrees or max(degrees) == 0:
            return 0.0
            
        # Coefficient of variation
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        cv = std_degree / (mean_degree + 1e-6)
        
        # Normalize to 0-1
        return min(cv / 2.0, 1.0)
        
    def _calculate_fragmentation_score(self, num_components: int, 
                                     total_agents: int) -> float:
        """Calculate how fragmented the communication network is."""
        if total_agents <= 1:
            return 0.0
            
        # Ideal is 1 component
        fragmentation = (num_components - 1) / total_agents
        return min(fragmentation, 1.0)
        
    def _calculate_communication_health(self, density: float, imbalance: float,
                                      fragmentation: float, 
                                      num_overloaded: int) -> float:
        """Calculate overall communication health score."""
        # Target density around 0.3-0.5 for healthy communication
        if density < 0.1:
            density_score = density * 10  # Too sparse
        elif density > 0.7:
            density_score = 1.0 - (density - 0.7) * 3.33  # Too dense
        else:
            density_score = 1.0  # Good range
            
        # Combine factors
        health = (
            0.3 * density_score +
            0.3 * (1.0 - imbalance) +
            0.2 * (1.0 - fragmentation) +
            0.2 * (1.0 - min(num_overloaded / 10.0, 1.0))
        )
        
        return float(max(0, min(health, 1.0)))
        
    # ==================== Risk Assessment ====================
    
    def get_bottlenecks(self, workflow_id: Optional[str] = None) -> List[str]:
        """Get current bottleneck agents."""
        bottlenecks = set()
        
        if workflow_id:
            # Specific workflow
            history = self.workflow_history.get(workflow_id, [])
            if history:
                latest = history[-1]
                bottlenecks.update(latest.bottleneck_agents)
        else:
            # All workflows
            for wf_history in self.workflow_history.values():
                if wf_history:
                    bottlenecks.update(wf_history[-1].bottleneck_agents)
                    
        return sorted(bottlenecks)
        
    def score_risk(self, workflow_id: str) -> float:
        """Get current risk score for workflow (0-1)."""
        history = self.workflow_history.get(workflow_id, [])
        if not history:
            return 0.0
            
        latest = history[-1]
        return latest.failure_risk
        
    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        # Check communication health
        if self.communication_history:
            comm_health = self.communication_history[-1].overall_health
        else:
            comm_health = 1.0
            
        # Check workflow risks
        max_risk = 0.0
        for history in self.workflow_history.values():
            if history:
                max_risk = max(max_risk, history[-1].failure_risk)
                
        # Combine
        overall_health = comm_health * (1.0 - max_risk)
        
        if overall_health > 0.8:
            return HealthStatus.EXCELLENT
        elif overall_health > 0.6:
            return HealthStatus.GOOD
        elif overall_health > 0.4:
            return HealthStatus.FAIR
        elif overall_health > 0.2:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
            
    async def get_features(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get latest features for workflow."""
        history = self.workflow_history.get(workflow_id, [])
        if not history:
            return None
            
        return history[-1].to_dict()


# ==================== Anomaly Detection ====================

class TopologicalAnomalyDetector:
    """Detect anomalies in topology evolution."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        self.anomaly_counter = 0
        
    async def check_workflow(self, features: WorkflowFeatures) -> List[TopologicalAnomaly]:
        """Check for workflow anomalies."""
        anomalies = []
        
        # Check for sudden bottleneck formation
        if features.bottleneck_score > 0.8:
            anomalies.append(TopologicalAnomaly(
                anomaly_id=f"anomaly_{self.anomaly_counter}",
                detected_at=time.time(),
                anomaly_type="severe_bottleneck",
                severity=features.bottleneck_score,
                affected_agents=features.bottleneck_agents[:3],
                description=f"Severe bottleneck detected with score {features.bottleneck_score:.2f}",
                recommended_action="Redistribute load from bottleneck agents"
            ))
            self.anomaly_counter += 1
            
        # Check for cycle formation
        if features.has_cycles:
            anomalies.append(TopologicalAnomaly(
                anomaly_id=f"anomaly_{self.anomaly_counter}",
                detected_at=time.time(),
                anomaly_type="cycle_detected",
                severity=0.7,
                affected_agents=features.critical_path_agents[:5],
                description="Circular dependencies detected in workflow",
                recommended_action="Remove circular dependencies to prevent deadlocks"
            ))
            self.anomaly_counter += 1
            
        # Check for instability
        if features.stability_index < 0.3:
            anomalies.append(TopologicalAnomaly(
                anomaly_id=f"anomaly_{self.anomaly_counter}",
                detected_at=time.time(),
                anomaly_type="topology_unstable",
                severity=0.6,
                affected_agents=[],
                description=f"Workflow topology is unstable (stability={features.stability_index:.2f})",
                recommended_action="Investigate rapid topology changes"
            ))
            self.anomaly_counter += 1
            
        return anomalies
        
    async def check_communications(self, features: CommunicationFeatures) -> List[TopologicalAnomaly]:
        """Check for communication anomalies."""
        anomalies = []
        
        # Check for fragmentation
        if features.fragmentation_score > 0.5:
            anomalies.append(TopologicalAnomaly(
                anomaly_id=f"anomaly_{self.anomaly_counter}",
                detected_at=time.time(),
                anomaly_type="network_fragmentation",
                severity=features.fragmentation_score,
                affected_agents=features.isolated_agents,
                description=f"Communication network is fragmented ({features.num_components} components)",
                recommended_action="Establish communication bridges between components"
            ))
            self.anomaly_counter += 1
            
        # Check for overloaded agents
        if len(features.overloaded_agents) > 0:
            anomalies.append(TopologicalAnomaly(
                anomaly_id=f"anomaly_{self.anomaly_counter}",
                detected_at=time.time(),
                anomaly_type="agent_overload",
                severity=min(len(features.overloaded_agents) / 5.0, 1.0),
                affected_agents=features.overloaded_agents[:5],
                description=f"{len(features.overloaded_agents)} agents are overloaded",
                recommended_action="Redistribute communication load"
            ))
            self.anomaly_counter += 1
            
        # Check for critical health
        if features.overall_health < 0.3:
            anomalies.append(TopologicalAnomaly(
                anomaly_id=f"anomaly_{self.anomaly_counter}",
                detected_at=time.time(),
                anomaly_type="critical_health",
                severity=0.9,
                affected_agents=features.hub_agents + features.bridge_agents,
                description=f"Communication health is critical ({features.overall_health:.2f})",
                recommended_action="Immediate intervention required"
            ))
            self.anomaly_counter += 1
            
        return anomalies


# Export main classes
__all__ = [
    "WorkflowFeatures",
    "CommunicationFeatures", 
    "TopologicalAnomaly",
    "HealthStatus",
    "AgentTopologyAnalyzer",
    "TopologicalAnomalyDetector"
]