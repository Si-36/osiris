"""
🏗️ Architect Agent - System Topology Intelligence
================================================

Specializes in:
- Dependency graph analysis with GPU acceleration
- System topology optimization
- Parallel scenario evaluation
- Scalability prediction
- Architecture pattern recommendations
"""

import asyncio
import time
import json
import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import structlog

# Try to import cuGraph for GPU graph processing
try:
    import cugraph
    import cudf
    CUGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("cuGraph not available, using NetworkX")
    CUGRAPH_AVAILABLE = False

from .test_agents import TestAgentBase, TestAgentConfig, Tool, AgentRole
from ..tda.algorithms import compute_persistence_diagram

logger = structlog.get_logger(__name__)


@dataclass
class ArchitectureAnalysis:
    """Result of architecture analysis"""
    system_id: str
    topology_metrics: Dict[str, float]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    scalability_forecast: Dict[str, Any]
    dependency_graph: Optional[nx.Graph] = None
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    patterns_detected: List[str] = field(default_factory=list)


@dataclass
class SystemSpec:
    """System specification"""
    components: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    constraints: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    current_metrics: Dict[str, Any] = field(default_factory=dict)


class GraphTopologyAnalyzer:
    """Analyze system topology using graph theory"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def analyze(self, graph: nx.Graph) -> Dict[str, float]:
        """Analyze graph topology metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = graph.number_of_nodes()
        metrics['num_edges'] = graph.number_of_edges()
        metrics['density'] = nx.density(graph)
        
        # Centrality metrics
        if graph.number_of_nodes() > 0:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
            metrics['max_degree_centrality'] = max(degree_centrality.values())
            
            # Betweenness centrality (for bottleneck detection)
            if graph.number_of_nodes() < 100:  # Expensive for large graphs
                betweenness = nx.betweenness_centrality(graph)
                metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
                metrics['max_betweenness'] = max(betweenness.values())
            
            # Clustering coefficient
            metrics['avg_clustering'] = nx.average_clustering(graph)
            
            # Connected components
            if graph.is_directed():
                num_strongly_connected = nx.number_strongly_connected_components(graph)
                num_weakly_connected = nx.number_weakly_connected_components(graph)
                metrics['strongly_connected_components'] = num_strongly_connected
                metrics['weakly_connected_components'] = num_weakly_connected
            else:
                num_connected = nx.number_connected_components(graph)
                metrics['connected_components'] = num_connected
                
            # Diameter and radius (if connected)
            if graph.is_directed():
                # Check if strongly connected
                if nx.is_strongly_connected(graph):
                    metrics['diameter'] = nx.diameter(graph)
                    metrics['radius'] = nx.radius(graph)
            else:
                if nx.is_connected(graph):
                    metrics['diameter'] = nx.diameter(graph)
                    metrics['radius'] = nx.radius(graph)
                    
        # Topological features for TDA
        if CUGRAPH_AVAILABLE and graph.number_of_nodes() > 10:
            topo_features = self._compute_gpu_topology(graph)
            metrics.update(topo_features)
            
        return metrics
        
    def _compute_gpu_topology(self, graph: nx.Graph) -> Dict[str, float]:
        """Compute topology features on GPU using cuGraph"""
        # Convert to cuGraph
        edge_list = list(graph.edges())
        
        if not edge_list:
            return {}
            
        # Create cuDF DataFrame
        df = cudf.DataFrame(edge_list, columns=['src', 'dst'])
        
        # Create cuGraph graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='src', destination='dst')
        
        # Compute GPU-accelerated metrics
        features = {}
        
        # PageRank
        pagerank = cugraph.pagerank(G)
        features['avg_pagerank'] = float(pagerank['pagerank'].mean())
        features['max_pagerank'] = float(pagerank['pagerank'].max())
        
        # Katz centrality
        try:
            katz = cugraph.katz_centrality(G)
            features['avg_katz'] = float(katz['katz_centrality'].mean())
        except:
            pass
            
        return features
        
    def detect_patterns(self, graph: nx.Graph) -> List[str]:
        """Detect architectural patterns"""
        patterns = []
        
        # Hub-and-spoke pattern
        degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
        if degree_sequence and degree_sequence[0] > len(graph) * 0.3:
            patterns.append("hub_and_spoke")
            
        # Layered architecture
        if self._is_layered(graph):
            patterns.append("layered")
            
        # Microservices pattern
        components = list(nx.connected_components(graph.to_undirected()))
        if len(components) > 5 and all(len(c) < len(graph) * 0.2 for c in components):
            patterns.append("microservices")
            
        # Pipeline pattern
        if self._is_pipeline(graph):
            patterns.append("pipeline")
            
        # Mesh pattern
        if nx.density(graph) > 0.3:
            patterns.append("mesh")
            
        return patterns
        
    def _is_layered(self, graph: nx.Graph) -> bool:
        """Check if graph has layered structure"""
        if not graph.is_directed():
            return False
            
        # Check if DAG
        if not nx.is_directed_acyclic_graph(graph):
            return False
            
        # Check layer structure
        try:
            topo_sort = list(nx.topological_sort(graph))
            
            # Assign layers
            layers = {}
            for node in topo_sort:
                # Node's layer is max of predecessors' layers + 1
                pred_layers = [layers.get(pred, -1) for pred in graph.predecessors(node)]
                layers[node] = max(pred_layers) + 1 if pred_layers else 0
                
            # Check if layers are well-defined
            num_layers = max(layers.values()) + 1
            
            # Layered if we have 3+ layers and reasonable distribution
            return num_layers >= 3 and num_layers < len(graph) * 0.5
            
        except:
            return False
            
    def _is_pipeline(self, graph: nx.Graph) -> bool:
        """Check if graph has pipeline structure"""
        if not graph.is_directed():
            return False
            
        # Pipeline has mostly linear flow
        avg_in_degree = sum(d for n, d in graph.in_degree()) / len(graph)
        avg_out_degree = sum(d for n, d in graph.out_degree()) / len(graph)
        
        # Most nodes should have degree 1 or 2
        return avg_in_degree < 1.5 and avg_out_degree < 1.5


class ParallelScenarioEvaluator:
    """Evaluate architectural scenarios in parallel"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network for scenario scoring
        self.scenario_scorer = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to(self.device)
        
    async def evaluate_scenarios(self,
                               system_spec: SystemSpec,
                               num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """Evaluate multiple architectural scenarios"""
        scenarios = []
        
        # Generate scenario variations
        base_features = self._extract_features(system_spec)
        
        # Parallel evaluation on GPU
        if self.device.type == "cuda":
            scenario_results = await self._gpu_parallel_evaluation(
                base_features,
                num_scenarios
            )
        else:
            scenario_results = await self._cpu_sequential_evaluation(
                base_features,
                num_scenarios
            )
            
        # Rank scenarios
        scenarios = sorted(scenario_results, key=lambda x: x['score'], reverse=True)
        
        return scenarios[:10]  # Return top 10
        
    def _extract_features(self, system_spec: SystemSpec) -> np.ndarray:
        """Extract features from system specification"""
        features = []
        
        # Component features
        features.append(len(system_spec.components))
        features.append(len(system_spec.connections))
        
        # Complexity features
        avg_connections = len(system_spec.connections) / max(len(system_spec.components), 1)
        features.append(avg_connections)
        
        # Constraint features
        features.append(len(system_spec.constraints))
        
        # Requirement features
        features.append(system_spec.requirements.get('min_throughput', 0))
        features.append(system_spec.requirements.get('max_latency', 100))
        features.append(system_spec.requirements.get('availability', 0.99))
        
        # Current performance
        features.append(system_spec.current_metrics.get('cpu_usage', 0))
        features.append(system_spec.current_metrics.get('memory_usage', 0))
        features.append(system_spec.current_metrics.get('response_time', 0))
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0)
            
        return np.array(features[:20], dtype=np.float32)
        
    async def _gpu_parallel_evaluation(self,
                                     base_features: np.ndarray,
                                     num_scenarios: int) -> List[Dict[str, Any]]:
        """Evaluate scenarios in parallel on GPU"""
        # Generate variations
        feature_tensor = torch.tensor(base_features, device=self.device)
        feature_batch = feature_tensor.unsqueeze(0).expand(num_scenarios, -1)
        
        # Add noise for variations
        noise = torch.randn_like(feature_batch) * 0.2
        varied_features = feature_batch + noise
        
        # Score all scenarios at once
        with torch.no_grad():
            scores = self.scenario_scorer(varied_features).squeeze()
            
        # Create scenario descriptions
        scenarios = []
        for i in range(num_scenarios):
            scenario = {
                "id": f"scenario_{i}",
                "features": varied_features[i].cpu().numpy(),
                "score": float(scores[i].item()),
                "changes": self._describe_changes(
                    base_features,
                    varied_features[i].cpu().numpy()
                )
            }
            scenarios.append(scenario)
            
        return scenarios
        
    async def _cpu_sequential_evaluation(self,
                                       base_features: np.ndarray,
                                       num_scenarios: int) -> List[Dict[str, Any]]:
        """Sequential evaluation fallback"""
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate variation
            noise = np.random.randn(len(base_features)) * 0.2
            varied_features = base_features + noise
            
            # Score
            feature_tensor = torch.tensor(varied_features, device=self.device)
            with torch.no_grad():
                score = self.scenario_scorer(feature_tensor.unsqueeze(0)).item()
                
            scenario = {
                "id": f"scenario_{i}",
                "features": varied_features,
                "score": score,
                "changes": self._describe_changes(base_features, varied_features)
            }
            scenarios.append(scenario)
            
        return scenarios
        
    def _describe_changes(self, original: np.ndarray, modified: np.ndarray) -> List[str]:
        """Describe changes between scenarios"""
        changes = []
        
        # Component changes
        if modified[0] > original[0]:
            changes.append(f"Add {int(modified[0] - original[0])} components")
        elif modified[0] < original[0]:
            changes.append(f"Remove {int(original[0] - modified[0])} components")
            
        # Connection changes
        if modified[1] > original[1] * 1.2:
            changes.append("Increase connectivity")
        elif modified[1] < original[1] * 0.8:
            changes.append("Reduce connectivity")
            
        # Performance targets
        if modified[4] > original[4]:
            changes.append(f"Increase throughput target to {modified[4]:.0f}")
            
        if modified[5] < original[5]:
            changes.append(f"Reduce latency target to {modified[5]:.0f}ms")
            
        return changes


class ScalabilityPredictor:
    """Predict system scalability using topology and metrics"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Scalability prediction model
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)  # 3 outputs: linear, sublinear, superlinear
        ).to(self.device)
        
    def predict(self,
               topology_metrics: Dict[str, float],
               scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict scalability characteristics"""
        # Extract features
        features = self._extract_scalability_features(topology_metrics, scenarios)
        
        # Predict
        feature_tensor = torch.tensor(features, device=self.device)
        
        with torch.no_grad():
            predictions = self.predictor(feature_tensor.unsqueeze(0))
            probabilities = torch.softmax(predictions, dim=-1).squeeze()
            
        scalability_types = ["linear", "sublinear", "superlinear"]
        predicted_type = scalability_types[torch.argmax(probabilities).item()]
        
        # Estimate scaling limits
        scaling_limits = self._estimate_limits(topology_metrics, predicted_type)
        
        return {
            "scalability_type": predicted_type,
            "confidence": float(torch.max(probabilities).item()),
            "probabilities": {
                t: float(p.item()) 
                for t, p in zip(scalability_types, probabilities)
            },
            "scaling_limits": scaling_limits,
            "bottleneck_components": self._identify_bottlenecks(topology_metrics),
            "recommendations": self._generate_recommendations(predicted_type, topology_metrics)
        }
        
    def _extract_scalability_features(self,
                                    topology_metrics: Dict[str, float],
                                    scenarios: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for scalability prediction"""
        features = []
        
        # Topology features
        features.extend([
            topology_metrics.get('num_nodes', 0),
            topology_metrics.get('num_edges', 0),
            topology_metrics.get('density', 0),
            topology_metrics.get('avg_degree_centrality', 0),
            topology_metrics.get('max_degree_centrality', 0),
            topology_metrics.get('avg_betweenness', 0),
            topology_metrics.get('max_betweenness', 0),
            topology_metrics.get('avg_clustering', 0),
            topology_metrics.get('connected_components', 1),
            topology_metrics.get('diameter', 0)
        ])
        
        # Scenario features
        if scenarios:
            scenario_scores = [s['score'] for s in scenarios]
            features.extend([
                np.mean(scenario_scores),
                np.std(scenario_scores),
                np.max(scenario_scores),
                np.min(scenario_scores)
            ])
        else:
            features.extend([0, 0, 0, 0])
            
        # Pad to fixed size
        while len(features) < 30:
            features.append(0)
            
        return np.array(features[:30], dtype=np.float32)
        
    def _estimate_limits(self, 
                        topology_metrics: Dict[str, float],
                        scalability_type: str) -> Dict[str, Any]:
        """Estimate scaling limits"""
        base_nodes = topology_metrics.get('num_nodes', 10)
        
        if scalability_type == "linear":
            max_scale = base_nodes * 100
            optimal_scale = base_nodes * 10
        elif scalability_type == "sublinear":
            max_scale = base_nodes * 20
            optimal_scale = base_nodes * 5
        else:  # superlinear
            max_scale = base_nodes * 1000
            optimal_scale = base_nodes * 50
            
        return {
            "current_scale": base_nodes,
            "optimal_scale": optimal_scale,
            "max_scale": max_scale,
            "scaling_factor": max_scale / base_nodes
        }
        
    def _identify_bottlenecks(self, topology_metrics: Dict[str, float]) -> List[str]:
        """Identify potential bottlenecks"""
        bottlenecks = []
        
        # High centrality indicates bottleneck
        if topology_metrics.get('max_betweenness', 0) > 0.5:
            bottlenecks.append("central_coordinator")
            
        # Low connectivity
        if topology_metrics.get('density', 1) < 0.1:
            bottlenecks.append("sparse_connectivity")
            
        # Many components
        if topology_metrics.get('connected_components', 1) > 5:
            bottlenecks.append("fragmented_architecture")
            
        return bottlenecks
        
    def _generate_recommendations(self,
                                scalability_type: str,
                                topology_metrics: Dict[str, float]) -> List[str]:
        """Generate scalability recommendations"""
        recommendations = []
        
        if scalability_type == "sublinear":
            recommendations.append("Consider horizontal partitioning")
            recommendations.append("Implement caching layers")
            recommendations.append("Reduce inter-component dependencies")
            
        if topology_metrics.get('max_betweenness', 0) > 0.5:
            recommendations.append("Distribute central coordinator responsibilities")
            recommendations.append("Implement load balancing")
            
        if topology_metrics.get('density', 1) > 0.5:
            recommendations.append("Consider service mesh for complex interactions")
            recommendations.append("Implement circuit breakers")
            
        return recommendations


class ArchitectAgent(TestAgentBase):
    """
    Specialized agent for system architecture analysis.
    
    Capabilities:
    - Dependency graph analysis
    - System topology optimization
    - Scalability prediction
    - Pattern detection
    - Risk assessment
    """
    
    def __init__(self, agent_id: str = "architect_agent_001", **kwargs):
        config = TestAgentConfig(
            agent_role=AgentRole.OBSERVER,
            specialty="architect",
            target_latency_ms=250.0,  # Higher for complex analysis
            **kwargs
        )
        
        super().__init__(agent_id=agent_id, config=config, **kwargs)
        
        # Initialize specialized components
        self.graph_analyzer = GraphTopologyAnalyzer()
        self.scenario_evaluator = ParallelScenarioEvaluator()
        self.scalability_predictor = ScalabilityPredictor()
        
        # Architecture patterns library
        self.pattern_library = {
            "layered": {
                "description": "Layered architecture with clear separation",
                "pros": ["Separation of concerns", "Easy to understand"],
                "cons": ["Performance overhead", "Rigid structure"]
            },
            "microservices": {
                "description": "Distributed microservices architecture",
                "pros": ["Independent scaling", "Technology diversity"],
                "cons": ["Complexity", "Network overhead"]
            },
            "event_driven": {
                "description": "Event-driven architecture",
                "pros": ["Loose coupling", "Scalability"],
                "cons": ["Eventual consistency", "Debugging complexity"]
            },
            "pipeline": {
                "description": "Pipeline/workflow architecture",
                "pros": ["Clear flow", "Easy to extend"],
                "cons": ["Bottlenecks", "Error propagation"]
            },
            "hub_and_spoke": {
                "description": "Centralized hub architecture",
                "pros": ["Simple routing", "Central control"],
                "cons": ["Single point of failure", "Hub bottleneck"]
            }
        }
        
        # Initialize tools
        self._init_architect_tools()
        
        logger.info("Architect Agent initialized",
                   agent_id=agent_id,
                   cugraph_available=CUGRAPH_AVAILABLE,
                   capabilities=["topology_analysis", "scalability_prediction", 
                               "pattern_detection", "risk_assessment"])
                   
    def _init_architect_tools(self):
        """Initialize architect-specific tools"""
        self.tools = {
            "analyze_topology": Tool(
                name="analyze_topology",
                description="Analyze system topology",
                func=self._tool_analyze_topology
            ),
            "evaluate_scenarios": Tool(
                name="evaluate_scenarios",
                description="Evaluate architectural scenarios",
                func=self._tool_evaluate_scenarios
            ),
            "predict_scalability": Tool(
                name="predict_scalability",
                description="Predict system scalability",
                func=self._tool_predict_scalability
            ),
            "detect_patterns": Tool(
                name="detect_patterns",
                description="Detect architectural patterns",
                func=self._tool_detect_patterns
            ),
            "assess_risks": Tool(
                name="assess_risks",
                description="Assess architectural risks",
                func=self._tool_assess_risks
            )
        }
        
    async def _handle_analyze(self, context: Dict[str, Any]) -> ArchitectureAnalysis:
        """Handle architecture analysis requests"""
        # Extract system specification
        system_data = context.get("original", {})
        
        # Build system specification
        system_spec = SystemSpec(
            components=system_data.get("components", []),
            connections=system_data.get("connections", []),
            constraints=system_data.get("constraints", {}),
            requirements=system_data.get("requirements", {}),
            current_metrics=system_data.get("metrics", {})
        )
        
        # Build dependency graph
        graph = self._build_dependency_graph(system_spec)
        
        # Analyze topology
        topology_metrics = self.graph_analyzer.analyze(graph)
        
        # Detect patterns
        patterns = self.graph_analyzer.detect_patterns(graph)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(graph, topology_metrics)
        
        # Evaluate scenarios
        scenarios = await self.scenario_evaluator.evaluate_scenarios(
            system_spec,
            num_scenarios=50
        )
        
        # Predict scalability
        scalability_forecast = self.scalability_predictor.predict(
            topology_metrics,
            scenarios
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            topology_metrics,
            patterns,
            bottlenecks,
            scalability_forecast
        )
        
        # Assess risks
        risk_assessment = self._assess_architectural_risks(
            system_spec,
            topology_metrics,
            patterns
        )
        
        # Store in memory
        if topology_metrics:
            # Create embedding from topology metrics
            embedding = self._create_topology_embedding(topology_metrics)
            
            await self.shape_memory.store(
                {
                    "type": "architecture_analysis",
                    "components": len(system_spec.components),
                    "patterns": patterns,
                    "scalability": scalability_forecast["scalability_type"],
                    "risks": risk_assessment["overall_risk"]
                },
                embedding=embedding
            )
            
        return ArchitectureAnalysis(
            system_id=f"system_{int(time.time())}",
            topology_metrics=topology_metrics,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            scalability_forecast=scalability_forecast,
            dependency_graph=graph,
            risk_assessment=risk_assessment,
            patterns_detected=patterns
        )
        
    async def _handle_generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle architecture generation requests"""
        requirements = context.get("original", {}).get("requirements", {})
        constraints = context.get("original", {}).get("constraints", {})
        
        # Generate architecture based on requirements
        architecture = await self._generate_architecture(requirements, constraints)
        
        # Analyze generated architecture
        system_spec = SystemSpec(
            components=architecture["components"],
            connections=architecture["connections"],
            constraints=constraints,
            requirements=requirements
        )
        
        analysis = await self._handle_analyze({
            "original": {
                "components": architecture["components"],
                "connections": architecture["connections"],
                "constraints": constraints,
                "requirements": requirements
            }
        })
        
        return {
            "architecture": architecture,
            "analysis": {
                "topology_metrics": analysis.topology_metrics,
                "patterns": analysis.patterns_detected,
                "scalability": analysis.scalability_forecast,
                "risks": analysis.risk_assessment
            }
        }
        
    def _build_dependency_graph(self, system_spec: SystemSpec) -> nx.DiGraph:
        """Build dependency graph from system specification"""
        graph = nx.DiGraph()
        
        # Add nodes for components
        for component in system_spec.components:
            graph.add_node(
                component["id"],
                **component.get("properties", {})
            )
            
        # Add edges for connections
        for connection in system_spec.connections:
            graph.add_edge(
                connection["from"],
                connection["to"],
                **connection.get("properties", {})
            )
            
        return graph
        
    def _identify_bottlenecks(self,
                            graph: nx.Graph,
                            metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # High betweenness centrality nodes
        if graph.number_of_nodes() > 0:
            betweenness = nx.betweenness_centrality(graph)
            
            # Find nodes with high betweenness
            threshold = metrics.get('avg_betweenness', 0) + \
                       2 * np.std(list(betweenness.values()))
                       
            for node, centrality in betweenness.items():
                if centrality > threshold:
                    bottlenecks.append({
                        "component": node,
                        "type": "high_betweenness",
                        "severity": "high" if centrality > threshold * 1.5 else "medium",
                        "metric": centrality,
                        "description": f"Component {node} is on many critical paths"
                    })
                    
        # Single points of failure
        if graph.is_directed():
            # Find articulation points in underlying undirected graph
            undirected = graph.to_undirected()
            articulation_points = list(nx.articulation_points(undirected))
            
            for node in articulation_points:
                bottlenecks.append({
                    "component": node,
                    "type": "single_point_of_failure",
                    "severity": "critical",
                    "description": f"Removing {node} would disconnect the system"
                })
                
        # Resource bottlenecks (if metrics available)
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            if 'cpu_usage' in node_data and node_data['cpu_usage'] > 0.8:
                bottlenecks.append({
                    "component": node,
                    "type": "resource_bottleneck",
                    "resource": "cpu",
                    "severity": "high",
                    "metric": node_data['cpu_usage'],
                    "description": f"Component {node} has high CPU usage"
                })
                
        return bottlenecks
        
    def _generate_recommendations(self,
                                topology_metrics: Dict[str, float],
                                patterns: List[str],
                                bottlenecks: List[Dict[str, Any]],
                                scalability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate architectural recommendations"""
        recommendations = []
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern in self.pattern_library:
                pattern_info = self.pattern_library[pattern]
                
                recommendations.append({
                    "category": "pattern",
                    "pattern": pattern,
                    "description": f"Detected {pattern} pattern",
                    "considerations": pattern_info
                })
                
        # Bottleneck-based recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "high_betweenness":
                recommendations.append({
                    "category": "bottleneck",
                    "component": bottleneck["component"],
                    "action": "distribute_load",
                    "description": f"Distribute load from {bottleneck['component']}",
                    "priority": bottleneck["severity"]
                })
            elif bottleneck["type"] == "single_point_of_failure":
                recommendations.append({
                    "category": "resilience",
                    "component": bottleneck["component"],
                    "action": "add_redundancy",
                    "description": f"Add redundancy for {bottleneck['component']}",
                    "priority": "critical"
                })
                
        # Scalability recommendations
        if scalability["scalability_type"] == "sublinear":
            recommendations.append({
                "category": "scalability",
                "action": "optimize_architecture",
                "description": "Architecture shows sublinear scaling",
                "suggestions": scalability["recommendations"]
            })
            
        # Topology-based recommendations
        if topology_metrics.get('density', 0) > 0.5:
            recommendations.append({
                "category": "complexity",
                "action": "reduce_coupling",
                "description": "High coupling detected between components",
                "metric": topology_metrics['density']
            })
            
        return recommendations
        
    def _assess_architectural_risks(self,
                                  system_spec: SystemSpec,
                                  topology_metrics: Dict[str, float],
                                  patterns: List[str]) -> Dict[str, Any]:
        """Assess architectural risks"""
        risks = {
            "technical": [],
            "operational": [],
            "scalability": [],
            "security": []
        }
        
        # Technical risks
        if topology_metrics.get('connected_components', 1) > 1:
            risks["technical"].append({
                "risk": "disconnected_components",
                "severity": "high",
                "description": "System has disconnected components",
                "mitigation": "Ensure all components are properly integrated"
            })
            
        # Operational risks
        if "hub_and_spoke" in patterns:
            risks["operational"].append({
                "risk": "central_point_of_failure",
                "severity": "critical",
                "description": "Hub component is a single point of failure",
                "mitigation": "Implement hub redundancy and failover"
            })
            
        # Scalability risks
        if topology_metrics.get('diameter', 0) > 10:
            risks["scalability"].append({
                "risk": "high_diameter",
                "severity": "medium",
                "description": "Long communication paths may impact performance",
                "mitigation": "Consider adding shortcuts or caching"
            })
            
        # Security risks
        if topology_metrics.get('density', 0) > 0.7:
            risks["security"].append({
                "risk": "large_attack_surface",
                "severity": "medium",
                "description": "High connectivity increases attack surface",
                "mitigation": "Implement network segmentation and access controls"
            })
            
        # Calculate overall risk
        all_risks = []
        for category, category_risks in risks.items():
            all_risks.extend(category_risks)
            
        if all_risks:
            severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            avg_severity = np.mean([
                severity_scores.get(r["severity"], 2) 
                for r in all_risks
            ])
            
            if avg_severity >= 3:
                overall_risk = "high"
            elif avg_severity >= 2:
                overall_risk = "medium"
            else:
                overall_risk = "low"
        else:
            overall_risk = "low"
            
        return {
            "categories": risks,
            "overall_risk": overall_risk,
            "total_risks": len(all_risks),
            "critical_risks": len([r for r in all_risks if r.get("severity") == "critical"])
        }
        
    def _create_topology_embedding(self, metrics: Dict[str, float]) -> np.ndarray:
        """Create embedding from topology metrics"""
        # Extract key metrics in consistent order
        embedding_values = [
            metrics.get('num_nodes', 0),
            metrics.get('num_edges', 0),
            metrics.get('density', 0),
            metrics.get('avg_degree_centrality', 0),
            metrics.get('max_degree_centrality', 0),
            metrics.get('avg_betweenness', 0),
            metrics.get('max_betweenness', 0),
            metrics.get('avg_clustering', 0),
            metrics.get('connected_components', 1),
            metrics.get('diameter', 0),
            metrics.get('radius', 0),
            metrics.get('avg_pagerank', 0),
            metrics.get('max_pagerank', 0)
        ]
        
        # Normalize and pad to standard size
        embedding = np.array(embedding_values, dtype=np.float32)
        
        # Normalize
        embedding = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-6)
        
        # Pad to 768 dimensions (standard embedding size)
        if len(embedding) < 768:
            padding = np.zeros(768 - len(embedding))
            embedding = np.concatenate([embedding, padding])
            
        return embedding[:768]
        
    async def _generate_architecture(self,
                                   requirements: Dict[str, Any],
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture based on requirements"""
        # Analyze requirements
        throughput = requirements.get('throughput', 1000)
        latency = requirements.get('latency', 100)
        availability = requirements.get('availability', 0.99)
        
        # Select pattern based on requirements
        if throughput > 10000 and latency < 50:
            pattern = "microservices"
        elif availability > 0.999:
            pattern = "event_driven"
        elif latency < 10:
            pattern = "pipeline"
        else:
            pattern = "layered"
            
        # Generate components based on pattern
        if pattern == "microservices":
            components = [
                {"id": "api_gateway", "type": "gateway"},
                {"id": "auth_service", "type": "service"},
                {"id": "user_service", "type": "service"},
                {"id": "data_service", "type": "service"},
                {"id": "cache_service", "type": "cache"},
                {"id": "message_queue", "type": "queue"}
            ]
            
            connections = [
                {"from": "api_gateway", "to": "auth_service"},
                {"from": "api_gateway", "to": "user_service"},
                {"from": "api_gateway", "to": "data_service"},
                {"from": "user_service", "to": "cache_service"},
                {"from": "data_service", "to": "cache_service"},
                {"from": "user_service", "to": "message_queue"},
                {"from": "data_service", "to": "message_queue"}
            ]
            
        elif pattern == "layered":
            components = [
                {"id": "presentation", "type": "layer"},
                {"id": "application", "type": "layer"},
                {"id": "business", "type": "layer"},
                {"id": "data_access", "type": "layer"},
                {"id": "database", "type": "storage"}
            ]
            
            connections = [
                {"from": "presentation", "to": "application"},
                {"from": "application", "to": "business"},
                {"from": "business", "to": "data_access"},
                {"from": "data_access", "to": "database"}
            ]
            
        else:
            # Default simple architecture
            components = [
                {"id": "frontend", "type": "ui"},
                {"id": "backend", "type": "api"},
                {"id": "database", "type": "storage"}
            ]
            
            connections = [
                {"from": "frontend", "to": "backend"},
                {"from": "backend", "to": "database"}
            ]
            
        return {
            "pattern": pattern,
            "components": components,
            "connections": connections,
            "metadata": {
                "generated_for": requirements,
                "constraints_applied": constraints
            }
        }
        
    # Tool implementations
    async def _tool_analyze_topology(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze topology tool"""
        # Build graph from data
        graph = nx.node_link_graph(graph_data) if 'nodes' in graph_data else nx.Graph()
        
        # Analyze
        metrics = self.graph_analyzer.analyze(graph)
        patterns = self.graph_analyzer.detect_patterns(graph)
        
        return {
            "metrics": metrics,
            "patterns": patterns
        }
        
    async def _tool_evaluate_scenarios(self,
                                     system_spec: Dict[str, Any],
                                     num_scenarios: int = 50) -> List[Dict[str, Any]]:
        """Evaluate scenarios tool"""
        spec = SystemSpec(
            components=system_spec.get("components", []),
            connections=system_spec.get("connections", []),
            constraints=system_spec.get("constraints", {}),
            requirements=system_spec.get("requirements", {})
        )
        
        scenarios = await self.scenario_evaluator.evaluate_scenarios(spec, num_scenarios)
        
        return scenarios
        
    async def _tool_predict_scalability(self,
                                      topology_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Predict scalability tool"""
        # Dummy scenarios for prediction
        scenarios = [{"score": np.random.random()} for _ in range(10)]
        
        prediction = self.scalability_predictor.predict(topology_metrics, scenarios)
        
        return prediction
        
    async def _tool_detect_patterns(self, graph_data: Dict[str, Any]) -> List[str]:
        """Detect patterns tool"""
        graph = nx.node_link_graph(graph_data) if 'nodes' in graph_data else nx.Graph()
        
        patterns = self.graph_analyzer.detect_patterns(graph)
        
        return patterns
        
    async def _tool_assess_risks(self,
                               system_spec: Dict[str, Any],
                               topology_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess risks tool"""
        spec = SystemSpec(
            components=system_spec.get("components", []),
            connections=system_spec.get("connections", [])
        )
        
        patterns = []  # Would detect patterns first
        
        risks = self._assess_architectural_risks(spec, topology_metrics, patterns)
        
        return risks


# Factory function
def create_architect_agent(agent_id: Optional[str] = None, **kwargs) -> ArchitectAgent:
    """Create an Architect Agent instance"""
    if agent_id is None:
        agent_id = f"architect_agent_{int(time.time())}"
        
    return ArchitectAgent(agent_id=agent_id, **kwargs)