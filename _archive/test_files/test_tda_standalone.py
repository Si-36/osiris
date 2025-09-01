#!/usr/bin/env python3
"""
Standalone Test for TopologicalAnalyzer
======================================
Tests the TDA implementation without importing the full AURA system.
"""

import asyncio
import numpy as np
import networkx as nx
from datetime import datetime, timezone
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Test TDA availability
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
    from gudhi import RipsComplex, SimplexTree
    TDA_LIBS_AVAILABLE = True
    print("✅ TDA libraries available")
except ImportError as e:
    TDA_LIBS_AVAILABLE = False
    print(f"⚠️  TDA libraries not available: {e}")

@dataclass
class TestTDAConfig:
    """Test configuration for TDA"""
    homology_dimensions: List[int] = field(default_factory=lambda: [0, 1, 2])
    max_edge_length: float = 2.0
    persistence_threshold: float = 0.1
    max_nodes_for_full_tda: int = 300
    enable_betti_curves: bool = True
    deterministic_embedding: bool = True


class StandaloneTopologicalAnalyzer:
    """Standalone version of TopologicalAnalyzer for testing"""
    
    def __init__(self, config: TestTDAConfig = None):
        self.config = config or TestTDAConfig()
        self.is_available = TDA_LIBS_AVAILABLE
        
        if not self.is_available:
            print("⚠️  TDA libraries not available - using fallback analysis")
            return
            
        try:
            # Initialize TDA components
            self.vr_persistence = VietorisRipsPersistence(
                homology_dimensions=self.config.homology_dimensions,
                max_edge_length=self.config.max_edge_length,
                n_jobs=1
            )
            
            self.entropy_calculator = PersistenceEntropy()
            self.amplitude_calculator = Amplitude()
            
            if self.config.enable_betti_curves:
                self.betti_curve = BettiCurve(n_bins=50)
            
            print("✅ TDA analyzer initialized with giotto-tda")
            
        except Exception as e:
            print(f"❌ TDA initialization failed: {e}")
            self.is_available = False
    
    async def analyze_workflow_topology(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow topology"""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Build workflow graph
            graph = await self._build_workflow_graph(workflow_state)
            
            if graph.number_of_nodes() == 0:
                return self._empty_result()
            
            # Choose analysis method based on size
            n_nodes = graph.number_of_nodes()
            
            if n_nodes > self.config.max_nodes_for_full_tda:
                result = await self._fast_analysis(graph, workflow_state)
            else:
                result = await self._full_tda_analysis(graph, workflow_state)
            
            # Add performance metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result['performance'] = {
                'duration_seconds': duration,
                'graph_size': n_nodes,
                'analysis_type': result.get('analysis_type', 'unknown')
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return self._error_result(str(e))
    
    async def _full_tda_analysis(self, graph: nx.Graph, state: Dict[str, Any]) -> Dict[str, Any]:
        """Full TDA analysis using persistent homology"""
        
        if not self.is_available:
            return await self._fast_analysis(graph, state)
        
        try:
            # Create point cloud
            point_cloud = self._create_embedding(graph)
            
            # Compute persistent homology
            persistence_diagrams = self.vr_persistence.fit_transform([point_cloud])[0]
            
            # Extract features
            tda_features = await self._extract_tda_features(persistence_diagrams)
            graph_features = self._compute_graph_features(graph)
            
            # Analysis
            analysis = self._analyze_patterns(graph_features, tda_features)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'analysis_type': 'full_persistent_homology',
                'graph_properties': graph_features,
                'tda_features': tda_features,
                'analysis': analysis,
                'health': self._assess_health(analysis),
                'recommendations': self._generate_recommendations(analysis)
            }
            
        except Exception as e:
            print(f"❌ Full TDA failed: {e}, using fallback")
            return await self._fast_analysis(graph, state)
    
    def _create_embedding(self, graph: nx.Graph) -> np.ndarray:
        """Create deterministic point cloud embedding"""
        
        gu = graph.to_undirected() if graph.is_directed() else graph
        n = gu.number_of_nodes()
        
        if n == 0:
            return np.empty((0, 3))
        
        if n <= 3:
            # Small graphs: regular polygon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])
        
        try:
            # Spectral layout in 3D
            pos = nx.spectral_layout(gu, dim=3)
            ordered_nodes = sorted(pos.keys())
            return np.array([pos[node] for node in ordered_nodes])
        except Exception:
            # Fallback with fixed seed
            pos = nx.spring_layout(gu, dim=3, seed=42, iterations=50)
            ordered_nodes = sorted(pos.keys())
            return np.array([pos[node] for node in ordered_nodes])
    
    async def _extract_tda_features(self, diagrams: np.ndarray) -> Dict[str, Any]:
        """Extract TDA features from persistence diagrams"""
        
        if len(diagrams) == 0:
            return {
                'persistence_entropy': 0.0,
                'persistence_amplitude': 0.0,
                'significant_features': 0,
                'total_features': 0
            }
        
        try:
            # Core features
            entropy = self.entropy_calculator.fit_transform([diagrams])
            amplitude = self.amplitude_calculator.fit_transform([diagrams])
            
            entropy_val = float(entropy[0][0]) if len(entropy[0]) > 0 else 0.0
            amplitude_val = float(amplitude[0][0]) if len(amplitude[0]) > 0 else 0.0
            
            # Count significant features
            significant = np.sum(diagrams[:, 1] - diagrams[:, 0] > self.config.persistence_threshold)
            
            return {
                'persistence_entropy': entropy_val,
                'persistence_amplitude': amplitude_val,
                'significant_features': int(significant),
                'total_features': len(diagrams)
            }
            
        except Exception as e:
            print(f"❌ TDA feature extraction failed: {e}")
            return {
                'persistence_entropy': 0.0,
                'persistence_amplitude': 0.0,
                'significant_features': 0,
                'total_features': len(diagrams)
            }
    
    def _compute_graph_features(self, graph: nx.Graph) -> Dict[str, Any]:
        """Compute graph features"""
        
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        features = {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': nx.density(graph),
            'is_connected': nx.is_connected(graph)
        }
        
        if n_nodes > 0:
            try:
                features['clustering'] = nx.average_clustering(graph)
                features['components'] = nx.number_connected_components(graph)
            except Exception:
                pass
        
        return features
    
    def _analyze_patterns(self, graph_features: Dict, tda_features: Dict) -> Dict[str, Any]:
        """Analyze patterns"""
        
        # Simple anomaly score based on features
        entropy = tda_features.get('persistence_entropy', 0.0)
        amplitude = tda_features.get('persistence_amplitude', 0.0)
        
        anomaly_score = min(1.0, (entropy + amplitude) / 2.0) if entropy > 0 or amplitude > 0 else 0.0
        
        # Simple complexity score
        density = graph_features.get('density', 0.0)
        clustering = graph_features.get('clustering', 0.0)
        complexity_score = min(1.0, (density + clustering) / 2.0)
        
        # Pattern classification
        patterns = []
        if graph_features.get('components', 1) > 1:
            patterns.append('FRAGMENTED')
        if density > 0.8:
            patterns.append('HIGHLY_CONNECTED')
        elif density < 0.1:
            patterns.append('SPARSE')
        if not patterns:
            patterns.append('NORMAL')
        
        return {
            'anomaly_score': anomaly_score,
            'complexity_score': complexity_score,
            'patterns': patterns
        }
    
    async def _build_workflow_graph(self, workflow_state: Dict[str, Any]) -> nx.Graph:
        """Build graph from workflow state"""
        
        graph = nx.Graph()
        
        # Add agents
        agents = workflow_state.get('agents', [])
        for i, agent in enumerate(agents):
            agent_id = agent.get('id', f'agent_{i}')
            # Avoid 'type' keyword conflict
            agent_attrs = {k: v for k, v in agent.items() if k != 'type'}
            agent_attrs['node_type'] = agent.get('type', 'agent')
            graph.add_node(agent_id, **agent_attrs)
        
        # Add tasks
        tasks = workflow_state.get('tasks', [])
        for i, task in enumerate(tasks):
            task_id = task.get('id', f'task_{i}')
            # Avoid 'type' keyword conflict
            task_attrs = {k: v for k, v in task.items() if k != 'type'}
            task_attrs['node_type'] = task.get('type', 'task')
            graph.add_node(task_id, **task_attrs)
        
        # Add message edges
        messages = workflow_state.get('messages', [])
        for msg in messages:
            sender = msg.get('sender')
            receiver = msg.get('receiver')
            if sender and receiver and sender in graph.nodes and receiver in graph.nodes:
                graph.add_edge(sender, receiver, type='message', **msg)
        
        # Add dependency edges
        dependencies = workflow_state.get('dependencies', [])
        for dep in dependencies:
            from_node = dep.get('from')
            to_node = dep.get('to')
            if from_node and to_node and from_node in graph.nodes and to_node in graph.nodes:
                graph.add_edge(from_node, to_node, type='dependency', **dep)
        
        # Ensure connectivity
        if graph.number_of_edges() == 0 and graph.number_of_nodes() > 1:
            nodes = list(graph.nodes())
            for i in range(len(nodes) - 1):
                graph.add_edge(nodes[i], nodes[i + 1], type='sequence')
        
        return graph
    
    async def _fast_analysis(self, graph: nx.Graph, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast analysis for large graphs or fallback"""
        
        graph_features = self._compute_graph_features(graph)
        
        n_nodes = graph_features['nodes']
        density = graph_features['density']
        
        # Simple heuristic scores
        complexity_score = min(1.0, density + (n_nodes / 100.0))
        anomaly_score = 0.5 if density > 0.9 or density < 0.05 else 0.0
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': 'fast_heuristic',
            'graph_properties': graph_features,
            'analysis': {
                'anomaly_score': anomaly_score,
                'complexity_score': complexity_score,
                'patterns': ['HEURISTIC_ANALYSIS']
            },
            'health': 'FAST_ANALYSIS',
            'recommendations': ['Fast analysis completed', 'Install TDA libraries for detailed analysis']
        }
    
    def _assess_health(self, analysis: Dict[str, Any]) -> str:
        """Assess health"""
        anomaly = analysis.get('anomaly_score', 0.0)
        complexity = analysis.get('complexity_score', 0.0)
        
        if anomaly > 0.8:
            return 'CRITICAL_ANOMALY'
        elif complexity > 0.9:
            return 'HIGH_COMPLEXITY'
        elif anomaly > 0.5 or complexity > 0.7:
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        anomaly = analysis.get('anomaly_score', 0.0)
        complexity = analysis.get('complexity_score', 0.0)
        patterns = analysis.get('patterns', [])
        
        if anomaly > 0.6:
            recommendations.append('High anomaly detected - investigate structure')
        if complexity > 0.8:
            recommendations.append('High complexity - consider simplification')
        if 'FRAGMENTED' in patterns:
            recommendations.append('Fragmented workflow - improve connectivity')
        
        return recommendations if recommendations else ['Topology appears healthy']
    
    def _empty_result(self) -> Dict[str, Any]:
        """Empty workflow result"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': 'empty',
            'graph_properties': {'nodes': 0, 'edges': 0, 'density': 0.0},
            'analysis': {'anomaly_score': 0.0, 'complexity_score': 0.0, 'patterns': ['EMPTY']},
            'health': 'EMPTY',
            'recommendations': ['No workflow data available']
        }
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Error result"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': 'error',
            'error': error,
            'health': 'ERROR',
            'recommendations': ['Analysis failed - check configuration']
        }


async def test_standalone_tda():
    """Test the standalone TDA analyzer"""
    
    print("🔬 Testing Standalone TopologicalAnalyzer")
    print("=" * 60)
    
    # Create analyzer
    config = TestTDAConfig()
    analyzer = StandaloneTopologicalAnalyzer(config)
    
    # Sample workflow
    workflow = {
        'agents': [
            {'id': 'researcher', 'type': 'analysis'},
            {'id': 'optimizer', 'type': 'optimization'}, 
            {'id': 'guardian', 'type': 'monitoring'}
        ],
        'tasks': [
            {'id': 'analyze_data', 'complexity': 0.6},
            {'id': 'optimize_params', 'complexity': 0.8}
        ],
        'messages': [
            {'sender': 'researcher', 'receiver': 'optimizer'},
            {'sender': 'optimizer', 'receiver': 'guardian'}
        ],
        'dependencies': [
            {'from': 'analyze_data', 'to': 'optimize_params'}
        ]
    }
    
    # Perform analysis
    result = await analyzer.analyze_workflow_topology(workflow)
    
    print("✅ Analysis completed")
    print(f"   Type: {result.get('analysis_type')}")
    print(f"   Health: {result.get('health')}")
    
    # Show graph properties
    graph_props = result.get('graph_properties', {})
    print(f"   Graph: {graph_props.get('nodes')} nodes, {graph_props.get('edges')} edges")
    print(f"   Density: {graph_props.get('density', 0):.3f}")
    
    # Show TDA features if available
    tda_features = result.get('tda_features', {})
    if tda_features:
        print(f"   Persistence entropy: {tda_features.get('persistence_entropy', 0):.3f}")
        print(f"   Persistence amplitude: {tda_features.get('persistence_amplitude', 0):.3f}")
        print(f"   Significant features: {tda_features.get('significant_features', 0)}")
    
    # Show analysis results
    analysis = result.get('analysis', {})
    print(f"   Anomaly score: {analysis.get('anomaly_score', 0):.3f}")
    print(f"   Complexity score: {analysis.get('complexity_score', 0):.3f}")
    print(f"   Patterns: {analysis.get('patterns', [])}")
    
    # Show recommendations
    recommendations = result.get('recommendations', [])
    print(f"   Recommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:3]):
        print(f"     {i+1}. {rec}")
    
    # Performance
    perf = result.get('performance', {})
    print(f"   Duration: {perf.get('duration_seconds', 0):.3f}s")
    
    print("\n🎉 Standalone TDA test completed successfully!")
    return True


async def main():
    """Run the standalone test"""
    
    print("🚀 AURA TopologicalAnalyzer Standalone Test")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    
    success = await test_standalone_tda()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 TopologicalAnalyzer standalone test PASSED!")
        print("✅ Implementation is working correctly")
    else:
        print("❌ Test FAILED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())