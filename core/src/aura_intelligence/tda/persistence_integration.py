"""
TDA Persistence Integration
==========================
Integrates Topological Data Analysis with causal persistence
"""

import numpy as np
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import structlog

from ..persistence.causal_state_manager import (
    get_causal_manager,
    StateType,
    CausalContext
)

logger = structlog.get_logger(__name__)

class PersistentTDA:
    """TDA with causal persistence capabilities"""
    
    def __init__(self, tda_id: str = "tda_analyzer"):
        self.tda_id = tda_id
        self._persistence_manager = None
        self._computation_history = []
        self._topology_cache = {}
        
    async def _ensure_persistence(self):
        """Ensure persistence manager is initialized"""
        if self._persistence_manager is None:
            self._persistence_manager = await get_causal_manager()
    
    async def save_persistence_diagram(self,
                                     diagram: np.ndarray,
                                     data_source: str,
                                     computation_params: Dict[str, Any],
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save persistence diagram with causal tracking"""
        await self._ensure_persistence()
        
        # Analyze diagram for causes/effects
        causes = self._extract_diagram_causes(diagram, computation_params)
        effects = self._predict_topology_effects(diagram, metadata)
        
        # Create causal context
        causal_context = CausalContext(
            causes=causes,
            effects=effects,
            confidence=self._calculate_topology_confidence(diagram),
            energy_cost=computation_params.get("computation_time", 0.1)
        )
        
        # Prepare diagram data
        diagram_data = {
            "diagram": diagram.tolist(),  # Convert to list for JSON
            "shape": diagram.shape,
            "data_source": data_source,
            "computation_params": computation_params,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "topological_features": self._extract_features(diagram)
        }
        
        # Generate embedding from diagram
        embedding = self._generate_diagram_embedding(diagram)
        
        # Save to persistence
        state_id = await self._persistence_manager.save_state(
            StateType.TDA_DIAGRAM,
            f"{self.tda_id}_{data_source}",
            diagram_data,
            causal_context=causal_context,
            embedding=embedding.tolist() if embedding is not None else None
        )
        
        # Update history
        self._computation_history.append({
            "state_id": state_id,
            "data_source": data_source,
            "timestamp": datetime.now(),
            "features": diagram_data["topological_features"]
        })
        
        # Cache for fast access
        self._topology_cache[data_source] = diagram
        
        logger.info("Saved persistence diagram with causality",
                   state_id=state_id,
                   data_source=data_source,
                   features=len(diagram))
        
        return state_id
    
    async def load_persistence_diagram(self,
                                     data_source: str,
                                     compute_fn: Optional[callable] = None) -> Optional[Dict[str, Any]]:
        """Load persistence diagram with optional computation"""
        await self._ensure_persistence()
        
        # Check cache first
        if data_source in self._topology_cache and not compute_fn:
            logger.info("Retrieved diagram from cache", data_source=data_source)
            return {
                "diagram": self._topology_cache[data_source],
                "cached": True
            }
        
        # Load from persistence
        diagram_data = await self._persistence_manager.load_state(
            StateType.TDA_DIAGRAM,
            f"{self.tda_id}_{data_source}",
            compute_on_retrieval=compute_fn
        )
        
        if diagram_data:
            # Reconstruct numpy array
            diagram_data["diagram"] = np.array(diagram_data["diagram"])
            
            # Update cache
            self._topology_cache[data_source] = diagram_data["diagram"]
            
            logger.info("Loaded persistence diagram",
                       data_source=data_source,
                       shape=diagram_data["diagram"].shape)
        
        return diagram_data
    
    async def compare_topologies(self,
                               source1: str,
                               source2: str,
                               metric: str = "bottleneck") -> Dict[str, Any]:
        """Compare two topological spaces with causal tracking"""
        await self._ensure_persistence()
        
        # Load both diagrams
        diagram1 = await self.load_persistence_diagram(source1)
        diagram2 = await self.load_persistence_diagram(source2)
        
        if not diagram1 or not diagram2:
            logger.warning("Could not load diagrams for comparison")
            return {"error": "Missing diagrams"}
        
        # Compute distance
        distance = self._compute_diagram_distance(
            diagram1["diagram"],
            diagram2["diagram"],
            metric
        )
        
        # Analyze differences
        analysis = {
            "source1": source1,
            "source2": source2,
            "metric": metric,
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance),  # Convert to similarity
            "timestamp": datetime.now().isoformat(),
            "feature_comparison": self._compare_features(
                diagram1.get("topological_features", {}),
                diagram2.get("topological_features", {})
            )
        }
        
        # Save comparison with causality
        causal_context = CausalContext(
            causes=["topology_comparison", f"metric_{metric}"],
            effects=self._predict_comparison_effects(distance),
            confidence=0.9
        )
        
        await self._persistence_manager.save_state(
            StateType.COMPONENT_STATE,
            f"{self.tda_id}_comparison_{source1}_{source2}",
            analysis,
            causal_context=causal_context
        )
        
        return analysis
    
    async def track_topology_evolution(self,
                                     data_source: str,
                                     time_window: int = 10) -> List[Dict[str, Any]]:
        """Track how topology evolves over time"""
        await self._ensure_persistence()
        
        # Get historical states
        states = await self._persistence_manager.get_state_history(
            StateType.TDA_DIAGRAM,
            f"{self.tda_id}_{data_source}"
        )
        
        # Analyze evolution
        evolution = []
        for i in range(1, len(states)):
            prev_state = states[i-1]
            curr_state = states[i]
            
            # Reconstruct diagrams
            prev_diagram = np.array(prev_state["data"]["diagram"])
            curr_diagram = np.array(curr_state["data"]["diagram"])
            
            # Compute changes
            distance = self._compute_diagram_distance(prev_diagram, curr_diagram)
            
            evolution.append({
                "timestamp": curr_state["timestamp"],
                "distance_from_previous": distance,
                "features_gained": self._find_new_features(prev_diagram, curr_diagram),
                "features_lost": self._find_lost_features(prev_diagram, curr_diagram),
                "causes": curr_state.get("causal_context", {}).get("causes", [])
            })
        
        return evolution[-time_window:]
    
    async def create_topology_experiment(self,
                                       base_source: str,
                                       experiment_name: str,
                                       transformation: callable) -> str:
        """Create experimental branch for topology exploration"""
        await self._ensure_persistence()
        
        # Create branch
        branch_id = await self._persistence_manager.create_branch(
            f"{self.tda_id}_{base_source}",
            experiment_name
        )
        
        # Load base topology
        base_data = await self.load_persistence_diagram(base_source)
        if not base_data:
            logger.error("Could not load base topology")
            return None
        
        # Apply transformation
        transformed_diagram = transformation(base_data["diagram"])
        
        # Save to branch
        causal_context = CausalContext(
            causes=["topology_experiment", f"transformation_{transformation.__name__}"],
            effects=["experimental_topology_created"],
            confidence=0.7  # Experimental, so lower confidence
        )
        
        experiment_data = {
            "diagram": transformed_diagram.tolist(),
            "shape": transformed_diagram.shape,
            "data_source": f"{base_source}_experiment",
            "base_source": base_source,
            "transformation": transformation.__name__,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "topological_features": self._extract_features(transformed_diagram)
        }
        
        state_id = await self._persistence_manager.save_state(
            StateType.TDA_DIAGRAM,
            f"{self.tda_id}_{base_source}_experiment",
            experiment_data,
            causal_context=causal_context,
            branch_id=branch_id
        )
        
        logger.info("Created topology experiment",
                   branch_id=branch_id,
                   experiment=experiment_name,
                   state_id=state_id)
        
        return branch_id
    
    def _extract_diagram_causes(self,
                              diagram: np.ndarray,
                              params: Dict[str, Any]) -> List[str]:
        """Extract what caused this topology"""
        causes = ["tda_computation"]
        
        # Parameter-based causes
        if params.get("max_dimension", 1) > 1:
            causes.append("high_dimensional_analysis")
        
        if params.get("max_edge_length", float('inf')) < 1.0:
            causes.append("fine_scale_analysis")
        
        # Data characteristics
        if len(diagram) > 100:
            causes.append("complex_topology")
        
        if len(diagram) == 0:
            causes.append("trivial_topology")
        
        return causes
    
    def _predict_topology_effects(self,
                                diagram: np.ndarray,
                                metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Predict effects of this topology"""
        effects = []
        
        # Structural effects
        if len(diagram) > 50:
            effects.append("high_complexity_detected")
        
        # Find significant features
        if len(diagram) > 0:
            max_persistence = np.max(diagram[:, 1] - diagram[:, 0])
            if max_persistence > 1.0:
                effects.append("significant_topological_feature")
        
        # Domain-specific effects
        if metadata:
            if metadata.get("domain") == "neural_network":
                effects.append("network_structure_analyzed")
            elif metadata.get("domain") == "time_series":
                effects.append("temporal_patterns_detected")
        
        return effects
    
    def _calculate_topology_confidence(self, diagram: np.ndarray) -> float:
        """Calculate confidence in topological analysis"""
        if len(diagram) == 0:
            return 0.5  # Empty diagram, neutral confidence
        
        # Base confidence on persistence values
        persistences = diagram[:, 1] - diagram[:, 0]
        
        # Higher persistence = higher confidence
        avg_persistence = np.mean(persistences)
        confidence = min(0.95, 0.5 + avg_persistence * 0.2)
        
        return float(confidence)
    
    def _extract_features(self, diagram: np.ndarray) -> Dict[str, Any]:
        """Extract topological features from diagram"""
        if len(diagram) == 0:
            return {
                "num_features": 0,
                "max_persistence": 0,
                "total_persistence": 0,
                "avg_birth": 0,
                "avg_death": 0
            }
        
        persistences = diagram[:, 1] - diagram[:, 0]
        
        return {
            "num_features": len(diagram),
            "max_persistence": float(np.max(persistences)),
            "total_persistence": float(np.sum(persistences)),
            "avg_birth": float(np.mean(diagram[:, 0])),
            "avg_death": float(np.mean(diagram[:, 1])),
            "persistence_entropy": float(self._compute_persistence_entropy(persistences))
        }
    
    def _generate_diagram_embedding(self, diagram: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding vector from persistence diagram"""
        if len(diagram) == 0:
            return np.zeros(768)  # Return zero vector for empty diagram
        
        # Simple embedding: statistics + persistence image
        features = []
        
        # Basic statistics
        persistences = diagram[:, 1] - diagram[:, 0]
        features.extend([
            np.mean(persistences),
            np.std(persistences),
            np.max(persistences),
            len(diagram)
        ])
        
        # Persistence image (simplified)
        # In practice, use gudhi or other library
        hist, _ = np.histogram(persistences, bins=20)
        features.extend(hist / np.sum(hist))
        
        # Pad to standard embedding size
        embedding = np.zeros(768)
        embedding[:len(features)] = features
        
        return embedding
    
    def _compute_diagram_distance(self,
                                diag1: np.ndarray,
                                diag2: np.ndarray,
                                metric: str = "bottleneck") -> float:
        """Compute distance between persistence diagrams"""
        # Simplified implementation
        # In practice, use gudhi.bottleneck_distance or similar
        
        if len(diag1) == 0 and len(diag2) == 0:
            return 0.0
        
        if len(diag1) == 0 or len(diag2) == 0:
            return 1.0
        
        # Simple Wasserstein-like distance
        pers1 = diag1[:, 1] - diag1[:, 0]
        pers2 = diag2[:, 1] - diag2[:, 0]
        
        # Match diagrams by persistence
        pers1_sorted = np.sort(pers1)[::-1]
        pers2_sorted = np.sort(pers2)[::-1]
        
        # Pad shorter array
        max_len = max(len(pers1_sorted), len(pers2_sorted))
        pers1_padded = np.pad(pers1_sorted, (0, max_len - len(pers1_sorted)))
        pers2_padded = np.pad(pers2_sorted, (0, max_len - len(pers2_sorted)))
        
        # Compute distance
        if metric == "bottleneck":
            return float(np.max(np.abs(pers1_padded - pers2_padded)))
        else:  # Wasserstein
            return float(np.mean(np.abs(pers1_padded - pers2_padded)))
    
    def _compare_features(self, features1: Dict, features2: Dict) -> Dict[str, Any]:
        """Compare topological features"""
        comparison = {}
        
        for key in features1:
            if key in features2:
                diff = features2[key] - features1[key]
                comparison[f"{key}_diff"] = diff
                comparison[f"{key}_ratio"] = features2[key] / features1[key] if features1[key] != 0 else float('inf')
        
        return comparison
    
    def _predict_comparison_effects(self, distance: float) -> List[str]:
        """Predict effects based on topology comparison"""
        effects = []
        
        if distance < 0.1:
            effects.append("topologies_nearly_identical")
        elif distance < 0.5:
            effects.append("topologies_similar")
        else:
            effects.append("topologies_significantly_different")
        
        return effects
    
    def _find_new_features(self, prev: np.ndarray, curr: np.ndarray) -> int:
        """Count new topological features"""
        # Simplified - in practice use proper matching
        return max(0, len(curr) - len(prev))
    
    def _find_lost_features(self, prev: np.ndarray, curr: np.ndarray) -> int:
        """Count lost topological features"""
        # Simplified - in practice use proper matching
        return max(0, len(prev) - len(curr))
    
    def _compute_persistence_entropy(self, persistences: np.ndarray) -> float:
        """Compute entropy of persistence values"""
        if len(persistences) == 0:
            return 0.0
        
        # Normalize to probabilities
        total = np.sum(persistences)
        if total == 0:
            return 0.0
        
        probs = persistences / total
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)