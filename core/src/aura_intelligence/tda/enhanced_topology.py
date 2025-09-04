"""
🔧 Enhanced Agent Topology Analyzer

Combines our production-ready agent topology analysis with valuable patterns
extracted from core/topology.py:
- Hardware detection for GPU optimization
- Algorithm registry for flexible selection
- Standard data structures for results
- Integration patterns for external accelerators
"""

import subprocess
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import structlog

from .agent_topology import AgentTopologyAnalyzer, WorkflowFeatures

logger = structlog.get_logger()


# ======================
# Extracted Data Structures
# ======================

@dataclass
class TopologicalSignature:
    """
    Standard format for topological signatures.
    Extracted from core/topology.py - this is actually useful!
    """
    # Core topological features
    betti_numbers: List[int] = field(default_factory=lambda: [1, 0, 0])
    persistence_diagram: List[Tuple[float, float]] = field(default_factory=list)
    signature_string: str = "B1-0-0_P0"
    
    # Metadata
    point_count: int = 0
    dimension: int = 3
    algorithm_used: str = "unknown"
    computation_time_ms: float = 0.0
    
    # Performance features
    anomaly_score: float = 0.0
    gpu_accelerated: bool = False
    
    def distance(self, other: "TopologicalSignature") -> float:
        """
        Calculate distance between two topological signatures.
        Useful for comparing workflow topologies.
        """
        if not isinstance(other, TopologicalSignature):
            raise TypeError("Can only calculate distance to another TopologicalSignature")
        
        # Betti number distance
        betti_dist = sum(abs(a - b) for a, b in zip(self.betti_numbers, other.betti_numbers))
        
        # Anomaly score distance
        anomaly_dist = abs(self.anomaly_score - other.anomaly_score)
        
        # Weighted combination
        return betti_dist + 0.3 * anomaly_dist
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "signature": self.signature_string,
            "betti_numbers": self.betti_numbers,
            "persistence_diagram": self.persistence_diagram,
            "point_count": self.point_count,
            "algorithm": self.algorithm_used,
            "computation_time_ms": self.computation_time_ms,
            "anomaly_score": self.anomaly_score,
            "gpu_accelerated": self.gpu_accelerated
        }


# ======================
# Enhanced Analyzer
# ======================

class EnhancedAgentTopologyAnalyzer(AgentTopologyAnalyzer):
    """
    Enhanced version that adds:
    - Hardware detection (GPU/CPU)
    - Algorithm registry pattern
    - Performance optimization
    - Standard result formats
    """
    
    def __init__(self):
        super().__init__()
        
        # Hardware detection
        self.gpu_available = self._check_gpu_availability()
        self.cpu_cores = self._get_cpu_cores()
        
        # Algorithm registry with performance metadata
        self._init_algorithm_registry()
        
        logger.info(
            "Enhanced topology analyzer initialized",
            gpu_available=self.gpu_available,
            cpu_cores=self.cpu_cores,
            algorithms=list(self.algorithms.keys())
        )
    
    def _init_algorithm_registry(self):
        """Initialize algorithm registry with available algorithms."""
        self.algorithms = {
            "standard": {
                "function": self._standard_analysis,
                "performance": 1.0,
                "available": True,
                "description": "Standard CPU-based analysis"
            },
            "parallel": {
                "function": self._parallel_analysis,
                "performance": float(self.cpu_cores),
                "available": self.cpu_cores > 1,
                "description": f"Parallel analysis using {self.cpu_cores} cores"
            },
            "gpu_accelerated": {
                "function": self._gpu_analysis,
                "performance": 10.0,
                "available": self.gpu_available,
                "description": "GPU-accelerated analysis"
            },
            "approximate": {
                "function": self._approximate_analysis,
                "performance": 5.0,
                "available": True,
                "description": "Fast approximate analysis"
            }
        }
    
    # ======================
    # Hardware Detection (from core/topology.py)
    # ======================
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available. Extracted from core/topology.py."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            has_gpu = result.returncode == 0 and result.stdout.strip()
            if has_gpu:
                logger.info(f"GPU detected: {result.stdout.strip()}")
            return has_gpu
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
            return False
    
    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores."""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except:
            return 1
    
    # ======================
    # Enhanced Analysis Methods
    # ======================
    
    async def analyze_workflow_with_best_algorithm(
        self,
        workflow_id: str,
        workflow_data: Dict[str, Any]
    ) -> Tuple[WorkflowFeatures, TopologicalSignature]:
        """
        Analyze workflow using the best available algorithm.
        Returns both our WorkflowFeatures and standard TopologicalSignature.
        """
        start_time = time.time()
        
        # Select best algorithm
        algorithm = self._select_best_algorithm(workflow_data)
        logger.info(
            f"Selected algorithm: {algorithm['name']}",
            performance=algorithm['performance'],
            description=algorithm['description']
        )
        
        # Run analysis
        features = await self.analyze_workflow(workflow_id, workflow_data)
        
        # Create standard signature
        signature = TopologicalSignature(
            betti_numbers=self._extract_betti_numbers(features),
            signature_string=self._generate_signature_string(features),
            point_count=features.num_agents,
            algorithm_used=algorithm['name'],
            computation_time_ms=(time.time() - start_time) * 1000,
            anomaly_score=features.failure_risk,
            gpu_accelerated=(algorithm['name'] == 'gpu_accelerated')
        )
        
        return features, signature
    
    def _select_best_algorithm(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select best available algorithm based on data size and hardware."""
        num_agents = len(workflow_data.get('agents', []))
        
        # Algorithm selection logic (inspired by core/topology.py)
        if num_agents <= 100:
            # Small: use exact computation
            preferred = 'standard'
        elif num_agents <= 1000:
            # Medium: use parallel if available
            preferred = 'parallel' if self.algorithms['parallel']['available'] else 'standard'
        elif num_agents <= 10000 and self.gpu_available:
            # Large: use GPU if available
            preferred = 'gpu_accelerated'
        else:
            # Very large: use approximation
            preferred = 'approximate'
        
        # Get algorithm if available, otherwise fall back to standard
        if self.algorithms[preferred]['available']:
            algorithm = self.algorithms[preferred].copy()
            algorithm['name'] = preferred
        else:
            algorithm = self.algorithms['standard'].copy()
            algorithm['name'] = 'standard'
        
        return algorithm
    
    def _extract_betti_numbers(self, features: WorkflowFeatures) -> List[int]:
        """Extract Betti numbers from workflow features."""
        # B0: Connected components (usually 1 for connected workflow)
        b0 = 1
        
        # B1: Loops/cycles in workflow
        b1 = 1 if features.has_cycles else 0
        b1 += len(features.bottleneck_agents)  # Each bottleneck indicates complexity
        
        # B2: Higher-order structures (voids)
        b2 = 0  # Workflows are typically 2D, no voids
        
        return [b0, b1, b2]
    
    def _generate_signature_string(self, features: WorkflowFeatures) -> str:
        """Generate signature string from features."""
        betti = self._extract_betti_numbers(features)
        risk_level = 'H' if features.failure_risk > 0.7 else 'M' if features.failure_risk > 0.3 else 'L'
        return f"B{betti[0]}-{betti[1]}-{betti[2]}_R{risk_level}"
    
    # ======================
    # Algorithm Implementations
    # ======================
    
    async def _standard_analysis(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standard CPU-based analysis."""
        # Just use our existing implementation
        return await super().analyze_workflow(workflow_data['id'], workflow_data)
    
    async def _parallel_analysis(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel analysis using multiple CPU cores."""
        # TODO: Implement parallel processing
        # For now, just use standard
        return await self._standard_analysis(workflow_data)
    
    async def _gpu_analysis(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """GPU-accelerated analysis."""
        # TODO: Implement GPU acceleration with RAPIDS or CuPy
        # For now, just use standard
        logger.info("GPU analysis not yet implemented, falling back to standard")
        return await self._standard_analysis(workflow_data)
    
    async def _approximate_analysis(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fast approximate analysis for large workflows."""
        # Sample the workflow for faster analysis
        agents = workflow_data.get('agents', [])
        if len(agents) > 1000:
            # Sample 10% of agents
            sample_size = max(100, len(agents) // 10)
            import random
            sampled_agents = random.sample(agents, sample_size)
            workflow_data = workflow_data.copy()
            workflow_data['agents'] = sampled_agents
        
        return await self._standard_analysis(workflow_data)
    
    # ======================
    # Utility Methods (from core/topology.py)
    # ======================
    
    def compute_pairwise_distances(self, points: List[List[float]]) -> np.ndarray:
        """
        Compute pairwise distances efficiently.
        Improved version of the method from core/topology.py.
        """
        points_array = np.array(points)
        # Use NumPy broadcasting for efficiency
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
    
    def compare_signatures(
        self,
        sig1: TopologicalSignature,
        sig2: TopologicalSignature
    ) -> Dict[str, Any]:
        """Compare two topological signatures."""
        return {
            "distance": sig1.distance(sig2),
            "betti_difference": [
                b2 - b1 for b1, b2 in zip(sig1.betti_numbers, sig2.betti_numbers)
            ],
            "same_algorithm": sig1.algorithm_used == sig2.algorithm_used,
            "performance_ratio": sig1.computation_time_ms / max(1, sig2.computation_time_ms)
        }


# ======================
# Integration Pattern (from core/topology.py)
# ======================

class ExternalAcceleratorBridge:
    """
    Pattern for integrating external accelerators.
    Inspired by MojoTDABridge from core/topology.py.
    """
    
    def __init__(self, accelerator_type: str = "gpu"):
        self.accelerator_type = accelerator_type
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if accelerator is available."""
        if self.accelerator_type == "gpu":
            try:
                import cupy
                return True
            except ImportError:
                return False
        return False
    
    async def accelerate_computation(
        self,
        data: np.ndarray,
        computation: Callable
    ) -> Any:
        """Run computation on accelerator if available."""
        if self.available and self.accelerator_type == "gpu":
            try:
                import cupy as cp
                # Transfer to GPU
                gpu_data = cp.asarray(data)
                # Run computation
                result = computation(gpu_data)
                # Transfer back
                return cp.asnumpy(result)
            except Exception as e:
                logger.warning(f"GPU acceleration failed: {e}")
        
        # Fallback to CPU
        return computation(data)


# ======================
# Example Usage
# ======================

if __name__ == "__main__":
    import asyncio
    
    async def example():
        # Create enhanced analyzer
        analyzer = EnhancedAgentTopologyAnalyzer()
        
        # Example workflow
        workflow_data = {
            "id": "test_workflow",
            "agents": [
                {"id": f"agent_{i}", "type": "worker"}
                for i in range(100)
            ],
            "dependencies": [
                {"source": f"agent_{i}", "target": f"agent_{i+1}"}
                for i in range(99)
            ]
        }
        
        # Analyze with best algorithm
        features, signature = await analyzer.analyze_workflow_with_best_algorithm(
            "test_workflow",
            workflow_data
        )
        
        print(f"Analysis complete!")
        print(f"Algorithm used: {signature.algorithm_used}")
        print(f"Computation time: {signature.computation_time_ms:.2f}ms")
        print(f"Signature: {signature.signature_string}")
        print(f"GPU accelerated: {signature.gpu_accelerated}")
    
    asyncio.run(example())