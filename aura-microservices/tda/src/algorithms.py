"""
TDA Algorithms - Extracted from AURA Intelligence
Production-grade topological data analysis algorithms
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to numpy

# Optional TDA libraries
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


@dataclass
class PersistenceDiagram:
    """Persistence diagram representation."""
    dimension: int
    birth_death_pairs: np.ndarray  # Shape: (n_features, 2)
    persistence: np.ndarray  # death - birth
    computation_time_ms: float
    algorithm: str
    metadata: Dict[str, Any]


class BaseTDAAlgorithm(ABC):
    """Base class for all TDA algorithms."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
    
    @abstractmethod
    def compute_persistence(
        self,
        data: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None
    ) -> List[PersistenceDiagram]:
        """Compute persistence diagrams."""
        pass
    
    def compute_persistence_image(
        self,
        diagram: PersistenceDiagram,
        resolution: Tuple[int, int] = (50, 50),
        sigma: float = 0.1
    ) -> np.ndarray:
        """Convert persistence diagram to persistence image."""
        if len(diagram.birth_death_pairs) == 0:
            return np.zeros(resolution)
        
        # Create grid
        x_min, x_max = 0, diagram.birth_death_pairs[:, 1].max()
        y_min, y_max = 0, diagram.persistence.max()
        
        x_grid = np.linspace(x_min, x_max, resolution[0])
        y_grid = np.linspace(y_min, y_max, resolution[1])
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        
        # Compute persistence image
        image = np.zeros(resolution)
        for i, (birth, death) in enumerate(diagram.birth_death_pairs):
            persistence = death - birth
            weight = persistence  # Can use other weighting schemes
            
            # Add Gaussian centered at (birth, persistence)
            gaussian = weight * np.exp(
                -((grid_x - birth)**2 + (grid_y - persistence)**2) / (2 * sigma**2)
            )
            image += gaussian
        
        return image


class VietorisRipsAlgorithm(BaseTDAAlgorithm):
    """Vietoris-Rips complex for persistence computation."""
    
    def compute_persistence(
        self,
        data: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None
    ) -> List[PersistenceDiagram]:
        """Compute Vietoris-Rips persistence."""
        start_time = time.time()
        
        if RIPSER_AVAILABLE:
            # Use ripser for fast computation
            result = ripser.ripser(
                data,
                maxdim=max_dimension,
                thresh=max_edge_length if max_edge_length else np.inf
            )
            
            diagrams = []
            for dim in range(max_dimension + 1):
                if f'dgms' in result and dim < len(result['dgms']):
                    dgm = result['dgms'][dim]
                    if len(dgm) > 0:
                        persistence = dgm[:, 1] - dgm[:, 0]
                        diagrams.append(PersistenceDiagram(
                            dimension=dim,
                            birth_death_pairs=dgm,
                            persistence=persistence,
                            computation_time_ms=(time.time() - start_time) * 1000,
                            algorithm="ripser",
                            metadata={"n_points": len(data)}
                        ))
            
            return diagrams
        
        elif GUDHI_AVAILABLE:
            # Use GUDHI as fallback
            rips_complex = gudhi.RipsComplex(points=data, max_edge_length=max_edge_length or float('inf'))
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
            simplex_tree.compute_persistence()
            
            diagrams = []
            for dim in range(max_dimension + 1):
                dgm = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(dgm) > 0:
                    dgm = np.array(dgm)
                    persistence = dgm[:, 1] - dgm[:, 0]
                    # Filter out infinite persistence
                    finite_mask = np.isfinite(dgm[:, 1])
                    dgm = dgm[finite_mask]
                    persistence = persistence[finite_mask]
                    
                    if len(dgm) > 0:
                        diagrams.append(PersistenceDiagram(
                            dimension=dim,
                            birth_death_pairs=dgm,
                            persistence=persistence,
                            computation_time_ms=(time.time() - start_time) * 1000,
                            algorithm="gudhi",
                            metadata={"n_points": len(data)}
                        ))
            
            return diagrams
        
        else:
            # Basic implementation without external libraries
            return self._compute_basic_persistence(data, max_dimension, max_edge_length, start_time)
    
    def _compute_basic_persistence(
        self,
        data: np.ndarray,
        max_dimension: int,
        max_edge_length: Optional[float],
        start_time: float
    ) -> List[PersistenceDiagram]:
        """Basic persistence computation without external libraries."""
        # Compute pairwise distances
        n_points = len(data)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(data[i] - data[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Find connected components (0-dimensional features)
        if max_edge_length is None:
            max_edge_length = distances.max()
        
        # Simple union-find for connected components
        parent = list(range(n_points))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        # Track when components merge
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distances[i, j] <= max_edge_length:
                    edges.append((distances[i, j], i, j))
        
        edges.sort()
        
        birth_death_0d = []
        components = set(range(n_points))
        
        for dist, i, j in edges:
            if union(i, j):
                # Component death
                birth_death_0d.append([0, dist])
                components.discard(find(i))
        
        # Remaining components have infinite persistence
        for _ in components:
            birth_death_0d.append([0, max_edge_length])
        
        if birth_death_0d:
            dgm_0d = np.array(birth_death_0d)
            persistence_0d = dgm_0d[:, 1] - dgm_0d[:, 0]
            
            return [PersistenceDiagram(
                dimension=0,
                birth_death_pairs=dgm_0d,
                persistence=persistence_0d,
                computation_time_ms=(time.time() - start_time) * 1000,
                algorithm="basic",
                metadata={"n_points": n_points}
            )]
        
        return []


class AlphaComplexAlgorithm(BaseTDAAlgorithm):
    """Alpha complex for more accurate persistence in low dimensions."""
    
    def compute_persistence(
        self,
        data: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None
    ) -> List[PersistenceDiagram]:
        """Compute Alpha complex persistence."""
        start_time = time.time()
        
        if GUDHI_AVAILABLE and data.shape[1] <= 3:  # Alpha complex works best in 2D/3D
            alpha_complex = gudhi.AlphaComplex(points=data)
            simplex_tree = alpha_complex.create_simplex_tree()
            simplex_tree.compute_persistence()
            
            diagrams = []
            for dim in range(min(max_dimension + 1, data.shape[1])):
                dgm = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(dgm) > 0:
                    dgm = np.array(dgm)
                    persistence = dgm[:, 1] - dgm[:, 0]
                    # Filter out infinite persistence
                    finite_mask = np.isfinite(dgm[:, 1])
                    dgm = dgm[finite_mask]
                    persistence = persistence[finite_mask]
                    
                    if len(dgm) > 0:
                        diagrams.append(PersistenceDiagram(
                            dimension=dim,
                            birth_death_pairs=dgm,
                            persistence=persistence,
                            computation_time_ms=(time.time() - start_time) * 1000,
                            algorithm="alpha",
                            metadata={"n_points": len(data)}
                        ))
            
            return diagrams
        
        else:
            # Fall back to Vietoris-Rips
            return VietorisRipsAlgorithm(self.use_gpu).compute_persistence(
                data, max_dimension, max_edge_length
            )


class WitnessComplexAlgorithm(BaseTDAAlgorithm):
    """Witness complex for large datasets."""
    
    def __init__(self, use_gpu: bool = False, n_landmarks: int = 100):
        super().__init__(use_gpu)
        self.n_landmarks = n_landmarks
    
    def compute_persistence(
        self,
        data: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None
    ) -> List[PersistenceDiagram]:
        """Compute Witness complex persistence."""
        start_time = time.time()
        
        if GUDHI_AVAILABLE:
            # Select landmarks
            n_points = len(data)
            n_landmarks = min(self.n_landmarks, n_points)
            landmark_indices = np.random.choice(n_points, n_landmarks, replace=False)
            landmarks = data[landmark_indices]
            
            # Compute witness complex
            witness_complex = gudhi.EuclideanWitnessComplex(
                witnesses=data,
                landmarks=landmarks
            )
            simplex_tree = witness_complex.create_simplex_tree(
                max_alpha_square=max_edge_length**2 if max_edge_length else float('inf'),
                limit_dimension=max_dimension
            )
            simplex_tree.compute_persistence()
            
            diagrams = []
            for dim in range(max_dimension + 1):
                dgm = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(dgm) > 0:
                    dgm = np.array(dgm)
                    persistence = dgm[:, 1] - dgm[:, 0]
                    # Filter out infinite persistence
                    finite_mask = np.isfinite(dgm[:, 1])
                    dgm = dgm[finite_mask]
                    persistence = persistence[finite_mask]
                    
                    if len(dgm) > 0:
                        diagrams.append(PersistenceDiagram(
                            dimension=dim,
                            birth_death_pairs=dgm,
                            persistence=persistence,
                            computation_time_ms=(time.time() - start_time) * 1000,
                            algorithm="witness",
                            metadata={
                                "n_points": n_points,
                                "n_landmarks": n_landmarks
                            }
                        ))
            
            return diagrams
        
        else:
            # Fall back to Vietoris-Rips with subsampling
            subsample_size = min(self.n_landmarks, len(data))
            subsample_indices = np.random.choice(len(data), subsample_size, replace=False)
            subsampled_data = data[subsample_indices]
            
            return VietorisRipsAlgorithm(self.use_gpu).compute_persistence(
                subsampled_data, max_dimension, max_edge_length
            )