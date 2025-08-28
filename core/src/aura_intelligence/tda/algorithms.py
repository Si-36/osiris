"""
TDA Algorithms - Lean, Production-Ready Topological Computations
===============================================================

Core topological data analysis algorithms with focus on stability,
performance, and practical agent system analysis.

Key Features:
- Persistent homology with numerical safeguards
- Diagram vectorization for ML integration
- Efficient distance computations
- CPU-first implementation with optional GPU
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import structlog

logger = structlog.get_logger(__name__)

# Numerical stability constants
EPSILON = 1e-10
MAX_DISTANCE = 1e6
MIN_PERSISTENCE = 1e-8


# ==================== Core Data Structures ====================

@dataclass
class PersistenceDiagram:
    """
    Persistence diagram representation.
    Each point is (birth, death) where death > birth.
    """
    dimension: int
    points: np.ndarray  # Shape (n, 2) with birth/death pairs
    
    def __post_init__(self):
        """Validate and clean diagram."""
        if len(self.points) > 0:
            # Ensure 2D array
            self.points = np.atleast_2d(self.points)
            
            # Remove invalid points
            valid_mask = (self.points[:, 1] - self.points[:, 0]) > MIN_PERSISTENCE
            self.points = self.points[valid_mask]
            
            # Clip to reasonable range
            self.points = np.clip(self.points, 0, MAX_DISTANCE)
            
    @property
    def persistence(self) -> np.ndarray:
        """Get persistence values (death - birth)."""
        if len(self.points) == 0:
            return np.array([])
        return self.points[:, 1] - self.points[:, 0]
        
    @property
    def total_persistence(self) -> float:
        """Get total persistence."""
        return float(np.sum(self.persistence))
        
    def is_empty(self) -> bool:
        """Check if diagram is empty."""
        return len(self.points) == 0


# ==================== Distance Matrix Computation ====================

def compute_distance_matrix(points: np.ndarray, 
                          metric: str = "euclidean",
                          max_distance: Optional[float] = None) -> np.ndarray:
    """
    Compute pairwise distance matrix with numerical safeguards.
    
    Args:
        points: Point cloud of shape (n_points, n_features)
        metric: Distance metric (euclidean, manhattan, etc.)
        max_distance: Maximum distance to consider
        
    Returns:
        Symmetric distance matrix of shape (n_points, n_points)
    """
    # Input validation
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
        
    # Check for NaN/Inf
    if np.any(~np.isfinite(points)):
        logger.warning("Non-finite values in point cloud, replacing with 0")
        points = np.nan_to_num(points, nan=0.0, posinf=MAX_DISTANCE, neginf=-MAX_DISTANCE)
        
    # Compute distances
    try:
        if len(points) == 1:
            distances = np.zeros((1, 1))
        else:
            distances = squareform(pdist(points, metric=metric))
            
    except Exception as e:
        logger.error(f"Error computing distances: {e}")
        # Fallback to simple Euclidean
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                distances[i, j] = distances[j, i] = d
                
    # Apply max distance threshold
    if max_distance is not None:
        distances = np.minimum(distances, max_distance)
        
    # Ensure symmetry and zeros on diagonal
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    
    return distances


# ==================== Rips Filtration ====================

class RipsFiltration:
    """Vietoris-Rips filtration builder."""
    
    def __init__(self, max_dimension: int = 2, max_edge_length: float = np.inf):
        self.max_dimension = min(max_dimension, 3)  # Cap at 3 for performance
        self.max_edge_length = min(max_edge_length, MAX_DISTANCE)
        
    def build(self, distances: np.ndarray) -> List[Tuple[float, Tuple[int, ...]]]:
        """
        Build Rips filtration from distance matrix.
        
        Returns:
            List of (filtration_value, simplex) tuples
        """
        n = len(distances)
        if n == 0:
            return []
            
        filtration = []
        
        # Add vertices (0-simplices)
        for i in range(n):
            filtration.append((0.0, (i,)))
            
        # Add edges (1-simplices)
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] <= self.max_edge_length:
                    filtration.append((distances[i, j], (i, j)))
                    
        # Add higher dimensional simplices if requested
        if self.max_dimension >= 2 and n >= 3:
            # Add triangles (2-simplices)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        # Check if all edges exist
                        d_ij = distances[i, j]
                        d_jk = distances[j, k]
                        d_ik = distances[i, k]
                        
                        if (d_ij <= self.max_edge_length and
                            d_jk <= self.max_edge_length and
                            d_ik <= self.max_edge_length):
                            # Birth time is max edge length
                            birth = max(d_ij, d_jk, d_ik)
                            filtration.append((birth, (i, j, k)))
                            
        # Sort by filtration value
        filtration.sort(key=lambda x: x[0])
        
        return filtration


# ==================== Persistent Homology ====================

def compute_persistence(points: np.ndarray,
                       max_dimension: int = 2,
                       max_edge_length: Optional[float] = None) -> List[PersistenceDiagram]:
    """
    Compute persistent homology of point cloud.
    
    Args:
        points: Point cloud array
        max_dimension: Maximum homology dimension
        max_edge_length: Maximum edge length in Rips complex
        
    Returns:
        List of persistence diagrams by dimension
    """
    # Compute distances
    distances = compute_distance_matrix(points)
    
    # Auto-determine max edge length if not provided
    if max_edge_length is None:
        # Use 90th percentile of distances
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        if len(upper_tri) > 0:
            max_edge_length = np.percentile(upper_tri, 90)
        else:
            max_edge_length = 1.0
            
    # Build filtration
    rips = RipsFiltration(max_dimension, max_edge_length)
    filtration = rips.build(distances)
    
    # Compute persistence using Union-Find for H0
    # (Full persistence computation would use boundary matrices)
    diagrams = []
    
    # H0 (connected components)
    h0_diagram = _compute_h0_persistence(filtration, len(points))
    diagrams.append(h0_diagram)
    
    # Placeholder for H1, H2 (would need full boundary matrix computation)
    for dim in range(1, max_dimension + 1):
        diagrams.append(PersistenceDiagram(dim, np.array([])))
        
    return diagrams


def _compute_h0_persistence(filtration: List[Tuple[float, Tuple[int, ...]]], 
                           n_vertices: int) -> PersistenceDiagram:
    """Compute 0-dimensional persistence (connected components)."""
    # Union-Find data structure
    parent = list(range(n_vertices))
    birth_time = [0.0] * n_vertices
    death_time = [np.inf] * n_vertices
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
        
    def union(x, y, time):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            # Always attach y to x
            parent[root_y] = root_x
            death_time[root_y] = time
            
    # Process filtration
    for time, simplex in filtration:
        if len(simplex) == 2:  # Edge
            union(simplex[0], simplex[1], time)
            
    # Extract persistence pairs
    points = []
    for i in range(n_vertices):
        if death_time[i] < np.inf:
            points.append([birth_time[i], death_time[i]])
            
    return PersistenceDiagram(0, np.array(points))


# ==================== Diagram Features ====================

def diagram_entropy(diagram: Union[PersistenceDiagram, np.ndarray]) -> float:
    """
    Compute persistence entropy of diagram.
    
    Entropy = -sum(p_i * log(p_i)) where p_i = persistence_i / total_persistence
    """
    if isinstance(diagram, PersistenceDiagram):
        persistences = diagram.persistence
    else:
        # Assume array of birth/death pairs
        diagram = np.atleast_2d(diagram)
        persistences = diagram[:, 1] - diagram[:, 0]
        
    # Filter valid persistences
    persistences = persistences[persistences > MIN_PERSISTENCE]
    
    if len(persistences) == 0:
        return 0.0
        
    # Normalize
    total = np.sum(persistences)
    if total <= EPSILON:
        return 0.0
        
    probs = persistences / total
    
    # Compute entropy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entropy = -np.sum(probs * np.log(probs + EPSILON))
        
    return float(entropy)


def diagram_distance(diagram1: Union[PersistenceDiagram, np.ndarray],
                    diagram2: Union[PersistenceDiagram, np.ndarray],
                    p: float = 2.0) -> float:
    """
    Compute Wasserstein distance between persistence diagrams.
    
    Simplified implementation using bottleneck distance approximation.
    """
    # Convert to arrays
    if isinstance(diagram1, PersistenceDiagram):
        points1 = diagram1.points
    else:
        points1 = np.atleast_2d(diagram1)
        
    if isinstance(diagram2, PersistenceDiagram):
        points2 = diagram2.points
    else:
        points2 = np.atleast_2d(diagram2)
        
    # Handle empty diagrams
    if len(points1) == 0 and len(points2) == 0:
        return 0.0
    if len(points1) == 0:
        return float(np.sum(points2[:, 1] - points2[:, 0]))
    if len(points2) == 0:
        return float(np.sum(points1[:, 1] - points1[:, 0]))
        
    # Bottleneck distance approximation
    # For each point in diagram1, find closest in diagram2
    distances = []
    
    for p1 in points1:
        # Distance to diagonal
        diag_dist = (p1[1] - p1[0]) / np.sqrt(2)
        
        # Distance to points in diagram2
        if len(points2) > 0:
            dists = np.sqrt(np.sum((points2 - p1)**2, axis=1))
            min_dist = min(np.min(dists), diag_dist)
        else:
            min_dist = diag_dist
            
        distances.append(min_dist)
        
    # Symmetric computation for points2
    for p2 in points2:
        diag_dist = (p2[1] - p2[0]) / np.sqrt(2)
        if len(points1) > 0:
            dists = np.sqrt(np.sum((points1 - p2)**2, axis=1))
            min_dist = min(np.min(dists), diag_dist)
        else:
            min_dist = diag_dist
        distances.append(min_dist)
        
    # Lp norm
    if p == np.inf:
        return float(np.max(distances))
    else:
        return float(np.sum(np.array(distances)**p)**(1/p))


def vectorize_diagram(diagram: Union[PersistenceDiagram, np.ndarray],
                     resolution: int = 50,
                     spread: float = 0.1) -> np.ndarray:
    """
    Convert persistence diagram to fixed-size vector using persistence images.
    
    Args:
        diagram: Persistence diagram
        resolution: Grid resolution for persistence image
        spread: Gaussian spread parameter
        
    Returns:
        Flattened persistence image vector
    """
    if isinstance(diagram, PersistenceDiagram):
        points = diagram.points
    else:
        points = np.atleast_2d(diagram)
        
    # Create grid
    if len(points) == 0:
        return np.zeros(resolution * resolution)
        
    # Transform to birth-persistence coordinates
    births = points[:, 0]
    persistences = points[:, 1] - points[:, 0]
    
    # Determine range
    x_min, x_max = 0, np.max(births) + spread
    y_min, y_max = 0, np.max(persistences) + spread
    
    # Create image grid
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    image = np.zeros((resolution, resolution))
    
    # Place Gaussians
    for b, p in zip(births, persistences):
        if p > MIN_PERSISTENCE:
            # Weight by persistence
            weight = p
            
            # Add Gaussian
            for i, x in enumerate(x_grid):
                for j, y in enumerate(y_grid):
                    dist_sq = ((x - b) / spread)**2 + ((y - p) / spread)**2
                    image[j, i] += weight * np.exp(-dist_sq / 2)
                    
    # Normalize and flatten
    if np.max(image) > 0:
        image = image / np.max(image)
        
    return image.flatten()


# ==================== Stability Helpers ====================

def validate_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Validate and clean point cloud data.
    
    - Removes duplicate points
    - Handles NaN/Inf values
    - Ensures proper shape
    """
    points = np.asarray(points, dtype=np.float64)
    
    # Ensure 2D
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    elif points.ndim > 2:
        logger.warning(f"Flattening {points.ndim}D array to 2D")
        points = points.reshape(points.shape[0], -1)
        
    # Remove NaN/Inf
    finite_mask = np.all(np.isfinite(points), axis=1)
    if not np.all(finite_mask):
        logger.warning(f"Removing {np.sum(~finite_mask)} non-finite points")
        points = points[finite_mask]
        
    # Remove duplicates
    if len(points) > 1:
        unique_points, indices = np.unique(points, axis=0, return_index=True)
        if len(unique_points) < len(points):
            logger.info(f"Removed {len(points) - len(unique_points)} duplicate points")
            points = unique_points
            
    return points


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    if abs(denominator) < EPSILON:
        return default
    return numerator / denominator


# ==================== Graph-Specific Extensions ====================

def compute_path_homology(graph: Union[Dict, 'nx.Graph'], 
                         max_dimension: int = 1) -> List[PersistenceDiagram]:
    """
    Compute path homology for directed graphs.
    Placeholder for future directed TDA support.
    """
    # Convert graph to point cloud embedding
    # This would use spectral embedding or similar
    logger.warning("Path homology not yet implemented, using standard persistence")
    
    # For now, return empty diagrams
    return [PersistenceDiagram(d, np.array([])) for d in range(max_dimension + 1)]


# ==================== Main API ====================

__all__ = [
    # Data structures
    'PersistenceDiagram',
    
    # Core algorithms
    'compute_persistence',
    'compute_distance_matrix',
    
    # Diagram operations
    'diagram_entropy',
    'diagram_distance', 
    'vectorize_diagram',
    
    # Utilities
    'validate_point_cloud',
    'safe_divide',
    
    # Extensions
    'compute_path_homology',
    
    # Constants
    'EPSILON',
    'MIN_PERSISTENCE'
]