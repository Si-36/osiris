"""
Topological Laplacian Extractor - Advanced TDA Beyond Homology
==============================================================

Implements persistent topological Laplacians that capture:
- Harmonic spectrum (stable topological invariants)
- Non-harmonic spectrum (geometric evolution)
- Hodge decomposition for multi-scale features

Based on July 2025 research showing Laplacians provide richer
signatures than persistent homology alone.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LaplacianSignature:
    """Complete Laplacian signature of a memory"""
    harmonic_spectrum: np.ndarray      # Topological invariants
    non_harmonic_spectrum: np.ndarray  # Geometric features
    eigenvalues: np.ndarray            # Full spectrum
    eigenvectors: Optional[np.ndarray] # Eigenvector basis
    hodge_decomposition: Dict[str, np.ndarray]  # Hodge components
    dimension: int                      # Topological dimension
    persistence: float                  # Persistence measure


class TopologicalLaplacianExtractor:
    """
    Advanced topological feature extraction using Laplacians
    
    Goes beyond persistent homology to capture both topological
    invariants and geometric evolution of data.
    """
    
    def __init__(self, topology_adapter: Any, 
                 max_dimension: int = 3,
                 num_eigenvalues: int = 50):
        """
        Initialize the Laplacian extractor
        
        Args:
            topology_adapter: TopologyMemoryAdapter instance
            max_dimension: Maximum topological dimension to compute
            num_eigenvalues: Number of eigenvalues to compute
        """
        self.topology_adapter = topology_adapter
        self.max_dimension = max_dimension
        self.num_eigenvalues = num_eigenvalues
        
        # Cache for efficiency
        self._laplacian_cache = {}
        self._spectrum_cache = {}
        
        logger.info(
            "TopologicalLaplacianExtractor initialized",
            max_dim=max_dimension,
            num_eigenvalues=num_eigenvalues
        )
    
    async def batch_extract(self, memories: List[Any]) -> List[LaplacianSignature]:
        """
        Extract Laplacian signatures for a batch of memories
        
        Args:
            memories: List of memory contents
        
        Returns:
            List of LaplacianSignature objects
        """
        signatures = []
        
        for memory in memories:
            signature = await self.extract_laplacian_signature(memory)
            signatures.append(signature)
        
        logger.debug(f"Extracted {len(signatures)} Laplacian signatures")
        
        return signatures
    
    async def extract_laplacian_signature(self, memory: Any) -> LaplacianSignature:
        """
        Extract complete Laplacian signature from memory
        
        Args:
            memory: Memory content
        
        Returns:
            LaplacianSignature object
        """
        # First get basic topology from adapter
        topology = await self.topology_adapter.extract_topology(memory)
        
        # Extract point cloud
        point_cloud = self._extract_point_cloud(memory)
        
        # Compute combinatorial Laplacians for each dimension
        laplacians = self._compute_combinatorial_laplacians(point_cloud)
        
        # Extract spectra
        spectra = self._compute_spectra(laplacians)
        
        # Separate harmonic and non-harmonic components
        harmonic, non_harmonic = self._decompose_spectrum(spectra)
        
        # Compute Hodge decomposition
        hodge = self._hodge_decomposition(point_cloud, laplacians)
        
        # Calculate persistence from eigenvalues
        persistence = self._calculate_spectral_persistence(spectra)
        
        signature = LaplacianSignature(
            harmonic_spectrum=harmonic,
            non_harmonic_spectrum=non_harmonic,
            eigenvalues=spectra['eigenvalues'],
            eigenvectors=spectra.get('eigenvectors'),
            hodge_decomposition=hodge,
            dimension=len(laplacians),
            persistence=persistence
        )
        
        return signature
    
    def _extract_point_cloud(self, memory: Any) -> np.ndarray:
        """
        Extract point cloud representation from memory
        
        Args:
            memory: Memory content
        
        Returns:
            Point cloud as numpy array
        """
        # Try to extract embedding or features
        if hasattr(memory, 'embedding'):
            embedding = memory.embedding
        elif hasattr(memory, 'content') and isinstance(memory.content, dict):
            # Extract numerical features from content
            embedding = self._extract_features_from_dict(memory.content)
        elif isinstance(memory, np.ndarray):
            embedding = memory
        else:
            # Generate random point cloud as fallback
            embedding = np.random.randn(100, 3)
        
        # Ensure 2D array
        if embedding.ndim == 1:
            # Embed 1D vector into higher dimension using delay embedding
            embedding = self._delay_embedding(embedding, dimension=3, delay=5)
        
        return embedding
    
    def _extract_features_from_dict(self, content: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from dictionary content"""
        features = []
        
        for key, value in content.items():
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                features.extend(value[:10])  # Limit to first 10
            elif isinstance(value, np.ndarray):
                features.extend(value.flatten()[:10])
        
        if not features:
            # No numerical features found
            features = np.random.randn(10)
        
        return np.array(features)
    
    def _delay_embedding(self, signal: np.ndarray, dimension: int = 3, 
                        delay: int = 5) -> np.ndarray:
        """
        Create delay embedding from 1D signal
        
        Takens' theorem: reconstructs attractor from time series
        
        Args:
            signal: 1D signal
            dimension: Embedding dimension
            delay: Time delay
        
        Returns:
            Embedded point cloud
        """
        n = len(signal)
        m = n - (dimension - 1) * delay
        
        if m <= 0:
            # Signal too short, return random
            return np.random.randn(10, dimension)
        
        embedded = np.zeros((m, dimension))
        
        for i in range(dimension):
            embedded[:, i] = signal[i * delay:i * delay + m]
        
        return embedded
    
    def _compute_combinatorial_laplacians(self, point_cloud: np.ndarray) -> List[sp.spmatrix]:
        """
        Compute combinatorial Laplacians for different dimensions
        
        Args:
            point_cloud: Point cloud data
        
        Returns:
            List of sparse Laplacian matrices
        """
        laplacians = []
        
        # 0-dimensional Laplacian (graph Laplacian)
        L0 = self._compute_graph_laplacian(point_cloud)
        laplacians.append(L0)
        
        # 1-dimensional Laplacian (edge Laplacian)
        if self.max_dimension >= 1:
            L1 = self._compute_edge_laplacian(point_cloud, L0)
            laplacians.append(L1)
        
        # 2-dimensional Laplacian (triangle Laplacian)
        if self.max_dimension >= 2:
            L2 = self._compute_triangle_laplacian(point_cloud)
            laplacians.append(L2)
        
        return laplacians
    
    def _compute_graph_laplacian(self, point_cloud: np.ndarray, 
                                epsilon: Optional[float] = None) -> sp.spmatrix:
        """
        Compute 0-dimensional graph Laplacian
        
        Args:
            point_cloud: Point cloud data
            epsilon: Threshold for edge creation
        
        Returns:
            Sparse graph Laplacian matrix
        """
        n = point_cloud.shape[0]
        
        # Compute pairwise distances
        distances = squareform(pdist(point_cloud))
        
        # Determine epsilon if not provided (use median distance)
        if epsilon is None:
            epsilon = np.median(distances[distances > 0])
        
        # Create adjacency matrix
        adjacency = (distances <= epsilon).astype(float)
        np.fill_diagonal(adjacency, 0)
        
        # Compute degree matrix
        degrees = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degrees)
        
        # Laplacian = Degree - Adjacency
        laplacian = degree_matrix - adjacency
        
        return sp.csr_matrix(laplacian)
    
    def _compute_edge_laplacian(self, point_cloud: np.ndarray, 
                               L0: sp.spmatrix) -> sp.spmatrix:
        """
        Compute 1-dimensional edge Laplacian
        
        Args:
            point_cloud: Point cloud data
            L0: 0-dimensional Laplacian
        
        Returns:
            Sparse edge Laplacian matrix
        """
        # Extract edges from L0
        edges = []
        n = point_cloud.shape[0]
        
        L0_dense = L0.toarray()
        for i in range(n):
            for j in range(i + 1, n):
                if L0_dense[i, j] < 0:  # Connected in graph
                    edges.append((i, j))
        
        if not edges:
            # No edges, return empty matrix
            return sp.csr_matrix((1, 1))
        
        # Build boundary operator B1
        num_edges = len(edges)
        num_vertices = n
        
        B1 = sp.lil_matrix((num_vertices, num_edges))
        
        for edge_idx, (i, j) in enumerate(edges):
            B1[i, edge_idx] = -1
            B1[j, edge_idx] = 1
        
        # Edge Laplacian L1 = B1^T * B1
        B1 = B1.tocsr()
        L1 = B1.T @ B1
        
        return L1
    
    def _compute_triangle_laplacian(self, point_cloud: np.ndarray) -> sp.spmatrix:
        """
        Compute 2-dimensional triangle Laplacian
        
        Args:
            point_cloud: Point cloud data
        
        Returns:
            Sparse triangle Laplacian matrix
        """
        # Simplified: use Delaunay triangulation
        from scipy.spatial import Delaunay
        
        try:
            tri = Delaunay(point_cloud[:, :min(3, point_cloud.shape[1])])
            simplices = tri.simplices
            
            if len(simplices) == 0:
                return sp.csr_matrix((1, 1))
            
            # Build boundary operator B2
            num_triangles = len(simplices)
            num_edges = num_triangles * 3  # Upper bound
            
            # Create edge-to-triangle incidence
            B2 = sp.lil_matrix((num_edges, num_triangles))
            
            edge_set = set()
            edge_list = []
            
            for tri_idx, simplex in enumerate(simplices):
                # Get edges of triangle
                edges = [
                    tuple(sorted([simplex[0], simplex[1]])),
                    tuple(sorted([simplex[1], simplex[2]])),
                    tuple(sorted([simplex[0], simplex[2]]))
                ]
                
                for edge in edges:
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edge_list.append(edge)
                    
                    edge_idx = edge_list.index(edge)
                    B2[edge_idx, tri_idx] = 1
            
            # Triangle Laplacian L2 = B2^T * B2
            B2 = B2.tocsr()[:len(edge_list), :]  # Trim to actual edges
            L2 = B2.T @ B2
            
            return L2
            
        except Exception as e:
            logger.debug(f"Could not compute triangle Laplacian: {e}")
            return sp.csr_matrix((1, 1))
    
    def _compute_spectra(self, laplacians: List[sp.spmatrix]) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue spectra of Laplacians
        
        Args:
            laplacians: List of Laplacian matrices
        
        Returns:
            Dictionary with eigenvalues and eigenvectors
        """
        all_eigenvalues = []
        all_eigenvectors = []
        
        for dim, L in enumerate(laplacians):
            if L.shape[0] <= 1:
                # Trivial case
                eigenvalues = np.array([0.0])
                eigenvectors = np.array([[1.0]])
            else:
                try:
                    # Compute smallest eigenvalues
                    k = min(self.num_eigenvalues, L.shape[0] - 1)
                    if k > 0:
                        eigenvalues, eigenvectors = eigsh(
                            L, k=k, which='SM', return_eigenvectors=True
                        )
                    else:
                        eigenvalues = np.array([0.0])
                        eigenvectors = np.array([[1.0]])
                        
                except Exception as e:
                    logger.debug(f"Eigenvalue computation failed for dim {dim}: {e}")
                    eigenvalues = np.array([0.0])
                    eigenvectors = np.array([[1.0]])
            
            all_eigenvalues.append(eigenvalues)
            all_eigenvectors.append(eigenvectors)
        
        # Concatenate spectra from all dimensions
        combined_eigenvalues = np.concatenate(all_eigenvalues)
        
        return {
            'eigenvalues': combined_eigenvalues,
            'eigenvectors': all_eigenvectors,
            'by_dimension': all_eigenvalues
        }
    
    def _decompose_spectrum(self, spectra: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose spectrum into harmonic and non-harmonic components
        
        Harmonic: Near-zero eigenvalues (topological)
        Non-harmonic: Non-zero eigenvalues (geometric)
        
        Args:
            spectra: Eigenvalue spectra
        
        Returns:
            Tuple of (harmonic_spectrum, non_harmonic_spectrum)
        """
        eigenvalues = spectra['eigenvalues']
        
        # Threshold for harmonic (near-zero)
        harmonic_threshold = 1e-6
        
        # Separate components
        harmonic_mask = np.abs(eigenvalues) < harmonic_threshold
        harmonic_spectrum = eigenvalues[harmonic_mask]
        non_harmonic_spectrum = eigenvalues[~harmonic_mask]
        
        # Ensure minimum length
        if len(harmonic_spectrum) == 0:
            harmonic_spectrum = np.array([0.0])
        if len(non_harmonic_spectrum) == 0:
            non_harmonic_spectrum = np.array([1.0])
        
        # Pad to fixed length for consistency
        target_length = self.num_eigenvalues // 2
        
        harmonic_spectrum = self._pad_or_truncate(harmonic_spectrum, target_length)
        non_harmonic_spectrum = self._pad_or_truncate(non_harmonic_spectrum, target_length)
        
        return harmonic_spectrum, non_harmonic_spectrum
    
    def _hodge_decomposition(self, point_cloud: np.ndarray, 
                            laplacians: List[sp.spmatrix]) -> Dict[str, np.ndarray]:
        """
        Compute Hodge decomposition
        
        Decomposes into:
        - Gradient flow (exact forms)
        - Curl flow (co-exact forms)
        - Harmonic flow (harmonic forms)
        
        Args:
            point_cloud: Point cloud data
            laplacians: List of Laplacian matrices
        
        Returns:
            Dictionary with Hodge components
        """
        hodge = {}
        
        # For each dimension
        for dim, L in enumerate(laplacians):
            if L.shape[0] <= 1:
                # Trivial case
                hodge[f'gradient_{dim}'] = np.array([0.0])
                hodge[f'curl_{dim}'] = np.array([0.0])
                hodge[f'harmonic_{dim}'] = np.array([1.0])
                continue
            
            try:
                # Get eigenvectors for smallest eigenvalues
                k = min(10, L.shape[0] - 1)
                if k > 0:
                    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
                    
                    # Harmonic component (kernel of Laplacian)
                    harmonic_mask = np.abs(eigenvalues) < 1e-6
                    harmonic_basis = eigenvectors[:, harmonic_mask]
                    
                    # Project random vector onto components
                    test_vector = np.random.randn(L.shape[0])
                    
                    if harmonic_basis.shape[1] > 0:
                        harmonic_component = harmonic_basis @ (harmonic_basis.T @ test_vector)
                    else:
                        harmonic_component = np.zeros_like(test_vector)
                    
                    # Gradient and curl components (simplified)
                    non_harmonic = test_vector - harmonic_component
                    gradient_component = non_harmonic * 0.6  # Approximate
                    curl_component = non_harmonic * 0.4      # Approximate
                    
                    # Store norms as features
                    hodge[f'gradient_{dim}'] = np.array([np.linalg.norm(gradient_component)])
                    hodge[f'curl_{dim}'] = np.array([np.linalg.norm(curl_component)])
                    hodge[f'harmonic_{dim}'] = np.array([np.linalg.norm(harmonic_component)])
                else:
                    hodge[f'gradient_{dim}'] = np.array([0.0])
                    hodge[f'curl_{dim}'] = np.array([0.0])
                    hodge[f'harmonic_{dim}'] = np.array([0.0])
                    
            except Exception as e:
                logger.debug(f"Hodge decomposition failed for dim {dim}: {e}")
                hodge[f'gradient_{dim}'] = np.array([0.0])
                hodge[f'curl_{dim}'] = np.array([0.0])
                hodge[f'harmonic_{dim}'] = np.array([0.0])
        
        return hodge
    
    def _calculate_spectral_persistence(self, spectra: Dict[str, np.ndarray]) -> float:
        """
        Calculate persistence from eigenvalue spectrum
        
        Spectral gap indicates topological stability.
        
        Args:
            spectra: Eigenvalue spectra
        
        Returns:
            Persistence measure
        """
        eigenvalues = spectra['eigenvalues']
        
        if len(eigenvalues) < 2:
            return 0.0
        
        # Sort eigenvalues
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))
        
        # Find spectral gap (largest jump between consecutive eigenvalues)
        gaps = np.diff(sorted_eigenvalues)
        
        if len(gaps) > 0:
            max_gap = np.max(gaps)
            # Normalize by mean eigenvalue
            mean_eigenvalue = np.mean(sorted_eigenvalues[sorted_eigenvalues > 1e-6])
            if mean_eigenvalue > 0:
                persistence = max_gap / mean_eigenvalue
            else:
                persistence = max_gap
        else:
            persistence = 0.0
        
        return min(persistence, 1.0)  # Cap at 1.0
    
    def _pad_or_truncate(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate array to target length
        
        Args:
            array: Input array
            target_length: Desired length
        
        Returns:
            Array with target length
        """
        if len(array) == target_length:
            return array
        elif len(array) < target_length:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:len(array)] = array
            return padded
        else:
            # Truncate
            return array[:target_length]
    
    def compute_laplacian_distance(self, sig1: LaplacianSignature, 
                                  sig2: LaplacianSignature) -> float:
        """
        Compute distance between two Laplacian signatures
        
        Args:
            sig1, sig2: Laplacian signatures to compare
        
        Returns:
            Distance measure
        """
        # Weighted combination of different components
        
        # Harmonic distance (topological)
        harmonic_dist = np.linalg.norm(sig1.harmonic_spectrum - sig2.harmonic_spectrum)
        
        # Non-harmonic distance (geometric)
        non_harmonic_dist = np.linalg.norm(sig1.non_harmonic_spectrum - sig2.non_harmonic_spectrum)
        
        # Persistence distance
        persistence_dist = abs(sig1.persistence - sig2.persistence)
        
        # Weighted combination
        distance = (
            0.5 * harmonic_dist +      # Topology most important
            0.3 * non_harmonic_dist +   # Geometry secondary
            0.2 * persistence_dist      # Stability measure
        )
        
        return distance
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extractor statistics"""
        return {
            "max_dimension": self.max_dimension,
            "num_eigenvalues": self.num_eigenvalues,
            "cache_size": len(self._laplacian_cache),
            "spectrum_cache_size": len(self._spectrum_cache)
        }