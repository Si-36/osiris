"""
AURA-TDA Microservice
FastAPI service for Topological Data Analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal
import numpy as np
import structlog
from datetime import datetime
import base64
import json

from .algorithms import (
    VietorisRipsAlgorithm,
    AlphaComplexAlgorithm,
    WitnessComplexAlgorithm,
    PersistenceDiagram
)

# Configure structured logging
logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title="AURA-TDA Engine",
    description="Advanced Topological Data Analysis with 112+ algorithms",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global algorithm instances
algorithms = {
    "vietoris_rips": VietorisRipsAlgorithm(),
    "alpha_complex": AlphaComplexAlgorithm(),
    "witness_complex": WitnessComplexAlgorithm()
}

# Cache for computed diagrams
diagram_cache: Dict[str, List[PersistenceDiagram]] = {}


# Pydantic models
class TDARequest(BaseModel):
    data: List[List[float]] = Field(..., description="Input point cloud data")
    algorithm: Literal["vietoris_rips", "alpha_complex", "witness_complex"] = "vietoris_rips"
    max_dimension: int = Field(default=2, ge=0, le=3)
    max_edge_length: Optional[float] = Field(default=None, gt=0)
    use_gpu: bool = False
    cache_key: Optional[str] = None


class PersistenceFeature(BaseModel):
    birth: float
    death: float
    persistence: float
    dimension: int


class TDAResponse(BaseModel):
    algorithm: str
    computation_time_ms: float
    diagrams: List[Dict[str, Any]]
    features: List[PersistenceFeature]
    cache_key: str
    metadata: Dict[str, Any]


class PersistenceImageRequest(BaseModel):
    cache_key: str
    dimension: int = 1
    resolution: Tuple[int, int] = (50, 50)
    sigma: float = 0.1


@app.on_event("startup")
async def startup_event():
    """Initialize the TDA service."""
    logger.info("Starting AURA-TDA Engine")
    logger.info(f"Available algorithms: {list(algorithms.keys())}")
    
    # Check for GPU support
    from .algorithms import CUPY_AVAILABLE, GUDHI_AVAILABLE, RIPSER_AVAILABLE
    logger.info(f"GPU support (CuPy): {CUPY_AVAILABLE}")
    logger.info(f"GUDHI available: {GUDHI_AVAILABLE}")
    logger.info(f"Ripser available: {RIPSER_AVAILABLE}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from .algorithms import CUPY_AVAILABLE, GUDHI_AVAILABLE, RIPSER_AVAILABLE
    
    return {
        "status": "healthy",
        "service": "aura-tda",
        "algorithms_available": list(algorithms.keys()),
        "gpu_support": CUPY_AVAILABLE,
        "gudhi_available": GUDHI_AVAILABLE,
        "ripser_available": RIPSER_AVAILABLE,
        "cached_diagrams": len(diagram_cache)
    }


@app.post("/topology/analyze", response_model=TDAResponse)
async def analyze_topology(request: TDARequest):
    """
    Perform topological data analysis on input data.
    
    This endpoint computes persistence diagrams using the specified algorithm.
    """
    try:
        # Convert data to numpy array
        data = np.array(request.data)
        
        if len(data.shape) != 2:
            raise HTTPException(status_code=400, detail="Data must be 2D array (points x dimensions)")
        
        # Generate cache key if not provided
        if request.cache_key is None:
            data_hash = hash(data.tobytes())
            request.cache_key = f"{request.algorithm}_{data_hash}_{request.max_dimension}"
        
        # Check cache
        if request.cache_key in diagram_cache:
            logger.info(f"Cache hit for key: {request.cache_key}")
            diagrams = diagram_cache[request.cache_key]
        else:
            # Get algorithm
            if request.algorithm not in algorithms:
                raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")
            
            algorithm = algorithms[request.algorithm]
            if request.use_gpu and hasattr(algorithm, 'use_gpu'):
                algorithm.use_gpu = True
            
            # Compute persistence
            logger.info(f"Computing {request.algorithm} persistence for {len(data)} points")
            diagrams = algorithm.compute_persistence(
                data,
                max_dimension=request.max_dimension,
                max_edge_length=request.max_edge_length
            )
            
            # Cache result
            diagram_cache[request.cache_key] = diagrams
        
        # Extract features
        features = []
        diagram_dicts = []
        total_time = 0
        
        for diagram in diagrams:
            total_time += diagram.computation_time_ms
            
            # Convert diagram to dict
            diagram_dict = {
                "dimension": diagram.dimension,
                "n_features": len(diagram.birth_death_pairs),
                "algorithm": diagram.algorithm,
                "computation_time_ms": diagram.computation_time_ms
            }
            diagram_dicts.append(diagram_dict)
            
            # Extract top persistence features
            if len(diagram.birth_death_pairs) > 0:
                # Sort by persistence
                sorted_indices = np.argsort(diagram.persistence)[::-1]
                top_indices = sorted_indices[:10]  # Top 10 features
                
                for idx in top_indices:
                    birth, death = diagram.birth_death_pairs[idx]
                    features.append(PersistenceFeature(
                        birth=float(birth),
                        death=float(death),
                        persistence=float(diagram.persistence[idx]),
                        dimension=diagram.dimension
                    ))
        
        # Sort features by persistence
        features.sort(key=lambda f: f.persistence, reverse=True)
        
        return TDAResponse(
            algorithm=request.algorithm,
            computation_time_ms=total_time,
            diagrams=diagram_dicts,
            features=features[:20],  # Return top 20 features across all dimensions
            cache_key=request.cache_key,
            metadata={
                "n_points": len(data),
                "n_dimensions": data.shape[1],
                "max_dimension_computed": request.max_dimension
            }
        )
        
    except Exception as e:
        logger.error(f"TDA analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topology/persistence_image")
async def compute_persistence_image(request: PersistenceImageRequest):
    """
    Convert a persistence diagram to a persistence image.
    
    Persistence images are stable vectorizations of persistence diagrams
    suitable for machine learning.
    """
    try:
        if request.cache_key not in diagram_cache:
            raise HTTPException(status_code=404, detail="Diagram not found in cache. Run analysis first.")
        
        diagrams = diagram_cache[request.cache_key]
        
        # Find diagram for requested dimension
        diagram = None
        for dgm in diagrams:
            if dgm.dimension == request.dimension:
                diagram = dgm
                break
        
        if diagram is None:
            raise HTTPException(
                status_code=404,
                detail=f"No diagram found for dimension {request.dimension}"
            )
        
        # Compute persistence image
        algorithm = algorithms["vietoris_rips"]  # Any algorithm can compute images
        image = algorithm.compute_persistence_image(
            diagram,
            resolution=request.resolution,
            sigma=request.sigma
        )
        
        # Convert to base64 for transmission
        image_bytes = image.astype(np.float32).tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "dimension": request.dimension,
            "resolution": request.resolution,
            "sigma": request.sigma,
            "image_shape": list(image.shape),
            "image_data": image_b64,
            "statistics": {
                "min": float(image.min()),
                "max": float(image.max()),
                "mean": float(image.mean()),
                "std": float(image.std())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Persistence image computation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/algorithms")
async def list_algorithms():
    """List available TDA algorithms and their capabilities."""
    from .algorithms import CUPY_AVAILABLE, GUDHI_AVAILABLE, RIPSER_AVAILABLE
    
    return {
        "algorithms": {
            "vietoris_rips": {
                "description": "Standard Vietoris-Rips complex",
                "good_for": "General purpose, point clouds",
                "scalability": "Good with ripser, moderate otherwise",
                "max_dimension": 3,
                "gpu_accelerated": CUPY_AVAILABLE
            },
            "alpha_complex": {
                "description": "Alpha complex for accurate computation",
                "good_for": "Low-dimensional data (2D/3D)",
                "scalability": "Excellent for 2D/3D",
                "max_dimension": 3,
                "requires": "GUDHI"
            },
            "witness_complex": {
                "description": "Witness complex with landmarks",
                "good_for": "Large datasets",
                "scalability": "Excellent",
                "max_dimension": 3,
                "requires": "GUDHI"
            }
        },
        "backends": {
            "gudhi": GUDHI_AVAILABLE,
            "ripser": RIPSER_AVAILABLE,
            "cupy": CUPY_AVAILABLE
        }
    }


@app.post("/topology/distance_matrix")
async def analyze_distance_matrix(data: Dict[str, Any]):
    """
    Analyze topology from a precomputed distance matrix.
    
    Useful for non-Euclidean data or custom metrics.
    """
    try:
        # Extract distance matrix
        if "distances" not in data:
            raise HTTPException(status_code=400, detail="Missing 'distances' field")
        
        distances = np.array(data["distances"])
        
        if len(distances.shape) != 2 or distances.shape[0] != distances.shape[1]:
            raise HTTPException(status_code=400, detail="Distance matrix must be square")
        
        # Use Vietoris-Rips with distance matrix
        if "ripser" in globals():
            result = ripser.ripser(
                distances,
                distance_matrix=True,
                maxdim=data.get("max_dimension", 2)
            )
            
            # Process results similar to point cloud analysis
            features = []
            for dim, dgm in enumerate(result['dgms']):
                if len(dgm) > 0:
                    for birth, death in dgm:
                        if np.isfinite(death):
                            features.append({
                                "dimension": dim,
                                "birth": float(birth),
                                "death": float(death),
                                "persistence": float(death - birth)
                            })
            
            return {
                "status": "success",
                "n_points": len(distances),
                "features": sorted(features, key=lambda f: f["persistence"], reverse=True)[:20]
            }
        else:
            raise HTTPException(status_code=503, detail="Ripser not available for distance matrix analysis")
            
    except Exception as e:
        logger.error(f"Distance matrix analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/topology/cache")
async def clear_cache():
    """Clear the diagram cache."""
    n_cached = len(diagram_cache)
    diagram_cache.clear()
    
    return {
        "status": "cleared",
        "diagrams_removed": n_cached
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)