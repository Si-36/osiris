"""
Memory Tiers Service API
Ultra-fast hierarchical memory with shape-aware indexing
Based on AURA Intelligence research and 2025 best practices
"""

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time
import numpy as np
import structlog
from typing import Optional, List, Dict, Any
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from ..models.advanced.memory_tiers_2025 import (
    MemoryProcessor, MemoryConfig, MemoryTier
)
from ..services.shape_analyzer import ShapeAnalysisService
from ..services.graph_manager import GraphRelationshipManager
from ..services.prefetch_engine import PredictivePrefetchEngine
from ..schemas.requests import (
    StoreRequest, RetrieveRequest, ShapeQueryRequest,
    GraphQueryRequest, TierMigrationRequest, BenchmarkRequest
)
from ..schemas.responses import (
    StoreResponse, RetrieveResponse, ShapeQueryResponse,
    MemoryStatsResponse, TierDistribution, EfficiencyReport
)
from ..middleware.observability import ObservabilityMiddleware
from ..middleware.security import SecurityMiddleware
from ..middleware.circuit_breaker import CircuitBreakerMiddleware
from ..utils.observability import setup_telemetry

# Initialize structured logging
logger = structlog.get_logger()

# Setup telemetry
setup_telemetry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    Handles initialization and cleanup of memory systems
    """
    logger.info("Starting Memory Tiers Service")
    
    # Initialize configuration
    config = MemoryConfig(
        # CXL 3.0 configuration
        tier_configs={
            MemoryTier.CXL_HOT: {"capacity_gb": 64, "latency_ns": 15},
            MemoryTier.PMEM_WARM: {"capacity_gb": 512, "latency_ns": 200},
            MemoryTier.NVME_COLD: {"capacity_gb": 2048, "latency_ns": 10000}
        },
        # Shape-aware features
        enable_shape_indexing=True,
        tda_feature_dim=128,
        persistence_threshold=0.1,
        # Predictive prefetching
        enable_predictive_prefetch=True,
        prefetch_window_size=64,
        prefetch_confidence_threshold=0.8,
        # Access patterns
        access_pattern_window=1000,
        tier_promotion_threshold=10,
        tier_demotion_threshold=100
    )
    
    # Initialize services
    app.state.processor = MemoryProcessor(config)
    await app.state.processor.initialize()
    
    app.state.shape_analyzer = ShapeAnalysisService()
    app.state.graph_manager = GraphRelationshipManager()
    app.state.prefetch_engine = PredictivePrefetchEngine()
    
    # Pre-warm memory pools
    logger.info("Pre-warming memory pools...")
    test_data = {"warmup": True}
    await app.state.processor.store_with_topology("warmup_key", test_data)
    
    # Start background tasks
    app.state.tier_optimization_task = asyncio.create_task(
        tier_optimization_loop(app.state.processor)
    )
    
    logger.info("Memory Tiers Service ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Memory Tiers Service")
    app.state.tier_optimization_task.cancel()
    
    # Save final stats
    final_report = await app.state.processor.get_efficiency_report()
    logger.info("Final efficiency report", **final_report)
    
    await app.state.processor.memory_tiers.cleanup()


async def tier_optimization_loop(processor: MemoryProcessor):
    """Background task for tier optimization"""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute
            stats = await processor.get_efficiency_report()
            
            # Log tier distribution
            for tier_name, tier_data in stats['tier_stats'].items():
                if tier_data['utilization'] > 0.9:
                    logger.warning(
                        "Tier nearly full",
                        tier=tier_name,
                        utilization=tier_data['utilization']
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Tier optimization error", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="AURA Memory Tiers Service",
    description="Hierarchical memory with CXL 3.0, shape-aware indexing, and predictive prefetching",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CircuitBreakerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Memory-Tier", "X-Latency-NS"]
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AURA Memory Tiers Service",
        "version": "2.0.0",
        "features": [
            "CXL 3.0 memory pooling (10-20ns latency)",
            "Intel Optane DC persistent memory",
            "Shape-aware topological indexing",
            "Hierarchical tiering (Hot/Warm/Cold)",
            "Predictive prefetching with TDA",
            "Neo4j graph relationships",
            "Real-time access pattern learning"
        ],
        "status": "operational"
    }


@app.get("/api/v1/health", response_model=MemoryStatsResponse, tags=["health"])
async def health_check(request: Request):
    """
    Comprehensive health check with memory statistics
    """
    processor = request.app.state.processor
    
    # Get current stats
    stats = await processor.get_efficiency_report()
    
    # Check health criteria
    is_healthy = (
        stats['hit_ratio'] > 0.5 and  # At least 50% cache hit ratio
        stats['utilization_percent'] < 90  # Not over capacity
    )
    
    return MemoryStatsResponse(
        status="healthy" if is_healthy else "degraded",
        total_capacity_bytes=stats['total_capacity_bytes'],
        total_usage_bytes=stats['total_usage_bytes'],
        utilization_percent=stats['utilization_percent'],
        hit_ratio=stats['hit_ratio'],
        average_latency_ns=stats['average_latency_ns'],
        tier_distribution=stats['tier_stats'],
        total_accesses=stats['total_accesses'],
        tier_promotions=stats['tier_promotions'],
        shape_index_size=stats.get('shape_index_size', 0)
    )


@app.post("/api/v1/store", response_model=StoreResponse, tags=["storage"])
async def store_data(
    request: StoreRequest,
    background_tasks: BackgroundTasks,
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor),
    shape_analyzer: ShapeAnalysisService = Depends(lambda r: r.app.state.shape_analyzer)
):
    """
    Store data with automatic tier placement and shape indexing
    
    Features:
    - Automatic tier selection based on size and access patterns
    - Shape-aware indexing for topological queries
    - Graph relationship tracking
    - Predictive prefetching setup
    """
    try:
        # Analyze shape if requested
        tda_features = None
        if request.enable_shape_analysis and request.data:
            tda_features = await shape_analyzer.analyze(request.data)
        
        # Store with all features
        start_time = time.perf_counter()
        
        key = await processor.store_with_topology(
            key=request.key or f"mem_{int(time.time()*1000000)}",
            data=request.data,
            tda_features=tda_features,
            relationships=request.relationships
        )
        
        latency_ns = (time.perf_counter() - start_time) * 1e9
        
        # Background task to update access patterns
        background_tasks.add_task(
            processor.memory_tiers._track_access,
            key
        )
        
        # Determine which tier it was stored in
        entry = await processor.memory_tiers._find_entry(key)
        tier = entry.tier.value if entry else "unknown"
        
        return StoreResponse(
            key=key,
            tier=tier,
            size_bytes=len(str(request.data).encode()),
            latency_ns=latency_ns,
            shape_indexed=tda_features is not None,
            relationships_stored=len(request.relationships or [])
        )
        
    except Exception as e:
        logger.error("Store operation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieve", response_model=RetrieveResponse, tags=["storage"])
async def retrieve_data(
    request: RetrieveRequest,
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor)
):
    """
    Retrieve data by key with automatic tier promotion
    
    Features:
    - Multi-tier search from fastest to slowest
    - Automatic promotion of hot data
    - Predictive prefetching of related data
    - Access pattern tracking
    """
    try:
        start_time = time.perf_counter()
        
        data = await processor.memory_tiers.retrieve(key=request.key)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"Key {request.key} not found")
        
        latency_ns = (time.perf_counter() - start_time) * 1e9
        
        # Get entry details
        entry = await processor.memory_tiers._find_entry(request.key)
        
        return RetrieveResponse(
            key=request.key,
            data=data,
            tier=entry.tier.value if entry else "unknown",
            latency_ns=latency_ns,
            access_count=entry.access_count if entry else 0,
            prefetched_keys=[]  # TODO: Track prefetched keys
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Retrieve operation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/shape", response_model=ShapeQueryResponse, tags=["query"])
async def query_by_shape(
    request: ShapeQueryRequest,
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor),
    shape_analyzer: ShapeAnalysisService = Depends(lambda r: r.app.state.shape_analyzer)
):
    """
    Query data by topological shape similarity
    
    Applications:
    - Find similar patterns in time series
    - Match molecular structures
    - Identify similar network topologies
    - Content-based retrieval
    """
    try:
        # Analyze query shape
        tda_features = await shape_analyzer.analyze(request.query_data)
        
        # Search by shape
        start_time = time.perf_counter()
        
        results = await processor.retrieve_by_shape(
            query_features=tda_features,
            k=request.k
        )
        
        latency_ns = (time.perf_counter() - start_time) * 1e9
        
        # Format results
        formatted_results = []
        for data, similarity in results:
            formatted_results.append({
                "data": data,
                "similarity_score": float(similarity),
                "distance": float(similarity)  # Convert distance to similarity
            })
        
        return ShapeQueryResponse(
            results=formatted_results,
            query_latency_ns=latency_ns,
            num_results=len(results),
            tda_features=tda_features
        )
        
    except Exception as e:
        logger.error("Shape query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/graph", tags=["query"])
async def query_by_graph(
    request: GraphQueryRequest,
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor),
    graph_manager: GraphRelationshipManager = Depends(lambda r: r.app.state.graph_manager)
):
    """
    Query data using graph relationships (Neo4j)
    
    Features:
    - Multi-hop graph traversal
    - Pattern matching
    - Community detection
    - PageRank-based importance
    """
    try:
        # This would implement actual Neo4j queries
        # For now, return placeholder
        return {
            "results": [],
            "query": request.cypher_query,
            "execution_time_ms": 0
        }
        
    except Exception as e:
        logger.error("Graph query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/tier/migrate", tags=["management"])
async def migrate_tier(
    request: TierMigrationRequest,
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor)
):
    """
    Manually migrate data between memory tiers
    
    Use cases:
    - Preload critical data to hot tier
    - Archive cold data
    - Rebalance tier distribution
    """
    try:
        entry = await processor.memory_tiers._find_entry(request.key)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Key {request.key} not found")
        
        old_tier = entry.tier
        new_tier = MemoryTier(request.target_tier)
        
        await processor.memory_tiers._migrate_entry(entry, new_tier)
        
        return {
            "key": request.key,
            "from_tier": old_tier.value,
            "to_tier": new_tier.value,
            "status": "migrated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Tier migration failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats/efficiency", response_model=EfficiencyReport, tags=["stats"])
async def get_efficiency_stats(
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor)
):
    """
    Get comprehensive memory efficiency statistics
    
    Metrics:
    - Tier utilization and distribution
    - Cache hit ratios
    - Average latency by tier
    - Bandwidth utilization
    - Cost efficiency
    """
    try:
        stats = await processor.get_efficiency_report()
        
        # Calculate cost efficiency
        total_cost = 0
        for tier_name, tier_data in stats['tier_stats'].items():
            tier = MemoryTier(tier_name)
            config = processor.config.tier_configs.get(tier)
            if config:
                cost = tier_data['usage_bytes'] / 1e9 * config.cost_per_gb
                total_cost += cost
        
        return EfficiencyReport(
            total_capacity_gb=stats['total_capacity_bytes'] / 1e9,
            total_usage_gb=stats['total_usage_bytes'] / 1e9,
            utilization_percent=stats['utilization_percent'],
            hit_ratio=stats['hit_ratio'],
            average_latency_ns=stats['average_latency_ns'],
            effective_bandwidth_gbps=stats['effective_bandwidth_gbps'],
            tier_distribution=stats['tier_stats'],
            total_accesses=stats['total_accesses'],
            tier_promotions=stats['tier_promotions'],
            tier_demotions=stats['tier_demotions'],
            cost_per_gb_effective=total_cost / (stats['total_usage_bytes'] / 1e9) if stats['total_usage_bytes'] > 0 else 0,
            shape_index_entries=stats.get('shape_index_size', 0)
        )
        
    except Exception as e:
        logger.error("Failed to get efficiency stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/benchmark", tags=["benchmark"])
async def run_benchmark(
    request: BenchmarkRequest,
    processor: MemoryProcessor = Depends(lambda r: r.app.state.processor)
):
    """
    Run comprehensive memory benchmarks
    
    Tests:
    - Latency distribution by tier
    - Throughput limits
    - Shape indexing performance
    - Tier migration overhead
    """
    try:
        # This would implement actual benchmarking
        # For now, return placeholder results
        return {
            "benchmark_type": request.benchmark_type,
            "iterations": request.iterations,
            "results": {
                "avg_store_latency_ns": 150,
                "avg_retrieve_latency_ns": 50,
                "shape_index_build_time_ms": 10,
                "tier_migration_time_us": 100
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Benchmark failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tiers", tags=["info"])
async def list_memory_tiers():
    """List all available memory tiers and their characteristics"""
    return {
        "tiers": {
            "l1_cache": {
                "latency_ns": 0.5,
                "capacity": "32KB-64KB per core",
                "bandwidth": "3.2TB/s",
                "persistence": False,
                "use_case": "CPU registers and immediate data"
            },
            "l2_cache": {
                "latency_ns": 2,
                "capacity": "256KB-1MB per core",
                "bandwidth": "1.6TB/s",
                "persistence": False,
                "use_case": "Frequently accessed small data"
            },
            "l3_cache": {
                "latency_ns": 10,
                "capacity": "8MB-128MB shared",
                "bandwidth": "800GB/s",
                "persistence": False,
                "use_case": "Shared cache for all cores"
            },
            "cxl_hot": {
                "latency_ns": 15,
                "capacity": "64GB-256GB",
                "bandwidth": "256GB/s",
                "persistence": False,
                "use_case": "CXL 3.0 attached memory for hot data"
            },
            "dram": {
                "latency_ns": 50,
                "capacity": "32GB-1TB",
                "bandwidth": "128GB/s",
                "persistence": False,
                "use_case": "Main memory"
            },
            "pmem_warm": {
                "latency_ns": 200,
                "capacity": "512GB-6TB",
                "bandwidth": "40GB/s",
                "persistence": True,
                "use_case": "Intel Optane for warm persistent data"
            },
            "nvme_cold": {
                "latency_ns": 10000,
                "capacity": "1TB-100TB",
                "bandwidth": "7GB/s",
                "persistence": True,
                "use_case": "NVMe SSD for cold data"
            },
            "hdd_archive": {
                "latency_ns": 10000000,
                "capacity": "10TB-100PB",
                "bandwidth": "200MB/s",
                "persistence": True,
                "use_case": "Archive and backup"
            }
        }
    }


@app.get("/api/v1/features", tags=["info"])
async def list_features():
    """List advanced memory features"""
    return {
        "shape_aware_indexing": {
            "description": "Topological indexing using TDA",
            "algorithms": ["Persistent homology", "Betti numbers", "Wasserstein distance"],
            "use_cases": ["Pattern matching", "Similarity search", "Anomaly detection"]
        },
        "predictive_prefetching": {
            "description": "ML-based prefetch prediction",
            "techniques": ["Access pattern learning", "Graph-based prediction", "Time series analysis"],
            "benefits": ["Reduced latency", "Better cache utilization"]
        },
        "tier_optimization": {
            "description": "Automatic data movement between tiers",
            "policies": ["LRU", "LFU", "Cost-aware", "ML-based"],
            "triggers": ["Access count", "Time-based", "Capacity"]
        },
        "graph_relationships": {
            "backend": "Neo4j",
            "features": ["Multi-hop queries", "Pattern matching", "Community detection"],
            "query_language": "Cypher"
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with correlation ID"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        path=request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": correlation_id,
            "message": str(exc) if request.app.debug else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )