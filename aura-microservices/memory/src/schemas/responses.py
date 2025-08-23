"""
Response schemas for Memory Tiers API
Comprehensive responses with performance metrics
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime


class StoreResponse(BaseModel):
    """Response from store operation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "sensor_data_123",
            "tier": "cxl_hot",
            "size_bytes": 1024,
            "latency_ns": 150,
            "shape_indexed": True,
            "relationships_stored": 2
        }
    })
    
    key: str = Field(..., description="Stored data key")
    tier: str = Field(..., description="Memory tier where data was stored")
    size_bytes: int = Field(..., description="Size of stored data")
    latency_ns: float = Field(..., description="Store operation latency in nanoseconds")
    shape_indexed: bool = Field(..., description="Whether shape indexing was applied")
    relationships_stored: int = Field(..., description="Number of graph relationships created")
    metadata_stored: Optional[Dict[str, Any]] = Field(default=None, description="Stored metadata")


class RetrieveResponse(BaseModel):
    """Response from retrieve operation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "sensor_data_123",
            "data": {"temperature": 25.5, "humidity": 60},
            "tier": "l3_cache",
            "latency_ns": 50,
            "access_count": 15,
            "prefetched_keys": ["sensor_data_124"]
        }
    })
    
    key: str = Field(..., description="Retrieved data key")
    data: Any = Field(..., description="Retrieved data")
    tier: str = Field(..., description="Tier from which data was retrieved")
    latency_ns: float = Field(..., description="Retrieval latency in nanoseconds")
    access_count: int = Field(..., description="Total access count for this data")
    prefetched_keys: List[str] = Field(default_factory=list, description="Keys that were prefetched")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Associated metadata")
    last_access_time: Optional[float] = Field(default=None, description="Unix timestamp of last access")


class ShapeQueryResult(BaseModel):
    """Single result from shape query"""
    data: Any = Field(..., description="Retrieved data")
    similarity_score: float = Field(..., description="Shape similarity score (0-1)")
    distance: float = Field(..., description="Topological distance")
    key: Optional[str] = Field(default=None, description="Data key if available")
    tier: Optional[str] = Field(default=None, description="Current storage tier")


class ShapeQueryResponse(BaseModel):
    """Response from shape-based query"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": [
                {"data": {"pattern": [1, 2, 1]}, "similarity_score": 0.95, "distance": 0.05},
                {"data": {"pattern": [1, 3, 1]}, "similarity_score": 0.87, "distance": 0.13}
            ],
            "query_latency_ns": 1500,
            "num_results": 2,
            "tda_features": {"betti_numbers": [1, 2, 0]}
        }
    })
    
    results: List[ShapeQueryResult] = Field(..., description="Similar data sorted by score")
    query_latency_ns: float = Field(..., description="Query execution time in nanoseconds")
    num_results: int = Field(..., description="Number of results returned")
    tda_features: Dict[str, Any] = Field(..., description="TDA features of query")
    index_size: Optional[int] = Field(default=None, description="Current shape index size")


class TierDistribution(BaseModel):
    """Distribution of data across tiers"""
    entries: int = Field(..., description="Number of entries in tier")
    usage_bytes: int = Field(..., description="Total bytes used in tier")
    capacity_bytes: int = Field(..., description="Total tier capacity")
    utilization: float = Field(..., description="Utilization ratio (0-1)")
    avg_access_count: Optional[float] = Field(default=None, description="Average access count")
    
    
class MemoryStatsResponse(BaseModel):
    """Memory system statistics"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "total_capacity_bytes": 549755813888,
            "total_usage_bytes": 27487790694,
            "utilization_percent": 5.0,
            "hit_ratio": 0.85,
            "average_latency_ns": 75.5
        }
    })
    
    status: str = Field(..., description="System health status")
    total_capacity_bytes: int = Field(..., description="Total memory capacity across all tiers")
    total_usage_bytes: int = Field(..., description="Total memory usage")
    utilization_percent: float = Field(..., description="Overall utilization percentage")
    hit_ratio: float = Field(..., description="Cache hit ratio (0-1)")
    average_latency_ns: float = Field(..., description="Average access latency")
    tier_distribution: Dict[str, TierDistribution] = Field(..., description="Distribution by tier")
    total_accesses: int = Field(..., description="Total number of accesses")
    tier_promotions: int = Field(..., description="Number of tier promotions")
    tier_demotions: int = Field(default=0, description="Number of tier demotions")
    shape_index_size: int = Field(..., description="Number of entries in shape index")


class EfficiencyReport(BaseModel):
    """Comprehensive efficiency report"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_capacity_gb": 549.76,
            "total_usage_gb": 27.49,
            "utilization_percent": 5.0,
            "hit_ratio": 0.85,
            "average_latency_ns": 75.5,
            "effective_bandwidth_gbps": 256.0,
            "cost_per_gb_effective": 2.5
        }
    })
    
    total_capacity_gb: float = Field(..., description="Total capacity in GB")
    total_usage_gb: float = Field(..., description="Total usage in GB")
    utilization_percent: float = Field(..., description="Overall utilization")
    hit_ratio: float = Field(..., description="Cache hit ratio")
    average_latency_ns: float = Field(..., description="Average latency in nanoseconds")
    effective_bandwidth_gbps: float = Field(..., description="Effective bandwidth in GB/s")
    tier_distribution: Dict[str, Any] = Field(..., description="Detailed tier distribution")
    total_accesses: int = Field(..., description="Total access count")
    tier_promotions: int = Field(..., description="Promotion count")
    tier_demotions: int = Field(..., description="Demotion count")
    cost_per_gb_effective: float = Field(..., description="Effective cost per GB based on tier usage")
    shape_index_entries: int = Field(..., description="Shape index size")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkResult(BaseModel):
    """Single benchmark result"""
    metric: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Metric unit")
    percentiles: Optional[Dict[str, float]] = Field(default=None, description="Percentile values")


class BenchmarkResponse(BaseModel):
    """Response from benchmark run"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "benchmark_type": "latency",
            "iterations": 1000,
            "results": [
                {
                    "metric": "avg_store_latency",
                    "value": 150,
                    "unit": "nanoseconds",
                    "percentiles": {"p50": 140, "p95": 200, "p99": 250}
                }
            ]
        }
    })
    
    benchmark_type: str = Field(..., description="Type of benchmark run")
    iterations: int = Field(..., description="Number of iterations")
    results: List[BenchmarkResult] = Field(..., description="Benchmark results")
    total_duration_ms: float = Field(..., description="Total benchmark duration")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Benchmark configuration")


class GraphQueryResponse(BaseModel):
    """Response from graph query"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": [
                {"key": "sensor_data_124", "data": {"value": 2}, "relationship": "RELATED_TO"},
                {"key": "sensor_data_125", "data": {"value": 3}, "relationship": "DERIVED_FROM"}
            ],
            "query": "MATCH (n:MemoryNode {key: $key})-[r]-(m) RETURN m, r",
            "execution_time_ms": 5.2,
            "nodes_visited": 3
        }
    })
    
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    query: str = Field(..., description="Executed query")
    execution_time_ms: float = Field(..., description="Query execution time")
    nodes_visited: int = Field(..., description="Number of nodes visited")
    relationships_traversed: Optional[int] = Field(default=None, description="Number of relationships traversed")


class TierMigrationResponse(BaseModel):
    """Response from tier migration"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "sensor_data_123",
            "from_tier": "pmem_warm",
            "to_tier": "cxl_hot",
            "status": "completed",
            "duration_us": 100,
            "size_bytes": 1024
        }
    })
    
    key: str = Field(..., description="Migrated key")
    from_tier: str = Field(..., description="Source tier")
    to_tier: str = Field(..., description="Destination tier")
    status: str = Field(..., description="Migration status")
    duration_us: float = Field(..., description="Migration duration in microseconds")
    size_bytes: int = Field(..., description="Size of migrated data")
    reason: Optional[str] = Field(default=None, description="Migration reason")


class PrefetchResponse(BaseModel):
    """Response from prefetch operation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prefetched_keys": ["sensor_data_123", "sensor_data_124"],
            "target_tier": "l3_cache",
            "total_size_bytes": 2048,
            "duration_us": 50,
            "related_keys_found": 5
        }
    })
    
    prefetched_keys: List[str] = Field(..., description="Keys that were prefetched")
    target_tier: str = Field(..., description="Tier where data was prefetched")
    total_size_bytes: int = Field(..., description="Total size of prefetched data")
    duration_us: float = Field(..., description="Prefetch duration in microseconds")
    related_keys_found: int = Field(..., description="Number of related keys discovered")
    success_rate: float = Field(default=1.0, description="Fraction of successful prefetches")