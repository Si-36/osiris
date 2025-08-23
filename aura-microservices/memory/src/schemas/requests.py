"""
Request schemas for Memory Tiers API
Pydantic V2 models with comprehensive validation
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Literal, Union
from enum import Enum


class MemoryTierEnum(str, Enum):
    """Available memory tiers"""
    L1_CACHE = "l1_cache"
    L2_CACHE = "l2_cache"
    L3_CACHE = "l3_cache"
    CXL_HOT = "cxl_hot"
    DRAM = "dram"
    PMEM_WARM = "pmem_warm"
    NVME_COLD = "nvme_cold"
    HDD_ARCHIVE = "hdd_archive"


class BenchmarkType(str, Enum):
    """Types of benchmarks"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SHAPE_INDEX = "shape_index"
    TIER_MIGRATION = "tier_migration"
    FULL = "full"


class StoreRequest(BaseModel):
    """Request to store data in memory tiers"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "sensor_data_123",
            "data": {"temperature": 25.5, "humidity": 60, "readings": [1, 2, 3]},
            "enable_shape_analysis": True,
            "relationships": ["sensor_data_122", "sensor_data_124"],
            "metadata": {"source": "IoT_sensor_1", "timestamp": 1234567890}
        }
    })
    
    key: Optional[str] = Field(
        default=None,
        description="Unique key for data. Auto-generated if not provided",
        pattern="^[a-zA-Z0-9_-]+$"
    )
    data: Any = Field(
        ...,
        description="Data to store (any JSON-serializable type)"
    )
    enable_shape_analysis: bool = Field(
        default=True,
        description="Enable topological shape analysis for similarity search"
    )
    relationships: Optional[List[str]] = Field(
        default=None,
        description="Related data keys for graph relationships"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to store"
    )
    preferred_tier: Optional[MemoryTierEnum] = Field(
        default=None,
        description="Preferred memory tier (auto-selected if not specified)"
    )
    
    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v):
        """Ensure relationship keys are valid"""
        if v:
            for key in v:
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Invalid relationship key: {key}")
        return v


class RetrieveRequest(BaseModel):
    """Request to retrieve data by key"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "sensor_data_123",
            "include_metadata": True,
            "prefetch_related": True
        }
    })
    
    key: str = Field(
        ...,
        description="Key to retrieve",
        pattern="^[a-zA-Z0-9_-]+$"
    )
    include_metadata: bool = Field(
        default=False,
        description="Include metadata in response"
    )
    prefetch_related: bool = Field(
        default=True,
        description="Enable predictive prefetching of related data"
    )


class ShapeQueryRequest(BaseModel):
    """Request to query by topological shape"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query_data": {"pattern": [1, 2, 1, 3, 2, 1], "frequency": 100},
            "k": 10,
            "distance_metric": "wasserstein",
            "filters": {"tier": "cxl_hot"}
        }
    })
    
    query_data: Any = Field(
        ...,
        description="Query data for shape analysis"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of similar results to return"
    )
    distance_metric: Literal["euclidean", "wasserstein", "bottleneck"] = Field(
        default="wasserstein",
        description="Distance metric for shape comparison"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filters (tier, size, age)"
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )


class GraphQueryRequest(BaseModel):
    """Request to query using graph relationships"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "start_key": "sensor_data_123",
            "cypher_query": "MATCH (n:MemoryNode {key: $key})-[:RELATED_TO*1..3]-(m) RETURN m",
            "max_hops": 3,
            "relationship_types": ["RELATED_TO", "DERIVED_FROM"]
        }
    })
    
    start_key: Optional[str] = Field(
        default=None,
        description="Starting node key"
    )
    cypher_query: Optional[str] = Field(
        default=None,
        description="Raw Cypher query for Neo4j"
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum graph traversal depth"
    )
    relationship_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by relationship types"
    )
    
    @field_validator('cypher_query')
    @classmethod
    def validate_cypher(cls, v):
        """Basic Cypher query validation"""
        if v:
            # Basic safety checks
            dangerous_keywords = ['DELETE', 'REMOVE', 'SET', 'CREATE', 'MERGE']
            if any(keyword in v.upper() for keyword in dangerous_keywords):
                raise ValueError("Only read queries are allowed")
        return v


class TierMigrationRequest(BaseModel):
    """Request to migrate data between tiers"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "sensor_data_123",
            "target_tier": "cxl_hot",
            "reason": "Frequent access detected"
        }
    })
    
    key: str = Field(
        ...,
        description="Key to migrate"
    )
    target_tier: MemoryTierEnum = Field(
        ...,
        description="Target memory tier"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for migration"
    )
    force: bool = Field(
        default=False,
        description="Force migration even if not optimal"
    )


class BenchmarkRequest(BaseModel):
    """Request to run memory benchmarks"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "benchmark_type": "latency",
            "iterations": 1000,
            "data_size_bytes": 1024,
            "tiers_to_test": ["cxl_hot", "dram", "pmem_warm"]
        }
    })
    
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.FULL,
        description="Type of benchmark to run"
    )
    iterations: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Number of iterations"
    )
    data_size_bytes: int = Field(
        default=1024,
        ge=1,
        le=1048576,  # Max 1MB for benchmarks
        description="Size of test data"
    )
    tiers_to_test: Optional[List[MemoryTierEnum]] = Field(
        default=None,
        description="Specific tiers to benchmark"
    )
    concurrent_operations: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of concurrent operations"
    )


class BulkStoreRequest(BaseModel):
    """Request to store multiple items"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "items": [
                {"key": "item1", "data": {"value": 1}},
                {"key": "item2", "data": {"value": 2}}
            ],
            "enable_shape_analysis": False,
            "batch_size": 100
        }
    })
    
    items: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Items to store in bulk"
    )
    enable_shape_analysis: bool = Field(
        default=False,
        description="Enable shape analysis for all items"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for processing"
    )


class PrefetchRequest(BaseModel):
    """Request to prefetch data"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "keys": ["sensor_data_123", "sensor_data_124"],
            "target_tier": "l3_cache",
            "recursive": True
        }
    })
    
    keys: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Keys to prefetch"
    )
    target_tier: Optional[MemoryTierEnum] = Field(
        default=MemoryTierEnum.L3_CACHE,
        description="Target tier for prefetched data"
    )
    recursive: bool = Field(
        default=False,
        description="Also prefetch related data"
    )
    max_depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum depth for recursive prefetch"
    )