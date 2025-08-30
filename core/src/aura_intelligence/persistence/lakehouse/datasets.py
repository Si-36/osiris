"""
Iceberg Dataset Definitions
==========================
Defines the core datasets for AURA's lakehouse architecture.
Each dataset has versioned schemas with evolution support.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SchemaField:
    """Field definition for Iceberg schema"""
    name: str
    type: str  # Iceberg type: boolean, int, long, float, double, string, binary, date, timestamp, etc.
    nullable: bool = True
    doc: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Iceberg field definition"""
        field_def = {
            'name': self.name,
            'type': self.type,
            'nullable': self.nullable
        }
        if self.doc:
            field_def['doc'] = self.doc
        return field_def


@dataclass
class PartitionField:
    """Partition field definition"""
    source_column: str
    transform: str  # identity, year, month, day, hour, bucket, truncate
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Iceberg partition spec"""
        spec = {
            'source-id': self.source_column,
            'transform': self.transform
        }
        if self.name:
            spec['name'] = self.name
        return spec


class BaseDataset(ABC):
    """Base class for all Iceberg datasets"""
    
    def __init__(self, namespace: str = "aura"):
        self.namespace = namespace
        self.table_name = self.get_table_name()
        self.schema_version = 1
        
    @abstractmethod
    def get_table_name(self) -> str:
        """Get the table name"""
        pass
        
    @abstractmethod
    def get_schema(self) -> List[SchemaField]:
        """Get the current schema"""
        pass
        
    @abstractmethod
    def get_partition_spec(self) -> List[PartitionField]:
        """Get partition specification"""
        pass
        
    def get_table_properties(self) -> Dict[str, str]:
        """Get table properties"""
        return {
            'write.format.default': 'parquet',
            'write.parquet.compression-codec': 'zstd',
            'write.metadata.compression-codec': 'gzip',
            'write.summary.partition-limit': '500',
            'write.metadata.delete-after-commit.enabled': 'true',
            'write.metadata.previous-versions-max': '10',
            'history.expire.max-snapshot-age-ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
            'commit.retry.num-retries': '10',
            'commit.retry.min-wait-ms': '100',
            'format-version': '2'  # Iceberg v2 for row-level deletes
        }
        
    def get_full_table_name(self) -> str:
        """Get fully qualified table name"""
        return f"{self.namespace}.{self.table_name}"
        
    def to_iceberg_schema(self) -> Dict[str, Any]:
        """Convert to Iceberg schema format"""
        return {
            'type': 'struct',
            'fields': [f.to_dict() for f in self.get_schema()]
        }
        
    def to_partition_spec(self) -> List[Dict[str, Any]]:
        """Convert to Iceberg partition spec"""
        return [p.to_dict() for p in self.get_partition_spec()]


class EventsDataset(BaseDataset):
    """
    Immutable event log for all system events.
    Forms the foundation of event sourcing pattern.
    """
    
    def get_table_name(self) -> str:
        return "events"
        
    def get_schema(self) -> List[SchemaField]:
        return [
            # Core fields
            SchemaField("event_id", "string", False, "Unique event identifier"),
            SchemaField("timestamp", "timestamp", False, "Event timestamp (microsecond precision)"),
            SchemaField("tenant_id", "string", True, "Tenant identifier for multi-tenancy"),
            SchemaField("event_type", "string", False, "Type of event"),
            SchemaField("event_version", "int", False, "Event schema version"),
            
            # Payload
            SchemaField("payload", "string", False, "JSON-encoded event payload"),
            SchemaField("payload_hash", "string", False, "SHA-256 hash of payload"),
            SchemaField("payload_size_bytes", "long", False, "Size of payload in bytes"),
            
            # Context
            SchemaField("source_component", "string", False, "Component that generated the event"),
            SchemaField("correlation_id", "string", True, "Correlation ID for tracing"),
            SchemaField("causation_id", "string", True, "ID of event that caused this event"),
            SchemaField("user_id", "string", True, "User who triggered the event"),
            
            # Temporal integration
            SchemaField("temporal_workflow_id", "string", True, "Temporal workflow ID"),
            SchemaField("temporal_run_id", "string", True, "Temporal run ID"),
            SchemaField("saga_id", "string", True, "Saga transaction ID"),
            
            # Policy and compliance
            SchemaField("policy_tags", "map<string, string>", True, "Policy tags for governance"),
            SchemaField("retention_days", "int", True, "Retention period in days"),
            SchemaField("encryption_key_id", "string", True, "KMS key ID for encryption"),
            
            # Processing metadata
            SchemaField("ingested_at", "timestamp", False, "Ingestion timestamp"),
            SchemaField("processing_time_ms", "long", True, "Processing time in milliseconds"),
            SchemaField("partition_key", "string", False, "Partition key for distribution")
        ]
        
    def get_partition_spec(self) -> List[PartitionField]:
        return [
            PartitionField("timestamp", "day", "event_day"),
            PartitionField("event_type", "identity"),
            PartitionField("tenant_id", "identity")
        ]
        
    def get_table_properties(self) -> Dict[str, str]:
        props = super().get_table_properties()
        props.update({
            'write.wap.enabled': 'true',  # Write-audit-publish pattern
            'write.object-storage.enabled': 'true',
            'write.object-storage.path': 's3://aura-lakehouse/events/objects/',
            'format.orc.bloom.filter.columns': 'event_id,correlation_id'
        })
        return props


class FeaturesDataset(BaseDataset):
    """
    Feature store for ML model inputs.
    Supports point-in-time correct feature retrieval.
    """
    
    def get_table_name(self) -> str:
        return "features"
        
    def get_schema(self) -> List[SchemaField]:
        return [
            # Entity identification
            SchemaField("entity_id", "string", False, "Entity identifier"),
            SchemaField("entity_type", "string", False, "Type of entity"),
            SchemaField("timestamp", "timestamp", False, "Feature timestamp"),
            
            # Feature data
            SchemaField("feature_name", "string", False, "Name of feature"),
            SchemaField("feature_value", "double", True, "Numeric feature value"),
            SchemaField("feature_vector", "list<double>", True, "Vector feature"),
            SchemaField("feature_string", "string", True, "String feature value"),
            SchemaField("feature_map", "map<string, double>", True, "Map features"),
            
            # Topological features
            SchemaField("persistence_diagram", "string", True, "JSON persistence diagram"),
            SchemaField("betti_numbers", "list<int>", True, "Betti numbers"),
            SchemaField("wasserstein_distance", "double", True, "Wasserstein distance"),
            SchemaField("topological_signature_id", "string", True, "Reference to signature"),
            
            # Model metadata
            SchemaField("model_version", "string", True, "Model that created feature"),
            SchemaField("encoder_version", "string", True, "Feature encoder version"),
            SchemaField("feature_importance", "double", True, "Feature importance score"),
            
            # Lineage
            SchemaField("source_event_id", "string", True, "Source event ID"),
            SchemaField("computation_dag", "string", True, "JSON computation graph"),
            SchemaField("created_at", "timestamp", False, "Creation timestamp"),
            
            # Quality
            SchemaField("quality_score", "double", True, "Data quality score"),
            SchemaField("is_validated", "boolean", False, "Validation status"),
            SchemaField("validation_errors", "list<string>", True, "Validation errors")
        ]
        
    def get_partition_spec(self) -> List[PartitionField]:
        return [
            PartitionField("timestamp", "hour", "feature_hour"),
            PartitionField("entity_type", "identity"),
            PartitionField("feature_name", "identity")
        ]


class EmbeddingsDataset(BaseDataset):
    """
    Vector embeddings with versioning and lineage.
    Supports efficient similarity search via external indexes.
    """
    
    def get_table_name(self) -> str:
        return "embeddings"
        
    def get_schema(self) -> List[SchemaField]:
        return [
            # Identification
            SchemaField("embedding_id", "string", False, "Unique embedding ID"),
            SchemaField("entity_id", "string", False, "Entity being embedded"),
            SchemaField("timestamp", "timestamp", False, "Embedding timestamp"),
            
            # Embedding data
            SchemaField("embedding_vector", "list<float>", False, "Embedding vector"),
            SchemaField("embedding_dim", "int", False, "Embedding dimension"),
            SchemaField("model_name", "string", False, "Model that created embedding"),
            SchemaField("model_version", "string", False, "Model version"),
            
            # Quantization support
            SchemaField("quantization_type", "string", True, "Quantization type (binary, int8, etc)"),
            SchemaField("quantized_vector", "binary", True, "Quantized representation"),
            SchemaField("scale_factors", "list<float>", True, "Quantization scale factors"),
            
            # Metadata
            SchemaField("content_hash", "string", True, "Hash of embedded content"),
            SchemaField("content_type", "string", True, "Type of embedded content"),
            SchemaField("metadata", "map<string, string>", True, "Additional metadata"),
            
            # Index hints
            SchemaField("index_partition", "int", True, "Partition for distributed index"),
            SchemaField("centroid_id", "string", True, "Nearest centroid for IVF"),
            SchemaField("hnsw_layer", "int", True, "HNSW graph layer"),
            
            # Lineage
            SchemaField("parent_embedding_id", "string", True, "Parent for incremental updates"),
            SchemaField("transformation_type", "string", True, "Transformation applied"),
            
            # Performance
            SchemaField("inference_time_ms", "long", True, "Time to generate embedding"),
            SchemaField("compression_ratio", "double", True, "Compression achieved")
        ]
        
    def get_partition_spec(self) -> List[PartitionField]:
        return [
            PartitionField("timestamp", "day", "embedding_day"),
            PartitionField("model_name", "identity"),
            PartitionField("index_partition", "identity")
        ]


class TopologyDataset(BaseDataset):
    """
    Topological signatures and persistence diagrams.
    Core dataset for AURA's shape-aware intelligence.
    """
    
    def get_table_name(self) -> str:
        return "topology"
        
    def get_schema(self) -> List[SchemaField]:
        return [
            # Identification
            SchemaField("signature_id", "string", False, "Unique signature ID"),
            SchemaField("entity_id", "string", False, "Entity this signature represents"),
            SchemaField("timestamp", "timestamp", False, "Computation timestamp"),
            
            # Topological data
            SchemaField("persistence_diagram", "string", False, "JSON persistence diagram"),
            SchemaField("betti_numbers", "list<int>", False, "Betti numbers by dimension"),
            SchemaField("persistence_entropy", "double", True, "Persistence entropy"),
            SchemaField("wasserstein_distances", "map<string, double>", True, "Distances to landmarks"),
            
            # Persistence features
            SchemaField("birth_times", "list<double>", True, "Birth times of features"),
            SchemaField("death_times", "list<double>", True, "Death times of features"),
            SchemaField("lifetimes", "list<double>", True, "Feature lifetimes"),
            SchemaField("persistence_landscape", "list<list<double>>", True, "Persistence landscape"),
            
            # Computation metadata
            SchemaField("algorithm", "string", False, "TDA algorithm used"),
            SchemaField("algorithm_params", "map<string, string>", True, "Algorithm parameters"),
            SchemaField("computation_time_ms", "long", False, "Computation time"),
            SchemaField("data_points", "int", False, "Number of input points"),
            
            # Shape classification
            SchemaField("shape_class", "string", True, "Classified shape type"),
            SchemaField("shape_confidence", "double", True, "Classification confidence"),
            SchemaField("anomaly_score", "double", True, "Topological anomaly score"),
            
            # Relationships
            SchemaField("parent_signature_id", "string", True, "Parent for hierarchical TDA"),
            SchemaField("similar_signatures", "list<string>", True, "Similar signatures"),
            
            # Storage optimization
            SchemaField("compressed_diagram", "binary", True, "Compressed representation"),
            SchemaField("diagram_size_bytes", "long", False, "Size of diagram")
        ]
        
    def get_partition_spec(self) -> List[PartitionField]:
        return [
            PartitionField("timestamp", "day", "topo_day"),
            PartitionField("algorithm", "identity"),
            PartitionField("shape_class", "identity")
        ]


class AuditDataset(BaseDataset):
    """
    Immutable audit log with WORM compliance.
    Protected by S3 Object Lock for regulatory requirements.
    """
    
    def get_table_name(self) -> str:
        return "audit"
        
    def get_schema(self) -> List[SchemaField]:
        return [
            # Core audit fields
            SchemaField("audit_id", "string", False, "Unique audit ID"),
            SchemaField("timestamp", "timestamp", False, "Audit timestamp"),
            SchemaField("action", "string", False, "Action performed"),
            SchemaField("resource_type", "string", False, "Type of resource accessed"),
            SchemaField("resource_id", "string", False, "Resource identifier"),
            
            # Actor information
            SchemaField("actor_id", "string", False, "Who performed the action"),
            SchemaField("actor_type", "string", False, "User, service, or system"),
            SchemaField("actor_role", "string", True, "Role at time of action"),
            SchemaField("delegation_chain", "list<string>", True, "Delegation chain"),
            
            # Request context
            SchemaField("request_id", "string", False, "Request correlation ID"),
            SchemaField("session_id", "string", True, "Session identifier"),
            SchemaField("ip_address", "string", True, "Client IP address"),
            SchemaField("user_agent", "string", True, "Client user agent"),
            
            # Action details
            SchemaField("operation", "string", False, "Specific operation"),
            SchemaField("parameters", "map<string, string>", True, "Operation parameters"),
            SchemaField("old_value", "string", True, "Previous value (encrypted)"),
            SchemaField("new_value", "string", True, "New value (encrypted)"),
            
            # Result
            SchemaField("result", "string", False, "Success, failure, or partial"),
            SchemaField("error_code", "string", True, "Error code if failed"),
            SchemaField("error_message", "string", True, "Error message"),
            
            # Compliance
            SchemaField("compliance_labels", "list<string>", True, "GDPR, HIPAA, etc"),
            SchemaField("data_classification", "string", True, "Data classification level"),
            SchemaField("retention_years", "int", False, "Required retention period"),
            SchemaField("legal_hold", "boolean", False, "Under legal hold"),
            
            # Integrity
            SchemaField("event_hash", "string", False, "SHA-256 of event data"),
            SchemaField("previous_hash", "string", True, "Hash chain for integrity"),
            SchemaField("signature", "string", True, "Digital signature"),
            
            # WORM compliance
            SchemaField("object_lock_mode", "string", True, "GOVERNANCE or COMPLIANCE"),
            SchemaField("object_lock_until", "timestamp", True, "Lock expiration"),
            SchemaField("immutable", "boolean", False, "Immutability flag")
        ]
        
    def get_partition_spec(self) -> List[PartitionField]:
        return [
            PartitionField("timestamp", "month", "audit_month"),  # Monthly for long retention
            PartitionField("resource_type", "identity"),
            PartitionField("actor_type", "identity")
        ]
        
    def get_table_properties(self) -> Dict[str, str]:
        props = super().get_table_properties()
        props.update({
            'write.object-storage.enabled': 'true',
            'write.object-storage.path': 's3://aura-lakehouse-audit/audit/objects/',
            's3.object-lock.enabled': 'true',
            's3.object-lock.mode': 'COMPLIANCE',
            's3.object-lock.retention-days': '2555',  # 7 years
            'table.immutable': 'true'  # No updates or deletes allowed
        })
        return props