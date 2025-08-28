"""
Query Builder - Unified Query Interface
======================================
Provides a consistent query API across all store types
with automatic translation to backend-specific queries.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from datetime import datetime
import json


class FilterOperator(Enum):
    """Supported filter operators"""
    EQ = "eq"              # Equal
    NEQ = "neq"            # Not equal
    GT = "gt"              # Greater than
    GTE = "gte"            # Greater than or equal
    LT = "lt"              # Less than
    LTE = "lte"            # Less than or equal
    IN = "in"              # In list
    NOT_IN = "not_in"      # Not in list
    CONTAINS = "contains"   # String contains
    STARTS_WITH = "starts_with"  # String starts with
    ENDS_WITH = "ends_with"      # String ends with
    EXISTS = "exists"      # Field exists
    NOT_EXISTS = "not_exists"    # Field doesn't exist
    REGEX = "regex"        # Regular expression match
    
    # Vector operations
    SIMILAR_TO = "similar_to"    # Vector similarity
    WITHIN_RADIUS = "within_radius"  # Distance threshold
    
    # Graph operations
    CONNECTED_TO = "connected_to"    # Graph connection
    PATH_EXISTS = "path_exists"      # Path between nodes
    
    # Time-series operations
    BETWEEN_TIME = "between_time"    # Time range
    SINCE = "since"                  # Time since
    UNTIL = "until"                  # Time until


class SortOrder(Enum):
    """Sort order options"""
    ASC = "asc"
    DESC = "desc"


class AggregationFunction(Enum):
    """Aggregation functions for queries"""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    PERCENTILE = "percentile"
    
    # Time-series specific
    RATE = "rate"
    DERIVATIVE = "derivative"
    INTEGRAL = "integral"
    
    # Graph specific
    DEGREE = "degree"
    CENTRALITY = "centrality"
    CLUSTERING = "clustering"


@dataclass
class FilterCondition:
    """Single filter condition"""
    field: str
    operator: FilterOperator
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'field': self.field,
            'operator': self.operator.value,
            'value': self.value
        }


@dataclass
class SortSpec:
    """Sort specification"""
    field: str
    order: SortOrder = SortOrder.ASC
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'field': self.field,
            'order': self.order.value
        }


@dataclass
class AggregationSpec:
    """Aggregation specification"""
    function: AggregationFunction
    field: Optional[str] = None
    alias: Optional[str] = None
    params: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'function': self.function.value,
            'field': self.field,
            'alias': self.alias,
            'params': self.params
        }


class QueryBuilder:
    """
    Fluent interface for building queries across different store types.
    Automatically translates to backend-specific query languages.
    """
    
    def __init__(self):
        # Core query components
        self.filters: List[FilterCondition] = []
        self.sort_specs: List[SortSpec] = []
        self.aggregations: List[AggregationSpec] = []
        
        # Pagination
        self.limit: int = 100
        self.cursor: Optional[str] = None
        self.offset: Optional[int] = None
        
        # Projections
        self.select_fields: Optional[List[str]] = None
        self.exclude_fields: Optional[List[str]] = None
        
        # Grouping
        self.group_by: Optional[List[str]] = None
        
        # Vector search
        self.vector_search: Optional[Dict[str, Any]] = None
        
        # Graph traversal
        self.graph_traversal: Optional[Dict[str, Any]] = None
        
        # Time-series
        self.time_window: Optional[Dict[str, Any]] = None
        
        # Raw query escape hatch
        self.raw_query: Optional[str] = None
        
    # Filtering methods
    
    def filter(self, field: str, operator: FilterOperator, value: Any) -> 'QueryBuilder':
        """Add a filter condition"""
        self.filters.append(FilterCondition(field, operator, value))
        return self
        
    def eq(self, field: str, value: Any) -> 'QueryBuilder':
        """Equal filter"""
        return self.filter(field, FilterOperator.EQ, value)
        
    def neq(self, field: str, value: Any) -> 'QueryBuilder':
        """Not equal filter"""
        return self.filter(field, FilterOperator.NEQ, value)
        
    def gt(self, field: str, value: Any) -> 'QueryBuilder':
        """Greater than filter"""
        return self.filter(field, FilterOperator.GT, value)
        
    def gte(self, field: str, value: Any) -> 'QueryBuilder':
        """Greater than or equal filter"""
        return self.filter(field, FilterOperator.GTE, value)
        
    def lt(self, field: str, value: Any) -> 'QueryBuilder':
        """Less than filter"""
        return self.filter(field, FilterOperator.LT, value)
        
    def lte(self, field: str, value: Any) -> 'QueryBuilder':
        """Less than or equal filter"""
        return self.filter(field, FilterOperator.LTE, value)
        
    def in_(self, field: str, values: List[Any]) -> 'QueryBuilder':
        """In list filter"""
        return self.filter(field, FilterOperator.IN, values)
        
    def not_in(self, field: str, values: List[Any]) -> 'QueryBuilder':
        """Not in list filter"""
        return self.filter(field, FilterOperator.NOT_IN, values)
        
    def contains(self, field: str, value: str) -> 'QueryBuilder':
        """String contains filter"""
        return self.filter(field, FilterOperator.CONTAINS, value)
        
    def starts_with(self, field: str, value: str) -> 'QueryBuilder':
        """String starts with filter"""
        return self.filter(field, FilterOperator.STARTS_WITH, value)
        
    def exists(self, field: str) -> 'QueryBuilder':
        """Field exists filter"""
        return self.filter(field, FilterOperator.EXISTS, True)
        
    def regex(self, field: str, pattern: str) -> 'QueryBuilder':
        """Regular expression filter"""
        return self.filter(field, FilterOperator.REGEX, pattern)
        
    # Sorting methods
    
    def sort(self, field: str, order: SortOrder = SortOrder.ASC) -> 'QueryBuilder':
        """Add sort specification"""
        self.sort_specs.append(SortSpec(field, order))
        return self
        
    def sort_asc(self, field: str) -> 'QueryBuilder':
        """Sort ascending"""
        return self.sort(field, SortOrder.ASC)
        
    def sort_desc(self, field: str) -> 'QueryBuilder':
        """Sort descending"""
        return self.sort(field, SortOrder.DESC)
        
    # Aggregation methods
    
    def aggregate(self, 
                 function: AggregationFunction,
                 field: Optional[str] = None,
                 alias: Optional[str] = None,
                 **params) -> 'QueryBuilder':
        """Add aggregation"""
        self.aggregations.append(AggregationSpec(function, field, alias, params))
        return self
        
    def count(self, alias: str = "count") -> 'QueryBuilder':
        """Count aggregation"""
        return self.aggregate(AggregationFunction.COUNT, alias=alias)
        
    def sum(self, field: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """Sum aggregation"""
        return self.aggregate(AggregationFunction.SUM, field, alias)
        
    def avg(self, field: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """Average aggregation"""
        return self.aggregate(AggregationFunction.AVG, field, alias)
        
    def group(self, *fields: str) -> 'QueryBuilder':
        """Group by fields"""
        self.group_by = list(fields)
        return self
        
    # Pagination methods
    
    def with_limit(self, limit: int) -> 'QueryBuilder':
        """Set result limit"""
        self.limit = limit
        return self
        
    def with_cursor(self, cursor: str) -> 'QueryBuilder':
        """Set pagination cursor"""
        self.cursor = cursor
        return self
        
    def with_offset(self, offset: int) -> 'QueryBuilder':
        """Set offset (alternative to cursor)"""
        self.offset = offset
        return self
        
    # Projection methods
    
    def select(self, *fields: str) -> 'QueryBuilder':
        """Select specific fields"""
        self.select_fields = list(fields)
        return self
        
    def exclude(self, *fields: str) -> 'QueryBuilder':
        """Exclude specific fields"""
        self.exclude_fields = list(fields)
        return self
        
    # Vector search methods
    
    def similar_to(self, 
                  embedding: List[float],
                  limit: int = 10,
                  score_threshold: Optional[float] = None) -> 'QueryBuilder':
        """Vector similarity search"""
        self.vector_search = {
            'embedding': embedding,
            'limit': limit,
            'score_threshold': score_threshold
        }
        self.limit = limit
        return self
        
    def within_radius(self,
                     embedding: List[float],
                     radius: float) -> 'QueryBuilder':
        """Vector radius search"""
        self.vector_search = {
            'embedding': embedding,
            'radius': radius
        }
        return self
        
    # Graph traversal methods
    
    def traverse(self,
                start_node: str,
                max_depth: int = 3,
                edge_types: Optional[List[str]] = None) -> 'QueryBuilder':
        """Graph traversal"""
        self.graph_traversal = {
            'start_node': start_node,
            'max_depth': max_depth,
            'edge_types': edge_types
        }
        return self
        
    def connected_to(self, node_id: str) -> 'QueryBuilder':
        """Filter nodes connected to specific node"""
        return self.filter('_connected_to', FilterOperator.CONNECTED_TO, node_id)
        
    # Time-series methods
    
    def between_time(self,
                    start_time: datetime,
                    end_time: datetime) -> 'QueryBuilder':
        """Time range filter"""
        self.time_window = {
            'start_time': start_time,
            'end_time': end_time
        }
        return self
        
    def since(self, time: datetime) -> 'QueryBuilder':
        """Time since filter"""
        self.time_window = {
            'start_time': time,
            'end_time': None
        }
        return self
        
    def last(self, duration: str) -> 'QueryBuilder':
        """Last duration (e.g. '1h', '7d')"""
        # Parse duration and convert to time window
        # This is simplified - real implementation would parse properly
        self.time_window = {
            'duration': duration
        }
        return self
        
    # Raw query
    
    def raw(self, query: str) -> 'QueryBuilder':
        """Set raw query for backend-specific needs"""
        self.raw_query = query
        return self
        
    # Building methods
    
    def build(self) -> Dict[str, Any]:
        """Build query as dictionary"""
        query = {}
        
        # Add filters
        if self.filters:
            query['filters'] = [f.to_dict() for f in self.filters]
            
        # Add sorting
        if self.sort_specs:
            query['sort'] = [s.to_dict() for s in self.sort_specs]
            
        # Add aggregations
        if self.aggregations:
            query['aggregations'] = [a.to_dict() for a in self.aggregations]
            
        # Add grouping
        if self.group_by:
            query['group_by'] = self.group_by
            
        # Add pagination
        query['limit'] = self.limit
        if self.cursor:
            query['cursor'] = self.cursor
        elif self.offset is not None:
            query['offset'] = self.offset
            
        # Add projections
        if self.select_fields:
            query['select'] = self.select_fields
        if self.exclude_fields:
            query['exclude'] = self.exclude_fields
            
        # Add special query types
        if self.vector_search:
            query['vector_search'] = self.vector_search
        if self.graph_traversal:
            query['graph_traversal'] = self.graph_traversal
        if self.time_window:
            query['time_window'] = self.time_window
            
        # Add raw query if specified
        if self.raw_query:
            query['raw'] = self.raw_query
            
        return query
        
    def to_json(self) -> str:
        """Build query as JSON string"""
        return json.dumps(self.build(), default=str)
        
    def __repr__(self) -> str:
        """String representation"""
        return f"QueryBuilder({self.to_json()})"


# Convenience factory functions

def query() -> QueryBuilder:
    """Create a new query builder"""
    return QueryBuilder()
    
def vector_query(embedding: List[float], limit: int = 10) -> QueryBuilder:
    """Create a vector similarity query"""
    return QueryBuilder().similar_to(embedding, limit)
    
def time_query(start: datetime, end: datetime) -> QueryBuilder:
    """Create a time-series query"""
    return QueryBuilder().between_time(start, end)
    
def graph_query(start_node: str, max_depth: int = 3) -> QueryBuilder:
    """Create a graph traversal query"""
    return QueryBuilder().traverse(start_node, max_depth)