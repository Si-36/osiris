"""
Store Registry
=============
Central registry for all persistence stores.
Provides unified access and lifecycle management.
"""

import asyncio
from typing import Dict, Type, Optional, Any, List
from dataclasses import dataclass
import logging

from ..core import (
    AbstractStore,
    StoreType,
    ConnectionConfig,
    VectorStore,
    GraphStore,
    TimeSeriesStore
)

from .kv import NATSKVStore, KVConfig
from .vector import (
    QdrantVectorStore,
    ClickHouseVectorStore,
    PgVectorStore,
    VectorIndexConfig
)
from .timeseries import (
    InfluxDB3Store,
    QuestDBStore,
    TimeSeriesConfig
)
from .graph import Neo4jGraphStore, GraphConfig
from .document import UnifiedDocumentStore, DocumentConfig
from .event import UnifiedEventStore, EventStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class StoreRegistration:
    """Registration entry for a store"""
    store_type: StoreType
    store_class: Type[AbstractStore]
    config_class: Type[ConnectionConfig]
    instance: Optional[AbstractStore] = None
    

class StoreRegistry:
    """
    Central registry for all persistence stores.
    Manages store lifecycle and provides unified access.
    """
    
    def __init__(self):
        # Store type to implementation mapping
        self._store_implementations: Dict[str, StoreRegistration] = {
            # KV stores
            'nats_kv': StoreRegistration(
                StoreType.KV,
                NATSKVStore,
                KVConfig
            ),
            
            # Vector stores
            'qdrant': StoreRegistration(
                StoreType.VECTOR,
                QdrantVectorStore,
                ConnectionConfig
            ),
            'clickhouse_vector': StoreRegistration(
                StoreType.VECTOR,
                ClickHouseVectorStore,
                ConnectionConfig
            ),
            'pgvector': StoreRegistration(
                StoreType.VECTOR,
                PgVectorStore,
                ConnectionConfig
            ),
            
            # Time-series stores
            'influxdb3': StoreRegistration(
                StoreType.TIMESERIES,
                InfluxDB3Store,
                TimeSeriesConfig
            ),
            'questdb': StoreRegistration(
                StoreType.TIMESERIES,
                QuestDBStore,
                TimeSeriesConfig
            ),
            
            # Graph stores
            'neo4j': StoreRegistration(
                StoreType.GRAPH,
                Neo4jGraphStore,
                GraphConfig
            ),
            
            # Document stores
            'document': StoreRegistration(
                StoreType.DOCUMENT,
                UnifiedDocumentStore,
                DocumentConfig
            ),
            
            # Event stores
            'event': StoreRegistration(
                StoreType.EVENT,
                UnifiedEventStore,
                EventStoreConfig
            )
        }
        
        # Active store instances
        self._stores: Dict[str, AbstractStore] = {}
        
        # Default stores by type
        self._default_stores: Dict[StoreType, str] = {
            StoreType.KV: 'nats_kv',
            StoreType.VECTOR: 'qdrant',
            StoreType.TIMESERIES: 'influxdb3',
            StoreType.GRAPH: 'neo4j',
            StoreType.DOCUMENT: 'document',
            StoreType.EVENT: 'event'
        }
        
    def configure(self, config: Dict[str, Any]):
        """Configure registry from configuration dict"""
        # Set default stores
        if 'defaults' in config:
            for store_type, impl_name in config['defaults'].items():
                self._default_stores[StoreType(store_type)] = impl_name
                
        # Configure specific stores
        if 'stores' in config:
            for store_name, store_config in config['stores'].items():
                # Would create and configure stores
                pass
                
    async def get_store(self,
                       store_name: Optional[str] = None,
                       store_type: Optional[StoreType] = None,
                       config: Optional[Dict[str, Any]] = None) -> AbstractStore:
        """Get or create a store instance"""
        # Determine store name
        if not store_name:
            if not store_type:
                raise ValueError("Either store_name or store_type must be provided")
            store_name = self._default_stores.get(store_type)
            if not store_name:
                raise ValueError(f"No default store for type {store_type}")
                
        # Check if already exists
        if store_name in self._stores:
            return self._stores[store_name]
            
        # Create new instance
        if store_name not in self._store_implementations:
            raise ValueError(f"Unknown store: {store_name}")
            
        registration = self._store_implementations[store_name]
        
        # Create config
        if config:
            store_config = registration.config_class(**config)
        else:
            store_config = registration.config_class()
            
        # Create store
        store = registration.store_class(store_config)
        
        # Initialize
        await store.initialize()
        
        # Cache instance
        self._stores[store_name] = store
        registration.instance = store
        
        logger.info(f"Created store: {store_name} ({registration.store_type.value})")
        
        return store
        
    async def get_vector_store(self,
                             store_name: Optional[str] = None,
                             config: Optional[Dict[str, Any]] = None) -> VectorStore:
        """Get a vector store"""
        store = await self.get_store(store_name or 'qdrant', StoreType.VECTOR, config)
        
        if not isinstance(store, VectorStore):
            raise TypeError(f"Store {store_name} is not a VectorStore")
            
        return store
        
    async def get_graph_store(self,
                            store_name: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None) -> GraphStore:
        """Get a graph store"""
        store = await self.get_store(store_name or 'neo4j', StoreType.GRAPH, config)
        
        if not isinstance(store, GraphStore):
            raise TypeError(f"Store {store_name} is not a GraphStore")
            
        return store
        
    async def get_timeseries_store(self,
                                 store_name: Optional[str] = None,
                                 config: Optional[Dict[str, Any]] = None) -> TimeSeriesStore:
        """Get a time-series store"""
        store = await self.get_store(store_name or 'influxdb3', StoreType.TIMESERIES, config)
        
        if not isinstance(store, TimeSeriesStore):
            raise TypeError(f"Store {store_name} is not a TimeSeriesStore")
            
        return store
        
    async def get_kv_store(self,
                         store_name: Optional[str] = None,
                         config: Optional[Dict[str, Any]] = None) -> NATSKVStore:
        """Get a KV store"""
        store = await self.get_store(store_name or 'nats_kv', StoreType.KV, config)
        
        if not isinstance(store, NATSKVStore):
            raise TypeError(f"Store {store_name} is not a KV store")
            
        return store
        
    def list_stores(self) -> Dict[str, Dict[str, Any]]:
        """List all registered stores"""
        result = {}
        
        for name, registration in self._store_implementations.items():
            result[name] = {
                'type': registration.store_type.value,
                'class': registration.store_class.__name__,
                'active': name in self._stores,
                'is_default': self._default_stores.get(registration.store_type) == name
            }
            
        return result
        
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check all active stores"""
        results = {}
        
        for name, store in self._stores.items():
            try:
                health = await store.health_check()
                results[name] = health
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'error': str(e)
                }
                
        return results
        
    async def close_all(self):
        """Close all active stores"""
        tasks = []
        
        for store in self._stores.values():
            tasks.append(store.close())
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._stores.clear()
        
        # Clear instances from registrations
        for registration in self._store_implementations.values():
            registration.instance = None
            
        logger.info("All stores closed")
        
    def get_metrics_all(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all active stores"""
        results = {}
        
        for name, store in self._stores.items():
            try:
                # Note: get_metrics is sync
                metrics = asyncio.run(store.get_metrics())
                results[name] = metrics
            except Exception as e:
                results[name] = {'error': str(e)}
                
        return results
        
    def register_store(self,
                      name: str,
                      store_class: Type[AbstractStore],
                      config_class: Type[ConnectionConfig],
                      store_type: StoreType):
        """Register a custom store implementation"""
        if name in self._store_implementations:
            raise ValueError(f"Store {name} already registered")
            
        self._store_implementations[name] = StoreRegistration(
            store_type=store_type,
            store_class=store_class,
            config_class=config_class
        )
        
        logger.info(f"Registered custom store: {name}")


# Global registry instance
_global_registry = StoreRegistry()


# Convenience functions
async def get_store(store_name: Optional[str] = None,
                   store_type: Optional[StoreType] = None,
                   config: Optional[Dict[str, Any]] = None) -> AbstractStore:
    """Get a store from the global registry"""
    return await _global_registry.get_store(store_name, store_type, config)
    
    
def register_store(name: str,
                  store_class: Type[AbstractStore],
                  config_class: Type[ConnectionConfig],
                  store_type: StoreType):
    """Register a custom store in the global registry"""
    _global_registry.register_store(name, store_class, config_class, store_type)