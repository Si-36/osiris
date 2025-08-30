"""
Event Store Implementation
=========================
Event sourcing store with projections and subscriptions.
Provides immutable event log with CQRS support.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, AsyncIterator, Tuple
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import json
import logging
from enum import Enum

from ..core import (
    AbstractStore,
    StoreType,
    QueryResult,
    WriteResult,
    TransactionContext,
    ConnectionConfig
)

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    SNAPSHOT = "snapshot"
    CUSTOM = "custom"


@dataclass
class EventStoreConfig(ConnectionConfig):
    """Configuration for event stores"""
    stream_name: str = "aura_events"
    
    # Event sourcing settings
    enable_snapshots: bool = True
    snapshot_frequency: int = 100
    
    # Projections
    enable_projections: bool = True
    projection_lag_ms: int = 100
    
    # Subscriptions
    enable_subscriptions: bool = True
    max_subscribers: int = 100
    
    # Retention
    event_retention_days: int = 365
    snapshot_retention_count: int = 5


@dataclass
class Event:
    """Immutable event in the event store"""
    event_id: str
    stream_id: str
    event_type: EventType
    event_version: int
    
    # Event data
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Causation/correlation
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'stream_id': self.stream_id,
            'event_type': self.event_type.value,
            'event_version': self.event_version,
            'data': self.data,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id
        }


@dataclass
class EventStream:
    """Stream of related events"""
    stream_id: str
    stream_type: str
    version: int = 0
    
    # Events in the stream
    events: List[Event] = field(default_factory=list)
    
    # Latest snapshot
    snapshot: Optional[Dict[str, Any]] = None
    snapshot_version: Optional[int] = None
    
    def append_event(self, event: Event):
        """Append event to stream"""
        event.event_version = self.version + 1
        self.events.append(event)
        self.version = event.event_version


class EventSubscription:
    """Subscription to event stream"""
    
    def __init__(self,
                 subscription_id: str,
                 stream_pattern: str,
                 callback: Callable[[Event], None],
                 start_from: Optional[int] = None):
        self.subscription_id = subscription_id
        self.stream_pattern = stream_pattern
        self.callback = callback
        self.start_from = start_from or 0
        self.current_position = start_from or 0
        self._task: Optional[asyncio.Task] = None
        
    async def process_event(self, event: Event):
        """Process single event"""
        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(event)
            else:
                self.callback(event)
                
            self.current_position = event.event_version
            
        except Exception as e:
            logger.error(f"Subscription {self.subscription_id} error: {e}")


class UnifiedEventStore(AbstractStore[str, Event]):
    """
    Event sourcing store with CQRS support.
    Provides immutable event log with projections.
    """
    
    def __init__(self, config: EventStoreConfig):
        super().__init__(StoreType.EVENT, config.__dict__)
        self.event_config = config
        
        # In-memory storage (would be persistent in production)
        self._streams: Dict[str, EventStream] = {}
        self._global_events: List[Event] = []
        
        # Subscriptions
        self._subscriptions: Dict[str, EventSubscription] = {}
        
        # Projections
        self._projections: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize event store"""
        self._initialized = True
        logger.info(f"Event store initialized: {self.event_config.stream_name}")
        
    async def health_check(self) -> Dict[str, Any]:
        """Check store health"""
        return {
            'healthy': self._initialized,
            'total_streams': len(self._streams),
            'total_events': len(self._global_events),
            'active_subscriptions': len(self._subscriptions)
        }
        
    async def close(self) -> None:
        """Close event store"""
        # Cancel all subscriptions
        for sub in self._subscriptions.values():
            if sub._task:
                sub._task.cancel()
                
        self._initialized = False
        
    async def append_events(self,
                          stream_id: str,
                          events: List[Dict[str, Any]],
                          expected_version: Optional[int] = None) -> WriteResult:
        """Append events to stream with optimistic concurrency"""
        try:
            # Get or create stream
            if stream_id not in self._streams:
                self._streams[stream_id] = EventStream(
                    stream_id=stream_id,
                    stream_type="default"
                )
                
            stream = self._streams[stream_id]
            
            # Check expected version (optimistic concurrency)
            if expected_version is not None and stream.version != expected_version:
                return WriteResult(
                    success=False,
                    error=f"Version mismatch: expected {expected_version}, got {stream.version}"
                )
                
            # Append events
            for event_data in events:
                event = Event(
                    event_id=f"{stream_id}:{stream.version + 1}",
                    stream_id=stream_id,
                    event_type=EventType(event_data.get('type', 'custom')),
                    event_version=0,  # Will be set by stream
                    data=event_data.get('data', {}),
                    metadata=event_data.get('metadata', {}),
                    correlation_id=event_data.get('correlation_id'),
                    causation_id=event_data.get('causation_id')
                )
                
                stream.append_event(event)
                self._global_events.append(event)
                
                # Notify subscribers
                await self._notify_subscribers(event)
                
            # Create snapshot if needed
            if (self.event_config.enable_snapshots and
                stream.version % self.event_config.snapshot_frequency == 0):
                await self._create_snapshot(stream)
                
            return WriteResult(
                success=True,
                id=stream_id,
                version=stream.version,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to append events: {e}")
            return WriteResult(success=False, error=str(e))
            
    async def get_events(self,
                        stream_id: str,
                        from_version: int = 0,
                        to_version: Optional[int] = None) -> List[Event]:
        """Get events from stream"""
        if stream_id not in self._streams:
            return []
            
        stream = self._streams[stream_id]
        events = []
        
        for event in stream.events:
            if event.event_version >= from_version:
                if to_version is None or event.event_version <= to_version:
                    events.append(event)
                    
        return events
        
    async def get_stream_version(self, stream_id: str) -> int:
        """Get current stream version"""
        if stream_id in self._streams:
            return self._streams[stream_id].version
        return 0
        
    async def subscribe(self,
                       subscription_id: str,
                       stream_pattern: str,
                       callback: Callable[[Event], None],
                       start_from: Optional[int] = None) -> str:
        """Subscribe to event stream"""
        if len(self._subscriptions) >= self.event_config.max_subscribers:
            raise RuntimeError("Max subscribers reached")
            
        subscription = EventSubscription(
            subscription_id=subscription_id,
            stream_pattern=stream_pattern,
            callback=callback,
            start_from=start_from
        )
        
        self._subscriptions[subscription_id] = subscription
        
        # Start processing existing events
        subscription._task = asyncio.create_task(
            self._process_subscription(subscription)
        )
        
        logger.info(f"Created subscription: {subscription_id}")
        return subscription_id
        
    async def unsubscribe(self, subscription_id: str):
        """Cancel subscription"""
        if subscription_id in self._subscriptions:
            sub = self._subscriptions[subscription_id]
            if sub._task:
                sub._task.cancel()
                
            del self._subscriptions[subscription_id]
            logger.info(f"Cancelled subscription: {subscription_id}")
            
    async def _notify_subscribers(self, event: Event):
        """Notify all matching subscribers"""
        for sub in self._subscriptions.values():
            # Simple pattern matching (would be more sophisticated)
            if (sub.stream_pattern == "*" or
                sub.stream_pattern == event.stream_id or
                event.stream_id.startswith(sub.stream_pattern.rstrip('*'))):
                
                await sub.process_event(event)
                
    async def _process_subscription(self, subscription: EventSubscription):
        """Process events for subscription"""
        try:
            # Process historical events
            for event in self._global_events:
                if event.event_version > subscription.current_position:
                    await subscription.process_event(event)
                    
            # Wait for new events (in production would use proper event bus)
            while True:
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
            
    async def _create_snapshot(self, stream: EventStream):
        """Create stream snapshot"""
        # Aggregate all events to create current state
        state = {}
        
        for event in stream.events:
            # Apply event to state (simplified)
            if event.event_type == EventType.CREATED:
                state.update(event.data)
            elif event.event_type == EventType.UPDATED:
                state.update(event.data)
            elif event.event_type == EventType.DELETED:
                state.clear()
                
        stream.snapshot = state
        stream.snapshot_version = stream.version
        
        logger.debug(f"Created snapshot for stream {stream.stream_id} at version {stream.version}")
        
    # Implement required abstract methods
    
    async def upsert(self, key: str, value: Event, context: Optional[TransactionContext] = None) -> WriteResult:
        """Not applicable - use append_events"""
        return WriteResult(success=False, error="Use append_events for event store")
        
    async def get(self, key: str, context: Optional[TransactionContext] = None) -> Optional[Event]:
        """Get single event by ID"""
        for event in self._global_events:
            if event.event_id == key:
                return event
        return None
        
    async def list(self, filter_dict: Optional[Dict[str, Any]] = None, limit: int = 100,
                   cursor: Optional[str] = None, context: Optional[TransactionContext] = None) -> QueryResult[Event]:
        """List events with filtering"""
        start_idx = int(cursor) if cursor else 0
        
        # Filter events
        filtered = self._global_events
        if filter_dict:
            if 'stream_id' in filter_dict:
                filtered = [e for e in filtered if e.stream_id == filter_dict['stream_id']]
            if 'event_type' in filter_dict:
                filtered = [e for e in filtered if e.event_type.value == filter_dict['event_type']]
                
        # Paginate
        data = filtered[start_idx:start_idx + limit]
        next_cursor = str(start_idx + limit) if len(filtered) > start_idx + limit else None
        
        return QueryResult(
            success=True,
            data=data,
            total_count=len(filtered),
            next_cursor=next_cursor
        )
        
    async def delete(self, key: str, context: Optional[TransactionContext] = None) -> WriteResult:
        """Events are immutable - cannot delete"""
        return WriteResult(success=False, error="Events are immutable")
        
    async def batch_upsert(self, items: List[Tuple[str, Event]], context: Optional[TransactionContext] = None) -> List[WriteResult]:
        """Not applicable - use append_events"""
        return [WriteResult(success=False, error="Use append_events") for _ in items]
        
    async def batch_get(self, keys: List[str], context: Optional[TransactionContext] = None) -> Dict[str, Optional[Event]]:
        """Batch get events"""
        result = {}
        for key in keys:
            result[key] = await self.get(key, context)
        return result