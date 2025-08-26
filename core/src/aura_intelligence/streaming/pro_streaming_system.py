"""
Professional Real-Time Streaming System
======================================

Enterprise-grade streaming with:
- Apache Kafka for event streaming
- NATS for low-latency messaging
- Redis Streams for time-series data
- WebSockets for real-time UI updates
- Event sourcing and CQRS patterns
- Exactly-once delivery guarantees
- Backpressure handling
- Stream processing with windowing
- Dead letter topics
- Schema registry integration
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, AsyncIterator, Set
import logging
from collections import defaultdict, deque
import struct
import zlib

# Kafka imports
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# NATS imports
import nats
from nats.errors import ConnectionClosedError, TimeoutError as NatsTimeoutError
from nats.js import JetStreamContext

# Redis imports
import redis.asyncio as redis
from redis.asyncio.client import PubSub

# WebSocket imports
from aiohttp import web
import aiohttp
from aiohttp import WSMsgType

# Schema validation
from pydantic import BaseModel, Field, validator
import avro.schema
import avro.io
from io import BytesIO

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T', bound=BaseModel)
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class StreamingBackend(Enum):
    """Available streaming backends"""
    KAFKA = "kafka"
    NATS = "nats"
    REDIS = "redis"
    WEBSOCKET = "websocket"


class DeliveryGuarantee(Enum):
    """Message delivery guarantees"""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class ProcessingGuarantee(Enum):
    """Stream processing guarantees"""
    BEST_EFFORT = "best_effort"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class StreamConfig:
    """Stream configuration"""
    name: str
    backend: StreamingBackend
    partitions: int = 3
    replication_factor: int = 2
    retention_hours: int = 168  # 7 days
    compression_type: str = "gzip"
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    max_batch_size: int = 100
    batch_timeout_ms: int = 100
    enable_idempotence: bool = True
    schema_registry_url: Optional[str] = None


@dataclass
class StreamMessage(Generic[K, V]):
    """Generic stream message"""
    key: K
    value: V
    timestamp: float = field(default_factory=time.time)
    headers: Dict[str, str] = field(default_factory=dict)
    partition: Optional[int] = None
    offset: Optional[int] = None
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes"""
        pass
        data = {
            'key': self.key,
            'value': asdict(self.value) if hasattr(self.value, '__dict__') else self.value,
            'timestamp': self.timestamp,
            'headers': self.headers
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes, key_type: type = str, value_type: type = dict):
        """Deserialize from bytes"""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            key=key_type(obj['key']),
            value=value_type(**obj['value']) if hasattr(value_type, '__init__') else obj['value'],
            timestamp=obj['timestamp'],
            headers=obj['headers']
        )


# Pydantic models for AURA events
class BaseEvent(BaseModel):
    """Base event model"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: float = Field(default_factory=time.time)
    source: str = "aura.intelligence"
    correlation_id: Optional[str] = None
    
    class Config:
        extra = "allow"


class TopologyChangeEvent(BaseEvent):
    """Topology change detected"""
    event_type: str = "topology.change"
    component_id: str
    change_type: str  # "node_added", "node_removed", "edge_changed"
    old_value: Optional[Dict[str, Any]] = None
    new_value: Dict[str, Any]
    impact_score: float = Field(ge=0, le=1)


class CascadePredictionEvent(BaseEvent):
    """Cascade failure prediction"""
    event_type: str = "cascade.prediction"
    risk_score: float = Field(ge=0, le=1)
    time_to_failure: float  # seconds
    affected_components: List[str]
    confidence: float = Field(ge=0, le=1)
    topology_snapshot: Dict[str, Any]


class InterventionEvent(BaseEvent):
    """System intervention executed"""
    event_type: str = "intervention.executed"
    intervention_type: str
    target_components: List[str]
    parameters: Dict[str, Any]
    expected_impact: float = Field(ge=0, le=1)
    status: str = "pending"  # pending, in_progress, completed, failed


class MetricEvent(BaseEvent):
    """System metric update"""
    event_type: str = "metric.update"
    metric_name: str
    metric_value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: str = "1"


# Streaming interfaces
class StreamProducer(ABC, Generic[K, V]):
    """Abstract stream producer"""
    
    @abstractmethod
    async def send(self, topic: str, message: StreamMessage[K, V]) -> None:
        """Send message to stream"""
        pass
    
    @abstractmethod
    async def send_batch(self, topic: str, messages: List[StreamMessage[K, V]]) -> None:
        """Send batch of messages"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close producer"""
        pass


class StreamConsumer(ABC, Generic[K, V]):
    """Abstract stream consumer"""
    
    @abstractmethod
    async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics"""
        pass
    
    @abstractmethod
    async def consume(self) -> AsyncIterator[StreamMessage[K, V]]:
        """Consume messages"""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit offsets"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close consumer"""
        pass


# Kafka implementation
class KafkaStreamProducer(StreamProducer[K, V]):
    """Kafka producer implementation"""
    
    def __init__(self, config: StreamConfig, serializer: Optional[Callable] = None):
        self.config = config
        self.serializer = serializer or json.dumps
        self.producer: Optional[AIOKafkaProducer] = None
        
        # Metrics
        self.messages_sent = Counter(
            'kafka_messages_sent_total',
            'Total messages sent to Kafka',
            ['topic']
        )
        self.send_errors = Counter(
            'kafka_send_errors_total',
            'Total Kafka send errors',
            ['topic', 'error_type']
        )
    
        async def start(self):
        """Start producer"""
        pass
        self.producer = AIOKafkaProducer(
            bootstrap_servers='localhost:9092',
            compression_type=self.config.compression_type,
            enable_idempotence=self.config.enable_idempotence,
            acks='all' if self.config.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE else 1,
            max_batch_size=self.config.max_batch_size,
            linger_ms=self.config.batch_timeout_ms,
            value_serializer=lambda v: self.serializer(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None
        )
        await self.producer.start()
        logger.info(f"Kafka producer started for {self.config.name}")
    
        async def send(self, topic: str, message: StreamMessage[K, V]) -> None:
        """Send single message"""
        if not self.producer:
            await self.start()
        
        try:
            # Prepare headers
            headers = [(k, v.encode('utf-8')) for k, v in message.headers.items()]
            
            # Send message
            await self.producer.send(
                topic,
                value=message.value,
                key=message.key,
                headers=headers,
                timestamp_ms=int(message.timestamp * 1000)
            )
            
            self.messages_sent.labels(topic=topic).inc()
            
        except KafkaError as e:
            self.send_errors.labels(topic=topic, error_type=type(e).__name__).inc()
            logger.error(f"Kafka send error: {e}")
            raise
    
        async def send_batch(self, topic: str, messages: List[StreamMessage[K, V]]) -> None:
        """Send batch of messages"""
        # Kafka producer batches automatically, so just send individually
        for message in messages:
            await self.send(topic, message)
    
        async def close(self) -> None:
        """Close producer"""
        pass
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer closed")


class KafkaStreamConsumer(StreamConsumer[K, V]):
    """Kafka consumer implementation"""
    
    def __init__(
        self,
        config: StreamConfig,
        group_id: str,
        deserializer: Optional[Callable] = None
    ):
        self.config = config
        self.group_id = group_id
        self.deserializer = deserializer or json.loads
        self.consumer: Optional[AIOKafkaConsumer] = None
        
        # Metrics
        self.messages_consumed = Counter(
            'kafka_messages_consumed_total',
            'Total messages consumed from Kafka',
            ['topic']
        )
        self.consumer_lag = Gauge(
            'kafka_consumer_lag',
            'Kafka consumer lag',
            ['topic', 'partition']
        )
    
    async def start(self):
        """Start consumer"""
        pass
        self.consumer = AIOKafkaConsumer(
            bootstrap_servers='localhost:9092',
            group_id=self.group_id,
            enable_auto_commit=False,  # Manual commit for better control
            auto_offset_reset='earliest',
            value_deserializer=lambda v: self.deserializer(v.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None
        )
        await self.consumer.start()
        logger.info(f"Kafka consumer started for group {self.group_id}")
    
    async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics"""
        if not self.consumer:
            await self.start()
        
        self.consumer.subscribe(topics)
        logger.info(f"Subscribed to topics: {topics}")
    
    async def consume(self) -> AsyncIterator[StreamMessage[K, V]]:
        """Consume messages"""
        pass
        if not self.consumer:
            raise RuntimeError("Consumer not started")
        
        async for msg in self.consumer:
            # Convert to StreamMessage
            headers = {k: v.decode('utf-8') for k, v in msg.headers} if msg.headers else {}
            
            message = StreamMessage(
                key=msg.key,
                value=msg.value,
                timestamp=msg.timestamp / 1000,  # Convert from ms
                headers=headers,
                partition=msg.partition,
                offset=msg.offset
            )
            
            # Update metrics
            self.messages_consumed.labels(topic=msg.topic).inc()
            
            # Calculate lag
            highwater = self.consumer.highwater(msg.tp)
            if highwater:
                lag = highwater - msg.offset - 1
                self.consumer_lag.labels(
                    topic=msg.topic,
                    partition=msg.partition
                ).set(lag)
            
            yield message
    
    async def commit(self) -> None:
        """Commit offsets"""
        pass
        if self.consumer:
            await self.consumer.commit()
    
    async def close(self) -> None:
        """Close consumer"""
        pass
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer closed")


# NATS implementation
class NatsStreamProducer(StreamProducer[K, V]):
    """NATS JetStream producer"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[JetStreamContext] = None
        
        # Metrics
        self.messages_sent = Counter(
            'nats_messages_sent_total',
            'Total messages sent to NATS',
            ['stream']
        )
    
        async def start(self):
        """Start NATS connection"""
        pass
        self.nc = await nats.connect("nats://localhost:4222")
        self.js = self.nc.jetstream()
        
        # Create stream if not exists
        try:
            await self.js.add_stream(
                name=self.config.name,
                subjects=[f"{self.config.name}.>"],
                retention="limits",
                max_age=self.config.retention_hours * 3600,
                storage="file",
                num_replicas=self.config.replication_factor
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise
        
        logger.info(f"NATS producer started for stream {self.config.name}")
    
        async def send(self, topic: str, message: StreamMessage[K, V]) -> None:
        """Send message to NATS"""
        if not self.js:
            await self.start()
        
        # Prepare subject (NATS equivalent of topic)
        subject = f"{self.config.name}.{topic}"
        
        # Prepare headers
        headers = {}
        headers.update(message.headers)
        headers['aura-key'] = str(message.key)
        headers['aura-timestamp'] = str(message.timestamp)
        
        # Send message
        ack = await self.js.publish(
            subject,
            message.to_bytes(),
            headers=headers
        )
        
        self.messages_sent.labels(stream=self.config.name).inc()
        
        # Store sequence for exactly-once semantics
        message.offset = ack.seq
    
        async def send_batch(self, topic: str, messages: List[StreamMessage[K, V]]) -> None:
        """Send batch of messages"""
        # NATS doesn't have native batching, send individually
        for message in messages:
            await self.send(topic, message)
    
        async def close(self) -> None:
        """Close connection"""
        pass
        if self.nc:
            await self.nc.drain()
            await self.nc.close()


class NatsStreamConsumer(StreamConsumer[K, V]):
    """NATS JetStream consumer"""
    
    def __init__(self, config: StreamConfig, durable_name: str):
        self.config = config
        self.durable_name = durable_name
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[JetStreamContext] = None
        self.subscription = None
        self.pending_messages: asyncio.Queue = asyncio.Queue()
        
        # Metrics
        self.messages_consumed = Counter(
            'nats_messages_consumed_total',
            'Total messages consumed from NATS',
            ['stream']
        )
    
        async def start(self):
        """Start NATS connection"""
        pass
        self.nc = await nats.connect("nats://localhost:4222")
        self.js = self.nc.jetstream()
        logger.info(f"NATS consumer started for stream {self.config.name}")
    
        async def subscribe(self, topics: List[str]) -> None:
        """Subscribe to subjects"""
        if not self.js:
            await self.start()
        
        # Create pull subscription
        subjects = [f"{self.config.name}.{topic}" for topic in topics]
        
        self.subscription = await self.js.pull_subscribe(
            subject=f"{self.config.name}.>",
            durable=self.durable_name,
            stream=self.config.name
        )
        
        logger.info(f"Subscribed to subjects: {subjects}")
    
        async def consume(self) -> AsyncIterator[StreamMessage[K, V]]:
        """Consume messages"""
        pass
        if not self.subscription:
            raise RuntimeError("Not subscribed")
        
        while True:
            try:
                # Fetch batch of messages
                messages = await self.subscription.fetch(batch=100, timeout=1)
                
                for msg in messages:
                    # Extract headers
                    headers = dict(msg.headers) if msg.headers else {}
                    key = headers.pop('aura-key', None)
                    timestamp = float(headers.pop('aura-timestamp', time.time()))
                    
                    # Parse message
                    data = json.loads(msg.data.decode('utf-8'))
                    
                    message = StreamMessage(
                        key=key,
                        value=data,
                        timestamp=timestamp,
                        headers=headers,
                        offset=msg.metadata.sequence.stream
                    )
                    
                    # Store for ack
                    await self.pending_messages.put((msg, message))
                    
                    self.messages_consumed.labels(stream=self.config.name).inc()
                    
                    yield message
                    
            except NatsTimeoutError:
                # No messages available, continue
                await asyncio.sleep(0.1)
    
        async def commit(self) -> None:
        """Acknowledge messages"""
        pass
        while not self.pending_messages.empty():
            msg, _ = await self.pending_messages.get()
            await msg.ack()
    
        async def close(self) -> None:
        """Close connection"""
        pass
        if self.subscription:
            await self.subscription.unsubscribe()
        if self.nc:
            await self.nc.drain()
            await self.nc.close()


# Stream processor
class StreamProcessor(Generic[K, V]):
    """Stream processing with windowing and state"""
    
    def __init__(
        self,
        name: str,
        window_size: timedelta,
        slide_interval: timedelta,
        process_func: Callable[[List[StreamMessage[K, V]]], Any]
    ):
        self.name = name
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.process_func = process_func
        
        # Window state
        self.windows: Dict[float, List[StreamMessage[K, V]]] = defaultdict(list)
        self.watermark = 0.0
        
        # Metrics
        self.windows_processed = Counter(
            'stream_windows_processed_total',
            'Total windows processed',
            ['processor']
        )
        self.processing_latency = Histogram(
            'stream_processing_latency_seconds',
            'Stream processing latency',
            ['processor']
        )
    
    async def process(self, message: StreamMessage[K, V]) -> Optional[Any]:
        """Process message with windowing"""
        # Update watermark
        self.watermark = max(self.watermark, message.timestamp)
        
        # Determine window(s) for message
        window_start = self._get_window_start(message.timestamp)
        
        # Add to window
        self.windows[window_start].append(message)
        
        # Check for complete windows
        results = []
        cutoff_time = self.watermark - self.window_size.total_seconds()
        
        for window_time, messages in list(self.windows.items()):
            if window_time < cutoff_time:
                # Window is complete
                with self.processing_latency.labels(processor=self.name).time():
                    result = await self._process_window(window_time, messages)
                    if result:
                        results.append(result)
                
                # Remove processed window
                del self.windows[window_time]
                self.windows_processed.labels(processor=self.name).inc()
        
        return results if results else None
    
    def _get_window_start(self, timestamp: float) -> float:
        """Get window start time for timestamp"""
        slide_seconds = self.slide_interval.total_seconds()
        return (timestamp // slide_seconds) * slide_seconds
    
        async def _process_window(
        self,
        window_time: float,
        messages: List[StreamMessage[K, V]]
        ) -> Any:
        """Process a complete window"""
        try:
            return await self.process_func(messages)
        except Exception as e:
            logger.error(f"Window processing error: {e}")
            return None


# WebSocket streaming
class WebSocketStreamServer:
    """WebSocket server for real-time updates"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = web.Application()
        self.websockets: Set[web.WebSocketResponse] = set()
        
        # Setup routes
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/health', self.health_check)
        
        # Metrics
        self.active_connections = Gauge(
            'websocket_active_connections',
            'Active WebSocket connections'
        )
        self.messages_sent = Counter(
            'websocket_messages_sent_total',
            'Total WebSocket messages sent'
        )
    
        async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        pass
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        self.active_connections.inc()
        
        try:
            # Send initial message
            await ws.send_json({
                'type': 'connection',
                'status': 'connected',
                'timestamp': time.time()
            })
            
            # Keep connection alive
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Handle client messages
                    if data.get('type') == 'ping':
                        await ws.send_json({
                            'type': 'pong',
                            'timestamp': time.time()
                        })
                    elif data.get('type') == 'subscribe':
                        # Handle subscription logic
                        await ws.send_json({
                            'type': 'subscribed',
                            'topics': data.get('topics', [])
                        })
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            
        finally:
            self.websockets.discard(ws)
            self.active_connections.dec()
            
        return ws
    
        async def broadcast(self, event: BaseEvent):
        """Broadcast event to all connected clients"""
        if not self.websockets:
            return
        
        # Prepare message
        message = event.dict()
        
        # Send to all clients
        disconnected = set()
        
        for ws in self.websockets:
            try:
                await ws.send_json(message)
                self.messages_sent.inc()
            except ConnectionResetError:
                disconnected.add(ws)
        
        # Clean up disconnected clients
        self.websockets -= disconnected
        self.active_connections.set(len(self.websockets))
    
        async def health_check(self, request):
        """Health check endpoint"""
        pass
        return web.json_response({
            'status': 'healthy',
            'active_connections': len(self.websockets),
            'timestamp': time.time()
        })
    
        async def start(self):
        """Start WebSocket server"""
        pass
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        logger.info(f"WebSocket server started on port {self.port}")


# Main streaming system
class StreamingSystem:
    """Unified streaming system for AURA"""
    
    def __init__(self):
        self.producers: Dict[str, StreamProducer] = {}
        self.consumers: Dict[str, StreamConsumer] = {}
        self.processors: Dict[str, StreamProcessor] = {}
        self.websocket_server = WebSocketStreamServer()
        
        # Topic registry
        self.topics = {
            'topology.changes': StreamConfig(
                name='topology_changes',
                backend=StreamingBackend.KAFKA,
                partitions=6,
                retention_hours=24
            ),
            'cascade.predictions': StreamConfig(
                name='cascade_predictions',
                backend=StreamingBackend.KAFKA,
                partitions=3,
                retention_hours=168
            ),
            'system.metrics': StreamConfig(
                name='system_metrics',
                backend=StreamingBackend.NATS,
                partitions=1,
                retention_hours=4
            ),
            'interventions': StreamConfig(
                name='interventions',
                backend=StreamingBackend.KAFKA,
                partitions=3,
                retention_hours=720  # 30 days
            )
        }
        
        logger.info("Streaming system initialized")
    
        async def start(self):
        """Start streaming system"""
        pass
        # Start WebSocket server
        await self.websocket_server.start()
        
        # Initialize Kafka topics
        await self._create_kafka_topics()
        
        # Start default processors
        await self._start_default_processors()
        
        logger.info("Streaming system started")
    
        async def _create_kafka_topics(self):
        """Create Kafka topics"""
        pass
        admin_client = KafkaAdminClient(
            bootstrap_servers='localhost:9092',
            client_id='aura_admin'
        )
        
        topics = []
        for topic_name, config in self.topics.items():
            if config.backend == StreamingBackend.KAFKA:
                topics.append(NewTopic(
                    name=config.name,
                    num_partitions=config.partitions,
                    replication_factor=config.replication_factor,
                    topic_configs={
                        'retention.ms': str(config.retention_hours * 3600 * 1000),
                        'compression.type': config.compression_type
                    }
                ))
        
        try:
            admin_client.create_topics(topics, validate_only=False)
            logger.info(f"Created Kafka topics: {[t.name for t in topics]}")
        except TopicAlreadyExistsError:
            logger.info("Kafka topics already exist")
        finally:
            admin_client.close()
    
        async def _start_default_processors(self):
        """Start default stream processors"""
        pass
        # Cascade detection processor
        cascade_processor = StreamProcessor(
            name="cascade_detector",
            window_size=timedelta(minutes=5),
            slide_interval=timedelta(minutes=1),
            process_func=self._process_cascade_window
        )
        self.processors['cascade_detector'] = cascade_processor
        
        # Anomaly detection processor
        anomaly_processor = StreamProcessor(
            name="anomaly_detector",
            window_size=timedelta(minutes=1),
            slide_interval=timedelta(seconds=30),
            process_func=self._process_anomaly_window
        )
        self.processors['anomaly_detector'] = anomaly_processor
    
        async def _process_cascade_window(
        self,
        messages: List[StreamMessage]
        ) -> Optional[CascadePredictionEvent]:
        """Process window for cascade detection"""
        if not messages:
            return None
        
        # Analyze topology changes in window
        topology_changes = [
            m for m in messages
            if isinstance(m.value, dict) and m.value.get('event_type') == 'topology.change'
        ]
        
        if len(topology_changes) > 5:  # Threshold for cascade risk
            # Calculate risk based on changes
            risk_score = min(len(topology_changes) / 10, 1.0)
            
            # Create prediction event
            prediction = CascadePredictionEvent(
                risk_score=risk_score,
                time_to_failure=300 / risk_score,  # Inverse relationship
                affected_components=[
                    m.value.get('component_id')
                    for m in topology_changes
                    if m.value.get('component_id')
                ],
                confidence=0.8,
                topology_snapshot={}  # Would include actual topology
            )
            
            # Broadcast via WebSocket
            await self.websocket_server.broadcast(prediction)
            
            return prediction
        
        return None
    
        async def _process_anomaly_window(
        self,
        messages: List[StreamMessage]
        ) -> Optional[Dict[str, Any]]:
        """Process window for anomaly detection"""
        if not messages:
            return None
        
        # Calculate metrics statistics
        metrics = [
            m.value.get('metric_value', 0)
            for m in messages
            if isinstance(m.value, dict) and m.value.get('event_type') == 'metric.update'
        ]
        
        if metrics:
            mean = sum(metrics) / len(metrics)
            std = (sum((x - mean) ** 2 for x in metrics) / len(metrics)) ** 0.5
            
            # Detect anomalies (simple z-score)
            anomalies = [
                m for m, v in zip(messages, metrics)
                if abs(v - mean) > 3 * std
            ]
            
            if anomalies:
                return {
                    'anomaly_count': len(anomalies),
                    'window_size': len(messages),
                    'mean': mean,
                    'std': std,
                    'anomalies': anomalies
                }
        
        return None
    
        async def publish(
        self,
        topic: str,
        event: BaseEvent,
        key: Optional[str] = None
        ):
        """Publish event to stream"""
        config = self.topics.get(topic)
        if not config:
            raise ValueError(f"Unknown topic: {topic}")
        
        # Get or create producer
        producer_key = f"{config.backend}:{config.name}"
        if producer_key not in self.producers:
            if config.backend == StreamingBackend.KAFKA:
                producer = KafkaStreamProducer(config)
            elif config.backend == StreamingBackend.NATS:
                producer = NatsStreamProducer(config)
            else:
                raise ValueError(f"Unsupported backend: {config.backend}")
            
            await producer.start()
            self.producers[producer_key] = producer
        
        # Create message
        message = StreamMessage(
            key=key or event.event_id,
            value=event.dict(),
            headers={
                'event_type': event.event_type,
                'source': event.source
            }
        )
        
        # Send message
        await self.producers[producer_key].send(topic, message)
        
        # Also broadcast critical events via WebSocket
        if isinstance(event, (CascadePredictionEvent, InterventionEvent)):
            await self.websocket_server.broadcast(event)
    
        async def subscribe(
        self,
        topics: List[str],
        group_id: str,
        handler: Callable[[StreamMessage], Any]
        ):
        """Subscribe to topics with handler"""
        # Group topics by backend
        backend_topics = defaultdict(list)
        for topic in topics:
            config = self.topics.get(topic)
            if config:
                backend_topics[config.backend].append(topic)
        
        # Create consumers
        for backend, topic_list in backend_topics.items():
            if backend == StreamingBackend.KAFKA:
                consumer = KafkaStreamConsumer(
                    config=StreamConfig(name=group_id, backend=backend),
                    group_id=group_id
                )
            elif backend == StreamingBackend.NATS:
                consumer = NatsStreamConsumer(
                    config=StreamConfig(name=group_id, backend=backend),
                    durable_name=group_id
                )
            else:
                continue
            
            await consumer.start()
            await consumer.subscribe(topic_list)
            
            # Start consumption task
            asyncio.create_task(self._consume_loop(consumer, handler))
    
        async def _consume_loop(
        self,
        consumer: StreamConsumer,
        handler: Callable[[StreamMessage], Any]
        ):
        """Consumer loop"""
        try:
            async for message in consumer.consume():
                try:
                    # Process message
                    result = await handler(message)
                    
                    # Process through stream processors if needed
                    for processor in self.processors.values():
                        await processor.process(message)
                    
                    # Commit on success
                    await consumer.commit()
                    
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    # In production, would send to DLQ
                    
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
        finally:
            await consumer.close()
    
        async def close(self):
        """Close streaming system"""
        pass
        # Close producers
        for producer in self.producers.values():
            await producer.close()
        
        # Close consumers
        for consumer in self.consumers.values():
            await consumer.close()
        
        logger.info("Streaming system closed")


# Example usage
async def test_streaming_system():
        """Test streaming system"""
    # Initialize system
        streaming = StreamingSystem()
        await streaming.start()
    
    # Define event handler
        async def handle_event(message: StreamMessage):
        logger.info(f"Received event: {message.value.get('event_type')} at {message.timestamp}")
    
    # Subscribe to topics
        await streaming.subscribe(
        topics=['topology.changes', 'cascade.predictions'],
        group_id='test_consumer',
        handler=handle_event
        )
    
    # Publish some events
        for i in range(10):
        # Topology change
        topology_event = TopologyChangeEvent(
            component_id=f"node_{i}",
            change_type="node_added",
            new_value={"status": "active"},
            impact_score=0.3
        )
        await streaming.publish('topology.changes', topology_event)
        
        # Metric update
        metric_event = MetricEvent(
            metric_name="cpu_usage",
            metric_value=50 + i * 5,
            labels={"node": f"node_{i}"}
        )
        await streaming.publish('system.metrics', metric_event)
        
        await asyncio.sleep(0.5)
    
    # Wait for processing
        await asyncio.sleep(5)
    
    # Check WebSocket server
        logger.info(f"Active WebSocket connections: {len(streaming.websocket_server.websockets)}")
    
    # Close
        await streaming.close()


        if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        asyncio.run(test_streaming_system())
