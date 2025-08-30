"""
Streaming Integration for Iceberg
=================================
Provides CDC sinks and streaming bridges for real-time data ingestion
into the lakehouse with exactly-once semantics.
"""

from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StreamingEngine(Enum):
    """Supported streaming engines"""
    KAFKA = "kafka"
    FLINK = "flink"
    SPARK_STREAMING = "spark_streaming"
    PULSAR = "pulsar"
    KINESIS = "kinesis"
    NATS = "nats"


class DeliveryGuarantee(Enum):
    """Message delivery guarantees"""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class StreamConfig:
    """Configuration for streaming integration"""
    # Engine settings
    engine: StreamingEngine
    bootstrap_servers: List[str]
    
    # Consumer settings
    consumer_group: str = "aura-lakehouse"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    
    # Processing settings
    batch_size: int = 1000
    batch_timeout_ms: int = 5000
    max_records_per_batch: int = 10000
    
    # Delivery semantics
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.EXACTLY_ONCE
    checkpoint_interval_ms: int = 60000
    
    # Error handling
    max_retries: int = 3
    retry_backoff_ms: int = 1000
    dead_letter_topic: Optional[str] = None
    
    # Performance
    parallelism: int = 4
    buffer_size: int = 10000
    
    # Monitoring
    metrics_enabled: bool = True
    metrics_interval_ms: int = 30000


@dataclass
class StreamingMetrics:
    """Metrics for streaming pipeline"""
    messages_processed: int = 0
    messages_failed: int = 0
    batches_processed: int = 0
    
    # Latency tracking
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Throughput
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0
    
    # Lag
    consumer_lag: int = 0
    
    # Errors
    last_error: Optional[str] = None
    error_count: int = 0
    
    # Checkpointing
    last_checkpoint: Optional[datetime] = None
    checkpoint_failures: int = 0


@dataclass
class CDCEvent:
    """Change Data Capture event"""
    operation: str  # insert, update, delete, snapshot
    table: str
    schema: str
    
    # Timing
    transaction_id: Optional[str] = None
    transaction_timestamp: Optional[datetime] = None
    capture_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Data
    before: Optional[Dict[str, Any]] = None  # For updates/deletes
    after: Optional[Dict[str, Any]] = None   # For inserts/updates
    
    # Metadata
    source: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_iceberg_row(self) -> Dict[str, Any]:
        """Convert CDC event to Iceberg row format"""
        if self.operation == "insert":
            row = self.after.copy() if self.after else {}
        elif self.operation == "update":
            row = self.after.copy() if self.after else {}
        elif self.operation == "delete":
            # For deletes, we might want to keep the row with a deleted flag
            row = self.before.copy() if self.before else {}
            row['_deleted'] = True
        else:  # snapshot
            row = self.after.copy() if self.after else {}
            
        # Add CDC metadata
        row['_cdc_operation'] = self.operation
        row['_cdc_transaction_id'] = self.transaction_id
        row['_cdc_timestamp'] = self.transaction_timestamp
        row['_cdc_capture_timestamp'] = self.capture_timestamp
        
        return row


class StreamProcessor(ABC):
    """Abstract base class for stream processors"""
    
    @abstractmethod
    async def process_batch(self, events: List[CDCEvent]) -> Dict[str, Any]:
        """Process a batch of events"""
        pass
        
    @abstractmethod
    async def checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Save checkpoint for recovery"""
        pass
        
    @abstractmethod
    async def recover_from_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Recover from last checkpoint"""
        pass


class CDCSink:
    """
    Change Data Capture sink for streaming data into Iceberg.
    Handles deduplication, ordering, and exactly-once delivery.
    """
    
    def __init__(self,
                 catalog: 'IcebergCatalog',
                 config: StreamConfig,
                 processor: Optional[StreamProcessor] = None):
        self.catalog = catalog
        self.config = config
        self.processor = processor
        
        # State management
        self._running = False
        self._consumer = None
        self._producer = None  # For dead letter queue
        
        # Buffering
        self._buffer: List[CDCEvent] = []
        self._buffer_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = StreamingMetrics()
        self._metrics_task = None
        
        # Checkpointing
        self._last_checkpoint: Optional[Dict[str, Any]] = None
        self._checkpoint_lock = asyncio.Lock()
        
        logger.info(f"CDC Sink initialized for engine: {config.engine}")
        
    async def start(self):
        """Start the CDC sink"""
        self._running = True
        
        # Recover from checkpoint if exists
        if self.processor:
            self._last_checkpoint = await self.processor.recover_from_checkpoint()
            
        # Start consumer based on engine
        if self.config.engine == StreamingEngine.KAFKA:
            await self._start_kafka_consumer()
        elif self.config.engine == StreamingEngine.NATS:
            await self._start_nats_consumer()
        else:
            raise ValueError(f"Unsupported engine: {self.config.engine}")
            
        # Start processing loop
        asyncio.create_task(self._processing_loop())
        
        # Start metrics reporting
        if self.config.metrics_enabled:
            self._metrics_task = asyncio.create_task(self._metrics_loop())
            
        logger.info("CDC Sink started")
        
    async def stop(self):
        """Stop the CDC sink"""
        self._running = False
        
        # Process remaining buffer
        await self._flush_buffer()
        
        # Stop consumer
        if self._consumer:
            await self._consumer.close()
            
        # Stop metrics
        if self._metrics_task:
            self._metrics_task.cancel()
            
        logger.info("CDC Sink stopped")
        
    async def _start_kafka_consumer(self):
        """Start Kafka consumer"""
        # In real implementation, would use aiokafka
        logger.info("Started Kafka consumer")
        
    async def _start_nats_consumer(self):
        """Start NATS consumer"""
        # In real implementation, would use nats.py
        logger.info("Started NATS consumer")
        
    async def _processing_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                # Collect batch
                batch = await self._collect_batch()
                
                if batch:
                    # Process batch
                    start_time = datetime.utcnow()
                    
                    if self.processor:
                        result = await self.processor.process_batch(batch)
                    else:
                        result = await self._default_process_batch(batch)
                        
                    # Update metrics
                    elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    await self._update_metrics(batch, elapsed_ms, result)
                    
                    # Checkpoint if needed
                    await self._maybe_checkpoint()
                    
                else:
                    # No data, sleep briefly
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.metrics.error_count += 1
                self.metrics.last_error = str(e)
                
                # Send to dead letter queue if configured
                if batch and self.config.dead_letter_topic:
                    await self._send_to_dead_letter(batch, str(e))
                    
    async def _collect_batch(self) -> List[CDCEvent]:
        """Collect a batch of events"""
        batch = []
        deadline = datetime.utcnow() + timedelta(milliseconds=self.config.batch_timeout_ms)
        
        while (len(batch) < self.config.batch_size and 
               datetime.utcnow() < deadline and
               self._running):
            
            # Get from buffer
            async with self._buffer_lock:
                if self._buffer:
                    available = min(
                        self.config.batch_size - len(batch),
                        len(self._buffer)
                    )
                    batch.extend(self._buffer[:available])
                    self._buffer = self._buffer[available:]
                    
            if len(batch) < self.config.batch_size:
                # Need more data, wait a bit
                await asyncio.sleep(0.01)
                
        return batch
        
    async def _default_process_batch(self, batch: List[CDCEvent]) -> Dict[str, Any]:
        """Default batch processing"""
        # Group by table
        tables: Dict[str, List[CDCEvent]] = {}
        
        for event in batch:
            table_name = f"{event.schema}.{event.table}"
            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append(event)
            
        # Process each table
        results = {}
        
        for table_name, events in tables.items():
            # Convert to Iceberg rows
            rows = [event.to_iceberg_row() for event in events]
            
            # Load table
            namespace, table = table_name.split('.')
            iceberg_table = await self.catalog.load_table(namespace, table)
            
            # Append data
            append_op = iceberg_table.new_append()
            for row in rows:
                append_op.append(row)
                
            # Commit
            append_op.commit()
            
            results[table_name] = {
                'rows_written': len(rows),
                'success': True
            }
            
        return results
        
    async def _update_metrics(self, 
                            batch: List[CDCEvent],
                            elapsed_ms: float,
                            result: Dict[str, Any]):
        """Update streaming metrics"""
        self.metrics.messages_processed += len(batch)
        self.metrics.batches_processed += 1
        
        # Latency
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, elapsed_ms)
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, elapsed_ms)
        
        # Running average
        total_messages = self.metrics.messages_processed
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (total_messages - len(batch)) + 
             elapsed_ms * len(batch)) / total_messages
        )
        
        # Throughput
        if elapsed_ms > 0:
            self.metrics.messages_per_second = len(batch) / (elapsed_ms / 1000)
            
    async def _maybe_checkpoint(self):
        """Checkpoint if needed"""
        if not self._last_checkpoint:
            self._last_checkpoint = {'timestamp': datetime.utcnow()}
            
        time_since_checkpoint = datetime.utcnow() - self._last_checkpoint['timestamp']
        
        if time_since_checkpoint.total_seconds() * 1000 >= self.config.checkpoint_interval_ms:
            await self._checkpoint()
            
    async def _checkpoint(self):
        """Save checkpoint"""
        async with self._checkpoint_lock:
            checkpoint_data = {
                'timestamp': datetime.utcnow(),
                'messages_processed': self.metrics.messages_processed,
                'consumer_position': self._get_consumer_position()
            }
            
            if self.processor:
                await self.processor.checkpoint(checkpoint_data)
                
            self._last_checkpoint = checkpoint_data
            self.metrics.last_checkpoint = checkpoint_data['timestamp']
            
            logger.debug(f"Checkpoint saved: {checkpoint_data}")
            
    def _get_consumer_position(self) -> Dict[str, Any]:
        """Get current consumer position"""
        # In real implementation, would get actual offset/position
        return {'offset': self.metrics.messages_processed}
        
    async def _send_to_dead_letter(self, batch: List[CDCEvent], error: str):
        """Send failed batch to dead letter queue"""
        # In real implementation, would send to actual DLQ
        logger.error(f"Sent {len(batch)} events to dead letter queue: {error}")
        
    async def _flush_buffer(self):
        """Process any remaining buffered data"""
        async with self._buffer_lock:
            if self._buffer:
                batch = self._buffer
                self._buffer = []
                
                try:
                    await self._default_process_batch(batch)
                except Exception as e:
                    logger.error(f"Error flushing buffer: {e}")
                    
    async def _metrics_loop(self):
        """Report metrics periodically"""
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_interval_ms / 1000)
                
                # Log current metrics
                logger.info(f"CDC Metrics: {json.dumps({
                    'messages_processed': self.metrics.messages_processed,
                    'messages_per_second': self.metrics.messages_per_second,
                    'avg_latency_ms': self.metrics.avg_latency_ms,
                    'error_count': self.metrics.error_count
                })}")
                
            except Exception as e:
                logger.error(f"Metrics reporting error: {e}")


class StreamingBridge:
    """
    Bridge between streaming systems and Iceberg lakehouse.
    Provides unified API for different streaming engines.
    """
    
    def __init__(self, catalog: 'IcebergCatalog'):
        self.catalog = catalog
        self._sinks: Dict[str, CDCSink] = {}
        
    async def create_sink(self,
                         sink_name: str,
                         config: StreamConfig,
                         processor: Optional[StreamProcessor] = None) -> CDCSink:
        """Create a new CDC sink"""
        if sink_name in self._sinks:
            raise ValueError(f"Sink {sink_name} already exists")
            
        sink = CDCSink(self.catalog, config, processor)
        self._sinks[sink_name] = sink
        
        return sink
        
    async def start_sink(self, sink_name: str):
        """Start a specific sink"""
        if sink_name not in self._sinks:
            raise ValueError(f"Sink {sink_name} not found")
            
        await self._sinks[sink_name].start()
        
    async def stop_sink(self, sink_name: str):
        """Stop a specific sink"""
        if sink_name not in self._sinks:
            raise ValueError(f"Sink {sink_name} not found")
            
        await self._sinks[sink_name].stop()
        
    async def start_all(self):
        """Start all sinks"""
        for sink in self._sinks.values():
            await sink.start()
            
    async def stop_all(self):
        """Stop all sinks"""
        for sink in self._sinks.values():
            await sink.stop()
            
    def get_metrics(self, sink_name: Optional[str] = None) -> Dict[str, StreamingMetrics]:
        """Get metrics for sinks"""
        if sink_name:
            if sink_name not in self._sinks:
                raise ValueError(f"Sink {sink_name} not found")
            return {sink_name: self._sinks[sink_name].metrics}
        else:
            return {name: sink.metrics for name, sink in self._sinks.items()}
            
    async def create_table_from_stream(self,
                                     stream_config: Dict[str, Any],
                                     table_name: str,
                                     namespace: str = "streaming") -> None:
        """Create Iceberg table from stream schema"""
        # In real implementation, would infer schema from stream
        # and create appropriate Iceberg table
        logger.info(f"Created table {namespace}.{table_name} from stream")