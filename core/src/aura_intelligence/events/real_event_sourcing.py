"""Real Event Sourcing with Kafka"""
import asyncio
from typing import Dict, Any
from datetime import datetime

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

class RealEventStore:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        if KAFKA_AVAILABLE:
            self.producer = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                value_serializer=lambda v: str(v).encode('utf-8')
            )
        else:
            self.producer = None
    
    async def store_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Store event in Kafka"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'id': f"evt_{int(datetime.utcnow().timestamp())}"
        }
        
        if self.producer:
            self.producer.send('aura_events', value=event)
            return event['id']
        else:
            return f"mock_{event['id']}"

def get_real_event_store():
    return RealEventStore()