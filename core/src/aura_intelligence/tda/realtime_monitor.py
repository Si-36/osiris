"""
Real-time Topology Monitor - Streaming Analysis for Agent Systems
===============================================================

Processes agent events in real-time to maintain up-to-date topology
features for routing decisions and anomaly detection.

Key Features:
- Event-driven topology updates
- Sliding window analysis
- Incremental feature computation
- Backpressure-aware processing
- Integration with feature stores
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Protocol
from enum import Enum
import numpy as np
import structlog

from .agent_topology import (
    AgentTopologyAnalyzer,
    WorkflowFeatures,
    CommunicationFeatures,
    TopologicalAnomaly
)

logger = structlog.get_logger(__name__)


# ==================== Event Types ====================

class EventType(str, Enum):
    """Types of system events."""
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    MESSAGE_SENT = "message_sent"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    DEPENDENCY_ADDED = "dependency_added"
    DEPENDENCY_REMOVED = "dependency_removed"


@dataclass
class SystemEvent:
    """Base event for system monitoring."""
    event_id: str
    event_type: EventType
    timestamp: float
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata
        }


# ==================== Window Management ====================

@dataclass
class TimeWindow:
    """Time window for analysis."""
    start_time: float
    end_time: float
    window_id: str
    
    @property
    def duration(self) -> float:
        """Window duration in seconds."""
        return self.end_time - self.start_time
        
    def contains(self, timestamp: float) -> bool:
        """Check if timestamp is in window."""
        return self.start_time <= timestamp < self.end_time
        
    def overlaps(self, other: 'TimeWindow') -> bool:
        """Check if windows overlap."""
        return not (self.end_time <= other.start_time or 
                   other.end_time <= self.start_time)


class WindowManager:
    """Manages sliding/tumbling windows for analysis."""
    
    def __init__(self, 
                 window_size: int = 300,  # 5 minutes
                 slide_interval: int = 60,  # 1 minute
                 max_windows: int = 10):
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.max_windows = max_windows
        
        self.active_windows: List[TimeWindow] = []
        self.window_counter = 0
        
    def update(self, current_time: float) -> List[TimeWindow]:
        """Update windows and return new ones to process."""
        new_windows = []
        
        # Check if we need a new window
        if not self.active_windows:
            # First window
            start_time = current_time
            end_time = start_time + self.window_size
            window = TimeWindow(
                start_time=start_time,
                end_time=end_time,
                window_id=f"window_{self.window_counter}"
            )
            self.active_windows.append(window)
            new_windows.append(window)
            self.window_counter += 1
            
        else:
            # Check if we need to slide
            latest_window = self.active_windows[-1]
            time_since_start = current_time - latest_window.start_time
            
            if time_since_start >= self.slide_interval:
                # Create new window
                start_time = latest_window.start_time + self.slide_interval
                end_time = start_time + self.window_size
                window = TimeWindow(
                    start_time=start_time,
                    end_time=end_time,
                    window_id=f"window_{self.window_counter}"
                )
                self.active_windows.append(window)
                new_windows.append(window)
                self.window_counter += 1
                
        # Remove old windows
        cutoff_time = current_time - (self.window_size * 2)
        self.active_windows = [
            w for w in self.active_windows 
            if w.end_time > cutoff_time
        ][:self.max_windows]
        
        return new_windows
        
    def get_active_windows(self, timestamp: float) -> List[TimeWindow]:
        """Get windows that contain the timestamp."""
        return [w for w in self.active_windows if w.contains(timestamp)]


# ==================== Event Aggregation ====================

@dataclass
class WindowData:
    """Aggregated data for a time window."""
    window: TimeWindow
    events: List[SystemEvent] = field(default_factory=list)
    
    # Workflow data
    workflow_agents: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    workflow_dependencies: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    workflow_events: Dict[str, List[SystemEvent]] = field(default_factory=lambda: defaultdict(list))
    
    # Communication data
    active_agents: Set[str] = field(default_factory=set)
    messages: List[Tuple[str, str, float]] = field(default_factory=list)  # (source, target, timestamp)
    
    # Metrics
    event_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agent_activity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_event(self, event: SystemEvent):
        """Add event to window data."""
        self.events.append(event)
        self.event_counts[event.event_type.value] += 1
        
        # Track agents
        if event.source_agent:
            self.active_agents.add(event.source_agent)
            self.agent_activity[event.source_agent] += 1
            
        if event.target_agent:
            self.active_agents.add(event.target_agent)
            self.agent_activity[event.target_agent] += 1
            
        # Process by type
        if event.event_type == EventType.MESSAGE_SENT:
            if event.source_agent and event.target_agent:
                self.messages.append((
                    event.source_agent,
                    event.target_agent,
                    event.timestamp
                ))
                
        elif event.event_type in [EventType.TASK_ASSIGNED, EventType.DEPENDENCY_ADDED]:
            if event.workflow_id and event.source_agent and event.target_agent:
                self.workflow_agents[event.workflow_id].add(event.source_agent)
                self.workflow_agents[event.workflow_id].add(event.target_agent)
                self.workflow_dependencies[event.workflow_id].append(
                    (event.source_agent, event.target_agent)
                )
                
        # Track workflow events
        if event.workflow_id:
            self.workflow_events[event.workflow_id].append(event)
            
    def to_workflow_data(self, workflow_id: str) -> Dict[str, Any]:
        """Convert to workflow analysis format."""
        agents = list(self.workflow_agents.get(workflow_id, set()))
        dependencies = self.workflow_dependencies.get(workflow_id, [])
        
        return {
            "agents": [{"id": agent} for agent in agents],
            "dependencies": [
                {"source": src, "target": tgt, "weight": 1.0}
                for src, tgt in dependencies
            ]
        }
        
    def to_communication_data(self) -> Dict[str, Any]:
        """Convert to communication analysis format."""
        return {
            "agents": [{"id": agent} for agent in self.active_agents],
            "messages": [
                {"source": src, "target": tgt, "timestamp": ts}
                for src, tgt, ts in self.messages
            ]
        }


# ==================== Feature Publisher ====================

class FeaturePublisher(Protocol):
    """Protocol for feature publishing."""
    
    async def publish(self, features: Dict[str, Any]) -> None:
        """Publish features to storage/stream."""
        ...


class InMemoryPublisher:
    """Simple in-memory feature store for testing."""
    
    def __init__(self):
        self.features: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    async def publish(self, features: Dict[str, Any]) -> None:
        """Store features in memory."""
        feature_type = features.get("type", "unknown")
        self.features[feature_type].append(features)
        
    def get_latest(self, feature_type: str) -> Optional[Dict[str, Any]]:
        """Get latest features of type."""
        if feature_type in self.features and self.features[feature_type]:
            return self.features[feature_type][-1]
        return None


# ==================== Real-time Monitor ====================

class RealtimeTopologyMonitor:
    """
    Main real-time topology monitoring system.
    Processes events and maintains current topology features.
    """
    
    def __init__(self,
                 analyzer: Optional[AgentTopologyAnalyzer] = None,
                 publisher: Optional[FeaturePublisher] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.config = config or {}
        self.analyzer = analyzer or AgentTopologyAnalyzer()
        self.publisher = publisher or InMemoryPublisher()
        
        # Window management
        self.window_manager = WindowManager(
            window_size=self.config.get("window_size", 300),
            slide_interval=self.config.get("slide_interval", 60)
        )
        
        # Window data storage
        self.window_data: Dict[str, WindowData] = {}
        
        # Event queue with backpressure
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.get("max_queue_size", 10000)
        )
        
        # Processing state
        self.is_running = False
        self.process_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.processed_events = 0
        self.dropped_events = 0
        self.published_features = 0
        
        # Incremental update support (future)
        self.enable_incremental = self.config.get("enable_incremental", False)
        
    async def start(self):
        """Start the monitor."""
        if self.is_running:
            return
            
        self.is_running = True
        self.process_task = asyncio.create_task(self._process_loop())
        logger.info("Realtime topology monitor started")
        
    async def stop(self):
        """Stop the monitor."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Realtime topology monitor stopped")
        
    async def process_event(self, event: SystemEvent) -> bool:
        """
        Process a system event.
        
        Returns:
            True if event was queued, False if dropped due to backpressure
        """
        try:
            self.event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped_events += 1
            logger.warning(
                "Event queue full, dropping event",
                event_type=event.event_type,
                dropped_total=self.dropped_events
            )
            return False
            
    async def _process_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get event with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check for window updates
                    await self._check_windows()
                    continue
                    
                # Process event
                await self._process_single_event(event)
                self.processed_events += 1
                
                # Periodic window check
                if self.processed_events % 100 == 0:
                    await self._check_windows()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(1)
                
    async def _process_single_event(self, event: SystemEvent):
        """Process single event."""
        # Get active windows for event
        windows = self.window_manager.get_active_windows(event.timestamp)
        
        # Add event to each window
        for window in windows:
            if window.window_id not in self.window_data:
                self.window_data[window.window_id] = WindowData(window)
                
            self.window_data[window.window_id].add_event(event)
            
            # Incremental update if enabled
            if self.enable_incremental:
                await self._incremental_update(window.window_id, event)
                
    async def _check_windows(self):
        """Check for completed windows and process them."""
        current_time = time.time()
        
        # Update windows
        new_windows = self.window_manager.update(current_time)
        
        # Process completed windows
        completed = []
        for window_id, data in self.window_data.items():
            if data.window.end_time <= current_time:
                completed.append(window_id)
                
        for window_id in completed:
            await self._process_window(window_id)
            del self.window_data[window_id]
            
    async def _process_window(self, window_id: str):
        """Process completed window."""
        data = self.window_data.get(window_id)
        if not data:
            return
            
        logger.info(
            f"Processing window {window_id}",
            events=len(data.events),
            workflows=len(data.workflow_agents)
        )
        
        # Analyze workflows
        workflow_features = []
        for workflow_id in data.workflow_agents:
            try:
                workflow_data = data.to_workflow_data(workflow_id)
                if workflow_data["agents"]:  # Has agents
                    features = await self.analyzer.analyze_workflow(
                        workflow_id, workflow_data
                    )
                    workflow_features.append(features)
                    
                    # Publish workflow features
                    await self._publish_features("workflow", features.to_dict())
                    
            except Exception as e:
                logger.error(f"Error analyzing workflow {workflow_id}: {e}")
                
        # Analyze communications
        try:
            comm_data = data.to_communication_data()
            if comm_data["agents"]:  # Has active agents
                comm_features = await self.analyzer.analyze_communications(comm_data)
                
                # Publish communication features
                await self._publish_features("communication", comm_features.to_dict())
                
        except Exception as e:
            logger.error(f"Error analyzing communications: {e}")
            
        # Publish window summary
        await self._publish_window_summary(window_id, data, workflow_features)
        
    async def _incremental_update(self, window_id: str, event: SystemEvent):
        """
        Perform incremental update for event.
        Placeholder for future optimization.
        """
        # This would update features incrementally rather than
        # recomputing from scratch at window end
        pass
        
    async def _publish_features(self, feature_type: str, features: Dict[str, Any]):
        """Publish features to store."""
        features["type"] = feature_type
        features["published_at"] = time.time()
        
        try:
            await self.publisher.publish(features)
            self.published_features += 1
        except Exception as e:
            logger.error(f"Error publishing features: {e}")
            
    async def _publish_window_summary(self, window_id: str, 
                                    data: WindowData,
                                    workflow_features: List[WorkflowFeatures]):
        """Publish window summary."""
        # Aggregate metrics
        total_bottleneck_score = 0.0
        total_failure_risk = 0.0
        all_bottlenecks = set()
        
        for features in workflow_features:
            total_bottleneck_score += features.bottleneck_score
            total_failure_risk += features.failure_risk
            all_bottlenecks.update(features.bottleneck_agents)
            
        num_workflows = len(workflow_features)
        
        summary = {
            "window_id": window_id,
            "window_start": data.window.start_time,
            "window_end": data.window.end_time,
            "total_events": len(data.events),
            "active_agents": len(data.active_agents),
            "active_workflows": num_workflows,
            "avg_bottleneck_score": total_bottleneck_score / num_workflows if num_workflows > 0 else 0,
            "avg_failure_risk": total_failure_risk / num_workflows if num_workflows > 0 else 0,
            "bottleneck_agents": list(all_bottlenecks),
            "event_counts": dict(data.event_counts),
            "health_status": self.analyzer.get_health_status().value
        }
        
        await self._publish_features("window_summary", summary)
        
    async def flush_window(self, timestamp: float):
        """Force flush windows up to timestamp."""
        # Process all windows ending before timestamp
        for window_id, data in list(self.window_data.items()):
            if data.window.end_time <= timestamp:
                await self._process_window(window_id)
                del self.window_data[window_id]
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitor metrics."""
        return {
            "processed_events": self.processed_events,
            "dropped_events": self.dropped_events,
            "published_features": self.published_features,
            "queue_size": self.event_queue.qsize(),
            "active_windows": len(self.window_data),
            "is_running": self.is_running
        }


# ==================== Event Adapters ====================

class EventAdapter:
    """Adapts external events to SystemEvent format."""
    
    def __init__(self):
        self.event_counter = 0
        
    def from_agent_lifecycle(self, agent_id: str, 
                           action: str,
                           timestamp: Optional[float] = None) -> SystemEvent:
        """Convert agent lifecycle event."""
        if action == "start":
            event_type = EventType.AGENT_STARTED
        elif action == "stop":
            event_type = EventType.AGENT_STOPPED
        else:
            raise ValueError(f"Unknown action: {action}")
            
        return SystemEvent(
            event_id=f"event_{self.event_counter}",
            event_type=event_type,
            timestamp=timestamp or time.time(),
            source_agent=agent_id
        )
        
    def from_task_event(self, task_id: str,
                       source_agent: str,
                       target_agent: Optional[str],
                       workflow_id: str,
                       status: str,
                       timestamp: Optional[float] = None) -> SystemEvent:
        """Convert task event."""
        if status == "assigned":
            event_type = EventType.TASK_ASSIGNED
        elif status == "completed":
            event_type = EventType.TASK_COMPLETED
        elif status == "failed":
            event_type = EventType.TASK_FAILED
        else:
            raise ValueError(f"Unknown status: {status}")
            
        self.event_counter += 1
        
        return SystemEvent(
            event_id=f"event_{self.event_counter}",
            event_type=event_type,
            timestamp=timestamp or time.time(),
            source_agent=source_agent,
            target_agent=target_agent,
            workflow_id=workflow_id,
            metadata={"task_id": task_id}
        )
        
    def from_message(self, source: str,
                    target: str,
                    message_type: str = "data",
                    timestamp: Optional[float] = None) -> SystemEvent:
        """Convert message event."""
        self.event_counter += 1
        
        return SystemEvent(
            event_id=f"event_{self.event_counter}",
            event_type=EventType.MESSAGE_SENT,
            timestamp=timestamp or time.time(),
            source_agent=source,
            target_agent=target,
            metadata={"message_type": message_type}
        )


# ==================== Integration Helpers ====================

async def create_monitor(config: Optional[Dict[str, Any]] = None,
                        publisher: Optional[FeaturePublisher] = None) -> RealtimeTopologyMonitor:
    """Create and start a topology monitor."""
    monitor = RealtimeTopologyMonitor(
        analyzer=AgentTopologyAnalyzer(config),
        publisher=publisher,
        config=config
    )
    await monitor.start()
    return monitor


# Export main classes
__all__ = [
    # Events
    "EventType",
    "SystemEvent",
    
    # Windows
    "TimeWindow",
    "WindowManager",
    "WindowData",
    
    # Publishing
    "FeaturePublisher",
    "InMemoryPublisher",
    
    # Monitor
    "RealtimeTopologyMonitor",
    "EventAdapter",
    
    # Helpers
    "create_monitor"
]