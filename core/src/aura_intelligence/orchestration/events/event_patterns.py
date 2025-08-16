"""
Event Pattern Recognition and Matching

Real-time semantic event pattern detection with TDA correlation.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Callable, Pattern
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of event patterns"""
    SEQUENCE = "sequence"        # Ordered sequence of events
    CONCURRENT = "concurrent"    # Events happening simultaneously
    FREQUENCY = "frequency"      # Event frequency patterns
    ANOMALY = "anomaly"         # Anomalous event patterns
    SEMANTIC = "semantic"       # Semantic content patterns

class PatternPriority(Enum):
    """Pattern matching priority levels"""
    CRITICAL = 1    # System-critical patterns
    HIGH = 2        # High-priority business patterns
    NORMAL = 3      # Standard operational patterns
    LOW = 4         # Informational patterns

@dataclass
class EventPattern:
    """Immutable event pattern definition"""
    pattern_id: str
    pattern_type: PatternType
    priority: PatternPriority
    conditions: Dict[str, Any]
    action: str
    timeout_seconds: int = 300
    max_matches: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PatternMatch:
    """Pattern match result"""
    pattern_id: str
    matched_events: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tda_correlation: Optional[str] = None

class PatternMatcher:
    """High-performance pattern matching engine (45 lines)"""
    
    def __init__(self):
        self.patterns: Dict[str, EventPattern] = {}
        self.event_buffer: List[Dict[str, Any]] = []
        self.match_history: List[PatternMatch] = []
        self.compiled_patterns: Dict[str, Pattern] = {}
        
        # Performance optimization
        self.buffer_size = 1000
        self.cleanup_interval = 3600  # 1 hour
        
        logger.info("Pattern matcher initialized")
    
    def register_pattern(self, pattern: EventPattern) -> None:
        """Register event pattern for matching"""
        self.patterns[pattern.pattern_id] = pattern
        
        # Pre-compile regex patterns for performance
        if pattern.pattern_type == PatternType.SEMANTIC:
            regex_pattern = pattern.conditions.get('regex')
            if regex_pattern:
                self.compiled_patterns[pattern.pattern_id] = re.compile(
                    regex_pattern, re.IGNORECASE
                )
        
        logger.info(f"Registered pattern {pattern.pattern_id} ({pattern.pattern_type.value})")
    
    def match_event(self, event: Dict[str, Any]) -> List[PatternMatch]:
        """Match single event against all patterns"""
        # Add to buffer
        self.event_buffer.append(event)
        self._cleanup_buffer()
        
        matches = []
        
        # Check each pattern
        for pattern in self.patterns.values():
            match = self._check_pattern(pattern, event)
            if match:
                matches.append(match)
                self.match_history.append(match)
        
        return matches
    
    def _check_pattern(self, pattern: EventPattern, event: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check if event matches pattern"""
        
        if pattern.pattern_type == PatternType.SEMANTIC:
            return self._check_semantic_pattern(pattern, event)
        elif pattern.pattern_type == PatternType.FREQUENCY:
            return self._check_frequency_pattern(pattern, event)
        elif pattern.pattern_type == PatternType.ANOMALY:
            return self._check_anomaly_pattern(pattern, event)
        elif pattern.pattern_type == PatternType.SEQUENCE:
            return self._check_sequence_pattern(pattern, event)
        
        return None
    
    def _check_semantic_pattern(self, pattern: EventPattern, event: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check semantic content patterns"""
        content = event.get('content', '')
        event_type = event.get('type', '')
        
        conditions = pattern.conditions
        
        # Check event type
        if 'event_types' in conditions:
            if event_type not in conditions['event_types']:
                return None
        
        # Check regex pattern
        if pattern.pattern_id in self.compiled_patterns:
            regex = self.compiled_patterns[pattern.pattern_id]
            if not regex.search(content):
                return None
        
        # Check keywords
        if 'keywords' in conditions:
            keywords = conditions['keywords']
            if not any(keyword.lower() in content.lower() for keyword in keywords):
                return None
        
        return PatternMatch(
            pattern_id=pattern.pattern_id,
            matched_events=[event],
            confidence=0.9,
            tda_correlation=event.get('tda_correlation_id')
        )
    
    def _check_frequency_pattern(self, pattern: EventPattern, event: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check frequency-based patterns"""
        conditions = pattern.conditions
        event_type = event.get('type', '')
        
        if 'target_event_type' not in conditions:
            return None
        
        if event_type != conditions['target_event_type']:
            return None
        
        # Count recent events of this type
        cutoff_time = datetime.utcnow() - timedelta(seconds=conditions.get('window_seconds', 60))
        recent_events = [
            e for e in self.event_buffer
            if e.get('timestamp', datetime.utcnow()) > cutoff_time
            and e.get('type') == event_type
        ]
        
        threshold = conditions.get('threshold', 10)
        if len(recent_events) >= threshold:
            return PatternMatch(
                pattern_id=pattern.pattern_id,
                matched_events=recent_events,
                confidence=min(1.0, len(recent_events) / threshold),
                tda_correlation=event.get('tda_correlation_id')
            )
        
        return None
    
    def _check_anomaly_pattern(self, pattern: EventPattern, event: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check TDA anomaly patterns"""
        tda_score = event.get('tda_anomaly_score', 0.0)
        threshold = pattern.conditions.get('anomaly_threshold', 0.8)
        
        if tda_score >= threshold:
            return PatternMatch(
                pattern_id=pattern.pattern_id,
                matched_events=[event],
                confidence=tda_score,
                tda_correlation=event.get('tda_correlation_id')
            )
        
        return None
    
    def _check_sequence_pattern(self, pattern: EventPattern, event: Dict[str, Any]) -> Optional[PatternMatch]:
        """Check sequence patterns"""
        conditions = pattern.conditions
        sequence = conditions.get('sequence', [])
        
        if not sequence:
            return None
        
        # Simple sequence matching (can be enhanced)
        event_type = event.get('type', '')
        if event_type == sequence[-1]:  # Last event in sequence
            # Check if we have the full sequence in recent events
            window_seconds = conditions.get('window_seconds', 300)
            cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
            
            recent_events = [
                e for e in self.event_buffer
                if e.get('timestamp', datetime.utcnow()) > cutoff_time
            ]
            
            # Simple sequence check
            sequence_found = self._find_sequence(recent_events, sequence)
            if sequence_found:
                return PatternMatch(
                    pattern_id=pattern.pattern_id,
                    matched_events=sequence_found,
                    confidence=0.8,
                    tda_correlation=event.get('tda_correlation_id')
                )
        
        return None
    
    def _find_sequence(self, events: List[Dict[str, Any]], sequence: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Find sequence in events"""
        if len(events) < len(sequence):
            return None
        
        # Simple implementation - can be optimized
        for i in range(len(events) - len(sequence) + 1):
            match = True
            matched_events = []
            
            for j, expected_type in enumerate(sequence):
                if i + j >= len(events):
                    match = False
                    break
                
                if events[i + j].get('type') != expected_type:
                    match = False
                    break
                
                matched_events.append(events[i + j])
            
            if match:
                return matched_events
        
        return None
    
    def _cleanup_buffer(self) -> None:
        """Clean up old events from buffer"""
        if len(self.event_buffer) > self.buffer_size:
            # Keep only recent events
            self.event_buffer = self.event_buffer[-self.buffer_size//2:]
        
        # Clean up old matches
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.cleanup_interval)
        self.match_history = [
            m for m in self.match_history
            if m.timestamp > cutoff_time
        ]
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern matching statistics"""
        total_matches = len(self.match_history)
        pattern_counts = {}
        
        for match in self.match_history:
            pattern_counts[match.pattern_id] = pattern_counts.get(match.pattern_id, 0) + 1
        
        return {
            'total_patterns': len(self.patterns),
            'total_matches': total_matches,
            'buffer_size': len(self.event_buffer),
            'pattern_match_counts': pattern_counts,
            'avg_confidence': sum(m.confidence for m in self.match_history) / max(1, total_matches)
        }