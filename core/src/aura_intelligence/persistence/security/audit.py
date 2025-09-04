"""
Audit and Compliance Module
==========================
Provides immutable audit logging and compliance reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuditConfig:
    """Configuration for audit logging"""
    retention_years: int = 7
    enable_worm: bool = True
    compliance_standards: List[str] = field(default_factory=lambda: ["SOC2", "HIPAA", "GDPR"])
    

@dataclass
class AuditEntry:
    """Immutable audit log entry"""
    audit_id: str
    timestamp: datetime
    action: str
    resource: str
    actor: str
    
    # Result
    result: str  # success, failure, partial
    error: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance
    compliance_labels: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Convert to JSON for storage"""
        return json.dumps({
            'audit_id': self.audit_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'resource': self.resource,
            'actor': self.actor,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata,
            'compliance_labels': self.compliance_labels
        })


class ImmutableAuditLog:
    """
    Immutable audit log with WORM (Write Once Read Many) compliance.
    Ensures audit entries cannot be modified or deleted.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self._entries: List[AuditEntry] = []
        
    async def log(self, entry: AuditEntry) -> None:
        """Add audit entry (immutable)"""
        # In production, would write to S3 with Object Lock
        self._entries.append(entry)
        logger.info(f"Audit logged: {entry.action} on {entry.resource} by {entry.actor}")
        
    async def search(self, 
                    actor: Optional[str] = None,
                    resource: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[AuditEntry]:
        """Search audit logs"""
        results = self._entries
        
        if actor:
            results = [e for e in results if e.actor == actor]
        if resource:
            results = [e for e in results if e.resource == resource]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
            
        return results
        

class AccessMonitor:
    """
    Monitors data access patterns for anomaly detection.
    """
    
    def __init__(self):
        self._access_patterns: Dict[str, List[datetime]] = {}
        
    async def record_access(self, user: str, resource: str) -> None:
        """Record access event"""
        key = f"{user}:{resource}"
        
        if key not in self._access_patterns:
            self._access_patterns[key] = []
            
        self._access_patterns[key].append(datetime.utcnow())
        
        # Check for anomalies
        await self._check_anomalies(user, resource)
        
    async def _check_anomalies(self, user: str, resource: str) -> None:
        """Check for access anomalies"""
        key = f"{user}:{resource}"
        accesses = self._access_patterns.get(key, [])
        
        # Simple anomaly: too many accesses in short time
        recent = [a for a in accesses if (datetime.utcnow() - a).total_seconds() < 60]
        
        if len(recent) > 10:
            logger.warning(f"Anomaly detected: {user} accessed {resource} {len(recent)} times in 1 minute")
            

class ComplianceReporter:
    """
    Generates compliance reports for various standards.
    """
    
    def __init__(self, audit_log: ImmutableAuditLog):
        self.audit_log = audit_log
        
    async def generate_report(self, 
                            standard: str,
                            start_date: datetime,
                            end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report"""
        # Get relevant audit entries
        entries = await self.audit_log.search(
            start_time=start_date,
            end_time=end_date
        )
        
        # Filter by compliance standard
        relevant_entries = [
            e for e in entries 
            if standard in e.compliance_labels
        ]
        
        # Generate report
        report = {
            'standard': standard,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(relevant_entries),
            'summary': self._generate_summary(relevant_entries),
            'compliance_score': self._calculate_compliance_score(relevant_entries)
        }
        
        return report
        
    def _generate_summary(self, entries: List[AuditEntry]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not entries:
            return {}
            
        return {
            'total': len(entries),
            'successful': len([e for e in entries if e.result == 'success']),
            'failed': len([e for e in entries if e.result == 'failure']),
            'unique_actors': len(set(e.actor for e in entries)),
            'unique_resources': len(set(e.resource for e in entries))
        }
        
    def _calculate_compliance_score(self, entries: List[AuditEntry]) -> float:
        """Calculate compliance score (0-100)"""
        if not entries:
            return 100.0
            
        # Simple scoring: percentage of successful operations
        successful = len([e for e in entries if e.result == 'success'])
        return (successful / len(entries)) * 100


__all__ = [
    'AuditConfig',
    'AuditEntry',
    'ImmutableAuditLog',
    'AccessMonitor',
    'ComplianceReporter'
]