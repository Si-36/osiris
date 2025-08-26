"""
Alert Management System

Production-ready alerting with escalation and notification channels.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_id: str
    channel_type: str  # "email", "slack", "webhook", "sms"
    config: Dict[str, Any]
    enabled: bool = True

class AlertManager:
    """
    Production alert management system.
    
    Handles alert lifecycle, escalation, and notifications.
    """
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Escalation rules
        self.escalation_rules: Dict[AlertSeverity, Dict[str, Any]] = {
            AlertSeverity.CRITICAL: {
                'immediate_notify': True,
                'escalate_after_minutes': 5,
                'max_escalations': 3
            },
            AlertSeverity.WARNING: {
                'immediate_notify': True,
                'escalate_after_minutes': 15,
                'max_escalations': 2
            },
            AlertSeverity.INFO: {
                'immediate_notify': False,
                'escalate_after_minutes': 60,
                'max_escalations': 1
            }
        }
        
        # Alert suppression
        self.suppression_rules: Dict[str, timedelta] = {}
        self.suppressed_alerts: Set[str] = set()
        
        # Metrics
        self.total_alerts = 0
        self.alerts_by_severity: Dict[str, int] = {
            'critical': 0, 'warning': 0, 'info': 0
        }
        
        logger.info("Alert Manager initialized")
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel"""
        self.notification_channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.channel_id}")
    
        async def create_alert(self, alert: Alert) -> str:
        """Create new alert"""
        # Check for suppression
        if self._is_suppressed(alert):
            logger.debug(f"Alert suppressed: {alert.alert_id}")
            return alert.alert_id
        
        # Check for duplicate active alerts
        existing_alert = self._find_similar_alert(alert)
        if existing_alert:
            logger.debug(f"Similar alert exists: {existing_alert.alert_id}")
            return existing_alert.alert_id
        
        # Create alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.total_alerts += 1
        self.alerts_by_severity[alert.severity.value] += 1
        
        logger.info(f"Created {alert.severity.value} alert: {alert.title}")
        
        # Handle immediate notification
        escalation_rule = self.escalation_rules[alert.severity]
        if escalation_rule['immediate_notify']:
            await self._send_notifications(alert)
        
        # Schedule escalation
        asyncio.create_task(self._handle_escalation(alert))
        
        return alert.alert_id
    
        async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        
        # Send acknowledgment notification
        await self._send_acknowledgment_notification(alert, acknowledged_by)
        
        return True
    
        async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        # Move to history
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        
        # Send resolution notification
        await self._send_resolution_notification(alert, resolved_by)
        
        return True
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        suppression_key = f"{alert.component}:{alert.title}"
        
        if suppression_key in self.suppressed_alerts:
            return True
        
        # Check suppression rules
        if alert.component in self.suppression_rules:
            suppression_duration = self.suppression_rules[alert.component]
            
            # Check recent similar alerts
            cutoff_time = datetime.utcnow() - suppression_duration
            recent_similar = [
                a for a in self.alert_history
                if (a.component == alert.component and 
                    a.title == alert.title and
                    a.created_at > cutoff_time)
            ]
            
            if len(recent_similar) > 0:
                self.suppressed_alerts.add(suppression_key)
                # Auto-remove suppression after duration
                asyncio.create_task(self._remove_suppression(suppression_key, suppression_duration))
                return True
        
        return False
    
        async def _remove_suppression(self, suppression_key: str, duration: timedelta) -> None:
        """Remove suppression after duration"""
        await asyncio.sleep(duration.total_seconds())
        self.suppressed_alerts.discard(suppression_key)
    
    def _find_similar_alert(self, alert: Alert) -> Optional[Alert]:
        """Find similar active alert"""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.component == alert.component and
                existing_alert.title == alert.title and
                existing_alert.severity == alert.severity):
                return existing_alert
        return None
    
        async def _handle_escalation(self, alert: Alert) -> None:
        """Handle alert escalation"""
        escalation_rule = self.escalation_rules[alert.severity]
        escalate_after = escalation_rule['escalate_after_minutes'] * 60  # Convert to seconds
        max_escalations = escalation_rule['max_escalations']
        
        escalation_count = 0
        
        while (alert.alert_id in self.active_alerts and 
               alert.status == AlertStatus.ACTIVE and
               escalation_count < max_escalations):
            
            # Wait for escalation time
            await asyncio.sleep(escalate_after)
            
            # Check if alert is still active and unacknowledged
            if (alert.alert_id in self.active_alerts and 
                alert.status == AlertStatus.ACTIVE):
                
                escalation_count += 1
                logger.warning(f"Escalating alert {alert.alert_id} (escalation {escalation_count})")
                
                # Send escalation notification
                await self._send_escalation_notification(alert, escalation_count)
    
        async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications to all channels"""
        for channel in self.notification_channels.values():
            if not channel.enabled:
                continue
            
            try:
                await self._send_to_channel(channel, alert, "alert")
            except Exception as e:
                logger.error(f"Failed to send notification to {channel.channel_id}: {e}")
    
        async def _send_acknowledgment_notification(self, alert: Alert, acknowledged_by: str) -> None:
        """Send acknowledgment notification"""
        for channel in self.notification_channels.values():
            if channel.enabled:
                try:
                    await self._send_to_channel(channel, alert, "acknowledgment", 
                                              extra_data={'acknowledged_by': acknowledged_by})
                except Exception as e:
                    logger.error(f"Failed to send acknowledgment to {channel.channel_id}: {e}")
    
        async def _send_resolution_notification(self, alert: Alert, resolved_by: str) -> None:
        """Send resolution notification"""
        for channel in self.notification_channels.values():
            if channel.enabled:
                try:
                    await self._send_to_channel(channel, alert, "resolution",
                                              extra_data={'resolved_by': resolved_by})
                except Exception as e:
                    logger.error(f"Failed to send resolution to {channel.channel_id}: {e}")
    
        async def _send_escalation_notification(self, alert: Alert, escalation_level: int) -> None:
        """Send escalation notification"""
        for channel in self.notification_channels.values():
            if channel.enabled:
                try:
                    await self._send_to_channel(channel, alert, "escalation",
                                              extra_data={'escalation_level': escalation_level})
                except Exception as e:
                    logger.error(f"Failed to send escalation to {channel.channel_id}: {e}")
    
        async def _send_to_channel(self, channel: NotificationChannel, alert: Alert,
        notification_type: str, extra_data: Dict[str, Any] = None) -> None:
        """Send notification to specific channel"""
        message_data = {
            'alert_id': alert.alert_id,
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'component': alert.component,
            'created_at': alert.created_at.isoformat(),
            'notification_type': notification_type
        }
        
        if extra_data:
            message_data.update(extra_data)
        
        if channel.channel_type == "webhook":
            await self._send_webhook(channel, message_data)
        elif channel.channel_type == "email":
            await self._send_email(channel, message_data)
        elif channel.channel_type == "slack":
            await self._send_slack(channel, message_data)
        else:
            logger.warning(f"Unsupported channel type: {channel.channel_type}")
    
        async def _send_webhook(self, channel: NotificationChannel, data: Dict[str, Any]) -> None:
        """Send webhook notification (mock implementation)"""
        webhook_url = channel.config.get('url')
        logger.info(f"Webhook notification to {webhook_url}: {data['title']}")
        # In production, this would make actual HTTP request
    
        async def _send_email(self, channel: NotificationChannel, data: Dict[str, Any]) -> None:
        """Send email notification (mock implementation)"""
        email_address = channel.config.get('email')
        logger.info(f"Email notification to {email_address}: {data['title']}")
        # In production, this would send actual email
    
        async def _send_slack(self, channel: NotificationChannel, data: Dict[str, Any]) -> None:
        """Send Slack notification (mock implementation)"""
        slack_channel = channel.config.get('channel')
        logger.info(f"Slack notification to {slack_channel}: {data['title']}")
        # In production, this would send to actual Slack
    
    def add_suppression_rule(self, component: str, duration: timedelta) -> None:
        """Add alert suppression rule"""
        self.suppression_rules[component] = duration
        logger.info(f"Added suppression rule for {component}: {duration}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        pass
        # Calculate alert resolution time
        resolved_alerts = [a for a in self.alert_history if a.resolved_at]
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.created_at).total_seconds()
                for a in resolved_alerts
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        else:
            avg_resolution_time = 0.0
        
        return {
            'active_alerts': len(self.active_alerts),
            'total_alerts': self.total_alerts,
            'alerts_by_severity': dict(self.alerts_by_severity),
            'notification_channels': len(self.notification_channels),
            'suppression_rules': len(self.suppression_rules),
            'avg_resolution_time_seconds': avg_resolution_time,
            'suppressed_alerts': len(self.suppressed_alerts)
        }

# Factory function
    def create_alert_manager() -> AlertManager:
        """Create alert manager"""
        return AlertManager()
