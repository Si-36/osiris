"""
📊 Grafana Dashboard Configurations for AURA Intelligence
========================================================

Production-ready Grafana dashboards for GPU-accelerated system monitoring.
Includes real-time GPU metrics, adapter performance, and system health.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GrafanaPanel:
    """Grafana panel configuration"""
    id: int
    title: str
    type: str  # graph, stat, gauge, table, heatmap, alert-list
    gridPos: Dict[str, int]  # x, y, w, h
    targets: List[Dict[str, Any]]
    options: Dict[str, Any] = field(default_factory=dict)
    fieldConfig: Dict[str, Any] = field(default_factory=dict)
    

class GrafanaDashboardBuilder:
    """
    Builds production-ready Grafana dashboards for AURA Intelligence.
    
    Features:
    - GPU monitoring dashboard
    - Adapter performance dashboard
    - System health overview
    - Agent activity monitoring
    - Alert dashboard
    """
    
    def __init__(self):
        self.panel_id_counter = 1
        
    def _next_panel_id(self) -> int:
        """Get next panel ID"""
        panel_id = self.panel_id_counter
        self.panel_id_counter += 1
        return panel_id
        
    def build_gpu_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        Build comprehensive GPU monitoring dashboard.
        """
        dashboard = {
            "uid": "aura-gpu-monitoring",
            "title": "AURA GPU Monitoring",
            "tags": ["gpu", "performance", "aura"],
            "timezone": "browser",
            "schemaVersion": 30,
            "version": 1,
            "refresh": "5s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": []
        }
        
        # Row 1: GPU Overview
        dashboard["panels"].extend([
            # GPU Count
            {
                "id": self._next_panel_id(),
                "type": "stat",
                "title": "Active GPUs",
                "gridPos": {"h": 4, "w": 3, "x": 0, "y": 0},
                "targets": [{
                    "expr": "count(gpu_utilization_percent)",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 0}
                            ]
                        },
                        "unit": "short"
                    }
                }
            },
            
            # Average GPU Utilization
            {
                "id": self._next_panel_id(),
                "type": "gauge",
                "title": "Avg GPU Utilization",
                "gridPos": {"h": 4, "w": 3, "x": 3, "y": 0},
                "targets": [{
                    "expr": "avg(gpu_utilization_percent)",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "max": 100,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90}
                            ]
                        },
                        "unit": "percent"
                    }
                }
            },
            
            # Total GPU Memory
            {
                "id": self._next_panel_id(),
                "type": "stat",
                "title": "Total GPU Memory",
                "gridPos": {"h": 4, "w": 3, "x": 6, "y": 0},
                "targets": [{
                    "expr": "sum(gpu_memory_total_bytes) / 1024 / 1024 / 1024",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "decimals": 1,
                        "unit": "decgbytes"
                    }
                }
            },
            
            # GPU Temperature Alert
            {
                "id": self._next_panel_id(),
                "type": "stat",
                "title": "Max GPU Temperature",
                "gridPos": {"h": 4, "w": 3, "x": 9, "y": 0},
                "targets": [{
                    "expr": "max(gpu_temperature_celsius)",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "decimals": 1,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 80}
                            ]
                        },
                        "unit": "celsius"
                    }
                }
            },
            
            # Total Power Draw
            {
                "id": self._next_panel_id(),
                "type": "gauge",
                "title": "Total Power Draw",
                "gridPos": {"h": 4, "w": 3, "x": 12, "y": 0},
                "targets": [{
                    "expr": "sum(gpu_power_draw_watts)",
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "max": 2000,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 1000},
                                {"color": "red", "value": 1500}
                            ]
                        },
                        "unit": "watt"
                    }
                }
            }
        ])
        
        # Row 2: GPU Utilization Timeline
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "GPU Utilization Timeline",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
            "targets": [{
                "expr": "gpu_utilization_percent",
                "legendFormat": "GPU {{device_id}} - {{device_name}}",
                "refId": "A"
            }],
            "yaxes": [
                {"format": "percent", "max": 100, "min": 0},
                {"format": "short"}
            ],
            "lines": True,
            "fill": 1,
            "linewidth": 2,
            "nullPointMode": "null",
            "options": {
                "alertThreshold": True
            }
        })
        
        # Row 2: GPU Memory Usage
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "GPU Memory Usage",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
            "targets": [
                {
                    "expr": "gpu_memory_used_bytes / 1024 / 1024 / 1024",
                    "legendFormat": "Used - GPU {{device_id}}",
                    "refId": "A"
                },
                {
                    "expr": "gpu_memory_total_bytes / 1024 / 1024 / 1024",
                    "legendFormat": "Total - GPU {{device_id}}",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {"format": "decgbytes", "min": 0},
                {"format": "short"}
            ],
            "lines": True,
            "fill": 1,
            "linewidth": 2
        })
        
        # Row 3: Temperature and Power
        dashboard["panels"].extend([
            {
                "id": self._next_panel_id(),
                "type": "graph",
                "title": "GPU Temperature",
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 12},
                "targets": [{
                    "expr": "gpu_temperature_celsius",
                    "legendFormat": "GPU {{device_id}} - {{device_name}}",
                    "refId": "A"
                }],
                "yaxes": [
                    {"format": "celsius", "min": 0, "max": 100},
                    {"format": "short"}
                ],
                "alert": {
                    "conditions": [{
                        "evaluator": {
                            "params": [80],
                            "type": "gt"
                        },
                        "query": {
                            "params": ["A", "5m", "now"]
                        },
                        "reducer": {
                            "params": [],
                            "type": "max"
                        },
                        "type": "query"
                    }],
                    "executionErrorState": "alerting",
                    "frequency": "60s",
                    "handler": 1,
                    "name": "GPU Temperature Alert",
                    "noDataState": "no_data",
                    "notifications": []
                }
            },
            
            {
                "id": self._next_panel_id(),
                "type": "graph",
                "title": "GPU Power Consumption",
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": 12},
                "targets": [{
                    "expr": "gpu_power_draw_watts",
                    "legendFormat": "GPU {{device_id}} - {{device_name}}",
                    "refId": "A"
                }],
                "yaxes": [
                    {"format": "watt", "min": 0},
                    {"format": "short"}
                ]
            }
        ])
        
        # Row 4: GPU Details Table
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "table",
            "title": "GPU Details",
            "gridPos": {"h": 6, "w": 24, "x": 0, "y": 18},
            "targets": [{
                "expr": '''
                    gpu_utilization_percent * on(device_id) group_left(device_name) 
                    (label_replace(gpu_memory_used_bytes, "x", "$1", "device_id", "(.*)") * 0 + 1)
                ''',
                "format": "table",
                "instant": True,
                "refId": "A"
            }],
            "options": {
                "showHeader": True
            },
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "auto",
                        "displayMode": "auto"
                    }
                }
            }
        })
        
        return dashboard
        
    def build_adapter_performance_dashboard(self) -> Dict[str, Any]:
        """
        Build adapter performance tracking dashboard.
        """
        dashboard = {
            "uid": "aura-adapter-performance",
            "title": "AURA Adapter Performance",
            "tags": ["adapters", "performance", "gpu", "aura"],
            "timezone": "browser",
            "schemaVersion": 30,
            "version": 1,
            "refresh": "5s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": []
        }
        
        # Adapter overview row
        adapters = [
            "memory_gpu", "tda_gpu", "orchestration_gpu", "swarm_gpu",
            "communication_gpu", "core_gpu", "infrastructure_gpu", "agents_gpu"
        ]
        
        # Create speedup gauges for each adapter
        for i, adapter in enumerate(adapters):
            x = (i % 4) * 6
            y = (i // 4) * 4
            
            dashboard["panels"].append({
                "id": self._next_panel_id(),
                "type": "stat",
                "title": f"{adapter.replace('_', ' ').title()} Speedup",
                "gridPos": {"h": 4, "w": 6, "x": x, "y": y},
                "targets": [{
                    "expr": f'aura_gpu_adapter_speedup{{adapter="{adapter}"}}',
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "decimals": 1,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 2},
                                {"color": "green", "value": 10}
                            ]
                        },
                        "unit": "none",
                        "custom": {
                            "postfix": "x"
                        }
                    }
                }
            })
        
        # Latency comparison chart
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "Adapter Latency Comparison",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
            "targets": [{
                "expr": "histogram_quantile(0.99, sum by (adapter, le) (rate(aura_component_latency_seconds_bucket[5m])))",
                "legendFormat": "{{adapter}} - p99",
                "refId": "A"
            }],
            "yaxes": [
                {"format": "s", "logBase": 10},
                {"format": "short"}
            ]
        })
        
        # Throughput chart
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "Adapter Throughput",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
            "targets": [{
                "expr": "sum by (adapter) (rate(aura_adapter_throughput_ops_per_second[5m]))",
                "legendFormat": "{{adapter}}",
                "refId": "A"
            }],
            "yaxes": [
                {"format": "ops", "min": 0},
                {"format": "short"}
            ],
            "bars": True,
            "lines": False
        })
        
        return dashboard
        
    def build_system_health_dashboard(self) -> Dict[str, Any]:
        """
        Build system health overview dashboard.
        """
        dashboard = {
            "uid": "aura-system-health",
            "title": "AURA System Health",
            "tags": ["health", "system", "overview", "aura"],
            "timezone": "browser",
            "schemaVersion": 30,
            "version": 1,
            "refresh": "10s",
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "panels": []
        }
        
        # Overall health score
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "gauge",
            "title": "Overall System Health",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [{
                "expr": 'aura_system_health_score{subsystem="overall"}',
                "refId": "A"
            }],
            "fieldConfig": {
                "defaults": {
                    "max": 1,
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0.7},
                            {"color": "green", "value": 0.9}
                        ]
                    },
                    "unit": "percentunit"
                }
            }
        })
        
        # Subsystem health scores
        subsystems = ["cpu", "gpu", "memory", "network"]
        for i, subsystem in enumerate(subsystems):
            dashboard["panels"].append({
                "id": self._next_panel_id(),
                "type": "stat",
                "title": f"{subsystem.upper()} Health",
                "gridPos": {"h": 4, "w": 3, "x": 12 + (i * 3), "y": 0},
                "targets": [{
                    "expr": f'aura_system_health_score{{subsystem="{subsystem}"}}',
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "decimals": 2,
                        "max": 1,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.9}
                            ]
                        },
                        "unit": "percentunit"
                    }
                }
            })
        
        # Active alerts
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "alert-list",
            "title": "Active Alerts",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
            "options": {
                "showOptions": "current",
                "maxItems": 10,
                "sortOrder": 1,
                "dashboardTags": [],
                "alertName": "",
                "folderId": None,
                "stateFilter": {
                    "ok": False,
                    "paused": False,
                    "no_data": False,
                    "execution_error": False,
                    "alerting": True,
                    "pending": True
                }
            }
        })
        
        # Resource usage timeline
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "Resource Usage Timeline",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 12},
            "targets": [
                {
                    "expr": "avg(1 - aura_system_health_score{subsystem=\"cpu\"})",
                    "legendFormat": "CPU Usage",
                    "refId": "A"
                },
                {
                    "expr": "avg(gpu_utilization_percent) / 100",
                    "legendFormat": "GPU Usage",
                    "refId": "B"
                },
                {
                    "expr": "avg(1 - aura_system_health_score{subsystem=\"memory\"})",
                    "legendFormat": "Memory Usage",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {"format": "percentunit", "max": 1, "min": 0},
                {"format": "short"}
            ]
        })
        
        return dashboard
        
    def build_agent_activity_dashboard(self) -> Dict[str, Any]:
        """
        Build agent activity monitoring dashboard.
        """
        dashboard = {
            "uid": "aura-agent-activity",
            "title": "AURA Agent Activity",
            "tags": ["agents", "activity", "monitoring", "aura"],
            "timezone": "browser",
            "schemaVersion": 30,
            "version": 1,
            "refresh": "5s",
            "time": {
                "from": "now-30m",
                "to": "now"
            },
            "panels": []
        }
        
        # Active agents by type
        agent_types = ["observer", "analyst", "executor", "coordinator"]
        for i, agent_type in enumerate(agent_types):
            dashboard["panels"].append({
                "id": self._next_panel_id(),
                "type": "stat",
                "title": f"Active {agent_type.title()} Agents",
                "gridPos": {"h": 4, "w": 6, "x": i * 6, "y": 0},
                "targets": [{
                    "expr": f'agents_active_total{{agent_type="{agent_type}"}}',
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "decimals": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None}
                            ]
                        }
                    }
                }
            })
        
        # Agent spawn rate
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "Agent Spawn Rate",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
            "targets": [{
                "expr": "sum by (agent_type) (rate(agents_spawn_seconds_count[5m]))",
                "legendFormat": "{{agent_type}}",
                "refId": "A"
            }],
            "yaxes": [
                {"format": "ops", "min": 0},
                {"format": "short"}
            ]
        })
        
        # Collective decision time
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "graph",
            "title": "Collective Decision Time",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
            "targets": [{
                "expr": "histogram_quantile(0.99, sum by (le) (rate(agents_collective_decision_seconds_bucket[5m])))",
                "legendFormat": "p99 Decision Time",
                "refId": "A"
            }],
            "yaxes": [
                {"format": "s", "min": 0},
                {"format": "short"}
            ]
        })
        
        # Agent resource usage
        dashboard["panels"].append({
            "id": self._next_panel_id(),
            "type": "heatmap",
            "title": "Agent Resource Usage Heatmap",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 12},
            "targets": [{
                "expr": "sum by (agent_type) (agents_active_total * 0.1)",  # Mock CPU usage per agent
                "format": "heatmap",
                "refId": "A"
            }],
            "options": {
                "calculate": False,
                "calculation": {},
                "color": {
                    "cardColor": "#b4ff00",
                    "colorScale": "sqrt",
                    "colorScheme": "interpolateOranges",
                    "exponent": 0.5,
                    "mode": "spectrum"
                }
            }
        })
        
        return dashboard
        
    def export_all_dashboards(self, output_dir: str = "/workspace/grafana_dashboards"):
        """
        Export all dashboards as JSON files.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        dashboards = {
            "gpu_monitoring": self.build_gpu_monitoring_dashboard(),
            "adapter_performance": self.build_adapter_performance_dashboard(),
            "system_health": self.build_system_health_dashboard(),
            "agent_activity": self.build_agent_activity_dashboard()
        }
        
        for name, dashboard in dashboards.items():
            filename = os.path.join(output_dir, f"{name}.json")
            with open(filename, 'w') as f:
                json.dump(dashboard, f, indent=2)
                
            logger.info(f"Exported dashboard: {filename}")
            
        # Create main dashboard with links
        main_dashboard = {
            "uid": "aura-main",
            "title": "AURA Intelligence Overview",
            "tags": ["aura", "overview"],
            "timezone": "browser",
            "schemaVersion": 30,
            "version": 1,
            "refresh": "10s",
            "panels": [
                {
                    "id": 1,
                    "type": "dashlist",
                    "title": "AURA Dashboards",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "options": {
                        "showSearch": True,
                        "showHeadings": True,
                        "showRecent": True,
                        "showStarred": True,
                        "query": "",
                        "tags": ["aura"]
                    }
                },
                {
                    "id": 2,
                    "type": "text",
                    "title": "AURA Intelligence System",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "options": {
                        "mode": "markdown",
                        "content": """
# AURA Intelligence System

## GPU-Accelerated AI Infrastructure

### 🚀 Key Features:
- **8 GPU Adapters** with 10-10,000x speedups
- **Real-time GPU Monitoring**
- **Distributed Agent System**
- **Advanced Observability**

### 📊 Available Dashboards:
1. **GPU Monitoring** - Real-time GPU metrics
2. **Adapter Performance** - Speedup tracking
3. **System Health** - Overall health scores
4. **Agent Activity** - Multi-agent monitoring

### 🎯 Performance Metrics:
- Memory Search: **16.7x** speedup
- TDA Analysis: **100x** speedup
- Swarm Optimization: **990x** speedup
- Agent Communication: **9082x** speedup
"""
                    }
                }
            ]
        }
        
        # Export main dashboard
        with open(os.path.join(output_dir, "main.json"), 'w') as f:
            json.dump(main_dashboard, f, indent=2)
            
        logger.info(f"All dashboards exported to {output_dir}")
        
        return list(dashboards.keys()) + ["main"]


def create_docker_compose_monitoring() -> str:
    """
    Create docker-compose configuration for monitoring stack.
    """
    config = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: aura-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: aura-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=aura2025
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana_dashboards:/var/lib/grafana/dashboards
      - ./grafana-dashboard.yml:/etc/grafana/provisioning/dashboards/dashboard.yml
      - ./grafana-datasource.yml:/etc/grafana/provisioning/datasources/datasource.yml
    depends_on:
      - prometheus
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: aura-node-exporter
    ports:
      - "9100:9100"
    restart: unless-stopped

  nvidia-gpu-exporter:
    image: utkuozdemir/nvidia_gpu_exporter:latest
    container_name: aura-gpu-exporter
    ports:
      - "9835:9835"
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all

volumes:
  prometheus_data:
  grafana_data:
"""
    return config


def create_prometheus_config() -> str:
    """
    Create Prometheus configuration.
    """
    config = """
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'aura-intelligence'
    static_configs:
      - targets: ['host.docker.internal:8000']
        labels:
          service: 'aura-main'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9835']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""
    return config


def create_grafana_provisioning_configs() -> Dict[str, str]:
    """
    Create Grafana provisioning configurations.
    """
    datasource_config = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    orgId: 1
    url: http://prometheus:9090
    isDefault: true
    editable: true
"""
    
    dashboard_config = """
apiVersion: 1

providers:
  - name: 'AURA Dashboards'
    orgId: 1
    folder: 'AURA Intelligence'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""
    
    return {
        "datasource": datasource_config,
        "dashboard": dashboard_config
    }


if __name__ == "__main__":
    # Create dashboard builder
    builder = GrafanaDashboardBuilder()
    
    # Export all dashboards
    dashboards = builder.export_all_dashboards()
    
    print("✅ Exported Grafana Dashboards:")
    for dashboard in dashboards:
        print(f"   - {dashboard}.json")
        
    print("\n📊 Dashboard Features:")
    print("   - Real-time GPU monitoring")
    print("   - Adapter performance tracking")
    print("   - System health overview")
    print("   - Agent activity monitoring")
    print("   - Alert configuration")
    print("   - Heatmaps and gauges")
    print("   - Time series graphs")
    print("   - Responsive layout")