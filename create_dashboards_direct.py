"""
✅ Direct Dashboard Creation
============================

Creates Grafana dashboards directly without problematic imports.
"""

import json
import os
import sys

# Add path and import only what we need
sys.path.insert(0, '/workspace/core/src/aura_intelligence/observability')

from grafana_dashboards import (
    GrafanaDashboardBuilder,
    create_docker_compose_monitoring,
    create_prometheus_config,
    create_grafana_provisioning_configs
)


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║        📊 CREATING GRAFANA DASHBOARDS 📊               ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Create dashboard builder
    builder = GrafanaDashboardBuilder()
    print("✅ Dashboard builder created")
    
    # Export all dashboards
    print("\n🎨 Creating and exporting dashboards...")
    output_dir = "/workspace/grafana_dashboards"
    exported = builder.export_all_dashboards(output_dir)
    
    print(f"\n✅ Successfully exported {len(exported)} dashboards:")
    for dashboard in exported:
        filepath = os.path.join(output_dir, f"{dashboard}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                panel_count = len(data.get('panels', []))
                print(f"   - {dashboard}.json ({panel_count} panels)")
    
    # Show dashboard details
    print("\n📊 Dashboard Details:")
    print("-" * 60)
    
    dashboards = {
        "gpu_monitoring": "GPU Monitoring Dashboard",
        "adapter_performance": "Adapter Performance Dashboard",
        "system_health": "System Health Dashboard",
        "agent_activity": "Agent Activity Dashboard",
        "main": "Main Overview Dashboard"
    }
    
    for filename, description in dashboards.items():
        filepath = os.path.join(output_dir, f"{filename}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            print(f"\n{description}:")
            print(f"   UID: {data.get('uid', 'N/A')}")
            print(f"   Title: {data.get('title', 'N/A')}")
            print(f"   Tags: {', '.join(data.get('tags', []))}")
            print(f"   Panels: {len(data.get('panels', []))}")
            
            # Show panel types
            panel_types = {}
            for panel in data.get('panels', []):
                ptype = panel.get('type', 'unknown')
                panel_types[ptype] = panel_types.get(ptype, 0) + 1
                
            if panel_types:
                print("   Panel Types:")
                for ptype, count in sorted(panel_types.items()):
                    print(f"      - {ptype}: {count}")
    
    # Show deployment files
    print("\n📁 Deployment Files Created:")
    print("-" * 60)
    
    files = [
        ("docker-compose.monitoring.yml", "Docker Compose stack"),
        ("prometheus.yml", "Prometheus configuration"),
        ("grafana-datasource.yml", "Grafana datasource config"),
        ("grafana-dashboard.yml", "Grafana dashboard provisioning")
    ]
    
    for filename, description in files:
        filepath = os.path.join("/workspace", filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   ✅ {filename} ({size} bytes) - {description}")
        else:
            print(f"   ❌ {filename} - Missing")
    
    print("\n🚀 Quick Start:")
    print("-" * 60)
    print("1. Start the monitoring stack:")
    print("   docker-compose -f docker-compose.monitoring.yml up -d")
    print("\n2. Access Grafana:")
    print("   http://localhost:3000 (admin/aura2025)")
    print("\n3. Dashboards will be auto-imported in 'AURA Intelligence' folder")
    
    print("\n✅ Dashboard creation complete!")


if __name__ == "__main__":
    main()