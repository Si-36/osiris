"""
🖼️ Dashboard Preview Demo
=========================

Shows a preview of what the dashboards will look like.
"""

import json
import os


def print_dashboard_preview():
    """Print ASCII art preview of dashboards"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    🖥️  AURA GPU MONITORING                          ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     ║
    ║  │ GPUs: 4 │ │ Util:   │ │ Memory: │ │ Temp:   │ │ Power:  │     ║
    ║  │    🟢   │ │  78.5%  │ │ 64.0 GB │ │  72°C   │ │ 1.2 kW  │     ║
    ║  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     ║
    ║                                                                    ║
    ║  GPU Utilization Timeline          │ GPU Memory Usage             ║
    ║  100% ┤                           │ 16GB ┤                        ║
    ║   75% ┤    ╱╲    ╱╲              │ 12GB ┤  ▄▄▄▄▄▄▄▄▄            ║
    ║   50% ┤ ╱╲╱  ╲╱╲╱  ╲            │  8GB ┤ █████████████         ║
    ║   25% ┤╱            ╲           │  4GB ┤                       ║
    ║    0% └──────────────────        │  0GB └──────────────────     ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                 ⚡ ADAPTER PERFORMANCE TRACKING                      ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             ║
    ║  │ Memory   │ │ TDA      │ │ Swarm    │ │ Comm     │             ║
    ║  │ 16.7x 🚀 │ │ 100x 🚀  │ │ 990x 🚀  │ │ 9082x 🚀 │             ║
    ║  └──────────┘ └──────────┘ └──────────┘ └──────────┘             ║
    ║                                                                    ║
    ║  Adapter Latency (p99)             │ Throughput (ops/sec)         ║
    ║  10ms ┤                           │ 100k ┤  ▐█▌                  ║
    ║   1ms ┤    ▄                      │  10k ┤  ▐█▌ ▐█▌              ║
    ║ 0.1ms ┤ ▄  █  ▄  ▄               │   1k ┤  ▐█▌ ▐█▌ ▐█▌ ▐█▌      ║
    ║       └─────────────────          │      └──────────────────     ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    💚 SYSTEM HEALTH OVERVIEW                        ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Overall Health: ████████████████░░ 92%                            ║
    ║                                                                    ║
    ║  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                 ║
    ║  │ CPU: 85%│ │ GPU: 95%│ │ MEM: 90%│ │ NET: 98%│                 ║
    ║  └─────────┘ └─────────┘ └─────────┘ └─────────┘                 ║
    ║                                                                    ║
    ║  Active Alerts:                    │ Resource Usage                ║
    ║  ⚠️  GPU 2 temp > 80°C             │ 100% ┤      ╱─╲              ║
    ║  ⚠️  Memory usage > 90%            │  50% ┤ ─╲╱─╱   ╲─           ║
    ║                                   │   0% └──────────────────     ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    🤖 AGENT ACTIVITY MONITOR                        ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             ║
    ║  │ Observer │ │ Analyst  │ │ Executor │ │ Coord.   │             ║
    ║  │   142    │ │    87    │ │    56    │ │    23    │             ║
    ║  └──────────┘ └──────────┘ └──────────┘ └──────────┘             ║
    ║                                                                    ║
    ║  Agent Spawn Rate                  │ Decision Time (p99)          ║
    ║  100/s ┤      ╱╲                   │ 100ms ┤                      ║
    ║   50/s ┤  ╱╲╱╱  ╲                  │  10ms ┤    ────────          ║
    ║    0/s └──────────────────         │   1ms └──────────────────    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


def show_metrics_available():
    """Show available metrics for monitoring"""
    
    print("\n📊 Available Prometheus Metrics:")
    print("=" * 70)
    
    metrics = [
        ("GPU Metrics", [
            "gpu_utilization_percent - GPU utilization percentage (0-100)",
            "gpu_memory_used_bytes - GPU memory usage in bytes",
            "gpu_temperature_celsius - GPU temperature in Celsius",
            "gpu_power_draw_watts - GPU power consumption in watts",
            "gpu_clock_speed_mhz - GPU/Memory clock speeds"
        ]),
        ("Adapter Metrics", [
            "aura_gpu_adapter_speedup - Speedup factor vs CPU baseline",
            "aura_component_latency_seconds - Operation latency histogram",
            "aura_adapter_throughput_ops_per_second - Operations per second"
        ]),
        ("System Health", [
            "aura_system_health_score - Health score (0-1) by subsystem",
            "agents_active_total - Active agents by type",
            "agents_spawn_seconds - Agent spawn time histogram",
            "agents_collective_decision_seconds - Decision time"
        ]),
        ("Additional Metrics", [
            "cuda_kernel_launches_total - CUDA kernel launch count",
            "tensor_core_utilization_percent - Tensor core usage",
            "memory_bandwidth_utilization_percent - Memory bandwidth",
            "pcie_bandwidth_gbps - PCIe throughput"
        ])
    ]
    
    for category, metric_list in metrics:
        print(f"\n{category}:")
        for metric in metric_list:
            print(f"   • {metric}")


def show_alert_rules():
    """Show configured alert rules"""
    
    print("\n🚨 Configured Alert Rules:")
    print("=" * 70)
    
    alerts = [
        {
            "name": "GPU Temperature Critical",
            "condition": "gpu_temperature_celsius > 80",
            "severity": "critical",
            "action": "Throttle GPU workload"
        },
        {
            "name": "GPU Memory Exhausted",
            "condition": "gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95",
            "severity": "warning",
            "action": "Clear GPU cache or reduce batch size"
        },
        {
            "name": "System Health Degraded",
            "condition": "aura_system_health_score{subsystem='overall'} < 0.7",
            "severity": "warning",
            "action": "Check subsystem health scores"
        },
        {
            "name": "Adapter Error Rate High",
            "condition": "rate(adapter_errors[5m]) > 0.05",
            "severity": "warning",
            "action": "Check adapter logs and fallback to CPU"
        },
        {
            "name": "Agent Spawn Failures",
            "condition": "rate(agents_spawn_failures[5m]) > 10",
            "severity": "critical",
            "action": "Check resource availability"
        }
    ]
    
    for alert in alerts:
        print(f"\n{alert['name']}:")
        print(f"   Condition: {alert['condition']}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Action: {alert['action']}")


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║      📊 GRAFANA DASHBOARD PREVIEW & METRICS 📊         ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Show dashboard preview
    print_dashboard_preview()
    
    # Show metrics
    show_metrics_available()
    
    # Show alerts
    show_alert_rules()
    
    # Show file locations
    print("\n📁 Generated Files:")
    print("=" * 70)
    
    dashboard_dir = "/workspace/grafana_dashboards"
    if os.path.exists(dashboard_dir):
        files = os.listdir(dashboard_dir)
        print(f"\nDashboards ({dashboard_dir}):")
        for file in sorted(files):
            if file.endswith('.json'):
                filepath = os.path.join(dashboard_dir, file)
                size = os.path.getsize(filepath)
                print(f"   • {file} ({size:,} bytes)")
    
    print("\nDeployment Files (/workspace):")
    deployment_files = [
        "docker-compose.monitoring.yml",
        "prometheus.yml",
        "grafana-datasource.yml",
        "grafana-dashboard.yml"
    ]
    
    for file in deployment_files:
        filepath = os.path.join("/workspace", file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   • {file} ({size:,} bytes)")
    
    print("\n✅ Dashboards ready for deployment!")
    print("   Run: docker-compose -f docker-compose.monitoring.yml up -d")


if __name__ == "__main__":
    main()