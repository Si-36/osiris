"""
AURA Advanced Monitoring System
Real-time monitoring with AI-specific metrics
"""

import asyncio
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
import plotext as plt
from collections import deque
import json

# Import AURA components
from ..core.system import AURASystem
from ..core.config import AURAConfig

console = Console()

class AURAMonitor:
    """Advanced monitoring system for AURA"""
    
    def __init__(self, system: Optional[AURASystem] = None):
        self.system = system or AURASystem(AURAConfig())
        self.metrics_history = {
            "cascade_risk": deque(maxlen=60),
            "active_agents": deque(maxlen=60),
            "interventions": deque(maxlen=60),
            "cpu_percent": deque(maxlen=60),
            "memory_mb": deque(maxlen=60),
            "latency_ms": deque(maxlen=60),
        }
        self.start_time = time.time()
        self.intervention_count = 0
        self.analysis_count = 0
        self.prediction_count = 0
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        
        # AURA metrics
        system_status = self.system.get_system_status()
        active_agents = len(self.system.agents)
        
        # Simulate cascade risk (would be calculated from topology)
        cascade_risk = np.random.uniform(0, 0.3) + (0.1 if active_agents > 80 else 0)
        
        # Latency simulation
        latency_ms = np.random.uniform(0.5, 2.0)
        
        metrics = {
            "cascade_risk": cascade_risk,
            "active_agents": active_agents,
            "interventions": self.intervention_count,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "latency_ms": latency_ms,
            "uptime": time.time() - self.start_time,
            "analysis_count": self.analysis_count,
            "prediction_count": self.prediction_count,
        }
        
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics
    
    def create_dashboard(self) -> Layout:
        """Create monitoring dashboard layout"""
        layout = Layout()
        
        # Main layout
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main into sections
        layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="graphs", ratio=2)
        )
        
        # Split graphs into multiple charts
        layout["graphs"].split_column(
            Layout(name="risk_chart"),
            Layout(name="performance_chart")
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render dashboard header"""
        metrics = self.get_system_metrics()
        uptime_str = str(timedelta(seconds=int(metrics["uptime"])))
        
        header_text = f"""[bold cyan]AURA Intelligence Monitoring System[/bold cyan]
[dim]Preventing agent failures through topological intelligence[/dim]
Uptime: {uptime_str} | Components: 213 | Version: 2.0.0"""
        
        return Panel(header_text, box_type="double", style="cyan")
    
    def render_metrics(self) -> Panel:
        """Render metrics table"""
        metrics = self.get_system_metrics()
        
        table = Table(title="System Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right", style="green")
        table.add_column("Status", justify="center", width=10)
        
        # Cascade Risk
        risk_status = "🟢" if metrics["cascade_risk"] < 0.3 else "🟡" if metrics["cascade_risk"] < 0.7 else "🔴"
        table.add_row("Cascade Risk", f"{metrics['cascade_risk']:.2%}", risk_status)
        
        # Active Agents
        agent_status = "🟢" if metrics["active_agents"] >= 90 else "🟡" if metrics["active_agents"] >= 50 else "🔴"
        table.add_row("Active Agents", str(metrics["active_agents"]), agent_status)
        
        # Interventions
        table.add_row("Interventions", str(metrics["interventions"]), "🔵")
        
        # CPU Usage
        cpu_status = "🟢" if metrics["cpu_percent"] < 50 else "🟡" if metrics["cpu_percent"] < 80 else "🔴"
        table.add_row("CPU Usage", f"{metrics['cpu_percent']:.1f}%", cpu_status)
        
        # Memory Usage
        mem_status = "🟢" if metrics["memory_mb"] < 1000 else "🟡" if metrics["memory_mb"] < 2000 else "🔴"
        table.add_row("Memory", f"{metrics['memory_mb']:.0f} MB", mem_status)
        
        # Latency
        lat_status = "🟢" if metrics["latency_ms"] < 1 else "🟡" if metrics["latency_ms"] < 3 else "🔴"
        table.add_row("Latency", f"{metrics['latency_ms']:.2f} ms", lat_status)
        
        # Analysis Count
        table.add_row("Analyses", str(metrics["analysis_count"]), "📊")
        
        # Prediction Count
        table.add_row("Predictions", str(metrics["prediction_count"]), "🔮")
        
        # Component Health
        table.add_section()
        components = self.system.get_all_components()
        table.add_row("TDA Algorithms", f"{len(components['tda_algorithms'])}/112", 
                     "🟢" if len(components['tda_algorithms']) == 112 else "🔴")
        table.add_row("Neural Networks", f"{len(components['neural_networks'])}/10",
                     "🟢" if len(components['neural_networks']) == 10 else "🔴")
        table.add_row("Memory Systems", f"{len(components['memory_components'])}/40",
                     "🟢" if len(components['memory_components']) == 40 else "🔴")
        table.add_row("Agent Systems", f"{len(components['agents'])}/100",
                     "🟢" if len(components['agents']) == 100 else "🔴")
        
        return Panel(table, title="[bold]Live Metrics[/bold]", border_style="green")
    
    def render_risk_chart(self) -> Panel:
        """Render cascade risk chart"""
        plt.clf()
        plt.theme("dark")
        plt.title("Cascade Risk Over Time")
        
        if len(self.metrics_history["cascade_risk"]) > 1:
            risk_values = list(self.metrics_history["cascade_risk"])
            x = list(range(len(risk_values)))
            
            plt.plot(x, risk_values, label="Risk Level", color="red")
            plt.plot(x, [0.3] * len(x), label="Warning", color="yellow", marker="dot")
            plt.plot(x, [0.7] * len(x), label="Critical", color="red", marker="dot")
            
            plt.ylim(0, 1)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Risk Level")
            
        chart = plt.build()
        return Panel(chart, title="[bold]Cascade Risk Analysis[/bold]", border_style="red")
    
    def render_performance_chart(self) -> Panel:
        """Render performance metrics chart"""
        plt.clf()
        plt.theme("dark")
        plt.title("System Performance")
        
        if len(self.metrics_history["latency_ms"]) > 1:
            latency_values = list(self.metrics_history["latency_ms"])
            cpu_values = [v/50 for v in self.metrics_history["cpu_percent"]]  # Scale to 0-2
            
            x = list(range(len(latency_values)))
            
            plt.plot(x, latency_values, label="Latency (ms)", color="cyan")
            plt.plot(x, cpu_values, label="CPU (scaled)", color="green")
            plt.plot(x, [1.0] * len(x), label="Target", color="yellow", marker="dot")
            
            plt.ylim(0, 3)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Performance")
            
        chart = plt.build()
        return Panel(chart, title="[bold]Performance Metrics[/bold]", border_style="cyan")
    
    def render_footer(self) -> Panel:
        """Render dashboard footer"""
        shortcuts = """[bold]Shortcuts:[/bold] Q=Quit | R=Reset | I=Intervene | A=Analyze | P=Pause
[dim]AURA Intelligence © 2025 | Project: bc-a397ac41-47c3-4620-a5ec-c56fb1f50fd0[/dim]"""
        
        return Panel(shortcuts, box_type="single", style="dim")
    
    async def update_dashboard(self, layout: Layout):
        """Update dashboard with latest data"""
        layout["header"].update(self.render_header())
        layout["metrics"].update(self.render_metrics())
        layout["risk_chart"].update(self.render_risk_chart())
        layout["performance_chart"].update(self.render_performance_chart())
        layout["footer"].update(self.render_footer())
        
        # Simulate some activity
        if np.random.random() < 0.1:
            self.analysis_count += 1
        if np.random.random() < 0.05:
            self.prediction_count += 1
        if np.random.random() < 0.02:
            self.intervention_count += 1
    
    async def run(self):
        """Run the monitoring dashboard"""
        layout = self.create_dashboard()
        
        with Live(layout, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    await self.update_dashboard(layout)
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

# Additional monitoring utilities

class MetricsCollector:
    """Collect and aggregate metrics for analysis"""
    
    def __init__(self):
        self.metrics_buffer = []
        self.aggregation_window = 60  # seconds
        
    def collect(self, metrics: Dict[str, Any]):
        """Collect metrics with timestamp"""
        metrics["timestamp"] = datetime.utcnow().isoformat()
        self.metrics_buffer.append(metrics)
        
        # Clean old metrics
        cutoff = datetime.utcnow() - timedelta(seconds=self.aggregation_window)
        self.metrics_buffer = [m for m in self.metrics_buffer 
                              if datetime.fromisoformat(m["timestamp"]) > cutoff]
    
    def get_aggregated(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        if not self.metrics_buffer:
            return {}
        
        # Calculate aggregates
        cascade_risks = [m.get("cascade_risk", 0) for m in self.metrics_buffer]
        latencies = [m.get("latency_ms", 0) for m in self.metrics_buffer]
        
        return {
            "avg_cascade_risk": np.mean(cascade_risks),
            "max_cascade_risk": np.max(cascade_risks),
            "avg_latency_ms": np.mean(latencies),
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            "sample_count": len(self.metrics_buffer),
            "time_window": self.aggregation_window,
        }

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            "cascade_risk": 0.7,
            "latency_ms": 5.0,
            "cpu_percent": 80.0,
            "memory_mb": 2000.0,
        }
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds"""
        new_alerts = []
        
        for metric, threshold in self.thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert = {
                    "metric": metric,
                    "value": metrics[metric],
                    "threshold": threshold,
                    "severity": "critical" if metrics[metric] > threshold * 1.5 else "warning",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"{metric} exceeded threshold: {metrics[metric]:.2f} > {threshold}"
                }
                new_alerts.append(alert)
                self.alerts.append(alert)
        
        return new_alerts

if __name__ == "__main__":
    # Run monitoring dashboard
    monitor = AURAMonitor()
    asyncio.run(monitor.run())