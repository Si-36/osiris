"""
Real Terminal Dashboard - Shows live system metrics
No fake data - connects to real API and displays actual metrics
"""
import time
import json
import requests
import websocket
import threading
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class AURADashboard:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.latest_metrics = None
        self.ws_connected = False
        self.history = []
        
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_metrics_summary(self):
        """Get metrics summary from API"""
        try:
            response = requests.get(f"{self.api_url}/metrics/summary", timeout=2)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def create_dashboard(self):
        """Create the dashboard layout"""
        layout = Layout()
        
        # Header
        header = Panel(
            Text("🚀 AURA Real-Time System Monitor", style="bold cyan", justify="center"),
            style="bold white on blue"
        )
        
        # System metrics table
        metrics_table = self.create_metrics_table()
        
        # Summary stats
        summary = self.create_summary_panel()
        
        # Status
        status = self.create_status_panel()
        
        # Arrange layout
        layout.split_column(
            Layout(header, size=3),
            Layout(name="main"),
            Layout(status, size=5)
        )
        
        layout["main"].split_row(
            Layout(metrics_table, name="metrics"),
            Layout(summary, name="summary")
        )
        
        return layout
    
    def create_metrics_table(self):
        """Create table showing current metrics"""
        table = Table(title="Current System Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        if self.latest_metrics:
            # CPU
            cpu_percent = self.latest_metrics.get('cpu', {}).get('percent', 0)
            cpu_status = "🔴 HIGH" if cpu_percent > 80 else "🟡 MEDIUM" if cpu_percent > 50 else "🟢 NORMAL"
            table.add_row("CPU Usage", f"{cpu_percent:.1f}%", cpu_status)
            
            # Memory
            mem_percent = self.latest_metrics.get('memory', {}).get('percent', 0)
            mem_status = "🔴 HIGH" if mem_percent > 80 else "🟡 MEDIUM" if mem_percent > 50 else "🟢 NORMAL"
            table.add_row("Memory Usage", f"{mem_percent:.1f}%", mem_status)
            
            # Disk
            disk_percent = self.latest_metrics.get('disk', {}).get('percent', 0)
            disk_status = "🔴 HIGH" if disk_percent > 90 else "🟡 MEDIUM" if disk_percent > 70 else "🟢 NORMAL"
            table.add_row("Disk Usage", f"{disk_percent:.1f}%", disk_status)
            
            # Network
            network = self.latest_metrics.get('network', {})
            bytes_sent = network.get('bytes_sent', 0) / 1024 / 1024  # Convert to MB
            bytes_recv = network.get('bytes_recv', 0) / 1024 / 1024
            table.add_row("Network Sent", f"{bytes_sent:.1f} MB", "📤")
            table.add_row("Network Received", f"{bytes_recv:.1f} MB", "📥")
            
            # Processes
            processes = self.latest_metrics.get('processes', 0)
            table.add_row("Active Processes", str(processes), "🔄")
            
            # Timestamp
            timestamp = self.latest_metrics.get('timestamp', '')
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                table.add_row("Last Update", dt.strftime("%H:%M:%S"), "⏰")
        else:
            table.add_row("No data", "Waiting for metrics...", "⏳")
        
        return Panel(table, title="📊 Live Metrics", border_style="green")
    
    def create_summary_panel(self):
        """Create summary statistics panel"""
        summary_data = self.get_metrics_summary()
        
        if summary_data:
            content = []
            content.append(f"📈 Sampling Period: {summary_data.get('period', 'N/A')}")
            content.append(f"📊 Total Samples: {summary_data.get('samples', 0)}")
            content.append("")
            
            # CPU stats
            cpu = summary_data.get('cpu', {})
            content.append("🖥️  CPU Statistics:")
            content.append(f"  Current: {cpu.get('current', 0):.1f}%")
            content.append(f"  Average: {cpu.get('avg', 0):.1f}%")
            content.append(f"  Min/Max: {cpu.get('min', 0):.1f}% / {cpu.get('max', 0):.1f}%")
            content.append("")
            
            # Memory stats
            mem = summary_data.get('memory', {})
            content.append("💾 Memory Statistics:")
            content.append(f"  Current: {mem.get('current', 0):.1f}%")
            content.append(f"  Average: {mem.get('avg', 0):.1f}%")
            content.append(f"  Min/Max: {mem.get('min', 0):.1f}% / {mem.get('max', 0):.1f}%")
            
            text = "\n".join(content)
        else:
            text = "📊 Loading statistics..."
        
        return Panel(text, title="📈 Statistics (Last Hour)", border_style="blue")
    
    def create_status_panel(self):
        """Create status panel"""
        health = self.check_api_health()
        
        if health:
            status = f"✅ API: {health.get('status', 'unknown')}"
            redis = "✅ Redis: Connected" if health.get('redis_connected') else "❌ Redis: Disconnected"
            uptime = f"⏱️  Uptime: {health.get('uptime_seconds', 0):.0f}s"
            ws = "✅ WebSocket: Connected" if self.ws_connected else "❌ WebSocket: Disconnected"
            
            text = f"{status} | {redis} | {ws} | {uptime}"
        else:
            text = "❌ API: Not responding - Make sure to run the API server!"
        
        return Panel(text, title="🔌 Connection Status", border_style="yellow")
    
    def start_websocket(self):
        """Start WebSocket connection for real-time updates"""
        def on_message(ws, message):
            try:
                self.latest_metrics = json.loads(message)
                self.ws_connected = True
            except:
                pass
        
        def on_error(ws, error):
            self.ws_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
        
        def on_open(ws):
            self.ws_connected = True
        
        try:
            ws_url = self.api_url.replace("http://", "ws://") + "/ws"
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except:
            self.ws_connected = False
    
    def run(self):
        """Run the dashboard"""
        console.print("[bold cyan]🚀 Starting AURA Real-Time Dashboard...[/bold cyan]")
        console.print("[yellow]Connecting to API at " + self.api_url + "[/yellow]")
        
        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=self.start_websocket, daemon=True)
        ws_thread.start()
        
        # Give WebSocket time to connect
        time.sleep(1)
        
        # Run live dashboard
        with Live(self.create_dashboard(), refresh_per_second=2, console=console) as live:
            try:
                while True:
                    live.update(self.create_dashboard())
                    time.sleep(0.5)
            except KeyboardInterrupt:
                console.print("\n[bold red]👋 Dashboard stopped[/bold red]")


if __name__ == "__main__":
    console.print("""
[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]
[bold white]          🚀 AURA REAL-TIME SYSTEM MONITOR 🚀              [/bold white]
[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]

[yellow]This dashboard shows REAL system metrics - no fake data![/yellow]

[green]To see data flowing:[/green]
1. Start Redis: [cyan]docker run -p 6379:6379 redis[/cyan]
2. Start Collector: [cyan]python real_aura/core/collector.py[/cyan]
3. Start API: [cyan]python real_aura/api/main.py[/cyan]
4. Watch the magic happen! ✨

[dim]Press Ctrl+C to exit[/dim]
""")
    
    dashboard = AURADashboard()
    dashboard.run()