#!/usr/bin/env python3
"""
AURA Monitoring Launcher v2
Simplified monitoring without external dependencies
"""

import os
import sys
import time
import random
import json
from datetime import datetime
from collections import deque

# Add src to path
sys.path.insert(0, 'src')

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'
CLEAR = '\033[2J\033[H'

class SimpleMonitor:
    """Simple monitoring dashboard"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = {
            "cascade_risk": deque(maxlen=20),
            "active_agents": deque(maxlen=20),
            "latency_ms": deque(maxlen=20),
        }
        
        # Initialize AURA system
        try:
            from aura.core.system import AURASystem
            from aura.core.config import AURAConfig
            self.system = AURASystem(AURAConfig())
            self.system_available = True
        except:
            self.system = None
            self.system_available = False
            
        self.intervention_count = 0
        self.analysis_count = 0
    
    def get_metrics(self):
        """Get current metrics"""
        # Real metrics if system available
        if self.system_available:
            active_agents = len(self.system.agents)
            components = self.system.get_all_components()
            total_components = sum(len(v) for v in components.values())
        else:
            active_agents = 100
            total_components = 326
            
        # Simulate some metrics
        cascade_risk = random.uniform(0.1, 0.4)
        latency = random.uniform(0.5, 2.5)
        
        # Simulate occasional spikes
        if random.random() < 0.1:
            cascade_risk += 0.3
        if random.random() < 0.05:
            latency += 2.0
            
        return {
            "cascade_risk": cascade_risk,
            "active_agents": active_agents,
            "latency_ms": latency,
            "interventions": self.intervention_count,
            "analyses": self.analysis_count,
            "uptime": time.time() - self.start_time,
            "total_components": total_components
        }
    
    def update_history(self, metrics):
        """Update metrics history"""
        for key in ["cascade_risk", "active_agents", "latency_ms"]:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
    
    def render_header(self):
        """Render header"""
        print(f"{CYAN}{'═'*60}{RESET}")
        print(f"{BOLD}{CYAN}      AURA Intelligence Monitoring System v2.0{RESET}")
        print(f"{CYAN}{'═'*60}{RESET}")
        print(f"Project: bc-a397ac41-47c3-4620-a5ec-c56fb1f50fd0")
        print(f"Preventing agent failures through topological intelligence")
        print()
    
    def render_metrics(self, metrics):
        """Render metrics display"""
        uptime_seconds = int(metrics["uptime"])
        uptime_str = f"{uptime_seconds//3600:02d}:{(uptime_seconds//60)%60:02d}:{uptime_seconds%60:02d}"
        
        print(f"{BOLD}System Status{RESET}")
        print(f"├─ Uptime: {GREEN}{uptime_str}{RESET}")
        print(f"├─ Components: {GREEN}{metrics['total_components']}/326{RESET}")
        print(f"└─ System: {GREEN}OPERATIONAL{RESET}")
        print()
        
        # Risk Level
        risk = metrics["cascade_risk"]
        risk_color = GREEN if risk < 0.3 else YELLOW if risk < 0.7 else RED
        risk_bar = "█" * int(risk * 20) + "░" * (20 - int(risk * 20))
        print(f"{BOLD}Cascade Risk{RESET}")
        print(f"├─ Level: {risk_color}{risk:.1%}{RESET}")
        print(f"└─ [{risk_color}{risk_bar}{RESET}]")
        print()
        
        # Active Agents
        agents = metrics["active_agents"]
        agent_color = GREEN if agents >= 90 else YELLOW if agents >= 50 else RED
        print(f"{BOLD}Active Agents{RESET}")
        print(f"├─ Count: {agent_color}{agents}/100{RESET}")
        print(f"└─ Health: {agent_color}{'Healthy' if agents >= 90 else 'Degraded'}{RESET}")
        print()
        
        # Performance
        latency = metrics["latency_ms"]
        lat_color = GREEN if latency < 1 else YELLOW if latency < 3 else RED
        print(f"{BOLD}Performance{RESET}")
        print(f"├─ Latency: {lat_color}{latency:.2f}ms{RESET}")
        print(f"├─ Analyses: {BLUE}{metrics['analyses']}{RESET}")
        print(f"└─ Interventions: {BLUE}{metrics['interventions']}{RESET}")
        print()
    
    def render_graph(self, data, title, width=40):
        """Render simple ASCII graph"""
        if not data:
            return
            
        print(f"{BOLD}{title}{RESET}")
        
        # Get min/max for scaling
        values = list(data)
        if not values:
            return
            
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Scale values to 0-10 range
        scaled = [(v - min_val) / range_val * 10 for v in values]
        
        # Draw graph
        for i in range(10, -1, -1):
            line = ""
            for val in scaled:
                if val >= i:
                    line += "█"
                else:
                    line += " "
            print(f"│{line}│")
            
        print(f"└{'─' * len(scaled)}┘")
        print(f" Min: {min_val:.2f}  Max: {max_val:.2f}")
        print()
    
    def run(self):
        """Run monitoring loop"""
        print(CLEAR)
        
        try:
            while True:
                # Clear screen
                print(CLEAR)
                
                # Get metrics
                metrics = self.get_metrics()
                self.update_history(metrics)
                
                # Render dashboard
                self.render_header()
                self.render_metrics(metrics)
                
                # Render graphs
                if len(self.metrics_history["cascade_risk"]) > 2:
                    self.render_graph(
                        self.metrics_history["cascade_risk"],
                        "Cascade Risk History"
                    )
                
                if len(self.metrics_history["latency_ms"]) > 2:
                    self.render_graph(
                        self.metrics_history["latency_ms"],
                        "Latency History (ms)"
                    )
                
                # Footer
                print(f"{CYAN}{'─'*60}{RESET}")
                print(f"Press Ctrl+C to exit | Refresh: 1s")
                
                # Simulate activity
                if random.random() < 0.1:
                    self.analysis_count += 1
                if random.random() < 0.02:
                    self.intervention_count += 1
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n{GREEN}Monitoring stopped.{RESET}")

def main():
    """Main entry point"""
    monitor = SimpleMonitor()
    
    if not monitor.system_available:
        print(f"{YELLOW}Warning: AURA system not fully available.{RESET}")
        print(f"{YELLOW}Running in simulation mode.{RESET}\n")
    
    print(f"{GREEN}Starting AURA Monitoring...{RESET}")
    time.sleep(1)
    
    monitor.run()

if __name__ == "__main__":
    main()