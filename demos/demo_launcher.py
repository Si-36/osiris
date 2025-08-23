#!/usr/bin/env python3
"""
AURA Intelligence Demo Launcher
Choose your demo to showcase different capabilities
"""

import subprocess
import sys
import os

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              🚀 AURA INTELLIGENCE DEMOS 🚀                ║
    ║                                                           ║
    ║  World-class GPU optimization meets real business value   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

def print_demos():
    print("""
    Choose a demo to run:
    
    1️⃣  Intelligence Extraction (demo_intelligence.py)
        📄 Process documents and extract actionable insights
        ⚡ 3.2ms processing with GPU acceleration
        🎯 Perfect for: Document analysis, compliance, research
        
    2️⃣  Real-Time Pattern Detection (demo_pattern_detection.py)
        🔍 Detect anomalies in streaming data
        📊 Live dashboard with risk scoring
        🎯 Perfect for: Monitoring, security, operations
        
    3️⃣  Edge AI Intelligence (demo_edge_intelligence.py)
        ⚡ Neuromorphic processing - 1000x energy efficiency
        🔋 Run AI on battery-powered devices for months
        🎯 Perfect for: IoT, autonomous systems, wearables
        
    4️⃣  Simple Working Demo (simple_demo.py)
        ✅ Basic demo that always works
        🧪 Test GPU capabilities
        🎯 Perfect for: Quick testing and validation
        
            5️⃣  🚀 ULTIMATE AURA Demo (demo_aura_ultimate.py) ⭐ NEW!
        🧠 ALL innovations in one unified system:
        - Liquid Neural Networks 2025 (self-modifying)
        - Spiking GNN (1000x energy efficiency)
        - Quantum-Enhanced TDA (112 algorithms)
        - Shape-Aware Memory V2 (8-tier CXL)
        - Byzantine Consensus (HotStuff)
        🎯 Perfect for: Showcasing EVERYTHING!
        
    6️⃣  🛡️ Agent Failure Prevention (demo_agent_failure_prevention.py) ⭐ CORE VISION!
        🧠 Your core hypothesis in action:
        - Multi-agent system with real-time topology visualization
        - Cascading failure prediction BEFORE it happens
        - Automatic intervention to prevent failures
        - Toggle AURA on/off to see the difference
        🎯 Perfect for: Demonstrating your CORE VALUE PROPOSITION!
    """)

def run_demo(demo_file):
    """Run the selected demo"""
    if not os.path.exists(demo_file):
        print(f"❌ Error: {demo_file} not found!")
        return
    
    print(f"\n🚀 Starting {demo_file}...")
    print("=" * 60)
    
    try:
        # Run the demo
        subprocess.run([sys.executable, demo_file], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Demo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running demo: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def main():
    print_banner()
    
    while True:
        print_demos()
        
        try:
            choice = input("\nEnter your choice (1-6) or 'q' to quit: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Thanks for using AURA Intelligence!")
                break
            
            demos = {
                '1': 'demo_intelligence.py',
                '2': 'demo_pattern_detection.py', 
                '3': 'demo_edge_intelligence.py',
                '4': 'simple_demo.py',
                '5': 'demo_aura_ultimate.py',
                '6': 'demo_agent_failure_prevention.py'
            }
            
            if choice in demos:
                run_demo(demos[choice])
                print("\n" + "=" * 60)
                input("\nPress Enter to continue...")
            else:
                print("\n❌ Invalid choice! Please enter 1-6 or 'q'")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()