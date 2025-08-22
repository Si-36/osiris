#!/usr/bin/env python3
"""Simple test to verify monitoring module imports"""

try:
    print("Testing business metrics import...")
    from core.src.aura_intelligence.monitoring.business_metrics import BusinessMetricsCollector
    print("✅ Business metrics imported successfully")
    
    print("Testing real-time dashboard import...")
    from core.src.aura_intelligence.monitoring.real_time_dashboard import RealTimeDashboard
    print("✅ Real-time dashboard imported successfully")
    
    print("Testing production monitor import...")
    from core.src.aura_intelligence.monitoring.production_monitor import ProductionMonitor
    print("✅ Production monitor imported successfully")
    
    print("🎉 All monitoring modules imported successfully!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()