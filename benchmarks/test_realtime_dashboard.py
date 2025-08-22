#!/usr/bin/env python3
"""
📊 REAL-TIME DASHBOARD TEST
Test the comprehensive real-time performance monitoring system
"""

import asyncio
import time
import json
from typing import Dict, Any
from pathlib import Path

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

from aura_intelligence.monitoring.realtime_dashboard import get_global_dashboard
from aura_intelligence.components.async_batch_processor import get_global_batch_processor, BatchType

class DashboardTester:
    def __init__(self):
        self.dashboard = None
        self.batch_processor = None
        
    async def initialize(self):
        """Initialize dashboard and dependencies"""
        print("🚀 Initializing Real-Time Dashboard...")
        try:
            self.dashboard = await get_global_dashboard()
            self.batch_processor = await get_global_batch_processor()
            print("✅ Dashboard initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize dashboard: {e}")
            return False
    
    async def create_system_load(self, duration: int = 30):
        """Create system load to test monitoring"""
        print(f"\n🏋️  Creating system load for {duration} seconds...")
        
        # Create various types of load
        tasks = []
        
        # CPU-intensive batch operations
        for i in range(50):
            task = self.batch_processor.add_operation(
                operation_id=f"cpu_load_{i}",
                batch_type=BatchType.NEURAL_FORWARD,
                input_data={"values": list(range(100))},  # Larger data
                component_id="load_test_neural",
                priority=0
            )
            tasks.append(task)
        
        # Memory-intensive operations
        for i in range(30):
            task = self.batch_processor.add_operation(
                operation_id=f"memory_load_{i}",
                batch_type=BatchType.BERT_ATTENTION,
                input_data={"text": "This is a long text for memory-intensive processing " * 20},
                component_id="load_test_bert",
                priority=1
            )
            tasks.append(task)
        
        # GPU-intensive operations
        for i in range(20):
            task = self.batch_processor.add_operation(
                operation_id=f"gpu_load_{i}",
                batch_type=BatchType.NEURAL_ODE,
                input_data={"values": [j * 0.01 for j in range(50)]},
                component_id="load_test_ode",
                priority=2
            )
            tasks.append(task)
        
        start_time = time.time()
        
        # Process load in waves
        while time.time() - start_time < duration:
            # Process current batch
            try:
                batch_results = await asyncio.gather(*tasks[:20], return_exceptions=True)
                tasks = tasks[20:]
                
                # Add more tasks if needed
                if len(tasks) < 10:
                    for i in range(10):
                        task = self.batch_processor.add_operation(
                            operation_id=f"continuous_load_{int(time.time())}_{i}",
                            batch_type=BatchType.GENERAL_COMPUTE,
                            input_data={"load": "continuous"},
                            component_id="continuous_load",
                            priority=0
                        )
                        tasks.append(task)
                
                print(f"   🔄 Load running... {time.time() - start_time:.1f}s elapsed")
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"   ⚠️  Load generation error: {e}")
                await asyncio.sleep(1)
        
        # Clean up remaining tasks
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass
                
        print(f"✅ System load completed after {duration} seconds")
    
    async def monitor_and_display(self, duration: int = 60):
        """Monitor system and display metrics"""
        print(f"\n📊 Monitoring system for {duration} seconds...")
        
        start_time = time.time()
        display_interval = 5  # Display every 5 seconds
        last_display = 0
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            if current_time - last_display >= display_interval:
                # Get current status
                status = self.dashboard.get_current_status()
                
                print(f"\n⏰ Time: {current_time:.1f}s")
                print(f"🏥 Health Score: {status.get('health_score', 0):.1f}/100")
                
                # System metrics
                system = status.get('system', {})
                print(f"💻 CPU: {system.get('cpu_percent', 0):.1f}% | Memory: {system.get('memory_percent', 0):.1f}% ({system.get('memory_used_gb', 0):.1f}GB)")
                
                # GPU metrics
                gpu = status.get('gpu', {})
                if gpu.get('available'):
                    gpu_memory_percent = 0
                    if gpu.get('memory_total_mb', 0) > 0:
                        gpu_memory_percent = (gpu.get('memory_used_mb', 0) / gpu.get('memory_total_mb', 1)) * 100
                    print(f"🎮 GPU: {gpu.get('utilization_percent', 0):.1f}% util | {gpu_memory_percent:.1f}% memory ({gpu.get('memory_used_mb', 0):.0f}MB)")
                else:
                    print("🎮 GPU: Not available")
                
                # Component metrics
                components = status.get('components', {})
                print(f"🔧 Components: {components.get('healthy', 0)}/{components.get('total', 0)} healthy | Avg response: {components.get('avg_response_time_ms', 0):.2f}ms")
                
                # Batch processing metrics
                batch = status.get('batch_processing', {})
                print(f"⚡ Batch: {batch.get('operations_per_second', 0):.0f} ops/sec | {batch.get('efficiency_percent', 0):.1f}% efficiency | {batch.get('pending_operations', 0)} pending")
                
                # Redis metrics
                redis = status.get('redis', {})
                redis_status = "✅" if redis.get('healthy') else "❌"
                print(f"📦 Redis: {redis_status} | {redis.get('operations_per_second', 0):.0f} ops/sec | {redis.get('cache_hit_rate', 0):.1f}% hit rate")
                
                # Recent alerts
                alerts = status.get('recent_alerts', [])
                if alerts:
                    print(f"🚨 Recent Alerts: {len(alerts)}")
                    for alert in alerts[-3:]:  # Show last 3 alerts
                        print(f"   [{alert.get('level', 'INFO')}] {alert.get('message', 'No message')}")
                
                last_display = current_time
            
            await asyncio.sleep(1)
    
    async def test_performance_trends(self):
        """Test performance trend analysis"""
        print(f"\n📈 Testing performance trend analysis...")
        
        # Wait for some data to accumulate
        await asyncio.sleep(10)
        
        trends = self.dashboard.get_performance_trends()
        
        print(f"📊 Trend Analysis Results:")
        print(f"   Analysis period: {trends.get('analysis_period_seconds', 0)} seconds")
        
        trend_data = trends.get('trends', {})
        for metric, trend in trend_data.items():
            direction = trend.get('direction', 'unknown')
            change = trend.get('change_percent', 0)
            print(f"   {metric}: {direction} ({change:+.1f}%)")
        
        averages = trends.get('averages', {})
        print(f"📈 Averages:")
        for metric, avg in averages.items():
            print(f"   {metric}: {avg:.2f}")
        
        alert_count = trends.get('alert_count_last_hour', 0)
        print(f"🚨 Alerts in last hour: {alert_count}")
        
        return trends
    
    async def test_metric_export(self):
        """Test metric export functionality"""
        print(f"\n💾 Testing metric export...")
        
        export_file = Path("dashboard_metrics_export.json")
        self.dashboard.export_metrics(str(export_file))
        
        # Verify export
        if export_file.exists():
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Export successful: {export_file}")
            print(f"   Performance history entries: {len(data.get('performance_history', []))}")
            print(f"   Alerts: {len(data.get('alerts', []))}")
            print(f"   Export size: {export_file.stat().st_size / 1024:.1f} KB")
            
            return str(export_file)
        else:
            print(f"❌ Export failed")
            return None
    
    async def run_comprehensive_test(self):
        """Run comprehensive dashboard test"""
        print("🔬 Real-Time Dashboard Comprehensive Test")
        print("=" * 50)
        
        if not await self.initialize():
            return {'error': 'Failed to initialize dashboard'}
        
        test_results = {}
        
        try:
            # Start monitoring in background
            monitor_task = asyncio.create_task(self.monitor_and_display(90))
            
            # Wait a bit for baseline
            await asyncio.sleep(5)
            test_results['baseline_status'] = self.dashboard.get_current_status()
            
            # Create system load
            load_task = asyncio.create_task(self.create_system_load(30))
            
            # Wait for load to complete
            await load_task
            
            # Get status under load
            test_results['load_status'] = self.dashboard.get_current_status()
            
            # Test trend analysis
            test_results['trends'] = await self.test_performance_trends()
            
            # Test export
            export_file = await self.test_metric_export()
            test_results['export_file'] = export_file
            
            # Wait for monitoring to complete
            await monitor_task
            
            # Final status
            test_results['final_status'] = self.dashboard.get_current_status()
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    async def close(self):
        """Clean up dashboard"""
        if self.dashboard:
            await self.dashboard.stop_monitoring()

async def main():
    """Run dashboard tests"""
    tester = DashboardTester()
    
    try:
        results = await tester.run_comprehensive_test()
        
        # Print summary
        print("\n" + "=" * 50)
        print("🏆 REAL-TIME DASHBOARD TEST SUMMARY")
        print("=" * 50)
        
        if 'error' not in results:
            # Health score comparison
            baseline_health = results.get('baseline_status', {}).get('health_score', 0)
            final_health = results.get('final_status', {}).get('health_score', 0)
            
            print(f"🏥 Health Score: {baseline_health:.1f} → {final_health:.1f} ({final_health - baseline_health:+.1f})")
            
            # Performance metrics
            if 'trends' in results:
                trends = results['trends']
                print(f"📊 Trend Analysis Period: {trends.get('analysis_period_seconds', 0)}s")
                
                avg_batch_ops = trends.get('averages', {}).get('batch_ops_per_second', 0)
                print(f"⚡ Average Batch Performance: {avg_batch_ops:.0f} ops/sec")
            
            # Export verification
            if 'export_file' in results and results['export_file']:
                print(f"💾 Metrics Export: ✅ {results['export_file']}")
            
            # Final system status
            final_status = results.get('final_status', {})
            components = final_status.get('components', {})
            batch = final_status.get('batch_processing', {})
            
            print(f"🔧 Final Component Health: {components.get('healthy', 0)}/{components.get('total', 0)}")
            print(f"⚡ Final Batch Efficiency: {batch.get('efficiency_percent', 0):.1f}%")
            
            # System verdict
            if final_health > 90:
                print("\n🎉 EXCELLENT - Dashboard monitoring system working perfectly!")
            elif final_health > 75:
                print("\n✅ GOOD - Dashboard providing effective monitoring!")
            elif final_health > 60:
                print("\n⚠️  FAIR - Dashboard functional with some issues")
            else:
                print("\n❌ POOR - Dashboard needs attention")
        
        # Save comprehensive results
        results_file = Path("dashboard_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Comprehensive test results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())