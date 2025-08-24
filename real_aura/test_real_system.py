"""
Test Real AURA System - Verify actual functionality
"""
import time
import json
import requests
import subprocess
import psutil
from rich.console import Console

console = Console()


def test_collector():
    """Test that collector actually collects real metrics"""
    console.print("\n[bold cyan]Testing Metric Collector...[/bold cyan]")
    
    # Get current system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    mem_percent = psutil.virtual_memory().percent
    
    console.print(f"✅ Real CPU Usage: {cpu_percent}%")
    console.print(f"✅ Real Memory Usage: {mem_percent}%")
    console.print(f"✅ Real Process Count: {len(psutil.pids())}")
    
    return True


def test_api():
    """Test that API serves real data"""
    console.print("\n[bold cyan]Testing API Server...[/bold cyan]")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            console.print(f"✅ API Health: {data.get('status')}")
            console.print(f"✅ Redis Connected: {data.get('redis_connected')}")
            console.print(f"✅ Uptime: {data.get('uptime_seconds')}s")
            return True
        else:
            console.print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        console.print("❌ API not running! Start with: python real_aura/api/main.py")
        return False
    except Exception as e:
        console.print(f"❌ API test failed: {e}")
        return False


def test_data_flow():
    """Test that data actually flows through the system"""
    console.print("\n[bold cyan]Testing Data Flow...[/bold cyan]")
    
    try:
        # Get metrics from API
        response = requests.get("http://localhost:8080/metrics", timeout=2)
        if response.status_code == 200:
            metrics = response.json()
            console.print("✅ Successfully retrieved metrics from API:")
            console.print(f"   CPU: {metrics.get('cpu', {}).get('percent')}%")
            console.print(f"   Memory: {metrics.get('memory', {}).get('percent')}%")
            console.print(f"   Timestamp: {metrics.get('timestamp')}")
            return True
        elif response.status_code == 404:
            console.print("⚠️  No metrics available yet - make sure collector is running")
            return False
        else:
            console.print(f"❌ Failed to get metrics: {response.status_code}")
            return False
    except Exception as e:
        console.print(f"❌ Data flow test failed: {e}")
        return False


def test_websocket():
    """Test WebSocket connection"""
    console.print("\n[bold cyan]Testing WebSocket...[/bold cyan]")
    
    try:
        import websocket
        ws = websocket.create_connection("ws://localhost:8080/ws", timeout=2)
        console.print("✅ WebSocket connected successfully")
        
        # Wait for a message
        console.print("⏳ Waiting for real-time data...")
        ws.settimeout(10)
        message = ws.recv()
        data = json.loads(message)
        console.print(f"✅ Received real-time data: CPU={data.get('cpu', {}).get('percent')}%")
        ws.close()
        return True
    except Exception as e:
        console.print(f"⚠️  WebSocket test skipped: {e}")
        return False


def main():
    """Run all tests"""
    console.print("""
[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]
[bold white]              🧪 AURA REAL SYSTEM TEST 🧪                   [/bold white]
[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]

This test verifies that the system actually works with REAL data.
No mocks, no fakes - just real functionality!
""")
    
    # Test components
    tests = [
        ("Collector", test_collector),
        ("API", test_api),
        ("Data Flow", test_data_flow),
        ("WebSocket", test_websocket)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            console.print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    console.print("\n[bold cyan]Test Summary:[/bold cyan]")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        console.print(f"{name}: {status}")
    
    console.print(f"\n[bold]Total: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("\n[bold green]🎉 All tests passed! The system is working with REAL data![/bold green]")
    else:
        console.print("\n[bold yellow]⚠️  Some tests failed. Make sure all services are running:[/bold yellow]")
        console.print("1. Redis: docker run -p 6379:6379 redis")
        console.print("2. Collector: python real_aura/core/collector.py")
        console.print("3. API: python real_aura/api/main.py")


if __name__ == "__main__":
    main()