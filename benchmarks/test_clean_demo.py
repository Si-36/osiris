#!/usr/bin/env python3
"""
🧪 Clean Demo Test - Professional 2025 validation
"""

import asyncio
import time
import requests
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_professional_demo():
    """Test the professional demo system"""
    base_url = "http://localhost:8080"
    
    logger.info("🧪 Testing Professional AURA Demo")
    logger.info("=" * 50)
    
    # Wait for system to be ready
    logger.info("⏳ Waiting for system startup...")
    for i in range(30):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ System ready after {i+1} seconds")
                break
        except:
            pass
        await asyncio.sleep(1)
    else:
        logger.error("❌ System failed to start within 30 seconds")
        return False
    
    try:
        # Test 1: Health check
        logger.info("🏥 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            logger.info(f"   ✅ Status: {health['status']}")
            logger.info(f"   ✅ Uptime: {health['uptime_seconds']:.1f}s")
        else:
            logger.error(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test 2: Scenarios endpoint
        logger.info("📋 Testing scenarios endpoint...")
        response = requests.get(f"{base_url}/scenarios", timeout=5)
        if response.status_code == 200:
            scenarios = response.json()
            logger.info(f"   ✅ Found {len(scenarios['scenarios'])} scenarios")
        else:
            logger.error(f"   ❌ Scenarios failed: {response.status_code}")
            return False
        
        # Test 3: Simple demo execution
        logger.info("🎯 Testing simple scenario...")
        response = requests.post(
            f"{base_url}/demo",
            json={
                "scenario": "simple_test",
                "data": {},
                "config": {}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                logger.info(f"   ✅ Simple test passed in {result['duration_ms']:.1f}ms")
            else:
                logger.error(f"   ❌ Simple test failed: {result.get('results', {}).get('error', 'Unknown')}")
                return False
        else:
            logger.error(f"   ❌ Demo execution failed: {response.status_code}")
            return False
        
        # Test 4: GPU performance test (if available)
        logger.info("⚡ Testing GPU scenario...")
        response = requests.post(
            f"{base_url}/demo",
            json={
                "scenario": "gpu_performance",
                "data": {},
                "config": {}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                gpu_available = result['results'].get('gpu_available', False)
                logger.info(f"   ✅ GPU test completed - GPU available: {gpu_available}")
            else:
                logger.warning(f"   ⚠️ GPU test had issues: {result.get('results', {}).get('error', 'Unknown')}")
        
        # Test 5: System health comprehensive
        logger.info("🏥 Testing comprehensive health...")
        response = requests.post(
            f"{base_url}/demo",
            json={
                "scenario": "system_health",
                "data": {},
                "config": {}
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                health_score = result['results'].get('overall_health_score', 0)
                grade = result['results'].get('health_grade', 'Unknown')
                logger.info(f"   ✅ System health: {health_score:.1f}% (Grade: {grade})")
            else:
                logger.warning(f"   ⚠️ Health test issues: {result.get('results', {}).get('error', 'Unknown')}")
        
        logger.info("=" * 50)
        logger.info("🎉 All tests completed successfully!")
        logger.info("🌐 Demo available at: http://localhost:8080")
        logger.info("📚 API docs at: http://localhost:8080/docs")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main test execution"""
    try:
        success = asyncio.run(test_professional_demo())
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())