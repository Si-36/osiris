#!/usr/bin/env python3
"""
Working AURA Core API - Testing what actually works
Uses only components that can successfully import
"""

import os
import sys
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging

# Add core to path
sys.path.insert(0, '/home/sina/projects/osiris-2/core/src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    name: str
    status: str  # SUCCESS, FAILED, PARTIAL
    error: Optional[str] = None
    import_time: Optional[float] = None

class WorkingComponentTester:
    """Test what AURA components actually import and work"""
    
    def __init__(self):
        self.results = []
        self.working_components = []
        
    def test_component(self, name: str, import_statement: str) -> ComponentStatus:
        """Test if a component can be imported"""
        start_time = time.time()
        try:
            exec(import_statement)
            import_time = time.time() - start_time
            status = ComponentStatus(name, "SUCCESS", None, import_time)
            self.working_components.append(name)
            logger.info(f"✅ {name}: SUCCESS ({import_time:.3f}s)")
            return status
        except Exception as e:
            import_time = time.time() - start_time
            error_msg = str(e)[:200]
            status = ComponentStatus(name, "FAILED", error_msg, import_time)
            logger.error(f"❌ {name}: {error_msg}")
            return status
    
    def test_all_components(self) -> Dict[str, Any]:
        """Test all key AURA components"""
        
        test_cases = [
            # Core Infrastructure
            ("FastAPI", "from fastapi import FastAPI"),
            ("Pydantic", "from pydantic import BaseModel"),
            ("Asyncio", "import asyncio"),
            
            # Basic Python Components (these should work)
            ("Redis Basic", "import redis"),
            ("JSON Logging", "import json, logging"),
            ("Time Utils", "import time, datetime"),
            
            # Try individual AURA files (bypassing the main __init__.py)
            ("AURA Config Base", "from aura_intelligence.config.base import BaseConfig"),
            ("AURA Memory Config", "from aura_intelligence.memory.config import MemoryConfig"),
            ("AURA Utils Decorators", "from aura_intelligence.utils.decorators import async_retry"),
            
            # Test some working files we identified
            ("AURA Resilience Types", "from aura_intelligence.resilience.metrics import ResilienceMetrics"),
            
        ]
        
        logger.info("🧪 Starting component import tests...")
        
        for name, import_stmt in test_cases:
            result = self.test_component(name, import_stmt)
            self.results.append(result)
            
        # Summary
        working_count = len([r for r in self.results if r.status == "SUCCESS"])
        failed_count = len([r for r in self.results if r.status == "FAILED"])
        
        summary = {
            "total_tested": len(self.results),
            "working": working_count,
            "failed": failed_count,
            "success_rate": working_count / len(self.results) if self.results else 0,
            "working_components": self.working_components,
            "detailed_results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "error": r.error,
                    "import_time": r.import_time
                }
                for r in self.results
            ]
        }
        
        logger.info(f"📊 Test Results: {working_count}/{len(self.results)} components working ({100*summary['success_rate']:.1f}%)")
        
        return summary

# Simple FastAPI server using only working components
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI(
        title="AURA Working Core API",
        description="API using only components that actually import successfully",
        version="0.1.0"
    )
    
    tester = WorkingComponentTester()
    
    @app.get("/")
    async def root():
        return {"message": "AURA Working Core API", "status": "operational", "timestamp": time.time()}
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "api": "working-core"
        }
    
    @app.get("/test-components")
    async def test_components():
        """Test what AURA components actually work"""
        try:
            results = tester.test_all_components()
            return results
        except Exception as e:
            logger.error(f"Component testing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Testing failed: {str(e)}")
    
    @app.get("/working-components")
    async def get_working_components():
        """Get list of components that successfully imported"""
        return {
            "working_components": tester.working_components,
            "count": len(tester.working_components),
            "timestamp": time.time()
        }
    
    if __name__ == "__main__":
        logger.info("🚀 Starting AURA Working Core API...")
        logger.info("📍 Available endpoints:")
        logger.info("   GET /              - Root endpoint")
        logger.info("   GET /health        - Health check")
        logger.info("   GET /test-components - Test AURA component imports")
        logger.info("   GET /working-components - Get working components list")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)

except ImportError as e:
    logger.error(f"❌ Cannot start FastAPI server: {e}")
    logger.info("📝 Running component tests without web server...")
    
    # Run tests without FastAPI
    tester = WorkingComponentTester()
    results = tester.test_all_components()
    
    # Save results to file
    with open('/home/sina/projects/osiris-2/component_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("💾 Results saved to component_test_results.json")
    print(json.dumps(results, indent=2))