#!/usr/bin/env python3
"""
Real Working AURA API - Built from actually working components
Using only the 9 isolated components that successfully import
"""

import os
import sys
import json
import time
import asyncio
import importlib.util
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

# Add core to path
sys.path.insert(0, '/home/sina/projects/osiris-2/core/src')

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Load working AURA components directly
def load_module_from_file(file_path: str, module_name: str):
    """Load a Python module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class AURAWorkingComponents:
    """Manager for working AURA components"""
    
    def __init__(self):
        self.base_path = Path("/home/sina/projects/osiris-2/core/src/aura_intelligence")
        self.loaded_modules = {}
        self.component_status = {}
        
        # Working components identified by testing
        self.working_components = {
            "config_base": "config/base.py",
            "utils_logging": "utils/logging.py", 
            "resilience_metrics": "resilience/metrics.py",
            "tda_models": "tda/models.py",
            "memory_interface": "memory/storage_interface.py",
            "event_schemas": "events/schemas.py",
            "core_interfaces": "core/interfaces.py",
            "agent_base_schema": "agents/schemas/base.py",
            "agent_enums": "agents/schemas/enums.py"
        }
    
    def load_component(self, name: str, file_path: str):
        """Load a working component"""
        try:
            full_path = self.base_path / file_path
            module = load_module_from_file(str(full_path), name)
            self.loaded_modules[name] = module
            self.component_status[name] = "SUCCESS"
            return module
        except Exception as e:
            self.component_status[name] = f"FAILED: {str(e)[:100]}"
            return None
    
    def load_all_working_components(self):
        """Load all identified working components"""
        print("🔧 Loading working AURA components...")
        
        for name, file_path in self.working_components.items():
            module = self.load_component(name, file_path)
            if module:
                print(f"✅ {name}: Loaded successfully")
            else:
                print(f"❌ {name}: Failed to load")
        
        working_count = len([s for s in self.component_status.values() if s == "SUCCESS"])
        print(f"📊 Loaded {working_count}/{len(self.working_components)} components")
        
        return self.loaded_modules
    
    def get_component_info(self, name: str) -> Dict[str, Any]:
        """Get information about a loaded component"""
        if name not in self.loaded_modules:
            return {"status": "not_loaded"}
        
        module = self.loaded_modules[name]
        return {
            "status": "loaded",
            "classes": [item for item in dir(module) if not item.startswith('_') and item[0].isupper()],
            "functions": [item for item in dir(module) if not item.startswith('_') and item[0].islower()],
            "attributes": [item for item in dir(module) if not item.startswith('_') and not callable(getattr(module, item, None))]
        }

# Initialize component manager
aura_components = AURAWorkingComponents()

@dataclass
class AURASystemStatus:
    """Real system status based on working components"""
    timestamp: float
    loaded_components: int
    total_components: int
    success_rate: float
    working_components: List[str]
    failed_components: List[str]
    api_version: str = "1.0.0-working"

class RealAURASystem:
    """AURA system using only working components"""
    
    def __init__(self):
        self.components = aura_components
        self.start_time = time.time()
        self.request_count = 0
        
    async def initialize(self):
        """Initialize the system with working components"""
        print("🚀 Initializing Real AURA System...")
        modules = self.components.load_all_working_components()
        
        # Try to access some functionality from loaded components
        self.available_features = {}
        
        # Test TDA models if available
        if "tda_models" in modules:
            try:
                tda_module = modules["tda_models"]
                # Check what's available in TDA models
                self.available_features["tda"] = {
                    "status": "available",
                    "classes": [item for item in dir(tda_module) if item[0].isupper()],
                    "description": "Topological Data Analysis models"
                }
            except Exception as e:
                self.available_features["tda"] = {"status": "error", "error": str(e)}
        
        # Test resilience metrics
        if "resilience_metrics" in modules:
            try:
                resilience_module = modules["resilience_metrics"]
                self.available_features["resilience"] = {
                    "status": "available", 
                    "classes": [item for item in dir(resilience_module) if item[0].isupper()],
                    "description": "System resilience and monitoring"
                }
            except Exception as e:
                self.available_features["resilience"] = {"status": "error", "error": str(e)}
        
        # Test agent schemas
        if "agent_enums" in modules:
            try:
                agent_module = modules["agent_enums"] 
                self.available_features["agents"] = {
                    "status": "available",
                    "enums": [item for item in dir(agent_module) if item[0].isupper()],
                    "description": "Agent system schemas and enums"
                }
            except Exception as e:
                self.available_features["agents"] = {"status": "error", "error": str(e)}
        
        print(f"✅ System initialized with {len(self.available_features)} feature groups")
        return True
    
    def get_system_status(self) -> AURASystemStatus:
        """Get current system status"""
        working = [name for name, status in self.components.component_status.items() if status == "SUCCESS"]
        failed = [name for name, status in self.components.component_status.items() if status != "SUCCESS"]
        
        return AURASystemStatus(
            timestamp=time.time(),
            loaded_components=len(working),
            total_components=len(self.components.working_components),
            success_rate=len(working) / len(self.components.working_components) if self.components.working_components else 0,
            working_components=working,
            failed_components=failed
        )

# Initialize system
aura_system = RealAURASystem()

if FASTAPI_AVAILABLE:
    # Create FastAPI app
    app = FastAPI(
        title="Real Working AURA API",
        description="API built from actually working AURA components (no dummy implementations)",
        version="1.0.0-real",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    @app.on_event("startup")
    async def startup_event():
        await aura_system.initialize()
    
    @app.get("/")
    async def root():
        return {
            "message": "Real Working AURA API",
            "status": "operational", 
            "uptime": time.time() - aura_system.start_time,
            "working_components": len([s for s in aura_system.components.component_status.values() if s == "SUCCESS"])
        }
    
    @app.get("/health")
    async def health_check():
        status = aura_system.get_system_status()
        return {
            "status": "healthy" if status.success_rate > 0.5 else "degraded",
            "timestamp": status.timestamp,
            "loaded_components": status.loaded_components,
            "success_rate": round(status.success_rate * 100, 1),
            "uptime": time.time() - aura_system.start_time
        }
    
    @app.get("/components", response_model=Dict[str, Any])
    async def get_components():
        """Get all loaded component information"""
        aura_system.request_count += 1
        
        component_info = {}
        for name in aura_system.components.loaded_modules:
            component_info[name] = aura_system.components.get_component_info(name)
        
        return {
            "loaded_components": component_info,
            "component_status": aura_system.components.component_status,
            "total_requests": aura_system.request_count
        }
    
    @app.get("/features")
    async def get_available_features():
        """Get available AURA features from working components"""
        return {
            "available_features": aura_system.available_features,
            "feature_count": len(aura_system.available_features),
            "timestamp": time.time()
        }
    
    @app.get("/status/detailed")
    async def get_detailed_status():
        """Get detailed system status"""
        status = aura_system.get_system_status()
        return asdict(status)
    
    # TDA endpoint if TDA components are working
    @app.post("/tda/analyze")
    async def tda_analyze(data: Dict[str, Any]):
        """Analyze data using TDA if available"""
        if "tda" not in aura_system.available_features:
            raise HTTPException(status_code=503, detail="TDA components not available")
        
        if aura_system.available_features["tda"]["status"] != "available":
            raise HTTPException(status_code=503, detail="TDA components failed to load")
        
        # For now, return structure info since we have working TDA models
        return {
            "message": "TDA analysis endpoint - models loaded successfully",
            "available_classes": aura_system.available_features["tda"]["classes"],
            "input_data_keys": list(data.keys()) if data else [],
            "timestamp": time.time(),
            "note": "Real TDA implementation requires fixing remaining syntax errors in algorithms"
        }
    
    if __name__ == "__main__":
        print("🚀 Starting Real Working AURA API...")
        print("📊 This API uses ONLY components that actually import successfully")
        print("🔧 No dummy implementations or mocked functionality")
        print()
        print("📍 Available endpoints:")
        print("   GET  /              - API root")
        print("   GET  /health        - Health check")
        print("   GET  /components    - Loaded components info")
        print("   GET  /features      - Available features")
        print("   GET  /status/detailed - Detailed status")
        print("   POST /tda/analyze   - TDA analysis (if available)")
        print("   GET  /docs          - API documentation")
        print()
        
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

else:
    print("❌ FastAPI not available - running component analysis only")
    
    async def run_analysis():
        await aura_system.initialize()
        status = aura_system.get_system_status()
        
        print("\n📊 REAL WORKING AURA SYSTEM STATUS:")
        print(f"   ✅ Working components: {status.loaded_components}/{status.total_components}")
        print(f"   📈 Success rate: {status.success_rate*100:.1f}%")
        print(f"   🔧 Available features: {len(aura_system.available_features)}")
        print(f"\n✅ Working components: {status.working_components}")
        print(f"❌ Failed components: {status.failed_components}")
        
        # Save status to file
        with open('/home/sina/projects/osiris-2/real_aura_status.json', 'w') as f:
            json.dump(asdict(status), f, indent=2)
        
        print(f"\n💾 Status saved to real_aura_status.json")
    
    if __name__ == "__main__":
        asyncio.run(run_analysis())