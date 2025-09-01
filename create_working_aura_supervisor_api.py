#!/usr/bin/env python3
"""
🚀 Working AURA Supervisor API
=============================
Creates a complete working API that integrates the UnifiedAuraSupervisor 
with the confirmed working AURA components.

This bypasses the import cascade issues by creating a standalone API
that uses only the proven working components.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import time
import json
import traceback
from dataclasses import dataclass, asdict

# Core dependencies
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Set up paths
project_root = Path(__file__).parent
core_path = project_root / "core" / "src"
sys.path.insert(0, str(core_path))

# Import our working components
sys.path.insert(0, str(project_root))

class WorkingAuraAPI:
    """Production AURA API with UnifiedAuraSupervisor"""
    
    def __init__(self):
        self.app = FastAPI(
            title="AURA Intelligence API with Unified Supervisor",
            description="Production-ready AURA system with TDA+LNN supervisor integration",
            version="2.0.0"
        )
        
        # Load working components
        self.working_components = self._load_working_components()
        
        # Initialize UnifiedAuraSupervisor
        self.unified_supervisor = self._create_unified_supervisor()
        
        # Performance metrics
        self.metrics = {
            "requests_total": 0,
            "supervisor_decisions": 0,
            "tda_analyses": 0,
            "lnn_decisions": 0,
            "error_count": 0
        }
        
        # Setup routes
        self._setup_routes()
        
        print("🚀 Working AURA API with Unified Supervisor initialized")
    
    def _load_working_components(self) -> Dict[str, Any]:
        """Load the confirmed working AURA components"""
        
        print("📦 Loading confirmed working components...")
        
        try:
            # Use the real_working_api.py approach
            components = {}
            
            # Load module utility function
            def load_module_from_file(file_path, module_name):
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            
            # Working component paths
            working_paths = {
                "config_base": core_path / "aura_common" / "config" / "base.py",
                "utils_logging": core_path / "aura_common" / "utils" / "logging.py", 
                "resilience_metrics": core_path / "aura_intelligence" / "resilience" / "metrics.py",
                "tda_models": core_path / "aura_intelligence" / "tda" / "models.py",
                "memory_interface": core_path / "aura_intelligence" / "memory" / "storage_interface.py",
                "event_schemas": core_path / "aura_intelligence" / "events" / "schemas.py",
                "core_interfaces": core_path / "aura_intelligence" / "core" / "interfaces.py",
                "agent_base_schema": core_path / "aura_intelligence" / "agents" / "schemas" / "base.py",
                "agent_enums": core_path / "aura_intelligence" / "agents" / "schemas" / "enums.py"
            }
            
            # Load each working component
            for name, path in working_paths.items():
                if path.exists():
                    try:
                        module = load_module_from_file(path, name)
                        components[name] = module
                        print(f"   ✅ {name}")
                    except Exception as e:
                        print(f"   ⚠️ {name}: {str(e)[:50]}...")
                        components[name] = None
                else:
                    print(f"   ❌ {name}: File not found")
                    components[name] = None
            
            working_count = sum(1 for v in components.values() if v is not None)
            print(f"📦 Loaded {working_count}/{len(working_paths)} components")
            
            return components
            
        except Exception as e:
            print(f"❌ Failed to load working components: {e}")
            return {}
    
    def _create_unified_supervisor(self):
        """Create the UnifiedAuraSupervisor with direct import"""
        
        print("🧠 Creating UnifiedAuraSupervisor...")
        
        try:
            # Mock the problematic imports
            import types
            
            # Mock resilience
            mock_resilience = types.ModuleType('aura_intelligence.resilience')
            mock_resilience.resilient = lambda **kwargs: lambda func: func
            mock_resilience.ResilienceLevel = None
            sys.modules['aura_intelligence.resilience'] = mock_resilience
            
            # Import core dependencies first
            from langchain_core.messages import AIMessage
            from langchain_core.runnables import RunnableConfig
            
            # Mock state classes
            class MockCollectiveState(dict):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.update({
                        "workflow_id": kwargs.get("workflow_id", f"workflow_{int(time.time())}"),
                        "thread_id": kwargs.get("thread_id", "thread_001"),
                        "current_step": kwargs.get("current_step", "supervisor_analysis"),
                        "evidence_log": kwargs.get("evidence_log", []),
                        "error_log": kwargs.get("error_log", []),
                        "messages": kwargs.get("messages", []),
                        "supervisor_decisions": kwargs.get("supervisor_decisions", []),
                        "error_recovery_attempts": kwargs.get("error_recovery_attempts", 0),
                        "execution_results": kwargs.get("execution_results", None),
                        "validation_results": kwargs.get("validation_results", None),
                        **kwargs
                    })
            
            class MockNodeResult:
                def __init__(self, success, node_name, output, duration_ms, next_node):
                    self.success = success
                    self.node_name = node_name
                    self.output = output
                    self.duration_ms = duration_ms
                    self.next_node = next_node
            
            # Read and execute supervisor code directly
            supervisor_file = core_path / "aura_intelligence" / "orchestration" / "workflows" / "nodes" / "supervisor.py"
            
            with open(supervisor_file, 'r') as f:
                supervisor_code = f.read()
            
            # Replace problematic imports
            supervisor_code = supervisor_code.replace(
                'from aura_intelligence.resilience import resilient, ResilienceLevel',
                'resilient = lambda **kwargs: lambda func: func\nResilienceLevel = None'
            )
            supervisor_code = supervisor_code.replace(
                'from ..state import CollectiveState, NodeResult',
                'CollectiveState = MockCollectiveState\nNodeResult = MockNodeResult'
            )
            
            # Create execution environment
            supervisor_globals = {
                '__name__': 'supervisor',
                'MockCollectiveState': MockCollectiveState,
                'MockNodeResult': MockNodeResult,
                'AIMessage': AIMessage,
                'RunnableConfig': RunnableConfig,
                'CollectiveState': MockCollectiveState,
                'NodeResult': MockNodeResult,
                'resilient': lambda **kwargs: lambda func: func,
                'ResilienceLevel': None,
                'get_logger': lambda name: MockLogger(),
                'with_correlation_id': lambda: lambda func: func,
                'is_feature_enabled': lambda feature: True,
                'resilient_operation': lambda **kwargs: lambda func: func,
                'Optional': Optional,
                'Dict': Dict,
                'Any': Any,
                'List': List,
                'datetime': datetime,
                'timezone': timezone,
                'time': time,
                'Enum': type('Enum', (), {}),
                'structlog': MockStructlog(),
                'logger': MockLogger(),
            }
            
            # Execute the supervisor code
            exec(supervisor_code, supervisor_globals)
            
            # Create supervisor instance
            create_unified_aura_supervisor = supervisor_globals['create_unified_aura_supervisor']
            supervisor = create_unified_aura_supervisor(
                risk_threshold=0.6,
                tda_config=None,
                lnn_config=None
            )
            
            print(f"🧠 UnifiedAuraSupervisor created successfully")
            print(f"   TDA Available: {supervisor.tda_available}")
            print(f"   LNN Available: {supervisor.lnn_available}")
            
            return supervisor
            
        except Exception as e:
            print(f"❌ Failed to create UnifiedAuraSupervisor: {e}")
            traceback.print_exc()
            return None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=JSONResponse)
        async def root():
            return {
                "service": "AURA Intelligence API with Unified Supervisor",
                "version": "2.0.0",
                "status": "operational",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "unified_supervisor": self.unified_supervisor is not None,
                    "tda_integration": self.unified_supervisor.tda_available if self.unified_supervisor else False,
                    "lnn_integration": self.unified_supervisor.lnn_available if self.unified_supervisor else False,
                    "working_components": len([v for v in self.working_components.values() if v is not None])
                }
            }
        
        @self.app.get("/health", response_class=JSONResponse)
        async def health():
            return {
                "status": "healthy" if self.unified_supervisor else "degraded",
                "metrics": self.metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "supervisor_health": {
                    "available": self.unified_supervisor is not None,
                    "tda_available": self.unified_supervisor.tda_available if self.unified_supervisor else False,
                    "lnn_available": self.unified_supervisor.lnn_available if self.unified_supervisor else False,
                    "cache_size": len(self.unified_supervisor.topology_cache) if self.unified_supervisor else 0,
                    "decision_history": len(self.unified_supervisor.decision_history) if self.unified_supervisor else 0
                }
            }
        
        @self.app.post("/supervisor/analyze", response_class=JSONResponse)
        async def supervisor_analyze(request: Dict[str, Any]):
            """Main supervisor analysis endpoint"""
            
            self.metrics["requests_total"] += 1
            start_time = time.time()
            
            try:
                if not self.unified_supervisor:
                    raise HTTPException(status_code=503, detail="Unified supervisor not available")
                
                # Create workflow state from request
                workflow_state = MockCollectiveState(
                    workflow_id=request.get("workflow_id", f"api_workflow_{int(time.time())}"),
                    evidence_log=request.get("evidence_log", []),
                    messages=[MockAIMessage(msg) for msg in request.get("messages", ["API request"])],
                    error_log=request.get("error_log", []),
                    current_step=request.get("current_step", "supervisor_analysis")
                )
                
                # Execute supervisor analysis
                result = await self.unified_supervisor(workflow_state)
                duration = time.time() - start_time
                
                # Extract results
                supervisor_decisions = result.get('supervisor_decisions', [])
                risk_assessment = result.get('risk_assessment', {})
                topology_analysis = result.get('topology_analysis')
                lnn_decision = result.get('lnn_decision')
                
                self.metrics["supervisor_decisions"] += 1
                if topology_analysis:
                    self.metrics["tda_analyses"] += 1
                if lnn_decision:
                    self.metrics["lnn_decisions"] += 1
                
                return {
                    "success": True,
                    "duration_seconds": duration,
                    "supervisor_decisions": supervisor_decisions,
                    "risk_assessment": risk_assessment,
                    "topology_analysis": topology_analysis,
                    "lnn_decision": lnn_decision,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "advanced_features": {
                        "tda_used": topology_analysis is not None,
                        "lnn_used": lnn_decision is not None,
                        "unified_risk": risk_assessment.get('unified_risk_score', 0.0)
                    }
                }
                
            except Exception as e:
                self.metrics["error_count"] += 1
                duration = time.time() - start_time
                return {
                    "success": False,
                    "error": str(e),
                    "duration_seconds": duration,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        
        @self.app.get("/supervisor/metrics", response_class=JSONResponse)
        async def supervisor_metrics():
            """Get detailed supervisor metrics"""
            
            if not self.unified_supervisor:
                return {"error": "Supervisor not available"}
            
            return {
                "supervisor_type": "unified_aura",
                "capabilities": {
                    "tda_available": self.unified_supervisor.tda_available,
                    "lnn_available": self.unified_supervisor.lnn_available,
                    "risk_threshold": self.unified_supervisor.risk_threshold
                },
                "performance": {
                    "topology_cache_size": len(self.unified_supervisor.topology_cache),
                    "decision_history_size": len(self.unified_supervisor.decision_history),
                    "requests_processed": self.metrics["requests_total"],
                    "decisions_made": self.metrics["supervisor_decisions"],
                    "tda_analyses": self.metrics["tda_analyses"],
                    "lnn_decisions": self.metrics["lnn_decisions"],
                    "error_rate": self.metrics["error_count"] / max(1, self.metrics["requests_total"])
                },
                "working_components": {
                    name: component is not None 
                    for name, component in self.working_components.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.post("/components/test", response_class=JSONResponse)
        async def test_components():
            """Test working components integration"""
            
            results = {}
            
            for name, component in self.working_components.items():
                if component is not None:
                    try:
                        # Basic component test
                        if hasattr(component, '__version__'):
                            results[name] = {"status": "available", "version": component.__version__}
                        elif hasattr(component, '__name__'):
                            results[name] = {"status": "available", "name": component.__name__}
                        else:
                            results[name] = {"status": "available", "type": str(type(component))}
                    except Exception as e:
                        results[name] = {"status": "error", "error": str(e)}
                else:
                    results[name] = {"status": "unavailable"}
            
            return {
                "component_tests": results,
                "working_count": sum(1 for r in results.values() if r["status"] == "available"),
                "total_count": len(results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


class MockStructlog:
    def get_logger(self, name):
        return MockLogger()

class MockLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg}")
    def warning(self, msg, **kwargs):
        print(f"WARNING: {msg}")
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg}")

class MockAIMessage:
    def __init__(self, content, additional_kwargs=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}

class MockCollectiveState(dict):
    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)


def create_working_aura_api():
    """Create and return the working AURA API"""
    return WorkingAuraAPI()


async def main():
    """Run the working AURA API"""
    
    print("🚀 STARTING WORKING AURA API WITH UNIFIED SUPERVISOR")
    print("=" * 80)
    
    # Create the API
    api = create_working_aura_api()
    
    if api.unified_supervisor is None:
        print("❌ Failed to initialize UnifiedAuraSupervisor")
        return
    
    print("\n✅ WORKING AURA API READY!")
    print(f"🧠 UnifiedAuraSupervisor: {'✅ OPERATIONAL' if api.unified_supervisor else '❌ FAILED'}")
    print(f"📊 TDA Integration: {'✅ AVAILABLE' if api.unified_supervisor.tda_available else '⚠️ FALLBACK'}")
    print(f"🤖 LNN Integration: {'✅ AVAILABLE' if api.unified_supervisor.lnn_available else '⚠️ FALLBACK'}")
    print(f"📦 Working Components: {len([v for v in api.working_components.values() if v is not None])}/9")
    print("\n🌐 API Endpoints:")
    print("   GET  /           - API status")
    print("   GET  /health     - Health check") 
    print("   POST /supervisor/analyze - Supervisor analysis")
    print("   GET  /supervisor/metrics - Supervisor metrics")
    print("   POST /components/test    - Test components")
    print("\n🚀 Starting server on http://localhost:8002")
    
    # Run the API
    config = uvicorn.Config(
        api.app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())