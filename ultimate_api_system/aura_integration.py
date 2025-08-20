"""
AURA Intelligence Integration with Ultimate API System
Connects core/src/aura_intelligence to ultimate_api_system
"""

import sys
from pathlib import Path

# Add aura_intelligence to path
aura_path = Path(__file__).parent.parent / "core" / "src"
sys.path.insert(0, str(aura_path))

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import your existing AURA components
from aura_intelligence.components.real_registry import get_real_registry, ComponentType
from aura_intelligence.consciousness.global_workspace import get_global_workspace
from aura_intelligence.coral.best_coral import get_best_coral
from aura_intelligence.dpo.preference_optimizer import get_dpo_optimizer
from aura_intelligence.bio_homeostatic.metabolic_manager import MetabolicManager
from aura_intelligence.streaming.kafka_integration import get_event_streaming, EventType
from aura_intelligence.graph.neo4j_integration import get_neo4j_integration
from aura_intelligence.lnn.core import LNNCore

# Request/Response models
class AURAProcessRequest(BaseModel):
    component_type: str
    data: Dict[str, Any]
    component_id: Optional[str] = None

class AURASystemRequest(BaseModel):
    data: Dict[str, Any]
    systems: List[str] = ["registry", "consciousness", "coral", "dpo"]

# Create router
aura_router = APIRouter(prefix="/aura", tags=["AURA Intelligence"])

# Initialize AURA systems
class AURAIntegration:
    def __init__(self):
        self.registry = None
        self.consciousness = None
        self.coral = None
        self.dpo = None
        self.metabolic = None
        self.streaming = None
        self.neo4j = None
        self.lnn = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all AURA systems"""
        if self.initialized:
            return
        
        try:
            # Core registry with 209 components
            self.registry = get_real_registry()
            
            # Consciousness system
            self.consciousness = get_global_workspace()
            await self.consciousness.start()
            
            # CoRaL system
            self.coral = get_best_coral()
            
            # DPO system
            self.dpo = get_dpo_optimizer()
            
            # Metabolic manager
            self.metabolic = MetabolicManager(registry=self.registry)
            
            # Event streaming
            self.streaming = get_event_streaming()
            await self.streaming.start_streaming()
            
            # Neo4j integration
            self.neo4j = get_neo4j_integration()
            
            # LNN Core
            self.lnn = LNNCore(input_size=10, output_size=5)
            
            self.initialized = True
            print("✅ AURA Intelligence systems initialized")
            
        except Exception as e:
            print(f"❌ AURA initialization error: {e}")
            raise
    
    async def process_component(self, component_type: str, component_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through specific component"""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.registry.process_data(component_id, data)
            return {
                'success': True,
                'component_id': component_id,
                'component_type': component_type,
                'result': result,
                'processing_time': self.registry.components[component_id].processing_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'component_id': component_id
            }
    
    async def process_unified(self, data: Dict[str, Any], systems: List[str]) -> Dict[str, Any]:
        """Process through multiple AURA systems"""
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # Registry processing
        if "registry" in systems and self.registry:
            try:
                neural_components = self.registry.get_components_by_type(ComponentType.NEURAL)
                if neural_components:
                    result = await self.registry.process_data(neural_components[0].id, data)
                    results['registry'] = result
            except Exception as e:
                results['registry'] = {'error': str(e)}
        
        # Consciousness processing
        if "consciousness" in systems and self.consciousness:
            try:
                from aura_intelligence.consciousness.global_workspace import WorkspaceContent
                content = WorkspaceContent(
                    content_id=f"ultimate_api_{datetime.now().timestamp()}",
                    source="ultimate_api_system",
                    data=data,
                    priority=1,
                    attention_weight=0.8
                )
                await self.consciousness.process_content(content)
                results['consciousness'] = self.consciousness.get_state()
            except Exception as e:
                results['consciousness'] = {'error': str(e)}
        
        # CoRaL processing
        if "coral" in systems and self.coral:
            try:
                coral_result = await self.coral.communicate([data])
                results['coral'] = coral_result
            except Exception as e:
                results['coral'] = {'error': str(e)}
        
        # DPO processing
        if "dpo" in systems and self.dpo:
            try:
                action = {
                    'action': 'ultimate_api_request',
                    'confidence': 0.8,
                    'risk_level': 'low'
                }
                dpo_result = await self.dpo.evaluate_action_preference(action, data)
                results['dpo'] = dpo_result
            except Exception as e:
                results['dpo'] = {'error': str(e)}
        
        return {
            'unified_processing': True,
            'systems_processed': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

# Global AURA integration instance
aura_integration = AURAIntegration()

# API endpoints
@aura_router.post("/process")
async def process_aura_component(request: AURAProcessRequest):
    """Process data through specific AURA component"""
    
    # Get component by type
    if request.component_id:
        component_id = request.component_id
    else:
        # Auto-select first component of type
        if not aura_integration.initialized:
            await aura_integration.initialize()
        
        if request.component_type == "neural":
            components = aura_integration.registry.get_components_by_type(ComponentType.NEURAL)
        elif request.component_type == "memory":
            components = aura_integration.registry.get_components_by_type(ComponentType.MEMORY)
        elif request.component_type == "agent":
            components = aura_integration.registry.get_components_by_type(ComponentType.AGENT)
        elif request.component_type == "tda":
            components = aura_integration.registry.get_components_by_type(ComponentType.TDA)
        elif request.component_type == "orchestration":
            components = aura_integration.registry.get_components_by_type(ComponentType.ORCHESTRATION)
        elif request.component_type == "observability":
            components = aura_integration.registry.get_components_by_type(ComponentType.OBSERVABILITY)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown component type: {request.component_type}")
        
        if not components:
            raise HTTPException(status_code=404, detail=f"No components found for type: {request.component_type}")
        
        component_id = components[0].id
    
    result = await aura_integration.process_component(
        request.component_type, 
        component_id, 
        request.data
    )
    
    return result

@aura_router.post("/unified")
async def process_unified_systems(request: AURASystemRequest):
    """Process through multiple AURA systems"""
    
    result = await aura_integration.process_unified(request.data, request.systems)
    return result

@aura_router.get("/components")
async def list_aura_components():
    """List all AURA components"""
    
    if not aura_integration.initialized:
        await aura_integration.initialize()
    
    components = []
    for comp_id, component in list(aura_integration.registry.components.items())[:50]:  # First 50
        components.append({
            'id': comp_id,
            'type': component.type.value,
            'status': component.status,
            'processing_time': component.processing_time,
            'data_processed': component.data_processed
        })
    
    return {
        'total_components': len(aura_integration.registry.components),
        'showing': len(components),
        'components': components,
        'stats': aura_integration.registry.get_component_stats()
    }

@aura_router.get("/systems/status")
async def aura_systems_status():
    """Get status of all AURA systems"""
    
    if not aura_integration.initialized:
        await aura_integration.initialize()
    
    return {
        'systems': {
            'registry': {
                'initialized': aura_integration.registry is not None,
                'components': len(aura_integration.registry.components) if aura_integration.registry else 0
            },
            'consciousness': {
                'initialized': aura_integration.consciousness is not None,
                'active': aura_integration.consciousness.active if aura_integration.consciousness else False
            },
            'coral': {
                'initialized': aura_integration.coral is not None,
                'rounds': aura_integration.coral.rounds if aura_integration.coral else 0
            },
            'dpo': {
                'initialized': aura_integration.dpo is not None,
                'training_steps': len(aura_integration.dpo.training_history) if aura_integration.dpo else 0
            },
            'streaming': {
                'initialized': aura_integration.streaming is not None
            },
            'neo4j': {
                'initialized': aura_integration.neo4j is not None
            }
        },
        'overall_health': 'healthy' if aura_integration.initialized else 'initializing'
    }

@aura_router.get("/lnn/process")
async def process_lnn(data: List[float]):
    """Process data through MIT LNN"""
    
    if not aura_integration.initialized:
        await aura_integration.initialize()
    
    import torch
    
    # Convert to tensor
    input_tensor = torch.tensor([data], dtype=torch.float32)
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(1)  # Add sequence dimension
    
    # Process through LNN
    with torch.no_grad():
        output = aura_integration.lnn(input_tensor)
    
    return {
        'lnn_output': output.squeeze().tolist(),
        'mit_research': True,
        'continuous_dynamics': True,
        'input_shape': list(input_tensor.shape),
        'output_shape': list(output.shape)
    }

# Initialize on import
async def initialize_aura_integration():
    """Initialize AURA integration"""
    await aura_integration.initialize()

# Export for use in main API
__all__ = ['aura_router', 'aura_integration', 'initialize_aura_integration']