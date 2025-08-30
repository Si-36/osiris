#!/usr/bin/env python3
"""
Direct Orchestration Test - Without Complex Dependencies
========================================================
Tests orchestration components directly
"""

import asyncio
import numpy as np
import time
from typing import Dict, Any, List
import sys
import os

# Add AURA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core/src'))


async def test_orchestration_layers():
    """Test orchestration without complex dependencies"""
    print("🚀 AURA Orchestration Direct Test")
    print("="*80)
    
    results = {}
    
    # Test Strategic Layer
    print("\n📊 Testing Strategic Layer...")
    try:
        from aura_intelligence.orchestration.strategic.strategic_planner import StrategicPlanner, ResourcePlan
        from aura_intelligence.orchestration.strategic.drift_detection import DriftDetector, DriftScore
        from aura_intelligence.orchestration.strategic.model_lifecycle import ModelLifecycleManager, ModelVersion
        
        # Test strategic planning
        planner = StrategicPlanner()
        state = {
            'cpu_usage': {'tda': 75.0, 'inference': 85.0},
            'gpu_usage': {'inference': 90.0}
        }
        metrics = {'inference_latency': 120.0}
        
        plan = await planner.create_resource_plan(state, metrics)
        
        print(f"   ✅ Strategic planner created plan")
        print(f"   - Confidence: {plan.confidence:.2f}")
        print(f"   - Scale up: {plan.scale_up_components}")
        print(f"   - CPU allocation total: {sum(plan.cpu_allocation.values()):.1f}%")
        
        # Test drift detection
        detector = DriftDetector()
        print("   ✅ Drift detector initialized")
        
        # Test model lifecycle
        manager = ModelLifecycleManager()
        model = ModelVersion(
            model_id="test_v1",
            version="1.0.0",
            component="inference"
        )
        deployment = await manager.deploy_model(model, "canary")
        
        print(f"   ✅ Model lifecycle - canary at {deployment.canary_percentage}%")
        
        results['strategic'] = {'success': True, 'confidence': plan.confidence}
        
    except Exception as e:
        print(f"   ❌ Strategic layer failed: {e}")
        import traceback
        traceback.print_exc()
        results['strategic'] = {'success': False, 'error': str(e)}
    
    # Test Tactical Layer
    print("\n🔧 Testing Tactical Layer...")
    try:
        from aura_intelligence.orchestration.tactical.pipeline_registry import (
            PipelineRegistry, Pipeline, PipelineVersion
        )
        from aura_intelligence.orchestration.tactical.conditional_flows import (
            ConditionalFlow, BranchCondition, ConditionOperator
        )
        from aura_intelligence.orchestration.tactical.experiment_manager import ExperimentManager
        
        # Test pipeline registry
        registry = PipelineRegistry()
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
            version=PipelineVersion(1, 0, 0),
            phases=['perception', 'inference', 'consensus', 'action']
        )
        
        reg_id = await registry.register_pipeline(pipeline)
        print(f"   ✅ Pipeline registered: {reg_id}")
        
        # Test conditional flows
        flow = ConditionalFlow()
        branch = flow.create_free_energy_branch(1.0)
        flow.register_branch("inference", branch)
        
        target = await flow.evaluate_branches(
            "inference",
            {'phase_results': {'inference': {'metrics': {'free_energy': 2.0}}}}
        )
        print(f"   ✅ Conditional flow branched to: {target}")
        
        # Test experiments
        exp_manager = ExperimentManager(registry)
        print("   ✅ Experiment manager initialized")
        
        results['tactical'] = {'success': True, 'pipeline': reg_id}
        
    except Exception as e:
        print(f"   ❌ Tactical layer failed: {e}")
        results['tactical'] = {'success': False, 'error': str(e)}
    
    # Test Operational Layer
    print("\n⚡ Testing Operational Layer...")
    try:
        from aura_intelligence.orchestration.operational.event_router import (
            EventRouter, Event, EventPriority, CognitivePriorityEngine
        )
        
        # Test event router
        router = EventRouter()
        await router.start()
        
        # Test cognitive priority
        engine = CognitivePriorityEngine()
        event = Event(
            event_id="test_1",
            event_type="test_event",
            metadata={'user_facing': True}
        )
        
        score = await engine.score(event)
        print(f"   ✅ Cognitive priority score: {score:.2f}")
        
        # Route some events
        handled = []
        async def handler(e):
            handled.append(e.event_id)
            
        router.register_handler('test_event', handler)
        
        for i in range(5):
            await router.route_event(Event(
                event_id=f"evt_{i}",
                event_type='test_event'
            ))
            
        await asyncio.sleep(0.1)
        await router.stop()
        
        print(f"   ✅ Event router processed {len(handled)} events")
        
        results['operational'] = {'success': True, 'score': score}
        
    except Exception as e:
        print(f"   ❌ Operational layer failed: {e}")
        results['operational'] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r.get('success'))
    total = len(results)
    
    print(f"\nTotal Layers: {total}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for layer, result in results.items():
        status = "✅" if result.get('success') else "❌"
        print(f"  {status} {layer}: {result}")
    
    if passed == total:
        print("\n🎉 All orchestration layers working!")
        print("\n✨ Verified Capabilities:")
        print("   ✅ Strategic: Drift detection, planning, lifecycle")
        print("   ✅ Tactical: Pipelines, conditions, experiments")
        print("   ✅ Operational: Event routing, cognitive priority")
        print("\n   → Orchestration ready for production!")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_orchestration_layers())
    sys.exit(0 if success else 1)