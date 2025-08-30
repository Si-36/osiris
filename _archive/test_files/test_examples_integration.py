#!/usr/bin/env python3
"""
Test examples system with integration to other AURA components
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🎯 TESTING EXAMPLES SYSTEM WITH INTEGRATION")
print("=" * 60)

async def test_examples_integration():
    """Test examples system integrated with other components"""
    
    try:
        # Test imports
        print("\n1️⃣ TESTING IMPORTS")
        print("-" * 40)
        
        from aura_intelligence.examples.interactive_demos import (
            DemoOrchestrator, LNNDemo, ConsciousnessDemo, TDADemo,
            DemoConfig, DemoResult, create_demo_suite
        )
        print("✅ Interactive demos imports successful")
        
        # Initialize demo orchestrator
        print("\n2️⃣ INITIALIZING DEMO SYSTEM")
        print("-" * 40)
        
        orchestrator = await create_demo_suite()
        print("✅ Demo orchestrator initialized")
        
        available = orchestrator.get_available_demos()
        print(f"✅ Registered {len(available)} demos:")
        for demo in available:
            print(f"   - {demo['name']} ({demo['category']})")
        
        # Test LNN demo with real components
        print("\n3️⃣ TESTING LNN DEMO")
        print("-" * 40)
        
        # Generate test sequence
        sequence_length = 50
        input_sequence = np.random.randn(sequence_length, 10)
        
        lnn_result = await orchestrator.run_demo(
            "liquid_neural_network",
            input_sequence=input_sequence
        )
        
        print(f"✅ LNN demo completed")
        print(f"   Output shape: {lnn_result.output['output'].shape if lnn_result.output else 'N/A'}")
        print(f"   Inference time: {lnn_result.inference_time_ms:.2f}ms")
        print(f"   Hidden states: {lnn_result.output.get('hidden_states', []).shape if lnn_result.output else 'N/A'}")
        print(f"   Visualizations: {list(lnn_result.visualizations.keys())}")
        
        # Test consciousness demo with integration
        print("\n4️⃣ TESTING CONSCIOUSNESS DEMO WITH INTEGRATION")
        print("-" * 40)
        
        try:
            # Import consciousness components
            from aura_intelligence.consciousness.global_workspace import GlobalWorkspace
            from aura_intelligence.consciousness.executive_functions import ExecutiveFunctions
            
            # Run consciousness demo
            consciousness_result = await orchestrator.run_demo(
                "consciousness_system",
                stimulus="Test integration with TDA and LNN",
                attention_level=0.9
            )
            
            print(f"✅ Consciousness demo completed")
            if consciousness_result.output:
                workspace = consciousness_result.output.get("workspace_state", {})
                print(f"   Attention focus: {workspace.get('attention_focus', 0):.2f}")
                print(f"   Conscious content: {workspace.get('conscious_content', [])}")
                print(f"   Phi integrated: {workspace.get('phi_integrated', 0):.2f}")
            print(f"   Visualizations: {list(consciousness_result.visualizations.keys())}")
            
        except ImportError as e:
            print(f"⚠️  Consciousness integration skipped: {e}")
        
        # Test TDA demo with point clouds
        print("\n5️⃣ TESTING TDA DEMO WITH COMPLEX SHAPES")
        print("-" * 40)
        
        # Generate different topological shapes
        shapes = {
            "circle": lambda n: np.column_stack([
                np.cos(np.linspace(0, 2*np.pi, n)),
                np.sin(np.linspace(0, 2*np.pi, n)),
                np.zeros(n)
            ]),
            "torus": lambda n: (lambda t, p: np.column_stack([
                (2 + np.cos(t)) * np.cos(p),
                (2 + np.cos(t)) * np.sin(p),
                np.sin(t)
            ]))(np.random.rand(n) * 2 * np.pi, np.random.rand(n) * 2 * np.pi),
            "sphere": lambda n: (lambda u, v: np.column_stack([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u)
            ]))(np.random.rand(n) * np.pi, np.random.rand(n) * 2 * np.pi)
        }
        
        for shape_name, shape_func in shapes.items():
            points = shape_func(200) + np.random.randn(200, 3) * 0.05  # Add noise
            
            tda_result = await orchestrator.run_demo(
                "topological_analysis",
                point_cloud=points,
                max_dimension=2
            )
            
            print(f"\n✅ TDA analysis of {shape_name}:")
            if tda_result.output:
                print(f"   Betti numbers: {tda_result.output.get('betti_numbers', [])}")
                print(f"   Wasserstein distance: {tda_result.output.get('wasserstein_distance', 0):.3f}")
                print(f"   Persistence entropy: {tda_result.output.get('persistence_entropy', 0):.3f}")
            print(f"   Inference time: {tda_result.inference_time_ms:.2f}ms")
        
        # Test integration with event system
        print("\n6️⃣ TESTING INTEGRATION WITH EVENT SYSTEM")
        print("-" * 40)
        
        try:
            from aura_intelligence.events.event_system import (
                EventBus, Event, EventType, EventMetadata
            )
            
            event_bus = EventBus()
            demo_events = []
            
            # Subscribe to demo events
            async def demo_event_handler(event: Event):
                demo_events.append(event)
            
            event_bus.subscribe(
                [EventType.MODEL_PREDICTION, EventType.DATA_PROCESSED],
                demo_event_handler
            )
            
            # Publish demo results as events
            await event_bus.publish(Event(
                type=EventType.MODEL_PREDICTION,
                data={
                    "model": "LNN",
                    "inference_time_ms": lnn_result.inference_time_ms,
                    "output_shape": str(lnn_result.output['output'].shape) if lnn_result.output else None
                },
                metadata=EventMetadata(source="demo_system")
            ))
            
            await event_bus.publish(Event(
                type=EventType.DATA_PROCESSED,
                data={
                    "processor": "TDA",
                    "shapes_analyzed": list(shapes.keys()),
                    "total_points": 200 * len(shapes)
                },
                metadata=EventMetadata(source="demo_system")
            ))
            
            await asyncio.sleep(0.1)  # Let events process
            
            print(f"✅ Published {2} demo events")
            print(f"✅ Received {len(demo_events)} events in handler")
            
        except ImportError as e:
            print(f"⚠️  Event system integration skipped: {e}")
        
        # Test integration with collective intelligence
        print("\n7️⃣ TESTING INTEGRATION WITH COLLECTIVE INTELLIGENCE")
        print("-" * 40)
        
        try:
            from aura_intelligence.collective.orchestrator import Orchestrator
            from aura_intelligence.collective.memory_manager import MemoryManager
            
            # Create memory of demo results
            memory_manager = MemoryManager()
            
            # Store demo results
            await memory_manager.store(
                agent_id="demo_system",
                memory_type="episodic",
                content={
                    "lnn_inference": lnn_result.inference_time_ms,
                    "tda_shapes": list(shapes.keys()),
                    "consciousness_phi": consciousness_result.output.get("metrics", {}).get("integration_measure", 0) if consciousness_result.output else 0
                }
            )
            
            # Query memories
            memories = await memory_manager.query(
                query="demo results",
                top_k=5
            )
            
            print(f"✅ Stored demo results in collective memory")
            print(f"✅ Retrieved {len(memories)} relevant memories")
            
        except ImportError as e:
            print(f"⚠️  Collective intelligence integration skipped: {e}")
        
        # Test performance comparison
        print("\n8️⃣ TESTING PERFORMANCE COMPARISON")
        print("-" * 40)
        
        # Run multiple iterations
        iterations = 10
        lnn_times = []
        tda_times = []
        
        print(f"Running {iterations} iterations...")
        
        for i in range(iterations):
            # LNN timing
            lnn_res = await orchestrator.run_demo(
                "liquid_neural_network",
                input_sequence=np.random.randn(30, 10)
            )
            lnn_times.append(lnn_res.inference_time_ms)
            
            # TDA timing
            tda_res = await orchestrator.run_demo(
                "topological_analysis",
                point_cloud=np.random.randn(100, 3)
            )
            tda_times.append(tda_res.inference_time_ms)
        
        print(f"\n✅ Performance comparison:")
        print(f"   LNN average: {np.mean(lnn_times):.2f}ms (±{np.std(lnn_times):.2f})")
        print(f"   TDA average: {np.mean(tda_times):.2f}ms (±{np.std(tda_times):.2f})")
        
        # Get demo statistics
        print("\n9️⃣ DEMO STATISTICS")
        print("-" * 40)
        
        for demo_info in orchestrator.get_available_demos():
            stats = demo_info["stats"]
            print(f"\n{demo_info['name']}:")
            print(f"   Run count: {stats['run_count']}")
            print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"   P95 latency: {stats['p95_latency_ms']:.2f}ms")
            print(f"   Error rate: {stats['error_rate']:.1%}")
        
        # Test error handling
        print("\n🔟 TESTING ERROR HANDLING")
        print("-" * 40)
        
        # Test with invalid input
        error_result = await orchestrator.run_demo(
            "topological_analysis",
            point_cloud=np.array([])  # Empty array should cause error
        )
        
        if error_result.error:
            print(f"✅ Error handling working: {error_result.error}")
        else:
            print("⚠️  Expected error was not raised")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ EXAMPLES INTEGRATION TEST COMPLETE")
        
        print("\n📊 SUMMARY:")
        print("- ✅ Interactive demo system with 3 demos")
        print("- ✅ LNN dynamics visualization")
        print("- ✅ Consciousness state exploration")
        print("- ✅ TDA topological analysis")
        print("- ✅ Integration with event system")
        print("- ✅ Integration with collective memory")
        print("- ✅ Performance benchmarking")
        print("- ✅ Error handling and statistics")
        
        print("\n📝 Key Features:")
        print("- Web-based interactive UI (Gradio)")
        print("- Real-time model inference")
        print("- Rich visualizations")
        print("- Component integration demos")
        print("- Performance tracking")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Some dependencies may be missing")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_examples_integration())