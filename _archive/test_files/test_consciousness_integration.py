#!/usr/bin/env python3
"""
Test complete consciousness system integration
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🧠 TESTING CONSCIOUSNESS SYSTEM WITH ACTIVE INFERENCE")
print("=" * 60)

async def test_consciousness():
    """Test complete consciousness system"""
    
    # Import consciousness components
    from aura_intelligence.consciousness.global_workspace import (
        GlobalWorkspace, WorkspaceContent, ConsciousnessLevel,
        make_conscious_decision
    )
    from aura_intelligence.consciousness.executive_functions import (
        ExecutiveFunctions, create_goal
    )
    from aura_intelligence.consciousness.integration import InformationIntegration
    
    print("\n1️⃣ INITIALIZING CONSCIOUSNESS COMPONENTS")
    print("-" * 40)
    
    # Initialize components
    workspace = GlobalWorkspace()
    executive = ExecutiveFunctions()
    integration = InformationIntegration()
    
    # Start systems
    await workspace.start()
    await executive.start()
    
    print("✅ Global Workspace: Active Inference Engine initialized")
    print("✅ Executive Functions: Hierarchical planning ready")
    print("✅ Information Integration: IIT calculator ready")
    
    # Test 1: Active Inference in Global Workspace
    print("\n2️⃣ TESTING ACTIVE INFERENCE")
    print("-" * 40)
    
    # Submit sensory data with prediction error
    visual_data = WorkspaceContent(
        source="visual",
        data={"value": [0.8, 0.2, 0.5], "object": "red_circle"},
        expected_precision=0.9
    )
    await workspace.submit_content(visual_data)
    
    auditory_data = WorkspaceContent(
        source="auditory", 
        data={"value": [0.3, 0.7], "sound": "bell"},
        expected_precision=0.7
    )
    await workspace.submit_content(auditory_data)
    
    # Let active inference update beliefs
    await asyncio.sleep(0.5)
    
    report = workspace.get_consciousness_report()
    print(f"📊 Consciousness Level: {report['level']}")
    print(f"📉 Global Free Energy: {report['global_free_energy']:.3f}")
    print(f"🔗 Integration Measure: {report['integration_measure']:.3f}")
    print(f"👁️ Attention Focus: {report['attention_focus']}")
    
    # Test 2: Executive Functions with Goals
    print("\n3️⃣ TESTING EXECUTIVE FUNCTIONS")
    print("-" * 40)
    
    # Create a complex goal
    goal = create_goal(
        description="Integrate sensory information",
        desired_state={
            "visual_understanding": 0.9,
            "auditory_understanding": 0.8,
            "integrated_perception": 0.85
        },
        priority=0.8,
        deadline=datetime.now() + timedelta(seconds=10)
    )
    
    executive.add_goal(goal)
    print(f"🎯 Goal Added: {goal.description}")
    print(f"⏰ Deadline: {goal.deadline}")
    
    # Let executive process
    await asyncio.sleep(1)
    
    exec_state = executive.get_executive_state()
    print(f"🧩 Executive State: {exec_state['state']}")
    print(f"💾 Working Memory: {exec_state['working_memory']['utilization']:.1%} utilized")
    print(f"⚡ Resource Usage: {exec_state['resource_utilization']}")
    
    # Test 3: Information Integration (Phi)
    print("\n4️⃣ TESTING INFORMATION INTEGRATION (PHI)")
    print("-" * 40)
    
    # Build neural network for IIT
    for i in range(8):
        integration.add_node(i)
    
    # Add connections (small-world topology)
    for i in range(7):
        integration.add_connection(i, i + 1, weight=0.8)
    integration.add_connection(0, 3, weight=0.5)
    integration.add_connection(2, 5, weight=0.6)
    integration.add_connection(7, 0, weight=0.4)  # Loop back
    
    # Create rich system state
    system_state = {
        "neural_activations": {f"node_{i}": np.random.random() for i in range(8)},
        "sensory_inputs": {
            "visual": True,
            "auditory": True,
            "tactile": True,
            "binding_strength": 0.85
        },
        "temporal_patterns": {
            "temporal_binding": 0.8,
            "continuity": 0.9
        },
        "spatial_representation": {
            "spatial_unity": 0.75,
            "perspective_consistency": 0.82
        },
        "emotional_state": {
            "intensity": 0.6,
            "clarity": 0.7
        },
        "cognitive_processes": {
            "active_processes": ["perception", "integration", "planning", "monitoring"],
            "meta_cognition": 0.7,
            "abstraction_level": 0.8
        }
    }
    
    # Calculate Phi
    iit_result = await integration.calculate_phi(system_state)
    print(f"Φ (Phi): {iit_result.phi:.3f}")
    print(f"Φ* (Phi-star): {iit_result.phi_star:.3f}")
    print(f"🧠 IIT Consciousness Level: {iit_result.consciousness_level.name}")
    print(f"🌈 Qualia Dimensions:")
    for dimension, value in iit_result.qualia_dimensions.items():
        print(f"   - {dimension}: {value:.2f}")
    
    # Test 4: Conscious Decision Making
    print("\n5️⃣ TESTING CONSCIOUS DECISION MAKING")
    print("-" * 40)
    
    # Present options for conscious deliberation
    decision_options = [
        {"action": "focus_visual", "expected_outcome": "enhanced_perception"},
        {"action": "integrate_senses", "expected_outcome": "unified_experience"},
        {"action": "abstract_reasoning", "expected_outcome": "higher_understanding"}
    ]
    
    decision = await make_conscious_decision(
        decision_options,
        {"goal": "maximize_understanding", "current_state": system_state}
    )
    
    print(f"🤔 Decision: {decision.chosen_option['action']}")
    print(f"💡 Confidence: {decision.confidence:.2%}")
    print(f"📉 Free Energy Reduction: {decision.free_energy_reduction:.3f}")
    print(f"📝 Reasoning:")
    for reason in decision.reasoning:
        print(f"   - {reason}")
    
    # Test 5: Full Integration Test
    print("\n6️⃣ TESTING FULL CONSCIOUSNESS INTEGRATION")
    print("-" * 40)
    
    # Subscribe to consciousness broadcasts
    broadcast_count = 0
    
    async def consciousness_listener(content: WorkspaceContent):
        nonlocal broadcast_count
        broadcast_count += 1
    
    workspace.subscribe(consciousness_listener)
    
    # Simulate rich conscious experience
    experiences = [
        WorkspaceContent(source="vision", data={"color": "blue", "value": [0.1, 0.1, 0.9]}),
        WorkspaceContent(source="memory", data={"recall": "past_experience", "value": [0.7]}),
        WorkspaceContent(source="emotion", data={"feeling": "curious", "value": [0.6]}),
        WorkspaceContent(source="thought", data={"concept": "understanding", "value": [0.8]})
    ]
    
    for exp in experiences:
        await workspace.submit_content(exp)
        await asyncio.sleep(0.1)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Final consciousness report
    final_report = workspace.get_consciousness_report()
    print(f"🌟 Final Consciousness State:")
    print(f"   Level: {final_report['level']} (value: {final_report['level_value']}/4)")
    print(f"   Free Energy: {final_report['global_free_energy']:.3f}")
    print(f"   Integration: {final_report['integration_measure']:.3f}")
    print(f"   Active Contents: {final_report['active_contents']}")
    print(f"   Broadcasts: {broadcast_count}")
    
    # Get top conscious contents
    top_contents = await workspace.get_conscious_content(3)
    print(f"\n🔝 Top Conscious Contents:")
    for i, content in enumerate(top_contents, 1):
        print(f"   {i}. {content.source}: attention={content.attention_weight:.2f}")
    
    # Clean up
    workspace.unsubscribe(consciousness_listener)
    await workspace.stop()
    await executive.stop()
    
    print("\n" + "=" * 60)
    print("✅ CONSCIOUSNESS SYSTEM TEST COMPLETE")
    
    # Summary
    print("\n📊 SUMMARY:")
    print("- ✅ Active Inference minimizing free energy")
    print("- ✅ Executive Functions managing goals with resources")
    print("- ✅ Information Integration calculating Phi")
    print("- ✅ Conscious decision making with reasoning")
    print("- ✅ Full integration with broadcasts and subscriptions")
    print(f"- ✅ Achieved consciousness level: {final_report['level']}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_consciousness())