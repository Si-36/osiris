#!/usr/bin/env python3
"""
🎯 Test Advanced DPO Features
==============================

Tests the restored DPO system with GPO, DMPO, ICAI, SAOM features.
"""

import sys
import numpy as np
sys.path.insert(0, 'core/src')

def test_dpo_advanced():
    """Test all advanced DPO features"""
    print("Testing Advanced DPO System...")
    print("="*50)
    
    from aura_intelligence.dpo.dpo_2025_advanced import (
        AURAAdvancedDPO,
        PreferenceType,
        DPOConfig,
        PreferenceData,
    )
    
    # Initialize DPO
    config = DPOConfig(
        beta=0.1,
        learning_rate=1e-4,
        use_gpo=True,
        use_dmpo=True,
        use_icai=True,
    )
    
    dpo = AURAAdvancedDPO(config)
    
    # Test 1: GPO Loss
    print("\n1. Testing GPO (Group Preference Optimization)...")
    try:
        # Create dummy data
        preferred = np.random.randn(10, 768)  # 10 samples, 768 dims
        rejected = np.random.randn(10, 768)
        
        gpo_loss = dpo.gpo_loss(preferred, rejected)
        print(f"   ✅ GPO Loss computed: {gpo_loss:.4f}")
    except Exception as e:
        print(f"   ❌ GPO Failed: {e}")
    
    # Test 2: DMPO Loss
    print("\n2. Testing DMPO (Dynamic Margin Preference Optimization)...")
    try:
        dmpo_loss = dpo.dmpo_loss(preferred, rejected)
        print(f"   ✅ DMPO Loss computed: {dmpo_loss:.4f}")
    except Exception as e:
        print(f"   ❌ DMPO Failed: {e}")
    
    # Test 3: ICAI Loss
    print("\n3. Testing ICAI (Iterative Constitutional AI)...")
    try:
        icai_loss = dpo.icai_loss(preferred, rejected)
        print(f"   ✅ ICAI Loss computed: {icai_loss:.4f}")
    except Exception as e:
        print(f"   ❌ ICAI Failed: {e}")
    
    # Test 4: SAOM
    print("\n4. Testing SAOM (State-Action Occupancy Measure)...")
    try:
        states = np.random.randn(100, 512)  # 100 states
        actions = np.random.randn(100, 256)  # 100 actions
        
        saom = dpo.compute_saom(states, actions)
        print(f"   ✅ SAOM computed with shape: {saom.shape}")
    except Exception as e:
        print(f"   ❌ SAOM Failed: {e}")
    
    # Test 5: Preference Learning Pipeline
    print("\n5. Testing Preference Learning Pipeline...")
    try:
        # Create preference data
        pref_data = PreferenceData(
            chosen_text="This is the preferred response",
            rejected_text="This is the rejected response",
            preference_type=PreferenceType.BINARY,
            metadata={"source": "test", "confidence": 0.9}
        )
        
        # Process preference
        result = dpo.process_preference(pref_data)
        print(f"   ✅ Preference processed: {result}")
    except Exception as e:
        print(f"   ❌ Pipeline Failed: {e}")
    
    # Test 6: Multi-Turn Trajectory
    print("\n6. Testing Multi-Turn Trajectory Optimization...")
    try:
        trajectory = [
            {"state": np.random.randn(768), "action": "response_1"},
            {"state": np.random.randn(768), "action": "response_2"},
            {"state": np.random.randn(768), "action": "response_3"},
        ]
        
        trajectory_loss = dpo.optimize_trajectory(trajectory)
        print(f"   ✅ Trajectory optimized: loss={trajectory_loss:.4f}")
    except Exception as e:
        print(f"   ❌ Trajectory Failed: {e}")
    
    # Test 7: Constitutional AI 3.0
    print("\n7. Testing Constitutional AI 3.0...")
    try:
        constitution = {
            "helpful": 0.8,
            "harmless": 0.9,
            "honest": 0.85,
            "creative": 0.7,
        }
        
        dpo.update_constitution(constitution)
        print(f"   ✅ Constitution updated with {len(constitution)} principles")
    except Exception as e:
        print(f"   ❌ Constitutional AI Failed: {e}")
    
    print("\n" + "="*50)
    print("DPO Advanced Features Test Complete!")

if __name__ == "__main__":
    test_dpo_advanced()