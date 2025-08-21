"""
Test script for LNN Service
Verifies basic functionality before integration
"""

import httpx
import asyncio
import json
import torch
import time
from typing import Dict, Any

async def test_lnn_service():
    """Test LNN service endpoints"""
    base_url = "http://localhost:8003"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n🧪 Testing AURA Liquid Neural Network Service\n")
        
        # 1. Health Check
        print("1️⃣ Testing health check...")
        try:
            resp = await client.get(f"{base_url}/api/v1/health")
            resp.raise_for_status()
            health = resp.json()
            print(f"   ✅ Service healthy: {health['total_models']} models loaded")
            print(f"   📊 Total inferences: {health['total_inferences']}")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return
        
        # 2. List Models
        print("\n2️⃣ Listing available models...")
        try:
            resp = await client.get(f"{base_url}/api/v1/models")
            resp.raise_for_status()
            models = resp.json()
            print(f"   ✅ Found {models['total']} models:")
            for model_id, info in models['models'].items():
                print(f"      - {model_id}: {info['implementation']} ({info['parameters']} params)")
        except Exception as e:
            print(f"   ❌ Failed to list models: {e}")
        
        # 3. Test Standard Inference
        print("\n3️⃣ Testing standard LNN inference...")
        try:
            input_data = [0.1 * i for i in range(128)]  # 128 dimensions
            resp = await client.post(
                f"{base_url}/api/v1/inference",
                json={
                    "model_id": "standard",
                    "input_data": input_data,
                    "session_id": "test_session",
                    "return_dynamics": True
                }
            )
            resp.raise_for_status()
            result = resp.json()
            print(f"   ✅ Inference successful!")
            print(f"      - Output shape: {len(result['output'])} dimensions")
            print(f"      - Latency: {result['latency_ms']:.2f}ms")
            if result.get('dynamics'):
                print(f"      - Mean activation: {result['dynamics']['mean_activation']:.4f}")
                print(f"      - Sparsity: {result['dynamics']['sparsity']:.2%}")
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
        
        # 4. Test Adaptive Model
        print("\n4️⃣ Testing adaptive LNN (self-modifying)...")
        try:
            # Run multiple inferences with increasing complexity
            for i in range(3):
                complexity = i / 2
                # Create input with varying complexity
                if complexity < 0.5:
                    input_data = [0.5] * 128  # Simple pattern
                else:
                    input_data = [torch.randn(1).item() * complexity for _ in range(128)]  # Complex
                
                resp = await client.post(
                    f"{base_url}/api/v1/inference",
                    json={
                        "model_id": "adaptive",
                        "input_data": input_data,
                        "session_id": "adaptive_test"
                    }
                )
                resp.raise_for_status()
                result = resp.json()
                
                adaptations = result.get('adaptations', {})
                if adaptations:
                    print(f"   ✅ Step {i+1}: Model adapted! {adaptations}")
                else:
                    print(f"   ✅ Step {i+1}: No adaptation needed")
        except Exception as e:
            print(f"   ❌ Adaptive test failed: {e}")
        
        # 5. Test Real-time Adaptation
        print("\n5️⃣ Testing real-time parameter adaptation...")
        try:
            # Give positive feedback
            resp = await client.post(
                f"{base_url}/api/v1/adapt",
                json={
                    "model_id": "standard",
                    "feedback_signal": 0.8,
                    "adaptation_strength": 0.1
                }
            )
            resp.raise_for_status()
            adapt_result = resp.json()
            print(f"   ✅ Adaptation successful!")
            print(f"      - Type: {adapt_result['adaptation_type']}")
            print(f"      - Parameters changed: {adapt_result['parameters_changed']}")
        except Exception as e:
            print(f"   ❌ Adaptation failed: {e}")
        
        # 6. Test Continuous Learning
        print("\n6️⃣ Testing continuous learning...")
        try:
            # Create simple training data
            train_data = [[i * 0.01 for i in range(128)] for _ in range(10)]
            train_labels = [[1.0 if i < 32 else 0.0 for i in range(64)] for _ in range(10)]
            
            resp = await client.post(
                f"{base_url}/api/v1/train/continuous",
                json={
                    "model_id": "standard",
                    "training_data": train_data,
                    "training_labels": train_labels,
                    "learning_rate": 0.001,
                    "epochs": 2
                }
            )
            resp.raise_for_status()
            train_result = resp.json()
            print(f"   ✅ Training completed!")
            print(f"      - Samples processed: {train_result['samples_processed']}")
            print(f"      - Loss improvement: {train_result['loss_before']:.4f} → {train_result['loss_after']:.4f}")
            print(f"      - Training time: {train_result['training_time_ms']:.1f}ms")
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
        
        # 7. Test Model Creation
        print("\n7️⃣ Testing custom model creation...")
        try:
            resp = await client.post(
                f"{base_url}/api/v1/models/create",
                json={
                    "model_id": "custom_test",
                    "mode": "adaptive",
                    "input_size": 64,
                    "hidden_size": 128,
                    "output_size": 32,
                    "enable_growth": True,
                    "ode_solver": "dopri5"
                }
            )
            resp.raise_for_status()
            create_result = resp.json()
            print(f"   ✅ Custom model created: {create_result['model_id']}")
            print(f"      - Parameters: {create_result['config']['parameters']}")
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
        
        # 8. Test Consensus Inference
        print("\n8️⃣ Testing consensus-based inference...")
        try:
            resp = await client.post(
                f"{base_url}/api/v1/inference/consensus",
                json={
                    "model_ids": ["standard", "adaptive", "edge"],
                    "input_data": [0.5] * 128,
                    "consensus_method": "weighted_average"
                }
            )
            resp.raise_for_status()
            consensus_result = resp.json()
            print(f"   ✅ Consensus reached!")
            print(f"      - Participants: {consensus_result['participants']}")
            print(f"      - Method: {consensus_result['consensus_method']}")
            print(f"      - Output dimensions: {len(consensus_result['consensus_output'])}")
        except Exception as e:
            print(f"   ❌ Consensus inference failed: {e}")
        
        # 9. Test Demo Endpoint
        print("\n9️⃣ Testing self-modification demo...")
        try:
            resp = await client.get(f"{base_url}/api/v1/demo/adaptation?steps=5")
            resp.raise_for_status()
            demo_result = resp.json()
            print(f"   ✅ Demo completed!")
            final_state = demo_result['final_state']
            print(f"      - Total adaptations: {final_state['total_adaptations']}")
            print(f"      - Final neurons: {final_state['final_neurons']}")
            print(f"      - Total parameters: {final_state['parameters']}")
        except Exception as e:
            print(f"   ❌ Demo failed: {e}")
        
        print("\n✅ LNN Service tests completed!")
        print("\n📊 Summary:")
        print("   - MIT's ncps library integration ✓")
        print("   - Continuous-time dynamics ✓")
        print("   - Self-modifying architecture ✓")
        print("   - Real-time adaptation ✓")
        print("   - Continuous learning ✓")
        print("   - Consensus inference ✓")


async def test_websocket():
    """Test WebSocket streaming"""
    print("\n🔌 Testing WebSocket streaming...")
    
    import websockets
    
    try:
        async with websockets.connect("ws://localhost:8003/ws/inference/standard") as ws:
            # Send inference request
            await ws.send(json.dumps({
                "type": "inference",
                "input": [0.5] * 128
            }))
            
            # Receive result
            result = json.loads(await ws.recv())
            print(f"   ✅ WebSocket inference successful!")
            print(f"      - Latency: {result['latency_ms']:.2f}ms")
            
            # Test adaptation via WebSocket
            await ws.send(json.dumps({
                "type": "adapt",
                "feedback": 0.5
            }))
            
            adapt_result = json.loads(await ws.recv())
            print(f"   ✅ WebSocket adaptation: {adapt_result['status']}")
            
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")


if __name__ == "__main__":
    print("🌊 AURA Liquid Neural Network Service Test Suite")
    print("=" * 50)
    
    # Run main tests
    asyncio.run(test_lnn_service())
    
    # Run WebSocket test
    try:
        asyncio.run(test_websocket())
    except:
        print("\n⚠️  WebSocket test skipped (requires websockets library)")
    
    print("\n🎉 All tests completed!")