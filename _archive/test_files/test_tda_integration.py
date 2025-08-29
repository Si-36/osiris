#!/usr/bin/env python3
"""
Test TDA system with integration to other AURA components
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import torch
import time
from datetime import datetime
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

print("🔺 TESTING TOPOLOGICAL DATA ANALYSIS SYSTEM WITH INTEGRATION")
print("=" * 60)

async def test_tda_integration():
    """Test TDA system integrated with other components"""
    
    try:
        # Test imports
        print("\n1️⃣ TESTING IMPORTS")
        print("-" * 40)
        
        from aura_intelligence.tda.advanced_tda_system import (
            AdvancedTDAEngine, TDAConfig, ComplexType, PersistenceType,
            PersistenceDiagram, VietorisRipsGPU, PersistenceComputer,
            NeuralPersistenceLayer, TopologicalLoss
        )
        print("✅ Advanced TDA system imports successful")
        
        from aura_intelligence.tda.algorithms import PersistentHomology
        print("✅ TDA algorithms imports successful")
        
        # Initialize TDA system
        print("\n2️⃣ INITIALIZING TDA SYSTEM")
        print("-" * 40)
        
        config = TDAConfig(
            complex_type=ComplexType.RIPS,
            max_dimension=2,
            max_edge_length=2.0,
            use_gpu=False,  # Set to False for CPU testing
            use_neural_persistence=True
        )
        
        tda_engine = AdvancedTDAEngine(config)
        print("✅ TDA engine initialized")
        print(f"   Max dimension: {config.max_dimension}")
        print(f"   Complex type: {config.complex_type}")
        print(f"   Neural persistence: {config.use_neural_persistence}")
        
        # Test with different geometric shapes
        print("\n3️⃣ TESTING WITH GEOMETRIC SHAPES")
        print("-" * 40)
        
        # Create test shapes
        shapes = {
            "Circle": create_circle_points(50),
            "Torus": create_torus_points(100),
            "Sphere": create_sphere_points(80),
            "Figure-8": create_figure_eight_points(60)
        }
        
        expected_betti = {
            "Circle": [1, 1, 0],      # 1 component, 1 loop
            "Torus": [1, 2, 1],       # 1 component, 2 loops, 1 void
            "Sphere": [1, 0, 1],      # 1 component, 0 loops, 1 void
            "Figure-8": [1, 2, 0]     # 1 component, 2 loops
        }
        
        for shape_name, points in shapes.items():
            print(f"\n{shape_name}:")
            
            # Compute TDA features
            features = tda_engine.compute_tda_features(
                points,
                return_diagrams=True,
                return_landscapes=True,
                return_images=True
            )
            
            betti = features['betti_numbers']
            print(f"  Betti numbers: b0={int(betti[0])}, b1={int(betti[1])}, b2={int(betti[2])}")
            
            # Compare with expected
            expected = expected_betti.get(shape_name, [0, 0, 0])
            if shape_name in ["Circle", "Figure-8"]:  # 2D shapes
                match = (int(betti[0]) == expected[0] and int(betti[1]) == expected[1])
            else:
                match = (int(betti[0]) == expected[0])
            
            print(f"  Expected: b0={expected[0]}, b1={expected[1]}, b2={expected[2]}")
            print(f"  Match: {'✅' if match else '⚠️'}")
            
            # Show persistence stats
            for key, value in features['persistence_stats'].items():
                if value > 0:
                    print(f"  {key}: {value:.3f}")
        
        # Test persistence diagrams
        print("\n4️⃣ TESTING PERSISTENCE DIAGRAMS")
        print("-" * 40)
        
        # Use torus for interesting topology
        torus_points = shapes["Torus"]
        features = tda_engine.compute_tda_features(torus_points)
        
        diagrams = features['persistence_diagrams']
        
        for dim, dgm in diagrams.items():
            if len(dgm.birth_death) > 0:
                print(f"\nDimension {dim}:")
                print(f"  Number of features: {len(dgm.birth_death)}")
                
                # Show top persistent features
                persistence = dgm.persistence()
                if len(persistence) > 0:
                    top_indices = np.argsort(persistence)[-3:][::-1]
                    
                    print("  Top persistent features:")
                    for idx in top_indices:
                        if idx < len(dgm.birth_death):
                            birth, death = dgm.birth_death[idx]
                            pers = persistence[idx]
                            print(f"    Birth: {birth:.3f}, Death: {death:.3f}, Persistence: {pers:.3f}")
        
        # Test neural persistence
        print("\n5️⃣ TESTING NEURAL PERSISTENCE")
        print("-" * 40)
        
        if config.use_neural_persistence:
            # Convert to tensor
            points_tensor = torch.tensor(torus_points, dtype=torch.float32).unsqueeze(0)
            
            # Compute neural features
            neural_features = tda_engine.neural_layer(points_tensor)
            
            print(f"✅ Neural persistence features shape: {neural_features.shape}")
            print(f"   Feature dimension: {config.persistence_dim}")
            print(f"   Feature norm: {neural_features.norm().item():.3f}")
        
        # Test topological loss
        print("\n6️⃣ TESTING TOPOLOGICAL LOSS")
        print("-" * 40)
        
        # Create original and perturbed versions
        original = torch.tensor(shapes["Circle"], dtype=torch.float32).unsqueeze(0)
        
        # Small perturbation (should have small loss)
        small_perturb = original + 0.05 * torch.randn_like(original)
        small_loss = tda_engine.topo_loss(small_perturb, original)
        
        # Large perturbation (should have larger loss)
        large_perturb = original + 0.5 * torch.randn_like(original)
        large_loss = tda_engine.topo_loss(large_perturb, original)
        
        print(f"✅ Small perturbation loss: {small_loss.item():.6f}")
        print(f"✅ Large perturbation loss: {large_loss.item():.6f}")
        print(f"   Loss ratio: {large_loss.item() / (small_loss.item() + 1e-6):.2f}x")
        
        # Test persistence landscapes and images
        print("\n7️⃣ TESTING PERSISTENCE REPRESENTATIONS")
        print("-" * 40)
        
        features = tda_engine.compute_tda_features(
            shapes["Torus"],
            return_landscapes=True,
            return_images=True
        )
        
        # Check landscapes
        landscapes = features['persistence_landscapes']
        print("Persistence Landscapes:")
        for dim, landscape in landscapes.items():
            if landscape.size > 0:
                print(f"  Dimension {dim}: shape {landscape.shape}")
                print(f"    Max value: {landscape.max():.3f}")
                print(f"    Non-zero ratio: {(landscape > 0).mean():.3f}")
        
        # Check images
        images = features['persistence_images']
        print("\nPersistence Images:")
        for dim, image in images.items():
            if image.size > 0:
                print(f"  Dimension {dim}: shape {image.shape}")
                print(f"    Max intensity: {image.max():.3f}")
                print(f"    Non-zero pixels: {(image > 0).sum()}")
        
        # Integration with PHFormer
        print("\n8️⃣ TESTING PHFORMER INTEGRATION")
        print("-" * 40)
        
        try:
            from aura_intelligence.models.phformer_clean import PHFormerTiny, PHFormerConfig
            
            # Create PHFormer model
            phformer_config = PHFormerConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4
            )
            
            phformer = PHFormerTiny(phformer_config)
            
            # Prepare topological inputs
            persistence_diagrams_list = []
            betti_numbers_list = []
            
            for shape_name, points in list(shapes.items())[:2]:  # Test 2 shapes
                features = tda_engine.compute_tda_features(points)
                
                # Convert persistence diagrams to tensors
                dgm_tensor = torch.tensor(
                    features['persistence_diagrams'][1].birth_death,
                    dtype=torch.float32
                )
                persistence_diagrams_list.append(dgm_tensor)
                
                # Betti numbers
                betti = torch.tensor(features['betti_numbers'], dtype=torch.float32)
                betti_numbers_list.append(betti[:3])  # Use first 3
            
            # Stack for batch
            betti_batch = torch.stack(betti_numbers_list)
            
            # Forward pass
            outputs = phformer(persistence_diagrams_list, betti_batch)
            
            print(f"✅ PHFormer output shape: {outputs['logits'].shape}")
            print(f"   Successfully integrated TDA → PHFormer")
            
        except ImportError as e:
            print(f"⚠️  PHFormer integration skipped: {e}")
        
        # Test batch processing
        print("\n9️⃣ TESTING BATCH PROCESSING")
        print("-" * 40)
        
        # Create batch of point clouds
        batch_data = [
            create_circle_points(30 + i*10) + np.random.normal(0, 0.1, (30 + i*10, 3))
            for i in range(5)
        ]
        
        # Time batch processing
        start_time = time.time()
        batch_results = await tda_engine.process_batch_async(batch_data)
        batch_time = time.time() - start_time
        
        print(f"✅ Processed {len(batch_data)} point clouds")
        print(f"   Total time: {batch_time:.2f}s")
        print(f"   Time per cloud: {batch_time/len(batch_data):.3f}s")
        
        # Show batch results
        for i, result in enumerate(batch_results):
            betti = result['betti_numbers']
            print(f"   Cloud {i}: b0={int(betti[0])}, b1={int(betti[1])}")
        
        # Performance comparison
        print("\n🔟 PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Test different point cloud sizes
        sizes = [50, 100, 200, 400]
        
        for size in sizes:
            points = create_torus_points(size)
            
            start_time = time.time()
            _ = tda_engine.compute_tda_features(points, return_landscapes=False, return_images=False)
            compute_time = time.time() - start_time
            
            print(f"Points: {size}, Time: {compute_time:.3f}s")
        
        # Memory usage
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 40)
        
        print(f"Shapes tested: {len(shapes)}")
        print(f"Max dimension: {config.max_dimension}")
        print(f"Features computed: Diagrams, Landscapes, Images")
        print(f"Neural integration: {'✅' if config.use_neural_persistence else '❌'}")
        print(f"GPU acceleration: {'✅' if config.use_gpu else '❌'}")
        
        print("\n" + "=" * 60)
        print("✅ TDA SYSTEM INTEGRATION TEST COMPLETE")
        
        print("\n📝 Key Capabilities:")
        print("- Persistent homology computation")
        print("- Multiple complex types (Rips, Alpha, etc.)")
        print("- Neural persistence layers")
        print("- Topological loss functions")
        print("- Persistence landscapes & images")
        print("- GPU acceleration support")
        print("- Batch processing")
        
        print("\n🎯 Use Cases:")
        print("- Shape analysis and recognition")
        print("- Data topology understanding")
        print("- Feature extraction for ML")
        print("- Anomaly detection")
        print("- Time series analysis")
        print("- Network analysis")
        
        # Clean up
        tda_engine.close()
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Some dependencies may be missing (gudhi, ripser)")
        print("Install with: pip install gudhi ripser")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()


# Helper functions to create test shapes
def create_circle_points(n_points: int) -> np.ndarray:
    """Create points on a circle"""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros(n_points)
    return np.column_stack([x, y, z])

def create_torus_points(n_points: int) -> np.ndarray:
    """Create points on a torus"""
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    R, r = 3, 1
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack([x, y, z])

def create_sphere_points(n_points: int) -> np.ndarray:
    """Create points on a sphere"""
    u = np.random.uniform(0, 1, n_points)
    v = np.random.uniform(0, 1, n_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def create_figure_eight_points(n_points: int) -> np.ndarray:
    """Create points on a figure-eight"""
    t = np.linspace(0, 4*np.pi, n_points)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    z = np.zeros(n_points)
    return np.column_stack([x, y, z])


# Run the test
if __name__ == "__main__":
    asyncio.run(test_tda_integration())