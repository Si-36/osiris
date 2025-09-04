"""
⚡ Standalone GPU LNN Test
=========================

Tests GPU-optimized LNN without problematic imports.
"""

import torch
import torch.nn as nn
import time
import sys

# Add only the LNN path
sys.path.insert(0, '/workspace/core/src/aura_intelligence/lnn')

from gpu_optimized_lnn import (
    create_gpu_lnn, GPULNNConfig, StreamingGPULNN,
    GPUOptimizedLNN, cfc_dynamics_fused
)


def test_basic_functionality():
    """Test basic GPU LNN functionality"""
    print("\n🔬 Testing Basic Functionality")
    print("-" * 50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = create_gpu_lnn(
        input_size=128,
        output_size=64,
        hidden_size=256,
        num_layers=3,
        use_jit=True
    ).to(device)
    
    print(f"\n✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 128).to(device)
    
    output = model(x)
    print(f"\n✅ Forward pass successful")
    print(f"   Input shape: {tuple(x.shape)}")
    print(f"   Output shape: {tuple(output.shape)}")
    
    # Test with info
    output, info = model(x, return_info=True)
    print(f"\n✅ Info extraction successful")
    print(f"   Complexity: {info['complexity'].mean().item():.3f}")
    print(f"   Tau bands: {info['tau_bands'].tolist()}")
    
    return model, device


def benchmark_performance(model, device):
    """Benchmark GPU vs CPU performance"""
    print("\n\n⚡ Performance Benchmark")
    print("-" * 50)
    
    # Test configurations
    configs = [
        (1, 10, "Single sample, short sequence"),
        (16, 50, "Small batch, medium sequence"),
        (32, 100, "Medium batch, long sequence"),
        (64, 50, "Large batch, medium sequence"),
        (128, 20, "Very large batch, short sequence"),
    ]
    
    results = []
    
    for batch_size, seq_len, desc in configs:
        print(f"\n📊 {desc}")
        print(f"   Batch: {batch_size}, Sequence: {seq_len}")
        
        # Create input
        x = torch.randn(batch_size, seq_len, model.input_size).to(device)
        
        # Warmup
        for _ in range(5):
            _ = model(x)
        
        # Time GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            _ = model(x)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        gpu_time = (time.time() - start) / 20
        
        # Time CPU (if on GPU)
        if device.type == 'cuda':
            # Create CPU model
            cpu_model = GPUOptimizedLNN(
                model.input_size,
                model.output_size,
                model.config
            ).cpu()
            x_cpu = x.cpu()
            
            # Warmup
            for _ in range(5):
                _ = cpu_model(x_cpu)
            
            start = time.time()
            for _ in range(10):  # Fewer runs for CPU
                _ = cpu_model(x_cpu)
            cpu_time = (time.time() - start) / 10
            
            speedup = cpu_time / gpu_time
            print(f"   GPU: {gpu_time*1000:.2f}ms")
            print(f"   CPU: {cpu_time*1000:.2f}ms")
            print(f"   Speedup: {speedup:.1f}x 🚀")
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'gpu_ms': gpu_time * 1000,
                'cpu_ms': cpu_time * 1000,
                'speedup': speedup
            })
        else:
            print(f"   Time: {gpu_time*1000:.2f}ms")
    
    return results


def test_streaming_mode():
    """Test streaming inference mode"""
    print("\n\n🌊 Testing Streaming Mode")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create streaming model
    model = StreamingGPULNN(
        input_size=128,
        output_size=64,
        hidden_size=256,
        num_layers=2
    ).to(device)
    
    print("✅ Streaming model created")
    
    # Test streaming
    batch_size = 32
    steps = 100
    
    # Reset state
    model.reset_state()
    
    # Warmup
    x = torch.randn(batch_size, 128).to(device)
    for _ in range(10):
        _ = model.step(x)
    
    # Time streaming
    outputs = []
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(steps):
        x = torch.randn(batch_size, 128).to(device)
        output = model.step(x)
        outputs.append(output)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start
    
    print(f"\n📈 Streaming Performance:")
    print(f"   Steps: {steps}")
    print(f"   Total time: {total_time*1000:.2f}ms")
    print(f"   Per step: {total_time/steps*1000:.2f}ms")
    print(f"   Throughput: {batch_size*steps/total_time:.0f} samples/sec")
    
    # Verify output consistency
    output_tensor = torch.stack(outputs)
    print(f"\n✅ Output shape: {tuple(output_tensor.shape)}")
    print(f"   Mean: {output_tensor.mean().item():.3f}")
    print(f"   Std: {output_tensor.std().item():.3f}")


def test_jit_compilation():
    """Test JIT compilation benefits"""
    print("\n\n🔧 Testing JIT Compilation")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models with and without JIT
    config_jit = GPULNNConfig(use_jit=True)
    config_no_jit = GPULNNConfig(use_jit=False)
    
    model_jit = GPUOptimizedLNN(128, 64, config_jit).to(device)
    model_no_jit = GPUOptimizedLNN(128, 64, config_no_jit).to(device)
    
    # Test input
    x = torch.randn(32, 50, 128).to(device)
    
    # Test JIT-compiled dynamics
    print("Testing fused CfC dynamics...")
    
    hidden = torch.randn(32, 256).to(device)
    W_rec = torch.randn(256, 256).to(device)
    W_in = torch.randn(128, 256).to(device)
    bias = torch.randn(256).to(device)
    tau = torch.ones(256).to(device) * 0.1
    
    # Time fused operation
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        result = cfc_dynamics_fused(
            x[:, 0], hidden, W_rec, W_in, bias, tau, 0.05
        )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    fused_time = (time.time() - start) / 100
    
    print(f"✅ Fused dynamics time: {fused_time*1000:.2f}ms")
    print(f"   Throughput: {32/fused_time:.0f} samples/sec")


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         ⚡ GPU-OPTIMIZED LNN TEST SUITE ⚡             ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    model, device = test_basic_functionality()
    
    if device.type == 'cuda':
        results = benchmark_performance(model, device)
        
        # Summary
        print("\n\n📊 Performance Summary")
        print("=" * 50)
        
        if results:
            avg_speedup = sum(r['speedup'] for r in results) / len(results)
            max_speedup = max(r['speedup'] for r in results)
            min_speedup = min(r['speedup'] for r in results)
            
            print(f"Average Speedup: {avg_speedup:.1f}x")
            print(f"Maximum Speedup: {max_speedup:.1f}x")
            print(f"Minimum Speedup: {min_speedup:.1f}x")
            
            # Best configuration
            best = max(results, key=lambda x: x['speedup'])
            print(f"\nBest Configuration:")
            print(f"   Batch: {best['batch_size']}, Seq: {best['seq_len']}")
            print(f"   Speedup: {best['speedup']:.1f}x 🏆")
    
    test_streaming_mode()
    test_jit_compilation()
    
    # Memory usage
    if device.type == 'cuda':
        print("\n\n💾 GPU Memory Usage")
        print("-" * 50)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    
    print("\n\n✅ All tests completed successfully!")
    
    # Save results
    if device.type == 'cuda' and results:
        import json
        with open('/workspace/lnn_gpu_results.json', 'w') as f:
            json.dump({
                'device': torch.cuda.get_device_name(0),
                'results': results,
                'avg_speedup': avg_speedup,
                'max_speedup': max_speedup
            }, f, indent=2)
        print("\n📄 Results saved to lnn_gpu_results.json")


if __name__ == "__main__":
    main()