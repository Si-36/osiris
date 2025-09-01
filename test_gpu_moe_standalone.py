"""
⚡ Standalone GPU MoE Test
=========================

Tests GPU-optimized MoE without problematic imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import numpy as np

# Add only the MoE path
sys.path.insert(0, '/workspace/core/src/aura_intelligence/moe')

from gpu_optimized_moe import (
    create_gpu_moe, GPUMoEConfig, StreamingGPUMoE,
    GPUOptimizedMoE, TRITON_AVAILABLE
)


def test_basic_functionality():
    """Test basic GPU MoE functionality"""
    print("\n🔬 Testing Basic MoE Functionality")
    print("-" * 50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    # Disable Triton on CPU
    use_triton = TRITON_AVAILABLE and device.type == 'cuda'
    
    model = create_gpu_moe(
        d_model=256,
        num_experts=16,
        use_triton=use_triton
    )
    
    print(f"\n✅ Model created successfully")
    print(f"   Experts: {model.config.num_experts}")
    print(f"   Model dimension: {model.config.d_model}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 256).to(device)
    
    output, info = model(x)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Input shape: {tuple(x.shape)}")
    print(f"   Output shape: {tuple(output.shape)}")
    print(f"   Active experts: {info['active_experts'].item()}")
    print(f"   Load balance loss: {info['load_balance_loss'].item():.4f}")
    
    return model, device


def benchmark_performance(model, device):
    """Benchmark MoE performance"""
    print("\n\n⚡ Performance Benchmark")
    print("-" * 50)
    
    # Test configurations
    configs = [
        (1, 128, "Single sample"),
        (8, 256, "Small batch"),
        (16, 512, "Medium batch"),
        (32, 256, "Large batch"),
        (64, 128, "Very large batch"),
    ]
    
    results = []
    
    for batch_size, seq_len, desc in configs:
        print(f"\n📊 {desc}")
        print(f"   Batch: {batch_size}, Sequence: {seq_len}")
        
        # Create input
        x = torch.randn(batch_size, seq_len, model.config.d_model).to(device)
        
        # Add complexity signal
        complexity = torch.randn(batch_size, 1).to(device)
        
        # Warmup
        for _ in range(5):
            _, _ = model(x, complexity)
        
        # Time MoE
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            output, info = model(x, complexity)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        moe_time = (time.time() - start) / 20
        
        # Compare with single FFN baseline
        ffn = nn.Sequential(
            nn.Linear(model.config.d_model, model.config.d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model.config.d_model * 4, model.config.d_model)
        ).to(device)
        
        # Time baseline
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            _ = ffn(x)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        baseline_time = (time.time() - start) / 20
        
        # Calculate metrics
        speedup = baseline_time / moe_time
        total_tokens = batch_size * seq_len
        throughput = total_tokens / moe_time
        
        print(f"   MoE time: {moe_time*1000:.2f}ms")
        print(f"   Baseline time: {baseline_time*1000:.2f}ms")
        print(f"   Speedup vs dense: {speedup:.2f}x")
        print(f"   Throughput: {throughput:.0f} tokens/sec")
        print(f"   Active experts: {info['active_experts'].item()}")
        
        results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'moe_ms': moe_time * 1000,
            'baseline_ms': baseline_time * 1000,
            'speedup': speedup,
            'throughput': throughput
        })
    
    return results


def test_routing_efficiency():
    """Test routing efficiency"""
    print("\n\n🎯 Routing Efficiency Test")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with different expert counts
    expert_counts = [4, 8, 16, 32]
    batch_size = 16
    seq_len = 256
    
    for num_experts in expert_counts:
        model = GPUOptimizedMoE(GPUMoEConfig(
            d_model=256,
            num_experts=num_experts,
            use_triton=False
        )).to(device)
        
        x = torch.randn(batch_size, seq_len, 256).to(device)
        
        # Measure routing overhead
        gates, indices = model.router(x)
        
        # Check distribution
        unique_experts = torch.unique(indices).numel()
        utilization = unique_experts / num_experts
        
        print(f"\n{num_experts} experts:")
        print(f"   Active: {unique_experts}")
        print(f"   Utilization: {utilization:.1%}")
        print(f"   Avg gate value: {gates.mean().item():.3f}")


def test_sparse_computation():
    """Test sparse vs dense computation"""
    print("\n\n🔍 Sparse Computation Test")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_triton = TRITON_AVAILABLE and device.type == 'cuda'
    model = create_gpu_moe(d_model=256, num_experts=32, use_triton=use_triton).to(device)
    
    batch_size = 32
    seq_len = 256
    x = torch.randn(batch_size, seq_len, 256).to(device)
    
    # Test sparse mode
    model.config.sparse_compute = True
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        _, info_sparse = model(x)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    sparse_time = (time.time() - start) / 10
    
    # Test dense mode  
    model.config.sparse_compute = False
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        _, info_dense = model(x)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    dense_time = (time.time() - start) / 10
    
    print(f"Sparse mode: {sparse_time*1000:.2f}ms")
    print(f"Dense mode: {dense_time*1000:.2f}ms")
    print(f"Sparse speedup: {dense_time/sparse_time:.2f}x")
    print(f"Active experts: {info_sparse['active_experts'].item()}/{model.config.num_experts}")


def test_streaming_mode():
    """Test streaming inference"""
    print("\n\n🌊 Streaming Mode Test")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StreamingGPUMoE(GPUMoEConfig(
        d_model=256,
        num_experts=16,
        use_triton=False
    )).to(device)
    
    batch_size = 32
    steps = 100
    
    # Stable complexity for caching test
    complexity = torch.randn(batch_size, 1).to(device)
    
    times = []
    
    for step in range(steps):
        x = torch.randn(batch_size, 256).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        output = model.step(x, complexity)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        times.append(time.time() - start)
    
    # Skip warmup
    avg_time = np.mean(times[10:]) * 1000
    std_time = np.std(times[10:]) * 1000
    
    print(f"Average step time: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"Throughput: {batch_size/np.mean(times[10:]):.0f} samples/sec")
    print(f"First step: {times[0]*1000:.2f}ms (includes routing)")
    print(f"Cached steps: {times[-1]*1000:.2f}ms")


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║        ⚡ GPU MoE PERFORMANCE TEST ⚡                   ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    model, device = test_basic_functionality()
    
    if model is not None:
        results = benchmark_performance(model, device)
        
        # Other tests
        test_routing_efficiency()
        test_sparse_computation()
        test_streaming_mode()
        
        # Summary
        print("\n\n📊 Performance Summary")
        print("=" * 60)
        
        if results:
            avg_throughput = np.mean([r['throughput'] for r in results])
            max_throughput = max(r['throughput'] for r in results)
            avg_speedup = np.mean([r['speedup'] for r in results])
            
            print(f"Average Throughput: {avg_throughput:.0f} tokens/sec")
            print(f"Maximum Throughput: {max_throughput:.0f} tokens/sec")
            print(f"Average Speedup vs Dense: {avg_speedup:.2f}x")
            
            print("\n🏆 Best Configuration:")
            best = max(results, key=lambda x: x['throughput'])
            print(f"   Batch: {best['batch_size']}, Seq: {best['seq_len']}")
            print(f"   Throughput: {best['throughput']:.0f} tokens/sec")
            print(f"   Time: {best['moe_ms']:.2f}ms")
        
        # Memory usage
        if device.type == 'cuda':
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()