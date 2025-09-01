"""
⚡ Test GPU-Optimized MoE Performance
====================================

Comprehensive benchmarks for GPU-accelerated Mixture of Experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import sys

# Add paths
sys.path.insert(0, '/workspace/core/src')


def test_basic_moe():
    """Test basic MoE functionality"""
    print("\n🔬 Testing Basic MoE Functionality")
    print("-" * 50)
    
    try:
        from aura_intelligence.moe.gpu_optimized_moe import (
            create_gpu_moe, GPUMoEConfig, StreamingGPUMoE,
            TRITON_AVAILABLE
        )
        print("✅ GPU MoE module imported successfully")
        print(f"   Triton available: {TRITON_AVAILABLE}")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return None
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = create_gpu_moe(
        d_model=256,
        num_experts=16,
        use_triton=TRITON_AVAILABLE
    )
    
    print(f"\n✅ Model created successfully")
    print(f"   Experts: {model.config.num_experts}")
    print(f"   Model dimension: {model.config.d_model}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, model.config.d_model).to(device)
    
    output, info = model(x)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Input shape: {tuple(x.shape)}")
    print(f"   Output shape: {tuple(output.shape)}")
    print(f"   Active experts: {info['active_experts'].item()}")
    print(f"   Load balance loss: {info['load_balance_loss'].item():.4f}")
    
    return model, device


def benchmark_routing_performance(model, device):
    """Benchmark routing performance"""
    print("\n\n⚡ Routing Performance Benchmark")
    print("-" * 50)
    
    configs = [
        (1, 128, "Single sample"),
        (16, 256, "Small batch"),
        (32, 512, "Medium batch"),
        (64, 256, "Large batch"),
        (128, 128, "Very large batch"),
    ]
    
    results = []
    
    for batch_size, seq_len, desc in configs:
        print(f"\n📊 {desc}: Batch={batch_size}, Seq={seq_len}")
        
        # Create input
        x = torch.randn(batch_size, seq_len, model.config.d_model).to(device)
        
        # Warmup
        for _ in range(5):
            _, _ = model(x)
        
        # Time routing only
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Full forward pass timing
        start = time.time()
        for _ in range(20):
            output, info = model(x)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        total_time = (time.time() - start) / 20
        
        # Calculate metrics
        total_tokens = batch_size * seq_len
        throughput = total_tokens / total_time
        time_per_token = total_time / total_tokens * 1000  # ms
        
        print(f"   Total time: {total_time*1000:.2f}ms")
        print(f"   Time per token: {time_per_token:.3f}ms")
        print(f"   Throughput: {throughput:.0f} tokens/sec")
        print(f"   Active experts: {info['active_experts'].item()}/{model.config.num_experts}")
        
        results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_time_ms': total_time * 1000,
            'time_per_token_ms': time_per_token,
            'throughput': throughput,
            'active_experts': info['active_experts'].item()
        })
    
    return results


def test_sparse_vs_dense(model, device):
    """Compare sparse vs dense computation"""
    print("\n\n🔍 Sparse vs Dense Computation")
    print("-" * 50)
    
    # Test configuration
    batch_size = 32
    seq_len = 256
    x = torch.randn(batch_size, seq_len, model.config.d_model).to(device)
    
    # Test sparse mode
    model.config.sparse_compute = True
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        _, _ = model(x)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    sparse_time = (time.time() - start) / 10
    
    # Test dense mode
    model.config.sparse_compute = False
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        _, _ = model(x)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    dense_time = (time.time() - start) / 10
    
    speedup = dense_time / sparse_time
    
    print(f"Sparse compute time: {sparse_time*1000:.2f}ms")
    print(f"Dense compute time: {dense_time*1000:.2f}ms")
    print(f"Sparse speedup: {speedup:.2f}x")
    
    # Reset to sparse
    model.config.sparse_compute = True
    
    return speedup


def test_expert_utilization():
    """Test expert utilization patterns"""
    print("\n\n📊 Expert Utilization Analysis")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with fewer experts for visualization
    model = create_gpu_moe(d_model=256, num_experts=8).to(device)
    
    # Generate diverse inputs
    batch_size = 64
    seq_len = 128
    
    # Different complexity patterns
    patterns = {
        'uniform': torch.randn(batch_size, seq_len, 256),
        'simple': torch.randn(batch_size, seq_len, 256) * 0.1,
        'complex': torch.randn(batch_size, seq_len, 256) * 10,
        'mixed': torch.cat([
            torch.randn(batch_size//2, seq_len, 256) * 0.1,
            torch.randn(batch_size//2, seq_len, 256) * 10
        ], dim=0)
    }
    
    utilization = {}
    
    for name, x in patterns.items():
        x = x.to(device)
        _, info = model(x)
        
        expert_counts = info['expert_counts'].cpu().numpy()
        utilization[name] = expert_counts / expert_counts.sum()
        
        print(f"\n{name.capitalize()} pattern:")
        print(f"   Expert distribution: {utilization[name]}")
        print(f"   Entropy: {-np.sum(utilization[name] * np.log(utilization[name] + 1e-8)):.3f}")
    
    return utilization


def test_streaming_mode():
    """Test streaming inference mode"""
    print("\n\n🌊 Testing Streaming Mode")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create streaming model
    from aura_intelligence.moe.gpu_optimized_moe import StreamingGPUMoE, GPUMoEConfig
    
    config = GPUMoEConfig(d_model=256, num_experts=16)
    model = StreamingGPUMoE(config).to(device)
    
    print("✅ Streaming model created")
    
    # Test streaming
    batch_size = 32
    steps = 100
    
    # Generate complexity signal that changes slowly
    complexity_base = torch.randn(batch_size, 1).to(device)
    
    outputs = []
    times = []
    
    for step in range(steps):
        # Slowly varying complexity
        complexity = complexity_base + 0.01 * step * torch.randn(batch_size, 1).to(device)
        x = torch.randn(batch_size, 256).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        output = model.step(x, complexity)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        times.append(time.time() - start)
        outputs.append(output)
    
    avg_time = np.mean(times[10:]) * 1000  # Skip warmup
    throughput = batch_size / np.mean(times[10:])
    
    print(f"\n📈 Streaming Performance:")
    print(f"   Average step time: {avg_time:.2f}ms")
    print(f"   Throughput: {throughput:.0f} samples/sec")
    print(f"   Latency variance: {np.std(times[10:])*1000:.2f}ms")


def compare_num_experts():
    """Compare performance with different number of experts"""
    print("\n\n🔢 Expert Scaling Analysis")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    expert_counts = [4, 8, 16, 32, 64]
    batch_size = 16
    seq_len = 256
    d_model = 256
    
    results = []
    
    for num_experts in expert_counts:
        print(f"\n📊 Testing with {num_experts} experts")
        
        # Create model
        model = create_gpu_moe(
            d_model=d_model,
            num_experts=num_experts
        ).to(device)
        
        # Test input
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # Warmup
        for _ in range(5):
            _, _ = model(x)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            output, info = model(x)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        time_taken = (time.time() - start) / 20
        
        results.append({
            'num_experts': num_experts,
            'time_ms': time_taken * 1000,
            'throughput': batch_size * seq_len / time_taken,
            'active_experts': info['active_experts'].item()
        })
        
        print(f"   Time: {time_taken*1000:.2f}ms")
        print(f"   Active experts: {info['active_experts'].item()}")
    
    # Show scaling
    print("\n📈 Scaling Summary:")
    for r in results:
        print(f"   {r['num_experts']:3d} experts: {r['time_ms']:6.2f}ms, {r['throughput']:6.0f} tok/s")
    
    return results


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         ⚡ GPU-OPTIMIZED MoE TEST SUITE ⚡             ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    model, device = test_basic_moe()
    
    if model is not None:
        # Performance benchmarks
        routing_results = benchmark_routing_performance(model, device)
        
        # Sparse vs dense
        sparse_speedup = test_sparse_vs_dense(model, device)
        
        # Expert utilization
        utilization = test_expert_utilization()
        
        # Streaming mode
        test_streaming_mode()
        
        # Scaling analysis
        scaling_results = compare_num_experts()
        
        # Summary
        print("\n\n📊 Performance Summary")
        print("=" * 60)
        
        if routing_results:
            avg_throughput = np.mean([r['throughput'] for r in routing_results])
            max_throughput = max(r['throughput'] for r in routing_results)
            
            print(f"Average Throughput: {avg_throughput:.0f} tokens/sec")
            print(f"Maximum Throughput: {max_throughput:.0f} tokens/sec")
            print(f"Sparse Speedup: {sparse_speedup:.2f}x")
            
            print("\n🏆 Best Configuration:")
            best = max(routing_results, key=lambda x: x['throughput'])
            print(f"   Batch: {best['batch_size']}, Seq: {best['seq_len']}")
            print(f"   Throughput: {best['throughput']:.0f} tokens/sec")
        
        # Memory usage
        if device.type == 'cuda':
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    
    print("\n✅ GPU MoE test complete!")


if __name__ == "__main__":
    main()