"""
⚡ Simple GPU LNN Performance Test
==================================

Demonstrates GPU acceleration for Liquid Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
import time
import numpy as np
from typing import Optional, Tuple, Dict


# Simplified CfC dynamics (JIT-compiled)
@jit.script
def cfc_update(
    x: torch.Tensor,        # [batch, features]
    hidden: torch.Tensor,   # [batch, hidden]
    W_in: torch.Tensor,     # [features, hidden]
    W_rec: torch.Tensor,    # [hidden, hidden]
    bias: torch.Tensor,     # [hidden]
    tau: torch.Tensor,      # [hidden]
    dt: float
) -> torch.Tensor:
    """Closed-form Continuous dynamics update"""
    # Exponential decay
    alpha = torch.exp(-dt / tau)
    
    # Input transformation
    h_in = torch.matmul(x, W_in)
    h_rec = torch.matmul(hidden, W_rec)
    h_total = torch.tanh(h_in + h_rec + bias)
    
    # CfC update
    new_hidden = alpha * hidden + (1 - alpha) * h_total
    
    return new_hidden


class SimpleLiquidLayer(nn.Module):
    """Simple GPU-optimized Liquid layer"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Parameters
        self.W_in = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.1)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through liquid layer"""
        batch_size = x.shape[0]
        seq_len = x.shape[1] if x.dim() == 3 else 1
        
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Initialize hidden state
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            hidden = cfc_update(
                x[:, t], hidden, 
                self.W_in, self.W_rec, self.bias, self.tau,
                dt=0.05
            )
            outputs.append(hidden)
        
        output = torch.stack(outputs, dim=1)
        return output.squeeze(1) if seq_len == 1 else output, hidden


class SimpleGPULNN(nn.Module):
    """Simple GPU-optimized LNN for testing"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Layers
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.liquid_layers = nn.ModuleList([
            SimpleLiquidLayer(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Project input
        x = self.input_proj(x)
        
        # Process through liquid layers
        hidden = None
        for layer in self.liquid_layers:
            x, hidden = layer(x, hidden)
        
        # Output projection
        if x.dim() == 3:
            x = x[:, -1]  # Take last timestep
        
        return self.output_proj(x)


class CPULiquidLayer(nn.Module):
    """CPU baseline for comparison"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Standard RNN-like parameters
        self.W_in = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.1)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU forward pass (no JIT)"""
        batch_size = x.shape[0]
        seq_len = x.shape[1] if x.dim() == 3 else 1
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Non-JIT implementation
            alpha = torch.exp(-0.05 / self.tau)
            h_in = torch.matmul(x[:, t], self.W_in)
            h_rec = torch.matmul(hidden, self.W_rec)
            h_total = torch.tanh(h_in + h_rec + self.bias)
            hidden = alpha * hidden + (1 - alpha) * h_total
            outputs.append(hidden)
        
        output = torch.stack(outputs, dim=1)
        return output.squeeze(1) if seq_len == 1 else output, hidden


def benchmark_gpu_vs_cpu():
    """Compare GPU-optimized vs CPU implementations"""
    print("\n⚡ GPU vs CPU LNN Benchmark")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  No GPU available, running CPU-only benchmark")
    
    # Test configurations
    configs = [
        (32, 10, 128, 256, 64),   # batch, seq, input, hidden, output
        (64, 50, 128, 256, 64),
        (128, 100, 128, 256, 64),
        (256, 50, 128, 512, 128),
    ]
    
    results = []
    
    for batch_size, seq_len, input_size, hidden_size, output_size in configs:
        print(f"\n📊 Config: Batch={batch_size}, Seq={seq_len}, Hidden={hidden_size}")
        
        # Create models
        gpu_model = SimpleGPULNN(input_size, hidden_size, output_size).to(device)
        
        # Create test input
        x = torch.randn(batch_size, seq_len, input_size).to(device)
        
        # Warmup
        for _ in range(5):
            _ = gpu_model(x)
        
        # Time GPU/JIT version
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            _ = gpu_model(x)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        gpu_time = (time.time() - start) / 20
        
        print(f"   JIT Time: {gpu_time*1000:.2f}ms")
        print(f"   Throughput: {batch_size*seq_len/gpu_time:.0f} samples/sec")
        
        # Compare with non-JIT version if on CPU
        if device.type == 'cpu':
            # Replace one layer with CPU version for comparison
            cpu_model = SimpleGPULNN(input_size, hidden_size, output_size)
            cpu_model.liquid_layers[0] = CPULiquidLayer(hidden_size, hidden_size)
            
            # Warmup
            for _ in range(5):
                _ = cpu_model(x)
            
            start = time.time()
            for _ in range(20):
                _ = cpu_model(x)
            cpu_time = (time.time() - start) / 20
            
            speedup = cpu_time / gpu_time
            print(f"   CPU Time: {cpu_time*1000:.2f}ms")
            print(f"   JIT Speedup: {speedup:.2f}x")
            
            results.append({
                'config': f"B{batch_size}_S{seq_len}_H{hidden_size}",
                'jit_ms': gpu_time * 1000,
                'cpu_ms': cpu_time * 1000,
                'speedup': speedup
            })
    
    return results


def test_features():
    """Test specific GPU features"""
    print("\n\n🔧 Feature Tests")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGPULNN(64, 128, 32).to(device)
    
    # Test 1: Different batch sizes
    print("\n1. Variable batch size handling:")
    for batch_size in [1, 16, 64, 256]:
        x = torch.randn(batch_size, 10, 64).to(device)
        output = model(x)
        print(f"   Batch {batch_size}: Output shape {tuple(output.shape)} ✓")
    
    # Test 2: Different sequence lengths
    print("\n2. Variable sequence length handling:")
    for seq_len in [1, 10, 50, 100]:
        x = torch.randn(32, seq_len, 64).to(device)
        output = model(x)
        print(f"   Seq {seq_len}: Output shape {tuple(output.shape)} ✓")
    
    # Test 3: Gradient flow
    print("\n3. Gradient flow test:")
    x = torch.randn(16, 20, 64, requires_grad=True).to(device)
    output = model(x)
    loss = output.sum()
    loss.backward()
    print(f"   Input grad norm: {x.grad.norm().item():.3f} ✓")
    
    # Test 4: Memory efficiency
    if device.type == 'cuda':
        print("\n4. GPU Memory usage:")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Large batch
        x = torch.randn(256, 100, 64).to(device)
        _ = model(x)
        
        print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        print(f"   Current memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║        ⚡ SIMPLE GPU LNN BENCHMARK ⚡                   ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Run benchmarks
    results = benchmark_gpu_vs_cpu()
    
    # Run feature tests
    test_features()
    
    # Summary
    if results:
        print("\n\n📊 Performance Summary")
        print("=" * 60)
        
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_speedup = max(r['speedup'] for r in results)
        
        print(f"Average JIT Speedup: {avg_speedup:.2f}x")
        print(f"Maximum JIT Speedup: {max_speedup:.2f}x")
        
        print("\n🏆 Best Configuration:")
        best = max(results, key=lambda x: x['speedup'])
        print(f"   {best['config']}: {best['speedup']:.2f}x speedup")
    
    print("\n✅ GPU LNN test complete!")


if __name__ == "__main__":
    main()