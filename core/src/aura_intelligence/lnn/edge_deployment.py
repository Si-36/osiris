"""
Edge Deployment for LNN - Ultra-Low Power Inference
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import time

try:
    from liquid_s4 import LiquidS4
    LIQUID_S4_AVAILABLE = True
except ImportError:
    LIQUID_S4_AVAILABLE = False

class EdgeLNNProcessor:
    def __init__(self, model_size='nano', power_budget_mw=50):
        self.power_budget = power_budget_mw
        self.model_size = model_size
        
        if LIQUID_S4_AVAILABLE:
            self.model = LiquidS4(
                d_model=64 if model_size == 'nano' else 128,
                d_state=32 if model_size == 'nano' else 64,
                power_budget=power_budget_mw
            )
            self.model_type = "LiquidS4-Edge"
        else:
            # Ultra-lightweight fallback
            self.model = nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 8)
            )
            self.model_type = "Lightweight-Fallback"
        
        self.model.eval()
        
        # Quantize for edge deployment
        if hasattr(torch.quantization, 'quantize_dynamic'):
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
    
    def edge_inference(self, context_data: np.ndarray) -> Dict[str, Any]:
        """Ultra-low power inference for edge devices"""
        start_time = time.perf_counter()
        
        # Prepare input
        if len(context_data.shape) == 1:
            input_tensor = torch.tensor(context_data, dtype=torch.float32)
        else:
            input_tensor = torch.tensor(context_data.flatten(), dtype=torch.float32)
        
        # Pad or truncate to expected size
        if len(input_tensor) != 64:
            if len(input_tensor) > 64:
                input_tensor = input_tensor[:64]
            else:
                padding = torch.zeros(64 - len(input_tensor))
                input_tensor = torch.cat([input_tensor, padding])
        
        input_tensor = input_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            if LIQUID_S4_AVAILABLE:
                output = self.model(input_tensor.unsqueeze(1))
                if isinstance(output, tuple):
                    output = output[0]
            else:
                output = self.model(input_tensor)
        
        # Process output
        output_np = output.squeeze().numpy()
        decision_score = float(np.mean(output_np))
        
        # Simple decision logic
        if decision_score > 0.1:
            decision = "approve"
            confidence = min(decision_score + 0.5, 1.0)
        elif decision_score < -0.1:
            decision = "reject"
            confidence = min(abs(decision_score) + 0.5, 1.0)
        else:
            decision = "neutral"
            confidence = 0.5
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Estimate power consumption
        estimated_power = self._estimate_power_consumption(inference_time)
        
        return {
            'decision': decision,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'estimated_power_mw': estimated_power,
            'model_type': self.model_type,
            'output_raw': output_np.tolist()[:8]  # First 8 values
        }
    
    def _estimate_power_consumption(self, inference_time_ms: float) -> float:
        """Estimate power consumption based on inference time"""
        # Simple power model: base power + computation power
        base_power = 10.0  # mW baseline
        compute_power = (inference_time_ms / 1000.0) * 30.0  # 30mW per second of compute
        
        total_power = base_power + compute_power
        return min(total_power, self.power_budget)
    
    def batch_inference(self, batch_data: np.ndarray, batch_size: int = 4) -> Dict[str, Any]:
        """Batch inference for improved efficiency"""
        
        if len(batch_data.shape) == 1:
            batch_data = batch_data.reshape(1, -1)
        
        n_samples = batch_data.shape[0]
        results = []
        
        start_time = time.perf_counter()
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_slice = batch_data[i:batch_end]
            
            batch_results = []
            for sample in batch_slice:
                result = self.edge_inference(sample)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        total_time = (time.perf_counter() - start_time) * 1000
        avg_time_per_sample = total_time / n_samples
        
        # Aggregate results
        decisions = [r['decision'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        return {
            'batch_results': results,
            'batch_size': n_samples,
            'total_time_ms': total_time,
            'avg_time_per_sample_ms': avg_time_per_sample,
            'decisions': decisions,
            'avg_confidence': np.mean(confidences),
            'power_efficiency': self.power_budget / avg_time_per_sample if avg_time_per_sample > 0 else 0
        }
    
    def get_edge_specs(self) -> Dict[str, Any]:
        """Get edge deployment specifications"""
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate memory usage (rough)
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
        
        return {
            'model_type': self.model_type,
            'model_size': self.model_size,
            'power_budget_mw': self.power_budget,
            'total_parameters': total_params,
            'memory_usage_mb': memory_mb,
            'quantized': 'qint8' in str(type(self.model)),
            'liquid_s4_available': LIQUID_S4_AVAILABLE,
            'target_platforms': ['ARM Cortex-M', 'RISC-V', 'Edge TPU', 'Mobile GPU']
        }

def get_edge_lnn_processor(model_size='nano', power_budget_mw=50):
    return EdgeLNNProcessor(model_size, power_budget_mw)