"""
MAX Engine Configuration
"""

from max.driver import Device

class MAXConfig:
    """MAX Engine configuration for optimal performance"""
    
    # Device selection (automatic GPU detection)
    device = Device.cpu()
    
    # Model paths (your AURA models)
    models = {
        "neural": "models/aura_neural.max",
        "tda": "models/aura_tda.max",
        "consciousness": "models/aura_consciousness.max",
        "memory": "models/aura_memory.max"
    }
    
    # Performance settings
    batch_size = 32
    num_threads = 8
    enable_graph_optimization = True
    enable_kernel_fusion = True
