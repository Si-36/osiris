"""
MAX Model Manager
"""

from typing import Dict, Any
import numpy as np

from max import engine
from max.driver import Tensor
from max.graph import Graph, Type, TensorType, DeviceRef
from max.graph import ops
from max.dtype import DType

from ultimate_api_system.max_config import MAXConfig

class MAXModelManager:
    """Professional MAX model management with hot-loading and caching"""
    
    def __init__(self):
        self.session = engine.InferenceSession()
        self.models: Dict[str, Any] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all MAX models with optimizations"""
        for name, path in MAXConfig.models.items():
            try:
                # Load model with MAX optimizations
                self.models[name] = self.session.load(path)
                print(f"✅ Loaded MAX model: {name} on {MAXConfig.device}")
            except Exception as e:
                print(f"⚠️  Model {name} not found, using dynamic compilation")
                # We'll compile on-demand if model doesn't exist
    
    async def execute(
        self, 
        model_name: str, 
        input_data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Execute model with MAX acceleration"""
        
        if model_name not in self.models:
            # Dynamic model compilation if not pre-loaded
            self.models[model_name] = await self._compile_model(model_name)
        
        model = self.models[model_name]
        
        # Convert numpy array to MAX Tensor
        max_tensor = Tensor.from_numpy(input_data)
        
        # Execute with MAX optimizations - use positional arguments
        output = model.execute(max_tensor)
        
        # Convert output back to numpy if it's a list of tensors
        if isinstance(output, list) and len(output) > 0:
            return output[0].to_numpy()
        elif hasattr(output, 'to_numpy'):
            return output.to_numpy()
        else:
            return output
    
    async def _compile_model(self, model_name: str) -> Any:
        """Dynamically compile a model using MAX Graph API"""
        
        # Build graph based on model type
        if model_name == "neural":
            graph = self._build_neural_graph()
        elif model_name == "tda":
            graph = self._build_tda_graph()
        else:
            graph = self._build_default_graph()
        
        # Compile with optimizations
        return self.session.load(graph)
    
    def _build_neural_graph(self) -> Graph:
        """Build neural network graph with MAX ops"""
        with Graph("neural", input_types=(TensorType(DType.float32, ("batch", 768), DeviceRef.CPU()),)) as graph:
            # Input tensor
            input_tensor, = graph.inputs
            
            # Simple neural processing
            x = ops.relu(input_tensor)
            
            # Output
            graph.output(x)
        return graph
    
    def _build_tda_graph(self) -> Graph:
        """Build Tda processing graph with MAX ops"""
        with Graph("tda", input_types=(TensorType(DType.float32, ("batch", 3), DeviceRef.CPU()),)) as graph:
            # Input point cloud
            input_points, = graph.inputs
            
            # Simple processing for now
            output = ops.relu(input_points)
            
            graph.output(output)
        return graph
    
    def _build_default_graph(self) -> Graph:
        """Default graph for general processing"""
        with Graph("default", input_types=(TensorType(DType.float32, ("batch", "sequence", "features"), DeviceRef.CPU()),)) as graph:
            input_tensor, = graph.inputs
            
            # Simple processing pipeline - just pass through for now
            # Use a simple linear transformation
            output = ops.relu(input_tensor)
            
            graph.output(output)
        return graph
