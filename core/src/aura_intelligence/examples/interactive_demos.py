"""
Interactive AI Demos - 2025 Implementation

Based on latest research:
- Interactive Jupyter-style demonstrations
- Real-time visualization with Gradio/Streamlit
- Live model inference showcases
- Component integration examples
- Performance benchmarking demos

Key features:
- Web-based interactive UI
- Real-time model responses
- Visualization of internal states
- A/B testing capabilities
- Multi-modal demonstrations
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import structlog
from pathlib import Path
import time

logger = structlog.get_logger(__name__)


@dataclass
class DemoConfig:
    """Configuration for interactive demos"""
    name: str
    description: str
    category: str
    
    # UI settings
    interface: str = "gradio"  # gradio, streamlit, fastapi
    port: int = 7860
    share: bool = False
    
    # Model settings
    model_path: Optional[str] = None
    max_batch_size: int = 32
    timeout: float = 30.0
    
    # Visualization
    enable_attention_viz: bool = True
    enable_metrics: bool = True
    enable_profiling: bool = False


@dataclass
class DemoResult:
    """Result from demo execution"""
    demo_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Results
    output: Any = None
    visualizations: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Metadata
    input_info: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class InteractiveDemo:
    """Base class for interactive demonstrations"""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.initialized = False
        self.run_count = 0
        
        # Performance tracking
        self.latencies: List[float] = []
        self.errors: List[str] = []
        
        logger.info(f"Demo initialized: {config.name}")
    
    async def initialize(self):
        """Initialize demo resources"""
        if self.initialized:
            return
        
        try:
            await self._setup_model()
            await self._setup_ui()
            self.initialized = True
            logger.info(f"Demo ready: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize demo: {e}")
            raise
    
    async def _setup_model(self):
        """Setup model for demo"""
        # Override in subclasses
        pass
    
    async def _setup_ui(self):
        """Setup UI interface"""
        # Override in subclasses
        pass
    
    async def run(self, **inputs) -> DemoResult:
        """Run the demo with given inputs"""
        start_time = time.time()
        result = DemoResult(demo_name=self.config.name)
        
        try:
            # Validate inputs
            validated_inputs = await self._validate_inputs(inputs)
            result.input_info = self._get_input_info(validated_inputs)
            
            # Run inference
            inference_start = time.time()
            output = await self._run_inference(validated_inputs)
            result.inference_time_ms = (time.time() - inference_start) * 1000
            
            # Process output
            result.output = await self._process_output(output)
            
            # Generate visualizations
            if self.config.enable_attention_viz:
                result.visualizations = await self._generate_visualizations(output)
            
            # Collect metrics
            if self.config.enable_metrics:
                result.metrics = await self._collect_metrics(output)
            
            # Update tracking
            self.run_count += 1
            result.total_time_ms = (time.time() - start_time) * 1000
            self.latencies.append(result.total_time_ms)
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            result.error = str(e)
            self.errors.append(str(e))
        
        return result
    
    async def _validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and preprocess inputs"""
        # Override in subclasses
        return inputs
    
    async def _run_inference(self, inputs: Dict[str, Any]) -> Any:
        """Run model inference"""
        # Override in subclasses
        raise NotImplementedError
    
    async def _process_output(self, output: Any) -> Any:
        """Process model output for display"""
        # Override in subclasses
        return output
    
    async def _generate_visualizations(self, output: Any) -> Dict[str, Any]:
        """Generate visualizations from output"""
        # Override in subclasses
        return {}
    
    async def _collect_metrics(self, output: Any) -> Dict[str, float]:
        """Collect performance metrics"""
        # Override in subclasses
        return {}
    
    def _get_input_info(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about inputs"""
        info = {}
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                info[key] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "min": float(value.min()),
                    "max": float(value.max())
                }
            elif isinstance(value, (list, tuple)):
                info[key] = {"length": len(value), "type": type(value).__name__}
            else:
                info[key] = {"type": type(value).__name__}
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get demo statistics"""
        return {
            "run_count": self.run_count,
            "avg_latency_ms": np.mean(self.latencies) if self.latencies else 0,
            "p95_latency_ms": np.percentile(self.latencies, 95) if self.latencies else 0,
            "error_rate": len(self.errors) / max(self.run_count, 1),
            "last_error": self.errors[-1] if self.errors else None
        }


class LNNDemo(InteractiveDemo):
    """Liquid Neural Network demonstration"""
    
    async def _setup_model(self):
        """Setup LNN model"""
        try:
            # Import actual LNN if available
            from aura_intelligence.lnn.liquid_neural_network import LiquidNeuralNetwork
            
            self.model = LiquidNeuralNetwork(
                input_size=10,
                hidden_size=64,
                output_size=5,
                num_layers=3
            )
            
            # Load pretrained weights if available
            if self.config.model_path and Path(self.config.model_path).exists():
                import torch
                self.model.load_state_dict(torch.load(self.config.model_path))
            
            self.model.eval()
            
        except ImportError:
            logger.warning("LNN module not available, using mock")
            self.model = None
    
    async def _run_inference(self, inputs: Dict[str, Any]) -> Any:
        """Run LNN inference"""
        if self.model is None:
            # Mock inference
            return {
                "output": np.random.randn(5),
                "hidden_states": np.random.randn(3, 64),
                "liquid_params": {
                    "tau": np.random.rand(3),
                    "A": np.random.randn(3, 64, 64)
                }
            }
        
        # Real inference
        import torch
        
        x = torch.tensor(inputs["input_sequence"], dtype=torch.float32)
        
        with torch.no_grad():
            output, hidden = self.model(x.unsqueeze(0))
            
            return {
                "output": output.squeeze().numpy(),
                "hidden_states": hidden.numpy(),
                "liquid_params": self.model.get_liquid_parameters()
            }
    
    async def _generate_visualizations(self, output: Any) -> Dict[str, Any]:
        """Generate LNN visualizations"""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        visualizations = {}
        
        # Hidden state evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        hidden_states = output["hidden_states"]
        
        for i in range(min(5, hidden_states.shape[1])):
            ax.plot(hidden_states[:, i], label=f"Neuron {i}")
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Activation")
        ax.set_title("Liquid Neural Network - Hidden State Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["hidden_states"] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Tau parameters heatmap
        if "tau" in output.get("liquid_params", {}):
            fig, ax = plt.subplots(figsize=(8, 6))
            tau = output["liquid_params"]["tau"]
            
            ax.bar(range(len(tau)), tau)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Time Constant (τ)")
            ax.set_title("LNN Time Constants by Layer")
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations["tau_params"] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        return visualizations


class ConsciousnessDemo(InteractiveDemo):
    """Consciousness system demonstration"""
    
    async def _setup_model(self):
        """Setup consciousness model"""
        try:
            from aura_intelligence.consciousness.global_workspace import GlobalWorkspace
            
            self.consciousness = GlobalWorkspace()
            await self.consciousness.initialize()
            
        except ImportError:
            logger.warning("Consciousness module not available, using mock")
            self.consciousness = None
    
    async def _run_inference(self, inputs: Dict[str, Any]) -> Any:
        """Run consciousness inference"""
        if self.consciousness is None:
            # Mock consciousness state
            return {
                "workspace_state": {
                    "attention_focus": np.random.rand(),
                    "conscious_content": ["perception", "memory", "planning"],
                    "phi_integrated": np.random.rand()
                },
                "executive_state": {
                    "active_goals": ["learn", "explore"],
                    "inhibited_processes": ["distraction"],
                    "working_memory": ["current_task", "context"]
                },
                "metrics": {
                    "integration_measure": np.random.rand(),
                    "complexity": np.random.rand(),
                    "coherence": np.random.rand()
                }
            }
        
        # Real consciousness processing
        stimulus = inputs.get("stimulus", "test_input")
        
        # Broadcast to workspace
        await self.consciousness.broadcast(stimulus)
        
        # Get current state
        state = await self.consciousness.get_state()
        
        return {
            "workspace_state": state.workspace_state,
            "executive_state": state.executive_state,
            "metrics": {
                "integration_measure": state.phi,
                "complexity": state.complexity,
                "coherence": state.coherence
            }
        }
    
    async def _generate_visualizations(self, output: Any) -> Dict[str, Any]:
        """Generate consciousness visualizations"""
        import matplotlib.pyplot as plt
        import networkx as nx
        import io
        import base64
        
        visualizations = {}
        
        # Consciousness metrics radar chart
        metrics = output["metrics"]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='purple')
        ax.fill(angles, values, alpha=0.25, color='purple')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Consciousness State Metrics", pad=20)
        ax.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["metrics_radar"] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Workspace content network
        fig, ax = plt.subplots(figsize=(10, 8))
        
        G = nx.Graph()
        
        # Add workspace nodes
        workspace = output["workspace_state"]
        for content in workspace.get("conscious_content", []):
            G.add_node(content, node_type="content")
        
        # Add executive nodes
        executive = output["executive_state"]
        for goal in executive.get("active_goals", []):
            G.add_node(goal, node_type="goal")
            # Connect to content
            for content in workspace.get("conscious_content", []):
                G.add_edge(goal, content, weight=np.random.rand())
        
        # Layout and draw
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes by type
        content_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "content"]
        goal_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "goal"]
        
        nx.draw_networkx_nodes(G, pos, nodelist=content_nodes, 
                              node_color='lightblue', node_size=1000, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=goal_nodes,
                              node_color='lightgreen', node_size=1000, ax=ax)
        
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title("Global Workspace Content Network", fontsize=16)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["workspace_network"] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return visualizations


class TDADemo(InteractiveDemo):
    """Topological Data Analysis demonstration"""
    
    async def _setup_model(self):
        """Setup TDA analyzer"""
        try:
            from aura_intelligence.core.topology import TopologyAnalyzer
            
            self.tda = TopologyAnalyzer()
            await self.tda.initialize()
            
        except ImportError:
            logger.warning("TDA module not available, using mock")
            self.tda = None
    
    async def _run_inference(self, inputs: Dict[str, Any]) -> Any:
        """Run TDA analysis"""
        if self.tda is None:
            # Mock TDA results
            n_points = 100
            return {
                "persistence_diagram": [
                    (0, 0.1), (0.05, 0.15), (0.1, np.inf),  # H0
                    (0.2, 0.4), (0.3, 0.5),  # H1
                    (0.6, 0.7)  # H2
                ],
                "betti_numbers": [3, 2, 1],
                "wasserstein_distance": 0.25,
                "bottleneck_distance": 0.15,
                "persistence_entropy": 0.8,
                "point_cloud": np.random.randn(n_points, 3)
            }
        
        # Real TDA analysis
        point_cloud = np.array(inputs["point_cloud"])
        
        result = await self.tda.analyze_topology(
            point_cloud,
            max_dimension=inputs.get("max_dimension", 2)
        )
        
        return result
    
    async def _generate_visualizations(self, output: Any) -> Dict[str, Any]:
        """Generate TDA visualizations"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import io
        import base64
        
        visualizations = {}
        
        # Persistence diagram
        fig, ax = plt.subplots(figsize=(10, 8))
        
        persistence = output["persistence_diagram"]
        
        # Separate by dimension
        h0 = [(b, d) for b, d in persistence if d - b < 0.2]  # Short-lived
        h1 = [(b, d) for b, d in persistence if 0.2 <= d - b < 0.5]
        h2 = [(b, d) for b, d in persistence if d - b >= 0.5 or d == np.inf]
        
        # Plot points
        for points, label, color in [(h0, "H0", "blue"), 
                                     (h1, "H1", "green"), 
                                     (h2, "H2", "red")]:
            if points:
                births = [p[0] for p in points]
                deaths = [p[1] if p[1] != np.inf else 1.0 for p in points]
                ax.scatter(births, deaths, label=label, color=color, s=50, alpha=0.7)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        ax.set_xlabel("Birth", fontsize=12)
        ax.set_ylabel("Death", fontsize=12)
        ax.set_title("Persistence Diagram", fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["persistence_diagram"] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Point cloud visualization (if 3D)
        if "point_cloud" in output and output["point_cloud"].shape[1] == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            points = output["point_cloud"]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap='viridis', alpha=0.6)
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Point Cloud Visualization", fontsize=16)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            visualizations["point_cloud"] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        # Betti numbers bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        betti = output["betti_numbers"]
        dimensions = list(range(len(betti)))
        
        bars = ax.bar(dimensions, betti, color=['blue', 'green', 'red'][:len(betti)])
        
        ax.set_xlabel("Dimension", fontsize=12)
        ax.set_ylabel("Betti Number", fontsize=12)
        ax.set_title("Topological Features by Dimension", fontsize=16)
        ax.set_xticks(dimensions)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, betti):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value)}', ha='center', va='bottom')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["betti_numbers"] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return visualizations


class DemoOrchestrator:
    """Orchestrates multiple demos with web interface"""
    
    def __init__(self):
        self.demos: Dict[str, InteractiveDemo] = {}
        self.initialized = False
        
        logger.info("Demo orchestrator initialized")
    
    def register_demo(self, demo: InteractiveDemo):
        """Register a demo"""
        self.demos[demo.config.name] = demo
        logger.info(f"Registered demo: {demo.config.name}")
    
    async def initialize_all(self):
        """Initialize all demos"""
        if self.initialized:
            return
        
        for name, demo in self.demos.items():
            try:
                await demo.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
        
        self.initialized = True
    
    async def run_demo(self, demo_name: str, **inputs) -> DemoResult:
        """Run a specific demo"""
        if demo_name not in self.demos:
            raise ValueError(f"Demo not found: {demo_name}")
        
        demo = self.demos[demo_name]
        return await demo.run(**inputs)
    
    def get_available_demos(self) -> List[Dict[str, Any]]:
        """Get list of available demos"""
        return [
            {
                "name": demo.config.name,
                "description": demo.config.description,
                "category": demo.config.category,
                "stats": demo.get_stats()
            }
            for demo in self.demos.values()
        ]
    
    async def launch_web_interface(self, port: int = 7860):
        """Launch Gradio web interface"""
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio not available, install with: pip install gradio")
            return
        
        # Create interface for each demo
        interfaces = []
        
        for name, demo in self.demos.items():
            if isinstance(demo, LNNDemo):
                interface = self._create_lnn_interface(demo)
            elif isinstance(demo, ConsciousnessDemo):
                interface = self._create_consciousness_interface(demo)
            elif isinstance(demo, TDADemo):
                interface = self._create_tda_interface(demo)
            else:
                continue
            
            interfaces.append((interface, name))
        
        # Create tabbed interface
        if interfaces:
            demo_dict = {name: interface for interface, name in interfaces}
            tabbed = gr.TabbedInterface(
                list(demo_dict.values()),
                list(demo_dict.keys()),
                title="AURA Intelligence Interactive Demos"
            )
            
            tabbed.launch(server_port=port, share=False)
    
    def _create_lnn_interface(self, demo: LNNDemo):
        """Create Gradio interface for LNN demo"""
        import gradio as gr
        
        async def run_lnn(sequence_length, hidden_size):
            # Generate input
            input_sequence = np.random.randn(int(sequence_length), 10)
            
            # Run demo
            result = await demo.run(input_sequence=input_sequence)
            
            if result.error:
                return f"Error: {result.error}", None, None
            
            # Format output
            output_text = f"Output shape: {result.output.shape}\n"
            output_text += f"Inference time: {result.inference_time_ms:.2f}ms\n"
            
            # Get visualizations
            hidden_viz = result.visualizations.get("hidden_states")
            tau_viz = result.visualizations.get("tau_params")
            
            return output_text, hidden_viz, tau_viz
        
        return gr.Interface(
            fn=run_lnn,
            inputs=[
                gr.Slider(10, 100, 50, label="Sequence Length"),
                gr.Slider(32, 256, 64, label="Hidden Size")
            ],
            outputs=[
                gr.Textbox(label="Results"),
                gr.Image(label="Hidden State Evolution"),
                gr.Image(label="Time Constants")
            ],
            title="Liquid Neural Network Demo",
            description="Explore the dynamics of Liquid Neural Networks"
        )
    
    def _create_consciousness_interface(self, demo: ConsciousnessDemo):
        """Create Gradio interface for consciousness demo"""
        import gradio as gr
        
        async def run_consciousness(stimulus_text, attention_level):
            # Run demo
            result = await demo.run(
                stimulus=stimulus_text,
                attention_level=float(attention_level)
            )
            
            if result.error:
                return f"Error: {result.error}", None, None
            
            # Format output
            state = result.output
            output_text = "Workspace State:\n"
            output_text += f"- Attention: {state['workspace_state'].get('attention_focus', 0):.2f}\n"
            output_text += f"- Content: {state['workspace_state'].get('conscious_content', [])}\n"
            output_text += f"- Phi: {state['metrics']['integration_measure']:.3f}\n"
            
            # Get visualizations
            radar_viz = result.visualizations.get("metrics_radar")
            network_viz = result.visualizations.get("workspace_network")
            
            return output_text, radar_viz, network_viz
        
        return gr.Interface(
            fn=run_consciousness,
            inputs=[
                gr.Textbox(label="Stimulus", value="Hello consciousness"),
                gr.Slider(0, 1, 0.8, label="Attention Level")
            ],
            outputs=[
                gr.Textbox(label="Consciousness State"),
                gr.Image(label="Metrics Radar"),
                gr.Image(label="Workspace Network")
            ],
            title="Consciousness System Demo",
            description="Interact with the Global Workspace consciousness model"
        )
    
    def _create_tda_interface(self, demo: TDADemo):
        """Create Gradio interface for TDA demo"""
        import gradio as gr
        
        async def run_tda(shape_type, n_points, noise_level):
            # Generate point cloud based on shape
            n = int(n_points)
            noise = float(noise_level)
            
            if shape_type == "Circle":
                theta = np.linspace(0, 2*np.pi, n)
                points = np.column_stack([
                    np.cos(theta) + np.random.randn(n) * noise,
                    np.sin(theta) + np.random.randn(n) * noise,
                    np.random.randn(n) * noise
                ])
            elif shape_type == "Torus":
                theta = np.random.rand(n) * 2 * np.pi
                phi = np.random.rand(n) * 2 * np.pi
                R, r = 2, 1
                points = np.column_stack([
                    (R + r * np.cos(theta)) * np.cos(phi) + np.random.randn(n) * noise,
                    (R + r * np.cos(theta)) * np.sin(phi) + np.random.randn(n) * noise,
                    r * np.sin(theta) + np.random.randn(n) * noise
                ])
            else:  # Random
                points = np.random.randn(n, 3)
            
            # Run demo
            result = await demo.run(point_cloud=points, max_dimension=2)
            
            if result.error:
                return f"Error: {result.error}", None, None, None
            
            # Format output
            output_text = f"Betti Numbers: {result.output['betti_numbers']}\n"
            output_text += f"Wasserstein Distance: {result.output['wasserstein_distance']:.3f}\n"
            output_text += f"Persistence Entropy: {result.output.get('persistence_entropy', 0):.3f}\n"
            
            # Get visualizations
            persistence_viz = result.visualizations.get("persistence_diagram")
            cloud_viz = result.visualizations.get("point_cloud")
            betti_viz = result.visualizations.get("betti_numbers")
            
            return output_text, persistence_viz, cloud_viz, betti_viz
        
        return gr.Interface(
            fn=run_tda,
            inputs=[
                gr.Dropdown(["Circle", "Torus", "Random"], label="Shape Type"),
                gr.Slider(50, 500, 100, label="Number of Points"),
                gr.Slider(0, 0.5, 0.1, label="Noise Level")
            ],
            outputs=[
                gr.Textbox(label="TDA Results"),
                gr.Image(label="Persistence Diagram"),
                gr.Image(label="Point Cloud"),
                gr.Image(label="Betti Numbers")
            ],
            title="Topological Data Analysis Demo",
            description="Analyze the topology of different point cloud shapes"
        )


# Example usage
async def create_demo_suite():
    """Create and configure demo suite"""
    orchestrator = DemoOrchestrator()
    
    # Register LNN demo
    lnn_config = DemoConfig(
        name="liquid_neural_network",
        description="Explore Liquid Neural Network dynamics and adaptability",
        category="neural_architectures"
    )
    orchestrator.register_demo(LNNDemo(lnn_config))
    
    # Register consciousness demo
    consciousness_config = DemoConfig(
        name="consciousness_system",
        description="Interact with the Global Workspace Theory implementation",
        category="cognitive_architectures"
    )
    orchestrator.register_demo(ConsciousnessDemo(consciousness_config))
    
    # Register TDA demo
    tda_config = DemoConfig(
        name="topological_analysis",
        description="Visualize topological features in data",
        category="analysis_tools"
    )
    orchestrator.register_demo(TDADemo(tda_config))
    
    # Initialize all demos
    await orchestrator.initialize_all()
    
    return orchestrator


async def run_example():
    """Run example demonstrations"""
    print("🎯 Creating AURA Intelligence Demo Suite")
    print("=" * 60)
    
    orchestrator = await create_demo_suite()
    
    # List available demos
    print("\nAvailable Demos:")
    for demo_info in orchestrator.get_available_demos():
        print(f"\n- {demo_info['name']}")
        print(f"  {demo_info['description']}")
        print(f"  Category: {demo_info['category']}")
        print(f"  Stats: {demo_info['stats']}")
    
    # Run a sample demo
    print("\n" + "=" * 60)
    print("Running TDA Demo...")
    
    # Generate sample torus
    n_points = 200
    theta = np.random.rand(n_points) * 2 * np.pi
    phi = np.random.rand(n_points) * 2 * np.pi
    R, r = 2, 1
    torus_points = np.column_stack([
        (R + r * np.cos(theta)) * np.cos(phi),
        (R + r * np.cos(theta)) * np.sin(phi),
        r * np.sin(theta)
    ])
    
    result = await orchestrator.run_demo(
        "topological_analysis",
        point_cloud=torus_points,
        max_dimension=2
    )
    
    print(f"\nResults:")
    print(f"- Betti numbers: {result.output['betti_numbers']}")
    print(f"- Inference time: {result.inference_time_ms:.2f}ms")
    print(f"- Total time: {result.total_time_ms:.2f}ms")
    
    # Launch web interface (commented out for testing)
    # print("\nLaunching web interface on http://localhost:7860")
    # await orchestrator.launch_web_interface()


if __name__ == "__main__":
    asyncio.run(run_example())