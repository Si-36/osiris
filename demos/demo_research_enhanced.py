#!/usr/bin/env python3
"""
AURA Research-Enhanced Demo - All 2025 Components
Shows PHFormer + Multi-Parameter + Streaming + Enhanced LNN
"""

import asyncio
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

class AURAResearchDemo:
    def __init__(self):
        self.frame_count = 0
        self.total_time = 0
        self.decisions = {"approve": 0, "reject": 0, "neutral": 0}
        self.enhanced_pipeline = None
        
    def initialize_pipeline(self):
        """Initialize the enhanced research pipeline"""
        try:
            from aura_intelligence.tda.enhanced_pipeline import get_enhanced_pipeline
            self.enhanced_pipeline = get_enhanced_pipeline('auto')
            return True
        except Exception as e:
            print(f"⚠️  Enhanced pipeline not available: {e}")
            return False
    
    def generate_research_data(self):
        """Generate diverse data patterns for research demo"""
        patterns = [
            ("Torus", self._generate_torus()),
            ("Sphere", self._generate_sphere()),
            ("Klein_Bottle", self._generate_klein_bottle()),
            ("Mobius_Strip", self._generate_mobius_strip()),
            ("Double_Helix", self._generate_double_helix()),
            ("Fractal_Tree", self._generate_fractal_tree()),
            ("Lorenz_Attractor", self._generate_lorenz_attractor()),
            ("Random_Graph", self._generate_random_graph())
        ]
        
        return patterns[self.frame_count % len(patterns)]
    
    def _generate_torus(self):
        """Generate torus point cloud"""
        n = 40
        u = np.random.uniform(0, 2*np.pi, n)
        v = np.random.uniform(0, 2*np.pi, n)
        R, r = 2.0, 1.0
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        return np.column_stack([x, y, z]) + np.random.randn(n, 3) * 0.1
    
    def _generate_sphere(self):
        """Generate sphere point cloud"""
        n = 35
        u = np.random.uniform(0, 2*np.pi, n)
        v = np.random.uniform(0, np.pi, n)
        
        x = np.sin(v) * np.cos(u)
        y = np.sin(v) * np.sin(u)
        z = np.cos(v)
        
        return np.column_stack([x, y, z]) + np.random.randn(n, 3) * 0.05
    
    def _generate_klein_bottle(self):
        """Generate Klein bottle point cloud"""
        n = 30
        u = np.random.uniform(0, 2*np.pi, n)
        v = np.random.uniform(0, 2*np.pi, n)
        
        x = (2 + np.cos(v/2) * np.sin(u) - np.sin(v/2) * np.sin(2*u)) * np.cos(v)
        y = (2 + np.cos(v/2) * np.sin(u) - np.sin(v/2) * np.sin(2*u)) * np.sin(v)
        z = np.sin(v/2) * np.sin(u) + np.cos(v/2) * np.sin(2*u)
        
        return np.column_stack([x, y, z]) + np.random.randn(n, 3) * 0.1
    
    def _generate_mobius_strip(self):
        """Generate Mobius strip point cloud"""
        n = 25
        u = np.random.uniform(0, 2*np.pi, n)
        v = np.random.uniform(-1, 1, n)
        
        x = (1 + v/2 * np.cos(u/2)) * np.cos(u)
        y = (1 + v/2 * np.cos(u/2)) * np.sin(u)
        z = v/2 * np.sin(u/2)
        
        return np.column_stack([x, y, z]) + np.random.randn(n, 3) * 0.05
    
    def _generate_double_helix(self):
        """Generate double helix point cloud"""
        n = 30
        t = np.linspace(0, 4*np.pi, n)
        
        # First helix
        x1 = np.cos(t)
        y1 = np.sin(t)
        z1 = t * 0.3
        
        # Second helix (offset)
        x2 = np.cos(t + np.pi)
        y2 = np.sin(t + np.pi)
        z2 = t * 0.3
        
        helix1 = np.column_stack([x1, y1, z1])
        helix2 = np.column_stack([x2, y2, z2])
        
        return np.vstack([helix1, helix2]) + np.random.randn(2*n, 3) * 0.05
    
    def _generate_fractal_tree(self):
        """Generate fractal tree structure"""
        points = []
        
        def add_branch(start, direction, length, depth):
            if depth <= 0 or length < 0.1:
                return
            
            end = start + direction * length
            points.append(end)
            
            # Add branches
            angle1 = np.pi / 6
            angle2 = -np.pi / 6
            
            # Rotate direction
            cos_a1, sin_a1 = np.cos(angle1), np.sin(angle1)
            cos_a2, sin_a2 = np.cos(angle2), np.sin(angle2)
            
            dir1 = np.array([
                direction[0] * cos_a1 - direction[1] * sin_a1,
                direction[0] * sin_a1 + direction[1] * cos_a1,
                direction[2]
            ])
            
            dir2 = np.array([
                direction[0] * cos_a2 - direction[1] * sin_a2,
                direction[0] * sin_a2 + direction[1] * cos_a2,
                direction[2]
            ])
            
            add_branch(end, dir1, length * 0.7, depth - 1)
            add_branch(end, dir2, length * 0.7, depth - 1)
        
        add_branch(np.array([0, 0, 0]), np.array([0, 0, 1]), 1.0, 4)
        
        return np.array(points) if points else np.random.randn(20, 3)
    
    def _generate_lorenz_attractor(self):
        """Generate Lorenz attractor point cloud"""
        def lorenz(x, y, z, s=10, r=28, b=2.667):
            dx = s * (y - x)
            dy = r * x - y - x * z
            dz = x * y - b * z
            return dx, dy, dz
        
        dt = 0.01
        n_steps = 30
        
        x, y, z = 0.1, 0.1, 0.1
        points = []
        
        for _ in range(n_steps):
            dx, dy, dz = lorenz(x, y, z)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            points.append([x, y, z])
        
        return np.array(points) * 0.1  # Scale down
    
    def _generate_random_graph(self):
        """Generate random graph structure"""
        n = 25
        # Random points
        points = np.random.randn(n, 3) * 2
        
        # Add some structure by connecting nearby points
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))
        threshold = np.percentile(distances, 30)
        
        # Add points along edges
        additional_points = []
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < threshold:
                    # Add midpoint
                    midpoint = (points[i] + points[j]) / 2
                    additional_points.append(midpoint)
        
        if additional_points:
            return np.vstack([points, additional_points])
        else:
            return points
    
    async def process_frame(self, pattern_name, data):
        """Process frame with enhanced research pipeline"""
        start_time = time.perf_counter()
        
        if self.enhanced_pipeline:
            # Use enhanced pipeline
            result = self.enhanced_pipeline.process_enhanced(data, pattern_name)
            
            # Enhanced LNN decision with rich features
            try:
                from aura_intelligence.lnn.edge_deployment import get_edge_lnn_processor
                
                edge_lnn = get_edge_lnn_processor('nano', power_budget_mw=25)
                
                # Rich feature vector from enhanced pipeline
                enhanced_features = result['enhanced_features']
                context_vector = np.array([
                    enhanced_features.get('betti_0', 0),
                    enhanced_features.get('betti_1', 0),
                    enhanced_features.get('topology_richness', 0),
                    enhanced_features.get('stability_score', 1.0),
                    enhanced_features.get('mp_entropy', 0.5),
                    enhanced_features.get('mp_complexity', 0.3),
                    enhanced_features.get('phformer_mean', 0.0),
                    enhanced_features.get('phformer_std', 1.0)
                ])
                
                lnn_result = edge_lnn.edge_inference(context_vector)
                
                result.update({
                    'decision': lnn_result['decision'],
                    'confidence': lnn_result['confidence'],
                    'lnn_time': lnn_result['inference_time_ms'],
                    'power_usage': lnn_result['estimated_power_mw']
                })
                
            except Exception as e:
                result.update({
                    'decision': 'neutral',
                    'confidence': 0.5,
                    'lnn_time': 0.5,
                    'power_usage': 10.0,
                    'lnn_error': str(e)
                })
        else:
            # Fallback processing
            processing_time = (time.perf_counter() - start_time) * 1000
            result = {
                'pattern': pattern_name,
                'betti_numbers': [1, 0],
                'decision': 'neutral',
                'confidence': 0.5,
                'processing_time_ms': processing_time,
                'method': 'Fallback',
                'power_usage': 10.0
            }
        
        return result
    
    def print_header(self):
        print("\n" + "="*90)
        print("🚀 AURA RESEARCH-ENHANCED DEMO - 2025 Advanced Components")
        print("="*90)
        print("🔬 PHFormer 2.0 + Multi-Parameter + Streaming + Enhanced LNN")
        print("🎯 Target: <10ms with advanced topological intelligence")
        print("-"*90)
    
    def print_frame_result(self, result):
        """Print enhanced research result"""
        decision_emoji = {"approve": "✅", "reject": "❌", "neutral": "⚪"}
        
        # Enhanced features display
        enhanced = result.get('enhanced_features', {})
        topology_richness = enhanced.get('topology_richness', 0)
        stability_score = enhanced.get('stability_score', 1.0)
        
        # Stability indicator
        if stability_score > 0.8:
            stability_emoji = "🟢"
        elif stability_score > 0.5:
            stability_emoji = "🟡"
        else:
            stability_emoji = "🔴"
        
        # Components indicator
        components = result.get('components', {})
        comp_indicators = []
        if components.get('phformer', False):
            comp_indicators.append("🧠PHF")
        if components.get('multi_parameter', False):
            comp_indicators.append("📊MP")
        if components.get('streaming', False):
            comp_indicators.append("🌊STR")
        
        comp_str = " ".join(comp_indicators) if comp_indicators else "💤BASIC"
        
        print(f"Frame {self.frame_count:3d} | "
              f"Pattern: {result['pattern']:15s} | "
              f"Betti: {str(result['betti_numbers']):12s} | "
              f"{decision_emoji[result['decision']]} {result['decision']:7s} | "
              f"Conf: {result['confidence']:.3f} | "
              f"Rich: {topology_richness:5.2f} | "
              f"{stability_emoji} Stab: {stability_score:.2f} | "
              f"Time: {result.get('processing_time_ms', 0):6.2f}ms | "
              f"{comp_str}")
    
    def print_summary(self):
        """Print research demo summary"""
        avg_time = self.total_time / max(self.frame_count, 1)
        fps = 1000 / max(avg_time, 1)
        
        print("-"*90)
        print("🔬 RESEARCH DEMO SUMMARY:")
        print(f"   Frames Processed: {self.frame_count}")
        print(f"   Average Time: {avg_time:.2f}ms")
        print(f"   Effective FPS: {fps:.1f}")
        print(f"   Decisions: ✅{self.decisions['approve']} ❌{self.decisions['reject']} ⚪{self.decisions['neutral']}")
        
        if self.enhanced_pipeline:
            pipeline_info = self.enhanced_pipeline.get_pipeline_info()
            print(f"   Enhanced Pipeline: {'✅ Active' if pipeline_info['components_loaded'] else '❌ Fallback'}")
        
        print(f"   Research Components: PHFormer 2.0 + Multi-Parameter + Streaming TDA")
        
        if avg_time < 10:
            print("   Performance: 🚀 RESEARCH TARGET ACHIEVED!")
        elif avg_time < 20:
            print("   Performance: ⚡ EXCELLENT")
        else:
            print("   Performance: 🔧 NEEDS OPTIMIZATION")
        
        print("="*90)
    
    async def run_demo(self, num_frames=15):
        """Run the research demo"""
        
        # Initialize pipeline
        pipeline_ready = self.initialize_pipeline()
        
        self.print_header()
        
        if not pipeline_ready:
            print("⚠️  Running in fallback mode - some research features unavailable")
            print("-"*90)
        
        for i in range(num_frames):
            try:
                # Generate research data
                pattern_name, data = self.generate_research_data()
                
                # Process through enhanced pipeline
                result = await self.process_frame(pattern_name, data)
                
                # Update stats
                self.frame_count += 1
                self.total_time += result.get('processing_time_ms', 0)
                self.decisions[result['decision']] += 1
                
                # Print result
                self.print_frame_result(result)
                
                # Small delay for readability
                await asyncio.sleep(0.2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in frame {i}: {e}")
        
        self.print_summary()

async def main():
    """Main research demo function"""
    
    print("""
🔬 AURA RESEARCH-ENHANCED DEMO 2025
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This demo showcases cutting-edge 2025 research integration:

🧠 PHFormer 2.0: Topology-aware transformers for rich shape understanding
📊 Multi-Parameter Persistence: Advanced TDA with temporal analysis  
🌊 Streaming Processing: Real-time topology with intelligent caching
⚡ Enhanced LNN: Context-aware decisions with 8D feature vectors
🎯 Complex Geometries: Torus, Klein bottle, Lorenz attractor, fractals

Press Ctrl+C to stop early...
    """)
    
    demo = AURAResearchDemo()
    
    try:
        await demo.run_demo(num_frames=20)
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
        demo.print_summary()

if __name__ == "__main__":
    asyncio.run(main())