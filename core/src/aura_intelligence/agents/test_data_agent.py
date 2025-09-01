"""
📊 Data Agent - GPU-Accelerated Data Intelligence
================================================

Specializes in:
- RAPIDS GPU data processing
- Multi-scale TDA computation
- Anomaly detection via topology
- Pattern recognition at scale
- Statistical analysis and visualization
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import structlog

# Try to import RAPIDS, fall back to pandas if not available
try:
    import cudf
    import cupy as cp
    from cuml import DBSCAN, PCA
    RAPIDS_AVAILABLE = True
except ImportError:
    logger.warning("RAPIDS not available, using CPU fallbacks")
    RAPIDS_AVAILABLE = False
    cudf = pd
    cp = np

from .test_agents import TestAgentBase, TestAgentConfig, Tool, AgentRole
from ..tda.algorithms import compute_persistence_diagram, compute_bottleneck_distance
from ..adapters.tda_adapter_gpu import TDAGPUAdapter

logger = structlog.get_logger(__name__)


@dataclass
class DataInsight:
    """Result of data analysis"""
    dataset_id: str
    shape: Tuple[int, int]
    topological_signature: np.ndarray
    anomalies: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0


class RAPIDSDataProcessor:
    """GPU-accelerated data processing using RAPIDS"""
    
    def __init__(self, gpu_memory_limit: int = 8 * 1024**3):  # 8GB default
        self.gpu_memory_limit = gpu_memory_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def process(self, data: Union[pd.DataFrame, Dict, List]) -> cudf.DataFrame:
        """Process data on GPU"""
        # Convert to cuDF DataFrame
        if isinstance(data, pd.DataFrame):
            if RAPIDS_AVAILABLE:
                gdf = cudf.from_pandas(data)
            else:
                gdf = data
        elif isinstance(data, dict):
            if RAPIDS_AVAILABLE:
                gdf = cudf.DataFrame(data)
            else:
                gdf = pd.DataFrame(data)
        elif isinstance(data, list):
            if RAPIDS_AVAILABLE:
                gdf = cudf.DataFrame(data)
            else:
                gdf = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        # Basic preprocessing
        gdf = await self._preprocess(gdf)
        
        return gdf
        
    async def _preprocess(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Basic preprocessing on GPU"""
        # Handle missing values
        for col in gdf.columns:
            if gdf[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                # Fill numeric columns with median
                if RAPIDS_AVAILABLE:
                    median_val = gdf[col].median()
                    gdf[col] = gdf[col].fillna(median_val)
                else:
                    gdf[col] = gdf[col].fillna(gdf[col].median())
                    
        return gdf
        
    async def compute_statistics(self, gdf: cudf.DataFrame) -> Dict[str, Any]:
        """Compute statistics on GPU"""
        stats = {}
        
        for col in gdf.columns:
            if gdf[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                if RAPIDS_AVAILABLE:
                    stats[col] = {
                        "mean": float(gdf[col].mean()),
                        "std": float(gdf[col].std()),
                        "min": float(gdf[col].min()),
                        "max": float(gdf[col].max()),
                        "median": float(gdf[col].median()),
                        "q25": float(gdf[col].quantile(0.25)),
                        "q75": float(gdf[col].quantile(0.75))
                    }
                else:
                    stats[col] = {
                        "mean": gdf[col].mean(),
                        "std": gdf[col].std(),
                        "min": gdf[col].min(),
                        "max": gdf[col].max(),
                        "median": gdf[col].median(),
                        "q25": gdf[col].quantile(0.25),
                        "q75": gdf[col].quantile(0.75)
                    }
                    
        return stats


class ParallelTDAEngine:
    """Multi-scale TDA computation on GPU"""
    
    def __init__(self, gpu_count: int = 1):
        self.gpu_count = gpu_count
        self.tda_adapter = TDAGPUAdapter()
        
    async def compute_multiscale(self, 
                                data: Union[cudf.DataFrame, pd.DataFrame],
                                scales: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]) -> Dict[str, Any]:
        """Compute TDA at multiple scales in parallel"""
        # Convert to numpy for TDA
        if hasattr(data, 'to_numpy'):
            points = data.to_numpy()
        else:
            points = data.values
            
        # Ensure numeric data only
        numeric_cols = []
        for i, dtype in enumerate(data.dtypes):
            if dtype in ['float32', 'float64', 'int32', 'int64']:
                numeric_cols.append(i)
                
        if not numeric_cols:
            return {"error": "No numeric columns found"}
            
        points = points[:, numeric_cols].astype(np.float32)
        
        # Compute persistence at each scale
        persistence_results = {}
        
        for scale in scales:
            # Scale the data
            scaled_points = points * scale
            
            # Compute persistence diagram
            persistence = await self.tda_adapter.compute_persistence_batch(
                [scaled_points],
                max_dim=2
            )
            
            if persistence and len(persistence) > 0:
                persistence_results[f"scale_{scale}"] = {
                    "diagram": persistence[0],
                    "features": self._extract_features(persistence[0])
                }
                
        # Compute multi-scale summary
        multiscale_features = self._combine_multiscale_features(persistence_results)
        
        return {
            "scales": scales,
            "persistence_by_scale": persistence_results,
            "multiscale_features": multiscale_features,
            "topological_complexity": self._compute_complexity(persistence_results)
        }
        
    def _extract_features(self, persistence_diagram: List) -> np.ndarray:
        """Extract features from persistence diagram"""
        if not persistence_diagram:
            return np.zeros(10)
            
        features = []
        
        for dim in range(3):  # dimensions 0, 1, 2
            dim_diagram = [p for p in persistence_diagram if p.get('dimension', 0) == dim]
            
            if dim_diagram:
                births = [p.get('birth', 0) for p in dim_diagram]
                deaths = [p.get('death', 1) for p in dim_diagram]
                persistences = [d - b for b, d in zip(births, deaths)]
                
                features.extend([
                    len(dim_diagram),  # number of features
                    np.mean(persistences) if persistences else 0,
                    np.std(persistences) if persistences else 0,
                    np.max(persistences) if persistences else 0
                ])
            else:
                features.extend([0, 0, 0, 0])
                
        return np.array(features[:10])  # Limit to 10 features
        
    def _combine_multiscale_features(self, persistence_results: Dict[str, Any]) -> np.ndarray:
        """Combine features across scales"""
        all_features = []
        
        for scale_result in persistence_results.values():
            if 'features' in scale_result:
                all_features.append(scale_result['features'])
                
        if all_features:
            # Stack and compute summary statistics
            feature_matrix = np.vstack(all_features)
            
            combined = np.concatenate([
                np.mean(feature_matrix, axis=0),
                np.std(feature_matrix, axis=0),
                np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0)
            ])
            
            return combined[:30]  # Limit size
        else:
            return np.zeros(30)
            
    def _compute_complexity(self, persistence_results: Dict[str, Any]) -> float:
        """Compute topological complexity score"""
        complexities = []
        
        for scale_result in persistence_results.values():
            if 'features' in scale_result:
                # Complexity based on number and persistence of features
                features = scale_result['features']
                complexity = np.sum(features[::4]) * np.mean(features[1::4])  # count * mean_persistence
                complexities.append(complexity)
                
        return np.mean(complexities) if complexities else 0.0


class TopologicalAnomalyDetector:
    """Detect anomalies using topological signatures"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.baseline_signatures = []
        
    async def detect(self, topological_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies based on topological features"""
        anomalies = []
        
        # Extract multiscale features
        features = topological_features.get('multiscale_features', np.zeros(30))
        
        # Compare against baseline if available
        if self.baseline_signatures:
            # Compute distance to baseline
            distances = [
                np.linalg.norm(features - baseline)
                for baseline in self.baseline_signatures
            ]
            
            min_distance = np.min(distances)
            threshold = np.percentile(distances, 100 * (1 - self.contamination))
            
            if min_distance > threshold:
                anomalies.append({
                    "type": "topological_outlier",
                    "score": float(min_distance),
                    "threshold": float(threshold),
                    "description": "Unusual topological structure detected"
                })
        
        # Check for specific patterns
        complexity = topological_features.get('topological_complexity', 0)
        
        if complexity > 100:  # High complexity threshold
            anomalies.append({
                "type": "high_complexity",
                "score": float(complexity),
                "description": "Unusually complex topological structure"
            })
            
        # Check scale consistency
        if 'persistence_by_scale' in topological_features:
            scale_variations = []
            
            for scale_data in topological_features['persistence_by_scale'].values():
                if 'features' in scale_data:
                    scale_variations.append(np.sum(scale_data['features']))
                    
            if scale_variations:
                variation_coef = np.std(scale_variations) / (np.mean(scale_variations) + 1e-6)
                
                if variation_coef > 2.0:  # High variation across scales
                    anomalies.append({
                        "type": "scale_inconsistency",
                        "score": float(variation_coef),
                        "description": "Topological features vary significantly across scales"
                    })
                    
        return anomalies
        
    def update_baseline(self, features: np.ndarray):
        """Update baseline signatures"""
        self.baseline_signatures.append(features)
        
        # Keep only recent signatures
        if len(self.baseline_signatures) > 1000:
            self.baseline_signatures = self.baseline_signatures[-1000:]


class DataAgent(TestAgentBase):
    """
    Specialized agent for data analysis and insights.
    
    Capabilities:
    - GPU-accelerated data processing (RAPIDS)
    - Multi-scale topological analysis
    - Anomaly detection
    - Pattern recognition
    - Statistical analysis
    - Visualization generation
    """
    
    def __init__(self, agent_id: str = "data_agent_001", **kwargs):
        config = TestAgentConfig(
            agent_role=AgentRole.ANALYST,
            specialty="data",
            target_latency_ms=120.0,
            **kwargs
        )
        
        super().__init__(agent_id=agent_id, config=config, **kwargs)
        
        # Initialize specialized components
        self.rapids_processor = RAPIDSDataProcessor()
        self.tda_engine = ParallelTDAEngine(gpu_count=self.config.gpu_count if hasattr(self.config, 'gpu_count') else 1)
        self.anomaly_detector = TopologicalAnomalyDetector()
        
        # Pattern library
        self.known_patterns = {
            "periodic": {"period_range": [2, 100], "strength_threshold": 0.7},
            "trending": {"window_size": 20, "correlation_threshold": 0.8},
            "clustering": {"min_cluster_size": 5, "max_clusters": 20},
            "outlier": {"contamination": 0.1, "method": "isolation_forest"}
        }
        
        # Initialize tools
        self._init_data_tools()
        
        logger.info("Data Agent initialized",
                   agent_id=agent_id,
                   rapids_available=RAPIDS_AVAILABLE,
                   capabilities=["data_processing", "tda_analysis", "anomaly_detection", "visualization"])
                   
    def _init_data_tools(self):
        """Initialize data-specific tools"""
        self.tools = {
            "load_data": Tool(
                name="load_data",
                description="Load data from various formats",
                func=self._tool_load_data
            ),
            "compute_statistics": Tool(
                name="compute_statistics", 
                description="Compute statistical summaries",
                func=self._tool_compute_statistics
            ),
            "detect_anomalies": Tool(
                name="detect_anomalies",
                description="Detect anomalies using TDA",
                func=self._tool_detect_anomalies
            ),
            "find_patterns": Tool(
                name="find_patterns",
                description="Find patterns in data",
                func=self._tool_find_patterns
            ),
            "generate_visualization": Tool(
                name="generate_visualization",
                description="Generate data visualizations",
                func=self._tool_generate_visualization
            )
        }
        
    async def _handle_analyze(self, context: Dict[str, Any]) -> DataInsight:
        """Handle data analysis requests"""
        start_time = time.perf_counter()
        
        # Extract data from context
        data = context.get("original", {}).get("data", None)
        analysis_type = context.get("original", {}).get("analysis_type", "full")
        
        if data is None:
            return DataInsight(
                dataset_id="empty",
                shape=(0, 0),
                topological_signature=np.zeros(30),
                anomalies=[],
                patterns=[],
                statistics={}
            )
            
        # Process data on GPU
        processed_data = await self.rapids_processor.process(data)
        
        # Compute statistics
        statistics = await self.rapids_processor.compute_statistics(processed_data)
        
        # Multi-scale TDA analysis
        tda_results = await self.tda_engine.compute_multiscale(processed_data)
        
        # Detect anomalies
        anomalies = await self.anomaly_detector.detect(tda_results)
        
        # Find patterns
        patterns = await self._find_patterns(processed_data, tda_results)
        
        # Store in shape-aware memory
        if 'multiscale_features' in tda_results:
            await self.shape_memory.store(
                {
                    "type": "data_analysis",
                    "shape": processed_data.shape,
                    "statistics": statistics,
                    "anomalies": anomalies,
                    "patterns": patterns
                },
                embedding=tda_results['multiscale_features']
            )
            
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return DataInsight(
            dataset_id=f"data_{int(time.time())}",
            shape=processed_data.shape,
            topological_signature=tda_results.get('multiscale_features', np.zeros(30)),
            anomalies=anomalies,
            patterns=patterns,
            statistics=statistics,
            processing_time_ms=processing_time
        )
        
    async def _handle_generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data generation requests"""
        generation_type = context.get("original", {}).get("type", "synthetic")
        params = context.get("original", {}).get("params", {})
        
        if generation_type == "synthetic":
            # Generate synthetic data with specific topological properties
            data = await self._generate_synthetic_data(params)
        elif generation_type == "augmented":
            # Augment existing data
            base_data = context.get("original", {}).get("base_data")
            data = await self._augment_data(base_data, params)
        else:
            data = pd.DataFrame()
            
        # Analyze generated data
        analysis = await self._handle_analyze({
            "original": {"data": data}
        })
        
        return {
            "generated_data": data.to_dict() if hasattr(data, 'to_dict') else {},
            "analysis": {
                "shape": analysis.shape,
                "topological_signature": analysis.topological_signature.tolist(),
                "statistics": analysis.statistics
            }
        }
        
    async def _find_patterns(self, 
                           data: Union[cudf.DataFrame, pd.DataFrame],
                           tda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns in data using TDA and statistical methods"""
        patterns = []
        
        # Convert to numpy for pattern detection
        if hasattr(data, 'to_numpy'):
            data_array = data.to_numpy()
        else:
            data_array = data.values
            
        # 1. Clustering patterns using DBSCAN
        if RAPIDS_AVAILABLE and data_array.shape[0] > 10:
            try:
                # Use cuML DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(data_array)
                
                n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 0:
                    patterns.append({
                        "type": "clustering",
                        "n_clusters": int(n_clusters),
                        "cluster_sizes": [int(np.sum(labels == i)) for i in range(n_clusters)],
                        "description": f"Found {n_clusters} distinct clusters"
                    })
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                
        # 2. Periodic patterns (simplified)
        if data_array.shape[1] > 0:
            # Check first numeric column for periodicity
            signal = data_array[:, 0]
            
            if len(signal) > 20:
                # Simple autocorrelation check
                autocorr = np.correlate(signal - np.mean(signal), 
                                      signal - np.mean(signal), 
                                      mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                # Find peaks in autocorrelation
                peaks = []
                for i in range(10, min(len(autocorr)//2, 100)):
                    if i < len(autocorr) - 1:
                        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.5:
                            peaks.append(i)
                            
                if peaks:
                    patterns.append({
                        "type": "periodic",
                        "periods": peaks[:3],  # Top 3 periods
                        "strength": float(np.mean([autocorr[p] for p in peaks[:3]])),
                        "description": f"Periodic pattern with period {peaks[0]}"
                    })
                    
        # 3. Topological patterns
        complexity = tda_results.get('topological_complexity', 0)
        
        if complexity > 50:
            patterns.append({
                "type": "complex_topology",
                "complexity_score": float(complexity),
                "description": "Complex topological structure indicating non-linear relationships"
            })
            
        # 4. Distribution patterns
        if 'statistics' in context:
            stats = context['statistics']
            
            # Check for skewness
            for col, col_stats in stats.items():
                if 'mean' in col_stats and 'median' in col_stats:
                    skewness = (col_stats['mean'] - col_stats['median']) / (col_stats['std'] + 1e-6)
                    
                    if abs(skewness) > 1.0:
                        patterns.append({
                            "type": "skewed_distribution",
                            "column": col,
                            "skewness": float(skewness),
                            "description": f"Column {col} shows {'positive' if skewness > 0 else 'negative'} skew"
                        })
                        
        return patterns
        
    async def _generate_synthetic_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic data with specific properties"""
        n_samples = params.get('n_samples', 1000)
        n_features = params.get('n_features', 10)
        pattern_type = params.get('pattern', 'random')
        
        if pattern_type == 'spiral':
            # Generate spiral data
            t = np.linspace(0, 4 * np.pi, n_samples)
            x = t * np.cos(t)
            y = t * np.sin(t)
            z = t
            
            # Add more features
            data = np.column_stack([x, y, z])
            
            # Add noise features
            noise = np.random.randn(n_samples, max(0, n_features - 3))
            data = np.column_stack([data, noise]) if n_features > 3 else data
            
        elif pattern_type == 'clusters':
            # Generate clustered data
            n_clusters = params.get('n_clusters', 5)
            data = []
            
            for i in range(n_clusters):
                center = np.random.randn(n_features) * 10
                cluster_data = np.random.randn(n_samples // n_clusters, n_features) + center
                data.append(cluster_data)
                
            data = np.vstack(data)
            np.random.shuffle(data)
            
        else:
            # Random data
            data = np.random.randn(n_samples, n_features)
            
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
        
        return df
        
    async def _augment_data(self, base_data: Any, params: Dict[str, Any]) -> pd.DataFrame:
        """Augment existing data"""
        if base_data is None:
            return pd.DataFrame()
            
        # Convert to DataFrame
        if isinstance(base_data, pd.DataFrame):
            df = base_data.copy()
        else:
            df = pd.DataFrame(base_data)
            
        augmentation_type = params.get('type', 'noise')
        
        if augmentation_type == 'noise':
            # Add Gaussian noise
            noise_level = params.get('noise_level', 0.1)
            
            for col in df.select_dtypes(include=[np.number]).columns:
                noise = np.random.randn(len(df)) * df[col].std() * noise_level
                df[col] = df[col] + noise
                
        elif augmentation_type == 'transform':
            # Apply transformations
            for col in df.select_dtypes(include=[np.number]).columns:
                if params.get('log_transform', False):
                    df[f'{col}_log'] = np.log1p(np.abs(df[col]))
                    
                if params.get('polynomial', False):
                    df[f'{col}_squared'] = df[col] ** 2
                    
        return df
        
    # Tool implementations
    async def _tool_load_data(self, 
                            file_path: Optional[str] = None,
                            data_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Load data tool"""
        if file_path:
            # Load from file
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                return {"error": f"Unsupported file format: {file_path}"}
        elif data_dict:
            # Load from dictionary
            data = pd.DataFrame(data_dict)
        else:
            return {"error": "No data source provided"}
            
        return {
            "loaded": True,
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        
    async def _tool_compute_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics tool"""
        gdf = await self.rapids_processor.process(data)
        stats = await self.rapids_processor.compute_statistics(gdf)
        
        return stats
        
    async def _tool_detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies tool"""
        # Process data
        gdf = await self.rapids_processor.process(data)
        
        # Compute TDA
        tda_results = await self.tda_engine.compute_multiscale(gdf)
        
        # Detect anomalies
        anomalies = await self.anomaly_detector.detect(tda_results)
        
        return anomalies
        
    async def _tool_find_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find patterns tool"""
        # Process data
        gdf = await self.rapids_processor.process(data)
        
        # Compute TDA
        tda_results = await self.tda_engine.compute_multiscale(gdf)
        
        # Find patterns
        patterns = await self._find_patterns(gdf, tda_results)
        
        return patterns
        
    async def _tool_generate_visualization(self, 
                                         data: pd.DataFrame,
                                         viz_type: str = "scatter") -> Dict[str, Any]:
        """Generate visualization tool"""
        # In practice, would generate actual visualizations
        # For now, return visualization specification
        
        viz_spec = {
            "type": viz_type,
            "data_shape": data.shape,
            "encoding": {}
        }
        
        if viz_type == "scatter" and data.shape[1] >= 2:
            viz_spec["encoding"] = {
                "x": data.columns[0],
                "y": data.columns[1],
                "color": data.columns[2] if data.shape[1] > 2 else None
            }
        elif viz_type == "histogram":
            viz_spec["encoding"] = {
                "x": data.columns[0],
                "bins": 30
            }
        elif viz_type == "heatmap":
            viz_spec["encoding"] = {
                "matrix": "correlation"
            }
            
        return viz_spec


# Factory function
def create_data_agent(agent_id: Optional[str] = None, **kwargs) -> DataAgent:
    """Create a Data Agent instance"""
    if agent_id is None:
        agent_id = f"data_agent_{int(time.time())}"
        
    return DataAgent(agent_id=agent_id, **kwargs)