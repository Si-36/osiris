"""
Real Failure Prediction using LNN
=================================
Production-grade failure prediction with Liquid Neural Networks
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

# Add paths for real AURA components
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/src')

# Import REAL LNN components
from aura.lnn.variants import (
    MITLiquidNN,
    LiquidNeuralNetwork,
    all_variants
)

# Try to import advanced LNN components
try:
    from core.src.aura_intelligence.neural.liquid_2025 import LiquidCouncilAgent2025
    from core.src.aura_intelligence.lnn.real_mit_lnn import RealMITLNN
    HAS_ADVANCED_LNN = True
except:
    HAS_ADVANCED_LNN = False


class RealFailurePredictor:
    """Production-grade infrastructure failure prediction using LNN"""
    
    def __init__(self, model_type: str = "mit_liquid_nn"):
        # Initialize real LNN
        self.lnn_wrapper = LiquidNeuralNetwork(model_type)
        
        # PyTorch LNN for advanced processing
        self.mit_lnn = MITLiquidNN("infrastructure_predictor")
        
        # Advanced LNN if available
        if HAS_ADVANCED_LNN:
            self.advanced_lnn = RealMITLNN()
            self.council_agent = LiquidCouncilAgent2025()
        else:
            self.advanced_lnn = None
            self.council_agent = None
        
        # Prediction parameters
        self.prediction_horizon_hours = 4
        self.confidence_threshold = 0.75
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
        
        # Historical tracking
        self.prediction_history = []
        self.accuracy_tracker = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Model state
        self.hidden_state = None
        self.model_initialized = False
    
    async def predict_failures(
        self,
        tda_features: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        raw_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict infrastructure failures using real LNN
        
        Args:
            tda_features: Topological features from TDA analysis
            historical_data: Historical metrics and predictions
            raw_metrics: Raw infrastructure metrics
            
        Returns:
            Comprehensive failure predictions with confidence
        """
        
        # 1. Prepare LNN input features
        input_features = self._prepare_input_features(
            tda_features,
            historical_data,
            raw_metrics
        )
        
        # 2. PyTorch LNN processing for temporal dynamics
        torch_predictions = await self._pytorch_lnn_predict(input_features)
        
        # 3. High-level prediction using wrapper
        risk_prediction = self.lnn_wrapper.predict_sync(input_features)
        
        # 4. Advanced LNN if available
        if self.advanced_lnn and HAS_ADVANCED_LNN:
            advanced_pred = await self._advanced_lnn_predict(input_features)
        else:
            advanced_pred = None
        
        # 5. Combine predictions
        combined_prediction = self._combine_predictions(
            torch_predictions,
            risk_prediction,
            advanced_pred
        )
        
        # 6. Generate failure scenarios
        failure_scenarios = self._generate_failure_scenarios(
            combined_prediction,
            tda_features
        )
        
        # 7. Calculate time to failure
        ttf_predictions = self._predict_time_to_failure(
            combined_prediction,
            tda_features
        )
        
        # 8. Generate actionable recommendations
        recommendations = self._generate_recommendations(
            combined_prediction,
            failure_scenarios,
            tda_features
        )
        
        # 9. Update history and accuracy
        self._update_prediction_history(combined_prediction)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_score': combined_prediction['risk_score'],
            'confidence': combined_prediction['confidence'],
            'risk_level': self._get_risk_level(combined_prediction['risk_score']),
            'failure_scenarios': failure_scenarios,
            'time_to_failure': ttf_predictions,
            'recommendations': recommendations,
            'feature_importance': combined_prediction.get('feature_importance', {}),
            'model_insights': {
                'pytorch_contribution': torch_predictions.get('contribution', 0.5),
                'wrapper_contribution': 0.3,
                'advanced_contribution': 0.2 if advanced_pred else 0.0
            },
            'historical_accuracy': self._get_accuracy_metrics()
        }
    
    def _prepare_input_features(
        self,
        tda_features: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        raw_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare comprehensive input features for LNN"""
        
        features = {
            # Topological features
            'components': tda_features.get('betti_0', 1),
            'loops': tda_features.get('betti_1', 0),
            'voids': tda_features.get('betti_2', 0),
            'connectivity': 1.0 / (1.0 + tda_features.get('betti_0', 1)),
            'complexity': tda_features.get('betti_1', 0) / 10.0,
            'persistence_entropy': tda_features.get('persistence_entropy', 0),
            'max_persistence': tda_features.get('max_persistence', 0),
            'anomaly_score': tda_features.get('anomaly_score', 0),
            
            # Topology vector for LNN
            'topology_vector': [
                tda_features.get('betti_0', 1),
                tda_features.get('betti_1', 0),
                tda_features.get('betti_2', 0)
            ],
            
            # Historical patterns
            'trend': self._calculate_trend(historical_data),
            'volatility': self._calculate_volatility(historical_data),
            'seasonality': self._detect_seasonality(historical_data)
        }
        
        # Add raw metrics if available
        if raw_metrics:
            features.update({
                'cpu_pressure': raw_metrics.get('cpu_util', 0) / 100.0,
                'memory_pressure': raw_metrics.get('mem_util', 0) / 100.0,
                'network_load': raw_metrics.get('net_util', 0) / 100.0,
                'disk_pressure': raw_metrics.get('disk_util', 0) / 100.0
            })
        
        return features
    
    async def _pytorch_lnn_predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Use PyTorch LNN for temporal dynamics prediction"""
        
        # Convert features to tensor
        feature_vector = torch.tensor([
            features['components'],
            features['loops'],
            features['connectivity'],
            features['complexity'],
            features['persistence_entropy'],
            features['anomaly_score'],
            features.get('cpu_pressure', 0.5),
            features.get('memory_pressure', 0.5),
            features.get('network_load', 0.5),
            features.get('disk_pressure', 0.5)
        ], dtype=torch.float32).unsqueeze(0)
        
        # Pad to match LNN input size
        if feature_vector.shape[1] < self.mit_lnn.input_size:
            padding = torch.zeros(1, self.mit_lnn.input_size - feature_vector.shape[1])
            feature_vector = torch.cat([feature_vector, padding], dim=1)
        
        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, self.mit_lnn.hidden_size)
        
        # Forward pass through LNN
        with torch.no_grad():
            output, self.hidden_state = self.mit_lnn(feature_vector, self.hidden_state)
        
        # Extract predictions from output
        risk_components = torch.sigmoid(output[0, :4])  # First 4 outputs for risk
        temporal_features = output[0, 4:8]  # Next 4 for temporal dynamics
        
        return {
            'risk_components': {
                'immediate': risk_components[0].item(),
                'short_term': risk_components[1].item(),
                'medium_term': risk_components[2].item(),
                'long_term': risk_components[3].item()
            },
            'temporal_dynamics': {
                'acceleration': temporal_features[0].item(),
                'deceleration': temporal_features[1].item(),
                'oscillation': temporal_features[2].item(),
                'stability': temporal_features[3].item()
            },
            'contribution': 0.5,
            'hidden_state_norm': torch.norm(self.hidden_state).item()
        }
    
    async def _advanced_lnn_predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Use advanced LNN components if available"""
        
        try:
            # Use council agent for decision
            decision = await self.council_agent.make_decision({
                'topology': features,
                'context': 'infrastructure_monitoring'
            })
            
            return {
                'advanced_risk': decision.get('risk_score', 0.5),
                'reasoning': decision.get('reasoning', ''),
                'confidence': decision.get('confidence', 0.5)
            }
        except Exception as e:
            return {
                'error': str(e),
                'advanced_risk': 0.5,
                'confidence': 0.0
            }
    
    def _combine_predictions(
        self,
        torch_pred: Dict[str, Any],
        wrapper_pred: Dict[str, Any],
        advanced_pred: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine predictions from multiple LNN models"""
        
        # Weighted combination
        weights = {
            'pytorch': 0.5,
            'wrapper': 0.3,
            'advanced': 0.2
        }
        
        # Calculate combined risk score
        risk_score = (
            weights['pytorch'] * torch_pred['risk_components']['immediate'] +
            weights['wrapper'] * wrapper_pred['prediction']
        )
        
        if advanced_pred and 'advanced_risk' in advanced_pred:
            risk_score = (
                (weights['pytorch'] + weights['wrapper']) * risk_score +
                weights['advanced'] * advanced_pred['advanced_risk']
            )
        
        # Calculate confidence
        confidence = wrapper_pred['confidence']
        if torch_pred['hidden_state_norm'] > 10.0:
            confidence *= 0.9  # Reduce confidence if hidden state is large
        
        return {
            'risk_score': min(risk_score, 0.99),
            'confidence': confidence,
            'feature_importance': wrapper_pred.get('features', {}),
            'temporal_dynamics': torch_pred['temporal_dynamics'],
            'risk_components': torch_pred['risk_components']
        }
    
    def _generate_failure_scenarios(
        self,
        prediction: Dict[str, Any],
        tda_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific failure scenarios based on predictions"""
        
        scenarios = []
        
        # Cascading failure scenario
        if prediction['risk_score'] > 0.8 and tda_features.get('betti_1', 0) > 5:
            scenarios.append({
                'type': 'cascading_failure',
                'probability': prediction['risk_score'],
                'description': 'Multiple service dependencies create cascading failure risk',
                'affected_services': f"~{tda_features['betti_1'] * 2} services",
                'mitigation': 'Break circular dependencies and add circuit breakers'
            })
        
        # Resource exhaustion scenario
        if prediction['risk_components']['immediate'] > 0.7:
            scenarios.append({
                'type': 'resource_exhaustion',
                'probability': prediction['risk_components']['immediate'],
                'description': 'Critical resource approaching limits',
                'affected_services': 'Core infrastructure services',
                'mitigation': 'Scale resources immediately or shed load'
            })
        
        # Network partition scenario
        if tda_features.get('betti_0', 1) > 1:
            scenarios.append({
                'type': 'network_partition',
                'probability': 0.8,
                'description': f"Network split into {tda_features['betti_0']} partitions",
                'affected_services': 'Cross-region communication',
                'mitigation': 'Check network connectivity and routing'
            })
        
        # Slow degradation scenario
        if prediction['temporal_dynamics']['deceleration'] > 0.5:
            scenarios.append({
                'type': 'slow_degradation',
                'probability': 0.6,
                'description': 'Performance gradually degrading',
                'affected_services': 'User-facing applications',
                'mitigation': 'Investigate memory leaks and resource cleanup'
            })
        
        return sorted(scenarios, key=lambda x: x['probability'], reverse=True)
    
    def _predict_time_to_failure(
        self,
        prediction: Dict[str, Any],
        tda_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict time to failure based on risk dynamics"""
        
        # Base time calculation
        if prediction['risk_score'] > 0.9:
            base_hours = 0.5
        elif prediction['risk_score'] > 0.8:
            base_hours = 2.0
        elif prediction['risk_score'] > 0.7:
            base_hours = 4.0
        else:
            base_hours = 24.0
        
        # Adjust based on acceleration
        acceleration = prediction['temporal_dynamics']['acceleration']
        if acceleration > 0:
            time_factor = 1.0 / (1.0 + acceleration)
        else:
            time_factor = 1.0
        
        predicted_hours = base_hours * time_factor
        
        return {
            'hours': predicted_hours,
            'formatted': self._format_time_to_failure(predicted_hours),
            'confidence': prediction['confidence'] * 0.8,  # TTF less certain
            'range': {
                'min_hours': predicted_hours * 0.5,
                'max_hours': predicted_hours * 2.0
            }
        }
    
    def _format_time_to_failure(self, hours: float) -> str:
        """Format time to failure for display"""
        
        if hours < 1:
            return f"{int(hours * 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            days = hours / 24
            return f"{days:.1f} days"
    
    def _generate_recommendations(
        self,
        prediction: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        tda_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Critical recommendations
        if prediction['risk_score'] > 0.9:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Initiate emergency response protocol',
                'reason': 'Imminent failure detected',
                'automated': True,
                'commands': ['kubectl scale --replicas=5', 'enable-circuit-breakers']
            })
        
        # High risk recommendations
        elif prediction['risk_score'] > 0.7:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Scale preventively and alert on-call',
                'reason': 'High failure risk detected',
                'automated': False,
                'commands': ['check-resource-usage', 'prepare-failover']
            })
        
        # Specific scenario recommendations
        for scenario in scenarios[:2]:  # Top 2 scenarios
            if scenario['type'] == 'cascading_failure':
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Isolate dependent services',
                    'reason': scenario['description'],
                    'automated': False,
                    'commands': ['enable-service-mesh-isolation']
                })
            elif scenario['type'] == 'resource_exhaustion':
                recommendations.append({
                    'priority': 'CRITICAL',
                    'action': 'Emergency resource allocation',
                    'reason': scenario['description'],
                    'automated': True,
                    'commands': ['scale-out-immediately']
                })
        
        return recommendations
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        
        for level, threshold in sorted(self.risk_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if risk_score >= threshold:
                return level.upper()
        return 'LOW'
    
    def _calculate_trend(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate trend from historical data"""
        
        if len(historical_data) < 2:
            return 0.0
        
        # Simple linear trend
        values = [d.get('risk_score', 0.5) for d in historical_data[-10:]]
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return float(np.clip(slope * 10, -1, 1))  # Normalize
    
    def _calculate_volatility(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate volatility from historical data"""
        
        if len(historical_data) < 2:
            return 0.0
        
        values = [d.get('risk_score', 0.5) for d in historical_data[-20:]]
        return float(np.std(values))
    
    def _detect_seasonality(self, historical_data: List[Dict[str, Any]]) -> float:
        """Detect seasonality patterns"""
        
        # Simplified seasonality detection
        if len(historical_data) < 24:  # Need at least 24 hours
            return 0.0
        
        # Check for hourly patterns
        hourly_avg = {}
        for i, data in enumerate(historical_data[-168:]):  # Last week
            hour = i % 24
            if hour not in hourly_avg:
                hourly_avg[hour] = []
            hourly_avg[hour].append(data.get('risk_score', 0.5))
        
        # Calculate variance across hours
        hour_means = [np.mean(hourly_avg.get(h, [0.5])) for h in range(24)]
        seasonality = np.std(hour_means)
        
        return float(min(seasonality * 10, 1.0))  # Normalize
    
    def _update_prediction_history(self, prediction: Dict[str, Any]):
        """Update prediction history for accuracy tracking"""
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'verified': False
        })
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
        
        self.accuracy_tracker['total_predictions'] += 1
    
    def _get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get current accuracy metrics"""
        
        total = self.accuracy_tracker['total_predictions']
        if total == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        correct = self.accuracy_tracker['correct_predictions']
        fp = self.accuracy_tracker['false_positives']
        fn = self.accuracy_tracker['false_negatives']
        
        accuracy = correct / total if total > 0 else 0.0
        precision = correct / (correct + fp) if (correct + fp) > 0 else 0.0
        recall = correct / (correct + fn) if (correct + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        }


async def demo_real_predictor():
    """Demo the real failure predictor"""
    
    print("🚀 Real LNN Failure Prediction Demo")
    print("=" * 60)
    
    predictor = RealFailurePredictor()
    
    # Simulate predictions over time
    for t in range(5):
        print(f"\n⏰ Time step {t+1}")
        
        # Mock TDA features (would come from real TDA analysis)
        if t < 3:
            # Normal operation
            tda_features = {
                'betti_0': 1,
                'betti_1': np.random.randint(0, 3),
                'persistence_entropy': np.random.rand() * 0.5,
                'anomaly_score': np.random.rand() * 0.5,
                'max_persistence': np.random.rand() * 0.5
            }
        else:
            # Anomaly developing
            tda_features = {
                'betti_0': np.random.randint(1, 4),
                'betti_1': np.random.randint(5, 15),
                'persistence_entropy': 1.5 + np.random.rand(),
                'anomaly_score': 2.0 + np.random.rand(),
                'max_persistence': 2.0 + np.random.rand()
            }
        
        # Make prediction
        prediction = await predictor.predict_failures(
            tda_features,
            [],  # Historical data
            {
                'cpu_util': 50 + t * 10,
                'mem_util': 40 + t * 15,
                'net_util': 30 + t * 5,
                'disk_util': 60 + t * 8
            }
        )
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Risk Score: {prediction['risk_score']:.2%}")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        print(f"   Time to Failure: {prediction['time_to_failure']['formatted']}")
        
        if prediction['failure_scenarios']:
            print(f"\n⚠️ Failure Scenarios:")
            for scenario in prediction['failure_scenarios'][:2]:
                print(f"   - {scenario['type']}: {scenario['probability']:.1%} probability")
                print(f"     {scenario['description']}")
        
        if prediction['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in prediction['recommendations'][:2]:
                print(f"   [{rec['priority']}] {rec['action']}")
                print(f"     Reason: {rec['reason']}")
        
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(demo_real_predictor())