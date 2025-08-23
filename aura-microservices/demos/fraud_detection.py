"""
AURA Fraud Detection Demo
Combines TDA + LNN + Byzantine Consensus for financial fraud detection
"""

import asyncio
import numpy as np
import httpx
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random


class FraudDetectionSystem:
    """
    Advanced fraud detection using:
    - TDA for anomaly topology
    - LNN for logical reasoning
    - Byzantine consensus for reliability
    """
    
    def __init__(self):
        self.lnn_url = "http://localhost:8001"
        self.tda_url = "http://localhost:8002"
        self.consensus_url = "http://localhost:8003"
        
    async def analyze_transaction_batch(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a batch of transactions for fraud patterns.
        
        Args:
            transactions: List of transaction dictionaries with amount, time, location, etc.
        
        Returns:
            Analysis results with fraud predictions and confidence scores
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Extract topological features from transaction patterns
            transaction_features = self._extract_features(transactions)
            
            print("🔍 Step 1: Analyzing transaction topology...")
            tda_response = await client.post(
                f"{self.tda_url}/topology/analyze",
                json={
                    "data": transaction_features,
                    "algorithm": "vietoris_rips",
                    "max_dimension": 2,
                    "max_edge_length": 10.0
                }
            )
            tda_result = tda_response.json()
            
            # Extract topological anomalies
            anomaly_score = self._compute_anomaly_score(tda_result["features"])
            print(f"   Topological anomaly score: {anomaly_score:.3f}")
            
            # Step 2: Use LNN for logical reasoning about fraud patterns
            print("\n🧠 Step 2: Applying logical neural network reasoning...")
            
            # Prepare input for LNN (combine transaction features with topology)
            lnn_input = self._prepare_lnn_input(transaction_features, tda_result["features"])
            
            lnn_response = await client.post(
                f"{self.lnn_url}/lnn/inference",
                json={
                    "model_id": "default",
                    "input_data": lnn_input,
                    "use_consensus": True,
                    "consensus_threshold": 0.75
                }
            )
            lnn_result = lnn_response.json()
            
            fraud_probability = self._extract_fraud_probability(lnn_result["output"])
            print(f"   Fraud probability: {fraud_probability:.3f}")
            print(f"   Consensus achieved: {lnn_result['consensus_achieved']}")
            
            # Step 3: Final decision with Byzantine consensus
            print("\n🛡️ Step 3: Byzantine consensus for final decision...")
            
            # Aggregate multiple detection signals
            detection_signals = {
                "topological_anomaly": anomaly_score > 0.7,
                "lnn_prediction": fraud_probability > 0.6,
                "rule_based": self._check_rule_based_flags(transactions),
                "velocity_check": self._check_velocity(transactions)
            }
            
            # In a real system, this would coordinate multiple nodes
            final_decision = self._byzantine_consensus(detection_signals)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "n_transactions": len(transactions),
                "fraud_detected": final_decision["is_fraud"],
                "confidence": final_decision["confidence"],
                "analysis": {
                    "topological_anomaly_score": anomaly_score,
                    "fraud_probability": fraud_probability,
                    "consensus_achieved": lnn_result["consensus_achieved"],
                    "detection_signals": detection_signals
                },
                "high_risk_transactions": self._identify_high_risk(
                    transactions, transaction_features, tda_result
                )
            }
    
    def _extract_features(self, transactions: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract numerical features from transactions for TDA analysis."""
        features = []
        
        for tx in transactions:
            # Extract relevant features
            amount = float(tx.get("amount", 0))
            hour = float(tx.get("timestamp", datetime.now()).hour)
            day_of_week = float(tx.get("timestamp", datetime.now()).weekday())
            
            # Location-based features (simplified)
            location_risk = float(tx.get("location_risk_score", 0.5))
            
            # Behavioral features
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
            is_night = 1.0 if hour < 6 or hour > 22 else 0.0
            
            # Normalize amount (log scale)
            normalized_amount = np.log1p(amount) / 10.0
            
            features.append([
                normalized_amount,
                hour / 24.0,
                day_of_week / 7.0,
                location_risk,
                is_weekend,
                is_night
            ])
        
        return features
    
    def _compute_anomaly_score(self, topological_features: List[Dict]) -> float:
        """Compute anomaly score based on topological features."""
        if not topological_features:
            return 0.0
        
        # Look for unusual persistence patterns
        high_persistence_features = [
            f for f in topological_features 
            if f["persistence"] > 5.0 and f["dimension"] > 0
        ]
        
        # Anomaly score based on number and strength of persistent features
        anomaly_score = min(1.0, len(high_persistence_features) * 0.2)
        
        # Adjust based on persistence values
        if high_persistence_features:
            max_persistence = max(f["persistence"] for f in high_persistence_features)
            anomaly_score = min(1.0, anomaly_score + (max_persistence - 5.0) * 0.1)
        
        return anomaly_score
    
    def _prepare_lnn_input(self, features: List[List[float]], topo_features: List[Dict]) -> List[List[float]]:
        """Prepare input for LNN by combining transaction and topological features."""
        # Create a fixed-size representation
        lnn_input = []
        
        # Aggregate transaction features
        features_array = np.array(features)
        avg_features = np.mean(features_array, axis=0).tolist()
        std_features = np.std(features_array, axis=0).tolist()
        
        # Topological summary
        topo_summary = [0.0] * 6  # 6 summary statistics
        if topo_features:
            topo_summary[0] = len(topo_features)  # Number of features
            topo_summary[1] = max(f["persistence"] for f in topo_features)  # Max persistence
            topo_summary[2] = sum(f["persistence"] for f in topo_features) / len(topo_features)  # Avg persistence
            
            for dim in range(3):
                dim_features = [f for f in topo_features if f["dimension"] == dim]
                topo_summary[3 + dim] = len(dim_features)
        
        # Combine all features
        combined = avg_features + std_features + topo_summary
        
        # Ensure correct dimensions for LNN
        while len(combined) < 128:  # Pad to expected input size
            combined.append(0.0)
        
        return [combined[:128]]  # Truncate if too long
    
    def _extract_fraud_probability(self, lnn_output: List[List[float]]) -> float:
        """Extract fraud probability from LNN output."""
        # Assuming LNN outputs a vector where first element is fraud score
        if lnn_output and lnn_output[0]:
            # Apply sigmoid to convert to probability
            raw_score = lnn_output[0][0]
            return 1 / (1 + np.exp(-raw_score))
        return 0.5
    
    def _check_rule_based_flags(self, transactions: List[Dict[str, Any]]) -> bool:
        """Check traditional rule-based fraud indicators."""
        flags = []
        
        for tx in transactions:
            # High amount transactions
            if tx.get("amount", 0) > 10000:
                flags.append("high_amount")
            
            # Unusual time
            hour = tx.get("timestamp", datetime.now()).hour
            if hour < 4 or hour > 23:
                flags.append("unusual_time")
            
            # Multiple transactions in short time
            # (In real system, would check against historical data)
            
        return len(flags) >= 2
    
    def _check_velocity(self, transactions: List[Dict[str, Any]]) -> bool:
        """Check transaction velocity (frequency)."""
        if len(transactions) < 2:
            return False
        
        # Sort by timestamp
        sorted_tx = sorted(transactions, key=lambda x: x.get("timestamp", datetime.now()))
        
        # Check time between transactions
        for i in range(1, len(sorted_tx)):
            time_diff = sorted_tx[i]["timestamp"] - sorted_tx[i-1]["timestamp"]
            if time_diff < timedelta(minutes=5):
                return True  # Suspicious velocity
        
        return False
    
    def _byzantine_consensus(self, detection_signals: Dict[str, bool]) -> Dict[str, Any]:
        """Simulate Byzantine consensus on fraud detection."""
        # In real system, this would involve multiple nodes voting
        # For demo, we use weighted voting
        
        weights = {
            "topological_anomaly": 0.3,
            "lnn_prediction": 0.4,
            "rule_based": 0.2,
            "velocity_check": 0.1
        }
        
        total_weight = sum(weights[k] for k, v in detection_signals.items() if v)
        confidence = total_weight
        
        return {
            "is_fraud": confidence > 0.5,
            "confidence": confidence,
            "votes": detection_signals
        }
    
    def _identify_high_risk(
        self,
        transactions: List[Dict[str, Any]],
        features: List[List[float]],
        tda_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify specific high-risk transactions."""
        high_risk = []
        
        for i, tx in enumerate(transactions):
            risk_score = 0.0
            
            # High amount
            if tx.get("amount", 0) > 5000:
                risk_score += 0.3
            
            # Unusual pattern in features
            if i < len(features) and features[i][0] > 0.8:  # High normalized amount
                risk_score += 0.2
            
            # Night transaction
            if features[i][5] > 0.5:  # is_night feature
                risk_score += 0.2
            
            if risk_score > 0.5:
                high_risk.append({
                    "transaction_id": tx.get("id", f"tx_{i}"),
                    "amount": tx.get("amount", 0),
                    "risk_score": risk_score,
                    "timestamp": tx.get("timestamp", datetime.now()).isoformat()
                })
        
        return sorted(high_risk, key=lambda x: x["risk_score"], reverse=True)[:5]


async def simulate_fraud_detection():
    """Run a fraud detection simulation with sample data."""
    detector = FraudDetectionSystem()
    
    # Generate sample transactions
    print("🏦 AURA Fraud Detection System Demo")
    print("=" * 50)
    
    # Normal transactions
    normal_transactions = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(20):
        normal_transactions.append({
            "id": f"normal_{i}",
            "amount": random.uniform(10, 500),
            "timestamp": base_time + timedelta(hours=i),
            "location_risk_score": random.uniform(0.1, 0.3),
            "merchant": f"merchant_{random.randint(1, 10)}"
        })
    
    # Inject fraudulent patterns
    fraud_transactions = []
    fraud_time = datetime.now() - timedelta(hours=3)
    
    # Pattern 1: Rapid small transactions (card testing)
    for i in range(5):
        fraud_transactions.append({
            "id": f"fraud_test_{i}",
            "amount": random.uniform(0.99, 9.99),
            "timestamp": fraud_time + timedelta(minutes=i*2),
            "location_risk_score": 0.8,
            "merchant": "suspicious_merchant_1"
        })
    
    # Pattern 2: Large transaction at unusual time
    fraud_transactions.append({
        "id": "fraud_large_1",
        "amount": 15000,
        "timestamp": datetime.now() - timedelta(hours=4),  # 4 AM
        "location_risk_score": 0.9,
        "merchant": "offshore_merchant"
    })
    
    # Mix transactions
    all_transactions = normal_transactions + fraud_transactions
    random.shuffle(all_transactions)
    
    print(f"\n📊 Analyzing {len(all_transactions)} transactions...")
    print(f"   - Normal: {len(normal_transactions)}")
    print(f"   - Suspicious: {len(fraud_transactions)}")
    print("\n" + "-" * 50 + "\n")
    
    # Run analysis
    try:
        result = await detector.analyze_transaction_batch(all_transactions)
        
        print("\n📈 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Fraud Detected: {'🚨 YES' if result['fraud_detected'] else '✅ NO'}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nDetection Signals:")
        for signal, detected in result['analysis']['detection_signals'].items():
            print(f"  - {signal}: {'🔴' if detected else '🟢'}")
        
        print(f"\n🎯 High Risk Transactions:")
        for tx in result['high_risk_transactions']:
            print(f"  - {tx['transaction_id']}: ${tx['amount']:.2f} (risk: {tx['risk_score']:.2f})")
        
        print("\n✨ Analysis Components:")
        print(f"  - Topological Anomaly Score: {result['analysis']['topological_anomaly_score']:.3f}")
        print(f"  - Neural Network Fraud Probability: {result['analysis']['fraud_probability']:.3f}")
        print(f"  - Byzantine Consensus: {'Achieved' if result['analysis']['consensus_achieved'] else 'Not Achieved'}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure all AURA services are running!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(simulate_fraud_detection())