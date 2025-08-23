#!/usr/bin/env python3
"""
🧠 AURA Knowledge Graph Integration Script

Connects the existing enhanced knowledge graph with the AURA system.
Tests all graph ML capabilities and ensures proper integration.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, List

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "core" / "src"))

from src.aura.core.system import AURASystem
from src.aura.core.config import AURAConfig
from aura_intelligence.enterprise.enhanced_knowledge_graph import EnhancedKnowledgeGraphService
from aura_intelligence.enterprise.data_structures import (
    TopologicalSignature, SystemEvent, AgentAction, Outcome
)
from aura_intelligence.utils.logger import get_logger

logger = get_logger(__name__)


class AURAKnowledgeGraphIntegration:
    """Integrates AURA system with the enhanced knowledge graph"""
    
    def __init__(self):
        self.aura_system = None
        self.kg_service = None
        self.config = AURAConfig()
        
    async def initialize(self):
        """Initialize AURA system and knowledge graph"""
        logger.info("🚀 Initializing AURA Knowledge Graph Integration...")
        
        # Initialize AURA system
        self.aura_system = AURASystem(self.config)
        logger.info("✅ AURA system initialized")
        
        # Initialize enhanced knowledge graph
        self.kg_service = EnhancedKnowledgeGraphService(
            uri=self.config.neo4j_uri,
            username=self.config.neo4j_user,
            password=self.config.neo4j_password
        )
        
        kg_initialized = await self.kg_service.initialize()
        if kg_initialized:
            logger.info("✅ Enhanced Knowledge Graph initialized with GDS 2.19")
        else:
            logger.error("❌ Failed to initialize Knowledge Graph")
            return False
            
        return True
    
    async def test_knowledge_graph_features(self):
        """Test all knowledge graph features"""
        logger.info("\n🧪 Testing Knowledge Graph Features...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # Test 1: Store topological signatures
        logger.info("\n1️⃣ Testing Topological Signature Storage...")
        try:
            # Create test signatures
            signatures = []
            for i in range(5):
                sig = TopologicalSignature(
                    hash=f"test_sig_{i}",
                    topology={"nodes": i+2, "edges": i+1},
                    persistence_0={f"comp_{j}": i*j for j in range(3)},
                    persistence_1={f"hole_{j}": i+j for j in range(2)},
                    timestamp=datetime.now(),
                    consciousness_level=0.7 + i*0.05,
                    metadata={"test": True, "index": i}
                )
                signatures.append(sig)
                
            # Store signatures
            for sig in signatures:
                stored = await self.kg_service.store_topological_signature(sig)
                logger.info(f"  ✓ Stored signature: {sig.hash}")
                
            test_results["tests"].append({
                "name": "Signature Storage",
                "status": "passed",
                "details": f"Stored {len(signatures)} signatures"
            })
        except Exception as e:
            logger.error(f"  ✗ Signature storage failed: {e}")
            test_results["tests"].append({
                "name": "Signature Storage",
                "status": "failed",
                "error": str(e)
            })
        
        # Test 2: Community Detection
        logger.info("\n2️⃣ Testing Community Detection...")
        try:
            communities = await self.kg_service.detect_signature_communities(
                consciousness_level=0.8
            )
            
            if "error" not in communities:
                logger.info(f"  ✓ Detected {communities['total_communities']} communities")
                logger.info(f"  ✓ Algorithm used: {communities['algorithm_used']}")
                logger.info(f"  ✓ Computation time: {communities['computation_time_ms']:.2f}ms")
                
                test_results["tests"].append({
                    "name": "Community Detection",
                    "status": "passed",
                    "details": communities
                })
            else:
                raise Exception(communities["error"])
                
        except Exception as e:
            logger.error(f"  ✗ Community detection failed: {e}")
            test_results["tests"].append({
                "name": "Community Detection",
                "status": "failed",
                "error": str(e)
            })
        
        # Test 3: Centrality Analysis
        logger.info("\n3️⃣ Testing Centrality Analysis...")
        try:
            centrality = await self.kg_service.analyze_centrality_patterns(
                consciousness_level=0.9
            )
            
            if "error" not in centrality:
                logger.info(f"  ✓ Analyzed {len(centrality['centrality_algorithms'])} algorithms")
                logger.info(f"  ✓ Top signatures: {len(centrality.get('top_signatures', []))}")
                logger.info(f"  ✓ Computation time: {centrality['computation_time_ms']:.2f}ms")
                
                test_results["tests"].append({
                    "name": "Centrality Analysis",
                    "status": "passed",
                    "details": {
                        "algorithms": centrality['centrality_algorithms'],
                        "computation_time_ms": centrality['computation_time_ms']
                    }
                })
            else:
                raise Exception(centrality["error"])
                
        except Exception as e:
            logger.error(f"  ✗ Centrality analysis failed: {e}")
            test_results["tests"].append({
                "name": "Centrality Analysis", 
                "status": "failed",
                "error": str(e)
            })
        
        # Test 4: Pattern Prediction
        logger.info("\n4️⃣ Testing Pattern Prediction...")
        try:
            prediction = await self.kg_service.predict_future_patterns(
                signature_hash="test_sig_0",
                consciousness_level=0.8
            )
            
            if "error" not in prediction:
                logger.info(f"  ✓ Generated {len(prediction.get('predictions', []))} predictions")
                logger.info(f"  ✓ ML algorithms: {prediction['ml_algorithms_used']}")
                logger.info(f"  ✓ Confidence: {prediction['prediction_confidence']:.2f}")
                
                test_results["tests"].append({
                    "name": "Pattern Prediction",
                    "status": "passed",
                    "details": {
                        "algorithms_used": prediction['ml_algorithms_used'],
                        "confidence": prediction['prediction_confidence']
                    }
                })
            else:
                raise Exception(prediction["error"])
                
        except Exception as e:
            logger.error(f"  ✗ Pattern prediction failed: {e}")
            test_results["tests"].append({
                "name": "Pattern Prediction",
                "status": "failed", 
                "error": str(e)
            })
        
        # Test 5: AURA System Integration
        logger.info("\n5️⃣ Testing AURA System Integration...")
        try:
            # Run AURA pipeline with knowledge graph
            test_data = {
                "topology": {
                    "nodes": [{"id": i, "type": "agent"} for i in range(10)],
                    "edges": [{"source": i, "target": (i+1)%10} for i in range(10)]
                },
                "metrics": {
                    "cascade_risk": 0.75,
                    "connectivity": 0.9
                }
            }
            
            # Execute AURA pipeline
            result = await self.aura_system.execute_pipeline(test_data)
            
            # Store result in knowledge graph
            outcome = Outcome(
                outcome_id=f"aura_test_{datetime.now().timestamp()}",
                outcome_type="prevention",
                success=result.get("failure_prevented", False),
                impact_metrics={
                    "prevented_failures": result.get("prevented_failures", 0),
                    "risk_reduction": result.get("risk_reduction", 0)
                },
                timestamp=datetime.now(),
                metadata={"test": True}
            )
            
            await self.kg_service.store_outcome(outcome)
            logger.info(f"  ✓ AURA pipeline executed and stored in KG")
            
            test_results["tests"].append({
                "name": "AURA Integration",
                "status": "passed",
                "details": {
                    "failure_prevented": result.get("failure_prevented", False),
                    "knowledge_graph_stored": True
                }
            })
            
        except Exception as e:
            logger.error(f"  ✗ AURA integration failed: {e}")
            test_results["tests"].append({
                "name": "AURA Integration",
                "status": "failed",
                "error": str(e)
            })
        
        # Test 6: Consciousness-Driven Analysis
        logger.info("\n6️⃣ Testing Consciousness-Driven Analysis...")
        try:
            consciousness_state = {
                "level": 0.85,
                "coherence": 0.9,
                "phase": "gamma"
            }
            
            analysis = await self.kg_service.consciousness_driven_analysis(consciousness_state)
            
            if analysis.get("success", False):
                logger.info(f"  ✓ Analysis depth: {analysis['analysis_depth']}")
                logger.info(f"  ✓ Communities: {analysis.get('communities', {}).get('total_communities', 0)}")
                logger.info(f"  ✓ Has centrality: {'centrality' in analysis}")
                logger.info(f"  ✓ Has predictions: {'predictions' in analysis}")
                
                test_results["tests"].append({
                    "name": "Consciousness-Driven Analysis",
                    "status": "passed",
                    "details": {
                        "analysis_depth": analysis['analysis_depth'],
                        "effective_consciousness": analysis['consciousness_level']
                    }
                })
            else:
                raise Exception(analysis.get("error", "Unknown error"))
                
        except Exception as e:
            logger.error(f"  ✗ Consciousness-driven analysis failed: {e}")
            test_results["tests"].append({
                "name": "Consciousness-Driven Analysis",
                "status": "failed",
                "error": str(e)
            })
        
        # Generate summary
        passed = sum(1 for t in test_results["tests"] if t["status"] == "passed")
        total = len(test_results["tests"])
        test_results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": f"{(passed/total)*100:.1f}%"
        }
        
        # Save results
        with open("knowledge_graph_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
            
        return test_results
    
    async def demonstrate_real_world_usage(self):
        """Demonstrate real-world usage scenario"""
        logger.info("\n🌟 Real-World Usage Demonstration...")
        
        # Simulate a multi-agent system experiencing issues
        logger.info("\n📊 Simulating multi-agent system with potential cascade failure...")
        
        # Create realistic agent topology
        agents = []
        for i in range(30):
            agent_type = ["predictor", "analyzer", "executor"][i % 3]
            agents.append({
                "id": f"agent_{i}",
                "type": agent_type,
                "load": 0.5 + (i % 5) * 0.1,
                "connections": [(i-1)%30, (i+1)%30, (i+15)%30]
            })
        
        # Simulate cascade risk detection
        topology_data = {
            "nodes": agents,
            "edges": [
                {"source": i, "target": j, "weight": 0.8}
                for i in range(30)
                for j in agents[i]["connections"]
            ]
        }
        
        # Run AURA analysis
        logger.info("🔍 Running AURA topological analysis...")
        tda_result = self.aura_system.analyze_topology(topology_data)
        
        # Store in knowledge graph
        signature = TopologicalSignature(
            hash=f"cascade_risk_{datetime.now().timestamp()}",
            topology=topology_data,
            persistence_0=tda_result.get("persistence_0", {}),
            persistence_1=tda_result.get("persistence_1", {}),
            timestamp=datetime.now(),
            consciousness_level=0.85,
            metadata={
                "scenario": "cascade_risk_detection",
                "agent_count": len(agents)
            }
        )
        
        await self.kg_service.store_topological_signature(signature)
        
        # Predict cascade failure
        logger.info("🔮 Predicting cascade failure patterns...")
        prediction = await self.aura_system.predict_failure(tda_result)
        
        if prediction["cascade_risk"] > 0.7:
            logger.warning(f"⚠️  High cascade risk detected: {prediction['cascade_risk']:.2f}")
            
            # Prevent cascade
            logger.info("🛡️  Activating AURA cascade prevention...")
            prevention_result = await self.aura_system.prevent_cascade(
                topology_data, 
                prediction
            )
            
            # Store outcome
            outcome = Outcome(
                outcome_id=f"prevention_{datetime.now().timestamp()}",
                outcome_type="cascade_prevention",
                success=prevention_result["success"],
                impact_metrics={
                    "agents_protected": prevention_result.get("agents_protected", 0),
                    "risk_reduction": prevention_result.get("risk_reduction", 0),
                    "response_time_ms": prevention_result.get("response_time", 0)
                },
                timestamp=datetime.now(),
                metadata={
                    "initial_risk": prediction["cascade_risk"],
                    "final_risk": prevention_result.get("final_risk", 0)
                }
            )
            
            await self.kg_service.store_outcome(outcome)
            
            logger.info(f"✅ Cascade prevention {'successful' if outcome.success else 'failed'}")
            logger.info(f"   - Agents protected: {outcome.impact_metrics['agents_protected']}")
            logger.info(f"   - Risk reduced by: {outcome.impact_metrics['risk_reduction']:.1%}")
        
        # Query causal chain
        logger.info("\n🔗 Querying causal chain from knowledge graph...")
        causal_chain = await self.kg_service.query_causal_chain(
            start_signature=signature.hash,
            max_depth=5
        )
        
        if causal_chain["chain"]:
            logger.info(f"📈 Found causal chain with {len(causal_chain['chain'])} steps")
            for i, step in enumerate(causal_chain["chain"][:3]):
                logger.info(f"   Step {i+1}: {step['type']} -> {step.get('outcome', 'pending')}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.kg_service:
            await self.kg_service.close()
        logger.info("🧹 Cleanup completed")


async def main():
    """Main integration script"""
    integration = AURAKnowledgeGraphIntegration()
    
    try:
        # Initialize
        if not await integration.initialize():
            logger.error("❌ Failed to initialize integration")
            return
        
        # Run tests
        test_results = await integration.test_knowledge_graph_features()
        
        logger.info(f"\n📊 Test Results: {test_results['summary']['success_rate']} success rate")
        logger.info(f"   - Passed: {test_results['summary']['passed']}")
        logger.info(f"   - Failed: {test_results['summary']['failed']}")
        
        # Demonstrate real usage
        await integration.demonstrate_real_world_usage()
        
        # Get enhanced stats
        stats = await integration.kg_service.get_enhanced_stats()
        logger.info(f"\n📈 Knowledge Graph Stats:")
        logger.info(f"   - Nodes: {stats.get('total_nodes', 0)}")
        logger.info(f"   - Relationships: {stats.get('total_relationships', 0)}")
        logger.info(f"   - GDS Version: {stats.get('gds_version', 'N/A')}")
        logger.info(f"   - ML Queries: {stats.get('ml_queries_executed', 0)}")
        
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await integration.cleanup()


if __name__ == "__main__":
    asyncio.run(main())