"""
🎯 Enhanced Saga Pattern Compensation Example

Demonstrates the enhanced saga pattern compensation system with TDA-aware
error correlation, intelligent recovery strategies, and comprehensive monitoring.

This example shows:
- Multi-agent workflow with potential failures
- TDA-correlated failure analysis
- Intelligent compensation strategy selection
- Error correlation and recovery recommendations
- Real-time monitoring and metrics
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

from .saga_patterns import (
    SagaOrchestrator, SagaStep, CompensationType, CompensationHandler
)
from ..semantic.tda_integration import MockTDAIntegration
from ..semantic.base_interfaces import TDAContext


class MultiAgentWorkflowExample:
    """
    Example multi-agent workflow with saga pattern compensation
    """
    
    def __init__(self):
        self.tda_integration = MockTDAIntegration()
        self.saga_orchestrator = SagaOrchestrator(tda_integration=self.tda_integration)
        self.setup_compensation_handlers()
    
    def setup_compensation_handlers(self):
        """Set up custom compensation handlers for different scenarios"""
        
        # Data processing compensation handler
        async def data_processing_compensation(input_data):
            print(f"🔄 Compensating data processing for step: {input_data['step_name']}")
            # Simulate cleanup of processed data
            await asyncio.sleep(0.1)
            return {"cleanup_completed": True, "data_restored": True}
        
        data_handler = CompensationHandler(
            handler_id="data_processing_handler",
            step_name="data_processing",
            compensation_type=CompensationType.ROLLBACK,
            handler_function=data_processing_compensation,
            parameters={"cleanup_strategy": "full_rollback"},
            priority=10
        )
        
        # Model inference compensation handler
        async def model_inference_compensation(input_data):
            print(f"🤖 Compensating model inference for step: {input_data['step_name']}")
            # Simulate model state cleanup
            await asyncio.sleep(0.2)
            return {"model_state_reset": True, "cache_cleared": True}
        
        model_handler = CompensationHandler(
            handler_id="model_inference_handler",
            step_name="model_inference",
            compensation_type=CompensationType.FORWARD_RECOVERY,
            handler_function=model_inference_compensation,
            parameters={"recovery_strategy": "graceful_degradation"},
            priority=8
        )
        
        # Result aggregation compensation handler
        async def aggregation_compensation(input_data):
            print(f"📊 Compensating result aggregation for step: {input_data['step_name']}")
            # Simulate partial result cleanup
            await asyncio.sleep(0.05)
            return {"partial_results_cleaned": True, "state_consistent": True}
        
        aggregation_handler = CompensationHandler(
            handler_id="aggregation_handler",
            step_name="result_aggregation",
            compensation_type=CompensationType.ROLLBACK,
            handler_function=aggregation_compensation,
            parameters={"aggregation_strategy": "incremental_cleanup"},
            priority=5
        )
        
        # Register handlers
        self.saga_orchestrator.register_compensation_handler("data_processing", data_handler)
        self.saga_orchestrator.register_compensation_handler("model_inference", model_handler)
        self.saga_orchestrator.register_compensation_handler("result_aggregation", aggregation_handler)
    
    async def create_multi_agent_workflow(self, failure_scenario: str = "none") -> List[SagaStep]:
        """
        Create a multi-agent workflow with configurable failure scenarios
        """
        
        # Step 1: Data Preprocessing Agent
        async def data_preprocessing_action(input_data):
            print("📥 Data Preprocessing Agent: Processing input data...")
            await asyncio.sleep(0.2)  # Simulate processing time
            
            if failure_scenario == "data_preprocessing":
                raise Exception("Data preprocessing failed: Invalid data format")
            
            return {
                "processed_data": {"records": 1000, "features": 50},
                "preprocessing_time": 0.2,
                "data_quality_score": 0.95
            }
        
        async def data_preprocessing_compensation(input_data):
            print("🔄 Compensating data preprocessing...")
            await asyncio.sleep(0.1)
            return {"data_cleanup": "completed", "temp_files_removed": True}
        
        # Step 2: Feature Engineering Agent
        async def feature_engineering_action(input_data):
            print("🔧 Feature Engineering Agent: Creating features...")
            await asyncio.sleep(0.3)
            
            if failure_scenario == "feature_engineering":
                raise Exception("Feature engineering failed: Memory allocation error")
            
            previous_data = input_data["previous_results"]["data_preprocessing"]
            return {
                "engineered_features": {"feature_count": 75, "new_features": 25},
                "feature_importance": {"top_features": ["feature_1", "feature_2", "feature_3"]},
                "processing_time": 0.3,
                "base_records": previous_data["processed_data"]["records"]
            }
        
        async def feature_engineering_compensation(input_data):
            print("🔄 Compensating feature engineering...")
            await asyncio.sleep(0.15)
            return {"feature_cache_cleared": True, "memory_released": True}
        
        # Step 3: Model Inference Agent
        async def model_inference_action(input_data):
            print("🤖 Model Inference Agent: Running inference...")
            await asyncio.sleep(0.4)
            
            if failure_scenario == "model_inference":
                raise Exception("Model inference failed: CUDA out of memory")
            
            previous_features = input_data["previous_results"]["feature_engineering"]
            return {
                "predictions": {"accuracy": 0.92, "confidence": 0.88},
                "inference_time": 0.4,
                "model_version": "v2.1.0",
                "processed_features": previous_features["engineered_features"]["feature_count"]
            }
        
        async def model_inference_compensation(input_data):
            print("🔄 Compensating model inference...")
            await asyncio.sleep(0.2)
            return {"gpu_memory_freed": True, "model_state_reset": True}
        
        # Step 4: Result Aggregation Agent
        async def result_aggregation_action(input_data):
            print("📊 Result Aggregation Agent: Aggregating results...")
            await asyncio.sleep(0.1)
            
            if failure_scenario == "result_aggregation":
                raise Exception("Result aggregation failed: Database connection timeout")
            
            all_results = input_data["previous_results"]
            return {
                "final_results": {
                    "total_records": all_results["data_preprocessing"]["processed_data"]["records"],
                    "model_accuracy": all_results["model_inference"]["predictions"]["accuracy"],
                    "processing_pipeline_time": 1.0
                },
                "aggregation_time": 0.1,
                "results_stored": True
            }
        
        async def result_aggregation_compensation(input_data):
            print("🔄 Compensating result aggregation...")
            await asyncio.sleep(0.05)
            return {"partial_results_cleaned": True, "database_rollback": True}
        
        # Create saga steps
        steps = [
            SagaStep(
                step_id="data_preprocessing",
                name="data_preprocessing",
                action=data_preprocessing_action,
                compensation_action=data_preprocessing_compensation,
                parameters={"input_source": "data_lake", "format": "parquet"},
                max_retries=2
            ),
            SagaStep(
                step_id="feature_engineering",
                name="feature_engineering", 
                action=feature_engineering_action,
                compensation_action=feature_engineering_compensation,
                parameters={"feature_strategy": "automated", "memory_limit": "8GB"},
                max_retries=1
            ),
            SagaStep(
                step_id="model_inference",
                name="model_inference",
                action=model_inference_action,
                compensation_action=model_inference_compensation,
                parameters={"model_type": "transformer", "batch_size": 32},
                max_retries=3
            ),
            SagaStep(
                step_id="result_aggregation",
                name="result_aggregation",
                action=result_aggregation_action,
                compensation_action=result_aggregation_compensation,
                parameters={"output_format": "json", "storage": "database"},
                max_retries=2
            )
        ]
        
        return steps
    
    async def run_workflow_example(self, failure_scenario: str = "none"):
        """
        Run the multi-agent workflow example with optional failure scenario
        """
        print(f"\n🚀 Starting Multi-Agent Workflow Example")
        print(f"📋 Failure Scenario: {failure_scenario}")
        print("=" * 60)
        
        # Create TDA context for the workflow
        tda_context = TDAContext(
            correlation_id=f"workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            pattern_confidence=0.85,
            anomaly_severity=0.3 if failure_scenario == "none" else 0.7,
            current_patterns={"workflow_pattern": 0.8, "agent_coordination": 0.9},
            temporal_window="1h",
            metadata={
                "workflow_type": "multi_agent_processing",
                "expected_duration": "1.0s",
                "failure_scenario": failure_scenario
            }
        )
        
        # Set up mock TDA integration
        self.tda_integration.contexts[tda_context.correlation_id] = tda_context
        
        # Create workflow steps
        workflow_steps = await self.create_multi_agent_workflow(failure_scenario)
        
        # Execute saga
        start_time = datetime.now(timezone.utc)
        
        try:
            result = await self.saga_orchestrator.execute_saga(
                saga_id=f"multi_agent_workflow_{failure_scenario}",
                steps=workflow_steps,
                tda_correlation_id=tda_context.correlation_id
            )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            print("📊 Workflow Execution Results")
            print("=" * 60)
            print(f"Status: {result['status']}")
            print(f"Execution Time: {execution_time:.2f}s")
            print(f"Steps Executed: {result['steps_executed']}")
            
            if result['status'] == 'completed':
                print("✅ Workflow completed successfully!")
                print("\n🎯 Final Results:")
                for step_name, step_result in result['results'].items():
                    print(f"  • {step_name}: {json.dumps(step_result, indent=2)}")
            else:
                print("❌ Workflow failed with compensation")
                print(f"Error: {result.get('error', 'Unknown error')}")
                if 'compensated_steps' in result:
                    print(f"Compensated Steps: {result['compensated_steps']}")
            
            # Show saga metrics
            print("\n📈 Saga Orchestrator Metrics:")
            metrics = self.saga_orchestrator.get_saga_metrics()
            for key, value in metrics.items():
                print(f"  • {key}: {value}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Workflow execution failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def demonstrate_compensation_strategies(self):
        """
        Demonstrate different compensation strategies based on failure types
        """
        print("\n🎭 Demonstrating Compensation Strategies")
        print("=" * 60)
        
        failure_scenarios = [
            ("none", "Normal execution (no failures)"),
            ("data_preprocessing", "Data preprocessing failure"),
            ("feature_engineering", "Feature engineering failure"), 
            ("model_inference", "Model inference failure"),
            ("result_aggregation", "Result aggregation failure")
        ]
        
        results = {}
        
        for scenario, description in failure_scenarios:
            print(f"\n🔍 Testing: {description}")
            print("-" * 40)
            
            result = await self.run_workflow_example(scenario)
            results[scenario] = result
            
            # Brief pause between scenarios
            await asyncio.sleep(0.5)
        
        # Summary
        print("\n📋 Compensation Strategy Summary")
        print("=" * 60)
        
        for scenario, result in results.items():
            status_emoji = "✅" if result["status"] == "completed" else "❌"
            print(f"{status_emoji} {scenario}: {result['status']}")
            
            if "compensated_steps" in result:
                print(f"   Compensated: {len(result['compensated_steps'])} steps")
        
        return results
    
    async def test_custom_compensation_handlers(self):
        """
        Test custom compensation handlers for specific scenarios
        """
        print("\n🛠️ Testing Custom Compensation Handlers")
        print("=" * 60)
        
        # Test data processing compensation
        result = await self.saga_orchestrator.execute_custom_compensation(
            saga_id="test_saga",
            step_name="data_processing",
            compensation_type=CompensationType.ROLLBACK,
            parameters={"test_data": "sample_data"},
            tda_correlation_id="test-correlation"
        )
        
        print(f"Data Processing Compensation: {result['status']}")
        print(f"Handler ID: {result.get('handler_id', 'N/A')}")
        
        # Test model inference compensation
        result = await self.saga_orchestrator.execute_custom_compensation(
            saga_id="test_saga",
            step_name="model_inference", 
            compensation_type=CompensationType.FORWARD_RECOVERY,
            parameters={"model_state": "corrupted"},
            tda_correlation_id="test-correlation"
        )
        
        print(f"Model Inference Compensation: {result['status']}")
        print(f"Handler ID: {result.get('handler_id', 'N/A')}")


async def main():
    """
    Main function to run the enhanced saga pattern compensation example
    """
    print("🎯 Enhanced Saga Pattern Compensation Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("• Multi-agent workflow coordination")
    print("• TDA-aware failure analysis")
    print("• Intelligent compensation strategies")
    print("• Error correlation and recovery")
    print("• Real-time monitoring and metrics")
    
    example = MultiAgentWorkflowExample()
    
    # Run normal workflow
    print("\n🌟 Running Normal Workflow (No Failures)")
    await example.run_workflow_example("none")
    
    # Demonstrate compensation strategies
    await example.demonstrate_compensation_strategies()
    
    # Test custom compensation handlers
    await example.test_custom_compensation_handlers()
    
    print("\n🎉 Enhanced Saga Pattern Compensation Example Complete!")


if __name__ == "__main__":
    asyncio.run(main())